"""Unified per-element model — Step 1: Mma + LoopReduce nodes + builder guards; Step 3: forward
interpreter loop-carried add-accumulation -> LoopReduce. CPU-only."""
from pyptx.parser import parse

from bitequiv.ptx.builder import collapse_balanced, tree_hash
from bitequiv.ptx.forward.interp import ForwardInterp, forward_module_descriptor
from bitequiv.ptx.treeir import FpOp, ITreeReduce, Leaf, LoopReduce, Mma
from bitequiv.ptx_reduction import _ensure_header


def _leaf(i):
    return Leaf(f"c{i}", frozenset({i}))


# -- Mma node -----------------------------------------------------------------


def test_mma_sig_and_hash():
    a = Mma("wgmma|k16|.f32,.f16,.f16", (), (_leaf(0), _leaf(1)))
    b = Mma("wgmma|k16|.f32,.f16,.f16", (), (_leaf(0), _leaf(1)))
    assert a.sig().startswith("MMA[wgmma|k16")
    assert tree_hash(a) == tree_hash(b)                       # same token + children -> same hash
    assert tree_hash(a) != tree_hash(Mma("wgmma|k32|.f32,.f16,.f16", (), (_leaf(0), _leaf(1))))  # K split


def test_mma_flags_split():
    assert tree_hash(Mma("t", ("1",), (_leaf(0),))) != tree_hash(Mma("t", ("-1",), (_leaf(0),)))


# -- LoopReduce node ----------------------------------------------------------


def _chunk():
    return FpOp("mul", (".f32", ), (_leaf(0), _leaf(1)))


def test_loopreduce_chunk_bearing_splits_step():
    a = LoopReduce("add.f32", _chunk(), (16, 64))
    b = LoopReduce("add.f32", _chunk(), (32, 32))
    assert tree_hash(a) != tree_hash(b)                       # chunk-bearing: different step -> split


def test_loopreduce_chunk_invariant_merges_step():
    a = LoopReduce("add.f32", _chunk(), ())                   # chunk-invariant (key dropped)
    b = LoopReduce("add.f32", _chunk(), ())
    assert tree_hash(a) == tree_hash(b)
    assert tree_hash(a) != tree_hash(LoopReduce("add.f32", _chunk(), (16, 64)))  # invariant != bearing


# -- builder guards: a reduction over Mma / LoopReduce must NOT collapse -------


def test_reduction_over_mma_does_not_collapse():
    t = FpOp("add", (".f32", ), (Mma("wgmma|k16|.f32", (), (_leaf(0), )),
                                 Mma("wgmma|k16|.f32", (), (_leaf(1), ))))
    assert not isinstance(collapse_balanced(t), ITreeReduce)  # Mma not layout-invariant -> no collapse


def test_reduction_over_loopreduce_does_not_collapse():
    lr0 = LoopReduce("add.f32", FpOp("mul", (".f32", ), (_leaf(0), _leaf(0))), (16, 64))
    lr1 = LoopReduce("add.f32", FpOp("mul", (".f32", ), (_leaf(1), _leaf(1))), (16, 64))
    assert not isinstance(collapse_balanced(FpOp("add", (".f32", ), (lr0, lr1))), ITreeReduce)


def test_loopreduce_root_survives_collapse():
    # single output = one LoopReduce -> collapse_balanced returns it unchanged (it IS the per-element sig)
    lr = LoopReduce("add.f32", Mma("tcgen05|.kind::f16", (), (_leaf(0), )), ())
    c = collapse_balanced(lr)
    assert isinstance(c, LoopReduce) and c.sig() == lr.sig()


# -- Step 3: forward interpreter loop-carried add-accumulation -> LoopReduce ---

# A chunked sum: each iteration loads one element and adds it to the running total `%f1`; the
# induction var steps by {step} (BLOCK_N). The forward walk should summarize this as one LoopReduce.
_LOOP_SUM = """
.visible .entry k(.param .u64 pin, .param .u64 pout) {{
  .reg .f32 %f<4>;
  .reg .b32 %r<4>;
  .reg .pred %p<2>;
  .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [pin];
  ld.param.u64 %rd2, [pout];
  mov.b32 %r1, 0;
  mov.f32 %f1, 0f00000000;
  cvta.to.global.u64 %rd3, %rd1;
$L__BB0_1:
  mul.wide.s32 %rd4, %r1, 4;
  add.s64 %rd5, %rd3, %rd4;
  ld.global.f32 %f2, [%rd5];
  add.f32 %f1, %f1, %f2;
  add.s32 %r1, %r1, {step};
  setp.lt.s32 %p1, %r1, 4096;
  @%p1 bra $L__BB0_1;
  cvta.to.global.u64 %rd6, %rd2;
  st.global.f32 [%rd6], %f1;
  ret;
}}
"""


def _interp(ptx):
    m = parse(_ensure_header(ptx))
    f = next(d for d in m.directives if getattr(d, "is_entry", False))
    interp = ForwardInterp(f)
    interp.run()
    return interp


def test_loop_add_emits_loopreduce():
    interp = _interp(_LOOP_SUM.format(step=256))
    lrs = [v for v in interp.regs.values() if isinstance(v, LoopReduce)]
    assert len(lrs) == 1                                       # the accumulator became a fold summary
    lr = lrs[0]
    assert lr.op == "add.f32" and isinstance(lr.chunk, Leaf) and lr.key == ((256, ), )  # chunk step key


def test_loop_step_splits_and_is_deterministic():
    d256 = forward_module_descriptor(_LOOP_SUM.format(step=256))
    assert d256 == forward_module_descriptor(_LOOP_SUM.format(step=256))    # deterministic
    assert d256 != forward_module_descriptor(_LOOP_SUM.format(step=128))    # different BLOCK_N -> split
