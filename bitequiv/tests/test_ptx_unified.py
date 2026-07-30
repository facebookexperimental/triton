"""Unified per-element model — Step 1: Mma + LoopReduce nodes + builder guards. CPU-only."""
from bitequiv.ptx.builder import collapse_balanced, tree_hash
from bitequiv.ptx.treeir import FpOp, ITreeReduce, Leaf, LoopReduce, Mma


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
