"""loops.py — loop-structure recovery (back-edges, nesting, accumulator detection). CPU-only."""
from pyptx.parser import parse

from bitequiv.ptx.forward import loops
from bitequiv.ptx_reduction import _ensure_header


def _entry(ptx):
    m = parse(_ensure_header(ptx))
    return next(d for d in m.directives if getattr(d, "is_entry", False))


# A loop that ACCUMULATES: `%f1 = %f1 + %f2` each iteration (a running total). Its step (32) is
# bit-relevant.
_ACC = """
.visible .entry k(.param .u64 p) {
  .reg .f32 %f<4>;
  .reg .b32 %r<4>;
  .reg .pred %p<2>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [p];
  mov.b32 %r1, 0;
  mov.f32 %f1, 0f00000000;
$L__BB0_1:
  add.f32 %f1, %f1, %f2;
  add.s32 %r1, %r1, 32;
  setp.lt.s32 %p1, %r1, 256;
  @%p1 bra $L__BB0_1;
  st.global.f32 [%rd1], %f1;
  ret;
}
"""

# A TILING loop: each iteration writes an independent output (no loop-carried fp total). Its step
# (64) is bit-free.
_TILE = """
.visible .entry k(.param .u64 p) {
  .reg .f32 %f<6>;
  .reg .b32 %r<4>;
  .reg .pred %p<2>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [p];
  mov.b32 %r2, 0;
$L__BB0_2:
  mul.f32 %f3, %f4, %f5;
  st.global.f32 [%rd1], %f3;
  add.s32 %r2, %r2, 64;
  setp.lt.s32 %p2, %r2, 512;
  @%p2 bra $L__BB0_2;
  ret;
}
"""


def test_back_edge_detected():
    insts, label_at = loops.instrs_and_labels(_entry(_ACC))
    assert len(loops.back_edges(insts, label_at)) == 1


def test_accumulating_loop_true():
    insts, lps = loops.find_loops(_entry(_ACC))
    assert loops.loop_accumulates(insts, lps[0], lps) is True


def test_tiling_loop_false():
    insts, lps = loops.find_loops(_entry(_TILE))
    assert loops.loop_accumulates(insts, lps[0], lps) is False


def test_loop_carried_accumulator_reg():
    insts, lps = loops.find_loops(_entry(_ACC))
    accs = loops.loop_carried_accumulators(insts, lps[0], lps)
    assert len(accs) == 1 and accs[0][1].opcode == "add"    # the self-accumulating add.f32


def test_innermost_and_own_body_pure():
    dummy = [None] * 10
    lps = [(0, 9), (3, 6)]                                  # outer loop + a nested loop
    assert loops.innermost_loop(4, lps) == (3, 6)
    assert loops.innermost_loop(1, lps) == (0, 9)
    assert loops.own_body(dummy, (0, 9), lps) == [0, 1, 2, 7, 8, 9]  # outer's own body = range minus nested
