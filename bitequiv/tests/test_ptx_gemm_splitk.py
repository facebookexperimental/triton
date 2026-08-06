"""Split-K soundness: a GEMM whose K sum is regrouped by an OUTER reduction loop (split-K combining
per-slice partials) is NOT bitwise-equivalent across split counts, yet its static MMA/op structure is
identical, so the tiling-invariant MMA fence alone over-merges the split counts. The checker must key on
the outer loop's TRIP (see ``bitequiv.ptx.forward.loops.reduction_trip_signature`` +
``_mma_entry_descriptor``). A PLAIN single-K-loop GEMM has no such nested reduction and must keep the
fence (num_warps / BLOCK_M / BLOCK_N recovered). Hand-crafted PTX, CPU-only."""
from pyptx.parser import parse

from bitequiv.ptx.forward import loops
from bitequiv.ptx.forward.interp import forward_module_descriptor as CHECK
from bitequiv.ptx_reduction import _ensure_header, ptx_header

_HDR = ptx_header()
_MMA = "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%r3}, %r4, %r5;\n"


def _entry(ptx):
    return next(d for d in parse(_ensure_header(ptx)).directives if getattr(d, "is_entry", False))


# split-K: an OUTER loop (acc += partial) around an INNER K-loop (the MMA). The outer bound is the
# split count; only it changes between the two below -> they must get DISTINCT descriptors.
def _split_ptx(num_splits):
    return _HDR + f"""
.visible .entry k(.param .u64 p) {{
  .reg .f32 %f<8>;
  .reg .b32 %r<8>;
  .reg .pred %p<4>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [p];
  mov.f32 %f1, 0f00000000;
  mov.b32 %r1, 0;
$L__OUTER:
  mov.b32 %r2, 0;
$L__INNER:
  {_MMA}  add.s32 %r2, %r2, 1;
  setp.lt.s32 %p1, %r2, 16;
  @%p1 bra $L__INNER;
  add.f32 %f1, %f1, %f2;
  add.s32 %r1, %r1, 1;
  setp.lt.s32 %p2, %r1, {num_splits};
  @%p2 bra $L__OUTER;
  st.global.f32 [%rd1], %f1;
  ret;
}}
"""


# plain GEMM: a SINGLE K-loop (the MMA), the acc += is a post-loop epilogue -> no nested reduction.
_PLAIN = _HDR + f"""
.visible .entry k(.param .u64 p) {{
  .reg .f32 %f<8>;
  .reg .b32 %r<8>;
  .reg .pred %p<4>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [p];
  mov.f32 %f1, 0f00000000;
  mov.b32 %r2, 0;
$L__INNER:
  {_MMA}  add.s32 %r2, %r2, 1;
  setp.lt.s32 %p1, %r2, 16;
  @%p1 bra $L__INNER;
  add.f32 %f1, %f1, %f2;
  st.global.f32 [%rd1], %f1;
  ret;
}}
"""


def test_nested_reduction_detected_for_split_not_plain():
    insts, lps = loops.find_loops(_entry(_split_ptx(4)))
    assert loops.outer_reduction_loops(insts, lps)          # split-K: outer reduction over the K-loop
    insts, lps = loops.find_loops(_entry(_PLAIN))
    assert not loops.outer_reduction_loops(insts, lps)      # plain GEMM: single K-loop, no nesting


def test_trip_signature_distinguishes_split_count():
    assert loops.reduction_trip_signature(_entry(_split_ptx(4))) \
        != loops.reduction_trip_signature(_entry(_split_ptx(8)))
    assert loops.reduction_trip_signature(_entry(_PLAIN)) is None


def test_split_counts_get_distinct_descriptors():
    # the soundness fix: different split counts must NOT collapse to one descriptor (was the over-merge)
    assert CHECK(_split_ptx(4)) != CHECK(_split_ptx(8))
    assert "splits=" in str(CHECK(_split_ptx(4)))


def test_plain_gemm_keeps_fence_no_splits():
    d = CHECK(_PLAIN)
    assert d and "splits=" not in str(d)                    # fence unchanged -> tiling recovery preserved
