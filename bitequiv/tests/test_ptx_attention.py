"""Two fail-closed MMA residuals that flash attention exposes, in hand-crafted PTX (CPU-only).

1. A fold whose SPLIT POINT moves with the output tile. Causal attention runs its fully-unmasked KV
   blocks in one loop bounded by ``BLOCK_M * program_id`` and the masked diagonal band in a second
   loop with DIFFERENT arithmetic. Both loops exist, with the same op mix, at every ``BLOCK_M`` — only
   the runtime bound moves — so every count-based term of the key is identical while the bits are not.
   :func:`~bitequiv.ptx.forward.loops.reduction_trip_signature` must carry the bound's affine form.
2. The matmul REDUCTION EXTENT. The fence drops the MMA count because a re-tiling issues a different
   count over the same dot products — a licence that comes from the pure-fold proof. Where that proof
   fails, the count is the only witness of how many products reach the accumulator (attention's head
   dim), so it must be back in the key; where it holds, it must stay out (BLOCK_K / v2==v5 recovery).
"""
from pyptx.parser import parse

from bitequiv.ptx.forward import loops
from bitequiv.ptx.forward.interp import forward_module_descriptor as CHECK
from bitequiv.ptx.mma import mma_token_counts
from bitequiv.ptx_reduction import _ensure_header, ptx_header

_HDR = ptx_header()
_MMA = "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%r3}, %r4, %r5;\n"
# A cross-lane butterfly makes the entry reduce over the MMA output -> the fail-closed branch.
_REDUCE = "shfl.sync.bfly.b32 %r6, %r7, 16, 31, -1;\nadd.f32 %f1, %f1, %f3;\n"


def _entry(ptx):
    return next(d for d in parse(_ensure_header(ptx)).directives if getattr(d, "is_entry", False))


def _tiled_fold(tile, nmma=1, first=0):
    """An accumulating loop whose bound is ``tile * program_id`` — the unmasked region of a
    tile-diagonal fold. ``first`` shifts every virtual register number (allocation noise)."""
    r = lambda n: f"%r{n + first}"  # noqa: E731
    return _HDR + f"""
.visible .entry k(.param .u64 p) {{
  .reg .f32 %f<8>;
  .reg .b32 %r<{16 + first}>;
  .reg .pred %p<4>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [p];
  mov.f32 %f1, 0f00000000;
  mov.u32 {r(8)}, %ctaid.x;
  mul.lo.s32 {r(9)}, {r(8)}, {tile};
  mov.b32 {r(1)}, 0;
$L__FOLD:
  {_MMA * nmma}  {_REDUCE}  add.s32 {r(1)}, {r(1)}, 1;
  setp.lt.s32 %p1, {r(1)}, {r(9)};
  @%p1 bra $L__FOLD;
  st.global.f32 [%rd1], %f1;
  ret;
}}
"""


def _param_fold(first=0):
    """Same fold, but bounded by a kernel PARAM (no tile scaling) — the non-causal shape."""
    r = lambda n: f"%r{n + first}"  # noqa: E731
    return _HDR + f"""
.visible .entry k(.param .u64 p, .param .u32 n) {{
  .reg .f32 %f<8>;
  .reg .b32 %r<{16 + first}>;
  .reg .pred %p<4>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 {r(9)}, [n];
  mov.f32 %f1, 0f00000000;
  mov.b32 {r(1)}, 0;
$L__FOLD:
  {_MMA}  {_REDUCE}  add.s32 {r(1)}, {r(1)}, 1;
  setp.lt.s32 %p1, {r(1)}, {r(9)};
  @%p1 bra $L__FOLD;
  st.global.f32 [%rd1], %f1;
  ret;
}}
"""


def _pure_fold(nmma):
    """A PROVEN-pure tensor-core fold (accumulator touched only by MMA, no reduction over the output):
    the case whose MMA count MUST stay out of the key so re-tiling / v2==v5 keep merging."""
    return _HDR + f"""
.visible .entry k(.param .u64 p) {{
  .reg .f32 %f<8>;
  .reg .b32 %r<8>;
  .reg .pred %p<4>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [p];
  mov.b32 %r1, 0;
$L__K:
  {_MMA * nmma}  add.s32 %r1, %r1, 16;
  setp.lt.s32 %p1, %r1, 256;
  @%p1 bra $L__K;
  st.global.f32 [%rd1], %f1;
  ret;
}}
"""


# -- 1. the fold's split point ------------------------------------------------------------------


def test_runtime_bound_recovered_as_affine():
    sig = loops.reduction_trip_signature(_entry(_tiled_fold(64)))
    assert ("bound", ("64*%ctaid.x", )) in sig


def test_tile_scale_of_a_runtime_bound_splits():
    # POSITIVE: same instructions, same counts -- only the tile the bound scales by differs.
    assert loops.reduction_trip_signature(_entry(_tiled_fold(64))) \
        != loops.reduction_trip_signature(_entry(_tiled_fold(128)))
    assert CHECK(_tiled_fold(64)) != CHECK(_tiled_fold(128))


def test_same_bound_does_not_split_on_register_noise():
    # NEGATIVE: one runtime bound, two register numberings. The tier is the bound's affine FORM, not
    # its text, so allocation noise must not separate two bit-identical configs.
    assert loops.reduction_trip_signature(_entry(_param_fold(0))) \
        == loops.reduction_trip_signature(_entry(_param_fold(32)))
    assert CHECK(_param_fold(0)) == CHECK(_param_fold(32))


def test_no_runtime_bound_leaves_the_signature_untouched():
    # NEGATIVE: a fold with no reduction loop at all keeps `None` -- a plain tiled GEMM must not
    # acquire a `splits=` fence (that would undo its tiling recovery).
    assert loops.reduction_trip_signature(_entry(_pure_fold(1))) is None
    assert "splits=" not in str(CHECK(_pure_fold(1)))


# -- 2. the matmul reduction extent -------------------------------------------------------------


def test_token_counts_count_every_matmul():
    assert mma_token_counts(_entry(_pure_fold(3))) == (("matmul|f16|cta1", 3), )


def test_reduction_extent_splits_when_the_fold_is_not_proven_pure():
    # POSITIVE: an entry that reduces over the MMA output and issues twice the matmuls is summing
    # twice the products -- a different value -- so it must not share the fail-closed key.
    a, b = CHECK(_param_fold()), CHECK(_param_fold().replace(_MMA, _MMA * 2, 1))
    assert "mma+fp|" in str(a) and a != b


def test_reduction_extent_stays_out_of_a_proven_pure_fold():
    # NEGATIVE: re-tiling a proven-pure fold issues a different MMA count over the SAME dot products.
    # The count must not leak into that key, or BLOCK_K and v2==v5 stop merging.
    assert "mma+fp|" not in str(CHECK(_pure_fold(1)))
    assert CHECK(_pure_fold(1)) == CHECK(_pure_fold(2))
