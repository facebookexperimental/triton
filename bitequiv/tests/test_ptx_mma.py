"""Tests for the shared MMA fence (bitequiv.ptx.mma). CPU-only, inline PTX."""
from pyptx.parser import parse

from bitequiv.ptx.linker import linearize
from bitequiv.ptx.mma import _is_mma, _mma_fence, _mma_token
from bitequiv.ptx_reduction import ptx_header

_HDR = ptx_header()


def _entry(body):
    ptx = _HDR + ".visible .entry k()\n{\n.reg .b32 %r<16>;\n.reg .f32 %f<16>;\n" + body + "ret;\n}\n"
    return [d for d in parse(ptx).directives if getattr(d, "is_entry", False)][0]


def _mma_insts(body):
    return [x for x in linearize(_entry(body)) if _is_mma(x)]


def _tok(body):
    (i, ) = _mma_insts(body)
    return _mma_token(i.opcode, i.modifiers)


_WGMMA = "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%r1}, %r2, %r3;\n"
_MMA_SYNC = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%r1}, {%r2}, {%r3}, {%r4};\n"


def test_native_k_and_tile_dropped_family_kept():
    # The canonical token keeps only the accumulation dtype FAMILY. The m/n tile AND the native k are
    # dropped: native k is fixed per family (carries no extra bit-relevant info) and keeping it would
    # block the v2==v5 merge, since tcgen05 has no k modifier.
    tok = _tok(_WGMMA)
    assert tok == "matmul|f16|cta1" and "k16" not in tok and "m64" not in tok


def test_wgmma_and_mma_sync_and_tcgen05_merge():
    # Form-invariant: mma.sync (v2), wgmma, and tcgen05 (v5) computing the same matmul get the SAME
    # token -- bit-identical for f16/bf16/tf32 (exact products + shared F=25 accumulate; see _mma_token).
    tcg = _tok("tcgen05.mma.cta_group::1.kind::f16 [%r1], %r2, %r3;\n")
    assert _tok(_WGMMA) == _tok(_MMA_SYNC) == tcg == "matmul|f16|cta1"


def test_tcgen05_canonical_and_fp8_kept_distinct():
    # Non-fp8 tcgen05 -> the canonical matmul token (merges with v2). fp8 (kind::f8f6f4) is the v2!=v5
    # exception: it keeps the native tcgen05 form so it never merges into matmul|.
    assert _tok("tcgen05.mma.cta_group::1.kind::f16 [%r1], %r2, %r3;\n") == "matmul|f16|cta1"
    fp8 = _tok("tcgen05.mma.cta_group::1.kind::f8f6f4 [%r1], %r2, %r3;\n")
    assert fp8.startswith("tcgen05|") and "kind::f8f6f4" in fp8


def test_wgmma_fence_wait_is_not_a_matmul():
    # A wgmma.wait_group / wgmma.fence is NOT a matmul: it must not enter the fence.
    assert _mma_insts("wgmma.wait_group.sync.aligned 0;\n") == []


def test_non_mma_entry_has_no_fence():
    assert _mma_fence(_entry("add.f32 %f1, %f2, %f3;\n")) is None


def test_proportional_tiling_reduces_to_same_fence():
    # n128 + 1 epilogue add  ==  2x n64 + 2 epilogue adds: same K/dtypes (m/n dropped) and the f32
    # epilogue PRESENCE (has_fma, has_addmul) is equal -> same fence (bit-irrelevant re-tiling).
    one = _mma_fence(_entry(_WGMMA + "add.f32 %f1, %f2, %f3;\n"))
    two = _mma_fence(_entry(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%r1}, %r2, %r3;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%r4}, %r5, %r6;\n"
        "add.f32 %f1, %f2, %f3;\n"
        "add.f32 %f4, %f5, %f6;\n"))
    assert one == two


def test_dtype_family_stays_distinct():
    # Different accumulation families must NOT merge (different rounding). f16 vs tf32. (The native tile
    # k is dropped -- it is fixed per family -- so it is no longer a splitter; the real K regrouping,
    # split-K, rides splits= and is covered in test_ptx_gemm_splitk.)
    f16 = _mma_fence(_entry(_WGMMA + "add.f32 %f1, %f2, %f3;\n"))
    tf32 = _mma_fence(_entry(
        "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 {%r1}, %r2, %r3;\n"
        "add.f32 %f1, %f2, %f3;\n"))
    assert f16 != tf32


def test_fp8_falls_back_to_conservative():
    f = _mma_fence(_entry(
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%r1}, %r2, %r3;\n"
        "add.f32 %f1, %f2, %f3;\n"))
    assert f is not None and f[0] == "mma-fp8"


def test_fp_fusion_splits_at_equal_op_count():
    # The measured gemm_bias_relu_fp_fusion over-merge: enable_fp_fusion on emits N fma, off emits N
    # add/mul -- the SAME total count -- so a single lumped fp count collides. Counting fma apart from
    # add/mul (ratio mma:fma:addmul) splits them (fma single-rounded vs mul+add double-rounded).
    fused = _mma_fence(_entry(_WGMMA + "fma.rn.f32 %f1, %f2, %f3, %f4;\nfma.rn.f32 %f5, %f6, %f7, %f8;\n"))
    unfused = _mma_fence(_entry(_WGMMA + "add.f32 %f1, %f2, %f3;\nadd.f32 %f5, %f6, %f7;\n"))
    assert fused != unfused  # presence (has_fma=1, has_addmul=0) vs (has_fma=0, has_addmul=1)


def test_epilogue_count_is_presence_not_scaled_count():
    # The split-K / bias_relu over-split: tcgen05's mma count stays 1 while the f32 epilogue add count
    # scales with the M/N tile (elements per thread), so a COUNT (even GCD-reduced against the constant
    # mma count) over-splits equivalent re-tilings. The fence records only PRESENCE, so same-token
    # entries differing ONLY in addmul COUNT (1 add vs 4 adds) merge -- the recovery this diff adds.
    _TCG = "tcgen05.mma.cta_group::1.kind::f16 [%r1], %r2, %r3;\n"
    one = _mma_fence(_entry(_TCG + "add.f32 %f1, %f2, %f3;\n"))
    four = _mma_fence(_entry(_TCG + "add.f32 %f1, %f2, %f3;\nadd.f32 %f4, %f5, %f6;\n"
                             "add.f32 %f7, %f8, %f9;\nadd.f32 %f10, %f11, %f12;\n"))
    assert one == four                                       # presence: both (has_fma=0, has_addmul=1)
    none = _mma_fence(_entry(_TCG))                          # no epilogue add at all
    assert none != one                                       # (has_addmul=0) still splits from (1)
