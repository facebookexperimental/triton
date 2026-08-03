"""Tests for the shared MMA fence (bitequiv.ptx.mma). CPU-only, inline PTX."""
from pyptx.parser import parse

from bitequiv.ptx.linker import linearize
from bitequiv.ptx.mma import _is_mma, _mma_fence, _mma_token

_HDR = ".version 8.5\n.target sm_90a\n.address_size 64\n"


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


def test_k_kept_tile_dropped_dtypes_kept():
    tok = _tok(_WGMMA)
    assert "k16" in tok and "m64" not in tok and "n128" not in tok and ".f16" in tok


def test_wgmma_vs_mma_sync_distinct():
    assert _tok(_WGMMA) != _tok(_MMA_SYNC)


def test_tcgen05_keeps_kind_and_cta_group():
    tok = _tok("tcgen05.mma.cta_group::1.kind::f16 [%r1], %r2, %r3;\n")
    assert "kind::f16" in tok and "cta_group::1" in tok and tok.startswith("tcgen05|")


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


def test_k_split_stays_distinct():
    k16 = _mma_fence(_entry(_WGMMA + "add.f32 %f1, %f2, %f3;\n"))
    k32 = _mma_fence(_entry(
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.f16.f16 {%r1}, %r2, %r3;\n"
        "add.f32 %f1, %f2, %f3;\n"))
    assert k16 != k32


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
