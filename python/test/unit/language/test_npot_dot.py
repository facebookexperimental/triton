import pytest
import torch
import triton
import triton.language as tl


def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda is not None


def get_cc():
    if not is_cuda():
        return (0, 0)
    return torch.cuda.get_device_capability()


requires_cuda = pytest.mark.skipif(not is_cuda(), reason="NVIDIA GPU required")
requires_sm80 = pytest.mark.skipif(not is_cuda() or get_cc() < (8, 0), reason="SM80+ required")
requires_sm90 = pytest.mark.skipif(not is_cuda() or get_cc() < (9, 0), reason="SM90+ required")


@pytest.fixture(autouse=True)
def enable_npot(monkeypatch):
    monkeypatch.setenv("TRITON_ALLOW_NPOT", "1")


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def _run(BK):
    """Run matmul with NPOT BLOCK_K using a standard loop pattern."""
    M, N, K = 128, 128, BK * 2
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    ref = torch.matmul(a, b)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BLOCK_M=64, BLOCK_N=64, BLOCK_K=BK)
    max_diff = (c - ref.to(torch.float16)).abs().max().item()
    rel_err = max_diff / ref.abs().max().item()
    return rel_err


def test_npot_k48():
    err = _run(48)
    print(f"BK=48: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"K=48 failed with rel_err={err}"


def test_npot_k96():
    err = _run(96)
    print(f"BK=96: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"K=96 failed with rel_err={err}"


def test_pow2_k32():
    err = _run(32)
    print(f"BK=32: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"K=32 failed with rel_err={err}"


def test_pow2_k64():
    err = _run(64)
    print(f"BK=64: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"K=64 failed with rel_err={err}"


def test_npot_k192():
    """K=192 -- DSv3 QK head dimension."""
    err = _run(192)
    print(f"BK=192: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"K=192 failed with rel_err={err}"


# ============================================================================
# NPOT N tests: non-power-of-2 BLOCK_N for accumulator dimension
# ============================================================================


@triton.jit
def matmul_npot_n_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def _run_npot_n(BN, BK=32):
    """Run matmul with NPOT BLOCK_N."""
    M, N, K = 128, BN, BK * 2
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    ref = torch.matmul(a, b)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, BN))
    matmul_npot_n_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                               c.stride(1), BLOCK_M=128, BLOCK_N=BN, BLOCK_K=BK)
    max_diff = (c - ref.to(torch.float16)).abs().max().item()
    rel_err = max_diff / ref.abs().max().item()
    return rel_err


@requires_sm80
def test_npot_n96():
    err = _run_npot_n(96)
    print(f"BN=96: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"N=96 failed with rel_err={err}"


@requires_sm80
def test_npot_n192():
    err = _run_npot_n(192)
    print(f"BN=192: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"N=192 failed with rel_err={err}"


@requires_sm80
def test_npot_n48():
    err = _run_npot_n(48)
    print(f"BN=48: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"N=48 failed with rel_err={err}"


def test_pow2_n64():
    err = _run_npot_n(64)
    print(f"BN=64: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"N=64 failed with rel_err={err}"


@requires_sm80
def test_npot_n48_diag():
    """Diagnostic: identify which elements are wrong for N=48."""
    BN, BK = 48, 32
    M, N, K = 128, BN, BK
    # Test 1: all-ones
    a = torch.ones(M, K, device='cuda', dtype=torch.float16)
    b = torch.ones(K, N, device='cuda', dtype=torch.float16)
    c = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    ref = torch.matmul(a, b)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, BN))
    matmul_npot_n_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                               c.stride(1), BLOCK_M=128, BLOCK_N=BN, BLOCK_K=BK)
    print(f"\n=== Diagnostic: all-ones, BN={BN}, BK={BK} ===")
    print(f"Expected value everywhere: {K}")
    diff = (c - ref.to(torch.float16)).abs()
    print(f"Max diff: {diff.max().item()}")

    # Print full output pattern
    for r in list(range(16)) + list(range(32, 48)) + list(range(64, 80)):
        vals = c[r, :].tolist()
        # Compact: show which cols are correct (32.0) and which are zero
        pattern = ''.join(['.' if v == 0.0 else '#' for v in vals])
        print(f"Row {r:3d}: {pattern}")

    # Which rows/cols are wrong
    wrong_mask = diff > 0.1
    if wrong_mask.any():
        wrong_cols = torch.any(wrong_mask, dim=0).nonzero().squeeze(-1).tolist()
        wrong_rows = torch.any(wrong_mask, dim=1).nonzero().squeeze(-1).tolist()
        if isinstance(wrong_cols, int): wrong_cols = [wrong_cols]
        if isinstance(wrong_rows, int): wrong_rows = [wrong_rows]
        print(f"Wrong cols ({len(wrong_cols)}): {wrong_cols}")
        print(f"Wrong rows ({len(wrong_rows)}): first 20 = {wrong_rows[:20]}")
        # Show wrong value pattern at row 0
        for col in range(BN):
            v = c[0, col].item()
            e = ref.to(torch.float16)[0, col].item()
            if abs(v - e) > 0.1:
                print(f"  Row0 col {col:3d}: got {v:.1f} expected {e:.1f}")
    else:
        print("ALL CORRECT!")

    # Test 2: column-pattern (each B col has distinct value)
    b2 = torch.zeros(K, N, device='cuda', dtype=torch.float16)
    for j in range(N):
        b2[:, j] = float(j + 1)
    c2 = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    ref2 = torch.matmul(a, b2)
    matmul_npot_n_kernel[grid](a, b2, c2, M, N, K, a.stride(0), a.stride(1), b2.stride(0), b2.stride(1), c2.stride(0),
                               c2.stride(1), BLOCK_M=128, BLOCK_N=BN, BLOCK_K=BK)
    print(f"\n=== Diagnostic: col-pattern, BN={BN}, BK={BK} ===")
    print(f"Row0 expected: {ref2.to(torch.float16)[0,:].tolist()}")
    print(f"Row0 actual:   {c2[0,:].tolist()}")
    # Show which column each output maps to
    print("Column mapping:")
    for j in range(BN):
        actual = c2[0, j].item()
        effective_col = actual / K if K > 0 else 0
        expected_col = j + 1
        if abs(effective_col - expected_col) > 0.5:
            print(f"  col {j:3d}: got {actual:.1f} -> effective_col={effective_col:.1f} expected_col={expected_col}")

    print("\n=== Diagnostic complete ===")
    # Both probes must be numerically correct (this is the Bug B regression
    # guard). all-ones expects K everywhere; col-pattern expects K*(j+1).
    err1 = diff.max().item() / max(ref.abs().max().item(), 1e-6)
    err2 = (c2 - ref2.to(torch.float16)).abs().max().item() / max(ref2.abs().max().item(), 1e-6)
    assert err1 < 0.05, f"N=48 all-ones probe wrong: rel_err={err1}"
    assert err2 < 0.05, f"N=48 col-pattern probe wrong: rel_err={err2}"


# ============================================================================
# NPOT M tests: non-power-of-2 BLOCK_M for WGMMA M dimension
# ============================================================================


@triton.jit
def matmul_npot_m_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def _run_npot_m(BM, BK=32):
    """Run matmul with NPOT BLOCK_M."""
    M, N, K = BM, 64, BK * 2
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    ref = torch.matmul(a, b)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, 64))
    matmul_npot_m_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                               c.stride(1), BLOCK_M=BM, BLOCK_N=64, BLOCK_K=BK)
    max_diff = (c - ref.to(torch.float16)).abs().max().item()
    rel_err = max_diff / ref.abs().max().item()
    return rel_err


@requires_sm80
def test_npot_m96():
    """M=96: WGMMA with 96/64 = 1.5 reps, partial rep handled by dead register fixup."""
    err = _run_npot_m(96)
    print(f"BM=96: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"M=96 failed with rel_err={err}"


@requires_sm80
def test_npot_m48():
    """M=48: WGMMA with 48 < 64, single rep covering more than needed."""
    err = _run_npot_m(48)
    print(f"BM=48: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"M=48 failed with rel_err={err}"


@requires_sm80
def test_npot_m80():
    """M=80: 80/64 = 1.25 reps."""
    err = _run_npot_m(80)
    print(f"BM=80: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"M=80 failed with rel_err={err}"


@requires_sm80
def test_npot_m112():
    """M=112: 112/64 = 1.75 reps."""
    err = _run_npot_m(112)
    print(f"BM=112: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"M=112 failed with rel_err={err}"


# ============================================================================
# MMAv5 NPOT M tests: force NPOT BLOCK_M onto the SM100 MMAv5/tcgen05 path.
#
# The matmul_npot_m_kernel above uses M,N=BM,64 K=64 fp16 -- with a 4-warp
# config and small N it may still pick MMAv5 on SM100, but to *guarantee* the
# tcgen05 path we use a GEMM-shaped problem: fp16 A/B, BLOCK_N=128, BLOCK_K=64,
# multiple K iterations, num_warps=4. getMMAVersionSafe selects version 5 on
# cc>=10.0 for these operands. Confirm via TTGIR that `tc_gen5_mma` and
# `#ttg.tensor_memory` appear (NOT FMA `tt.dot`/`#blocked`).
# ============================================================================

requires_sm100 = pytest.mark.skipif(not is_cuda() or get_cc() < (10, 0), reason="SM100+ (Blackwell) required")


@triton.jit
def matmul_mmav5_m_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                          stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def _run_mmav5_m(BM, BN=128, BK=64, K=256):
    """Run an fp16 GEMM with NPOT BLOCK_M on the MMAv5/tcgen05 path.

    Returns (rel_err, compiled_kernel) so callers can additionally assert on
    the lowered TTGIR (e.g. that tc_gen5_mma was actually selected).
    """
    M, N = BM, BN
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    ref = torch.matmul(a, b)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    compiled = matmul_mmav5_m_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                           c.stride(0), c.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=4)
    max_diff = (c - ref.to(torch.float16)).abs().max().item()
    rel_err = max_diff / ref.abs().max().item()
    return rel_err, compiled


@requires_sm100
@pytest.mark.parametrize("BM", [64, 128, 80, 96, 112])
def test_mmav5_npot_m(BM):
    """NPOT BLOCK_M <= 128 on the SM100 MMAv5/tcgen05 path.

    64/128 are pow2 baselines; 80/96/112 are NPOT M that fit in a single
    MMAv5 row-block (blockM rounds up to 64 or 128). The accumulator lives
    in TMEM (tc_gen5_mma) and the NPOT row dim is handled modularly by
    tensorMemoryToLinearLayout + applyNpotReductionForTmem.
    """
    err, compiled = _run_mmav5_m(BM)
    print(f"MMAv5 BM={BM}: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"MMAv5 M={BM} failed with rel_err={err}"
    # Guard against a silent FMA fallback: the kernel MUST actually lower to the
    # tcgen05 MMAv5 path (otherwise this test would pass even if NPOT-M MMAv5
    # support regressed). Inspect the lowered TTGIR for the tc_gen5_mma op.
    ttgir = compiled.asm["ttgir"]
    assert "tc_gen5_mma" in ttgir, (f"MMAv5 M={BM} did not use tc_gen5_mma (likely FMA fallback); "
                                    f"NPOT-M MMAv5 lowering regressed")


@requires_sm100
@pytest.mark.parametrize("BM", [144, 160, 176, 192])
def test_mmav5_npot_m_multiblock(BM):
    """NPOT BLOCK_M in (128, 256) spans 2 MMAv5 row-blocks.

    These do NOT use MMAv5: the TMEM-load distributed #linear layout would
    over-span to the pow2-rounded row footprint (e.g. 256 for M=192), making
    the convert_layout->store SMEM staging misaligned. supportMMA therefore
    rejects NPOT M > 128 on MMAv5 (see lib/Analysis/Utility.cpp), so they fall
    back to the FMA path, which is correct (rel_err well under 0.05). Enabling
    MMAv5 for multi-row-block NPOT M needs dedicated TMEM multi-block layout
    work (tracked separately). This test guards correctness of the fallback.
    """
    err, compiled = _run_mmav5_m(BM)
    print(f"NPOT-M(multiblock,FMA) BM={BM}: rel_err={err:.6f} "
          f"{'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"NPOT M={BM} (FMA fallback) failed with rel_err={err}"
    # Documented behavior: NPOT M > 128 is rejected by supportMMA and falls
    # back to FMA, so tc_gen5_mma must NOT appear. If it does, the fallback
    # guard regressed (and the result may be wrong for the wrong reason).
    ttgir = compiled.asm["ttgir"]
    assert "tc_gen5_mma" not in ttgir, (f"NPOT M={BM} unexpectedly used tc_gen5_mma; supportMMA should reject "
                                        f"multi-row-block NPOT M and fall back to FMA")


# ============================================================================
# Edge case tests: primes, non-swizzle-compatible, combined NPOT dims
# ============================================================================


def test_npot_k33():
    """K=33: not swizzle-compatible (33 * 2 = 66 bytes, not divisible by 128).
    Should fall back to pow2 rounding."""
    err = _run(33)
    print(f"BK=33: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"K=33 failed with rel_err={err}"


def test_npot_k17():
    """K=17: prime number, not swizzle-compatible."""
    err = _run(17)
    print(f"BK=17: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"K=17 failed with rel_err={err}"


@requires_sm80
def test_npot_n24():
    """N=24: multiple NPOT factors (24 = 3 * 8)."""
    err = _run_npot_n(24)
    print(f"BN=24: rel_err={err:.6f} {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, f"N=24 failed with rel_err={err}"


@requires_sm80
def test_npot_k48_n96():
    """Combined NPOT K and N dimensions."""
    BK, BN = 48, 96
    M, N, K = 128, BN, BK * 2
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    ref = torch.matmul(a, b)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, BN))
    matmul_npot_n_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                               c.stride(1), BLOCK_M=128, BLOCK_N=BN, BLOCK_K=BK)
    max_diff = (c - ref.to(torch.float16)).abs().max().item()
    rel_err = max_diff / ref.abs().max().item()
    print(f"BK={BK}, BN={BN}: rel_err={rel_err:.6f} {'PASS' if rel_err < 0.05 else 'FAIL'}")
    # Bug B regression guard: combined NPOT K=48 + N=96 must be numerically
    # correct, not merely crash-free.
    assert rel_err < 0.05, f"K=48,N=96 failed with rel_err={rel_err}"


# ============================================================================
# MMAv2 NPOT tests: force MMAv2 (Ampere path) with DISABLE_MMA_V3
# MMAv2 uses m16n8 atoms with non-pow2 rep counts.
# ============================================================================

import os


@triton.jit
def matmul_f32_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
    acc += tl.dot(a, b)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def _run_mmav2(M, N, K, BM, BN, BK, num_warps=4):
    """Run matmul forcing MMAv2 and check NPOT correctness."""
    # Save/restore DISABLE_MMA_V3 in a finally so it never leaks into other
    # tests even if the kernel launch or compare raises.
    _prev_mma_v3 = os.environ.get("DISABLE_MMA_V3")
    os.environ["DISABLE_MMA_V3"] = "1"
    try:
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        c = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        ref = torch.matmul(a.float(), b.float())
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        matmul_f32_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                                c.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=num_warps)
        max_diff = (c - ref).abs().max().item()
        rel_err = max_diff / ref.abs().max().item()
    finally:
        if _prev_mma_v3 is None:
            os.environ.pop("DISABLE_MMA_V3", None)
        else:
            os.environ["DISABLE_MMA_V3"] = _prev_mma_v3
    return rel_err


@requires_sm80
def test_mmav2_npot_k48():
    err = _run_mmav2(64, 64, 48, 64, 64, 48)
    assert err < 0.01, f"MMAv2 NPOT K=48 failed: rel_err={err}"


@requires_sm80
def test_mmav2_npot_k96():
    err = _run_mmav2(64, 64, 96, 64, 64, 96)
    assert err < 0.01, f"MMAv2 NPOT K=96 failed: rel_err={err}"


@requires_sm80
def test_mmav2_npot_m96():
    err = _run_mmav2(96, 64, 64, 96, 64, 64)
    assert err < 0.01, f"MMAv2 NPOT M=96 failed: rel_err={err}"


@requires_sm80
def test_mmav2_npot_n48():
    err = _run_mmav2(64, 48, 64, 64, 48, 64)
    assert err < 0.01, f"MMAv2 NPOT N=48 failed: rel_err={err}"


@requires_sm80
def test_mmav2_npot_all():
    err = _run_mmav2(96, 48, 48, 96, 48, 48)
    assert err < 0.01, f"MMAv2 all NPOT failed: rel_err={err}"


@requires_sm80
def test_mmav2_pow2_baseline():
    err = _run_mmav2(64, 64, 64, 64, 64, 64)
    assert err < 0.01, f"MMAv2 pow2 baseline failed: rel_err={err}"
