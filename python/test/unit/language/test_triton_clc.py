"""Tests for the core Triton CLC (Cluster Launch Control) tile scheduler.

CLC is a Blackwell (SM100+) feature. These tests exercise the standalone
``tl.clc_tile_scheduler`` API on a plain (non warp-specialized) persistent
kernel -- no ``tl.range(..., warp_specialize=True)`` is used, so AutoWS never
runs.
"""
import pytest
import torch

import triton
import triton.language as tl
from triton import knobs


def is_blackwell():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


requires_blackwell = pytest.mark.skipif(not is_blackwell(), reason="CLC requires Blackwell (SM100+)")


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _clc_count_kernel(counts_ptr, num_tiles):
    # Scheduler-only kernel: claim tiles via CLC and count how often each is seen.
    sched = tl.clc_tile_scheduler()
    while sched.is_valid():
        tid = sched.tile_id[0]
        sched.try_cancel()
        tl.atomic_add(counts_ptr + tid, 1, mask=tid < num_tiles)
        sched = sched.advance()


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _clc_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  #
                       stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                       GROUP_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    sched = tl.clc_tile_scheduler()
    while sched.is_valid():
        tile_id = sched.tile_id[0]
        # Issue the CLC request for the next tile early to overlap the matmul.
        sched.try_cancel()

        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M)
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_rem = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = acc.to(c_ptr.dtype.element_ty)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

        sched = sched.advance()


# ---------------------------------------------------------------------------
# Explicit CLC GEMM (warp_specialize=False -- plain persistent kernel)
# ---------------------------------------------------------------------------
@requires_blackwell
@pytest.mark.parametrize("M, N, K", [(256, 256, 128), (1000, 1000, 500)])
def test_clc_gemm(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 32, 8
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (num_tiles, )
    _clc_matmul_kernel[grid](
        a, b, c, M, N, K,  #
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),  #
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    ref = torch.matmul(a, b)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


# ---------------------------------------------------------------------------
# Core unit tests
# ---------------------------------------------------------------------------
@requires_blackwell
@pytest.mark.parametrize("num_tiles", [1, 7, 133, 1000])
def test_clc_scheduler_visits_each_tile_once(num_tiles):
    # Every tile in the grid must be claimed exactly once across all CTAs.
    counts = torch.zeros(num_tiles, device="cuda", dtype=torch.int32)
    _clc_count_kernel[(num_tiles, )](counts, num_tiles)
    counts = counts.cpu()
    assert counts.min().item() == 1, "some tile was never claimed"
    assert counts.max().item() == 1, "some tile was claimed more than once"


@requires_blackwell
def test_clc_emits_expected_ops_in_while_loop():
    counts = torch.zeros(4, device="cuda", dtype=torch.int32)
    kernel = _clc_count_kernel[(4, )](counts, 4)
    ttir = kernel.asm["ttir"]
    for op in ("ttng.clc_try_cancel", "ttng.clc_load_result", "ttng.clc_is_canceled", "ttng.clc_get_program_id"):
        assert op in ttir, f"expected {op} in TTIR"
    assert "scf.while" in ttir, "expected the CLC scheduler to form an scf.while persistent loop"


@requires_blackwell
def test_clc_drops_unused_tile_dims():
    # The count kernel only reads tile_id[0]; y/z decodes must never be emitted.
    counts = torch.zeros(4, device="cuda", dtype=torch.int32)
    kernel = _clc_count_kernel[(4, )](counts, 4)
    ttir = kernel.asm["ttir"]
    assert ttir.count("ttng.clc_get_program_id") == 1, "unused tile dimensions should not be decoded"


def test_clc_depth_gt1_is_rejected(monkeypatch):
    # Deeper prefetch needs drain-on-exit, which is not implemented yet: the API
    # must fail loudly rather than silently miscompile.
    monkeypatch.setattr(knobs.nvidia, "ws_tile_prefetch_depth", 2)

    @triton.jit
    def _k(counts_ptr, num_tiles):
        sched = tl.clc_tile_scheduler()
        while sched.is_valid():
            tid = sched.tile_id[0]
            sched.try_cancel()
            tl.atomic_add(counts_ptr + tid, 1, mask=tid < num_tiles)
            sched = sched.advance()

    counts = torch.zeros(4, device="cuda", dtype=torch.int32)
    with pytest.raises(triton.compiler.errors.CompilationError, match="TRITON_WS_TILE_PREFETCH_DEPTH"):
        _k[(4, )](counts, 4)
