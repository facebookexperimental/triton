"""
Pure Triton 2-CTA matmul test (no TLX).

The user writes standard Triton code:
  - Loads FULL B (BLOCK_K x BLOCK_N) via TMA descriptor
  - Calls tl.dot(a, b, acc, two_ctas=True)
  - Launches with ctas_per_cga=(2, 1, 1)

The compiler handles 2-CTA automatically:
  - Transform2CTALoads: splits B load so each CTA loads BLOCK_N/2 columns
  - Insert2CTASync: inserts cross-CTA "arrive remote, wait local" sync
  - AccelerateMatmul: emits tcgen05.mma.cta_group::2

Tests:
  1. Non-WS: Compilation + correctness with ctas_per_cga=(2, 1, 1)
  2. Auto-WS + 2CTA: Compilation + correctness
  3. Performance: Pure Triton 2-CTA vs TLX 2-CTA
"""

import os
from contextlib import contextmanager
from typing import Optional

import pytest
import torch
import triton
import triton.language as tl
from triton.language.extra.tlx.tutorials.blackwell_gemm_2cta import (
    tcgen5_dot_kernel2cta_tma as _tlx_2cta_kernel, )


def alloc_fn(size: int, align: int, stream: Optional[int]):
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(alloc_fn)

# ---------------------------------------------------------------------------
# Non-WS 2-CTA kernel
# ---------------------------------------------------------------------------


@triton.jit  # noqa: TR001 — test kernel with fixed block sizes
def matmul_2cta_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_tiles = tl.cdiv(K, BLOCK_K)
    for k in range(k_tiles):
        offs_k = k * BLOCK_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, accumulator, two_ctas=True)

    c = accumulator.to(tl.float16)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def matmul_2cta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (max(triton.cdiv(M, BLOCK_M), 2), triton.cdiv(N, BLOCK_N))

    matmul_2cta_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_stages=1,
        ctas_per_cga=(2, 1, 1),
    )
    return c


@pytest.mark.parametrize("M,N,K", [
    (128, 128, 64),
    (256, 256, 128),
])
def test_matmul_2cta_correctness(M, N, K):
    """Test that 2-CTA matmul produces correct results."""
    torch.manual_seed(42)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    ref = torch.matmul(a, b)
    out = matmul_2cta(a, b)

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


def test_matmul_2cta_compilation():
    """Test that the kernel compiles without errors."""
    torch.manual_seed(42)
    M, N, K = 128, 128, 64
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    out = matmul_2cta(a, b)
    assert out.shape == (M, N)
    assert out.dtype == torch.float16


# ---------------------------------------------------------------------------
# Auto-WS + 2-CTA tests
# ---------------------------------------------------------------------------


@contextmanager
def ws_env():
    """Set Meta WS env vars for the duration of the context."""
    old = {k: os.environ.get(k) for k in ["TRITON_USE_META_WS", "TRITON_USE_META_PARTITION", "TRITON_ALWAYS_COMPILE"]}
    os.environ["TRITON_USE_META_WS"] = "1"
    os.environ["TRITON_USE_META_PARTITION"] = "1"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@triton.jit  # noqa: TR001 — test kernel with fixed block sizes
def matmul_2cta_ws_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_tiles = tl.cdiv(K, BLOCK_K)
    for k in tl.range(0, k_tiles, warp_specialize=True):
        offs_k = k * BLOCK_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, accumulator, two_ctas=True)

    c = accumulator.to(tl.float16)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def matmul_2cta_ws(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (max(triton.cdiv(M, BLOCK_M), 2), triton.cdiv(N, BLOCK_N))

    with ws_env():
        matmul_2cta_ws_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_stages=1,
            ctas_per_cga=(2, 1, 1),
        )
    return c


@pytest.mark.parametrize("M,N,K", [(128, 128, 128),  # Single cluster, 2 K iters
                                   (256, 256, 64),  # Multi cluster, 1 K iter
                                   (128, 256, 128),  # 2 clusters in Y only, 2 K iters
                                   (256, 256, 128),  # Multi cluster both dims, 2 K iters
                                   ])
def test_matmul_2cta_ws_correctness(M, N, K):
    """Test WS + 2-CTA matmul produces correct results."""
    torch.manual_seed(42)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    ref = torch.matmul(a, b)
    out = matmul_2cta_ws(a, b)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


def test_matmul_2cta_ws_compilation():
    """Test that WS + 2-CTA kernel compiles and runs without errors."""
    torch.manual_seed(42)
    M, N, K = 256, 256, 128
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    out = matmul_2cta_ws(a, b)
    torch.cuda.synchronize()
    assert out.shape == (M, N)
    assert out.dtype == torch.float16


# ---------------------------------------------------------------------------
# Performance benchmark: Triton 1CTA vs Triton 2CTA vs TLX 2CTA
# Shows the perf benefit of 2CTA (this diff's contribution).
# Config: BLOCK=128x128x128, num_stages=2, ctas_per_cga=(2,1,1).
#
# Note on 2CTA+WS vs 1CTA+WS:
#   2CTA+WS may not outperform 1CTA+WS because Insert2CTASync adds
#   +6 mbarrier ops and +3 barrier.cluster ops (cross-CTA sync) on top
#   of the WS pipeline barriers. This extra sync overhead offsets the
#   2CTA MMA bandwidth benefit when WS is already pipelining effectively.
#   Merging Insert2CTASync barriers with WS pipeline barriers is a
#   follow-up optimization.
# ---------------------------------------------------------------------------


@triton.jit  # noqa: TR001 — test kernel with fixed block sizes
def matmul_1cta_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
                                       block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
                                       block_shape=[BLOCK_K, BLOCK_N])
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, accumulator)
    c = accumulator.to(tl.float16)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


@triton.jit  # noqa: TR001 — test kernel with fixed block sizes
def matmul_1cta_ws_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
                                       block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
                                       block_shape=[BLOCK_K, BLOCK_N])
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=True):
        offs_k = k * BLOCK_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, accumulator)
    c = accumulator.to(tl.float16)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def _launch_kernel(a, b, kernel, num_stages=2, ctas_per_cga=None, use_ws_env=False, BLOCK_M=128, BLOCK_N=128,
                   BLOCK_K=128):
    """Generic launcher for perf benchmarks."""
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (max(triton.cdiv(M, BLOCK_M), 2), triton.cdiv(N, BLOCK_N))
    kwargs = dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_stages=num_stages)
    if ctas_per_cga is not None:
        kwargs["ctas_per_cga"] = ctas_per_cga
    ctx = ws_env() if use_ws_env else contextmanager(lambda: (yield))()
    with ctx:
        kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            **kwargs,
        )
    return c


def _launch_tlx_2cta(a, b):
    """Launch TLX manual 2-CTA kernel with same config."""
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
    grid = (max(M // BLOCK_M, 2), N // BLOCK_N)
    _tlx_2cta_kernel[grid](
        a,
        a.stride(0),
        a.stride(1),
        b,
        b.stride(0),
        b.stride(1),
        c,
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        OUT_DTYPE=tl.float32,
        M=M,
        N=N,
        K=K,
        num_stages=2,
        ctas_per_cga=(2, 1, 1),
    )
    return c


@pytest.mark.parametrize("M,N,K", [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
])
def test_matmul_2cta_perf(M, N, K):
    """Compare 1CTA vs 2CTA vs TLX-2CTA to show the perf benefit of 2-CTA MMA.

    All variants use BLOCK=128x128x128, num_stages=2.
    """
    torch.manual_seed(42)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    def tflops(ms):
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    t_1cta = triton.testing.do_bench(lambda: _launch_kernel(a, b, matmul_1cta_kernel), warmup=500, rep=2000)
    t_1cta_ws = triton.testing.do_bench(lambda: _launch_kernel(a, b, matmul_1cta_ws_kernel, use_ws_env=True),
                                        warmup=500, rep=2000)
    t_2cta = triton.testing.do_bench(lambda: _launch_kernel(a, b, matmul_2cta_kernel, ctas_per_cga=(2, 1, 1)),
                                     warmup=500, rep=2000)
    t_2cta_ws = triton.testing.do_bench(
        lambda: _launch_kernel(a, b, matmul_2cta_ws_kernel, ctas_per_cga=(2, 1, 1), use_ws_env=True), warmup=500,
        rep=2000)

    print(f"\n  ({M}x{N}x{K})"
          f"  1CTA: {tflops(t_1cta):.0f}"
          f"  |  1CTA+WS: {tflops(t_1cta_ws):.0f}"
          f"  |  2CTA: {tflops(t_2cta):.0f}"
          f"  |  2CTA+WS: {tflops(t_2cta_ws):.0f} TFLOPS")
