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
  1. Compilation succeeds with ctas_per_cga=(2, 1, 1)
  2. Correctness against PyTorch reference
"""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
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

    # Create TMA descriptors for A and B (full block shapes).
    # The compiler's Transform2CTALoads pass will clone B's descriptor
    # with half-width block shape (BLOCK_K x BLOCK_N//2) and add
    # CTA-based offset so each CTA loads its half.
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
        # two_ctas=True tells the compiler to generate cta_group::2 MMA.
        # Transform2CTALoads handles B splitting; Insert2CTASync adds barriers.
        accumulator = tl.dot(a, b, accumulator, two_ctas=True)

    # Store result
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

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

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

    # fp16 accumulation tolerance
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


def test_matmul_2cta_compilation():
    """Test that the kernel compiles without errors."""
    torch.manual_seed(42)
    M, N, K = 128, 128, 64
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # This should not raise
    out = matmul_2cta(a, b)
    assert out.shape == (M, N)
    assert out.dtype == torch.float16
