"""auto-TMA promotion of multi-dim strided loads (non-WS).

1. ``test_auto_tma_2d_strided_numerics``: a 2D strided masked block load
   `x[:, :]` (row-major, contiguous innermost) is promoted by PromoteLoadToTMA
   to a host-side TMA descriptor and runs correctly on GPU. Exercises the
   decomposePointer extensions (addptr chain walk, stride-after-expand_dims,
   dense-zero OOB fill, broadcast-wrapped mask cmp).
2. ``test_auto_tma_gemm_nows``: auto-TMA on both DOT operands of a GEMM, without
   warp specialization -- isolates the dot-operand promotion path (NVMMAShared
   swizzle read from the final descriptor encoding).

Requires sm_90+ (auto-TMA gating). Validated on Hopper (sm_90) and Blackwell
(sm_100).
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_cuda


def _is_sm90plus() -> bool:
    if not is_cuda():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9  # Hopper (sm_90) and Blackwell (sm_100)


# ---------------------------------------------------------------------------
# 2D strided promotion + numerics
# ---------------------------------------------------------------------------
@triton.jit
def _scale_2d_strided(x_ptr, out_ptr, M, N, stride_xm, stride_om, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Row-major x: innermost (offs_n) is contiguous -> auto-TMA eligible.
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :], mask=mask, other=0.0)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], x * 2.0, mask=mask)


@pytest.mark.skipif(not _is_sm90plus(), reason="auto-TMA promotion requires sm_90+")
def test_auto_tma_2d_strided_numerics():
    M, N = 512, 384
    BLOCK_M, BLOCK_N = 64, 64
    dtype = torch.float16
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=dtype, device="cuda")
    out = torch.empty((M, N), dtype=dtype, device="cuda")

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.auto_tma = True
        kernel = _scale_2d_strided[grid](x, out, M, N, x.stride(0), out.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    assert "tt.descriptor_load" in kernel.asm["ttir"], \
        "expected the 2D strided masked load to be auto-TMA promoted"
    torch.testing.assert_close(out, x * 2.0)


@triton.jit
def _gemm_nows(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bn, stride_cm, BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in range(tl.cdiv(K, BLOCK_K)):
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc = tl.dot(a, b.T, acc)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :], acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@pytest.mark.skipif(not _is_sm90plus(), reason="auto-TMA promotion requires sm_90+")
def test_auto_tma_gemm_nows():
    """Isolates the auto-TMA DOT-operand path WITHOUT warp specialization."""
    M, N, K = 256, 256, 256
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    dtype = torch.float16
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=dtype, device="cuda")
    B = torch.randn((N, K), dtype=dtype, device="cuda")
    C = torch.empty((M, N), dtype=dtype, device="cuda")

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.auto_tma = True
        kernel = _gemm_nows[grid](A, B, C, M, N, K, A.stride(0), B.stride(0), C.stride(0), BLOCK_M=BLOCK_M,
                                  BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

    assert "tt.descriptor_load" in kernel.asm["ttir"], "expected auto-TMA promotion"
    ref = (A.to(torch.float32) @ B.T.to(torch.float32)).to(dtype)
    torch.testing.assert_close(ref, C, atol=0.05, rtol=0.05)
