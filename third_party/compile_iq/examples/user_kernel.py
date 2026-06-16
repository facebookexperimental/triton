"""A normal Triton kernel + run — NO compile_iq / CompileIQ imports or references.

This is exactly what a user would write. It demonstrates Stage-1 collection's
**zero user surface**: nothing in this file is aware of compile_iq. Collection is
triggered entirely by the environment, with no change to this code:

    FBTRITON_COMPILE_IQ_COLLECT=1 \
    COMPILE_IQ_TASK_DIR=/tmp/ciq_tasks \
    TRITON_PTXAS_BLACKWELL_PATH=/path/to/ptxas13.3 \
    python user_kernel.py

After the run, a compileIQ task appears under $COMPILE_IQ_TASK_DIR.
"""

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


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
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M = BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BLOCK_M, BLOCK_N, BLOCK_K)
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn((2048, 2048), device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn((2048, 2048), device=DEVICE, dtype=torch.bfloat16)
    c = matmul(a, b)
    torch.cuda.synchronize()
    print("matmul ran:", tuple(c.shape), c.dtype)
