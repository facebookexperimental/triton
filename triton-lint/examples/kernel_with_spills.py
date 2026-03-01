# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Example Triton kernel configured to demonstrate spill analysis.

**IMPORTANT NOTE:**
Modern GPUs (especially NVIDIA Hopper/SM90) have MASSIVE register files:
- 256KB of registers per SM
- 65,536 32-bit registers total
- Triton compiler is excellent at optimization

As a result, most reasonable kernels do NOT spill on modern hardware.
This is GOOD NEWS for performance!

Register spills typically only occur when:
1. Extremely large tile sizes (BLOCK_M/N > 256)
2. Very high NUM_STAGES (> 7) for pipelining
3. Complex nested loops with many live values
4. Targeting older GPUs (SM70/Volta, SM80/Ampere)

This example shows proper usage of the spill analyzer. If you see
"No register spills detected" on Hopper, your kernel is well-optimized!

To see actual spills, you would need to:
- Use BLOCK_M=256, BLOCK_N=256 (very large tiles)
- Use NUM_STAGES=10 (extreme pipelining)
- Target older architecture: target="cuda:80" (Ampere)
"""

import triton
import triton.language as tl
import torch
"""
A test to show triton always report >0 register spills for sin/cos.
Output:
    n_regs 24, n_spills 8
"""


@triton.jit
def sin_kernel(in_ptr, out_ptr, numel, XBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < numel
    t0 = tl.load(in_ptr + xindex, xmask)
    t1 = tl.sin(t0)
    tl.store(out_ptr + xindex, t1, xmask)


def test_sin_kernel():
    n_elements = 128 * 1024 * 1024
    x = torch.randn(n_elements).cuda()
    y = torch.empty(n_elements).cuda()
    grid = lambda meta: (triton.cdiv(n_elements, meta['XBLOCK']), )
    sin_kernel[grid](x, y, n_elements, XBLOCK=1024)
    assert torch.allclose(x.sin(), y)


@triton.jit
def matmul_with_spills(
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
    NUM_STAGES: tl.constexpr,
):
    """
    Matrix multiplication with aggressive tiling to cause register spills.

    Using large BLOCK_M, BLOCK_N and high NUM_STAGES will cause spills
    on most hardware.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create tile offsets - using large tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Multiple accumulators - keep many values live simultaneously
    # This dramatically increases register pressure
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop with pipelining - NUM_STAGES increases register pressure
    for k in range(0, K, BLOCK_K):
        # Load tiles - convert to fp32 immediately for consistency
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)

        # Convert to fp32 to match accumulator type
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # Keep many transformed versions live - forces register pressure
        # Each transformation creates new live values
        a_scaled1 = a * 1.1
        a_scaled2 = a * 0.9
        a_scaled3 = a * 1.05
        a_scaled4 = a * 0.95

        b_scaled1 = b * 1.1
        b_scaled2 = b * 0.9
        b_scaled3 = b * 1.05
        b_scaled4 = b * 0.95

        # Additional transformations with math operations
        # These create even more live values that need registers
        a_sqrt = tl.sqrt(tl.abs(a))
        a_exp = tl.exp(a * 0.01)
        a_log = tl.log(tl.abs(a) + 1.0)

        b_sqrt = tl.sqrt(tl.abs(b))
        b_exp = tl.exp(b * 0.01)
        b_log = tl.log(tl.abs(b) + 1.0)

        # Multiple accumulations with different transformations
        # All these intermediate values need to be kept in registers
        # This is the KEY to creating register pressure: many accumulators
        acc1 += tl.dot(a, b)
        acc2 += tl.dot(a_scaled1, b_scaled1)
        acc3 += tl.dot(a_scaled2, b_scaled2)
        acc4 += tl.dot(a_scaled3, b_scaled3)
        acc5 += tl.dot(a_scaled4, b_scaled4)

        # Use math-transformed versions
        acc6 += tl.dot(a_sqrt, b_sqrt) * 0.1
        acc1 += tl.dot(a_exp, b_exp) * 0.01
        acc2 += tl.dot(a_log, b_log) * 0.05

        # Cross products to keep more values live
        acc3 += tl.dot(a_scaled1, b_scaled2) * 0.5
        acc4 += tl.dot(a_scaled2, b_scaled1) * 0.5
        acc5 += tl.dot(a_sqrt, b_exp) * 0.05
        acc6 += tl.dot(a_exp, b_sqrt) * 0.05

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Combine all accumulators - keeps all live until the end
    # This is the key: all acc1-6 must be kept in registers throughout
    final = (acc1 * 0.25 + acc2 * 0.20 + acc3 * 0.20 + acc4 * 0.15 + acc5 * 0.10 + acc6 * 0.10)

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, final, mask=c_mask)


def test_matmul_with_spills():
    """
    Test function to compile the kernel with settings that cause spills.

    Using BLOCK_M=128, BLOCK_N=128 with NUM_STAGES=5 should cause
    register spills on most GPUs.
    """
    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    # Aggressive settings to force spills:
    # - Large tile sizes (128x128)
    # - High pipelining (NUM_STAGES=5)
    # This combination requires many registers for:
    # - Accumulator: 128*128 = 16K floats
    # - Pipelined loads: multiple stages of A and B tiles
    matmul_with_spills[grid](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M=128,
        BLOCK_N=128, BLOCK_K=32, NUM_STAGES=5,  # High pipelining increases register pressure
    )

    return c


if __name__ == "__main__":
    # This will compile the kernel when the module is executed
    result = test_matmul_with_spills()
    print(f"Kernel executed successfully. Output shape: {result.shape}")
    print(f"Note: Check for spills with: python3 triton_lint_simple.py --analyze-spills {__file__}")
