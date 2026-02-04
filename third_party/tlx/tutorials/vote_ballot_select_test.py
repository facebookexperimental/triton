"""
Test case extracted from blackwell-fa-ws-pipelined-persistent_test.py
demonstrating vote_ballot_sync usage to guard conditional computation.

This pattern is used in Flash Attention for conditional rescaling:
- Use vote_ballot_sync to check if ANY thread in the warp needs rescaling
- If the ballot result is zero (no thread needs rescaling), skip the computation
- This avoids computing the scaled value when all threads have alpha >= 1.0
"""

import torch
import triton
import triton.language as tl

try:
    import triton.language.extra.tlx as tlx

    HAS_TLX = True
except ImportError:
    HAS_TLX = False


@triton.jit
def _mul_f32x2(a, b):
    """Multiply two f32 values element-wise (simulating f32x2 packed ops)."""
    return a * b


@triton.jit
def vote_ballot_select_kernel(
    # Pointers
    acc_ptr,
    alpha_ptr,
    out_ptr,
    # Shape
    M: tl.constexpr,
    N: tl.constexpr,
    # Strides
    stride_acc_m,
    stride_acc_n,
    stride_alpha_m,
    stride_out_m,
    stride_out_n,
    # Config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RESCALE_OPT: tl.constexpr,
):
    """
    Kernel demonstrating vote_ballot_sync pattern for conditional rescaling.

    For each block:
    1. Load alpha values (shape: BLOCK_M x 1)
    2. Use vote_ballot_sync to check if any alpha < 1.0
    3. Conditionally scale acc values based on ballot result

    This pattern enables branch optimization when all threads in a warp
    have alpha >= 1.0 (ballot result is 0), skipping the multiplication.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create pointers
    acc_ptrs = acc_ptr + offs_m[:, None] * stride_acc_m + offs_n[None, :] * stride_acc_n
    alpha_ptrs = alpha_ptr + offs_m * stride_alpha_m
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n

    # Masks for boundary checking
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load accumulator (BLOCK_M x BLOCK_N)
    acc = tl.load(acc_ptrs, mask=mask, other=0.0)

    # Load alpha (BLOCK_M x 1) - broadcast to BLOCK_M x BLOCK_N for operations
    alpha_1 = tl.load(alpha_ptrs, mask=mask_m, other=1.0)[:, None]

    if RESCALE_OPT:
        # Key pattern: Use vote_ballot_sync to check if ANY thread needs rescaling
        #
        # pred: tensor<BLOCK_M x 1 x i1> - True where alpha < 1.0
        # ballot_result: tensor<BLOCK_M x 1 x i32> - Warp ballot result
        #   - All elements contain the same warp-level ballot value
        #   - Non-zero means at least one thread has alpha_1 < 1.0
        # should_rescale: tensor<BLOCK_M x 1 x i1> - True if any rescaling needed
        pred = alpha_1 < 1.0
        ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
        should_rescale = ballot_result != 0

        # Conditional scaling using tl.where
        # When should_rescale is False (ballot_result == 0), skip multiplication
        scaled_acc = _mul_f32x2(acc, alpha_1)
        acc = tl.where(should_rescale, scaled_acc, acc)
    else:
        # Always rescale when optimization is disabled
        acc = _mul_f32x2(acc, alpha_1)

    # Store result
    tl.store(out_ptrs, acc, mask=mask)


@triton.jit
def vote_ballot_select_tmem_kernel(
    # Pointers
    acc_ptr,
    alpha_ptr,
    out_ptr,
    # Shape
    M: tl.constexpr,
    N: tl.constexpr,
    # Strides
    stride_acc_m,
    stride_acc_n,
    stride_alpha_m,
    stride_out_m,
    stride_out_n,
    # Config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RESCALE_OPT: tl.constexpr,
):
    """
    Extended kernel demonstrating vote_ballot_sync with TMEM operations.

    This kernel simulates the pattern from Flash Attention where:
    1. Data is loaded from global memory to TMEM (tensor memory)
    2. vote_ballot_sync determines if rescaling is needed
    3. Conditionally perform tmem_load -> compute -> tmem_store

    The goal is to convert the tl.where into an if-branch that guards
    the entire tmem_load + computation + tmem_store sequence.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create pointers
    acc_ptrs = acc_ptr + offs_m[:, None] * stride_acc_m + offs_n[None, :] * stride_acc_n
    alpha_ptrs = alpha_ptr + offs_m * stride_alpha_m
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n

    # Masks for boundary checking
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load accumulator (BLOCK_M x BLOCK_N)
    acc = tl.load(acc_ptrs, mask=mask, other=0.0)

    # Load alpha (BLOCK_M x 1)
    alpha_1 = tl.load(alpha_ptrs, mask=mask_m, other=1.0)[:, None]

    if RESCALE_OPT:
        # Pattern from FA: vote_ballot to check rescaling need
        pred = alpha_1 < 1.0
        ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
        should_rescale = ballot_result != 0

        # DESIRED OPTIMIZATION:
        # Convert this tl.where into an if-branch at LLVM level:
        #
        # Current lowering:
        #   scaled_acc = acc * alpha_1  // Always computed
        #   result = select(should_rescale, scaled_acc, acc)
        #
        # Desired lowering (when should_rescale is warp-uniform):
        #   if (any_thread_in_warp(should_rescale)) {
        #       result = acc * alpha_1
        #   } else {
        #       result = acc
        #   }
        #
        # Benefits:
        # - When all alpha >= 1.0 in a warp, skip multiplication entirely
        # - No warp divergence since ballot result is uniform across warp

        scaled_acc = _mul_f32x2(acc, alpha_1)
        acc = tl.where(should_rescale, scaled_acc, acc)
    else:
        acc = _mul_f32x2(acc, alpha_1)

    # Store result
    tl.store(out_ptrs, acc, mask=mask)


def test_vote_ballot_select():
    """Test the vote_ballot_sync + select pattern."""
    if not HAS_TLX:
        print("SKIP: tlx not available")
        return

    torch.manual_seed(42)

    # Test dimensions
    M, N = 128, 64
    BLOCK_M, BLOCK_N = 32, 32

    # Create test data
    acc = torch.randn(M, N, dtype=torch.float32, device="cuda")
    # Mix of alpha values: some < 1.0 (need rescaling), some >= 1.0 (no rescaling)
    alpha = torch.ones(M, dtype=torch.float32, device="cuda")
    # Set some values < 1.0 to trigger rescaling in some warps
    alpha[:M // 4] = 0.5  # First quarter needs rescaling
    alpha[M // 2:3 * M // 4] = 0.8  # Third quarter needs rescaling

    out_opt = torch.empty_like(acc)
    out_ref = torch.empty_like(acc)

    # Grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Run with RESCALE_OPT=True (uses vote_ballot_sync)
    vote_ballot_select_kernel[grid](
        acc,
        alpha,
        out_opt,
        M,
        N,
        acc.stride(0),
        acc.stride(1),
        alpha.stride(0),
        out_opt.stride(0),
        out_opt.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        RESCALE_OPT=True,
    )

    # Run with RESCALE_OPT=False (reference, always rescales)
    vote_ballot_select_kernel[grid](
        acc,
        alpha,
        out_ref,
        M,
        N,
        acc.stride(0),
        acc.stride(1),
        alpha.stride(0),
        out_ref.stride(0),
        out_ref.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        RESCALE_OPT=False,
    )

    # Verify results match
    torch.testing.assert_close(out_opt, out_ref, rtol=1e-5, atol=1e-5)
    print("PASS: vote_ballot_select_kernel correctness verified")

    # Test edge cases
    # Case 1: All alpha >= 1.0 (no rescaling needed)
    alpha_no_rescale = torch.ones(M, dtype=torch.float32, device="cuda")
    out_no_rescale = torch.empty_like(acc)
    vote_ballot_select_kernel[grid](
        acc,
        alpha_no_rescale,
        out_no_rescale,
        M,
        N,
        acc.stride(0),
        acc.stride(1),
        alpha_no_rescale.stride(0),
        out_no_rescale.stride(0),
        out_no_rescale.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        RESCALE_OPT=True,
    )
    # When alpha=1.0, output should equal input (no rescaling)
    torch.testing.assert_close(out_no_rescale, acc, rtol=1e-5, atol=1e-5)
    print("PASS: No rescaling case (all alpha >= 1.0)")

    # Case 2: All alpha < 1.0 (all need rescaling)
    alpha_all_rescale = torch.full((M, ), 0.5, dtype=torch.float32, device="cuda")
    out_all_rescale = torch.empty_like(acc)
    vote_ballot_select_kernel[grid](
        acc,
        alpha_all_rescale,
        out_all_rescale,
        M,
        N,
        acc.stride(0),
        acc.stride(1),
        alpha_all_rescale.stride(0),
        out_all_rescale.stride(0),
        out_all_rescale.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        RESCALE_OPT=True,
    )
    expected = acc * alpha_all_rescale[:, None]
    torch.testing.assert_close(out_all_rescale, expected, rtol=1e-5, atol=1e-5)
    print("PASS: All rescaling case (all alpha < 1.0)")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_vote_ballot_select()
