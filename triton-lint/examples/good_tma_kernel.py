"""
Example: Good kernel that SHOULD use TMA on Hopper.

This kernel uses block pointers which are the primary target for TMA.
When compiled for Hopper with optimizations, Triton should automatically
insert TMA operations.
"""

import triton
import triton.language as tl


@triton.jit
def matmul_kernel_block_ptr(
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
    """
    Matrix multiplication using block pointers (TMA-eligible).

    This should automatically use TMA on Hopper GPUs.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create block pointers for A and B
    # These SHOULD trigger TMA on Hopper
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))

    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop - loads should use TMA
    for k in range(0, K, BLOCK_K):
        # These loads should become TMA operations
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)

        # Matrix multiplication
        acc += tl.dot(a, b)

        # Advance block pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # Store result using block pointer (should also use TMA)
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N),
                                    order=(1, 0))

    tl.store(c_block_ptr, acc)


@triton.jit
def matmul_with_tma_descriptors(
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
    """
    Matrix multiplication using TMA with on-device descriptors.

    This kernel uses tl.make_tensor_descriptor to create on-device TMA descriptors,
    then uses the descriptor's .load() and .store() methods for TMA operations.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create TMA descriptors directly using tl.make_tensor_descriptor
    # This creates on-device descriptors that map directly to TMA operations
    desc_a = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )

    desc_b = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop with explicit TMA descriptor loads
    for k in range(0, K, BLOCK_K):
        # Load using TMA descriptors - descriptor.load() method
        # These will generate triton_nvidia_gpu.async_tma_copy_global_to_local
        a = desc_a.load([pid_m * BLOCK_M, k])
        b = desc_b.load([k, pid_n * BLOCK_N])

        # Matrix multiplication
        acc += tl.dot(a, b)
        acc = tl.sin(acc)

    # Create output TMA descriptor
    desc_c = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    # Store using TMA descriptor - descriptor.store() method
    # This will generate triton_nvidia_gpu.async_tma_copy_local_to_global
    desc_c.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc)


# Test functions to run the kernels
def test_matmul_kernel_block_ptr(verbose=False):
    """Run matmul_kernel_block_ptr to populate cache for TMA verification."""
    try:
        import torch
    except ImportError:
        if verbose:
            print("PyTorch not available - skipping test")
        return False

    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available - skipping test")
        return False

    if verbose:
        print("Running matmul_kernel_block_ptr...")

    try:
        # Create test tensors
        M, N, K = 256, 256, 256
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64

        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        c = torch.empty(M, N, device='cuda', dtype=torch.float32)

        # Define grid
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

        # Run kernel
        matmul_kernel_block_ptr[grid](
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
        )

        torch.cuda.synchronize()

        # Verify result
        expected = torch.matmul(a, b)
        torch.testing.assert_close(c, expected, rtol=1e-2, atol=1e-2)

        if verbose:
            print(f"  ✓ Kernel executed successfully and result is correct")
        return True
    except Exception as e:
        if verbose:
            print(f"  ✗ Kernel failed: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
        return False


def test_matmul_with_tma_descriptors(verbose=False):
    """Run matmul_with_tma_descriptors to test explicit TMA with tl.make_tensor_descriptor."""
    try:
        import torch
    except ImportError:
        if verbose:
            print("PyTorch not available - skipping test")
        return False

    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available - skipping test")
        return False

    if verbose:
        print("Running matmul_with_tma_descriptors...")

    try:
        # Set up allocator for TMA descriptor memory
        # TMA descriptors require runtime memory allocation
        triton.set_allocator(lambda size, align, stream: torch.cuda.caching_allocator_alloc(size, stream))

        # Create test tensors
        # Note: Using smaller block sizes to avoid shared memory limit
        M, N, K = 256, 256, 256
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        c = torch.empty(M, N, device='cuda', dtype=torch.float32)

        # Define grid
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

        # Run kernel with TMA descriptors
        matmul_with_tma_descriptors[grid](
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
        )

        torch.cuda.synchronize()

        # Verify result
        expected = torch.matmul(a, b)
        torch.testing.assert_close(c, expected, rtol=1e-2, atol=1e-2)

        if verbose:
            print(f"  ✓ Kernel with tl.make_tensor_descriptor executed successfully")
            print(f"  ℹ️  Uses descriptor.load() and descriptor.store() methods")
        return True
    except Exception as e:
        if verbose:
            print(f"  ✗ Kernel failed: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
        return False


def test_optimized_matmul(verbose=False):
    """Run optimized_matmul (with autotune) to populate cache for TMA verification."""
    try:
        import torch
    except ImportError:
        if verbose:
            print("PyTorch not available - skipping test")
        return False

    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available - skipping test")
        return False

    if verbose:
        print("Running optimized_matmul (with autotune)...")

    try:
        # Create test tensors
        M, N, K = 256, 256, 256

        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        c = torch.empty(M, N, device='cuda', dtype=torch.float32)

        # The autotune decorator will handle grid and BLOCK_* parameters
        # We just need to pass the data
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )

        # Run kernel - autotune will try different configurations
        optimized_matmul[grid](
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
        )

        torch.cuda.synchronize()

        # Verify result
        expected = torch.matmul(a, b)
        torch.testing.assert_close(c, expected, rtol=1e-2, atol=1e-2)

        if verbose:
            print(f"  ✓ Kernel executed successfully and result is correct")
        return True
    except Exception as e:
        if verbose:
            print(f"  ✗ Kernel failed: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing good_tma_kernel.py")
    print("=" * 60)
    print()

    # Test all kernels with verbose output when run directly
    success1 = test_matmul_kernel_block_ptr(verbose=True)
    print()
    success2 = test_matmul_with_tma_descriptors(verbose=True)
    print()
    success3 = test_optimized_matmul(verbose=True)

    print()
    print("=" * 60)
    if success1 and success2 and success3:
        print("✅ All kernels passed!")
    else:
        print("❌ Some kernels failed")
    print()
    print("Kernels compiled and cached!")
    print()
    print("TMA Approaches Demonstrated:")
    print("  1. matmul_kernel_block_ptr - Block pointers (implicit TMA)")
    print("  2. matmul_with_tma_descriptors - tl.make_tensor_descriptor (explicit TMA)")
    print("  3. optimized_matmul - Autotuned with block pointers")
    print()
    print("Verify TMA with:")
    print("  python3 triton_lint_simple.py --verify-tma examples/good_tma_kernel.py")
    print("=" * 60)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def optimized_matmul(
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
    """
    Well-written matmul with autotune and block pointers.

    This is the pattern Triton TMA passes are designed for.
    """
    # Same implementation as above
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))

    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N),
                                    order=(1, 0))

    tl.store(c_block_ptr, acc)
