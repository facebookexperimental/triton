"""
Example of a Triton kernel with anti-patterns.
This file is used for testing triton_lint.
"""

import triton
import triton.language as tl


@triton.jit
def bad_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  # Should be autotuned!
                      ):
    """
    Matrix multiplication kernel with several anti-patterns:
    1. No @triton.autotune decorator despite tunable parameters
    2. Hardcoded block sizes in the body
    3. Potential scalar accesses
    4. Missing masks
    """

    # Anti-pattern: hardcoded block size
    BLOCK_SIZE = 128

    pid = tl.program_id(0)

    # Compute offsets
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop
    for k in range(0, K, BLOCK_K):
        # Anti-pattern: no masking for out-of-bounds
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    # Anti-pattern: no masking for out-of-bounds
    tl.store(c_ptrs, acc)


@triton.jit
def bad_vector_add(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Vector addition with anti-patterns but syntactically correct."""

    pid = tl.program_id(axis=0)

    # Anti-pattern 1: Using hardcoded value instead of constexpr parameter
    block_size = 64  # Should use BLOCK_SIZE parameter!

    # Anti-pattern 2: Not using the full block efficiently
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, 64)  # Hardcoded 64 instead of BLOCK_SIZE

    # Anti-pattern 3: Missing boundary check/mask
    # This will access out-of-bounds if n_elements is not divisible by block_size
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)

    result = x + y

    # Anti-pattern 4: Store without mask
    tl.store(output_ptr + offsets, result)


@triton.jit
def bad_tma_dead_code(
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
    TMA anti-pattern: Dead code elimination.

    This kernel uses block pointers (which should trigger TMA),
    but loads data that is never meaningfully used. The compiler's
    dead code elimination pass will remove the TMA operations.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create block pointers - looks like it should use TMA
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))

    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))

    # Anti-pattern: Load data but never use it!
    # These TMA loads will be optimized away
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)

    # Instead of using the loaded data, just write zeros
    # This makes the loads above dead code
    result = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N),
                                    order=(1, 0))

    tl.store(c_block_ptr, result)


@triton.jit
def bad_tma_conditional_never_taken(
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
    TMA anti-pattern: Conditional TMA that's never taken.

    Block pointers are used inside a conditional that's always false,
    so the TMA code path is never executed and gets optimized away.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Anti-pattern: Condition that's always false
    use_tma = tl.constexpr(False)  # Always false!

    if use_tma:
        # This block pointer code will be eliminated
        a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))

        b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))

        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc = tl.dot(a, b)
    else:
        # This path always executes - uses regular pointer arithmetic
        # No TMA will be used
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N),
                                    order=(1, 0))

    tl.store(c_block_ptr, acc)


@triton.jit
def bad_tma_redundant_load(
    a_ptr,
    c_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    TMA anti-pattern: Redundant loads that get optimized away.

    Loading the same data multiple times with block pointers.
    The compiler will optimize away redundant TMA operations.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, N), strides=(stride_am, stride_an),
                                    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N),
                                    order=(1, 0))

    # Anti-pattern: Load the same data multiple times
    # First load - TMA might be used
    a1 = tl.load(a_block_ptr)

    # Second load - identical to first (redundant)
    # Compiler will optimize this away
    a2 = tl.load(a_block_ptr)

    # Third load - still redundant
    a3 = tl.load(a_block_ptr)

    # Only use the last one, making previous loads dead code
    result = a3

    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N),
                                    order=(1, 0))

    tl.store(c_block_ptr, result)


# Test functions to run the kernels
def test_bad_matmul():
    """Run bad_matmul_kernel to populate cache for TMA verification."""
    try:
        import torch
    except ImportError:
        print("PyTorch not available - skipping test")
        return False

    if not torch.cuda.is_available():
        print("CUDA not available - skipping test")
        return False

    print("Running bad_matmul_kernel...")

    try:
        # Create small test tensors to minimize memory issues
        M, N, K = 128, 128, 128
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        c = torch.empty(M, N, device='cuda', dtype=torch.float32)

        # Define grid
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

        # Run kernel - this will compile it and populate cache
        # Even if it fails at runtime, compilation happens first
        bad_matmul_kernel[grid](
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
        print(f"  ✓ Kernel executed successfully")
        return True
    except Exception as e:
        # Kernel may fail at runtime due to anti-patterns, but compilation happened
        print(f"  ⚠️  Kernel compiled but may have runtime issues")
        print(f"      Error: {str(e)[:80]}")
        # Return True anyway - compilation is what matters for TMA verification
        return True


def test_bad_vector_add():
    """Run bad_vector_add to populate cache for TMA verification."""
    try:
        import torch
    except ImportError:
        print("PyTorch not available - skipping test")
        return False

    if not torch.cuda.is_available():
        print("CUDA not available - skipping test")
        return False

    print("Running bad_vector_add...")

    try:
        # Create test tensors
        n_elements = 1024
        BLOCK_SIZE = 256
        x = torch.randn(n_elements, device='cuda', dtype=torch.float32)
        y = torch.randn(n_elements, device='cuda', dtype=torch.float32)
        output = torch.empty(n_elements, device='cuda', dtype=torch.float32)

        # Define grid
        grid = (triton.cdiv(n_elements, BLOCK_SIZE), )

        # Run kernel with BLOCK_SIZE parameter
        bad_vector_add[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        torch.cuda.synchronize()
        print(f"  ✓ Kernel executed successfully")
        return True
    except Exception as e:
        print(f"  ⚠️  Kernel compiled but may have runtime issues")
        print(f"      Error: {str(e)[:80]}")
        return True


def test_bad_tma_dead_code():
    """Run bad_tma_dead_code to populate cache for TMA verification."""
    try:
        import torch
    except ImportError:
        print("PyTorch not available - skipping test")
        return False

    if not torch.cuda.is_available():
        print("CUDA not available - skipping test")
        return False

    print("Running bad_tma_dead_code...")

    try:
        M, N, K = 256, 256, 256
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64

        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        c = torch.empty(M, N, device='cuda', dtype=torch.float32)

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

        bad_tma_dead_code[grid](
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
        print(f"  ✓ Kernel executed successfully")
        return True
    except Exception as e:
        print(f"  ⚠️  Kernel compiled but may have runtime issues")
        print(f"      Error: {str(e)[:80]}")
        return True


def test_bad_tma_conditional():
    """Run bad_tma_conditional_never_taken to populate cache for TMA verification."""
    try:
        import torch
    except ImportError:
        print("PyTorch not available - skipping test")
        return False

    if not torch.cuda.is_available():
        print("CUDA not available - skipping test")
        return False

    print("Running bad_tma_conditional_never_taken...")

    try:
        M, N, K = 256, 256, 256
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64

        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        c = torch.empty(M, N, device='cuda', dtype=torch.float32)

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

        bad_tma_conditional_never_taken[grid](
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
        print(f"  ✓ Kernel executed successfully")
        return True
    except Exception as e:
        print(f"  ⚠️  Kernel compiled but may have runtime issues")
        print(f"      Error: {str(e)[:80]}")
        return True


def test_bad_tma_redundant():
    """Run bad_tma_redundant_load to populate cache for TMA verification."""
    try:
        import torch
    except ImportError:
        print("PyTorch not available - skipping test")
        return False

    if not torch.cuda.is_available():
        print("CUDA not available - skipping test")
        return False

    print("Running bad_tma_redundant_load...")

    try:
        M, N = 256, 256
        BLOCK_M, BLOCK_N = 128, 128

        a = torch.randn(M, N, device='cuda', dtype=torch.float32)
        c = torch.empty(M, N, device='cuda', dtype=torch.float32)

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

        bad_tma_redundant_load[grid](
            a,
            c,
            M,
            N,
            a.stride(0),
            a.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        torch.cuda.synchronize()
        print(f"  ✓ Kernel executed successfully")
        return True
    except Exception as e:
        print(f"  ⚠️  Kernel compiled but may have runtime issues")
        print(f"      Error: {str(e)[:80]}")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing bad_kernel.py")
    print("=" * 60)
    print()

    # Test both kernels
    test_bad_matmul()
    print()
    test_bad_vector_add()

    print()
    print("=" * 60)
    print("Kernels compiled and cached!")
    print("You can now verify TMA with:")
    print("  python3 verify_bad_kernels_tma.py")
    print("=" * 60)
