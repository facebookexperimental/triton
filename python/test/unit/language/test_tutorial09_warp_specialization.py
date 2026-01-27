"""
Explicit unit tests for all warp-specialized variations of Tutorial 09 (Persistent Matmul).

These tests validate the warp specialization feature for persistent matmul kernels
with both Flatten=True and Flatten=False configurations. Tests are restricted to
Blackwell GPUs only.
"""
import torch
import pytest
import triton
import triton.language as tl

from triton._internal_testing import is_blackwell
from triton.tools.tensor_descriptor import TensorDescriptor


# Helper function from tutorial 09
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# ============================================================================
# Kernel 1: matmul_kernel_tma - TMA-based matmul with warp specialization
# This kernel uses warp_specialize in the K-loop (inner loop)
# ============================================================================
@triton.jit
def matmul_kernel_tma_ws(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """TMA-based matmul with warp specialization in K-loop (always enabled)."""
    dtype = tl.float16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Always use warp_specialize=True
    for k in tl.range(k_tiles, warp_specialize=True, disallow_acc_multi_buffer=True):
        offs_k = k * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], c)


# ============================================================================
# Kernel 2: matmul_kernel_tma_persistent - Persistent TMA matmul with warp spec
# This kernel uses warp_specialize in the outer tile loop with flatten parameter
# ============================================================================
@triton.jit
def matmul_kernel_tma_persistent_ws(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    """Persistent TMA matmul with warp specialization (always enabled)."""
    dtype = tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Always use warp_specialize=True with configurable flatten
    for tile_id in tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            flatten=FLATTEN,
            warp_specialize=True,
            disallow_acc_multi_buffer=True,
    ):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)


# ============================================================================
# Kernel 3: matmul_kernel_descriptor_persistent - Device-side TMA descriptors
# Uses warp_specialize with flatten in outer tile loop
# ============================================================================
@triton.jit
def matmul_kernel_descriptor_persistent_ws(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    """Persistent matmul with device-side TMA descriptors and warp specialization (always enabled)."""
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2],
    )

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Always use warp_specialize=True with configurable flatten
    for tile_id in tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            flatten=FLATTEN,
            warp_specialize=True,
            disallow_acc_multi_buffer=True,
    ):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


# ============================================================================
# Test 1: matmul_kernel_tma warp specialization (K-loop based)
# ============================================================================
@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (512, 512, 256), (1024, 1024, 512)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64, 128])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_matmul_tma_warp_specialize(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages,
                                               num_warps):
    """Test matmul_kernel_tma with warp_specialize=True (K-loop based)."""
    # Skip configurations that exceed hardware resource limits
    if BLOCK_SIZE_N == 256 and num_stages == 3 and num_warps == 8:
        pytest.skip(
            "Out of resources: shared memory and tensor memory exceeded for BLOCK_SIZE_N=256, num_stages=3, num_warps=8"
        )
    if num_warps == 4:
        pytest.skip("Fails in AutoWS. TODO: Support")

    # Use scope() to set use_meta_ws and automatically restore on exit
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        device = "cuda"

        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=dtype, device=device)
        B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        # Set up tensor descriptors
        a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N])

        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

        kernel = matmul_kernel_tma_ws[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        # Verify IR contains warp_specialize
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        # Verify correctness
        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Test 2: matmul_kernel_tma_persistent warp specialization (tile-loop based)
# Tests both Flatten=True and Flatten=False
# ============================================================================
@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (512, 512, 256), (1024, 1024, 512)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64, 128])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("FLATTEN", [True, False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_matmul_tma_persistent_warp_specialize(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages,
                                                          num_warps, FLATTEN):
    """Test matmul_kernel_tma_persistent with warp_specialize=True for both Flatten values."""
    # Skip configurations that exceed hardware resource limits
    if BLOCK_SIZE_N == 256 and num_stages == 3:
        pytest.skip("Out of resources: tensor memory exceeded for BLOCK_SIZE_N=256, num_stages=3")

    # Skip FLATTEN=False - not yet supported in AutoWS
    if not FLATTEN:
        pytest.skip("Fails in AutoWS. TODO: Support")

    # EPILOGUE_SUBTILE only works with flatten=True
    EPILOGUE_SUBTILE = False

    # Use scope() to set use_meta_ws and automatically restore on exit
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=dtype, device=device)
        B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        # Set up tensor descriptors
        a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N])

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = matmul_kernel_tma_persistent_ws[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            FLATTEN=FLATTEN,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        # Verify IR contains warp_specialize
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        # Verify correctness
        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Test 3: matmul_kernel_descriptor_persistent warp specialization (device-side TMA)
# Tests both Flatten=True and Flatten=False
# ============================================================================
@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (512, 512, 256), (1024, 1024, 512)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64, 128])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("FLATTEN", [True, False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_matmul_descriptor_persistent_warp_specialize(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                                                 num_stages, num_warps, FLATTEN):
    """Test matmul_kernel_descriptor_persistent with warp_specialize=True for both Flatten values."""
    # Skip configurations that exceed hardware resource limits
    if BLOCK_SIZE_N == 256 and num_stages == 3:
        pytest.skip("Out of resources: tensor memory exceeded for BLOCK_SIZE_N=256, num_stages=3")

    # Skip FLATTEN=False - not yet supported in AutoWS
    if not FLATTEN:
        pytest.skip("Fails in AutoWS. TODO: Support")

    # EPILOGUE_SUBTILE only works with flatten=True
    EPILOGUE_SUBTILE = False

    # Use scope() to set use_meta_ws and automatically restore on exit
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        # For device-side TMA descriptors, we need contiguous memory layout
        A = torch.randn((M, K), dtype=dtype, device=device).contiguous()
        B = torch.randn((N, K), dtype=dtype, device=device).contiguous()
        C = torch.empty((M, N), dtype=dtype, device=device).contiguous()

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = matmul_kernel_descriptor_persistent_ws[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            FLATTEN=FLATTEN,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        # Verify IR contains warp_specialize
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        # Verify correctness
        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Test 4: matmul_kernel_tma_persistent with EPILOGUE_SUBTILE (only with Flatten=True)
# ============================================================================
@pytest.mark.parametrize("M, N, K", [(256, 256, 256), (1024, 1024, 512)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [256])  # EPILOGUE_SUBTILE needs BLOCK_SIZE_N >= 2
@pytest.mark.parametrize("BLOCK_SIZE_K", [64, 128])
@pytest.mark.parametrize("num_stages", [2])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_matmul_tma_persistent_epilogue_subtile(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                                           num_stages, num_warps):
    """Test matmul_kernel_tma_persistent with EPILOGUE_SUBTILE=True (requires Flatten=True)."""
    # EPILOGUE_SUBTILE requires Flatten=True
    FLATTEN = True
    EPILOGUE_SUBTILE = True

    # Use scope() to set use_meta_ws and automatically restore on exit
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=dtype, device=device)
        B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        # Set up tensor descriptors - note the special block shape for epilogue subtiling
        a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = matmul_kernel_tma_persistent_ws[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            FLATTEN=FLATTEN,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        # Verify IR contains warp_specialize
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"

        # Verify correctness
        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)
