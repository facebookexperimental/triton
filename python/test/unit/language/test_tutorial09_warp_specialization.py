"""
Explicit unit tests for all warp-specialized variations of Tutorial 09 (Persistent Matmul).

These tests validate the warp specialization feature for persistent matmul kernels
with both Flatten=True and Flatten=False configurations. Tests are restricted to
Blackwell GPUs only.
"""

import pytest
import torch
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
    A_COL_MAJOR: tl.constexpr,
    B_COL_MAJOR: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,
    SMEM_ALLOC_ALGO: tl.constexpr,
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
    for k in tl.range(
            k_tiles,
            warp_specialize=True,
            data_partition_factor=DATA_PARTITION_FACTOR,
            smem_alloc_algo=SMEM_ALLOC_ALGO,
            separate_epilogue_store=True,
    ):
        offs_k = k * BLOCK_SIZE_K
        if A_COL_MAJOR:
            a = a_desc.load([offs_k, offs_am]).T
        else:
            a = a_desc.load([offs_am, offs_k])
        if B_COL_MAJOR:
            b = b_desc.load([offs_k, offs_bn]).T
        else:
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
    A_COL_MAJOR: tl.constexpr,
    B_COL_MAJOR: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,
    SMEM_ALLOC_ALGO: tl.constexpr,
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
            data_partition_factor=DATA_PARTITION_FACTOR,
            smem_alloc_algo=SMEM_ALLOC_ALGO,
            separate_epilogue_store=True,
    ):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            if A_COL_MAJOR:
                a = a_desc.load([offs_k, offs_am]).T
            else:
                a = a_desc.load([offs_am, offs_k])
            if B_COL_MAJOR:
                b = b_desc.load([offs_k, offs_bn]).T
            else:
                b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE == 1:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)
        elif EPILOGUE_SUBTILE == 2:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        elif EPILOGUE_SUBTILE == 4:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            acc00, acc01 = tl.split(tl.permute(tl.reshape(acc0, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 4)), (0, 2, 1)))
            acc10, acc11 = tl.split(tl.permute(tl.reshape(acc1, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 4)), (0, 2, 1)))
            c00 = acc00.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c00)
            c01 = acc01.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 4], c01)
            c10 = acc10.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + 2 * (BLOCK_SIZE_N // 4)], c10)
            c11 = acc11.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + 3 * (BLOCK_SIZE_N // 4)], c11)


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
    A_COL_MAJOR: tl.constexpr,
    B_COL_MAJOR: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,
    SMEM_ALLOC_ALGO: tl.constexpr,
):
    """Persistent matmul with device-side TMA descriptors and warp specialization (always enabled)."""
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    if A_COL_MAJOR:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[K, M],
            strides=[M, 1],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_M],
        )
    else:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
    if B_COL_MAJOR:
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[K, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        )
    else:
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
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // EPILOGUE_SUBTILE,
        ],
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
            data_partition_factor=DATA_PARTITION_FACTOR,
            smem_alloc_algo=SMEM_ALLOC_ALGO,
            separate_epilogue_store=True,
    ):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            if A_COL_MAJOR:
                a = a_desc.load([offs_k, offs_am]).T
            else:
                a = a_desc.load([offs_am, offs_k])
            if B_COL_MAJOR:
                b = b_desc.load([offs_k, offs_bn]).T
            else:
                b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE == 1:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)
        elif EPILOGUE_SUBTILE == 2:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        elif EPILOGUE_SUBTILE == 4:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            acc00, acc01 = tl.split(tl.permute(tl.reshape(acc0, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 4)), (0, 2, 1)))
            acc10, acc11 = tl.split(tl.permute(tl.reshape(acc1, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 4)), (0, 2, 1)))
            c00 = acc00.to(dtype)
            c_desc.store([offs_cm, offs_cn], c00)
            c01 = acc01.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 4], c01)
            c10 = acc10.to(dtype)
            c_desc.store([offs_cm, offs_cn + 2 * (BLOCK_SIZE_N // 4)], c10)
            c11 = acc11.to(dtype)
            c_desc.store([offs_cm, offs_cn + 3 * (BLOCK_SIZE_N // 4)], c11)


# ============================================================================
# Test 1: matmul_kernel_tma warp specialization (K-loop based)
# ============================================================================
@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (512, 512, 256), (8192, 8192, 1024)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64, 128])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("A_col_major", [False, True])
@pytest.mark.parametrize("B_col_major", [False, True])
@pytest.mark.parametrize("use_early_tma_store_lowering", [True, False])
@pytest.mark.parametrize("DATA_PARTITION_FACTOR", [1, 2])
@pytest.mark.parametrize("SMEM_ALLOC_ALGO", [0, 1])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_matmul_tma_warp_specialize(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    num_stages,
    num_warps,
    A_col_major,
    B_col_major,
    use_early_tma_store_lowering,
    DATA_PARTITION_FACTOR,
    SMEM_ALLOC_ALGO,
):
    """Test matmul_kernel_tma with warp_specialize=True (K-loop based)."""
    # DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256
    if DATA_PARTITION_FACTOR != 1 and BLOCK_SIZE_M != 256:
        pytest.skip("DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256")

    if DATA_PARTITION_FACTOR == 1 and BLOCK_SIZE_M == 256 and BLOCK_SIZE_N == 256:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    # Skip configurations that exceed hardware resource limits
    if BLOCK_SIZE_N == 256 and BLOCK_SIZE_K == 128 and (num_stages == 3 or num_warps == 4):
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    # Use scope() to set use_meta_ws and automatically restore on exit
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        device = "cuda"

        torch.manual_seed(42)
        if A_col_major:
            A = torch.randn((K, M), dtype=dtype, device=device).t()
        else:
            A = torch.randn((M, K), dtype=dtype, device=device)
        if B_col_major:
            B = torch.randn((K, N), dtype=dtype, device=device).t()
        else:
            B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        # Set up tensor descriptors (swap dims for col-major so contiguous dim is last)
        if A_col_major:
            a_desc = TensorDescriptor(A, [K, M], [M, 1], [BLOCK_SIZE_K, BLOCK_SIZE_M])
        else:
            a_desc = TensorDescriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
        if B_col_major:
            b_desc = TensorDescriptor(B, [K, N], [N, 1], [BLOCK_SIZE_K, BLOCK_SIZE_N])
        else:
            b_desc = TensorDescriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
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
            A_COL_MAJOR=A_col_major,
            B_COL_MAJOR=B_col_major,
            DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
            SMEM_ALLOC_ALGO=SMEM_ALLOC_ALGO,
            num_stages=num_stages,
            num_warps=num_warps,
            early_tma_store_lowering=use_early_tma_store_lowering,
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
@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (512, 512, 256), (8192, 8192, 1024)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64, 128])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("FLATTEN", [True, False])
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [1, 2, 4])
@pytest.mark.parametrize("A_col_major", [False, True])
@pytest.mark.parametrize("B_col_major", [False, True])
@pytest.mark.parametrize("use_early_tma_store_lowering", [True, False])
@pytest.mark.parametrize("DATA_PARTITION_FACTOR", [1, 2])
@pytest.mark.parametrize("SMEM_ALLOC_ALGO", [0, 1])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_matmul_tma_persistent_warp_specialize(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    num_stages,
    num_warps,
    FLATTEN,
    EPILOGUE_SUBTILE,
    A_col_major,
    B_col_major,
    use_early_tma_store_lowering,
    DATA_PARTITION_FACTOR,
    SMEM_ALLOC_ALGO,
):
    """Test matmul_kernel_tma_persistent with warp_specialize=True for both Flatten values."""
    # DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256
    if DATA_PARTITION_FACTOR != 1 and BLOCK_SIZE_M != 256:
        pytest.skip("DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256")

    if (DATA_PARTITION_FACTOR == 1 and BLOCK_SIZE_M == 256
            and (BLOCK_SIZE_N == 256 or (BLOCK_SIZE_K == 128 and not FLATTEN))):
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    if DATA_PARTITION_FACTOR == 1 and BLOCK_SIZE_M == 256 and num_stages == 3 and FLATTEN:
        pytest.skip("Out of resources: tensor memory exceeded (BLOCK_SIZE_M=256 with num_stages=3 and FLATTEN)")

    # Skip configurations that exceed hardware resource limits
    if BLOCK_SIZE_N == 256 and BLOCK_SIZE_K == 128 and (num_stages == 3 or num_warps == 4) and not FLATTEN:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    if BLOCK_SIZE_N == 256 and BLOCK_SIZE_K == 128 and num_stages == 3 and EPILOGUE_SUBTILE == 1:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    if BLOCK_SIZE_N == 256 and num_stages == 3 and FLATTEN:
        pytest.skip("Out of resources: tensor memory exceeded")

    if DATA_PARTITION_FACTOR == 2 and BLOCK_SIZE_M == 256 and BLOCK_SIZE_N == 256 and FLATTEN and SMEM_ALLOC_ALGO == 0:
        pytest.skip("Out of resources: tensor memory exceeded")

    if DATA_PARTITION_FACTOR == 2 and BLOCK_SIZE_M == 256 and num_stages == 3 and FLATTEN and SMEM_ALLOC_ALGO == 0:
        pytest.skip("Out of resources: tensor memory exceeded")

    if (DATA_PARTITION_FACTOR == 2 and SMEM_ALLOC_ALGO == 0 and BLOCK_SIZE_M == 256 and BLOCK_SIZE_N == 256
            and BLOCK_SIZE_K == 64 and not FLATTEN):
        pytest.skip("Out of resources: shared memory exceeded")

    if (DATA_PARTITION_FACTOR == 2 and SMEM_ALLOC_ALGO == 1 and BLOCK_SIZE_M == 256 and FLATTEN
            and EPILOGUE_SUBTILE == 4 and (BLOCK_SIZE_N == 256 or num_stages == 3)):
        pytest.skip("Out of resources: tensor memory exceeded")

    if (DATA_PARTITION_FACTOR == 2 and SMEM_ALLOC_ALGO == 1 and BLOCK_SIZE_M == 256 and FLATTEN
            and EPILOGUE_SUBTILE in (1, 2)):
        pytest.skip("Out of resources: tensor memory exceeded")

    # Use scope() to set use_meta_ws and automatically restore on exit
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        if A_col_major:
            A = torch.randn((K, M), dtype=dtype, device=device).t()
        else:
            A = torch.randn((M, K), dtype=dtype, device=device)
        if B_col_major:
            B = torch.randn((K, N), dtype=dtype, device=device).t()
        else:
            B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        # Set up tensor descriptors (swap dims for col-major so contiguous dim is last)
        if A_col_major:
            a_desc = TensorDescriptor(A, [K, M], [M, 1], [BLOCK_SIZE_K, BLOCK_SIZE_M])
        else:
            a_desc = TensorDescriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
        if B_col_major:
            b_desc = TensorDescriptor(B, [K, N], [N, 1], [BLOCK_SIZE_K, BLOCK_SIZE_N])
        else:
            b_desc = TensorDescriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(
            C,
            C.shape,
            C.stride(),
            [BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE],
        )

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
            A_COL_MAJOR=A_col_major,
            B_COL_MAJOR=B_col_major,
            DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
            SMEM_ALLOC_ALGO=SMEM_ALLOC_ALGO,
            num_stages=num_stages,
            num_warps=num_warps,
            early_tma_store_lowering=use_early_tma_store_lowering,
        )

        # Verify IR contains expected ops
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
@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (512, 512, 256), (8192, 8192, 1024)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64, 128])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("FLATTEN", [True, False])
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [1, 2, 4])
@pytest.mark.parametrize("A_col_major", [False, True])
@pytest.mark.parametrize("B_col_major", [False, True])
@pytest.mark.parametrize("use_early_tma_store_lowering", [True, False])
@pytest.mark.parametrize("DATA_PARTITION_FACTOR", [1, 2])
@pytest.mark.parametrize("SMEM_ALLOC_ALGO", [0, 1])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_matmul_descriptor_persistent_warp_specialize(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    num_stages,
    num_warps,
    FLATTEN,
    EPILOGUE_SUBTILE,
    A_col_major,
    B_col_major,
    use_early_tma_store_lowering,
    DATA_PARTITION_FACTOR,
    SMEM_ALLOC_ALGO,
):
    """Test matmul_kernel_descriptor_persistent with warp_specialize=True for both Flatten values."""
    # DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256
    if DATA_PARTITION_FACTOR != 1 and BLOCK_SIZE_M != 256:
        pytest.skip("DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256")

    if DATA_PARTITION_FACTOR == 1 and BLOCK_SIZE_M == 256 and num_stages == 3 and FLATTEN:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    if (DATA_PARTITION_FACTOR == 1 and BLOCK_SIZE_M == 256
            and (BLOCK_SIZE_N == 256 or (BLOCK_SIZE_K == 128 and not FLATTEN))):
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    # Skip configurations that exceed hardware resource limits
    if BLOCK_SIZE_N == 256 and BLOCK_SIZE_K == 128 and (num_stages == 3 or num_warps == 4) and not FLATTEN:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    if BLOCK_SIZE_N == 256 and BLOCK_SIZE_K == 128 and num_stages == 3 and EPILOGUE_SUBTILE == 1:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    if BLOCK_SIZE_N == 256 and num_stages == 3 and FLATTEN:
        pytest.skip("Out of resources: tensor memory exceeded")

    # Use scope() to set use_meta_ws and automatically restore on exit
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        if A_col_major:
            A = torch.randn((K, M), dtype=dtype, device=device).t()
        else:
            A = torch.randn((M, K), dtype=dtype, device=device)
        if B_col_major:
            B = torch.randn((K, N), dtype=dtype, device=device).t()
        else:
            B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

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
            A_COL_MAJOR=A_col_major,
            B_COL_MAJOR=B_col_major,
            DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
            SMEM_ALLOC_ALGO=SMEM_ALLOC_ALGO,
            num_stages=num_stages,
            num_warps=num_warps,
            early_tma_store_lowering=use_early_tma_store_lowering,
        )

        # Verify IR contains expected ops
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        # Verify correctness
        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Test 4: Multi-copy epilogue buffers with epilogue subtiling
# Focused test for the Phase 4.5 memory planner feature: with algo 1 and
# numBuffers capped at 2, 4 epilogue channels share 2 buffer copies.
# FLATTEN=True is not supported because the flattened loop generates
# scf.IfOp with else blocks, which the autoWS pass cannot handle yet.
# ============================================================================
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_multi_epilogue_subtile():
    """Test multi-copy epilogue buffers: 4 epilogue channels with 2 buffer copies."""
    M, N, K = 8192, 8192, 8192
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128
    EPILOGUE_SUBTILE = 4
    SMEM_ALLOC_ALGO = 1
    num_stages = 2
    num_warps = 4

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True

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

        a_desc = TensorDescriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_desc = TensorDescriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(
            C,
            C.shape,
            C.stride(),
            [BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE],
        )

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
            FLATTEN=False,
            A_COL_MAJOR=False,
            B_COL_MAJOR=False,
            DATA_PARTITION_FACTOR=1,
            SMEM_ALLOC_ALGO=SMEM_ALLOC_ALGO,
            num_stages=num_stages,
            num_warps=num_warps,
            early_tma_store_lowering=True,
        )

        # Verify warp specialization actually ran (ttg.warp_return is only
        # emitted by the WS code partition pass)
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_return" in ttgir, "Expected warp specialization to run"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"

        # Verify correctness
        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)
