"""
Unit tests for addmm (bias + A @ B.T) with automatic warp specialization.

Based on test_tutorial09_matmul_tma_persistent_warp_specialize from
test_tutorial09_warp_specialization.py, with an added bias load in the epilogue.
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


@triton.jit
def addmm_kernel_tma_persistent_ws(
    a_desc,
    b_desc,
    c_desc,
    bias_desc,
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
    """Persistent TMA addmm (bias + matmul) with warp specialization."""
    dtype = tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            flatten=FLATTEN,
            warp_specialize=True,
            disallow_acc_multi_buffer=True,
            data_partition_factor=DATA_PARTITION_FACTOR,
            smem_alloc_algo=SMEM_ALLOC_ALGO,
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
            # Load full bias tile via TMA, add in float32, then downcast
            bias = bias_desc.load([offs_cm, offs_cn]).to(tl.float32)
            accumulator = accumulator + bias
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)
        elif EPILOGUE_SUBTILE == 2:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            # Load bias halves via TMA, add in float32, then downcast
            bias0 = bias_desc.load([offs_cm, offs_cn]).to(tl.float32)
            acc0 = acc0 + bias0
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            bias1 = bias_desc.load([offs_cm, offs_cn + BLOCK_SIZE_N // 2]).to(tl.float32)
            acc1 = acc1 + bias1
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        elif EPILOGUE_SUBTILE == 4:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            acc00, acc01 = tl.split(tl.permute(tl.reshape(acc0, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 4)), (0, 2, 1)))
            acc10, acc11 = tl.split(tl.permute(tl.reshape(acc1, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 4)), (0, 2, 1)))
            # Load bias quarters via TMA, add in float32, then downcast
            bias00 = bias_desc.load([offs_cm, offs_cn]).to(tl.float32)
            acc00 = acc00 + bias00
            c00 = acc00.to(dtype)
            c_desc.store([offs_cm, offs_cn], c00)
            bias01 = bias_desc.load([offs_cm, offs_cn + BLOCK_SIZE_N // 4]).to(tl.float32)
            acc01 = acc01 + bias01
            c01 = acc01.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 4], c01)
            bias10 = bias_desc.load([offs_cm, offs_cn + 2 * (BLOCK_SIZE_N // 4)]).to(tl.float32)
            acc10 = acc10 + bias10
            c10 = acc10.to(dtype)
            c_desc.store([offs_cm, offs_cn + 2 * (BLOCK_SIZE_N // 4)], c10)
            bias11 = bias_desc.load([offs_cm, offs_cn + 3 * (BLOCK_SIZE_N // 4)]).to(tl.float32)
            acc11 = acc11 + bias11
            c11 = acc11.to(dtype)
            c_desc.store([offs_cm, offs_cn + 3 * (BLOCK_SIZE_N // 4)], c11)


@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (512, 512, 256), (1024, 1024, 512)])
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
def test_autows_addmm_tma_persistent(
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
    """Test addmm kernel (bias + matmul) with warp_specialize=True."""
    # DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256
    if DATA_PARTITION_FACTOR != 1 and BLOCK_SIZE_M != 256:
        pytest.skip("DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256")

    if BLOCK_SIZE_M == 256 and DATA_PARTITION_FACTOR == 1:
        # TODO: Re-enable in some cases
        pytest.skip("BLOCK_SIZE_M=256 with DATA_PARTITION_FACTOR=1 not supported")

    if use_early_tma_store_lowering:
        # This fails due to NaN with < 0.1% of values.
        # Is this the previous register issue?
        pytest.skip("TODO: Fix early TMA store lowering for addmm")

    # Skip configurations that exceed hardware resource limits
    if BLOCK_SIZE_N == 256 and not FLATTEN:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    if BLOCK_SIZE_N == 256 and BLOCK_SIZE_K == 128 and num_stages == 3:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    if not FLATTEN and BLOCK_SIZE_K == 128 and B_col_major and not A_col_major:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True
        triton.knobs.nvidia.use_early_tma_store_lowering = use_early_tma_store_lowering

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
        bias = torch.randn((M, N), dtype=dtype, device=device)
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
        bias_desc = TensorDescriptor(
            bias,
            [M, N],
            [N, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE],
        )

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = addmm_kernel_tma_persistent_ws[grid](
            a_desc,
            b_desc,
            c_desc,
            bias_desc,
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
        )

        # Verify IR contains expected ops
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        # Verify correctness: bias + A @ B.T
        ref_out = (torch.matmul(A.to(torch.float32), B.T.to(torch.float32)) + bias.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)
