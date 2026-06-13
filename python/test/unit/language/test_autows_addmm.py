"""
Unit tests for addmm (bias + A @ B.T) with automatic warp specialization.

Based on test_tutorial09_matmul_tma_persistent_warp_specialize from
test_tutorial09_warp_specialization.py, with an added bias load in the epilogue.
"""

import re

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
from triton.language.extra.subtile_ops import _split_n_2D
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

        acc_slices = _split_n_2D(accumulator, EPILOGUE_SUBTILE)
        slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
        for slice_id in tl.static_range(0, EPILOGUE_SUBTILE):
            offs_cn = offs_bn + slice_id * slice_size
            bias = bias_desc.load([offs_am, offs_cn]).to(tl.float32)
            c = (acc_slices[slice_id] + bias).to(dtype)
            c_desc.store([offs_am, offs_cn], c)


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 8192)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64])
@pytest.mark.parametrize("num_stages", [3])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("FLATTEN", [True, False])
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [1, 2, 4])
@pytest.mark.parametrize("A_col_major", [False, True])
@pytest.mark.parametrize("B_col_major", [False, True])
@pytest.mark.parametrize("DATA_PARTITION_FACTOR", [1, 2])
@pytest.mark.parametrize("SMEM_ALLOC_ALGO", [0, 1])
@pytest.mark.parametrize("generate_subtiled_region", [True, False])
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
    DATA_PARTITION_FACTOR,
    SMEM_ALLOC_ALGO,
    generate_subtiled_region,
):
    """Test addmm kernel (bias + matmul) with warp_specialize=True."""
    if FLATTEN:
        pytest.skip("FLATTEN will not WarpSpecialize although it will otherwise pass.")

    if generate_subtiled_region:
        pytest.skip("TODO: enable generate_subtiled_region=True")

    # DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256
    if DATA_PARTITION_FACTOR != 1 and BLOCK_SIZE_M != 256:
        pytest.skip("DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256")

    if BLOCK_SIZE_M == 256 and not FLATTEN and SMEM_ALLOC_ALGO == 0:
        pytest.skip("Out of resources: shared memory exceeded")

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

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
            generate_subtiled_region=generate_subtiled_region,
        )

        # Verify IR contains expected ops
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        # Verify correctness: bias + A @ B.T
        ref_out = (torch.matmul(A.to(torch.float32), B.T.to(torch.float32)) + bias.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


@triton.jit
def addmm_kernel_1d_bias_ws(
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
    NUM_SMS: tl.constexpr,
):
    """Persistent TMA addmm with 1D bias broadcast and warp specialization."""
    dtype = tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            flatten=False,
            warp_specialize=True,
            disallow_acc_multi_buffer=True,
            separate_epilogue_store=True,
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

        # 1D bias: load [1, BLOCK_SIZE_N], broadcast to [BLOCK_SIZE_M, BLOCK_SIZE_N]
        bias_tile = bias_desc.load([0, offs_bn]).to(tl.float32)
        bias_tile = tl.broadcast_to(bias_tile, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        accumulator = accumulator + bias_tile
        c = accumulator.to(dtype)
        c_desc.store([offs_am, offs_bn], c)


def _run_addmm_1d_bias_ws():
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        M, N, K = 1024, 1024, 512
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 128
        num_warps = 4
        num_stages = 6
        dtype = torch.float16
        GROUP_SIZE_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=dtype, device=device)
        B = torch.randn((N, K), dtype=dtype, device=device)
        bias_1d = torch.randn((N, ), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        a_desc = TensorDescriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_desc = TensorDescriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N])
        bias_desc = TensorDescriptor(bias_1d, [1, N], [N, 1], [1, BLOCK_SIZE_N])

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = addmm_kernel_1d_bias_ws[grid](
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
            NUM_SMS=NUM_SMS,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        ref_out = (torch.matmul(A.to(torch.float32), B.T.to(torch.float32)) + bias_1d.to(torch.float32)).to(dtype)
        return kernel, C, ref_out


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_autows_addmm_hoist_convert_before_broadcast():
    """Test that convert_layout is hoisted before broadcast in the addmm epilogue.

    Config: BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, EPILOGUE_SUBTILE=1,
    num_warps=4, num_stages=6.

    This config previously OOM'd because the convert_layout on the 128x128
    accumulator (64KB SMEM scratch) plus the A/B tile buffers (~197KB)
    exceeded the 232KB B200 SMEM limit. With the hoist fix, the convert happens
    on the 1x128 bias (512B scratch) instead, and the kernel fits.
    """
    kernel, C, ref_out = _run_addmm_1d_bias_ws()

    # Verify the TTGIR has the convert before broadcast pattern
    ttgir = kernel.asm["ttgir"]
    assert "partition0" in ttgir or "partition1" in ttgir, "Expected warp specialization partitions in IR"
    assert "ttng.async_tma_copy_local_to_global" in ttgir, "Expected TMA store copy in IR"
    assert "ttng.async_tma_store_token_wait" in ttgir, "Expected TMA store token wait in IR"
    assert "tt.descriptor_store" not in ttgir, "Expected descriptor stores to be lowered"
    assert "can_rotate_by_buffer_count" not in ttgir, "Expected TMA store wait rotation to be resolved"

    # Check that convert_layout on bias happens before broadcast (on 1xN, not MxN)
    cvt_before_bc = re.search(r"convert_layout.*tensor<1x\d+xf32.*\n.*tt\.broadcast.*tensor<1x\d+xf32", ttgir)
    bc_before_cvt = re.search(
        r"tt\.broadcast.*tensor<1x\d+xf32.*tensor<128x\d+xf32.*\n.*convert_layout.*tensor<128x\d+xf32", ttgir)
    assert cvt_before_bc or not bc_before_cvt, "Expected convert_layout before broadcast (on small 1xN tensor)"

    # Verify correctness: bias_1d + A @ B.T
    torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)
