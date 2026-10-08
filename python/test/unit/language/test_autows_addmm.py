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
from triton._internal_testing import is_blackwell, is_hopper
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
@pytest.mark.parametrize("generate_subtiled_region", [True, False])
@pytest.mark.skipif(not (is_hopper() or is_blackwell()), reason="Requires Hopper or Blackwell")
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
    generate_subtiled_region,
):
    """Test addmm kernel (bias + matmul) with warp_specialize=True."""
    if FLATTEN:
        pytest.skip("FLATTEN will not WarpSpecialize although it will otherwise pass.")

    if is_hopper():
        if EPILOGUE_SUBTILE != 1:
            pytest.skip("EPILOGUE_SUBTILE is only supported for Blackwell.")

        if BLOCK_SIZE_M == 256:
            pytest.skip("BLOCK_SIZE_M == 256 runs out of shared memory for Hopper")

    # DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256
    if DATA_PARTITION_FACTOR != 1 and BLOCK_SIZE_M != 256:
        pytest.skip("DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256")

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
            num_stages=num_stages,
            num_warps=num_warps,
            generate_subtiled_region=generate_subtiled_region,
        )

        # Verify IR contains expected ops
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"
        if is_blackwell():
            assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"
        else:
            assert "ttng.warp_group_dot" in ttgir, "Expected Hopper MMA instruction"

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

    unified_waits = re.search(
        r"ttng\.wait_barrier[^\n]*dstTask = 3[^\n]*\n\s*"
        r"ttng\.wait_barrier[^\n]*dstTask = 1[^\n]*",
        ttgir,
    )
    assert unified_waits, "Expected adjacent bias-TMA and accumulator-TMEM waits"
    unified_region = ttgir[unified_waits.end():]
    tmem_load = unified_region.find("ttng.tmem_load")
    bias_load = unified_region.find("ttg.local_load")
    assert 0 <= tmem_load < bias_load, "Expected the TMEM load before the bias load after wait unification"

    # Check that convert_layout on bias happens before broadcast (on 1xN, not MxN)
    cvt_before_bc = re.search(r"convert_layout.*tensor<1x\d+xf32.*\n.*tt\.broadcast.*tensor<1x\d+xf32", ttgir)
    bc_before_cvt = re.search(
        r"tt\.broadcast.*tensor<1x\d+xf32.*tensor<128x\d+xf32.*\n.*convert_layout.*tensor<128x\d+xf32", ttgir)
    assert cvt_before_bc or not bc_before_cvt, "Expected convert_layout before broadcast (on small 1xN tensor)"

    # Verify correctness: bias_1d + A @ B.T
    torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


@triton.jit
def addmm_kernel_pointer_bias_ws(
    bias_ptr,
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent addmm with a pointer-loaded bias and pointer store epilogue."""
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    a_desc = tl.make_tensor_descriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=False, warp_specialize=True,
                            data_partition_factor=1, separate_epilogue_store=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            a = a_desc.load([pid_m * BLOCK_SIZE_M, ki * BLOCK_SIZE_K])
            b = b_desc.load([pid_n * BLOCK_SIZE_N, ki * BLOCK_SIZE_K])
            accumulator += tl.dot(a, b.T)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]
        mask = (rm < M) & (rn < N)
        bias = tl.load(bias_ptr + tl.broadcast_to(rn, [BLOCK_SIZE_M, BLOCK_SIZE_N]), mask)
        tl.store(C + rm * N + rn, (accumulator + bias.to(tl.float32)).to(tl.bfloat16), mask)


@pytest.mark.parametrize("K", [128, 256])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_autows_addmm_pointer_bias_single_k_tile(K):
    """With a single K tile the accumulator is not loop-carried: the bias is
    folded into the MMA and the TMEM accumulator is initialised every tile.
    The loop scheduler puts that init and the epilogue load in different
    stages, so the accumulator needs one TMEM buffer per in-flight tile.
    With one buffer, the next tile's init overwrites the accumulator before
    the epilogue reads it, and tiles beyond the first wave come out wrong."""
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        BLOCK = 128
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        N = 1024
        # More tiles than SMs, with a ragged last M tile.
        M = (NUM_SMS // (N // BLOCK) + 1) * BLOCK + BLOCK // 2
        torch.manual_seed(0)
        bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        C = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        num_tiles = triton.cdiv(M, BLOCK) * (N // BLOCK)
        kernel = addmm_kernel_pointer_bias_ws[(min(NUM_SMS, num_tiles), )](bias, A, B, C, M, N, K, BLOCK, BLOCK, BLOCK,
                                                                           NUM_SMS, num_warps=4, num_stages=3)
        assert "ttg.warp_specialize" in kernel.asm["ttgir"], "Expected warp specialization in IR"
        ref_out = (A.double() @ B.double().T + bias.double()).to(torch.bfloat16)
        torch.testing.assert_close(C, ref_out, atol=1e-2, rtol=1e-2)


@triton.jit
def addmm_kernel_tma_bias_ws(
    bias_ptr,
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent addmm whose epilogue reads the bias with a 1-D TMA load."""
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    a_desc = tl.make_tensor_descriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(C, [M, N], [N, 1], [BLOCK_SIZE_M, BLOCK_SIZE_N])
    bias_desc = tl.make_tensor_descriptor(bias_ptr, [N], [1], [BLOCK_SIZE_N])
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=False, warp_specialize=True,
                            data_partition_factor=1, separate_epilogue_store=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            a = a_desc.load([pid_m * BLOCK_SIZE_M, ki * BLOCK_SIZE_K])
            b = b_desc.load([pid_n * BLOCK_SIZE_N, ki * BLOCK_SIZE_K])
            accumulator += tl.dot(a, b.T)
        bias = bias_desc.load([pid_n * BLOCK_SIZE_N])[None, :]
        c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], (accumulator + bias.to(tl.float32)).to(tl.bfloat16))


@pytest.mark.parametrize("BLOCK_N", [128, 32])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_autows_addmm_tma_bias_single_k_tile(K, BLOCK_N):
    """With a single K tile the bias is folded into the MMA: the epilogue
    partition stores the TMA-loaded bias into the TMEM accumulator and the
    MMA accumulates onto it. The MMA must wait for that store, or the output
    tile is just the bias. A 32-wide bf16 bias tile is 64 bytes, so its TMA
    landing buffer must also stay single-buffered (TMA needs 128-byte aligned
    destinations)."""
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        BLOCK_M, BLOCK_K = 128, 64
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        M, N = 5588, 256
        torch.manual_seed(0)
        bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        C = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        num_tiles = triton.cdiv(M, BLOCK_M) * (N // BLOCK_N)
        kernel = addmm_kernel_tma_bias_ws[(min(NUM_SMS, num_tiles), )](bias, A, B, C, M, N, K, BLOCK_M, BLOCK_N,
                                                                       BLOCK_K, NUM_SMS, num_warps=8, num_stages=3)
        assert "ttg.warp_specialize" in kernel.asm["ttgir"], "Expected warp specialization in IR"
        ref_out = (A.double() @ B.double().T + bias.double()).to(torch.bfloat16)
        torch.testing.assert_close(C, ref_out, atol=1e-2, rtol=1e-2)


@triton.jit
def addmm_kernel_tma_bias_subtiled_ws(
    bias_ptr,
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent addmm whose epilogue is split into two subtiles, each adding
    a bias slice read with a 1-D TMA load."""
    SUB: tl.constexpr = BLOCK_SIZE_N // 2
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    a_desc = tl.make_tensor_descriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(C, [M, N], [N, 1], [BLOCK_SIZE_M, SUB])
    bias_desc = tl.make_tensor_descriptor(bias_ptr, [N], [1], [SUB])
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=False, warp_specialize=True,
                            data_partition_factor=1, separate_epilogue_store=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            a = a_desc.load([pid_m * BLOCK_SIZE_M, ki * BLOCK_SIZE_K])
            b = b_desc.load([pid_n * BLOCK_SIZE_N, ki * BLOCK_SIZE_K])
            accumulator += tl.dot(a, b.T)
        acc0, acc1 = tl.split(tl.permute(tl.reshape(accumulator, (BLOCK_SIZE_M, 2, SUB)), (0, 2, 1)))
        for i in tl.static_range(2):
            sub = acc0 if i == 0 else acc1
            offs_n = pid_n * BLOCK_SIZE_N + i * SUB
            bias = bias_desc.load([offs_n])[None, :]
            c_desc.store([pid_m * BLOCK_SIZE_M, offs_n], (sub + bias.to(tl.float32)).to(tl.bfloat16))


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_autows_addmm_tma_bias_subtiled_epilogue():
    """The epilogue reads each bias slice from a two-slot TMA ring with a
    local_load and then hands the slot back to the load partition. The next
    TMA load writes through the async proxy, so the release needs a proxy
    fence; without it CTAs with three or more tiles add a later tile's bias."""
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        M, N, K = 5588, 8192, 64
        torch.manual_seed(0)
        bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        num_tiles = triton.cdiv(M, BLOCK_M) * (N // BLOCK_N)
        ref_out = (A.double() @ B.double().T + bias.double()).to(torch.bfloat16)
        # The race is timing dependent, so run the kernel a few times.
        for _ in range(3):
            C = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
            kernel = addmm_kernel_tma_bias_subtiled_ws[(min(NUM_SMS,
                                                            num_tiles), )](bias, A, B, C, M, N, K, BLOCK_M, BLOCK_N,
                                                                           BLOCK_K, NUM_SMS, num_warps=8, num_stages=2)
            assert "ttg.warp_specialize" in kernel.asm["ttgir"], "Expected warp specialization in IR"
            torch.testing.assert_close(C, ref_out, atol=1e-2, rtol=1e-2)


@triton.jit
def addmm_kernel_tma_store_bias_ws(
    bias_ptr,
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent addmm with a pointer-loaded bias and a TMA store epilogue."""
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    a_desc = tl.make_tensor_descriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(C, [M, N], [N, 1], [BLOCK_SIZE_M, BLOCK_SIZE_N])
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=False, warp_specialize=True,
                            data_partition_factor=1, separate_epilogue_store=False):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            a = a_desc.load([pid_m * BLOCK_SIZE_M, ki * BLOCK_SIZE_K])
            b = b_desc.load([pid_n * BLOCK_SIZE_N, ki * BLOCK_SIZE_K])
            accumulator += tl.dot(a, b.T)
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]
        bias = tl.load(bias_ptr + rn)
        c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], (accumulator + bias.to(tl.float32)).to(tl.bfloat16))


@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_autows_addmm_tma_store_short_trip_count(num_warps):
    """With num_stages=4 the epilogue partition is peeled into a three-deep
    drain. CTAs with fewer tiles than that run dead drain iterations, whose
    staging-buffer wait is predicated off; the local_store into the
    single-slot TMA staging buffer must be predicated too, or it overwrites
    the tile the TMA store partition is still reading."""
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        BLOCK = 128
        K = 128
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        N = 1024
        num_pid_n = N // BLOCK
        # Most CTAs get two tiles, a few get three.
        M = triton.cdiv(2 * NUM_SMS + 32, num_pid_n) * BLOCK - BLOCK // 2
        torch.manual_seed(0)
        bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        C = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        num_tiles = triton.cdiv(M, BLOCK) * num_pid_n
        kernel = addmm_kernel_tma_store_bias_ws[(min(NUM_SMS,
                                                     num_tiles), )](bias, A, B, C, M, N, K, BLOCK, BLOCK, BLOCK,
                                                                    NUM_SMS, num_warps=num_warps, num_stages=4)
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir
        assert "ttng.async_tma_copy_local_to_global" in ttgir
        ref_out = (A.double() @ B.double().T + bias.double()).to(torch.bfloat16)
        torch.testing.assert_close(C, ref_out, atol=1e-2, rtol=1e-2)
