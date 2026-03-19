# TLX GEMM kernel optimized for Blackwell Warp Specialization
# This is design for GEMM with small M where we can fit all
# or nearly all K block values for a particular N_BLOCK_SIZE
# in shared memory. This allows us to minimize reloading B.

import functools

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor


# Cached SM count — never changes during program lifetime.
# Calling torch.cuda.get_device_properties() on every matmul() call
# adds measurable overhead that degrades benchmark throughput on fast kernels.
@functools.lru_cache(maxsize=1)
def _get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": g,
                "NUM_SMEM_BUFFERS": s,
                "NUM_TMEM_BUFFERS": t,
                "EPILOGUE_SUBTILE": subtile,
                "USE_WARP_BARRIER": uwb,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
        )
        for BM in [128]
        for BN in [64, 128, 256]
        for BK in [64, 128]
        for g in [1, 4, 8]
        for s in [2, 3, 4, 5, 6, 7]
        for t in [1, 2, 3, 4]
        for subtile in [1, 2, 4, 8]
        for uwb in [False, True]
    ]


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)
    nargs["c_desc"].block_shape = [
        BLOCK_M,
        BLOCK_N // EPILOGUE_SUBTILE,
    ]
    nargs["NEXT_POW2_K"] = triton.next_power_of_2(nargs["K"])


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


@triton.jit
def _swizzle_pid_m(pid_m, num_pid_m, GROUP_SIZE_M):
    """Swizzle pid_m within groups of GROUP_SIZE_M for better L2 locality."""
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m % group_size_m)
    return pid_m


@triton.jit
def _compute_grid_info(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    return start_pid, num_pid_m, num_pid_n, k_tiles


@triton.jit
def _process_tile_epilogue_inner(
    pid_m,
    pid_n,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    EPILOGUE_SUBTILE,
    c_desc,
    c_smem_buffers,
    tmem_buffers,
    tmem_full_bars,
    tmem_empty_bars,
    cur_tmem_buf,
    tmem_read_phase,
):
    """Process epilogue for a single tile."""
    offs_bn = pid_n * BLOCK_SIZE_N

    # Wait for TMEM to be filled
    tlx.barrier_wait(tmem_full_bars[cur_tmem_buf], tmem_read_phase)

    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    # load the result from TMEM to registers
    acc_tmem = tmem_buffers[cur_tmem_buf]
    offs_am = pid_m * BLOCK_SIZE_M
    for slice_id in tl.static_range(EPILOGUE_SUBTILE):
        acc_tmem_subslice = tlx.local_slice(
            acc_tmem,
            [0, slice_id * slice_size],
            [BLOCK_SIZE_M, slice_size],
        )
        result = tlx.local_load(acc_tmem_subslice)
        tlx.barrier_arrive(tmem_empty_bars[cur_tmem_buf], 1)
        c = result.to(tlx.dtype_of(c_desc))
        c_smem = c_smem_buffers[slice_id % 2]
        tlx.async_descriptor_store_wait(1)
        tlx.local_store(c_smem, c)
        tlx.fence_async_shared()
        tlx.async_descriptor_store(
            c_desc,
            c_smem,
            [offs_am, offs_bn + slice_id * slice_size],
            eviction_policy="evict_first",
        )

    # Wait for all TMA stores to complete
    tlx.async_descriptor_store_wait(0)


@triton.jit
def _process_tile_mma_inner(
    NUM_SMEM_BUFFERS,
    buffers_A,
    buffers_B,
    tmem_buffers,
    A_smem_full_bars,
    A_smem_empty_bars,
    tmem_full_bars,
    cur_tmem_buf,
    tmem_empty_bars,
    tmem_write_phase,
    smem_accum_cnt,
    NUM_K_TILES: tl.constexpr,
):
    """Process MMA for a single tile over [k_tile_start, k_tile_end). Returns updated smem_accum_cnt."""
    # Peeled first K-iteration: wait for data before acquiring TMEM
    buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

    b_buf = 0

    # Wait for this A subtile buffer to be loaded
    tlx.barrier_wait(A_smem_full_bars[buf], phase)

    # Wait for epilogue to be done with all TMEM buffers (after data is ready)
    tlx.barrier_wait(tmem_empty_bars[cur_tmem_buf], tmem_write_phase ^ 1)

    # Perform MMA: use_acc=False for first K iteration (clears accumulator)
    tlx.async_dot(
        buffers_A[buf],
        buffers_B[b_buf],
        tmem_buffers[cur_tmem_buf],
        use_acc=False,
        mBarriers=[A_smem_empty_bars[buf]],
        out_dtype=tl.float32,
    )

    smem_accum_cnt += 1

    # Remaining K iterations with use_acc=True
    for b_buf in range(1, NUM_K_TILES):
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

        # Wait for this A subtile buffer to be loaded
        tlx.barrier_wait(A_smem_full_bars[buf], phase)

        # Perform MMA: use_acc=True for remaining K iterations
        tlx.async_dot(
            buffers_A[buf],
            buffers_B[b_buf],
            tmem_buffers[cur_tmem_buf],
            use_acc=True,
            mBarriers=[A_smem_empty_bars[buf]],
            out_dtype=tl.float32,
        )

        smem_accum_cnt += 1

    tlx.tcgen05_commit(tmem_full_bars[cur_tmem_buf])

    return smem_accum_cnt


@triton.jit
def _process_tile_producer_inner(
    pid_m,
    pid_n,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    NUM_SMEM_BUFFERS,
    a_desc,
    b_desc,
    buffers_A,
    buffers_B,
    A_smem_full_bars,
    A_smem_empty_bars,
    smem_accum_cnt,
    NUM_K_TILES: tl.constexpr,
    LOAD_B: tl.constexpr,
):
    """Process TMA loads for a single tile with all subtiles over [k_tile_start, k_tile_end)."""
    dsize: tl.constexpr = tlx.size_of(tlx.dtype_of(b_desc))
    offs_bn = pid_n * BLOCK_SIZE_N

    # Iterate along K dimension for this split's range
    for k in range(0, NUM_K_TILES):
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
        offs_k = k * BLOCK_SIZE_K

        # Load A always. B only loads for the first iteration.
        tlx.barrier_wait(A_smem_empty_bars[buf], phase ^ 1)
        offs_am = pid_m * BLOCK_SIZE_M
        # We persist over B, so we only load in the first iteration.
        A_SIZE: tl.constexpr = dsize * BLOCK_SIZE_M * BLOCK_SIZE_K
        B_SIZE: tl.constexpr = dsize * BLOCK_SIZE_N * BLOCK_SIZE_K
        if LOAD_B:
            TOTAL_SIZE: tl.constexpr = A_SIZE + B_SIZE
        else:
            TOTAL_SIZE: tl.constexpr = A_SIZE

        tlx.barrier_expect_bytes(A_smem_full_bars[buf], TOTAL_SIZE)
        tlx.async_descriptor_load(a_desc, buffers_A[buf], [offs_am, offs_k], A_smem_full_bars[buf],
                                  eviction_policy="evict_last")
        if LOAD_B:
            # Evict first because B tiles are fully partitioned.
            # We reuse A's barrier.
            tlx.async_descriptor_load(b_desc, buffers_B[k], [offs_k, offs_bn], A_smem_full_bars[buf],
                                      eviction_policy="evict_first")

        smem_accum_cnt += 1

    return smem_accum_cnt


TORCH_DTYPE_TO_TRITON = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_tma_ws_blackwell(
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
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NEXT_POW2_K: tl.constexpr,
    USE_WARP_BARRIER: tl.constexpr = False,
    NUM_SMS: tl.constexpr = 1,
):
    NUM_K_TILES: tl.constexpr = NEXT_POW2_K // BLOCK_SIZE_K

    # allocate NUM_SMEM_BUFFERS buffers
    buffers_A = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_K),
        tlx.dtype_of(a_desc),
        NUM_SMEM_BUFFERS,
    )
    # B is a fixed number of buffers that are fully loaded.
    buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_desc), NUM_K_TILES)
    # NUM_TMEM_BUFFERS (overlaps MMA and epilogue)
    # Each buffer holds one subtile: BLOCK_M_SPLIT x BLOCK_SIZE_N
    # Total buffers: NUM_TMEM_BUFFERS * NUM_MMA_GROUPS
    tmem_buffers = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS,
        tlx.storage_kind.tmem,
    )

    # Allocate SMEM buffers for epilogue TMA store (at least 2 for multi-buffering)
    NUM_EPILOGUE_SMEM_BUFFERS: tl.constexpr = 2
    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    c_smem_buffers = tlx.local_alloc(
        (BLOCK_SIZE_M, slice_size),
        tlx.dtype_of(c_desc),
        NUM_EPILOGUE_SMEM_BUFFERS,
    )

    # allocate barriers - each subtile needs its own barriers
    # NUM_SMEM_BUFFERS barriers per subtile for synchronization
    A_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    A_smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    # NUM_TMEM_BUFFERS (overlaps MMA and epilogue)
    if USE_WARP_BARRIER:
        tmem_full_bars = tlx.alloc_warp_barrier(num_barriers=NUM_TMEM_BUFFERS, num_warps=1)
        tmem_empty_bars = tlx.alloc_warp_barrier(num_barriers=NUM_TMEM_BUFFERS, num_warps=4,
                                                 num_arrivals=EPILOGUE_SUBTILE)
    else:
        tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
        tmem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=EPILOGUE_SUBTILE)

    with tlx.async_tasks():
        with tlx.async_task("default"):  # epilogue consumer
            start_pid, num_pid_m, num_pid_n, k_tiles = _compute_grid_info(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N,
                                                                          BLOCK_SIZE_K)

            tmem_accum_cnt = 0
            pid_n = start_pid // num_pid_m
            pid_m = start_pid % num_pid_m

            while pid_m < num_pid_m:
                swizzled_pid_m = _swizzle_pid_m(pid_m, num_pid_m, GROUP_SIZE_M)
                cur_tmem_buf, tmem_read_phase = _get_bufidx_phase(tmem_accum_cnt, NUM_TMEM_BUFFERS)
                _process_tile_epilogue_inner(
                    pid_m=swizzled_pid_m,
                    pid_n=pid_n,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
                    c_desc=c_desc,
                    c_smem_buffers=c_smem_buffers,
                    tmem_buffers=tmem_buffers,
                    tmem_full_bars=tmem_full_bars,
                    tmem_empty_bars=tmem_empty_bars,
                    cur_tmem_buf=cur_tmem_buf,
                    tmem_read_phase=tmem_read_phase,
                )
                tmem_accum_cnt += 1
                pid_m += NUM_SMS

        with tlx.async_task(num_warps=1, num_regs=24):  # MMA consumer
            start_pid, num_pid_m, num_pid_n, k_tiles = _compute_grid_info(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N,
                                                                          BLOCK_SIZE_K)

            tmem_accum_cnt = 0
            smem_accum_cnt = 0
            pid_m = start_pid % num_pid_m

            while pid_m < num_pid_m:
                cur_tmem_buf, tmem_write_phase = _get_bufidx_phase(tmem_accum_cnt, NUM_TMEM_BUFFERS)
                smem_accum_cnt = _process_tile_mma_inner(
                    NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                    buffers_A=buffers_A,
                    buffers_B=buffers_B,
                    tmem_buffers=tmem_buffers,
                    A_smem_full_bars=A_smem_full_bars,
                    A_smem_empty_bars=A_smem_empty_bars,
                    tmem_full_bars=tmem_full_bars,
                    cur_tmem_buf=cur_tmem_buf,
                    tmem_empty_bars=tmem_empty_bars,
                    tmem_write_phase=tmem_write_phase,
                    smem_accum_cnt=smem_accum_cnt,
                    NUM_K_TILES=NUM_K_TILES,
                )
                tmem_accum_cnt += 1
                pid_m += NUM_SMS

        with tlx.async_task(num_warps=1, num_regs=24):  # producer, TMA load
            start_pid, num_pid_m, num_pid_n, k_tiles = _compute_grid_info(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N,
                                                                          BLOCK_SIZE_K)

            smem_accum_cnt = 0
            pid_n = start_pid // num_pid_m
            pid_m = start_pid % num_pid_m
            swizzled_pid_m = _swizzle_pid_m(pid_m, num_pid_m, GROUP_SIZE_M)

            # First M-tile: load both A and B
            smem_accum_cnt = _process_tile_producer_inner(
                pid_m=swizzled_pid_m,
                pid_n=pid_n,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                a_desc=a_desc,
                b_desc=b_desc,
                buffers_A=buffers_A,
                buffers_B=buffers_B,
                A_smem_full_bars=A_smem_full_bars,
                A_smem_empty_bars=A_smem_empty_bars,
                smem_accum_cnt=smem_accum_cnt,
                NUM_K_TILES=NUM_K_TILES,
                LOAD_B=True,
            )
            pid_m += NUM_SMS

            # Remaining M-tiles: only load A (B persists in SMEM)
            while pid_m < num_pid_m:
                swizzled_pid_m = _swizzle_pid_m(pid_m, num_pid_m, GROUP_SIZE_M)
                smem_accum_cnt = _process_tile_producer_inner(
                    pid_m=swizzled_pid_m,
                    pid_n=pid_n,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                    a_desc=a_desc,
                    b_desc=b_desc,
                    buffers_A=buffers_A,
                    buffers_B=buffers_B,
                    A_smem_full_bars=A_smem_full_bars,
                    A_smem_empty_bars=A_smem_empty_bars,
                    smem_accum_cnt=smem_accum_cnt,
                    NUM_K_TILES=NUM_K_TILES,
                    LOAD_B=False,
                )
                pid_m += NUM_SMS


def matmul(a, b):
    """Matrix multiplication using TLX GEMM kernel.

    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
    Returns:
        Output matrix C of shape (M, N)
    """
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    NUM_SMS = _get_num_sms()

    def grid(META):
        num_pid_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
        return (min(num_pid_m, NUM_SMS), )

    matmul_kernel_tma_ws_blackwell[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        NUM_SMS=NUM_SMS,
    )
    return c
