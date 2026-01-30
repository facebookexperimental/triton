# TLX GEMM kernel optimized for Blackwell Warp Specialization
from typing import Optional

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert align == 128
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device=DEVICE)


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
                "NUM_SMEM_BUFFERS": s,
                "NUM_TMEM_BUFFERS": t,
                "EPILOGUE_SUBTILE": subtile,
            },
            num_warps=4,
            num_stages=1,
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [2, 3, 4]
        for t in [2, 3]
        for subtile in [True]
    ]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_tma_ws_blackwell_clc(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                                       stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                       BLOCK_SIZE_K: tl.constexpr,  #
                                       GROUP_SIZE_M: tl.constexpr,  #
                                       NUM_SMEM_BUFFERS: tl.constexpr,  #
                                       NUM_TMEM_BUFFERS: tl.constexpr,  #
                                       NUM_SMS: tl.constexpr,  #
                                       NUM_CLC_STAGES: tl.constexpr,  #
                                       EPILOGUE_SUBTILE: tl.constexpr,  #
                                       ):
    # Initialize TMA descriptors
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    if EPILOGUE_SUBTILE:
        c_desc = tl.make_tensor_descriptor(
            c_ptr,
            shape=[M, N],
            strides=[stride_cm, stride_cn],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // 2],
        )
    else:
        c_desc = tl.make_tensor_descriptor(
            c_ptr,
            shape=[M, N],
            strides=[stride_cm, stride_cn],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )
    # allocate NUM_SMEM_BUFFERS buffers
    buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float16, NUM_SMEM_BUFFERS)
    buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tl.float16, NUM_SMEM_BUFFERS)
    # use multiple TMEM buffers to overlap MMA and epilogue
    tmem_buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem)

    # allocate barriers
    smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)

    clc_context = tlx.clc_create_context(NUM_CLC_STAGES, 3)

    with tlx.async_tasks():
        with tlx.async_task("default"):  # epilogue consumer
            # common code duplicated for each region to avoid SMEM overhead
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            # end of common code

            tmem_read_phase = 0
            cur_tmem_buf = 0

            tile_id = start_pid

            clc_phase_producer = 1
            clc_phase_consumer = 0
            clc_buf = 0
            while tile_id != -1:
                clc_buf = clc_buf % NUM_CLC_STAGES
                # Debug prints
                # if tlx.thread_id(axis=0) == 0:
                # tl.device_print("Default WG Processing CtaID", tile_id)
                # producer
                tlx.clc_producer(clc_context, clc_buf, clc_phase_producer)
                # clc_phase_producer ^= 1
                clc_phase_producer = clc_phase_producer ^ (clc_buf == (NUM_CLC_STAGES - 1))

                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn = pid_n * BLOCK_SIZE_N

                tlx.barrier_wait(tmem_full_bars[cur_tmem_buf], tmem_read_phase)
                # flip phase at the end of a round of using TMEM barriers
                tmem_read_phase = tmem_read_phase ^ (cur_tmem_buf == NUM_TMEM_BUFFERS - 1)

                # load the result from TMEM to registers
                acc_tmem = tmem_buffers[cur_tmem_buf]

                if EPILOGUE_SUBTILE:
                    # We load/store the result half by half to reduce SMEM pressure
                    acc_tmem_subslice1 = tlx.subslice(acc_tmem, 0, BLOCK_SIZE_N // 2)
                    result = tlx.local_load(acc_tmem_subslice1)
                    c = result.to(tl.float16)
                    c_desc.store([offs_am, offs_bn], c)

                    acc_tmem_subslice2 = tlx.subslice(acc_tmem, BLOCK_SIZE_N // 2, BLOCK_SIZE_N // 2)
                    result = tlx.local_load(acc_tmem_subslice2)
                    c = result.to(tl.float16)
                    c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c)
                else:
                    result = tlx.local_load(acc_tmem)
                    c = result.to(tl.float16)
                    c_desc.store([offs_am, offs_bn], c)

                # done storing this buffer, signal MMA consumer to resume writing to it
                tlx.barrier_arrive(tmem_empty_bars[cur_tmem_buf], 1)

                cur_tmem_buf = (cur_tmem_buf + 1) % NUM_TMEM_BUFFERS

                tile_id = tlx.clc_consumer(clc_context, clc_buf, clc_phase_consumer)
                # clc_phase_consumer ^= 1
                clc_phase_consumer = clc_phase_consumer ^ (clc_buf == (NUM_CLC_STAGES - 1))
                clc_buf += 1

                # Debug-only: verifying that CLC steals workloads successfully
                # if tlx.thread_id(axis=0) == 0:
                # tl.device_print("Extracted CtaID", tile_id)

        with tlx.async_task(num_warps=1, num_regs=232):  # MMA consumer
            # common code duplicated for each region to avoid SMEM overhead
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            # end of common code

            dot_phase = 0  # the current phase of dot op
            tmem_write_phase = 1  # sync between epilogue consumer and MMA consumer
            cur_tmem_buf = 0

            processed_k_iters = 0
            tile_id = start_pid
            clc_phase = 0
            clc_buf = 0
            while tile_id != -1:
                clc_buf = clc_buf % NUM_CLC_STAGES
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn = pid_n * BLOCK_SIZE_N

                # wait epilogue consumer to be done with the buffer before reusing it
                tlx.barrier_wait(tmem_empty_bars[cur_tmem_buf], tmem_write_phase)
                # flip phase at the end of a round of using TMEM barriers
                tmem_write_phase = tmem_write_phase ^ (cur_tmem_buf == NUM_TMEM_BUFFERS - 1)

                # now iterate along K to compute result for the block
                for k in range(0, k_tiles):
                    # processed_k_iters + k means we use the immediate next buffer slot of tile_id x when we start tile_id x+1
                    buf = (processed_k_iters + k) % NUM_SMEM_BUFFERS
                    # wait for current phase(round) of load for this buf
                    tlx.barrier_wait(smem_full_bars[buf], dot_phase)
                    # buffer is now ready with loaded data, tlx.async_dot will signal `mBarrier` when done
                    tlx.async_dot(buffers_A[buf], buffers_B[buf], tmem_buffers[cur_tmem_buf], use_acc=k > 0,
                                  mBarriers=[smem_empty_bars[buf]], out_dtype=tl.float32)
                    # flip phase at the end of a round
                    dot_phase = dot_phase ^ (buf == NUM_SMEM_BUFFERS - 1)

                # wait for last mma to complete
                last_buf = (processed_k_iters + k_tiles - 1) % NUM_SMEM_BUFFERS
                # in case phase was flipped, we should use the phase value when dot op was issued
                last_dot_phase = dot_phase ^ (last_buf == NUM_SMEM_BUFFERS - 1)
                tlx.barrier_wait(smem_empty_bars[last_buf], last_dot_phase)

                # done filling this buffer, signal epilogue consumer
                tlx.barrier_arrive(tmem_full_bars[cur_tmem_buf], 1)

                # possibly enter next iteration (next tile) without waiting for epilogue
                cur_tmem_buf = (cur_tmem_buf + 1) % NUM_TMEM_BUFFERS
                processed_k_iters += k_tiles
                tile_id = tlx.clc_consumer(clc_context, clc_buf, clc_phase)
                # clc_phase ^= 1
                clc_phase = clc_phase ^ (clc_buf == (NUM_CLC_STAGES - 1))
                clc_buf += 1

        with tlx.async_task(num_warps=1, num_regs=232):  # producer, TMA load
            # common code duplicated for each region to avoid SMEM overhead
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            # end of common code

            load_phase = 0  # the current phase of TMA load
            # we virtually "flatten" the two layer loop as if we're performing tma loads on
            # one big list of data
            processed_k_iters = 0

            tile_id = start_pid
            clc_phase = 0
            clc_buf = 0
            while tile_id != -1:
                clc_buf = clc_buf % NUM_CLC_STAGES
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn = pid_n * BLOCK_SIZE_N

                for k in range(0, k_tiles):
                    # processed_k_iters + k means we use the immediate next buffer slot of tile_id x when we start tile_id x+1
                    buf = (processed_k_iters + k) % NUM_SMEM_BUFFERS
                    # wait for previous phase(round) of dot for this buf
                    tlx.barrier_wait(smem_empty_bars[buf], load_phase ^ 1)
                    # buffer is now ready to be used again
                    offs_k = k * BLOCK_SIZE_K
                    tlx.barrier_expect_bytes(smem_full_bars[buf],
                                             2 * (BLOCK_SIZE_M + BLOCK_SIZE_N) * BLOCK_SIZE_K)  # float16
                    tlx.async_descriptor_load(a_desc, buffers_A[buf], [offs_am, offs_k], smem_full_bars[buf])
                    tlx.async_descriptor_load(b_desc, buffers_B[buf], [offs_k, offs_bn], smem_full_bars[buf])
                    # flip phase at the end of a round
                    load_phase = load_phase ^ (buf == NUM_SMEM_BUFFERS - 1)
                processed_k_iters += k_tiles
                tile_id = tlx.clc_consumer(clc_context, clc_buf, clc_phase)
                # clc_phase ^= 1
                clc_phase = clc_phase ^ (clc_buf == (NUM_CLC_STAGES - 1))
                clc_buf += 1


def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Initialize TMA descriptor storage allocator
    triton.set_allocator(alloc_fn)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # Persistent kernel to have thread block resident in SM as long as possible
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel_tma_ws_blackwell_clc[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        NUM_CLC_STAGES=1,  #
    )

    return c
