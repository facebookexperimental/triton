# TLX GEMM kernel optimized for Blackwell Warp Specialization
import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor
from triton._internal_testing import is_blackwell

from tlx_kernel_utils import (
    compute_tile_position,
    compute_grid_info,
    compute_tile_offsets,
    flip_phase_on_boundary,
    async_load_ab_tiles,
    async_mma_with_barrier,
    epilogue_store_half_half,
    epilogue_store_full,
)

DEVICE = triton.runtime.driver.active.get_active_torch_device()


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
            pre_hook=matmul_tma_set_block_size_hook,
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [2, 3, 4]
        for t in [2, 3]
        for subtile in [True]
    ]


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


# Use compute_tile_position from tlx_kernel_utils instead of local _compute_pid


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_tma_ws_blackwell_clc(a_desc, b_desc, c_desc, M, N, K, BLOCK_SIZE_M: tl.constexpr,
                                       BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
                                       GROUP_SIZE_M: tl.constexpr,  #
                                       NUM_SMEM_BUFFERS: tl.constexpr,  #
                                       NUM_TMEM_BUFFERS: tl.constexpr,  #
                                       NUM_SMS: tl.constexpr,  #
                                       NUM_CLC_STAGES: tl.constexpr,  #
                                       EPILOGUE_SUBTILE: tl.constexpr,  #
                                       ):
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
            # Use compute_grid_info from tlx_kernel_utils
            start_pid, num_pid_m, num_pid_n, num_pid_in_group, _, k_tiles = compute_grid_info(
                M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
            )

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
                clc_phase_producer = flip_phase_on_boundary(clc_phase_producer, clc_buf, NUM_CLC_STAGES)

                pid_m, pid_n = compute_tile_position(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am, offs_bn = compute_tile_offsets(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)

                tlx.barrier_wait(tmem_full_bars[cur_tmem_buf], tmem_read_phase)
                tmem_read_phase = flip_phase_on_boundary(tmem_read_phase, cur_tmem_buf, NUM_TMEM_BUFFERS)

                # load the result from TMEM to registers
                acc_tmem = tmem_buffers[cur_tmem_buf]

                if EPILOGUE_SUBTILE:
                    # We load/store the result half by half to reduce SMEM pressure
                    epilogue_store_half_half(acc_tmem, c_desc, offs_am, offs_bn, BLOCK_SIZE_N)
                else:
                    epilogue_store_full(acc_tmem, c_desc, offs_am, offs_bn)

                # done storing this buffer, signal MMA consumer to resume writing to it
                tlx.barrier_arrive(tmem_empty_bars[cur_tmem_buf], 1)

                cur_tmem_buf = (cur_tmem_buf + 1) % NUM_TMEM_BUFFERS

                tile_id = tlx.clc_consumer(clc_context, clc_buf, clc_phase_consumer)
                clc_phase_consumer = flip_phase_on_boundary(clc_phase_consumer, clc_buf, NUM_CLC_STAGES)
                clc_buf += 1

                # Debug-only: verifying that CLC steals workloads successfully
                # if tlx.thread_id(axis=0) == 0:
                # tl.device_print("Extracted CtaID", tile_id)

        with tlx.async_task(num_warps=1, num_regs=232):  # MMA consumer
            # Use compute_grid_info from tlx_kernel_utils
            start_pid, num_pid_m, num_pid_n, num_pid_in_group, _, k_tiles = compute_grid_info(
                M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
            )
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
                pid_m, pid_n = compute_tile_position(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am, offs_bn = compute_tile_offsets(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)

                # wait epilogue consumer to be done with the buffer before reusing it
                tlx.barrier_wait(tmem_empty_bars[cur_tmem_buf], tmem_write_phase)
                tmem_write_phase = flip_phase_on_boundary(tmem_write_phase, cur_tmem_buf, NUM_TMEM_BUFFERS)

                # now iterate along K to compute result for the block
                for k in range(0, k_tiles):
                    # processed_k_iters + k means we use the immediate next buffer slot of tile_id x when we start tile_id x+1
                    buf = (processed_k_iters + k) % NUM_SMEM_BUFFERS
                    # wait for current phase(round) of load for this buf
                    tlx.barrier_wait(smem_full_bars[buf], dot_phase)
                    # buffer is now ready with loaded data, tlx.async_dot will signal `mBarrier` when done
                    async_mma_with_barrier(
                        buffers_A[buf],
                        buffers_B[buf],
                        tmem_buffers[cur_tmem_buf],
                        use_acc=k > 0,
                        done_barrier=smem_empty_bars[buf],
                    )
                    dot_phase = flip_phase_on_boundary(dot_phase, buf, NUM_SMEM_BUFFERS)

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
                clc_phase = flip_phase_on_boundary(clc_phase, clc_buf, NUM_CLC_STAGES)
                clc_buf += 1

        with tlx.async_task(num_warps=1, num_regs=232):  # producer, TMA load
            # Use compute_grid_info from tlx_kernel_utils
            start_pid, num_pid_m, num_pid_n, num_pid_in_group, _, k_tiles = compute_grid_info(
                M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
            )

            load_phase = 0  # the current phase of TMA load
            # we virtually "flatten" the two layer loop as if we're performing tma loads on
            # one big list of data
            processed_k_iters = 0

            tile_id = start_pid
            clc_phase = 0
            clc_buf = 0
            while tile_id != -1:
                clc_buf = clc_buf % NUM_CLC_STAGES
                pid_m, pid_n = compute_tile_position(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am, offs_bn = compute_tile_offsets(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)

                for k in range(0, k_tiles):
                    # processed_k_iters + k means we use the immediate next buffer slot of tile_id x when we start tile_id x+1
                    buf = (processed_k_iters + k) % NUM_SMEM_BUFFERS
                    # wait for previous phase(round) of dot for this buf
                    tlx.barrier_wait(smem_empty_bars[buf], load_phase ^ 1)
                    # buffer is now ready to be used again
                    offs_k = k * BLOCK_SIZE_K
                    async_load_ab_tiles(
                        a_desc,
                        b_desc,
                        buffers_A[buf],
                        buffers_B[buf],
                        offs_am,
                        offs_k,
                        offs_bn,
                        smem_full_bars[buf],
                        BLOCK_SIZE_M,
                        BLOCK_SIZE_N,
                        BLOCK_SIZE_K,
                    )
                    load_phase = flip_phase_on_boundary(load_phase, buf, NUM_SMEM_BUFFERS)
                processed_k_iters += k_tiles
                tile_id = tlx.clc_consumer(clc_context, clc_buf, clc_phase)
                clc_phase = flip_phase_on_boundary(clc_phase, clc_buf, NUM_CLC_STAGES)
                clc_buf += 1


def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # Persistent kernel to have thread block resident in SM as long as possible
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel_tma_ws_blackwell_clc[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        NUM_SMS=NUM_SMS,  #
        NUM_CLC_STAGES=1,  #
        # launch_cluster=True,
    )

    return c


@pytest.mark.skipif(
    not is_blackwell(),
    reason="Requires Blackwell GPU",
)
def test_op():
    torch.manual_seed(0)
    M, N, K = 8192, 8192, 8192
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    torch_output = torch.matmul(a, b)
    triton_output = matmul(a, b)
    print(f"torch_output_with_fp16_inputs={torch_output}")
    print(f"triton_output_with_fp16_inputs={triton_output}")
    rtol = 0
    torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=rtol)


ref_lib = "cuBLAS"


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[ref_lib.lower(), "triton_clc"],  # Label name for the lines
        line_names=[ref_lib, "Triton (CLC)"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-" + ("fp16"),  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles, warmup=2000,
                                                     rep=2000)

    if provider == "triton_clc":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, True), quantiles=quantiles, warmup=2000,
                                                     rep=2000)

    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    if is_blackwell():
        print("Running benchmarks...")
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no Blackwell GPU found.")
