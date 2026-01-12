import pytest
from typing import Optional

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_blackwell

from tlx_kernel_utils import (
    compute_tile_position,
    init_tmem_accumulator,
    flip_phase_on_boundary,
)

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8, "NUM_STAGES" : s}) \
        for BM in [128] \
        for BN in [128, 256] \
        for BK in [64,128] \
        for s in ([2,4])
    ]


def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert align == 128
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device=DEVICE)


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_tma_pipelined_blackwell(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
                                          stride_bk, stride_bn,  #
                                          stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                          BLOCK_SIZE_K: tl.constexpr,  #
                                          GROUP_SIZE_M: tl.constexpr,  #
                                          NUM_STAGES: tl.constexpr  #
                                          ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    pid_m, pid_n = compute_tile_position(pid, num_pid_in_group, num_pid_m, GROUP_SIZE_M)

    # Initialize TMA descriptors
    desc_a = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    desc_b = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    desc_c = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    # allocate NUM_STAGES buffers
    buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float16, NUM_STAGES)
    buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tl.float16, NUM_STAGES)
    # allocate barriers
    dot_bars = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    load_bars = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    phase = 0

    # prefetch (pipelining) for NUM_STAGES - 1 buffers
    for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        a = tlx.local_view(buffers_A, i)
        b = tlx.local_view(buffers_B, i)
        load_bar = tlx.local_view(load_bars, i)
        tlx.barrier_expect_bytes(load_bar, 2 * (BLOCK_SIZE_M + BLOCK_SIZE_N) * BLOCK_SIZE_K)  # float16
        tlx.async_descriptor_load(desc_a, a, [pid_m * BLOCK_SIZE_M, i * BLOCK_SIZE_K], load_bar)
        tlx.async_descriptor_load(desc_b, b, [i * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N], load_bar)

    # main K loop - init accumulator to 0 (in TMEM)
    acc_tmem = init_tmem_accumulator(BLOCK_SIZE_M, BLOCK_SIZE_N)

    num_iter = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_iter):
        # identify the buffer index for the current iteration
        buf = k % NUM_STAGES
        a_k = tlx.local_view(buffers_A, buf)
        b_k = tlx.local_view(buffers_B, buf)

        # wait for buffers to be ready at `phase`
        load_bar = tlx.local_view(load_bars, buf)
        tlx.barrier_wait(load_bar, phase)

        # issue the async mma "with `phase`"
        dot_bar = tlx.local_view(dot_bars, buf)
        # mmav5 can take A and B from SMEM, and accumulate result into TMEM
        tlx.async_dot(a_k, b_k, acc_tmem, mBarriers=[dot_bar], out_dtype=tl.float32)

        # prefetch for i-th iteration, i.e, NUM_STAGES - 1 ahead
        i = k + NUM_STAGES - 1
        # wait for the previous iteration's MMA using the buffer to complete
        prev_dot_bar = tlx.local_view(dot_bars, i % NUM_STAGES)
        # if the previous MMA was issued in previous round of the buffers/barrier use, `phase` was flipped in last iteration,
        # meaning the previous MMA was issued "with `phase ^ 1`"
        prev_phase = phase ^ 1 if (i % NUM_STAGES == NUM_STAGES - 1) else phase
        # wait for dot op k-1 to complete before prefetching for its buffer for next time
        tlx.barrier_wait(prev_dot_bar, prev_phase)

        if i < num_iter:
            a_next = tlx.local_view(buffers_A, i % NUM_STAGES)
            b_next = tlx.local_view(buffers_B, i % NUM_STAGES)
            next_load_bar = tlx.local_view(load_bars, i % NUM_STAGES)
            # prefetch
            # if i % NUM_STAGES == NUM_STAGES - 1, we are prefetching for the buffer with current `phase`
            # otherwise, we are prefetching for the buffer with next phase (`phase ^ 1`)
            tlx.barrier_expect_bytes(next_load_bar, 2 * (BLOCK_SIZE_M + BLOCK_SIZE_N) * BLOCK_SIZE_K)  # float16
            tlx.async_descriptor_load(desc_a, a_next, [pid_m * BLOCK_SIZE_M, i * BLOCK_SIZE_K], next_load_bar)
            tlx.async_descriptor_load(desc_b, b_next, [i * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N], next_load_bar)

        phase = flip_phase_on_boundary(phase, buf, NUM_STAGES)

    # wait for last mma to complete
    i = num_iter - 1
    prev_dot_bar = tlx.local_view(dot_bars, i % NUM_STAGES)
    prev_phase = phase ^ 1 if (i % NUM_STAGES == NUM_STAGES - 1) else phase
    tlx.barrier_wait(prev_dot_bar, prev_phase)

    # load the result from TMEM to registers
    result = tlx.local_load(acc_tmem)
    c = result.to(tl.float16)

    # store the result to SMEM to prepare for TMA store (TMEM -> GMEM)
    c_buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float16, tl.constexpr(1))
    c_smem = tlx.local_view(c_buffers, 0)
    tlx.local_store(c_smem, c)
    tlx.async_descriptor_store(desc_c, c_smem, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N])


def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Initialize TMA descriptor storgae allocator
    triton.set_allocator(alloc_fn)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_tma_pipelined_blackwell[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c


@pytest.mark.skipif(
    not is_blackwell(),
    reason="Requires Blackwell GPU",
)
def test_op():
    torch.manual_seed(0)
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    torch_output = torch.matmul(a, b)
    triton_output = matmul(a, b)
    print(f"torch_output_with_fp16_inputs={torch_output}")
    print(f"triton_output_with_fp16_inputs={triton_output}")
    rtol = 0
    torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol)


ref_lib = 'cuBLAS'

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[ref_lib.lower(), "triton"],  # Label name for the lines
        line_names=[ref_lib, "Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-" + ("fp16"),  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    if is_blackwell():
        print("Running benchmarks...")
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no Blackwell GPU found.")
