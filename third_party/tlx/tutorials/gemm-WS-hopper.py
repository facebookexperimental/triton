import pytest
import torch

import triton
import triton.language as tl
import triton.tlx.language as tlx
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

M, N, K = (8192, 8192, 8192)


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_cdna2():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert align == 128
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device=DEVICE)


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BM"]
    BLOCK_N = nargs["BN"]
    BLOCK_K = nargs["BK"]
    nargs["desc_in_1"].block_shape = [BLOCK_M // 2, BLOCK_K]
    nargs["desc_in_2"].block_shape = [BLOCK_K, BLOCK_N]
    nargs["desc_out"].block_shape = [BLOCK_M // 2, BLOCK_N]


@triton.autotune(
    configs=[
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64, 'NUM_STAGES': 2,}, num_warps=4, pre_hook=matmul_tma_set_block_size_hook),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32, 'NUM_STAGES': 4,}, num_warps=4, pre_hook=matmul_tma_set_block_size_hook),
    ],
    key=["K",],
)
@triton.jit
def matmul_kernel_tlx_ws(
    desc_in_1, desc_in_2, desc_out, #
    K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,  #
):
    # Descriptor
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offset_am = pid_m * BM
    offset_bn = pid_n * BN

    # Need NUM_STAGES sets of SMEM buffers for A and B
    # where each set contains two for A and one for B.
    # Split A into two in M-dimension to have two consumer tasks for wgmma
    a = tlx.local_alloc((BM // 2, BK), tl.float16, NUM_STAGES * 2)
    b = tlx.local_alloc((BK, BN), tl.float16, NUM_STAGES)

    # Need NUM_STAGES sets of mbarriers for A and B
    # where each set contains two for A and one for B.
    # Do the above for both empty states and full states respectively.
    bars_empty_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * 2,)
    bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * 2, arrive_count=1)
    bars_empty_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=2)
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    # Warp specilization
    with tlx.async_tasks():
        # Producer (async load)
        with tlx.async_task("default"):
            # Assuming NUM_STAGES = 2
            # p should be 1, 1, 0, 0, 1, 1, 0, 0, ...
            p = 1

            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES
                offset_k = k * BK

                # Async load to a[buf]
                empty_a_1st = tlx.local_view(bars_empty_a, buf)  # mbar
                full_a_1st = tlx.local_view(bars_full_a, buf)  # mbar
                tlx.barrier_wait(bar=empty_a_1st, phase=p)  # EmptyBar A1 wait
                tlx.barrier_expect_bytes(full_a_1st, (BM//2) * BK * 2)
                data_a_1st = tlx.local_view(a, buf)  # smem data
                tlx.async_descriptor_load(
                    desc_in_1,
                    data_a_1st,
                    [offset_am, offset_k],
                    full_a_1st)

                # Async load to b[buf]
                empty_b = tlx.local_view(bars_empty_b, buf)
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=empty_b, phase=p)
                tlx.barrier_expect_bytes(full_b, BN * BK * 2)
                data_b = tlx.local_view(b, buf)
                tlx.async_descriptor_load(
                    desc_in_2,
                    data_b,
                    [offset_k, offset_bn],
                    full_b)

                # Async load to a[buf+NUM_STAGES]
                empty_a_2nd = tlx.local_view(bars_empty_a, buf+NUM_STAGES)
                full_a_2nd = tlx.local_view(bars_full_a, buf+NUM_STAGES)
                tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                tlx.barrier_expect_bytes(bar=full_a_2nd, size=(BM//2) * BK * 2)
                data_a_2nd = tlx.local_view(a, buf+NUM_STAGES)  # smem data
                tlx.async_descriptor_load(
                    desc_in_1,
                    data_a_2nd,
                    [offset_am + (BM//2), offset_k],
                    full_a_2nd)

                # Flip phase after every NUM_STAGES iterations finish
                p = p ^ (buf == (NUM_STAGES-1))

        # consumers (wgmma + async store)
        with tlx.async_task(num_warps=4, replicate=2):
            p = 0
            # Assuming NUM_STAGES = 2
            # p should be 0, 0, 1, 1, 0, 0, ...
            acc = tl.zeros([BM//2, BN], dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES

                # Wait for TMA load
                full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=full_a, phase=p)
                tlx.barrier_wait(bar=full_b, phase=p)

                # async_dot
                data_a = tlx.local_view(a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
                data_b = tlx.local_view(b, buf)
                acc = tlx.async_dot(
                    data_a,
                    data_b,
                    acc,
                )
                # async_wait
                acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                # Release buffers
                empty_a = tlx.local_view(bars_empty_a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
                empty_b = tlx.local_view(bars_empty_b, buf)
                tlx.barrier_arrive(empty_a)  # EmptyBar A1 arrive
                tlx.barrier_arrive(empty_b)

                # Flip phase after every NUM_STAGES iterations finish
                p = p ^ (buf == (NUM_STAGES-1))

            desc_out.store([offset_am + (BM // 2) * tlx.async_task_replica_id(), offset_bn], acc.to(tlx.dtype_of(desc_out)))  # noqa


def matmul(a, b,):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])
    c = torch.zeros((M, N), dtype=torch.float16, device=DEVICE, )

    dummy_block = [1, 1]
    desc_in_1 = TensorDescriptor(
        a,
        shape=[M, K],
        strides=[K, 1],
        block_shape=dummy_block,
    )

    desc_in_2 = TensorDescriptor(
        b,
        shape=[K, N],
        strides=[N, 1],
        block_shape=dummy_block,
    )
    desc_out = TensorDescriptor(
        c,
        shape=[M, N],
        strides=[N, 1],
        block_shape=dummy_block,
    )

    grid = lambda META: (  # noqa E731
        triton.cdiv(M, META['BM']), triton.cdiv(N, META['BN']),
    )
    matmul_kernel_tlx_ws[grid](
        desc_in_1, desc_in_2, desc_out,  #
        K=K,
    )
    return c

triton.set_allocator(alloc_fn)

torch.manual_seed(0)

a = torch.randn((M, K), dtype=torch.float16, device=DEVICE)
b = torch.randn((K, N), dtype=torch.float16, device=DEVICE)

rtol = 1e-2 if is_hip_cdna2() else 0
output = matmul(a, b,)
output_ref = torch.matmul(a, b)

if torch.allclose(output, output_ref, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

TORCH_HAS_FP8 = False

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

# Benchmarking
configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True, diff_col=True)
