import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

M, N, K = (2176, 2176, 2176)


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
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N]


def matmul_get_configs():
    return [
        triton.Config({'BM': BM, 'BN': BN, "BK": BK, "GROUP_SIZE_M": 8, "NUM_STAGES": num_stage,
                "NUM_MMA_WARPS": 8,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": True,},
                      num_stages=0, num_warps=4, pre_hook=matmul_tma_set_block_size_hook) \
        for BM, BN in [(128, 256), (256, 128)] \
        for BK in [64] \
        for num_stage in [3, 4]
    ]


@triton.autotune(
    # Autotune configs can be reused or adapted
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@triton.jit
def matmul_kernel_tlx_ws_persistent(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
):
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    a = tlx.local_alloc((BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS)
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)
    bars_empty_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_empty_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS)
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    with tlx.async_tasks():
        # Producer (async load)
        with tlx.async_task("default"):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_tiles = num_pid_m * num_pid_n
            num_pid_in_group = GROUP_SIZE_M * num_pid_n

            p = 1
            buf = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m

                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                for k in range(0, tl.cdiv(K, BK)):
                    offset_k = k * BK

                    # Async load to a[buf]
                    empty_a_1st = tlx.local_view(bars_empty_a, buf)
                    full_a_1st = tlx.local_view(bars_full_a, buf)
                    tlx.barrier_wait(bar=empty_a_1st, phase=p)
                    tlx.barrier_expect_bytes(full_a_1st, BLOCK_M_SPLIT * BK * 2)
                    data_a_1st = tlx.local_view(a, buf)
                    tlx.async_descriptor_load(a_desc, data_a_1st, [offset_am, offset_k], full_a_1st)

                    # Async load to b[buf]
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p)
                    tlx.barrier_expect_bytes(full_b, BN * BK * 2)
                    data_b = tlx.local_view(b, buf)
                    tlx.async_descriptor_load(b_desc, data_b, [offset_k, offset_bn], full_b)

                    # Async load to a[buf+NUM_STAGES]
                    empty_a_2nd = tlx.local_view(bars_empty_a, buf + NUM_STAGES)
                    full_a_2nd = tlx.local_view(bars_full_a, buf + NUM_STAGES)
                    tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                    tlx.barrier_expect_bytes(bar=full_a_2nd, size=BLOCK_M_SPLIT * BK * 2)
                    data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)
                    tlx.async_descriptor_load(a_desc, data_a_2nd, [offset_am + BLOCK_M_SPLIT, offset_k], full_a_2nd)

                    p = p ^ (buf == (NUM_STAGES - 1))
                    buf = (buf + 1) % NUM_STAGES

        # Consumers (wgmma + async store)
        with tlx.async_task(num_warps=4, replicate=2, registers=232):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_tiles = num_pid_m * num_pid_n
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            cid: tl.constexpr = tlx.async_task_replica_id()

            p = 0
            buf = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m

                offset_am = pid_m * BM
                offset_bn = pid_n * BN
                last_buf = buf
                full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * cid)
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=full_a, phase=p)
                tlx.barrier_wait(bar=full_b, phase=p)

                data_a = tlx.local_view(a, buf + NUM_STAGES * cid)
                data_b = tlx.local_view(b, buf)

                acc = tlx.async_dot(data_a, data_b)

                p = p ^ (buf == (NUM_STAGES - 1))
                buf = (buf + 1) % NUM_STAGES

                for k in range(1, tl.cdiv(K, BK)):

                    full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * cid)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    data_a = tlx.local_view(a, buf + NUM_STAGES * cid)
                    data_b = tlx.local_view(b, buf)

                    acc = tlx.async_dot(data_a, data_b, acc)
                    acc = tlx.async_dot_wait(1, acc)

                    empty_a = tlx.local_view(bars_empty_a, last_buf + NUM_STAGES * cid)
                    empty_b = tlx.local_view(bars_empty_b, last_buf)
                    tlx.barrier_arrive(empty_a)
                    tlx.barrier_arrive(empty_b)

                    last_buf = buf
                    p = p ^ (buf == (NUM_STAGES - 1))
                    buf = (buf + 1) % NUM_STAGES

                offset_cm = offset_am + BLOCK_M_SPLIT * cid

                acc = tlx.async_dot_wait(0, acc)
                empty_a = tlx.local_view(bars_empty_a, last_buf + NUM_STAGES * cid)
                empty_b = tlx.local_view(bars_empty_b, last_buf)
                tlx.barrier_arrive(empty_a)
                tlx.barrier_arrive(empty_b)

                if EPILOGUE_SUBTILE:
                    acc = tl.reshape(acc, (BLOCK_M_SPLIT, 2, BN // 2))
                    acc = tl.permute(acc, (0, 2, 1))
                    acc0, acc1 = tl.split(acc)
                    c0 = acc0.to(tlx.dtype_of(c_desc))
                    c_desc.store([offset_cm, offset_bn], c0)
                    c1 = acc1.to(tlx.dtype_of(c_desc))
                    c_desc.store([offset_cm, offset_bn + BN // 2], c1)
                else:
                    c_desc.store([offset_cm, offset_bn], acc.to(tlx.dtype_of(c_desc)))


def matmul_tlx_ws_persistent(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])
    c = torch.zeros((M, N), dtype=torch.float16, device=DEVICE)

    NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count

    dummy_block = [1, 1]
    desc_in_1 = TensorDescriptor(a, shape=[M, K], strides=[K, 1], block_shape=dummy_block)
    desc_in_2 = TensorDescriptor(b, shape=[K, N], strides=[N, 1], block_shape=dummy_block)
    desc_out = TensorDescriptor(c, shape=[M, N], strides=[N, 1], block_shape=dummy_block)

    def grid(META):
        num_m_blocks = triton.cdiv(M, META['BM'])
        num_n_blocks = triton.cdiv(N, META['BN'])
        total_blocks = num_m_blocks * num_n_blocks
        return (min(NUM_SMS, total_blocks), )

    matmul_kernel_tlx_ws_persistent[grid](
        desc_in_1,
        desc_in_2,
        desc_out,
        M,
        N,
        K,
        NUM_SMS=NUM_SMS,
    )
    return c


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires Hopper GPU",
)
def test_op():
    triton.set_allocator(alloc_fn)

    torch.manual_seed(0)

    a = torch.randn((M, K), dtype=torch.float16, device=DEVICE)
    b = torch.randn((K, N), dtype=torch.float16, device=DEVICE)

    rtol = 1e-2 if is_hip_cdna2() else 0
    output = matmul_tlx_ws_persistent(
        a,
        b,
    )
    output_ref = torch.matmul(a, b)

    torch.allclose(output, output_ref, atol=1e-2, rtol=rtol)

    print("Test passed!")


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
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles, warmup=200,
                                                     rep=200)
    if provider == 'triton':
        _ = matmul_tlx_ws_persistent(a, b)  # run to compile
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_tlx_ws_persistent(a, b), quantiles=quantiles,
                                                     warmup=200, rep=200)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    test_op()
    if is_cuda() and torch.cuda.get_device_capability()[0] == 9:
        print("Running benchmarks...")
        benchmark.run(show_plots=True, print_data=True, diff_col=True)
    else:
        print("Skipping benchmarks, no Hopper GPU found.")
