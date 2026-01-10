import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_cuda, is_hip_cdna2, is_hip

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_hip_autotune_config_full():
    configs = [
        triton.Config({
            'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k, 'GROUP_SIZE_M': group_size_m,
            'NUM_STAGES': num_stages
        } | {'kpack': kpack, 'matrix_instr_nonkdim': mat_dim_non_k, 'waves_per_eu': waves_per_eu}, num_warps=num_warps)
        for block_m in [32, 64, 128, 256]
        for block_n in [32, 64, 128, 256]
        for block_k in [32, 64, 128]
        for group_size_m in [1, 2, 4, 8]
        for waves_per_eu in [0, 2, 4]
        for num_stages in [2]
        for num_warps in [4, 8]
        for mat_dim_non_k in [16]
        for kpack in [1, 2]
    ]
    return configs


def is_invalid_config(config, N, M, K, mfma):
    """
    Contains all of the configuration checks for prune_configs
    that will result in an invalid result if select as the config.

    This is done to ensure that if no config is "optimal" for a given
    shape we don't accidentally select
    """
    BLOCK_SIZE_M = config.kwargs.get("BLOCK_SIZE_M")
    BLOCK_SIZE_N = config.kwargs.get("BLOCK_SIZE_N")
    BLOCK_SIZE_K = config.kwargs.get("BLOCK_SIZE_K")
    matrix_instr_nonkdim = config.kwargs.get("matrix_instr_nonkdim")
    if matrix_instr_nonkdim > mfma:
        return True
    if mfma == 4 and BLOCK_SIZE_K < 64:
        return True
    # some layouts could not work properly in case
    # number elements per thread is less 1
    if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
        return True
    if BLOCK_SIZE_M < matrix_instr_nonkdim or BLOCK_SIZE_N < matrix_instr_nonkdim:
        return True
    if M <= matrix_instr_nonkdim and BLOCK_SIZE_M != matrix_instr_nonkdim:
        return True
    if N <= matrix_instr_nonkdim and BLOCK_SIZE_N != matrix_instr_nonkdim:
        return True
    return False


def prune_configs(configs, named_args, **kwargs):

    pruned_configs = []
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    elemBytes_a = named_args["a_ptr"].element_size()
    elemBytes_b = named_args["b_ptr"].element_size()

    if M < 32 or N < 32:
        mfma = 16
    else:
        mfma = 32

    for config in configs:
        BLOCK_SIZE_M = config.kwargs.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.kwargs.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.kwargs.get("BLOCK_SIZE_K")
        GROUP_SIZE_M = config.kwargs.get("GROUP_SIZE_M")
        if is_invalid_config(config, N, M, K, mfma):
            continue
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if BLOCK_SIZE_M > M * 2 and BLOCK_SIZE_M != 16:
            continue
        if BLOCK_SIZE_N > N * 2 and BLOCK_SIZE_N != 16:
            continue
        # skip large GROUP_SIZE_M
        if GROUP_SIZE_M * BLOCK_SIZE_M >= M and GROUP_SIZE_M != 1:
            continue
        # out of shared memory resource
        LDS = (BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a + BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b)
        if LDS > 65536:
            continue
        pruned_configs.append(config)

    print(f"{len(configs)=} {len(pruned_configs)=} for {M=} {N=} {K=}")
    if len(pruned_configs) == 0:
        raise RuntimeError("No valid configs left after pruning! Consider autotuning further with TritonBench")
    return pruned_configs


full_tune = False
hip_configs = [
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 2, 'NUM_STAGES': 2}
                  | {'kpack': 2, 'matrix_instr_nonkdim': 16, 'waves_per_eu': 0}, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'NUM_STAGES': 2}
                  | {'kpack': 2, 'matrix_instr_nonkdim': 16, 'waves_per_eu': 4}, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'NUM_STAGES': 2}
                  | {'kpack': 2, 'matrix_instr_nonkdim': 16, 'waves_per_eu': 2}, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'NUM_STAGES': 2}
                  | {'kpack': 1, 'matrix_instr_nonkdim': 16, 'waves_per_eu': 0}, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6, 'NUM_STAGES': 2}
                  | {'matrix_instr_nonkdim': 16, 'waves_per_eu': 0}, num_warps=8, num_stages=2)
]

configs = get_hip_autotune_config_full() if full_tune else hip_configs


@triton.autotune(
    prune_configs_by={
        "early_config_prune": prune_configs,
    },
    configs=configs,
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_pipelined_mi300(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
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
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # offset computation
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    K_ITERS = tl.cdiv(K, BLOCK_SIZE_K)

    # NUM_STAGES-1 because we use tl.load that buffers results in registers
    # In general, when using tl.load + local_store
    # num buffers = pipeline-stage(local-store) - pipeline-stage(local-load)
    NUM_BUFFERS = NUM_STAGES - 1
    buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_STAGES - 1)
    buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_STAGES - 1)

    # Pipeline Prologue. (NUM_STAGES - 1) iterations
    for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        a_smem_view = tlx.local_view(buffers_A, i)
        b_smem_view = tlx.local_view(buffers_B, i)
        a_load_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K)
        b_load_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K)
        tlx.local_store(a_smem_view, a_load_reg)
        tlx.local_store(b_smem_view, b_load_reg)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Pipeline Kernel Main Loop.
    # BLOCK_SIZE_K - (NUM_STAGES - 1) iterations
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Disable auto-pipelining with num_stages=0
    for k in tl.range(NUM_STAGES - 1, K_ITERS, num_stages=0):
        # prefetch data for k into regs, this is NUM_STAGES - 1 ahead of the k in the following tl.dot
        a_k_smem_view = tlx.local_view(buffers_A, k % NUM_BUFFERS)
        b_k_smem_view = tlx.local_view(buffers_B, k % NUM_BUFFERS)
        a_load_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_load_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K)

        # do compute on data fetched ahead by NUM_STAGES - 1
        buf = (k - NUM_STAGES - 1) % NUM_BUFFERS
        a_k_prev_shmem = tlx.local_view(buffers_A, buf)
        b_k_prev_shmem = tlx.local_view(buffers_B, buf)
        a_k_prev_reg = tlx.local_load(a_k_prev_shmem)
        b_k_prev_reg = tlx.local_load(b_k_prev_shmem)
        acc = tl.dot(a_k_prev_reg, b_k_prev_reg, acc)

        # store data for k from regs to shmem, this is NUM_STAGES - 1 ahead of the k in the prev tl.dot
        tlx.local_store(a_k_smem_view, a_load_reg)
        tlx.local_store(b_k_smem_view, b_load_reg)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Epilogue
    for k in tl.range(K_ITERS - (NUM_STAGES - 1), K_ITERS, loop_unroll_factor=NUM_STAGES - 1):
        # do compute on data fetched ahead by NUM_STAGES - 1 in Main Loop
        buf = k % NUM_BUFFERS
        a_k_prev_shmem = tlx.local_view(buffers_A, buf)
        b_k_prev_shmem = tlx.local_view(buffers_B, buf)
        a_k_prev_reg = tlx.local_load(a_k_prev_shmem)
        b_k_prev_reg = tlx.local_load(b_k_prev_shmem)
        acc = tl.dot(a_k_prev_reg, b_k_prev_reg, acc)

    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_pipelined_mi300[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c


@pytest.mark.skipif(
    not is_hip(),
    reason="Requires AMD GPU",
)
def test_op():
    torch.manual_seed(0)
    a = torch.randn((8192, 8192), device=DEVICE, dtype=torch.float16)
    b = torch.randn((8192, 8192), device=DEVICE, dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    rtol = 1e-2 if is_hip_cdna2() else 1e-4
    # TODO. rtol 1e-5 failed while 1e-4 passed on Hopper
    torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol)


TORCH_HAS_FP8 = False

# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of cuBLAS or rocBLAS. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[256, 512, 1024, 2048, 4096],  # Different possible values for `x_name`
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
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles, rep=1000)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles, rep=1000)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    print("Running benchmarks...")
    benchmark.run(print_data=True)
