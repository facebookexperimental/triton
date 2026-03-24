"""
Multi-CTA Layer Normalization
=============================

This tutorial demonstrates a multi-CTA (Cooperative Thread Array) implementation
of Layer Normalization using TLX primitives. The kernel distributes the reduction
across multiple CTAs within a cluster, enabling efficient processing of large
feature dimensions.

Key TLX features demonstrated:
- Cluster-level synchronization with `tlx.cluster_cta_rank()` and `tlx.cluster_barrier()`
- Local shared memory allocation with `tlx.local_alloc()`
- Cross-CTA communication with `tlx.async_remote_shmem_store()`
- Barrier-based synchronization with `tlx.alloc_barriers()` and `tlx.barrier_wait()`
- Async memory operations with `tlx.async_load()` and `tlx.async_load_wait_group()`
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from torch._inductor.runtime.triton_compat import libdevice
from triton._internal_testing import is_blackwell

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def compute_multi_cta_sum(
    x,
    cta_cluster_rank,
    barrier,
    phase,
    BLOCK_SIZE_M: tl.constexpr,
    num_reduction_ctas: tl.constexpr,
):
    dtype_x = tlx.dtype_of(x)
    local_buff = tlx.local_alloc((BLOCK_SIZE_M, 1), dtype_x, num_reduction_ctas)

    local_partial_sum = tl.sum(x, axis=1, keep_dims=True)
    # store local sum to shmem and read it back in cluster_rank order
    # in the second (final_sum) loop. This is required to preserve
    # preserve the order of the reduction, without using a branch in
    # the final_sum loop.
    tlx.local_store(local_buff[cta_cluster_rank], local_partial_sum)

    for i in tl.static_range(num_reduction_ctas):
        if cta_cluster_rank != i:
            tlx.async_remote_shmem_store(
                dst=local_buff[cta_cluster_rank],
                src=local_partial_sum,
                remote_cta_rank=i,
                barrier=barrier,
            )
    tlx.barrier_wait(barrier, phase=phase)

    final_sum = tl.zeros((BLOCK_SIZE_M, 1), dtype=dtype_x)
    for i in tl.static_range(num_reduction_ctas):
        remote_local_buff_view = tlx.local_view(local_buff, i)
        final_sum += tlx.local_load(remote_local_buff_view)
    return final_sum


# Autotune configs - BLOCK_SIZE_N and masking flags are computed during config pruning.
# NOTE: We cannot use @triton.heuristics decorator in triton_pytest targets
# because Buck's bytecode precompilation breaks inspect.getsourcelines().
# Instead, we compute heuristics in the prune_and_update_configs function.

# Generate base configs (with placeholder values that will be updated by prune_configs)
kernel_configs_multi_cta = [
    triton.Config(
        {
            "BLOCK_SIZE_M": m,
            "BLOCK_SIZE_N": 8192,
            "num_reduction_ctas": ctas,
            "SHOULD_MASK_ROW": False,
            "SHOULD_MASK_COL": False,
        },
        num_warps=nw,
        ctas_per_cga=(1, ctas, 1),
    ) for m in [1, 2] for nw in [4, 8, 16, 32] for ctas in [2, 4, 8]
]


def prune_and_update_configs(configs, named_args, **kwargs):
    """Prune invalid configs and update heuristic values."""
    N = kwargs["N"]
    M = kwargs["M"]

    pruned_configs = []
    for conf in configs:
        num_ctas = conf.kwargs.get("num_reduction_ctas")
        block_size_m = conf.kwargs.get("BLOCK_SIZE_M")

        # Compute BLOCK_SIZE_N using the same formula as @triton.heuristics
        blocksize_n = triton.next_power_of_2(N // num_ctas)

        # Skip if rounding up reduces num_ctas (tail CTAs won't have work)
        if triton.cdiv(N, blocksize_n) != num_ctas:
            continue

        # cp.async does not support transfers smaller than 4 bytes per thread
        element_size = 2  # float16
        num_threads = conf.num_warps * 32
        bytes_per_thread = (block_size_m * blocksize_n * element_size) // num_threads
        if bytes_per_thread < 4:
            continue

        # Update the config with computed values
        conf.kwargs["BLOCK_SIZE_N"] = blocksize_n
        conf.kwargs["SHOULD_MASK_ROW"] = M % block_size_m != 0
        conf.kwargs["SHOULD_MASK_COL"] = N % blocksize_n != 0

        pruned_configs.append(conf)

    return pruned_configs


@triton.autotune(
    configs=kernel_configs_multi_cta,
    prune_configs_by={"early_config_prune": prune_and_update_configs},
    key=["M", "N"],
)
@triton.jit
def kernel_layernorm_multi_cta(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean_out,  # pointer to the mean
    Rstd_out,  # pointer to the 1/std
    row_stride,  # input row stride
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    num_reduction_ctas: tl.constexpr,
    SHOULD_MASK_ROW: tl.constexpr,
    SHOULD_MASK_COL: tl.constexpr,
):
    cta_cluster_rank = tlx.cluster_cta_rank()
    COMPUTE_DTYPE = tl.float32

    # alloc buffers for staging
    x_buffer = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), X.dtype.element_ty, 1)
    x_buf = tlx.local_view(x_buffer, 0)

    # alloc barriers for synchronizing remote stores
    barriers = tlx.alloc_barriers(num_barriers=2)
    cross_cta_reduction_expected_bytes: tl.constexpr = (BLOCK_SIZE_M * tlx.size_of(COMPUTE_DTYPE) *
                                                        (num_reduction_ctas - 1))
    tlx.barrier_expect_bytes(
        barriers[0],
        size=cross_cta_reduction_expected_bytes,
    )
    tlx.barrier_expect_bytes(
        barriers[1],
        size=cross_cta_reduction_expected_bytes,
    )
    tlx.cluster_barrier()

    # offsets
    row_offsets = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    read_write_offsets = (row_offsets[:, None] * row_stride) + col_offsets[None, :]
    x_ptrs = X + read_write_offsets
    y_ptrs = Y + read_write_offsets
    w_ptrs = W + col_offsets
    b_ptrs = B + col_offsets

    # mask calculation
    mask_row = None
    if SHOULD_MASK_ROW:
        mask_row = row_offsets < M
    else:
        if SHOULD_MASK_COL:
            mask_row = tl.full([BLOCK_SIZE_M], True, dtype=tl.int1)

    mask_col = None
    if SHOULD_MASK_COL:
        mask_col = col_offsets < N
    else:
        if SHOULD_MASK_ROW:
            mask_col = tl.full([BLOCK_SIZE_N], True, dtype=tl.int1)

    read_write_mask = None
    SHOULD_MASK: tl.constexpr = SHOULD_MASK_ROW or SHOULD_MASK_COL
    if SHOULD_MASK:
        read_write_mask = mask_row[:, None] & mask_col[None, :]
    other = 0.0 if SHOULD_MASK else None

    # async load x
    token_x = tlx.async_load(x_ptrs, x_buf, mask=read_write_mask, other=other)
    tlx.async_load_commit_group([token_x])
    tlx.async_load_wait_group(0)
    x = tlx.local_load(x_buf).to(COMPUTE_DTYPE)

    # N dim reduction across multiple CTAs
    # to compute sum
    multi_cta_sum = compute_multi_cta_sum(
        x,
        cta_cluster_rank,
        barriers[0],
        phase=0,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        num_reduction_ctas=num_reduction_ctas,
    )
    mean = multi_cta_sum / N
    if SHOULD_MASK:
        x_minus_mean = tl.where(read_write_mask, x - mean, 0.0)
    else:
        x_minus_mean = x - mean
    x_minus_mean_sq = x_minus_mean * x_minus_mean

    # N dim reduction across multiple CTAs
    # to compute reduction of (x - mean)^2
    multi_cta_sum_x_minus_mean_sq = compute_multi_cta_sum(
        x_minus_mean_sq,
        cta_cluster_rank,
        barriers[1],
        phase=0,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        num_reduction_ctas=num_reduction_ctas,
    )
    var = multi_cta_sum_x_minus_mean_sq / N
    rstd = libdevice.rsqrt(var + eps)
    mean_1d = tl.reshape(mean, (BLOCK_SIZE_M, ))
    tl.store(Mean_out + row_offsets, mean_1d, mask=mask_row)

    rstd_1d = tl.reshape(rstd, (BLOCK_SIZE_M, ))

    w = tl.load(w_ptrs, mask=mask_col).to(COMPUTE_DTYPE)
    b = tl.load(b_ptrs, mask=mask_col).to(COMPUTE_DTYPE)
    tl.store(Rstd_out + row_offsets, rstd_1d, mask=mask_row)

    x = tlx.local_load(x_buffer[0]).to(COMPUTE_DTYPE)

    x_hat = (x - mean) * rstd
    y = x_hat * w + b
    y = tl.cast(y, y_ptrs.dtype.element_ty)
    tl.store(y_ptrs, y, mask=read_write_mask)


def multi_cta_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    TLX Multi-CTA Layer Normalization Forward Pass.

    Args:
        x: Input tensor of shape [*, N] where * is any number of leading dimensions
        weight: Weight tensor of shape [N]
        bias: Bias tensor of shape [N]
        eps: Small epsilon for numerical stability

    Returns:
        out: Normalized output of same shape as input
        mean: Mean tensor of shape [M] where M is the product of leading dimensions
        rstd: Reciprocal standard deviation of shape [M]
    """
    original_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    m, n = x.size()

    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {n}"

    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    mean = torch.empty([m], dtype=torch.float32, device=x.device)
    rstd = torch.empty([m], dtype=torch.float32, device=x.device)

    def grid_2d(meta):
        return (
            triton.cdiv(m, meta["BLOCK_SIZE_M"]),
            triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

    kernel_layernorm_multi_cta[grid_2d](
        X=x,
        Y=out,
        W=weight,
        B=bias,
        Mean_out=mean,
        Rstd_out=rstd,
        row_stride=x.stride(0),
        M=m,
        N=n,
        eps=eps,
    )

    out = out.view(original_shape)
    return out, mean, rstd


def _torch_layernorm_impl(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """Reference PyTorch implementation of layer normalization."""
    return torch.nn.functional.layer_norm(x, (x.shape[-1], ), weight, bias, eps)


torch_layernorm = torch.compile(_torch_layernorm_impl)


@pytest.mark.skipif(
    not is_blackwell(),
    reason="Requires Blackwell GPU for multi-CTA support",
)
@pytest.mark.parametrize(
    "M,N",
    [(4, 16384), (4, 32768), (4, 65536), (1152, 16384), (1152, 32768), (1152, 65536)],
)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_op(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, device=DEVICE, dtype=dtype)
    weight = torch.randn(N, device=DEVICE, dtype=dtype)
    bias = torch.randn(N, device=DEVICE, dtype=dtype)
    eps = 1e-5

    # PyTorch reference
    output_torch = torch_layernorm(x, weight, bias, eps)

    # TLX implementation
    output_triton, mean_triton, rstd_triton = multi_cta_layernorm(x, weight, bias, eps)

    # Check output
    rtol = 1e-2 if dtype == torch.float16 else 1e-3
    atol = 1e-2 if dtype == torch.float16 else 1e-3

    max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
    print(f"[M={M}, N={N}, dtype={dtype}] Max difference: {max_diff}")

    assert torch.allclose(output_torch, output_triton, rtol=rtol,
                          atol=atol), (f"Output mismatch: max diff = {max_diff}")


# %%
# Benchmark
# ---------
#
# We benchmark our multi-CTA layer normalization kernel against PyTorch's native
# implementation across various tensor sizes.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(9, 15)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["TLX", "PyTorch"],  # Label name for the lines.
        styles=[("blue", "-"), ("red", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="multi-cta-layernorm-performance",  # Name for the plot.
        args={"M": 1024},  # Fixed arguments.
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
    weight = torch.randn(N, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(N, device=DEVICE, dtype=torch.float16)
    eps = 1e-5

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_layernorm(x, weight, bias, eps), quantiles=quantiles)
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: multi_cta_layernorm(x, weight, bias, eps),
                                                     quantiles=quantiles)

    # Calculate bandwidth: read x, weight, bias; write output, mean, rstd
    total_bytes = (
        x.numel() * x.element_size() * 2  # read x, write output
        + weight.numel() * weight.element_size()  # read weight
        + bias.numel() * bias.element_size()  # read bias
        + M * 4 * 2  # write mean and rstd (float32)
    )
    gbps = lambda ms: total_bytes * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    if is_blackwell():
        test_op(M=64, N=2048, dtype=torch.float16)
        benchmark.run(print_data=True, show_plots=True)
    else:
        print("Skipping: Multi-CTA layer normalization requires Blackwell GPU")
