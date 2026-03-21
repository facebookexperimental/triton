"""
Multi-CTA Layer Normalization
==============================

This tutorial demonstrates how to use ``multi_cta=True`` on ``tl.range`` to
automatically distribute a reduction across multiple CTAs in a cluster, enabling
efficient processing of large feature dimensions (N ≥ 4096).

When ``multi_cta=True`` is set on a loop and the kernel is launched with
``ctas_per_cga`` > (1,1,1), the Triton compiler automatically:

1. Partitions loop iterations across CTAs in the cluster
2. Performs a local partial reduction within each CTA
3. Exchanges partial results via Distributed Shared Memory (DSM)
4. Aggregates the final result across all CTAs

The user writes standard Triton code — the only change from a normal layernorm
kernel is adding ``multi_cta=True`` to the accumulation loops.

.. note::
    Multi-CTA reduction requires SM90+ (Hopper/Blackwell) GPUs and
    ``ctas_per_cga`` to be set in the kernel launch config.
    CTAs must cluster on dim 1 (not dim 0) so that all CTAs in a cluster
    share the same ``program_id(0)`` (row).
"""

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# %%
# Single-CTA Layer Norm (Baseline)
# ----------------------------------
# This is the standard layernorm kernel from tutorial 05, limited to N ≤ 32K.


@triton.jit
def _layer_norm_fwd_single_cta(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


# %%
# Multi-CTA Layer Norm
# ---------------------
# The **only** change: ``multi_cta=True`` on the three ``tl.range`` loops.
# The compiler automatically distributes the loop iterations across CTAs
# and aggregates reductions via DSM.


@triton.jit
def _layer_norm_fwd_multi_cta(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # Accumulate mean — distributed across CTAs
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    # Accumulate variance — distributed across CTAs
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Normalize — distributed across CTAs
    for off in tl.range(0, N, BLOCK_SIZE, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


# %%
# Wrapper Functions
# ------------------


def single_cta_layernorm(x, weight, bias, eps=1e-5):
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    y = torch.empty_like(x)
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("Single-CTA layernorm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _layer_norm_fwd_single_cta[(M, )](
        x_arg,
        y,
        weight,
        bias,
        mean,
        rstd,
        x_arg.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y, mean, rstd


def multi_cta_layernorm(x, weight, bias, eps=1e-5, NUM_CTAS=2):
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    y = torch.empty_like(x)
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    # Compute BLOCK_SIZE: must be power-of-2 and divide chunk = N//NUM_CTAS
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    chunk = N // NUM_CTAS
    while BLOCK_SIZE > chunk or chunk % BLOCK_SIZE != 0:
        BLOCK_SIZE //= 2
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # Grid dim 1 = NUM_CTAS: CTAs cluster on dim 1 so all CTAs in a
    # cluster share the same program_id(0) (row).
    _layer_norm_fwd_multi_cta[(M, NUM_CTAS)](
        x_arg,
        y,
        weight,
        bias,
        mean,
        rstd,
        x_arg.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        ctas_per_cga=(1, NUM_CTAS, 1),
    )
    return y, mean, rstd


# %%
# Correctness Test
# -----------------


def test_multi_cta_layernorm(M=4, N=16384, dtype=torch.float16, eps=1e-5):
    torch.manual_seed(0)
    x = torch.randn(M, N, device=DEVICE, dtype=dtype)
    weight = torch.randn(N, device=DEVICE, dtype=dtype)
    bias = torch.randn(N, device=DEVICE, dtype=dtype)

    # PyTorch reference
    y_ref = torch.nn.functional.layer_norm(x, (N, ), weight, bias, eps)

    # Test with different NUM_CTAS values
    for nc in [1, 2, 4]:
        y_tri, _, _ = multi_cta_layernorm(x, weight, bias, eps, NUM_CTAS=nc)
        max_diff = torch.max(torch.abs(y_ref - y_tri)).item()
        passed = torch.allclose(y_ref, y_tri, rtol=1e-2, atol=1e-2)
        status = "✓" if passed else "✗"
        print(f"  {status} [M={M}, N={N}, NUM_CTAS={nc}] Max diff: {max_diff}")
        assert passed, f"Mismatch with NUM_CTAS={nc}: max diff = {max_diff}"
    print("✓ Correctness test passed for all NUM_CTAS values!")


# %%
# Benchmark
# ----------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 16)],
        x_log=True,
        line_arg="provider",
        line_vals=["single_cta", "multi_cta_2", "multi_cta_4", "torch"],
        line_names=["Single-CTA Triton", "Multi-CTA (2)", "Multi-CTA (4)", "PyTorch"],
        styles=[("blue", "-"), ("red", "-"), ("orange", "--"), ("green", "-")],
        ylabel="GB/s",
        plot_name="multi-cta-layernorm-performance",
        args={"M": 1152},
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
    weight = torch.randn(N, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(N, device=DEVICE, dtype=torch.float16)
    eps = 1e-5

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.layer_norm(x, (N, ), weight, bias, eps), quantiles=quantiles)
    elif provider == "single_cta":
        if N > 32768:  # fp16 limit for single CTA
            return 0, 0, 0
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: single_cta_layernorm(x, weight, bias, eps),
                                                     quantiles=quantiles)
    elif provider == "multi_cta_2":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: multi_cta_layernorm(x, weight, bias, eps, NUM_CTAS=2),
                                                     quantiles=quantiles)
    elif provider == "multi_cta_4":
        if N < 4 * 256:  # Need at least 256 elements per CTA
            return 0, 0, 0
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: multi_cta_layernorm(x, weight, bias, eps, NUM_CTAS=4),
                                                     quantiles=quantiles)

    total_bytes = (
        x.numel() * x.element_size() * 2 + weight.numel() * weight.element_size() + bias.numel() * bias.element_size() +
        M * 4 * 2  # mean and rstd (float32)
    )
    gbps = lambda ms: total_bytes * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_multi_cta_layernorm()
    benchmark.run(print_data=True, show_plots=True)
