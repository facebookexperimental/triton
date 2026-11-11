"""
Multi-CTA Layer Normalization
==============================

This tutorial demonstrates reduce-marked multi-CTA collectives. The kernel
explicitly assigns rank-strided column tiles to each CTA, computes a CTA-local
partial, and marks only the final ``tl.sum`` with ``multi_cta=True``. The
compiler combines those partials across the physical cluster with DSMEM; it
does not rewrite loop bounds or stores.

.. note::
    Multi-CTA reduction requires SM90, SM100, or SM103 and ``ctas_per_cga``.
    CTAs cluster on dimension 1 so every participant shares the same row ID.
    The cluster-local rank is ``tl.program_id(1) % NUM_CTAS``.
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
# The kernel owns work partitioning. Only the CTA-local reduction is marked.


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
    NUM_CTAS: tl.constexpr,
):
    row = tl.program_id(0)
    local_rank = tl.program_id(1) % NUM_CTAS
    Y += row * stride
    X += row * stride

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            N,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += values
    mean = tl.sum(_mean, axis=0, multi_cta=True) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            N,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        values = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        centered = tl.where(mask, values - mean, 0.0)
        _var += centered * centered
    var = tl.sum(_var, axis=0, multi_cta=True) / N
    rstd = 1 / tl.sqrt(var + eps)

    is_leader = local_rank == 0
    tl.store(Mean + row, mean, mask=is_leader)
    tl.store(Rstd + row, rstd, mask=is_leader)

    for off in tl.range(
            local_rank * BLOCK_SIZE,
            N,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        values = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        output = (values - mean) * rstd * w + b
        tl.store(Y + cols, output, mask=mask)


# %%
# Multi-CTA Layer Norm with 2D Blocks
# -------------------------------------
# Each CTA handles ``BLOCK_SIZE_M`` rows while the physical cluster partitions
# columns. The marked axis-1 reductions produce replicated per-row tensors.


@triton.jit
def _layer_norm_fwd_multi_cta_2d(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride,
    M,
    N,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    pid = tl.program_id(0)
    local_rank = tl.program_id(1) % NUM_CTAS
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < M
    X += rows[:, None] * stride
    Y += rows[:, None] * stride

    _mean = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE_N,
            N,
            NUM_CTAS * BLOCK_SIZE_N,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        values = tl.load(X + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        _mean += values
    mean = tl.sum(_mean, axis=1, multi_cta=True) / N

    _var = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE_N,
            N,
            NUM_CTAS * BLOCK_SIZE_N,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        values = tl.load(X + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        centered = tl.where(mask, values - mean[:, None], 0.0)
        _var += centered * centered
    var = tl.sum(_var, axis=1, multi_cta=True) / N
    rstd = 1 / tl.sqrt(var + eps)

    leader_mask = row_mask & (local_rank == 0)
    tl.store(Mean + rows, mean, mask=leader_mask)
    tl.store(Rstd + rows, rstd, mask=leader_mask)

    for off in tl.range(
            local_rank * BLOCK_SIZE_N,
            N,
            NUM_CTAS * BLOCK_SIZE_N,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        w = tl.load(W + cols[None, :], mask=cols[None, :] < N)
        b = tl.load(B + cols[None, :], mask=cols[None, :] < N)
        values = tl.load(X + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        output = (values - mean[:, None]) * rstd[:, None] * w + b
        tl.store(Y + cols[None, :], output, mask=mask)


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
    if NUM_CTAS not in (1, 2, 4):
        raise ValueError(f"NUM_CTAS must be 1, 2, or 4; got {NUM_CTAS}")
    if N <= 0:
        raise ValueError("LayerNorm feature dimension must be positive")
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(triton.cdiv(N, NUM_CTAS)))
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
        NUM_CTAS=NUM_CTAS,
        num_warps=num_warps,
        ctas_per_cga=(1, NUM_CTAS, 1),
    )
    return y, mean, rstd


def multi_cta_layernorm_2d(x, weight, bias, eps=1e-5, NUM_CTAS=2, BLOCK_SIZE_M=4):
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    y = torch.empty_like(x)
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    if NUM_CTAS not in (1, 2, 4):
        raise ValueError(f"NUM_CTAS must be 1, 2, or 4; got {NUM_CTAS}")
    if N <= 0:
        raise ValueError("LayerNorm feature dimension must be positive")
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(triton.cdiv(N, NUM_CTAS)))
    num_warps = min(max(BLOCK_SIZE_N // 256, 1), 8)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), NUM_CTAS)
    _layer_norm_fwd_multi_cta_2d[grid](
        x_arg,
        y,
        weight,
        bias,
        mean,
        rstd,
        x_arg.stride(0),
        M,
        N,
        eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_CTAS=NUM_CTAS,
        num_warps=num_warps,
        ctas_per_cga=(1, NUM_CTAS, 1),
    )
    return y, mean, rstd


# %%
# Correctness Test
# -----------------


def test_multi_cta_layernorm(M=4, N=16381, dtype=torch.float16, eps=1e-5):
    torch.manual_seed(0)
    x = torch.randn(M, N, device=DEVICE, dtype=dtype)
    weight = torch.randn(N, device=DEVICE, dtype=dtype)
    bias = torch.randn(N, device=DEVICE, dtype=dtype)

    y_ref = torch.nn.functional.layer_norm(x, (N, ), weight, bias, eps)
    x_f32 = x.float()
    mean_ref = x_f32.mean(dim=1)
    rstd_ref = torch.rsqrt(((x_f32 - mean_ref[:, None])**2).mean(dim=1) + eps)

    for num_ctas in (1, 2, 4):
        y_tri, mean_tri, rstd_tri = multi_cta_layernorm(x, weight, bias, eps, NUM_CTAS=num_ctas)
        torch.testing.assert_close(y_tri, y_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(mean_tri, mean_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(rstd_tri, rstd_ref, rtol=1e-3, atol=1e-3)

    for num_ctas in (2, 4):
        y_tri, mean_tri, rstd_tri = multi_cta_layernorm_2d(x, weight, bias, eps, NUM_CTAS=num_ctas)
        torch.testing.assert_close(y_tri, y_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(mean_tri, mean_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(rstd_tri, rstd_ref, rtol=1e-3, atol=1e-3)


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
