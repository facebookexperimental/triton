"""
Multi-CTA Layer Normalization kernels (importable module for testing).

Provides both 1D (one row per CTA) and 2D (BLOCK_SIZE_M rows per CTA) variants
of the multi-CTA layer normalization kernel. The compiler MultiCTAReduction pass
automatically partitions loop iterations across CTAs and generates cross-CTA
DSM exchange for reduction results.
"""

import torch

import triton
import triton.language as tl

# =============================================================================
# 1D variant: one row per CTA, BLOCK_SIZE columns per iteration
# =============================================================================


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

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

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

    for off in tl.range(0, N, BLOCK_SIZE, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


def multi_cta_layernorm(x, weight, bias, eps=1e-5, NUM_CTAS=2):
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    y = torch.empty_like(x)
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    chunk = N // NUM_CTAS
    while BLOCK_SIZE > chunk or chunk % BLOCK_SIZE != 0:
        BLOCK_SIZE //= 2
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
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


# =============================================================================
# 2D variant: BLOCK_SIZE_M rows per CTA, BLOCK_SIZE_N columns per iteration
# =============================================================================


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
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < M
    X += rows[:, None] * stride
    Y += rows[:, None] * stride

    _mean = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE_N, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        a = tl.load(X + cols[None, :], mask=mask, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N

    _var = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE_N, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        x = tl.load(X + cols[None, :], mask=mask, other=0.).to(tl.float32)
        x = tl.where(mask, x - mean[:, None], 0.)
        _var += x * x
    var = tl.sum(_var, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + rows, mean, mask=row_mask)
    tl.store(Rstd + rows, rstd, mask=row_mask)

    for off in tl.range(0, N, BLOCK_SIZE_N, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        w = tl.load(W + cols[None, :], mask=cols[None, :] < N)
        b = tl.load(B + cols[None, :], mask=cols[None, :] < N)
        x = tl.load(X + cols[None, :], mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean[:, None]) * rstd[:, None]
        y = x_hat * w + b
        tl.store(Y + cols[None, :], y, mask=mask)


def multi_cta_layernorm_2d(x, weight, bias, eps=1e-5, NUM_CTAS=2, BLOCK_SIZE_M=4):
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    y = torch.empty_like(x)
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    chunk = N // NUM_CTAS
    while BLOCK_SIZE_N > chunk or chunk % BLOCK_SIZE_N != 0:
        BLOCK_SIZE_N //= 2
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
        num_warps=num_warps,
        ctas_per_cga=(1, NUM_CTAS, 1),
    )
    return y, mean, rstd
