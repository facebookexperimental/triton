"""Reduce-marked multi-CTA LayerNorm kernels for TLX testing.

The kernels explicitly partition columns by physical CTA rank and mark only
the CTA-local mean and variance reductions. The compiler forms the DSMEM
collective without rewriting loop bounds or store ownership.
"""

import torch

import triton
import triton.language as tl


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
