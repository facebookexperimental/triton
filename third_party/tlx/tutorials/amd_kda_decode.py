"""Generic indexed Kimi Delta Attention decode for gfx950.

The kernel consumes normalized Q/K, per-token log decays, and sigmoid-applied
beta coefficients. It processes at most one packed token per graph row and
directly reads/writes an FP32, V-major ``[slot, H, V, K]`` recurrent-state
pool.

Read and write indices are independent.  A negative or out-of-range read
produces zero output and suppresses the update; a negative or out-of-range
write suppresses only the state store.  Empty ``cu_seqlens`` rows produce zero
output and never mutate the pool.  State slots may have padding between them,
but each slot's ``[H, V, K]`` payload must be contiguous.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import tlx


@triton.jit
def _kda_recurrent_decode_kernel(
    Q,
    KIn,
    VIn,
    G,
    Beta,
    StatePool,
    ReadIndices,
    WriteIndices,
    Output,
    CuSeqLens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    SCALE: tl.constexpr,
    Q_TOKEN_STRIDE: tl.constexpr,
    K_TOKEN_STRIDE: tl.constexpr,
    V_TOKEN_STRIDE: tl.constexpr,
    G_TOKEN_STRIDE: tl.constexpr,
    BETA_TOKEN_STRIDE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    TOKENS: tl.constexpr,
    STATE_SLOT_STRIDE: tl.constexpr,
):
    value_block = tl.program_id(0)
    sequence_head = tl.program_id(1)
    sequence = sequence_head // H
    head = sequence_head % H

    begin = tl.load(CuSeqLens + sequence).to(tl.int32)
    end = tl.load(CuSeqLens + sequence + 1).to(tl.int32)
    active = (end == begin + 1) & (begin >= 0) & (begin < TOKENS)
    read_index = tl.load(ReadIndices + sequence).to(tl.int32)
    write_index = tl.load(WriteIndices + sequence).to(tl.int32)
    valid_read = active & (read_index >= 0) & (read_index < NUM_SLOTS)
    valid_write = valid_read & (write_index >= 0) & (write_index < NUM_SLOTS)
    safe_read = tl.where(valid_read, read_index, 0)
    safe_write = tl.where(valid_write, write_index, 0)

    key_offsets = tl.arange(0, BK)
    value_offsets = value_block * BV + tl.arange(0, BV)
    key_mask = key_offsets < K
    value_mask = value_offsets < V
    token = begin

    q = tl.load(
        Q + token * Q_TOKEN_STRIDE + head * K + key_offsets,
        mask=valid_read & key_mask,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        KIn + token * K_TOKEN_STRIDE + head * K + key_offsets,
        mask=valid_read & key_mask,
        other=0.0,
    ).to(tl.float32)
    log_decay = tl.load(
        G + token * G_TOKEN_STRIDE + head * K + key_offsets,
        mask=valid_read & key_mask,
        other=0.0,
    ).to(tl.bfloat16).to(tl.float32)
    decay = tl.exp(log_decay)
    beta = tl.load(
        Beta + token * BETA_TOKEN_STRIDE + head,
        mask=valid_read,
        other=0.0,
    ).to(tl.float32)

    q *= SCALE

    read_base = safe_read * STATE_SLOT_STRIDE + head * V * K
    state_offsets = value_offsets[:, None] * K + key_offsets[None, :]
    state_mask = valid_read & value_mask[:, None] & key_mask[None, :]
    state_buffer = tlx.local_alloc((BV, BK), StatePool.dtype.element_ty, 1)
    state_token = tlx.async_load(
        StatePool + read_base + state_offsets,
        tlx.local_view(state_buffer, 0),
        mask=state_mask,
    )
    tlx.async_load_commit_group([state_token])

    value = tl.load(
        VIn + token * V_TOKEN_STRIDE + head * V + value_offsets,
        mask=valid_read & value_mask,
        other=0.0,
    ).to(tl.float32)
    wait_token = tlx.async_load_wait_group(0)
    running = tlx.local_load(tlx.local_view(state_buffer, 0), token=wait_token).to(tl.float32)
    running = tl.where(state_mask, running, 0.0)
    running *= decay[None, :]
    prediction = tl.sum(running * k[None, :], axis=1)
    delta = beta * (value - prediction)
    running += delta[:, None] * k[None, :]
    result = tl.sum(running * q[None, :], axis=1)

    output_offsets = (sequence * H + head) * V + value_offsets
    tl.store(
        Output + output_offsets,
        tl.where(valid_read, result, 0.0).to(Output.dtype.element_ty),
        mask=value_mask,
    )
    write_base = safe_write * STATE_SLOT_STRIDE + head * V * K
    tl.store(
        StatePool + write_base + state_offsets,
        running,
        mask=valid_write & value_mask[:, None] & key_mask[None, :],
    )


def _validate_decode_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[int, int, int]:
    tensors = (
        q,
        k,
        v,
        g,
        beta,
        state_pool,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gfx950 TLX KDA decode requires GPU tensors")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("KDA data tensors must be on the same GPU")
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("q must have shape [1, batch, H, K]")
    if q.shape != k.shape or q.shape != g.shape:
        raise ValueError("q, k, and g must have identical shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("v must match q through the head dimension")
    _, batch, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if not (1 <= key_dim <= 128 and 1 <= value_dim <= 128):
        raise ValueError("gfx950 TLX KDA decode supports 1 <= K,V <= 128")
    if beta.shape != (1, batch, heads):
        raise ValueError("beta must have shape [1, batch, H]")
    if q.dtype != torch.bfloat16 or any(t.dtype != q.dtype for t in (k, v)):
        raise ValueError("q, k, and v must be BF16")
    if not g.dtype.is_floating_point or not beta.dtype.is_floating_point:
        raise ValueError("g and beta must use floating-point dtypes")
    if state_pool.dtype != torch.float32:
        raise ValueError("state_pool must be FP32")
    if state_pool.ndim != 4 or state_pool.shape[1:] != (heads, value_dim, key_dim):
        raise ValueError("state_pool must have V-major shape [slots, H, V, K]")
    if state_pool.stride()[1:] != (value_dim * key_dim, key_dim, 1):
        raise ValueError("state_pool inner [H, V, K] dimensions must be contiguous")
    if state_pool.stride(0) < heads * value_dim * key_dim:
        raise ValueError("state_pool slots must not overlap")
    if read_indices.ndim != 1 or read_indices.shape != write_indices.shape:
        raise ValueError("read_indices and write_indices must be matching vectors")
    if read_indices.numel() != batch:
        raise ValueError("decode requires one graph row per index")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() != batch + 1:
        raise ValueError("cu_seqlens must contain one boundary per graph row")
    for tensor, width in ((q, key_dim), (k, key_dim), (g, key_dim), (v, value_dim)):
        if tensor.stride(-1) != 1 or tensor.stride(-2) != width:
            raise ValueError("KDA input head vectors must be contiguous")
    if beta.stride(-1) != 1:
        raise ValueError("beta must have contiguous heads")
    return batch, heads, key_dim


def kda_recurrent_decode_tlx(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float = 1.0,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Run prepared-input KDA recurrence against a persistent state pool."""
    batch, heads, key_dim = _validate_decode_inputs(
        q,
        k,
        v,
        g,
        beta,
        state_pool,
        read_indices,
        write_indices,
        cu_seqlens,
    )
    value_dim = v.shape[-1]
    g = g.contiguous()
    beta = beta.to(dtype=torch.float32).contiguous()
    read_indices = read_indices.to(device=q.device, dtype=torch.int32).contiguous()
    write_indices = write_indices.to(device=q.device, dtype=torch.int32).contiguous()
    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()

    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    block_key = max(8, triton.next_power_of_2(key_dim))
    # Narrow value panels expose enough independent programs at small H/batch.
    block_value = 8
    _kda_recurrent_decode_kernel[(triton.cdiv(value_dim, block_value), batch * heads)](
        q,
        k,
        v,
        g,
        beta,
        state_pool,
        read_indices,
        write_indices,
        output,
        cu_seqlens,
        H=heads,
        K=key_dim,
        V=value_dim,
        SCALE=scale,
        Q_TOKEN_STRIDE=q.stride(1),
        K_TOKEN_STRIDE=k.stride(1),
        V_TOKEN_STRIDE=v.stride(1),
        G_TOKEN_STRIDE=g.stride(1),
        BETA_TOKEN_STRIDE=beta.stride(1),
        BK=block_key,
        BV=block_value,
        NUM_SLOTS=state_pool.shape[0],
        TOKENS=batch,
        STATE_SLOT_STRIDE=state_pool.stride(0),
        num_warps=1,
        num_stages=2,
    )
    return output


__all__ = [
    "kda_recurrent_decode_tlx",
]
