"""Stateful chunk-parallel Kimi Delta Attention prefill for gfx950.

The kernel accepts normalized Q/K, per-token log decays, and sigmoid-applied
beta coefficients. Recurrent states use the physical ``[N, H, V, K]``
V-major layout.

The implementation uses 64-token chunks and 16-token triangular solves:

* consume normalized Q/K and form per-channel chunk-local cumulative log decays;
* build causal QK and beta-scaled KK products with CDNA4 ``tl.dot`` MFMAs;
* solve four 16x16 diagonal systems and merge their 64x64 WY inverse;
* form W/U and scan the recurrent state sequentially between chunks;
* add the parallel causal within-chunk output.

TLX local-memory async copies stage the shared WY and causal matrices.
"""

from __future__ import annotations

from itertools import pairwise

import torch
import triton
import triton.language as tl
from triton.language.extra import tlx

CHUNK_SIZE = 64
SUBCHUNK_SIZE = 16
_KEY_DIM = 128
_VALUE_DIM = 128


def _chunk_pairs(boundaries: list[int], chunk_size: int) -> list[tuple[int, int]]:
    return [
        (sequence, local_chunk)
        for sequence, (begin, end) in enumerate(pairwise(boundaries))
        for local_chunk in range((end - begin + chunk_size - 1) // chunk_size)
    ]


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int = CHUNK_SIZE) -> torch.Tensor:
    """Map packed global chunks to ``(sequence, local_chunk)``."""
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be a vector")
    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    pairs = _chunk_pairs(boundaries, chunk_size)
    if not pairs:
        return torch.empty((0, 2), dtype=torch.int32, device=cu_seqlens.device)
    return torch.tensor(pairs, dtype=torch.int32, device=cu_seqlens.device)


_PREFILL_METADATA_CACHE = []


def _prepare_prefill_metadata(cu_seqlens: torch.Tensor, device: torch.device, total_tokens: int):
    """Cache the latest packed schedule by tensor identity and version."""
    version = cu_seqlens._version
    if _PREFILL_METADATA_CACHE:
        source, cached_version, cached_device, cached_tokens, value = _PREFILL_METADATA_CACHE
        if source is cu_seqlens and (cached_version, cached_device, cached_tokens) == (version, device, total_tokens):
            return value

    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != total_tokens:
        raise ValueError("cu_seqlens must start at zero and end at the packed token count")
    if any(end < begin for begin, end in pairwise(boundaries)):
        raise ValueError("cu_seqlens must be nondecreasing")

    cu_device = cu_seqlens
    if cu_device.device != device or cu_device.dtype != torch.int32 or not cu_device.is_contiguous():
        cu_device = cu_device.to(device=device, dtype=torch.int32).contiguous()
    pairs = _chunk_pairs(boundaries, CHUNK_SIZE)
    chunk_indices = (torch.tensor(pairs, dtype=torch.int32, device=device) if pairs else
                     torch.empty((0, 2), dtype=torch.int32, device=device))
    value = (cu_device, chunk_indices)
    _PREFILL_METADATA_CACHE[:] = [cu_seqlens, version, device, total_tokens, value]
    return value


@triton.jit
def _load_chunk_block(
    buffer,
    block: tl.constexpr,
    BC: tl.constexpr,
    D: tl.constexpr,
    layout: tl.constexpr,
):
    view = tlx.local_slice(tlx.local_view(buffer, 0), [block * BC, 0], [BC, D])
    return tlx.local_load(view, layout=layout)


@triton.jit
def _preprocess_chunk_kernel(
    Q,
    K,
    G,
    Beta,
    Bg,
    Qg,
    Akk,
    Aqk,
    CuSeqLens,
    ChunkIndices,
    SCALE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    sequence = tl.load(ChunkIndices + chunk * 2).to(tl.int32)
    local_chunk = tl.load(ChunkIndices + chunk * 2 + 1).to(tl.int32)
    begin = tl.load(CuSeqLens + sequence).to(tl.int32)
    end = tl.load(CuSeqLens + sequence + 1).to(tl.int32)
    length = end - begin
    token0 = local_chunk * BT

    rows = tl.arange(0, BT)
    keys = tl.arange(0, D)
    tokens = token0 + rows
    mask = (tokens[:, None] < length) & (keys[None, :] < D)
    offsets = ((begin + tokens[:, None]) * H + head) * D + keys[None, :]
    q = tl.load(Q + offsets, mask=mask, other=0.0)
    k = tl.load(K + offsets, mask=mask, other=0.0)
    log_decay = tl.load(G + offsets, mask=mask, other=0.0).to(tl.bfloat16).to(tl.float32)
    log_decay = tl.where(mask, log_decay, 0.0)
    cumulative = tl.cumsum(log_decay, axis=0)
    gated_q = q.to(tl.float32) * tl.exp(cumulative) * SCALE
    tl.store(Bg + offsets, cumulative, mask=mask)
    tl.store(Qg + offsets, gated_q.to(Qg.dtype.element_ty), mask=mask)

    q_shared_layout: tl.constexpr = tlx.swizzled_layout(3, 3, 5, order=[1, 0])
    bg_shared_layout: tl.constexpr = tlx.swizzled_layout(3, 2, 5, order=[1, 0])
    q_buffer = tlx.local_alloc((BT, D), tl.bfloat16, 1, layout=q_shared_layout)
    k_buffer = tlx.local_alloc((BT, D), tl.bfloat16, 1, layout=q_shared_layout)
    bg_buffer = tlx.local_alloc((BT, D), tl.float32, 1, layout=bg_shared_layout)
    tlx.local_store(tlx.local_view(q_buffer, 0), q)
    tlx.local_store(tlx.local_view(k_buffer, 0), k)
    tlx.local_store(tlx.local_view(bg_buffer, 0), cumulative)
    tl.debug_barrier()

    mfma_layout: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    a_layout: tl.constexpr = tlx.dot_operand_layout(0, mfma_layout, k_width=8)
    b_layout: tl.constexpr = tlx.dot_operand_layout(1, mfma_layout, k_width=8)
    beta_layout: tl.constexpr = tlx.slice_layout(a_layout, 1)
    reference_layout: tl.constexpr = tlx.slice_layout(a_layout, 0)
    out_rows = tl.arange(0, BC)
    out_cols = tl.arange(0, BC)
    sub_rows = tl.arange(0, BC)

    for row_block in tl.static_range(0, BT // BC):
        row_q = _load_chunk_block(q_buffer, row_block, BC, D, a_layout).to(tl.float32)
        row_k = _load_chunk_block(k_buffer, row_block, BC, D, a_layout).to(tl.float32)
        row_bg = _load_chunk_block(bg_buffer, row_block, BC, D, a_layout)
        reference = tl.sum(
            tl.where(sub_rows[:, None] == 0, tlx.release_layout(row_bg), 0.0),
            axis=0,
        )
        reference = tlx.require_layout(reference, reference_layout, pin=False)
        row_gate = tl.exp(row_bg - reference[None, :])
        row_q *= row_gate
        row_k *= row_gate

        row_tokens = token0 + row_block * BC + sub_rows
        row_mask = row_tokens < length
        row_beta = tl.load(
            Beta + (begin + row_tokens) * H + head,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        row_beta = tlx.require_layout(row_beta, beta_layout, pin=False)
        lhs_k = tlx.require_layout(
            (row_k * row_beta[:, None]).to(tl.bfloat16),
            a_layout,
            pin=False,
        )
        lhs_q = tlx.require_layout(row_q.to(tl.bfloat16), a_layout, pin=False)

        for col_block in tl.static_range(0, row_block + 1):
            col_k = _load_chunk_block(k_buffer, col_block, BC, D, a_layout).to(tl.float32)
            col_bg = _load_chunk_block(bg_buffer, col_block, BC, D, a_layout)
            col_k *= tl.exp(reference[None, :] - col_bg)
            rhs = tlx.require_layout(tl.trans(col_k.to(tl.bfloat16)), b_layout, pin=False)
            acc_k = tlx.zeros((BC, BC), tl.float32, layout=mfma_layout)
            acc_q = tlx.zeros((BC, BC), tl.float32, layout=mfma_layout)
            acc_k = tl.dot(lhs_k, rhs, acc=acc_k, out_dtype=tl.float32)
            acc_q = tl.dot(lhs_q, rhs, acc=acc_q, out_dtype=tl.float32)

            if col_block == row_block:
                lower_mask = tlx.require_layout(
                    (out_rows[:, None] > out_cols[None, :]).to(tl.int8),
                    mfma_layout,
                    pin=False,
                ) != 0
                causal_mask = tlx.require_layout(
                    (out_rows[:, None] >= out_cols[None, :]).to(tl.int8),
                    mfma_layout,
                    pin=False,
                ) != 0
                acc_k = tl.where(lower_mask, acc_k, 0.0)
                acc_q = tl.where(causal_mask, acc_q, 0.0)

            out_offsets = (
                ((begin + token0 + row_block * BC + out_rows[:, None]) * H + head) * BT
                + col_block * BC
                + out_cols[None, :]
            ).to(tl.int32)
            out_offsets = tlx.require_layout(out_offsets, mfma_layout, pin=False)
            out_mask = tlx.require_layout(
                (token0 + row_block * BC + out_rows[:, None] < length).to(tl.int8),
                mfma_layout,
                pin=False,
            ) != 0
            tlx.buffer_store(acc_k, Akk, out_offsets, mask=out_mask)
            tlx.buffer_store(
                (acc_q * SCALE).to(Aqk.dtype.element_ty),
                Aqk,
                out_offsets,
                mask=out_mask,
            )


@triton.jit
def _solve_16x16_kernel(
    Akk,
    DiagonalInverse,
    CuSeqLens,
    ChunkIndices,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    chunk_block = tl.program_id(0)
    head = tl.program_id(1)
    chunk = chunk_block // (BT // BC)
    block = chunk_block % (BT // BC)
    sequence = tl.load(ChunkIndices + chunk * 2).to(tl.int32)
    local_chunk = tl.load(ChunkIndices + chunk * 2 + 1).to(tl.int32)
    begin = tl.load(CuSeqLens + sequence).to(tl.int32)
    end = tl.load(CuSeqLens + sequence + 1).to(tl.int32)
    length = end - begin
    token0 = local_chunk * BT + block * BC

    rows = tl.arange(0, BC)
    cols = tl.arange(0, BC)
    offsets = ((begin + token0 + rows[:, None]) * H + head) * BT + block * BC + cols[None, :]
    matrix = tl.load(Akk + offsets, mask=token0 + rows[:, None] < length, other=0.0).to(tl.float32)
    inverse = -tl.where(rows[:, None] > cols[None, :], matrix, 0.0)
    for row in tl.static_range(2, BC):
        coefficients = -tl.load(
            Akk + ((begin + token0 + row) * H + head) * BT + block * BC + cols,
            mask=token0 + row < length,
            other=0.0,
        ).to(tl.float32)
        coefficients = tl.where(cols < row, coefficients, 0.0)
        solved = coefficients + tl.sum(coefficients[:, None] * inverse, axis=0)
        inverse = tl.where((rows == row)[:, None], solved[None, :], inverse)
    inverse += rows[:, None] == cols[None, :]

    out_offsets = ((begin + token0 + rows[:, None]) * H + head) * BC + cols[None, :]
    tl.store(
        DiagonalInverse + out_offsets,
        inverse,
        mask=token0 + rows[:, None] < length,
    )


@triton.jit
def _load_akk_block(
    Akk,
    begin,
    length,
    token0,
    head,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    rows = tl.arange(0, BC)
    cols = tl.arange(0, BC)
    offsets = (
        ((begin + token0 + ROW_BLOCK * BC + rows[:, None]) * H + head) * BT
        + COL_BLOCK * BC
        + cols[None, :]
    )
    return tl.load(
        Akk + offsets,
        mask=token0 + ROW_BLOCK * BC + rows[:, None] < length,
        other=0.0,
    ).to(tl.float32)


@triton.jit
def _load_diagonal_block(
    DiagonalInverse,
    begin,
    length,
    token0,
    head,
    BLOCK: tl.constexpr,
    H: tl.constexpr,
    BC: tl.constexpr,
):
    rows = tl.arange(0, BC)
    cols = tl.arange(0, BC)
    offsets = ((begin + token0 + BLOCK * BC + rows[:, None]) * H + head) * BC + cols[None, :]
    return tl.load(
        DiagonalInverse + offsets,
        mask=token0 + BLOCK * BC + rows[:, None] < length,
        other=0.0,
    ).to(tl.float32)


@triton.jit
def _mm16(lhs, rhs):
    return tl.dot(lhs.to(tl.bfloat16), rhs.to(tl.bfloat16))


@triton.jit
def _store_inverse_block(
    TInverse,
    value,
    begin,
    length,
    token0,
    head,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    rows = tl.arange(0, BC)
    cols = tl.arange(0, BC)
    offsets = (
        ((begin + token0 + ROW_BLOCK * BC + rows[:, None]) * H + head) * BT
        + COL_BLOCK * BC
        + cols[None, :]
    )
    tl.store(
        TInverse + offsets,
        value.to(TInverse.dtype.element_ty),
        mask=token0 + ROW_BLOCK * BC + rows[:, None] < length,
    )


@triton.jit
def _merge_inverse_64_kernel(
    Akk,
    DiagonalInverse,
    TInverse,
    CuSeqLens,
    ChunkIndices,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    sequence = tl.load(ChunkIndices + chunk * 2).to(tl.int32)
    local_chunk = tl.load(ChunkIndices + chunk * 2 + 1).to(tl.int32)
    begin = tl.load(CuSeqLens + sequence).to(tl.int32)
    end = tl.load(CuSeqLens + sequence + 1).to(tl.int32)
    length = end - begin
    token0 = local_chunk * BT

    i00 = _load_diagonal_block(DiagonalInverse, begin, length, token0, head, 0, H, BC)
    i11 = _load_diagonal_block(DiagonalInverse, begin, length, token0, head, 1, H, BC)
    i22 = _load_diagonal_block(DiagonalInverse, begin, length, token0, head, 2, H, BC)
    i33 = _load_diagonal_block(DiagonalInverse, begin, length, token0, head, 3, H, BC)
    a10 = _load_akk_block(Akk, begin, length, token0, head, 1, 0, H, BT, BC)
    a20 = _load_akk_block(Akk, begin, length, token0, head, 2, 0, H, BT, BC)
    a21 = _load_akk_block(Akk, begin, length, token0, head, 2, 1, H, BT, BC)
    a30 = _load_akk_block(Akk, begin, length, token0, head, 3, 0, H, BT, BC)
    a31 = _load_akk_block(Akk, begin, length, token0, head, 3, 1, H, BT, BC)
    a32 = _load_akk_block(Akk, begin, length, token0, head, 3, 2, H, BT, BC)

    i10 = -_mm16(_mm16(i11, a10), i00)
    i21 = -_mm16(_mm16(i22, a21), i11)
    i20 = -_mm16(i22, _mm16(a20, i00) + _mm16(a21, i10))
    i32 = -_mm16(_mm16(i33, a32), i22)
    i31 = -_mm16(i33, _mm16(a31, i11) + _mm16(a32, i21))
    i30 = -_mm16(i33, _mm16(a30, i00) + _mm16(a31, i10) + _mm16(a32, i20))

    _store_inverse_block(TInverse, i00, begin, length, token0, head, 0, 0, H, BT, BC)
    _store_inverse_block(TInverse, i10, begin, length, token0, head, 1, 0, H, BT, BC)
    _store_inverse_block(TInverse, i11, begin, length, token0, head, 1, 1, H, BT, BC)
    _store_inverse_block(TInverse, i20, begin, length, token0, head, 2, 0, H, BT, BC)
    _store_inverse_block(TInverse, i21, begin, length, token0, head, 2, 1, H, BT, BC)
    _store_inverse_block(TInverse, i22, begin, length, token0, head, 2, 2, H, BT, BC)
    _store_inverse_block(TInverse, i30, begin, length, token0, head, 3, 0, H, BT, BC)
    _store_inverse_block(TInverse, i31, begin, length, token0, head, 3, 1, H, BT, BC)
    _store_inverse_block(TInverse, i32, begin, length, token0, head, 3, 2, H, BT, BC)
    _store_inverse_block(TInverse, i33, begin, length, token0, head, 3, 3, H, BT, BC)


@triton.jit
def _form_w_u_kernel(
    TInverse,
    K,
    V,
    Bg,
    Beta,
    U,
    W,
    Kg,
    CuSeqLens,
    ChunkIndices,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BO: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    out_block = tl.program_id(2)
    sequence = tl.load(ChunkIndices + chunk * 2).to(tl.int32)
    local_chunk = tl.load(ChunkIndices + chunk * 2 + 1).to(tl.int32)
    begin = tl.load(CuSeqLens + sequence).to(tl.int32)
    end = tl.load(CuSeqLens + sequence + 1).to(tl.int32)
    length = end - begin
    token0 = local_chunk * BT

    rows = tl.arange(0, BT)
    cols = tl.arange(0, BT)
    t_offsets = ((begin + token0 + rows[:, None]) * H + head) * BT + cols[None, :]
    t_mask = token0 + rows[:, None] < length
    t_buffer = tlx.local_alloc((BT, BT), TInverse.dtype.element_ty, 1)
    token = tlx.async_load(TInverse + t_offsets, tlx.local_view(t_buffer, 0), mask=t_mask)
    tlx.async_load_commit_group([token])

    out_cols = out_block * BO + tl.arange(0, BO)
    data_offsets = ((begin + token0 + rows[:, None]) * H + head) * D + out_cols[None, :]
    data_mask = (token0 + rows[:, None] < length) & (out_cols[None, :] < D)
    beta = tl.load(
        Beta + (begin + token0 + rows) * H + head,
        mask=token0 + rows < length,
        other=0.0,
    ).to(tl.float32)
    values = tl.load(V + data_offsets, mask=data_mask, other=0.0)
    keys = tl.load(K + data_offsets, mask=data_mask, other=0.0)
    gates = tl.load(Bg + data_offsets, mask=data_mask, other=0.0).to(tl.float32)

    wait_token = tlx.async_load_wait_group(0)
    inverse = tlx.local_load(tlx.local_view(t_buffer, 0), token=wait_token)
    u = tl.dot(inverse, (values * beta[:, None]).to(values.dtype))
    w = tl.dot(
        inverse,
        (keys.to(tl.float32) * beta[:, None] * tl.exp(gates)).to(keys.dtype),
    )
    last_token = tl.minimum(token0 + BT, length) - 1
    last_gate = tl.load(Bg + ((begin + last_token) * H + head) * D + out_cols)
    kg = keys.to(tl.float32) * tl.exp(last_gate[None, :] - gates)
    tl.store(U + data_offsets, u.to(U.dtype.element_ty), mask=data_mask)
    tl.store(W + data_offsets, w.to(W.dtype.element_ty), mask=data_mask)
    tl.store(Kg + data_offsets, kg.to(Kg.dtype.element_ty), mask=data_mask)


@triton.jit
def _state_scan_kernel(
    W,
    U,
    Kg,
    Qg,
    Bg,
    InitialState,
    VNew,
    Output,
    FinalState,
    CuSeqLens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    value_block = tl.program_id(0)
    sequence_head = tl.program_id(1)
    sequence = sequence_head // H
    head = sequence_head % H
    begin = tl.load(CuSeqLens + sequence).to(tl.int32)
    end = tl.load(CuSeqLens + sequence + 1).to(tl.int32)
    length = end - begin
    num_chunks = tl.cdiv(length, BT)
    BK: tl.constexpr = K // 2

    uv_layout: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    state_layout: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    uv_a_layout: tl.constexpr = tlx.dot_operand_layout(0, uv_layout, k_width=8)
    uv_b_layout: tl.constexpr = tlx.dot_operand_layout(1, uv_layout, k_width=8)
    state_a_layout: tl.constexpr = tlx.dot_operand_layout(0, state_layout, k_width=8)
    state_b_layout: tl.constexpr = tlx.dot_operand_layout(1, state_layout, k_width=8)

    state_key_layout: tl.constexpr = tlx.slice_layout(state_layout, 0)

    state_values = tl.arange(0, BV)
    state_keys = tl.arange(0, BK)
    values = value_block * BV + state_values
    state_offsets = values[:, None] * K + state_keys[None, :]
    state_offsets = tlx.require_layout(state_offsets.to(tl.int32), state_layout, pin=False)
    state_mask = tlx.require_layout(
        ((values[:, None] < V) & (state_keys[None, :] < BK)).to(tl.int8),
        state_layout,
        pin=False,
    ) != 0
    state_base = sequence_head * K * V
    state0 = tlx.buffer_load(
        InitialState + state_base,
        state_offsets,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    state1 = tlx.buffer_load(
        InitialState + state_base + BK,
        state_offsets,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    state0 = tlx.require_layout(state0, state_layout, pin=False)
    state1 = tlx.require_layout(state1, state_layout, pin=False)

    qw_keys = tl.arange(0, BK)
    qw_rows = tl.arange(0, BT)
    kg_rows = tl.arange(0, BT)
    kg_keys = tl.arange(0, BK)
    uv_values = tl.arange(0, BV)
    uv_rows = tl.arange(0, BT)
    out_values = value_block * BV + uv_values
    key_base = (begin * H + head) * K
    value_base = (begin * H + head) * V

    for local_chunk in range(num_chunks):
        token0 = local_chunk * BT
        qw_offsets0 = ((token0 + qw_rows[None, :]) * H * K + qw_keys[:, None]).to(tl.int32)
        qw_offsets0 = tlx.require_layout(qw_offsets0, uv_b_layout, pin=False)
        qw_offsets1 = qw_offsets0 + BK
        qw_mask = tlx.require_layout(
            ((token0 + qw_rows[None, :] < length) & (qw_keys[:, None] < BK)).to(tl.int8),
            uv_b_layout,
            pin=False,
        ) != 0
        q0 = tlx.buffer_load(Qg + key_base, qw_offsets0, mask=qw_mask, other=0.0)
        q1 = tlx.buffer_load(Qg + key_base, qw_offsets1, mask=qw_mask, other=0.0)
        w0 = tlx.buffer_load(W + key_base, qw_offsets0, mask=qw_mask, other=0.0)
        w1 = tlx.buffer_load(W + key_base, qw_offsets1, mask=qw_mask, other=0.0)
        q0 = tlx.require_layout(q0, uv_b_layout, pin=False)
        q1 = tlx.require_layout(q1, uv_b_layout, pin=False)
        w0 = tlx.require_layout(w0, uv_b_layout, pin=False)
        w1 = tlx.require_layout(w1, uv_b_layout, pin=False)

        state_lhs0 = tlx.require_layout(state0.to(tl.bfloat16), uv_a_layout, pin=False)
        state_lhs1 = tlx.require_layout(state1.to(tl.bfloat16), uv_a_layout, pin=False)
        inter = tlx.zeros((BV, BT), tl.float32, layout=uv_layout)
        inter = tl.dot(state_lhs0, q0, acc=inter, out_dtype=tl.float32)
        inter = tl.dot(state_lhs1, q1, acc=inter, out_dtype=tl.float32)
        prediction = tlx.zeros((BV, BT), tl.float32, layout=uv_layout)
        prediction = tl.dot(state_lhs0, w0, acc=prediction, out_dtype=tl.float32)
        prediction = tl.dot(state_lhs1, w1, acc=prediction, out_dtype=tl.float32)

        result_offsets = ((token0 + uv_rows[None, :]) * H * V + out_values[:, None]).to(tl.int32)
        result_offsets = tlx.require_layout(result_offsets, uv_layout, pin=False)
        result_mask = tlx.require_layout(
            ((token0 + uv_rows[None, :] < length) & (out_values[:, None] < V)).to(tl.int8),
            uv_layout,
            pin=False,
        ) != 0
        u_value = tlx.buffer_load(
            U + value_base,
            result_offsets,
            mask=result_mask,
            other=0.0,
        ).to(tl.float32)
        u_value = tlx.require_layout(u_value, uv_layout, pin=False)
        new_value = u_value - prediction
        tlx.buffer_store(
            inter.to(Output.dtype.element_ty),
            Output + value_base,
            result_offsets,
            mask=result_mask,
        )
        tlx.buffer_store(
            new_value.to(tl.bfloat16),
            VNew + value_base,
            result_offsets,
            mask=result_mask,
        )

        kg_offsets0 = ((token0 + kg_rows[:, None]) * H * K + kg_keys[None, :]).to(tl.int32)
        kg_offsets0 = tlx.require_layout(kg_offsets0, state_b_layout, pin=False)
        kg_offsets1 = kg_offsets0 + BK
        kg_mask = tlx.require_layout(
            ((token0 + kg_rows[:, None] < length) & (kg_keys[None, :] < BK)).to(tl.int8),
            state_b_layout,
            pin=False,
        ) != 0
        kg0 = tlx.buffer_load(Kg + key_base, kg_offsets0, mask=kg_mask, other=0.0)
        kg1 = tlx.buffer_load(Kg + key_base, kg_offsets1, mask=kg_mask, other=0.0)
        kg0 = tlx.require_layout(kg0, state_b_layout, pin=False)
        kg1 = tlx.require_layout(kg1, state_b_layout, pin=False)

        last_token = tl.minimum(token0 + BT, length) - 1
        bg_offsets = (last_token * H * K + state_keys).to(tl.int32)
        bg_offsets = tlx.require_layout(bg_offsets, state_key_layout, pin=False)
        decay0 = tlx.buffer_load(Bg + key_base, bg_offsets).to(tl.float32)
        decay1 = tlx.buffer_load(Bg + key_base, bg_offsets + BK).to(tl.float32)
        decay0 = tlx.require_layout(tl.exp(decay0), state_key_layout, pin=False)
        decay1 = tlx.require_layout(tl.exp(decay1), state_key_layout, pin=False)
        state0 *= decay0[None, :]
        state1 *= decay1[None, :]
        state_lhs = tlx.require_layout(new_value.to(tl.bfloat16), state_a_layout, pin=False)
        state0 = tl.dot(state_lhs, kg0, acc=state0, out_dtype=tl.float32)
        state1 = tl.dot(state_lhs, kg1, acc=state1, out_dtype=tl.float32)

    tlx.buffer_store(state0, FinalState + state_base, state_offsets, mask=state_mask)
    tlx.buffer_store(state1, FinalState + state_base + BK, state_offsets, mask=state_mask)


@triton.jit
def _output_tail_kernel(
    Aqk,
    VNew,
    Output,
    CuSeqLens,
    ChunkIndices,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    value_block = tl.program_id(2)
    sequence = tl.load(ChunkIndices + chunk * 2).to(tl.int32)
    local_chunk = tl.load(ChunkIndices + chunk * 2 + 1).to(tl.int32)
    begin = tl.load(CuSeqLens + sequence).to(tl.int32)
    end = tl.load(CuSeqLens + sequence + 1).to(tl.int32)
    length = end - begin
    token0 = local_chunk * BT

    rows = tl.arange(0, BT)
    cols = tl.arange(0, BT)
    a_offsets = ((begin + token0 + rows[:, None]) * H + head) * BT + cols[None, :]
    row_mask = token0 + rows < length
    a_buffer = tlx.local_alloc((BT, BT), Aqk.dtype.element_ty, 1)
    token = tlx.async_load(
        Aqk + a_offsets,
        tlx.local_view(a_buffer, 0),
        mask=row_mask[:, None],
    )
    tlx.async_load_commit_group([token])

    value_cols = value_block * BV + tl.arange(0, BV)
    value_offsets = (
        ((begin + token0 + cols[:, None]) * H + head) * V + value_cols[None, :]
    )
    value_mask = row_mask[:, None] & (value_cols[None, :] < V)
    vnew = tl.load(VNew + value_offsets, mask=value_mask, other=0.0)
    wait_token = tlx.async_load_wait_group(0)
    causal = tlx.local_load(tlx.local_view(a_buffer, 0), token=wait_token)
    intra = tl.dot(causal, vnew)
    output_offsets = (
        ((begin + token0 + rows[:, None]) * H + head) * V + value_cols[None, :]
    )
    old = tl.load(Output + output_offsets, mask=value_mask, other=0.0).to(tl.float32)
    tl.store(
        Output + output_offsets,
        (old + intra).to(Output.dtype.element_ty),
        mask=value_mask,
    )


def _validate_prefill_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[int, int, int, int]:
    tensors = (q, k, v, g, beta, initial_state)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gfx950 TLX KDA prefill requires GPU tensors")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("KDA data tensors must be on the same GPU")
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("q must be a packed [1, T, H, K] tensor")
    if q.shape != k.shape or q.shape != g.shape:
        raise ValueError("q, k, and g must have identical shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("v must match q through the head dimension")
    if beta.shape != q.shape[:-1]:
        raise ValueError("beta must have shape [1, T, H]")
    _, total_tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if (key_dim, value_dim) != (_KEY_DIM, _VALUE_DIM):
        raise ValueError("gfx950 TLX KDA prefill specializes K=V=128")
    if q.dtype != torch.bfloat16 or any(t.dtype != q.dtype for t in (k, v)):
        raise ValueError("q, k, and v must be BF16")
    if not g.dtype.is_floating_point or not beta.dtype.is_floating_point:
        raise ValueError("g and beta must use floating-point dtypes")
    if initial_state.dtype != torch.float32:
        raise ValueError("initial_state must be FP32")
    if initial_state.ndim != 4 or initial_state.shape[1:] != (heads, value_dim, key_dim):
        raise ValueError("initial_state must have V-major shape [N, H, V, K]")
    if initial_state.stride()[-2:] != (key_dim, 1):
        raise ValueError("initial_state inner [V, K] dimensions must be contiguous")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() != initial_state.shape[0] + 1:
        raise ValueError("cu_seqlens and initial_state must describe the same batch")
    return total_tokens, heads, key_dim, value_dim


def kda_paged_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run prepared-input KDA prefill and return output plus final state."""
    total_tokens, heads, key_dim, value_dim = _validate_prefill_inputs(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
    )
    q = q[0].contiguous()
    k = k[0].contiguous()
    v = v[0].contiguous()
    g = g[0].contiguous()
    beta = beta[0].to(dtype=torch.float32).contiguous()
    initial_state = initial_state.contiguous()
    cu_seqlens, chunk_indices = _prepare_prefill_metadata(cu_seqlens, q.device, total_tokens)
    num_chunks = chunk_indices.shape[0]
    num_sequences = cu_seqlens.numel() - 1
    if num_chunks == 0:
        return torch.empty_like(v).unsqueeze(0), initial_state.clone()

    cumulative_gate = torch.empty_like(g, dtype=torch.float32)
    gated_query = torch.empty_like(q)
    akk = torch.empty(
        (total_tokens, heads, CHUNK_SIZE),
        dtype=torch.float32,
        device=q.device,
    )
    aqk = torch.zeros(
        (total_tokens, heads, CHUNK_SIZE),
        dtype=torch.bfloat16,
        device=q.device,
    )

    _preprocess_chunk_kernel[(num_chunks, heads)](
        q,
        k,
        g,
        beta,
        cumulative_gate,
        gated_query,
        akk,
        aqk,
        cu_seqlens,
        chunk_indices,
        SCALE=scale,
        H=heads,
        D=key_dim,
        BT=CHUNK_SIZE,
        BC=SUBCHUNK_SIZE,
        num_warps=4,
    )

    diagonal_inverse = torch.empty(
        (total_tokens, heads, SUBCHUNK_SIZE),
        dtype=torch.float32,
        device=q.device,
    )
    _solve_16x16_kernel[(num_chunks * 4, heads)](
        akk,
        diagonal_inverse,
        cu_seqlens,
        chunk_indices,
        H=heads,
        BT=CHUNK_SIZE,
        BC=SUBCHUNK_SIZE,
        num_warps=1,
    )
    inverse = torch.zeros_like(aqk)
    _merge_inverse_64_kernel[(num_chunks, heads)](
        akk,
        diagonal_inverse,
        inverse,
        cu_seqlens,
        chunk_indices,
        H=heads,
        BT=CHUNK_SIZE,
        BC=SUBCHUNK_SIZE,
        num_warps=4,
    )

    u = torch.empty_like(v)
    w = torch.empty_like(k)
    gated_key = torch.empty_like(k)
    _form_w_u_kernel[(num_chunks, heads, 2)](
        inverse,
        k,
        v,
        cumulative_gate,
        beta,
        u,
        w,
        gated_key,
        cu_seqlens,
        chunk_indices,
        H=heads,
        D=key_dim,
        BT=CHUNK_SIZE,
        BO=64,
        num_warps=4,
    )

    v_new = torch.empty_like(v)
    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    state_block_value = 32 if heads >= 12 and num_sequences >= 4 else 16
    state_waves_per_eu = 1 if heads >= 12 and num_sequences >= 8 else 2
    _state_scan_kernel[(triton.cdiv(value_dim, state_block_value), num_sequences * heads)](
        w,
        u,
        gated_key,
        gated_query,
        cumulative_gate,
        initial_state,
        v_new,
        output,
        final_state,
        cu_seqlens,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=CHUNK_SIZE,
        BV=state_block_value,
        num_warps=4,
        num_stages=2,
        waves_per_eu=state_waves_per_eu,
    )
    _output_tail_kernel[(num_chunks, heads, 2)](
        aqk,
        v_new,
        output,
        cu_seqlens,
        chunk_indices,
        H=heads,
        V=value_dim,
        BT=CHUNK_SIZE,
        BV=64,
        num_warps=4,
        num_stages=2,
        waves_per_eu=4,
    )
    return output.unsqueeze(0), final_state


__all__ = [
    "CHUNK_SIZE",
    "SUBCHUNK_SIZE",
    "kda_paged_prefill",
    "prepare_chunk_indices",
]
