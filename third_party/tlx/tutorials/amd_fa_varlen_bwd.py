"""Packed variable-length BF16 D128 attention backward for gfx950.

Each workgroup owns a KV tile and a subset of its mapped query heads.  The
baseline path uses BN128/BM16 phases and supports non-causal MHA/GQA plus
causal self-attention MHA; long non-causal split-GQA uses masked BN256/BM32
phases and forms dQ as two native BM16 accumulator chains.  Split workgroups
reduce their BF16 dK/dV partials in FP32.  Independent KV owners combine dQ
contributions with BF16 atomics in a guarded native layout, followed by a
conversion to packed THD order.

Call :func:`prepare_varlen_backward` once and reuse the resulting plan with
:func:`fa_varlen_backward`.  Plan creation performs one device-to-host
synchronization to construct compact schedules; execution itself does not copy
offsets to the host.  Treat every plan-owned offset and schedule tensor as
immutable after preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from triton.language.extra.tlx.tutorials.amd_fa_bwd import _attn_bwd_gqa_front

_BLOCK_M = 16
_BLOCK_N = 128
_WIDE_BLOCK_M = 32
_WIDE_BLOCK_N = 256
_HEAD_DIM = 128
_I32_BUFFER_BF16_ELEMENTS = 1 << 30
_VARLEN_GQA_SPLIT_WORK_THRESHOLD = 1024


@dataclass(frozen=True)
class VarlenBackwardPlan:
    """Reusable launch metadata whose tensor contents are caller-immutable.

    ``frozen=True`` prevents field rebinding, but PyTorch tensors remain
    mutable.  Do not modify any plan-owned offset or schedule tensor after
    preparation.
    """

    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    q_block_sequence: torch.Tensor
    q_block_start: torch.Tensor
    kv_block_sequence: torch.Tensor
    kv_block_start: torch.Tensor
    wide_kv_start: torch.Tensor
    wide_q_start: torch.Tensor
    wide_dq_start: torch.Tensor
    wide_q_len: torch.Tensor
    wide_kv_valid: torch.Tensor
    batch: int
    total_q: int
    total_kv: int
    max_q: int
    num_full_kv_blocks: int
    qk_offsets_equal: bool


def _copy_and_validate_cu_seqlens(name: str, value: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    if value.ndim != 1 or value.numel() < 2:
        raise ValueError(f"{name} must be a rank-1 tensor with at least two elements")
    if value.dtype is not torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be on a CUDA device")
    owned = value.detach().clone(memory_format=torch.contiguous_format)
    offsets = owned.cpu().tolist()
    if offsets[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if any(end <= begin for begin, end in zip(offsets, offsets[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return owned, offsets


def _make_block_schedule(lengths: list[int], block: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences: list[int] = []
    starts: list[int] = []
    for sequence, length in enumerate(lengths):
        for start in range(0, length, block):
            sequences.append(sequence)
            starts.append(start)
    return (
        torch.tensor(sequences, dtype=torch.int32, device=device),
        torch.tensor(starts, dtype=torch.int32, device=device),
    )


def _make_partitioned_block_schedule(lengths: list[int], block: int,
                                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    full_sequences: list[int] = []
    full_starts: list[int] = []
    tail_sequences: list[int] = []
    tail_starts: list[int] = []
    for sequence, length in enumerate(lengths):
        full_end = length // block * block
        for start in range(0, full_end, block):
            full_sequences.append(sequence)
            full_starts.append(start)
        if full_end < length:
            tail_sequences.append(sequence)
            tail_starts.append(full_end)
    return (
        torch.tensor(full_sequences + tail_sequences, dtype=torch.int32, device=device),
        torch.tensor(full_starts + tail_starts, dtype=torch.int32, device=device),
        len(full_sequences),
    )


def _make_wide_kv_schedule(q_offsets: list[int], k_offsets: list[int],
                           device: torch.device) -> tuple[torch.Tensor, ...]:
    """Build masked BN256 tasks, front-loading sequences with the most Q work."""
    tasks: list[tuple[int, int, int, int, int]] = []
    sequences = sorted(
        range(len(q_offsets) - 1),
        key=lambda sequence: q_offsets[sequence + 1] - q_offsets[sequence],
        reverse=True,
    )
    for sequence in sequences:
        q_start = q_offsets[sequence]
        q_len = q_offsets[sequence + 1] - q_start
        kv_start = k_offsets[sequence]
        kv_len = k_offsets[sequence + 1] - kv_start
        dq_start = q_start + sequence * (_BLOCK_M - 1)
        for block_start in range(0, kv_len, _WIDE_BLOCK_N):
            tasks.append((
                kv_start + block_start,
                q_start,
                dq_start,
                q_len,
                min(_WIDE_BLOCK_N, kv_len - block_start),
            ))
    columns = tuple(zip(*tasks, strict=True))
    return tuple(torch.tensor(column, dtype=torch.int32, device=device) for column in columns)


def prepare_varlen_backward(cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor) -> VarlenBackwardPlan:
    """Prepare reusable compact schedules for immutable packed offsets."""
    if cu_seqlens_q.device != cu_seqlens_k.device:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be on the same device")
    owned_q, q_offsets = _copy_and_validate_cu_seqlens("cu_seqlens_q", cu_seqlens_q)
    owned_k, k_offsets = _copy_and_validate_cu_seqlens("cu_seqlens_k", cu_seqlens_k)
    if len(q_offsets) != len(k_offsets):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch")

    q_lengths = [end - begin for begin, end in zip(q_offsets, q_offsets[1:])]
    k_lengths = [end - begin for begin, end in zip(k_offsets, k_offsets[1:])]
    q_block_sequence, q_block_start = _make_block_schedule(q_lengths, _BLOCK_M, cu_seqlens_q.device)
    kv_block_sequence, kv_block_start, num_full_kv_blocks = _make_partitioned_block_schedule(
        k_lengths, _BLOCK_N, cu_seqlens_q.device)
    wide_kv_start, wide_q_start, wide_dq_start, wide_q_len, wide_kv_valid = _make_wide_kv_schedule(
        q_offsets, k_offsets, cu_seqlens_q.device)

    return VarlenBackwardPlan(
        cu_seqlens_q=owned_q,
        cu_seqlens_k=owned_k,
        q_block_sequence=q_block_sequence,
        q_block_start=q_block_start,
        kv_block_sequence=kv_block_sequence,
        kv_block_start=kv_block_start,
        wide_kv_start=wide_kv_start,
        wide_q_start=wide_q_start,
        wide_dq_start=wide_dq_start,
        wide_q_len=wide_q_len,
        wide_kv_valid=wide_kv_valid,
        batch=len(q_lengths),
        total_q=q_offsets[-1],
        total_kv=k_offsets[-1],
        max_q=max(q_lengths),
        num_full_kv_blocks=num_full_kv_blocks,
        qk_offsets_equal=q_offsets == k_offsets,
    )


def _validate_i32_buffer_offsets(*, total_q: int, total_kv: int, batch: int, q_heads: int, kv_heads: int) -> None:
    if total_kv * kv_heads * _HEAD_DIM > _I32_BUFFER_BF16_ELEMENTS:
        raise ValueError("KV tensor size exceeds the signed 32-bit byte-offset range")
    total_q_padded = total_q + batch * (_BLOCK_M - 1)
    if total_q_padded * q_heads * _HEAD_DIM > _I32_BUFFER_BF16_ELEMENTS:
        raise ValueError("padded dQ size exceeds the signed 32-bit byte-offset range")


def _select_varlen_kv_splits(max_q: int, group_size: int) -> int:
    query_blocks = (max_q + _BLOCK_M - 1) // _BLOCK_M
    if group_size <= 1 or group_size * query_blocks < _VARLEN_GQA_SPLIT_WORK_THRESHOLD:
        return 1
    for kv_splits in range(min(group_size, 4), 1, -1):
        if group_size % kv_splits == 0:
            return kv_splits
    return 1


def _select_varlen_kernel_blocks(group_size: int, kv_splits: int) -> tuple[int, int]:
    if group_size > 1 and kv_splits > 1:
        return _WIDE_BLOCK_M, _WIDE_BLOCK_N
    return _BLOCK_M, _BLOCK_N


def _allocate_varlen_dkdv_partials(k: torch.Tensor, kv_splits: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if kv_splits == 1 or k.numel() * kv_splits > _I32_BUFFER_BF16_ELEMENTS:
        return None, None
    total_kv, kv_heads, head_dim = k.shape
    dk_part = torch.empty((total_kv, kv_heads, kv_splits, head_dim), dtype=k.dtype, device=k.device)
    return dk_part, torch.empty_like(dk_part)


@triton.jit
def _varlen_bwd_preprocess(
    O,
    DO,
    Delta,
    CuQ,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // HEADS
    head = batch_head % HEADS
    q_start = tl.load(CuQ + batch).to(tl.int64)
    q_end = tl.load(CuQ + batch + 1).to(tl.int64)
    q_len = q_end - q_start
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask = offs_m[:, None] < q_len
    token = q_start + offs_m
    offsets = (token[:, None] * HEADS + head) * D + offs_d[None, :]
    o = tl.load(O + offsets, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(DO + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(Delta + token * HEADS + head, tl.sum(o * do, axis=1), mask=offs_m < q_len)


@triton.jit
def _issue_qdo_async(
    q_dst,
    do_dst,
    Q,
    DO,
    q_start,
    q_len,
    q_head,
    step,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = step * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    global_rows = q_start + rows
    offsets = (global_rows[:, None] * HQ + q_head) * D + dims[None, :]
    valid = rows[:, None] < q_len
    q_token = tlx.async_load(Q + offsets, q_dst, mask=valid, other=0.0)
    do_token = tlx.async_load(DO + offsets, do_dst, mask=valid, other=0.0)
    tlx.async_load_commit_group([q_token, do_token])


@triton.jit
def _store_dq_native(
    dq,
    DQ_ACC,
    dq_base,
    q_len,
    step,
    SM_SCALE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MMA_MD: tl.constexpr,
):
    dq = tlx.require_layout(dq, MMA_MD, pin=False)
    scale = tlx.require_layout(
        tl.full((BLOCK_M, D), SM_SCALE, dtype=tl.float32),
        MMA_MD,
        pin=False,
    )
    dq *= scale
    dq_row_remat_group: tl.constexpr = 40
    dq_column_remat_group: tl.constexpr = 41
    local_m = tlx.rematerialized_range(0, BLOCK_M, dq_row_remat_group, placement=step)
    offs_d = tlx.rematerialized_range(0, D, dq_column_remat_group, placement=step)
    valid = tl.broadcast_to((step * BLOCK_M + local_m < q_len)[:, None], (BLOCK_M, D))
    valid = tlx.require_layout(valid, MMA_MD, pin=False)
    d_swizzled = ((offs_d & 1)
                  | ((offs_d & 2) << 6)
                  | ((offs_d & 12) << 3)
                  | ((offs_d & 48) << 5)
                  | ((offs_d & 64) << 2))
    tile_offset = step * BLOCK_M * D
    offsets = dq_base + tile_offset + ((local_m[:, None] << 1) | d_swizzled[None, :])
    offsets = tl.max_contiguous(offsets.to(tl.int32), [1, 2])
    offsets = tlx.require_layout(offsets, MMA_MD, pin=False)
    tlx.buffer_atomic_add(
        DQ_ACC,
        offsets,
        dq.to(tl.bfloat16),
        mask=valid,
        sem="relaxed",
        contiguity=2,
    )


@triton.jit
def _issue_qdo_bm32_async(
    q_dst,
    do_dst,
    Q,
    DO,
    q_start,
    q_len,
    q_head,
    outer_block,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ASYNC_LAYOUT: tl.constexpr,
):
    outer_slice = tl.arange(0, 1)
    rows = outer_block * BLOCK_M + outer_slice[:, None] * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
    dims = tl.arange(0, D)
    valid = tl.broadcast_to((rows < q_len)[:, :, None], (1, BLOCK_M, D))
    valid = tlx.require_layout(valid, ASYNC_LAYOUT, pin=False)
    qdo_base = (q_start * HQ + q_head.to(tl.int64)) * D
    offsets = (rows[:, :, None] * HQ * D + dims[None, None, :]).to(tl.int32)
    offsets = tlx.require_layout(offsets, ASYNC_LAYOUT, pin=False)
    other = tlx.zeros((1, BLOCK_M, D), tl.bfloat16, layout=ASYNC_LAYOUT)
    q_token = tlx.buffer_load_to_local(
        q_dst,
        tl.multiple_of(Q + qdo_base, 16),
        offsets,
        mask=valid,
        other=other,
    )
    tlx.async_load_commit_group([q_token])
    do_token = tlx.buffer_load_to_local(
        do_dst,
        tl.multiple_of(DO + qdo_base, 16),
        offsets,
        mask=valid,
        other=other,
    )
    tlx.async_load_commit_group([do_token])


@triton.jit
def _varlen_gqa_phase_bm32(
    q_tiles,
    do_tiles,
    ds_buffer,
    k_buffer,
    v_operand,
    dk,
    dv,
    outer_block,
    LSE,
    Delta,
    q_start,
    q_len,
    q_head,
    TOTAL_Q,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MMA_NM: tl.constexpr,
    MMA_ND: tl.constexpr,
    K_NM_LAYOUT: tl.constexpr,
    QT_LAYOUT: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
):
    q_slice = tlx.local_view(q_tiles, 0)
    do_slice = tlx.local_view(do_tiles, 0)
    rows = outer_block * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = rows < q_len
    global_rows = q_start + rows
    lse_values = tl.load(
        LSE + q_head * TOTAL_Q + global_rows,
        mask=valid,
        other=float("inf"),
    )
    delta_values = tl.load(
        Delta + global_rows * HQ + q_head,
        mask=valid,
        other=0.0,
    )
    dv, dk_lhs, dk_rhs, dv_lhs, dv_rhs = _attn_bwd_gqa_front(
        dv,
        q_slice,
        do_slice,
        k_buffer,
        v_operand,
        ds_buffer,
        lse_values,
        delta_values,
        0,
        0,
        SM_SCALE,
        BLOCK_M,
        BLOCK_M,
        BLOCK_N,
        False,
        MMA_NM,
        MMA_ND,
        K_NM_LAYOUT,
        QT_LAYOUT,
        P_ND_LAYOUT,
        Q_OUT_LAYOUT,
        False,
    )
    dv_lhs = tlx.require_layout(dv_lhs, P_ND_LAYOUT, pin=False)
    dv_rhs = tlx.require_layout(dv_rhs, Q_OUT_LAYOUT, pin=False)
    dv = tlx.require_layout(dv, MMA_ND, pin=False)
    dv = tl.dot(dv_lhs, dv_rhs, dv)
    dk_lhs = tlx.require_layout(dk_lhs, P_ND_LAYOUT, pin=False)
    dk_rhs = tlx.require_layout(dk_rhs, Q_OUT_LAYOUT, pin=False)
    dk = tlx.require_layout(dk, MMA_ND, pin=False)
    dk = tl.dot(dk_lhs, dk_rhs, dk)
    tl.debug_barrier()
    return (
        tlx.require_layout(dk, MMA_ND, pin=False),
        tlx.require_layout(dv, MMA_ND, pin=False),
        tlx.require_layout(v_operand, K_NM_LAYOUT, pin=False),
    )


@triton.jit
def _varlen_gqa_dq_bm32(
    ds_buffer,
    k_buffer,
    v_operand,
    MMA_MD: tl.constexpr,
    DS_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
    V_LAYOUT: tl.constexpr,
):
    v_operand = tlx.require_layout(v_operand, V_LAYOUT, pin=False)
    k = tlx.local_load(k_buffer, layout=K_MD_LAYOUT, relaxed=True)
    ds_lo = tlx.local_load(
        tlx.local_slice(ds_buffer, [0, 0], [16, 256]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    ds_hi = tlx.local_load(
        tlx.local_slice(ds_buffer, [16, 0], [16, 256]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    dq_lo = tlx.amd_scheduled_mfma(
        ds_lo,
        k,
        tlx.zeros((16, 128), tl.float32, layout=MMA_MD),
        resident_operand=1,
        accumulator_role="transient",
        initialize=True,
    )
    dq_hi = tlx.amd_scheduled_mfma(
        ds_hi,
        k,
        tlx.zeros((16, 128), tl.float32, layout=MMA_MD),
        resident_operand=1,
        accumulator_role="transient",
        initialize=True,
    )
    dq_lo, dq_hi, v_operand = tlx.amd_mfma_commit((dq_lo, dq_hi), v_operand)
    return (
        tlx.require_layout(dq_lo, MMA_MD, pin=False),
        tlx.require_layout(dq_hi, MMA_MD, pin=False),
        tlx.require_layout(v_operand, V_LAYOUT, pin=False),
    )


@triton.jit
def _store_dq_bm32_native(
    dq_lo,
    dq_hi,
    DQ_ACC,
    dq_base,
    q_len,
    outer_block,
    SM_SCALE: tl.constexpr,
    D: tl.constexpr,
    MMA_MD: tl.constexpr,
):
    _store_dq_native(dq_lo, DQ_ACC, dq_base, q_len, outer_block * 2, SM_SCALE, D, 16, MMA_MD)
    _store_dq_native(dq_hi, DQ_ACC, dq_base, q_len, outer_block * 2 + 1, SM_SCALE, D, 16, MMA_MD)


@triton.jit
def _varlen_bwd_interleaved_bm32_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    KVGlobalStart,
    QStart,
    DQScratchStart,
    QLen,
    KVValidRows,
    DQ_ACC,
    DK,
    DV,
    SM_SCALE: tl.constexpr,
    TOTAL_Q,
    TOTAL_Q_PADDED,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_SPLITS: tl.constexpr,
):
    """Process masked BN256 KV owners in BM32 phases with two native BM16 dQ chains."""
    tl.static_assert(D == 128)
    tl.static_assert(BLOCK_M == 32)
    tl.static_assert(BLOCK_N == 256)
    tl.static_assert(HQ % HKV == 0)
    tl.static_assert(HQ // HKV > 1)
    tl.static_assert(KV_SPLITS > 1)
    tl.static_assert(KV_SPLITS <= 4)
    tl.static_assert((HQ // HKV) % KV_SPLITS == 0)

    task = tl.program_id(1)
    kv_head_split = tl.program_id(0)
    kv_head = kv_head_split // KV_SPLITS
    split = kv_head_split % KV_SPLITS
    group_size: tl.constexpr = HQ // HKV
    heads_per_split: tl.constexpr = group_size // KV_SPLITS
    kv_global_start = tl.load(KVGlobalStart + task).to(tl.int64)
    q_start = tl.load(QStart + task).to(tl.int64)
    q_scratch_start = tl.load(DQScratchStart + task).to(tl.int64)
    q_len = tl.load(QLen + task)
    kv_valid_rows = tl.load(KVValidRows + task)
    outer_blocks = (q_len + BLOCK_M - 1) // BLOCK_M
    total_outer_steps = heads_per_split * outer_blocks
    first_q_head = kv_head * group_size + split * heads_per_split

    mma_nm: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_nd: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_md: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    k_nm_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    qt_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_nm, k_width=8)
    p_nd_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_nd, k_width=8)
    q_out_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_nd, k_width=8)
    ds_md_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_md, k_width=8)
    k_md_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_md, k_width=8)

    qdo_async_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
        stride=((8, 16, 32, 128, 64, 512, 256, 1024), (1, 2, 4, 2048)),
    )
    qdo_smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32), (1024, 16)],
        [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [0, 0, 64], [0, 4, 0],
         [0, 2, 0], [0, 8, 0], [0, 16, 0]],
        [1, BLOCK_M, D],
    )
    qdo_slice_smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32), (1024, 16)],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [0, 64], [4, 0], [2, 0], [8, 0], [16, 0]],
        [BLOCK_M, D],
    )
    ds_smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 16)],
        [[1, 0], [2, 0], [0, 1], [0, 2], [4, 0], [0, 8], [8, 0], [0, 32], [0, 16], [0, 4], [0, 64], [0, 128], [16, 0]],
        [BLOCK_M, BLOCK_N],
    )
    k_raw_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0],
                      [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0], [128, 0, 0]],
        block_bases=[],
        alignment=16,
    )
    k_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 64], [1, 0], [2, 0], [4, 0], [8, 64], [0, 16], [0, 32],
                      [16, 0], [32, 0], [64, 0], [128, 0]],
        block_bases=[],
        alignment=16,
    )
    k_raw_async_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (8, 8, 2)),
        stride=((8, 512), (1, 2048, 16384)),
    )
    kv_native_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2)),
        stride=((128, 256, 512, 1024, 8192, 4, 2048, 4096), (1, 2, 8, 16, 32, 64, 16384)),
    )

    k_raw_buffer = tlx.local_alloc((BLOCK_N, D // 8, 8), tl.bfloat16, 1, layout=k_raw_smem_layout)
    k_buffer = tlx.local_reinterpret(
        tlx.local_view(k_raw_buffer, 0),
        tl.bfloat16,
        [BLOCK_N, D],
        layout=k_smem_layout,
    )
    q_buffers = tlx.local_alloc((1, BLOCK_M, D), tl.bfloat16, 2, layout=qdo_smem_layout)
    do_buffers = tlx.local_alloc((1, BLOCK_M, D), tl.bfloat16, 2, layout=qdo_smem_layout)
    ds_buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.bfloat16, 1, layout=ds_smem_layout)

    raw_n = tl.arange(0, BLOCK_N)
    raw_dg = tl.arange(0, D // 8)
    raw_v = tl.arange(0, 8)
    k_phys = raw_n[:, None, None] * D + raw_dg[None, :, None] * 8
    k_d_base = ((k_phys & 0x8) | (((k_phys >> 9) & 0x3) << 4) | ((((k_phys >> 4) ^ (k_phys >> 8)) & 0x1) << 6))
    k_n = (((k_phys >> 5) & 0x7) | (((k_phys >> 8) & 0x1) << 3) | (((k_phys >> 11) & 0xf) << 4))
    kv_tile_base = (kv_global_start * HKV + kv_head.to(tl.int64)) * D
    k_ptr = tl.multiple_of(K + kv_tile_base, 16)
    v_ptr = tl.multiple_of(V + kv_tile_base, 16)
    k_offsets = (k_n * HKV * D + k_d_base + raw_v[None, None, :]).to(tl.int32)
    k_offsets = tl.multiple_of(k_offsets, [1, 1, 8])
    k_offsets = tl.max_contiguous(k_offsets, [1, 1, 8])
    k_offsets = tlx.require_layout(k_offsets, k_raw_async_layout, pin=False)
    k_valid = tl.broadcast_to(k_n < kv_valid_rows, (BLOCK_N, D // 8, 8))
    k_valid = tlx.require_layout(k_valid, k_raw_async_layout, pin=False)
    k_zero = tlx.zeros((BLOCK_N, D // 8, 8), tl.bfloat16, layout=k_raw_async_layout)
    k_token = tlx.buffer_load_to_local(
        tlx.local_view(k_raw_buffer, 0),
        k_ptr,
        k_offsets,
        mask=k_valid,
        other=k_zero,
    )
    tlx.async_load_commit_group([k_token])

    _issue_qdo_bm32_async(
        tlx.local_view(q_buffers, 0),
        tlx.local_view(do_buffers, 0),
        Q,
        DO,
        q_start,
        q_len,
        first_q_head,
        0,
        HQ,
        D,
        BLOCK_M,
        qdo_async_layout,
    )
    second_step = tl.minimum(1, total_outer_steps - 1)
    second_group = second_step // outer_blocks
    second_outer = second_step % outer_blocks
    _issue_qdo_bm32_async(
        tlx.local_view(q_buffers, 1),
        tlx.local_view(do_buffers, 1),
        Q,
        DO,
        q_start,
        q_len,
        first_q_head + second_group,
        second_outer,
        HQ,
        D,
        BLOCK_M,
        qdo_async_layout,
    )
    tlx.async_load_wait_group(2)
    tl.debug_barrier()

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    v_offsets = (offs_n[:, None] * HKV * D + offs_d[None, :]).to(tl.int32)
    v_offsets = tlx.require_layout(v_offsets, k_nm_layout, pin=False)
    v_valid = tl.broadcast_to((offs_n < kv_valid_rows)[:, None], (BLOCK_N, D))
    v_valid = tlx.require_layout(v_valid, k_nm_layout, pin=False)
    v_zero = tlx.zeros((BLOCK_N, D), tl.bfloat16, layout=k_nm_layout)
    v_operand = tlx.buffer_load(v_ptr, v_offsets, mask=v_valid, other=v_zero)
    v_operand = tlx.require_layout(v_operand, k_nm_layout, pin=False)
    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    DQ_ACC = DQ_ACC + q_scratch_start * D

    for outer_step in tl.range(0, total_outer_steps, loop_unroll_factor=1):
        outer_stage = outer_step % 2
        group_index = outer_step // outer_blocks
        outer_block = outer_step % outer_blocks
        q_head = first_q_head + group_index
        q_outer = tlx.local_view(q_buffers, outer_stage)
        do_outer = tlx.local_view(do_buffers, outer_stage)
        q_tiles = tlx.local_reinterpret(
            q_outer,
            tl.bfloat16,
            [1, BLOCK_M, D],
            layout=qdo_slice_smem_layout,
        )
        do_tiles = tlx.local_reinterpret(
            do_outer,
            tl.bfloat16,
            [1, BLOCK_M, D],
            layout=qdo_slice_smem_layout,
        )
        dq_base = (q_head.to(tl.int64) * TOTAL_Q_PADDED * D).to(tl.int32)
        dk, dv, v_operand = _varlen_gqa_phase_bm32(
            q_tiles,
            do_tiles,
            tlx.local_view(ds_buffers, 0),
            k_buffer,
            v_operand,
            dk,
            dv,
            outer_block,
            LSE,
            Delta,
            q_start,
            q_len,
            q_head,
            TOTAL_Q,
            SM_SCALE,
            HQ,
            D,
            BLOCK_M,
            BLOCK_N,
            mma_nm,
            mma_nd,
            k_nm_layout,
            qt_layout,
            p_nd_layout,
            q_out_layout,
        )
        next_step = (outer_step + 2) % total_outer_steps
        next_group = next_step // outer_blocks
        next_outer = next_step % outer_blocks
        _issue_qdo_bm32_async(
            q_outer,
            do_outer,
            Q,
            DO,
            q_start,
            q_len,
            first_q_head + next_group,
            next_outer,
            HQ,
            D,
            BLOCK_M,
            qdo_async_layout,
        )
        dq_lo, dq_hi, v_operand = _varlen_gqa_dq_bm32(
            tlx.local_view(ds_buffers, 0),
            k_buffer,
            v_operand,
            mma_md,
            ds_md_layout,
            k_md_layout,
            k_nm_layout,
        )
        _store_dq_bm32_native(
            dq_lo,
            dq_hi,
            DQ_ACC,
            dq_base,
            q_len,
            outer_block,
            SM_SCALE,
            D,
            mma_md,
        )
        tlx.async_load_wait_group(2)
        tl.debug_barrier()

    tlx.async_load_wait_group(0)
    dk = tlx.require_layout(dk, mma_nd, pin=False)
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    dk = tl.reshape(dk, (2, 2, 2, 2, 16, D))
    dk = tl.permute(dk, (0, 3, 1, 2, 4, 5))
    dk = tl.reshape(dk, (BLOCK_N, D))
    dk = tlx.require_layout(dk, kv_native_layout, pin=False)
    dk *= SM_SCALE
    dv = tl.reshape(dv, (2, 2, 2, 2, 16, D))
    dv = tl.permute(dv, (0, 3, 1, 2, 4, 5))
    dv = tl.reshape(dv, (BLOCK_N, D))
    dv = tlx.require_layout(dv, kv_native_layout, pin=False)
    store_n = tlx.rematerialized_range(0, BLOCK_N, 30)
    store_d = tlx.rematerialized_range(0, D, 31)
    partial_base = ((kv_global_start * HKV + kv_head.to(tl.int64)) * KV_SPLITS + split) * D
    output_ptr = tl.multiple_of(DK + partial_base, 16)
    output_v_ptr = tl.multiple_of(DV + partial_base, 16)
    output_offsets = (store_n[:, None] * HKV * KV_SPLITS * D + store_d[None, :]).to(tl.int32)
    output_offsets = tlx.require_layout(output_offsets, kv_native_layout, pin=False)
    output_mask = tl.broadcast_to((store_n < kv_valid_rows)[:, None], (BLOCK_N, D))
    output_mask = tlx.require_layout(output_mask, kv_native_layout, pin=False)
    tlx.buffer_store(dk.to(tl.bfloat16), output_ptr, output_offsets, mask=output_mask)
    tlx.buffer_store(dv.to(tl.bfloat16), output_v_ptr, output_offsets, mask=output_mask)


@triton.jit
def _varlen_bwd_interleaved_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    CuQ,
    CuKV,
    KVBlockSequence,
    KVBlockStart,
    DQ_ACC,
    DK,
    DV,
    TASK_OFFSET,
    SM_SCALE: tl.constexpr,
    TOTAL_Q,
    TOTAL_Q_PADDED,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FULL_KV_TILE: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    V_STRIDE_T: tl.constexpr,
):
    """Compute current dK/dV while consuming the preceding dS phase for dQ."""
    tl.static_assert(D == 128)
    tl.static_assert(BLOCK_M == 16)
    tl.static_assert(BLOCK_N == 128)
    tl.static_assert(HQ % HKV == 0)
    tl.static_assert(KV_SPLITS > 0)
    tl.static_assert(KV_SPLITS <= 4)
    tl.static_assert((HQ // HKV) % KV_SPLITS == 0)

    task = tl.program_id(0) + TASK_OFFSET
    kv_head_split = tl.program_id(1)
    kv_head = kv_head_split // KV_SPLITS
    split = kv_head_split % KV_SPLITS
    group_size: tl.constexpr = HQ // HKV
    heads_per_split: tl.constexpr = group_size // KV_SPLITS
    batch = tl.load(KVBlockSequence + task)
    n0 = tl.load(KVBlockStart + task)
    q_start = tl.load(CuQ + batch).to(tl.int64)
    q_end = tl.load(CuQ + batch + 1).to(tl.int64)
    kv_start = tl.load(CuKV + batch).to(tl.int64)
    kv_end = tl.load(CuKV + batch + 1).to(tl.int64)
    q_len = (q_end - q_start).to(tl.int32)
    kv_len = (kv_end - kv_start).to(tl.int32)
    q_blocks = (q_len + BLOCK_M - 1) // BLOCK_M
    first_q_block = n0 // BLOCK_M if IS_CAUSAL else 0
    active_q_blocks = q_blocks - first_q_block
    total_steps = heads_per_split * active_q_blocks

    qdo_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [8, 0], [1, 0], [2, 0], [4, 0]],
        [BLOCK_M, D],
    )
    k_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0],
         [8, 0]],
        [BLOCK_N, D],
    )
    ds_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 16)],
        [[1, 0], [2, 0], [0, 1], [0, 2], [4, 0], [0, 8], [8, 0], [0, 32], [0, 16], [0, 4], [0, 64]],
        [BLOCK_M, BLOCK_N],
    )
    mma_md: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    mma_nm: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_nd: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[2, 2],
    )
    k_nm_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    qt_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_nm, k_width=8)
    p_nd_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_nd, k_width=8)
    q_nd_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_nd, k_width=8)
    ds_md_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_md, k_width=8)
    k_md_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_md, k_width=8)

    k_buffer = tlx.local_alloc((BLOCK_N, D), tl.bfloat16, 1, layout=k_layout)
    q_buffers = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 2, layout=qdo_layout)
    do_buffers = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 2, layout=qdo_layout)
    ds_buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.bfloat16, 2, layout=ds_layout)

    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    global_n = kv_start + offs_n
    kv_offsets = (global_n[:, None] * HKV + kv_head) * D + offs_d[None, :]
    if FULL_KV_TILE:
        k_token = tlx.async_load(K + kv_offsets, tlx.local_view(k_buffer, 0))
    else:
        kv_mask = offs_n[:, None] < kv_len
        k_token = tlx.async_load(K + kv_offsets, tlx.local_view(k_buffer, 0), mask=kv_mask, other=0.0)
    tlx.async_load_commit_group([k_token])
    _issue_qdo_async(
        tlx.local_view(q_buffers, 0),
        tlx.local_view(do_buffers, 0),
        Q,
        DO,
        q_start,
        q_len,
        kv_head * group_size + split * heads_per_split,
        first_q_block,
        HQ,
        D,
        BLOCK_M,
    )
    initial_wait = tlx.async_load_wait_group(0)
    tl.debug_barrier()

    if IS_CAUSAL:
        v_base = V + kv_start * V_STRIDE_T + kv_head.to(tl.int64) * D
        v_offsets = offs_n[:, None] * V_STRIDE_T + offs_d[None, :]
    else:
        v_base = V
        v_offsets = kv_offsets
    v_offsets = tlx.require_layout(v_offsets.to(tl.int32), k_nm_layout, pin=False)
    if FULL_KV_TILE:
        v_tile = tlx.buffer_load(v_base, v_offsets)
    else:
        v_valid = tlx.require_layout(tl.broadcast_to(kv_mask, (BLOCK_N, D)), k_nm_layout, pin=False)
        v_zero = tlx.zeros((BLOCK_N, D), tl.bfloat16, layout=k_nm_layout)
        v_tile = tlx.buffer_load(v_base, v_offsets, mask=v_valid, other=v_zero)
    v_tile = tlx.require_layout(v_tile, k_nm_layout, pin=False)
    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    q_scratch_start = q_start + batch.to(tl.int64) * (BLOCK_M - 1)
    log2e: tl.constexpr = 1.4426950408889634

    for step in tl.range(0, total_steps, num_stages=1):
        current_slot = step % 2
        next_slot = 1 - current_slot
        if KV_SPLITS > 1 and heads_per_split == 1:
            q_step = first_q_block + step
            q_head = kv_head * group_size + split
            next_step = tl.minimum(step + 1, total_steps - 1)
            next_q_step = first_q_block + next_step
            next_q_head = q_head
        else:
            group_index = step // active_q_blocks
            q_step = first_q_block + step % active_q_blocks
            q_head = kv_head * group_size + split * heads_per_split + group_index
            next_step = tl.minimum(step + 1, total_steps - 1)
            next_group_index = next_step // active_q_blocks
            next_q_step = first_q_block + next_step % active_q_blocks
            next_q_head = kv_head * group_size + split * heads_per_split + next_group_index
        _issue_qdo_async(
            tlx.local_view(q_buffers, next_slot),
            tlx.local_view(do_buffers, next_slot),
            Q,
            DO,
            q_start,
            q_len,
            next_q_head,
            next_q_step,
            HQ,
            D,
            BLOCK_M,
        )
        tlx.async_load_wait_group(1)
        tl.debug_barrier()

        q_view = tlx.local_view(q_buffers, current_slot)
        do_view = tlx.local_view(do_buffers, current_slot)
        q_tile = tlx.local_load(q_view, layout=q_nd_layout)
        q_t = tlx.local_load(tlx.local_trans(q_view), layout=qt_layout)
        do_tile = tlx.local_load(do_view, layout=q_nd_layout)
        do_t = tlx.local_load(tlx.local_trans(do_view), layout=qt_layout)
        k_tile = tlx.local_load(tlx.local_view(k_buffer, 0), token=initial_wait, layout=k_nm_layout)
        score_acc = tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=mma_nm)
        scores_t = tl.dot(k_tile, q_t, acc=score_acc, out_dtype=score_acc.dtype)
        rows = q_step * BLOCK_M + tl.arange(0, BLOCK_M)
        global_m = q_start + rows
        lse = tl.load(LSE + q_head * TOTAL_Q + global_m, mask=rows < q_len, other=0.0)
        delta = tl.load(Delta + global_m * HQ + q_head, mask=rows < q_len, other=0.0)
        score_scale = tlx.require_layout(
            tl.full((BLOCK_N, BLOCK_M), SM_SCALE * log2e, tl.float32),
            mma_nm,
            pin=False,
        )
        lse_full = tlx.require_layout(
            tl.broadcast_to((lse * log2e)[None, :], (BLOCK_N, BLOCK_M)),
            mma_nm,
            pin=False,
        )
        scores_t = scores_t * score_scale - lse_full
        if FULL_KV_TILE:
            valid = tl.broadcast_to(rows[None, :] < q_len, (BLOCK_N, BLOCK_M))
        else:
            valid = (offs_n[:, None] < kv_len) & (rows[None, :] < q_len)
        valid = tlx.require_layout(valid, mma_nm, pin=False)
        if IS_CAUSAL:
            query_fragment = q_step - first_q_block
            if query_fragment < BLOCK_N // BLOCK_M:
                causal_n = n0 + tlx.rematerialized_range(0, BLOCK_N, 32, placement=step)
                causal_m = q_step * BLOCK_M + tlx.rematerialized_range(0, BLOCK_M, 33, placement=step)
                causal_valid = causal_n[:, None] <= causal_m[None, :]
                causal_valid = tlx.require_layout(causal_valid, mma_nm, pin=False)
                valid = valid & causal_valid
            neg_inf = tlx.require_layout(
                tl.full((BLOCK_N, BLOCK_M), float("-inf"), dtype=tl.float32),
                mma_nm,
                pin=False,
            )
            scores_t = tl.where(valid, scores_t, neg_inf)
            scores_t = tlx.require_layout(scores_t, mma_nm, pin=False)
            p_t = tlx.require_layout(tl.math.exp2(scores_t), mma_nm, pin=False)
        else:
            p_t = tlx.require_layout(tl.where(valid, tl.math.exp2(scores_t), 0.0), mma_nm, pin=False)
        dp_acc = tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=mma_nm)
        dp_t = tl.dot(v_tile, do_t, acc=dp_acc, out_dtype=dp_acc.dtype)
        delta_full = tlx.require_layout(
            tl.broadcast_to(delta[None, :], (BLOCK_N, BLOCK_M)),
            mma_nm,
            pin=False,
        )
        ds_t = p_t * (dp_t - delta_full)
        ds_bf16 = ds_t.to(tl.bfloat16)
        current_ds = tlx.local_view(ds_buffers, current_slot)
        tlx.local_store(current_ds, tl.trans(ds_bf16))
        p_nd = tlx.require_layout(p_t.to(tl.bfloat16), p_nd_layout, pin=False)
        ds_nd = tlx.require_layout(ds_bf16, p_nd_layout, pin=False)
        dv = tl.dot(p_nd, do_tile, acc=dv, out_dtype=dv.dtype)
        dk = tl.dot(ds_nd, q_tile, acc=dk, out_dtype=dk.dtype)

        if step > 0:
            previous_step = step - 1
            if KV_SPLITS > 1 and heads_per_split == 1:
                previous_q_step = first_q_block + previous_step
                previous_q_head = kv_head * group_size + split
            else:
                previous_group_index = previous_step // active_q_blocks
                previous_q_step = first_q_block + previous_step % active_q_blocks
                previous_q_head = kv_head * group_size + split * heads_per_split + previous_group_index
            previous_dq_acc_base = ((previous_q_head.to(tl.int64) * TOTAL_Q_PADDED + q_scratch_start) * D).to(tl.int32)
            previous_ds = tlx.local_load(tlx.local_view(ds_buffers, 1 - current_slot), layout=ds_md_layout)
            k_for_dq = tlx.local_load(tlx.local_view(k_buffer, 0), token=initial_wait, layout=k_md_layout)
            dq_acc = tlx.zeros((BLOCK_M, D), tl.float32, layout=mma_md)
            dq_part = tl.dot(previous_ds, k_for_dq, acc=dq_acc, out_dtype=dq_acc.dtype)
            _store_dq_native(
                dq_part,
                DQ_ACC,
                previous_dq_acc_base,
                q_len,
                previous_q_step,
                SM_SCALE,
                D,
                BLOCK_M,
                mma_md,
            )
        tl.debug_barrier()

    tlx.async_load_wait_group(0)
    tl.debug_barrier()
    last_step = total_steps - 1
    if KV_SPLITS > 1 and heads_per_split == 1:
        last_q_step = first_q_block + last_step
        last_q_head = kv_head * group_size + split
    else:
        last_group_index = last_step // active_q_blocks
        last_q_step = first_q_block + last_step % active_q_blocks
        last_q_head = kv_head * group_size + split * heads_per_split + last_group_index
    last_dq_acc_base = ((last_q_head.to(tl.int64) * TOTAL_Q_PADDED + q_scratch_start) * D).to(tl.int32)
    last_ds = tlx.local_load(tlx.local_view(ds_buffers, last_step % 2), layout=ds_md_layout)
    k_for_dq = tlx.local_load(tlx.local_view(k_buffer, 0), layout=k_md_layout)
    dq_acc = tlx.zeros((BLOCK_M, D), tl.float32, layout=mma_md)
    dq_part = tl.dot(last_ds, k_for_dq, acc=dq_acc, out_dtype=dq_acc.dtype)
    _store_dq_native(
        dq_part,
        DQ_ACC,
        last_dq_acc_base,
        q_len,
        last_q_step,
        SM_SCALE,
        D,
        BLOCK_M,
        mma_md,
    )

    dk_scale = tlx.require_layout(tl.full((BLOCK_N, D), SM_SCALE, dtype=tl.float32), mma_nd, pin=False)
    dk *= dk_scale
    if KV_SPLITS == 1:
        output_offsets = tlx.require_layout(kv_offsets.to(tl.int32), mma_nd, pin=False)
    else:
        partial_offsets = (((global_n[:, None] * HKV + kv_head) * KV_SPLITS + split) * D + offs_d[None, :])
        output_offsets = tlx.require_layout(partial_offsets.to(tl.int32), mma_nd, pin=False)
    if FULL_KV_TILE:
        tlx.buffer_store(dk.to(tl.bfloat16), DK, output_offsets)
        tlx.buffer_store(dv.to(tl.bfloat16), DV, output_offsets)
    else:
        output_mask = tlx.require_layout(tl.broadcast_to(kv_mask, (BLOCK_N, D)), mma_nd, pin=False)
        tlx.buffer_store(dk.to(tl.bfloat16), DK, output_offsets, mask=output_mask)
        tlx.buffer_store(dv.to(tl.bfloat16), DV, output_offsets, mask=output_mask)


@triton.jit
def _varlen_dkdv_reduce_kernel(
    DK_PART,
    DV_PART,
    DK,
    DV,
    TOTAL_KV,
    HKV: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(KV_SPLITS > 1)
    tl.static_assert(KV_SPLITS <= 4)
    pid_n = tl.program_id(0)
    kv_head = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    valid = tl.broadcast_to((offs_n < TOTAL_KV)[:, None], (BLOCK_N, D))
    partial_base = ((offs_n[:, None] * HKV + kv_head) * KV_SPLITS) * D + offs_d[None, :]
    dk = tl.zeros((BLOCK_N, D), tl.float32)
    dv = tl.zeros((BLOCK_N, D), tl.float32)
    for split in tl.static_range(0, KV_SPLITS):
        dk += tl.load(DK_PART + partial_base + split * D, mask=valid, other=0.0).to(tl.float32)
        dv += tl.load(DV_PART + partial_base + split * D, mask=valid, other=0.0).to(tl.float32)
    output_offsets = (offs_n[:, None] * HKV + kv_head) * D + offs_d[None, :]
    tl.store(DK + output_offsets, dk.to(tl.bfloat16), mask=valid)
    tl.store(DV + output_offsets, dv.to(tl.bfloat16), mask=valid)


@triton.jit
def _varlen_dq_convert_kernel(
    DQ_ACC,
    CuQ,
    QBlockSequence,
    QBlockStart,
    DQ,
    TOTAL_Q_PADDED,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    task = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.load(QBlockSequence + task)
    start_m = tl.load(QBlockStart + task)
    q_start = tl.load(CuQ + batch).to(tl.int64)
    q_end = tl.load(CuQ + batch + 1).to(tl.int64)
    q_len = (q_end - q_start).to(tl.int32)

    local_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    d_swizzled = ((offs_d & 1)
                  | ((offs_d & 2) << 6)
                  | ((offs_d & 12) << 3)
                  | ((offs_d & 48) << 5)
                  | ((offs_d & 64) << 2))
    native_offsets = ((local_m[:, None] << 1) | d_swizzled[None, :]).to(tl.int32)
    native_offsets = tl.max_contiguous(native_offsets, [1, 2])
    valid = tl.broadcast_to((start_m + local_m < q_len)[:, None], (BLOCK_M, D))
    q_scratch_start = q_start + batch.to(tl.int64) * (BLOCK_M - 1)
    native_base = (head.to(tl.int64) * TOTAL_Q_PADDED + q_scratch_start + start_m) * D
    values = tl.load(DQ_ACC + native_base + native_offsets, mask=valid, other=0.0)

    global_m = q_start + start_m + local_m
    output_offsets = (global_m[:, None] * HEADS + head) * D + offs_d[None, :]
    tl.store(DQ + output_offsets, values, mask=valid)


def _validate_backward_inputs(q, k, v, o, do, lse, plan, sm_scale, causal):
    if not isinstance(plan, VarlenBackwardPlan):
        raise TypeError("plan must be a VarlenBackwardPlan")
    if not math.isfinite(float(sm_scale)):
        raise ValueError("sm_scale must be finite")
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must be rank-3 packed THD tensors")
    total_q, heads, head_dim = q.shape
    total_kv, kv_heads, kv_dim = k.shape
    if (total_q, total_kv) != (plan.total_q, plan.total_kv):
        raise ValueError("q and k token counts must match the prepared plan")
    if heads == 0 or kv_heads == 0:
        raise ValueError("packed backward requires positive Q and KV head counts")
    if heads % kv_heads != 0:
        raise ValueError("packed D128 backward requires Q heads divisible by KV heads")
    if causal and not plan.qk_offsets_equal:
        raise ValueError("causal packed backward requires identical Q and KV cumulative offsets")
    if causal and heads != kv_heads:
        raise ValueError("causal packed backward currently requires equal Q and KV head counts")
    if head_dim != 128 or kv_dim != 128:
        raise ValueError("packed backward currently requires head dimension 128")
    _validate_i32_buffer_offsets(
        total_q=total_q,
        total_kv=total_kv,
        batch=plan.batch,
        q_heads=heads,
        kv_heads=kv_heads,
    )
    q_tensors = {"q": q, "o": o, "do": do}
    for name, tensor in q_tensors.items():
        if tensor.shape != q.shape or tensor.device != q.device:
            raise ValueError(f"{name} must match q shape and device")
        if tensor.dtype is not torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous bfloat16 THD")
    if k.shape != (total_kv, kv_heads, head_dim) or k.device != q.device:
        raise ValueError("k must match its packed THD shape and q device")
    if k.dtype is not torch.bfloat16 or not k.is_contiguous():
        raise ValueError("k must be contiguous bfloat16 THD")
    if v.shape != k.shape or v.device != q.device:
        raise ValueError("v must match k shape and q device")
    if v.dtype is not torch.bfloat16:
        raise ValueError("v must be bfloat16 THD")
    if causal:
        dense_head_axes = v.stride(-1) == 1 and v.stride(-2) == head_dim
        nonoverlapping_tokens = v.stride(0) >= kv_heads * head_dim
        if not dense_head_axes or not nonoverlapping_tokens:
            raise ValueError("causal v must have dense head/D axes and a positive non-overlapping token stride")
        if plan.max_q * v.stride(0) > _I32_BUFFER_BF16_ELEMENTS:
            raise ValueError("causal V sequence stride exceeds the signed 32-bit byte-offset range")
    elif not v.is_contiguous():
        raise ValueError("v must be contiguous bfloat16 THD")
    if plan.cu_seqlens_q.device != q.device or plan.cu_seqlens_k.device != q.device:
        raise ValueError("the prepared plan and inputs must be on the same device")
    if lse.shape != (heads, total_q) or lse.device != q.device or lse.dtype is not torch.float32:
        raise ValueError("lse must be contiguous FP32 with shape (heads, total_q)")
    if not lse.is_contiguous():
        raise ValueError("lse must be contiguous FP32 with shape (heads, total_q)")
    arch = torch.cuda.get_device_properties(q.device).gcnArchName
    if not arch.startswith("gfx950"):
        raise ValueError(f"gfx950 is required, got {arch}")


def fa_varlen_backward(q, k, v, o, do, lse, plan, sm_scale, causal=False):
    """Run packed BF16 D128 backward using a prepared immutable-offset plan.

    Non-causal mode supports MHA/GQA with independent Q and KV offsets.  Causal
    mode is limited to self-attention MHA, so Q/KV offsets and head counts must
    match.  Its V input may have a larger token stride when the head and D axes
    remain dense, as in TritonBench's ``v_storage[:, 0]`` view.
    """
    _validate_backward_inputs(q, k, v, o, do, lse, plan, sm_scale, causal)
    total_q, heads, head_dim = q.shape
    total_kv, kv_heads, _ = k.shape
    group_size = heads // kv_heads
    kv_splits = _select_varlen_kv_splits(plan.max_q, group_size)
    total_q_padded = total_q + plan.batch * (_BLOCK_M - 1)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(k)
    dk_part, dv_part = _allocate_varlen_dkdv_partials(k, kv_splits)
    if dk_part is None:
        kv_splits = 1
    block_m, block_n = _select_varlen_kernel_blocks(group_size, kv_splits)
    dk_target = dk if dk_part is None else dk_part
    dv_target = dv if dv_part is None else dv_part
    delta = torch.empty((total_q, heads), dtype=torch.float32, device=q.device)
    dq_acc = torch.zeros((heads, total_q_padded, head_dim), dtype=torch.bfloat16, device=q.device)

    _varlen_bwd_preprocess[(triton.cdiv(plan.max_q, 64), plan.batch * heads)](
        o,
        do,
        delta,
        plan.cu_seqlens_q,
        HEADS=heads,
        D=head_dim,
        BLOCK_M=64,
        num_warps=4,
    )
    if (block_m, block_n) == (_WIDE_BLOCK_M, _WIDE_BLOCK_N):
        _varlen_bwd_interleaved_bm32_kernel[(kv_heads * kv_splits, plan.wide_kv_start.numel())](
            q,
            k,
            v,
            do,
            lse,
            delta,
            plan.wide_kv_start,
            plan.wide_q_start,
            plan.wide_dq_start,
            plan.wide_q_len,
            plan.wide_kv_valid,
            dq_acc,
            dk_target,
            dv_target,
            SM_SCALE=sm_scale,
            TOTAL_Q=total_q,
            TOTAL_Q_PADDED=total_q_padded,
            HQ=heads,
            HKV=kv_heads,
            D=head_dim,
            BLOCK_M=_WIDE_BLOCK_M,
            BLOCK_N=_WIDE_BLOCK_N,
            KV_SPLITS=kv_splits,
            num_warps=4,
            num_stages=1,
            matrix_instr_nonkdim=16,
        )
    else:
        kv_launches = (
            (0, plan.num_full_kv_blocks, True),
            (plan.num_full_kv_blocks, plan.kv_block_sequence.numel() - plan.num_full_kv_blocks, False),
        )
        for task_offset, task_count, full_kv_tile in kv_launches:
            if task_count == 0:
                continue
            _varlen_bwd_interleaved_kernel[(task_count, kv_heads * kv_splits)](
                q,
                k,
                v,
                do,
                lse,
                delta,
                plan.cu_seqlens_q,
                plan.cu_seqlens_k,
                plan.kv_block_sequence,
                plan.kv_block_start,
                dq_acc,
                dk_target,
                dv_target,
                task_offset,
                SM_SCALE=sm_scale,
                TOTAL_Q=total_q,
                TOTAL_Q_PADDED=total_q_padded,
                HQ=heads,
                HKV=kv_heads,
                D=head_dim,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                FULL_KV_TILE=full_kv_tile,
                KV_SPLITS=kv_splits,
                IS_CAUSAL=causal,
                V_STRIDE_T=v.stride(0),
                num_warps=4,
                num_stages=1,
                matrix_instr_nonkdim=16,
            )
    if kv_splits > 1:
        _varlen_dkdv_reduce_kernel[(triton.cdiv(total_kv, _BLOCK_M), kv_heads)](
            dk_part,
            dv_part,
            dk,
            dv,
            total_kv,
            HKV=kv_heads,
            D=head_dim,
            KV_SPLITS=kv_splits,
            BLOCK_N=_BLOCK_M,
            num_warps=4,
        )
    _varlen_dq_convert_kernel[(plan.q_block_sequence.numel(), heads)](
        dq_acc,
        plan.cu_seqlens_q,
        plan.q_block_sequence,
        plan.q_block_start,
        dq,
        TOTAL_Q_PADDED=total_q_padded,
        HEADS=heads,
        D=head_dim,
        BLOCK_M=_BLOCK_M,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    return dq, dk, dv
