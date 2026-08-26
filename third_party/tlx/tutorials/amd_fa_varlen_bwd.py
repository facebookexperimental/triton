"""Packed variable-length BF16 D128 attention backward for gfx950.

Each workgroup owns one 128-row KV tile and walks the corresponding sequence
in 16-row query phases.  The current phase accumulates dK/dV and publishes dS
through LDS; the preceding dS phase is consumed for dQ.  Independent KV owners
combine their dQ contributions with BF16 atomics in a guarded native layout,
followed by a conversion to packed THD order.

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

_BLOCK_M = 16
_BLOCK_N = 128
_HEAD_DIM = 128
_I32_BUFFER_BF16_ELEMENTS = 1 << 30


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
    batch: int
    total_q: int
    total_kv: int
    max_q: int


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
    kv_block_sequence, kv_block_start = _make_block_schedule(k_lengths, _BLOCK_N, cu_seqlens_q.device)

    return VarlenBackwardPlan(
        cu_seqlens_q=owned_q,
        cu_seqlens_k=owned_k,
        q_block_sequence=q_block_sequence,
        q_block_start=q_block_start,
        kv_block_sequence=kv_block_sequence,
        kv_block_start=kv_block_start,
        batch=len(q_lengths),
        total_q=q_offsets[-1],
        total_kv=k_offsets[-1],
        max_q=max(q_lengths),
    )


def _validate_i32_buffer_offsets(*, total_q: int, total_kv: int, batch: int, heads: int) -> None:
    if total_kv * heads * _HEAD_DIM > _I32_BUFFER_BF16_ELEMENTS:
        raise ValueError("KV tensor size exceeds the signed 32-bit byte-offset range")
    total_q_padded = total_q + batch * (_BLOCK_M - 1)
    if total_q_padded * heads * _HEAD_DIM > _I32_BUFFER_BF16_ELEMENTS:
        raise ValueError("padded dQ size exceeds the signed 32-bit byte-offset range")


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
    head,
    step,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = step * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    global_rows = q_start + rows
    offsets = (global_rows[:, None] * HEADS + head) * D + dims[None, :]
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
    SM_SCALE: tl.constexpr,
    TOTAL_Q,
    TOTAL_Q_PADDED,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute current dK/dV while consuming the preceding dS phase for dQ."""
    tl.static_assert(D == 128)
    tl.static_assert(BLOCK_M == 16)
    tl.static_assert(BLOCK_N == 128)

    task = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.load(KVBlockSequence + task)
    n0 = tl.load(KVBlockStart + task)
    q_start = tl.load(CuQ + batch).to(tl.int64)
    q_end = tl.load(CuQ + batch + 1).to(tl.int64)
    kv_start = tl.load(CuKV + batch).to(tl.int64)
    kv_end = tl.load(CuKV + batch + 1).to(tl.int64)
    q_len = (q_end - q_start).to(tl.int32)
    kv_len = (kv_end - kv_start).to(tl.int32)

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
    kv_offsets = (global_n[:, None] * HEADS + head) * D + offs_d[None, :]
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
        head,
        0,
        HEADS,
        D,
        BLOCK_M,
    )
    initial_wait = tlx.async_load_wait_group(0)
    tl.debug_barrier()

    v_offsets = tlx.require_layout(kv_offsets.to(tl.int32), k_nm_layout, pin=False)
    v_valid = tlx.require_layout(tl.broadcast_to(kv_mask, (BLOCK_N, D)), k_nm_layout, pin=False)
    v_zero = tlx.zeros((BLOCK_N, D), tl.bfloat16, layout=k_nm_layout)
    v_tile = tlx.buffer_load(V, v_offsets, mask=v_valid, other=v_zero)
    v_tile = tlx.require_layout(v_tile, k_nm_layout, pin=False)
    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    q_blocks = (q_len + BLOCK_M - 1) // BLOCK_M
    q_scratch_start = q_start + batch.to(tl.int64) * (BLOCK_M - 1)
    dq_acc_base = ((head.to(tl.int64) * TOTAL_Q_PADDED + q_scratch_start) * D).to(tl.int32)
    log2e: tl.constexpr = 1.4426950408889634

    for step in tl.range(0, q_blocks, num_stages=1):
        current_slot = step % 2
        next_slot = 1 - current_slot
        _issue_qdo_async(
            tlx.local_view(q_buffers, next_slot),
            tlx.local_view(do_buffers, next_slot),
            Q,
            DO,
            q_start,
            q_len,
            head,
            step + 1,
            HEADS,
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
        rows = step * BLOCK_M + tl.arange(0, BLOCK_M)
        global_m = q_start + rows
        lse = tl.load(LSE + head * TOTAL_Q + global_m, mask=rows < q_len, other=0.0)
        delta = tl.load(Delta + global_m * HEADS + head, mask=rows < q_len, other=0.0)
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
        valid = (offs_n[:, None] < kv_len) & (rows[None, :] < q_len)
        valid = tlx.require_layout(valid, mma_nm, pin=False)
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
            previous_ds = tlx.local_load(tlx.local_view(ds_buffers, 1 - current_slot), layout=ds_md_layout)
            k_for_dq = tlx.local_load(tlx.local_view(k_buffer, 0), token=initial_wait, layout=k_md_layout)
            dq_acc = tlx.zeros((BLOCK_M, D), tl.float32, layout=mma_md)
            dq_part = tl.dot(previous_ds, k_for_dq, acc=dq_acc, out_dtype=dq_acc.dtype)
            _store_dq_native(
                dq_part,
                DQ_ACC,
                dq_acc_base,
                q_len,
                step - 1,
                SM_SCALE,
                D,
                BLOCK_M,
                mma_md,
            )
        tl.debug_barrier()

    tlx.async_load_wait_group(0)
    tl.debug_barrier()
    last_ds = tlx.local_load(tlx.local_view(ds_buffers, (q_blocks - 1) % 2), layout=ds_md_layout)
    k_for_dq = tlx.local_load(tlx.local_view(k_buffer, 0), layout=k_md_layout)
    dq_acc = tlx.zeros((BLOCK_M, D), tl.float32, layout=mma_md)
    dq_part = tl.dot(last_ds, k_for_dq, acc=dq_acc, out_dtype=dq_acc.dtype)
    _store_dq_native(
        dq_part,
        DQ_ACC,
        dq_acc_base,
        q_len,
        q_blocks - 1,
        SM_SCALE,
        D,
        BLOCK_M,
        mma_md,
    )

    dk_scale = tlx.require_layout(tl.full((BLOCK_N, D), SM_SCALE, dtype=tl.float32), mma_nd, pin=False)
    dk *= dk_scale
    output_offsets = tlx.require_layout(kv_offsets.to(tl.int32), mma_nd, pin=False)
    output_mask = tlx.require_layout(tl.broadcast_to(kv_mask, (BLOCK_N, D)), mma_nd, pin=False)
    tlx.buffer_store(dk.to(tl.bfloat16), DK, output_offsets, mask=output_mask)
    tlx.buffer_store(dv.to(tl.bfloat16), DV, output_offsets, mask=output_mask)


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


def _validate_backward_inputs(q, k, v, o, do, lse, plan, sm_scale):
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
    if heads != kv_heads:
        raise ValueError("packed D128 backward currently requires equal Q and KV head counts")
    if heads == 0:
        raise ValueError("packed backward requires a positive head count")
    if head_dim != 128 or kv_dim != 128:
        raise ValueError("packed backward currently requires head dimension 128")
    _validate_i32_buffer_offsets(
        total_q=total_q,
        total_kv=total_kv,
        batch=plan.batch,
        heads=heads,
    )
    q_tensors = {"q": q, "o": o, "do": do}
    for name, tensor in q_tensors.items():
        if tensor.shape != q.shape or tensor.device != q.device:
            raise ValueError(f"{name} must match q shape and device")
        if tensor.dtype is not torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous bfloat16 THD")
    for name, tensor in {"k": k, "v": v}.items():
        if tensor.shape != k.shape or tensor.device != q.device:
            raise ValueError(f"{name} must match k shape and q device")
        if tensor.dtype is not torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous bfloat16 THD")
    if plan.cu_seqlens_q.device != q.device or plan.cu_seqlens_k.device != q.device:
        raise ValueError("the prepared plan and inputs must be on the same device")
    if lse.shape != (heads, total_q) or lse.device != q.device or lse.dtype is not torch.float32:
        raise ValueError("lse must be contiguous FP32 with shape (heads, total_q)")
    if not lse.is_contiguous():
        raise ValueError("lse must be contiguous FP32 with shape (heads, total_q)")
    arch = torch.cuda.get_device_properties(q.device).gcnArchName
    if not arch.startswith("gfx950"):
        raise ValueError(f"gfx950 is required, got {arch}")


def fa_varlen_backward(q, k, v, o, do, lse, plan, sm_scale):
    """Run packed BF16 D128 non-causal MHA backward using a prepared plan."""
    _validate_backward_inputs(q, k, v, o, do, lse, plan, sm_scale)
    total_q, heads, head_dim = q.shape
    total_q_padded = total_q + plan.batch * (_BLOCK_M - 1)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
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
    _varlen_bwd_interleaved_kernel[(plan.kv_block_sequence.numel(), heads)](
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
        dk,
        dv,
        SM_SCALE=sm_scale,
        TOTAL_Q=total_q,
        TOTAL_Q_PADDED=total_q_padded,
        HEADS=heads,
        D=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
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
