"""Packed variable-length BF16 D128 attention backward for gfx950.

Each workgroup owns a KV tile and a subset of its mapped query heads.  The
baseline path uses BN128/BM16 phases and supports non-causal MHA/GQA plus
causal self-attention MHA; long non-causal split-GQA uses masked BN256/BM32
phases and forms dQ as two native BM16 accumulator chains.  Split workgroups
preserve FP32 dK/dV partials through their final reduction.  Independent KV
owners combine dQ contributions with BF16 atomics in a guarded native layout,
followed by a conversion to packed THD order.

Call :func:`prepare_varlen_backward` once and reuse the resulting plan with
:func:`fa_varlen_backward` on the same CUDA stream. When token metadata is
supplied, plan creation builds compact schedules asynchronously without copying
offsets or task counts to the host. The legacy two-argument path copies offsets
to the CPU to infer metadata. Treat every plan-owned offset and schedule tensor
as immutable after preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

_BLOCK_M = 16
_BLOCK_N = 128
_WIDE_BLOCK_M = 32
_WIDE_BLOCK_N = 256
_HEAD_DIM = 128
_I32_BUFFER_BF16_ELEMENTS = 1 << 30
_I32_BUFFER_FP32_ELEMENTS = _I32_BUFFER_BF16_ELEMENTS // 2
# This is an empirical gfx950 crossover, not a hardware limit.  The work metric
# is group_size * ceil(max_q / 16), the BM16 query-block phases per KV head.
# Local sweeps kept max-Q 300/400 on BM16/BN128, while max-Q 5662 GQA3/GQA8
# amortized the BM32/BN256 split kernel and its dK/dV reduction overhead.
_VARLEN_GQA_SPLIT_WORK_THRESHOLD = 1024

# Slots in the device-resident task_counts vector: schedule lengths first,
# followed by general and causal offset-validation flags.
_Q_TASK_COUNT = tl.constexpr(0)
_FULL_KV_TASK_COUNT = tl.constexpr(1)
_TAIL_KV_TASK_COUNT = tl.constexpr(2)
_WIDE_KV_TASK_COUNT = tl.constexpr(3)
_PLAN_ERROR = tl.constexpr(4)
_CAUSAL_PLAN_ERROR = tl.constexpr(5)
_NUM_TASK_COUNTS = tl.constexpr(6)


@dataclass(frozen=True)
class VarlenBackwardPlan:
    """Reusable device-built launch metadata for immutable packed offsets.

    ``cu_seqlens_q`` and ``cu_seqlens_k`` are prefix sums delimiting each
    sequence in the packed Q and KV tensors. The compact schedules are parallel
    arrays: ``q_block_*`` maps each dQ-conversion task to a sequence and a
    sequence-local Q row, while ``full_kv_block_*`` and ``tail_kv_block_*`` do
    the same for complete and masked-tail KV tiles.

    The ``wide_*`` arrays describe one split-GQA task per wide KV tile: its
    packed KV and Q starts, its padded dQ-scratch start, the Q length, and the
    number of valid KV rows. Schedule buffers are allocated to conservative
    capacity bounds; ``task_counts`` holds their device-produced valid lengths
    as well as general and causal offset-validation flags.

    ``batch``, token totals, and maximum sequence lengths are host launch
    metadata. ``qk_offsets_equal`` is ``True`` or ``False`` when equality was
    established on the host, and ``None`` when causal compatibility remains
    device-validated. The optional dQ fields retain compatibility with optimized
    legacy plans.

    ``frozen=True`` prevents field rebinding, but tensor contents remain
    mutable. Prepare and consume the plan on the same CUDA stream, and do not
    modify its offset, schedule, or count tensors after preparation.
    """

    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    q_block_sequence: torch.Tensor
    q_block_start: torch.Tensor
    full_kv_block_sequence: torch.Tensor
    full_kv_block_start: torch.Tensor
    tail_kv_block_sequence: torch.Tensor
    tail_kv_block_start: torch.Tensor
    wide_kv_start: torch.Tensor
    wide_q_start: torch.Tensor
    wide_dq_start: torch.Tensor
    wide_q_len: torch.Tensor
    wide_kv_valid: torch.Tensor
    task_counts: torch.Tensor
    batch: int
    total_q: int
    total_kv: int
    max_q: int
    max_kv: int
    qk_offsets_equal: bool | None
    # Device-built plans do not materialize the optional tail-finalization
    # schedule, so they retain the separate dQ conversion path.
    dq_full_kv_sequence: torch.Tensor | None = None
    dq_full_kv_start: torch.Tensor | None = None
    dq_tail_k96: bool = False
    # Exact host-known count used only by the legacy rolling-owner candidate.
    wide_task_count: int | None = None


@triton.jit
def _varlen_validate_offsets(
    CuQ,
    CuK,
    TaskCounts,
    TOTAL_Q,
    TOTAL_KV,
    MAX_Q,
    MAX_KV,
    PLAN_ERROR_INDEX: tl.constexpr,
    CAUSAL_PLAN_ERROR_INDEX: tl.constexpr,
):
    sequence = tl.program_id(0)
    q_start = tl.load(CuQ + sequence)
    q_end = tl.load(CuQ + sequence + 1)
    kv_start = tl.load(CuK + sequence)
    kv_end = tl.load(CuK + sequence + 1)
    q_len = q_end - q_start
    kv_len = kv_end - kv_start
    is_first = sequence == 0
    is_last = sequence == tl.num_programs(0) - 1
    invalid = ((q_start < 0)
               | (kv_start < 0)
               | (q_len <= 0)
               | (kv_len <= 0)
               | (q_end > TOTAL_Q)
               | (kv_end > TOTAL_KV)
               | (q_len > MAX_Q)
               | (kv_len > MAX_KV)
               | (is_first & ((q_start != 0) | (kv_start != 0)))
               | (is_last & ((q_end != TOTAL_Q) | (kv_end != TOTAL_KV))))
    qk_mismatch = (q_start != kv_start) | (q_end != kv_end)
    if invalid:
        tl.atomic_xchg(TaskCounts + PLAN_ERROR_INDEX, 1, sem="relaxed")
    if invalid | qk_mismatch:
        tl.atomic_xchg(TaskCounts + CAUSAL_PLAN_ERROR_INDEX, 1, sem="relaxed")


@triton.jit
def _varlen_build_compact_schedules(
    CuQ,
    CuK,
    QBlockSequence,
    QBlockStart,
    FullKVBlockSequence,
    FullKVBlockStart,
    TailKVBlockSequence,
    TailKVBlockStart,
    WideKVStart,
    WideQStart,
    WideDQStart,
    WideQLen,
    WideKVValid,
    TaskCounts,
    Q_TASK_COUNT_INDEX: tl.constexpr,
    FULL_KV_TASK_COUNT_INDEX: tl.constexpr,
    TAIL_KV_TASK_COUNT_INDEX: tl.constexpr,
    WIDE_KV_TASK_COUNT_INDEX: tl.constexpr,
    PLAN_ERROR_INDEX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WIDE_BLOCK_N: tl.constexpr,
):
    if tl.load(TaskCounts + PLAN_ERROR_INDEX) != 0:
        return
    sequence = tl.program_id(0)
    q_start = tl.load(CuQ + sequence)
    q_end = tl.load(CuQ + sequence + 1)
    kv_start = tl.load(CuK + sequence)
    kv_end = tl.load(CuK + sequence + 1)
    q_len = q_end - q_start
    kv_len = kv_end - kv_start
    q_blocks = tl.cdiv(q_len, BLOCK_M)
    full_kv_blocks = kv_len // BLOCK_N
    has_kv_tail = kv_len % BLOCK_N != 0
    wide_kv_blocks = tl.cdiv(kv_len, WIDE_BLOCK_N)

    q_base = tl.atomic_add(TaskCounts + Q_TASK_COUNT_INDEX, q_blocks, sem="relaxed")
    for block in range(0, q_blocks):
        tl.store(QBlockSequence + q_base + block, sequence)
        tl.store(QBlockStart + q_base + block, block * BLOCK_M)

    full_kv_base = tl.atomic_add(TaskCounts + FULL_KV_TASK_COUNT_INDEX, full_kv_blocks, sem="relaxed")
    for block in range(0, full_kv_blocks):
        tl.store(FullKVBlockSequence + full_kv_base + block, sequence)
        tl.store(FullKVBlockStart + full_kv_base + block, block * BLOCK_N)

    if has_kv_tail:
        tail_kv_task = tl.atomic_add(TaskCounts + TAIL_KV_TASK_COUNT_INDEX, 1, sem="relaxed")
        tl.store(TailKVBlockSequence + tail_kv_task, sequence)
        tl.store(TailKVBlockStart + tail_kv_task, full_kv_blocks * BLOCK_N)

    wide_base = tl.atomic_add(TaskCounts + WIDE_KV_TASK_COUNT_INDEX, wide_kv_blocks, sem="relaxed")
    for block in range(0, wide_kv_blocks):
        block_start = block * WIDE_BLOCK_N
        task = wide_base + block
        tl.store(WideKVStart + task, kv_start + block_start)
        tl.store(WideQStart + task, q_start)
        tl.store(WideDQStart + task, q_start + sequence * (BLOCK_M - 1))
        tl.store(WideQLen + task, q_len)
        tl.store(WideKVValid + task, tl.minimum(WIDE_BLOCK_N, kv_len - block_start))


def _validate_cu_seqlens_metadata(name: str, value: torch.Tensor) -> None:
    if value.ndim != 1 or value.numel() < 2:
        raise ValueError(f"{name} must be a rank-1 tensor with at least two elements")
    if value.dtype is not torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be on a CUDA device")


def _read_cu_seqlens(name: str, value: torch.Tensor) -> tuple[list[int], list[int]]:
    offsets = value.detach().cpu().tolist()
    if offsets[0] != 0:
        raise ValueError(f"{name} must start at zero")
    lengths = [end - begin for begin, end in zip(offsets, offsets[1:])]
    if any(length <= 0 for length in lengths):
        raise ValueError(f"{name} must be strictly increasing")
    return offsets, lengths


def prepare_varlen_backward(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_q: int | None = None,
    total_kv: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> VarlenBackwardPlan:
    """Build compact schedules, using a device-only path when metadata is supplied.

    The legacy two-argument path clones strided offsets into contiguous plan
    storage. The metadata-supplied path requires contiguous offsets so its
    validation and schedule kernels can address them directly.
    """
    if cu_seqlens_q.device != cu_seqlens_k.device:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be on the same device")
    _validate_cu_seqlens_metadata("cu_seqlens_q", cu_seqlens_q)
    _validate_cu_seqlens_metadata("cu_seqlens_k", cu_seqlens_k)
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch")

    metadata = (total_q, total_kv, max_seqlen_q, max_seqlen_k)
    legacy_path = all(value is None for value in metadata)
    if legacy_path:
        shared_offsets = cu_seqlens_q is cu_seqlens_k
        cu_seqlens_q = cu_seqlens_q.detach().clone(memory_format=torch.contiguous_format)
        cu_seqlens_k = (cu_seqlens_q if shared_offsets else cu_seqlens_k.detach().clone(
            memory_format=torch.contiguous_format))
        q_offsets, q_lengths = _read_cu_seqlens("cu_seqlens_q", cu_seqlens_q)
        if shared_offsets:
            k_offsets, k_lengths = q_offsets, q_lengths
        else:
            k_offsets, k_lengths = _read_cu_seqlens("cu_seqlens_k", cu_seqlens_k)
        total_q = q_offsets[-1]
        total_kv = k_offsets[-1]
        max_seqlen_q = max(q_lengths)
        max_seqlen_k = max(k_lengths)
        qk_offsets_equal = q_offsets == k_offsets
        wide_task_count = sum(triton.cdiv(length, _WIDE_BLOCK_N) for length in k_lengths)
    elif any(value is None for value in metadata):
        raise ValueError("token totals and maximum sequence lengths must be supplied together")
    else:
        for name, value in (
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_k", cu_seqlens_k),
        ):
            if not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous when token metadata is supplied")
        qk_offsets_equal = True if cu_seqlens_q.data_ptr() == cu_seqlens_k.data_ptr() else None
        wide_task_count = None

    dq_full_kv_sequence = dq_full_kv_start = None
    dq_tail_k96 = False

    assert total_q is not None
    assert total_kv is not None
    assert max_seqlen_q is not None
    assert max_seqlen_k is not None
    if total_q <= 0 or total_kv <= 0:
        raise ValueError("packed token counts must be positive")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError("maximum sequence lengths must be positive")

    batch = cu_seqlens_q.numel() - 1
    if legacy_path:
        q_capacity = sum(triton.cdiv(length, _BLOCK_M) for length in q_lengths)
        full_kv_capacity = sum(length // _BLOCK_N for length in k_lengths)
        tail_kv_capacity = sum(length % _BLOCK_N != 0 for length in k_lengths)
        wide_kv_capacity = wide_task_count
    else:
        q_uniform = total_q == batch * max_seqlen_q
        kv_uniform = total_kv == batch * max_seqlen_k
        q_capacity = (batch * triton.cdiv(max_seqlen_q, _BLOCK_M) if q_uniform else triton.cdiv(total_q, _BLOCK_M) +
                      batch)
        full_kv_capacity = 0 if max_seqlen_k < _BLOCK_N else total_kv // _BLOCK_N
        tail_kv_capacity = 0 if kv_uniform and max_seqlen_k % _BLOCK_N == 0 else batch
        wide_kv_capacity = (batch * triton.cdiv(max_seqlen_k, _WIDE_BLOCK_N)
                            if kv_uniform else triton.cdiv(total_kv, _WIDE_BLOCK_N) + batch)
    device = cu_seqlens_q.device
    q_block_sequence = torch.empty(q_capacity, dtype=torch.int32, device=device)
    q_block_start = torch.empty_like(q_block_sequence)
    full_kv_block_sequence = torch.empty(full_kv_capacity, dtype=torch.int32, device=device)
    full_kv_block_start = torch.empty_like(full_kv_block_sequence)
    tail_kv_block_sequence = torch.empty(tail_kv_capacity, dtype=torch.int32, device=device)
    tail_kv_block_start = torch.empty_like(tail_kv_block_sequence)
    wide_kv_start = torch.empty(wide_kv_capacity, dtype=torch.int32, device=device)
    wide_q_start = torch.empty_like(wide_kv_start)
    wide_dq_start = torch.empty_like(wide_kv_start)
    wide_q_len = torch.empty_like(wide_kv_start)
    wide_kv_valid = torch.empty_like(wide_kv_start)
    task_counts = torch.zeros(_NUM_TASK_COUNTS, dtype=torch.int32, device=device)

    _varlen_validate_offsets[(batch, )](
        cu_seqlens_q,
        cu_seqlens_k,
        task_counts,
        total_q,
        total_kv,
        max_seqlen_q,
        max_seqlen_k,
        PLAN_ERROR_INDEX=_PLAN_ERROR,
        CAUSAL_PLAN_ERROR_INDEX=_CAUSAL_PLAN_ERROR,
        num_warps=1,
    )
    _varlen_build_compact_schedules[(batch, )](
        cu_seqlens_q,
        cu_seqlens_k,
        q_block_sequence,
        q_block_start,
        full_kv_block_sequence,
        full_kv_block_start,
        tail_kv_block_sequence,
        tail_kv_block_start,
        wide_kv_start,
        wide_q_start,
        wide_dq_start,
        wide_q_len,
        wide_kv_valid,
        task_counts,
        Q_TASK_COUNT_INDEX=_Q_TASK_COUNT,
        FULL_KV_TASK_COUNT_INDEX=_FULL_KV_TASK_COUNT,
        TAIL_KV_TASK_COUNT_INDEX=_TAIL_KV_TASK_COUNT,
        WIDE_KV_TASK_COUNT_INDEX=_WIDE_KV_TASK_COUNT,
        PLAN_ERROR_INDEX=_PLAN_ERROR,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        WIDE_BLOCK_N=_WIDE_BLOCK_N,
        num_warps=1,
    )
    return VarlenBackwardPlan(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        q_block_sequence=q_block_sequence,
        q_block_start=q_block_start,
        full_kv_block_sequence=full_kv_block_sequence,
        full_kv_block_start=full_kv_block_start,
        tail_kv_block_sequence=tail_kv_block_sequence,
        tail_kv_block_start=tail_kv_block_start,
        wide_kv_start=wide_kv_start,
        wide_q_start=wide_q_start,
        wide_dq_start=wide_dq_start,
        wide_q_len=wide_q_len,
        wide_kv_valid=wide_kv_valid,
        task_counts=task_counts,
        batch=batch,
        total_q=total_q,
        total_kv=total_kv,
        max_q=max_seqlen_q,
        max_kv=max_seqlen_k,
        qk_offsets_equal=qk_offsets_equal,
        dq_full_kv_sequence=dq_full_kv_sequence,
        dq_full_kv_start=dq_full_kv_start,
        dq_tail_k96=dq_tail_k96,
        wide_task_count=wide_task_count,
    )


def validate_varlen_backward_plan(plan: VarlenBackwardPlan, *, causal: bool = False) -> None:
    """Synchronize once and raise if device-side offset validation failed."""
    error_index = _CAUSAL_PLAN_ERROR if causal else _PLAN_ERROR
    if plan.task_counts[error_index.value].item() != 0:
        requirement = " and match between Q and KV" if causal else ""
        raise ValueError("cu_seqlens must start at zero, be strictly increasing, end at "
                         "the packed token total, respect the maximum sequence length"
                         f"{requirement}")


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
    if kv_splits == 1 or k.numel() * kv_splits > _I32_BUFFER_FP32_ELEMENTS:
        return None, None
    total_kv, kv_heads, head_dim = k.shape
    dk_part = torch.empty((total_kv, kv_heads, kv_splits, head_dim), dtype=torch.float32, device=k.device)
    return dk_part, torch.empty_like(dk_part)


@triton.jit
def _varlen_bwd_preprocess(
    O,
    DO,
    Delta,
    CuQ,
    DQ_ACC,
    TOTAL_Q_PADDED,
    TaskCounts,
    PLAN_ERROR_INDEX: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ZERO_DQ: tl.constexpr,
    DQ_PAD_ROWS: tl.constexpr = 16,
    LSE=None,
    TOTAL_Q=None,
    PACK_STATS: tl.constexpr = False,
):
    if tl.load(TaskCounts + PLAN_ERROR_INDEX) != 0:
        return
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
    delta_values = tl.sum(o * do, axis=1)
    if PACK_STATS:
        tl.static_assert(DQ_PAD_ROWS == 32)
        tl.static_assert(BLOCK_M % 32 == 0)
        # Each BM32 tile stores [FP32 log2 LSE rows0..31 | Delta rows0..31].
        # The sequence/head bases need not be aligned to a 32-row boundary.
        stats_sequence_start = q_start + batch * (DQ_PAD_ROWS - 1)
        stats_base = 2 * (head.to(tl.int64) * TOTAL_Q_PADDED + stats_sequence_start)
        stats_offsets = stats_base + (offs_m // 32) * 64 + offs_m % 32
        raw_lse = tl.load(LSE + head.to(tl.int64) * TOTAL_Q + token, mask=offs_m < q_len, other=0.0)
        # Preserve the core's separately rounded scalar FP32 multiply.
        lse_log2 = tl.inline_asm_elementwise(
            "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
            "=v,v",
            [raw_lse],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(Delta + stats_offsets, lse_log2, mask=offs_m < q_len)
        tl.store(Delta + stats_offsets + 32, delta_values, mask=offs_m < q_len)
    else:
        tl.store(Delta + token * HEADS + head, delta_values, mask=offs_m < q_len)
    if ZERO_DQ:
        tl.static_assert(DQ_PAD_ROWS == 16 or DQ_PAD_ROWS == 32)
        tl.static_assert(BLOCK_M % DQ_PAD_ROWS == 0)
        # Wide dQ stores use both native BM16 halves, including padded rows.
        padded_q_len = tl.cdiv(q_len, DQ_PAD_ROWS) * DQ_PAD_ROWS
        scratch_start = q_start + batch.to(tl.int64) * (DQ_PAD_ROWS - 1)
        scratch_base = (head.to(tl.int64) * TOTAL_Q_PADDED + scratch_start) * D
        scratch_offsets = offs_m[:, None] * D + offs_d[None, :]
        tl.store(DQ_ACC + scratch_base + scratch_offsets, 0.0, mask=offs_m[:, None] < padded_q_len)


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
    USE_BUFFER_LOADS: tl.constexpr,
    QDO_BANKPERM: tl.constexpr = False,
):
    rows = step * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    if USE_BUFFER_LOADS:
        # Each thread copies eight adjacent BF16 values directly into LDS.
        # Keep the sequence base scalar and the tile offsets in 32 bits.
        if QDO_BANKPERM:
            # Match the full-tile shared D6/R0 permutation. Low register
            # bits still copy eight adjacent BF16 values per thread.
            load_layout: tl.constexpr = tlx.layout(
                shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
                stride=((8, 16, 32, 128, 1024, 64, 256, 512), (1, 2, 4)),
            )
        else:
            load_layout: tl.constexpr = tlx.layout(
                shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
                stride=((8, 16, 32, 64, 1024, 128, 256, 512), (1, 2, 4)),
            )
        base = (q_start * HQ + q_head) * D
        offsets = (rows[:, None] * HQ * D + dims[None, :]).to(tl.int32)
        valid = tl.broadcast_to(rows[:, None] < q_len, (BLOCK_M, D))
        offsets = tlx.require_layout(offsets, load_layout, pin=False)
        valid = tlx.require_layout(valid, load_layout, pin=False)
        zero = tlx.zeros((BLOCK_M, D), tl.bfloat16, layout=load_layout)
        q_token = tlx.buffer_load_to_local(q_dst, Q + base, offsets, mask=valid, other=zero)
        do_token = tlx.buffer_load_to_local(do_dst, DO + base, offsets, mask=valid, other=zero)
    else:
        global_rows = q_start + rows
        offsets = (global_rows[:, None] * HQ + q_head) * D + dims[None, :]
        valid = rows[:, None] < q_len
        q_token = tlx.async_load(Q + offsets, q_dst, mask=valid, other=0.0)
        do_token = tlx.async_load(DO + offsets, do_dst, mask=valid, other=0.0)
    tlx.async_load_commit_group([q_token, do_token])


@triton.jit
def _compute_dq_tail_k96(ds_shared, k_shared, MMA_MD: tl.constexpr, K_TOKEN=None):
    # Rows 96..127 are padding in the guarded tail. Retain the original
    # dependent K0 -> K32 -> K64 MFMA accumulation using power-of-two slices.
    ds_layout: tl.constexpr = tlx.dot_operand_layout(0, MMA_MD, k_width=8)
    k_layout: tl.constexpr = tlx.dot_operand_layout(1, MMA_MD, k_width=8)
    ds64 = tlx.local_load(tlx.local_slice(ds_shared, (0, 0), (16, 64)), layout=ds_layout)
    k64 = tlx.local_load(tlx.local_slice(k_shared, (0, 0), (64, 128)), token=K_TOKEN, layout=k_layout)
    acc = tlx.zeros((16, 128), tl.float32, layout=MMA_MD)
    acc = tl.dot(ds64, k64, acc=acc, out_dtype=acc.dtype)
    ds32 = tlx.local_load(tlx.local_slice(ds_shared, (0, 64), (16, 32)), layout=ds_layout)
    k32 = tlx.local_load(tlx.local_slice(k_shared, (64, 0), (32, 128)), token=K_TOKEN, layout=k_layout)
    return tl.dot(ds32, k32, acc=acc, out_dtype=acc.dtype)


@triton.jit
def _store_dkdv_tail_n96(
    dk,
    dv,
    DK,
    DV,
    kv_start,
    n0,
    kv_len,
    kv_head,
    SM_SCALE: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    CHUNK_N: tl.constexpr,
    ROW_BASE: tl.constexpr,
    MMA_ND: tl.constexpr,
):
    tl.static_assert(D == 128)
    tl.static_assert((CHUNK_N == 64 and ROW_BASE == 0) or (CHUNK_N == 32 and ROW_BASE == 64))
    # Preserve each output's FP32 query-step chain and its final rounding.
    dk_scale = tlx.require_layout(tl.full((CHUNK_N, D), SM_SCALE, dtype=tl.float32), MMA_ND, pin=False)
    dk *= dk_scale
    dk = dk.to(tl.bfloat16)
    dv = dv.to(tl.bfloat16)

    # Keep each wave's native 32-column D stripe. Eight adjacent BF16 values
    # per thread permit coalesced stores without a cross-wave exchange.
    store_layout: tl.constexpr = tlx.layout(
        shape=((4, 16, 4), (8, CHUNK_N // 16)),
        stride=((8, 128, 32), (1, 2048)),
    )
    dk = tlx.require_layout(dk, store_layout, pin=False)
    dv = tlx.require_layout(dv, store_layout, pin=False)
    offs_n = n0 + ROW_BASE + tl.arange(0, CHUNK_N)
    offs_d = tl.arange(0, D)
    global_n = kv_start + offs_n
    output_offsets = (global_n[:, None] * HKV + kv_head) * D + offs_d[None, :]
    output_offsets = tlx.require_layout(output_offsets.to(tl.int32), store_layout, pin=False)
    output_mask = tlx.require_layout(tl.broadcast_to(offs_n[:, None] < kv_len, (CHUNK_N, D)), store_layout, pin=False)
    tlx.buffer_store(dk, DK, output_offsets, mask=output_mask)
    tlx.buffer_store(dv, DV, output_offsets, mask=output_mask)


@triton.jit
def _load_dq_tail_previous(
    DQ_ACC,
    dq_base,
    q_len,
    step,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MMA_MD: tl.constexpr,
):
    # This is the finalizer's existing native BF16 read, moved earlier only
    # for the guarded steady phase. Its value stays packed until final add.
    local_m = tlx.rematerialized_range(0, BLOCK_M, 40, placement=step)
    offs_d = tlx.rematerialized_range(0, D, 41, placement=step)
    valid = tl.broadcast_to((step * BLOCK_M + local_m < q_len)[:, None], (BLOCK_M, D))
    valid = tlx.require_layout(valid, MMA_MD, pin=True)
    d_swizzled = ((offs_d & 1)
                  | ((offs_d & 2) << 6)
                  | ((offs_d & 12) << 3)
                  | ((offs_d & 48) << 5)
                  | ((offs_d & 64) << 2))
    offsets = dq_base + step * BLOCK_M * D + ((local_m[:, None] << 1) | d_swizzled[None, :])
    offsets = tl.max_contiguous(offsets.to(tl.int32), [1, 2])
    offsets = tlx.require_layout(offsets, MMA_MD, pin=True)
    zero = tlx.zeros((BLOCK_M, D), tl.bfloat16, layout=MMA_MD)
    previous = tlx.buffer_load(DQ_ACC, offsets, mask=valid, other=zero, contiguity=2)
    previous = tlx.require_layout(previous, MMA_MD, pin=True)
    return previous


@triton.jit
def _store_dq_tail_final_preloaded(
    dq,
    previous,
    DQ_OUTPUT,
    q_start,
    q_head,
    q_len,
    step,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MMA_MD: tl.constexpr,
):
    # Keep the parent's two BF16 roundings and final packed-output layout.
    # previous is a required BF16 tensor, never a None-valued loop input.
    dq = tlx.require_layout(dq, MMA_MD, pin=False)
    scale = tlx.require_layout(tl.full((BLOCK_M, D), SM_SCALE, dtype=tl.float32), MMA_MD, pin=False)
    partial = (dq * scale).to(tl.bfloat16)
    partial = tlx.require_layout(partial, MMA_MD, pin=True)
    combined = (previous.to(tl.float32) + partial.to(tl.float32)).to(tl.bfloat16)
    combined = tlx.require_layout(combined, MMA_MD, pin=True)

    store_layout: tl.constexpr = tlx.layout(shape=((2, 16, 2, 4), (8, )), stride=((8, 128, 64, 16), (1, )))
    combined = tlx.require_layout(tlx.release_layout(combined), store_layout, pin=False)
    store_m = tlx.rematerialized_range(0, BLOCK_M, 52, placement=step)
    store_d = tlx.rematerialized_range(0, D, 53, placement=step)
    rows = step * BLOCK_M + store_m
    output_base = (q_start * HQ + q_head.to(tl.int64)) * D
    output_offsets = (rows[:, None] * HQ * D + store_d[None, :]).to(tl.int32)
    output_offsets = tlx.require_layout(output_offsets, store_layout, pin=False)
    output_mask = tlx.require_layout(tl.broadcast_to((rows < q_len)[:, None], (BLOCK_M, D)), store_layout, pin=False)
    tlx.buffer_store(combined, tl.multiple_of(DQ_OUTPUT + output_base, 16), output_offsets, mask=output_mask)


@triton.jit
def _store_dq_tail_final(
    dq,
    DQ_ACC,
    DQ_OUTPUT,
    dq_base,
    q_start,
    q_head,
    q_len,
    step,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MMA_MD: tl.constexpr,
    DEFER_SCRATCH: tl.constexpr = True,
):
    # Earlier full-KV launches have finished. This MHA tail CTA is the only
    # remaining owner of this sequence/head, so it can write final dQ.
    dq = tlx.require_layout(dq, MMA_MD, pin=False)
    scale = tlx.require_layout(tl.full((BLOCK_M, D), SM_SCALE, dtype=tl.float32), MMA_MD, pin=False)
    # Match the atomic path's first rounding before adding its BF16 partial.
    partial = (dq * scale).to(tl.bfloat16)
    partial = tlx.require_layout(partial, MMA_MD, pin=True)
    # Retire dQ MFMA operands before loading the prior scratch partial.
    # Allow VALU/SALU and LDS scheduling across this compiler-only boundary.
    if DEFER_SCRATCH:
        tlx.amd_sched_barrier(0xFC6)
    local_m = tlx.rematerialized_range(0, BLOCK_M, 40, placement=step)
    offs_d = tlx.rematerialized_range(0, D, 41, placement=step)
    valid = tl.broadcast_to((step * BLOCK_M + local_m < q_len)[:, None], (BLOCK_M, D))
    valid = tlx.require_layout(valid, MMA_MD, pin=True)
    d_swizzled = ((offs_d & 1)
                  | ((offs_d & 2) << 6)
                  | ((offs_d & 12) << 3)
                  | ((offs_d & 48) << 5)
                  | ((offs_d & 64) << 2))
    offsets = dq_base + step * BLOCK_M * D + ((local_m[:, None] << 1) | d_swizzled[None, :])
    offsets = tl.max_contiguous(offsets.to(tl.int32), [1, 2])
    offsets = tlx.require_layout(offsets, MMA_MD, pin=True)
    zero = tlx.zeros((BLOCK_M, D), tl.bfloat16, layout=MMA_MD)
    # Keep adjacent native BF16 pairs together through the load and add.
    previous = tlx.buffer_load(DQ_ACC, offsets, mask=valid, other=zero, contiguity=2)
    previous = tlx.require_layout(previous, MMA_MD, pin=True)
    combined = (previous.to(tl.float32) + partial.to(tl.float32)).to(tl.bfloat16)
    combined = tlx.require_layout(combined, MMA_MD, pin=True)

    # Keep each wave's two native 16-column D stripes during final output.
    store_layout: tl.constexpr = tlx.layout(shape=((2, 16, 2, 4), (8, )), stride=((8, 128, 64, 16), (1, )))
    combined = tlx.require_layout(tlx.release_layout(combined), store_layout, pin=False)
    store_m = tlx.rematerialized_range(0, BLOCK_M, 52, placement=step)
    store_d = tlx.rematerialized_range(0, D, 53, placement=step)
    rows = step * BLOCK_M + store_m
    output_base = (q_start * HQ + q_head.to(tl.int64)) * D
    output_offsets = (rows[:, None] * HQ * D + store_d[None, :]).to(tl.int32)
    output_offsets = tlx.require_layout(output_offsets, store_layout, pin=False)
    output_mask = tlx.require_layout(tl.broadcast_to((rows < q_len)[:, None], (BLOCK_M, D)), store_layout, pin=False)
    tlx.buffer_store(combined, tl.multiple_of(DQ_OUTPUT + output_base, 16), output_offsets, mask=output_mask)


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
    REMATERIALIZE_COORDS: tl.constexpr = True,
    MASK_ROWS: tl.constexpr = True,
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
    if REMATERIALIZE_COORDS:
        local_m = tlx.rematerialized_range(0, BLOCK_M, dq_row_remat_group, placement=step)
        offs_d = tlx.rematerialized_range(0, D, dq_column_remat_group, placement=step)
    else:
        local_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
    if MASK_ROWS:
        valid = tl.broadcast_to((step * BLOCK_M + local_m < q_len)[:, None], (BLOCK_M, D))
        valid = tlx.require_layout(valid, MMA_MD, pin=False)
    else:
        valid = None
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
def _bm32_cat_cols(left, right, LAYOUT: tl.constexpr):
    left = tlx.require_layout(left, LAYOUT, pin=False)
    right = tlx.require_layout(right, LAYOUT, pin=False)
    joined = tl.permute(tl.join(left, right), (0, 2, 1))
    result = tl.reshape(joined, (left.shape[0], 2 * left.shape[1]), can_reorder=False)
    return tlx.require_layout(result, LAYOUT, pin=False)


@triton.jit
def _bm32_cat_rows(low, high, LAYOUT: tl.constexpr):
    low = tlx.require_layout(low, LAYOUT, pin=False)
    high = tlx.require_layout(high, LAYOUT, pin=False)
    joined = tl.permute(tl.join(low, high), (2, 0, 1))
    result = tl.reshape(joined, (2 * low.shape[0], low.shape[1]), can_reorder=False)
    return tlx.require_layout(result, LAYOUT, pin=False)


@triton.jit
def _bm32_cat_stats(low, high, STATS_LAYOUT: tl.constexpr):
    joined = tl.permute(tl.join(low, high), (1, 0))
    values = tl.reshape(joined, (32, ), can_reorder=False)
    return tlx.require_layout(values, STATS_LAYOUT, pin=False)


@triton.jit
def _bm32_qdo_stage_slice(buffers, slot, QDO_SLICE_LAYOUT: tl.constexpr):
    stage = tlx.local_view(buffers, slot)
    tiles = tlx.local_reinterpret(stage, tl.bfloat16, [1, 32, 128], layout=QDO_SLICE_LAYOUT)
    return tlx.local_view(tiles, 0)


@triton.jit
def _bm32_load_stat_half(
    tile,
    HALF: tl.constexpr,
    STATS_LAYOUT: tl.constexpr,
    BASE_OFFSET: tl.constexpr = 0,
):
    # Packed Delta uses the second 32-word field of the same 64-word stage.
    return tlx.local_load(
        tlx.local_slice(tile, [BASE_OFFSET + 16 * HALF], [16]),
        layout=STATS_LAYOUT,
        relaxed=True,
    )


@triton.jit
def _bm32_load_score_prefix(tile, HALF: tl.constexpr, QT_LAYOUT: tl.constexpr):
    prefix = tlx.local_slice(tile, [16 * HALF, 0], [16, 32])
    value = tlx.local_load(tlx.local_trans(prefix), layout=QT_LAYOUT, relaxed=True)
    return tlx.require_layout(value, QT_LAYOUT, pin=True)


@triton.jit
def _bm32_load_score_band(tile, BAND: tl.constexpr, QT_LAYOUT: tl.constexpr):
    band = tlx.local_slice(tile, [0, 32 * BAND], [32, 32])
    value = tlx.local_load(tlx.local_trans(band), layout=QT_LAYOUT, relaxed=True)
    return tlx.require_layout(value, QT_LAYOUT, pin=True)


@triton.jit
def _bm32_score_with_prefix(tile, prefix_lo, prefix_hi, QT_LAYOUT: tl.constexpr):
    # The carried first D32 band replaces its LDS reads; load only bands 1..3.
    band0 = _bm32_cat_cols(prefix_lo, prefix_hi, QT_LAYOUT)
    band1 = _bm32_load_score_band(tile, 1, QT_LAYOUT)
    band2 = _bm32_load_score_band(tile, 2, QT_LAYOUT)
    band3 = _bm32_load_score_band(tile, 3, QT_LAYOUT)
    operand = _bm32_cat_rows(_bm32_cat_rows(band0, band1, QT_LAYOUT), _bm32_cat_rows(band2, band3, QT_LAYOUT),
                             QT_LAYOUT)
    return tlx.require_layout(operand, QT_LAYOUT, pin=False)


@triton.jit
def _bm32_cat_k_cols_pinned(left, right, K_NM_LAYOUT: tl.constexpr):
    # Keep each full-N dot-operand fragment in its original lane/wave ownership.
    left = tlx.require_layout(left, K_NM_LAYOUT, pin=True)
    right = tlx.require_layout(right, K_NM_LAYOUT, pin=True)
    joined = tl.permute(tl.join(left, right), (0, 2, 1))
    result = tl.reshape(joined, (left.shape[0], 2 * left.shape[1]), can_reorder=False)
    return tlx.require_layout(result, K_NM_LAYOUT, pin=True)


@triton.jit
def _varlen_gqa_front_bm32(
    dv,
    q_slice,
    do_slice,
    k_buffer,
    k_prefix32,
    v_operand,
    ds_stage,
    lse_values,
    delta_values,
    q0,
    q1,
    do0,
    do1,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MMA_NM: tl.constexpr,
    MMA_ND: tl.constexpr,
    K_NM_LAYOUT: tl.constexpr,
    QT_LAYOUT: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
):
    """Noncausal GQA front with the current Q/dO D32 prefixes already loaded."""
    dv = tlx.require_layout(dv, MMA_ND, pin=False)
    v_operand = tlx.require_layout(v_operand, K_NM_LAYOUT, pin=False)
    # The immutable first D32 strip was loaded once after K publication.
    k_band1 = tlx.local_load(
        tlx.local_slice(k_buffer, [0, 32], [BLOCK_N, 32]),
        layout=K_NM_LAYOUT,
        relaxed=True,
    )
    k_band1 = tlx.require_layout(k_band1, K_NM_LAYOUT, pin=True)
    k_band23 = tlx.local_load(
        tlx.local_slice(k_buffer, [0, 64], [BLOCK_N, 64]),
        layout=K_NM_LAYOUT,
        relaxed=True,
    )
    k_band23 = tlx.require_layout(k_band23, K_NM_LAYOUT, pin=True)
    k_half = _bm32_cat_k_cols_pinned(k_prefix32, k_band1, K_NM_LAYOUT)
    k_nm = _bm32_cat_k_cols_pinned(k_half, k_band23, K_NM_LAYOUT)
    q_t = _bm32_score_with_prefix(q_slice, q0, q1, QT_LAYOUT)
    scores = tl.dot(
        k_nm,
        q_t,
        tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=MMA_NM),
    )
    scores = tlx.amd_register_resident(scores, register_class="vgpr", registers_per_group=16)
    do_t = _bm32_score_with_prefix(do_slice, do0, do1, QT_LAYOUT)
    dp = tl.dot(
        v_operand,
        do_t,
        tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=MMA_NM),
    )
    dp = tlx.amd_register_resident(dp, register_class="vgpr", registers_per_group=16)

    q_out = tlx.local_load(q_slice, layout=Q_OUT_LAYOUT, relaxed=True)
    q_out = tlx.amd_register_resident(q_out, register_class="vgpr", registers_per_group=4)
    do_out = tlx.local_load(do_slice, layout=Q_OUT_LAYOUT, relaxed=True)
    # The prologue or preceding dQ phase already scaled LSE by log2(e).
    # This front consumes that carried value without another multiply.
    lse_log2 = lse_values
    lse_full = tlx.require_layout(
        tl.broadcast_to(lse_log2[None, :], (BLOCK_N, BLOCK_M)),
        MMA_NM,
        pin=False,
    )
    scale_full = tlx.require_layout(
        tl.full((BLOCK_N, BLOCK_M), SM_SCALE * 1.4426950408889634, tl.float32),
        MMA_NM,
        pin=False,
    )
    scaled_scores = scores * scale_full - lse_full
    p = tl.math.exp2(scaled_scores)
    delta_full = tlx.require_layout(
        tl.broadcast_to(delta_values[None, :], (BLOCK_N, BLOCK_M)),
        MMA_NM,
        pin=False,
    )
    ds = p * (dp - delta_full)

    p_nd = tl.reshape(p.to(tl.bfloat16), (2, 2, 2, 2, 16, BLOCK_M))
    p_nd = tl.permute(p_nd, (0, 2, 3, 1, 4, 5))
    p_nd = tl.reshape(p_nd, (BLOCK_N, BLOCK_M))
    p_nd = tlx.require_layout(p_nd, P_ND_LAYOUT, pin=False)
    ds_bf16 = ds.to(tl.bfloat16)
    tlx.local_store(ds_stage, tl.trans(ds_bf16))
    ds_nd = tl.reshape(ds_bf16, (2, 2, 2, 2, 16, BLOCK_M))
    ds_nd = tl.permute(ds_nd, (0, 2, 3, 1, 4, 5))
    ds_nd = tl.reshape(ds_nd, (BLOCK_N, BLOCK_M))
    ds_nd = tlx.require_layout(ds_nd, P_ND_LAYOUT, pin=False)
    return dv, ds_nd, q_out, p_nd, do_out


@triton.jit
def _issue_qdo_bm32_async(
    q_dst,
    do_dst,
    lse_dst,
    delta_dst,
    Q,
    DO,
    LSE,
    Delta,
    q_start,
    q_len,
    q_head,
    outer_block,
    TOTAL_Q,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ASYNC_LAYOUT: tl.constexpr,
    STATS_ASYNC_LAYOUT: tl.constexpr,
    ZERO_FILL_QDO: tl.constexpr = False,
    base_q_head=0,
    REUSE_HEAD_BASE: tl.constexpr = False,
    PACK_STATS: tl.constexpr = False,
    packed_q_start=None,
    packed_total_q=None,
):
    outer_slice = tl.arange(0, 1)
    rows = outer_block * BLOCK_M + outer_slice[:, None] * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
    dims = tl.arange(0, D)
    valid = tl.broadcast_to((rows < q_len)[:, :, None], (1, BLOCK_M, D))
    valid = tlx.require_layout(valid, ASYNC_LAYOUT, pin=False)
    if REUSE_HEAD_BASE:
        head_delta = (q_head - base_q_head).to(tl.int32)
        qdo_base = (q_start * HQ + base_q_head.to(tl.int64)) * D
        offsets = (rows[:, :, None] * HQ * D + dims[None, None, :] + head_delta * D).to(tl.int32)
    else:
        qdo_base = (q_start * HQ + q_head.to(tl.int64)) * D
        offsets = (rows[:, :, None] * HQ * D + dims[None, None, :]).to(tl.int32)
    offsets = tlx.require_layout(offsets, ASYNC_LAYOUT, pin=False)
    if (outer_block + 1) * BLOCK_M > q_len:
        # Keep the explicit tail clear and barrier even with ZERO_FILL_QDO=True.
        # This boundary preserves the validated schedule for reusing each stage.
        tlx.local_store(q_dst, tl.zeros((1, BLOCK_M, D), tl.bfloat16))
        tlx.local_store(do_dst, tl.zeros((1, BLOCK_M, D), tl.bfloat16))
        tl.debug_barrier()
    qdo_zero = tlx.zeros((1, BLOCK_M, D), tl.bfloat16, layout=ASYNC_LAYOUT) if ZERO_FILL_QDO else None
    q_token = tlx.buffer_load_to_local(
        q_dst,
        tl.multiple_of(Q + qdo_base, 16),
        offsets,
        mask=valid,
        other=qdo_zero,
    )
    tlx.async_load_commit_group([q_token])
    # A packed stage has one full wave: log2 LSE then Delta, 32 words each.
    # The fallback retains two separate stages with their original masks.
    stats_i = tl.arange(0, 64)
    stats_zero = tlx.zeros((64, ), tl.float32, layout=STATS_ASYNC_LAYOUT)
    if PACK_STATS:
        tl.static_assert(BLOCK_M == 32)
        stats_rows = outer_block * BLOCK_M + stats_i % BLOCK_M
        stats_valid = tlx.require_layout(stats_rows < q_len, STATS_ASYNC_LAYOUT, pin=False)
        if REUSE_HEAD_BASE:
            stats_base = 2 * (base_q_head.to(tl.int64) * packed_total_q + packed_q_start)
            stats_offsets = (outer_block * 64 + stats_i + 2 * head_delta * packed_total_q).to(tl.int32)
        else:
            stats_base = 2 * (q_head.to(tl.int64) * packed_total_q + packed_q_start)
            stats_offsets = (outer_block * 64 + stats_i).to(tl.int32)
        stats_offsets = tlx.require_layout(stats_offsets, STATS_ASYNC_LAYOUT, pin=False)
        # lse_dst is the whole 64-word allocation, not its 32-word consumer view.
        stats_token = tlx.buffer_load_to_local(lse_dst, Delta + stats_base, stats_offsets, mask=stats_valid,
                                               other=stats_zero)
    else:
        stats_rows = outer_block * BLOCK_M + stats_i
        stats_valid = tlx.require_layout((stats_i < BLOCK_M) & (stats_rows < q_len), STATS_ASYNC_LAYOUT, pin=False)
        if REUSE_HEAD_BASE:
            lse_offsets = tlx.require_layout((stats_rows + head_delta * TOTAL_Q).to(tl.int32), STATS_ASYNC_LAYOUT,
                                             pin=False)
            delta_offsets = tlx.require_layout((stats_rows * HQ + head_delta).to(tl.int32), STATS_ASYNC_LAYOUT,
                                               pin=False)
        else:
            lse_offsets = tlx.require_layout(stats_rows.to(tl.int32), STATS_ASYNC_LAYOUT, pin=False)
            delta_offsets = tlx.require_layout((stats_rows * HQ).to(tl.int32), STATS_ASYNC_LAYOUT, pin=False)
        if REUSE_HEAD_BASE:
            lse_token = tlx.buffer_load_to_local(lse_dst, LSE + base_q_head * TOTAL_Q + q_start, lse_offsets,
                                                 mask=stats_valid, other=stats_zero)
            delta_token = tlx.buffer_load_to_local(delta_dst, Delta + q_start * HQ + base_q_head, delta_offsets,
                                                   mask=stats_valid, other=stats_zero)
        else:
            lse_token = tlx.buffer_load_to_local(lse_dst, LSE + q_head * TOTAL_Q + q_start, lse_offsets,
                                                 mask=stats_valid, other=stats_zero)
            delta_token = tlx.buffer_load_to_local(delta_dst, Delta + q_start * HQ + q_head, delta_offsets,
                                                   mask=stats_valid, other=stats_zero)
    do_token = tlx.buffer_load_to_local(
        do_dst,
        tl.multiple_of(DO + qdo_base, 16),
        offsets,
        mask=valid,
        other=qdo_zero,
    )
    if PACK_STATS:
        tlx.async_load_commit_group([stats_token, do_token])
    else:
        tlx.async_load_commit_group([lse_token, delta_token, do_token])


@triton.jit
def _varlen_gqa_phase_bm32(
    q_tiles,
    do_tiles,
    ds_buffer,
    k_buffer,
    k_prefix32,
    v_operand,
    dk,
    dv,
    outer_block,
    lse0,
    lse1,
    delta0,
    delta1,
    q0,
    q1,
    do0,
    do1,
    q_len,
    SM_SCALE: tl.constexpr,
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
    stats_layout: tl.constexpr = tlx.slice_layout(MMA_NM, 0)
    lse_values = _bm32_cat_stats(lse0, lse1, stats_layout)
    delta_values = _bm32_cat_stats(delta0, delta1, stats_layout)
    # The async producer zero-fills masked rows. Restore +inf after the
    # carried log2 scaling; valid rows retain the same one FP32 multiply.
    lse_values = tl.where(valid, lse_values, float("inf"))
    dv, dk_lhs, dk_rhs, dv_lhs, dv_rhs = _varlen_gqa_front_bm32(
        dv,
        q_slice,
        do_slice,
        k_buffer,
        k_prefix32,
        v_operand,
        ds_buffer,
        lse_values,
        delta_values,
        q0,
        q1,
        do0,
        do1,
        SM_SCALE,
        BLOCK_M,
        BLOCK_N,
        MMA_NM,
        MMA_ND,
        K_NM_LAYOUT,
        QT_LAYOUT,
        P_ND_LAYOUT,
        Q_OUT_LAYOUT,
    )
    dv_lhs = tlx.require_layout(dv_lhs, P_ND_LAYOUT, pin=False)
    dv_rhs = tlx.require_layout(dv_rhs, Q_OUT_LAYOUT, pin=False)
    dv = tlx.require_layout(dv, MMA_ND, pin=False)
    dv = tl.dot(dv_lhs, dv_rhs, dv)
    dk_lhs = tlx.require_layout(dk_lhs, P_ND_LAYOUT, pin=False)
    dk_rhs = tlx.require_layout(dk_rhs, Q_OUT_LAYOUT, pin=False)
    dk = tlx.require_layout(dk, MMA_ND, pin=False)
    dk = tl.dot(dk_lhs, dk_rhs, dk)
    tlx.amd_iglp_opt(3)
    # Complete publication of the opposite stage before its late carried loads.
    tlx.async_load_wait_group(0)
    tl.debug_barrier()
    return (
        tlx.require_layout(dk, MMA_ND, pin=False),
        tlx.require_layout(dv, MMA_ND, pin=False),
        tlx.require_layout(v_operand, K_NM_LAYOUT, pin=False),
    )


@triton.jit
def _bm32_load_dq_k(k_buffer, BAND: tl.constexpr, PANEL: tl.constexpr, K_MD_LAYOUT: tl.constexpr):
    return tlx.local_load(
        tlx.local_slice(k_buffer, [32 * BAND, 64 * PANEL], [32, 64]),
        layout=K_MD_LAYOUT,
        relaxed=True,
    )


@triton.jit
def _bm32_load_dq_s(ds_buffer, BAND: tl.constexpr, HALF: tl.constexpr, DS_MD_LAYOUT: tl.constexpr):
    return tlx.local_load(
        tlx.local_slice(ds_buffer, [16 * HALF, 32 * BAND], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )


@triton.jit
def _varlen_gqa_dq_bm32(
    ds_buffer,
    k_buffer,
    k_band0_panel0,
    k_band0_panel1,
    k_band1_panel0,
    k_band1_panel1,
    v_operand,
    next_q_slice,
    next_do_slice,
    next_lse_tile,
    next_delta_tile,
    MMA_MD: tl.constexpr,
    DS_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
    V_LAYOUT: tl.constexpr,
    QT_LAYOUT: tl.constexpr,
    STATS_LAYOUT: tl.constexpr,
    DELTA_OFFSET: tl.constexpr = 0,
    LSE_PRESCALED: tl.constexpr = False,
):
    """Eight ordered K32 updates per dQ fragment, with two bands in flight."""
    v_operand = tlx.require_layout(v_operand, V_LAYOUT, pin=False)
    c00 = tlx.zeros((16, 64), tl.float32, layout=MMA_MD)
    c01 = tlx.zeros((16, 64), tl.float32, layout=MMA_MD)
    c10 = tlx.zeros((16, 64), tl.float32, layout=MMA_MD)
    c11 = tlx.zeros((16, 64), tl.float32, layout=MMA_MD)

    # Bands0/1 K are persistent; four dS loads request eight startup LDS reads.
    tlx.amd_sched_barrier(0)
    k0 = tlx.require_layout(k_band0_panel0, K_MD_LAYOUT, pin=False)
    k1 = tlx.require_layout(k_band0_panel1, K_MD_LAYOUT, pin=False)
    s0 = _bm32_load_dq_s(ds_buffer, 0, 0, DS_MD_LAYOUT)
    s1 = _bm32_load_dq_s(ds_buffer, 0, 1, DS_MD_LAYOUT)
    next_k0 = tlx.require_layout(k_band1_panel0, K_MD_LAYOUT, pin=False)
    next_k1 = tlx.require_layout(k_band1_panel1, K_MD_LAYOUT, pin=False)
    next_s0 = _bm32_load_dq_s(ds_buffer, 1, 0, DS_MD_LAYOUT)
    next_s1 = _bm32_load_dq_s(ds_buffer, 1, 1, DS_MD_LAYOUT)
    tlx.amd_sched_barrier(0)

    # Two startup markers, 32 MFMA markers and 24 future-load markers = 58.
    # Late carries occupy source MFMA ordinals 23..30, before its marker.
    # LSE loads stay at source MFMA ordinals 23/24 (zero-based). Raw fallbacks
    # retain their scalar scales at 28/29; packed values were scaled in PRE.
    for band in tl.static_range(0, 8):
        c00 = tlx.amd_scheduled_mfma(s0, k0, c00, resident_operand=1, accumulator_role="transient",
                                     initialize=band == 0)
        if band == 6:
            next_lse1 = _bm32_load_stat_half(next_lse_tile, 1, STATS_LAYOUT)
        if band == 7:
            next_q1 = _bm32_load_score_prefix(next_q_slice, 1, QT_LAYOUT)
            # Scalar math prevents LLVM from packing an LSE lane with a score
            # fragment and adding a false dependency between independent fragments.
            if not LSE_PRESCALED:
                next_lse0 = tl.inline_asm_elementwise(
                    "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
                    "=v,v",
                    [next_lse0],  # noqa: F821 - Loaded in static band 5 before use in band 7.
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
        tlx.amd_sched_barrier(0)
        if band < 6:
            future_k0 = _bm32_load_dq_k(k_buffer, band + 2, 0, K_MD_LAYOUT)
            tlx.amd_sched_barrier(0)

        c01 = tlx.amd_scheduled_mfma(s0, k1, c01, resident_operand=1, accumulator_role="transient",
                                     initialize=band == 0)
        if band == 6:
            next_delta0 = _bm32_load_stat_half(next_delta_tile, 0, STATS_LAYOUT, BASE_OFFSET=DELTA_OFFSET)
        if band == 7:
            next_do0 = _bm32_load_score_prefix(next_do_slice, 0, QT_LAYOUT)
            if not LSE_PRESCALED:
                next_lse1 = tl.inline_asm_elementwise(
                    "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
                    "=v,v",
                    [next_lse1],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
        tlx.amd_sched_barrier(0)
        if band < 6:
            future_k1 = _bm32_load_dq_k(k_buffer, band + 2, 1, K_MD_LAYOUT)
            tlx.amd_sched_barrier(0)

        c10 = tlx.amd_scheduled_mfma(s1, k0, c10, resident_operand=1, accumulator_role="transient",
                                     initialize=band == 0)
        if band == 6:
            next_delta1 = _bm32_load_stat_half(next_delta_tile, 1, STATS_LAYOUT, BASE_OFFSET=DELTA_OFFSET)
        if band == 7:
            next_do1 = _bm32_load_score_prefix(next_do_slice, 1, QT_LAYOUT)
        tlx.amd_sched_barrier(0)
        if band < 6:
            future_s0 = _bm32_load_dq_s(ds_buffer, band + 2, 0, DS_MD_LAYOUT)
            tlx.amd_sched_barrier(0)

        c11 = tlx.amd_scheduled_mfma(s1, k1, c11, resident_operand=1, accumulator_role="transient",
                                     initialize=band == 0)
        if band == 5:
            next_lse0 = _bm32_load_stat_half(next_lse_tile, 0, STATS_LAYOUT)
        if band == 6:
            next_q0 = _bm32_load_score_prefix(next_q_slice, 0, QT_LAYOUT)
        tlx.amd_sched_barrier(0)
        if band < 6:
            future_s1 = _bm32_load_dq_s(ds_buffer, band + 2, 1, DS_MD_LAYOUT)
            tlx.amd_sched_barrier(0)

        if band < 7:
            k0, k1, s0, s1 = next_k0, next_k1, next_s0, next_s1
        if band < 6:
            next_k0, next_k1, next_s0, next_s1 = future_k0, future_k1, future_s0, future_s1

    dq_lo = tlx.require_layout(_bm32_cat_cols(c00, c01, MMA_MD), MMA_MD, pin=False)
    dq_hi = tlx.require_layout(_bm32_cat_cols(c10, c11, MMA_MD), MMA_MD, pin=False)
    dq_lo, dq_hi, v_operand = tlx.amd_mfma_commit((dq_lo, dq_hi), v_operand)
    return (
        tlx.require_layout(dq_lo, MMA_MD, pin=False),
        tlx.require_layout(dq_hi, MMA_MD, pin=False),
        tlx.require_layout(v_operand, V_LAYOUT, pin=False),
        next_lse0,
        next_lse1,
        next_delta0,
        next_delta1,
        next_q0,
        next_q1,
        next_do0,
        next_do1,
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
    MASK_ROWS: tl.constexpr = True,
):
    _store_dq_native(dq_lo, DQ_ACC, dq_base, q_len, outer_block * 2, SM_SCALE, D, 16, MMA_MD,
                     REMATERIALIZE_COORDS=False, MASK_ROWS=MASK_ROWS)
    _store_dq_native(dq_hi, DQ_ACC, dq_base, q_len, outer_block * 2 + 1, SM_SCALE, D, 16, MMA_MD,
                     REMATERIALIZE_COORDS=False, MASK_ROWS=MASK_ROWS)


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
    TaskCounts,
    SM_SCALE: tl.constexpr,
    TOTAL_Q,
    TOTAL_Q_PADDED,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TASK_COUNT_INDEX: tl.constexpr,
    PLAN_ERROR_INDEX: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    PAD_DQ_TO_BM32: tl.constexpr = False,
    PACK_STATS: tl.constexpr = False,
    K_PREFIX_REGISTER_CLASS: tl.constexpr = None,
):
    """Process masked BN256 KV owners in BM32 phases with two native BM16 dQ chains."""
    # Four BM16-by-D64 accumulator fragments form two native BM16 output tiles.
    tl.static_assert(D == 128)
    tl.static_assert(BLOCK_M == 32)
    tl.static_assert(BLOCK_N == 256)
    tl.static_assert(HQ % HKV == 0)
    tl.static_assert(HQ // HKV > 1)
    tl.static_assert(KV_SPLITS > 1)
    tl.static_assert(KV_SPLITS <= 4)
    tl.static_assert((HQ // HKV) % KV_SPLITS == 0)
    tl.static_assert(not PACK_STATS or PAD_DQ_TO_BM32)
    tl.static_assert(
        K_PREFIX_REGISTER_CLASS is None or K_PREFIX_REGISTER_CLASS == "vgpr" or K_PREFIX_REGISTER_CLASS == "agpr",
        "K_PREFIX_REGISTER_CLASS must be None, 'vgpr', or 'agpr'",
    )

    # Arrange head/split work for eight XCDs in groups of up to 32 schedule
    # slots. Capacity-launched slots beyond the device-built count exit below.
    HS: tl.constexpr = HKV * KV_SPLITS
    w = tl.program_id(0) + HS * tl.program_id(1)
    group = w // (HS * 32)
    first_task = group * 32
    count = tl.minimum(32, tl.num_programs(1) - first_task)
    rank = w % (HS * 32)
    W = count * HS
    xcd = rank % 8
    local = rank // 8
    mapped = xcd * (W // 8) + tl.minimum(xcd, W % 8) + local
    kv_head_split = mapped // count
    task = first_task + mapped % count
    if tl.load(TaskCounts + PLAN_ERROR_INDEX) != 0:
        return
    if task >= tl.load(TaskCounts + TASK_COUNT_INDEX):
        return
    kv_head = kv_head_split // KV_SPLITS
    split = kv_head_split % KV_SPLITS
    group_size: tl.constexpr = HQ // HKV
    heads_per_split: tl.constexpr = group_size // KV_SPLITS
    kv_global_start = tl.load(KVGlobalStart + task).to(tl.int64)
    q_start = tl.load(QStart + task).to(tl.int64)
    q_scratch_start = tl.load(DQScratchStart + task).to(tl.int64)
    if PAD_DQ_TO_BM32:
        # Plans retain the BM16 base CuQ[sequence] + 15*sequence. Derive the
        # sequence once and extend its scratch gap to 31 rows for this launch.
        sequence = (q_scratch_start - q_start).to(tl.int32) // 15
        q_scratch_start += sequence.to(tl.int64) * 16
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
    stats_async_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), ()),
        stride=((1, 0), ()),
    )
    stats_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[1], [2], [4], [8], [16], [32]],
        block_bases=[],
        alignment=16,
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
    lse_buffers = tlx.local_alloc((64, ), tl.float32, 2, layout=stats_smem_layout)
    if PACK_STATS:
        delta_buffers = lse_buffers
    else:
        delta_buffers = tlx.local_alloc((64, ), tl.float32, 2, layout=stats_smem_layout)

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
        tlx.local_view(lse_buffers, 0),
        tlx.local_view(delta_buffers, 0),
        Q,
        DO,
        LSE,
        Delta,
        q_start,
        q_len,
        first_q_head,
        0,
        TOTAL_Q,
        HQ,
        D,
        BLOCK_M,
        qdo_async_layout,
        stats_async_layout,
        base_q_head=first_q_head,
        REUSE_HEAD_BASE=heads_per_split == 2,
        PACK_STATS=PACK_STATS,
        packed_q_start=q_scratch_start,
        packed_total_q=TOTAL_Q_PADDED,
    )
    second_step = tl.minimum(1, total_outer_steps - 1)
    second_group = second_step // outer_blocks
    second_outer = second_step % outer_blocks
    _issue_qdo_bm32_async(
        tlx.local_view(q_buffers, 1),
        tlx.local_view(do_buffers, 1),
        tlx.local_view(lse_buffers, 1),
        tlx.local_view(delta_buffers, 1),
        Q,
        DO,
        LSE,
        Delta,
        q_start,
        q_len,
        first_q_head + second_group,
        second_outer,
        TOTAL_Q,
        HQ,
        D,
        BLOCK_M,
        qdo_async_layout,
        stats_async_layout,
        base_q_head=first_q_head,
        REUSE_HEAD_BASE=heads_per_split == 2,
        PACK_STATS=PACK_STATS,
        packed_q_start=q_scratch_start,
        packed_total_q=TOTAL_Q_PADDED,
    )
    tlx.async_load_wait_group(2)
    tl.debug_barrier()

    # K is immutable for this owner. Retain the full-N first D32 strip once.
    k_prefix32 = tlx.local_load(
        tlx.local_slice(k_buffer, [0, 0], [BLOCK_N, 32]),
        layout=k_nm_layout,
        relaxed=True,
    )
    k_prefix32 = tlx.require_layout(k_prefix32, k_nm_layout, pin=True)
    # None leaves register placement to the compiler without removing the layout pin.
    if K_PREFIX_REGISTER_CLASS is not None:
        k_prefix32 = tlx.amd_register_resident(k_prefix32, register_class=K_PREFIX_REGISTER_CLASS,
                                               registers_per_group=4)

    # Reuse immutable band0 K directly in the dQ operand layout on every phase.
    dq_k_band0_panel0 = _bm32_load_dq_k(k_buffer, 0, 0, k_md_layout)
    dq_k_band0_panel0 = tlx.require_layout(dq_k_band0_panel0, k_md_layout, pin=True)
    dq_k_band0_panel0 = tlx.amd_register_resident(dq_k_band0_panel0, register_class="agpr", registers_per_group=4)
    dq_k_band0_panel1 = _bm32_load_dq_k(k_buffer, 0, 1, k_md_layout)
    dq_k_band0_panel1 = tlx.require_layout(dq_k_band0_panel1, k_md_layout, pin=True)
    dq_k_band0_panel1 = tlx.amd_register_resident(dq_k_band0_panel1, register_class="agpr", registers_per_group=4)

    # Extend the immutable dQ cache to band1 with the same prologue layout pins.
    dq_k_band1_panel0 = _bm32_load_dq_k(k_buffer, 1, 0, k_md_layout)
    dq_k_band1_panel0 = tlx.require_layout(dq_k_band1_panel0, k_md_layout, pin=True)
    dq_k_band1_panel0 = tlx.amd_register_resident(dq_k_band1_panel0, register_class="agpr", registers_per_group=4)
    dq_k_band1_panel1 = _bm32_load_dq_k(k_buffer, 1, 1, k_md_layout)
    dq_k_band1_panel1 = tlx.require_layout(dq_k_band1_panel1, k_md_layout, pin=True)
    dq_k_band1_panel1 = tlx.amd_register_resident(dq_k_band1_panel1, register_class="agpr", registers_per_group=4)

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

    # The prologue wait has completed stage 0. Carry its raw statistics and
    # first Q/dO D32 band into the first phase in the existing consumer layouts.
    stats_layout: tl.constexpr = tlx.slice_layout(mma_nm, 0)
    initial_q_slice = _bm32_qdo_stage_slice(q_buffers, 0, qdo_slice_smem_layout)
    initial_do_slice = _bm32_qdo_stage_slice(do_buffers, 0, qdo_slice_smem_layout)
    initial_lse_tile = tlx.local_view(lse_buffers, 0)
    initial_delta_tile = tlx.local_view(delta_buffers, 0)
    lse0 = _bm32_load_stat_half(initial_lse_tile, 0, stats_layout)
    lse1 = _bm32_load_stat_half(initial_lse_tile, 1, stats_layout)
    delta0 = _bm32_load_stat_half(initial_delta_tile, 0, stats_layout, BASE_OFFSET=32 if PACK_STATS else 0)
    delta1 = _bm32_load_stat_half(initial_delta_tile, 1, stats_layout, BASE_OFFSET=32 if PACK_STATS else 0)
    q0 = _bm32_load_score_prefix(initial_q_slice, 0, qt_layout)
    q1 = _bm32_load_score_prefix(initial_q_slice, 1, qt_layout)
    do0 = _bm32_load_score_prefix(initial_do_slice, 0, qt_layout)
    do1 = _bm32_load_score_prefix(initial_do_slice, 1, qt_layout)
    # Packed LSE was scaled in PRE. Raw fallback halves keep their original
    # scalar operation and rounding before the first front.
    if not PACK_STATS:
        lse0 = tl.inline_asm_elementwise(
            "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
            "=v,v",
            [lse0],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        lse1 = tl.inline_asm_elementwise(
            "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
            "=v,v",
            [lse1],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    for outer_step in tl.range(0, total_outer_steps, loop_unroll_factor=1):
        outer_stage = outer_step % 2
        group_index = outer_step // outer_blocks
        outer_block = outer_step % outer_blocks
        q_head = first_q_head + group_index
        q_outer = tlx.local_view(q_buffers, outer_stage)
        do_outer = tlx.local_view(do_buffers, outer_stage)
        lse_outer = tlx.local_view(lse_buffers, outer_stage)
        delta_outer = tlx.local_view(delta_buffers, outer_stage)
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
            k_prefix32,
            v_operand,
            dk,
            dv,
            outer_block,
            lse0,
            lse1,
            delta0,
            delta1,
            q0,
            q1,
            do0,
            do1,
            q_len,
            SM_SCALE,
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
            lse_outer,
            delta_outer,
            Q,
            DO,
            LSE,
            Delta,
            q_start,
            q_len,
            first_q_head + next_group,
            next_outer,
            TOTAL_Q,
            HQ,
            D,
            BLOCK_M,
            qdo_async_layout,
            stats_async_layout,
            ZERO_FILL_QDO=True,
            base_q_head=first_q_head,
            REUSE_HEAD_BASE=heads_per_split == 2,
            PACK_STATS=PACK_STATS,
            packed_q_start=q_scratch_start,
            packed_total_q=TOTAL_Q_PADDED,
        )
        # The just-issued refill targets the current slot (i+2); the next
        # consumer takes i+1 from the opposite slot, including the final drain.
        next_stage = outer_stage ^ 1
        next_q_slice = _bm32_qdo_stage_slice(q_buffers, next_stage, qdo_slice_smem_layout)
        next_do_slice = _bm32_qdo_stage_slice(do_buffers, next_stage, qdo_slice_smem_layout)
        next_lse_tile = tlx.local_view(lse_buffers, next_stage)
        next_delta_tile = tlx.local_view(delta_buffers, next_stage)
        (dq_lo, dq_hi, v_operand, next_lse0, next_lse1, next_delta0, next_delta1, next_q0, next_q1, next_do0,
         next_do1) = _varlen_gqa_dq_bm32(
             tlx.local_view(ds_buffers, 0),
             k_buffer,
             dq_k_band0_panel0,
             dq_k_band0_panel1,
             dq_k_band1_panel0,
             dq_k_band1_panel1,
             v_operand,
             next_q_slice,
             next_do_slice,
             next_lse_tile,
             next_delta_tile,
             mma_md,
             ds_md_layout,
             k_md_layout,
             k_nm_layout,
             qt_layout,
             stats_layout,
             DELTA_OFFSET=32 if PACK_STATS else 0,
             LSE_PRESCALED=PACK_STATS,
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
            MASK_ROWS=not PAD_DQ_TO_BM32,
        )
        tlx.async_load_wait_group(2)
        tl.debug_barrier()
        lse0, lse1 = next_lse0, next_lse1
        delta0, delta1 = next_delta0, next_delta1
        q0, q1 = next_q0, next_q1
        do0, do1 = next_do0, next_do1

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
    tlx.buffer_store(dk, output_ptr, output_offsets, mask=output_mask)
    tlx.buffer_store(dv, output_v_ptr, output_offsets, mask=output_mask)


@triton.jit
def _issue_mha_stats_async(lse_dst, delta_dst, LSE, Delta, q_start, q_len, q_head, TOTAL_Q, HQ: tl.constexpr):
    """Stage one short MHA sequence's immutable statistics before its Q loop."""
    rows = tl.arange(0, 512)
    load_layout: tl.constexpr = tlx.layout(shape=((256, ), (2, )), stride=((1, ), (256, )))
    lse_offsets = tlx.require_layout(rows.to(tl.int32), load_layout, pin=False)
    delta_offsets = tlx.require_layout((rows * HQ).to(tl.int32), load_layout, pin=False)
    valid = tlx.require_layout(rows < q_len, load_layout, pin=False)
    zero = tlx.zeros((512, ), tl.float32, layout=load_layout)
    lse_token = tlx.buffer_load_to_local(lse_dst, LSE + q_head * TOTAL_Q + q_start, lse_offsets, mask=valid, other=zero)
    delta_token = tlx.buffer_load_to_local(delta_dst, Delta + q_start * HQ + q_head, delta_offsets, mask=valid,
                                           other=zero)
    tlx.async_load_commit_group([lse_token, delta_token])


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
    TaskCounts,
    SM_SCALE: tl.constexpr,
    TOTAL_Q,
    TOTAL_Q_PADDED,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TASK_COUNT_INDEX: tl.constexpr,
    PLAN_ERROR_INDEX: tl.constexpr,
    FULL_KV_TILE: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    V_STRIDE_T: tl.constexpr,
    QDO_ALIGNED: tl.constexpr = False,
    CACHE_MHA_STATS: tl.constexpr = False,
    DQ_OUTPUT=None,
    FINALIZE_DQ: tl.constexpr = False,
    DQ_TAIL_K96: tl.constexpr = False,
):
    """Compute current dK/dV while consuming the preceding dS phase for dQ."""
    tl.static_assert(D == 128)
    tl.static_assert(BLOCK_M == 16)
    tl.static_assert(BLOCK_N == 128)
    tl.static_assert(HQ % HKV == 0)
    tl.static_assert(KV_SPLITS > 0)
    tl.static_assert(KV_SPLITS <= 4)
    tl.static_assert((HQ // HKV) % KV_SPLITS == 0)

    # Preserve the established causal LDS layout and copy schedule.
    NONCAUSAL_MHA: tl.constexpr = HQ == HKV and not IS_CAUSAL
    tl.static_assert(not CACHE_MHA_STATS or NONCAUSAL_MHA)
    tl.static_assert(not FINALIZE_DQ
                     or (NONCAUSAL_MHA and KV_SPLITS == 1 and not FULL_KV_TILE and CACHE_MHA_STATS and QDO_ALIGNED))
    tl.static_assert(not DQ_TAIL_K96 or FINALIZE_DQ)
    if NONCAUSAL_MHA:
        task = tl.program_id(1)
        kv_head_split = tl.program_id(0)
    else:
        task = tl.program_id(0)
        kv_head_split = tl.program_id(1)
    if tl.load(TaskCounts + PLAN_ERROR_INDEX) != 0:
        return
    if task >= tl.load(TaskCounts + TASK_COUNT_INDEX):
        return
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

    QDO_BANKPERM: tl.constexpr = NONCAUSAL_MHA and CACHE_MHA_STATS and QDO_ALIGNED and FULL_KV_TILE
    if QDO_BANKPERM:
        qdo_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [(512, 16)],
            [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [8, 0], [0, 64], [2, 0], [4, 0]],
            [BLOCK_M, D],
        )
    else:
        qdo_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [(512, 16 if NONCAUSAL_MHA else 32)],
            [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [8, 0], [1, 0], [2, 0], [4, 0]],
            [BLOCK_M, D],
        )
    k_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 8 if NONCAUSAL_MHA else 32)],
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
        warps_per_cta=[1, 4] if DQ_TAIL_K96 else [2, 2],
    )
    k_nm_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    qt_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_nm, k_width=8)
    p_nd_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_nd, k_width=8)
    q_nd_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_nd, k_width=8)
    ds_md_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_md, k_width=8)
    k_md_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_md, k_width=8)

    if CACHE_MHA_STATS:
        stats_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
            offset_bases=[[1], [2], [4], [8], [16], [32], [64], [128], [256]], block_bases=[], alignment=16)
        lse_buffer = tlx.local_alloc((512, ), tl.float32, 1, layout=stats_layout)
        delta_buffer = tlx.local_alloc((512, ), tl.float32, 1, layout=stats_layout)
    k_buffer = tlx.local_alloc((BLOCK_N, D), tl.bfloat16, 1, layout=k_layout)
    q_buffers = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 2, layout=qdo_layout)
    do_buffers = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 2, layout=qdo_layout)
    if NONCAUSAL_MHA:
        p_buffer = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.bfloat16, 1, layout=ds_layout)
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
        NONCAUSAL_MHA and QDO_ALIGNED,
        QDO_BANKPERM,
    )
    if CACHE_MHA_STATS:
        _issue_mha_stats_async(tlx.local_view(lse_buffer, 0), tlx.local_view(delta_buffer, 0), LSE, Delta, q_start,
                               q_len, kv_head, TOTAL_Q, HQ)
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
    if DQ_TAIL_K96:
        # Partition output rows, retaining the complete Q16 reduction and
        # ordered FP32 query-step updates for every dK/dV element.
        dk64 = tlx.zeros((64, D), tl.float32, layout=mma_nd)
        dv64 = tlx.zeros((64, D), tl.float32, layout=mma_nd)
        dk32 = tlx.zeros((32, D), tl.float32, layout=mma_nd)
        dv32 = tlx.zeros((32, D), tl.float32, layout=mma_nd)
    else:
        dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
        dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    q_scratch_start = q_start + batch.to(tl.int64) * (BLOCK_M - 1)
    log2e: tl.constexpr = 1.4426950408889634

    # Separate startup from the cached path's steady loop, where every
    # iteration computes preceding dQ and issues four atomics. Complete the
    # startup prefetch before loop entry so LLVM can preserve the longer
    # VMEM event distance on the backedge.
    phase_count: tl.constexpr = 2 if CACHE_MHA_STATS else 1
    for phase in tl.static_range(0, phase_count):
        if CACHE_MHA_STATS:
            if phase == 0:
                step_begin = 0
                step_end = 1
            else:
                tlx.async_load_wait_group(0)
                tl.debug_barrier()
                step_begin = 1
                step_end = total_steps
        else:
            step_begin = 0
            step_end = total_steps
        for step in tl.range(step_begin, step_end, num_stages=1):
            current_slot = step % 2
            next_slot = 1 - current_slot
            if heads_per_split == 1:
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
                NONCAUSAL_MHA and QDO_ALIGNED,
                QDO_BANKPERM,
            )
            tlx.async_load_wait_group(1)
            tl.debug_barrier()

            if DQ_TAIL_K96 and phase > 0:
                # Cached MHA has one query head per owner. Define this tensor
                # inside each steady iteration, after the current Q/dO wait.
                preload_dq_base = ((q_head.to(tl.int64) * TOTAL_Q_PADDED + q_scratch_start) * D).to(tl.int32)
                preloaded_previous = _load_dq_tail_previous(DQ_ACC, preload_dq_base, q_len, step - 1, D, BLOCK_M,
                                                            mma_md)
                # Pin VMEM reads before subsequent MFMA work; allow ALU/LDS.
                # Native review must also verify these reads stay below the
                # preceding wait/barrier and do not force an early wait.
                tlx.amd_sched_barrier(0xFC6)

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
            if CACHE_MHA_STATS:
                stats_offset = tl.multiple_of((q_step * BLOCK_M).to(tl.int32), BLOCK_M)
                lse_view = tlx.local_slice(tlx.local_view(lse_buffer, 0), [stats_offset], [BLOCK_M])
                delta_view = tlx.local_slice(tlx.local_view(delta_buffer, 0), [stats_offset], [BLOCK_M])
                lse = tlx.local_load(lse_view, token=initial_wait)
                delta = tlx.local_load(delta_view, token=initial_wait)
            else:
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
            if NONCAUSAL_MHA:
                # Exchange P and dS together. The dS buffer also feeds the next
                # iteration's dQ, so its existing storage serves both layouts.
                p_shared = tlx.local_view(p_buffer, 0)
                tlx.local_store(p_shared, tl.trans(p_t.to(tl.bfloat16)))
                tl.debug_barrier()
                if DQ_TAIL_K96:
                    p_nd64 = tlx.local_load(tlx.local_trans(tlx.local_slice(p_shared, (0, 0), (BLOCK_M, 64))),
                                            layout=p_nd_layout)
                    ds_nd64 = tlx.local_load(tlx.local_trans(tlx.local_slice(current_ds, (0, 0), (BLOCK_M, 64))),
                                             layout=p_nd_layout)
                    dv64 = tl.dot(p_nd64, do_tile, acc=dv64, out_dtype=dv64.dtype)
                    dk64 = tl.dot(ds_nd64, q_tile, acc=dk64, out_dtype=dk64.dtype)
                    p_nd32 = tlx.local_load(tlx.local_trans(tlx.local_slice(p_shared, (0, 64), (BLOCK_M, 32))),
                                            layout=p_nd_layout)
                    ds_nd32 = tlx.local_load(tlx.local_trans(tlx.local_slice(current_ds, (0, 64), (BLOCK_M, 32))),
                                             layout=p_nd_layout)
                    dv32 = tl.dot(p_nd32, do_tile, acc=dv32, out_dtype=dv32.dtype)
                    dk32 = tl.dot(ds_nd32, q_tile, acc=dk32, out_dtype=dk32.dtype)
                else:
                    p_nd = tlx.local_load(tlx.local_trans(p_shared), layout=p_nd_layout)
                    ds_nd = tlx.local_load(tlx.local_trans(current_ds), layout=p_nd_layout)
            else:
                p_nd = tlx.require_layout(p_t.to(tl.bfloat16), p_nd_layout, pin=False)
                ds_nd = tlx.require_layout(ds_bf16, p_nd_layout, pin=False)
            if not DQ_TAIL_K96:
                dv = tl.dot(p_nd, do_tile, acc=dv, out_dtype=dv.dtype)
                dk = tl.dot(ds_nd, q_tile, acc=dk, out_dtype=dk.dtype)

            if not CACHE_MHA_STATS or phase > 0:
                if CACHE_MHA_STATS or step > 0:
                    previous_step = step - 1
                    if heads_per_split == 1:
                        previous_q_step = first_q_block + previous_step
                        previous_q_head = kv_head * group_size + split
                    else:
                        previous_group_index = previous_step // active_q_blocks
                        previous_q_step = first_q_block + previous_step % active_q_blocks
                        previous_q_head = kv_head * group_size + split * heads_per_split + previous_group_index
                    previous_dq_acc_base = ((previous_q_head.to(tl.int64) * TOTAL_Q_PADDED + q_scratch_start) * D).to(
                        tl.int32)
                    if DQ_TAIL_K96:
                        dq_part = _compute_dq_tail_k96(tlx.local_view(ds_buffers, 1 - current_slot),
                                                       tlx.local_view(k_buffer, 0), mma_md, initial_wait)
                    else:
                        previous_ds = tlx.local_load(tlx.local_view(ds_buffers, 1 - current_slot), layout=ds_md_layout)
                        k_for_dq = tlx.local_load(tlx.local_view(k_buffer, 0), token=initial_wait, layout=k_md_layout)
                        dq_acc = tlx.zeros((BLOCK_M, D), tl.float32, layout=mma_md)
                        dq_part = tl.dot(previous_ds, k_for_dq, acc=dq_acc, out_dtype=dq_acc.dtype)
                    if FINALIZE_DQ:
                        if DQ_TAIL_K96 and phase > 0:
                            _store_dq_tail_final_preloaded(
                                dq_part,
                                preloaded_previous,
                                DQ_OUTPUT,
                                q_start,
                                previous_q_head,
                                q_len,
                                previous_q_step,
                                SM_SCALE,
                                HQ,
                                D,
                                BLOCK_M,
                                mma_md,
                            )
                        else:
                            _store_dq_tail_final(
                                dq_part,
                                DQ_ACC,
                                DQ_OUTPUT,
                                previous_dq_acc_base,
                                q_start,
                                previous_q_head,
                                q_len,
                                previous_q_step,
                                SM_SCALE,
                                HQ,
                                D,
                                BLOCK_M,
                                mma_md,
                                DEFER_SCRATCH=not DQ_TAIL_K96,
                            )
                    else:
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
    if heads_per_split == 1:
        last_q_step = first_q_block + last_step
        last_q_head = kv_head * group_size + split
    else:
        last_group_index = last_step // active_q_blocks
        last_q_step = first_q_block + last_step % active_q_blocks
        last_q_head = kv_head * group_size + split * heads_per_split + last_group_index
    last_dq_acc_base = ((last_q_head.to(tl.int64) * TOTAL_Q_PADDED + q_scratch_start) * D).to(tl.int32)
    if DQ_TAIL_K96:
        dq_part = _compute_dq_tail_k96(tlx.local_view(ds_buffers, last_step % 2), tlx.local_view(k_buffer, 0), mma_md)
    else:
        last_ds = tlx.local_load(tlx.local_view(ds_buffers, last_step % 2), layout=ds_md_layout)
        k_for_dq = tlx.local_load(tlx.local_view(k_buffer, 0), layout=k_md_layout)
        dq_acc = tlx.zeros((BLOCK_M, D), tl.float32, layout=mma_md)
        dq_part = tl.dot(last_ds, k_for_dq, acc=dq_acc, out_dtype=dq_acc.dtype)
    if FINALIZE_DQ:
        _store_dq_tail_final(
            dq_part,
            DQ_ACC,
            DQ_OUTPUT,
            last_dq_acc_base,
            q_start,
            last_q_head,
            q_len,
            last_q_step,
            SM_SCALE,
            HQ,
            D,
            BLOCK_M,
            mma_md,
            DEFER_SCRATCH=not DQ_TAIL_K96,
        )
    else:
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

    if DQ_TAIL_K96:
        _store_dkdv_tail_n96(
            dk64,
            dv64,
            DK,
            DV,
            kv_start,
            n0,
            kv_len,
            kv_head,
            SM_SCALE,
            HKV,
            D,
            64,
            0,
            mma_nd,
        )
        _store_dkdv_tail_n96(
            dk32,
            dv32,
            DK,
            DV,
            kv_start,
            n0,
            kv_len,
            kv_head,
            SM_SCALE,
            HKV,
            D,
            32,
            64,
            mma_nd,
        )
    else:
        dk_scale = tlx.require_layout(tl.full((BLOCK_N, D), SM_SCALE, dtype=tl.float32), mma_nd, pin=False)
        dk *= dk_scale
        if NONCAUSAL_MHA:
            # Pack eight BF16 values per lane and place neighboring lanes on
            # neighboring columns for coalesced 128-bit output stores.
            store_layout: tl.constexpr = tlx.layout(shape=((16, 16), (8, 8)), stride=((8, 128), (1, 2048)))
        else:
            store_layout: tl.constexpr = mma_nd
        if KV_SPLITS == 1:
            output_offsets = tlx.require_layout(kv_offsets.to(tl.int32), store_layout, pin=False)
        else:
            partial_offsets = (((global_n[:, None] * HKV + kv_head) * KV_SPLITS + split) * D + offs_d[None, :])
            output_offsets = tlx.require_layout(partial_offsets.to(tl.int32), store_layout, pin=False)
        if KV_SPLITS == 1:
            dk = dk.to(tl.bfloat16)
            dv = dv.to(tl.bfloat16)
        dk = tlx.require_layout(dk, store_layout, pin=False)
        dv = tlx.require_layout(dv, store_layout, pin=False)
        if FULL_KV_TILE:
            tlx.buffer_store(dk, DK, output_offsets)
            tlx.buffer_store(dv, DV, output_offsets)
        else:
            output_mask = tlx.require_layout(tl.broadcast_to(kv_mask, (BLOCK_N, D)), store_layout, pin=False)
            tlx.buffer_store(dk, DK, output_offsets, mask=output_mask)
            tlx.buffer_store(dv, DV, output_offsets, mask=output_mask)


@triton.jit
def _varlen_dkdv_reduce_kernel(
    DK_PART,
    DV_PART,
    DK,
    DV,
    TaskCounts,
    TOTAL_KV,
    PLAN_ERROR_INDEX: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(KV_SPLITS > 1)
    tl.static_assert(KV_SPLITS <= 4)
    if tl.load(TaskCounts + PLAN_ERROR_INDEX) != 0:
        return
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
    # Match the FP32 load layout so the BF16 stores do not introduce a
    # whole-tile cross-warp conversion through LDS.
    output_offsets = tl.max_contiguous(output_offsets, [1, 4])
    tl.store(DK + output_offsets, dk.to(tl.bfloat16), mask=valid)
    tl.store(DV + output_offsets, dv.to(tl.bfloat16), mask=valid)


@triton.jit
def _varlen_dq_convert_kernel(
    DQ_ACC,
    CuQ,
    QBlockSequence,
    QBlockStart,
    TaskCounts,
    DQ,
    TOTAL_Q_PADDED,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TASK_COUNT_INDEX: tl.constexpr,
    PLAN_ERROR_INDEX: tl.constexpr,
):
    task = tl.program_id(0)
    if tl.load(TaskCounts + PLAN_ERROR_INDEX) != 0:
        return
    if task >= tl.load(TaskCounts + TASK_COUNT_INDEX):
        return
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


@gluon.jit
def _varlen_mha_dq_convert_coalesced_kernel(
    DQ_ACC,
    CuQ,
    QBlockSequence,
    QBlockStart,
    TaskCounts,
    DQ,
    TOTAL_Q_PADDED,
    HEADS: ttgl.constexpr,
    D: ttgl.constexpr,
    BLOCK_M: ttgl.constexpr,
    TASK_COUNT_INDEX: ttgl.constexpr,
    PLAN_ERROR_INDEX: ttgl.constexpr,
    DQ_PAD_ROWS: ttgl.constexpr = 16,
):
    ttgl.static_assert(BLOCK_M == 16 and D == 128)
    ttgl.static_assert(DQ_PAD_ROWS == 16 or DQ_PAD_ROWS == 32)
    task = ttgl.program_id(0)
    if ttgl.load(TaskCounts + PLAN_ERROR_INDEX) != 0:
        return
    if task >= ttgl.load(TaskCounts + TASK_COUNT_INDEX):
        return
    head = ttgl.program_id(1)
    batch = ttgl.load(QBlockSequence + task)
    start_m = ttgl.load(QBlockStart + task)
    q_start = ttgl.load(CuQ + batch).to(ttgl.int64)
    q_end = ttgl.load(CuQ + batch + 1).to(ttgl.int64)
    q_len = (q_end - q_start).to(ttgl.int32)
    q_scratch_start = q_start + batch.to(ttgl.int64) * (DQ_PAD_ROWS - 1)
    native_base = (head.to(ttgl.int64) * TOTAL_Q_PADDED + q_scratch_start + start_m) * D

    # Compact tasks read native BM16 tiles from the initialized footprint.
    # With BM32 padding, a wholly padded extra high tile needs no conversion.
    native_layout: ttgl.constexpr = ttgl.BlockedLayout([8], [64], [4], [0])
    native_offsets = ttgl.arange(0, BLOCK_M * D, layout=native_layout)
    native_values = ttgl.load(DQ_ACC + native_base + native_offsets)
    # Native bit order: [d5,d4,d6,d1,d3,d2,m3,m2,m1,m0,d0].
    # Reorder dimensions into logical [m3,m2,m1,m0,d6,d5,d4,d3,d2,d1,d0].
    values = native_values.reshape((4, 2, 2, 4, 16, 2))
    values = values.permute((4, 1, 0, 3, 2, 5)).reshape((BLOCK_M, D))
    output_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    values = ttgl.convert_layout(values, output_layout)
    local_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, output_layout))
    offs_d = ttgl.arange(0, D, layout=ttgl.SliceLayout(0, output_layout))
    global_m = q_start + start_m + local_m
    output_offsets = (global_m[:, None] * HEADS + head) * D + offs_d[None, :]
    valid = (start_m + local_m[:, None] < q_len)
    ttgl.store(DQ + output_offsets, values, mask=valid)


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
    if causal and plan.qk_offsets_equal is False:
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


@triton.jit
def _varlen_bwd_preprocess_dynamic_owner_queue(
    O,
    DO,
    Delta,
    CuQ,
    DQ_ACC,
    TOTAL_Q_PADDED,
    TaskCounts,
    OWNER_NEXT,
    INITIAL_OWNER,
    PLAN_ERROR_INDEX: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ZERO_DQ: tl.constexpr,
    DQ_PAD_ROWS: tl.constexpr = 16,
    LSE=None,
    TOTAL_Q=None,
    PACK_STATS: tl.constexpr = False,
):
    if tl.load(TaskCounts + PLAN_ERROR_INDEX) != 0:
        return
    if (tl.program_id(0) == 0) & (tl.program_id(1) == 0):
        tl.store(OWNER_NEXT, INITIAL_OWNER)
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
    delta_values = tl.sum(o * do, axis=1)
    if PACK_STATS:
        tl.static_assert(DQ_PAD_ROWS == 32)
        tl.static_assert(BLOCK_M % 32 == 0)
        # Each BM32 tile stores [FP32 log2 LSE rows0..31 | Delta rows0..31].
        # The sequence/head bases need not be aligned to a 32-row boundary.
        stats_sequence_start = q_start + batch * (DQ_PAD_ROWS - 1)
        stats_base = 2 * (head.to(tl.int64) * TOTAL_Q_PADDED + stats_sequence_start)
        stats_offsets = stats_base + (offs_m // 32) * 64 + offs_m % 32
        raw_lse = tl.load(LSE + head.to(tl.int64) * TOTAL_Q + token, mask=offs_m < q_len, other=0.0)
        # Preserve the core's separately rounded scalar FP32 multiply.
        lse_log2 = tl.inline_asm_elementwise(
            "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
            "=v,v",
            [raw_lse],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(Delta + stats_offsets, lse_log2, mask=offs_m < q_len)
        tl.store(Delta + stats_offsets + 32, delta_values, mask=offs_m < q_len)
    else:
        tl.store(Delta + token * HEADS + head, delta_values, mask=offs_m < q_len)
    if ZERO_DQ:
        tl.static_assert(DQ_PAD_ROWS == 16 or DQ_PAD_ROWS == 32)
        tl.static_assert(BLOCK_M % DQ_PAD_ROWS == 0)
        # Wide dQ stores use both native BM16 halves, including padded rows.
        padded_q_len = tl.cdiv(q_len, DQ_PAD_ROWS) * DQ_PAD_ROWS
        scratch_start = q_start + batch.to(tl.int64) * (DQ_PAD_ROWS - 1)
        scratch_base = (head.to(tl.int64) * TOTAL_Q_PADDED + scratch_start) * D
        scratch_offsets = offs_m[:, None] * D + offs_d[None, :]
        tl.store(DQ_ACC + scratch_base + scratch_offsets, 0.0, mask=offs_m[:, None] < padded_q_len)


@triton.jit
def _bm32_load_initial_score_prefix_owner(tile, HALF: tl.constexpr, QT_LAYOUT: tl.constexpr):
    prefix = tlx.local_slice(tile, [16 * HALF, 0], [16, 32])
    value = tlx.local_load(
        tlx.local_trans(prefix),
        layout=QT_LAYOUT,
        relaxed=True,
        rematerialize_coordinates_group=200,
    )
    return tlx.require_layout(value, QT_LAYOUT, pin=True)


@triton.jit
def _varlen_gqa_dq_bm32_owner_reload_panel(
    ds_buffer,
    k_buffer,
    k_band0_panel0,
    k_band0_panel1,
    k_band1_panel0,
    v_operand,
    next_q_slice,
    next_do_slice,
    next_lse_tile,
    next_delta_tile,
    MMA_MD: tl.constexpr,
    DS_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
    V_LAYOUT: tl.constexpr,
    QT_LAYOUT: tl.constexpr,
    STATS_LAYOUT: tl.constexpr,
    DELTA_OFFSET: tl.constexpr = 0,
    LSE_PRESCALED: tl.constexpr = False,
):
    """Eight ordered K32 updates per dQ fragment, with two bands in flight."""
    v_operand = tlx.require_layout(v_operand, V_LAYOUT, pin=False)
    c00 = tlx.zeros((16, 64), tl.float32, layout=MMA_MD)
    c01 = tlx.zeros((16, 64), tl.float32, layout=MMA_MD)
    c10 = tlx.zeros((16, 64), tl.float32, layout=MMA_MD)
    c11 = tlx.zeros((16, 64), tl.float32, layout=MMA_MD)

    # Keep three K panels cached; reload band1/panel1 for each ordered dQ phase.
    tlx.amd_sched_barrier(0)
    k0 = tlx.require_layout(k_band0_panel0, K_MD_LAYOUT, pin=False)
    k1 = tlx.require_layout(k_band0_panel1, K_MD_LAYOUT, pin=False)
    s0 = _bm32_load_dq_s(ds_buffer, 0, 0, DS_MD_LAYOUT)
    s1 = _bm32_load_dq_s(ds_buffer, 0, 1, DS_MD_LAYOUT)
    next_k0 = tlx.require_layout(k_band1_panel0, K_MD_LAYOUT, pin=False)
    next_k1 = tlx.require_layout(_bm32_load_dq_k(k_buffer, 1, 1, K_MD_LAYOUT), K_MD_LAYOUT, pin=False)
    next_s0 = _bm32_load_dq_s(ds_buffer, 1, 0, DS_MD_LAYOUT)
    next_s1 = _bm32_load_dq_s(ds_buffer, 1, 1, DS_MD_LAYOUT)
    tlx.amd_sched_barrier(0)

    # Two startup markers, 32 MFMA markers and 24 future-load markers = 58.
    # Late carries occupy source MFMA ordinals 23..30, before its marker.
    # LSE loads stay at source MFMA ordinals 23/24 (zero-based). Raw fallbacks
    # retain their scalar scales at 28/29; packed values were scaled in PRE.
    for band in tl.static_range(0, 8):
        c00 = tlx.amd_scheduled_mfma(s0, k0, c00, resident_operand=1, accumulator_role="transient",
                                     initialize=band == 0)
        if band == 6:
            next_lse1 = _bm32_load_stat_half(next_lse_tile, 1, STATS_LAYOUT)
        if band == 7:
            next_q1 = _bm32_load_score_prefix(next_q_slice, 1, QT_LAYOUT)
            # Scalar math prevents LLVM from packing an LSE lane with a score
            # fragment and adding a false dependency between independent fragments.
            if not LSE_PRESCALED:
                next_lse0 = tl.inline_asm_elementwise(
                    "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
                    "=v,v",
                    [next_lse0],  # noqa: F821 - Loaded in static band 5 before use in band 7.
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
        tlx.amd_sched_barrier(0)
        if band < 6:
            future_k0 = _bm32_load_dq_k(k_buffer, band + 2, 0, K_MD_LAYOUT)
            tlx.amd_sched_barrier(0)

        c01 = tlx.amd_scheduled_mfma(s0, k1, c01, resident_operand=1, accumulator_role="transient",
                                     initialize=band == 0)
        if band == 6:
            next_delta0 = _bm32_load_stat_half(next_delta_tile, 0, STATS_LAYOUT, BASE_OFFSET=DELTA_OFFSET)
        if band == 7:
            next_do0 = _bm32_load_score_prefix(next_do_slice, 0, QT_LAYOUT)
            if not LSE_PRESCALED:
                next_lse1 = tl.inline_asm_elementwise(
                    "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
                    "=v,v",
                    [next_lse1],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
        tlx.amd_sched_barrier(0)
        if band < 6:
            future_k1 = _bm32_load_dq_k(k_buffer, band + 2, 1, K_MD_LAYOUT)
            tlx.amd_sched_barrier(0)

        c10 = tlx.amd_scheduled_mfma(s1, k0, c10, resident_operand=1, accumulator_role="transient",
                                     initialize=band == 0)
        if band == 6:
            next_delta1 = _bm32_load_stat_half(next_delta_tile, 1, STATS_LAYOUT, BASE_OFFSET=DELTA_OFFSET)
        if band == 7:
            next_do1 = _bm32_load_score_prefix(next_do_slice, 1, QT_LAYOUT)
        tlx.amd_sched_barrier(0)
        if band < 6:
            future_s0 = _bm32_load_dq_s(ds_buffer, band + 2, 0, DS_MD_LAYOUT)
            tlx.amd_sched_barrier(0)

        c11 = tlx.amd_scheduled_mfma(s1, k1, c11, resident_operand=1, accumulator_role="transient",
                                     initialize=band == 0)
        if band == 5:
            next_lse0 = _bm32_load_stat_half(next_lse_tile, 0, STATS_LAYOUT)
        if band == 6:
            next_q0 = _bm32_load_score_prefix(next_q_slice, 0, QT_LAYOUT)
        tlx.amd_sched_barrier(0)
        if band < 6:
            future_s1 = _bm32_load_dq_s(ds_buffer, band + 2, 1, DS_MD_LAYOUT)
            tlx.amd_sched_barrier(0)

        if band < 7:
            k0, k1, s0, s1 = next_k0, next_k1, next_s0, next_s1
        if band < 6:
            next_k0, next_k1, next_s0, next_s1 = future_k0, future_k1, future_s0, future_s1

    dq_lo = tlx.require_layout(_bm32_cat_cols(c00, c01, MMA_MD), MMA_MD, pin=False)
    dq_hi = tlx.require_layout(_bm32_cat_cols(c10, c11, MMA_MD), MMA_MD, pin=False)
    dq_lo, dq_hi, v_operand = tlx.amd_mfma_commit((dq_lo, dq_hi), v_operand)
    return (
        tlx.require_layout(dq_lo, MMA_MD, pin=False),
        tlx.require_layout(dq_hi, MMA_MD, pin=False),
        tlx.require_layout(v_operand, V_LAYOUT, pin=False),
        next_lse0,
        next_lse1,
        next_delta0,
        next_delta1,
        next_q0,
        next_q1,
        next_do0,
        next_do1,
    )


@triton.jit
def _group_owner_rolling_fp32_boundary_s3(STATE):
    """Complete ordinary VMEM on every wave, then expose a CTA-wide boundary."""
    tlx.async_load_wait_group(0)
    drained_state = tl.inline_asm_elementwise("s_waitcnt vmcnt(0);", "=s,0,~{memory}", [STATE], dtype=tl.int32,
                                              is_pure=False, pack=1)
    tl.debug_barrier()
    # Consumers use a scalar made opaque after all waves reach the boundary.
    return tl.inline_asm_elementwise("s_mov_b32 $0, $1;", "=s,s,~{memory}", [drained_state], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def _group_owner_rolling_fp32_chunk_s3(
    current_part,
    R,
    OUT,
    kv_global_start,
    kv_head,
    kv_valid_rows,
    split,
    READY_OWNER,
    ROW_HALF: tl.constexpr,
    COL_HALF: tl.constexpr,
    GRAD_ID: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    CHUNK_LAYOUT: tl.constexpr,
):
    """Store P0, update R01, or emit BF16(R01 + P2) for one 128x64 chunk."""
    tl.static_assert(D == 128)
    tl.static_assert(KV_SPLITS == 3)
    current_part = tlx.require_layout(current_part, CHUNK_LAYOUT, pin=False)
    # Each owner's dK scale must round to FP32 before any owner-fold addition.
    # This opaque register identity also leaves dV's original FP32 value intact.
    current_part = tl.inline_asm_elementwise("", "=v,0", [current_part], dtype=tl.float32, is_pure=False, pack=1)
    current_part = tlx.require_layout(current_part, CHUNK_LAYOUT, pin=False)
    local_n = tlx.rematerialized_range(0, 128, 300 + 100 * GRAD_ID + 2 * ROW_HALF + COL_HALF, placement=READY_OWNER)
    local_d = tlx.rematerialized_range(0, 64, 310 + 100 * GRAD_ID + 2 * ROW_HALF + COL_HALF, placement=READY_OWNER)
    local_n = local_n + 128 * ROW_HALF
    local_d = local_d + 64 * COL_HALF
    output_base = (kv_global_start * HKV + kv_head.to(tl.int64)) * D
    rolling_ptr = tl.multiple_of(R + output_base, 16)
    output_ptr = tl.multiple_of(OUT + output_base, 16)
    offsets = (local_n[:, None] * HKV * D + local_d[None, :]).to(tl.int32)
    offsets = tlx.require_layout(offsets, CHUNK_LAYOUT, pin=False)
    offsets = tl.max_contiguous(offsets, [1, 4])
    valid = tl.broadcast_to((local_n < kv_valid_rows)[:, None], (128, 64))
    valid = tlx.require_layout(valid, CHUNK_LAYOUT, pin=False)
    zero = tlx.zeros((128, 64), tl.float32, layout=CHUNK_LAYOUT)

    tlx.amd_sched_barrier(0)
    # The split is scalar-uniform, and the queue always executes 0, 1, then 2.
    if split == 0:
        # In particular, an empty owner still initializes every valid R row.
        tlx.buffer_store(current_part, rolling_ptr, offsets, mask=valid)
    else:
        previous = tlx.buffer_load(rolling_ptr, offsets, mask=valid, other=zero, contiguity=4)
        if split == 1:
            total = zero + previous
            total = total + current_part
            tlx.buffer_store(total, rolling_ptr, offsets, mask=valid)
        else:
            total = previous + current_part
            final_bf16 = tlx.require_layout(total.to(tl.bfloat16), CHUNK_LAYOUT, pin=False)
            tlx.buffer_store(final_bf16, output_ptr, offsets, mask=valid)
    tlx.amd_sched_barrier(0)


@triton.jit
def _group_owner_rolling_fp32_gradient_s3(
    current_part,
    R,
    OUT,
    kv_global_start,
    kv_head,
    kv_valid_rows,
    split,
    READY_OWNER,
    GRAD_ID: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
):
    """Visit four native register quadrants without exchanging lane ownership."""
    chunk_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((64, 128, 256, 512, 4096, 4, 1024, 2048), (1, 2, 8, 16, 32)),
    )
    quadrants = tl.permute(tl.reshape(current_part, (2, 128, 2, 64)), (1, 3, 0, 2))
    cols0, cols1 = tl.split(quadrants)
    rows0_cols0, rows1_cols0 = tl.split(cols0)
    rows0_cols1, rows1_cols1 = tl.split(cols1)
    _group_owner_rolling_fp32_chunk_s3(rows0_cols0, R, OUT, kv_global_start, kv_head, kv_valid_rows, split, READY_OWNER,
                                       0, 0, GRAD_ID, HKV, D, KV_SPLITS, chunk_layout)
    _group_owner_rolling_fp32_chunk_s3(rows0_cols1, R, OUT, kv_global_start, kv_head, kv_valid_rows, split, READY_OWNER,
                                       0, 1, GRAD_ID, HKV, D, KV_SPLITS, chunk_layout)
    _group_owner_rolling_fp32_chunk_s3(rows1_cols0, R, OUT, kv_global_start, kv_head, kv_valid_rows, split, READY_OWNER,
                                       1, 0, GRAD_ID, HKV, D, KV_SPLITS, chunk_layout)
    _group_owner_rolling_fp32_chunk_s3(rows1_cols1, R, OUT, kv_global_start, kv_head, kv_valid_rows, split, READY_OWNER,
                                       1, 1, GRAD_ID, HKV, D, KV_SPLITS, chunk_layout)


@triton.jit
def _varlen_bwd_interleaved_bm32_rolling_fp32_owner_s3(
    OWNER_ID,
    K_RAW_BUFFER,
    LOAD_K,
    NUM_TASKS,
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
    R_DK,
    R_DV,
    DK_FINAL,
    DV_FINAL,
    SM_SCALE: tl.constexpr,
    TOTAL_Q,
    TOTAL_Q_PADDED,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    PAD_DQ_TO_BM32: tl.constexpr = False,
    PACK_STATS: tl.constexpr = False,
):
    """Process masked BN256 KV owners in BM32 phases with two native BM16 dQ chains."""
    # Four BM16-by-D64 accumulator fragments form two native BM16 output tiles.
    tl.static_assert(D == 128)
    tl.static_assert(BLOCK_M == 32)
    tl.static_assert(BLOCK_N == 256)
    tl.static_assert(HQ % HKV == 0)
    tl.static_assert(HQ // HKV > 1)
    tl.static_assert(KV_SPLITS > 1)
    tl.static_assert(KV_SPLITS <= 4)
    tl.static_assert((HQ // HKV) % KV_SPLITS == 0)
    tl.static_assert(not PACK_STATS or PAD_DQ_TO_BM32)

    # Arrange head/split work for eight XCDs in groups of up to 32 compact
    # KV tasks. The group width was chosen by measurement on gfx950.
    # For W=8q+r, residue xcd owns q+(xcd<r) consecutive mapped entries;
    # min(xcd,r) keeps their segments adjacent in an incomplete final group.
    HS: tl.constexpr = HKV * KV_SPLITS
    w = OWNER_ID
    group = w // (HS * 32)
    first_task = group * 32
    count = tl.minimum(32, NUM_TASKS - first_task)
    rank = w % (HS * 32)
    W = count * HS
    xcd = rank % 8
    local = rank // 8
    mapped = xcd * (W // 8) + tl.minimum(xcd, W % 8) + local
    kv_head_split = mapped // count
    task = first_task + mapped % count
    kv_head = kv_head_split // KV_SPLITS
    split = kv_head_split % KV_SPLITS
    group_size: tl.constexpr = HQ // HKV
    heads_per_split: tl.constexpr = group_size // KV_SPLITS
    kv_global_start = tl.load(KVGlobalStart + task).to(tl.int64)
    q_start = tl.load(QStart + task).to(tl.int64)
    q_scratch_start = tl.load(DQScratchStart + task).to(tl.int64)
    if PAD_DQ_TO_BM32:
        # Plans retain the BM16 base CuQ[sequence] + 15*sequence. Derive the
        # sequence once and extend its scratch gap to 31 rows for this launch.
        sequence = (q_scratch_start - q_start).to(tl.int32) // 15
        q_scratch_start += sequence.to(tl.int64) * 16
    q_len = tl.load(QLen + task)
    kv_valid_rows = tlx.assume_uniform(tl.load(KVValidRows + task))
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
    stats_async_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), ()),
        stride=((1, 0), ()),
    )
    stats_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[1], [2], [4], [8], [16], [32]],
        block_bases=[],
        alignment=16,
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

    # Keep selected LDS bases local to each owner without changing their values.
    owner_zero = tl.inline_asm_elementwise(
        "s_mov_b32 $0, 0;",
        "=s,s",
        [OWNER_ID],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    k_raw_buffer = K_RAW_BUFFER
    k_buffer = tlx.local_reinterpret(
        tlx.local_view(k_raw_buffer, owner_zero),
        tl.bfloat16,
        [BLOCK_N, D],
        layout=k_smem_layout,
    )
    q_buffers = tlx.local_alloc((1, BLOCK_M, D), tl.bfloat16, 2, layout=qdo_smem_layout)
    do_buffers = tlx.local_alloc((1, BLOCK_M, D), tl.bfloat16, 2, layout=qdo_smem_layout)
    ds_buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.bfloat16, 1, layout=ds_smem_layout)
    lse_buffers = tlx.local_alloc((64, ), tl.float32, 2, layout=stats_smem_layout)
    if PACK_STATS:
        delta_buffers = lse_buffers
    else:
        delta_buffers = tlx.local_alloc((64, ), tl.float32, 2, layout=stats_smem_layout)

    raw_n = tlx.rematerialized_range(0, BLOCK_N, 100, placement=OWNER_ID)
    raw_dg = tlx.rematerialized_range(0, D // 8, 101, placement=OWNER_ID)
    raw_v = tlx.rematerialized_range(0, 8, 102, placement=OWNER_ID)
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
    # Initialize the whole K tile on split0, including empty-owner tails.
    if LOAD_K:
        k_token = tlx.buffer_load_to_local(
            tlx.local_view(k_raw_buffer, owner_zero),
            k_ptr,
            k_offsets,
            mask=k_valid,
            other=k_zero,
        )
        tlx.async_load_commit_group([k_token])

    _issue_qdo_bm32_async(
        tlx.local_view(q_buffers, owner_zero),
        tlx.local_view(do_buffers, owner_zero),
        tlx.local_view(lse_buffers, 0),
        tlx.local_view(delta_buffers, 0),
        Q,
        DO,
        LSE,
        Delta,
        q_start,
        q_len,
        first_q_head,
        0,
        TOTAL_Q,
        HQ,
        D,
        BLOCK_M,
        qdo_async_layout,
        stats_async_layout,
        base_q_head=first_q_head,
        REUSE_HEAD_BASE=heads_per_split == 2,
        PACK_STATS=PACK_STATS,
        packed_q_start=q_scratch_start,
        packed_total_q=TOTAL_Q_PADDED,
    )
    second_step = tl.minimum(1, total_outer_steps - 1)
    second_group = second_step // outer_blocks
    second_outer = second_step % outer_blocks
    _issue_qdo_bm32_async(
        tlx.local_view(q_buffers, 1 + owner_zero),
        tlx.local_view(do_buffers, 1 + owner_zero),
        tlx.local_view(lse_buffers, 1),
        tlx.local_view(delta_buffers, 1),
        Q,
        DO,
        LSE,
        Delta,
        q_start,
        q_len,
        first_q_head + second_group,
        second_outer,
        TOTAL_Q,
        HQ,
        D,
        BLOCK_M,
        qdo_async_layout,
        stats_async_layout,
        base_q_head=first_q_head,
        REUSE_HEAD_BASE=heads_per_split == 2,
        PACK_STATS=PACK_STATS,
        packed_q_start=q_scratch_start,
        packed_total_q=TOTAL_Q_PADDED,
    )
    tlx.async_load_wait_group(2)
    tl.debug_barrier()

    # K is immutable for this owner. Retain the full-N first D32 strip once.
    k_prefix32 = tlx.local_load(
        tlx.local_slice(k_buffer, [0, 0], [BLOCK_N, 32]),
        layout=k_nm_layout,
        relaxed=True,
    )
    k_prefix32 = tlx.require_layout(k_prefix32, k_nm_layout, pin=True)

    # Reuse immutable band0 K directly in the dQ operand layout on every phase.
    dq_k_band0_panel0 = _bm32_load_dq_k(k_buffer, 0, 0, k_md_layout)
    dq_k_band0_panel0 = tlx.require_layout(dq_k_band0_panel0, k_md_layout, pin=True)
    dq_k_band0_panel1 = _bm32_load_dq_k(k_buffer, 0, 1, k_md_layout)
    dq_k_band0_panel1 = tlx.require_layout(dq_k_band0_panel1, k_md_layout, pin=True)

    # Extend the immutable dQ cache to band1 with the same prologue layout pins.
    dq_k_band1_panel0 = _bm32_load_dq_k(k_buffer, 1, 0, k_md_layout)
    dq_k_band1_panel0 = tlx.require_layout(dq_k_band1_panel0, k_md_layout, pin=True)

    offs_n = tlx.rematerialized_range(0, BLOCK_N, 103, placement=OWNER_ID)
    offs_d = tlx.rematerialized_range(0, D, 104, placement=OWNER_ID)
    offs_n = tlx.require_layout(offs_n, tlx.slice_layout(k_nm_layout, 1), pin=True)
    offs_d = tlx.require_layout(offs_d, tlx.slice_layout(k_nm_layout, 0), pin=True)
    v_offsets = (offs_n[:, None] * HKV * D + offs_d[None, :]).to(tl.int32)
    v_offsets = tlx.require_layout(v_offsets, k_nm_layout, pin=False)
    v_valid = tl.broadcast_to((offs_n < kv_valid_rows)[:, None], (BLOCK_N, D))
    v_valid = tlx.require_layout(v_valid, k_nm_layout, pin=True)
    v_zero = tlx.zeros((BLOCK_N, D), tl.bfloat16, layout=k_nm_layout)
    v_operand = tlx.buffer_load(v_ptr, v_offsets, mask=v_valid, other=v_zero)
    v_operand = tlx.require_layout(v_operand, k_nm_layout, pin=False)
    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    DQ_ACC = DQ_ACC + q_scratch_start * D

    # The prologue wait has completed stage 0. Carry its raw statistics and
    # first Q/dO D32 band into the first phase in the existing consumer layouts.
    stats_layout: tl.constexpr = tlx.slice_layout(mma_nm, 0)
    initial_q_slice = _bm32_qdo_stage_slice(q_buffers, owner_zero, qdo_slice_smem_layout)
    initial_do_slice = _bm32_qdo_stage_slice(do_buffers, owner_zero, qdo_slice_smem_layout)
    initial_lse_tile = tlx.local_view(lse_buffers, 0)
    initial_delta_tile = tlx.local_view(delta_buffers, 0)
    lse0 = _bm32_load_stat_half(initial_lse_tile, 0, stats_layout)
    lse1 = _bm32_load_stat_half(initial_lse_tile, 1, stats_layout)
    delta0 = _bm32_load_stat_half(initial_delta_tile, 0, stats_layout, BASE_OFFSET=32 if PACK_STATS else 0)
    delta1 = _bm32_load_stat_half(initial_delta_tile, 1, stats_layout, BASE_OFFSET=32 if PACK_STATS else 0)
    q0 = _bm32_load_initial_score_prefix_owner(initial_q_slice, 0, qt_layout)
    q1 = _bm32_load_initial_score_prefix_owner(initial_q_slice, 1, qt_layout)
    do0 = _bm32_load_initial_score_prefix_owner(initial_do_slice, 0, qt_layout)
    do1 = _bm32_load_initial_score_prefix_owner(initial_do_slice, 1, qt_layout)
    # Packed LSE was scaled in PRE. Raw fallback halves keep their original
    # scalar operation and rounding before the first front.
    if not PACK_STATS:
        lse0 = tl.inline_asm_elementwise(
            "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
            "=v,v",
            [lse0],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        lse1 = tl.inline_asm_elementwise(
            "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
            "=v,v",
            [lse1],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    for outer_step in tl.range(0, total_outer_steps, loop_unroll_factor=1):
        outer_stage = outer_step % 2
        group_index = outer_step // outer_blocks
        outer_block = outer_step % outer_blocks
        q_head = first_q_head + group_index
        q_outer = tlx.local_view(q_buffers, outer_stage)
        do_outer = tlx.local_view(do_buffers, outer_stage)
        lse_outer = tlx.local_view(lse_buffers, outer_stage)
        delta_outer = tlx.local_view(delta_buffers, outer_stage)
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
            k_prefix32,
            v_operand,
            dk,
            dv,
            outer_block,
            lse0,
            lse1,
            delta0,
            delta1,
            q0,
            q1,
            do0,
            do1,
            q_len,
            SM_SCALE,
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
            lse_outer,
            delta_outer,
            Q,
            DO,
            LSE,
            Delta,
            q_start,
            q_len,
            first_q_head + next_group,
            next_outer,
            TOTAL_Q,
            HQ,
            D,
            BLOCK_M,
            qdo_async_layout,
            stats_async_layout,
            ZERO_FILL_QDO=True,
            base_q_head=first_q_head,
            REUSE_HEAD_BASE=heads_per_split == 2,
            PACK_STATS=PACK_STATS,
            packed_q_start=q_scratch_start,
            packed_total_q=TOTAL_Q_PADDED,
        )
        # The just-issued refill targets the current slot (i+2); the next
        # consumer takes i+1 from the opposite slot, including the final drain.
        next_stage = outer_stage ^ 1
        next_q_slice = _bm32_qdo_stage_slice(q_buffers, next_stage, qdo_slice_smem_layout)
        next_do_slice = _bm32_qdo_stage_slice(do_buffers, next_stage, qdo_slice_smem_layout)
        next_lse_tile = tlx.local_view(lse_buffers, next_stage)
        next_delta_tile = tlx.local_view(delta_buffers, next_stage)
        (dq_lo, dq_hi, v_operand, next_lse0, next_lse1, next_delta0, next_delta1, next_q0, next_q1, next_do0,
         next_do1) = _varlen_gqa_dq_bm32_owner_reload_panel(
             tlx.local_view(ds_buffers, 0),
             k_buffer,
             dq_k_band0_panel0,
             dq_k_band0_panel1,
             dq_k_band1_panel0,
             v_operand,
             next_q_slice,
             next_do_slice,
             next_lse_tile,
             next_delta_tile,
             mma_md,
             ds_md_layout,
             k_md_layout,
             k_nm_layout,
             qt_layout,
             stats_layout,
             DELTA_OFFSET=32 if PACK_STATS else 0,
             LSE_PRESCALED=PACK_STATS,
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
            MASK_ROWS=not PAD_DQ_TO_BM32,
        )
        tlx.async_load_wait_group(2)
        tl.debug_barrier()
        lse0, lse1 = next_lse0, next_lse1
        delta0, delta1 = next_delta0, next_delta1
        q0, q1 = next_q0, next_q1
        do0, do1 = next_do0, next_do1

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
    # No rolling FP32 state enters a query loop. All owners finish their
    # independent MFMA chains and separately rounded dK scale above this point.
    ready_owner = _group_owner_rolling_fp32_boundary_s3(OWNER_ID)
    _group_owner_rolling_fp32_gradient_s3(dk, R_DK, DK_FINAL, kv_global_start, kv_head, kv_valid_rows, split,
                                          ready_owner, 0, HKV, D, KV_SPLITS)
    # dK and dV use disjoint compact buffers. Drain dK stores on every wave
    # before entering dV's chunks; this boundary also closes dK VMEM lifetimes.
    ready_dv = _group_owner_rolling_fp32_boundary_s3(ready_owner)
    _group_owner_rolling_fp32_gradient_s3(dv, R_DV, DV_FINAL, kv_global_start, kv_head, kv_valid_rows, split, ready_dv,
                                          1, HKV, D, KV_SPLITS)


@triton.jit
def _varlen_bwd_interleaved_bm32_rolling_fp32_queue_s3(
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
    TaskCounts,
    DK_FINAL,
    DV_FINAL,
    OWNER_NEXT,
    NUM_TASKS,
    NUM_GROUPS,
    SM_SCALE: tl.constexpr,
    TOTAL_Q,
    TOTAL_Q_PADDED,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TASK_COUNT_INDEX: tl.constexpr,
    PLAN_ERROR_INDEX: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    PAD_DQ_TO_BM32: tl.constexpr = False,
    PACK_STATS: tl.constexpr = False,
):
    """Claim a KV-task/head group and execute its three original split owners."""
    tl.static_assert(HQ == 12)
    tl.static_assert(HKV == 4)
    tl.static_assert(KV_SPLITS == 3)
    if tl.load(TaskCounts + PLAN_ERROR_INDEX) != 0:
        return
    if NUM_TASKS != tl.load(TaskCounts + TASK_COUNT_INDEX):
        return
    k_raw_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0],
                      [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0], [128, 0, 0]],
        block_bases=[],
        alignment=16,
    )
    # K remains live across all three owners; the other LDS stays per-owner.
    group_k_raw_buffer = tlx.local_alloc((BLOCK_N, D // 8, 8), tl.bfloat16, 1, layout=k_raw_smem_layout)
    HS: tl.constexpr = HKV * KV_SPLITS
    state = KV_SPLITS * tl.program_id(0)
    while tl.condition(state < KV_SPLITS * NUM_GROUPS, disable_licm=True):
        ticket = state // KV_SPLITS
        split_index = state % KV_SPLITS
        task = ticket // HKV
        kv_head = ticket % HKV
        task_group = task // 32
        group_tasks = tl.minimum(32, NUM_TASKS - 32 * task_group)
        local_task = task % 32
        # Invert the original eight-residue owner permutation. For HS=12,
        # odd task-group widths give four longer and four shorter segments.
        mapped = (kv_head * KV_SPLITS + split_index) * group_tasks + local_task
        group_owners = group_tasks * HS
        quotient = group_owners // 8
        remainder = group_owners % 8
        cut = remainder * (quotient + 1)
        xcd = tl.where(
            mapped < cut,
            mapped // (quotient + 1),
            remainder + (mapped - cut) // quotient,
        )
        local = mapped - (xcd * quotient + tl.minimum(xcd, remainder))
        owner = HS * 32 * task_group + 8 * local + xcd
        # One physical owner body; derived group geometry ends at this call.
        _varlen_bwd_interleaved_bm32_rolling_fp32_owner_s3(
            owner,
            group_k_raw_buffer,
            split_index == 0,
            NUM_TASKS,
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
            DK_FINAL,
            DV_FINAL,
            SM_SCALE,
            TOTAL_Q,
            TOTAL_Q_PADDED,
            HQ,
            HKV,
            D,
            BLOCK_M,
            BLOCK_N,
            KV_SPLITS,
            PAD_DQ_TO_BM32,
            PACK_STATS,
        )
        # Publish both compact FP32 buffers after EVERY owner, including
        # empty owners. Split2 publishes final BF16 stores before a new ticket.
        # Async-copy waits alone do not drain ordinary global stores.
        state = _group_owner_rolling_fp32_boundary_s3(state)
        state += 1
        if state % KV_SPLITS == 0:
            state = KV_SPLITS * tl.atomic_add(OWNER_NEXT, 1, sem="relaxed", scope="gpu")


@triton.jit
def _group_owner_rolling_fp32_boundary_s4(STATE):
    """Complete ordinary VMEM on every wave, then expose a CTA-wide boundary."""
    tlx.async_load_wait_group(0)
    drained_state = tl.inline_asm_elementwise("s_waitcnt vmcnt(0);", "=s,0,~{memory}", [STATE], dtype=tl.int32,
                                              is_pure=False, pack=1)
    tl.debug_barrier()
    # Consumers use a scalar made opaque after all waves reach the boundary.
    return tl.inline_asm_elementwise("s_mov_b32 $0, $1;", "=s,s,~{memory}", [drained_state], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def _group_owner_rolling_fp32_chunk_s4(
    current_part,
    R,
    OUT,
    kv_global_start,
    kv_head,
    kv_valid_rows,
    split,
    READY_OWNER,
    ROW_HALF: tl.constexpr,
    COL_HALF: tl.constexpr,
    GRAD_ID: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    CHUNK_LAYOUT: tl.constexpr,
):
    """Store P0, update R01/R012, or emit BF16(R012 + P3) for one 128x64 chunk."""
    tl.static_assert(D == 128)
    tl.static_assert(KV_SPLITS == 4)
    current_part = tlx.require_layout(current_part, CHUNK_LAYOUT, pin=False)
    # Each owner's dK scale must round to FP32 before any owner-fold addition.
    # This opaque register identity also leaves dV's original FP32 value intact.
    current_part = tl.inline_asm_elementwise("", "=v,0", [current_part], dtype=tl.float32, is_pure=False, pack=1)
    current_part = tlx.require_layout(current_part, CHUNK_LAYOUT, pin=False)
    local_n = tlx.rematerialized_range(0, 128, 300 + 100 * GRAD_ID + 2 * ROW_HALF + COL_HALF, placement=READY_OWNER)
    local_d = tlx.rematerialized_range(0, 64, 310 + 100 * GRAD_ID + 2 * ROW_HALF + COL_HALF, placement=READY_OWNER)
    local_n = local_n + 128 * ROW_HALF
    local_d = local_d + 64 * COL_HALF
    output_base = (kv_global_start * HKV + kv_head.to(tl.int64)) * D
    rolling_ptr = tl.multiple_of(R + output_base, 16)
    output_ptr = tl.multiple_of(OUT + output_base, 16)
    offsets = (local_n[:, None] * HKV * D + local_d[None, :]).to(tl.int32)
    offsets = tlx.require_layout(offsets, CHUNK_LAYOUT, pin=False)
    offsets = tl.max_contiguous(offsets, [1, 4])
    valid = tl.broadcast_to((local_n < kv_valid_rows)[:, None], (128, 64))
    valid = tlx.require_layout(valid, CHUNK_LAYOUT, pin=False)
    zero = tlx.zeros((128, 64), tl.float32, layout=CHUNK_LAYOUT)

    tlx.amd_sched_barrier(0)
    # The split is scalar-uniform, and the queue always executes 0, 1, 2, then 3.
    if split == 0:
        # In particular, an empty owner still initializes every valid R row.
        tlx.buffer_store(current_part, rolling_ptr, offsets, mask=valid)
    else:
        previous = tlx.buffer_load(rolling_ptr, offsets, mask=valid, other=zero, contiguity=4)
        if split == 1:
            total = zero + previous
            total = total + current_part
            tlx.buffer_store(total, rolling_ptr, offsets, mask=valid)
        elif split == 2:
            total = previous + current_part
            tlx.buffer_store(total, rolling_ptr, offsets, mask=valid)
        else:
            total = previous + current_part
            final_bf16 = tlx.require_layout(total.to(tl.bfloat16), CHUNK_LAYOUT, pin=False)
            tlx.buffer_store(final_bf16, output_ptr, offsets, mask=valid)
    tlx.amd_sched_barrier(0)


@triton.jit
def _group_owner_rolling_fp32_gradient_s4(
    current_part,
    R,
    OUT,
    kv_global_start,
    kv_head,
    kv_valid_rows,
    split,
    READY_OWNER,
    GRAD_ID: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
):
    """Visit four native register quadrants without exchanging lane ownership."""
    chunk_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((64, 128, 256, 512, 4096, 4, 1024, 2048), (1, 2, 8, 16, 32)),
    )
    quadrants = tl.permute(tl.reshape(current_part, (2, 128, 2, 64)), (1, 3, 0, 2))
    cols0, cols1 = tl.split(quadrants)
    rows0_cols0, rows1_cols0 = tl.split(cols0)
    rows0_cols1, rows1_cols1 = tl.split(cols1)
    _group_owner_rolling_fp32_chunk_s4(rows0_cols0, R, OUT, kv_global_start, kv_head, kv_valid_rows, split, READY_OWNER,
                                       0, 0, GRAD_ID, HKV, D, KV_SPLITS, chunk_layout)
    _group_owner_rolling_fp32_chunk_s4(rows0_cols1, R, OUT, kv_global_start, kv_head, kv_valid_rows, split, READY_OWNER,
                                       0, 1, GRAD_ID, HKV, D, KV_SPLITS, chunk_layout)
    _group_owner_rolling_fp32_chunk_s4(rows1_cols0, R, OUT, kv_global_start, kv_head, kv_valid_rows, split, READY_OWNER,
                                       1, 0, GRAD_ID, HKV, D, KV_SPLITS, chunk_layout)
    _group_owner_rolling_fp32_chunk_s4(rows1_cols1, R, OUT, kv_global_start, kv_head, kv_valid_rows, split, READY_OWNER,
                                       1, 1, GRAD_ID, HKV, D, KV_SPLITS, chunk_layout)


@triton.jit
def _varlen_bwd_interleaved_bm32_rolling_fp32_owner_s4(
    OWNER_ID,
    K_RAW_BUFFER,
    LOAD_K,
    NUM_TASKS,
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
    R_DK,
    R_DV,
    DK_FINAL,
    DV_FINAL,
    SM_SCALE: tl.constexpr,
    TOTAL_Q,
    TOTAL_Q_PADDED,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    PAD_DQ_TO_BM32: tl.constexpr = False,
    PACK_STATS: tl.constexpr = False,
):
    """Process masked BN256 KV owners in BM32 phases with two native BM16 dQ chains."""
    # Four BM16-by-D64 accumulator fragments form two native BM16 output tiles.
    tl.static_assert(D == 128)
    tl.static_assert(BLOCK_M == 32)
    tl.static_assert(BLOCK_N == 256)
    tl.static_assert(HQ % HKV == 0)
    tl.static_assert(HQ // HKV > 1)
    tl.static_assert(KV_SPLITS > 1)
    tl.static_assert(KV_SPLITS <= 4)
    tl.static_assert((HQ // HKV) % KV_SPLITS == 0)
    tl.static_assert(not PACK_STATS or PAD_DQ_TO_BM32)

    # Arrange head/split work for eight XCDs in groups of up to 32 compact
    # KV tasks. The group width was chosen by measurement on gfx950.
    # For W=8q+r, residue xcd owns q+(xcd<r) consecutive mapped entries;
    # min(xcd,r) keeps their segments adjacent in an incomplete final group.
    HS: tl.constexpr = HKV * KV_SPLITS
    w = OWNER_ID
    group = w // (HS * 32)
    first_task = group * 32
    count = tl.minimum(32, NUM_TASKS - first_task)
    rank = w % (HS * 32)
    W = count * HS
    xcd = rank % 8
    local = rank // 8
    mapped = xcd * (W // 8) + tl.minimum(xcd, W % 8) + local
    kv_head_split = mapped // count
    task = first_task + mapped % count
    kv_head = kv_head_split // KV_SPLITS
    split = kv_head_split % KV_SPLITS
    group_size: tl.constexpr = HQ // HKV
    heads_per_split: tl.constexpr = group_size // KV_SPLITS
    kv_global_start = tl.load(KVGlobalStart + task).to(tl.int64)
    q_start = tl.load(QStart + task).to(tl.int64)
    q_scratch_start = tl.load(DQScratchStart + task).to(tl.int64)
    if PAD_DQ_TO_BM32:
        # Plans retain the BM16 base CuQ[sequence] + 15*sequence. Derive the
        # sequence once and extend its scratch gap to 31 rows for this launch.
        sequence = (q_scratch_start - q_start).to(tl.int32) // 15
        q_scratch_start += sequence.to(tl.int64) * 16
    q_len = tl.load(QLen + task)
    kv_valid_rows = tlx.assume_uniform(tl.load(KVValidRows + task))
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
    stats_async_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), ()),
        stride=((1, 0), ()),
    )
    stats_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[1], [2], [4], [8], [16], [32]],
        block_bases=[],
        alignment=16,
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

    # Keep selected LDS bases local to each owner without changing their values.
    owner_zero = tl.inline_asm_elementwise(
        "s_mov_b32 $0, 0;",
        "=s,s",
        [OWNER_ID],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    k_raw_buffer = K_RAW_BUFFER
    k_buffer = tlx.local_reinterpret(
        tlx.local_view(k_raw_buffer, owner_zero),
        tl.bfloat16,
        [BLOCK_N, D],
        layout=k_smem_layout,
    )
    q_buffers = tlx.local_alloc((1, BLOCK_M, D), tl.bfloat16, 2, layout=qdo_smem_layout)
    do_buffers = tlx.local_alloc((1, BLOCK_M, D), tl.bfloat16, 2, layout=qdo_smem_layout)
    ds_buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.bfloat16, 1, layout=ds_smem_layout)
    lse_buffers = tlx.local_alloc((64, ), tl.float32, 2, layout=stats_smem_layout)
    if PACK_STATS:
        delta_buffers = lse_buffers
    else:
        delta_buffers = tlx.local_alloc((64, ), tl.float32, 2, layout=stats_smem_layout)

    raw_n = tlx.rematerialized_range(0, BLOCK_N, 100, placement=OWNER_ID)
    raw_dg = tlx.rematerialized_range(0, D // 8, 101, placement=OWNER_ID)
    raw_v = tlx.rematerialized_range(0, 8, 102, placement=OWNER_ID)
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
    # Initialize the whole K tile on split0, including empty-owner tails.
    if LOAD_K:
        k_token = tlx.buffer_load_to_local(
            tlx.local_view(k_raw_buffer, owner_zero),
            k_ptr,
            k_offsets,
            mask=k_valid,
            other=k_zero,
        )
        tlx.async_load_commit_group([k_token])

    _issue_qdo_bm32_async(
        tlx.local_view(q_buffers, owner_zero),
        tlx.local_view(do_buffers, owner_zero),
        tlx.local_view(lse_buffers, 0),
        tlx.local_view(delta_buffers, 0),
        Q,
        DO,
        LSE,
        Delta,
        q_start,
        q_len,
        first_q_head,
        0,
        TOTAL_Q,
        HQ,
        D,
        BLOCK_M,
        qdo_async_layout,
        stats_async_layout,
        base_q_head=first_q_head,
        REUSE_HEAD_BASE=heads_per_split == 2,
        PACK_STATS=PACK_STATS,
        packed_q_start=q_scratch_start,
        packed_total_q=TOTAL_Q_PADDED,
    )
    second_step = tl.minimum(1, total_outer_steps - 1)
    second_group = second_step // outer_blocks
    second_outer = second_step % outer_blocks
    _issue_qdo_bm32_async(
        tlx.local_view(q_buffers, 1 + owner_zero),
        tlx.local_view(do_buffers, 1 + owner_zero),
        tlx.local_view(lse_buffers, 1),
        tlx.local_view(delta_buffers, 1),
        Q,
        DO,
        LSE,
        Delta,
        q_start,
        q_len,
        first_q_head + second_group,
        second_outer,
        TOTAL_Q,
        HQ,
        D,
        BLOCK_M,
        qdo_async_layout,
        stats_async_layout,
        base_q_head=first_q_head,
        REUSE_HEAD_BASE=heads_per_split == 2,
        PACK_STATS=PACK_STATS,
        packed_q_start=q_scratch_start,
        packed_total_q=TOTAL_Q_PADDED,
    )
    tlx.async_load_wait_group(2)
    tl.debug_barrier()

    # K is immutable for this owner. Retain the full-N first D32 strip once.
    k_prefix32 = tlx.local_load(
        tlx.local_slice(k_buffer, [0, 0], [BLOCK_N, 32]),
        layout=k_nm_layout,
        relaxed=True,
    )
    k_prefix32 = tlx.require_layout(k_prefix32, k_nm_layout, pin=True)
    k_prefix32 = tlx.amd_register_resident(k_prefix32, register_class="agpr", registers_per_group=4)

    # Reuse immutable band0 K directly in the dQ operand layout on every phase.
    dq_k_band0_panel0 = _bm32_load_dq_k(k_buffer, 0, 0, k_md_layout)
    dq_k_band0_panel0 = tlx.require_layout(dq_k_band0_panel0, k_md_layout, pin=True)
    dq_k_band0_panel0 = tlx.amd_register_resident(dq_k_band0_panel0, register_class="agpr", registers_per_group=4)
    dq_k_band0_panel1 = _bm32_load_dq_k(k_buffer, 0, 1, k_md_layout)
    dq_k_band0_panel1 = tlx.require_layout(dq_k_band0_panel1, k_md_layout, pin=True)

    # Extend the immutable dQ cache to band1 with the same prologue layout pins.
    dq_k_band1_panel0 = _bm32_load_dq_k(k_buffer, 1, 0, k_md_layout)
    dq_k_band1_panel0 = tlx.require_layout(dq_k_band1_panel0, k_md_layout, pin=True)
    dq_k_band1_panel0 = tlx.amd_register_resident(dq_k_band1_panel0, register_class="agpr", registers_per_group=4)

    offs_n = tlx.rematerialized_range(0, BLOCK_N, 103, placement=OWNER_ID)
    offs_d = tlx.rematerialized_range(0, D, 104, placement=OWNER_ID)
    offs_n = tlx.require_layout(offs_n, tlx.slice_layout(k_nm_layout, 1), pin=True)
    offs_d = tlx.require_layout(offs_d, tlx.slice_layout(k_nm_layout, 0), pin=True)
    v_offsets = (offs_n[:, None] * HKV * D + offs_d[None, :]).to(tl.int32)
    v_offsets = tlx.require_layout(v_offsets, k_nm_layout, pin=False)
    v_valid = tl.broadcast_to((offs_n < kv_valid_rows)[:, None], (BLOCK_N, D))
    v_valid = tlx.require_layout(v_valid, k_nm_layout, pin=True)
    v_zero = tlx.zeros((BLOCK_N, D), tl.bfloat16, layout=k_nm_layout)
    v_operand = tlx.buffer_load(v_ptr, v_offsets, mask=v_valid, other=v_zero)
    v_operand = tlx.require_layout(v_operand, k_nm_layout, pin=False)
    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    DQ_ACC = DQ_ACC + q_scratch_start * D

    # The prologue wait has completed stage 0. Carry its raw statistics and
    # first Q/dO D32 band into the first phase in the existing consumer layouts.
    stats_layout: tl.constexpr = tlx.slice_layout(mma_nm, 0)
    initial_q_slice = _bm32_qdo_stage_slice(q_buffers, owner_zero, qdo_slice_smem_layout)
    initial_do_slice = _bm32_qdo_stage_slice(do_buffers, owner_zero, qdo_slice_smem_layout)
    initial_lse_tile = tlx.local_view(lse_buffers, 0)
    initial_delta_tile = tlx.local_view(delta_buffers, 0)
    lse0 = _bm32_load_stat_half(initial_lse_tile, 0, stats_layout)
    lse1 = _bm32_load_stat_half(initial_lse_tile, 1, stats_layout)
    delta0 = _bm32_load_stat_half(initial_delta_tile, 0, stats_layout, BASE_OFFSET=32 if PACK_STATS else 0)
    delta1 = _bm32_load_stat_half(initial_delta_tile, 1, stats_layout, BASE_OFFSET=32 if PACK_STATS else 0)
    q0 = _bm32_load_initial_score_prefix_owner(initial_q_slice, 0, qt_layout)
    q1 = _bm32_load_initial_score_prefix_owner(initial_q_slice, 1, qt_layout)
    do0 = _bm32_load_initial_score_prefix_owner(initial_do_slice, 0, qt_layout)
    do1 = _bm32_load_initial_score_prefix_owner(initial_do_slice, 1, qt_layout)
    # Packed LSE was scaled in PRE. Raw fallback halves keep their original
    # scalar operation and rounding before the first front.
    if not PACK_STATS:
        lse0 = tl.inline_asm_elementwise(
            "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
            "=v,v",
            [lse0],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        lse1 = tl.inline_asm_elementwise(
            "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
            "=v,v",
            [lse1],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    for outer_step in tl.range(0, total_outer_steps, loop_unroll_factor=1):
        outer_stage = outer_step % 2
        group_index = outer_step // outer_blocks
        outer_block = outer_step % outer_blocks
        q_head = first_q_head + group_index
        q_outer = tlx.local_view(q_buffers, outer_stage)
        do_outer = tlx.local_view(do_buffers, outer_stage)
        lse_outer = tlx.local_view(lse_buffers, outer_stage)
        delta_outer = tlx.local_view(delta_buffers, outer_stage)
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
            k_prefix32,
            v_operand,
            dk,
            dv,
            outer_block,
            lse0,
            lse1,
            delta0,
            delta1,
            q0,
            q1,
            do0,
            do1,
            q_len,
            SM_SCALE,
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
            lse_outer,
            delta_outer,
            Q,
            DO,
            LSE,
            Delta,
            q_start,
            q_len,
            first_q_head + next_group,
            next_outer,
            TOTAL_Q,
            HQ,
            D,
            BLOCK_M,
            qdo_async_layout,
            stats_async_layout,
            ZERO_FILL_QDO=True,
            base_q_head=first_q_head,
            REUSE_HEAD_BASE=heads_per_split == 2,
            PACK_STATS=PACK_STATS,
            packed_q_start=q_scratch_start,
            packed_total_q=TOTAL_Q_PADDED,
        )
        # The just-issued refill targets the current slot (i+2); the next
        # consumer takes i+1 from the opposite slot, including the final drain.
        next_stage = outer_stage ^ 1
        next_q_slice = _bm32_qdo_stage_slice(q_buffers, next_stage, qdo_slice_smem_layout)
        next_do_slice = _bm32_qdo_stage_slice(do_buffers, next_stage, qdo_slice_smem_layout)
        next_lse_tile = tlx.local_view(lse_buffers, next_stage)
        next_delta_tile = tlx.local_view(delta_buffers, next_stage)
        (dq_lo, dq_hi, v_operand, next_lse0, next_lse1, next_delta0, next_delta1, next_q0, next_q1, next_do0,
         next_do1) = _varlen_gqa_dq_bm32_owner_reload_panel(
             tlx.local_view(ds_buffers, 0),
             k_buffer,
             dq_k_band0_panel0,
             dq_k_band0_panel1,
             dq_k_band1_panel0,
             v_operand,
             next_q_slice,
             next_do_slice,
             next_lse_tile,
             next_delta_tile,
             mma_md,
             ds_md_layout,
             k_md_layout,
             k_nm_layout,
             qt_layout,
             stats_layout,
             DELTA_OFFSET=32 if PACK_STATS else 0,
             LSE_PRESCALED=PACK_STATS,
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
            MASK_ROWS=not PAD_DQ_TO_BM32,
        )
        tlx.async_load_wait_group(2)
        tl.debug_barrier()
        lse0, lse1 = next_lse0, next_lse1
        delta0, delta1 = next_delta0, next_delta1
        q0, q1 = next_q0, next_q1
        do0, do1 = next_do0, next_do1

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
    # No rolling FP32 state enters a query loop. All owners finish their
    # independent MFMA chains and separately rounded dK scale above this point.
    ready_owner = _group_owner_rolling_fp32_boundary_s4(OWNER_ID)
    _group_owner_rolling_fp32_gradient_s4(dk, R_DK, DK_FINAL, kv_global_start, kv_head, kv_valid_rows, split,
                                          ready_owner, 0, HKV, D, KV_SPLITS)
    # dK and dV use disjoint compact buffers. Drain dK stores on every wave
    # before entering dV's chunks; this boundary also closes dK VMEM lifetimes.
    ready_dv = _group_owner_rolling_fp32_boundary_s4(ready_owner)
    _group_owner_rolling_fp32_gradient_s4(dv, R_DV, DV_FINAL, kv_global_start, kv_head, kv_valid_rows, split, ready_dv,
                                          1, HKV, D, KV_SPLITS)


@triton.jit
def _varlen_bwd_interleaved_bm32_rolling_fp32_queue_s4(
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
    TaskCounts,
    DK_FINAL,
    DV_FINAL,
    OWNER_NEXT,
    NUM_TASKS,
    NUM_GROUPS,
    SM_SCALE: tl.constexpr,
    TOTAL_Q,
    TOTAL_Q_PADDED,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TASK_COUNT_INDEX: tl.constexpr,
    PLAN_ERROR_INDEX: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    PAD_DQ_TO_BM32: tl.constexpr = False,
    PACK_STATS: tl.constexpr = False,
):
    """Claim a KV-task/head group and execute its four original split owners."""
    tl.static_assert(HQ == 64)
    tl.static_assert(HKV == 8)
    tl.static_assert(KV_SPLITS == 4)
    if tl.load(TaskCounts + PLAN_ERROR_INDEX) != 0:
        return
    if NUM_TASKS != tl.load(TaskCounts + TASK_COUNT_INDEX):
        return
    k_raw_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0],
                      [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0], [128, 0, 0]],
        block_bases=[],
        alignment=16,
    )
    # K remains live across all four owners; the other LDS stays per-owner.
    group_k_raw_buffer = tlx.local_alloc((BLOCK_N, D // 8, 8), tl.bfloat16, 1, layout=k_raw_smem_layout)
    state = KV_SPLITS * tl.program_id(0)
    while tl.condition(state < KV_SPLITS * NUM_GROUPS, disable_licm=True):
        ticket = state // KV_SPLITS
        split_index = state % KV_SPLITS
        task = ticket // HKV
        kv_head = ticket % HKV
        task_group = task // 32
        group_tasks = tl.minimum(32, NUM_TASKS - 32 * task_group)
        local_task = task % 32
        owner = 1024 * task_group + 8 * (split_index * group_tasks + local_task) + kv_head
        # One physical owner body; derived group geometry ends at this call.
        _varlen_bwd_interleaved_bm32_rolling_fp32_owner_s4(
            owner,
            group_k_raw_buffer,
            split_index == 0,
            NUM_TASKS,
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
            DK_FINAL,
            DV_FINAL,
            SM_SCALE,
            TOTAL_Q,
            TOTAL_Q_PADDED,
            HQ,
            HKV,
            D,
            BLOCK_M,
            BLOCK_N,
            KV_SPLITS,
            PAD_DQ_TO_BM32,
            PACK_STATS,
        )
        # Publish both compact FP32 buffers after EVERY owner, including
        # empty owners. Split3 publishes final BF16 stores before a new ticket.
        # Async-copy waits alone do not drain ordinary global stores.
        state = _group_owner_rolling_fp32_boundary_s4(state)
        state += 1
        if state % KV_SPLITS == 0:
            state = KV_SPLITS * tl.atomic_add(OWNER_NEXT, 1, sem="relaxed", scope="gpu")


def fa_varlen_backward(q, k, v, o, do, lse, plan, sm_scale, causal=False):
    """Run packed BF16 D128 backward using a prepared immutable-offset plan.

    Non-causal mode supports MHA/GQA with independent Q and KV offsets.  Causal
    mode is limited to self-attention MHA, so Q/KV offsets and head counts must
    match.  Its V input may have a larger token stride when the head and D axes
    remain dense, as in TritonBench's ``v_storage[:, 0]`` view.
    """
    _validate_backward_inputs(q, k, v, o, do, lse, plan, sm_scale, causal)
    plan_error_index = _CAUSAL_PLAN_ERROR if causal else _PLAN_ERROR
    total_q, heads, head_dim = q.shape
    total_kv, kv_heads, _ = k.shape
    group_size = heads // kv_heads
    kv_splits = _select_varlen_kv_splits(plan.max_q, group_size)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(k)
    rolling_fp32_case = (not causal and
                         (total_q, total_kv, plan.max_q, getattr(plan, "max_kv", None)) == (50754, 100696, 5662, 10414)
                         and plan.wide_task_count is not None and plan.batch == 19
                         and (heads, kv_heads, head_dim, kv_splits) in ((12, 4, 128, 3), (64, 8, 128, 4))
                         and q.dtype is torch.bfloat16 and k.dtype is torch.bfloat16
                         and _select_varlen_kernel_blocks(group_size, kv_splits) == (_WIDE_BLOCK_M, _WIDE_BLOCK_N)
                         and k.numel() <= _I32_BUFFER_FP32_ELEMENTS)
    if rolling_fp32_case:
        # These compact FP32 buffers replace the three- or four-slot partials.
        # Their public-launch locals keep the established positional plumbing.
        dk_part = torch.empty((total_kv, kv_heads, head_dim), dtype=torch.float32, device=k.device)
        dv_part = torch.empty_like(dk_part)
    else:
        dk_part, dv_part = _allocate_varlen_dkdv_partials(k, kv_splits)
    if dk_part is None:
        kv_splits = 1
    block_m, block_n = _select_varlen_kernel_blocks(group_size, kv_splits)
    # Retain masked BM16 scratch if the larger footprint exceeds the existing
    # signed-byte-offset limit. Input validation already proved the BM16 size.
    pad_dq_to_bm32 = ((block_m, block_n) == (_WIDE_BLOCK_M, _WIDE_BLOCK_N)
                      and (total_q + plan.batch * (_WIDE_BLOCK_M - 1)) * heads * head_dim <= _I32_BUFFER_BF16_ELEMENTS)
    dq_pad_rows = _WIDE_BLOCK_M if pad_dq_to_bm32 else _BLOCK_M
    total_q_padded = total_q + plan.batch * (dq_pad_rows - 1)
    use_dq_aux = group_size == 1 or (block_m, block_n) == (_WIDE_BLOCK_M, _WIDE_BLOCK_N)
    dk_target = dk if dk_part is None else dk_part
    dv_target = dv if dv_part is None else dv_part
    pack_stats = pad_dq_to_bm32 and 2 * heads * total_q_padded <= _I32_BUFFER_FP32_ELEMENTS
    delta_shape = (heads, 2 * total_q_padded) if pack_stats else (total_q, heads)
    delta = torch.empty(delta_shape, dtype=torch.float32, device=q.device)
    allocate_dq_acc = torch.empty if use_dq_aux else torch.zeros
    dq_acc = allocate_dq_acc((heads, total_q_padded, head_dim), dtype=torch.bfloat16, device=q.device)

    owner_tasks = plan.wide_task_count if rolling_fp32_case and plan.wide_task_count is not None else 0
    owner_groups = kv_heads * owner_tasks
    owner_workers = min(256, owner_groups)
    owner_next = torch.empty((1, ), dtype=torch.int32, device=q.device) if rolling_fp32_case else None
    preprocess_fn = _varlen_bwd_preprocess_dynamic_owner_queue if rolling_fp32_case else _varlen_bwd_preprocess
    preprocess_extra = {"OWNER_NEXT": owner_next, "INITIAL_OWNER": owner_workers} if rolling_fp32_case else {}

    preprocess_fn[(triton.cdiv(plan.max_q, 64), plan.batch * heads)](
        o,
        do,
        delta,
        plan.cu_seqlens_q,
        dq_acc,
        total_q_padded,
        plan.task_counts,
        PLAN_ERROR_INDEX=plan_error_index,
        HEADS=heads,
        D=head_dim,
        BLOCK_M=64,
        ZERO_DQ=use_dq_aux,
        DQ_PAD_ROWS=dq_pad_rows,
        LSE=lse if pack_stats else None,
        TOTAL_Q=total_q if pack_stats else None,
        PACK_STATS=pack_stats,
        num_warps=4,
        **preprocess_extra,
    )
    finalize_tail_dq = False
    if (block_m, block_n) == (_WIDE_BLOCK_M, _WIDE_BLOCK_N):
        wide_core = _varlen_bwd_interleaved_bm32_kernel
        if rolling_fp32_case:
            wide_core = (_varlen_bwd_interleaved_bm32_rolling_fp32_queue_s3
                         if kv_splits == 3 else _varlen_bwd_interleaved_bm32_rolling_fp32_queue_s4)
        wide_grid = (owner_workers, 1, 1) if rolling_fp32_case else (kv_heads * kv_splits, plan.wide_kv_start.numel())
        wide_extra = {
            "DK_FINAL": dk,
            "DV_FINAL": dv,
            "OWNER_NEXT": owner_next,
            "NUM_TASKS": owner_tasks,
            "NUM_GROUPS": owner_groups,
        } if rolling_fp32_case else {}
        wide_core[wide_grid](
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
            plan.task_counts,
            SM_SCALE=sm_scale,
            TOTAL_Q=total_q,
            TOTAL_Q_PADDED=total_q_padded,
            HQ=heads,
            HKV=kv_heads,
            D=head_dim,
            BLOCK_M=_WIDE_BLOCK_M,
            BLOCK_N=_WIDE_BLOCK_N,
            TASK_COUNT_INDEX=_WIDE_KV_TASK_COUNT,
            PLAN_ERROR_INDEX=plan_error_index,
            KV_SPLITS=kv_splits,
            PAD_DQ_TO_BM32=pad_dq_to_bm32,
            PACK_STATS=pack_stats,
            num_warps=4,
            num_stages=1,
            matrix_instr_nonkdim=16,
            **wide_extra,
        )
    else:
        # Direct-to-LDS copies need 16-byte aligned Q/dO bases for this layout.
        # Contiguous views with shifted storage retain the generic copy path.
        qdo_aligned = not causal and group_size == 1 and q.data_ptr() % 16 == 0 and do.data_ptr() % 16 == 0
        cache_mha_stats = not causal and group_size == 1 and plan.max_q <= 512
        # Causal tail owners skip earlier Q rows, which still need conversion.
        finalize_tail_dq = (not causal and cache_mha_stats and qdo_aligned and plan.dq_full_kv_sequence is not None
                            and plan.dq_full_kv_start is not None)
        # The peeled MHA loop exposes independent score/dP and gradient work.
        core_llvm_attrs = (("amdgpu-sched-strategy", "max-ilp"), ) if cache_mha_stats else ()
        kv_launches = (
            (
                plan.full_kv_block_sequence,
                plan.full_kv_block_start,
                _FULL_KV_TASK_COUNT,
                True,
            ),
            (
                plan.tail_kv_block_sequence,
                plan.tail_kv_block_start,
                _TAIL_KV_TASK_COUNT,
                False,
            ),
        )
        for block_sequence, block_start, task_count_index, full_kv_tile in kv_launches:
            task_capacity = block_sequence.numel()
            if task_capacity == 0:
                continue
            grid = ((kv_heads * kv_splits, task_capacity) if group_size == 1 and not causal else
                    (task_capacity, kv_heads * kv_splits))
            _varlen_bwd_interleaved_kernel[grid](
                q,
                k,
                v,
                do,
                lse,
                delta,
                plan.cu_seqlens_q,
                plan.cu_seqlens_k,
                block_sequence,
                block_start,
                dq_acc,
                dk_target,
                dv_target,
                plan.task_counts,
                SM_SCALE=sm_scale,
                TOTAL_Q=total_q,
                TOTAL_Q_PADDED=total_q_padded,
                HQ=heads,
                HKV=kv_heads,
                D=head_dim,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                TASK_COUNT_INDEX=task_count_index,
                PLAN_ERROR_INDEX=plan_error_index,
                FULL_KV_TILE=full_kv_tile,
                KV_SPLITS=kv_splits,
                IS_CAUSAL=causal,
                V_STRIDE_T=v.stride(0),
                QDO_ALIGNED=qdo_aligned,
                CACHE_MHA_STATS=cache_mha_stats,
                DQ_OUTPUT=dq if finalize_tail_dq and not full_kv_tile else None,
                FINALIZE_DQ=finalize_tail_dq and not full_kv_tile,
                DQ_TAIL_K96=finalize_tail_dq and not full_kv_tile and plan.dq_tail_k96,
                num_warps=4,
                num_stages=1,
                matrix_instr_nonkdim=16,
                llvm_fn_attrs=core_llvm_attrs,
            )
    if kv_splits > 1 and not rolling_fp32_case:
        _varlen_dkdv_reduce_kernel[(triton.cdiv(total_kv, _BLOCK_M), kv_heads)](
            dk_part,
            dv_part,
            dk,
            dv,
            plan.task_counts,
            total_kv,
            PLAN_ERROR_INDEX=plan_error_index,
            HKV=kv_heads,
            D=head_dim,
            KV_SPLITS=kv_splits,
            BLOCK_N=_BLOCK_M,
            num_warps=4,
        )
    if use_dq_aux:
        dq_sequence = plan.dq_full_kv_sequence if finalize_tail_dq else plan.q_block_sequence
        dq_start = plan.dq_full_kv_start if finalize_tail_dq else plan.q_block_start
        if dq_sequence is not None and dq_start is not None and dq_sequence.numel():
            _varlen_mha_dq_convert_coalesced_kernel[(dq_sequence.numel(), heads)](
                dq_acc,
                plan.cu_seqlens_q,
                dq_sequence,
                dq_start,
                plan.task_counts,
                dq,
                TOTAL_Q_PADDED=total_q_padded,
                HEADS=heads,
                D=head_dim,
                BLOCK_M=_BLOCK_M,
                TASK_COUNT_INDEX=_Q_TASK_COUNT,
                PLAN_ERROR_INDEX=plan_error_index,
                DQ_PAD_ROWS=dq_pad_rows,
                num_warps=4,
            )
    else:
        _varlen_dq_convert_kernel[(plan.q_block_sequence.numel(), heads)](
            dq_acc,
            plan.cu_seqlens_q,
            plan.q_block_sequence,
            plan.q_block_start,
            plan.task_counts,
            dq,
            TOTAL_Q_PADDED=total_q_padded,
            HEADS=heads,
            D=head_dim,
            BLOCK_M=_BLOCK_M,
            TASK_COUNT_INDEX=_Q_TASK_COUNT,
            PLAN_ERROR_INDEX=plan_error_index,
            num_warps=4,
            matrix_instr_nonkdim=16,
        )
    return dq, dk, dv
