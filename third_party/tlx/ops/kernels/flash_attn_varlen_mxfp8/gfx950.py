# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""Packed-varlen MXFP8 attention forward on the 4-wave scaled-MFMA core.

Packed [tokens, 4, 128] BF16 Q/K/V, non-causal. One preparation kernel
quantizes K and V into head-major buffers in which every sequence owns a span
of zero-padded keys that starts at a multiple of 256 keys and is a multiple of
256 keys long, in the layouts the core reads (K row-major with permuted scale
words, V transposed with keys reordered inside 32-key groups). Q is quantized
in registers, or ahead of time for ``forward_prequantized``. Work items are
128-row query-tile/head pairs, cost-balanced over a persistent grid. The
padding sits at both ends of a span (see ``_front``), so only an item's first
and last 128-key tiles hold padding keys, which get scores of -1e30 or less.
The sequence-offset tensors must not be modified in place while their prepared
metadata and work schedule are reused.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import weakref
from typing import Any, cast

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from ..flash_attn_mxfp8.gfx950 import _K_LDS_BASES, _V_LDS_BASES
from . import mxfp8_attention_core_gfx950 as core

_HEADS = 4
_HEAD_DIM = 128
_BLOCK_M = 128
# Sequences averaging at least this many keys are long: the kernel then reads
# each work item's metadata again instead of carrying it over from the prefetch,
# and the approximate exponential is the default.
_LONG_KV = 4096
# Each sequence's keys are padded to a multiple of two 128-key tiles: the
# kernel takes the tiles in pairs, one per LDS slot.
_PAD = 256
# Keys per program of the K/V preparation kernel.
_PREP_ROWS = 64
_LOG2E = 1.4426950408889634

_LAUNCH_OPTIONS = {
    "num_warps": 4,
    "num_stages": 1,
    # VGPR-only register allocation, and no IEEE mode: max then needs no
    # canonicalizing v_max x, x of its operands.
    "llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"), ("amdgpu-ieee", "false")),
    # core._mfma_stage places its own scheduling barriers. Set explicitly: an
    # unset option follows the environment knob.
    "enable_sched_group_barrier_scheduler": False,
}


@triton.jit
def _prepare_tile(K, V, K8, KS, V8, VS, src, dst, lo, hi, HEADS: tl.constexpr):
    """Quantize 64 keys of one head; rows outside [lo, hi) are written as zeros."""
    _rows_tile(K, K8, KS, src, dst, lo, hi, HEADS)
    _cols_tile(V, V8, VS, src, dst, lo, hi, HEADS)


@triton.jit
def _rows_tile(K, K8, KS, src, dst, lo, hi, HEADS: tl.constexpr):
    """64 rows in the K operand layout: row-major, one E8M0 scale per 32x32
    tile, scale words permuted for the core's _ld_ks."""
    rows = tl.arange(0, 64)
    x = tlx.buffer_load(
        K + src,
        rows[:, None] * (HEADS * 128) + tl.arange(0, 128)[None, :],
        ((rows >= lo) & (rows < hi))[:, None],
        0.0,
        contiguity=8,
    )
    _rows_store(x, K8, KS, dst)


@triton.jit
def _rows_store(x, K8, KS, dst):
    """_rows_tile's quantization and stores of 64 loaded rows."""
    BLOCK: tl.constexpr = 64
    rows = tl.arange(0, BLOCK)
    columns = tl.arange(0, 128)
    magnitude = ((x.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.uint16).reshape((BLOCK // 32, 32, 4, 32)))
    maximum_bits = tl.max(tl.max(magnitude, 3), 1)
    maximum = maximum_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    byte = core._mx_scale(maximum)
    operand_scale = (byte << 23).to(tl.float32, bitcast=True)
    operand_scale = tl.broadcast_to(operand_scale[:, None, :, None], (BLOCK // 32, 32, 4, 32)).reshape((BLOCK, 128))
    tlx.buffer_store(
        core._to_e4m3(x, operand_scale),
        K8 + dst * 128,
        rows[:, None] * 128 + columns[None, :],
    )
    scales = tl.broadcast_to(byte[:, None, :], (BLOCK // 32, 32, 4)).to(tl.uint8)
    scales = (scales.reshape((BLOCK // 64, 2, 32, 2, 2)).permute(0, 4, 2, 3, 1).reshape((BLOCK, 4)))
    tlx.buffer_store(
        scales,
        KS + dst * 4,
        rows[:, None] * 4 + tl.arange(0, 4)[None, :],
    )


@triton.jit
def _cols_tile(V, V8, VS, src, dst, lo, hi, HEADS: tl.constexpr):
    """64 rows in the V operand layout: transposed per 64 rows with rows
    reordered inside 32-row groups, one E8M0 scale per column and 32 rows."""
    rows = tl.arange(0, 64)
    source_rows = (rows & ~20) | (((rows >> 2) & 1) << 4) | (((rows >> 4) & 1) << 2)
    v = tlx.buffer_load(
        V + src,
        source_rows[:, None] * (HEADS * 128) + tl.arange(0, 128)[None, :],
        ((source_rows >= lo) & (source_rows < hi))[:, None],
        0.0,
        contiguity=8,
    )
    _cols_store(v, V8, VS, dst)


@triton.jit
def _cols_store(v, V8, VS, dst):
    """_cols_tile's quantization and stores of 64 loaded rows, already in
    its reordered row order."""
    BLOCK: tl.constexpr = 64
    rows = tl.arange(0, BLOCK)
    columns = tl.arange(0, 128)
    v_bits = ((v.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.uint16).reshape((BLOCK // 32, 32, 128)))
    v_max = tl.max(v_bits, 1).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    v_byte = core._mx_scale(v_max)
    v_operand_scale = (v_byte << 23).to(tl.float32, bitcast=True)
    v_operand_scale = tl.broadcast_to(tl.trans(v_operand_scale)[:, :, None], (128, BLOCK // 32, 32)).reshape(
        (128, BLOCK))
    v_cast_layout: tl.constexpr = tlx.layout(
        shape=((16, 4, tlx.num_warps()), (4, 128 // (4 * tlx.num_warps()))),
        stride=((4, 64, 256), (1, 256 * tlx.num_warps())),
    )
    v_transposed = tlx.require_layout(tl.trans(v), v_cast_layout, pin=True)
    v_operand_scale = tlx.require_layout(v_operand_scale, v_cast_layout, pin=True)
    tlx.buffer_store(
        core._to_e4m3(v_transposed, v_operand_scale),
        V8 + dst * 128,
        columns[:, None] * BLOCK + rows[None, :],
    )
    v_scales = (v_byte.to(tl.uint8).reshape((BLOCK // 32, 4, 32)).permute(0, 2, 1).reshape((BLOCK, 4)))
    tlx.buffer_store(
        v_scales,
        VS + dst * 4,
        rows[:, None] * 4 + tl.arange(0, 4)[None, :],
    )


@triton.jit
def _front(length, padded):
    # Padding keys before the sequence: only the first and last 128-key tiles
    # then hold padding, and the pipelined middle tiles need no mask.
    return tl.maximum(padded - length - 128, 0)


@triton.jit
def _prepare_kv(
    K,
    V,
    K8,
    KS,
    V8,
    VS,
    CuKV,
    CuPad,
    TileSequence,
    CAPACITY,
    BATCH,
    NPAD,
    HEADS: tl.constexpr,
):
    """Quantize one padded 64-key tile of one sequence and head."""
    tile = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    sequence = tl.load(TileSequence + tile, mask=tile < CAPACITY, other=BATCH)
    if sequence >= BATCH:
        return
    first = tl.load(CuKV + sequence)
    length = tl.load(CuKV + sequence + 1) - first
    pad_start = tl.load(CuPad + sequence)
    row = tile * 64 - pad_start
    key = row - _front(length, tl.load(CuPad + sequence + 1) - pad_start)
    src = (first + key).to(tl.int64) * (HEADS * 128) + head * 128
    dst = head * NPAD + pad_start + row
    _prepare_tile(K, V, K8, KS, V8, VS, src, dst, -key, length - key, HEADS)


@triton.jit
def _item(
    item,
    ItemTile,
    CuQTile,
    CuQ,
    CuKV,
    CuPad,
    HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Work item ``item``: one head of one BLOCK_M-row query tile.

    Returns (head, first query row, query rows left in the sequence, padding
    keys ahead of the sequence, valid keys of the last 128-key tile, start of
    the sequence's padded key span, number of 128-key tiles in it)."""
    tile = item // HEADS
    sequence = tl.load(ItemTile + tile)
    q0 = tl.load(CuQ + sequence) + (tile - tl.load(CuQTile + sequence)) * BLOCK_M
    q_rows = tl.load(CuQ + sequence + 1) - q0
    kv_len = tl.load(CuKV + sequence + 1) - tl.load(CuKV + sequence)
    pad_start = tl.load(CuPad + sequence)
    padded = tl.load(CuPad + sequence + 1) - pad_start
    front = _front(kv_len, padded)
    # Valid keys of the last tile.
    tail = 128 - (padded - kv_len - front)
    return (
        (item % HEADS).to(tl.int64),
        q0,
        q_rows,
        front,
        tail,
        pad_start,
        padded // 128,
    )


@triton.jit
def _kv_pointers(K8, V8, KS, VS, head, pad_start, NPAD, k_off, v_off, ks_off):
    """Per-element pointers to the first 64-key half of a padded key span, for
    K, V and their E8M0 scales taken as one int32 word per key."""
    key = head * NPAD + pad_start
    return (
        K8 + key * 128 + k_off,
        V8 + key * 128 + v_off,
        KS.to(tl.pointer_type(tl.int32)) + key + ks_off,
        VS.to(tl.pointer_type(tl.int32)) + key + ks_off,
    )


@triton.jit
def _mask_front(s0, s1, front):
    # Branch-free (-1e30 or less): the QK MFMAs get the softmax's
    # one-warp-per-32-rows layout only when the backend sees their scores
    # reach a PV MFMA, and it does not look through scf.if results.
    # Arithmetic rather than selects: per-element compares would each take an
    # SGPR pair and spill. The last tile is masked in core._mfma_stage.
    # Subtract in integers: LLVM then hoists only the lane's base key, not a
    # float per element.
    keys = tl.arange(0, 64)[None, :] - front
    return (
        s0 + tl.minimum(keys, 0).to(tl.float32) * 1e30,
        s1 + tl.minimum(keys + 64, 0).to(tl.float32) * 1e30,
    )


@triton.jit
def _mfma_softmax_loop_4w(
    q_lo,
    q_hi,
    qs_lo,
    qs_hi,
    ka,
    kb,
    ksa,
    ksb,
    va,
    vb,
    vsa,
    vsb,
    s0,
    s1,
    p0,
    p1,
    m_i,
    l_i,
    acc0,
    acc1,
    acc2,
    acc3,
    n_tiles,
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    c_qk,
    qk_scale,
    FAST_EXP: tl.constexpr,
):
    """An item's middle tiles, two per iteration.

    On entry P holds tile 0 and LDS slot 0 holds K tile 1 and V tile 0. Each
    stage takes QK of the next tile and PV of the current one from one slot
    while it refills the other, so the slots alternate; the softmax of the new
    scores follows. On exit the scores and P are those of tile n_tiles - 2, and
    slot 0 holds K tile n_tiles - 1 and V tile n_tiles - 2 for the caller's
    last stage."""
    OFF: tl.constexpr = core._score_offset(FAST_EXP)
    HN: tl.constexpr = 64
    D: tl.constexpr = 128
    k_step = 128 * D
    v_step = 128 * D
    ks_step = 128
    k_n1 = HN * D
    v_n1 = HN * D
    ks_n1 = HN
    for it in tl.range(0, (n_tiles - 2) // 2, num_stages=1):
        t = 2 * it + 2
        s0, s1, acc0, acc1, acc2, acc3, l_i = core._mfma_stage(
            q_lo,
            q_hi,
            qs_lo,
            qs_hi,
            ka[0],
            kb[0],
            ksa[0],
            ksb[0],
            va[0],
            vb[0],
            vsa[0],
            vsb[0],
            p0,
            p1,
            acc0,
            acc1,
            acc2,
            acc3,
            l_i,
            (OFF - m_i) * c_qk,
            qk_scale,
            k_ptr,
            v_ptr,
            ks_ptr,
            vs_ptr,
            ka[1],
            kb[1],
            va[1],
            vb[1],
            ksa[1],
            ksb[1],
            vsa[1],
            vsb[1],
            t,
            t - 1,
            k_step,
            v_step,
            ks_step,
            k_n1,
            v_n1,
            ks_n1,
        )
        p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = core._softmax(s0, s1, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale,
                                                                 FAST_EXP)
        t = 2 * it + 3
        s0, s1, acc0, acc1, acc2, acc3, l_i = core._mfma_stage(
            q_lo,
            q_hi,
            qs_lo,
            qs_hi,
            ka[1],
            kb[1],
            ksa[1],
            ksb[1],
            va[1],
            vb[1],
            vsa[1],
            vsb[1],
            p0,
            p1,
            acc0,
            acc1,
            acc2,
            acc3,
            l_i,
            (OFF - m_i) * c_qk,
            qk_scale,
            k_ptr,
            v_ptr,
            ks_ptr,
            vs_ptr,
            ka[0],
            kb[0],
            va[0],
            vb[0],
            ksa[0],
            ksb[0],
            vsa[0],
            vsb[0],
            t,
            t - 1,
            k_step,
            v_step,
            ks_step,
            k_n1,
            v_n1,
            ks_n1,
        )
        p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = core._softmax(s0, s1, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale,
                                                                 FAST_EXP)
    return s0, s1, p0, p1, m_i, l_i, acc0, acc1, acc2, acc3


@triton.jit
def _item_uniform(item, ItemTile, CuQTile, CuQ, CuKV, CuPad, HEADS: tl.constexpr, BLOCK_M: tl.constexpr):
    # Work-item metadata is loaded from memory, which leaves each scalar (and
    # every loop bound and offset derived from it) in a VGPR. The values are
    # block-uniform, so move them to SGPRs.
    head, q0, q_rows, front, tail, pad_start, n_tiles = _item(item, ItemTile, CuQTile, CuQ, CuKV, CuPad, HEADS, BLOCK_M)
    return (
        tlx.assume_uniform(head),
        tlx.assume_uniform(q0),
        tlx.assume_uniform(q_rows),
        tlx.assume_uniform(front),
        tlx.assume_uniform(tail),
        tlx.assume_uniform(pad_start),
        tlx.assume_uniform(n_tiles),
    )


@triton.jit
# Triton TR001: BLOCK_M is a host-selected production contract.
def _mxfp8_varlen_extended_kernel_4(  # noqa: TR001
    Q,
    QScales,
    WorkOrder,
    K8,
    V8,
    KS,
    VS,
    Out,
    Lse,
    CuQ,
    CuKV,
    CuPad,
    CuQTile,
    ItemTile,
    BATCH,
    TOTAL_Q,
    NPAD,
    qk_scale,
    HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    FAST_EXP: tl.constexpr,
    PREQUANTIZED_Q: tl.constexpr,
    REUSE_METADATA: tl.constexpr,
    USE_WORK_ORDER: tl.constexpr,
):
    """Persistent 4-wave forward: each program walks its share of the work items.

    Q is BF16 and quantized here, or with PREQUANTIZED_Q E4M3 with its E8M0
    bytes in QScales. K8, V8, KS and VS are prepare_kv's buffers, and CuPad,
    CuQTile and ItemTile are _metadata's tables. qk_scale is the softmax scale
    in the log2 domain. With USE_WORK_ORDER the items come in WorkOrder's order,
    else in index order; REUSE_METADATA then also carries the metadata read for
    an item's prefetch over to the item instead of reading it again.

    An item's tiles alternate between two LDS slots. QK of tile 0 comes first,
    masked ahead of the sequence start, then core._mfma_stage takes QK of tile
    j + 1 with PV of tile j for every later tile, the last one masked past the
    sequence end, and core._pv_last closes with PV of the last tile. The next
    item's first copies are issued behind the last stage, so they land during
    the item's epilogue."""
    HN: tl.constexpr = 64
    HD: tl.constexpr = 64
    D: tl.constexpr = 128
    ROW: tl.constexpr = HEADS * D
    pid = tl.program_id(0)
    n_ctas = tl.num_programs(0)
    n_items = tlx.assume_uniform(tl.load(CuQTile + BATCH)) * HEADS
    base = tlx.assume_uniform(pid % 8 * (n_ctas // 8) + pid // 8)
    if base >= n_items:
        return
    n_mine = tlx.assume_uniform((n_items - base + n_ctas - 1) // n_ctas)
    offs_d = tl.arange(0, D)
    offs_hd = tl.arange(0, HD)
    offs_n = tl.arange(0, HN)
    offs_mm = tl.arange(0, BLOCK_M)
    C: tl.constexpr = core._score_scale(FAST_EXP)
    OFF: tl.constexpr = core._score_offset(FAST_EXP)
    c_qk = tlx.assume_uniform(C / qk_scale)
    k_off = offs_d[:, None] + offs_n[None, :] * D
    v_off = offs_n[:, None] + offs_d[None, :] * HN
    ks_off = offs_n[:, None]
    q_off = offs_mm[:, None] * ROW + offs_hd[None, :]
    k_step = 128 * D
    v_step = 128 * D
    ks_step = 128
    k_n1 = HN * D
    v_n1 = HN * D
    ks_n1 = HN
    mfma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], True, [tlx.num_warps(), 1])
    q_layout: tl.constexpr = tlx.dot_operand_layout(0, mfma, k_width=16)
    q_local = tlx.require_layout(q_off.to(tl.int32), q_layout)
    scale_layout: tl.constexpr = tlx.layout(
        shape=((32, 2, tlx.num_warps()), (1, )),
        stride=((2, 1, 64), (1, )),
    )
    # The fixed-length op's LDS layouts of a 64-key half tile: K [feature, key]
    # with a key's features contiguous, V [key, feature] with a feature's keys
    # contiguous (V comes transposed).
    k_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(1024, 16)], _K_LDS_BASES, [D, HN])
    v_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(1024, 16)], _V_LDS_BASES, [HN, D])
    ka = tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout)
    kb = tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout)
    va = tlx.local_alloc((HN, D), tl.float8e4nv, 2, layout=v_layout)
    vb = tlx.local_alloc((HN, D), tl.float8e4nv, 2, layout=v_layout)
    ksa = tlx.local_alloc((HN, 1), tl.int32, 2)
    ksb = tlx.local_alloc((HN, 1), tl.int32, 2)
    vsa = tlx.local_alloc((HN, 1), tl.int32, 2)
    vsb = tlx.local_alloc((HN, 1), tl.int32, 2)
    item = base
    if USE_WORK_ORDER:
        item = tlx.assume_uniform(tl.load(WorkOrder + base))
    head, q0, q_rows, front, tail, pad_start, n_tiles = _item_uniform(item, ItemTile, CuQTile, CuQ, CuKV, CuPad, HEADS,
                                                                      BLOCK_M)
    k_ptr, v_ptr, ks_ptr, vs_ptr = _kv_pointers(K8, V8, KS, VS, head, pad_start, NPAD, k_off, v_off, ks_off)
    # The first item's opening copies, as three commit groups: K tile 0 into
    # slot 1, then the two halves of K tile 1 and V tile 0 into slot 0.
    tlx.async_load(k_ptr, ka[1])
    tlx.async_load(k_ptr + k_n1, kb[1])
    tlx.async_load(ks_ptr, ksa[1])
    tlx.async_load(ks_ptr + ks_n1, ksb[1])
    tlx.async_load_commit_group()
    core._issue_tile(
        k_ptr,
        v_ptr,
        ks_ptr,
        vs_ptr,
        ka[0],
        kb[0],
        va[0],
        vb[0],
        ksa[0],
        ksb[0],
        vsa[0],
        vsb[0],
        1,
        0,
        k_step,
        v_step,
        ks_step,
        k_n1,
        v_n1,
        ks_n1,
    )
    for w in tl.range(0, n_mine, num_stages=1):
        if not (REUSE_METADATA and USE_WORK_ORDER):
            item = base + w * n_ctas
            if USE_WORK_ORDER:
                item = tlx.assume_uniform(tl.load(WorkOrder + item))
            head, q0, q_rows, front, tail, pad_start, n_tiles = _item_uniform(item, ItemTile, CuQTile, CuQ, CuKV, CuPad,
                                                                              HEADS, BLOCK_M)
        q_base = Q + q0.to(tl.int64) * ROW + head * D
        q_mask = q_local < q_rows * ROW
        if PREQUANTIZED_Q:
            q_lo = tlx.require_layout(
                tlx.buffer_load(q_base, q_local, q_mask, 0.0, contiguity=16),
                q_layout,
            )
            q_hi = tlx.require_layout(
                tlx.buffer_load(q_base + HD, q_local, q_mask, 0.0, contiguity=16),
                q_layout,
            )
            if not (REUSE_METADATA and USE_WORK_ORDER):
                k_ptr, v_ptr, ks_ptr, vs_ptr = _kv_pointers(K8, V8, KS, VS, head, pad_start, NPAD, k_off, v_off, ks_off)
            scale_base = QScales + q0.to(tl.int64) * HEADS * 4 + head * 4
            scale_offsets = (offs_mm[:, None] * (HEADS * 4) + tl.arange(0, 2)[None, :])
            scale_mask = (offs_mm < q_rows)[:, None]
            qs_lo = tlx.require_layout(tl.load(scale_base + scale_offsets, scale_mask, 0), scale_layout)
            qs_hi = tlx.require_layout(
                tl.load(scale_base + scale_offsets + 2, scale_mask, 0),
                scale_layout,
            )
        else:
            q_reg0 = tlx.buffer_load(q_base, q_local, q_mask, 0.0, contiguity=8)
            q_reg1 = tlx.buffer_load(q_base + HD, q_local, q_mask, 0.0, contiguity=8)
            tlx.amd_sched_barrier(0)
            if not (REUSE_METADATA and USE_WORK_ORDER):
                k_ptr, v_ptr, ks_ptr, vs_ptr = _kv_pointers(K8, V8, KS, VS, head, pad_start, NPAD, k_off, v_off, ks_off)
            tlx.amd_sched_barrier(0)
            q_lo, qs_lo = core._quantize_q_half(q_reg0)
            q_lo = tlx.require_layout(q_lo, q_layout)
            q_hi, qs_hi = core._quantize_q_half(q_reg1)
            q_hi = tlx.require_layout(q_hi, q_layout)
        if FAST_EXP:
            qs_lo = (tl.maximum(qs_lo, 9) - 9).to(tl.uint8)
            qs_hi = (tl.maximum(qs_hi, 9) - 9).to(tl.uint8)
        acc0 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        acc2 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        acc3 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        # K tile 0 is in slot 1; only slot 0's two groups may be outstanding.
        tlx.async_load_wait_group(2)
        s0, s1 = core._qk_first(q_lo, q_hi, qs_lo, qs_hi, ka[1], kb[1], ksa[1], ksb[1], qk_scale)
        s0, s1 = _mask_front(s0, s1, front)
        p0, p1, m_i, l_i = core._softmax_first(s0, s1, FAST_EXP)
        s0, s1, p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = _mfma_softmax_loop_4w(
            q_lo,
            q_hi,
            qs_lo,
            qs_hi,
            ka,
            kb,
            ksa,
            ksb,
            va,
            vb,
            vsa,
            vsb,
            s0,
            s1,
            p0,
            p1,
            m_i,
            l_i,
            acc0,
            acc1,
            acc2,
            acc3,
            n_tiles,
            k_ptr,
            v_ptr,
            ks_ptr,
            vs_ptr,
            c_qk,
            qk_scale,
            FAST_EXP=FAST_EXP,
        )
        # The last stage: QK of the last tile, masked past its tail valid keys,
        # with PV of the tile before it. V's last tile goes to slot 1.
        k_final_a, k_final_b = ka[0], kb[0]
        ks_final_a, ks_final_b = ksa[0], ksb[0]
        s0, s1, acc0, acc1, acc2, acc3, l_i = core._mfma_stage(
            q_lo,
            q_hi,
            qs_lo,
            qs_hi,
            k_final_a,
            k_final_b,
            ks_final_a,
            ks_final_b,
            va[0],
            vb[0],
            vsa[0],
            vsb[0],
            p0,
            p1,
            acc0,
            acc1,
            acc2,
            acc3,
            l_i,
            (OFF - m_i) * c_qk,
            qk_scale,
            k_ptr,
            v_ptr,
            ks_ptr,
            vs_ptr,
            # No K refill: the item has no further QK tile, and the next
            # item's first K tile is issued into these buffers below.
            None,
            None,
            va[1],
            vb[1],
            None,
            None,
            vsa[1],
            vsb[1],
            0,
            n_tiles - 1,
            k_step,
            v_step,
            ks_step,
            k_n1,
            v_n1,
            ks_n1,
            tail,
        )
        # The next item's opening copies, in the same three groups as the
        # first item's; after the last item they repeat it and go unread. Slot
        # 1's K buffers have no reader left in this item, and the last stage
        # has issued all of this item's reads of slot 0.
        nxt = tl.where(w + 1 < n_mine, item + n_ctas, item)
        if USE_WORK_ORDER:
            next_slot = base + tl.where(w + 1 < n_mine, w + 1, w) * n_ctas
            nxt = tlx.assume_uniform(tl.load(WorkOrder + next_slot))
        n_head, n_q0, n_q_rows, n_front, n_tail, n_pad_start, n_tiles_next = _item_uniform(
            nxt, ItemTile, CuQTile, CuQ, CuKV, CuPad, HEADS, BLOCK_M)
        nk_ptr, nv_ptr, nks_ptr, nvs_ptr = _kv_pointers(K8, V8, KS, VS, n_head, n_pad_start, NPAD, k_off, v_off, ks_off)
        tlx.async_load(nk_ptr, ka[1])
        tlx.async_load(nk_ptr + k_n1, kb[1])
        tlx.async_load(nks_ptr, ksa[1])
        tlx.async_load(nks_ptr + ks_n1, ksb[1])
        tlx.async_load_commit_group()
        core._issue_tile(
            nk_ptr,
            nv_ptr,
            nks_ptr,
            nvs_ptr,
            ka[0],
            kb[0],
            va[0],
            vb[0],
            ksa[0],
            ksb[0],
            vsa[0],
            vsb[0],
            1,
            0,
            k_step,
            v_step,
            ks_step,
            k_n1,
            v_n1,
            ks_n1,
        )
        # V's last tile and the next item's K tile 0 are in slot 1.
        tlx.async_load_wait_group(2)
        p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = core._softmax(s0, s1, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale,
                                                                 FAST_EXP)
        acc0, acc1, acc2, acc3, l_i = core._pv_last(p0, p1, va[1], vb[1], vsa[1], vsb[1], acc0, acc1, acc2, acc3, l_i)
        row_layout: tl.constexpr = tlx.slice_layout(mfma, 1)
        total = tlx.require_layout(tl.sum(l_i, 1), row_layout)
        scale = (1.0 / total)[:, None]
        valid = offs_mm < q_rows
        # P is 2**(x - m_i + 4) with either exponential.
        lse = (tlx.require_layout(m_i, row_layout) - 4.0 + tl.log2(total)) * 0.6931471805599453
        tl.store(Lse + head * TOTAL_Q + q0 + offs_mm, lse, mask=valid)
        offs_c = tl.arange(0, 32)
        o_rows = Out + (q0.to(tl.int64) + offs_mm[:, None]) * ROW + head * D
        o_mask = valid[:, None]
        tl.store(
            o_rows + offs_c[None, :],
            (acc0 * scale).to(tl.bfloat16),
            mask=o_mask,
        )
        tl.store(
            o_rows + offs_c[None, :] + 32,
            (acc1 * scale).to(tl.bfloat16),
            mask=o_mask,
        )
        tl.store(
            o_rows + offs_c[None, :] + 64,
            (acc2 * scale).to(tl.bfloat16),
            mask=o_mask,
        )
        tl.store(
            o_rows + offs_c[None, :] + 96,
            (acc3 * scale).to(tl.bfloat16),
            mask=o_mask,
        )
        if REUSE_METADATA and USE_WORK_ORDER:
            item = nxt
            head, q0, q_rows = n_head, n_q0, n_q_rows
            front, tail, pad_start, n_tiles = (
                n_front,
                n_tail,
                n_pad_start,
                n_tiles_next,
            )
            k_ptr, v_ptr, ks_ptr, vs_ptr = nk_ptr, nv_ptr, nks_ptr, nvs_ptr
    # The copies issued after the last item must land before the program ends.
    tlx.async_load_wait_group(0)


@triton.jit
# Triton TR001: BLOCK_M matches the host-selected attention tile.
def _offsets(  # noqa: TR001
        CuQ,
        CuKV,
        CuPad,
        CuQTile,
        BATCH,
        BLOCK_M: tl.constexpr,
        PAD: tl.constexpr,
        BLOCK: tl.constexpr,
):
    """Prefix sums of padded key counts and of query tiles, in one program."""
    tl.store(CuPad, 0)
    tl.store(CuQTile, 0)
    pad_carry = 0
    tile_carry = 0
    for start in range(0, BATCH, BLOCK):
        sequence = start + tl.arange(0, BLOCK)
        valid = sequence < BATCH
        kv = tl.load(CuKV + sequence + 1, valid, 0) - tl.load(CuKV + sequence, valid, 0)
        q = tl.load(CuQ + sequence + 1, valid, 0) - tl.load(CuQ + sequence, valid, 0)
        padded = (kv + PAD - 1) // PAD * PAD
        tiles = tl.where(kv > 0, (q + BLOCK_M - 1) // BLOCK_M, 0)
        tl.store(CuPad + sequence + 1, tl.cumsum(padded, 0) + pad_carry, valid)
        tl.store(CuQTile + sequence + 1, tl.cumsum(tiles, 0) + tile_carry, valid)
        pad_carry += tl.sum(padded, 0)
        tile_carry += tl.sum(tiles, 0)


@triton.jit
def _owners(CuPad, CuQTile, TileSequence, ItemTile, TILE_CAPACITY, BATCH, BLOCK: tl.constexpr):
    """Sequence of every 64-key preparation tile and query tile;
    program BATCH marks the unused preparation tiles."""
    sequence = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    owner = tl.full((BLOCK, ), 0, tl.int32) + sequence
    first = tl.load(CuPad + sequence) // 64
    last = tl.where(sequence < BATCH, tl.load(CuPad + sequence + 1) // 64, TILE_CAPACITY)
    for start in range(first, last, BLOCK):
        tl.store(TileSequence + start + lanes, owner, start + lanes < last)
    if sequence < BATCH:
        last = tl.load(CuQTile + sequence + 1)
        for start in range(tl.load(CuQTile + sequence), last, BLOCK):
            tl.store(ItemTile + start + lanes, owner, start + lanes < last)


@triton.jit(do_not_specialize=["TOTAL_Q"])
def _fill_empty(Out, Lse, CuQ, CuKV, TOTAL_Q, HEADS: tl.constexpr, ROWS: tl.constexpr):
    """Zero output and -inf LSE for a sequence with queries but no keys."""
    sequence = tl.program_id(0)
    if tl.load(CuKV + sequence + 1) == tl.load(CuKV + sequence):
        lo = tl.load(CuQ + sequence)
        hi = tl.load(CuQ + sequence + 1)
        rows = tl.arange(0, ROWS)
        cols = tl.arange(0, HEADS * 128)
        zeros = tl.zeros((ROWS, HEADS * 128), tl.bfloat16)
        for start in range(lo, hi, ROWS):
            row = start + rows
            tl.store(
                Out + row.to(tl.int64)[:, None] * (HEADS * 128) + cols[None, :],
                zeros,
                mask=(row < hi)[:, None],
            )
            for head in tl.static_range(HEADS):
                tl.store(Lse + head * TOTAL_Q + row, float("-inf"), mask=row < hi)


# Metadata per (cu_seqlens_q, cu_seqlens_k, sizes): a model's layers reuse
# the same sequence offsets, so each set is built once. Entries hold weak
# references and are checked against tensor identity and version.
_METADATA: dict[tuple[Any, ...], tuple[Any, Any, dict[str, Any]]] = {}


def _metadata(cu_seqlens_q, cu_seqlens_k, total_q, total_kv, block_m, pad=_PAD):
    """Device tables: padded key offsets, prep tiles and query-tile items."""
    key = (
        id(cu_seqlens_q),
        cu_seqlens_q._version,
        id(cu_seqlens_k),
        cu_seqlens_k._version,
        total_q,
        total_kv,
        block_m,
        pad,
    )
    hit = _METADATA.get(key)
    if hit is not None and hit[0]() is cu_seqlens_q and hit[1]() is cu_seqlens_k:
        return hit[2]
    meta = _build_metadata(cu_seqlens_q, cu_seqlens_k, total_q, total_kv, block_m, pad)
    if len(_METADATA) >= 64:
        _METADATA.pop(next(iter(_METADATA)))
    _METADATA[key] = (weakref.ref(cu_seqlens_q), weakref.ref(cu_seqlens_k), meta)
    return meta


def _build_metadata(cu_seqlens_q, cu_seqlens_k, total_q, total_kv, block_m, pad):
    """Build _metadata's tables on the device, without reading the offsets on
    the host: the table sizes are upper bounds from the totals."""
    batch = cu_seqlens_k.numel() - 1
    device = cu_seqlens_k.device
    tile_capacity = total_kv // _PREP_ROWS + batch * pad // _PREP_ROWS
    item_capacity = total_q // block_m + batch
    cu_pad = torch.empty(batch + 1, device=device, dtype=torch.int32)
    cu_q_tile = torch.empty_like(cu_pad)
    tile_sequence = torch.empty(tile_capacity, device=device, dtype=torch.int32)
    item_tile = torch.empty(item_capacity, device=device, dtype=torch.int32)
    cast(Any, _offsets)[(1, )](
        cu_seqlens_q,
        cu_seqlens_k,
        cu_pad,
        cu_q_tile,
        batch,
        BLOCK_M=block_m,
        PAD=pad,
        BLOCK=1024,
        num_warps=4,
    )
    cast(Any, _owners)[(batch + 1, )](
        cu_pad,
        cu_q_tile,
        tile_sequence,
        item_tile,
        tile_capacity,
        batch,
        BLOCK=256,
        num_warps=1,
    )
    return {
        "cu_pad": cu_pad,
        "tile_sequence": tile_sequence,
        "tile_capacity": tile_capacity,
        "cu_q_tile": cu_q_tile,
        "item_tile": item_tile,
        "item_capacity": item_capacity,
        # An upper bound of the padded key count.
        "npad": triton.cdiv(total_kv + (pad - 1) * batch, pad) * pad,
    }


def prepare_kv(k, v, cu_seqlens_k, meta=None):
    """Quantize packed K and V into the core's padded head-major layouts.

    Returns ``(k8, k_scales, v8, v_scales)``, the ``kv_state`` that ``forward``
    accepts. ``meta`` is optional for callers preparing a reusable K/V state
    directly.
    """
    if meta is None:
        dummy_cu_q = torch.zeros_like(cu_seqlens_k)
        meta = _build_metadata(dummy_cu_q, cu_seqlens_k, 0, k.shape[0], _BLOCK_M, _PAD)
    npad = meta["npad"]
    device = k.device
    k8 = torch.empty((_HEADS, npad, _HEAD_DIM), device=device, dtype=torch.float8_e4m3fn)
    # V transposed per 64-key block: [head, block, feature, key].
    v8 = torch.empty((_HEADS, npad // 64, _HEAD_DIM, 64), device=device, dtype=torch.float8_e4m3fn)
    ks = torch.empty((_HEADS, npad, 4), device=device, dtype=torch.uint8)
    vs = torch.empty_like(ks)
    cast(Any, _prepare_kv)[(meta["tile_capacity"], _HEADS)](
        k,
        v,
        k8,
        ks,
        v8,
        vs,
        cu_seqlens_k,
        meta["cu_pad"],
        meta["tile_sequence"],
        meta["tile_capacity"],
        cu_seqlens_k.numel() - 1,
        npad,
        HEADS=_HEADS,
        num_warps=1,
        num_stages=1,
        llvm_fn_attrs=(("amdgpu-waves-per-eu", "2,2"), ),
    )
    return k8, ks, v8, vs


def _check_offsets(offsets, device):
    """Require contiguous int32 cumulative offsets on the input device."""
    if (offsets.ndim != 1 or offsets.numel() < 2 or offsets.dtype != torch.int32 or not offsets.is_contiguous()
            or offsets.device != device):
        raise ValueError("Requires contiguous int32 sequence offsets on the input device")


@dataclass(frozen=True)
class ForwardWorkSchedule:
    """Assignment of work items to the persistent grid.

    ``item_order[base + n * num_ctas]`` is the n-th work item of the program
    with slot ``base``. ``total_q``, ``total_kv`` and ``batch`` identify the
    lengths the schedule was built for. The offset tensors and their versions
    tie the schedule to the exact per-sequence lengths, not just their totals.
    ``needs_empty_fill`` is set when a sequence has queries but no keys; such
    sequences have no work items and get their rows from a separate fill.
    ``item_count`` is the number of work items: the kernel follows
    ``item_order`` only when it covers every item, and otherwise takes the
    items in index order."""

    item_order: torch.Tensor
    num_ctas: int
    total_q: int
    total_kv: int
    batch: int
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q_version: int
    cu_seqlens_k_version: int
    needs_empty_fill: bool = True
    item_count: int | None = None


def prepare_work_schedule(
    query_lengths,
    key_lengths,
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    num_ctas=512,
):
    """Build a reusable KV-cost-balanced persistent work schedule.

    Every program gets a fixed share of the work items. The items are taken
    longest key span first, each going to the least loaded program that still
    has room. The lengths are host lists, so building a schedule does not read
    device memory. The schedule is valid only with the immutable offset tensors
    passed here."""
    if num_ctas < 8 or num_ctas % 8:
        raise ValueError("Requires a multiple of eight CTAs")
    if not isinstance(query_lengths, (list, tuple)) or not isinstance(key_lengths, (list, tuple)):
        raise ValueError("Requires CPU lists or tuples of sequence lengths")
    if len(query_lengths) != len(key_lengths) or not query_lengths:
        raise ValueError("Query and key lengths must have the same nonzero batch size")
    if any(not isinstance(n, int) or n < 0 for n in (*query_lengths, *key_lengths)):
        raise ValueError("Sequence lengths must be nonnegative integers")
    _check_offsets(cu_seqlens_q, cu_seqlens_q.device)
    _check_offsets(cu_seqlens_k, cu_seqlens_q.device)
    if cu_seqlens_q.shape != cu_seqlens_k.shape or cu_seqlens_q.numel() != len(query_lengths) + 1:
        raise ValueError("Offset tensors must match the supplied sequence lengths")
    costs = []
    for q_length, k_length in zip(query_lengths, key_lengths):
        if k_length:
            costs.extend([triton.cdiv(k_length, _PAD)] * (triton.cdiv(q_length, _BLOCK_M) * _HEADS))
    count = len(costs)
    capacities = [triton.cdiv(count - base, num_ctas) for base in range(min(num_ctas, count))]
    bins = [(0, base, 0) for base in range(len(capacities))]
    heapq.heapify(bins)
    order = [0] * count
    for item in sorted(range(count), key=lambda index: (-costs[index], index)):
        load, base, assigned = heapq.heappop(bins)
        order[base + assigned * num_ctas] = item
        assigned += 1
        if assigned < capacities[base]:
            heapq.heappush(bins, (load + costs[item], base, assigned))
    return ForwardWorkSchedule(
        torch.tensor(order, dtype=torch.int32, device=cu_seqlens_q.device),
        num_ctas,
        sum(query_lengths),
        sum(key_lengths),
        len(query_lengths),
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_q._version,
        cu_seqlens_k._version,
        any(q_length > 0 and k_length == 0 for q_length, k_length in zip(query_lengths, key_lengths)),
        count,
    )


def _check_work_schedule(schedule, q, total_kv, batch, num_ctas, cu_seqlens_q, cu_seqlens_k):
    """Reject a schedule not tied to these exact immutable offsets."""
    if schedule is None:
        return
    if (schedule.total_q != q.shape[0] or schedule.total_kv != total_kv or schedule.batch != batch
            or (num_ctas is not None and num_ctas != schedule.num_ctas) or schedule.item_order.device != q.device
            or schedule.cu_seqlens_q is not cu_seqlens_q or schedule.cu_seqlens_k is not cu_seqlens_k
            or schedule.cu_seqlens_q_version != cu_seqlens_q._version
            or schedule.cu_seqlens_k_version != cu_seqlens_k._version):
        raise ValueError("Work schedule must match the exact input offsets, device, and grid")


_AUTOMATIC_SCHEDULES: dict[tuple[Any, ...], tuple[Any, Any, ForwardWorkSchedule]] = {}


def _automatic_work_schedule(cu_seqlens_q, cu_seqlens_k, device, *, num_ctas):
    """Create the balanced schedule used by the public forward paths.

    Reading the lengths synchronizes with the device, so schedules are cached
    like _metadata's tables. With num_ctas None the grid is one program per
    work item, rounded up to a multiple of eight and capped at 512."""
    key = (
        id(cu_seqlens_q),
        cu_seqlens_q._version,
        id(cu_seqlens_k),
        cu_seqlens_k._version,
        num_ctas,
    )
    hit = _AUTOMATIC_SCHEDULES.get(key)
    if hit is not None and hit[0]() is cu_seqlens_q and hit[1]() is cu_seqlens_k:
        return hit[2]
    query_lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).detach().cpu().tolist()
    key_lengths = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).detach().cpu().tolist()
    if num_ctas is None:
        work_items = sum(triton.cdiv(q, _BLOCK_M) * _HEADS for q, k in zip(query_lengths, key_lengths) if k)
        num_ctas = min(512, max(8, triton.cdiv(work_items, 8) * 8))
    schedule = prepare_work_schedule(
        query_lengths,
        key_lengths,
        cu_seqlens_q,
        cu_seqlens_k,
        num_ctas=num_ctas,
    )
    if len(_AUTOMATIC_SCHEDULES) >= 64:
        _AUTOMATIC_SCHEDULES.pop(next(iter(_AUTOMATIC_SCHEDULES)))
    _AUTOMATIC_SCHEDULES[key] = (weakref.ref(cu_seqlens_q), weakref.ref(cu_seqlens_k), schedule)
    return schedule


@triton.jit
def _prepare_q_forward(Q, Q8, QS, CuQ, HEADS: tl.constexpr):
    """Quantize one sequence and head of Q, 64 rows at a time: E4M3 in Q8 at
    Q's offsets, and in QS one E8M0 byte per row and 32 features, shared by
    the rows of a 32-row block counted from the sequence start."""
    sequence = tl.program_id(0)
    head = tl.program_id(1)
    first = tl.load(CuQ + sequence)
    last = tl.load(CuQ + sequence + 1)
    rows = tl.arange(0, 64)
    columns = tl.arange(0, 128)
    for start in range(first, last, 64):
        row = start + rows
        base = start.to(tl.int64) * (HEADS * 128) + head * 128
        offsets = (rows[:, None] * (HEADS * 128) + columns[None, :])
        values = tlx.buffer_load(
            Q + base,
            offsets,
            (row < last)[:, None],
            0.0,
            contiguity=8,
        )
        magnitude = ((values.to(tl.uint16, bitcast=True) & 0x7FFF).reshape((2, 32, 4, 32)))
        maximum_bits = tl.max(tl.max(magnitude, 3), 1)
        maximum = (maximum_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32))
        byte = core._mx_scale(maximum)
        scale = (byte << 23).to(tl.float32, bitcast=True)
        scale = tl.broadcast_to(scale[:, None, :, None], (2, 32, 4, 32)).reshape((64, 128))
        tlx.buffer_store(
            core._to_e4m3(values, scale),
            Q8 + base,
            offsets,
            (row < last)[:, None],
        )
        scale_offsets = (row.to(tl.int64)[:, None] * (HEADS * 4) + head * 4 + tl.arange(0, 4)[None, :])
        scales = tl.broadcast_to(byte[:, None, :], (2, 32, 4)).reshape((64, 4)).to(tl.uint8)
        tl.store(QS + scale_offsets, scales, (row < last)[:, None])


def prepare_q(q, cu_seqlens_q):
    """Prepare packed E4M3 Q and row-wise E8M0 scales.

    Returns ``(q8, q_scales)`` for ``forward_prequantized``: E4M3
    ``[tokens, 4, 128]`` and uint8 ``[tokens, 4, 4]``."""
    if (q.ndim != 3 or q.shape[1:] != (_HEADS, _HEAD_DIM) or q.dtype != torch.bfloat16 or not q.is_contiguous()):
        raise ValueError("Requires contiguous BF16 [tokens, 4, 128] Q")
    _check_offsets(cu_seqlens_q, q.device)
    q8 = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    scales = torch.empty((q.shape[0], _HEADS, 4), dtype=torch.uint8, device=q.device)
    cast(Any, _prepare_q_forward)[(cu_seqlens_q.numel() - 1, _HEADS)](
        q,
        q8,
        scales,
        cu_seqlens_q,
        HEADS=_HEADS,
        num_warps=1,
        num_stages=1,
    )
    return q8, scales


def forward_prequantized(
    q8,
    q_scales,
    k8,
    k_scales,
    v8,
    v_scales,
    cu_seqlens_q,
    cu_seqlens_k,
    sm_scale,
    *,
    total_kv,
    fast_exp=None,
    num_ctas=None,
    work_schedule=None,
):
    """Return output and natural-log LSE from prepared MXFP8 inputs.

    ``q8`` and ``q_scales`` are ``prepare_q``'s outputs, the K/V payloads and
    scales ``prepare_kv``'s, and ``total_kv`` is the number of key rows before
    padding. ``fast_exp`` is as in ``forward``."""
    if (q8.ndim != 3 or q8.shape[1:] != (_HEADS, _HEAD_DIM) or q8.dtype != torch.float8_e4m3fn
            or not q8.is_contiguous()):
        raise ValueError("Requires contiguous E4M3 [tokens, 4, 128] Q")
    device = q8.device
    _check_offsets(cu_seqlens_q, device)
    _check_offsets(cu_seqlens_k, device)
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("Query and key sequence offsets must have the same shape")
    if (not isinstance(total_kv, int) or total_kv < 0 or not math.isfinite(sm_scale)):
        raise ValueError("total_kv must be nonnegative and sm_scale must be finite")
    if num_ctas is not None and (num_ctas < 8 or num_ctas % 8):
        raise ValueError("num_ctas must be a positive multiple of eight")
    batch = cu_seqlens_k.numel() - 1
    if work_schedule is None:
        work_schedule = _automatic_work_schedule(
            cu_seqlens_q,
            cu_seqlens_k,
            device,
            num_ctas=num_ctas,
        )
    _check_work_schedule(
        work_schedule,
        q8,
        total_kv,
        batch,
        num_ctas,
        cu_seqlens_q,
        cu_seqlens_k,
    )
    meta = _metadata(cu_seqlens_q, cu_seqlens_k, q8.shape[0], total_kv, _BLOCK_M)
    npad = meta["npad"]
    tensors = (
        (q_scales, (q8.shape[0], _HEADS, 4), torch.uint8),
        (k8, (_HEADS, npad, _HEAD_DIM), torch.float8_e4m3fn),
        (
            v8,
            (_HEADS, npad // 64, _HEAD_DIM, 64),
            torch.float8_e4m3fn,
        ),
        (k_scales, (_HEADS, npad, 4), torch.uint8),
        (v_scales, (_HEADS, npad, 4), torch.uint8),
    )
    for tensor, shape, dtype in tensors:
        if (tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous() or tensor.device != device):
            raise ValueError("Prepared payloads and scales must match the documented "
                             "shapes, dtypes, and device")
    return _launch_extended_forward(
        q8,
        q_scales,
        k8,
        k_scales,
        v8,
        v_scales,
        cu_seqlens_q,
        cu_seqlens_k,
        sm_scale,
        meta,
        fast_exp,
        True,
        work_schedule,
    )


def _check(q, k, v, sm_scale, kv_contiguous=True):
    """Validate forward's BF16 inputs; K and V may be non-contiguous when a
    prepared kv_state stands in for their data."""
    for t in (q, k, v):
        if (t.ndim != 3 or t.shape[1:] != (_HEADS, _HEAD_DIM) or t.dtype != torch.bfloat16
                or not (t.is_contiguous() or (t is not q and not kv_contiguous))):
            raise ValueError("Requires contiguous BF16 [tokens, 4, 128]")
    if k.shape != v.shape or not math.isfinite(sm_scale):
        raise ValueError("K and V must match and sm_scale must be finite")


def forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    sm_scale,
    *,
    fast_exp=None,
    kv_state=None,
    num_ctas=None,
    work_schedule=None,
):
    """Return (out, lse, kv_state); lse is natural-log FP32 [4, total_q].

    Query tiles have 128 rows on four warps. ``fast_exp`` selects the
    approximate exponential; left as None, it is used when the sequences
    average at least 4096 keys. Sequences with no keys get zero
    output rows and -inf LSE. The sequence-offset tensors must not be
    modified in place while their prepared metadata and work schedule are
    reused."""
    # With a given kv_state, K and V only supply their shape.
    _check(q, k, v, sm_scale, kv_contiguous=kv_state is None)
    batch = cu_seqlens_k.numel() - 1
    if work_schedule is None:
        work_schedule = _automatic_work_schedule(
            cu_seqlens_q,
            cu_seqlens_k,
            q.device,
            num_ctas=num_ctas,
        )
    _check_work_schedule(
        work_schedule,
        q,
        k.shape[0],
        batch,
        num_ctas,
        cu_seqlens_q,
        cu_seqlens_k,
    )
    meta = _metadata(cu_seqlens_q, cu_seqlens_k, q.shape[0], k.shape[0], _BLOCK_M)
    if kv_state is None:
        kv_state = prepare_kv(k, v, cu_seqlens_k, meta)
    k8, ks, v8, vs = kv_state[:4]
    out, lse = _launch_extended_forward(
        q,
        None,
        k8,
        ks,
        v8,
        vs,
        cu_seqlens_q,
        cu_seqlens_k,
        sm_scale,
        meta,
        fast_exp,
        False,
        work_schedule,
    )
    return out, lse, kv_state


def flash_attn_varlen_mxfp8(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    sm_scale,
    *,
    fast_exp=None,
):
    """Catalog entry point of ``tlx.ops.flash_attn_varlen_mxfp8``: ``forward``'s
    output alone. The maximum lengths are unused."""
    del max_seqlen_k
    del max_seqlen_q
    out, _, _ = forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        sm_scale,
        fast_exp=fast_exp,
    )
    return out


def _launch_extended_forward(
    q,
    q_scales,
    k8,
    ks,
    v8,
    vs,
    cu_seqlens_q,
    cu_seqlens_k,
    sm_scale,
    meta,
    fast_exp,
    prequantized,
    work_schedule,
):
    """Launch the attention kernel, then the fill for sequences with queries
    but no keys. Returns (out, lse). ``q`` is BF16, or E4M3 with ``q_scales``
    when ``prequantized``."""
    batch = cu_seqlens_k.numel() - 1
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((_HEADS, q.shape[0]), dtype=torch.float32, device=q.device)
    grid = work_schedule.num_ctas
    long_kv = work_schedule.total_kv >= _LONG_KV * batch
    if fast_exp is None:
        fast_exp = long_kv
    kernel = cast(Any, _mxfp8_varlen_extended_kernel_4)
    use_work_order = (work_schedule.item_count is not None
                      and work_schedule.item_count == work_schedule.item_order.numel())
    kernel[(grid, )](
        q,
        q_scales if q_scales is not None else q,
        work_schedule.item_order,
        k8,
        v8,
        ks,
        vs,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        meta["cu_pad"],
        meta["cu_q_tile"],
        meta["item_tile"],
        batch,
        q.shape[0],
        meta["npad"],
        sm_scale * _LOG2E,
        HEADS=_HEADS,
        BLOCK_M=_BLOCK_M,
        FAST_EXP=fast_exp,
        PREQUANTIZED_Q=prequantized,
        REUSE_METADATA=not long_kv,
        USE_WORK_ORDER=use_work_order,
        **_LAUNCH_OPTIONS,
    )
    if work_schedule.needs_empty_fill:
        cast(Any, _fill_empty)[(batch, )](
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            q.shape[0],
            HEADS=_HEADS,
            ROWS=8,
            num_warps=4,
        )
    return out, lse
