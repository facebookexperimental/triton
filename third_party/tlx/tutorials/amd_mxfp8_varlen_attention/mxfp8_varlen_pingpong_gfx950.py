# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Packed-varlen MXFP8 attention forward on the D122867768 ping-pong core.

Packed [tokens, 4, 128] BF16 Q/K/V, non-causal. One preparation kernel
quantizes K and V into head-major buffers in which every sequence starts at
a multiple of 256 keys and is zero-padded to one, in the layouts the core
reads (K row-major with permuted scale words, V transposed with keys
reordered inside 32-key groups). Q is quantized in registers. Work items are
(256-row query tile, head) pairs, walked by a persistent grid; keys past the
sequence end get -1e30 scores, which only the last two 128-key tiles need.
"""

from __future__ import annotations

import math
import weakref
from typing import Any, cast

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from . import mxfp8_attention_core_gfx950 as core

_HEADS = 4
_HEAD_DIM = 128
_PAD = 256
_PREP_ROWS = 64
_LOG2E = 1.4426950408889634

_LAUNCH_OPTIONS = {
    "MFMA_PRIO": 1,
    "SOFTMAX_PRIO": 0,
    "num_warps": 8,
    "num_stages": 1,
    "llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"), ("amdgpu-ieee", "false")),
    "enable_sched_group_barrier_scheduler": False,
    "sched_group_barrier_mfma_per_dwordx4": 4,
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
    magnitude = (
        (x.to(tl.uint16, bitcast=True) & 0x7FFF)
        .to(tl.uint16)
        .reshape((BLOCK // 32, 32, 4, 32))
    )
    maximum_bits = tl.max(tl.max(magnitude, 3), 1)
    maximum = maximum_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    byte = core._mx_scale(maximum)
    operand_scale = (byte << 23).to(tl.float32, bitcast=True)
    operand_scale = tl.broadcast_to(
        operand_scale[:, None, :, None], (BLOCK // 32, 32, 4, 32)
    ).reshape((BLOCK, 128))
    tlx.buffer_store(
        core._to_e4m3(x, operand_scale),
        K8 + dst * 128,
        rows[:, None] * 128 + columns[None, :],
    )
    scales = tl.broadcast_to(byte[:, None, :], (BLOCK // 32, 32, 4)).to(tl.uint8)
    scales = (
        scales.reshape((BLOCK // 64, 2, 32, 2, 2))
        .permute(0, 4, 2, 3, 1)
        .reshape((BLOCK, 4))
    )
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
    v_bits = (
        (v.to(tl.uint16, bitcast=True) & 0x7FFF)
        .to(tl.uint16)
        .reshape((BLOCK // 32, 32, 128))
    )
    v_max = tl.max(v_bits, 1).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    v_byte = core._mx_scale(v_max)
    v_operand_scale = (v_byte << 23).to(tl.float32, bitcast=True)
    v_operand_scale = tl.broadcast_to(
        tl.trans(v_operand_scale)[:, :, None], (128, BLOCK // 32, 32)
    ).reshape((128, BLOCK))
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
    v_scales = (
        v_byte.to(tl.uint8)
        .reshape((BLOCK // 32, 4, 32))
        .permute(0, 2, 1)
        .reshape((BLOCK, 4))
    )
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
    VR8,
    VRS,
    CuKV,
    CuPad,
    TileSequence,
    CAPACITY,
    BATCH,
    NPAD,
    HEADS: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    """One padded 64-key tile of one sequence and head; with BACKWARD, also V
    in the K layout for the MXFP8 backward."""
    tile = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    if BACKWARD and tile >= CAPACITY:
        # Four extra programs zero the spare last 256 keys, which serve the
        # backward's work items without keys.
        dst = head * NPAD + NPAD - 256 + (tile - CAPACITY) * 64
        _rows_tile(K, K8, KS, 0, dst, 0, 0, HEADS)
        _rows_tile(V, VR8, VRS, 0, dst, 0, 0, HEADS)
        return
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
    if BACKWARD:
        _rows_tile(V, VR8, VRS, src, dst, -key, length - key, HEADS)


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
def _mfma_softmax_loop_8w(
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
    MFMA_PRIO: tl.constexpr,
    SOFTMAX_PRIO: tl.constexpr,
    FAST_EXP: tl.constexpr,
):
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
        with tlx.warp_pipeline_stage("mfma", priority=MFMA_PRIO):
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
                True,
                True,
                True,
            )
        tlx.async_load_wait_group(0)
        with tlx.warp_pipeline_stage("softmax", priority=SOFTMAX_PRIO):
            p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = core._pingpong_softmax(
                s0, s1, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale, FAST_EXP
            )
        with tlx.warp_pipeline_stage("mfma", priority=MFMA_PRIO):
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
                True,
                True,
                True,
            )
        tlx.async_load_wait_group(0)
        with tlx.warp_pipeline_stage("softmax", priority=SOFTMAX_PRIO):
            p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = core._pingpong_softmax(
                s0, s1, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale, FAST_EXP
            )
    return s0, s1, p0, p1, m_i, l_i, acc0, acc1, acc2, acc3


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
            True,
            True,
            False,
        )
        p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = core._pingpong_softmax(
            s0, s1, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale, FAST_EXP
        )
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
            True,
            True,
            False,
        )
        p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = core._pingpong_softmax(
            s0, s1, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale, FAST_EXP
        )
    return s0, s1, p0, p1, m_i, l_i, acc0, acc1, acc2, acc3


@triton.jit
# Triton TR001: BLOCK_M is a host-selected production contract.
def _mxfp8_varlen_kernel(  # noqa: TR001
    Q,
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
    MFMA_PRIO: tl.constexpr,
    SOFTMAX_PRIO: tl.constexpr,
    FAST_EXP: tl.constexpr,
):
    HN: tl.constexpr = 64
    HD: tl.constexpr = 64
    D: tl.constexpr = 128
    ROW: tl.constexpr = HEADS * D
    pid = tl.program_id(0)
    n_ctas = tl.num_programs(0)
    n_items = tl.load(CuQTile + BATCH) * HEADS
    # Workgroups go round-robin to the 8 XCDs: give each XCD a contiguous run
    # of items so the query tiles sharing a K/V stream share its L2.
    base = pid % 8 * (n_ctas // 8) + pid // 8
    if base >= n_items:
        return
    n_mine = (n_items - base + n_ctas - 1) // n_ctas
    offs_d = tl.arange(0, D)
    offs_hd = tl.arange(0, HD)
    offs_n = tl.arange(0, HN)
    offs_mm = tl.arange(0, BLOCK_M)
    C: tl.constexpr = core._score_scale(FAST_EXP)
    OFF: tl.constexpr = core._score_offset(FAST_EXP)
    c_qk = C / qk_scale
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
    mfma: tl.constexpr = tlx.amd_mfma_layout(
        4, [32, 32, 64], True, [tlx.num_warps(), 1]
    )
    q_layout: tl.constexpr = tlx.dot_operand_layout(0, mfma, k_width=16)
    q_local = tlx.require_layout(q_off.to(tl.int32), q_layout)
    k_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(1024, 16)], core._K_LDS_BASES, [D, HN]
    )
    v_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(1024, 16)], core._V_LDS_BASES, [HN, D]
    )
    ka = tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout)
    kb = tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout)
    va = tlx.local_alloc((HN, D), tl.float8e4nv, 2, layout=v_layout)
    vb = tlx.local_alloc((HN, D), tl.float8e4nv, 2, layout=v_layout)
    ksa = tlx.local_alloc((HN, 1), tl.int32, 2)
    ksb = tlx.local_alloc((HN, 1), tl.int32, 2)
    vsa = tlx.local_alloc((HN, 1), tl.int32, 2)
    vsb = tlx.local_alloc((HN, 1), tl.int32, 2)

    head, q0, q_rows, front, tail, pad_start, n_tiles = _item(
        base, ItemTile, CuQTile, CuQ, CuKV, CuPad, HEADS, BLOCK_M
    )
    k_ptr, v_ptr, ks_ptr, vs_ptr = _kv_pointers(
        K8, V8, KS, VS, head, pad_start, NPAD, k_off, v_off, ks_off
    )
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
        True,
        True,
    )
    for w in tl.range(0, n_mine, num_stages=1):
        item = base + w * n_ctas
        head, q0, q_rows, front, tail, pad_start, n_tiles = _item(
            item, ItemTile, CuQTile, CuQ, CuKV, CuPad, HEADS, BLOCK_M
        )
        q_base = Q + q0.to(tl.int64) * ROW + head * D
        q_mask = q_local < q_rows * ROW
        q_reg0 = tlx.buffer_load(q_base, q_local, q_mask, 0.0, contiguity=8)
        q_reg1 = tlx.buffer_load(q_base + HD, q_local, q_mask, 0.0, contiguity=8)
        tlx.amd_sched_barrier(0)
        k_ptr, v_ptr, ks_ptr, vs_ptr = _kv_pointers(
            K8, V8, KS, VS, head, pad_start, NPAD, k_off, v_off, ks_off
        )
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
        tlx.async_load_wait_group(2)
        s0, s1 = core._qk_first(
            q_lo, q_hi, qs_lo, qs_hi, ka[1], kb[1], ksa[1], ksb[1], qk_scale
        )
        s0, s1 = _mask_front(s0, s1, front)
        p0, p1, m_i, l_i = core._pingpong_softmax_first(s0, s1, FAST_EXP)
        warps: tl.constexpr = tlx.num_warps()
        if warps == 8:
            s0, s1, p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = _mfma_softmax_loop_8w(
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
                MFMA_PRIO=MFMA_PRIO,
                SOFTMAX_PRIO=SOFTMAX_PRIO,
                FAST_EXP=FAST_EXP,
            )
        else:
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
            0,
            n_tiles - 1,
            k_step,
            v_step,
            ks_step,
            k_n1,
            v_n1,
            ks_n1,
            False,
            True,
            warps == 8,
            tail,
        )
        # Buffer 0 and buffer 1's K half are free: copy the next item's first
        # tiles behind this epilogue (the last item re-copies its own).
        nxt = tl.where(w + 1 < n_mine, item + n_ctas, item)
        n_head, _, _, _, _, n_pad_start, _ = _item(
            nxt, ItemTile, CuQTile, CuQ, CuKV, CuPad, HEADS, BLOCK_M
        )
        nk_ptr, nv_ptr, nks_ptr, nvs_ptr = _kv_pointers(
            K8, V8, KS, VS, n_head, n_pad_start, NPAD, k_off, v_off, ks_off
        )
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
            True,
            True,
        )
        tlx.async_load_wait_group(2)
        p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = core._pingpong_softmax(
            s0, s1, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale, FAST_EXP
        )
        acc0, acc1, acc2, acc3, l_i = core._pv_last(
            p0, p1, va[1], vb[1], vsa[1], vsb[1], acc0, acc1, acc2, acc3, l_i
        )
        row_layout: tl.constexpr = tlx.slice_layout(mfma, 1)
        total = tlx.require_layout(tl.sum(l_i, 1), row_layout)
        scale = (1.0 / total)[:, None]
        valid = offs_mm < q_rows
        lse = (
            tlx.require_layout(m_i, row_layout) - OFF + tl.log2(total)
        ) * 0.6931471805599453
        tl.store(Lse + head * TOTAL_Q + q0 + offs_mm, lse, mask=valid)
        offs_c = tl.arange(0, 32)
        o_rows = Out + (q0.to(tl.int64) + offs_mm[:, None]) * ROW + head * D
        o_mask = valid[:, None]
        tl.store(o_rows + offs_c[None, :], (acc0 * scale).to(tl.bfloat16), mask=o_mask)
        tl.store(
            o_rows + offs_c[None, :] + 32, (acc1 * scale).to(tl.bfloat16), mask=o_mask
        )
        tl.store(
            o_rows + offs_c[None, :] + 64, (acc2 * scale).to(tl.bfloat16), mask=o_mask
        )
        tl.store(
            o_rows + offs_c[None, :] + 96, (acc3 * scale).to(tl.bfloat16), mask=o_mask
        )
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
def _owners(
    CuPad, CuQTile, TileSequence, ItemTile, TILE_CAPACITY, BATCH, BLOCK: tl.constexpr
):
    """Sequence of every 64-key preparation tile and query tile;
    program BATCH marks the unused preparation tiles."""
    sequence = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    owner = tl.full((BLOCK,), 0, tl.int32) + sequence
    first = tl.load(CuPad + sequence) // 64
    last = tl.where(
        sequence < BATCH, tl.load(CuPad + sequence + 1) // 64, TILE_CAPACITY
    )
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


# Metadata per (cu_seqlens_q, cu_seqlens_k, sizes): the block's layers reuse
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
    batch = cu_seqlens_k.numel() - 1
    device = cu_seqlens_k.device
    tile_capacity = total_kv // _PREP_ROWS + batch * pad // _PREP_ROWS
    item_capacity = total_q // block_m + batch
    cu_pad = torch.empty(batch + 1, device=device, dtype=torch.int32)
    cu_q_tile = torch.empty_like(cu_pad)
    tile_sequence = torch.empty(tile_capacity, device=device, dtype=torch.int32)
    item_tile = torch.empty(item_capacity, device=device, dtype=torch.int32)
    cast(Any, _offsets)[(1,)](
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
    cast(Any, _owners)[(batch + 1,)](
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
        # One spare tile keeps prefetches of empty work items in bounds.
        "npad": (triton.cdiv(total_kv + (pad - 1) * batch, pad) + 1) * pad,
    }


def prepare_kv(k, v, cu_seqlens_k, meta, backward=False):
    """Quantize packed K and V into the core's padded head-major layouts.

    Returns (k8, ks, v8, vs), followed with backward by V in the K layout
    (vr8, vrs)."""
    npad = meta["npad"]
    device = k.device
    k8 = torch.empty(
        (_HEADS, npad, _HEAD_DIM), device=device, dtype=torch.float8_e4m3fn
    )
    # V transposed per 64-key block: [head, block, feature, key].
    v8 = torch.empty(
        (_HEADS, npad // 64, _HEAD_DIM, 64), device=device, dtype=torch.float8_e4m3fn
    )
    ks = torch.empty((_HEADS, npad, 4), device=device, dtype=torch.uint8)
    vs = torch.empty_like(ks)
    extra = (torch.empty_like(k8), torch.empty_like(ks)) if backward else (k8, ks)
    prepare_attrs = () if backward else (("amdgpu-waves-per-eu", "2,2"),)
    cast(Any, _prepare_kv)[(meta["tile_capacity"] + 4 * backward, _HEADS)](
        k,
        v,
        k8,
        ks,
        v8,
        vs,
        *extra,
        cu_seqlens_k,
        meta["cu_pad"],
        meta["tile_sequence"],
        meta["tile_capacity"],
        cu_seqlens_k.numel() - 1,
        npad,
        HEADS=_HEADS,
        BACKWARD=backward,
        num_warps=1,
        num_stages=1,
        llvm_fn_attrs=prepare_attrs,
    )
    return (k8, ks, v8, vs, *extra) if backward else (k8, ks, v8, vs)


def _check(q, k, v, sm_scale, kv_contiguous=True):
    for t in (q, k, v):
        if (
            t.ndim != 3
            or t.shape[1:] != (_HEADS, _HEAD_DIM)
            or t.dtype != torch.bfloat16
            or not (t.is_contiguous() or (t is not q and not kv_contiguous))
        ):
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
    fast_exp=False,
    kv_state=None,
    num_ctas=None,
    block_m=None,
    backward_state=False,
):
    """Return (out, lse, kv_state); lse is natural-log FP32 [4, total_q].

    Query tiles have 256 rows (8 warps), or 128 (4 warps) when sequences
    average under 256 queries and 4096 keys. Sequences with no keys get zero
    output rows and -inf LSE. With backward_state, kv_state also carries the
    K/V operands of tlx_mxfp8_varlen_backward_gfx950.backward."""
    # With a given kv_state, K and V only supply their shape.
    _check(q, k, v, sm_scale, kv_contiguous=kv_state is None)
    batch = cu_seqlens_k.numel() - 1
    if block_m is None:
        short = q.shape[0] < 256 * batch and k.shape[0] < 4096 * batch
        block_m = 128 if short else 256
    meta = _metadata(cu_seqlens_q, cu_seqlens_k, q.shape[0], k.shape[0], block_m)
    if kv_state is None:
        kv_state = prepare_kv(k, v, cu_seqlens_k, meta, backward=backward_state)
    k8, ks, v8, vs = kv_state[:4]
    out = torch.empty_like(q)
    lse = torch.empty((_HEADS, q.shape[0]), dtype=torch.float32, device=q.device)
    # Two 4-warp workgroups fit a CU as one 8-warp workgroup does.
    ctas = 256 * (256 // block_m)
    grid = num_ctas or min(ctas, triton.cdiv(meta["item_capacity"] * _HEADS, 8) * 8)
    cast(Any, _mxfp8_varlen_kernel)[(grid,)](
        q,
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
        BLOCK_M=block_m,
        FAST_EXP=fast_exp,
        **{**_LAUNCH_OPTIONS, "num_warps": block_m // 32},
    )
    cast(Any, _fill_empty)[(batch,)](
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        q.shape[0],
        HEADS=_HEADS,
        ROWS=8,
        num_warps=4,
    )
    return out, lse, kv_state


def emulate(q, k, v, cu_seqlens_q, cu_seqlens_k, sm_scale):
    """FP64 reference of the recipe: Q/K scaled per 32x32 tile, V per feature
    and 32 keys, P per row and 32 keys; key blocks follow the padded layout."""

    def quantize(x, rows, cols, front=0):
        n, h, d = x.shape
        g = torch.nn.functional.pad(
            x.double(), (0, 0, 0, 0, front, (-n - front) % rows)
        )
        g = g.reshape(-1, rows, h, d // cols, cols)
        amax = g.abs().amax((1, 4), keepdim=True).float()
        byte = ((amax / 448.0).view(torch.int32) + 0x7FFFFF) >> 23
        scale = torch.exp2(byte.double() - 127)
        data = (g / scale).float().to(torch.float8_e4m3fn).double() * scale
        return data.reshape(-1, h, d)[front : front + n]

    out = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for s in range(len(cq) - 1):
        qs, ks_, vs_ = q[cq[s] : cq[s + 1]], k[ck[s] : ck[s + 1]], v[ck[s] : ck[s + 1]]
        n = ks_.shape[0]
        if qs.shape[0] == 0 or n == 0:
            continue
        front = max((n + _PAD - 1) // _PAD * _PAD - n - 128, 0)
        qd = quantize(qs, 32, 32)
        kd, vd = quantize(ks_, 32, 32, front), quantize(vs_, 32, 1, front)
        scores = torch.einsum("qhd,khd->hqk", qd, kd) * sm_scale
        p = torch.exp(scores - scores.amax(-1, keepdim=True))
        pp = torch.nn.functional.pad(p, (front, (-n - front) % 32)).unflatten(
            -1, (-1, 32)
        )
        amax = pp.amax(-1, keepdim=True).float()
        byte = ((amax / 448.0).view(torch.int32) + 0x7FFFFF) >> 23
        scale = torch.exp2(byte.double() - 127)
        pq = ((pp / scale).float().to(torch.float8_e4m3fn).double() * scale).flatten(-2)
        pq = pq[..., front : front + n]
        # The kernel's denominator sums the FP8 probabilities.
        out[cq[s] : cq[s + 1]] = torch.einsum(
            "hqk,khd->qhd", pq / pq.sum(-1, keepdim=True), vd
        )
    return out
