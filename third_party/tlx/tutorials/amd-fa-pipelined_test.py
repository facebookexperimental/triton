"""
AMD Flash Attention Forward — Async DMA Kernel (CDNA4)
============================================================

Usage:
    # Defaults: -b 1 -hq 64 -sq 1024 8192 16384 -d 64 128 -causal false --kernel async_simple
    python amd-fa-pipelined_test.py

    # Sweep sequence lengths and head dims
    python amd-fa-pipelined_test.py -sq 512 1024 4096 -d 64 128

    # Both causal modes, multiple batch sizes
    python amd-fa-pipelined_test.py -b 1 2 -causal true false

    # Multiple kernels
    python amd-fa-pipelined_test.py --kernel async_simple --dtype fp16
"""

import argparse
import math
import pytest
import torch
import torch.nn.functional as F

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _assume_strides(
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
):
    tl.assume(stride_qz >= 0)
    tl.assume(stride_qh >= 0)
    tl.assume(stride_qm > 0)
    tl.assume(stride_qk >= 0)
    tl.assume(stride_kz >= 0)
    tl.assume(stride_kh >= 0)
    tl.assume(stride_kn > 0)
    tl.assume(stride_kk >= 0)
    tl.assume(stride_vz >= 0)
    tl.assume(stride_vh >= 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_vk >= 0)
    tl.assume(stride_oz >= 0)
    tl.assume(stride_oh >= 0)
    tl.assume(stride_om > 0)
    tl.assume(stride_ok >= 0)


@triton.jit
def _attn_fwd_async_simple(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    off_z = pid_hz // H
    off_h = pid_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX,
                other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    QK_SCALE = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, 1)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, 1)

    k_base = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_base = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    for start_n in tl.range(0, hi, BLOCK_N, num_stages=0):
        kn = start_n + offs_n
        k_mask = kn[:, None] < N_CTX
        v_mask = kn[:, None] < N_CTX

        tok_k = tlx.async_load(k_base + start_n * stride_kn, tlx.local_view(k_buf, 0), mask=k_mask)
        tok_v = tlx.async_load(v_base + start_n * stride_vn, tlx.local_view(v_buf, 0), mask=v_mask)
        tlx.async_load_commit_group([tok_k, tok_v])

        wait_tok = tlx.async_load_wait_group(0)
        # Transpose K at the memdesc level (metadata-only) so local_load lands
        # directly in dot_op(opIdx=1) layout — skips the register-shuffle + LDS
        # round-trip that `tl.dot(q, k_cur.T)` would otherwise emit.
        kt_view = tlx.local_trans(tlx.local_view(k_buf, 0))
        kt_cur = tlx.local_load(kt_view, token=wait_tok, relaxed=True)
        v_cur = tlx.local_load(tlx.local_view(v_buf, 0), token=wait_tok, relaxed=True)

        qk = tl.dot(q, kt_cur)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk = qk * QK_SCALE - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


@triton.jit
def _attn_fwd_async_prefetch(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Prefetch flash attention with explicit modulo-scheduled prologue,
    hot loop (steady state), and epilogue.

    Design notes:
      * K and V are both double-buffered (2 LDS slots, alternating index
        i%2).
      * `local_trans` is applied to K so `local_load` lands directly in
        dot-operand layout 1, skipping the per-iter ds_write+barrier+
        ds_read shuffle that `tl.dot(q, k_cur.T)` would emit.
    Prologue:
    t = 0
    [GLDS_KV]

    Steady State (Hot Loop):
    t = i               t = i+1
    [LR_KV]
    [QK, SM0, SM1, PV]  [GLDS_KV],
    Epilogue:
                        t = i+1
                        [LR_KV]
                        [QK (masked), SM0, SM1, PV]
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    off_z = pid_hz // H
    off_h = pid_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX,
                other=0.0)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    # K and V: 2 LDS slots each (double-buffered) -- avoids both the
    # memdesc_trans alias race for K and any single-buf RAW hazards.
    NUM_BUFFERS: tl.constexpr = 2
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_BUFFERS)

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = (hi + BLOCK_N - 1) // BLOCK_N
    n_main = tl.maximum(n_blocks - 1, 0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    """
    Prologue:
    t = 0
    [GLDS_KV]
    """
    tok_k0 = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0), mask=offs_n[:, None] < N_CTX)
    tok_v0 = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0), mask=offs_n[:, None] < N_CTX)
    tlx.async_load_commit_group([tok_k0, tok_v0])
    """
    Steady State (Hot Loop):
    t = i               t = i+1
    [LR_KV]
    [QK, SM0, SM1, PV]  [GLDS_KV],
    """
    for block_id in tl.range(0, n_main * BLOCK_N, BLOCK_N, num_stages=0):
        next_off = block_id + BLOCK_N
        kn = block_id + offs_n
        next_mask = (next_off + offs_n[:, None]) < N_CTX

        i = block_id // BLOCK_N
        slot_cur = i % 2
        slot_nxt = (i + 1) % 2

        # LR_KV_ti
        wait_tok = tlx.async_load_wait_group(0)
        kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_cur))
        kt_cur = tlx.local_load(kt_view, token=wait_tok, relaxed=True)
        v_cur = tlx.local_load(tlx.local_view(v_buf, slot_cur), token=wait_tok, relaxed=True)

        # GLDS_KV_t(i+1), prefetch tile i+1 into the *other* slots.
        tok_k = tlx.async_load(k_ptrs + next_off * stride_kn, tlx.local_view(k_buf, slot_nxt), mask=next_mask)
        tok_v = tlx.async_load(v_ptrs + next_off * stride_vn, tlx.local_view(v_buf, slot_nxt), mask=next_mask)
        tlx.async_load_commit_group([tok_k, tok_v])

        # QK_ti
        qk = tl.dot(q, kt_cur)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))

        # SM_ti
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # PV_ti
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)
    """
    Epilogue:
    t = i+1
    [LR_KV]
    [QK (masked), SM0, SM1, PV]
    """
    # Consume tile n_main from slot (n_main % 2).
    wait_tok = tlx.async_load_wait_group(0)
    slot_last = n_main % 2
    kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_last))
    kt_cur = tlx.local_load(kt_view, token=wait_tok, relaxed=True)
    v_cur = tlx.local_load(tlx.local_view(v_buf, slot_last), token=wait_tok, relaxed=True)

    # QK_t(i+1) — with boundary + causal masking
    kn_last = n_main * BLOCK_N + offs_n
    qk = tl.dot(q, kt_cur)
    qk = tl.where(kn_last[None, :] < N_CTX, qk, float("-inf"))
    if IS_CAUSAL:
        qk = tl.where(offs_m[:, None] >= kn_last[None, :], qk, float("-inf"))

    # SM0_t(i+1)
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
    p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
    l_ij = tl.sum(p, 1)

    # SM1_t(i+1)
    alpha = tl.math.exp2(m_i - m_ij)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    m_i = m_ij

    # PV_t(i+1)
    acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

    # Store output
    acc = acc / l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


@triton.jit
def _async_prefetch_attn_tile(
    pid_m,
    q_off,
    k_off,
    v_off,
    o_off,
    Q,
    K,
    V,
    Out,
    k_buf,
    v_buf,
    stride_qm,
    stride_qk,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    stride_om,
    stride_ok,
    N_CTX_Q,
    N_CTX_K,
    DIAG_OFFSET,
    QK_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """
    Function to compute Compute one output m-tile — causal (peeled mask) or non-causal (full).
    Using similar algorithm/optimizations to previous async_prefetch, but with additions of:

    - Peeled last iteration of loop to have steady state not need masking or OOB check.
    - Supports `q_len != kv_len` (cross-attention / decode). Query row `qpos`
      attends key `kpos` iff `kpos <= qpos + DIAG_OFFSET`, where
      `DIAG_OFFSET = kv_len - q_len` (bottom-right alignment).

    """
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX_Q,
                other=0.0)

    if IS_CAUSAL:
        # Largest key any query in this tile may attend, +1 (exclusive bound).
        hi = min(N_CTX_K, (pid_m + 1) * BLOCK_M + DIAG_OFFSET)
    else:
        hi = N_CTX_K

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = (hi + BLOCK_N - 1) // BLOCK_N
    if IS_CAUSAL:
        # Blocks fully below the diagonal for the *smallest* query row in the
        # tile (conservative: any block we skip masking on is unmasked for every
        # row). Reduces to (pid_m*BLOCK_M)//BLOCK_N when DIAG_OFFSET == 0.
        n_unmasked = (pid_m * BLOCK_M + DIAG_OFFSET) // BLOCK_N
        n_unmasked = tl.maximum(tl.minimum(n_unmasked, n_blocks), 0)
    elif EVEN_N:
        # No mask anywhere -> the whole range is the steady-state loop.
        n_unmasked = n_blocks
    else:
        # Only the final ragged block needs a boundary mask.
        n_unmasked = tl.maximum(n_blocks - 1, 0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Prologue: prefetch block 0.
    tok_k0 = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0), mask=offs_n[:, None] < N_CTX_K)
    tok_v0 = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0), mask=offs_n[:, None] < N_CTX_K)
    tlx.async_load_commit_group([tok_k0, tok_v0])
    """
    Unmasked steady-state loop
    """
    for block_id in tl.range(0, n_unmasked * BLOCK_N, BLOCK_N, num_stages=0):
        next_off = block_id + BLOCK_N
        i = block_id // BLOCK_N
        slot_cur = i % 2
        slot_nxt = (i + 1) % 2

        wait_tok = tlx.async_load_wait_group(0)
        kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_cur))
        kt_cur = tlx.local_load(kt_view, token=wait_tok, relaxed=True)
        v_cur = tlx.local_load(tlx.local_view(v_buf, slot_cur), token=wait_tok, relaxed=True)

        next_mask = (next_off + offs_n[:, None]) < N_CTX_K
        tok_k = tlx.async_load(k_ptrs + next_off * stride_kn, tlx.local_view(k_buf, slot_nxt), mask=next_mask)
        tok_v = tlx.async_load(v_ptrs + next_off * stride_vn, tlx.local_view(v_buf, slot_nxt), mask=next_mask)
        tlx.async_load_commit_group([tok_k, tok_v])

        qk = tl.dot(q, kt_cur)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)
    """
    Masked peeled epilogue
    """
    for block_id in tl.range(n_unmasked * BLOCK_N, n_blocks * BLOCK_N, BLOCK_N, num_stages=0):
        next_off = block_id + BLOCK_N
        kn = block_id + offs_n
        i = block_id // BLOCK_N
        slot_cur = i % 2
        slot_nxt = (i + 1) % 2

        wait_tok = tlx.async_load_wait_group(0)
        kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_cur))
        kt_cur = tlx.local_load(kt_view, token=wait_tok, relaxed=True)
        v_cur = tlx.local_load(tlx.local_view(v_buf, slot_cur), token=wait_tok, relaxed=True)

        if next_off < hi:
            next_mask = (next_off + offs_n[:, None]) < N_CTX_K
            tok_k = tlx.async_load(k_ptrs + next_off * stride_kn, tlx.local_view(k_buf, slot_nxt), mask=next_mask)
            tok_v = tlx.async_load(v_ptrs + next_off * stride_vn, tlx.local_view(v_buf, slot_nxt), mask=next_mask)
            tlx.async_load_commit_group([tok_k, tok_v])

        qk = tl.dot(q, kt_cur)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] + DIAG_OFFSET >= kn[None, :], qk, float("-inf"))
        if not EVEN_N:
            qk = tl.where(kn[None, :] < N_CTX_K, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < N_CTX_Q) & (offs_d[None, :] < HEAD_DIM))


@triton.jit
def _attn_fwd_persistent(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX_Q,
    N_CTX_K,
    DIAG_OFFSET,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """
    Persistent FA with these key features:
    1. Aysnc prefetch work tile
    2. XCD-pinned head-batch dim for improved L2 locality
    3. zig-zag ordering of output/work tile for each workunit wihtin a WG,
       for equal work distribution per WG and improved causal performance.

    Ownership hierarchy (coarse -> fine), all derived from one flat `unit` index.
    Examples use N=1024 causal defaults (BLOCK_M=256 -> 4 tiles/head, 64 heads,
    8 XCDs, 32 programs/XCD, TILES_PER_UNIT=2 -> units_per_head=2, 16 units/XCD):

    - XCD owns `heads_per_xcd` batch-heads (those with `hz % NUM_XCDS == xcd`),
      i.e. a flat pile of `units = heads_per_xcd * units_per_head` work-units.
        e.g. XCD 0 owns heads {0,8,16,...,56} -> 16 units.

    - Local (one of the XCD's `NUM_LOCAL` workgroups) owns
      a round-robin slice of that pile: units `local, local+NUM_LOCAL, ...`.
        e.g. WG local=2 owns unit 2 (here 16 units < 32 programs, so local>=16 idle).

    - Unit -> global work-unit id within the whole XCD; `bundle` = which work-unit within a head): `local_head = unit //
      units_per_head` picks the head (global `pid_hz = xcd + local_head *
      NUM_XCDS`), and `bundle = unit % units_per_head` picks the tile-group
      within that head.
        e.g. unit 2 -> local_head=1 -> head 8, bundle=0.

    - Bundle owns `TILES_PER_UNIT` zig-zag tiles (causal: a {light, heavy} fold
      pair of constant cost; non-causal: a single tile).
        e.g. bundle 0 -> zig-zag {0,1} -> tiles {m0, m3} of head 8.

    - Tile = `BLOCK_M` query rows of that head, streamed over its `BLOCK_N`
      K-blocks (the inner `_async_prefetch_attn_tile` loop) to produce one output tile.
        e.g. m0 -> rows 0..255, 4 K-blocks; m3 -> rows 768..1023, 16 (sum=20, flat).
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    # Note since it is persistent, we only launch 1-dim of PIDs (NUM_SMs,).
    pid = tl.program_id(0)
    xcd = pid % NUM_XCDS
    local = pid // NUM_XCDS
    NUM_LOCAL: tl.constexpr = NUM_SMS // NUM_XCDS

    NUM_BUFFERS: tl.constexpr = 2
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_BUFFERS)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    # Causal: 2-tile {light, heavy} fold bundles (constant cost). Non-causal:
    # 1-tile units (tiles are already equal-cost, so no bundling is needed).
    TILES_PER_UNIT: tl.constexpr = 2 if IS_CAUSAL else 1

    # We flattened head and batch dim into a single flat HZ work dim.
    # Then, we pin specific HZ ids to a XCD for K/V L2 locality.
    units_per_hz: tl.constexpr = (NUM_M_BLOCKS + TILES_PER_UNIT - 1) // TILES_PER_UNIT
    hz_per_xcd = (Z * H + NUM_XCDS - 1) // NUM_XCDS
    units = hz_per_xcd * units_per_hz
    for unit in tl.range(local, units, NUM_LOCAL, num_stages=0):
        local_hz = unit // units_per_hz  # which head-batch id (quotient)
        bundle = unit % units_per_hz  # quer_seq/m tiles within head-batch id (remainder)
        pid_hz = xcd + local_hz * NUM_XCDS  # global head-batch id.
        if pid_hz < Z * H:
            off_z = pid_hz // H
            off_h = pid_hz % H
            q_off = off_z * stride_qz + off_h * stride_qh
            k_off = off_z * stride_kz + off_h * stride_kh
            v_off = off_z * stride_vz + off_h * stride_vh
            o_off = off_z * stride_oz + off_h * stride_oh

            # Run the bundle's TILES_PER_UNIT consecutive zig-zag tiles.
            # If causal and TILES_PER_UNIT=2, then head's m-tiles are walked
            # as `0, N-1, 1, N-2, …`. This interleaves the lightest and heaviest
            # causal tiles, making each WG do ~same amount of work.
            for j in tl.static_range(TILES_PER_UNIT):
                idx = bundle * TILES_PER_UNIT + j
                if idx < NUM_M_BLOCKS:
                    half = idx // 2
                    pid_m = tl.where(idx % 2 == 0, half, NUM_M_BLOCKS - 1 - half)
                    _async_prefetch_attn_tile(pid_m, q_off, k_off, v_off, o_off, Q, K, V, Out, k_buf, v_buf, stride_qm,
                                              stride_qk, stride_kn, stride_kk, stride_vn, stride_vk, stride_om,
                                              stride_ok, N_CTX_Q, N_CTX_K, DIAG_OFFSET, QK_SCALE=QK_SCALE,
                                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, IS_CAUSAL=IS_CAUSAL,
                                              EVEN_N=EVEN_N)


# ═══════════════════════════════════════════════════════════════════════════
# Host wrapper
# ═══════════════════════════════════════════════════════════════════════════


def flash_attn_async_simple(q, k, v, sm_scale, causal=False, **kw):
    """Launch with K in original BHND layout — stride_kk=1 avoids alignment issues."""
    B, H, N_CTX, D = q.shape
    o = torch.empty_like(q)

    BLOCK_M = kw.pop("BLOCK_M", 256)
    BLOCK_N = kw.pop("BLOCK_N", 64)
    num_warps = kw.pop("num_warps", 4)

    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)
    _attn_fwd_async_simple[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        B,
        H,
        N_CTX,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        **kw,
    )
    return o


def flash_attn_async_prefetch(q, k, v, sm_scale, causal=False, **kw):
    """Prefetch FA with modulo-scheduled prologue/hot-loop/epilogue."""
    B, H, N_CTX, D = q.shape
    o = torch.empty_like(q)

    BLOCK_M = kw.pop("BLOCK_M", 256)
    # BLOCK_N=128 wins at D=64 nocausal (more compute per barrier),
    # but the diagonal masking cost overwhelms that for causal, and at
    # D=128 it blows the 64KB LDS budget for double-buffered K+V.
    BLOCK_N = kw.pop("BLOCK_N", 128 if (D <= 64 and not causal) else 64)
    num_warps = kw.pop("num_warps", 4)

    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)
    _attn_fwd_async_prefetch[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        B,
        H,
        N_CTX,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        **kw,
    )
    return o


# ═══════════════════════════════════════════════════════════════════════════
# Kernel registry — add new kernel wrappers here
# ═══════════════════════════════════════════════════════════════════════════


def flash_attn_persistent(q, k, v, sm_scale, causal=False, **kw):
    """persistent FA with workgroup scheduler who is persistent and have a zig-zag tile order.

    FA with a workgroup scheduler who is persistent, XCD-grouped,
    with constant-cost fold-bundling scheduler (zig-zag tile order + `TILES_PER_UNIT` bundle).
    Causal uses 2-tile fold bundles non-causal uses 1-tile
    units (tiles are already equal-cost). A program runs a *variable* number of
    tiles, not a fixed 2.

    Supports `q_len != kv_len` (cross-attention / decode): the causal diagonal is
    anchored bottom-right via `DIAG_OFFSET = kv_len - q_len` (matching PyTorch
    SDPA). Q heads and K/V heads must match (no GQA yet).

    Falls back to the square baseline for partial-block N (`N % BLOCK_N != 0`) to
    dodge the modulo-decode `iota_range` compiler crash (not perf-critical).
    """
    B, H, N_CTX_Q, D = q.shape
    N_CTX_K = k.shape[2]
    diag_offset = N_CTX_K - N_CTX_Q

    BLOCK_M = kw.pop("BLOCK_M", 256)
    # Peeled causal + non-causal both prefer BLOCK_N=128 at D<=64 (more compute
    # per LDS barrier); D=128 must stay 64 for the double-buffered K+V LDS budget.
    BLOCK_N = kw.pop("BLOCK_N", 128 if D <= 64 else 64)
    num_warps = kw.pop("num_warps", 4)

    if N_CTX_K % BLOCK_N != 0:
        # Partial-block kv hits the modulo-decode iota_range compiler crash.
        # The square baseline only handles q_len == kv_len, so require that here.
        assert N_CTX_Q == N_CTX_K, ("persistent: partial-block kv_len with "
                                    "q_len != kv_len is not supported "
                                    f"(q_len={N_CTX_Q}, kv_len={N_CTX_K}, BLOCK_N={BLOCK_N})")
        return flash_attn_async_prefetch(q, k, v, sm_scale, causal=causal, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                         num_warps=num_warps, **kw)

    o = torch.empty_like(q)
    num_m_blocks = triton.cdiv(N_CTX_Q, BLOCK_M)
    num_xcds = kw.pop("NUM_XCDS", 8)
    cu_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    num_sms = kw.pop("NUM_SMS", (cu_count // num_xcds) * num_xcds)
    grid = (num_sms, )
    _attn_fwd_persistent[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        B,
        H,
        N_CTX_Q,
        N_CTX_K,
        diag_offset,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
        NUM_M_BLOCKS=num_m_blocks,
        NUM_SMS=num_sms,
        NUM_XCDS=num_xcds,
        EVEN_N=(N_CTX_K % BLOCK_N == 0),
        num_warps=num_warps,
        **kw,
    )
    return o


KERNEL_REGISTRY = {
    "async_simple": flash_attn_async_simple,
    "async_prefetch": flash_attn_async_prefetch,
    "persistent": flash_attn_persistent,
}


def get_kernel(name):
    if name not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel: {name!r}. "
                         f"Available: {list(KERNEL_REGISTRY.keys())}")
    return KERNEL_REGISTRY[name]


# ═══════════════════════════════════════════════════════════════════════════
# Reference, verification, and Misc Utils
# ═══════════════════════════════════════════════════════════════════════════


def print_summary_table(results, kernel_names):
    """Print a markdown-style summary table of benchmark results."""
    providers = ["Torch SDPA"] + list(kernel_names)

    rows = []
    for key in sorted(results.keys()):
        B, H, D, N, causal = key
        rows.append((f"B={B}, H={H}, D={D}, N={N}, causal={causal}", results[key]))

    cfg_w = max(len("Config"), *(len(lbl) for lbl, _ in rows)) if rows else len("Config")
    col_w = max(14, *(len(p) for p in providers))

    hdr = f"| {'Config':<{cfg_w}} |" + "".join(f" {p:>{col_w}} |" for p in providers)
    sep = f"|{'-' * (cfg_w + 2)}|" + "".join(f"{'-' * (col_w + 2)}|" for _ in providers)

    print(f"\n{'=' * len(sep)}")
    print("Summary (TFLOPS)")
    print(f"{'=' * len(sep)}")
    print(hdr)
    print(sep)

    for label, prov in rows:
        vals = (f"{prov[p]['tflops']:>{col_w}.1f}" if p in prov else f"{'—':>{col_w}}" for p in providers)
        print(f"| {label:<{cfg_w}} |" + "".join(f" {v} |" for v in vals))

    print(f"{'=' * len(sep)}\n")


def ref_sdpa(q, k, v, sm_scale, causal=False):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)


def verify(name, got, ref, atol=2e-2, rtol=2e-2, log=True):
    diff = (got.float() - ref.float()).abs()
    ok = torch.allclose(ref, got, atol=atol, rtol=rtol)
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    status = "PASS" if ok else "FAIL"
    if log:
        print(f"  {name:<28} {status}  max={max_err:.6f}  mean={mean_err:.6f}")
    return ok


def run_correctness_check(kernel_fn, dtype, causal, B=2, H=4, N=512, D=128):
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    sm = 1.0 / math.sqrt(D)
    ref = ref_sdpa(q, k, v, sm, causal)
    tag = f"causal={causal} N={N}"
    out = kernel_fn(q, k, v, sm, causal)
    return verify(f"{kernel_fn.__name__} [{tag}]", out, ref)


@pytest.mark.parametrize("causal", [False, True], ids=["nocausal", "causal"])
@pytest.mark.parametrize("n_test", [128, 192, 256, 500, 512, 1024])
@pytest.mark.parametrize("kernel_name", list(KERNEL_REGISTRY.keys()))
def test_fa_correctness(kernel_name, causal, n_test, dtype=torch.bfloat16, D=128):
    kernel_fn = get_kernel(kernel_name)
    ok = run_correctness_check(kernel_fn, dtype, causal, B=1, H=4, N=n_test, D=D)
    assert ok, f"Correctness failed: kernel={kernel_name} causal={causal} N={n_test}"


def _ref_bottomright(q, k, v, sm, causal):
    """Reference for q_len != kv_len: bottom-right causal alignment
    (key j attends iff j <= i + (kv_len - q_len)) — the decode/KV-cache and
    FlashAttention convention. Reduces to is_causal when q_len == kv_len."""
    Lq, Lk = q.shape[2], k.shape[2]
    if not causal:
        return F.scaled_dot_product_attention(q, k, v, scale=sm)
    i = torch.arange(Lq, device=q.device)[:, None]
    j = torch.arange(Lk, device=q.device)[None, :]
    bias = torch.zeros(Lq, Lk, device=q.device, dtype=q.dtype).masked_fill(~(j <= i + (Lk - Lq)), float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=sm)


@pytest.mark.parametrize("causal", [False, True], ids=["nocausal", "causal"])
@pytest.mark.parametrize("q_len,kv_len", [(256, 1024), (1024, 256), (1, 1024), (1024, 1024)],
                         ids=["cross_qlt", "cross_qgt", "decode", "square"])
def test_fa_persistent_cross_attention(q_len, kv_len, causal, dtype=torch.bfloat16, D=128):
    """persistent kernel with q_len != kv_len (cross-attention / decode)."""
    torch.manual_seed(42)
    B, H = 1, 8
    q = torch.randn(B, H, q_len, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, kv_len, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, kv_len, D, device=DEVICE, dtype=dtype)
    sm = 1.0 / math.sqrt(D)
    ref = _ref_bottomright(q, k, v, sm, causal)
    out = flash_attn_persistent(q, k, v, sm, causal)
    valid = ~torch.isnan(ref.float())  # fully-masked rows (q_len > kv_len) are undefined
    ok = torch.allclose(out.float()[valid], ref.float()[valid], atol=2e-2, rtol=2e-2)
    assert ok, f"cross-attn failed: q_len={q_len} kv_len={kv_len} causal={causal}"


# ═══════════════════════════════════════════════════════════════════════════
# Performance regression guard
# ═══════════════════════════════════════════════════════════════════════════

# Baseline TFLOPS measured on gfx950/MI350 (bf16, B=1, H=64). A kernel must stay
# within PERF_TOL below its baseline or the perf test fails (catches regressions).
# Regenerate the numbers with:
#   python amd-fa-pipelined_test.py -b 1 -hq 64 -sq 4096 8192 16384 -d 64 128 \
#       -causal true false --kernel async_simple async_prefetch persistent
# then copy the printed Summary table values here.
PERF_TOL = 0.15  # allow 15% slack below baseline (run-to-run noise + minor drift)

# key: (kernel, D, N, causal) -> TFLOPS
PERF_BASELINE_TFLOPS = {
    ("async_simple", 64, 4096, False): 614,
    ("async_simple", 64, 4096, True): 344,
    ("async_simple", 64, 8192, False): 644,
    ("async_simple", 64, 8192, True): 371,
    ("async_simple", 64, 16384, False): 668,
    ("async_simple", 64, 16384, True): 432,
    ("async_simple", 128, 4096, False): 646,
    ("async_simple", 128, 4096, True): 341,
    ("async_simple", 128, 8192, False): 697,
    ("async_simple", 128, 8192, True): 366,
    ("async_simple", 128, 16384, False): 724,
    ("async_simple", 128, 16384, True): 475,
    ("async_prefetch", 64, 4096, False): 677,
    ("async_prefetch", 64, 4096, True): 334,
    ("async_prefetch", 64, 8192, False): 710,
    ("async_prefetch", 64, 8192, True): 348,
    ("async_prefetch", 64, 16384, False): 732,
    ("async_prefetch", 64, 16384, True): 439,
    ("async_prefetch", 128, 4096, False): 782,
    ("async_prefetch", 128, 4096, True): 447,
    ("async_prefetch", 128, 8192, False): 832,
    ("async_prefetch", 128, 8192, True): 475,
    ("async_prefetch", 128, 16384, False): 861,
    ("async_prefetch", 128, 16384, True): 544,
    ("persistent", 64, 4096, False): 695,
    ("persistent", 64, 4096, True): 620,
    ("persistent", 64, 8192, False): 724,
    ("persistent", 64, 8192, True): 693,
    ("persistent", 64, 16384, False): 744,
    ("persistent", 64, 16384, True): 741,
    ("persistent", 128, 4096, False): 828,
    ("persistent", 128, 4096, True): 697,
    ("persistent", 128, 8192, False): 876,
    ("persistent", 128, 8192, True): 796,
    ("persistent", 128, 16384, False): 894,
    ("persistent", 128, 16384, True): 837,
}


def measure_tflops(kernel_fn, B, H, N, D, causal, dtype=torch.bfloat16):
    """Verify correctness then return achieved TFLOPS for one config."""
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    sm = 1.0 / math.sqrt(D)
    fn = lambda: kernel_fn(q, k, v, sm, causal)
    assert verify("", fn(), ref_sdpa(q, k, v, sm, causal), log=False), "correctness check failed"
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    valid_el = N * (N + 1) // 2 if causal else N * N
    total_flops = 2 * 2.0 * B * H * valid_el * D
    return total_flops / ms * 1e-9


@pytest.mark.parametrize("causal", [False, True], ids=["nocausal", "causal"])
@pytest.mark.parametrize("N", [4096, 8192, 16384])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("kernel_name", ["async_simple", "async_prefetch", "persistent"])
def test_fa_performance(kernel_name, D, N, causal):
    """Guard against perf regressions: achieved TFLOPS must stay within PERF_TOL
    of the recorded gfx950 baseline (B=1, H=64, bf16)."""
    baseline = PERF_BASELINE_TFLOPS[(kernel_name, D, N, causal)]
    floor = baseline * (1 - PERF_TOL)
    got = measure_tflops(get_kernel(kernel_name), 1, 64, N, D, causal)
    print(f"  {kernel_name:14s} D={D:3d} N={N:5d} {'causal' if causal else 'nc':6s} "
          f"{got:7.1f} TFLOPS (baseline {baseline}, floor {floor:.1f})")
    assert got >= floor, (f"perf regression: {kernel_name} D={D} N={N} causal={causal} -> "
                          f"{got:.1f} TFLOPS < floor {floor:.1f} (baseline {baseline}, tol {PERF_TOL:.0%})")


def run_perf_test(args):
    """CLI entry: run the perf guard over all baseline configs, print a report,
    and return True iff every kernel is within PERF_TOL of its baseline."""
    tol = args.perf_tol
    all_ok = True
    print(f"\nPerformance regression test (floor = baseline * (1 - {tol:.0%}))")
    print("-" * 78)
    for (kernel_name, D, N, causal), baseline in sorted(PERF_BASELINE_TFLOPS.items()):
        floor = baseline * (1 - tol)
        try:
            got = measure_tflops(get_kernel(kernel_name), 1, 64, N, D, causal)
        except Exception as e:
            all_ok = False
            print(f"  [FAIL] {kernel_name:14s} D={D:3d} N={N:5d} {'causal' if causal else 'nc':6s} -> ERROR ({e})")
            continue
        ok = got >= floor
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {kernel_name:14s} D={D:3d} N={N:5d} "
              f"{'causal' if causal else 'nc':6s} {got:7.1f} TFLOPS (baseline {baseline}, floor {floor:.1f})")
    print("-" * 78)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    assert all_ok


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════


def run_benchmark(args):
    causal_modes = [s.lower() in ("true", "1", "yes") for s in args.causal]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    results = {}

    for kernel_name in args.kernel:
        kernel_fn = get_kernel(kernel_name)
        for B in args.b:
            for H in args.hq:
                for D in args.d:
                    for N in args.sq:
                        for causal in causal_modes:
                            torch.manual_seed(42)
                            q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
                            k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
                            v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
                            sm = 1.0 / math.sqrt(D)

                            if causal:
                                valid_el = N * (N + 1) // 2
                            else:
                                valid_el = N * N
                            total_flops = 2 * 2.0 * B * H * valid_el * D

                            causal_str = "causal" if causal else "nc"
                            ref_sdpa_lambda = lambda: F.scaled_dot_product_attention(
                                q, k, v, is_causal=causal, scale=sm)

                            try:
                                tlx_sdpa_lambda = lambda: kernel_fn(q, k, v, sm, causal)
                                ref_out = ref_sdpa_lambda()
                                tlx_out = tlx_sdpa_lambda()
                                assert verify("", tlx_out, ref_out, log=False)
                            except Exception as e:
                                print(f"  {kernel_name:20s} D={D} N={N:5d} {causal_str:6s} -> SKIPPED ({e})")
                                continue

                            key = (B, H, D, N, causal)
                            if key not in results:
                                results[key] = {}

                            if "Torch SDPA" not in results[key]:
                                ms = triton.testing.do_bench(ref_sdpa_lambda, warmup=25, rep=100)
                                tflops = total_flops / ms * 1e-9
                                results[key]["Torch SDPA"] = {"ms": ms, "tflops": tflops}

                            ms = triton.testing.do_bench(tlx_sdpa_lambda, warmup=25, rep=100)
                            tflops = total_flops / ms * 1e-9
                            results[key][kernel_name] = {"ms": ms, "tflops": tflops}

    print_summary_table(results, args.kernel)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(prog="AMD TLX FA Pipelined")
    p.add_argument("-b", type=int, nargs="+", default=[1])
    p.add_argument("-hq", type=int, nargs="+", default=[64])
    p.add_argument("-sq", type=int, nargs="+", default=[1024, 8192, 16384])
    p.add_argument("-d", type=int, nargs="+", default=[64, 128])
    p.add_argument("-causal", type=str, nargs="+", default=["false"],
                   help="Causal modes to benchmark (e.g. -causal true false)")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--kernel", type=str, nargs="+", default=["async_simple", "async_prefetch", "persistent"],
                   help="Kernel variants to benchmark")
    p.add_argument("--mode", choices=["benchmark", "perf_test"], default="benchmark",
                   help="benchmark: print TFLOPS table; perf_test: assert TFLOPS within PERF_TOL of baseline")
    p.add_argument("--perf-tol", dest="perf_tol", type=float, default=PERF_TOL,
                   help="perf_test slack below baseline (e.g. 0.15 = allow 15%% regression)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "perf_test":
        run_perf_test(args)
    else:
        run_benchmark(args)
