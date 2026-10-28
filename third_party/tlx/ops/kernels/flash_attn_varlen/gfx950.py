"""Packed variable-length flash attention forward for gfx950 (MI350). FP16/BF16, head dim 128, inference only.

Q/K/V are packed ``(tokens, heads, 128)``; sequence ``b`` owns query rows
``cu_seqlens_q[b]:cu_seqlens_q[b + 1]`` and key rows
``cu_seqlens_k[b]:cu_seqlens_k[b + 1]``. Causal masking is bottom-right
aligned: query row ``i`` of a sequence sees keys ``j <= i + k_len - q_len``.

Each workgroup is one 64-row query tile of one (sequence, head), on 4 warps of
16 rows. Short sequences give few tiles (B = 10 x 1000 tokens is 160 for 256
CUs), so the tiles stay small and the time is per-tile latency:

- Workgroups go round-robin to the 8 XCDs; each XCD gets a contiguous run of
  work items, so most tiles of a (sequence, head) share its K/V in that XCD's
  L2 (one can straddle two runs).
- K and V are copied straight from global memory into LDS, double buffered;
  one copy group per step (K(b + 2), V(b + 1)) lands during step b.
- Each step issues QK(b + 1) and PV(b) MFMA by MFMA, with K and V read from
  LDS in 32-wide pieces, each read issued one MFMA before its use (as in
  flash_attn_mxfp8/gfx950.py's ``_mfma_stage``); scheduling barriers keep
  LLVM from sinking the reads to their uses.
- Key blocks that every row of the tile sees in full run unmasked; only the
  sequence's last partial block and the causal diagonal are masked.
- Causal: tile m sees m + 1 key blocks, so the last tiles of a sequence set
  the time. When CUs are spare, tile m is split by key range into
  m // SPLIT + 1 work items. The tile's last split merges: the others store
  their unnormalized fp32 partial and release a per-tile counter; the merger
  acquires it, folds the partials in, writes the output and resets the
  counter. A tile's splits are adjacent items of one XCD's run, so the merger
  is dispatched after the splits it waits for.
"""

import functools

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

BLOCK_M = 64
BLOCK_N = 64
NUM_WARPS = 4
# Causal split: the smallest SPLIT >= 4 that gives every work item its own CU
# and stores at most 64 partials (each partial's release writes back its XCD's
# L2, so more splits than that cost more than they save).
MIN_SPLIT = 4
MAX_PARTIALS = 64
# Each stream gets its own counters; they are zeroed in batches of 16 so that a
# stream first seen while capturing a graph finds some ready.
COUNTER_ROWS = 16


@triton.jit
def _qk(q0, q1, q2, q3, k_slot, BLOCK_N: tl.constexpr):
    # Q arrives as four 32-wide head-dim chunks: extract_slice on a
    # layout-pinned helper argument crashes the TLX fixup pass.
    mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[4, 1])
    kl: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    n: tl.constexpr = k_slot.shape[0]
    s = tlx.zeros((q0.shape[0], BLOCK_N), tl.float32, layout=mma)
    k0 = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_slice(k_slot, [0, 0], [n, 32])), relaxed=True), kl,
                            pin=False)
    k1 = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_slice(k_slot, [0, 32], [n, 32])), relaxed=True),
                            kl, pin=False)
    k2 = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_slice(k_slot, [0, 64], [n, 32])), relaxed=True),
                            kl, pin=False)
    k3 = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_slice(k_slot, [0, 96], [n, 32])), relaxed=True),
                            kl, pin=False)
    s = tl.dot(q0, k0, s)
    s = tl.dot(q1, k1, s)
    s = tl.dot(q2, k2, s)
    return tl.dot(q3, k3, s)


@triton.jit
def _pv(p, v_slot, acc0, acc1, acc2, acc3):
    mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[4, 1])
    vl: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    n: tl.constexpr = v_slot.shape[0]
    p = tlx.require_layout(p, tlx.dot_operand_layout(0, mma, k_width=4), pin=False)
    v0 = tlx.require_layout(tlx.local_load(tlx.local_slice(v_slot, [0, 0], [n, 32]), relaxed=True), vl, pin=False)
    v1 = tlx.require_layout(tlx.local_load(tlx.local_slice(v_slot, [0, 32], [n, 32]), relaxed=True), vl, pin=False)
    v2 = tlx.require_layout(tlx.local_load(tlx.local_slice(v_slot, [0, 64], [n, 32]), relaxed=True), vl, pin=False)
    v3 = tlx.require_layout(tlx.local_load(tlx.local_slice(v_slot, [0, 96], [n, 32]), relaxed=True), vl, pin=False)
    return tl.dot(p, v0, acc0), tl.dot(p, v1, acc1), tl.dot(p, v2, acc2), tl.dot(p, v3, acc3)


@triton.jit
def _mfma_stage(q0, q1, q2, q3, p, k_slot, v_slot, acc0, acc1, acc2, acc3, BLOCK_N: tl.constexpr):
    # QK(b + 1) from k_slot and PV(b) from v_slot, alternating 32-wide chunks.
    mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[4, 1])
    kl: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    vl: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    n: tl.constexpr = k_slot.shape[0]
    p = tlx.require_layout(p, tlx.dot_operand_layout(0, mma, k_width=4), pin=False)
    s = tlx.zeros((q0.shape[0], BLOCK_N), tl.float32, layout=mma)
    k0 = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_slice(k_slot, [0, 0], [n, 32])), relaxed=True), kl,
                            pin=False)
    v0 = tlx.require_layout(tlx.local_load(tlx.local_slice(v_slot, [0, 0], [n, 32]), relaxed=True), vl, pin=False)
    tlx.amd_sched_barrier(0)
    s = tl.dot(q0, k0, s)
    k1 = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_slice(k_slot, [0, 32], [n, 32])), relaxed=True),
                            kl, pin=False)
    tlx.amd_sched_barrier(0)
    acc0 = tl.dot(p, v0, acc0)
    v1 = tlx.require_layout(tlx.local_load(tlx.local_slice(v_slot, [0, 32], [n, 32]), relaxed=True), vl, pin=False)
    tlx.amd_sched_barrier(0)
    s = tl.dot(q1, k1, s)
    k2 = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_slice(k_slot, [0, 64], [n, 32])), relaxed=True),
                            kl, pin=False)
    tlx.amd_sched_barrier(0)
    acc1 = tl.dot(p, v1, acc1)
    v2 = tlx.require_layout(tlx.local_load(tlx.local_slice(v_slot, [0, 64], [n, 32]), relaxed=True), vl, pin=False)
    tlx.amd_sched_barrier(0)
    s = tl.dot(q2, k2, s)
    k3 = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_slice(k_slot, [0, 96], [n, 32])), relaxed=True),
                            kl, pin=False)
    tlx.amd_sched_barrier(0)
    acc2 = tl.dot(p, v2, acc2)
    v3 = tlx.require_layout(tlx.local_load(tlx.local_slice(v_slot, [0, 96], [n, 32]), relaxed=True), vl, pin=False)
    tlx.amd_sched_barrier(0)
    s = tl.dot(q3, k3, s)
    tlx.amd_sched_barrier(0)
    acc3 = tl.dot(p, v3, acc3)
    return s, acc0, acc1, acc2, acc3


@triton.jit
def _softmax(s, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale, dtype: tl.constexpr):
    # Online softmax on unscaled scores; m_i is in scaled (log2) units, so the
    # scale folds into the exponent's FMA.
    m_new = tl.maximum(m_i, tl.max(s, 1) * qk_scale)
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.math.exp2(s * qk_scale - m_safe[:, None])
    alpha = tl.math.exp2(m_i - m_safe)
    l_i = l_i * alpha + tl.sum(p, 1)
    a = alpha[:, None]
    return m_new, l_i, p.to(dtype), acc0 * a, acc1 * a, acc2 * a, acc3 * a


@triton.jit
def _split_item(r, SPLIT: tl.constexpr):
    # Item r of a (sequence, head) -> (query tile, split, splits): tiles
    # [j * SPLIT, (j + 1) * SPLIT) have j + 1 splits each.
    j = 0
    while r >= SPLIT * (j + 1):
        r -= SPLIT * (j + 1)
        j += 1
    return j * SPLIT + r // (j + 1), r % (j + 1), j + 1


@triton.jit
def _tile_start(t, n_items, items_per_bh, SPLIT: tl.constexpr):
    # The first item at or after t that is a tile's first split.
    _, split, n_splits = _split_item(t % items_per_bh, SPLIT)
    return tl.minimum(tl.where(split == 0, t, t + n_splits - split), n_items)


@triton.jit
def _store_partial(Part, slot, m_i, l_i, acc0, acc1, acc2, acc3, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    # Unnormalized accumulator, then the row max (scaled log2 units) and sum.
    base = Part + slot.to(tl.int64) * (BLOCK_M * (HEAD_DIM + 2))
    rows = tl.arange(0, BLOCK_M)
    ptrs = base + rows[:, None] * HEAD_DIM + tl.arange(0, 32)[None, :]
    tl.store(ptrs, tlx.release_layout(acc0))
    tl.store(ptrs + 32, tlx.release_layout(acc1))
    tl.store(ptrs + 64, tlx.release_layout(acc2))
    tl.store(ptrs + 96, tlx.release_layout(acc3))
    tl.store(base + BLOCK_M * HEAD_DIM + rows, m_i)
    tl.store(base + BLOCK_M * (HEAD_DIM + 1) + rows, l_i)


@triton.jit
def _fold_partials(Part, slot0, count, m_i, l_i, acc0, acc1, acc2, acc3, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 32)
    for s in range(count):
        base = Part + (slot0 + s).to(tl.int64) * (BLOCK_M * (HEAD_DIM + 2))
        m_s = tl.load(base + BLOCK_M * HEAD_DIM + rows)
        l_s = tl.load(base + BLOCK_M * (HEAD_DIM + 1) + rows)
        m_new = tl.maximum(m_i, m_s)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        a = tl.math.exp2(m_i - m_safe)
        w = tl.math.exp2(m_s - m_safe)
        l_i = l_i * a + l_s * w
        ptrs = base + rows[:, None] * HEAD_DIM + cols[None, :]
        acc0 = acc0 * a[:, None] + tl.load(ptrs) * w[:, None]
        acc1 = acc1 * a[:, None] + tl.load(ptrs + 32) * w[:, None]
        acc2 = acc2 * a[:, None] + tl.load(ptrs + 64) * w[:, None]
        acc3 = acc3 * a[:, None] + tl.load(ptrs + 96) * w[:, None]
        m_i = m_new
    return l_i, acc0, acc1, acc2, acc3


@triton.jit
def _flash_attn_varlen_fwd(Q, K, V, Out, CuQ, CuK, Part, Count, stride_qt, stride_qh, stride_kt, stride_kh, stride_vt,
                           stride_vh, stride_ot, stride_oh, qk_scale, H, kv_group, num_m_blocks, items_per_bh, n_items,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
                           IS_CAUSAL: tl.constexpr, SPLIT: tl.constexpr):
    tl.static_assert(HEAD_DIM == 128)
    pid = tl.program_id(0)
    # Workgroups go round-robin to the 8 XCDs: give each XCD a contiguous run
    # of (sequence, head, query tile[, split]) items.
    per_xcd = tl.cdiv(n_items, 8)
    if SPLIT == 0:
        item = (pid % 8) * per_xcd + pid // 8
        if item >= n_items:
            return
        m_block = item % num_m_blocks
        bh = item // num_m_blocks
        split = 0
        n_splits = 1
    else:
        # Runs start and end on a tile's first split.
        item = _tile_start((pid % 8) * per_xcd, n_items, items_per_bh, SPLIT) + pid // 8
        if item >= _tile_start((pid % 8 + 1) * per_xcd, n_items, items_per_bh, SPLIT):
            return
        bh = item // items_per_bh
        m_block, split, n_splits = _split_item(item % items_per_bh, SPLIT)
    off_h = bh % H
    off_kh = off_h // kv_group
    b = bh // H
    q_start = tl.load(CuQ + b)
    q_len = tl.load(CuQ + b + 1) - q_start
    k_start = tl.load(CuK + b)
    k_len = tl.load(CuK + b + 1) - k_start
    if m_block * BLOCK_M >= q_len:
        return

    mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[4, 1])
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < q_len
    q = tl.load(Q + (q_start + offs_m)[:, None] * stride_qt + off_h * stride_qh + offs_d[None, :], mask=row_ok[:, None],
                other=0.0)
    q = tlx.require_layout(q, tlx.dot_operand_layout(0, mma, k_width=8), pin=False)
    q0 = tlx.extract_slice(q, [BLOCK_M, 32], [0, 0])
    q1 = tlx.extract_slice(q, [BLOCK_M, 32], [0, 32])
    q2 = tlx.extract_slice(q, [BLOCK_M, 32], [0, 64])
    q3 = tlx.extract_slice(q, [BLOCK_M, 32], [0, 96])
    k_ptrs = K + k_start * stride_kt + off_kh * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :]
    v_ptrs = V + k_start * stride_vt + off_kh * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :]
    k_step = BLOCK_N * stride_kt
    v_step = BLOCK_N * stride_vt

    # Blocks [lo, full) are unmasked for every row of the tile; [full, n_blocks)
    # hold the sequence's partial last block and the causal diagonal.
    causal_off = k_len - q_len
    if IS_CAUSAL:
        hi = tl.maximum(tl.minimum(k_len, (m_block + 1) * BLOCK_M + causal_off), 0)
        full = tl.minimum(k_len, tl.maximum(m_block * BLOCK_M + causal_off + 1, 0)) // BLOCK_N
    else:
        hi = k_len
        full = k_len // BLOCK_N
    n_blocks = tl.cdiv(hi, BLOCK_N)
    full = tl.minimum(full, n_blocks)
    lo = 0
    if SPLIT > 0:
        # Split s of n walks key blocks [s * n_blocks // n, (s + 1) * n_blocks // n).
        lo = split * n_blocks // n_splits
        n_blocks = (split + 1) * n_blocks // n_splits
        full = tl.maximum(tl.minimum(full, n_blocks), lo)

    # Padded rows and the dense bases [[0, 1] .. [0, 64], [16, 0], [32, 0],
    # [1, 0] .. [8, 0]] keep the K^T and V chunk reads free of bank conflicts.
    kv_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 16)],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [1, 0], [2, 0], [4, 0], [8, 0]],
        [BLOCK_N, HEAD_DIM])
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, 2, layout=kv_layout)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, 2, layout=kv_layout)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc0 = tlx.zeros((BLOCK_M, 32), tl.float32, layout=mma)
    acc1 = tlx.zeros((BLOCK_M, 32), tl.float32, layout=mma)
    acc2 = tlx.zeros((BLOCK_M, 32), tl.float32, layout=mma)
    acc3 = tlx.zeros((BLOCK_M, 32), tl.float32, layout=mma)
    if full > lo:
        # Counting from lo: K(0) into slot 0, then K(1) and V(0); step b reads
        # K(b + 1) and V(b) and copies K(b + 2) and V(b + 1) into the slots
        # step b - 1 freed. The K copies past the last unmasked block repeat it.
        kp = k_ptrs + lo * k_step
        vp = v_ptrs + lo * v_step
        last = full - lo - 1
        tok = tlx.async_load(kp, tlx.local_view(k_buf, 0))
        tlx.async_load_commit_group([tok])
        tlx.async_load_wait_group(0)
        tl.debug_barrier()
        tok_k = tlx.async_load(kp + tl.minimum(1, last) * k_step, tlx.local_view(k_buf, 1))
        tok_v = tlx.async_load(vp, tlx.local_view(v_buf, 0))
        tlx.async_load_commit_group([tok_k, tok_v])
        s = _qk(q0, q1, q2, q3, tlx.local_view(k_buf, 0), BLOCK_N)
        m_i, l_i, p, acc0, acc1, acc2, acc3 = _softmax(s, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale,
                                                       Q.dtype.element_ty)
        for blk in tl.range(0, last, num_stages=1):
            cur = blk % 2
            nxt = 1 - cur
            tlx.async_load_wait_group(0)
            tl.debug_barrier()
            tok_k = tlx.async_load(kp + tl.minimum(blk + 2, last) * k_step, tlx.local_view(k_buf, cur))
            tok_v = tlx.async_load(vp + (blk + 1) * v_step, tlx.local_view(v_buf, nxt))
            tlx.async_load_commit_group([tok_k, tok_v])
            s, acc0, acc1, acc2, acc3 = _mfma_stage(q0, q1, q2, q3, p, tlx.local_view(k_buf, nxt),
                                                    tlx.local_view(v_buf, cur), acc0, acc1, acc2, acc3, BLOCK_N)
            m_i, l_i, p, acc0, acc1, acc2, acc3 = _softmax(s, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale,
                                                           Q.dtype.element_ty)
        tlx.async_load_wait_group(0)
        tl.debug_barrier()
        acc0, acc1, acc2, acc3 = _pv(p, tlx.local_view(v_buf, last % 2), acc0, acc1, acc2, acc3)
        tl.debug_barrier()

    for blk in tl.range(full, n_blocks, num_stages=1):
        cols = blk * BLOCK_N + offs_n
        kv_ok = (cols < k_len)[:, None]
        tok_k = tlx.async_load(k_ptrs + blk * k_step, tlx.local_view(k_buf, 0), mask=kv_ok)
        tok_v = tlx.async_load(v_ptrs + blk * v_step, tlx.local_view(v_buf, 0), mask=kv_ok)
        tlx.async_load_commit_group([tok_k, tok_v])
        tlx.async_load_wait_group(0)
        tl.debug_barrier()
        s = _qk(q0, q1, q2, q3, tlx.local_view(k_buf, 0), BLOCK_N)
        valid = cols[None, :] < k_len
        if IS_CAUSAL:
            valid = valid & (cols[None, :] <= offs_m[:, None] + causal_off)
        s = tl.where(valid, s, float("-inf"))
        m_i, l_i, p, acc0, acc1, acc2, acc3 = _softmax(s, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale,
                                                       Q.dtype.element_ty)
        acc0, acc1, acc2, acc3 = _pv(p, tlx.local_view(v_buf, 0), acc0, acc1, acc2, acc3)
        tl.debug_barrier()

    if SPLIT > 0:
        tile = bh * num_m_blocks + m_block
        if split < n_splits - 1:
            _store_partial(Part, item, m_i, l_i, acc0, acc1, acc2, acc3, BLOCK_M, HEAD_DIM)
            tl.debug_barrier()
            tl.atomic_add(Count + tile, 1, sem="release", scope="gpu")
        elif n_splits > 1:
            while tl.atomic_add(Count + tile, 0, sem="relaxed", scope="gpu") < n_splits - 1:
                pass
            tl.atomic_add(Count + tile, 0, sem="acquire", scope="gpu")
            tl.store(Count + tile, 0)
            l_i, acc0, acc1, acc2, acc3 = _fold_partials(Part, item - split, n_splits - 1, m_i, l_i, acc0, acc1, acc2,
                                                         acc3, BLOCK_M, HEAD_DIM)

    if split == n_splits - 1:
        # A row that sees no keys (causal, q_len > k_len) has l_i = 0 and writes 0.
        inv = (1.0 / tl.where(l_i == 0.0, 1.0, l_i))[:, None]
        o_ptrs = Out + (q_start + offs_m)[:, None] * stride_ot + off_h * stride_oh + tl.arange(0, 32)[None, :]
        out_ty = Out.dtype.element_ty
        tl.store(o_ptrs, tlx.release_layout((acc0 * inv).to(out_ty)), mask=row_ok[:, None])
        tl.store(o_ptrs + 32, tlx.release_layout((acc1 * inv).to(out_ty)), mask=row_ok[:, None])
        tl.store(o_ptrs + 64, tlx.release_layout((acc2 * inv).to(out_ty)), mask=row_ok[:, None])
        tl.store(o_ptrs + 96, tlx.release_layout((acc3 * inv).to(out_ty)), mask=row_ok[:, None])


def _run_length(n_items, max_splits):
    # Items per XCD; a run that ends inside a tile extends to the tile's end.
    return triton.cdiv(n_items, 8) + max_splits - 1


@functools.lru_cache(maxsize=None)
def _split_plan(bh, num_m_blocks, cus):
    """``(SPLIT, items per (sequence, head), most splits of a tile)`` of a causal launch; SPLIT 0 is unsplit."""
    # No split fits once the unsplit tiles fill the CUs.
    for split in range(MIN_SPLIT, num_m_blocks if bh * num_m_blocks < cus else 0):
        items_per_bh = sum(m // split + 1 for m in range(num_m_blocks))
        max_splits = (num_m_blocks - 1) // split + 1
        if (_run_length(bh * items_per_bh, max_splits) <= cus // 8
                and bh * (items_per_bh - num_m_blocks) <= MAX_PARTIALS):
            return split, items_per_bh, max_splits
    return 0, num_m_blocks, 1


@functools.lru_cache(maxsize=None)
def _num_cus(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


_free_counters = {}
_scratch = {}


def _split_scratch(device, cus):
    """This stream's fp32 partials and per-tile counters; None if capturing and no zeroed counters are left.

    The counters start at zero and every launch returns them to zero.
    """
    key = (device, triton.runtime.driver.active.get_current_stream(device.index))
    scratch = _scratch.get(key)
    if scratch is None:
        free = _free_counters.setdefault(device, [])
        if not free:
            if torch.cuda.is_current_stream_capturing():
                return None
            free.extend(torch.zeros((COUNTER_ROWS, cus), device=device, dtype=torch.int32).unbind())
        # Per item: accumulator (64 x 128), row max, row sum.
        part = torch.empty((cus, BLOCK_M * (128 + 2)), device=device, dtype=torch.float32)
        scratch = _scratch[key] = (part, free.pop())
    return scratch


def flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, sm_scale, causal):
    """Launch the forward; inputs are validated by ``tlx.ops.flash_attn_varlen``."""
    heads, head_dim = q.shape[1], q.shape[2]
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    out = torch.empty_like(q)
    num_m_blocks = triton.cdiv(max_seqlen_q, BLOCK_M)
    bh = (cu_seqlens_q.shape[0] - 1) * heads
    if bh * num_m_blocks == 0:
        return out
    split, items_per_bh, max_splits = 0, num_m_blocks, 1
    part = count = None
    if causal:
        cus = _num_cus(q.device)
        split, items_per_bh, max_splits = _split_plan(bh, num_m_blocks, cus)
    if split:
        scratch = _split_scratch(q.device, cus)
        if scratch is None:
            split, items_per_bh, max_splits = 0, num_m_blocks, 1
        else:
            part, count = scratch
    n_items = bh * items_per_bh
    grid = (_run_length(n_items, max_splits) * 8, )
    _flash_attn_varlen_fwd[grid](q, k, v, out, cu_seqlens_q, cu_seqlens_k, part, count, q.stride(0), q.stride(1),
                                 k.stride(0), k.stride(1), v.stride(0), v.stride(1), out.stride(0), out.stride(1),
                                 sm_scale * 1.4426950408889634, heads, heads // k.shape[1], num_m_blocks, items_per_bh,
                                 n_items, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim, IS_CAUSAL=causal,
                                 SPLIT=split, num_warps=NUM_WARPS, waves_per_eu=0, matrix_instr_nonkdim=16, kpack=1)
    return out
