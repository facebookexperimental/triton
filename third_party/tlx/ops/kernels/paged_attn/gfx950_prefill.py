"""Paged-KV causal prefill attention for gfx950 (MI350). BF16/FP16, inference only.

Q is packed ``(tokens, q_heads, D)``; sequence ``b`` owns query rows
``cu_seqlens_q[b]:cu_seqlens_q[b + 1]``, which are the last ``q_len`` of its
``context_lens[b]`` keys (any earlier keys are a cached prefix). K/V live in a
paged cache ``(blocks, page_size, kv_heads, D)``, SGLang's paged pool layout,
addressed through
``block_tables[b]``. Query row ``i`` sees keys ``j <= ctx - q_len + i`` and,
with a sliding window ``w > 0``, only ``j > ctx - q_len + i - w``.

Each program computes ``ROWS`` query rows of one sequence for all ``GROUP``
query heads that share one KV head, so every K/V tile it loads feeds
``ROWS * GROUP`` MFMA rows. Key blocks that every row sees in full run
unmasked; only the causal diagonal and the window's lower edge are masked.

``paged_attention_prefill`` is the TLX kernel (head dim 64): Q is staged once
through LDS; K and V are gathered page by page with ``async_load`` into double
buffered LDS; each unmasked step pairs the QK MFMAs of the next key block with
the PV MFMAs of the current one, and the LDS reads and next copies ride with
them in intra-wave warp-pipeline stages. ``paged_attention_prefill_triton`` is
the plain-Triton reference for any head dim.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _load_kv_block(K, V, bt_row, kv_head, blk, lo, ctx, stride_kp, stride_ks, stride_kh, stride_vp, stride_vs,
                   stride_vh, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, PAGE: tl.constexpr, MASKED: tl.constexpr):
    kpos = blk * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    if MASKED:
        # Keys before the tile's earliest window start may sit in recycled pages; never read them.
        ok = (kpos >= lo) & (kpos < ctx)
        page = tl.load(bt_row + kpos // PAGE, mask=ok, other=0).to(tl.int64)
        slot = kpos % PAGE
        k = tl.load(K + (page * stride_kp + slot * stride_ks + kv_head * stride_kh)[:, None] + offs_d[None, :],
                    mask=ok[:, None], other=0.0)
        v = tl.load(V + (page * stride_vp + slot * stride_vs + kv_head * stride_vh)[:, None] + offs_d[None, :],
                    mask=ok[:, None], other=0.0)
    else:
        page = tl.load(bt_row + kpos // PAGE).to(tl.int64)
        slot = kpos % PAGE
        k = tl.load(K + (page * stride_kp + slot * stride_ks + kv_head * stride_kh)[:, None] + offs_d[None, :])
        v = tl.load(V + (page * stride_vp + slot * stride_vs + kv_head * stride_vh)[:, None] + offs_d[None, :])
    return k, v


@triton.jit
def _attend(q, k, v, m_i, l_i, acc, qk_scale, valid, MASKED: tl.constexpr):
    s = tl.dot(q, tl.trans(k)) * qk_scale
    if MASKED:
        s = tl.where(valid, s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    m_use = m_new
    if MASKED:
        # A window-edge block can mask a row entirely before its first visible key.
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.math.exp2(m_i - m_use)
    p = tl.math.exp2(s - m_use[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return m_new, l_i, acc


@triton.jit
def _paged_prefill_kernel(Q, K, V, Out, CuQ, CtxLens, BlockTables, stride_qt, stride_qh, stride_kp, stride_ks,
                          stride_kh, stride_vp, stride_vs, stride_vh, stride_ot, stride_oh, stride_bt, qk_scale, window,
                          num_m_blocks, GROUP: tl.constexpr, ROWS: tl.constexpr, BLOCK_N: tl.constexpr,
                          HEAD_DIM: tl.constexpr, PAGE: tl.constexpr, HAS_WINDOW: tl.constexpr):
    BLOCK_M: tl.constexpr = ROWS * GROUP
    # Causal tiles near the end of a sequence see the most keys; launch them first.
    m_block = num_m_blocks - 1 - tl.program_id(0)
    kv_head = tl.program_id(1)
    b = tl.program_id(2)
    q_start = tl.load(CuQ + b)
    q_len = tl.load(CuQ + b + 1) - q_start
    if m_block * ROWS >= q_len:
        return
    ctx = tl.load(CtxLens + b)
    q_off = ctx - q_len
    bt_row = BlockTables + b.to(tl.int64) * stride_bt

    # MFMA row m is query row m // GROUP of head kv_head * GROUP + m % GROUP.
    offs_m = tl.arange(0, BLOCK_M)
    row = m_block * ROWS + offs_m // GROUP
    head = kv_head * GROUP + offs_m % GROUP
    row_ok = row < q_len
    offs_d = tl.arange(0, HEAD_DIM)
    q_rows = (q_start + row).to(tl.int64)
    q = tl.load(Q + (q_rows * stride_qt + head * stride_qh)[:, None] + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    qpos = q_off + row

    first_q = q_off + m_block * ROWS
    last_q = q_off + tl.minimum((m_block + 1) * ROWS, q_len) - 1
    k_lo = 0
    full_lo = 0
    if HAS_WINDOW:
        k_lo = tl.maximum(first_q - window + 1, 0)
        full_lo = tl.maximum(last_q - window + 1, 0)
    blk_lo = k_lo // BLOCK_N
    blk_hi = last_q // BLOCK_N + 1
    # Blocks [f_lo, f_hi) hold only keys every row of the tile sees.
    f_lo = tl.minimum(tl.maximum(tl.cdiv(full_lo, BLOCK_N), blk_lo), blk_hi)
    f_hi = tl.minimum(tl.maximum((first_q + 1) // BLOCK_N, f_lo), blk_hi)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)

    for blk in range(blk_lo, f_lo):
        k, v = _load_kv_block(K, V, bt_row, kv_head, blk, k_lo, ctx, stride_kp, stride_ks, stride_kh, stride_vp,
                              stride_vs, stride_vh, BLOCK_N, HEAD_DIM, PAGE, True)
        kpos = blk * BLOCK_N + offs_n
        valid = (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] > qpos[:, None] - window)
        m_i, l_i, acc = _attend(q, k, v, m_i, l_i, acc, qk_scale, valid, True)
    for blk in range(f_lo, f_hi):
        k, v = _load_kv_block(K, V, bt_row, kv_head, blk, k_lo, ctx, stride_kp, stride_ks, stride_kh, stride_vp,
                              stride_vs, stride_vh, BLOCK_N, HEAD_DIM, PAGE, False)
        m_i, l_i, acc = _attend(q, k, v, m_i, l_i, acc, qk_scale, None, False)
    for blk in range(f_hi, blk_hi):
        k, v = _load_kv_block(K, V, bt_row, kv_head, blk, k_lo, ctx, stride_kp, stride_ks, stride_kh, stride_vp,
                              stride_vs, stride_vh, BLOCK_N, HEAD_DIM, PAGE, True)
        kpos = blk * BLOCK_N + offs_n
        valid = kpos[None, :] <= qpos[:, None]
        if HAS_WINDOW:
            valid = valid & (kpos[None, :] > qpos[:, None] - window)
        m_i, l_i, acc = _attend(q, k, v, m_i, l_i, acc, qk_scale, valid, True)

    out = acc / l_i[:, None]
    tl.store(Out + (q_rows * stride_ot + head * stride_oh)[:, None] + offs_d[None, :], out.to(Out.dtype.element_ty),
             mask=row_ok[:, None])


def paged_attention_prefill_triton(query, key_cache, value_cache, cu_seqlens_q, context_lens, block_tables,
                                   max_seqlen_q, sm_scale=None, sliding_window=0, out=None, *, rows=32, block_n=64,
                                   num_warps=4, num_stages=2, waves_per_eu=2):
    """Plain-Triton reference launch (no TLX); any head dim the cache supports."""
    num_q_heads, head_dim = query.shape[1], query.shape[2]
    page_size, num_kv_heads = key_cache.shape[1], key_cache.shape[2]
    group = num_q_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    if out is None:
        out = torch.empty_like(query)
    batch = cu_seqlens_q.shape[0] - 1
    num_m_blocks = triton.cdiv(max_seqlen_q, rows)
    if batch == 0 or num_m_blocks == 0:
        return out
    extra = {} if waves_per_eu is None else {"waves_per_eu": waves_per_eu}
    grid = (num_m_blocks, num_kv_heads, batch)
    _paged_prefill_kernel[grid](query, key_cache, value_cache, out, cu_seqlens_q, context_lens, block_tables,
                                query.stride(0), query.stride(1), key_cache.stride(0), key_cache.stride(1),
                                key_cache.stride(2), value_cache.stride(0),
                                value_cache.stride(1), value_cache.stride(2), out.stride(0), out.stride(1),
                                block_tables.stride(0), sm_scale * 1.4426950408889634, sliding_window, num_m_blocks,
                                GROUP=group, ROWS=rows, BLOCK_N=block_n, HEAD_DIM=head_dim, PAGE=page_size,
                                HAS_WINDOW=sliding_window > 0, num_warps=num_warps, num_stages=num_stages, **extra)
    return out


# ---------------------------------------------------------------------------
# TLX kernel: head dim 64, 4 warps, 32x32x16 MFMA (16x16x32 behind mfma32=False).
# ---------------------------------------------------------------------------


@triton.jit
def _read_k(k_slot, col: tl.constexpr, MFMA32: tl.constexpr):
    # 32 head-dim columns of every key in the LDS K block, transposed into the B operand of a QK dot.
    if MFMA32:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 16], transposed=True,
                                                warps_per_cta=[4, 1])
    else:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                warps_per_cta=[4, 1])
    n: tl.constexpr = k_slot.shape[0]
    return tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_slice(k_slot, [0, col], [n, 32])), relaxed=True),
                              tlx.dot_operand_layout(1, mma, k_width=8), pin=False)


@triton.jit
def _read_v(v_slot, col: tl.constexpr, MFMA32: tl.constexpr):
    # 32 head-dim columns of every key in the LDS V block, as the B operand of the PV dot into one accumulator.
    if MFMA32:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 16], transposed=True,
                                                warps_per_cta=[4, 1])
    else:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                warps_per_cta=[4, 1])
    n: tl.constexpr = v_slot.shape[0]
    return tlx.require_layout(tlx.local_load(tlx.local_slice(v_slot, [0, col], [n, 32]), relaxed=True),
                              tlx.dot_operand_layout(1, mma, k_width=4), pin=False)


@triton.jit
def _qk(q0, q1, k_slot, BLOCK_N: tl.constexpr, MFMA32: tl.constexpr):
    if MFMA32:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 16], transposed=True,
                                                warps_per_cta=[4, 1])
    else:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                warps_per_cta=[4, 1])
    s = tlx.zeros((q0.shape[0], BLOCK_N), tl.float32, layout=mma)
    s = tl.dot(q0, _read_k(k_slot, 0, MFMA32), s)
    return tl.dot(q1, _read_k(k_slot, 32, MFMA32), s)


@triton.jit
def _pv(p, v_slot, acc0, acc1, MFMA32: tl.constexpr):
    if MFMA32:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 16], transposed=True,
                                                warps_per_cta=[4, 1])
    else:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                warps_per_cta=[4, 1])
    p = tlx.require_layout(p, tlx.dot_operand_layout(0, mma, k_width=4), pin=False)
    return tl.dot(p, _read_v(v_slot, 0, MFMA32), acc0), tl.dot(p, _read_v(v_slot, 32, MFMA32), acc1)


@triton.jit
def _qk_interleaved(q0, q1, k_slot, v_src, v_dst, BLOCK_N: tl.constexpr, MFMA32: tl.constexpr):
    # QK of the first unmasked block, with the V copy of that block riding the first dot.
    if MFMA32:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 16], transposed=True,
                                                warps_per_cta=[4, 1])
    else:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                warps_per_cta=[4, 1])
    s = tlx.zeros((q0.shape[0], BLOCK_N), tl.float32, layout=mma)
    k0 = _read_k(k_slot, 0, MFMA32)
    with tlx.warp_pipeline_stage("v_copy_read_k1", scope="intra_wave", pair=0):
        tlx.async_load_commit_group([tlx.async_load(v_src, v_dst)])
        k1 = _read_k(k_slot, 32, MFMA32)
    with tlx.warp_pipeline_stage("qk0", scope="intra_wave", pair=0):
        s = tl.dot(q0, k0, s)
    return tl.dot(q1, k1, s)


@triton.jit
def _qk_pv_interleaved(q0, q1, p, k_slot, v_slot, acc0, acc1, v_src, v_dst, k_src, k_dst, BLOCK_N: tl.constexpr,
                       MFMA32: tl.constexpr):
    # QK of the next key block and PV of the current one. Each dot is paired with the LDS read the next
    # dot needs, and the next V and K copies ride with the first two dots.
    if MFMA32:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 16], transposed=True,
                                                warps_per_cta=[4, 1])
    else:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                warps_per_cta=[4, 1])
    p = tlx.require_layout(p, tlx.dot_operand_layout(0, mma, k_width=4), pin=False)
    s = tlx.zeros((q0.shape[0], BLOCK_N), tl.float32, layout=mma)
    k0 = _read_k(k_slot, 0, MFMA32)
    with tlx.warp_pipeline_stage("v_copy_read_v0", scope="intra_wave", pair=0):
        tlx.async_load_commit_group([tlx.async_load(v_src, v_dst)])
        v0 = _read_v(v_slot, 0, MFMA32)
    with tlx.warp_pipeline_stage("qk0", scope="intra_wave", pair=0):
        s = tl.dot(q0, k0, s)
    with tlx.warp_pipeline_stage("k_copy_read_k1", scope="intra_wave", pair=1):
        tlx.async_load_commit_group([tlx.async_load(k_src, k_dst)])
        k1 = _read_k(k_slot, 32, MFMA32)
    with tlx.warp_pipeline_stage("pv0", scope="intra_wave", pair=1):
        acc0 = tl.dot(p, v0, acc0)
    with tlx.warp_pipeline_stage("read_v1", scope="intra_wave", pair=2):
        v1 = _read_v(v_slot, 32, MFMA32)
    with tlx.warp_pipeline_stage("qk1", scope="intra_wave", pair=2):
        s = tl.dot(q1, k1, s)
    acc1 = tl.dot(p, v1, acc1)
    return s, acc0, acc1


@triton.jit
def _softmax(s, m_i, l_i, acc0, acc1, qk_scale, dtype: tl.constexpr):
    # Online softmax on unscaled scores; m_i is in scaled log2 units, so the scale folds into the exponent's FMA.
    m_new = tl.maximum(m_i, tl.max(s, 1) * qk_scale)
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.math.exp2(s * qk_scale - m_safe[:, None])
    alpha = tl.math.exp2(m_i - m_safe)
    l_i = l_i * alpha + tl.sum(p, 1)
    a = alpha[:, None]
    return m_new, l_i, p.to(dtype), acc0 * a, acc1 * a


@triton.jit
def _kv_ptrs(Base, bt_row, kv_head, blk, lo, ctx, stride_p, stride_s, stride_h, BLOCK_N: tl.constexpr,
             HEAD_DIM: tl.constexpr, PAGE: tl.constexpr):
    # Row r of key block blk is token blk * BLOCK_N + r, in slot (token % PAGE) of page block_tables[token // PAGE].
    kpos = blk * BLOCK_N + tl.arange(0, BLOCK_N)
    page = tl.load(bt_row + kpos // PAGE, mask=(kpos >= lo) & (kpos < ctx), other=0).to(tl.int64)
    rows = page * stride_p + (kpos % PAGE) * stride_s + kv_head * stride_h
    return Base + rows[:, None] + tl.arange(0, HEAD_DIM)[None, :]


@triton.jit
def _masked_step(q0, q1, qpos, K, V, k_slot, v_slot, bt_row, kv_head, blk, k_lo, ctx, window, m_i, l_i, acc0, acc1,
                 qk_scale, stride_kp, stride_ks, stride_kh, stride_vp, stride_vs, stride_vh, BLOCK_N: tl.constexpr,
                 HEAD_DIM: tl.constexpr, PAGE: tl.constexpr, HAS_WINDOW: tl.constexpr, dtype: tl.constexpr,
                 MFMA32: tl.constexpr):
    kpos = blk * BLOCK_N + tl.arange(0, BLOCK_N)
    # Keys before the tile's earliest window start may sit in recycled pages; never read them.
    ok = ((kpos >= k_lo) & (kpos < ctx))[:, None]
    k_src = _kv_ptrs(K, bt_row, kv_head, blk, k_lo, ctx, stride_kp, stride_ks, stride_kh, BLOCK_N, HEAD_DIM, PAGE)
    v_src = _kv_ptrs(V, bt_row, kv_head, blk, k_lo, ctx, stride_vp, stride_vs, stride_vh, BLOCK_N, HEAD_DIM, PAGE)
    # Masked rows land as zeros, so unread keys never put NaN into PV.
    tlx.async_load_commit_group([tlx.async_load(k_src, k_slot, mask=ok), tlx.async_load(v_src, v_slot, mask=ok)])
    tlx.async_load_wait_group(0)
    tl.debug_barrier()
    s = _qk(q0, q1, k_slot, BLOCK_N, MFMA32)
    valid = kpos[None, :] <= qpos[:, None]
    if HAS_WINDOW:
        valid = valid & (kpos[None, :] > qpos[:, None] - window)
    s = tl.where(valid, s, float("-inf"))
    m_i, l_i, p, acc0, acc1 = _softmax(s, m_i, l_i, acc0, acc1, qk_scale, dtype)
    acc0, acc1 = _pv(p, v_slot, acc0, acc1, MFMA32)
    tl.debug_barrier()
    return m_i, l_i, acc0, acc1


@triton.jit
def _paged_prefill_tlx_kernel(Q, K, V, Out, CuQ, CtxLens, BlockTables, stride_qt, stride_qh, stride_kp, stride_ks,
                              stride_kh, stride_vp, stride_vs, stride_vh, stride_ot, stride_oh, stride_bt, qk_scale,
                              window, num_kv_heads, num_m_blocks, n_items, GROUP: tl.constexpr, ROWS: tl.constexpr,
                              BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, PAGE: tl.constexpr,
                              HAS_WINDOW: tl.constexpr, SEQ_MAJOR: tl.constexpr, K_LAYOUT: tl.constexpr,
                              V_LAYOUT: tl.constexpr, MFMA32: tl.constexpr):
    tl.static_assert(HEAD_DIM == 64)
    BLOCK_M: tl.constexpr = ROWS * GROUP
    dtype: tl.constexpr = Q.dtype.element_ty
    pid = tl.program_id(0)
    if SEQ_MAJOR:
        # The tiles of one (sequence, KV head) back to back, longest first: with cached prefixes of different
        # lengths a tile's cost no longer follows its index, and adjacent tiles share the prefix in cache.
        if pid >= n_items:
            return
        m_block = num_m_blocks - 1 - pid % num_m_blocks
        bh = pid // num_m_blocks
    else:
        # Without prefixes a tile's cost grows with its index: longest tiles first across every
        # (sequence, KV head), so the causal tail is short.
        if pid >= n_items:
            return
        n_bh = n_items // num_m_blocks
        m_block = num_m_blocks - 1 - pid // n_bh
        bh = pid % n_bh
    kv_head = bh % num_kv_heads
    b = bh // num_kv_heads
    q_start = tl.load(CuQ + b)
    q_len = tl.load(CuQ + b + 1) - q_start
    if m_block * ROWS >= q_len:
        return
    ctx = tl.load(CtxLens + b)
    q_off = ctx - q_len
    bt_row = BlockTables + b.to(tl.int64) * stride_bt

    if MFMA32:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 16], transposed=True,
                                                warps_per_cta=[4, 1])
    else:
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                warps_per_cta=[4, 1])
    # MFMA row m is query row m // GROUP of head kv_head * GROUP + m % GROUP.
    offs_m = tl.arange(0, BLOCK_M)
    row = m_block * ROWS + offs_m // GROUP
    head = kv_head * GROUP + offs_m % GROUP
    row_ok = row < q_len
    q_rows = (q_start + row).to(tl.int64)
    q_base = q_rows * stride_qt + head * stride_qh
    qpos = q_off + row

    first_q = q_off + m_block * ROWS
    last_q = q_off + tl.minimum((m_block + 1) * ROWS, q_len) - 1
    k_lo = 0
    full_lo = 0
    if HAS_WINDOW:
        k_lo = tl.maximum(first_q - window + 1, 0)
        full_lo = tl.maximum(last_q - window + 1, 0)
    blk_lo = k_lo // BLOCK_N
    blk_hi = last_q // BLOCK_N + 1
    # Blocks [f_lo, f_hi) hold only keys every row of the tile sees.
    f_lo = tl.minimum(tl.maximum(tl.cdiv(full_lo, BLOCK_N), blk_lo), blk_hi)
    f_hi = tl.minimum(tl.maximum((first_q + 1) // BLOCK_N, f_lo), blk_hi)

    q_buf = tlx.local_alloc((BLOCK_M, HEAD_DIM), dtype, 1)
    if K_LAYOUT is None:
        k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), dtype, 2)
        v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), dtype, 2)
    else:
        k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), dtype, 2, layout=K_LAYOUT)
        v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), dtype, 2, layout=V_LAYOUT)
    q_slot = tlx.local_view(q_buf, 0)
    tlx.async_load_commit_group(
        [tlx.async_load(Q + q_base[:, None] + tl.arange(0, HEAD_DIM)[None, :], q_slot, mask=row_ok[:, None])])
    tlx.async_load_wait_group(0)
    tl.debug_barrier()
    q = tlx.require_layout(tlx.local_load(q_slot, relaxed=True), tlx.dot_operand_layout(0, mma, k_width=8), pin=False)
    q0 = tlx.extract_slice(q, [BLOCK_M, 32], [0, 0])
    q1 = tlx.extract_slice(q, [BLOCK_M, 32], [0, 32])
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc0 = tlx.zeros((BLOCK_M, 32), tl.float32, layout=mma)
    acc1 = tlx.zeros((BLOCK_M, 32), tl.float32, layout=mma)

    for blk in tl.range(blk_lo, f_lo, num_stages=1):
        m_i, l_i, acc0, acc1 = _masked_step(q0, q1, qpos, K, V, tlx.local_view(k_buf, 0), tlx.local_view(v_buf, 0),
                                            bt_row, kv_head, blk, k_lo, ctx, window, m_i, l_i, acc0, acc1, qk_scale,
                                            stride_kp, stride_ks, stride_kh, stride_vp, stride_vs, stride_vh, BLOCK_N,
                                            HEAD_DIM, PAGE, HAS_WINDOW, dtype, MFMA32)

    if f_hi > f_lo:
        # Counting from f_lo: K(0) into slot 0, then K(1) and V(0); step i reads K(i + 1) and V(i) and copies
        # K(i + 2) and V(i + 1) into the slots step i - 1 freed. Copies past the last block repeat it.
        last = f_hi - f_lo - 1
        tlx.async_load_commit_group([
            tlx.async_load(
                _kv_ptrs(K, bt_row, kv_head, f_lo, k_lo, ctx, stride_kp, stride_ks, stride_kh, BLOCK_N, HEAD_DIM, PAGE),
                tlx.local_view(k_buf, 0))
        ])
        tlx.async_load_commit_group([
            tlx.async_load(
                _kv_ptrs(K, bt_row, kv_head, f_lo + tl.minimum(1, last), k_lo, ctx, stride_kp, stride_ks, stride_kh,
                         BLOCK_N, HEAD_DIM, PAGE), tlx.local_view(k_buf, 1))
        ])
        v_src = _kv_ptrs(V, bt_row, kv_head, f_lo, k_lo, ctx, stride_vp, stride_vs, stride_vh, BLOCK_N, HEAD_DIM, PAGE)
        tlx.async_load_wait_group(1)
        tl.debug_barrier()
        s = _qk_interleaved(q0, q1, tlx.local_view(k_buf, 0), v_src, tlx.local_view(v_buf, 0), BLOCK_N, MFMA32)
        m_i, l_i, p, acc0, acc1 = _softmax(s, m_i, l_i, acc0, acc1, qk_scale, dtype)
        for i in tl.range(0, last, num_stages=1):
            cur = i % 2
            nxt = 1 - cur
            # Block-table loads for the next copies overlap the copies still in flight.
            v_next = _kv_ptrs(V, bt_row, kv_head, f_lo + i + 1, k_lo, ctx, stride_vp, stride_vs, stride_vh, BLOCK_N,
                              HEAD_DIM, PAGE)
            k_next = _kv_ptrs(K, bt_row, kv_head, f_lo + tl.minimum(i + 2, last), k_lo, ctx, stride_kp, stride_ks,
                              stride_kh, BLOCK_N, HEAD_DIM, PAGE)
            tlx.async_load_wait_group(0)
            tl.debug_barrier()
            s, acc0, acc1 = _qk_pv_interleaved(q0, q1, p, tlx.local_view(k_buf, nxt), tlx.local_view(v_buf, cur), acc0,
                                               acc1, v_next, tlx.local_view(v_buf, nxt), k_next,
                                               tlx.local_view(k_buf, cur), BLOCK_N, MFMA32)
            m_i, l_i, p, acc0, acc1 = _softmax(s, m_i, l_i, acc0, acc1, qk_scale, dtype)
        tlx.async_load_wait_group(0)
        tl.debug_barrier()
        acc0, acc1 = _pv(p, tlx.local_view(v_buf, last % 2), acc0, acc1, MFMA32)
        tl.debug_barrier()

    for blk in tl.range(f_hi, blk_hi, num_stages=1):
        m_i, l_i, acc0, acc1 = _masked_step(q0, q1, qpos, K, V, tlx.local_view(k_buf, 0), tlx.local_view(v_buf, 0),
                                            bt_row, kv_head, blk, k_lo, ctx, window, m_i, l_i, acc0, acc1, qk_scale,
                                            stride_kp, stride_ks, stride_kh, stride_vp, stride_vs, stride_vh, BLOCK_N,
                                            HEAD_DIM, PAGE, HAS_WINDOW, dtype, MFMA32)

    inv = (1.0 / tl.where(l_i == 0.0, 1.0, l_i))[:, None]
    o_ptrs = Out + (q_rows * stride_ot + head * stride_oh)[:, None] + tl.arange(0, 32)[None, :]
    out_ty = Out.dtype.element_ty
    tl.store(o_ptrs, tlx.release_layout((acc0 * inv).to(out_ty)), mask=row_ok[:, None])
    tl.store(o_ptrs + 32, tlx.release_layout((acc1 * inv).to(out_ty)), mask=row_ok[:, None])


@lru_cache(maxsize=None)
def _kv_layout(pad):
    """LDS layout of a 64x64 K/V block: rows r, r + 16, r + 32 and r + 48 sit next to each other, and ``pad``
    elements follow every 8 rows, so the 32-wide K^T and V chunk reads spread over the LDS banks."""
    if pad == 0:
        return None
    bases = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0], [2, 0], [4, 0], [8, 0]]
    return tlx.padded_shared_layout_encoding.with_bases([(512, pad)], bases, [64, 64])


def paged_attention_prefill(query, key_cache, value_cache, cu_seqlens_q, context_lens, block_tables, max_seqlen_q,
                            sm_scale=None, sliding_window=0, out=None, *, max_context_len=None, rows=None,
                            waves_per_eu=2, seq_major=None, k_pad=32, v_pad=32, mfma32=True):
    """Launch the TLX forward (head dim 64); inputs are validated by ``tlx.ops.paged_attention_prefill``."""
    num_q_heads, head_dim = query.shape[1], query.shape[2]
    page_size, num_kv_heads = key_cache.shape[1], key_cache.shape[2]
    group = num_q_heads // num_kv_heads
    if rows is None:
        # 256 MFMA rows per K/V tile: 32 query rows of a GQA-8 group.
        rows = max(1, 256 // group)
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    if out is None:
        out = torch.empty_like(query)
    batch = cu_seqlens_q.shape[0] - 1
    num_m_blocks = triton.cdiv(max_seqlen_q, rows)
    n_items = batch * num_kv_heads * num_m_blocks
    if n_items == 0:
        return out
    extra = {} if waves_per_eu is None else {"waves_per_eu": waves_per_eu}
    if seq_major is None:
        seq_major = batch > 1 and max_context_len is not None and max_context_len > max_seqlen_q
    _paged_prefill_tlx_kernel[(n_items, )](query, key_cache, value_cache, out, cu_seqlens_q, context_lens, block_tables,
                                           query.stride(0), query.stride(1), key_cache.stride(0), key_cache.stride(1),
                                           key_cache.stride(2), value_cache.stride(0), value_cache.stride(1),
                                           value_cache.stride(2), out.stride(0), out.stride(1), block_tables.stride(0),
                                           sm_scale * 1.4426950408889634, sliding_window, num_kv_heads, num_m_blocks,
                                           n_items, GROUP=group, ROWS=rows, BLOCK_N=64, HEAD_DIM=head_dim,
                                           PAGE=page_size, HAS_WINDOW=sliding_window > 0, SEQ_MAJOR=seq_major,
                                           K_LAYOUT=_kv_layout(k_pad), V_LAYOUT=_kv_layout(v_pad), MFMA32=mfma32,
                                           num_warps=4, num_stages=1, **extra)
    return out


def ref_prefill(query, key_cache, value_cache, cu_seqlens_q, context_lens, block_tables, sm_scale=None,
                sliding_window=0):
    """FP32 reference: gathers each sequence's pages and applies the causal (and window) mask."""
    num_q_heads, head_dim = query.shape[1], query.shape[2]
    page_size, num_kv_heads = key_cache.shape[1], key_cache.shape[2]
    group = num_q_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    out = torch.empty_like(query)
    cu = cu_seqlens_q.tolist()
    for b, ctx in enumerate(context_lens.tolist()):
        q_len = cu[b + 1] - cu[b]
        if q_len == 0:
            continue
        pages = block_tables[b, :triton.cdiv(ctx, page_size)].long()
        k = key_cache[pages].reshape(-1, num_kv_heads, head_dim)[:ctx].float()
        v = value_cache[pages].reshape(-1, num_kv_heads, head_dim)[:ctx].float()
        k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
        q = query[cu[b]:cu[b + 1]].float()
        s = torch.einsum("qhd,khd->hqk", q, k) * sm_scale
        qpos = torch.arange(ctx - q_len, ctx, device=query.device)[:, None]
        kpos = torch.arange(ctx, device=query.device)[None, :]
        mask = kpos <= qpos
        if sliding_window > 0:
            mask &= kpos > qpos - sliding_window
        s = s.masked_fill(~mask, float("-inf"))
        out[cu[b]:cu[b + 1]] = torch.einsum("hqk,khd->qhd", s.softmax(-1), v).to(query.dtype)
    return out


def build_inputs(q_lens, ctx_lens, num_q_heads, num_kv_heads, head_dim, page_size, dtype=torch.bfloat16, device="cuda",
                 extra_pages=0, seed=0):
    """Random packed Q and a paged cache with each sequence's pages scattered across the pool."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    pages_per_seq = [triton.cdiv(c, page_size) for c in ctx_lens]
    num_pages = sum(pages_per_seq) + extra_pages
    perm = torch.randperm(num_pages, generator=gen)
    max_pages = max(pages_per_seq)
    block_tables = torch.zeros((len(ctx_lens), max_pages), dtype=torch.int32)
    used = 0
    for b, n in enumerate(pages_per_seq):
        block_tables[b, :n] = perm[used:used + n].int()
        used += n
    tokens = sum(q_lens)
    query = (torch.randn((tokens, num_q_heads, head_dim), generator=gen) * 0.5).to(dtype)
    key_cache = torch.randn((num_pages, page_size, num_kv_heads, head_dim), generator=gen).to(dtype)
    value_cache = torch.randn((num_pages, page_size, num_kv_heads, head_dim), generator=gen).to(dtype)
    cu = torch.zeros(len(q_lens) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(q_lens).cumsum(0)
    return (query.to(device), key_cache.to(device), value_cache.to(device), cu.to(device),
            torch.tensor(ctx_lens, dtype=torch.int32, device=device), block_tables.to(device))
