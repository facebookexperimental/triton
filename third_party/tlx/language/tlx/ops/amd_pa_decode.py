"""Production AMD CDNA4 paged-attention decode operator written with TLX.

Two-phase split-K decode, ported from the algorithm in ROCm/aiter
``pa_decode_gluon`` but expressed with Triton + TLX low-level primitives.
Phase 1 (``_pa_decode_partition_kernel``) has each program handle one
``(sequence, kv_head, split)``: it streams the split's KV pages into LDS with
``tlx.async_load`` (double-buffered), does ``q @ k^T`` via MFMA, a base-2
online softmax, then ``p @ v``, and writes a normalized partial output plus a
base-2 log-sum-exp for that split. Phase 2 (``_pa_decode_reduce_kernel``) is a
plain ``@triton.jit`` kernel that merges the per-split partials for each output
``(token, query_head)`` via the standard LSE trick.

For tuned D64 saturated shapes, phase 1 instead vector-loads packed K and V
while QK/softmax remains register-distributed.  Its qlen1/group8
``gluon_compat`` specialization pins Gluon's exact packed load ownership, so
the final K/V MFMA conversions are register-only rather than LDS round trips.

Scope: bf16/fp16 KV cache, GQA, and MTP query_length in 1..4 (causal across the
query positions). Not covered: FP8, sliding window, ALiBi, sinks, per-token
quantization. The kernel accepts either contiguous 4-D K/V caches
``[num_blocks, num_kv_heads, PAGE_SIZE, HEAD_DIM]`` or AITER-compatible packed
5-D caches::

    K: [num_blocks, num_kv_heads, HEAD_DIM / X, PAGE_SIZE, X]
    V: [num_blocks, num_kv_heads, PAGE_SIZE / X, HEAD_DIM, X]

where ``X`` is the number of cache elements in 16 bytes (8 for bf16/fp16).

Consumed by the correctness suite (``test_correctness.py``) and the perf script
(``test_amd_pa_decode_perf.py``).
"""

from dataclasses import dataclass
from functools import lru_cache

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

BUF_DEPTH = tl.constexpr(2)


@lru_cache(maxsize=None)
def _make_qk_softmax_layout(block_n, m_pow2, num_warps=4):
    """Build lane/register ownership for a register-resident ``[M, N]`` QK tile."""
    assert block_n in (64, 128, 256, 512)
    assert m_pow2 in (16, 32, 64)
    assert num_warps in (2, 4)
    thread_column_strides = (4, 8, 16) if num_warps == 2 else (4, 8, 16, 32)
    thread_column_bits = {64: 6, 128: 7, 256: 8, 512: 9}[block_n]
    owned_column_bits = {stride.bit_length() - 1 for stride in thread_column_strides}
    column_register_strides = tuple(1 << bit for bit in range(thread_column_bits) if bit not in owned_column_bits)
    row_register_bits = (m_pow2 // 16).bit_length() - 1
    row_register_strides = tuple((16 << bit) * block_n for bit in range(row_register_bits))
    value_strides = column_register_strides + row_register_strides
    return tlx.layout(
        shape=((2, ) * (4 + len(thread_column_strides)), (2, ) * len(value_strides)),
        stride=((block_n, 2 * block_n, 4 * block_n, 8 * block_n) + thread_column_strides, value_strides),
    )


@lru_cache(maxsize=None)
def _make_gluon_qk_layout(block_n):
    """Gluon's qlen1/group8 blocked score ownership for a four-wave CTA.

    Lanes 0..1 select one of eight query rows, lanes 1..5 span 32
    consecutive score columns, and the four warps own adjacent 2-row groups.
    Each thread retains eight columns in registers for ``BLOCK_N=256``.
    """
    assert block_n == 256
    return tlx.layout(
        shape=((2, 32, 4), (8, )),
        stride=((block_n, 1, 2 * block_n), (32, )),
    )


@lru_cache(maxsize=None)
def _make_gluon_k_load_layout(page_size):
    """Exact Gluon blocked packed-K ownership for page sizes 16 and 64."""
    if page_size == 16:
        return tlx.layout(
            shape=((64, 4), (8, 2, 4)),
            stride=((8, 1024), (1, 512, 4096)),
        )
    assert page_size == 64
    return tlx.layout(
        shape=((16, 4, 4), (8, 2, 4)),
        stride=((8, 512, 128), (1, 2048, 4096)),
    )


@lru_cache(maxsize=None)
def _make_gluon_v_load_layout(page_size):
    """Exact Gluon blocked packed-V ownership for page sizes 16 and 64."""
    if page_size == 16:
        return tlx.layout(
            shape=((16, 4, 4), (8, 2, 4)),
            stride=((8, 1024, 128), (1, 512, 4096)),
        )
    assert page_size == 64
    return tlx.layout(
        shape=((16, 4, 4), (8, 2, 4)),
        stride=((8, 512, 128), (1, 2048, 4096)),
    )


@lru_cache(maxsize=None)
def _make_gluon_shared_layout(vector_size, max_phase):
    return tlx.swizzled_shared_layout_encoding(
        vectorSize=vector_size,
        perPhase=1,
        maxPhase=max_phase,
        order=[1, 0],
        numCTAs=[1, 1],
        numCTAsPerCGA=[1, 1],
        numCTASplit=[1, 1],
        numCTAOrder=[1, 0],
    )


@lru_cache(maxsize=1)
def _make_v_shared_layout():
    return tlx.swizzled_shared_layout_encoding.make_default(2).make_permute((1, 0))


@triton.jit
def _pa_decode_partition_kernel(
    Q,  # [num_tokens, num_q_heads, HEAD_DIM]
    Kc,  # [num_blocks, num_kv_heads, PAGE_SIZE, HEAD_DIM]
    Vc,  # [num_blocks, num_kv_heads, PAGE_SIZE, HEAD_DIM]
    BlockTables,  # [num_seqs, max_pages]
    CtxLens,  # [num_seqs]
    Mid,  # [num_seqs, num_kv_heads, NUM_SPLITS, M_POW2, HEAD_DIM] (fp32)
    Lse,  # [num_seqs, num_kv_heads, NUM_SPLITS, M_POW2] (fp32)
    Out,  # [num_tokens, num_q_heads, HEAD_DIM] (used only when FUSED)
    sm_scale,
    num_splits,  # runtime int (== grid dim 2)
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kc_b,
    stride_kc_h,
    stride_kc_p,
    stride_kc_d,
    stride_kc_x,
    stride_vc_b,
    stride_vc_h,
    stride_vc_p,
    stride_vc_d,
    stride_vc_x,
    stride_bt_s,
    stride_bt_p,
    stride_mid_s,
    stride_mid_h,
    stride_mid_k,
    stride_mid_m,
    stride_mid_d,
    stride_lse_s,
    stride_lse_h,
    stride_lse_k,
    stride_lse_m,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGES_PER_TILE: tl.constexpr,
    BUFFER_DEPTH: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    GROUP_POW2: tl.constexpr,
    QLEN: tl.constexpr,
    QLEN_POW2: tl.constexpr,
    M_POW2: tl.constexpr,
    FUSED: tl.constexpr,
    CACHE_5D: tl.constexpr,
    CACHE_PACK_X: tl.constexpr,
    V_LAYOUT: tl.constexpr,
    QK_SOFTMAX_LAYOUT: tl.constexpr,
    STREAMING_KV: tl.constexpr,
    STREAM_WARPS: tl.constexpr,
    GLUON_COMPAT: tl.constexpr,
    GLUON_K_LOAD_LAYOUT: tl.constexpr,
    GLUON_V_LOAD_LAYOUT: tl.constexpr,
    GLUON_Q_SHARED_LAYOUT: tl.constexpr,
    GLUON_P_SHARED_LAYOUT: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)

    ctx_len = tl.load(CtxLens + seq)
    num_pages = tl.cdiv(ctx_len, PAGE_SIZE)
    pages_per_split = tl.cdiv(num_pages, num_splits)
    start_page = split * pages_per_split
    end_page = tl.minimum(num_pages, start_page + pages_per_split)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_g = tl.arange(0, GROUP_POW2)
    offs_ql = tl.arange(0, QLEN_POW2)
    offs_n = tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, M_POW2)

    # Load Q for this (seq, kv_head): [QLEN_POW2, GROUP_POW2, HEAD_DIM].
    q_head = kv_head * QUERY_GROUP_SIZE + offs_g  # [GROUP_POW2]
    q_tok = seq * QLEN + offs_ql  # [QLEN_POW2]
    q_ptrs = (Q + q_tok[:, None, None] * stride_q_t + q_head[None, :, None] * stride_q_h +
              offs_d[None, None, :] * stride_q_d)
    q_mask = (offs_ql[:, None, None] < QLEN) & (offs_g[None, :, None] < QUERY_GROUP_SIZE)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    q = tl.reshape(q, (M_POW2, HEAD_DIM))

    QK_SCALE = sm_scale * 1.44269504089  # 1/log(2), for exp2-based softmax
    # The general path pre-scales Q once.  The compatibility path deliberately
    # follows Gluon and scales the score tile after QK.
    if GLUON_COMPAT:
        q = q.to(Kc.dtype.element_ty)
    else:
        q = (q.to(tl.float32) * QK_SCALE).to(Kc.dtype.element_ty)
    m_qpos = offs_m // GROUP_POW2  # query position per row
    vis_limit = ctx_len - QLEN  # min visible abs key index over rows (m_qpos >= 0)

    m_i = tl.full([M_POW2], float("-inf"), tl.float32)
    l_i = tl.zeros([M_POW2], tl.float32)
    if STREAMING_KV:
        # QK softmax stays distributed across lanes/waves, avoiding a full
        # score-tile LDS spill.  GLUON_COMPAT additionally pins the packed K/V
        # load ownership so their MFMA conversions require no LDS traffic.
        mfma_k: tl.constexpr = 16 if GLUON_COMPAT else 32
        pv_k_width: tl.constexpr = 8 if GLUON_COMPAT else 16
        qk_mfma_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, mfma_k], transposed=True,
                                                           warps_per_cta=[1, STREAM_WARPS])
        qk_lhs_layout: tl.constexpr = tlx.dot_operand_layout(0, qk_mfma_layout, k_width=8)
        qk_rhs_layout: tl.constexpr = tlx.dot_operand_layout(1, qk_mfma_layout, k_width=8)
        pv_mfma_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, mfma_k], transposed=True,
                                                           warps_per_cta=[1, STREAM_WARPS])
        pv_lhs_layout: tl.constexpr = tlx.dot_operand_layout(0, pv_mfma_layout, k_width=pv_k_width)
        pv_rhs_layout: tl.constexpr = tlx.dot_operand_layout(1, pv_mfma_layout, k_width=pv_k_width)
        # Cooperative packed-V VMEM ownership. Lanes span D, warps span
        # adjacent token groups, and each lane owns the contiguous X=8 bf16
        # values. The two-wave N128 ownership is the short-context qlen1 path;
        # qlen4 uses four waves and the matching four-group register extent.
        if BLOCK_N == 128 and STREAM_WARPS == 2:
            v_load_layout: tl.constexpr = tlx.layout(shape=((64, 2), (8, 8)), stride=((8, 512), (1, 1024)))
        elif BLOCK_N == 128:
            v_load_layout: tl.constexpr = tlx.layout(shape=((64, 4), (8, 4)), stride=((8, 512), (1, 2048)))
        elif BLOCK_N == 64:
            v_load_layout: tl.constexpr = tlx.layout(shape=((64, 4), (8, 2)), stride=((8, 512), (1, 2048)))
        else:
            v_load_layout: tl.constexpr = tlx.layout(shape=((64, 4), (8, 8)), stride=((8, 512), (1, 2048)))
        qk_softmax_layout: tl.constexpr = QK_SOFTMAX_LAYOUT
        if GLUON_COMPAT:
            q_buf = tlx.local_alloc((M_POW2, HEAD_DIM), Kc.dtype.element_ty, 1, layout=GLUON_Q_SHARED_LAYOUT)
            p_buf = tlx.local_alloc((M_POW2, BLOCK_N), Kc.dtype.element_ty, 1, layout=GLUON_P_SHARED_LAYOUT)
            tlx.local_store(tlx.local_view(q_buf, 0), q)
            q_register = tlx.local_load(tlx.local_view(q_buf, 0), layout=qk_lhs_layout)
        else:
            q_register = tlx.require_layout(q, qk_lhs_layout, pin=False)
        acc = tlx.zeros((M_POW2, HEAD_DIM), tl.float32, layout=pv_mfma_layout)
    else:
        acc = tl.zeros([M_POW2, HEAD_DIM], tl.float32)
        k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), Kc.dtype.element_ty, BUFFER_DEPTH)
        if CACHE_5D:
            # Packed V is physically contiguous in X-token groups, so pin the LDS
            # order to logical N,D. This lets the async-copy coalescer issue 16-byte
            # direct-to-LDS loads even though two copies share this double buffer.
            v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), Vc.dtype.element_ty, BUFFER_DEPTH, layout=V_LAYOUT)
        else:
            v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), Vc.dtype.element_ty, BUFFER_DEPTH)

    # Decouple the KV compute tile (BLOCK_N keys) from PAGE_SIZE: row r of a tile
    # maps to logical page (tile_page0 + r // PAGE_SIZE) at offset (r % PAGE_SIZE),
    # gathered in one load so the MFMA K-dim stays BLOCK_N-wide even at small
    # PAGE_SIZE. The LDS path below uses a depth-2 software pipeline; the packed
    # register path relies on instruction scheduling across current K/V/QK/PV.
    row_page = offs_n // PAGE_SIZE
    row_in_page = offs_n % PAGE_SIZE
    num_tiles = tl.cdiv(end_page - start_page, PAGES_PER_TILE)

    if STREAMING_KV and num_tiles > 0:
        offs_tile_page = tl.arange(0, PAGES_PER_TILE)
        offs_head_group = tl.arange(0, HEAD_DIM // CACHE_PACK_X)
        offs_value_group = tl.arange(0, PAGE_SIZE // CACHE_PACK_X)
        offs_page = tl.arange(0, PAGE_SIZE)
        offs_x = tl.arange(0, CACHE_PACK_X)
        if GLUON_COMPAT:
            # Match Gluon's loop-carried K pipeline: fetch tile zero before the
            # loop, then fetch K[i+1] after softmax and before the current PV.
            tile_pages = start_page + offs_tile_page
            tile_physical = tl.load(BlockTables + seq * stride_bt_s +
                                    tl.where(tile_pages < end_page, tile_pages, end_page - 1) * stride_bt_p)
            k_offsets_4d = (tile_physical[:, None, None, None] * stride_kc_b + kv_head * stride_kc_h +
                            offs_head_group[None, :, None, None] * stride_kc_d +
                            offs_page[None, None, :, None] * stride_kc_p +
                            offs_x[None, None, None, :] * stride_kc_x)
            k_offsets_4d = tl.multiple_of(k_offsets_4d, (1, 1, 1, CACHE_PACK_X))
            k_offsets_4d = tl.max_contiguous(k_offsets_4d, (1, 1, 1, CACHE_PACK_X))
            k_offsets_4d = tlx.require_layout(k_offsets_4d.to(tl.int32), GLUON_K_LOAD_LAYOUT)
            k_raw = tlx.buffer_load(Kc, k_offsets_4d, cache=".cg")
        for tidx in tl.range(0, num_tiles):
            tile_page0 = start_page + tidx * PAGES_PER_TILE
            if not GLUON_COMPAT:
                tile_pages = tile_page0 + offs_tile_page
                tile_physical = tl.load(BlockTables + seq * stride_bt_s +
                                        tl.where(tile_pages < end_page, tile_pages, end_page - 1) * stride_bt_p)
                # Keep the native packed cache axes through the global load so X=8
                # bf16 values form one 16-byte vector. Flattening to [D,N] before
                # this point makes the MFMA ownership scalarize VMEM.
                k_offsets_4d = (tile_physical[:, None, None, None] * stride_kc_b + kv_head * stride_kc_h +
                                offs_head_group[None, :, None, None] * stride_kc_d +
                                offs_page[None, None, :, None] * stride_kc_p +
                                offs_x[None, None, None, :] * stride_kc_x)
                k_offsets_4d = tl.multiple_of(k_offsets_4d, (1, 1, 1, CACHE_PACK_X))
                k_offsets_4d = tl.max_contiguous(k_offsets_4d, (1, 1, 1, CACHE_PACK_X))
                if QLEN == 4:
                    # At M=32/N=128, inheriting QK-dot ownership at the load
                    # scalarizes packed X. Preserve the coalesced packed ownership
                    # through VMEM, then convert the loaded registers to QK RHS.
                    k_offsets_4d = tlx.require_layout(k_offsets_4d.to(tl.int32), v_load_layout)
                    k_raw = tlx.buffer_load(Kc, k_offsets_4d, cache=".cg")
                else:
                    k_raw = tlx.buffer_load(Kc, k_offsets_4d.to(tl.int32), cache=".cg")
            kt = tl.permute(k_raw, (1, 3, 0, 2))
            kt = tl.reshape(kt, (HEAD_DIM, BLOCK_N))
            kt = tlx.require_layout(kt, qk_rhs_layout, pin=False)

            # Issue the packed V vector load before QK, matching the Gluon
            # schedule so VMEM can overlap the current QK MFMA/softmax. In
            # logical [N, D] order adjacent D values are strided by X; keeping
            # [page, token_group, D, X] through VMEM preserves 16-byte loads.
            v_offsets_4d = (tile_physical[:, None, None, None] * stride_vc_b + kv_head * stride_vc_h +
                            offs_value_group[None, :, None, None] * stride_vc_p +
                            offs_d[None, None, :, None] * stride_vc_d + offs_x[None, None, None, :] * stride_vc_x)
            v_offsets_4d = tl.multiple_of(v_offsets_4d, (1, 1, 1, CACHE_PACK_X))
            v_offsets_4d = tl.max_contiguous(v_offsets_4d, (1, 1, 1, CACHE_PACK_X))
            if GLUON_COMPAT:
                v_offsets_4d = tlx.require_layout(v_offsets_4d.to(tl.int32), GLUON_V_LOAD_LAYOUT)
            else:
                v_offsets_4d = tlx.require_layout(v_offsets_4d.to(tl.int32), v_load_layout)
            v_raw = tlx.buffer_load(Vc, v_offsets_4d, cache=".cg")
            v = tl.permute(v_raw, (0, 1, 3, 2))
            v = tl.reshape(v, (BLOCK_N, HEAD_DIM))
            v = tlx.require_layout(v, pv_rhs_layout, pin=False)

            qk_acc = tlx.zeros((M_POW2, BLOCK_N), tl.float32, layout=qk_mfma_layout)
            qk = tl.dot(q_register, kt, acc=qk_acc, out_dtype=tl.float32)
            qk = tlx.require_layout(qk, qk_softmax_layout, pin=False)
            if GLUON_COMPAT:
                qk = qk * QK_SCALE

            tile_max_abs = tile_page0 * PAGE_SIZE + (BLOCK_N - 1)
            is_full = (tile_max_abs <= vis_limit) & ((tile_page0 + PAGES_PER_TILE) <= end_page)
            if is_full:
                qks = qk
            else:
                page_ok = (tile_page0 + row_page) < end_page
                kt_abs = tile_page0 * PAGE_SIZE + offs_n
                vis = page_ok[None, :] & (kt_abs[None, :] <= (vis_limit + m_qpos[:, None]))
                qks = tl.where(vis, qk, float("-inf"))
            qks = tlx.require_layout(qks, qk_softmax_layout, pin=False)

            # Reduce register-local column bits, then same-row lanes, then the
            # four wave partials. Only row scalars cross waves; the QK tile never
            # makes a round trip through LDS.
            if GLUON_COMPAT:
                m_ij = tl.max(qks, axis=1)
            elif BLOCK_N == 64:
                qk_reduce = tl.reshape(qks, (M_POW2, 4, 4, 4))
                qk_reduce = tlx.require_layout(qk_reduce, qk_softmax_layout, pin=False)
                qk_reduce = tl.max(qk_reduce, axis=3)
            elif BLOCK_N == 128:
                qk_reduce = tl.reshape(qks, (M_POW2, 2, 4, 4, 4))
                qk_reduce = tlx.require_layout(qk_reduce, qk_softmax_layout, pin=False)
                qk_reduce = tl.max(qk_reduce, axis=4)
                qk_reduce = tl.max(qk_reduce, axis=1)
            elif BLOCK_N == 256:
                qk_reduce = tl.reshape(qks, (M_POW2, 4, 4, 4, 4))
                qk_reduce = tlx.require_layout(qk_reduce, qk_softmax_layout, pin=False)
                qk_reduce = tl.max(qk_reduce, axis=4)
                qk_reduce = tl.max(qk_reduce, axis=1)
            else:
                qk_reduce = tl.reshape(qks, (M_POW2, 8, 4, 4, 4))
                qk_reduce = tlx.require_layout(qk_reduce, qk_softmax_layout, pin=False)
                qk_reduce = tl.max(qk_reduce, axis=4)
                qk_reduce = tl.max(qk_reduce, axis=1)
            if not GLUON_COMPAT:
                qk_reduce = tl.max(qk_reduce, axis=2)
                m_ij = tl.max(qk_reduce, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            m_new_qk = tl.broadcast_to(m_new[:, None], (M_POW2, BLOCK_N))
            m_new_qk = tlx.require_layout(m_new_qk, qk_softmax_layout, pin=False)
            p = tl.math.exp2(qks - m_new_qk)
            alpha = tl.math.exp2(m_i - m_new)

            if GLUON_COMPAT:
                p_sum = tl.sum(p, axis=1)
            elif BLOCK_N == 64:
                p_reduce = tl.reshape(p, (M_POW2, 4, 4, 4))
                p_reduce = tlx.require_layout(p_reduce, qk_softmax_layout, pin=False)
                p_reduce = tl.sum(p_reduce, axis=3)
            elif BLOCK_N == 128:
                p_reduce = tl.reshape(p, (M_POW2, 2, 4, 4, 4))
                p_reduce = tlx.require_layout(p_reduce, qk_softmax_layout, pin=False)
                p_reduce = tl.sum(p_reduce, axis=4)
                p_reduce = tl.sum(p_reduce, axis=1)
            elif BLOCK_N == 256:
                p_reduce = tl.reshape(p, (M_POW2, 4, 4, 4, 4))
                p_reduce = tlx.require_layout(p_reduce, qk_softmax_layout, pin=False)
                p_reduce = tl.sum(p_reduce, axis=4)
                p_reduce = tl.sum(p_reduce, axis=1)
            else:
                p_reduce = tl.reshape(p, (M_POW2, 8, 4, 4, 4))
                p_reduce = tlx.require_layout(p_reduce, qk_softmax_layout, pin=False)
                p_reduce = tl.sum(p_reduce, axis=4)
                p_reduce = tl.sum(p_reduce, axis=1)
            if not GLUON_COMPAT:
                p_reduce = tl.sum(p_reduce, axis=2)
                p_sum = tl.sum(p_reduce, axis=1)
            l_i = l_i * alpha + p_sum

            if GLUON_COMPAT:
                tlx.local_store(tlx.local_view(p_buf, 0), p.to(Vc.dtype.element_ty))
                p_operand = tlx.local_load(tlx.local_view(p_buf, 0), layout=pv_lhs_layout)
                # Keep next-K below softmax while allowing the backend to braid
                # its VMEM instructions through the following PV MFMA sequence.
                if tidx + 1 < num_tiles:
                    tlx.amd_sched_barrier()
                    next_tile_page0 = tile_page0 + PAGES_PER_TILE
                    next_tile_pages = next_tile_page0 + offs_tile_page
                    next_tile_physical = tl.load(
                        BlockTables + seq * stride_bt_s +
                        tl.where(next_tile_pages < end_page, next_tile_pages, end_page - 1) * stride_bt_p)
                    next_k_offsets_4d = (
                        next_tile_physical[:, None, None, None] * stride_kc_b + kv_head * stride_kc_h +
                        offs_head_group[None, :, None, None] * stride_kc_d +
                        offs_page[None, None, :, None] * stride_kc_p +
                        offs_x[None, None, None, :] * stride_kc_x)
                    next_k_offsets_4d = tl.multiple_of(next_k_offsets_4d, (1, 1, 1, CACHE_PACK_X))
                    next_k_offsets_4d = tl.max_contiguous(next_k_offsets_4d, (1, 1, 1, CACHE_PACK_X))
                    next_k_offsets_4d = tlx.require_layout(next_k_offsets_4d.to(tl.int32), GLUON_K_LOAD_LAYOUT)
                    k_raw = tlx.buffer_load(Kc, next_k_offsets_4d, cache=".cg")
                    tile_physical = next_tile_physical
            else:
                p_operand = tlx.require_layout(p.to(Vc.dtype.element_ty), pv_lhs_layout, pin=False)
            pv_acc = tlx.zeros((M_POW2, HEAD_DIM), tl.float32, layout=pv_mfma_layout)
            pv = tl.dot(p_operand, v, acc=pv_acc, out_dtype=tl.float32)
            alpha_acc = tlx.require_layout(alpha[:, None], pv_mfma_layout, pin=False)
            acc = acc * alpha_acc + pv
            m_i = m_new

    if not STREAMING_KV and num_tiles > 0:
        physical = tl.load(BlockTables + seq * stride_bt_s +
                           tl.where(row_page < end_page - start_page, start_page + row_page, end_page - 1) *
                           stride_bt_p)
        if CACHE_5D:
            k_ptrs = (Kc + physical[:, None] * stride_kc_b + kv_head * stride_kc_h +
                      (offs_d[None, :] // CACHE_PACK_X) * stride_kc_d + row_in_page[:, None] * stride_kc_p +
                      (offs_d[None, :] % CACHE_PACK_X) * stride_kc_x)
            v_ptrs = (Vc + physical[:, None] * stride_vc_b + kv_head * stride_vc_h +
                      (row_in_page[:, None] // CACHE_PACK_X) * stride_vc_p + offs_d[None, :] * stride_vc_d +
                      (row_in_page[:, None] % CACHE_PACK_X) * stride_vc_x)
            # Loading the page table hides these facts from AxisInfo: within
            # each packed subgroup K is contiguous in D and V is contiguous in
            # N. Restore the 16-byte vector/alignment facts for direct-to-LDS.
            k_ptrs = tl.multiple_of(k_ptrs, (2, 16))
            k_ptrs = tl.max_contiguous(k_ptrs, (1, CACHE_PACK_X))
            v_ptrs = tl.multiple_of(v_ptrs, (16, 2))
            v_ptrs = tl.max_contiguous(v_ptrs, (CACHE_PACK_X, 1))
        else:
            k_ptrs = (Kc + physical[:, None] * stride_kc_b + kv_head * stride_kc_h +
                      row_in_page[:, None] * stride_kc_p + offs_d[None, :] * stride_kc_d)
            v_ptrs = (Vc + physical[:, None] * stride_vc_b + kv_head * stride_vc_h +
                      row_in_page[:, None] * stride_vc_p + offs_d[None, :] * stride_vc_d)
        tok_k = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0))
        tok_v = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0))
        tlx.async_load_commit_group([tok_k, tok_v])

        for tidx in tl.range(0, num_tiles):
            slot = tidx % BUFFER_DEPTH
            nxt = tidx + 1
            if nxt < num_tiles:
                nslot = nxt % BUFFER_DEPTH
                n_page_of_row = start_page + nxt * PAGES_PER_TILE + row_page
                n_physical = tl.load(BlockTables + seq * stride_bt_s +
                                     tl.where(n_page_of_row < end_page, n_page_of_row, end_page - 1) * stride_bt_p)
                if CACHE_5D:
                    nk_ptrs = (Kc + n_physical[:, None] * stride_kc_b + kv_head * stride_kc_h +
                               (offs_d[None, :] // CACHE_PACK_X) * stride_kc_d + row_in_page[:, None] * stride_kc_p +
                               (offs_d[None, :] % CACHE_PACK_X) * stride_kc_x)
                    nv_ptrs = (Vc + n_physical[:, None] * stride_vc_b + kv_head * stride_vc_h +
                               (row_in_page[:, None] // CACHE_PACK_X) * stride_vc_p + offs_d[None, :] * stride_vc_d +
                               (row_in_page[:, None] % CACHE_PACK_X) * stride_vc_x)
                    nk_ptrs = tl.multiple_of(nk_ptrs, (2, 16))
                    nk_ptrs = tl.max_contiguous(nk_ptrs, (1, CACHE_PACK_X))
                    nv_ptrs = tl.multiple_of(nv_ptrs, (16, 2))
                    nv_ptrs = tl.max_contiguous(nv_ptrs, (CACHE_PACK_X, 1))
                else:
                    nk_ptrs = (Kc + n_physical[:, None] * stride_kc_b + kv_head * stride_kc_h +
                               row_in_page[:, None] * stride_kc_p + offs_d[None, :] * stride_kc_d)
                    nv_ptrs = (Vc + n_physical[:, None] * stride_vc_b + kv_head * stride_vc_h +
                               row_in_page[:, None] * stride_vc_p + offs_d[None, :] * stride_vc_d)
                ntok_k = tlx.async_load(nk_ptrs, tlx.local_view(k_buf, nslot))
                ntok_v = tlx.async_load(nv_ptrs, tlx.local_view(v_buf, nslot))
                tlx.async_load_commit_group([ntok_k, ntok_v])
                tlx.async_load_wait_group(1)
            else:
                tlx.async_load_wait_group(0)

            kt = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, slot)))
            v = tlx.local_load(tlx.local_view(v_buf, slot))

            tile_page0 = start_page + tidx * PAGES_PER_TILE
            qk = tl.dot(q, kt)  # q pre-scaled -> qk already in log2 units
            # An interior tile fully at/below the causal limit skips
            # the per-element visibility compare + select; only boundary tiles pay.
            tile_max_abs = tile_page0 * PAGE_SIZE + (BLOCK_N - 1)
            is_full = (tile_max_abs <= vis_limit) & ((tile_page0 + PAGES_PER_TILE) <= end_page)
            if is_full:
                qks = qk
            else:
                page_ok = (tile_page0 + row_page) < end_page
                kt_abs = tile_page0 * PAGE_SIZE + offs_n
                vis = page_ok[None, :] & (kt_abs[None, :] <= (vis_limit + m_qpos[:, None]))
                qks = tl.where(vis, qk, float("-inf"))
            m_ij = tl.max(qks, 1)
            m_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp2(qks - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

    has_kv = l_i > 0.0
    if STREAMING_KV:
        has_kv_acc = tlx.require_layout(has_kv[:, None], pv_mfma_layout, pin=False)
        denom = tl.where(has_kv[:, None], l_i[:, None], 1.0)
        denom_acc = tlx.require_layout(denom, pv_mfma_layout, pin=False)
        zero_acc = tlx.zeros((M_POW2, HEAD_DIM), tl.float32, layout=pv_mfma_layout)
        o_part = tl.where(has_kv_acc, acc / denom_acc, zero_acc)
    else:
        o_part = tl.where(has_kv[:, None], acc / tl.where(has_kv[:, None], l_i[:, None], 1.0), 0.0)

    if FUSED:
        # Single split -> o_part is already the final normalized output, so
        # write it straight to Out and skip the Mid/LSE round-trip + reduce launch.
        qpos_m = offs_m // GROUP_POW2
        hgrp_m = offs_m % GROUP_POW2
        valid_m = (qpos_m < QLEN) & (hgrp_m < QUERY_GROUP_SIZE)
        gt_m = seq * QLEN + qpos_m
        qh_m = kv_head * QUERY_GROUP_SIZE + hgrp_m
        out_ptrs = (Out + gt_m[:, None] * stride_o_t + qh_m[:, None] * stride_o_h + offs_d[None, :] * stride_o_d)
        if STREAMING_KV:
            out_ptrs = tlx.require_layout(out_ptrs, pv_mfma_layout, pin=False)
            valid_m = tlx.require_layout(valid_m[:, None], pv_mfma_layout, pin=False)
            tl.store(out_ptrs, o_part.to(Out.dtype.element_ty), mask=valid_m)
        else:
            tl.store(out_ptrs, o_part.to(Out.dtype.element_ty), mask=valid_m[:, None])
    else:
        # Store the normalized partial output + base-2 lse for this split.
        lse_part = tl.where(has_kv, m_i + tl.math.log2(tl.where(has_kv, l_i, 1.0)), float("-inf"))
        mid_ptrs = (Mid + seq * stride_mid_s + kv_head * stride_mid_h + split * stride_mid_k +
                    offs_m[:, None] * stride_mid_m + offs_d[None, :] * stride_mid_d)
        if STREAMING_KV:
            mid_ptrs = tlx.require_layout(mid_ptrs, pv_mfma_layout, pin=False)
        tl.store(mid_ptrs, o_part)
        lse_ptrs = Lse + seq * stride_lse_s + kv_head * stride_lse_h + split * stride_lse_k + offs_m * stride_lse_m
        tl.store(lse_ptrs, lse_part)


@triton.jit
def _pa_decode_reduce_kernel(
    Out,  # [num_tokens, num_q_heads, HEAD_DIM]
    Mid,  # [num_seqs, num_kv_heads, NUM_SPLITS, M_POW2, HEAD_DIM]
    Lse,  # [num_seqs, num_kv_heads, NUM_SPLITS, M_POW2]
    num_splits,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_mid_s,
    stride_mid_h,
    stride_mid_k,
    stride_mid_m,
    stride_mid_d,
    stride_lse_s,
    stride_lse_h,
    stride_lse_k,
    stride_lse_m,
    HEAD_DIM: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    GROUP_POW2: tl.constexpr,
    QLEN: tl.constexpr,
    SPLITS_POW2: tl.constexpr,
):
    gt = tl.program_id(0)  # global token = seq * QLEN + qpos
    qh = tl.program_id(1)  # query head

    seq = gt // QLEN
    qpos = gt % QLEN
    kv_head = qh // QUERY_GROUP_SIZE
    hgrp = qh % QUERY_GROUP_SIZE
    m_row = qpos * GROUP_POW2 + hgrp

    offs_k = tl.arange(0, SPLITS_POW2)
    offs_d = tl.arange(0, HEAD_DIM)
    kmask = offs_k < num_splits

    lse = tl.load(
        Lse + seq * stride_lse_s + kv_head * stride_lse_h + offs_k * stride_lse_k + m_row * stride_lse_m,
        mask=kmask,
        other=float("-inf"),
    )
    gmax = tl.max(lse, 0)
    gmax_safe = tl.where(gmax == float("-inf"), 0.0, gmax)
    w = tl.where(kmask, tl.math.exp2(lse - gmax_safe), 0.0)  # [SPLITS_POW2]
    wsum = tl.sum(w, 0)

    o = tl.load(
        Mid + seq * stride_mid_s + kv_head * stride_mid_h + offs_k[:, None] * stride_mid_k + m_row * stride_mid_m +
        offs_d[None, :] * stride_mid_d,
        mask=kmask[:, None],
        other=0.0,
    )  # [SPLITS_POW2, HEAD_DIM]
    out = tl.sum(o * w[:, None], 0) / tl.where(wsum > 0, wsum, 1.0)

    tl.store(Out + gt * stride_o_t + qh * stride_o_h + offs_d * stride_o_d, out.to(Out.dtype.element_ty))


def _next_pow2(x):
    return 1 << (max(1, x) - 1).bit_length()


@triton.jit
def _reshape_and_cache_5d_kernel(
    Key,  # [num_tokens, num_kv_heads, HEAD_DIM]
    Value,  # [num_tokens, num_kv_heads, HEAD_DIM]
    KeyCache,  # [num_blocks, num_kv_heads, HEAD_DIM / X, PAGE_SIZE, X]
    ValueCache,  # [num_blocks, num_kv_heads, PAGE_SIZE / X, HEAD_DIM, X]
    SlotMapping,  # [num_tokens], slot = physical_block * PAGE_SIZE + offset
    num_blocks,
    num_kv_heads,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_kcb,
    stride_kch,
    stride_kcd,
    stride_kcp,
    stride_kcx,
    stride_vcb,
    stride_vch,
    stride_vcp,
    stride_vcd,
    stride_vcx,
    stride_slot,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CACHE_PACK_X: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CONTIGUOUS_INNER: tl.constexpr,
    CONTIGUOUS_CACHE: tl.constexpr,
):
    token = tl.program_id(0)
    kv_head = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    slot = tl.load(SlotMapping + token * stride_slot)
    physical_block = slot // PAGE_SIZE
    page_offset = slot % PAGE_SIZE

    offs_d = tl.arange(0, BLOCK_D)
    valid_slot = (slot >= 0) & (physical_block < num_blocks)
    mask = (kv_head[:, None] < num_kv_heads) & (offs_d[None, :] < HEAD_DIM) & valid_slot

    if CONTIGUOUS_INNER:
        source_inner = kv_head[:, None] * HEAD_DIM + offs_d[None, :]
        key_src = Key + token * stride_kt + source_inner
        value_src = Value + token * stride_vt + source_inner
    else:
        key_src = Key + token * stride_kt + kv_head[:, None] * stride_kh + offs_d[None, :] * stride_kd
        value_src = Value + token * stride_vt + kv_head[:, None] * stride_vh + offs_d[None, :] * stride_vd
    key = tl.load(key_src, mask=mask, other=0.0)
    value = tl.load(value_src, mask=mask, other=0.0)

    if CONTIGUOUS_CACHE:
        block_stride = num_kv_heads * HEAD_DIM * PAGE_SIZE
        head_stride = HEAD_DIM * PAGE_SIZE
        key_dst = (KeyCache + physical_block * block_stride + kv_head[:, None] * head_stride +
                   (offs_d[None, :] // CACHE_PACK_X) * PAGE_SIZE * CACHE_PACK_X + page_offset * CACHE_PACK_X +
                   (offs_d[None, :] % CACHE_PACK_X))
        value_dst = (ValueCache + physical_block * block_stride + kv_head[:, None] * head_stride +
                     (page_offset // CACHE_PACK_X) * HEAD_DIM * CACHE_PACK_X + offs_d[None, :] * CACHE_PACK_X +
                     (page_offset % CACHE_PACK_X))
    else:
        key_dst = (KeyCache + physical_block * stride_kcb + kv_head[:, None] * stride_kch +
                   (offs_d[None, :] // CACHE_PACK_X) * stride_kcd + page_offset * stride_kcp +
                   (offs_d[None, :] % CACHE_PACK_X) * stride_kcx)
        value_dst = (ValueCache + physical_block * stride_vcb + kv_head[:, None] * stride_vch +
                     (page_offset // CACHE_PACK_X) * stride_vcp + offs_d[None, :] * stride_vcd +
                     (page_offset % CACHE_PACK_X) * stride_vcx)
    tl.store(key_dst, key, mask=mask)
    tl.store(value_dst, value, mask=mask)


def allocate_5d_kv_cache(num_blocks, num_kv_heads, page_size, head_dim, dtype=torch.bfloat16, device="cuda"):
    """Allocate the packed K/V cache layout consumed by TLX and AITER.

    The returned tensors own their 5-D storage; no 4-D cache or repacking copy is
    involved. Cache slots are intentionally left uninitialized.
    """
    assert dtype in (torch.bfloat16, torch.float16)
    element_size = torch.empty((), dtype=dtype).element_size()
    assert element_size <= 16 and 16 % element_size == 0
    x = 16 // element_size
    assert head_dim % x == 0 and page_size % x == 0
    key_cache = torch.empty(
        (num_blocks, num_kv_heads, head_dim // x, page_size, x),
        dtype=dtype,
        device=device,
    )
    value_cache = torch.empty(
        (num_blocks, num_kv_heads, page_size // x, head_dim, x),
        dtype=dtype,
        device=device,
    )
    return key_cache, value_cache


def reshape_and_cache_5d(key, value, key_cache, value_cache, slot_mapping, num_warps=1):
    """Write new token K/V directly into arbitrary slots of a packed 5-D cache.

    ``key`` and ``value`` have shape ``[num_tokens, num_kv_heads, head_dim]``.
    Each non-negative entry in ``slot_mapping`` selects
    ``physical_block * page_size + page_offset``; negative entries are padding
    and are skipped. The source values are loaded into VGPRs and scattered
    directly to their final packed cache addresses.
    """
    assert key.ndim == value.ndim == 3 and key.shape == value.shape
    assert key.dtype == value.dtype == key_cache.dtype == value_cache.dtype
    assert key.device == value.device == key_cache.device == value_cache.device == slot_mapping.device
    assert slot_mapping.ndim == 1 and slot_mapping.shape[0] == key.shape[0]
    assert slot_mapping.dtype in (torch.int32, torch.int64)
    assert key_cache.ndim == value_cache.ndim == 5

    num_tokens, num_kv_heads, head_dim = key.shape
    num_blocks = key_cache.shape[0]
    x = 16 // key.element_size()
    page_size = key_cache.shape[3]
    assert key_cache.shape == (num_blocks, num_kv_heads, head_dim // x, page_size, x)
    assert value_cache.shape == (num_blocks, num_kv_heads, page_size // x, head_dim, x)
    if num_tokens == 0:
        return key_cache, value_cache

    head_block = min(4, _next_pow2(num_kv_heads))
    contiguous_inner = (key.stride(1) == value.stride(1) == head_dim and key.stride(2) == value.stride(2) == 1)
    contiguous_cache = key_cache.is_contiguous() and value_cache.is_contiguous()
    _reshape_and_cache_5d_kernel[(num_tokens, triton.cdiv(num_kv_heads, head_block))](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        num_blocks,
        num_kv_heads,
        *key.stride(),
        *value.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        slot_mapping.stride(0),
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        CACHE_PACK_X=x,
        HEAD_BLOCK=head_block,
        BLOCK_D=_next_pow2(head_dim),
        CONTIGUOUS_INNER=contiguous_inner,
        CONTIGUOUS_CACHE=contiguous_cache,
        num_warps=num_warps,
    )
    return key_cache, value_cache


def pack_5d_kv_cache(key_cache, value_cache):
    """Pack an existing 4-D cache (compatibility/offline conversion helper)."""
    assert key_cache.ndim == value_cache.ndim == 4
    assert key_cache.shape == value_cache.shape
    x = 16 // key_cache.element_size()
    num_blocks, num_kv_heads, page_size, head_dim = key_cache.shape
    assert head_dim % x == 0 and page_size % x == 0

    key_cache = key_cache.view(num_blocks, num_kv_heads, page_size, head_dim // x, x)
    key_cache = key_cache.permute(0, 1, 3, 2, 4).contiguous()
    value_cache = value_cache.view(num_blocks, num_kv_heads, page_size // x, x, head_dim)
    value_cache = value_cache.permute(0, 1, 2, 4, 3).contiguous()
    return key_cache, value_cache


def unpack_5d_kv_cache(key_cache, value_cache):
    """Return logical 4-D views copied from packed 5-D K/V caches."""
    assert key_cache.ndim == value_cache.ndim == 5
    num_blocks, num_kv_heads, head_groups, page_size, x = key_cache.shape
    head_dim = head_groups * x
    assert value_cache.shape == (num_blocks, num_kv_heads, page_size // x, head_dim, x)
    key = key_cache.permute(0, 1, 3, 2, 4).reshape(num_blocks, num_kv_heads, page_size, head_dim)
    value = value_cache.permute(0, 1, 2, 4, 3).reshape(num_blocks, num_kv_heads, page_size, head_dim)
    return key.contiguous(), value.contiguous()


def get_num_splits(
    num_seqs,
    num_kv_heads,
    max_ctx_len=None,
    page_size=None,
    pages_per_tile=1,
    cap=64,
    target_waves=1,
):
    """Choose the KV split-K count.

    Default rule: pick enough splits to fill the requested CTA waves, capped at
    ``cap``. The qlen=1 caller requests one wave because extra partial-output and
    reduction traffic costs more than it gains once all MI350 CUs are occupied;
    MTP callers request two waves to hide their heavier per-CTA compute.

    One case needs more splits: a medium batch of ~one wave (``progs ~ num_cu``)
    only gets ~2 splits from the rule above, so each split has to walk a long
    serial chain of KV tiles. When the tiles are also narrow (small
    ``pages_per_tile``, so each tile's gather is cheap), we add splits to keep that
    chain short, up to ~eight waves.

    Notes: context length only tightens the bound here; the caller further clamps
    splits to the KV tile count, and any split that ends up with no keys is dropped
    by the kernel's ``has_kv`` path.
    """
    props = torch.cuda.get_device_properties(0)
    num_cu = props.multi_processor_count
    progs = max(1, num_seqs * num_kv_heads)
    splits = max(1, (num_cu * target_waves) // progs)

    if target_waves > 1 and max_ctx_len is not None and page_size is not None:
        num_pages = max(1, (max_ctx_len + page_size - 1) // page_size)
        num_tiles = max(1, (num_pages + pages_per_tile - 1) // pages_per_tile)
        by_tail = max(1, num_tiles // 128)
        hi = max(1, (num_cu * 4) // progs)
        splits = max(splits, min(hi, by_tail))

    if (target_waves == 1 and max_ctx_len is not None and page_size is not None and pages_per_tile <= 2
            and num_cu <= progs <= num_cu * 2):
        num_pages = max(1, (max_ctx_len + page_size - 1) // page_size)
        num_tiles = max(1, (num_pages + pages_per_tile - 1) // pages_per_tile)
        by_tail = max(1, num_tiles // 128)  # keep very long serial tails in check
        hi = max(1, (num_cu * 8) // progs)  # never exceed ~eight waves
        splits = max(splits, min(hi, by_tail))

    return max(1, min(cap, splits))


@dataclass(frozen=True)
class PagedDecodeConfig:
    """Host-side launch configuration selected for one paged-decode shape."""

    page_size: int
    cache_pack_x: int
    query_length: int
    num_splits: int
    splits_pow2: int
    block_n: int
    pages_per_tile: int
    num_warps: int
    waves_per_eu: int
    streaming_kv: bool
    gluon_compat: bool
    query_group_size: int
    qlen_pow2: int
    group_pow2: int
    m_pow2: int

    @property
    def fused(self):
        return self.num_splits == 1

    def workspace_shapes(self, num_seqs, num_kv_heads, head_dim):
        """Return logical ``(mid, lse)`` shapes, or ``None`` for fused decode."""
        if self.fused:
            return None
        return (
            (num_seqs, num_kv_heads, self.num_splits, self.m_pow2, head_dim),
            (num_seqs, num_kv_heads, self.num_splits, self.m_pow2),
        )


@lru_cache(maxsize=None)
def _is_gfx950(device_index):
    props = torch.cuda.get_device_properties(device_index)
    return getattr(props, "gcnArchName", "").split(":", 1)[0] == "gfx950"


def can_use_pa_decode_tlx(query, key_cache, value_cache, query_length=1, sliding_window=0, sinks=None):
    """Whether the production packed-cache TLX decode supports this call.

    This intentionally describes the conservative SGLang integration surface.
    The operator itself also retains a 4-D compatibility path for tests and
    experimentation.
    """
    if not all(isinstance(tensor, torch.Tensor) for tensor in (query, key_cache, value_cache)):
        return False
    if query.ndim != 3 or key_cache.ndim != 5 or value_cache.ndim != 5:
        return False
    if query.device.type != "cuda" or key_cache.device != query.device or value_cache.device != query.device:
        return False
    if query.dtype not in (torch.bfloat16, torch.float16):
        return False
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        return False
    if query.shape[2] != 64 or query_length not in (1, 2, 3, 4):
        return False
    if sliding_window not in (None, 0) or sinks is not None:
        return False
    if key_cache.shape[0:2] != value_cache.shape[0:2]:
        return False
    num_kv_heads = key_cache.shape[1]
    if num_kv_heads == 0 or query.shape[1] % num_kv_heads != 0:
        return False
    if query_length * (query.shape[1] // num_kv_heads) > 64:
        return False
    x = 16 // key_cache.element_size()
    page_size = key_cache.shape[3]
    if page_size not in (16, 64):
        return False
    if key_cache.shape[2:] != (64 // x, page_size, x):
        return False
    if value_cache.shape[2:] != (page_size // x, 64, x):
        return False
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _is_gfx950(device_index)


def get_pa_decode_config(query, key_cache, value_cache, block_tables, query_length=1, num_splits=None,
                         max_context_len=None, num_warps=None, waves_per_eu=None, streaming_kv=None,
                         block_n=None, gluon_compat=None):
    """Select the exact launch and workspace configuration without launching."""
    num_tokens, num_q_heads, head_dim = query.shape
    assert key_cache.ndim == value_cache.ndim and key_cache.ndim in (4, 5)
    cache_5d = key_cache.ndim == 5
    num_blocks, num_kv_heads = key_cache.shape[:2]
    cache_pack_x = 16 // key_cache.element_size()
    if cache_5d:
        page_size = key_cache.shape[3]
        assert key_cache.shape == (num_blocks, num_kv_heads, head_dim // cache_pack_x, page_size, cache_pack_x)
        assert value_cache.shape == (num_blocks, num_kv_heads, page_size // cache_pack_x, head_dim, cache_pack_x)
    else:
        page_size = key_cache.shape[2]
        assert key_cache.shape[3] == head_dim
        assert value_cache.shape == key_cache.shape

    assert num_tokens % query_length == 0
    num_seqs = num_tokens // query_length
    assert num_q_heads % num_kv_heads == 0
    query_group_size = num_q_heads // num_kv_heads
    # B1 needs 32 independent partitions to fill all 256 MI350 CUs
    # (1 sequence * 8 KV heads * 32 splits).  The compact probability-LDS
    # pipeline is faster than the generic full-tile LDS path at both short
    # production contexts.  Keep this tri-state so an explicit False remains
    # a useful debugging/benchmark override.
    if gluon_compat is None:
        gluon_compat = (
            cache_5d and head_dim == 64 and query_length == 1 and query_group_size == 8 and
            num_seqs == 1 and page_size in (16, 64) and max_context_len in (8192, 32768) and
            num_splits is None and num_warps is None and streaming_kv is None and block_n is None
        )
    if gluon_compat:
        assert cache_5d and head_dim == 64
        assert query_length == 1 and query_group_size == 8
        assert page_size in (16, 64)
        streaming_kv = True
        block_n = 256
        num_warps = 4
    short_qlen1 = query_length == 1 and num_seqs == 32 and max_context_len == 8192
    if streaming_kv is None:
        streaming_kv = (cache_5d and head_dim == 64 and max_context_len is not None
                        and ((query_length == 1 and ((num_seqs == 8 and max_context_len == 8192) or short_qlen1 or
                                                     (num_seqs >= 128 and max_context_len >= 8192) or
                                                     (num_seqs >= 32 and max_context_len >= 32768) or
                                                     (num_seqs == 8 and max_context_len >= 32768))) or
                             (query_length == 4 and ((num_seqs >= 32 and max_context_len >= 8192) or
                                                     (num_seqs >= 8 and max_context_len >= 32768)))))
    short_streaming = streaming_kv and query_length == 1 and (block_n == 128 or (block_n is None and short_qlen1))
    if num_warps is None:
        num_warps = (2 if short_streaming else 4) if streaming_kv else (2 if query_length > 1 else 4)
    if streaming_kv:
        assert cache_5d and query_length in (1, 4) and head_dim == 64
        expected_warps = 2 if short_streaming else 4
        assert num_warps == expected_warps, (
            f"packed qlen={query_length} path requires num_warps={expected_warps} "
            f"for BLOCK_N={block_n or (128 if query_length == 4 or short_streaming else 256)}")

    qlen_pow2 = _next_pow2(query_length)
    group_pow2 = _next_pow2(query_group_size) if gluon_compat else max(16 // qlen_pow2,
                                                                     _next_pow2(query_group_size))
    m_pow2 = qlen_pow2 * group_pow2
    assert m_pow2 >= (8 if gluon_compat else 16), f"M_POW2={m_pow2} is too small for the selected MFMA path"
    assert query_length * query_group_size <= 64

    if block_n is None:
        if streaming_kv:
            target_tile_elements = 8192 if (query_length == 4 or short_streaming) else 16384
        elif (query_length == 1 and head_dim == 64 and num_seqs >= 8 and max_context_len is not None
              and max_context_len >= 32768):
            target_tile_elements = 16384
        else:
            target_tile_elements = 4096 if query_length > 1 else 8192
        target_block_n = target_tile_elements // head_dim
        pages_per_tile = max(1, (target_block_n + page_size - 1) // page_size)
        block_n = pages_per_tile * page_size
    else:
        assert block_n >= page_size and block_n % page_size == 0
        assert block_n & (block_n - 1) == 0
        pages_per_tile = block_n // page_size
    if streaming_kv:
        expected_block_n = 128 if (query_length == 4 or short_streaming) else 256
        assert block_n == expected_block_n, (
            f"packed qlen={query_length} path is specialized for a {expected_block_n}x64 KV tile")

    if num_splits is None:
        if gluon_compat and num_seqs == 1 and max_context_len in (8192, 32768):
            num_splits = 32
        elif streaming_kv:
            medium_decode = (query_length == 1 and num_seqs == 8 and max_context_len is not None
                             and 32768 <= max_context_len < 131072)
            short_decode = query_length == 1 and short_qlen1
            if gluon_compat:
                split_cap, target_waves = 8, 2
            elif short_decode:
                split_cap, target_waves = 4, 4
            elif medium_decode:
                split_cap, target_waves = 16, 4
            else:
                split_cap, target_waves = 8, 2
            num_splits = get_num_splits(num_seqs, num_kv_heads, max_context_len, page_size, pages_per_tile,
                                        cap=split_cap, target_waves=target_waves)
        else:
            num_splits = get_num_splits(num_seqs, num_kv_heads, max_context_len, page_size, pages_per_tile,
                                        target_waves=2 if query_length > 1 else 1)
        if (not gluon_compat and cache_5d and head_dim == 64 and query_length == 1 and num_seqs == 1 and
                max_context_len == 8192):
            num_splits = 8 if page_size == 16 else 4
        max_pages = block_tables.shape[1]
        max_useful_splits = max(1, (max_pages + pages_per_tile - 1) // pages_per_tile)
        num_splits = min(num_splits, max_useful_splits)

    if waves_per_eu is None:
        tuned_wpe2 = streaming_kv and (
            (page_size == 64 and num_seqs == 128 and max_context_len == 8192 and num_splits == 1) or
            (page_size == 16 and num_seqs == 8 and max_context_len == 32768 and num_splits == 16))
        waves_per_eu = 2 if tuned_wpe2 else 0

    return PagedDecodeConfig(
        page_size=page_size,
        cache_pack_x=cache_pack_x,
        query_length=query_length,
        num_splits=num_splits,
        splits_pow2=_next_pow2(num_splits),
        block_n=block_n,
        pages_per_tile=pages_per_tile,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        streaming_kv=streaming_kv,
        gluon_compat=gluon_compat,
        query_group_size=query_group_size,
        qlen_pow2=qlen_pow2,
        group_pow2=group_pow2,
        m_pow2=m_pow2,
    )


def allocate_pa_decode_workspace(query, key_cache, config):
    """Allocate reusable FP32 split-decode workspace for ``config``.

    B1/D64 uses a four-element physical row pad for the partial output.  The
    non-power-of-two split stride prevents MI350 memory-partition camping that
    otherwise makes performance depend strongly on the allocator's base
    address.  ``pa_decode_tlx`` slices this capacity tensor to its logical shape
    while retaining the padded strides.
    """
    num_seqs = query.shape[0] // config.query_length
    shapes = config.workspace_shapes(num_seqs, key_cache.shape[1], query.shape[2])
    if shapes is None:
        return None
    mid_shape, lse_shape = shapes
    mid_capacity_shape = mid_shape
    if num_seqs == 1 and config.query_length == 1 and query.shape[2] == 64:
        mid_capacity_shape = (*mid_shape[:-1], mid_shape[-1] + 4)
    return (
        torch.empty(mid_capacity_shape, dtype=torch.float32, device=query.device),
        torch.empty(lse_shape, dtype=torch.float32, device=query.device),
    )


def pa_decode_tlx(output,  # [num_tokens, num_q_heads, HEAD_DIM]
                  query,  # [num_tokens, num_q_heads, HEAD_DIM]
                  key_cache,  # 4-D NHD or packed 5-D [block, head, dim/X, page, X]
                  value_cache,  # 4-D NHD or packed 5-D [block, head, page/X, dim, X]
                  context_lens,  # [num_seqs] int32
                  block_tables,  # [num_seqs, max_pages] int32
                  sm_scale, query_length=1, num_splits=None, max_context_len=None, num_warps=None, waves_per_eu=None,
                  streaming_kv=None,  # None auto-selects the tuned packed K/V register path
                  block_n=None,  # Experimental tile-width override; defaults to ~8192 KV elements
                  gluon_compat=None,  # None auto-selects the tuned qlen1/group8 probability-LDS path
                  workspace=None,  # Optional reusable (mid, lse) float32 tensors for split decode
                  config=None,  # Optional cached PagedDecodeConfig; bypasses host-side reselection
                  ):
    num_tokens, num_q_heads, head_dim = query.shape
    assert key_cache.ndim == value_cache.ndim and key_cache.ndim in (4, 5)
    cache_5d = key_cache.ndim == 5
    num_blocks, num_kv_heads = key_cache.shape[:2]
    if config is None:
        config = get_pa_decode_config(
            query,
            key_cache,
            value_cache,
            block_tables,
            query_length=query_length,
            num_splits=num_splits,
            max_context_len=max_context_len,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            streaming_kv=streaming_kv,
            block_n=block_n,
            gluon_compat=gluon_compat,
        )
    else:
        assert isinstance(config, PagedDecodeConfig)
        assert config.query_length == query_length
        assert num_splits is None or config.num_splits == num_splits
        assert num_warps is None or config.num_warps == num_warps
        assert waves_per_eu is None or config.waves_per_eu == waves_per_eu
        assert streaming_kv is None or config.streaming_kv == streaming_kv
        assert block_n is None or config.block_n == block_n
        assert not gluon_compat or config.gluon_compat

    page_size = config.page_size
    cache_pack_x = config.cache_pack_x
    num_splits = config.num_splits
    splits_pow2 = config.splits_pow2
    block_n = config.block_n
    pages_per_tile = config.pages_per_tile
    num_warps = config.num_warps
    waves_per_eu = config.waves_per_eu
    streaming_kv = config.streaming_kv
    gluon_compat = config.gluon_compat
    query_group_size = config.query_group_size
    qlen_pow2 = config.qlen_pow2
    group_pow2 = config.group_pow2
    m_pow2 = config.m_pow2
    num_seqs = num_tokens // query_length

    assert num_tokens % query_length == 0
    assert num_q_heads % num_kv_heads == 0
    assert query_group_size == num_q_heads // num_kv_heads
    if cache_5d:
        assert page_size == key_cache.shape[3]
        assert key_cache.shape == (num_blocks, num_kv_heads, head_dim // cache_pack_x, page_size, cache_pack_x)
        assert value_cache.shape == (num_blocks, num_kv_heads, page_size // cache_pack_x, head_dim, cache_pack_x)
        # Keep stride names semantic: kc_p addresses page and kc_d addresses
        # packed head-dimension groups. V's dim 2 addresses token groups.
        key_cache_strides = (key_cache.stride(0), key_cache.stride(1), key_cache.stride(3), key_cache.stride(2),
                             key_cache.stride(4))
        value_cache_strides = tuple(value_cache.stride(i) for i in range(5))
    else:
        assert page_size == key_cache.shape[2]
        assert key_cache.shape[3] == head_dim
        assert value_cache.shape == key_cache.shape
        key_cache_strides = (*key_cache.stride(), 0)
        value_cache_strides = (*value_cache.stride(), 0)

    # One-shot fused path: with a single split the partition output is already
    # the final normalized result, so write it straight to `output` and skip both
    # the Mid/LSE HBM round-trip and the separate reduce launch.
    fused = config.fused
    if fused:
        mid = lse = output
        mid_strides = (0, 0, 0, 0, 0)
        lse_strides = (0, 0, 0, 0)
    else:
        mid_shape = (num_seqs, num_kv_heads, num_splits, m_pow2, head_dim)
        lse_shape = (num_seqs, num_kv_heads, num_splits, m_pow2)
        if workspace is None:
            mid = torch.empty(mid_shape, dtype=torch.float32, device=query.device)
            lse = torch.empty(lse_shape, dtype=torch.float32, device=query.device)
        else:
            assert isinstance(workspace, (tuple, list)) and len(workspace) == 2, \
                "workspace must be a (mid, lse) pair"
            mid, lse = workspace
            assert mid.ndim == 5 and lse.ndim == 4
            assert all(got >= need for got, need in zip(mid.shape, mid_shape))
            assert all(got >= need for got, need in zip(lse.shape, lse_shape))
            assert mid.dtype == lse.dtype == torch.float32
            assert mid.device == lse.device == query.device
            # Capacity-sized buffers let serving runtimes reserve one stable
            # workspace before graph capture and reuse it across batch shapes.
            if tuple(mid.shape) != mid_shape:
                mid = mid[tuple(slice(0, size) for size in mid_shape)]
            if tuple(lse.shape) != lse_shape:
                lse = lse[tuple(slice(0, size) for size in lse_shape)]
        mid_strides = (mid.stride(0), mid.stride(1), mid.stride(2), mid.stride(3), mid.stride(4))
        lse_strides = (lse.stride(0), lse.stride(1), lse.stride(2), lse.stride(3))

    grid_p = (num_seqs, num_kv_heads, num_splits)
    _pa_decode_partition_kernel[grid_p](
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        mid,
        lse,
        output,
        sm_scale,
        num_splits,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        *key_cache_strides,
        *value_cache_strides,
        block_tables.stride(0),
        block_tables.stride(1),
        *mid_strides,
        *lse_strides,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_N=block_n,
        PAGES_PER_TILE=pages_per_tile,
        BUFFER_DEPTH=BUF_DEPTH,
        QUERY_GROUP_SIZE=query_group_size,
        GROUP_POW2=group_pow2,
        QLEN=query_length,
        QLEN_POW2=qlen_pow2,
        M_POW2=m_pow2,
        FUSED=fused,
        CACHE_5D=cache_5d,
        CACHE_PACK_X=cache_pack_x,
        V_LAYOUT=_make_v_shared_layout(),
        QK_SOFTMAX_LAYOUT=_make_qk_softmax_layout(
            block_n if streaming_kv else 64, m_pow2, num_warps if streaming_kv else 4
        ) if not gluon_compat else _make_gluon_qk_layout(block_n),
        STREAMING_KV=streaming_kv,
        STREAM_WARPS=num_warps,
        GLUON_COMPAT=gluon_compat,
        GLUON_K_LOAD_LAYOUT=_make_gluon_k_load_layout(page_size),
        GLUON_V_LOAD_LAYOUT=_make_gluon_v_load_layout(page_size),
        GLUON_Q_SHARED_LAYOUT=_make_gluon_shared_layout(8, 16),
        GLUON_P_SHARED_LAYOUT=_make_gluon_shared_layout(8, 8),
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    if fused:
        return output

    grid_r = (num_tokens, num_q_heads)
    _pa_decode_reduce_kernel[grid_r](
        output,
        mid,
        lse,
        num_splits,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        mid.stride(3),
        mid.stride(4),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        lse.stride(3),
        HEAD_DIM=head_dim,
        QUERY_GROUP_SIZE=query_group_size,
        GROUP_POW2=group_pow2,
        QLEN=query_length,
        SPLITS_POW2=splits_pow2,
    )
    return output


# Test/benchmark helpers: paged inputs + a dense fp32 reference, consumed by the
# correctness suite (test_correctness.py) and the perf harness
# (test_amd_pa_decode_perf.py).
def build_inputs(num_seqs, ctx_lens, num_q_heads, num_kv_heads, head_dim, page_size, query_length=1,
                 dtype=torch.bfloat16, device="cuda", seed=0, pool_pages=None, cache_layout="4d"):
    """Build paged decode inputs. If ``pool_pages`` is set, physical pages are
    drawn from a shared pool of that size (bounds memory for large sweeps); the
    dense reference uses the same ``block_tables`` so correctness is unaffected.
    ``cache_layout="5d"`` allocates and initializes packed storage natively;
    it never constructs or repacks a 4-D cache.
    """
    torch.manual_seed(seed)
    assert len(ctx_lens) == num_seqs
    num_tokens = num_seqs * query_length

    query = torch.randn(num_tokens, num_q_heads, head_dim, dtype=dtype, device=device) * 0.2

    max_pages = (max(ctx_lens) + page_size - 1) // page_size
    distinct = num_seqs * max_pages
    total_pages = distinct if pool_pages is None else min(distinct, pool_pages)
    assert cache_layout in ("4d", "5d")
    if cache_layout == "4d":
        cache_shape = (total_pages, num_kv_heads, page_size, head_dim)
        key_cache = torch.empty(cache_shape, dtype=dtype, device=device).normal_().mul_(0.2)
        value_cache = torch.empty(cache_shape, dtype=dtype, device=device).normal_().mul_(0.2)
    else:
        key_cache, value_cache = allocate_5d_kv_cache(total_pages, num_kv_heads, page_size, head_dim, dtype=dtype,
                                                      device=device)
        key_cache.normal_().mul_(0.2)
        value_cache.normal_().mul_(0.2)

    block_tables = torch.zeros(num_seqs, max_pages, dtype=torch.int32, device=device)
    for s in range(num_seqs):
        npag = (ctx_lens[s] + page_size - 1) // page_size
        for p in range(max_pages):
            phys = (s * max_pages + (p if p < npag else 0)) % total_pages
            block_tables[s, p] = phys
    context_lens = torch.tensor(ctx_lens, dtype=torch.int32, device=device)
    return query, key_cache, value_cache, context_lens, block_tables


def ref_decode(query, key_cache, value_cache, context_lens, block_tables, sm_scale, num_q_heads, num_kv_heads,
               query_length):
    """Dense fp32 reference: gather full K/V from the page table, causal over qlen."""
    if key_cache.ndim == 5:
        key_cache, value_cache = unpack_5d_kv_cache(key_cache, value_cache)
    head_dim = query.shape[-1]
    page_size = key_cache.shape[2]
    group = num_q_heads // num_kv_heads
    num_seqs = query.shape[0] // query_length
    out = torch.empty_like(query, dtype=torch.float32)

    for s in range(num_seqs):
        ctx = int(context_lens[s].item())
        npag = (ctx + page_size - 1) // page_size
        phys = block_tables[s, :npag]
        k = key_cache[phys].to(torch.float32)  # [npag, kvh, page, d]
        v = value_cache[phys].to(torch.float32)
        k = k.permute(1, 0, 2, 3).reshape(num_kv_heads, npag * page_size, head_dim)[:, :ctx]
        v = v.permute(1, 0, 2, 3).reshape(num_kv_heads, npag * page_size, head_dim)[:, :ctx]
        for qpos in range(query_length):
            gt = s * query_length + qpos
            limit = ctx - query_length + qpos  # inclusive last visible key index
            for qh in range(num_q_heads):
                kvh = qh // group
                q = query[gt, qh].to(torch.float32)  # [d]
                scores = (q[None, :] * k[kvh]).sum(-1) * sm_scale  # [ctx]
                scores = scores[:limit + 1]
                p = torch.softmax(scores, dim=0)
                out[gt, qh] = (p[:, None] * v[kvh, :limit + 1]).sum(0)
    return out
