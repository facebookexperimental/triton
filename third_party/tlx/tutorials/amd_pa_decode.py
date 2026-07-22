"""AMD CDNA4 paged-attention decode kernel written with TLX.

Two-phase split-K decode, ported from the algorithm in ROCm/aiter
``pa_decode_gluon`` but expressed with Triton + TLX low-level primitives.
Phase 1 (``_pa_decode_partition_kernel``) has each program handle one
``(sequence, kv_head, split)``: it streams the split's KV pages into LDS with
``tlx.async_load`` (double-buffered), does ``q @ k^T`` via MFMA, a base-2
online softmax, then ``p @ v``, and writes a normalized partial output plus a
base-2 log-sum-exp for that split. Phase 2 (``_pa_decode_reduce_kernel``) is a
plain ``@triton.jit`` kernel that merges the per-split partials for each output
``(token, query_head)`` via the standard LSE trick.

Scope: bf16/fp16 KV cache, GQA, and MTP query_length in 1..4 (causal across the
query positions). Not covered: FP8, sliding window, ALiBi, sinks, per-token
quantization. KV cache layout is contiguous
``[num_blocks, num_kv_heads, PAGE_SIZE, HEAD_DIM]``.

Consumed by the correctness suite (``test_correctness.py``) and the perf script
(``test_amd_pa_decode_perf.py``).
"""

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

BUF_DEPTH = tl.constexpr(2)


@triton.jit
def _pa_decode_partition_kernel(
    Q,  # [num_tokens, num_q_heads, HEAD_DIM]
    Kc,  # [num_blocks, num_kv_heads, PAGE_SIZE, HEAD_DIM]
    Vc,  # [num_blocks, num_kv_heads, PAGE_SIZE, HEAD_DIM]
    BlockTables,  # [num_seqs, max_pages]
    CtxLens,  # [num_seqs]
    Mid,  # [num_seqs, num_kv_heads, NUM_SPLITS, M_POW2, HEAD_DIM] (fp32)
    Lse,  # [num_seqs, num_kv_heads, NUM_SPLITS, M_POW2] (fp32)
    sm_scale,
    num_splits,  # runtime int (== grid dim 2)
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kc_b,
    stride_kc_h,
    stride_kc_p,
    stride_kc_d,
    stride_vc_b,
    stride_vc_h,
    stride_vc_p,
    stride_vc_d,
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
    offs_p = tl.arange(0, PAGE_SIZE)
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
    m_qpos = offs_m // GROUP_POW2  # query position per row

    m_i = tl.full([M_POW2], float("-inf"), tl.float32)
    l_i = tl.zeros([M_POW2], tl.float32)
    acc = tl.zeros([M_POW2, HEAD_DIM], tl.float32)

    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), Kc.dtype.element_ty, BUFFER_DEPTH)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), Vc.dtype.element_ty, BUFFER_DEPTH)

    if PAGES_PER_TILE == 4:
        for pidx in tl.range(start_page, end_page, PAGES_PER_TILE):
            k_view = tlx.local_view(k_buf, 0)
            v_view = tlx.local_view(v_buf, 0)
            logical_page = pidx + offs_n // PAGE_SIZE
            safe_page = tl.minimum(logical_page, end_page - 1)
            physical = tl.load(BlockTables + seq * stride_bt_s + safe_page * stride_bt_p)
            page_row = offs_n % PAGE_SIZE
            k_ptrs = (Kc + physical[:, None] * stride_kc_b + kv_head * stride_kc_h + page_row[:, None] * stride_kc_p +
                      offs_d[None, :] * stride_kc_d)
            v_ptrs = (Vc + physical[:, None] * stride_vc_b + kv_head * stride_vc_h + page_row[:, None] * stride_vc_p +
                      offs_d[None, :] * stride_vc_d)
            tlx.local_store(k_view, tl.load(k_ptrs))
            tlx.local_store(v_view, tl.load(v_ptrs))
            tl.debug_barrier()
            kt = tlx.local_load(tlx.local_trans(k_view))
            v = tlx.local_load(v_view)

            qk = tl.dot(q, kt)
            kt_abs = pidx * PAGE_SIZE + offs_n
            vis = ((kt_abs[None, :] < end_page * PAGE_SIZE) & (kt_abs[None, :] <= (ctx_len - QLEN + m_qpos[:, None])))
            qk = tl.where(vis, qk * QK_SCALE, float("-inf"))
            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp2(qk - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
            tl.debug_barrier()
    elif start_page < end_page:
        physical = tl.load(BlockTables + seq * stride_bt_s + start_page * stride_bt_p)
        k_ptrs = (Kc + physical * stride_kc_b + kv_head * stride_kc_h + offs_p[:, None] * stride_kc_p +
                  offs_d[None, :] * stride_kc_d)
        v_ptrs = (Vc + physical * stride_vc_b + kv_head * stride_vc_h + offs_p[:, None] * stride_vc_p +
                  offs_d[None, :] * stride_vc_d)
        tok_k = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0))
        tok_v = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0))
        tlx.async_load_commit_group([tok_k, tok_v])

        num_pages_this_split = end_page - start_page
        for rel_page in tl.range(0, num_pages_this_split - 1, num_stages=0):
            slot_cur = rel_page % 2
            slot_nxt = (rel_page + 1) % 2
            wait_tok = tlx.async_load_wait_group(0)
            kt = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, slot_cur)), token=wait_tok)
            v = tlx.local_load(tlx.local_view(v_buf, slot_cur), token=wait_tok)

            next_page = start_page + rel_page + 1
            physical_next = tl.load(BlockTables + seq * stride_bt_s + next_page * stride_bt_p)
            k_ptrs_next = (Kc + physical_next * stride_kc_b + kv_head * stride_kc_h + offs_p[:, None] * stride_kc_p +
                           offs_d[None, :] * stride_kc_d)
            v_ptrs_next = (Vc + physical_next * stride_vc_b + kv_head * stride_vc_h + offs_p[:, None] * stride_vc_p +
                           offs_d[None, :] * stride_vc_d)
            tok_k = tlx.async_load(k_ptrs_next, tlx.local_view(k_buf, slot_nxt))
            tok_v = tlx.async_load(v_ptrs_next, tlx.local_view(v_buf, slot_nxt))
            tlx.async_load_commit_group([tok_k, tok_v])

            pidx = start_page + rel_page
            qk = tl.dot(q, kt)
            kt_abs = pidx * PAGE_SIZE + offs_n
            qk = tl.where(kt_abs[None, :] <= (ctx_len - QLEN + m_qpos[:, None]), qk * QK_SCALE, float("-inf"))
            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp2(qk - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        last_rel_page = num_pages_this_split - 1
        wait_tok = tlx.async_load_wait_group(0)
        kt = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, last_rel_page % 2)), token=wait_tok)
        v = tlx.local_load(tlx.local_view(v_buf, last_rel_page % 2), token=wait_tok)
        last_page = start_page + last_rel_page
        qk = tl.dot(q, kt)
        kt_abs = last_page * PAGE_SIZE + offs_n
        qk = tl.where(kt_abs[None, :] <= (ctx_len - QLEN + m_qpos[:, None]), qk * QK_SCALE, float("-inf"))
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Store the normalized partial output + base-2 lse for this split.
    has_kv = l_i > 0.0
    o_part = tl.where(has_kv[:, None], acc / tl.where(has_kv[:, None], l_i[:, None], 1.0), 0.0)
    lse_part = tl.where(has_kv, m_i + tl.math.log2(tl.where(has_kv, l_i, 1.0)), float("-inf"))

    mid_ptrs = (Mid + seq * stride_mid_s + kv_head * stride_mid_h + split * stride_mid_k +
                offs_m[:, None] * stride_mid_m + offs_d[None, :] * stride_mid_d)
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


def get_num_splits(num_seqs, num_kv_heads, max_ctx_len=None, page_size=None, cap=128):
    """Choose enough splits to expose CTA parallelism without creating empty work.

    When the caller knows the maximum context length on the host, target roughly
    six CTAs per CU and cap the result by the physical page count. Otherwise,
    preserve the original occupancy-only policy and its cap of eight splits to
    avoid introducing a device-to-host synchronization in production callers.
    """
    props = torch.cuda.get_device_properties(0)
    num_cu = props.multi_processor_count
    base_programs = max(1, num_seqs * num_kv_heads)
    if max_ctx_len is None or page_size is None:
        return max(1, min(8, (num_cu * 2) // base_programs))

    num_pages = max(1, (max_ctx_len + page_size - 1) // page_size)
    splits_for_occupancy = _next_pow2((num_cu * 6 + base_programs - 1) // base_programs)
    return max(1, min(cap, num_pages, splits_for_occupancy))


def pa_decode_tlx(
    output,  # [num_tokens, num_q_heads, HEAD_DIM]
    query,  # [num_tokens, num_q_heads, HEAD_DIM]
    key_cache,  # [num_blocks, num_kv_heads, PAGE_SIZE, HEAD_DIM]
    value_cache,  # [num_blocks, num_kv_heads, PAGE_SIZE, HEAD_DIM]
    context_lens,  # [num_seqs] int32
    block_tables,  # [num_seqs, max_pages] int32
    sm_scale,
    query_length=1,
    num_splits=None,
    max_context_len=None,
    num_warps=4,
    waves_per_eu=0,
):
    num_tokens, num_q_heads, head_dim = query.shape
    num_blocks, num_kv_heads, page_size, _ = key_cache.shape
    num_seqs = num_tokens // query_length
    query_group_size = num_q_heads // num_kv_heads

    qlen_pow2 = _next_pow2(query_length)
    group_pow2 = max(16 // qlen_pow2, _next_pow2(query_group_size))
    m_pow2 = qlen_pow2 * group_pow2
    assert m_pow2 >= 16, f"M_POW2={m_pow2} must be >= 16 for MFMA"
    assert query_length * query_group_size <= 64

    if num_splits is None:
        num_splits = get_num_splits(num_seqs, num_kv_heads, max_context_len, page_size)
    splits_pow2 = _next_pow2(num_splits)

    mid = torch.empty((num_seqs, num_kv_heads, num_splits, m_pow2, head_dim), dtype=torch.float32, device=query.device)
    lse = torch.empty((num_seqs, num_kv_heads, num_splits, m_pow2), dtype=torch.float32, device=query.device)

    grid_p = (num_seqs, num_kv_heads, num_splits)
    _pa_decode_partition_kernel[grid_p](
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        mid,
        lse,
        sm_scale,
        num_splits,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
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
        PAGE_SIZE=page_size,
        BLOCK_N=64 if page_size == 16 else page_size,
        PAGES_PER_TILE=4 if page_size == 16 else 1,
        BUFFER_DEPTH=1 if page_size == 16 else BUF_DEPTH,
        QUERY_GROUP_SIZE=query_group_size,
        GROUP_POW2=group_pow2,
        QLEN=query_length,
        QLEN_POW2=qlen_pow2,
        M_POW2=m_pow2,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

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
                 dtype=torch.bfloat16, device="cuda", seed=0, pool_pages=None):
    """Build paged decode inputs. If ``pool_pages`` is set, physical pages are
    drawn from a shared pool of that size (bounds memory for large sweeps); the
    dense reference uses the same ``block_tables`` so correctness is unaffected.
    """
    torch.manual_seed(seed)
    assert len(ctx_lens) == num_seqs
    num_tokens = num_seqs * query_length

    query = torch.randn(num_tokens, num_q_heads, head_dim, dtype=dtype, device=device) * 0.2

    max_pages = (max(ctx_lens) + page_size - 1) // page_size
    distinct = num_seqs * max_pages
    total_pages = distinct if pool_pages is None else min(distinct, pool_pages)
    key_cache = torch.randn(total_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device) * 0.2
    value_cache = torch.randn(total_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device) * 0.2

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
