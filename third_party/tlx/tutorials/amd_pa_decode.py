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
    # Pre-scale q once (M_POW2 x HEAD_DIM) so the q@k^T dot already yields
    # scaled scores, instead of scaling the M_POW2 x BLOCK_N score matrix per tile.
    q = (q.to(tl.float32) * QK_SCALE).to(Kc.dtype.element_ty)
    m_qpos = offs_m // GROUP_POW2  # query position per row
    vis_limit = ctx_len - QLEN  # min visible abs key index over rows (m_qpos >= 0)

    m_i = tl.full([M_POW2], float("-inf"), tl.float32)
    l_i = tl.zeros([M_POW2], tl.float32)
    acc = tl.zeros([M_POW2, HEAD_DIM], tl.float32)

    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), Kc.dtype.element_ty, BUFFER_DEPTH)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), Vc.dtype.element_ty, BUFFER_DEPTH)

    # Decouple the KV compute tile (BLOCK_N keys) from PAGE_SIZE: row r of a tile
    # maps to logical page (tile_page0 + r // PAGE_SIZE) at offset (r % PAGE_SIZE),
    # gathered in one async load so the MFMA K-dim stays BLOCK_N-wide even at
    # small PAGE_SIZE. Depth-2 software pipeline: prefetch tile N+1 while computing
    # tile N (wait_group(1) keeps only the prefetch outstanding).
    row_page = offs_n // PAGE_SIZE
    row_in_page = offs_n % PAGE_SIZE
    num_tiles = tl.cdiv(end_page - start_page, PAGES_PER_TILE)
    if num_tiles > 0:
        physical = tl.load(BlockTables + seq * stride_bt_s +
                           tl.where(row_page < end_page - start_page, start_page + row_page, end_page - 1) *
                           stride_bt_p)
        tok_k = tlx.async_load(
            Kc + physical[:, None] * stride_kc_b + kv_head * stride_kc_h + row_in_page[:, None] * stride_kc_p +
            offs_d[None, :] * stride_kc_d, tlx.local_view(k_buf, 0))
        tok_v = tlx.async_load(
            Vc + physical[:, None] * stride_vc_b + kv_head * stride_vc_h + row_in_page[:, None] * stride_vc_p +
            offs_d[None, :] * stride_vc_d, tlx.local_view(v_buf, 0))
        tlx.async_load_commit_group([tok_k, tok_v])

        for tidx in tl.range(0, num_tiles):
            slot = tidx % BUFFER_DEPTH
            nxt = tidx + 1
            if nxt < num_tiles:
                nslot = nxt % BUFFER_DEPTH
                n_page_of_row = start_page + nxt * PAGES_PER_TILE + row_page
                n_physical = tl.load(BlockTables + seq * stride_bt_s +
                                     tl.where(n_page_of_row < end_page, n_page_of_row, end_page - 1) * stride_bt_p)
                ntok_k = tlx.async_load(
                    Kc + n_physical[:, None] * stride_kc_b + kv_head * stride_kc_h +
                    row_in_page[:, None] * stride_kc_p + offs_d[None, :] * stride_kc_d, tlx.local_view(k_buf, nslot))
                ntok_v = tlx.async_load(
                    Vc + n_physical[:, None] * stride_vc_b + kv_head * stride_vc_h +
                    row_in_page[:, None] * stride_vc_p + offs_d[None, :] * stride_vc_d, tlx.local_view(v_buf, nslot))
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
        tl.store(out_ptrs, o_part.to(Out.dtype.element_ty), mask=valid_m[:, None])
    else:
        # Store the normalized partial output + base-2 lse for this split.
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


def get_num_splits(num_seqs, num_kv_heads, max_ctx_len=None, page_size=None, pages_per_tile=1, cap=64):
    """Choose the KV split-K count.

    Default rule: pick enough splits to fill the CUs (~two waves of CTAs), capped
    at ``cap``. This is right for small and large batches.

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
    splits = max(1, (num_cu * 2) // progs)

    if (max_ctx_len is not None and page_size is not None and pages_per_tile <= 2 and num_cu <= progs <= num_cu * 2):
        num_pages = max(1, (max_ctx_len + page_size - 1) // page_size)
        num_tiles = max(1, (num_pages + pages_per_tile - 1) // pages_per_tile)
        by_tail = max(1, num_tiles // 32)  # keep the serial tail <= ~32 tiles/split
        hi = max(1, (num_cu * 8) // progs)  # never exceed ~eight waves
        splits = max(splits, min(hi, by_tail))

    return max(1, min(cap, splits))


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

    # KV compute tile width in keys, independent of PAGE_SIZE. Target a fixed tile
    # of BLOCK_N * HEAD_DIM ~= 8192 elements (fits the LDS budget double-buffered),
    # rounded up to a whole number of pages so tiles stay page-aligned.
    target_block_n = 8192 // head_dim
    pages_per_tile = max(1, (target_block_n + page_size - 1) // page_size)
    block_n = pages_per_tile * page_size

    if num_splits is None:
        num_splits = get_num_splits(num_seqs, num_kv_heads, max_context_len, page_size, pages_per_tile)
        # Never launch more splits than there are KV tiles to cover (host-only, no
        # device sync), so short contexts never over-split.
        max_pages = block_tables.shape[1]
        max_useful_splits = max(1, (max_pages + pages_per_tile - 1) // pages_per_tile)
        num_splits = min(num_splits, max_useful_splits)
    splits_pow2 = _next_pow2(num_splits)

    # One-shot fused path: with a single split the partition output is already
    # the final normalized result, so write it straight to `output` and skip both
    # the Mid/LSE HBM round-trip and the separate reduce launch.
    fused = num_splits == 1
    if fused:
        mid = lse = output
        mid_strides = (0, 0, 0, 0, 0)
        lse_strides = (0, 0, 0, 0)
    else:
        mid = torch.empty((num_seqs, num_kv_heads, num_splits, m_pow2, head_dim), dtype=torch.float32,
                          device=query.device)
        lse = torch.empty((num_seqs, num_kv_heads, num_splits, m_pow2), dtype=torch.float32, device=query.device)
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
