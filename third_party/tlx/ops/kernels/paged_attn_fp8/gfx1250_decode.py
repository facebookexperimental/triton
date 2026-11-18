"""gfx1250 paged attention over E4M3FN K/V with per-KV-head scales.

Cache layout and terminology
----------------------------
``page_size`` / ``PAGE_SIZE`` is the number of cached tokens per physical
cache block. K is [blocks, kv_heads, D/16, page_size, 16] and V is
[blocks, kv_heads, page_size/16, D, 16]. The inner 16 is a packing factor,
independent of page size. A logical token at position t uses block-table
entry t // page_size and offset t % page_size within that physical block.
An LDS buffer slot instead selects the local staging buffer, usually 0 or 1.

Execution paths
---------------
* ``_matrix_decode`` groups query rows sharing a KV head and uses WMMA for
  QK and PV, with FP32 online softmax. ``matrix_tdm_flat`` copies contiguous
  packed head/page bytes into double-buffered LDS on page sizes 64/128;
  ``matrix`` uses masked loads for other supported page sizes. The internal
  ``matrix_precise`` control uses compensated BF16 PV. Split partitions
  write FP32 normalized partial values and log-sum-exp for ``_merge``; a single
  split writes the output directly.
* ``_small_window_decode`` specializes a window no larger than one page.
  Its block-uniform alignment check permits one page-sized QK/softmax/PV
  with one K/V buffer each; short, empty or unaligned contexts call the
  ordinary masked matrix path. ``_window_decode`` similarly specializes
  a 128-token window over two complete 64-token pages, sharing one FP32
  normalization across both page-sized products. These are window fast
  paths of the same attention operation, selected privately in ``decode``.
* ``_vector_decode`` computes QK/PV with vector reductions instead of WMMA.
  ``vector`` supports arbitrary valid page sizes using masked indexed loads;
  the internal ``vector_page`` control loads complete packed physical pages.
  Automatic dispatch uses the vector fallback for tiny contexts on pages
  other than 64/128. Larger contexts on those pages use masked ``matrix``.

``_select_decode_config`` chooses the path, split count and warp count from
the geometry. Window paths require one split, single-token BF16 Q and their
documented shape/alignment guards. FP8 Q and unsupported fast-path shapes
retain the ordinary matrix/vector paths. These choices are not public API.
Automatic batched single-token decode on packed pages uses actual device
lengths for partitioning. Eligible larger batches use ``_batched_decode`` to
pack those partitions onto a bounded work grid; smaller batches keep flat
dispatch to avoid queue-loop overhead. Both wrap the same matrix computation. Single-partition requests
write output directly; ``_merge`` skips these requests and reads only active
FP32 partials for the others. Prefix scans cover at most 64 requests, and a
device reduction divides a parallelism budget among requests worth splitting.
Small active merges use a compact reduction tile; large merges stay vectorized.
Reserved workspace capacity still covers the configured bound. Explicit split overrides, windows and multi-query calls
retain the fixed plan; single-request scheduling is unchanged.

All paths retain per-head scaling and produce BF16 output before projection;
this is plain FP8, not E8M0 block-scaled MXFP8.

Design references: AITER's split unified attention and packed-cache movement,
TokenSpeed's grouped-head WMMA decode, vLLM's online softmax/scale placement,
and SGLang's split-KV merge and gfx1250 numerical precautions. This file is an
independent implementation; pinned reference bodies live outside this checkout.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@lru_cache(maxsize=None)
def _vector_layouts(block_n, dim, num_warps):
    k = tlx.layout(shape=((4, 8, num_warps), (16, dim // 64, block_n // (8 * num_warps))),
                   stride=((16, dim, 8 * dim), (1, 64, 8 * num_warps * dim)))
    v = tlx.layout(shape=((2, 16, num_warps), (16, block_n // 32, dim // (16 * num_warps))),
                   stride=((16 * dim, 1, 16), (dim, 32 * dim, 16 * num_warps)))
    return k, v


@lru_cache(maxsize=None)
def _packed_operand_layouts(page_size, dim, fp8_qk):
    if fp8_qk:
        k = tlx.layout(shape=((16, 2), (16, dim // 32, page_size // 16)),
                       stride=((16, page_size * 16), (1, page_size * 32, 256)))
    else:
        k = tlx.layout(shape=((16, 2), (8, dim // 16, page_size // 16)), stride=((16, 8), (1, page_size * 16, 256)))
    v = tlx.layout(shape=((16, 2), (8, page_size // 16, dim // 16)), stride=((16, 8), (1, dim * 16, 256)))
    return k, v


@triton.jit
def _vector_decode(
    Q,
    K,
    V,
    KS,
    VS,
    QS,
    BT,
    LEN,
    PART,
    LSE,
    OUT,
    qst,
    qsh,
    qsd,
    bst,
    kb,
    kh,
    kg,
    kp,
    vb,
    vh,
    vp,
    vd,
    scale,
    D: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    QLEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    BN: tl.constexpr,
    Q_FP8: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    K_LAYOUT: tl.constexpr,
    V_LAYOUT: tl.constexpr,
    PHYSICAL: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    seq = token // QLEN
    kvh = head // G
    length = tl.load(LEN + seq)
    qpos = length - QLEN + token % QLEN
    end = length
    if CAUSAL:
        end = tl.minimum(end, qpos + 1)
    begin = 0
    if WINDOW > 0:
        begin = tl.maximum(0, qpos + 1 - WINDOW)
    attention_begin = begin
    if PHYSICAL:
        begin = begin // PAGE_SIZE * PAGE_SIZE
    active = tl.maximum(0, end - begin)
    span = tl.cdiv(active, SPLITS * BN) * BN
    lo = begin + split * span
    hi = tl.minimum(end, lo + span)
    d = tl.arange(0, D)
    n = tl.arange(0, BN)
    q = tl.load(Q + token * qst + head * qsh + d * qsd).to(tl.float32)
    qs = 1.0
    if Q_FP8:
        qs = tl.load(QS + head)
    ks = tl.load(KS + kvh)
    vs = tl.load(VS + kvh)
    multiplier = scale * ks * qs * 1.4426950408889634
    m = -float("inf")
    l = 0.0
    acc = tl.full((D, ), 0, tl.float32)
    for start in range(lo, hi, BN):
        pos = start + n
        valid = pos < hi
        block = tl.load(BT + seq * bst + pos // PAGE_SIZE, valid, other=0).to(tl.int64)
        slot = pos % PAGE_SIZE
        if PHYSICAL:
            physical = tl.load(BT + seq * bst + start // PAGE_SIZE).to(tl.int64)
            offsets = tl.arange(0, PAGE_SIZE * D)
            kr = tl.load(K + physical * kb + kvh * kh + offsets)
            vr = tl.load(V + physical * vb + kvh * vh + offsets)
            kval = kr.reshape((D // 16, PAGE_SIZE, 16)).trans(1, 0, 2).reshape((PAGE_SIZE, D))
            vval = vr.reshape((PAGE_SIZE // 16, D, 16)).trans(0, 2, 1).reshape((PAGE_SIZE, D))
            valid = valid & (pos >= attention_begin)
            k = tl.where(valid[:, None], kval, 0).to(tl.float32)
        else:
            ko = block[:, None] * kb + kvh * kh + (d[None, :] // 16) * kg + slot[:, None] * kp + d[None, :] % 16
            vo = block[:, None] * vb + kvh * vh + (slot[:, None] // 16) * vp + d[None, :] * vd + slot[:, None] % 16
            k = tlx.require_layout(tl.load(K + ko, valid[:, None], other=0.0), K_LAYOUT).to(tl.float32)
        score = tlx.release_layout(tl.sum(k * q[None, :], 1)) * multiplier
        score = tl.where(valid, score, -float("inf"))
        next_m = tl.maximum(m, tl.max(score, 0))
        alpha = tl.exp2(m - next_m)
        p = tl.exp2(score - next_m)
        if PHYSICAL:
            v = tl.where(valid[:, None], vval, 0).to(tl.float32)
        else:
            v = tlx.require_layout(tl.load(V + vo, valid[:, None], other=0.0), V_LAYOUT).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        l = l * alpha + tl.sum(p, 0)
        m = next_m
    value = acc * vs / tl.maximum(l, 1.0e-20)
    if SPLITS == 1:
        tl.store(OUT + (token * HQ + head) * D + d, value)
    else:
        row = (token * HQ + head) * SPLITS + split
        tl.store(PART + row * D + d, value)
        tl.store(LSE + row, tl.where(l > 0, m + tl.log2(l), -float("inf")))


@triton.jit
def _issue_kv_page(KD, VD, BT, kbuf, vbuf, seq, bst, kvh, start, hi, buffer_slot, HKV: tl.constexpr,
                   PAGE_SIZE: tl.constexpr, LOAD_K: tl.constexpr = True, LOAD_V: tl.constexpr = True):
    """Queue selected K/V copies for one logical page into an LDS buffer slot.

    PAGE_SIZE counts tokens; BT maps start // PAGE_SIZE to a physical cache
    block. LOAD_K and LOAD_V select which tensors to copy, independently of
    page size and buffer_slot. One-warp callers can issue K and V separately
    to overlap their readiness with QK/PV. Multi-warp callers copy both with
    a fused transfer using disjoint warp hints.
    """
    valid = start < hi
    physical = tl.load(BT + seq * bst + start // PAGE_SIZE, valid, other=0)
    # TDM coordinates are int32 even when the caller uses int64 page ids.
    # The descriptor still addresses the cache through its 64-bit base/strides.
    descriptor_row = (physical * HKV + kvh).to(tl.int32)
    kd = tlx.update_tensor_descriptor(KD, add_offsets=[descriptor_row, 0], pred=valid, clamp_bounds=False)
    vd = tlx.update_tensor_descriptor(VD, add_offsets=[descriptor_row, 0], pred=valid, clamp_bounds=False)
    if tlx.num_warps() == 1:
        if LOAD_K:
            tlx.async_amd_descriptor_load(kd, tlx.local_view(kbuf, buffer_slot))
        if LOAD_V:
            tlx.async_amd_descriptor_load(vd, tlx.local_view(vbuf, buffer_slot))
    else:
        tlx.async_amd_descriptor_load_fused([(kd, tlx.local_view(kbuf, buffer_slot), 1),
                                             (vd, tlx.local_view(vbuf, buffer_slot), 2)])


@triton.jit
def _pair_window_value(q, K, V, BT, seq, bst, kvh, length, qk_scale, vs, D: tl.constexpr, PAGE_SIZE: tl.constexpr,
                       HKV: tl.constexpr, BLOCKS: tl.constexpr, KL: tl.constexpr, VL: tl.constexpr):
    # Both pages are full and inside the active window, as proved by the caller.
    # Keep page-sized WMMA layouts while sharing one stable FP32 normalization.
    kd = tl.make_tensor_descriptor(K, shape=[BLOCKS * HKV, PAGE_SIZE * D], strides=[PAGE_SIZE * D,
                                                                                    tl.constexpr(1)],
                                   block_shape=[1, PAGE_SIZE * D])
    vd = tl.make_tensor_descriptor(V, shape=[BLOCKS * HKV, PAGE_SIZE * D], strides=[PAGE_SIZE * D,
                                                                                    tl.constexpr(1)],
                                   block_shape=[1, PAGE_SIZE * D])
    kb = tlx.local_alloc((1, PAGE_SIZE * D), K.dtype.element_ty, 2)
    vb = tlx.local_alloc((1, PAGE_SIZE * D), V.dtype.element_ty, 2)
    first = length - 2 * PAGE_SIZE
    _issue_kv_page(kd, vd, BT, kb, vb, seq, bst, kvh, first, length, 0, HKV, PAGE_SIZE)
    _issue_kv_page(kd, vd, BT, kb, vb, seq, bst, kvh, first + PAGE_SIZE, length, 1, HKV, PAGE_SIZE)
    tlx.async_amd_descriptor_wait(3)
    kr0 = tlx.local_load(tlx.local_view(kb, 0), layout=KL, relaxed=True)
    k0 = kr0.reshape((D // 16, PAGE_SIZE, 16)).trans(1, 0, 2).reshape((PAGE_SIZE, D))
    score0 = tl.dot(q.to(tl.bfloat16), k0.trans().to(tl.bfloat16)) * qk_scale[:, None]
    tlx.async_amd_descriptor_wait(1)
    kr1 = tlx.local_load(tlx.local_view(kb, 1), layout=KL, relaxed=True)
    k1 = kr1.reshape((D // 16, PAGE_SIZE, 16)).trans(1, 0, 2).reshape((PAGE_SIZE, D))
    score1 = tl.dot(q.to(tl.bfloat16), k1.trans().to(tl.bfloat16)) * qk_scale[:, None]
    m = tl.max(tl.maximum(score0, score1), 1)
    p0 = tl.exp2(score0 - m[:, None])
    p1 = tl.exp2(score1 - m[:, None])
    denom = tl.sum(p0 + p1, 1)
    vr0 = tlx.local_load(tlx.local_view(vb, 0), layout=VL, relaxed=True)
    v0 = vr0.reshape((PAGE_SIZE // 16, D, 16)).trans(0, 2, 1).reshape((PAGE_SIZE, D)).to(tl.float16)
    acc = tl.dot((p0 * 32768.).to(tl.float16), v0)
    tlx.async_amd_descriptor_wait(0)
    vr1 = tlx.local_load(tlx.local_view(vb, 1), layout=VL, relaxed=True)
    v1 = vr1.reshape((PAGE_SIZE // 16, D, 16)).trans(0, 2, 1).reshape((PAGE_SIZE, D)).to(tl.float16)
    acc = tl.dot((p1 * 32768.).to(tl.float16), v1, acc=acc)
    return acc * (vs / 32768.) / tl.maximum(denom[:, None], 1.e-20)


@triton.jit
def _active_split_count(length, capacity: tl.constexpr, chunk: tl.constexpr, budget=None):
    # A device-side scalar per sequence, shared by attention and merge. Empty
    # requests retain one partition so they write a deterministic zero output.
    if chunk > 0:
        count = tl.where(length <= 512, 1, tl.minimum(capacity, tl.cdiv(length, chunk)))
        if budget is not None:
            count = tl.minimum(count, budget)
        # Count is bounded by the small static split capacity, regardless of
        # whether context metadata uses int32 or int64.
        return count.to(tl.int32)
    else:
        return capacity


@triton.jit
def _split_budget(lengths, HEAD_TASKS: tl.constexpr, TARGET: tl.constexpr):
    # Short requests write directly. Allocate the desired parallelism across
    # the requests which can benefit from splitting, using device lengths only.
    tasks = tl.sum((lengths > 512).to(tl.int32), 0) * HEAD_TASKS
    return tl.maximum(1, TARGET // tl.maximum(1, tasks))


@triton.jit
def _matrix_decode(
    Q,
    K,
    V,
    KS,
    VS,
    QS,
    BT,
    LEN,
    PART,
    LSE,
    OUT,
    qst,
    qsh,
    qsd,
    bst,
    kb,
    kh,
    kg,
    kp,
    vb,
    vh,
    vp,
    vd,
    scale,
    D: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    QLEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    Q_FP8: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    ROW_BLOCKS: tl.constexpr,
    FP8_QK: tl.constexpr,
    PRECISE_PV: tl.constexpr,
    TDM: tl.constexpr,
    BLOCKS: tl.constexpr,
    K_LOAD_LAYOUT: tl.constexpr,
    V_LOAD_LAYOUT: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
    PREFETCH_PAIR: tl.constexpr,
    SPLIT_CHUNK: tl.constexpr = 0,
    MAPPED_TASK=None,
    MAPPED_KV_HEAD=None,
    MAPPED_SPLIT=None,
    SPLIT_BUDGET=None,
):
    task = tl.program_id(0) if MAPPED_TASK is None else MAPPED_TASK
    kvh = tl.program_id(1) if MAPPED_KV_HEAD is None else MAPPED_KV_HEAD
    split = tl.program_id(2) if MAPPED_SPLIT is None else MAPPED_SPLIT
    seq = task // ROW_BLOCKS
    row_block = task % ROW_BLOCKS
    length = tl.load(LEN + seq)
    active_splits = _active_split_count(length, SPLITS, SPLIT_CHUNK, SPLIT_BUDGET)
    # This predicate is uniform across the logical block. Inactive partitions
    # skip Q/scales, descriptors, LDS setup and stores; merge uses the same count.
    if SPLIT_CHUNK == 0 or split < active_splits:
        row = row_block * BM + tl.arange(0, BM)
        query = row // G
        head = kvh * G + row % G
        token = seq * QLEN + query
        row_valid = query < QLEN
        d = tl.arange(0, D)
        n = tl.arange(0, BN)
        q = tl.load(Q + token[:, None] * qst + head[:, None] * qsh + d[None, :] * qsd, row_valid[:, None], other=0.0)
        qs = tl.full((BM, ), 1, tl.float32)
        if Q_FP8:
            qs = tl.load(QS + head)
        ks = tl.load(KS + kvh)
        vs = tl.load(VS + kvh)
        pv_dtype: tl.constexpr = tl.bfloat16 if PRECISE_PV else tl.float16
        qk_scale = scale * ks * qs * 1.4426950408889634
        qpos = length - QLEN + query
        max_pos = length - QLEN + tl.minimum(QLEN - 1, (row_block * BM + BM - 1) // G)
        min_pos = length - QLEN + row_block * BM // G
        end = length
        if CAUSAL:
            end = tl.minimum(length, max_pos + 1)
        begin = 0
        if WINDOW > 0:
            begin = tl.maximum(0, min_pos + 1 - WINDOW)
        if TDM:
            begin = begin // PAGE_SIZE * PAGE_SIZE
        span = tl.cdiv(tl.maximum(0, end - begin), active_splits * BN) * BN
        lo = begin + split * span
        hi = tl.minimum(lo + span, end)
        m = tl.full((BM, ), -float("inf"), tl.float32)
        l = tl.full((BM, ), 0, tl.float32)
        acc = tl.full((BM, D), 0, tl.float32)
        if TDM:
            kd = tl.make_tensor_descriptor(K, shape=[BLOCKS * (HQ // G), PAGE_SIZE * D],
                                           strides=[PAGE_SIZE * D, tl.constexpr(1)], block_shape=[1, PAGE_SIZE * D])
            vdsc = tl.make_tensor_descriptor(V, shape=[BLOCKS * (HQ // G), PAGE_SIZE * D],
                                             strides=[PAGE_SIZE * D, tl.constexpr(1)], block_shape=[1, PAGE_SIZE * D])
            kbuf = tlx.local_alloc((1, PAGE_SIZE * D), K.dtype.element_ty, 2)
            vbuf = tlx.local_alloc((1, PAGE_SIZE * D), V.dtype.element_ty, 2)
            _issue_kv_page(kd, vdsc, BT, kbuf, vbuf, seq, bst, kvh, lo, hi, 0, HQ // G, PAGE_SIZE)
            if PREFETCH_PAIR:
                # Queue K0/V0/K1/V1 before computing page0. Its first K/V waits
                # leave three/two copies pending; page1 resumes the normal loop.
                if lo + BN < hi:
                    _issue_kv_page(kd, vdsc, BT, kbuf, vbuf, seq, bst, kvh, lo + BN, hi, 1, HQ // G, PAGE_SIZE)
        else:
            kbuf = tlx.local_alloc((BN, D), K.dtype.element_ty, 1)
        INDEX_TILES: tl.constexpr = tlx.num_warps() == 1
        for tile_start in range(0 if SINGLE_TILE or INDEX_TILES else lo,
                                1 if SINGLE_TILE else tl.cdiv(hi - lo, BN) if INDEX_TILES else hi,
                                1 if SINGLE_TILE or INDEX_TILES else BN):
            start = lo if SINGLE_TILE else lo + tile_start * BN if INDEX_TILES else tile_start
            pos = start + n
            valid = pos < hi
            block = tl.load(BT + seq * bst + pos // PAGE_SIZE, valid, other=0).to(tl.int64)
            slot = pos % PAGE_SIZE
            if TDM:
                buffer_slot = (((start - lo) // BN) % 2).to(tl.int32)
                if PREFETCH_PAIR and start == lo and start + BN < hi:
                    tlx.async_amd_descriptor_wait(3)
                else:
                    tlx.async_amd_descriptor_wait(1 if tlx.num_warps() == 1 else 0)
                kr = tlx.local_load(tlx.local_view(kbuf, buffer_slot), layout=K_LOAD_LAYOUT,
                                    relaxed=tlx.num_warps() == 1)
                if tlx.num_warps() != 1:
                    vr = tlx.local_load(tlx.local_view(vbuf, buffer_slot), layout=V_LOAD_LAYOUT,
                                        relaxed=tlx.num_warps() == 1)
                if (not PREFETCH_PAIR or start != lo) and (tlx.num_warps() != 1 or
                                                           (not SINGLE_TILE and start + BN < hi)):
                    _issue_kv_page(kd, vdsc, BT, kbuf, vbuf, seq, bst, kvh, start + BN, hi, 1 - buffer_slot, HQ // G,
                                   PAGE_SIZE, LOAD_V=tlx.num_warps() != 1)
                kval = kr.reshape((D // 16, PAGE_SIZE, 16)).trans(1, 0, 2).reshape((PAGE_SIZE, D))
                # Invalid K columns are removed from logits after QK. Invalid V
                # must be zero before PV: zero probability does not sanitize NaN.
                kt = tl.trans(kval)
                if tlx.num_warps() != 1:
                    vval = vr.reshape((PAGE_SIZE // 16, D, 16)).trans(0, 2, 1).reshape((PAGE_SIZE, D))
                    if start + BN <= hi:
                        v = vval.to(pv_dtype)
                    else:
                        v = tl.where(valid[:, None], vval, 0).to(pv_dtype)
            else:
                ko = block[:, None] * kb + kvh * kh + (d[None, :] // 16) * kg + slot[:, None] * kp + d[None, :] % 16
                vo = block[:, None] * vb + kvh * vh + (slot[:, None] // 16) * vp + d[None, :] * vd + slot[:, None] % 16
                kcopy = tlx.async_load(K + ko, tlx.local_view(kbuf, 0), mask=valid[:, None])
                tlx.async_load_commit_group([kcopy])
                v = tl.load(V + vo, valid[:, None], other=0.0).to(pv_dtype)
                ready = tlx.async_load_wait_group(0)
                kt = tlx.local_load(tlx.local_trans(tlx.local_view(kbuf, 0)), token=ready)
            if FP8_QK:
                score = tl.dot(q.to(kt.dtype), kt)
            else:
                score = tl.dot(q.to(tl.bfloat16), kt.to(tl.bfloat16))
            score *= qk_scale[:, None]
            mask = valid[None, :]
            if QLEN == 1:
                if WINDOW > 0:
                    mask &= pos[None, :] >= length - WINDOW
            else:
                if CAUSAL:
                    mask = mask & (pos[None, :] <= qpos[:, None])
                if WINDOW > 0:
                    mask = mask & (pos[None, :] > qpos[:, None] - WINDOW)
            score = tl.where(mask, score, -float("inf"))
            next_m = tl.maximum(m, tl.max(score, 1))
            safe_m = tl.where(next_m == -float("inf"), 0.0, next_m)
            alpha = tl.exp2(m - safe_m)
            p = tl.exp2(score - safe_m[:, None])
            l = l * alpha + tl.sum(p, 1)
            acc *= alpha[:, None]
            if TDM and tlx.num_warps() == 1:
                if PREFETCH_PAIR and start == lo and start + BN < hi:
                    tlx.async_amd_descriptor_wait(2)
                elif not SINGLE_TILE and start + BN < hi:
                    tlx.async_amd_descriptor_wait(1)
                else:
                    tlx.async_amd_descriptor_wait(0)
                vr = tlx.local_load(tlx.local_view(vbuf, buffer_slot), layout=V_LOAD_LAYOUT,
                                    relaxed=tlx.num_warps() == 1)
                if (not PREFETCH_PAIR or start != lo) and not SINGLE_TILE and start + BN < hi:
                    _issue_kv_page(kd, vdsc, BT, kbuf, vbuf, seq, bst, kvh, start + BN, hi, 1 - buffer_slot, HQ // G,
                                   PAGE_SIZE, LOAD_K=False)
                vval = vr.reshape((PAGE_SIZE // 16, D, 16)).trans(0, 2, 1).reshape((PAGE_SIZE, D))
                if start + BN <= hi:
                    v = vval.to(pv_dtype)
                else:
                    v = tl.where(valid[:, None], vval, 0).to(pv_dtype)
            if PRECISE_PV:
                p_hi = p.to(tl.bfloat16)
                acc += tl.dot(p_hi, v)
                p_lo = (p - p_hi.to(tl.float32)).to(tl.bfloat16)
                acc += tl.dot(p_lo, v)
            else:
                acc += tl.dot((p * 32768.0).to(tl.float16), v)
            m = next_m
        numerator_scale: tl.constexpr = 1.0 if PRECISE_PV else 1.0 / 32768.0
        value = acc * (vs * numerator_scale) / tl.maximum(l[:, None], 1.0e-20)
        if SPLITS == 1 or (SPLIT_CHUNK > 0 and active_splits == 1):
            tl.store(OUT + (token[:, None] * HQ + head[:, None]) * D + d[None, :], value, row_valid[:, None])
        else:
            offset = ((token[:, None] * HQ + head[:, None]) * SPLITS + split) * D + d[None, :]
            tl.store(PART + offset, value, row_valid[:, None])
            tl.store(LSE + (token * HQ + head) * SPLITS + split, tl.where(l > 0, m + tl.log2(l), -float("inf")),
                     row_valid)


@triton.jit
def _batched_decode(Q, K, V, KS, VS, QS, BT, LEN, PART, LSE, OUT, QST: tl.constexpr, QSH: tl.constexpr,
                    QSD: tl.constexpr, BST: tl.constexpr, scale, D: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
                    BATCH: tl.constexpr, PAGE_SIZE: tl.constexpr, SPLITS: tl.constexpr, BM: tl.constexpr,
                    BLOCKS: tl.constexpr, KL: tl.constexpr, VL: tl.constexpr, Q_FP8: tl.constexpr, FP8_QK: tl.constexpr,
                    PREFETCH_PAIR: tl.constexpr, SPLIT_CHUNK: tl.constexpr, GRID: tl.constexpr, SEQUENCES: tl.constexpr,
                    TARGET: tl.constexpr):
    # Every block computes the same device-side work prefix. No host length
    # read, global queue counter, preparation kernel or graph update is needed.
    seq_base = tl.program_id(1) * SEQUENCES
    seq_ids = seq_base + tl.arange(0, SEQUENCES)
    lengths = tl.load(LEN + seq_ids, seq_ids < BATCH, other=0)
    ROWS: tl.constexpr = triton.cdiv(G, BM)
    HEAD_TASKS: tl.constexpr = (HQ // G) * ROWS
    budget = _split_budget(lengths, HEAD_TASKS, TARGET)
    counts = tl.where(seq_ids < BATCH, _active_split_count(lengths, SPLITS, SPLIT_CHUNK, budget), 0)
    ends = tl.cumsum(counts, 0)
    partitions = tl.sum(counts, 0)
    tasks = partitions * HEAD_TASKS
    for work in range(tl.program_id(0), tasks, GRID):
        partition = work // HEAD_TASKS
        seq = seq_base + tl.sum(((partition >= ends) & (seq_ids < BATCH)).to(tl.int32), 0)
        first = tl.sum(tl.where(seq_ids < seq, counts, 0), 0)
        task = seq * ROWS + work % ROWS
        kv_head = work // ROWS % (HQ // G)
        _matrix_decode(Q, K, V, KS, VS, QS, BT, LEN, PART, LSE, OUT, QST, QSH, QSD, BST, (HQ // G) * PAGE_SIZE * D,
                       PAGE_SIZE * D, PAGE_SIZE * 16, 16, (HQ // G) * PAGE_SIZE * D, PAGE_SIZE * D, D * 16, 16, scale,
                       D, HQ, G, 1, PAGE_SIZE, SPLITS, BM, PAGE_SIZE, Q_FP8, True, 0, ROWS, FP8_QK, False, True, BLOCKS,
                       KL, VL, False, PREFETCH_PAIR, SPLIT_CHUNK, task, kv_head, partition - first, budget)


@triton.jit
def _small_window_decode(Q, K, V, KS, VS, BT, LEN, OUT, scale, QST: tl.constexpr, QSH: tl.constexpr, QSD: tl.constexpr,
                         BST: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, PAGE_SIZE: tl.constexpr,
                         BLOCKS: tl.constexpr, BM: tl.constexpr, ROWS: tl.constexpr, KL: tl.constexpr, VL: tl.constexpr,
                         WINDOW: tl.constexpr):
    # An aligned context and WINDOW <= PAGE_SIZE need only the final physical page.
    # The block-uniform guard preserves the masked path for empty/ragged inputs.
    task = tl.program_id(0)
    kvh = tl.program_id(1)
    seq = task // ROWS
    length = tl.load(LEN + seq)
    if length >= PAGE_SIZE and length % PAGE_SIZE == 0:
        row = (task % ROWS) * BM + tl.arange(0, BM)
        row_valid = row < G
        head = kvh * G + row % G
        d = tl.arange(0, D)
        n = tl.arange(0, PAGE_SIZE)
        q = tl.load(Q + seq * QST + head[:, None] * QSH + d[None, :] * QSD, row_valid[:, None], other=0.)
        ks = tl.load(KS + kvh)
        vs = tl.load(VS + kvh)
        kd = tl.make_tensor_descriptor(K, shape=[BLOCKS * (HQ // G), PAGE_SIZE * D],
                                       strides=[PAGE_SIZE * D, tl.constexpr(1)], block_shape=[1, PAGE_SIZE * D])
        vd = tl.make_tensor_descriptor(V, shape=[BLOCKS * (HQ // G), PAGE_SIZE * D],
                                       strides=[PAGE_SIZE * D, tl.constexpr(1)], block_shape=[1, PAGE_SIZE * D])
        kb = tlx.local_alloc((1, PAGE_SIZE * D), K.dtype.element_ty, 1)
        vb = tlx.local_alloc((1, PAGE_SIZE * D), V.dtype.element_ty, 1)
        _issue_kv_page(kd, vd, BT, kb, vb, seq, BST, kvh, length - PAGE_SIZE, length, 0, HQ // G, PAGE_SIZE)
        tlx.async_amd_descriptor_wait(0)
        kr = tlx.local_load(tlx.local_view(kb, 0), layout=KL, relaxed=tlx.num_warps() == 1)
        kval = kr.reshape((D // 16, PAGE_SIZE, 16)).trans(1, 0, 2).reshape((PAGE_SIZE, D))
        score = tl.dot(q.to(tl.bfloat16), kval.trans().to(tl.bfloat16)) * (ks * scale * 1.4426950408889634)
        active = n >= PAGE_SIZE - WINDOW
        score = tl.where(active[None, :], score, -float('inf'))
        m = tl.max(score, 1)
        p = tl.exp2(score - m[:, None])
        denom = tl.sum(p, 1)
        vr = tlx.local_load(tlx.local_view(vb, 0), layout=VL, relaxed=tlx.num_warps() == 1)
        vval = vr.reshape((PAGE_SIZE // 16, D, 16)).trans(0, 2, 1).reshape((PAGE_SIZE, D))
        vval = tl.where(active[:, None], vval, 0).to(tl.float16)
        acc = tl.dot((p * 32768.).to(tl.float16), vval)
        value = acc * (vs / 32768.) / denom[:, None]
        tl.store(OUT + (seq * HQ + head[:, None]) * D + d[None, :], value, row_valid[:, None])
    else:
        _matrix_decode(Q, K, V, KS, VS, KS, BT, LEN, OUT, OUT, OUT, QST, QSH, QSD, BST, (HQ // G) * PAGE_SIZE * D,
                       PAGE_SIZE * D, PAGE_SIZE * 16, 16, (HQ // G) * PAGE_SIZE * D, PAGE_SIZE * D, D * 16, 16, scale,
                       D, HQ, G, 1, PAGE_SIZE, 1, BM, PAGE_SIZE, False, True, WINDOW, ROWS, False, False, True, BLOCKS,
                       KL, VL, False, False)


@triton.jit
def _window_decode(Q, K, V, KS, VS, BT, LEN, OUT, scale, QST: tl.constexpr, QSH: tl.constexpr, QSD: tl.constexpr,
                   BST: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, PAGE_SIZE: tl.constexpr,
                   BLOCKS: tl.constexpr, BM: tl.constexpr, ROWS: tl.constexpr, KL: tl.constexpr, VL: tl.constexpr):
    # A smaller launch signature avoids loading unused split/stride arguments.
    # This condition is uniform across each logical block, including its waits.
    task = tl.program_id(0)
    kvh = tl.program_id(1)
    seq = task // ROWS
    length = tl.load(LEN + seq)
    if length >= 2 * PAGE_SIZE and length % PAGE_SIZE == 0:
        row = (task % ROWS) * BM + tl.arange(0, BM)
        valid = row < G
        head = kvh * G + row % G
        d = tl.arange(0, D)
        q = tl.load(Q + seq * QST + head[:, None] * QSH + d[None, :] * QSD, valid[:, None], other=0.)
        ks = tl.load(KS + kvh)
        vs = tl.load(VS + kvh)
        qk_scale = tl.full((BM, ), 1., tl.float32) * (ks * scale * 1.4426950408889634)
        value = _pair_window_value(q, K, V, BT, seq, BST, kvh, length, qk_scale, vs, D, PAGE_SIZE, HQ // G, BLOCKS, KL,
                                   VL)
        tl.store(OUT + (seq * HQ + head[:, None]) * D + d[None, :], value, valid[:, None])
    else:
        # Packed K/V strides are fixed by the contiguous physical cache contract.
        _matrix_decode(Q, K, V, KS, VS, KS, BT, LEN, OUT, OUT, OUT, QST, QSH, QSD, BST, (HQ // G) * PAGE_SIZE * D,
                       PAGE_SIZE * D, PAGE_SIZE * 16, 16, (HQ // G) * PAGE_SIZE * D, PAGE_SIZE * D, D * 16, 16, scale,
                       D, HQ, G, 1, PAGE_SIZE, 1, BM, PAGE_SIZE, False, True, 2 * PAGE_SIZE, ROWS, False, False, True,
                       BLOCKS, KL, VL, False, False)


@triton.jit
def _merge_values(PART, LSE, OUT, row, active_splits, D: tl.constexpr, SPLITS: tl.constexpr, BS: tl.constexpr,
                  BD: tl.constexpr):
    s = tl.arange(0, BS)
    if BD == D:
        d = tl.arange(0, D)
    else:
        d = tl.program_id(1) * BD + tl.arange(0, BD)
    lse = tl.load(LSE + row * SPLITS + s, s < active_splits, other=-float("inf"))
    m = tl.max(lse, 0)
    m = tl.where(m == -float("inf"), 0.0, m)
    p = tl.exp2(lse - m)
    vals = tl.load(PART + (row * SPLITS + s[:, None]) * D + d[None, :], s[:, None] < active_splits, other=0.0)
    value = tl.sum(vals * p[:, None], 0) / tl.maximum(tl.sum(p, 0), 1.0e-20)
    tl.store(OUT + row * D + d, value)


@triton.jit
def _merge(PART, LSE, OUT, LEN, HQ: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr, BS: tl.constexpr,
           BD: tl.constexpr, SPLIT_CHUNK: tl.constexpr = 0, BATCH: tl.constexpr = 0, SEQUENCES: tl.constexpr = 0,
           HEAD_TASKS: tl.constexpr = 0, TARGET: tl.constexpr = 0):
    row = tl.program_id(0)
    active_splits = SPLITS
    if SPLIT_CHUNK > 0:
        seq = row // HQ
        length = tl.load(LEN + seq)
        active_splits = 1
        if length > 512:
            if TARGET > 0:
                ids = seq // SEQUENCES * SEQUENCES + tl.arange(0, SEQUENCES)
                lengths = tl.load(LEN + ids, ids < BATCH, other=0)
                budget = _split_budget(lengths, HEAD_TASKS, TARGET)
                active_splits = _active_split_count(length, SPLITS, SPLIT_CHUNK, budget)
            else:
                active_splits = _active_split_count(length, SPLITS, SPLIT_CHUNK)
    # One-partition sequences were written directly by the attention kernel.
    if SPLIT_CHUNK == 0 or active_splits > 1:
        if SPLIT_CHUNK > 0 and BS > 32:
            # Small active counts need a small reduction tile even when the
            # graph reserved hundreds of partitions. Keep large merges vectorized.
            if active_splits <= 32:
                _merge_values(PART, LSE, OUT, row, active_splits, D, SPLITS, 32, BD)
            else:
                _merge_values(PART, LSE, OUT, row, active_splits, D, SPLITS, BS, BD)
        else:
            _merge_values(PART, LSE, OUT, row, active_splits, D, SPLITS, BS, BD)


def decode(q, k, v, lengths, table, ks, vs, *, q_scale, query_length, causal, window_size, sm_scale, num_splits,
           variant, out=None, workspace=None, block_n=64, block_m=16, num_warps=4, prefetch_pair=True, split_chunk=0,
           partition_grid=0, sequence_group=0, split_target=0):
    """Launch the selected matrix/vector path or an eligible window fast path.

    The window branches below are specializations of matrix_tdm_flat, not
    independent public variants. Their runtime guards fall back to masked
    _matrix_decode without reading context lengths on the host. See the
    module docstring for path/layout semantics. Multi-split paths then call
    _merge; normalized partials, softmax, log-sum-exp and merge arithmetic
    remain FP32. Keeping partials in FP32 preserves small residuals when
    positive and negative partitions cancel.

    prefetch_pair primes the pipeline with two KV pages in the prologue before
    computation begins for eligible single-token decode configurations.
    """
    tokens, heads, dim = q.shape
    hkv, page_size = k.shape[1], k.shape[3]
    group = heads // hkv
    if out is None:
        out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    if out.shape != q.shape or out.dtype != torch.bfloat16 or out.device != q.device or not out.is_contiguous():
        raise ValueError("out must be contiguous BF16 with Q's shape/device")
    mid_shape, lse_shape = (tokens, heads, num_splits, dim), (tokens, heads, num_splits)
    if workspace is None:
        if num_splits > 1:
            mid = torch.empty(mid_shape, device=q.device, dtype=torch.float32)
            lse = torch.empty(lse_shape, device=q.device, dtype=torch.float32)
        else:
            mid = lse = out  # constexpr dead pointers, never dereferenced
    else:
        mid, lse = workspace
        if (mid.shape != mid_shape or lse.shape != lse_shape or mid.dtype != torch.float32 or lse.dtype != torch.float32
                or mid.device != q.device or lse.device != q.device or not mid.is_contiguous()
                or not lse.is_contiguous()):
            raise ValueError("workspace must be contiguous FP32 partial/LSE tensors for this workload")
    strides = (*q.stride(), table.stride(0), *k.stride()[:4], *v.stride()[:4])
    if variant in ("vector", "vector_page"):
        kl, vl = _vector_layouts(block_n, dim, num_warps)
        _vector_decode[(tokens, heads, num_splits)](q, k, v, ks, vs, q_scale if q_scale is not None else ks, table,
                                                    lengths, mid, lse, out, *strides, sm_scale, dim, heads, group,
                                                    query_length, page_size, num_splits, block_n,
                                                    q.dtype == torch.float8_e4m3fn, causal, window_size, kl, vl,
                                                    variant == "vector_page", num_warps=num_warps)
    else:
        rows = triton.cdiv(query_length * group, block_m)
        fp8_qk = variant != "matrix_precise" and q.dtype == torch.float8_e4m3fn and dim == 64
        kload, vload = _packed_operand_layouts(
            page_size, dim, fp8_qk) if variant == "matrix_tdm_flat" and num_warps == 1 else (None, None)
        visible_bound = min(table.shape[1] * page_size, window_size + page_size -
                            1) if window_size else table.shape[1] * page_size
        single_tile = variant == "matrix_tdm_flat" and query_length == 1 and num_warps == 1 and visible_bound <= num_splits * block_n
        prefetch_pair_enabled = (prefetch_pair and variant == "matrix_tdm_flat" and num_warps == 1 and not single_tile
                                 and q.dtype == torch.bfloat16 and window_size == 0 and query_length == 1
                                 and visible_bound >= 8192
                                 and ((page_size == 64 and hkv <= 2) or heads == hkv and hkv >= 64))
        if (variant == "matrix_tdm_flat" and query_length == 1 and 0 < window_size <= page_size
                and (page_size == 128 or dim == 64 and heads == hkv and hkv >= 32) and num_splits == 1
                and q.dtype == torch.bfloat16):
            # The D128 one-warp fallback benefits from this allocation policy.
            _small_window_decode[(tokens * rows, hkv, 1)](q, k, v, ks, vs, table, lengths, out, sm_scale, *q.stride(),
                                                          table.stride(0), dim, heads, group, page_size, k.shape[0],
                                                          block_m, rows, kload, vload, window_size, num_warps=num_warps,
                                                          matrix_instr_nonkdim=16, reverse_local_assignment=dim == 128
                                                          and num_warps == 1)
        elif (variant == "matrix_tdm_flat" and query_length == 1 and window_size == 128 and page_size == 64
              and dim == 128 and num_splits == 1 and num_warps == 1 and q.dtype == torch.bfloat16):
            _window_decode[(tokens * rows, hkv, 1)](q, k, v, ks, vs, table, lengths, out, sm_scale, *q.stride(),
                                                    table.stride(0), dim, heads, group, page_size, k.shape[0], block_m,
                                                    rows, kload, vload, num_warps=1, matrix_instr_nonkdim=16)
        elif partition_grid:
            _batched_decode[(partition_grid, triton.cdiv(tokens, sequence_group))](
                q, k, v, ks, vs, q_scale if q_scale is not None else ks, table, lengths, mid, lse, out, *q.stride(),
                table.stride(0), sm_scale, dim, heads, group, tokens, page_size, num_splits, block_m, k.shape[0], kload,
                vload, q.dtype == torch.float8_e4m3fn, fp8_qk, prefetch_pair_enabled, split_chunk, partition_grid,
                sequence_group, split_target, num_warps=num_warps, matrix_instr_nonkdim=16)
        else:
            _matrix_decode[(tokens // query_length * rows, hkv, num_splits)](
                q, k, v, ks, vs, q_scale if q_scale is not None else ks, table, lengths, mid, lse, out, *strides,
                sm_scale, dim, heads, group, query_length, page_size, num_splits, block_m, block_n,
                q.dtype == torch.float8_e4m3fn, causal, window_size, rows, fp8_qk,
                variant == "matrix_precise", variant == "matrix_tdm_flat", k.shape[0], kload, vload, single_tile,
                prefetch_pair_enabled, split_chunk, num_warps=num_warps, matrix_instr_nonkdim=16)
    if num_splits > 1:
        # Dimension tiles spread sparse merges across more CUs. D64 needs
        # enough partitions to amortize the extra workgroup setup.
        merge_d = 32 if (tokens * heads <= 32 and q.dtype == torch.bfloat16 and num_splits >= 16 and
                         (dim == 128 or dim == 64 and query_length == 1 and group > 1 and num_splits >= 128)) else dim
        _merge[(tokens * heads, triton.cdiv(dim, merge_d))](
            mid, lse, out, lengths, heads, dim, num_splits, triton.next_power_of_2(num_splits), merge_d, split_chunk,
            tokens, sequence_group, hkv * triton.cdiv(group, 16), split_target,
            num_warps=(1 if num_splits * merge_d <= 8192 else 4 if num_splits * merge_d <= 32768 else 8))
    return out


def _select_decode_config(q, key_cache, *, max_context_len, query_length=1, window_size=0, num_splits=None):
    """Choose a private execution path and split/warp geometry.

    page_size counts cache tokens per physical block, not bytes or a load
    selector. Page sizes 64/128 use packed TDM matrix paths; tiny contexts
    on other valid page sizes use vector reductions, and larger ones use
    masked matrix loads. Window specializations are chosen in decode after
    this policy establishes a one-split matrix configuration. Shapes outside
    the measured families retain heuristic scheduling; model metadata is
    never inferred. For automatic batched single-token packed decode, the bound
    sets partition capacity while actual device lengths determine active work.
    An explicit num_splits retains a fixed split plan.
    """
    tokens, heads, dim = q.shape
    kv_heads, page_size = key_cache.shape[1], key_cache.shape[3]
    effective = max(1, min(max_context_len, window_size or max_context_len))
    group = heads // kv_heads
    if query_length == 1 and effective <= 128:
        cfg = dict(
            variant="matrix_tdm_flat" if page_size in (64, 128) else "vector",
            block_n=page_size if page_size in (64, 128) else min(128, max(32, triton.next_power_of_2(effective))),
            block_m=16, num_warps=1 if (dim == 128 and effective > 64) or (dim == 64 and kv_heads >= 32) else 4,
            num_splits=1)
    elif query_length == 1 and effective >= 65536 and kv_heads <= 8:
        cfg = dict(variant="matrix_tdm_flat" if page_size in (64, 128) else "matrix",
                   block_n=page_size if page_size in (64, 128) else 32, block_m=16, num_warps=1,
                   num_splits=min(512, max(128, triton.cdiv(768 if dim == 128 else 512, kv_heads))))
    elif page_size in (64, 128):
        bm = 16 if query_length == 1 else 64
        tasks = tokens // query_length * kv_heads * triton.cdiv(query_length * group, bm)
        if query_length == 1:
            if kv_heads >= 32:
                chunk = 1024 if page_size == 128 and effective >= 8192 else 3 * page_size if dim == 128 else 256
                split_cap = 128
            else:
                # One-page partitions expose more parallel work in medium
                # BF16-Q/D128 grouped decode; longer loops retain amortization.
                chunk = 2 * page_size if dim == 128 and page_size == 64 and effective >= 4096 and (
                    q.dtype != torch.bfloat16 or effective > 8192 or tasks > 4) else page_size
                split_cap = 256 if page_size == 128 and dim == 64 else 128
            splits = min(split_cap, triton.cdiv(effective, chunk))
        else:
            splits = min(triton.cdiv(effective, page_size), max(1, triton.cdiv(384, tasks)), 128)
        if query_length == 1 and q.dtype == torch.bfloat16 and effective > 128:
            if kv_heads >= 32 and effective >= 8192:
                # Long dense partitions amortize descriptor/setup traffic;
                # excessive splitting increases scheduling and merge work.
                # Maintain enough CTAs for smaller KV-head grids.
                if dim == 128:
                    splits = max(12, triton.cdiv(768, tasks))
                elif page_size == 64:
                    splits = min(32, max(triton.cdiv(1024, tasks), triton.cdiv(effective, 1024)))
                else:
                    splits = min(32, max(12, triton.cdiv(1024, tasks), triton.cdiv(effective, 4096)))
            elif dim == 64 and kv_heads < 32 and page_size == 64:
                # Sparse grouped grids need more independent page partitions.
                pages = triton.cdiv(effective, page_size)
                splits = min(256, triton.next_power_of_2(pages))
                if tokens > query_length:
                    # Independent requests already supply parallel work.
                    splits = min(splits, pages)
        cfg = dict(
            variant="matrix_tdm_flat", block_n=page_size, block_m=bm, num_warps=1 if query_length == 1 and
            (kv_heads >= 32 or dim == 128 or page_size == 128 or q.dtype == torch.bfloat16 and effective > 128)
            and not (dim == 128 and q.dtype == torch.float8_e4m3fn and kv_heads < 32 and page_size == 64) else 4,
            num_splits=splits)
    else:
        cfg = dict(variant="matrix", block_n=32, block_m=16, num_warps=4,
                   num_splits=min(128, triton.cdiv(effective, 128)))
    # Tiny grouped D64 grids benefit from their existing page parallelism;
    # adapting the two-request control costs more than its small merge saves.
    tiny_grouped = dim == 64 and tokens * kv_heads * triton.cdiv(group, 16) <= 4
    if (num_splits is None and query_length == 1 and tokens > 1 and page_size in (64, 128) and window_size == 0
            and effective > 128 and not tiny_grouped):
        # Bound launch size while retaining enough partition capacity for a
        # long outlier. The device work prefix packs actual per-sequence splits
        # onto this grid. Explicit overrides and single-request plans are fixed.
        chunk = 256
        target = (512 if dim == 128 else 1024) if kv_heads < 32 else (3072 if dim == 128 else 4096)
        cfg["num_splits"] = 1 if effective <= 512 else min(cfg["num_splits"], triton.cdiv(effective, chunk))
        if cfg["num_splits"] > 1:
            cfg["split_chunk"] = chunk
            if tokens >= 8:
                # Small batches keep flat dispatch; its leaner loop wins the
                # measured long-context controls. Larger batches pack requests
                # in groups so the prefix cost stays bounded.
                seq_group = min(64, triton.next_power_of_2(tokens))
                groups = triton.cdiv(tokens, seq_group)
                group_tasks = min(tokens, seq_group) * kv_heads * triton.cdiv(group, 16)
                cfg["sequence_group"] = seq_group
                cfg["split_target"] = max(1, target // groups)
                # A request needing splits has at least HEAD_TASKS live work.
                # Its runtime budget therefore cannot exceed this static cap.
                # Clip unreachable scratch slots without changing any partition.
                head_tasks = kv_heads * triton.cdiv(group, 16)
                cfg["num_splits"] = min(cfg["num_splits"], max(1, cfg["split_target"] // head_tasks))
                cfg["partition_grid"] = min(group_tasks * cfg["num_splits"], max(cfg["split_target"], group_tasks))
                if cfg["num_splits"] == 1:
                    for option in ("split_chunk", "sequence_group", "split_target", "partition_grid"):
                        cfg.pop(option)
    if num_splits is not None:
        cfg["num_splits"] = num_splits
    return cfg


def paged_attention_decode_fp8(query, key_cache, value_cache, context_lens, block_tables, k_scale, v_scale, *,
                               max_context_len, q_scale=None, query_length=1, causal=True, sliding_window=0,
                               sm_scale=None, num_splits=None, out=None, workspace=None):
    config = _select_decode_config(query, key_cache, max_context_len=max_context_len, query_length=query_length,
                                   window_size=sliding_window, num_splits=num_splits)
    return decode(query, key_cache, value_cache, context_lens, block_tables, k_scale, v_scale, q_scale=q_scale,
                  query_length=query_length, causal=causal, window_size=sliding_window, sm_scale=sm_scale, out=out,
                  workspace=workspace, **config)


def paged_attention_decode_fp8_workspace(query, key_cache, value_cache, block_tables, *, max_context_len,
                                         query_length=1, sliding_window=0, num_splits=None):
    config = _select_decode_config(query, key_cache, max_context_len=max_context_len, query_length=query_length,
                                   window_size=sliding_window, num_splits=num_splits)
    splits = config["num_splits"]
    if splits == 1:
        return None
    tokens, heads, dim = query.shape
    return (torch.empty((tokens, heads, splits, dim), dtype=torch.float32, device=query.device),
            torch.empty((tokens, heads, splits), dtype=torch.float32, device=query.device))
