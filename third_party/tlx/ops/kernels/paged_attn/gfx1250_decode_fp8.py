"""gfx1250 implementation for TLX paged FP8 attention with per-head scales.

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
def _packed_operand_layouts(page, dim, fp8_qk):
    if fp8_qk:
        k = tlx.layout(shape=((16, 2), (16, dim // 32, page // 16)), stride=((16, page * 16), (1, page * 32, 256)))
    else:
        k = tlx.layout(shape=((16, 2), (8, dim // 16, page // 16)), stride=((16, 8), (1, page * 16, 256)))
    v = tlx.layout(shape=((16, 2), (8, page // 16, dim // 16)), stride=((16, 8), (1, dim * 16, 256)))
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
    PAGE: tl.constexpr,
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
        begin = begin // PAGE * PAGE
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
        block = tl.load(BT + seq * bst + pos // PAGE, valid, other=0).to(tl.int64)
        slot = pos % PAGE
        if PHYSICAL:
            physical = tl.load(BT + seq * bst + start // PAGE).to(tl.int64)
            offsets = tl.arange(0, PAGE * D)
            kr = tl.load(K + physical * kb + kvh * kh + offsets)
            vr = tl.load(V + physical * vb + kvh * vh + offsets)
            kval = kr.reshape((D // 16, PAGE, 16)).trans(1, 0, 2).reshape((PAGE, D))
            vval = vr.reshape((PAGE // 16, D, 16)).trans(0, 2, 1).reshape((PAGE, D))
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
    output_scale = 1.0 if SPLITS > 1 and PART.dtype.element_ty == tl.float16 else vs
    value = acc * output_scale / tl.maximum(l, 1.0e-20)
    if SPLITS == 1:
        tl.store(OUT + (token * HQ + head) * D + d, value)
    else:
        row = (token * HQ + head) * SPLITS + split
        tl.store(PART + row * D + d, value)
        tl.store(LSE + row, tl.where(l > 0, m + tl.log2(l), -float("inf")))


@triton.jit
def _issue_flat_page(KD, VD, BT, kbuf, vbuf, seq, bst, kvh, start, hi, slot, HKV: tl.constexpr, PAGE: tl.constexpr,
                     WHICH: tl.constexpr = 0):
    valid = start < hi
    physical = tl.load(BT + seq * bst + start // PAGE, valid, other=0)
    kd = tlx.update_tensor_descriptor(KD, add_offsets=[physical * HKV + kvh, 0], pred=valid, clamp_bounds=False)
    vd = tlx.update_tensor_descriptor(VD, add_offsets=[physical * HKV + kvh, 0], pred=valid, clamp_bounds=False)
    if tlx.num_warps() == 1:
        if WHICH != 2:
            tlx.async_amd_descriptor_load(kd, tlx.local_view(kbuf, slot))
        if WHICH != 1:
            tlx.async_amd_descriptor_load(vd, tlx.local_view(vbuf, slot))
    else:
        tlx.async_amd_descriptor_load_fused([(kd, tlx.local_view(kbuf, slot), 1), (vd, tlx.local_view(vbuf, slot), 2)])


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
    PAGE: tl.constexpr,
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
    PREFILL_PAIR: tl.constexpr,
):
    task = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    seq = task // ROW_BLOCKS
    row_block = task % ROW_BLOCKS
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
    length = tl.load(LEN + seq)
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
        begin = begin // PAGE * PAGE
    span = tl.cdiv(tl.maximum(0, end - begin), SPLITS * BN) * BN
    lo = begin + split * span
    hi = tl.minimum(lo + span, end)
    m = tl.full((BM, ), -float("inf"), tl.float32)
    l = tl.full((BM, ), 0, tl.float32)
    acc = tl.full((BM, D), 0, tl.float32)
    if TDM:
        kd = tl.make_tensor_descriptor(K, shape=[BLOCKS * (HQ // G), PAGE * D], strides=[PAGE * D,
                                                                                         tl.constexpr(1)],
                                       block_shape=[1, PAGE * D])
        vdsc = tl.make_tensor_descriptor(V, shape=[BLOCKS * (HQ // G), PAGE * D], strides=[PAGE * D,
                                                                                           tl.constexpr(1)],
                                         block_shape=[1, PAGE * D])
        kbuf = tlx.local_alloc((1, PAGE * D), K.dtype.element_ty, 2)
        vbuf = tlx.local_alloc((1, PAGE * D), V.dtype.element_ty, 2)
        _issue_flat_page(kd, vdsc, BT, kbuf, vbuf, seq, bst, kvh, lo, hi, 0, HQ // G, PAGE)
        if PREFILL_PAIR:
            # Queue K0/V0/K1/V1 before computing page0. Its first K/V waits
            # leave three/two copies pending; page1 resumes the normal loop.
            if lo + BN < hi:
                _issue_flat_page(kd, vdsc, BT, kbuf, vbuf, seq, bst, kvh, lo + BN, hi, 1, HQ // G, PAGE)
    else:
        kbuf = tlx.local_alloc((BN, D), K.dtype.element_ty, 1)
    INDEX_TILES: tl.constexpr = tlx.num_warps() == 1
    for tile_start in range(0 if SINGLE_TILE or INDEX_TILES else lo,
                            1 if SINGLE_TILE else tl.cdiv(hi - lo, BN) if INDEX_TILES else hi,
                            1 if SINGLE_TILE or INDEX_TILES else BN):
        start = lo if SINGLE_TILE else lo + tile_start * BN if INDEX_TILES else tile_start
        pos = start + n
        valid = pos < hi
        block = tl.load(BT + seq * bst + pos // PAGE, valid, other=0).to(tl.int64)
        slot = pos % PAGE
        if TDM:
            buffer_slot = ((start - lo) // BN) % 2
            if PREFILL_PAIR and start == lo and start + BN < hi:
                tlx.async_amd_descriptor_wait(3)
            else:
                tlx.async_amd_descriptor_wait(1 if tlx.num_warps() == 1 else 0)
            kr = tlx.local_load(tlx.local_view(kbuf, buffer_slot), layout=K_LOAD_LAYOUT, relaxed=tlx.num_warps() == 1)
            if tlx.num_warps() != 1:
                vr = tlx.local_load(tlx.local_view(vbuf, buffer_slot), layout=V_LOAD_LAYOUT,
                                    relaxed=tlx.num_warps() == 1)
            if (not PREFILL_PAIR or start != lo) and (tlx.num_warps() != 1 or (not SINGLE_TILE and start + BN < hi)):
                _issue_flat_page(kd, vdsc, BT, kbuf, vbuf, seq, bst, kvh, start + BN, hi, 1 - buffer_slot, HQ // G,
                                 PAGE, 1 if tlx.num_warps() == 1 else 0)
            kval = kr.reshape((D // 16, PAGE, 16)).trans(1, 0, 2).reshape((PAGE, D))
            # Invalid K columns are removed from logits after QK. Invalid V
            # must be zero before PV: zero probability does not sanitize NaN.
            kt = tl.trans(kval)
            if tlx.num_warps() != 1:
                vval = vr.reshape((PAGE // 16, D, 16)).trans(0, 2, 1).reshape((PAGE, D))
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
            if PREFILL_PAIR and start == lo and start + BN < hi:
                tlx.async_amd_descriptor_wait(2)
            elif not SINGLE_TILE and start + BN < hi:
                tlx.async_amd_descriptor_wait(1)
            else:
                tlx.async_amd_descriptor_wait(0)
            vr = tlx.local_load(tlx.local_view(vbuf, buffer_slot), layout=V_LOAD_LAYOUT, relaxed=tlx.num_warps() == 1)
            if (not PREFILL_PAIR or start != lo) and not SINGLE_TILE and start + BN < hi:
                _issue_flat_page(kd, vdsc, BT, kbuf, vbuf, seq, bst, kvh, start + BN, hi, 1 - buffer_slot, HQ // G,
                                 PAGE, 2)
            vval = vr.reshape((PAGE // 16, D, 16)).trans(0, 2, 1).reshape((PAGE, D))
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
    output_scale = 1.0 if SPLITS > 1 and PART.dtype.element_ty == tl.float16 else vs
    value = acc * (output_scale * numerator_scale) / tl.maximum(l[:, None], 1.0e-20)
    if SPLITS == 1:
        tl.store(OUT + (token[:, None] * HQ + head[:, None]) * D + d[None, :], value, row_valid[:, None])
    else:
        offset = ((token[:, None] * HQ + head[:, None]) * SPLITS + split) * D + d[None, :]
        tl.store(PART + offset, value, row_valid[:, None])
        tl.store(LSE + (token * HQ + head) * SPLITS + split, tl.where(l > 0, m + tl.log2(l), -float("inf")), row_valid)


@triton.jit
def _merge(PART, LSE, OUT, D: tl.constexpr, SPLITS: tl.constexpr, BS: tl.constexpr, VS, HQ: tl.constexpr,
           G: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, BS)
    d = tl.arange(0, D)
    lse = tl.load(LSE + row * SPLITS + s, s < SPLITS, other=-float("inf"))
    m = tl.max(lse, 0)
    m = tl.where(m == -float("inf"), 0.0, m)
    p = tl.exp2(lse - m)
    vals = tl.load(PART + (row * SPLITS + s[:, None]) * D + d[None, :], s[:, None] < SPLITS, other=0.0)
    value = tl.sum(vals * p[:, None], 0) / tl.maximum(tl.sum(p, 0), 1.0e-20)
    if PART.dtype.element_ty == tl.float16:
        value *= tl.load(VS + (row % HQ) // G)
    tl.store(OUT + row * D + d, value)


def decode(q, k, v, lengths, table, ks, vs, *, q_scale, query_length, causal, window_size, sm_scale, num_splits,
           variant, out=None, workspace=None, block_n=64, block_m=16, num_warps=4, prefill_pair=True):
    tokens, heads, dim = q.shape
    hkv, page = k.shape[1], k.shape[3]
    group = heads // hkv
    if out is None:
        out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    if out.shape != q.shape or out.dtype != torch.bfloat16 or out.device != q.device or not out.is_contiguous():
        raise ValueError("out must be contiguous BF16 with Q's shape/device")
    mid_shape, lse_shape = (tokens, heads, num_splits, dim), (tokens, heads, num_splits)
    if workspace is None:
        if num_splits > 1:
            mid = torch.empty(mid_shape, device=q.device, dtype=torch.float16 if num_splits >= 8 else torch.float32)
            lse = torch.empty(lse_shape, device=q.device, dtype=torch.float32)
        else:
            mid = lse = out  # constexpr dead pointers, never dereferenced
    else:
        mid, lse = workspace
        if (mid.shape != mid_shape or lse.shape != lse_shape or mid.dtype != torch.float32 or lse.dtype != torch.float32
                or mid.device != q.device or lse.device != q.device or not mid.is_contiguous()
                or not lse.is_contiguous()):
            raise ValueError("workspace must be contiguous FP32 partial/LSE tensors for this workload")
        if num_splits >= 8:
            mid = mid.view(torch.float16)
    strides = (*q.stride(), table.stride(0), *k.stride()[:4], *v.stride()[:4])
    if variant in ("vector", "vector_page"):
        kl, vl = _vector_layouts(block_n, dim, num_warps)
        _vector_decode[(tokens, heads, num_splits)](q, k, v, ks, vs, q_scale if q_scale is not None else ks, table,
                                                    lengths, mid, lse, out, *strides, sm_scale, dim, heads, group,
                                                    query_length, page, num_splits, block_n,
                                                    q.dtype == torch.float8_e4m3fn, causal, window_size, kl, vl,
                                                    variant == "vector_page", num_warps=num_warps)
    else:
        rows = triton.cdiv(query_length * group, block_m)
        fp8_qk = variant != "matrix_precise" and q.dtype == torch.float8_e4m3fn and dim == 64
        kload, vload = _packed_operand_layouts(
            page, dim, fp8_qk) if variant == "matrix_tdm_flat" and num_warps == 1 else (None, None)
        visible_bound = min(table.shape[1] * page, window_size + page - 1) if window_size else table.shape[1] * page
        single_tile = variant == "matrix_tdm_flat" and query_length == 1 and num_warps == 1 and visible_bound <= num_splits * block_n
        prefill_enabled = (prefill_pair and variant == "matrix_tdm_flat" and num_warps == 1 and not single_tile
                           and q.dtype == torch.bfloat16 and window_size == 0 and query_length == 1
                           and visible_bound >= 8192 and ((page == 64 and hkv <= 2) or heads == hkv and hkv >= 64))
        _matrix_decode[(tokens // query_length * rows, hkv,
                        num_splits)](q, k, v, ks, vs, q_scale if q_scale is not None else ks, table, lengths, mid, lse,
                                     out, *strides, sm_scale, dim, heads, group, query_length, page, num_splits,
                                     block_m, block_n, q.dtype == torch.float8_e4m3fn, causal, window_size, rows,
                                     fp8_qk, variant == "matrix_precise", variant == "matrix_tdm_flat", k.shape[0],
                                     kload, vload, single_tile, prefill_enabled, num_warps=num_warps,
                                     matrix_instr_nonkdim=16)
    if num_splits > 1:
        _merge[(tokens * heads, )](mid, lse, out, dim, num_splits, triton.next_power_of_2(num_splits), vs, heads, group,
                                   num_warps=(1 if num_splits * dim <= 8192 else 4 if num_splits * dim <= 32768 else 8))
    return out


def _select_decode_config(q, key_cache, *, max_context_len, query_length=1, window_size=0, num_splits=None):
    """Return the measured shape policy; unknown model metadata is never inferred.

    Pass this configuration explicitly or use variant="auto". Shapes outside
    the measured families use the same semantic kernel with a heuristic config.
    """
    tokens, heads, dim = q.shape
    kv_heads, page = key_cache.shape[1], key_cache.shape[3]
    effective = max(1, min(max_context_len, window_size or max_context_len))
    group = heads // kv_heads
    if query_length == 1 and effective <= 128:
        cfg = dict(variant="matrix_tdm_flat" if page in (64, 128) else "vector",
                   block_n=page if page in (64, 128) else min(128, max(32, triton.next_power_of_2(effective))),
                   block_m=16, num_warps=1 if (dim == 128 and effective > 64) or (dim == 64 and kv_heads >= 32) else 4,
                   num_splits=1)
    elif query_length == 1 and effective >= 65536 and kv_heads <= 8:
        cfg = dict(variant="matrix_tdm_flat" if page in (64, 128) else "matrix",
                   block_n=page if page in (64, 128) else 32, block_m=16, num_warps=1,
                   num_splits=min(512, max(128, triton.cdiv(768 if dim == 128 else 512, kv_heads))))
    elif page in (64, 128):
        bm = 16 if query_length == 1 else 64
        tasks = tokens // query_length * kv_heads * triton.cdiv(query_length * group, bm)
        if query_length == 1:
            if kv_heads >= 32:
                chunk = 1024 if page == 128 and effective >= 8192 else 3 * page if dim == 128 else 256
                split_cap = 128
            else:
                chunk = 2 * page if dim == 128 and page == 64 and effective >= 4096 else page
                split_cap = 256 if page == 128 and dim == 64 else 128
            splits = min(split_cap, triton.cdiv(effective, chunk))
        else:
            splits = min(triton.cdiv(effective, page), max(1, triton.cdiv(384, tasks)), 128)
        if query_length == 1 and q.dtype == torch.bfloat16 and effective > 128:
            if kv_heads >= 32 and effective >= 8192:
                # Long dense partitions amortize descriptor/setup traffic;
                # excessive splitting increases scheduling and merge work.
                # Maintain enough CTAs for smaller KV-head grids.
                if dim == 128:
                    splits = max(12, triton.cdiv(768, tasks))
                elif page == 64:
                    splits = min(32, max(triton.cdiv(1024, tasks), triton.cdiv(effective, 1024)))
                else:
                    splits = min(32, max(12, triton.cdiv(1024, tasks), triton.cdiv(effective, 4096)))
            elif dim == 64 and kv_heads < 32 and page == 64:
                # Sparse grouped grids need more independent page partitions.
                pages = triton.cdiv(effective, page)
                splits = min(256, triton.next_power_of_2(pages))
                if tokens > query_length:
                    # Independent requests already supply parallel work.
                    splits = min(splits, pages)
        cfg = dict(
            variant="matrix_tdm_flat", block_n=page, block_m=bm, num_warps=1 if query_length == 1 and
            (kv_heads >= 32 or dim == 128 or page == 128 or q.dtype == torch.bfloat16 and effective > 128)
            and not (dim == 128 and q.dtype == torch.float8_e4m3fn and kv_heads < 32 and page == 64) else 4,
            num_splits=splits)
    else:
        cfg = dict(variant="matrix", block_n=32, block_m=16, num_warps=4,
                   num_splits=min(128, triton.cdiv(effective, 128)))
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
