"""Preliminary gfx1250 TLX decode attention with E4M3 KV and per-head scales.

Physical caches follow amd_pa_decode.py, with 16-byte packing (X=16):
  K: [blocks, kv_heads, head_dim // X, page_size, X]
  V: [blocks, kv_heads, page_size // X, head_dim, X]
Q/O: [batch * query_length, query_heads, head_dim]; O is BF16.

Each request has query_length queries. Causal queries are aligned to the last
query_length positions of context_lens (which INCLUDES the new tokens). For
independent decode requests, query_length=1. Use causal=False for a dense
query sweep without this temporal interpretation. window_size counts the
current token and is supported with causal=True only.

K/V are plain E4M3FN, NOT MXFP8: k = k_fp8 * k_scale[kv_head], likewise V.
An FP8 query additionally uses q_scale[query_head]. Scales are FP32,
positive and finite. Inputs must be finite. No output projection is included.

The default auto path selects grouped WMMA candidates with packed TDM cache
loads on supported pages. The separate baseline variant uses TLX LDS staging
and FP32 vector QK/PV reductions to establish an independent implementation of
the cache/scaling contract. It handles one (query token, query head, KV split)
per CTA; a second kernel merges normalized partials using log-sum-exp values.

Run the embedded correctness suite (no performance benchmark):
  pytest -s --tb=short third_party/tlx/tutorials/amd_pa_decode_fp8.py
"""

import math
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _decode_fp8_partitions(
    Q,
    K,
    V,
    KScale,
    VScale,
    QScale,
    BlockTable,
    ContextLens,
    Partial,
    LSE,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_bt,
    stride_kb,
    stride_kh,
    stride_kg,
    stride_kp,
    stride_kx,
    stride_vb,
    stride_vh,
    stride_vp,
    stride_vd,
    stride_vx,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    QUERY_LENGTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    Q_FP8: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    seq = token // QUERY_LENGTH
    kv_head = head // GROUP_SIZE
    context = tl.load(ContextLens + seq)
    query_pos = context - QUERY_LENGTH + token % QUERY_LENGTH
    split_size = tl.cdiv(context, NUM_SPLITS * BLOCK_N) * BLOCK_N
    lo = split * split_size
    hi = tl.minimum(lo + split_size, context)
    if CAUSAL:
        hi = tl.minimum(hi, query_pos + 1)
    if WINDOW_SIZE > 0:
        lo = tl.maximum(lo, query_pos + 1 - WINDOW_SIZE)
    lo = tl.maximum(lo, 0)

    d = tl.arange(0, BLOCK_D)
    n = tl.arange(0, BLOCK_N)
    q = tl.load(Q + token * stride_qt + head * stride_qh + d * stride_qd, mask=d < HEAD_DIM, other=0.0).to(tl.float32)
    if Q_FP8:
        q *= tl.load(QScale + head)
    k_scale = tl.load(KScale + kv_head)
    v_scale = tl.load(VScale + kv_head)

    k_smem = tlx.local_alloc((BLOCK_N, BLOCK_D), K.dtype.element_ty, 1)
    v_smem = tlx.local_alloc((BLOCK_N, BLOCK_D), V.dtype.element_ty, 1)
    max_score = -float("inf")
    denominator = 0.0
    accumulator = tl.full((BLOCK_D, ), 0, tl.float32)
    for start in range(lo, hi, BLOCK_N):
        positions = start + n
        valid = positions < hi
        blocks = tl.load(BlockTable + seq * stride_bt + positions // PAGE_SIZE, mask=valid, other=0).to(tl.int64)
        slots = positions % PAGE_SIZE
        k_offsets = (blocks[:, None] * stride_kb + kv_head * stride_kh + (d[None, :] // 16) * stride_kg +
                     slots[:, None] * stride_kp + (d[None, :] % 16) * stride_kx)
        v_offsets = (blocks[:, None] * stride_vb + kv_head * stride_vh + (slots[:, None] // 16) * stride_vp +
                     d[None, :] * stride_vd + (slots[:, None] % 16) * stride_vx)
        mask = valid[:, None] & (d[None, :] < HEAD_DIM)
        k_copy = tlx.async_load(K + k_offsets, tlx.local_view(k_smem, 0), mask=mask)
        v_tile = tl.load(V + v_offsets, mask=mask, other=0.0)
        tlx.local_store(tlx.local_view(v_smem, 0), v_tile)
        tlx.async_load_commit_group([k_copy])
        ready = tlx.async_load_wait_group(0)
        k = tlx.local_load(tlx.local_view(k_smem, 0), token=ready).to(tl.float32)
        v = tlx.local_load(tlx.local_view(v_smem, 0)).to(tl.float32)

        scores = tl.sum(k * q[None, :], 1) * (sm_scale * k_scale * 1.4426950408889634)
        scores = tl.where(valid, scores, -float("inf"))
        new_max = tl.maximum(max_score, tl.max(scores, 0))
        alpha = tl.exp2(max_score - new_max)
        weights = tl.exp2(scores - new_max)
        denominator = denominator * alpha + tl.sum(weights, 0)
        accumulator = accumulator * alpha + tl.sum(weights[:, None] * v, 0)
        max_score = new_max

    partial_id = (token * NUM_Q_HEADS + head) * NUM_SPLITS + split
    normalized = accumulator * v_scale / tl.maximum(denominator, 1.0e-20)
    tl.store(Partial + partial_id * HEAD_DIM + d, normalized, mask=d < HEAD_DIM)
    lse = tl.where(denominator > 0, max_score + tl.log2(denominator), -float("inf"))
    tl.store(LSE + partial_id, lse)


@triton.jit
def _decode_fp8_merge(Partial, LSE, Out, HEAD_DIM: tl.constexpr, NUM_SPLITS: tl.constexpr, BLOCK_S: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, BLOCK_D)
    lse = tl.load(LSE + row * NUM_SPLITS + s, mask=s < NUM_SPLITS, other=-float("inf"))
    max_lse = tl.max(lse, 0)
    # All splits can be empty (zero context or a fully masked query).
    max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    weights = tl.exp2(lse - max_lse)
    partial = tl.load(Partial + (row * NUM_SPLITS + s[:, None]) * HEAD_DIM + d[None, :],
                      mask=(s[:, None] < NUM_SPLITS) & (d[None, :] < HEAD_DIM), other=0)
    result = tl.sum(partial * weights[:, None], 0) / tl.maximum(tl.sum(weights, 0), 1.0e-20)
    tl.store(Out + row * HEAD_DIM + d, result, mask=d < HEAD_DIM)


@lru_cache(maxsize=1)
def _optimized_module():
    path = Path(__file__).with_name("amd_pa_decode_fp8_optimized.py")
    spec = importlib.util.spec_from_file_location("tlx_fp8_decode_optimized", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_decode_config(q, key_cache, *, max_context_len, query_length=1, window_size=0, num_splits=None):
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
        cfg = dict(
            variant="matrix_tdm_flat", block_n=page, block_m=bm,
            num_warps=1 if query_length == 1 and (kv_heads >= 32 or dim == 128 or page == 128)
            and not (dim == 128 and q.dtype == torch.float8_e4m3fn and kv_heads < 32 and page == 64) else 4,
            num_splits=splits)
    else:
        cfg = dict(variant="matrix", block_n=32, block_m=16, num_warps=4,
                   num_splits=min(128, triton.cdiv(effective, 128)))
    if num_splits is not None:
        cfg["num_splits"] = num_splits
    return cfg


def fp8_kv_decode(q, key_cache, value_cache, context_lens, block_tables, k_scale, v_scale, *, max_context_len,
                  q_scale=None, query_length=1, causal=True, window_size=0, sm_scale=None, num_splits=None, block_n=32,
                  variant="auto", block_m=16, num_warps=4, out=None, workspace=None):
    """Return BF16 attention output from packed FP8 caches and per-head scales.

    context_lens must lie in [0, max_context_len], and each referenced block
    table entry must index an allocated cache block. These device-side value
    invariants are the caller's responsibility; validation does not synchronize
    the host. A zero context or fully masked query produces zero output.
    """
    if query_length < 1 or q.ndim != 3 or q.shape[0] % query_length:
        raise ValueError("Q must be [batch * query_length, query_heads, head_dim]")
    tokens, hq, dim = q.shape
    if not tokens or not hq or dim not in (64, 128):
        raise ValueError("nonempty Q with head_dim 64 or 128 is required")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("Q must be BF16 or E4M3FN")
    if key_cache.ndim != 5 or value_cache.ndim != 5:
        raise ValueError("K/V must use the tutorial's packed 5D layout")
    blocks, hkv, groups, page, x = key_cache.shape
    if not blocks or not hkv or hq % hkv or x != 16 or groups * x != dim or page < 16 or page % 16:
        raise ValueError("invalid K layout, head grouping, or page size (must be divisible by 16)")
    if value_cache.shape != (blocks, hkv, page // x, dim, x):
        raise ValueError("V must be [blocks, kv_heads, page_size/16, head_dim, 16]")
    if key_cache.dtype != torch.float8_e4m3fn or value_cache.dtype != torch.float8_e4m3fn:
        raise TypeError("K/V must be plain E4M3FN, not MXFP8 or FNUZ")
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        raise ValueError("packed K/V caches must be contiguous")
    batch = tokens // query_length
    if context_lens.shape != (batch, ) or context_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("context_lens must be an integer tensor [batch]")
    if (block_tables.ndim != 2 or block_tables.shape[0] != batch or block_tables.dtype not in (torch.int32, torch.int64)
            or block_tables.stride(1) != 1):
        raise ValueError("block_tables must be integer [batch, max_pages] with contiguous rows")
    if max_context_len < 1 or block_tables.shape[1] < triton.cdiv(max_context_len, page):
        raise ValueError("block table capacity must cover positive max_context_len")
    if window_size < 0 or (window_size and not causal):
        raise ValueError("window_size must be nonnegative; sliding windows require causal=True")
    if variant == "auto":
        cfg = select_decode_config(q, key_cache, max_context_len=max_context_len, query_length=query_length,
                                   window_size=window_size, num_splits=num_splits)
        variant, block_n, block_m, num_warps, num_splits = (cfg[name] for name in ("variant", "block_n", "block_m",
                                                                                   "num_warps", "num_splits"))
    if block_n not in (16, 32, 64, 128):
        raise ValueError("block_n must be 16, 32, 64, or 128")
    if variant not in ("baseline", "vector", "vector_page", "matrix", "matrix_fp8", "matrix_precise",
                       "matrix_tdm_flat"):
        raise ValueError("unknown decode variant")
    if variant != "baseline" and (block_n < 32 or num_warps not in (1, 2, 4) or block_m not in (16, 32, 64)):
        raise ValueError("candidates require block_n >= 32, num_warps 1/2/4 and block_m 16/32/64")
    if variant == "vector_page" and (page not in (64, 128) or block_n != page):
        raise ValueError("vector_page requires page=block_n=64 or 128")
    if variant == "matrix_tdm_flat" and (page not in (64, 128) or block_n != page):
        raise ValueError("matrix_tdm_flat requires page=block_n=64 or 128")
    if q.dtype == torch.float8_e4m3fn and q_scale is None:
        raise ValueError("FP8 Q requires per-query-head q_scale")
    if q.dtype == torch.bfloat16 and q_scale is not None:
        raise ValueError("q_scale applies only to FP8 Q")
    for name, scale, heads in (("k_scale", k_scale, hkv), ("v_scale", v_scale, hkv), ("q_scale", q_scale, hq)):
        if scale is None and name == "q_scale":
            continue
        if not isinstance(scale, torch.Tensor) or scale.shape != (heads, ) or scale.dtype != torch.float32:
            raise ValueError(f"{name} must be FP32 [{heads}]")
        if not scale.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    tensors = [key_cache, value_cache, context_lens, block_tables, k_scale, v_scale]
    if q_scale is not None:
        tensors.append(q_scale)
    if q.device.type != "cuda" or any(t.device != q.device for t in tensors):
        raise ValueError("all inputs must be on the same HIP device")
    if not torch.version.hip or torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx1250":
        raise ValueError("this preliminary kernel targets gfx1250")
    sm_scale = dim**-0.5 if sm_scale is None else float(sm_scale)
    if not math.isfinite(sm_scale):
        raise ValueError("sm_scale must be finite")
    num_splits = min(32, triton.cdiv(max_context_len, 2048)) if num_splits is None else num_splits
    if not isinstance(num_splits, int) or not 1 <= num_splits <= 1024:
        raise ValueError("num_splits must be an integer in [1, 1024]")
    if variant != "baseline":
        return _optimized_module().decode(q, key_cache, value_cache, context_lens, block_tables, k_scale, v_scale,
                                          q_scale=q_scale, query_length=query_length, causal=causal,
                                          window_size=window_size, sm_scale=sm_scale, num_splits=num_splits,
                                          variant=variant, out=out, workspace=workspace, block_n=block_n,
                                          block_m=block_m, num_warps=num_warps)
    if out is not None or workspace is not None:
        raise ValueError("preallocated out/workspace are supported by candidate variants")
    partial = torch.empty((tokens, hq, num_splits, dim), dtype=torch.float32, device=q.device)
    lse = torch.empty((tokens, hq, num_splits), dtype=torch.float32, device=q.device)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    _decode_fp8_partitions[(tokens, hq, num_splits)](
        q,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        q_scale if q_scale is not None else k_scale,
        block_tables,
        context_lens,
        partial,
        lse,
        *q.stride(),
        block_tables.stride(0),
        *key_cache.stride(),
        *value_cache.stride(),
        sm_scale,
        dim,
        hq,
        hq // hkv,
        query_length,
        page,
        num_splits,
        block_n,
        triton.next_power_of_2(dim),
        q.dtype == torch.float8_e4m3fn,
        causal,
        window_size,
        num_warps=4,
    )
    _decode_fp8_merge[(tokens * hq, )](
        partial,
        lse,
        out,
        dim,
        num_splits,
        triton.next_power_of_2(num_splits),
        triton.next_power_of_2(dim),
        num_warps=4,
    )
    return out


def _gfx1250_available():
    return bool(torch.cuda.is_available() and torch.version.hip
                and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1250")


def _make_test_inputs(lengths, hq=16, hkv=2, dim=64, page=64, query_length=1, fp8_q=False, seed=182):
    """Quantize with distinct per-head scales, shuffle pages, poison unused slots."""
    torch.manual_seed(seed)
    batch = len(lengths)
    max_len = max(1, max(lengths))
    pages = triton.cdiv(max_len, page)
    blocks = batch * pages + 3
    ks = torch.linspace(0.025, 0.08, hkv, device="cuda")
    vs = torch.linspace(0.04, 0.12, hkv, device="cuda").flip(0).contiguous()
    qscale = torch.linspace(0.02, 0.06, hq, device="cuda") if fp8_q else None
    q = torch.randn(batch * query_length, hq, dim, device="cuda") * 0.4
    q = (q / qscale[None, :, None]).to(torch.float8_e4m3fn) if fp8_q else q.to(torch.bfloat16)
    k = torch.randn(blocks, hkv, page, dim, device="cuda") * 0.5
    v = torch.randn_like(k) * 0.8
    table = torch.randperm(blocks, device="cuda")[:batch * pages].reshape(batch, pages).to(torch.int32)
    offsets = torch.arange(pages * page, device="cuda").reshape(1, pages, page)
    invalid = offsets >= torch.tensor(lengths, device="cuda")[:, None, None]
    invalid = invalid.reshape(batch * pages, 1, page, 1)
    physical_pages = table.long().flatten()
    k[physical_pages] = k[physical_pages].masked_fill(invalid, float("nan"))
    v[physical_pages] = v[physical_pages].masked_fill(invalid, float("nan"))
    k = (k / ks[None, :, None, None]).to(torch.float8_e4m3fn)
    v = (v / vs[None, :, None, None]).to(torch.float8_e4m3fn)
    # Pack independently of the kernel address calculation, following the tutorial.
    kp = k.reshape(blocks, hkv, page, dim // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
    vp = v.reshape(blocks, hkv, page // 16, 16, dim).permute(0, 1, 2, 4, 3).contiguous()
    contexts = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return q, kp, vp, contexts, table, ks, vs, qscale, max_len, k, v


def _reference(q, k, v, table, lengths, ks, vs, qs, query_length, causal, window_size):
    """Independent dense FP32 attention over the actual quantized inputs."""
    q = q.float() if qs is None else q.float() * qs[None, :, None]
    hq, dim = q.shape[1:]
    hkv, page = k.shape[1:3]
    out = torch.zeros_like(q)
    for seq, length in enumerate(lengths):
        if not length:
            continue
        blocks = table[seq, :triton.cdiv(length, page)].long()
        kd = k[blocks].float().permute(1, 0, 2, 3).reshape(hkv, -1, dim)[:, :length]
        vd = v[blocks].float().permute(1, 0, 2, 3).reshape(hkv, -1, dim)[:, :length]
        kd *= ks[:, None, None]
        vd *= vs[:, None, None]
        mapping = torch.arange(hq, device=q.device) // (hq // hkv)
        kd, vd = kd[mapping], vd[mapping]
        qr = q[seq * query_length:(seq + 1) * query_length].transpose(0, 1)
        scores = torch.matmul(qr, kd.transpose(1, 2)) * dim**-0.5
        positions = torch.arange(length, device=q.device)
        qpos = length - query_length + torch.arange(query_length, device=q.device)
        mask = torch.ones(query_length, length, device=q.device, dtype=torch.bool)
        if causal:
            mask &= positions[None, :] <= qpos[:, None]
        if window_size:
            mask &= positions[None, :] > qpos[:, None] - window_size
        probabilities = torch.softmax(scores.masked_fill(~mask[None], -float("inf")), -1).nan_to_num()
        out[seq * query_length:(seq + 1) * query_length] = torch.matmul(probabilities, vd).transpose(0, 1)
    return out


@pytest.mark.skipif(not _gfx1250_available(), reason="Requires gfx1250")
@pytest.mark.parametrize(
    "variant",
    ["auto", "baseline", "vector", "vector_page", "matrix", "matrix_precise", "matrix_fp8", "matrix_tdm_flat"])
@pytest.mark.parametrize("fp8_q", [False, True], ids=["bf16-q", "fp8-q"])
@pytest.mark.parametrize("dim,page,hq,hkv,qlen,lengths,causal,window,splits", [
    (64, 64, 16, 2, 1, [2304], True, 0, 1),
    (64, 64, 16, 2, 1, [2304], True, 0, 4),
    (64, 64, 16, 2, 1, [67, 8193], True, 0, 384),
    (64, 64, 128, 128, 1, [2304], True, 0, 4),
    (128, 32, 4, 4, 1, [0, 1, 31, 97], True, 0, 8),
    (128, 128, 8, 2, 1, [0, 67, 8193], True, 0, 384),
    (128, 64, 64, 64, 1, [0, 67, 2304], True, 0, 12),
    (128, 128, 64, 64, 1, [0, 67, 2304], True, 0, 6),
    (128, 64, 64, 64, 1, [0, 67, 257], True, 128, 1),
    (64, 128, 16, 1, 1, [257, 2303], True, 127, 4),
    (64, 64, 6, 2, 8, [1, 113], True, 37, 4),
    (64, 16, 4, 2, 32, [17], False, 0, 3),
    (64, 64, 4, 2, 1, [8192, 32768], True, 0, 8),
    (64, 64, 2, 1, 1, [131072, 262144], True, 0, 32),
])
def test_fp8_kv_decode(variant, fp8_q, dim, page, hq, hkv, qlen, lengths, causal, window, splits):
    if variant in ("vector_page", "matrix_tdm_flat") and page not in (64, 128):
        pytest.skip("packed-page candidate requires page64/128")
    q, kp, vp, contexts, table, ks, vs, qs, max_len, k, v = _make_test_inputs(lengths, hq, hkv, dim, page, qlen, fp8_q)
    actual = fp8_kv_decode(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len, q_scale=qs, query_length=qlen,
                           causal=causal, window_size=window, num_splits=splits, variant=variant,
                           block_n=page if variant in ("vector_page", "matrix_tdm_flat") else 32)
    expected = _reference(q, k, v, table, lengths, ks, vs, qs, qlen, causal, window)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=5e-4, rtol=1e-2)


@pytest.mark.skipif(not _gfx1250_available(), reason="Requires gfx1250")
def test_fp8_kv_decode_validation():
    q, kp, vp, contexts, table, ks, vs, qs, max_len, _, _ = _make_test_inputs([33], fp8_q=True)
    with pytest.raises(ValueError, match="q_scale"):
        fp8_kv_decode(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len)
    with pytest.raises(ValueError, match="k_scale"):
        fp8_kv_decode(q, kp, vp, contexts, table, ks[:1], vs, max_context_len=max_len, q_scale=qs)
    with pytest.raises(ValueError, match="sliding windows"):
        fp8_kv_decode(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len, q_scale=qs, causal=False,
                      window_size=16)


@pytest.mark.skipif(not _gfx1250_available(), reason="Requires gfx1250")
@pytest.mark.parametrize("variant", ["auto", "baseline", "matrix", "matrix_tdm_flat"])
@pytest.mark.parametrize("tokens", [8, 32, 128, 512, 2048, 8192])
def test_fp8_kv_decode_query_sweep(tokens, variant):
    # Exercise T independently from L: a dense query set over one shared cache.
    # These are deliberately noncausal, since T can exceed the cache length.
    lengths = [2304]
    q, kp, vp, contexts, table, ks, vs, _, max_len, k, v = _make_test_inputs(lengths, hq=16, hkv=2, query_length=tokens)
    actual = fp8_kv_decode(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len, query_length=tokens,
                           causal=False, num_splits=1, variant=variant, block_n=64)
    # Chunk the independent reference to bound memory at the largest T.
    for first in range(0, tokens, 128):
        qr = q[first:first + 128]
        expected = _reference(qr, k, v, table, lengths, ks, vs, None, qr.shape[0], False, 0)
        torch.testing.assert_close(actual[first:first + 128].float(), expected, atol=5e-4, rtol=1e-2)


@pytest.mark.skipif(not _gfx1250_available(), reason="Requires gfx1250")
@pytest.mark.parametrize("block_n", [16, 64])
def test_fp8_kv_decode_tile_sizes(block_n):
    lengths = [67, 129]
    q, kp, vp, contexts, table, ks, vs, qs, max_len, k, v = _make_test_inputs(lengths, fp8_q=True)
    actual = fp8_kv_decode(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len, q_scale=qs, num_splits=3,
                           block_n=block_n, window_size=37, variant="baseline")
    expected = _reference(q, k, v, table, lengths, ks, vs, qs, 1, True, 37)
    torch.testing.assert_close(actual.float(), expected, atol=5e-4, rtol=1e-2)


@pytest.mark.skipif(not _gfx1250_available(), reason="Requires gfx1250")
@pytest.mark.parametrize("variant", ["vector_page", "matrix", "matrix_precise", "matrix_fp8", "matrix_tdm_flat"])
@pytest.mark.parametrize("fp8_q", [False, True])
@pytest.mark.parametrize("splits", [4, 8, 12, 384])
@pytest.mark.parametrize("seed", [0, 19, 182])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_fp8_kv_decode_peaked_and_workspace(variant, fp8_q, splits, seed, dim, num_warps):
    q, kp, vp, contexts, table, ks, vs, qs, max_len, k, v = _make_test_inputs([0, 67, 257], hq=16, hkv=2, dim=dim,
                                                                              query_length=8, fp8_q=fp8_q, seed=seed)
    ks = ks * torch.tensor([1.0, 64.0], device=q.device)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    workspace = (torch.empty((*q.shape[:2], splits, q.shape[2]),
                             device=q.device), torch.empty((*q.shape[:2], splits), device=q.device))
    actual = fp8_kv_decode(q, kp, vp, contexts, table, ks, vs, max_context_len=max_len, q_scale=qs, query_length=8,
                           window_size=37, num_splits=splits, variant=variant, block_n=64, num_warps=num_warps, out=out,
                           workspace=workspace)
    expected = _reference(q, k, v, table, [0, 67, 257], ks, vs, qs, 8, True, 37)
    assert actual.data_ptr() == out.data_ptr()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=5e-4, rtol=1e-2)


@pytest.mark.skipif(not _gfx1250_available(), reason="Requires gfx1250")
@pytest.mark.parametrize("variant,num_warps", [("auto", 4), ("matrix_tdm_flat", 1)])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("fp8_q", [False, True])
def test_fp8_kv_decode_tiny_probability_mass(variant, num_warps, dim, page, fp8_q):
    # Individually tiny probabilities can have a visible aggregate after V
    # scaling. A fast path must not discard them by quantizing P too early.
    q, _, _, lengths, table, _, _, _, max_len, k, _ = _make_test_inputs([128], hq=16, hkv=2, dim=dim, page=page,
                                                                        fp8_q=fp8_q)
    qf = torch.zeros(q.shape, device=q.device)
    qf[..., 0] = 1.0
    q = qf.to(q.dtype)
    kf = torch.zeros(k.shape, device=q.device)
    kf[..., 0] = -144.0 if dim == 64 else -192.0
    vf = torch.full(k.shape, 448.0, device=q.device)
    first_page = table[0, 0].long()
    kf[first_page, :, 0, 0] = 0.0
    vf[first_page, :, 0, :] = 0.0
    k = kf.to(torch.float8_e4m3fn)
    v = vf.to(torch.float8_e4m3fn)
    blocks = k.shape[0]
    kp = k.reshape(blocks, 2, page, dim // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
    vp = v.reshape(blocks, 2, page // 16, 16, dim).permute(0, 1, 2, 4, 3).contiguous()
    ks = torch.tensor([1.0, 1.125], device=q.device)
    vs = torch.tensor([1.0, 8.0], device=q.device)
    qs = torch.ones(16, device=q.device) if fp8_q else None
    actual = fp8_kv_decode(q, kp, vp, lengths, table, ks, vs, max_context_len=max_len, q_scale=qs, variant=variant,
                           num_warps=num_warps, block_n=page, num_splits=1)
    expected = _reference(q, k, v, table, [128], ks, vs, qs, 1, True, 0)
    assert float(expected.abs().max()) > 7e-4
    torch.testing.assert_close(actual.float(), expected, atol=5e-4, rtol=1e-2)
