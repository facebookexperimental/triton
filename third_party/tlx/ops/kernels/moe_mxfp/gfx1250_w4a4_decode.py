"""Small-batch MXFP4 W4A4 routed MoE for gfx1250 (decode): two launches per call, no routing kernels, no per-call
allocations.

Numerics: each token row is quantized to MXFP4 (one E8M0 scale per 32 along K, AITER / Quark "even" rounding: scale
2^(floor(log2(amax rounded to 2 mantissa bits)) - 2), E2M1 round to nearest even, saturating at 6), gate and up are
rounded to BF16, SiLU(gate) * up is requantized to MXFP4, the down projection is rounded to BF16, and the finalize
applies the optional post-norm (``(y * (w + gain)) -> BF16``, then RMS) and the routing-weighted sum in FP32.

At decode sizes every routed slot is one row of one expert, so each GEMM is a matrix-vector product whose cost is
reading the expert's weights. Launch 1 runs one program per (slot, BLOCK_N gate rows + the matching up rows): it reads
the expert id from ``topk_ids`` (slots outside the local window exit), quantizes the token row in registers and feeds
the packed operands to ``tl.dot_scaled(e2m1, e2m1)`` (``v_wmma_scale_f32_32x16x128_f4``) with the row broadcast to the
WMMA height, then stores the requantized SiLU(gate) * up tile. Launch 2 runs one program per (slot, BLOCK_H rows of the
down projection) on the packed tile; the last program of a token to finish (an atomic ticket, no waiting) applies the
finalize and resets the ticket. Tokens with no local slot get zeros.

The workspace (requantized activations, expert outputs, tickets) is cached per device and shape, so calls on one device
must be ordered on one stream. Weights use the checkpoint's natural (Quark) layout: ``w13`` uint8 ``[E, 2I, K/2]`` with
``w13_scale`` uint8 ``[E, 2I, K/32]`` (gate rows first), ``w2`` uint8 ``[E, H, I/2]`` with ``w2_scale`` uint8
``[E, H, I/32]``; packed bytes hold the even element in the low nibble.
"""

import torch
import triton
import triton.language as tl

from triton.tlx.ops.kernels.moe_mxfp.gfx1250 import _check_post_norm

_BLOCK_N = 32
_BLOCK_H = 32
_BLOCK_K = 512
_WMMA_M = 32
_NEG_LOG2E = -1.4426950408889634


@triton.jit
def _e2m1_rne(a):
    """E2M1 magnitude code (0-7) of a non-negative scaled value: round to nearest even, saturating at 6."""
    c = tl.where(a > 0.25, 1, 0)
    c = tl.where(a >= 0.75, 2, c)
    c = tl.where(a > 1.25, 3, c)
    c = tl.where(a >= 1.75, 4, c)
    c = tl.where(a > 2.5, 5, c)
    c = tl.where(a >= 3.5, 6, c)
    return tl.where(a > 5.0, 7, c)


@triton.jit
def _quant(v, N: tl.constexpr):
    """MXFP4 of fp32 v [N]: packed uint8 [N / 2] (even element in the low nibble) and E8M0 uint8 [N / 32]."""
    vb = tl.reshape(v, (N // 32, 32))
    amax = tl.max(tl.abs(vb), axis=1)
    e = ((amax.to(tl.int32, bitcast=True) + 0x200000) & 0x7F800000) >> 23
    unb = tl.minimum(tl.maximum(e - 129, -127), 127)
    q = vb * tl.exp2((-unb).to(tl.float32))[:, None]
    code = _e2m1_rne(tl.abs(q)) | tl.where(q < 0, 8, 0)
    lo, hi = tl.split(tl.reshape(code, (N // 2, 2)))
    return (lo | (hi << 4)).to(tl.uint8), (unb + 127).to(tl.uint8)


@triton.jit
def _row0(acc, M: tl.constexpr):
    """Row 0 of acc [M, N]: every row of the broadcast A operand gives the same products."""
    return tl.sum(tl.where((tl.arange(0, M) == 0)[:, None], acc, 0.0), axis=0)


@triton.jit
def _local(ids_ptr, slot, expert_start, E):
    e = tl.load(ids_ptr + slot).to(tl.int32) - expert_start
    return e, (e >= 0) & (e < E)


@triton.jit
def _gate_up_kernel(x_ptr, stride_xm, ids_ptr, w_ptr, ws_ptr, hq_ptr, hs_ptr, expert_start, E, stride_we, stride_wse,
                    neg_log2e, K: tl.constexpr, I: tl.constexpr, TOP_K: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr, WMMA_M: tl.constexpr):
    slot = tl.program_id(0)
    e, ok = _local(ids_ptr, slot, expert_start, E)
    if ok:
        tok = (slot // TOP_K).to(tl.int64)
        n0 = tl.program_id(1) * BLOCK_N
        rows = n0 + tl.arange(0, BLOCK_N)
        wp = w_ptr + e.to(tl.int64) * stride_we
        sp = ws_ptr + e.to(tl.int64) * stride_wse
        acc_g = tl.zeros((WMMA_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((WMMA_M, BLOCK_N), dtype=tl.float32)
        for k0 in tl.range(0, K, BLOCK_K, num_stages=1):
            xq, xs = _quant(tl.load(x_ptr + tok * stride_xm + k0 + tl.arange(0, BLOCK_K)).to(tl.float32), BLOCK_K)
            a = tl.broadcast_to(xq[None, :], (WMMA_M, BLOCK_K // 2))
            sa = tl.broadcast_to(xs[None, :], (WMMA_M, BLOCK_K // 32))
            kb = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            ks = k0 // 32 + tl.arange(0, BLOCK_K // 32)
            bg = tl.load(wp + rows[:, None] * (K // 2) + kb[None, :])
            sg = tl.load(sp + rows[:, None] * (K // 32) + ks[None, :])
            bu = tl.load(wp + (rows + I)[:, None] * (K // 2) + kb[None, :])
            su = tl.load(sp + (rows + I)[:, None] * (K // 32) + ks[None, :])
            acc_g = tl.dot_scaled(a, sa, "e2m1", tl.trans(bg), sg, "e2m1", acc_g)
            acc_u = tl.dot_scaled(a, sa, "e2m1", tl.trans(bu), su, "e2m1", acc_u)
        g = _row0(acc_g, WMMA_M).to(tl.bfloat16).to(tl.float32)
        u = _row0(acc_u, WMMA_M).to(tl.bfloat16).to(tl.float32)
        hq, hs = _quant(g / (1.0 + tl.exp2(g * neg_log2e)) * u, BLOCK_N)
        tl.store(hq_ptr + slot.to(tl.int64) * (I // 2) + n0 // 2 + tl.arange(0, BLOCK_N // 2), hq)
        tl.store(hs_ptr + slot.to(tl.int64) * (I // 32) + n0 // 32 + tl.arange(0, BLOCK_N // 32), hs)


@triton.jit
def _finalize_token(y_ptr, scores_ptr, norm_ptr, out_ptr, ids_ptr, tok, expert_start, E, eps, gain, H: tl.constexpr,
                    TOP_K: tl.constexpr, HAS_NORM: tl.constexpr):
    offs = tl.arange(0, H)
    if HAS_NORM:
        w = tl.load(norm_ptr + offs)
        scale = (w.to(tl.float32) + gain).to(norm_ptr.dtype.element_ty).to(tl.float32)
    out = tl.zeros((H, ), dtype=tl.float32)
    for j in tl.static_range(TOP_K):
        _, okj = _local(ids_ptr, tok * TOP_K + j, expert_start, E)
        if okj:
            sc = tl.load(scores_ptr + tok * TOP_K + j).to(tl.float32)
            y = tl.load(y_ptr + (tok * TOP_K + j).to(tl.int64) * H + offs, volatile=True).to(tl.float32)
            if HAS_NORM:
                y = (y * scale).to(tl.bfloat16).to(tl.float32)
                out += y * (sc / tl.sqrt(tl.sum(y * y, axis=0) / H + eps))
            else:
                out += y * sc
    tl.store(out_ptr + tok.to(tl.int64) * H + offs, out.to(out_ptr.dtype.element_ty))


@triton.jit
def _down_finalize_kernel(hq_ptr, hs_ptr, ids_ptr, w_ptr, ws_ptr, y_ptr, scores_ptr, norm_ptr, out_ptr, cnt_ptr,
                          expert_start, E, stride_we, stride_wse, eps, gain, I: tl.constexpr, H: tl.constexpr,
                          TOP_K: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr, WMMA_M: tl.constexpr,
                          HAS_NORM: tl.constexpr):
    slot = tl.program_id(0)
    ht = tl.program_id(1)
    tok = slot // TOP_K
    e, ok = _local(ids_ptr, slot, expert_start, E)
    if ok:
        rows = ht * BLOCK_H + tl.arange(0, BLOCK_H)
        wp = w_ptr + e.to(tl.int64) * stride_we
        sp = ws_ptr + e.to(tl.int64) * stride_wse
        acc = tl.zeros((WMMA_M, BLOCK_H), dtype=tl.float32)
        for k0 in tl.range(0, I, BLOCK_K, num_stages=1):
            kb = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            ks = k0 // 32 + tl.arange(0, BLOCK_K // 32)
            a = tl.broadcast_to(tl.load(hq_ptr + slot.to(tl.int64) * (I // 2) + kb)[None, :], (WMMA_M, BLOCK_K // 2))
            sa = tl.broadcast_to(tl.load(hs_ptr + slot.to(tl.int64) * (I // 32) + ks)[None, :], (WMMA_M, BLOCK_K // 32))
            b = tl.load(wp + rows[:, None] * (I // 2) + kb[None, :])
            sb = tl.load(sp + rows[:, None] * (I // 32) + ks[None, :])
            acc = tl.dot_scaled(a, sa, "e2m1", tl.trans(b), sb, "e2m1", acc)
        tl.store(y_ptr + slot.to(tl.int64) * H + rows, _row0(acc, WMMA_M).to(tl.bfloat16))
    n_local = 0
    for j in tl.static_range(TOP_K):
        _, okj = _local(ids_ptr, tok * TOP_K + j, expert_start, E)
        n_local += okj.to(tl.int32)
    if n_local == 0:
        if (slot % TOP_K == 0) & (ht == 0):
            tl.store(out_ptr + tok.to(tl.int64) * H + tl.arange(0, H), tl.zeros((H, ), dtype=out_ptr.dtype.element_ty))
    elif ok:
        # One thread does the atomic, and the barrier Triton puts before it orders only LDS: this barrier makes every
        # wave wait for its y stores first. The release then publishes the whole tile; acquire makes the other tiles
        # visible to the last arriver.
        tl.debug_barrier()
        if tl.atomic_add(cnt_ptr + tok, 1, sem="acq_rel", scope="gpu") == n_local * (H // BLOCK_H) - 1:
            _finalize_token(y_ptr, scores_ptr, norm_ptr, out_ptr, ids_ptr, tok, expert_start, E, eps, gain, H, TOP_K,
                            HAS_NORM)
            tl.atomic_xchg(cnt_ptr + tok, 0, sem="relaxed", scope="gpu")


def _alloc(shape, dtype, device):
    """Workspace allocation (tests replace it to place guard regions around the op's buffers)."""
    return torch.empty(shape, dtype=dtype, device=device)


_WORKSPACE = {}


def _workspace(device, T, topk, I, H):
    key = (device, T, topk, I, H)
    ws = _WORKSPACE.get(key)
    if ws is None:
        slots = T * topk
        ws = (_alloc((slots, I // 2), torch.uint8, device), _alloc(
            (slots, I // 32), torch.uint8, device), _alloc((slots, H), torch.bfloat16,
                                                           device), torch.zeros(T, dtype=torch.int32, device=device))
        _WORKSPACE[key] = ws
    return ws


def _check(x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale, post_norm_weight, out):
    T, K = x.shape
    E, I2, K2 = w13.shape
    I, H = I2 // 2, w2.shape[1]
    if x.dtype != torch.bfloat16 or x.stride(1) != 1:
        raise ValueError("moe_mxfp4_w4a4_decode: x must be BF16 with contiguous rows")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != T:
        raise ValueError("moe_mxfp4_w4a4_decode: topk_ids and topk_weights must both be [T, topk]")
    if not (topk_ids.is_contiguous() and topk_weights.is_contiguous()):
        raise ValueError("moe_mxfp4_w4a4_decode: topk_ids and topk_weights must be contiguous")
    if K2 * 2 != K or w2.shape != (E, H, I // 2) or w13_scale.shape != (E, I2, K // 32) or w2_scale.shape != (E, H,
                                                                                                              I // 32):
        raise ValueError("moe_mxfp4_w4a4_decode: weight shapes do not match x [T, K], w13 [E, 2I, K/2], w2 [E, H, I/2]")
    for name, w in (("w13", w13), ("w13_scale", w13_scale), ("w2", w2), ("w2_scale", w2_scale)):
        if w.dtype != torch.uint8 or w.stride(2) != 1 or w.stride(1) != w.shape[2]:
            raise ValueError(f"moe_mxfp4_w4a4_decode: {name} must be uint8 with contiguous rows")
    if K % _BLOCK_K or I % _BLOCK_K or I % _BLOCK_N or H % _BLOCK_H or H & (H - 1):
        raise ValueError(
            f"moe_mxfp4_w4a4_decode: K={K} and I={I} must be multiples of {_BLOCK_K}, H={H} a power of two "
            f">= {_BLOCK_H}")
    _check_post_norm("moe_mxfp4_w4a4_decode", post_norm_weight, H, x.device)
    if post_norm_weight is not None and not post_norm_weight.is_contiguous():
        raise ValueError(f"moe_mxfp4_w4a4_decode: post_norm_weight must be contiguous [{H}]")
    if out is not None and (out.shape != (T, H) or out.dtype != torch.bfloat16 or not out.is_contiguous()
                            or out.device != x.device):
        raise ValueError(f"moe_mxfp4_w4a4_decode: out must be a contiguous bf16 [{T}, {H}] tensor on {x.device}")
    return T, K, E, I, H


def moe_mxfp4_w4a4_decode(x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale, expert_start=0,
                          post_norm_weight=None, post_norm_eps=1e-6, post_norm_gain=0.0, out=None):
    """BF16 x [T, K] through the local experts [expert_start, expert_start + E) for decode-size T; returns BF16
    out [T, H] = sum over local slots of topk_weights * post_norm(expert(x)). ``topk_ids`` / ``topk_weights`` are
    [T, topk] (global expert ids, routing weights)."""
    T, K, E, I, H = _check(x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale, post_norm_weight, out)
    topk = topk_ids.shape[1]
    hq, hs, y, cnt = _workspace(x.device, T, topk, I, H)
    if out is None:
        out = torch.empty((T, H), dtype=torch.bfloat16, device=x.device)
    has_norm = post_norm_weight is not None
    slots = T * topk
    _gate_up_kernel[(slots, I // _BLOCK_N)](x, x.stride(0), topk_ids, w13, w13_scale, hq, hs, expert_start, E,
                                            w13.stride(0), w13_scale.stride(0), _NEG_LOG2E, K=K, I=I, TOP_K=topk,
                                            BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K, WMMA_M=_WMMA_M, num_warps=4)
    _down_finalize_kernel[(slots, H // _BLOCK_H)](hq, hs, topk_ids, w2, w2_scale, y, topk_weights,
                                                  post_norm_weight if has_norm else out, out, cnt, expert_start, E,
                                                  w2.stride(0), w2_scale.stride(0), float(post_norm_eps),
                                                  float(post_norm_gain), I=I, H=H, TOP_K=topk, BLOCK_H=_BLOCK_H,
                                                  BLOCK_K=_BLOCK_K, WMMA_M=_WMMA_M, HAS_NORM=has_norm, num_warps=4)
    return out
