"""Experimental bounded-window TLX vector attention; independent oracle required."""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def short_decode(Q, K, V, KS, VS, QS, BT, LEN, O, qst, qsh, qsd, bst, kb, kh, kg, kp, vb, vh, vp, vd, scale,
                 D: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, QLEN: tl.constexpr, PAGE: tl.constexpr,
                 BN: tl.constexpr, Q_FP8: tl.constexpr, CAUSAL: tl.constexpr, WINDOW: tl.constexpr, KL: tl.constexpr,
                 VL: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    seq = token // QLEN
    kvh = head // G
    length = tl.load(LEN + seq)
    qpos = length - QLEN + token % QLEN
    end = tl.minimum(length, qpos + 1) if CAUSAL else length
    begin = tl.maximum(0, qpos + 1 - WINDOW) if WINDOW > 0 else 0
    n = begin + tl.arange(0, BN)
    d = tl.arange(0, D)
    valid = n < end
    physical = tl.load(BT + seq * bst + n // PAGE, valid, other=0).to(tl.int64)
    slot = n % PAGE
    ko = physical[:, None] * kb + kvh * kh + (d[None, :] // 16) * kg + slot[:, None] * kp + d[None, :] % 16
    vo = physical[:, None] * vb + kvh * vh + (slot[:, None] // 16) * vp + d[None, :] * vd + slot[:, None] % 16
    q = tl.load(Q + token * qst + head * qsh + d * qsd).to(tl.float32)
    qs = tl.load(QS + head) if Q_FP8 else 1.0
    ks = tl.load(KS + kvh)
    vs = tl.load(VS + kvh)
    k = tlx.require_layout(tl.load(K + ko, valid[:, None], other=0.0), KL).to(tl.float32)
    score = tlx.release_layout(tl.sum(k * q[None, :], 1)) * (scale * ks * qs * 1.4426950408889634)
    score = tl.where(valid, score, -float('inf'))
    m = tl.max(score, 0)
    m = tl.where(m == -float('inf'), 0.0, m)
    p = tl.exp2(score - m)
    den = tl.maximum(tl.sum(p, 0), 1e-20)
    v = tlx.require_layout(tl.load(V + vo, valid[:, None], other=0.0), VL).to(tl.float32)
    value = tlx.release_layout(tl.sum(p[:, None] * v, 0)) * (vs / den)
    tl.store(O + (token * HQ + head) * D + d, value)


def make_call(x, a, layouts):
    if a.splits != 1: raise ValueError('short candidate requires one split')
    if (a.window or a.length) > a.block_n: raise ValueError('visible bound exceeds tile')
    out = torch.empty_like(x.q, dtype=torch.bfloat16)
    kl, vl = layouts(a.block_n, a.dim, a.warps)
    return lambda: (short_decode[(a.tokens, a.heads)]
                    (x.q, x.kp, x.vp, x.ks, x.vs, x.qs, x.table, x.contexts, out, *x.q.stride(), x.table.stride(
                        0), *x.kp.stride()[:4], *x.vp.stride()[:4], a.dim**-.5, a.dim, a.heads, a.heads // a.kv_heads, a
                     .tokens, a.page, a.block_n, a.fp8_q, a.causal, a.window, kl, vl, num_warps=a.warps), out)[1]
