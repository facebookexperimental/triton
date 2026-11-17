"""Gated sliding-window self-attention with low-rank projections, forward, for sm90 (H100). FP16.

Over packed variable-length sequences ``x`` of shape ``(tokens, d_model)``:

    h = rms_norm(x)
    lora(p) = rms_norm(h @ p.down.T) @ p.up.T
    q = lora(q_proj), (k, v) = per-head halves of lora(kv_proj), gate = sigmoid(lora(gate_proj))
    out = x + window_attention(q, k, v) * gate

where query ``i`` attends keys ``i - left .. i + right`` of its own sequence. Work is split into blocks of 64 rows that
overlap their neighbours by ``left`` and ``right`` rows: a block recomputes the projections of those neighbours and
stores only its middle rows, so no projection leaves the chip. Intermediates are rounded to FP16 where an eager PyTorch
module rounds them.

With two heads, and when the weights fit in shared memory, the TLX kernel `_mla_window_attn_ws` runs: a persistent CTA
keeps every weight in shared memory and two consumer warpgroups walk alternate blocks independently. Otherwise each
block is one program of `_mla_window_attn_fwd`, which reads the weights through registers.
"""

import functools
import re

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from torch._library.triton import wrap_triton
from torch._subclasses.fake_tensor import FakeTensor
from triton.language.extra.libdevice import fast_dividef, fast_expf
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK = 64
NUM_WARPS = 4
CONSUMER_REGISTERS = 240
# Weights plus the consumers' buffers the TLX kernel may hold: d_model 256 with ranks 64/32/64 fills it.
SHARED_BYTES = 212992
LOG2E = 1.4426950408889634


@functools.cache
def _ptxas_before_13():
    """Whether Triton's ptxas predates 13; CUDA 12.4's, which production lowering bundles, needs `SPLIT_KV_DESC`."""
    from triton.backends.nvidia.compiler import get_ptxas_version

    m = re.search(r"release (\d+)\.(\d+)", get_ptxas_version(90))
    return m is None or int(m.group(1)) < 13


@triton.jit
def _sigmoid(x):
    """FP32 sigmoid with libdevice's fast exp and divide. Its error is far below the FP16 rounding that follows, also
    for very negative x, where 0.5 * tanh(x / 2) + 0.5 in FP16 would round small probabilities to zero."""
    return fast_dividef(1.0, 1.0 + fast_expf(-x))


@triton.jit
def _rms(x, w_ptr, N: tl.constexpr, EPS: tl.constexpr):
    """Row RMS norm of an FP32 tile, scaled by an FP16 weight and rounded to FP16."""
    w = tl.load(w_ptr + tl.arange(0, N)).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / N + EPS)
    return (w[None, :] * (x * inv[:, None])).to(tl.float16)


@triton.jit
def _lora_down(h, w_down, w_norm, R: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    """rms_norm(h @ w_down.T) for an FP16 tile h; w_down is (R, D). The product is rounded to FP16 first."""
    r = tl.arange(0, R)
    d = tl.arange(0, D)
    wt = tl.load(w_down + r[None, :] * D + d[:, None])
    return _rms(tl.dot(h, wt).to(tl.float16).to(tl.float32), w_norm, R, EPS)


@triton.jit
def _up(c, w_up, row0, R: tl.constexpr, N: tl.constexpr):
    """c @ w_up[row0:row0 + N].T rounded to FP16; w_up is (*, R)."""
    n = tl.arange(0, N)
    r = tl.arange(0, R)
    wt = tl.load(w_up + (row0 + n[None, :]) * R + r[:, None])
    return tl.dot(c, wt).to(tl.float16)


@triton.jit
def _key_range(Cu, num_seqs, rows, LEFT: tl.constexpr, RIGHT: tl.constexpr):
    """First and last key each row may attend: its window clipped to its sequence. A row's sequence is the number of
    sequence starts after the first that are at or before it, counted 32 starts per load."""
    SEQ_CHUNK: tl.constexpr = 32
    j = tl.arange(0, SEQ_CHUNK)
    b = tl.zeros(rows.shape, dtype=tl.int32)
    for c in range(1, num_seqs, SEQ_CHUNK):
        starts = tl.load(Cu + c + j, mask=c + j < num_seqs, other=2147483647)
        b += tl.sum((starts[None, :] <= rows[:, None]).to(tl.int32), 1)
    b = tl.minimum(b, num_seqs - 1)
    key_lo = tl.maximum(tl.load(Cu + b), rows - LEFT)
    key_hi = tl.minimum(tl.load(Cu + b + 1) - 1, rows + RIGHT)
    return key_lo, key_hi


@triton.jit
def _mla_window_attn_fwd(X, Out, Cu, tokens, num_seqs, w_norm, q_down, q_norm, q_up, kv_down, kv_norm, kv_up, g_down,
                         g_norm, g_up, qk_scale, D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, QR: tl.constexpr,
                         KVR: tl.constexpr, GR: tl.constexpr, LEFT: tl.constexpr, RIGHT: tl.constexpr,
                         NORM_EPS: tl.constexpr, LORA_EPS: tl.constexpr, BLOCK: tl.constexpr):
    STRIDE: tl.constexpr = BLOCK - LEFT - RIGHT
    rows = tl.program_id(0) * STRIDE - LEFT + tl.arange(0, BLOCK)
    inside = (rows >= 0) & (rows < tokens)
    rows64 = rows.to(tl.int64)
    d = tl.arange(0, D)
    x = tl.load(X + rows64[:, None] * D + d[None, :], mask=inside[:, None], other=0.0).to(tl.float32)
    h = _rms(x, w_norm, D, NORM_EPS)
    cq = _lora_down(h, q_down, q_norm, QR, D, LORA_EPS)
    ckv = _lora_down(h, kv_down, kv_norm, KVR, D, LORA_EPS)
    cg = _lora_down(h, g_down, g_norm, GR, D, LORA_EPS)
    key_lo, key_hi = _key_range(Cu, num_seqs, rows, LEFT, RIGHT)
    visible = (rows[None, :] >= key_lo[:, None]) & (rows[None, :] <= key_hi[:, None])
    local = tl.arange(0, BLOCK)
    keep = (local >= LEFT) & (local < LEFT + STRIDE) & inside
    e = tl.arange(0, HD)
    for hh in tl.static_range(H):
        q = _up(cq, q_up, hh * HD, QR, HD)
        k = _up(ckv, kv_up, hh * 2 * HD, KVR, HD)
        v = _up(ckv, kv_up, hh * 2 * HD + HD, KVR, HD)
        gate = _sigmoid(_up(cg, g_up, hh * HD, GR, HD).to(tl.float32)).to(tl.float16)
        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(visible, qk * qk_scale, -float("inf"))
        # Every real row sees itself; only rows outside [0, tokens) see no key, and they are not stored.
        m = tl.max(qk, 1)
        m = tl.where(m == -float("inf"), 0.0, m)
        p = tl.math.exp2(qk - m[:, None])
        l = tl.sum(p, 1)
        o = tl.dot(p.to(tl.float16), v)
        o = (o * tl.fdiv(1.0, tl.where(l == 0.0, 1.0, l), ieee_rounding=False)[:, None]).to(tl.float16)
        ptrs = rows64[:, None] * D + hh * HD + e[None, :]
        xh = tl.load(X + ptrs, mask=keep[:, None], other=0.0)
        tl.store(Out + ptrs, xh + o * gate, mask=keep[:, None])


@triton.jit
def _mla_window_attn_ws(X, Out, Cu, tokens, num_seqs, w_norm, d_q_down, q_norm, d_q_up, d_kv_down, kv_norm, d_kv_up,
                        d_g_down, g_norm, d_g_up, qk_scale, NUM_CTAS: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
                        HD: tl.constexpr, QR: tl.constexpr, KVR: tl.constexpr, GR: tl.constexpr, LEFT: tl.constexpr,
                        RIGHT: tl.constexpr, NORM_EPS: tl.constexpr, LORA_EPS: tl.constexpr,
                        CONSUMER_REGISTERS: tl.constexpr, SPLIT_KV_DESC: tl.constexpr):
    """Persistent, warp-specialized form of `_mla_window_attn_fwd` for heads of d_model / 2 columns.

    The default task loads every weight into shared memory once and exits. Two consumer warpgroups then walk
    alternate blocks of rows independently. A consumer's only buffer, one (64, d_model / 2) tile, holds in turn the
    two halves of its normalized input, then per head k, v and the gated attention output; the low-rank projections,
    q, the scores and the probabilities stay in registers as left operands of the tensor-core products."""
    BLOCK: tl.constexpr = 64
    HALF: tl.constexpr = D // 2
    STRIDE: tl.constexpr = BLOCK - LEFT - RIGHT
    num_tiles = tl.cdiv(tokens, STRIDE)
    # The q and gate down weights are stacked: they share the left operand, so with equal ranks one product gives
    # [cq | cg], half the wgmma instructions of two products.
    MERGE_QG: tl.constexpr = QR == GR
    if MERGE_QG:
        w_qg_down = tlx.local_alloc((QR + GR, D), tl.float16, 1)[0]
        w_q_down = tlx.local_slice(w_qg_down, [0, 0], [QR, D])
        w_g_down = tlx.local_slice(w_qg_down, [QR, 0], [GR, D])
    else:
        w_q_down = tlx.local_alloc((QR, D), tl.float16, 1)[0]
        w_g_down = tlx.local_alloc((GR, D), tl.float16, 1)[0]
    w_kv_down = tlx.local_alloc((KVR, D), tl.float16, 1)
    w_q_up = tlx.local_alloc((HD, QR), tl.float16, H)
    w_k_up = tlx.local_alloc((HD, KVR), tl.float16, H)
    w_v_up = tlx.local_alloc((HD, KVR), tl.float16, H)
    w_g_up = tlx.local_alloc((HD, GR), tl.float16, H)
    tile_s = tlx.local_alloc((BLOCK, HALF), tl.float16, 2)
    loaded = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    with tlx.async_tasks(exclusive=True):
        with tlx.async_task("default"):
            tlx.barrier_expect_bytes(loaded[0], 2 * ((QR + KVR + GR) * D + H * HD * (QR + 2 * KVR + GR)))
            tlx.async_descriptor_load(d_q_down, w_q_down, [0, 0], loaded[0])
            tlx.async_descriptor_load(d_kv_down, w_kv_down[0], [0, 0], loaded[0])
            tlx.async_descriptor_load(d_g_down, w_g_down, [0, 0], loaded[0])
            for hh in tl.static_range(H):
                tlx.async_descriptor_load(d_q_up, w_q_up[hh], [hh * HD, 0], loaded[0])
                tlx.async_descriptor_load(d_kv_up, w_k_up[hh], [hh * 2 * HD, 0], loaded[0])
                tlx.async_descriptor_load(d_kv_up, w_v_up[hh], [hh * 2 * HD + HD, 0], loaded[0])
                tlx.async_descriptor_load(d_g_up, w_g_up[hh], [hh * HD, 0], loaded[0])
        with tlx.async_task(num_warps=4, replicate=2, registers=CONSUMER_REGISTERS):
            cid: tl.constexpr = tlx.async_task_replica_id()
            buf = tile_s[cid]
            kv_buf = buf
            if SPLIT_KV_DESC:
                # ptxas 12 miscompiles the k and v products when they reach the tile through the shared-memory
                # descriptor it already built for the down products (wrong scores, out-of-range shared addresses).
                # A view indexed by a value neither compiler can fold gives them a descriptor of their own; the
                # program id stays opaque where an argument would not (Triton specializes integers equal to 1).
                kv_buf = tile_s[cid + (tl.program_id(0) >= NUM_CTAS).to(tl.int32)]
            tlx.barrier_wait(loaded[0], 0)
            for tile in range(2 * tl.program_id(0) + cid, num_tiles, 2 * NUM_CTAS):
                local = tl.arange(0, BLOCK)
                c = tl.arange(0, HALF)
                rows = tile * STRIDE - LEFT + local
                inside = (rows >= 0) & (rows < tokens)
                offs = rows.to(tl.int64)[:, None] * D + c[None, :]
                lo = tl.load(X + offs, mask=inside[:, None], other=0.0).to(tl.float32)
                hi = tl.load(X + offs + HALF, mask=inside[:, None], other=0.0).to(tl.float32)
                inv = tl.rsqrt((tl.sum(lo * lo, 1) + tl.sum(hi * hi, 1)) / D + NORM_EPS)[:, None]
                h_hi = (tl.load(w_norm + HALF + c).to(tl.float32)[None, :] * (hi * inv)).to(tl.float16)
                # The down products take the normalized input one column half at a time through the buffer.
                tlx.local_store(buf, (tl.load(w_norm + c).to(tl.float32)[None, :] * (lo * inv)).to(tl.float16))
                tlx.fence_async_shared()
                if MERGE_QG:
                    cqg = tlx.async_dot(buf, tlx.local_trans(tlx.local_slice(w_qg_down, [0, 0], [QR + GR, HALF])))
                else:
                    cq = tlx.async_dot(buf, tlx.local_trans(tlx.local_slice(w_q_down, [0, 0], [QR, HALF])))
                    cg = tlx.async_dot(buf, tlx.local_trans(tlx.local_slice(w_g_down, [0, 0], [GR, HALF])))
                ckv = tlx.async_dot(buf, tlx.local_trans(tlx.local_slice(w_kv_down[0], [0, 0], [KVR, HALF])))
                if MERGE_QG:
                    cqg = tlx.async_dot_wait(0, cqg)
                else:
                    cq = tlx.async_dot_wait(0, cq)
                    cg = tlx.async_dot_wait(0, cg)
                ckv = tlx.async_dot_wait(0, ckv)
                tlx.local_store(buf, h_hi)
                tlx.fence_async_shared()
                if MERGE_QG:
                    cqg = tlx.async_dot(buf, tlx.local_trans(tlx.local_slice(w_qg_down, [0, HALF], [QR + GR, HALF])),
                                        cqg)
                else:
                    cq = tlx.async_dot(buf, tlx.local_trans(tlx.local_slice(w_q_down, [0, HALF], [QR, HALF])), cq)
                    cg = tlx.async_dot(buf, tlx.local_trans(tlx.local_slice(w_g_down, [0, HALF], [GR, HALF])), cg)
                ckv = tlx.async_dot(buf, tlx.local_trans(tlx.local_slice(w_kv_down[0], [0, HALF], [KVR, HALF])), ckv)
                if MERGE_QG:
                    cqg = tlx.async_dot_wait(0, cqg)
                    cq, cg = tl.split(tl.permute(tl.reshape(cqg, (BLOCK, 2, QR)), (0, 2, 1)))
                else:
                    cq = tlx.async_dot_wait(0, cq)
                    cg = tlx.async_dot_wait(0, cg)
                ckv = tlx.async_dot_wait(0, ckv)
                cq = _rms(cq.to(tl.float16).to(tl.float32), q_norm, QR, LORA_EPS)
                ckv = _rms(ckv.to(tl.float16).to(tl.float32), kv_norm, KVR, LORA_EPS)
                cg = _rms(cg.to(tl.float16).to(tl.float32), g_norm, GR, LORA_EPS)
                key_lo, key_hi = _key_range(Cu, num_seqs, rows, LEFT, RIGHT)
                keep = (local >= LEFT) & (local < LEFT + STRIDE) & inside
                e = tl.arange(0, HD)
                # A rolled loop: two unrolled copies of the head body per consumer overflow the instruction cache.
                for head in tl.range(H, loop_unroll_factor=1):
                    k = tlx.async_dot(ckv, tlx.local_trans(w_k_up[head]))
                    k = tlx.async_dot_wait(0, k)
                    tlx.local_store(kv_buf, k.to(tl.float16))
                    tlx.fence_async_shared()
                    q = tlx.async_dot(cq, tlx.local_trans(w_q_up[head]))
                    q = tlx.async_dot_wait(0, q).to(tl.float16)
                    qk = tlx.async_dot(q, tlx.local_trans(kv_buf))
                    qk = tlx.async_dot_wait(0, qk)
                    visible = (rows[None, :] >= key_lo[:, None]) & (rows[None, :] <= key_hi[:, None])
                    qk = tl.where(visible, qk * qk_scale, -float("inf"))
                    m = tl.max(qk, 1)
                    m = tl.where(m == -float("inf"), 0.0, m)
                    p = tl.math.exp2(qk - m[:, None])
                    l = tl.sum(p, 1)
                    p = (p * tl.fdiv(1.0, tl.where(l == 0.0, 1.0, l), ieee_rounding=False)[:, None])
                    p = p.to(tl.float16)
                    v = tlx.async_dot(ckv, tlx.local_trans(w_v_up[head]))
                    v = tlx.async_dot_wait(0, v)
                    tlx.local_store(kv_buf, v.to(tl.float16))
                    tlx.fence_async_shared()
                    o = tlx.async_dot(p, kv_buf)
                    o = tlx.async_dot_wait(0, o).to(tl.float16)
                    g = tlx.async_dot(cg, tlx.local_trans(w_g_up[head]))
                    g = tlx.async_dot_wait(0, g)
                    gate = _sigmoid(g).to(tl.float16)
                    tlx.local_store(buf, o * gate)
                    ptrs = rows.to(tl.int64)[:, None] * D + head * HD + e[None, :]
                    xh = tl.load(X + ptrs, mask=keep[:, None], other=0.0)
                    tl.store(Out + ptrs, xh + tlx.local_load(buf), mask=keep[:, None])


def mla_window_attn(x, cu_seqlens, norm_weight, q_proj, kv_proj, gate_proj, num_heads, window_size, norm_eps, lora_eps):
    """Launch the forward; inputs are validated by ``tlx.ops.mla_window_attn``."""
    if isinstance(x, FakeTensor):
        return torch.empty(x.shape, device=x.device, dtype=x.dtype)
    tokens, d_model = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    if tokens == 0:
        return out
    head_dim = d_model // num_heads
    left, right = window_size
    cu_seqlens, norm_weight = cu_seqlens.contiguous(), norm_weight.contiguous()
    (q_down, q_norm, q_up), (kv_down, kv_norm, kv_up), (g_down, g_norm, g_up) = (tuple(t.contiguous()
                                                                                       for t in p)
                                                                                 for p in (q_proj, kv_proj, gate_proj))
    qr, kvr, gr = q_down.shape[0], kv_down.shape[0], g_down.shape[0]
    weight_bytes = 2 * ((qr + kvr + gr) * d_model + num_heads * head_dim * (qr + 2 * kvr + gr))
    if num_heads == 2 and weight_bytes + 2 * BLOCK * d_model <= SHARED_BYTES:
        desc = TensorDescriptor.from_tensor
        # One CTA per SM whatever the token count, so a compiled graph needs no guard on it.
        num_ctas = torch.cuda.get_device_properties(x.device).multi_processor_count
        with torch.cuda.device(x.device):
            wrap_triton(_mla_window_attn_ws)[(num_ctas, )](
                x, out, cu_seqlens, tokens, cu_seqlens.shape[0] - 1, norm_weight, desc(q_down, [qr, d_model]), q_norm,
                desc(q_up, [head_dim, qr]), desc(kv_down, [kvr, d_model]), kv_norm, desc(kv_up, [head_dim, kvr]),
                desc(g_down, [gr, d_model]), g_norm, desc(g_up, [head_dim, gr]), head_dim**-0.5 * LOG2E,
                NUM_CTAS=num_ctas, D=d_model, H=num_heads, HD=head_dim, QR=qr, KVR=kvr, GR=gr, LEFT=left, RIGHT=right,
                NORM_EPS=norm_eps, LORA_EPS=lora_eps, CONSUMER_REGISTERS=CONSUMER_REGISTERS,
                SPLIT_KV_DESC=_ptxas_before_13(), num_warps=4, num_stages=1)
        return out
    grid = (triton.cdiv(tokens, BLOCK - left - right), )
    with torch.cuda.device(x.device):
        wrap_triton(_mla_window_attn_fwd)[grid](x, out, cu_seqlens, tokens, cu_seqlens.shape[0] - 1, norm_weight,
                                                q_down, q_norm, q_up, kv_down, kv_norm, kv_up, g_down, g_norm, g_up,
                                                head_dim**-0.5 * LOG2E, D=d_model, H=num_heads, HD=head_dim, QR=qr,
                                                KVR=kvr, GR=gr, LEFT=left, RIGHT=right, NORM_EPS=norm_eps,
                                                LORA_EPS=lora_eps, BLOCK=BLOCK, num_warps=NUM_WARPS, num_stages=1)
    return out
