"""RMS-normalized SwiGLU feed-forward with residual, forward, for sm90 (H100). FP16.

    out = x + (silu(n @ w1.T) * (n @ w3.T)) @ w2.T,   n = rms_norm(x)

with ``x`` of shape ``(tokens, d_model)`` and square ``(d_model, d_model)`` weights. The kernel is persistent and
warp-specialized: a producer streams column chunks of w1/w3 (rows) and w2 (columns) through a ring of TMA buffers, and
two consumer warpgroups, each on its own block of 64 rows, share every chunk. The w1 and w3 rows of a chunk land in one
buffer, so a single product gives [a | b]. A consumer keeps n in shared memory and feeds silu(a) * b to the w2 product
from registers. Intermediates are rounded to FP16 where an eager PyTorch module rounds them.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from torch._library.triton import wrap_triton
from torch._subclasses.fake_tensor import FakeTensor
from triton.language.extra.libdevice import fast_dividef, fast_expf
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK = 64
CHUNK = 32
STAGES = 2
CONSUMER_REGISTERS = 240


@triton.jit
def _swiglu_ffn_fwd(X, Out, w_norm, desc_w1, desc_w3, desc_w2, tokens, eps, NUM_CTAS: tl.constexpr, D: tl.constexpr,
                    BLOCK: tl.constexpr, CHUNK: tl.constexpr, STAGES: tl.constexpr, CONSUMER_REGISTERS: tl.constexpr):
    NCH: tl.constexpr = D // CHUNK
    HALF: tl.constexpr = D // 2
    num_rounds = tl.cdiv(tokens, 2 * BLOCK)
    n_s = tlx.local_alloc((BLOCK, D), tl.float16, 2)
    w13_s = tlx.local_alloc((2 * CHUNK, D), tl.float16, STAGES)
    w2_s = tlx.local_alloc((D, CHUNK), tl.float16, STAGES)
    full = tlx.alloc_barriers(num_barriers=STAGES, arrive_count=1)
    empty = tlx.alloc_barriers(num_barriers=STAGES, arrive_count=2)
    with tlx.async_tasks(exclusive=True):
        with tlx.async_task("default"):
            buf = 0
            p = 1
            for _ in range(tl.program_id(0), num_rounds, NUM_CTAS):
                for j in range(NCH):
                    tlx.barrier_wait(empty[buf], p)
                    tlx.barrier_expect_bytes(full[buf], 3 * CHUNK * D * 2)
                    tlx.async_descriptor_load(desc_w1, tlx.local_slice(w13_s[buf], [0, 0], [CHUNK, D]), [j * CHUNK, 0],
                                              full[buf])
                    tlx.async_descriptor_load(desc_w3, tlx.local_slice(w13_s[buf], [CHUNK, 0], [CHUNK, D]),
                                              [j * CHUNK, 0], full[buf])
                    tlx.async_descriptor_load(desc_w2, w2_s[buf], [0, j * CHUNK], full[buf])
                    p = p ^ (buf == STAGES - 1)
                    buf = (buf + 1) % STAGES
        with tlx.async_task(num_warps=4, replicate=2, registers=CONSUMER_REGISTERS):
            cid: tl.constexpr = tlx.async_task_replica_id()
            c = tl.arange(0, HALF)
            buf = 0
            p = 0
            for r in range(tl.program_id(0), num_rounds, NUM_CTAS):
                rows = (2 * r + cid) * BLOCK + tl.arange(0, BLOCK)
                inside = rows < tokens
                offs = rows.to(tl.int64)[:, None] * D + c[None, :]
                # The norm runs over two column halves to bound the registers it holds.
                lo = tl.load(X + offs, mask=inside[:, None], other=0.0).to(tl.float32)
                hi = tl.load(X + offs + HALF, mask=inside[:, None], other=0.0).to(tl.float32)
                inv = tl.rsqrt((tl.sum(lo * lo, 1) + tl.sum(hi * hi, 1)) / D + eps)[:, None]
                tlx.local_store(tlx.local_slice(n_s[cid], [0, 0], [BLOCK, HALF]),
                                (tl.load(w_norm + c).to(tl.float32)[None, :] * (lo * inv)).to(tl.float16))
                tlx.local_store(tlx.local_slice(n_s[cid], [0, HALF], [BLOCK, HALF]),
                                (tl.load(w_norm + HALF + c).to(tl.float32)[None, :] * (hi * inv)).to(tl.float16))
                tlx.fence_async_shared()
                acc = tl.zeros([BLOCK, D], dtype=tl.float32)
                for j in range(NCH):
                    tlx.barrier_wait(full[buf], p)
                    # One N = 2 * CHUNK product: half as many wgmma instructions, which matters most under ptxas 12.4,
                    # which serializes every wgmma in this kernel.
                    ab = tlx.async_dot(n_s[cid], tlx.local_trans(w13_s[buf]))
                    ab = tlx.async_dot_wait(0, ab)
                    a, b = tl.split(tl.permute(tl.reshape(ab, (BLOCK, 2, CHUNK)), (0, 2, 1)))
                    a = a.to(tl.float16).to(tl.float32)
                    # Fast sigmoid: its error is far below the FP16 rounding that follows.
                    u = (a * fast_dividef(1.0, 1.0 + fast_expf(-a))).to(tl.float16) * b.to(tl.float16)
                    acc = tlx.async_dot(u, tlx.local_trans(w2_s[buf]), acc)
                    acc = tlx.async_dot_wait(0, acc)
                    tlx.barrier_arrive(empty[buf])
                    p = p ^ (buf == STAGES - 1)
                    buf = (buf + 1) % STAGES
                # n is no longer needed: the FP16 product goes through its buffer, read back one half at a time.
                tlx.local_store(n_s[cid], acc.to(tl.float16))
                for half in tl.static_range(2):
                    y = tlx.local_load(tlx.local_slice(n_s[cid], [0, half * HALF], [BLOCK, HALF]))
                    x = tl.load(X + offs + half * HALF, mask=inside[:, None], other=0.0)
                    tl.store(Out + offs + half * HALF, x + y, mask=inside[:, None])


def swiglu_ffn(x, norm_weight, w1, w2, w3, eps):
    """Launch the forward; inputs are validated by ``tlx.ops.swiglu_ffn``."""
    if isinstance(x, FakeTensor):
        return torch.empty(x.shape, device=x.device, dtype=x.dtype)
    tokens, d_model = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    if tokens == 0:
        return out
    norm_weight, w1, w2, w3 = (w.contiguous() for w in (norm_weight, w1, w2, w3))
    # One CTA per SM whatever the token count, so a compiled graph needs no guard on it.
    num_ctas = torch.cuda.get_device_properties(x.device).multi_processor_count
    with torch.cuda.device(x.device):
        wrap_triton(_swiglu_ffn_fwd)[(num_ctas, )](x, out, norm_weight,
                                                   TensorDescriptor.from_tensor(w1, [CHUNK, d_model]),
                                                   TensorDescriptor.from_tensor(w3, [CHUNK, d_model]),
                                                   TensorDescriptor.from_tensor(w2, [d_model, CHUNK]), tokens, eps,
                                                   NUM_CTAS=num_ctas, D=d_model, BLOCK=BLOCK, CHUNK=CHUNK,
                                                   STAGES=STAGES, CONSUMER_REGISTERS=CONSUMER_REGISTERS, num_warps=4,
                                                   num_stages=1)
    return out
