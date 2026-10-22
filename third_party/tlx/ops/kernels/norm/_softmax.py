from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import _stage_load


@triton.jit
def _softmax(X, D, Y, M, N: tl.constexpr, SX: tl.constexpr, SD: tl.constexpr, SY: tl.constexpr, BM: tl.constexpr,
             BN: tl.constexpr, NW: tl.constexpr, TPR: tl.constexpr, V: tl.constexpr, CHUNKS: tl.constexpr,
             BACKWARD: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN).to(tl.int64)
    xs = ()
    ds = ()
    total = tl.full((BM, ), 0.0 if BACKWARD else -float('inf'), tl.float32)
    for i in tl.static_range(CHUNKS):
        c = i * BN + cols
        mask = (rows[:, None] < M) & (c[None, :] < N)
        x = _stage_load(X, rows[:, None] * SX[0] + c[None, :] * SX[1], mask, BM, BN, DIRECT=True, NW=NW, TPR=TPR,
                        V=V).to(tl.float32)
        if BACKWARD:
            d = _stage_load(D, rows[:, None] * SD[0] + c[None, :] * SD[1], mask, BM, BN, DIRECT=True, NW=NW, TPR=TPR,
                            V=V).to(tl.float32)
            total += tl.sum(x * d, 1)
            ds += (d, )
        else:
            x = tl.where(c[None, :] < N, x, -float('inf'))
            total = tl.maximum(total, tl.max(x, 1))
        xs += (x, )
    if not BACKWARD:
        es = ()
        denominator = tl.full((BM, ), 0.0, tl.float32)
        for i in tl.static_range(CHUNKS):
            e = tl.exp2((xs[i] - total[:, None]) * 1.4426950408889634)
            denominator += tl.sum(e, 1)
            es += (e, )
    for i in tl.static_range(CHUNKS):
        c = i * BN + cols
        if BACKWARD:
            value = xs[i] * (ds[i] - total[:, None])
        else:
            value = es[i] / denominator[:, None]
        tl.store(Y + rows[:, None] * SY[0] + c[None, :] * SY[1], value, (rows[:, None] < M) & (c[None, :] < N))


@triton.jit
def _wide(X, D, Y, PMax, PSum, N: tl.constexpr, SX: tl.constexpr, SD: tl.constexpr, SY: tl.constexpr,
          SPLITS: tl.constexpr, BS: tl.constexpr, BLOCK: tl.constexpr):
    row, part = (tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64))
    c = part * BLOCK + tl.arange(0, BLOCK)
    x = _stage_load(X, row * SX[0] + c[None, :] * SX[1], c[None, :] < N, 1, BLOCK, DIRECT=True, TPR=128).to(tl.float32)
    if D is not None:
        d = _stage_load(D, row * SD[0] + c[None, :] * SD[1], c[None, :] < N, 1, BLOCK, DIRECT=True,
                        TPR=128).to(tl.float32)
    if Y is None:
        if D is not None:
            total = tl.sum(x * d, 1)
        else:
            x = tl.where(c[None, :] < N, x, -float('inf'))
            maximum = tl.max(x, 1)
            total = tl.sum(tl.exp(x - maximum[:, None]), 1)
            tl.store(PMax + row * SPLITS + part, tl.sum(maximum, 0))
        tl.store(PSum + row * SPLITS + part, tl.sum(total, 0))
    else:
        b = tl.arange(0, BS)
        sums = tl.load(PSum + row * SPLITS + b, b < SPLITS, 0)
        if D is not None:
            value = x * (d - tl.sum(sums, 0))
        else:
            maxima = tl.load(PMax + row * SPLITS + b, b < SPLITS, -float('inf'))
            maximum = tl.max(maxima, 0)
            denominator = tl.sum(sums * tl.exp(maxima - maximum), 0)
            value = tl.exp(x - maximum) / denominator
        tl.store(Y + row * SY[0] + c[None, :] * SY[1], value, c[None, :] < N)


def _run(x, dy=None, out=None):
    if out is None:
        out = torch.empty_like(x)
    m, n = x.shape
    if not m or not n:
        return out
    backward = dy is not None
    sd = (0, 0) if dy is None else dy.stride()
    with torch.cuda.device(x.device):
        if n <= 32768:
            bn, chunks, warps = (triton.next_power_of_2(n), 1, 4)
            if n == 384 and x.is_contiguous() and out.is_contiguous() and (dy is None or dy.is_contiguous()):
                bn, chunks = (128, 3)
                if m <= 4096 or (backward and m <= 65536):
                    bm, tpr, vec = (4, 32, 2 if backward else 4)
                else:
                    bm, tpr, vec = (16, 8, 2) if backward else (8, 16, 4)
            elif n <= 128:
                tpr = min(16, bn)
                bm, vec = (128 // tpr, min(4, bn // tpr))
            elif n <= 512:
                bm, tpr, vec = (4, 32, 4)
            elif n <= 2048:
                bm, tpr, vec = (2, 64, 4 if backward else 2)
            elif n <= 16384:
                bm, tpr, vec = (1, 128, 8)
            else:
                bm, tpr, vec, warps = (1, 256, 8, 8)
            _softmax[
                triton.cdiv(m, bm),
            ](x, dy, out, m, n, x.stride(), sd, out.stride(), bm, bn, warps, tpr, vec, chunks, backward,
              num_warps=warps, enable_fp_fusion=False)
        else:
            block = 4096
            splits = triton.cdiv(n, block)
            ps = torch.empty((m, splits), device=x.device, dtype=torch.float32)
            pm = None if backward else torch.empty_like(ps)
            for target in (None, out):
                _wide[m, splits](x, dy, target, pm, ps, n, x.stride(), sd, out.stride(), splits,
                                 triton.next_power_of_2(splits), block, num_warps=4, enable_fp_fusion=False)
    return out


def softmax_fwd(x, *, out=None):
    return _run(x, out=out)


def softmax_bwd(dy, y, *, out=None):
    return _run(y, dy, out=out)


class SoftmaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = softmax_fwd(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        y, = ctx.saved_tensors
        return softmax_bwd(dy, y)


def softmax(x):
    return SoftmaxFunction.apply(x)
