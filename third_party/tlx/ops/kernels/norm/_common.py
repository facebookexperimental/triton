from __future__ import annotations

import triton
import triton.language as tl
from triton.language import core
from triton.language.extra import tlx


@core.builtin
def pin(value, layout, _semantic=None):
    layout = core._unwrap_if_constexpr(layout)
    encoding = layout.to_ir(_semantic.builder, value.shape, value.dtype)
    anchored = _semantic.builder.create_require_layout(value.handle, encoding)
    return core.tensor(_semantic.builder.create_release_layout(anchored), value.type)


@triton.jit
def _read_load(P, indices, mask, BM: tl.constexpr, BN: tl.constexpr, ROW_STRIDE: tl.constexpr, N: tl.constexpr,
               DIRECT: tl.constexpr, NW: tl.constexpr, TPR: tl.constexpr, V: tl.constexpr, USE_LAYOUT: tl.constexpr):
    if USE_LAYOUT:
        return _stage_load(P, indices, mask, BM, BN, ROW_STRIDE=ROW_STRIDE, N=N, DIRECT=DIRECT, NW=NW, TPR=TPR, V=V)
    return tl.load(P + _row_offset(indices, ROW_STRIDE, N), mask=mask, other=0)


@triton.jit
def _read_x(X, Residual, RW, RB, RM, RR, row, cols, mask, N: tl.constexpr, BN: tl.constexpr, XS: tl.constexpr,
            RS: tl.constexpr, BM: tl.constexpr = 1, DIRECT: tl.constexpr = True, NW: tl.constexpr = 4,
            TPR: tl.constexpr = 128, V: tl.constexpr = 4, USE_LAYOUT: tl.constexpr = True, ROW_MASK=True):
    row = tl.full((BM, ), 0, tl.int64) + row.to(tl.int64)
    idx = row[:, None] * N + cols[None, :]
    x = _read_load(X, idx, mask, BM, BN, ROW_STRIDE=XS, N=N, DIRECT=DIRECT, NW=NW, TPR=TPR, V=V,
                   USE_LAYOUT=USE_LAYOUT).to(tl.float32)
    if Residual is not None:
        r = _read_load(Residual, idx, mask, BM, BN, ROW_STRIDE=RS, N=N, DIRECT=DIRECT, NW=NW, TPR=TPR, V=V,
                       USE_LAYOUT=USE_LAYOUT).to(tl.float32)
        if RW is not None:
            mu = tl.reshape(tl.load(RM + row, mask=ROW_MASK, other=0), [BM, 1])
            rr = tl.reshape(tl.load(RR + row, mask=ROW_MASK, other=0), [BM, 1])
            w = tl.load(RW + cols, mask=cols < N, other=0)
            b = tl.load(RB + cols, mask=cols < N, other=0)
            rn = tl.fma(r, rr, -mu * rr).to(tl.bfloat16)
            r = _bf16_affine(rn, w[None, :], b[None, :]).to(tl.float32)
        x = x + r
    return tl.where(mask, x, 0.0)


@triton.jit
def _read_bwd(X, Dy, Addend, DoutAddend, row, cols, mask, N: tl.constexpr, BN: tl.constexpr, XS: tl.constexpr,
              DS: tl.constexpr, AS: tl.constexpr, DAS: tl.constexpr, BM: tl.constexpr = 1, DIRECT: tl.constexpr = True,
              NW: tl.constexpr = 4, TPR: tl.constexpr = 128, V: tl.constexpr = 4, USE_LAYOUT: tl.constexpr = True):
    row = tl.full((BM, ), 0, tl.int64) + row.to(tl.int64)
    idx = row[:, None] * N + cols[None, :]
    x = _read_load(X, idx, mask, BM, BN, ROW_STRIDE=XS, N=N, DIRECT=DIRECT, NW=NW, TPR=TPR, V=V,
                   USE_LAYOUT=USE_LAYOUT).to(tl.float32)
    dy = _read_load(Dy, idx, mask, BM, BN, ROW_STRIDE=DS, N=N, DIRECT=DIRECT, NW=NW, TPR=TPR, V=V,
                    USE_LAYOUT=USE_LAYOUT).to(tl.float32)
    if Addend is not None:
        x = x + _read_load(Addend, idx, mask, BM, BN, ROW_STRIDE=AS, N=N, DIRECT=DIRECT, NW=NW, TPR=TPR, V=V,
                           USE_LAYOUT=USE_LAYOUT).to(tl.float32)
    if DoutAddend is not None:
        dy = (dy + _read_load(DoutAddend, idx, mask, BM, BN, ROW_STRIDE=DAS, N=N, DIRECT=DIRECT, NW=NW, TPR=TPR, V=V,
                              USE_LAYOUT=USE_LAYOUT).to(tl.float32)).to(Dy.dtype.element_ty).to(tl.float32)
    return (x, dy)


@triton.jit
def _row_offset(indices, ROW_STRIDE: tl.constexpr, N: tl.constexpr):
    if ROW_STRIDE != N:
        return indices // N * ROW_STRIDE + indices % N
    return indices


@triton.jit
def _stage_load(P, indices, mask, BM: tl.constexpr, BN: tl.constexpr, ROW_STRIDE: tl.constexpr = None,
                N: tl.constexpr = 1, DIRECT: tl.constexpr = False, NW: tl.constexpr = 4, TPR: tl.constexpr = 32,
                V: tl.constexpr = 4):
    if ROW_STRIDE is not None:
        indices = _row_offset(indices, ROW_STRIDE, N)
    if DIRECT:
        layout: tl.constexpr = tlx.layout(shape=((TPR, NW * 32 // TPR), (V, BN // (TPR * V), BM // (NW * 32 // TPR))),
                                          stride=((V, BN), (1, TPR * V, NW * 32 // TPR * BN)))
        out = pin(tl.load(P + indices, mask=mask, other=0), layout)
    else:
        buf = tlx.local_alloc((BM, BN), tlx.dtype_of(P), 1)
        tlx.async_load(P + indices, buf[0], mask=mask, other=0)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(0)
        tlx.fence_async_shared()
        tl.debug_barrier()
        out = tlx.local_load(buf[0])
        tl.debug_barrier()
    return out


@triton.jit
def _bf16_affine(a, w, b):
    return tl.inline_asm_elementwise('fma.rn.bf16x2 $0, $1, $2, $3;', constraints='=r,r,r,r',
                                     args=[a.to(tl.bfloat16), w.to(tl.bfloat16),
                                           b.to(tl.bfloat16)], dtype=tl.bfloat16, is_pure=True, pack=2)
