from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.tlx.mxfp8_utils import _cvt_e4m3x4_f32

from ._common import _bf16_affine, _stage_load
from ._mxfp8 import _core, _e8m0_to_f32, build_padded_cta_metadata
from .sm100 import norm_bwd


@triton.jit
def _stage_fp8(P, S, indices, scale_indices, mask, BM: tl.constexpr, BN: tl.constexpr):
    raw = _stage_load(P.to(tl.pointer_type(tl.uint8)), indices, mask, BM, BN)
    scale = _e8m0_to_f32(tl.load(S + scale_indices, mask, 127).to(tl.uint8))
    return (raw.to(P.dtype.element_ty, bitcast=True).to(tl.float32), scale)


@triton.jit
def _prepare_backward(X, W, B, Dy, Dv, SfDv, SfX, DyAdd, Rstd, Mean, Xhat, Grad, Saved, M, N: tl.constexpr,
                      BN: tl.constexpr, IS_RMS: tl.constexpr, FROM_NORMALIZED: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    idx = rows[:, None] * N + cols[None, :]
    scale_idx = rows[:, None] * (N // 32) + (cols // 32)[None, :]
    if SfX is not None:
        x, scale = _stage_fp8(X, SfX, idx, scale_idx, mask, 4, BN)
        x = x * scale
    else:
        x = _stage_load(X, idx, mask, 4, BN).to(tl.float32)
    dy = _stage_load(Dy, idx, mask, 4, BN).to(tl.float32)
    w = tl.load(W + cols, mask=cols < N, other=1).to(tl.float32)
    if B is not None:
        b = tl.load(B + cols, mask=cols < N, other=0).to(tl.float32)
    rr = tl.load(Rstd + rows, mask=rows < M, other=0)[:, None]
    if FROM_NORMALIZED:
        affine = x
        if B is not None:
            affine = affine - b[None, :]
        xhat = affine / w[None, :]
    elif IS_RMS:
        xhat = x * rr
    else:
        mu = tl.load(Mean + rows, mask=rows < M, other=0)[:, None]
        xhat = (x - mu) * rr
    if DyAdd is not None:
        dy = (dy + _stage_load(DyAdd, idx, mask, 4, BN).to(tl.float32)).to(Dy.dtype.element_ty).to(tl.float32)
    if Dv is not None:
        dv, scale = _stage_fp8(Dv, SfDv, idx, scale_idx, mask, 4, BN)
        dy = tl.fma(dv, scale, dy) if N == 384 or N == 512 else dy + dv * scale
    tl.store(Xhat + idx, xhat, mask=mask)
    tl.store(Grad + idx, dy, mask=mask)
    if Saved is not None:
        if FROM_NORMALIZED:
            saved = x
        else:
            saved = xhat * w[None, :]
            if B is not None:
                saved = saved + b[None, :]
        tl.store(Saved + idx, saved, mask=mask)


def _saved_backward(dy, x, w, rstd, *, norm, mean=None, save_x_norm=False, dv_fp8=None, sf_dv=None, sf_x_norm=None,
                    dy_bf16_addend=None, norm_bias=None, dx_out=None, x_norm_out=None, dx_bf16_addend=None,
                    from_normalized=False):
    m, n = x.shape
    dx_dtype = dy.dtype if sf_x_norm is not None else x.dtype
    prepare = from_normalized or dv_fp8 is not None or dy_bf16_addend is not None or save_x_norm
    if prepare:
        xhat = torch.empty_like(x, dtype=torch.float32)
        grad = torch.empty_like(xhat)
        save_materialized = save_x_norm and (not from_normalized or sf_x_norm is not None)
        saved = (x_norm_out if x_norm_out is not None else torch.empty_like(
            x, dtype=torch.bfloat16)) if save_materialized else None
        if m:
            with torch.cuda.device(x.device):
                bn = min(512, triton.next_power_of_2(n))
                _prepare_backward[triton.cdiv(m, 4),
                                  triton.cdiv(n, bn)](x, w, norm_bias, dy, dv_fp8, sf_dv, sf_x_norm, dy_bf16_addend,
                                                      rstd, mean, xhat, grad, saved, m, N=n, BN=bn,
                                                      IS_RMS=norm == 'rms', FROM_NORMALIZED=from_normalized,
                                                      num_warps=4, enable_fp_fusion=False)
    dx, dw, db = norm_bwd(xhat if prepare else x, w, grad if prepare else dy, rstd, norm=norm,
                          mean=None if prepare else mean, from_normalized=prepare, has_bias=norm == 'layer',
                          dx_dtype=dx_dtype, dw_dtype=torch.float32, db_dtype=torch.float32, dx_out=dx_out,
                          dx_accum=dx_bf16_addend, round_dx_before_accum=dx_bf16_addend is not None)
    out = (dx, dw, db) if norm == 'layer' else (dx, dw)
    if save_x_norm:
        out += (x if from_normalized and sf_x_norm is None else saved, )
    return out


def rmsnorm_backward(dy, x, w, rstd, save_x_norm=False, dv_fp8=None, sf_dv=None, dy_bf16_addend=None, dx_out=None,
                     x_norm_out=None, dx_bf16_addend=None):
    return _saved_backward(dy, x, w, rstd, norm='rms', save_x_norm=save_x_norm, dv_fp8=dv_fp8, sf_dv=sf_dv,
                           dy_bf16_addend=dy_bf16_addend, dx_out=dx_out, x_norm_out=x_norm_out,
                           dx_bf16_addend=dx_bf16_addend)


def rmsnorm_backward_from_x_norm(dy, x_norm, w, rstd, save_x_norm=False, dv_fp8=None, sf_dv=None, sf_x_norm=None,
                                 dy_bf16_addend=None, dx_out=None, x_norm_out=None, dx_bf16_addend=None):
    return _saved_backward(dy, x_norm, w, rstd, norm='rms', from_normalized=True, save_x_norm=save_x_norm,
                           dv_fp8=dv_fp8, sf_dv=sf_dv, sf_x_norm=sf_x_norm, dy_bf16_addend=dy_bf16_addend,
                           dx_out=dx_out, x_norm_out=x_norm_out, dx_bf16_addend=dx_bf16_addend)


def layernorm_backward(dy, x, w, rstd, mean, save_x_norm=False, dv_fp8=None, sf_dv=None, dy_bf16_addend=None,
                       norm_bias=None, dx_out=None, x_norm_out=None, dx_bf16_addend=None):
    return _saved_backward(dy, x, w, rstd, norm='layer', mean=mean, save_x_norm=save_x_norm, dv_fp8=dv_fp8, sf_dv=sf_dv,
                           dy_bf16_addend=dy_bf16_addend, norm_bias=norm_bias, dx_out=dx_out, x_norm_out=x_norm_out,
                           dx_bf16_addend=dx_bf16_addend)


def layernorm_backward_from_x_norm(dy, x_norm, w, rstd, mean, save_x_norm=False, dv_fp8=None, sf_dv=None,
                                   sf_x_norm=None, dy_bf16_addend=None, norm_bias=None, dx_out=None, x_norm_out=None,
                                   dx_bf16_addend=None, *, early_param_accum=None, sm_count_multiple=None):
    return _saved_backward(dy, x_norm, w, rstd, norm='layer', mean=mean, from_normalized=True, save_x_norm=save_x_norm,
                           dv_fp8=dv_fp8, sf_dv=sf_dv, sf_x_norm=sf_x_norm, dy_bf16_addend=dy_bf16_addend,
                           norm_bias=norm_bias, dx_out=dx_out, x_norm_out=x_norm_out, dx_bf16_addend=dx_bf16_addend)


@triton.jit
def _bf16_mul(a, b):
    return tl.inline_asm_elementwise('mul.rn.bf16x2 $0, $1, $2;', '=r,r,r', [a, b], dtype=tl.bfloat16, is_pure=True,
                                     pack=2)


@triton.jit
def _split_four(x):
    a, b = tl.split(tl.reshape(x, [x.shape[0], x.shape[1], 2, 2]))
    a0, a2 = tl.split(a)
    a1, a3 = tl.split(b)
    return (a0, a1, a2, a3)


@triton.jit
def _bf16_dot8(x, y, BM: tl.constexpr, BN: tl.constexpr):
    xe, xo = tl.split(tl.reshape(x, [BM, BN // 8, 4, 2]))
    ye, yo = tl.split(tl.reshape(y, [BM, BN // 8, 4, 2]))
    x0, x1, x2, x3 = _split_four(xe)
    y0, y1, y2, y3 = _split_four(ye)
    pe = _bf16_mul(x0, y0)
    pe = _bf16_affine(x1, y1, pe)
    pe = _bf16_affine(x2, y2, pe)
    pe = _bf16_affine(x3, y3, pe)
    x0, x1, x2, x3 = _split_four(xo)
    y0, y1, y2, y3 = _split_four(yo)
    po = _bf16_mul(x0, y0)
    po = _bf16_affine(x1, y1, po)
    po = _bf16_affine(x2, y2, po)
    po = _bf16_affine(x3, y3, po)
    return tl.sum(pe.to(tl.float32) + po.to(tl.float32), axis=1, keep_dims=True)


@triton.jit
def _bf16_quant_backward(X, Dy, W, Rstd, Dz, DwPart, Starts, Counts, TILES, N: tl.constexpr, BN: tl.constexpr):
    cols = tl.arange(0, BN)
    lr = tl.arange(0, 32)
    w = tl.load(W + cols, mask=cols < N, other=0).to(tl.bfloat16)
    accum = tl.full([32, BN], 0, tl.bfloat16)
    for tile in range(tl.program_id(0), TILES, tl.num_programs(0)):
        src = tl.load(Starts + tile)
        valid = lr < tl.load(Counts + tile)
        rows = src.to(tl.int64) + lr
        mask = valid[:, None] & (cols < N)[None, :]
        idx = rows[:, None] * N + cols[None, :]
        x = _stage_load(X, idx, mask, 32, BN).to(tl.bfloat16)
        dy = _stage_load(Dy, idx, mask, 32, BN).to(tl.bfloat16)
        rr = tl.load(Rstd + rows, mask=valid, other=0).to(tl.bfloat16)[:, None]
        xhat = _bf16_mul(x, rr)
        wdy = _bf16_mul(dy, w[None, :])
        dot = (_bf16_dot8(xhat, wdy, 32, BN) / N).to(tl.bfloat16)
        neg = (-dot.to(tl.float32)).to(tl.bfloat16)
        dx = _bf16_mul(_bf16_affine(xhat, neg, wdy), rr)
        tl.store(Dz + idx, dx, mask=mask)
        accum = _bf16_affine(dy, xhat, accum)
    tl.store(DwPart + tl.program_id(0) * N + cols, tl.sum(accum.to(tl.float32), axis=0), mask=cols < N)


@triton.jit
def _subtract_and_static_q(Final, Raw, Mean, Rstd, Xhat, Q, M, N: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)
    cols = tl.arange(0, BN)
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    idx = rows[:, None] * N + cols[None, :]
    raw = _stage_load(Raw, idx, mask, 4, BN).to(tl.float32)
    final = _stage_load(Final, idx, mask, 4, BN).to(tl.float32)
    tl.store(Xhat + idx, (final - raw).to(tl.bfloat16).to(tl.float32), mask=mask)
    mean = tl.load(Mean + rows, mask=rows < M, other=0)[:, None]
    rr = tl.load(Rstd + rows, mask=rows < M, other=0)[:, None]
    y = tl.fma(raw, rr, -mean * rr) * 16.0
    tl.store(Q + idx, _cvt_e4m3x4_f32(y), mask=mask)


def _backward_quant(dy, z, weight, rstd, n_heads, head_dim, offsets, padded_offsets, total_padded,
                    precomputed_src_starts=None, precomputed_valid_counts=None, *, bf16_math=False, raw=None,
                    ln_mean=None, ln_rstd=None):
    m, n = z.shape
    if precomputed_src_starts is None:
        starts, counts = build_padded_cta_metadata(offsets, padded_offsets, total_padded)
    else:
        starts, counts = (precomputed_src_starts, precomputed_valid_counts)
    if raw is not None:
        xhat = torch.empty_like(z, dtype=torch.float32)
        static_q = torch.empty_like(z, dtype=torch.float8_e4m3fn)
        if m:
            with torch.cuda.device(z.device):
                _subtract_and_static_q[
                    triton.cdiv(m, 4),
                ](z, raw, ln_mean, ln_rstd, xhat, static_q, m, N=n, BN=triton.next_power_of_2(n), num_warps=4,
                  enable_fp_fusion=False)
        dz, _, _ = norm_bwd(xhat, None, dy, rstd, norm='rms', from_normalized=True, dx_dtype=torch.bfloat16)
        second = static_q.view(m, n_heads, head_dim)
    elif bf16_math:
        tiles = total_padded // 32
        np = min(tiles, torch.cuda.get_device_properties(z.device).multi_processor_count * 5)
        dz = torch.empty_like(z)
        parts = torch.empty((np, n), device=z.device, dtype=torch.float32)
        if m and tiles:
            with torch.cuda.device(z.device):
                _bf16_quant_backward[
                    np,
                ](z, dy, weight, rstd, dz, parts, starts, counts, tiles, N=n, BN=triton.next_power_of_2(n), num_warps=8,
                  enable_fp_fusion=False)
                second = parts.sum(0)
        else:
            second = torch.zeros(n, device=z.device, dtype=torch.float32)
    else:
        dz, second, _ = norm_bwd(z, weight, dy, rstd, norm='rms', dw_dtype=torch.float32)
    quant = _core(dz, torch.ones_like(weight), None, n_heads, head_dim, norm='rms', skip_norm=True, offsets=offsets,
                  padded_offsets=padded_offsets, total_padded=total_padded, head_major=True,
                  precomputed_src_starts=starts, precomputed_valid_counts=counts)
    sfdo, sfdo_dv = _backward_scale_layouts(quant['q'], quant['k'], quant['t'], offsets, padded_offsets, total_padded,
                                            n_heads, head_dim)
    return (dz, second, quant['q'].view(m, n_heads, head_dim), sfdo, sfdo_dv)


def _backward_scale_layouts(q, k, t, offsets, padded_offsets, total_padded, n_heads, head_dim):
    if q.shape[0] == 0:
        size = n_heads * (total_padded // 128) * 512
        z = torch.zeros(size, dtype=torch.uint8, device=q.device)
        return (z, torch.zeros_like(z))
    return (k, t.flatten())


def rmsnorm_backward_quantize_dx_mxfp8(dy, z, weight, rstd, n_heads, head_dim, offsets, padded_offsets, total_padded):
    return _backward_quant(dy, z, weight, rstd, n_heads, head_dim, offsets, padded_offsets, total_padded)


def rmsnorm_backward_quantize_dx_mxfp8_precomputed(dy, z, weight, rstd, n_heads, head_dim, offsets, padded_offsets,
                                                   total_padded, precomputed_src_starts, precomputed_valid_counts):
    return _backward_quant(dy, z, weight, rstd, n_heads, head_dim, offsets, padded_offsets, total_padded,
                           precomputed_src_starts, precomputed_valid_counts)


def rmsnorm_backward_quantize_dx_mxfp8_bf16_precomputed(dy, z, weight, rstd, n_heads, head_dim, offsets, padded_offsets,
                                                        total_padded, precomputed_src_starts, precomputed_valid_counts):
    return _backward_quant(dy, z, weight, rstd, n_heads, head_dim, offsets, padded_offsets, total_padded,
                           precomputed_src_starts, precomputed_valid_counts, bf16_math=True)


def rmsnorm_backward_quantize_dx_mxfp8_from_output_recompute_q_precomputed(dy, final_output, raw, rms_rstd, n_heads,
                                                                           head_dim, offsets, padded_offsets,
                                                                           total_padded, precomputed_src_starts,
                                                                           precomputed_valid_counts, ln_mean, ln_rstd):
    return _backward_quant(dy, final_output,
                           torch.empty(final_output.shape[1], device=final_output.device, dtype=final_output.dtype),
                           rms_rstd, n_heads, head_dim, offsets, padded_offsets, total_padded, precomputed_src_starts,
                           precomputed_valid_counts, raw=raw, ln_mean=ln_mean, ln_rstd=ln_rstd)
