from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import _scale_layout as slu
from ._common import _stage_load as _stage


@triton.jit
def _scale_offset(row, col, rest_k):
    return slu.mma_scale_offset(row.to(tl.int64), col.to(tl.int64), rest_k)


@triton.jit
def _mx_scale(amax):
    return tl.inline_asm_elementwise(
        '{ .reg .f32 v; .reg .b16 p; .reg .u32 e, bits; mul.rn.f32 v, $2, 0f3E2AAAAB; cvt.rp.satfinite.ue8m0x2.f32 p, v, v; cvt.u32.u16 e, p; and.b32 $0, e, 255; sub.u32 e, 254, $0; shl.b32 bits, e, 23; mov.b32 $1, bits; }',
        constraints='=r,=f,f', args=[amax], dtype=(tl.uint32, tl.float32), is_pure=True, pack=1)


@triton.jit
def _nv_scale(amax, global_scale):
    return tl.inline_asm_elementwise(
        '{ .reg .f32 v, sf; .reg .b16 p, h; .reg .b32 hp; .reg .u32 e; mul.rn.f32 v, $2, 0f3E2AAAAB; mul.rn.f32 v, v, $3; cvt.rn.satfinite.e4m3x2.f32 p, v, v; cvt.u32.u16 e, p; and.b32 $0, e, 255; cvt.rn.f16x2.e4m3x2 hp, p; mov.b32 {h, _}, hp; cvt.f32.f16 sf, h; div.rn.f32 $1, $3, sf; }',
        constraints='=r,=f,f,f', args=[amax, global_scale], dtype=(tl.uint32, tl.float32), is_pure=True, pack=1)


@triton.jit
def _pack_pair(lo, hi):
    return tl.inline_asm_elementwise('{ .reg .b8 p; cvt.rn.satfinite.e2m1x2.f32 p, $2, $1; cvt.u32.u8 $0, p; }',
                                     constraints='=r,f,f', args=[lo, hi], dtype=tl.uint32, is_pure=True, pack=1)


@triton.jit
def _pack_stochastic(v0, v1, v2, v3, random_bits):
    return tl.inline_asm_elementwise(
        '{ .reg .b16 p; cvt.rs.satfinite.e2m1x4.f32 p, {$4, $3, $2, $1}, $5; cvt.u32.u16 $0, p; }',
        constraints='=r,f,f,f,f,r', args=[v0, v1, v2, v3, random_bits], dtype=tl.uint32, is_pure=True, pack=1)


@triton.jit
def _fht32_step(values, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pairs = tl.reshape(values, (BM, BN // (2 * D), 2, D))
    lo, hi = tl.split(tl.trans(pairs, (0, 1, 3, 2)))
    added, subtracted = tl.inline_asm_elementwise('add.rn.bf16x2 $0, $2, $3; sub.rn.bf16x2 $1, $2, $3;',
                                                  constraints='=r,=r,r,r', args=[lo, hi],
                                                  dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=2)
    return tl.reshape(tl.trans(tl.join(added, subtracted), (0, 1, 3, 2)), (BM, BN))


@triton.jit
def _fht32(y, BM: tl.constexpr, BN: tl.constexpr):
    values = y.to(tl.bfloat16)
    for stage in tl.static_range(5):
        values = _fht32_step(values, BM, BN, 1 << stage)
    return (values.to(tl.float32) * 0.1767766953125).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _epilogue(y, Q, S, rows, col0, M, N: tl.constexpr, rest_k, global_scale, BM: tl.constexpr, BN: tl.constexpr,
              MX: tl.constexpr, QROWS: tl.constexpr, STOCH: tl.constexpr):
    if MX:
        y = _fht32(y, BM, BN)
        groups = tl.reshape(y, (BM, BN // 32, 32))
        amax = tl.max(tl.abs(groups), axis=2)
        sf, inverse = _mx_scale(amax)
        scaled = tl.reshape(groups * inverse[:, :, None], (BM, BN))
        scale_col = col0 // 32 + tl.arange(0, BN // 32)
    else:
        y = y.to(tl.bfloat16).to(tl.float32)
        groups = tl.reshape(y, (BM, BN // 16, 16))
        amax = tl.max(tl.abs(groups), axis=2)
        if QROWS == 16:
            am = tl.reshape(amax, (BM // 16, 16, BN // 16))
            block_amax = tl.max(am, axis=1)
            amax = tl.reshape(tl.broadcast_to(block_amax[:, None, :], (BM // 16, 16, BN // 16)), (BM, BN // 16))
        sf, inverse = _nv_scale(amax, global_scale)
        scaled = tl.reshape(groups * inverse[:, :, None], (BM, BN))
        scale_col = col0 // 16 + tl.arange(0, BN // 16)
    scale_offsets = _scale_offset(rows[:, None], scale_col[None, :], rest_k)
    block_size: tl.constexpr = 32 if MX else 16
    tl.store(S + scale_offsets, sf.to(tl.uint8), (rows < M)[:, None] & (scale_col * block_size < N)[None, :])
    if STOCH:
        pieces = tl.reshape(scaled, (BM, BN // 4, 2, 2))
        v02, v13 = tl.split(pieces)
        v0, v2 = tl.split(v02)
        v1, v3 = tl.split(v13)
        cols4 = col0 // 4 + tl.arange(0, BN // 4)
        cols8 = cols4 // 2 * 8
        bits = rows[:, None].to(tl.uint32) * 2654435761 + cols8[None, :].to(tl.uint32) * 2246822519 ^ 3735928559
        bits = bits ^ bits << 13
        bits = bits ^ bits >> 17
        bits = bits ^ bits << 5
        packed = _pack_stochastic(v0, v1, v2, v3, bits)
        tl.store(Q + rows[:, None] * (N // 4) + cols4[None, :], packed.to(tl.uint16),
                 (rows < M)[:, None] & (cols4 * 4 < N)[None, :])
    else:
        lo, hi = tl.split(tl.reshape(scaled, (BM, BN // 2, 2)))
        packed = _pack_pair(lo, hi)
        cols2 = col0 // 2 + tl.arange(0, BN // 2)
        tl.store(Q + rows[:, None] * (N // 2) + cols2[None, :], packed.to(tl.uint8),
                 (rows < M)[:, None] & (cols2 * 2 < N)[None, :])
    return sf


@triton.jit
def _norm_quant(X, W, B, Q, S, Mean, Rstd, Normalized, M, N: tl.constexpr, eps, global_scale, rest_k, BM: tl.constexpr,
                BN: tl.constexpr, RMS: tl.constexpr, MX: tl.constexpr, GEMMA: tl.constexpr, QROWS: tl.constexpr,
                STOCH: tl.constexpr, WIDE: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    col0 = tl.program_id(1).to(tl.int64) * BN if WIDE else 0
    cols = col0 + tl.arange(0, BN)
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    x = _stage(X, rows[:, None] * N + cols[None, :], mask, BM, BN, DIRECT=BN >= 1024 and BM >= 4).to(tl.float32)
    if WIDE:
        rstd = tl.load(Rstd + rows, rows < M, 0)
        centered = x
        if not RMS:
            centered -= tl.load(Mean + rows, rows < M, 0)[:, None]
    else:
        if RMS:
            mean = tl.zeros((BM, ), tl.float32)
            variance = tl.sum(x * x, axis=1) / N
            centered = x
        else:
            mean = tl.sum(x, axis=1) / N
            centered = tl.where((cols < N)[None, :], x - mean[:, None], 0.0)
            if MX:
                variance = tl.sum(centered * centered, axis=1) / N
            else:
                variance = tl.sum(x * x, axis=1) / N - mean * mean
        rstd = libdevice.rsqrt(variance + eps)
    weight = tl.load(W + cols, cols < N, 0).to(tl.float32)
    if GEMMA:
        weight = weight + 1.0
    y = centered * rstd[:, None] * weight[None, :]
    if B is not None:
        y += tl.load(B + cols, cols < N, 0).to(tl.float32)[None, :]
    y = tl.where(mask, y, 0.0).to(tl.bfloat16).to(tl.float32)
    if not WIDE and Rstd is not None:
        tl.store(Rstd + rows, rstd, rows < M)
        if not RMS:
            tl.store(Mean + rows, mean, rows < M)
    if Normalized is not None:
        tl.store(Normalized + rows[:, None] * N + cols[None, :], y, mask)
    _epilogue(y, Q, S, rows, col0, M, N, rest_k, global_scale, BM, BN, MX, QROWS, STOCH)


@triton.jit
def _partial_stats(X, Mean, PSum, PSq, M, N: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr,
                   CENTERED: tl.constexpr):
    row, part = (tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64))
    cols = part * BLOCK + tl.arange(0, BLOCK)
    x = _stage(X, cols[None, :] + row * N, (cols < N)[None, :], 1, BLOCK).to(tl.float32)
    if CENTERED:
        mean = tl.load(Mean + row)
        x = tl.where((cols < N)[None, :], x - mean, 0.0)
    tl.store(PSum + row * PARTS + part, tl.sum(x))
    tl.store(PSq + row * PARTS + part, tl.sum(x * x))


@triton.jit
def _finish_stats(PSum, PSq, Mean, Rstd, N: tl.constexpr, eps, PARTS: tl.constexpr, BP: tl.constexpr, RMS: tl.constexpr,
                  CENTERED: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BP)
    sq = tl.sum(tl.load(PSq + row * PARTS + c, c < PARTS, 0)) / N
    if RMS:
        mu = 0.0
        variance = sq
    elif CENTERED:
        mu = tl.load(Mean + row)
        variance = sq
    else:
        mu = tl.sum(tl.load(PSum + row * PARTS + c, c < PARTS, 0)) / N
        variance = sq - mu * mu
        tl.store(Mean + row, mu)
    tl.store(Rstd + row, libdevice.rsqrt(variance + eps))


def get_mma_scale_size(m, n, block=16):
    return slu.mma_scale_numel(m, n // block)


def _outputs(x, mx, q, s):
    m, n = x.shape
    block = 32 if mx else 16
    if q is None:
        q = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    if s is None:
        s = torch.empty((get_mma_scale_size(m, n, block), ), dtype=torch.uint8, device=x.device)
        if m % 128 or n % (4 * block):
            s.fill_(127 if mx else 56)
    return (q, s)


def compute_norm_nvfp4_global_scale(x, weight, bias=None, use_layernorm=False):
    estimate = float(weight.detach().abs().max().item()) * math.sqrt(2 * math.log(max(weight.shape[-1], 2)))
    if use_layernorm and bias is not None:
        estimate += float(bias.detach().abs().max().item())
    return 2688.0 / max(estimate, 1e-12)


def _run(x, weight, bias=None, *, mx=False, rms=False, eps=1e-05, global_scale=1.0, stats=False, gemma=False,
         stochastic=False, qrows=1, out_fp4=None, out_scales=None, rstd_out=None, mean_out=None, normalized_out=None):
    m, n = x.shape
    block = 32 if mx else 16
    q, s = _outputs(x, mx, out_fp4, out_scales)
    wide = n > (2048 if qrows == 16 else 8192)
    rr = rstd_out
    mu = mean_out
    if stats or wide:
        if rr is None:
            rr = torch.empty(m, dtype=torch.float32, device=x.device)
        if mu is None:
            mu = torch.empty_like(rr)
    gs = 1.0 if mx else float(global_scale)
    if m:
        with torch.cuda.device(x.device):
            bn = triton.next_power_of_2(n)
            bm = 16 if qrows == 16 else max(1, min(16, 2048 // bn))
            if not mx and n == 384 and (m >= 4096) and (qrows == 1):
                bm = 8
            if wide:
                parts = triton.cdiv(n, 4096)
                ps = torch.empty((m, parts), dtype=torch.float32, device=x.device)
                pq = torch.empty_like(ps)
                for centered in (False, True) if mx and (not rms) else (False, ):
                    _partial_stats[m, parts](x, mu, ps, pq, m, n, parts, 4096, centered, num_warps=4)
                    _finish_stats[
                        m,
                    ](ps, pq, mu, rr, n, eps, parts, triton.next_power_of_2(parts), rms, centered, num_warps=4)
                bm, bn = (16 if qrows == 16 else 4, 128)
            _norm_quant[triton.cdiv(m, bm),
                        triton.cdiv(n, bn)](x, weight, bias, q.view(torch.uint16) if stochastic else q, s,
                                            mu if stats or wide else None,
                                            rr if stats or wide else None, normalized_out, m, n, eps, gs,
                                            triton.cdiv(n // block, 4), bm, bn, rms, mx, gemma, qrows, stochastic, wide,
                                            num_warps=4)
    return (q, s, rr, mu)


def layernorm_mxfp4(x, weight, hadamard_matrix=None, eps=1e-06, return_rstd=False, return_mean=False,
                    stochastic_rounding=False):
    q, s, rr, mu = _run(x, weight, mx=True, eps=eps, stats=return_rstd or return_mean, stochastic=stochastic_rounding)
    return (q, s) + ((rr, ) if return_rstd else ()) + ((mu, ) if return_mean else ())


def rmsnorm_mxfp4(x, weight, hadamard_matrix=None, eps=1e-06, gemma=False, return_rstd=False,
                  stochastic_rounding=False):
    q, s, rr, _ = _run(x, weight, mx=True, rms=True, eps=eps, stats=return_rstd, gemma=gemma,
                       stochastic=stochastic_rounding)
    return (q, s) + ((rr, ) if return_rstd else ())


def layernorm_nvfp4_quantize(x, weight, eps=1e-05, bias=None, global_scale=None, save_rstd=False, use_layernorm=True,
                             quant_block_shape=(1, 16)):
    gs = compute_norm_nvfp4_global_scale(x, weight, bias, use_layernorm) if global_scale is None else global_scale
    q, s, rr, mu = _run(x, weight, bias, rms=not use_layernorm, eps=eps, global_scale=gs, stats=save_rstd,
                        qrows=quant_block_shape[0])
    return (q, s) + ((rr, mu) if save_rstd and use_layernorm else (rr, ) if save_rstd else ())


def rmsnorm_nvfp4_quantize(x, weight, eps=1e-05, global_scale=None, save_rstd=False, quant_block_shape=(1, 16)):
    return layernorm_nvfp4_quantize(x, weight, eps, None, global_scale, save_rstd, False, quant_block_shape)


def layernorm_nvfp4_quantize_inplace(x, weight, out_fp4, out_scales, global_scale, *, eps=1e-05, bias=None,
                                     use_layernorm=True, rstd_out=None, mean_out=None, quant_block_shape=(1, 16)):
    _run(x, weight, bias, rms=not use_layernorm, eps=eps, global_scale=global_scale, stats=rstd_out is not None
         or mean_out is not None, qrows=quant_block_shape[0], out_fp4=out_fp4, out_scales=out_scales, rstd_out=rstd_out,
         mean_out=mean_out)


@triton.jit
def _quantize_bf16(Y, Q, S, Logical, M, N: tl.constexpr, global_scale, REST_K: tl.constexpr, BM: tl.constexpr,
                   BN: tl.constexpr, QROWS: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    col0 = tl.program_id(1).to(tl.int64) * BN
    cols = col0 + tl.arange(0, BN)
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    y = _stage(Y, rows[:, None] * N + cols[None, :], mask, BM, BN).to(tl.float32)
    sf = _epilogue(y, Q, S, rows, col0, M, N, REST_K, global_scale, BM, BN, False, QROWS, False)
    if Logical is not None:
        sc = col0 // 16 + tl.arange(0, BN // 16)
        tl.store(Logical + rows[:, None] * (N // 16) + sc[None, :], sf.to(tl.uint8),
                 (rows < M)[:, None] & (sc * 16 < N)[None, :])


def nvfp4_quantize_normalized(y, global_scale, quant_block_shape=(1, 16), *, out_fp4=None, out_scales=None,
                              return_logical_scales=False):
    m, n = y.shape
    q, s = _outputs(y, False, out_fp4, out_scales)
    logical = torch.empty((m, n // 16), device=y.device, dtype=torch.uint8) if return_logical_scales else None
    if m:
        bm = 16 if quant_block_shape[0] == 16 else 4
        with torch.cuda.device(y.device):
            _quantize_bf16[triton.cdiv(m, bm), triton.cdiv(n, 128)](y, q, s, logical, m, n, float(global_scale),
                                                                    triton.cdiv(n // 16, 4), bm, 128,
                                                                    quant_block_shape[0], num_warps=4)
    return (q, s, logical) if return_logical_scales else (q, s)


@triton.jit
def _find_sequence(Offsets, row, SEQUENCES: tl.constexpr, ITERS: tl.constexpr):
    lo = tl.full(row.shape, 0, tl.int32)
    hi = tl.full(row.shape, SEQUENCES, tl.int32)
    for _ in tl.static_range(ITERS):
        mid = (lo + hi) // 2
        start = tl.load(Offsets + mid)
        lower = start <= row
        lo = tl.where(lower, mid, lo)
        hi = tl.where(lower, hi, mid)
    return tl.minimum(lo, SEQUENCES - 1)


@triton.jit
def _transpose_quant(Y, Offsets, PaddedOffsets, QT, ST, M, PADDED, GEMM_K: tl.constexpr, global_scale,
                     REST_K: tl.constexpr, FEATURES: tl.constexpr, SEQUENCES: tl.constexpr, ITERS: tl.constexpr):
    feature_tile = tl.program_id(0) % (384 // FEATURES)
    padded_tile = tl.program_id(0) // (384 // FEATURES)
    features = feature_tile.to(tl.int64) * FEATURES + tl.arange(0, FEATURES)
    padded_base = padded_tile.to(tl.int64) * 16
    sequence = _find_sequence(PaddedOffsets, padded_base, SEQUENCES, ITERS)
    start = tl.load(Offsets + sequence)
    length = tl.load(Offsets + sequence + 1) - start
    within = padded_base - tl.load(PaddedOffsets + sequence) + tl.arange(0, 16)
    compact = start + within
    valid = (within >= 0) & (within < length) & (padded_base + tl.arange(0, 16) < PADDED)
    pair_features = feature_tile.to(tl.int64) * (FEATURES // 2) + tl.arange(0, FEATURES // 2)
    mask = valid[:, None] & (pair_features < 192)[None, :]
    loaded_words = _stage(Y, compact[:, None] * 192 + pair_features[None, :], mask, 16, FEATURES // 2)
    low = (loaded_words & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    high = (loaded_words >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    loaded = tl.reshape(tl.join(low, high), (16, FEATURES))
    y = tl.trans(loaded).to(tl.float32)
    amax = tl.max(tl.abs(y), axis=1)
    sf, inv = _nv_scale(amax, global_scale)
    lo, hi = tl.split(tl.reshape(y * inv[:, None], (FEATURES, 8, 2)))
    packed = _pack_pair(lo, hi)
    active = tl.sum(valid.to(tl.int32), axis=0) > 0
    pairs = padded_base // 2 + tl.arange(0, 8)
    tl.store(QT + features[:, None] * (GEMM_K // 2) + pairs[None, :], packed.to(tl.uint8),
             (features < 384)[:, None] & active)
    sf_offset = _scale_offset(features, padded_base // 16, REST_K)
    tl.store(ST + sf_offset, sf.to(tl.uint8), (features < 384) & active)


def layernorm_nvfp4_quantize_q0_jagged_transpose_inplace(x, weight, offsets, padded_offsets, outputs, *, padded_tokens,
                                                         max_seq_len, eps=1e-05, bias=None, global_scale=1.0,
                                                         transpose_global_scale=64.0):
    m, n = x.shape
    sequences = offsets.numel() - 1
    gemm_k = triton.cdiv(padded_tokens, 64) * 64
    gs, tg = (float(global_scale), float(transpose_global_scale))
    outputs[1].fill_(56)
    outputs[3].zero_()
    outputs[4].fill_(56)
    outputs[5].fill_(1.0 / tg)
    y = torch.empty_like(x)
    _run(x, weight, bias, eps=eps, global_scale=gs, stats=True, out_fp4=outputs[0], out_scales=outputs[1],
         rstd_out=outputs[2], normalized_out=y)
    if m:
        with torch.cuda.device(x.device):
            _transpose_quant[
                triton.cdiv(n, 128) * triton.cdiv(gemm_k, 16),
            ](y.view(torch.uint32), offsets, padded_offsets, outputs[3], outputs[4], m, padded_tokens, gemm_k, tg,
              triton.cdiv(gemm_k // 16, 4), 128, sequences, max(1, sequences.bit_length()), num_warps=4)


def layernorm_nvfp4_quantize_q0_jagged_transpose(x, weight, offsets, padded_offsets, *, padded_tokens, max_seq_len,
                                                 eps=1e-05, bias=None, global_scale=1.0, transpose_global_scale=64.0):
    m, n = x.shape
    gemm_k = triton.cdiv(padded_tokens, 64) * 64
    shapes = ((m, n // 2), (get_mma_scale_size(m, n), ), (m, ), (n, gemm_k // 2), (get_mma_scale_size(n,
                                                                                                      gemm_k), ), (1, ))
    outputs = tuple((torch.empty(shape, device=x.device, dtype=torch.float32 if i in (2, 5) else torch.uint8)
                     for i, shape in enumerate(shapes)))
    layernorm_nvfp4_quantize_q0_jagged_transpose_inplace(x, weight, offsets, padded_offsets, outputs,
                                                         padded_tokens=padded_tokens, max_seq_len=max_seq_len, eps=eps,
                                                         bias=bias, global_scale=global_scale,
                                                         transpose_global_scale=transpose_global_scale)
    return outputs
