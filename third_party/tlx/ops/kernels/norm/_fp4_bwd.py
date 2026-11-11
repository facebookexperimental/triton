from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._common import _stage_load as _stage
from ._fp4 import _find_sequence, _scale_offset
from .sm100 import norm_backward_from_normalized


@triton.jit
def _reconstruct_nvfp4(Q, S, W, B, DY, Offsets, PaddedOffsets, XHat, CompactDY, M, N: tl.constexpr, inv_global_scale,
                       REST_K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BIAS: tl.constexpr,
                       SEQUENCES: tl.constexpr, ITERS: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    cols = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    words = tl.program_id(1).to(tl.int64) * (BN // 8) + tl.arange(0, BN // 8)
    pmask = (rows < M)[:, None] & (words * 8 < N)[None, :]
    packed = _stage(Q, rows[:, None] * (N // 8) + words[None, :], pmask, BM, BN // 8)
    nibbles = tl.reshape(packed[:, :, None] >> (4 * tl.arange(0, 8))[None, None, :] & 15, (BM, BN))
    code = nibbles & 7
    magnitude = tl.where(code >= 6, 2.0 * (code.to(tl.float32) - 4.0),
                         tl.where(code >= 4,
                                  code.to(tl.float32) - 2.0,
                                  code.to(tl.float32) * 0.5))
    v = tl.where(nibbles & 8 != 0, -magnitude, magnitude)
    sf = tl.load(S + _scale_offset(rows[:, None], (cols // 16)[None, :], REST_K),
                 (rows < M)[:, None] & (cols < N)[None, :], 0)
    scale = sf.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    w = tl.load(W + cols, cols < N, 1).to(tl.float32)
    x_hat = v * (scale * tl.div_rn(inv_global_scale, w[None, :]))
    if BIAS:
        b = tl.load(B + cols, cols < N, 0).to(tl.float32)
        x_hat += tl.div_rn(-b[None, :], w[None, :])
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    tl.store(XHat + rows[:, None] * N + cols[None, :], x_hat, mask)
    sequence = _find_sequence(Offsets, tl.minimum(rows, M - 1), SEQUENCES, ITERS)
    start = tl.load(Offsets + sequence)
    padded = tl.load(PaddedOffsets + sequence)
    padded_rows = rows + padded - start
    dy = _stage(DY, padded_rows[:, None] * N + cols[None, :], mask, BM, BN)
    tl.store(CompactDY + rows[:, None] * N + cols[None, :], dy, mask)


def layernorm_nvfp4_backward_from_x_norm_padded_dy(dy_padded, offsets, padded_offsets, x_norm_fp4, scales_mma,
                                                   global_scale, weight, rstd, norm_bias=None, dx_out=None,
                                                   min_abs_weight=2.0**(-10), residual_grad=None):
    if min_abs_weight is not None:
        minimum = weight.detach().abs().min().item()
        if not (math.isfinite(minimum) and minimum >= min_abs_weight > 0):
            raise ValueError('Saved normalized state cannot recover x_hat for near-zero weight')
    m, n = (x_norm_fp4.shape[0], dy_padded.shape[1])
    sequences = offsets.numel() - 1
    x_hat = torch.empty((m, n), dtype=torch.float32, device=dy_padded.device)
    dy = torch.empty((m, n), dtype=dy_padded.dtype, device=dy_padded.device)
    with torch.cuda.device(dy_padded.device):
        _reconstruct_nvfp4[triton.cdiv(m, 4),
                           triton.cdiv(n, 128)](x_norm_fp4.view(torch.uint32), scales_mma, weight, norm_bias, dy_padded,
                                                offsets, padded_offsets, x_hat, dy, m, n, 1.0 / float(global_scale),
                                                triton.cdiv(n // 16, 4), 4, 128, norm_bias is not None, sequences,
                                                max(1, sequences.bit_length()), num_warps=4, enable_fp_fusion=False)
    dx, dw, db = norm_backward_from_normalized(x_hat, weight, dy, rstd, norm='layer', has_bias=True, dx_dtype=dy.dtype,
                                               dw_dtype=torch.float32, db_dtype=torch.float32,
                                               residual_grad=residual_grad, round_dx_before_accum=True)
    if dx_out is not None:
        dx_out.copy_(dx)
        dx = dx_out
    return (dx, dw, db)
