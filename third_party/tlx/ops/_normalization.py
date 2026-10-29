"""Row normalization on CUDA: plain, MXFP8, MXFP4, NVFP4 and softmax.

Plain APIs accept FP16/BF16/FP32 [rows, features] with unit feature stride.
Quantized APIs accept contiguous matrices and preserve GEO's packed scale
layouts. Only layernorm, layernorm_add, rmsnorm, softmax and the MXFP8
with_backward APIs build autograd graphs. Other backward APIs return explicit
first-order gradients. space selects the fixed heuristic (the only space).
Quantized outputs use E4M3 FP8 or packed UINT8 FP4 payloads with MMA/cuBLAS
scale buffers; NVFP4 scales are E4M3 bytes and MX scales are E8M0 bytes.
Saved affine-state backward APIs require nonzero weights to recover x_hat.
Sequence offsets must be monotonic, start at zero and end at the compact/padded
row counts; padded sequence lengths must align to each kernel's row tile.
Supplying total_padded and precomputed metadata avoids device synchronization.
"""
from __future__ import annotations

import math

import torch

from ._catalog import InvalidInput, check_backward, check_inputs, impl_for


def _dispatch(op, *, space=None, **params):
    x = next(iter(params.values()))
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise InvalidInput('normalization expects a rank-2 input tensor')
    if space not in (None, 'heuristic'):
        raise InvalidInput(f"tlx.ops.{op} supports only space='heuristic'")
    fn, spec = impl_for(op, device=x.device)
    check_inputs(spec, dtype=x.dtype)
    tensors = {k: v for k, v in params.items() if isinstance(v, torch.Tensor) and k != 'cpu_offsets'}
    if any(v.device != x.device for v in tensors.values()):
        raise InvalidInput('normalization tensors must be on the same device')
    if 'q0_jagged_transpose' in op and (x.shape[1] != 384 or x.dtype != torch.bfloat16):
        raise InvalidInput('jagged NVFP4 transpose requires BF16 [rows, 384]')
    if op.startswith('softmax') and any(v.dtype != x.dtype for v in tensors.values()):
        raise InvalidInput('softmax tensors must share the input dtype')
    m, n = x.shape
    if op == 'layernorm_nvfp4_backward_from_x_norm_padded_dy':
        m = params['x_norm_fp4'].shape[0]
    if 'softmax' not in op and (n == 0 or x.stride(1) != 1 or x.stride(0) < n):
        raise InvalidInput('normalization requires positive features, unit feature stride and nonoverlapping rows')
    quantized = 'fp' in op or op.startswith(('layernorm_backward', 'rmsnorm_backward'))
    if quantized and not x.is_contiguous():
        raise InvalidInput('quantized normalization requires contiguous inputs')
    if quantized and n % (16 if 'nvfp4' in op else 32):
        raise InvalidInput('quantized feature dimension must be divisible by its scale block')
    for key in ('weight', 'w', 'bias', 'norm_bias', 'residual_norm_weight', 'residual_norm_bias'):
        v = params.get(key)
        if v is not None and (not isinstance(v, torch.Tensor) or v.shape != (n, ) or v.stride() != (1, ) or
                              (v.dtype not in (torch.float16, torch.bfloat16, torch.float32))):
            raise InvalidInput(f'{key} must be a contiguous floating point [features] tensor')
    if 'n_heads' in params:
        heads, dim = (params['n_heads'], params['head_dim'])
        if heads <= 0 or dim <= 0 or dim % 32 or (heads * dim != n):
            raise InvalidInput('positive n_heads * head_dim must equal features; head_dim must be divisible by 32')
    if params.get('norm', 'layer') not in ('layer', 'rms'):
        raise InvalidInput("norm must be 'layer' or 'rms'")
    residual_state = [
        params.get(k)
        for k in ('residual_norm_weight', 'residual_norm_bias', 'residual_norm_mean', 'residual_norm_rstd')
    ]
    if any(v is not None for v in residual_state) and any(v is None for v in residual_state):
        raise InvalidInput('residual LayerNorm reconstruction requires weight, bias, mean and rstd')
    for key in ('out_dtype', 'residual_dtype', 'dx_dtype', 'dw_dtype', 'db_dtype'):
        if params.get(key) not in (None, torch.float16, torch.bfloat16, torch.float32):
            raise InvalidInput(f'{key} must be FP16, BF16 or FP32')
    if 'eps' in params and (not math.isfinite(params['eps']) or params['eps'] < 0):
        raise InvalidInput('eps must be finite and nonnegative')
    if params.get('hadamard_matrix') is not None:
        raise InvalidInput('MXFP4 uses the fixed 32-point Hadamard transform')
    if params.get('quant_block_shape', (1, 16)) not in ((1, 16), (16, 16)):
        raise InvalidInput('NVFP4 quant_block_shape must be (1, 16) or (16, 16)')
    for key in ('global_scale', 'transpose_global_scale', 'nvfp4_global_scale', 'max_norm_rcp'):
        value = params.get(key)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise InvalidInput(f'{key} must be finite and positive')
    for key in ('dout', 'dy', 'y', 'x', 'x_norm', 'z', 'raw', 'final_output', 'addend', 'dout_addend', 'dx_accum',
                'residual', 'post_norm_addend', 'out', 'dx_out', 'residual_grad', 'dy_bf16_addend', 'dx_bf16_addend',
                'x_norm_out', 'dv_fp8'):
        v = params.get(key)
        if v is not None and (not isinstance(v, torch.Tensor) or v.shape != (m, n)):
            raise InvalidInput(f'{key} must have the input shape')
        if v is not None and v.ndim == 2:
            if 'softmax' not in op and (v.stride(1) != 1 or v.stride(0) < n):
                raise InvalidInput(f'{key} requires unit feature stride and nonoverlapping rows')
            if quantized and not v.is_contiguous():
                raise InvalidInput(f'{key} must be contiguous for quantized normalization')
            if key in ('out', 'dx_out') and m and n:
                a, b = sorted(zip(v.stride(), v.shape))
                if a[0] <= 0 or b[0] < a[0] * a[1]:
                    raise InvalidInput(f'{key} must have nonoverlapping output elements')
    for key in ('rstd', 'mean', 'mean_out', 'rstd_out', 'rms_rstd', 'ln_mean', 'ln_rstd', 'residual_norm_mean',
                'residual_norm_rstd'):
        v = params.get(key)
        if v is not None and (v.numel() != m or not v.is_contiguous() or v.dtype != torch.float32):
            raise InvalidInput(f'{key} must have one contiguous FP32 value per row')
    for key in ('offsets', 'padded_offsets', 'cu_seqlens', 'padded_cu_seqlens', 'precomputed_src_starts',
                'precomputed_valid_counts'):
        v = params.get(key)
        if v is not None and (v.ndim != 1 or not v.is_contiguous() or v.dtype not in (torch.int32, torch.int64)):
            raise InvalidInput(f'{key} must be a contiguous integer vector')
    for left, right in (('offsets', 'padded_offsets'), ('cu_seqlens', 'padded_cu_seqlens'),
                        ('precomputed_src_starts', 'precomputed_valid_counts')):
        a, b = params.get(left), params.get(right)
        if (a is None) != (b is None) or (a is not None and
                                          (a.shape != b.shape
                                           or a.numel() < (2 if 'offsets' in left or 'seqlens' in left else 0))):
            raise InvalidInput(f'{left} and {right} must be supplied together with matching lengths')
    total = params.get('total_padded', params.get('padded_tokens'))
    if total is not None and (not isinstance(total, int) or total < m or total % 32):
        raise InvalidInput('padded token count must cover the input rows and be divisible by 32')
    if params.get('precomputed_src_starts') is not None and (total is None or params['precomputed_src_starts'].numel()
                                                             != total // 32):
        raise InvalidInput('precomputed metadata requires one entry per padded 32-row tile')
    for key, data in (('sf_x_norm', params.get('x_norm')), ('sf_dv', params.get('dv_fp8'))):
        sf = params.get(key)
        if sf is not None and (data is None or data.dtype != torch.float8_e4m3fn or sf.shape != (m, n // 32)
                               or not sf.is_contiguous() or sf.dtype not in (torch.uint8, torch.float8_e8m0fnu)):
            raise InvalidInput(f'{key} requires FP8 data and contiguous E8M0 [rows, features/32] scales')
    if params.get('dv_fp8') is not None and params.get('sf_dv') is None:
        raise InvalidInput('dv_fp8 requires sf_dv')
    for key, shape in (('out_fp4', (m, n // 2)), ('x_norm_fp4', (m, n // 2))):
        v = params.get(key)
        if v is not None and (v.shape != shape or v.dtype != torch.uint8 or not v.is_contiguous()):
            raise InvalidInput(f'{key} must be contiguous packed UINT8 [rows, features/2]')
    scale_numel = ((m + 127) // 128) * ((n // 16 + 3) // 4) * 512
    for key in ('out_scales', 'scales_mma'):
        v = params.get(key)
        if v is not None and (v.numel() != scale_numel or v.dtype != torch.uint8 or not v.is_contiguous()):
            raise InvalidInput(f'{key} has an invalid packed NVFP4 scale buffer')
    if 'outputs' in params:
        k = ((params['padded_tokens'] + 63) // 64) * 64
        shapes = ((m, n // 2), (scale_numel, ), (m, ), (n, k // 2), (((n + 127) // 128) * ((k // 16 + 3) // 4) * 512, ),
                  (1, ))
        outputs = params['outputs']
        if len(outputs) != 6 or any(not isinstance(v, torch.Tensor) or v.device != x.device or not v.is_contiguous()
                                    or v.shape != shape or v.dtype != (torch.float32 if i in (2, 5) else torch.uint8)
                                    for i, (v, shape) in enumerate(zip(outputs, shapes))):
            raise InvalidInput('jagged NVFP4 outputs have invalid shapes, dtypes or devices')
    if params.get('n_scores', 0):
        fragments, const, ns = params.get('score_fragments'), params.get('score_const'), params['n_scores']
        if n != 384 or not 0 < ns <= 8 or fragments is None or const is None or fragments.dtype != torch.uint32 or not fragments.is_contiguous(
        ) or fragments.numel() != 3072 or const.dtype != torch.float32 or const.numel(
        ) < ns or not const.is_contiguous():
            raise InvalidInput(
                'row scores require 384 features, 1..8 scores, packed UINT32 fragments and FP32 constants')
    if params.get('return_nvfp4_cta_scale_max') and params.get('nvfp4_global_scale') is None:
        raise InvalidInput('NVFP4 CTA maxima require nvfp4_global_scale')
    if 'backward' not in op and (not op.endswith('_bwd')):
        check_backward(spec, *tensors.values())
    with torch.cuda.device(x.device):
        return fn(**params)


def layernorm(x, weight, eps=1e-06, bias=None, *, space=None):
    """Return (y, rstd, mean) with optional bias and first-order autograd."""
    return _dispatch('layernorm', **locals())


def layernorm_add(x, addend, weight, eps=1e-06, bias=None, *, space=None):
    """Normalize x + addend; return (y, rstd, mean) with first-order autograd."""
    return _dispatch('layernorm_add', **locals())


def rmsnorm(x, weight, bias=None, residual=None, post_norm_addend=None, eps=1e-05, *, space=None):
    """Return (y, residual_out, rstd) with optional residual, bias and post addend."""
    return _dispatch('rmsnorm', **locals())


def norm_fwd(x, weight=None, bias=None, *, norm='layer', eps=1e-06, residual=None, post_norm_addend=None,
             out_dtype=None, residual_dtype=None, store_stats=True, residual_norm_weight=None, residual_norm_bias=None,
             residual_norm_rstd=None, residual_norm_mean=None, out=None, mean_out=None, rstd_out=None,
             return_residual=None, space=None):
    """Return (out, residual_out, mean, rstd); supplied output and statistic buffers are reused."""
    return _dispatch('norm_fwd', **locals())


def norm_bwd(x, weight, dout, rstd, *, mean=None, norm='rms', has_bias=False, addend=None, dout_addend=None,
             dx_accum=None, dx_out=None, from_normalized=False, dx_dtype=None, dw_dtype=None, db_dtype=None,
             round_dx_before_accum=False, space=None):
    """Return explicit first-order (dx, dw, db) from raw input and saved statistics."""
    return _dispatch('norm_bwd', **locals())


def norm_backward_from_normalized(x_hat, weight, dout, rstd, *, norm='layer', has_bias=False, dx_dtype=None,
                                  dw_dtype=torch.float32, db_dtype=torch.float32, residual_grad=None,
                                  round_dx_before_accum=False, space=None):
    """Return explicit first-order (dx, dw, db) from pre-affine x_hat and saved rstd."""
    return _dispatch('norm_backward_from_normalized', **locals())


def softmax(x, *, space=None):
    """Return row softmax with first-order autograd."""
    return _dispatch('softmax', **locals())


def softmax_fwd(x, *, out=None, space=None):
    """Write row softmax to out or return a new tensor."""
    return _dispatch('softmax_fwd', **locals())


def softmax_bwd(dy, y, *, out=None, space=None):
    """Return y * (dy - sum(y * dy)) along each row."""
    return _dispatch('softmax_bwd', **locals())


def layernorm_mxfp4(x, weight, hadamard_matrix=None, eps=1e-06, return_rstd=False, return_mean=False,
                    stochastic_rounding=False, *, space=None):
    return _dispatch('layernorm_mxfp4', **locals())


def rmsnorm_mxfp4(x, weight, hadamard_matrix=None, eps=1e-06, gemma=False, return_rstd=False, stochastic_rounding=False,
                  *, space=None):
    return _dispatch('rmsnorm_mxfp4', **locals())


def layernorm_nvfp4_quantize(x, weight, eps=1e-05, bias=None, global_scale=None, save_rstd=False, use_layernorm=True,
                             quant_block_shape=(1, 16), *, space=None):
    return _dispatch('layernorm_nvfp4_quantize', **locals())


def rmsnorm_nvfp4_quantize(x, weight, eps=1e-05, global_scale=None, save_rstd=False, quant_block_shape=(1, 16), *,
                           space=None):
    return _dispatch('rmsnorm_nvfp4_quantize', **locals())


def layernorm_nvfp4_quantize_inplace(x, weight, out_fp4, out_scales, global_scale, *, eps=1e-05, bias=None,
                                     use_layernorm=True, rstd_out=None, mean_out=None, quant_block_shape=(1, 16),
                                     space=None):
    return _dispatch('layernorm_nvfp4_quantize_inplace', **locals())


def nvfp4_quantize_normalized(y, global_scale, quant_block_shape=(1, 16), *, out_fp4=None, out_scales=None,
                              return_logical_scales=False, space=None):
    return _dispatch('nvfp4_quantize_normalized', **locals())


def layernorm_nvfp4_quantize_q0_jagged_transpose(x, weight, offsets, padded_offsets, *, padded_tokens, max_seq_len,
                                                 eps=1e-05, bias=None, global_scale=1.0, transpose_global_scale=64.0,
                                                 space=None):
    return _dispatch('layernorm_nvfp4_quantize_q0_jagged_transpose', **locals())


def layernorm_nvfp4_quantize_q0_jagged_transpose_inplace(x, weight, offsets, padded_offsets, outputs, *, padded_tokens,
                                                         max_seq_len, eps=1e-05, bias=None, global_scale=1.0,
                                                         transpose_global_scale=64.0, space=None):
    return _dispatch('layernorm_nvfp4_quantize_q0_jagged_transpose_inplace', **locals())


def compute_norm_nvfp4_global_scale(x, weight, bias=None, use_layernorm=False, *, space=None):
    return _dispatch('compute_norm_nvfp4_global_scale', **locals())


def layernorm_nvfp4_backward_from_x_norm_padded_dy(dy_padded, offsets, padded_offsets, x_norm_fp4, scales_mma,
                                                   global_scale, weight, rstd, norm_bias=None, dx_out=None,
                                                   min_abs_weight=2.0**(-10), residual_grad=None, *, space=None):
    return _dispatch('layernorm_nvfp4_backward_from_x_norm_padded_dy', **locals())


def norm_quant_mxfp8_fwd(
    x: torch.Tensor, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None, eps: float = 1e-05,
    norm: str = 'layer', amax_block: str = '1x32', scale_layout: str = 'rowmajor', return_norm_out: bool = True,
    return_stats: bool = True, return_colsum: bool = False, offsets: torch.Tensor | None = None,
    padded_offsets: torch.Tensor | None = None, total_padded: int | None = None, groups_per_tile: int | None = None, *,
    space=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (y, mean, rstd, fp8, scales, scales_t, colsum); disabled outputs are empty."""
    return _dispatch('norm_quant_mxfp8_fwd', **locals())


def rmsnorm_mxfp8_quantize(x, weight, n_heads, head_dim, eps=1e-05, save_rstd=False, offsets=None, padded_offsets=None,
                           total_padded=None, return_colsum=False, return_represented_v=False, *, space=None):
    return _dispatch('rmsnorm_mxfp8_quantize', **locals())


def layernorm_mxfp8_quantize(x, weight, n_heads, head_dim, eps=1e-05, save_rstd=False, offsets=None,
                             padded_offsets=None, total_padded=None, bias=None, max_norm_rcp=None,
                             split_scale_stores=None, return_colsum=False, cutlass_head_major_scales=False,
                             initialize_scale_storage=True, return_normalized=False, return_represented_v=False,
                             precomputed_src_starts=None, precomputed_valid_counts=None, nvfp4_global_scale=None,
                             emit_kblock_scales=True, nvfp4_padded_scale_output=False, return_nvfp4_cta_scale_max=False,
                             score_fragments=None, score_const=None, n_scores=0, *, space=None):
    return _dispatch('layernorm_mxfp8_quantize', **locals())


def layernorm_mxfp8_quantize_numerics_preserving(x, weight, n_heads, head_dim, offsets, padded_offsets, total_padded,
                                                 eps=1e-05, bias=None, max_norm_rcp=1 / 256,
                                                 prepare_forward_scales=True, write_represented_v=False,
                                                 split_scale_stores=False, return_colsum=False, *, space=None):
    return _dispatch('layernorm_mxfp8_quantize_numerics_preserving', **locals())


def layernorm_mxfp8_with_backward(x, weight, bias, n_heads, head_dim, *, space=None):
    """Return (x.clone(), q, k, t); the carrier routes its gradient through LayerNorm."""
    return _dispatch('layernorm_mxfp8_with_backward', **locals())


def layernorm_mxfp8_gdpa_with_backward(x, weight, bias, n_heads, head_dim, offsets, padded_offsets, cpu_offsets,
                                       total_padded, precomputed_src_starts=None, precomputed_valid_counts=None, *,
                                       space=None):
    """Return (normalized_y, q, head-major_k, t) with first-order LayerNorm autograd."""
    return _dispatch('layernorm_mxfp8_gdpa_with_backward', **locals())


def layernorm_mxfp8_gdpa_precomputed_no_normalized(x, weight, bias, n_heads, head_dim, offsets, padded_offsets,
                                                   cpu_offsets, total_padded, precomputed_src_starts,
                                                   precomputed_valid_counts, *, space=None):
    return _dispatch('layernorm_mxfp8_gdpa_precomputed_no_normalized', **locals())


def rmsnorm_mxfp8_quantize_3232_2pass(x, weight, cu_seqlens, padded_cu_seqlens, n_heads, head_dim, eps=1e-05,
                                      total_padded=None, save_rstd=False, jagged_output=False, *, space=None):
    return _dispatch('rmsnorm_mxfp8_quantize_3232_2pass', **locals())


def rmsnorm_backward(dy, x, w, rstd, save_x_norm=False, dv_fp8=None, sf_dv=None, dy_bf16_addend=None, dx_out=None,
                     x_norm_out=None, dx_bf16_addend=None, *, space=None):
    return _dispatch('rmsnorm_backward', **locals())


def rmsnorm_backward_from_x_norm(dy, x_norm, w, rstd, save_x_norm=False, dv_fp8=None, sf_dv=None, sf_x_norm=None,
                                 dy_bf16_addend=None, dx_out=None, x_norm_out=None, dx_bf16_addend=None, *, space=None):
    return _dispatch('rmsnorm_backward_from_x_norm', **locals())


def layernorm_backward(dy, x, w, rstd, mean, save_x_norm=False, dv_fp8=None, sf_dv=None, dy_bf16_addend=None,
                       norm_bias=None, dx_out=None, x_norm_out=None, dx_bf16_addend=None, *, space=None):
    return _dispatch('layernorm_backward', **locals())


def layernorm_backward_from_x_norm(dy, x_norm, w, rstd, mean, save_x_norm=False, dv_fp8=None, sf_dv=None,
                                   sf_x_norm=None, dy_bf16_addend=None, norm_bias=None, dx_out=None, x_norm_out=None,
                                   dx_bf16_addend=None, *, early_param_accum=None, sm_count_multiple=None, space=None):
    return _dispatch('layernorm_backward_from_x_norm', **locals())


def rmsnorm_backward_quantize_dx_mxfp8(dy, z, weight, rstd, n_heads, head_dim, offsets, padded_offsets, total_padded, *,
                                       space=None):
    return _dispatch('rmsnorm_backward_quantize_dx_mxfp8', **locals())


def rmsnorm_backward_quantize_dx_mxfp8_precomputed(dy, z, weight, rstd, n_heads, head_dim, offsets, padded_offsets,
                                                   total_padded, precomputed_src_starts, precomputed_valid_counts, *,
                                                   space=None):
    return _dispatch('rmsnorm_backward_quantize_dx_mxfp8_precomputed', **locals())


def rmsnorm_backward_quantize_dx_mxfp8_bf16_precomputed(dy, z, weight, rstd, n_heads, head_dim, offsets, padded_offsets,
                                                        total_padded, precomputed_src_starts, precomputed_valid_counts,
                                                        *, space=None):
    return _dispatch('rmsnorm_backward_quantize_dx_mxfp8_bf16_precomputed', **locals())


def rmsnorm_backward_quantize_dx_mxfp8_from_output_recompute_q_precomputed(dy, final_output, raw, rms_rstd, n_heads,
                                                                           head_dim, offsets, padded_offsets,
                                                                           total_padded, precomputed_src_starts,
                                                                           precomputed_valid_counts, ln_mean, ln_rstd,
                                                                           *, space=None):
    return _dispatch('rmsnorm_backward_quantize_dx_mxfp8_from_output_recompute_q_precomputed', **locals())
