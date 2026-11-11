from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice, tlx
from triton.language.extra.cuda.inline_ptx_lib import _fma_f32x2, _mul_f32x2, _sub_f32x2
from triton.language.extra.tlx.mxfp8_utils import (
    _cvt_e4m3x4_f32,
    _fused_amax_to_e8m0,
    _to_mxfp8_block,
)

from . import _scale_layout as slu
from ._common import _bf16_affine, _stage_load
from .sm100 import norm_fwd

_AMAX_1X32 = 0
_AMAX_32X1 = 1
_AMAX_32X32 = 2
_AMAX_MODES = {'1x32': _AMAX_1X32, '32x1': _AMAX_32X1, '32x32': _AMAX_32X32}
_SCALE_LAYOUTS = ('rowmajor', 'mma', 'cublas', 'dual')
_NORMS = ('layer', 'rms')


@triton.jit
def _kernel_build_jagged_tile_metadata(Offsets, PaddedOffsets, SrcStarts, ValidCounts, num_tiles, num_seqs,
                                       BLOCK_M: tl.constexpr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    """Precompute (compact source row, valid row count) for every padded tile."""
    tile = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = tile < num_tiles
    padded_row = tile.to(tl.int64) * BLOCK_M
    lo = tl.zeros([BLOCK], dtype=tl.int32)
    hi = tl.full([BLOCK], num_seqs - 1, dtype=tl.int32)
    for _ in tl.static_range(ITERS):
        mid = (lo + hi + 1) // 2
        take = tl.load(PaddedOffsets + mid, mask=live, other=0) <= padded_row
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)
    local = padded_row - tl.load(PaddedOffsets + lo, mask=live, other=0)
    seq_start = tl.load(Offsets + lo, mask=live, other=0)
    seq_len = tl.load(Offsets + lo + 1, mask=live, other=0) - seq_start
    nvalid = tl.minimum(tl.maximum(seq_len - local, 0), BLOCK_M)
    tl.store(SrcStarts + tile, seq_start + local, mask=live)
    tl.store(ValidCounts + tile, nvalid, mask=live)


@triton.jit
def _norm_quant_epilogue(y, OutFp8, ScaleK, ScaleM, rows, cols, row_blocks, row_stride, scale_k_stride, scale_m_stride,
                         n_cb_k, n_cb_m, n_row_blocks, row_mask, col_offset, VEC: tl.constexpr,
                         GROUPS_PER_TILE: tl.constexpr, AMAX_MODE: tl.constexpr, LAYOUT: tl.constexpr,
                         DUAL: tl.constexpr, JAGGED: tl.constexpr) -> None:
    """Quantize one N-chunk of the normalized output and store data + scales."""
    BLOCK_M: tl.constexpr = y.shape[0]
    CHUNK_N: tl.constexpr = y.shape[1]
    CG: tl.constexpr = CHUNK_N // VEC
    if AMAX_MODE == 0:
        q, sc = _to_mxfp8_block(y, VEC, tl.float8e4nv)
    elif AMAX_MODE == 1:
        y_t = tl.trans(y)
        q_t, sc_t = _to_mxfp8_block(y_t, VEC, tl.float8e4nv)
        q = tl.trans(q_t)
        sc = tl.trans(sc_t)
    else:
        absr = tl.reshape(tl.abs(y), [BLOCK_M, CG, VEC])
        per_row = tl.max(absr, axis=2)
        pr_g = tl.reshape(per_row, [GROUPS_PER_TILE, VEC, CG])
        tile_amax = tl.max(pr_g, axis=1)
        exp_g, inv_g = _fused_amax_to_e8m0(tile_amax, 1.0 / 448.0)
        sc_g = exp_g.to(tl.uint8)
        inv_rows = tl.reshape(tl.broadcast_to(inv_g[:, None, :], [GROUPS_PER_TILE, VEC, CG]), [BLOCK_M, CG])
        scaled = _mul_f32x2(tl.reshape(y, [BLOCK_M, CG, VEC]), inv_rows[:, :, None])
        q = tl.reshape(_cvt_e4m3x4_f32(scaled), [BLOCK_M, CHUNK_N])
        sc = tl.reshape(tl.broadcast_to(sc_g[:, None, :], [GROUPS_PER_TILE, VEC, CG]), [BLOCK_M, CG])
    tl.store(OutFp8 + rows[:, None] * row_stride + cols[None, :], q, mask=row_mask[:, None], cache_modifier='.cs')
    gcols = col_offset // VEC + tl.arange(0, CG)
    blk_mask = row_blocks < n_row_blocks
    if AMAX_MODE == 1:
        slu.store_scales(ScaleM, cols[:, None], row_blocks[None, :], tl.trans(sc), blk_mask[None, :], scale_m_stride, 1,
                         n_cb_m, LAYOUT=LAYOUT)
    elif AMAX_MODE == 0 or not DUAL:
        slu.store_scales(ScaleK, rows[:, None], gcols[None, :], sc, row_mask[:, None], scale_k_stride, 1, n_cb_k,
                         LAYOUT=LAYOUT)
    else:
        sc_wide = tl.reshape(tl.broadcast_to(tl.expand_dims(sc_g, 2), [GROUPS_PER_TILE, CG, VEC]),
                             [GROUPS_PER_TILE, CHUNK_N]).to(tl.uint8)
        slu.store_mma_scales(ScaleK, rows[:, None], gcols[None, :], sc, n_cb_k, row_mask[:, None])
        slu.store_cublas_blocked_scales(ScaleM, cols[:, None], row_blocks[None, :], tl.trans(sc_wide), n_cb_m,
                                        blk_mask[None, :])


@triton.jit
def kernel_norm_quant_mxfp8_ws(X, W, B, NormOut, Mean, Rstd, OutFp8, ScaleK, ScaleM, ColsumPartials, SrcStarts,
                               ValidCounts, row_stride, scale_k_stride, scale_m_stride, n_cb_k, n_cb_m, M, n_row_blocks,
                               num_blocks, eps, N: tl.constexpr, BLOCK_M: tl.constexpr, VEC: tl.constexpr,
                               GROUPS_PER_TILE: tl.constexpr, CHUNK_N: tl.constexpr, NUM_CHUNKS: tl.constexpr,
                               NB: tl.constexpr, IS_RMS: tl.constexpr, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr,
                               AMAX_MODE: tl.constexpr, LAYOUT: tl.constexpr, DUAL: tl.constexpr,
                               RETURN_NORM_OUT: tl.constexpr, RETURN_STATS: tl.constexpr, RETURN_COLSUM: tl.constexpr,
                               JAGGED: tl.constexpr) -> None:
    """Persistent, warp-specialized norm + MXFP8 quantize."""
    COMPUTE_DTYPE: tl.constexpr = tl.float32
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    x_buffer = tlx.local_alloc((BLOCK_M, CHUNK_N), tlx.dtype_of(X), NB * NUM_CHUNKS)
    x_full = tlx.alloc_barriers(num_barriers=NB)
    x_empty = tlx.alloc_barriers(num_barriers=NB)
    x_desc = tl.make_tensor_descriptor(X, shape=[M, N], strides=[row_stride, 1], block_shape=[BLOCK_M, CHUNK_N])
    expected_bytes: tl.constexpr = BLOCK_M * CHUNK_N * NUM_CHUNKS * tlx.size_of(tlx.dtype_of(X))
    with tlx.async_tasks():
        with tlx.async_task(num_warps=1, num_regs=24):
            for it in range(pid, num_blocks, nprog):
                _it = (it - pid) // nprog
                slot = _it % NB
                phase = _it // NB & 1
                if JAGGED:
                    src = tl.load(SrcStarts + it).to(tl.int64)
                else:
                    src = it.to(tl.int64) * BLOCK_M
                tlx.barrier_wait(x_empty[slot], phase ^ 1)
                tlx.barrier_expect_bytes(x_full[slot], expected_bytes)
                for c in tl.static_range(NUM_CHUNKS):
                    tlx.async_descriptor_load(x_desc, tlx.local_view(x_buffer, slot * NUM_CHUNKS + c),
                                              [src.to(tl.int32), c * CHUNK_N], x_full[slot])
        with tlx.async_task('default'):
            lrow = tl.arange(0, BLOCK_M)
            for it in range(pid, num_blocks, nprog):
                _it = (it - pid) // nprog
                slot = _it % NB
                phase = _it // NB & 1
                if JAGGED:
                    src = tl.load(SrcStarts + it).to(tl.int64)
                    rmask = lrow < tl.load(ValidCounts + it)
                else:
                    src = it.to(tl.int64) * BLOCK_M
                    rmask = src + lrow < M
                rows = src + lrow.to(tl.int64)
                row_blocks = it.to(tl.int64) * GROUPS_PER_TILE + tl.arange(0, GROUPS_PER_TILE).to(tl.int64)
                tlx.barrier_wait(x_full[slot], phase)
                sum_x = tl.zeros([BLOCK_M, 1], dtype=COMPUTE_DTYPE)
                sum_sq = tl.zeros([BLOCK_M, 1], dtype=COMPUTE_DTYPE)
                for c in tl.static_range(NUM_CHUNKS):
                    xc = tlx.local_load(tlx.local_view(x_buffer, slot * NUM_CHUNKS + c)).to(COMPUTE_DTYPE)
                    if not IS_RMS:
                        sum_x += tl.sum(xc, axis=1, keep_dims=True)
                    sum_sq += tl.sum(xc * xc, axis=1, keep_dims=True)
                if IS_RMS:
                    mean = tl.zeros([BLOCK_M, 1], dtype=COMPUTE_DTYPE)
                    rstd = libdevice.rsqrt(sum_sq / N + eps)
                else:
                    mean = sum_x / N
                    rstd = libdevice.rsqrt(sum_sq / N - mean * mean + eps)
                if RETURN_STATS:
                    tl.store(Rstd + rows, tl.reshape(rstd, [BLOCK_M]), mask=rmask, cache_modifier='.cs')
                    if not IS_RMS:
                        tl.store(Mean + rows, tl.reshape(mean, [BLOCK_M]), mask=rmask, cache_modifier='.cs')
                for c in tl.static_range(NUM_CHUNKS):
                    ccols = c * CHUNK_N + tl.arange(0, CHUNK_N)
                    xc = tlx.local_load(tlx.local_view(x_buffer, slot * NUM_CHUNKS + c)).to(COMPUTE_DTYPE)
                    if IS_RMS:
                        yc = xc * rstd
                    else:
                        yc = (xc - mean) * rstd
                    if HAS_WEIGHT:
                        yc = yc * tl.load(W + ccols).to(COMPUTE_DTYPE)
                    if HAS_BIAS:
                        yc = yc + tl.load(B + ccols).to(COMPUTE_DTYPE)
                    yc = tl.where(rmask[:, None], yc, 0.0)
                    if RETURN_NORM_OUT:
                        tl.store(NormOut + rows[:, None] * row_stride + ccols[None, :], yc.to(tl.bfloat16),
                                 mask=rmask[:, None], cache_modifier='.cs')
                    if RETURN_COLSUM:
                        tl.atomic_add(ColsumPartials + pid * N + ccols, tl.sum(yc, axis=0))
                    _norm_quant_epilogue(yc, OutFp8, ScaleK, ScaleM, rows, ccols, row_blocks, row_stride,
                                         scale_k_stride, scale_m_stride, n_cb_k, n_cb_m, n_row_blocks, rmask,
                                         c * CHUNK_N, VEC, GROUPS_PER_TILE, AMAX_MODE, LAYOUT, DUAL, JAGGED)
                tlx.barrier_arrive(x_empty[slot], 1)


@triton.jit
def kernel_norm_quant_mxfp8_tma_pipeline(
        X, W, B, NormOut, Mean, Rstd, OutFp8, ScaleK, ScaleM, ColsumPartials, SrcStarts, ValidCounts, row_stride,
        scale_k_stride, scale_m_stride, n_cb_k, n_cb_m, M, n_row_blocks, num_blocks, eps, N: tl.constexpr,
        BLOCK_M: tl.constexpr, VEC: tl.constexpr, GROUPS_PER_TILE: tl.constexpr, CHUNK_N: tl.constexpr,
        NUM_CHUNKS: tl.constexpr, NB: tl.constexpr, IS_RMS: tl.constexpr, HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr, AMAX_MODE: tl.constexpr, LAYOUT: tl.constexpr, DUAL: tl.constexpr,
        RETURN_NORM_OUT: tl.constexpr, RETURN_STATS: tl.constexpr, RETURN_COLSUM: tl.constexpr, JAGGED: tl.constexpr,
        PACKED_MATH: tl.constexpr, BF16_QUANTIZATION: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    x_buffer = tlx.local_alloc((BLOCK_M, CHUNK_N), tlx.dtype_of(X), 2 * NUM_CHUNKS)
    x_full = tlx.alloc_barriers(num_barriers=2)
    x_desc = tl.make_tensor_descriptor(X, shape=[M, N], strides=[row_stride, 1], block_shape=[BLOCK_M, CHUNK_N])
    expected_bytes: tl.constexpr = BLOCK_M * CHUNK_N * NUM_CHUNKS * tlx.size_of(tlx.dtype_of(X))
    lrow = tl.arange(0, BLOCK_M)
    ccols = tl.arange(0, CHUNK_N)
    load_layout: tl.constexpr = None
    if pid < num_blocks:
        if JAGGED:
            first_src = tl.load(SrcStarts + pid)
        else:
            first_src = pid * BLOCK_M
        tlx.barrier_expect_bytes(x_full[0], expected_bytes)
        for c in tl.static_range(NUM_CHUNKS):
            tlx.async_descriptor_load(x_desc, tlx.local_view(x_buffer, c), [first_src.to(tl.int32), c * CHUNK_N],
                                      x_full[0])
    ordinal = 0
    for it in range(pid, num_blocks, nprog):
        slot = ordinal & 1
        phase = ordinal // 2 & 1
        if JAGGED:
            src = tl.load(SrcStarts + it).to(tl.int64)
            nvalid = tl.load(ValidCounts + it)
            rmask = lrow < nvalid
        else:
            src = it.to(tl.int64) * BLOCK_M
            nvalid = BLOCK_M
            rmask = src + lrow < M
        rows = src + lrow.to(tl.int64)
        row_blocks = it.to(tl.int64) * GROUPS_PER_TILE + tl.arange(0, GROUPS_PER_TILE).to(tl.int64)
        next_it = it + nprog
        if next_it < num_blocks:
            next_slot = slot ^ 1
            if JAGGED:
                next_src = tl.load(SrcStarts + next_it)
            else:
                next_src = next_it * BLOCK_M
            tlx.barrier_expect_bytes(x_full[next_slot], expected_bytes)
            for c in tl.static_range(NUM_CHUNKS):
                tlx.async_descriptor_load(x_desc, tlx.local_view(x_buffer, next_slot * NUM_CHUNKS + c),
                                          [next_src.to(tl.int32), c * CHUNK_N], x_full[next_slot])
        tlx.barrier_wait(x_full[slot], phase)
        raw0 = tlx.local_load(tlx.local_view(x_buffer, slot * NUM_CHUNKS), layout=load_layout)
        raw1 = tlx.local_load(tlx.local_view(x_buffer, slot * NUM_CHUNKS + 1), layout=load_layout)
        raw2 = tlx.local_load(tlx.local_view(x_buffer, slot * NUM_CHUNKS + 2), layout=load_layout)
        tl.debug_barrier()
        x0 = raw0.to(tl.float32)
        x1 = raw1.to(tl.float32)
        x2 = raw2.to(tl.float32)
        sum_sq = tl.zeros([BLOCK_M, 1], tl.float32)
        if PACKED_MATH:
            sum_sq += tl.sum(_mul_f32x2(x0, x0), axis=1, keep_dims=True)
            sum_sq += tl.sum(_mul_f32x2(x1, x1), axis=1, keep_dims=True)
            sum_sq += tl.sum(_mul_f32x2(x2, x2), axis=1, keep_dims=True)
        else:
            sum_sq += tl.sum(x0 * x0, axis=1, keep_dims=True)
            sum_sq += tl.sum(x1 * x1, axis=1, keep_dims=True)
            sum_sq += tl.sum(x2 * x2, axis=1, keep_dims=True)
        if IS_RMS:
            mean = tl.zeros([BLOCK_M, 1], tl.float32)
            rstd = libdevice.rsqrt(sum_sq / N + eps)
        else:
            sum_x = tl.zeros([BLOCK_M, 1], tl.float32)
            sum_x += tl.sum(x0, axis=1, keep_dims=True)
            sum_x += tl.sum(x1, axis=1, keep_dims=True)
            sum_x += tl.sum(x2, axis=1, keep_dims=True)
            mean = sum_x / N
            rstd = libdevice.rsqrt(sum_sq / N - mean * mean + eps)
        if RETURN_STATS:
            tl.store(Rstd + rows, tl.reshape(rstd, [BLOCK_M]), mask=rmask, cache_modifier='.cs')
            if not IS_RMS:
                tl.store(Mean + rows, tl.reshape(mean, [BLOCK_M]), mask=rmask, cache_modifier='.cs')
        chunks = (x0, x1, x2)
        for c in tl.static_range(NUM_CHUNKS):
            cols = c * CHUNK_N + ccols
            xc = chunks[c]
            if BF16_QUANTIZATION:
                if IS_RMS:
                    yc = _mul_f32x2(xc, rstd)
                    if HAS_WEIGHT:
                        yc = _mul_f32x2(yc, tl.load(W + cols).to(tl.float32))
                    yc = yc.to(tl.bfloat16).to(tl.float32)
                elif HAS_BIAS:
                    neg_mean_rstd = _sub_f32x2(tl.full([BLOCK_M, 1], 0, tl.float32), _mul_f32x2(mean, rstd))
                    normalized = _fma_f32x2(xc, rstd, neg_mean_rstd).to(tl.bfloat16)
                    weight_bf16 = tl.load(W + cols) if HAS_WEIGHT else tl.full([CHUNK_N], 1, tl.bfloat16)
                    yc = _bf16_affine(normalized, weight_bf16[None, :], tl.load(B + cols)[None, :]).to(tl.float32)
                else:
                    yc = _mul_f32x2(_sub_f32x2(xc, mean), rstd)
                    if HAS_WEIGHT:
                        yc = _mul_f32x2(yc, tl.load(W + cols).to(tl.float32))
                    yc = yc.to(tl.bfloat16).to(tl.float32)
            elif PACKED_MATH:
                if IS_RMS:
                    yc = _mul_f32x2(xc, rstd)
                else:
                    yc = _mul_f32x2(_sub_f32x2(xc, mean), rstd)
                if HAS_WEIGHT and HAS_BIAS:
                    yc = _fma_f32x2(yc, tl.load(W + cols).to(tl.float32), tl.load(B + cols).to(tl.float32))
                elif HAS_WEIGHT:
                    yc = _mul_f32x2(yc, tl.load(W + cols).to(tl.float32))
                elif HAS_BIAS:
                    if IS_RMS:
                        yc = _fma_f32x2(xc, rstd, tl.load(B + cols).to(tl.float32))
                    else:
                        yc = _fma_f32x2(_sub_f32x2(xc, mean), rstd, tl.load(B + cols).to(tl.float32))
            else:
                if IS_RMS:
                    yc = xc * rstd
                else:
                    yc = (xc - mean) * rstd
                if HAS_WEIGHT:
                    yc = yc * tl.load(W + cols).to(tl.float32)
                if HAS_BIAS:
                    yc = yc + tl.load(B + cols).to(tl.float32)
            yc = tl.where(rmask[:, None], yc, 0.0)
            if RETURN_NORM_OUT:
                tl.store(NormOut + rows[:, None] * row_stride + cols[None, :], yc.to(tl.bfloat16), mask=rmask[:, None],
                         cache_modifier='.cs')
            if RETURN_COLSUM:
                tl.atomic_add(ColsumPartials + pid * N + cols, tl.sum(yc, axis=0))
            _norm_quant_epilogue(yc, OutFp8, ScaleK, ScaleM, rows, cols, row_blocks, row_stride, scale_k_stride,
                                 scale_m_stride, n_cb_k, n_cb_m, n_row_blocks, rmask, c * CHUNK_N, VEC, GROUPS_PER_TILE,
                                 AMAX_MODE, LAYOUT, DUAL, JAGGED)
        ordinal += 1  # noqa: SIM113 - Triton does not support enumerate.


def _pick_chunk_n(n: int) -> int:
    """Largest power of two <= 256 that DIVIDES n, so chunks tile n exactly."""
    chunk_n = 256
    while n % chunk_n != 0:
        chunk_n //= 2
    return chunk_n


def _norm_quant_mxfp8_fwd_impl(x, weight, bias, eps, norm, amax_block, scale_layout, return_norm_out, return_stats,
                               return_colsum, offsets, padded_offsets, total_padded, groups_per_tile,
                               bf16_quantization=False):
    if norm not in _NORMS:
        raise ValueError(f'norm must be one of {_NORMS}, got {norm!r}')
    if amax_block not in _AMAX_MODES:
        raise ValueError(f'amax_block must be one of {tuple(_AMAX_MODES)}, got {amax_block!r}')
    if scale_layout not in _SCALE_LAYOUTS:
        raise ValueError(f'scale_layout must be one of {_SCALE_LAYOUTS}, got {scale_layout!r}')
    amax_mode = _AMAX_MODES[amax_block]
    dual = scale_layout == 'dual'
    if dual and amax_mode != _AMAX_32X32:
        raise ValueError("scale_layout='dual' requires amax_block='32x32'")
    layout_id = slu.SCALE_LAYOUT_IDS[scale_layout]
    vec = slu.MXFP8_BLOCK_SIZE
    assert x.dim() == 2 and x.shape[1] % vec == 0, f'expected 2D x with N % {vec} == 0, got {tuple(x.shape)}'
    m, n = x.shape
    device = x.device
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    jagged = offsets is not None
    if jagged:
        default_gpt = 1
    elif dual:
        pipeline_dense = n == 384 and x.dtype == torch.bfloat16 and x.is_contiguous() and (not return_colsum)
        if pipeline_dense and norm == 'layer':
            default_gpt = 2 if m >= 16384 else 1
        else:
            default_gpt = 2 if m >= 8192 else 1
    else:
        default_gpt = 2
    gpt = default_gpt if groups_per_tile is None else int(groups_per_tile)
    assert gpt in (1, 2, 4), f'groups_per_tile must be 1, 2 or 4, got {gpt}'
    block_m = vec * gpt
    chunk_n = _pick_chunk_n(n)
    num_chunks = n // chunk_n
    use_pipeline = bool(n == 384 and dual and (amax_mode == _AMAX_32X32) and (gpt in (1, 2)) and (not return_colsum)
                        and (x.dtype == torch.bfloat16) and x.is_contiguous())
    if use_pipeline:
        num_warps = 4
    else:
        num_warps = 4 if amax_mode == _AMAX_32X1 else 8
    packed_math = norm == 'rms'
    num_seqs = 0
    bsearch_iters = 1
    if jagged:
        if padded_offsets is None or total_padded is None:
            raise ValueError(
                'jagged mode needs offsets, padded_offsets and total_padded; deriving the padded block count from offsets would force a device sync'
            )
        if total_padded % vec != 0:
            raise ValueError(f'total_padded must be a multiple of {vec}')
        if total_padded % block_m != 0:
            raise ValueError(
                f"groups_per_tile={gpt} needs total_padded ({total_padded}) to be a multiple of {block_m}; every sequence's padded length must also be, or a tile would straddle two sequences"
            )
        num_seqs = offsets.numel() - 1
        num_blocks = total_padded // block_m
        n_row_blocks = total_padded // vec
        bsearch_iters = max(1, (num_seqs - 1).bit_length())
    else:
        offsets = x
        padded_offsets = x
        num_blocks = triton.cdiv(m, block_m)
        n_row_blocks = triton.cdiv(m, vec)
    scale_cols_k = slu.num_scale_cols(n)
    n_cb_k = slu.num_col_blocks(scale_cols_k)
    n_cb_m = slu.num_col_blocks(n_row_blocks)

    def _alloc(rows, scale_cols, layout, fully_covered=False):
        fill = None if fully_covered else slu.E8M0_PAD_SCALE
        return slu.alloc_scales(layout, rows, scale_cols, device, fill=fill)

    ph_u8 = torch.empty(1, dtype=torch.uint8, device=device)
    if dual:
        k_fully_covered = not jagged and m % 128 == 0 and (scale_cols_k % 4 == 0)
        m_fully_covered = not jagged and n % 128 == 0 and (n_row_blocks % 4 == 0)
        partial_fill = bool(use_pipeline and jagged and (n % 128 == 0) and (scale_cols_k % 4 == 0)
                            and (n_row_blocks % 4 == 0))
        scale_k = _alloc(m, scale_cols_k, slu.SCALE_LAYOUT_MMA, k_fully_covered or partial_fill)
        if partial_fill and m % 128 != 0:
            tail_bytes = n_cb_k * slu.SF_BLOCK_BYTES
            scale_k[-tail_bytes:].fill_(slu.E8M0_PAD_SCALE)
        scale_m = _alloc(n, n_row_blocks, slu.SCALE_LAYOUT_CUBLAS, m_fully_covered or partial_fill)
    elif amax_mode == _AMAX_32X1:
        scale_k = ph_u8
        scale_m = _alloc(n, n_row_blocks, layout_id)
    else:
        scale_k = _alloc(m, scale_cols_k, layout_id)
        scale_m = ph_u8
    out_fp8 = torch.empty(m, n, dtype=torch.float8_e4m3fn, device=device)

    def _slot(cond, shape, dtype):
        """Real tensor when the output is on, else a 1-element backing store."""
        buf = torch.empty(shape if cond else (1, ), dtype=dtype, device=device)
        return (buf, buf if cond else buf[:0])

    norm_out_buf, norm_out = _slot(return_norm_out, (m, n), torch.bfloat16)
    mean_buf, mean = _slot(return_stats and norm == 'layer', (m, 1), torch.float32)
    rstd_buf, rstd = _slot(return_stats, (m, 1), torch.float32)
    ring_bytes = 2 * block_m * chunk_n * num_chunks * x.element_size()
    ctas_per_sm = max(1, min(4, 227 * 1024 // ring_bytes))
    if use_pipeline:
        factor = 2
        ctas_per_sm = max(1, min(factor, 227 * 1024 // ring_bytes))
        num_persist = max(1, min(num_blocks, sms * ctas_per_sm))
    elif dual and (not jagged):
        num_persist = max(1, min(num_blocks, sms))
        while num_blocks % num_persist != 0:
            num_persist -= 1
    else:
        num_persist = max(1, min(num_blocks, sms * ctas_per_sm))
    if return_colsum:
        colsum_partials = torch.zeros(num_persist, n, dtype=torch.float32, device=device)
    else:
        colsum_partials = torch.empty(1, dtype=torch.float32, device=device)
    has_weight = weight is not None
    has_bias = bias is not None
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device=device))
    if num_blocks > 0:
        with torch.cuda.device(x.device):
            if jagged:
                src_starts = torch.empty(num_blocks, dtype=offsets.dtype, device=device)
                valid_counts = torch.empty(num_blocks, dtype=torch.int32, device=device)
                meta_block = 256
                _kernel_build_jagged_tile_metadata[
                    triton.cdiv(num_blocks, meta_block),
                ](offsets, padded_offsets, src_starts, valid_counts, num_blocks, num_seqs, BLOCK_M=block_m,
                  ITERS=bsearch_iters, BLOCK=meta_block)
            else:
                src_starts = x
                valid_counts = x
            kernel = kernel_norm_quant_mxfp8_tma_pipeline if use_pipeline else kernel_norm_quant_mxfp8_ws
            pipeline_kwargs = {'PACKED_MATH': packed_math, 'BF16_QUANTIZATION': bf16_quantization
                               } if use_pipeline else {}
            kernel[
                num_persist,
            ](x, weight if has_weight else x, bias if has_bias else x, norm_out_buf, mean_buf,
              rstd_buf, out_fp8, scale_k, scale_m, colsum_partials, src_starts, valid_counts, x.stride(0),
              scale_k.stride(0) if scale_k.dim() == 2 else 0, scale_m.stride(0) if scale_m.dim() == 2 else 0, n_cb_k,
              n_cb_m, m, n_row_blocks, num_blocks, eps, N=n, BLOCK_M=block_m, VEC=vec, GROUPS_PER_TILE=gpt,
              CHUNK_N=chunk_n, NUM_CHUNKS=num_chunks, NB=2, IS_RMS=norm == 'rms', HAS_WEIGHT=has_weight,
              HAS_BIAS=has_bias, AMAX_MODE=amax_mode, LAYOUT=layout_id if not dual else slu.SCALE_LAYOUT_MMA, DUAL=dual,
              RETURN_NORM_OUT=return_norm_out, RETURN_STATS=return_stats, RETURN_COLSUM=return_colsum, JAGGED=jagged,
              **pipeline_kwargs, num_warps=num_warps)
    if return_colsum:
        colsum = colsum_partials.sum(0)
    else:
        colsum = colsum_partials[:0]
    scale = scale_m if amax_mode == _AMAX_32X1 else scale_k
    scale_t = scale_m if dual else ph_u8[:0]
    return (norm_out, mean, rstd, out_fp8, scale, scale_t, colsum)


def padded_cu_seqlens(offsets: torch.Tensor, alignment: int = 32) -> torch.Tensor:
    """``alignment``-aligned padded cu_seqlens for the jagged path."""
    seqlens = offsets[1:].to(torch.int64) - offsets[:-1].to(torch.int64)
    padded = (seqlens + alignment - 1) // alignment * alignment
    out = torch.zeros_like(offsets)
    out[1:] = padded.cumsum(0)
    return out


def norm_quant_mxfp8_fwd(
    x: torch.Tensor, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None, eps: float = 1e-05,
    norm: str = 'layer', amax_block: str = '1x32', scale_layout: str = 'rowmajor', return_norm_out: bool = True,
    return_stats: bool = True, return_colsum: bool = False, offsets: torch.Tensor | None = None,
    padded_offsets: torch.Tensor | None = None, total_padded: int | None = None, groups_per_tile: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RMS/Layer norm + MXFP8 quantize (warp-specialized TLX)."""
    return _norm_quant_mxfp8_fwd_impl(x, weight, bias, eps, norm, amax_block, scale_layout, return_norm_out,
                                      return_stats, return_colsum, offsets, padded_offsets, total_padded,
                                      groups_per_tile)


def _backward_module():
    from . import _mxfp8_bwd
    return _mxfp8_bwd


@triton.jit
def _e8m0_to_f32(s):
    bits = s.to(tl.uint32) << 23
    bits = tl.where(s == 0, 4194304, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _emit_quant(y, Q, K, T, Row, Sfv, Represented, rows, padded_rows, valid, tile_id, cols, TOTAL_PADDED, NK, NT,
                N: tl.constexpr, BC: tl.constexpr, HEAD_DIM: tl.constexpr, RCP: tl.constexpr, EMIT_K: tl.constexpr,
                HEAD_MAJOR: tl.constexpr, K_PADDED: tl.constexpr, EMIT_ROW: tl.constexpr, EMIT_SFV: tl.constexpr,
                EMIT_REPRESENTED: tl.constexpr, EMPTY_T_ONE: tl.constexpr, ZERO_SCALE_ONE: tl.constexpr = False):
    CG: tl.constexpr = BC // 32
    groups = tl.min(cols, axis=0) // 32 + tl.arange(0, CG)
    yr = tl.reshape(y, [32, CG, 32])
    amax = tl.max(tl.max(tl.abs(yr), axis=2), axis=0)
    exponent, inverse = _fused_amax_to_e8m0(amax, RCP)
    scale = exponent.to(tl.uint8)
    if ZERO_SCALE_ONE:
        scale = tl.minimum(tl.maximum(exponent, 1), 254).to(tl.uint8)
        inverse = 1.0 / _e8m0_to_f32(scale)
    scaled = yr * inverse[None, :, None]
    q = tl.reshape(_cvt_e4m3x4_f32(scaled), [32, BC])
    mask = valid[:, None] & (cols < N)[None, :]
    tl.store(Q + rows[:, None] * N + cols[None, :], q, mask=mask)
    if EMIT_REPRESENTED:
        reconstructed = tl.reshape(q.to(tl.float32), [32, CG, 32]) * _e8m0_to_f32(scale)[None, :, None]
        tl.store(Represented + rows[:, None] * N + cols[None, :], tl.reshape(reconstructed, [32, BC]), mask=mask)
    replicated = tl.broadcast_to(scale[None, :], [32, CG])
    gm = groups < N // 32
    if EMIT_ROW:
        tl.store(Row + rows[:, None] * (N // 32) + groups[None, :], replicated, mask=valid[:, None] & gm[None, :])
    if EMIT_K:
        if HEAD_MAJOR:
            head = groups // (HEAD_DIM // 32)
            local = groups % (HEAD_DIM // 32)
            kr = head[None, :] * TOTAL_PADDED + padded_rows[:, None]
            slu.store_mma_scales(K, kr, local[None, :], tl.where(valid[:, None], replicated, 0), NK,
                                 gm[None, :] & (padded_rows < TOTAL_PADDED)[:, None])
        else:
            kr = padded_rows if K_PADDED else rows
            kv = tl.where(tl.sum(valid.to(tl.int32), axis=0) > 0, replicated, 0)
            km = (padded_rows < TOTAL_PADDED)[:, None] if K_PADDED else valid[:, None]
            slu.store_mma_scales(K, kr[:, None], groups[None, :], kv, NK, km & gm[None, :])
    if EMIT_SFV:
        head = groups // (HEAD_DIM // 32)
        local = groups % (HEAD_DIM // 32)
        sr = head[None, :] * TOTAL_PADDED + padded_rows[:, None]
        slu.store_mma_scales(Sfv, sr, local[None, :], tl.where(valid[:, None], replicated, 1),
                             tl.cdiv(HEAD_DIM // 32, 4), gm[None, :] & (padded_rows < TOTAL_PADDED)[:, None])
    expanded = tl.reshape(tl.broadcast_to(scale[:, None], [CG, 32]), [BC])
    empty = 1 if EMPTY_T_ONE else 0
    tv = tl.where(tl.sum(valid.to(tl.int32), axis=0) > 0, expanded, empty)
    slu.store_cublas_blocked_scales(T, cols, tile_id, tv, NT, cols < N)


@triton.jit
def _fused_forward(X, W, B, Y, Mean, Rstd, Q, K, T, Row, Sfv, Represented, GlobalAmax, ColPart, Starts, Counts, M,
                   TILES, TOTAL_PADDED, NK, NT, eps, N: tl.constexpr, BN: tl.constexpr, HEAD_DIM: tl.constexpr,
                   IS_RMS: tl.constexpr, HAS_BIAS: tl.constexpr, CENTERED: tl.constexpr, SKIP_NORM: tl.constexpr,
                   JAGGED: tl.constexpr, SAVE_Y: tl.constexpr, SAVE_STATS: tl.constexpr, SAVE_AMAX: tl.constexpr,
                   SAVE_COLSUM: tl.constexpr, RCP: tl.constexpr, EMIT_K: tl.constexpr, HEAD_MAJOR: tl.constexpr,
                   K_PADDED: tl.constexpr, EMIT_ROW: tl.constexpr, EMIT_SFV: tl.constexpr,
                   EMIT_REPRESENTED: tl.constexpr, SINGLE_ITER: tl.constexpr = False):
    pid = tl.program_id(0)
    cols = tl.arange(0, BN)
    local_rows = tl.arange(0, 32)
    w = tl.load(W + cols, mask=cols < N, other=0).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=cols < N, other=0)
    for iteration in range(1 if SINGLE_ITER else tl.cdiv(TILES - pid, tl.num_programs(0))):
        tile = pid + iteration * tl.num_programs(0)
        padded_rows = tile * 32 + local_rows
        if JAGGED:
            src = tl.load(Starts + tile)
            count = tl.load(Counts + tile)
            valid = local_rows < count
        else:
            src = tile * 32
            valid = padded_rows < M
        rows = src.to(tl.int64) + local_rows
        mask = valid[:, None] & (cols < N)[None, :]
        x = _stage_load(X, rows[:, None] * N + cols[None, :], mask, 32, BN).to(tl.float32)
        if SKIP_NORM:
            y = x
        else:
            if IS_RMS:
                mu = tl.zeros([32], tl.float32)
                var = tl.sum(x * x, axis=1) / N
            else:
                mu = tl.sum(x, axis=1) / N
                if CENTERED:
                    xc = tl.where((cols < N)[None, :], x - mu[:, None], 0.0)
                    var = tl.sum(xc * xc, axis=1) / N
                else:
                    var = tl.sum(x * x, axis=1) / N - mu * mu
            rr = libdevice.rsqrt(var + eps)
            if not IS_RMS and HAS_BIAS and (not CENTERED):
                normalized = tl.fma(x, rr[:, None], (-mu * rr)[:, None]).to(tl.bfloat16)
                y = _bf16_affine(normalized, w[None, :].to(tl.bfloat16), b[None, :]).to(tl.float32)
            else:
                y = (x - mu[:, None]) * rr[:, None] * w[None, :]
                if HAS_BIAS:
                    y = y + b[None, :].to(tl.float32)
                y = y.to(tl.bfloat16).to(tl.float32)
            if SAVE_STATS:
                tl.store(Rstd + rows, rr, mask=valid)
                if not IS_RMS:
                    tl.store(Mean + rows, mu, mask=valid)
        y = tl.where(mask, y, 0.0)
        if SAVE_Y:
            tl.store(Y + rows[:, None] * N + cols[None, :], y, mask=mask)
        if SAVE_COLSUM:
            tl.store(ColPart + tile.to(tl.int64) * N + cols, tl.sum(y, axis=0), mask=cols < N)
        if SAVE_AMAX:
            tl.atomic_max(GlobalAmax, tl.max(tl.max(tl.abs(y), axis=1), axis=0))
        _emit_quant(y, Q, K, T, Row, Sfv, Represented, rows, padded_rows, valid, tile, cols, TOTAL_PADDED, NK, NT, N,
                    BN, HEAD_DIM, RCP, EMIT_K, HEAD_MAJOR, K_PADDED, EMIT_ROW, EMIT_SFV, EMIT_REPRESENTED, CENTERED)


@triton.jit
def _row_forward(X, W, B, Y, Mean, Rstd, M, eps, N: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr,
                 IS_RMS: tl.constexpr, HAS_BIAS: tl.constexpr, CENTERED: tl.constexpr):
    cols = tl.arange(0, BN)
    for row0 in range(tl.program_id(0) * BM, M, tl.num_programs(0) * BM):
        rows = row0 + tl.arange(0, BM)
        mask = (rows < M)[:, None] & (cols < N)[None, :]
        x = _stage_load(X, rows[:, None].to(tl.int64) * N + cols[None, :], mask, BM, BN).to(tl.float32)
        w = tl.load(W + cols, mask=cols < N, other=0).to(tl.float32)
        mu = tl.zeros([BM], tl.float32)
        if not IS_RMS:
            mu = tl.sum(x, axis=1) / N
        if CENTERED and (not IS_RMS):
            z = tl.where((cols < N)[None, :], x - mu[:, None], 0.0)
            var = tl.sum(z * z, axis=1) / N
        else:
            var = tl.sum(x * x, axis=1) / N - mu * mu
        rr = libdevice.rsqrt(var + eps)
        if HAS_BIAS and (not IS_RMS) and (not CENTERED):
            z = tl.fma(x, rr[:, None], (-mu * rr)[:, None]).to(tl.bfloat16)
            y = _bf16_affine(z, w[None, :].to(tl.bfloat16), tl.load(B + cols, mask=cols < N, other=0)[None, :])
        else:
            y = (x - mu[:, None]) * rr[:, None] * w[None, :]
            if HAS_BIAS:
                y += tl.load(B + cols, mask=cols < N, other=0).to(tl.float32)[None, :]
        tl.store(Y + rows[:, None].to(tl.int64) * N + cols[None, :], y, mask=mask)
        tl.store(Rstd + rows, rr, mask=rows < M)
        if not IS_RMS:
            tl.store(Mean + rows, mu, mask=rows < M)


@triton.jit
def _quant_wide(Y, Q, K, T, Row, Sfv, Represented, Starts, Counts, M, TILES, TOTAL_PADDED, NK, NT, N: tl.constexpr,
                HEAD_DIM: tl.constexpr, JAGGED: tl.constexpr, RCP: tl.constexpr, EMIT_K: tl.constexpr,
                HEAD_MAJOR: tl.constexpr, K_PADDED: tl.constexpr, EMIT_ROW: tl.constexpr, EMIT_SFV: tl.constexpr,
                EMIT_REPRESENTED: tl.constexpr, CENTERED: tl.constexpr):
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    lr = tl.arange(0, 32)
    for tile in range(tl.program_id(0), TILES, tl.num_programs(0)):
        pr = tile * 32 + lr
        if JAGGED:
            src = tl.load(Starts + tile)
            valid = lr < tl.load(Counts + tile)
        else:
            src = tile * 32
            valid = pr < M
        rows = src.to(tl.int64) + lr
        mask = valid[:, None] & (cols < N)[None, :]
        y = _stage_load(Y, rows[:, None] * N + cols[None, :], mask, 32, 128).to(tl.float32)
        _emit_quant(y, Q, K, T, Row, Sfv, Represented, rows, pr, valid, tile, cols, TOTAL_PADDED, NK, NT, N, 128,
                    HEAD_DIM, RCP, EMIT_K, HEAD_MAJOR, K_PADDED, EMIT_ROW, EMIT_SFV, EMIT_REPRESENTED, CENTERED)


def build_padded_cta_metadata(offsets, padded_offsets, total_padded):
    tiles = total_padded // 32
    starts = torch.empty(tiles, device=offsets.device, dtype=offsets.dtype)
    counts = torch.empty_like(starts, dtype=torch.int32)
    if tiles:
        with torch.cuda.device(offsets.device):
            nseq = offsets.numel() - 1
            _kernel_build_jagged_tile_metadata[
                triton.cdiv(tiles, 256),
            ](offsets, padded_offsets, starts, counts, tiles, nseq, BLOCK_M=32,
              ITERS=max(1,
                        math.ceil(math.log2(max(nseq, 1))) + 1), BLOCK=256)
    return (starts, counts)


def _core(x, weight, bias, n_heads, head_dim, *, norm, eps=1e-05, offsets=None, padded_offsets=None, total_padded=None,
          save_stats=False, save_y=False, represented=False, colsum=False, centered=False, rcp=1 / 448, emit_k=True,
          head_major=False, k_padded=False, row_scales=False, sfv=False, skip_norm=False, precomputed_src_starts=None,
          precomputed_valid_counts=None):
    m, n = x.shape
    fused = n <= 256 or (n <= 512 and m >= 4096)
    jagged = offsets is not None
    if jagged and total_padded is None:
        total_padded = int(padded_offsets[-1].item())
    mp = total_padded if jagged else triton.cdiv(m, 32) * 32
    tiles = mp // 32
    if jagged:
        if precomputed_src_starts is None:
            starts, counts = build_padded_cta_metadata(offsets, padded_offsets, mp)
        else:
            starts, counts = (precomputed_src_starts, precomputed_valid_counts)
    else:
        starts = counts = x
    empty = lambda dtype: torch.empty(0, device=x.device, dtype=dtype)
    k_rows = n_heads * mp if head_major else mp if k_padded else m
    k_cols = head_dim // 32 if head_major else n // 32
    k = torch.empty(
        (slu.mma_scale_numel(k_rows, k_cols), ), device=x.device, dtype=torch.uint8) if emit_k else empty(torch.uint8)
    if emit_k and (jagged or m % 128 or (head_dim if head_major else n) % 128):
        k.fill_(1)
    trans = torch.empty(slu.mblock_scale_shape(mp if jagged else m, n), device=x.device, dtype=torch.uint8)
    if jagged or m % 128 or n % 128:
        trans.fill_(1)
    row = torch.empty((m, n // 32), device=x.device, dtype=torch.uint8) if row_scales else empty(torch.uint8)
    sfv_out = torch.full(slu.cublas_blocked_scale_shape(n_heads * mp, head_dim // 32), 1, device=x.device,
                         dtype=torch.uint8) if sfv else empty(torch.uint8)
    y = torch.empty_like(x, dtype=torch.bfloat16) if save_y or not fused else empty(torch.bfloat16)
    rstd = torch.empty(m, device=x.device, dtype=torch.float32) if save_stats or not fused else empty(torch.float32)
    mean = torch.empty_like(rstd) if norm == 'layer' else empty(torch.float32)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    rep = torch.empty_like(x, dtype=torch.bfloat16) if represented else empty(torch.bfloat16)
    amax = torch.zeros(1, device=x.device, dtype=torch.float32) if centered else empty(torch.float32)
    parts = torch.empty((tiles, n), device=x.device, dtype=torch.float32) if colsum else empty(torch.float32)
    nk, nt = (triton.cdiv(k_cols, 4), triton.cdiv(triton.cdiv(mp if jagged else m, 32), 4))
    if tiles and m:
        with torch.cuda.device(x.device):
            sms = torch.cuda.get_device_properties(x.device).multi_processor_count
            grid = min(tiles, sms * 2)
            quant_kw = {
                'N': n, 'HEAD_DIM': head_dim, 'JAGGED': jagged, 'RCP': rcp, 'EMIT_K': emit_k, 'HEAD_MAJOR': head_major,
                'K_PADDED': k_padded, 'EMIT_ROW': row_scales, 'EMIT_SFV': sfv, 'EMIT_REPRESENTED': represented,
                'CENTERED': centered, 'num_warps': 8 if fused and n == 512 else 4, 'enable_fp_fusion': False
            }
            if fused:
                _fused_forward[
                    grid,
                ](x, weight, bias if bias is not None else x, y, mean, rstd, q, k, trans, row, sfv_out, rep, amax,
                  parts, starts, counts, m, tiles, mp, nk, nt, eps, BN=triton.next_power_of_2(n), IS_RMS=norm == 'rms',
                  HAS_BIAS=bias is not None, SKIP_NORM=skip_norm, SAVE_Y=save_y, SAVE_STATS=save_stats,
                  SAVE_AMAX=centered, SAVE_COLSUM=colsum, SINGLE_ITER=grid == tiles, **quant_kw)
            else:
                if skip_norm:
                    y = x
                else:
                    bn = triton.next_power_of_2(n)
                    bm = max(1, min(4, 16384 // bn))
                    _row_forward[
                        min(triton.cdiv(m, bm), sms * 2),
                    ](x, weight, bias if bias is not None else x, y, mean, rstd, m, eps, N=n, BN=bn, BM=bm,
                      IS_RMS=norm == 'rms', HAS_BIAS=bias is not None, CENTERED=centered, num_warps=4,
                      enable_fp_fusion=False)
                _quant_wide[grid, triton.cdiv(n, 128)](y, q, k, trans, row, sfv_out, rep, starts, counts, m, tiles, mp,
                                                       nk, nt, **quant_kw)
                if centered:
                    amax = y.float().abs().amax().reshape(1)
    if colsum:
        sums = parts.sum(0) if fused and m else y.float().sum(0) if m else torch.zeros(
            n, device=x.device, dtype=torch.float32)
    else:
        sums = empty(torch.float32)
    return {
        'y': y, 'mean': mean, 'rstd': rstd, 'q': q, 'k': k, 't': trans, 'row': row, 'sfv': sfv_out, 'represented': rep,
        'amax': amax, 'colsum': sums
    }


def _generic(x, weight, bias, n_heads, head_dim, *, norm, eps=1e-05, save_rstd=False, offsets=None, padded_offsets=None,
             total_padded=None, return_colsum=False, return_normalized=False, return_represented_v=False,
             max_norm_rcp=None, cutlass_head_major_scales=False, emit_kblock_scales=True, precomputed_src_starts=None,
             precomputed_valid_counts=None, nvfp4_global_scale=None, nvfp4_padded_scale_output=False,
             return_nvfp4_cta_scale_max=False, score_fragments=None, score_const=None, n_scores=0):
    extended = nvfp4_global_scale is not None or n_scores or return_nvfp4_cta_scale_max
    compatible = (not extended and x.shape[1] == 384 and (x.dtype == torch.bfloat16) and (not return_colsum) and
                  (not return_represented_v) and (max_norm_rcp is None) and
                  (not cutlass_head_major_scales) and emit_kblock_scales and (precomputed_src_starts is None) and
                  (x.shape[0] > 0)) and x.is_contiguous()
    if compatible:
        y, mean, rr, q, k, t, _ = _norm_quant_mxfp8_fwd_impl(x, weight, bias, eps, norm, '32x32', 'dual',
                                                             return_normalized, save_rstd, False, offsets,
                                                             padded_offsets, total_padded,
                                                             2 if offsets is not None else None, True)
        values = {
            'y': y, 'mean': mean.flatten(), 'rstd': rr.flatten(), 'q': q, 'k': k, 't': t, 'colsum': None, 'represented':
            None
        }
    else:
        values = _core(x, weight, bias, n_heads, head_dim, norm=norm, eps=eps, save_stats=save_rstd,
                       save_y=return_normalized or bool(extended), offsets=offsets, padded_offsets=padded_offsets,
                       total_padded=total_padded, colsum=return_colsum, represented=return_represented_v,
                       rcp=1 / 448 if max_norm_rcp is None else max_norm_rcp, head_major=cutlass_head_major_scales,
                       emit_k=emit_kblock_scales, precomputed_src_starts=precomputed_src_starts,
                       precomputed_valid_counts=precomputed_valid_counts)
    out = (values['q'], values['k'], values['t'].view(torch.float8_e8m0fnu))
    if save_rstd:
        out += (values['rstd'], ) if norm == 'rms' else (values['rstd'], values['mean'])
    if return_normalized:
        out = (values['y'], *out)
    if return_colsum:
        out += (values['colsum'], )
    if return_represented_v:
        out += (values['represented'], )
    if nvfp4_global_scale is not None:
        extra = _nvfp4_emission(values['y'], nvfp4_global_scale, offsets=offsets, padded_offsets=padded_offsets,
                                total_padded=total_padded, padded_scale_output=nvfp4_padded_scale_output,
                                return_cta_scale_max=return_nvfp4_cta_scale_max)
        out += tuple(extra)
    if n_scores:
        scores = torch.empty((n_scores, x.shape[0]), device=x.device, dtype=torch.float32)
        if x.shape[0]:
            with torch.cuda.device(x.device):
                _row_scores[
                    triton.cdiv(x.shape[0], 4),
                ](values['y'], score_fragments, score_const, scores, x.shape[0], N=384, BN=512, NS=n_scores,
                  num_warps=4, enable_fp_fusion=False)
        out += (scores, )
    return out


def rmsnorm_mxfp8_quantize(x, weight, n_heads, head_dim, eps=1e-05, save_rstd=False, offsets=None, padded_offsets=None,
                           total_padded=None, return_colsum=False, return_represented_v=False):
    return _generic(x, weight, None, n_heads, head_dim, norm='rms', eps=eps, save_rstd=save_rstd, offsets=offsets,
                    padded_offsets=padded_offsets, total_padded=total_padded, return_colsum=return_colsum,
                    return_represented_v=return_represented_v)


def layernorm_mxfp8_quantize(x, weight, n_heads, head_dim, eps=1e-05, save_rstd=False, offsets=None,
                             padded_offsets=None, total_padded=None, bias=None, max_norm_rcp=None,
                             split_scale_stores=None, return_colsum=False, cutlass_head_major_scales=False,
                             initialize_scale_storage=True, return_normalized=False, return_represented_v=False,
                             precomputed_src_starts=None, precomputed_valid_counts=None, nvfp4_global_scale=None,
                             emit_kblock_scales=True, nvfp4_padded_scale_output=False, return_nvfp4_cta_scale_max=False,
                             score_fragments=None, score_const=None, n_scores=0):
    return _generic(x, weight, bias, n_heads, head_dim, norm='layer', eps=eps, save_rstd=save_rstd, offsets=offsets,
                    padded_offsets=padded_offsets, total_padded=total_padded, return_colsum=return_colsum,
                    return_normalized=return_normalized, return_represented_v=return_represented_v,
                    max_norm_rcp=max_norm_rcp, cutlass_head_major_scales=cutlass_head_major_scales,
                    emit_kblock_scales=emit_kblock_scales, precomputed_src_starts=precomputed_src_starts,
                    precomputed_valid_counts=precomputed_valid_counts, nvfp4_global_scale=nvfp4_global_scale,
                    nvfp4_padded_scale_output=nvfp4_padded_scale_output,
                    return_nvfp4_cta_scale_max=return_nvfp4_cta_scale_max, score_fragments=score_fragments,
                    score_const=score_const, n_scores=n_scores)


def layernorm_mxfp8_quantize_numerics_preserving(x, weight, n_heads, head_dim, offsets, padded_offsets, total_padded,
                                                 eps=1e-05, bias=None, max_norm_rcp=1 / 256,
                                                 prepare_forward_scales=True, write_represented_v=False,
                                                 split_scale_stores=False, return_colsum=False):
    v = _core(x, weight, bias, n_heads, head_dim, norm='layer', eps=eps, offsets=offsets, padded_offsets=padded_offsets,
              total_padded=total_padded, save_stats=True, centered=True, rcp=max_norm_rcp,
              emit_k=prepare_forward_scales, k_padded=True, row_scales=True, sfv=write_represented_v,
              represented=write_represented_v, colsum=return_colsum)
    bundle = torch.cat((v['t'].flatten(), v['row'].flatten()))
    empty = torch.empty(0, dtype=torch.uint8, device=x.device)
    rep = v['represented'].view(x.shape[0], n_heads, head_dim) if write_represented_v else v['represented']
    out = (v['q'], empty, bundle, v['k'], v['rstd'], v['mean'], v['amax'], rep, v['sfv'])
    return (*out, v['colsum']) if return_colsum else out


@triton.jit
def _row_scores(Y, Fragments, Const, Scores, M, N: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    cols = tl.arange(0, BN)
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    y = _stage_load(Y, rows[:, None].to(tl.int64) * N + cols[None, :], mask, 4, BN).to(tl.float32)
    for score in tl.static_range(NS):
        lane = score * 4 + cols % 8 // 2
        base = (cols // 16 * 2 * 32 + lane) * 2 + cols % 16 // 8
        hi_word = tl.load(Fragments + base, mask=cols < N, other=0).to(tl.uint32)
        lo_word = tl.load(Fragments + base + 64, mask=cols < N, other=0).to(tl.uint32)
        shift = cols % 2 * 16
        hi = ((hi_word >> shift & 65535) << 16).to(tl.float32, bitcast=True)
        lo = ((lo_word >> shift & 65535) << 16).to(tl.float32, bitcast=True)
        value = tl.sum(y * hi[None, :], axis=1) + tl.sum(y * lo[None, :], axis=1) + tl.load(Const + score)
        tl.store(Scores + score * M + rows, value, mask=rows < M)


@triton.jit
def _nvfp4_scale_relayout(Logical, Starts, Counts, Scales, Maxima, TOTAL_PADDED, REST_K, NG: tl.constexpr,
                          BG: tl.constexpr, WRITE_SCALES: tl.constexpr, WRITE_MAX: tl.constexpr):
    tile = tl.program_id(0)
    lr = tl.arange(0, 32)
    groups = tl.arange(0, BG)
    src = tl.load(Starts + tile)
    count = tl.load(Counts + tile)
    valid = lr < count
    s = tl.load(Logical + (src.to(tl.int64) + lr[:, None]) * NG + groups[None, :],
                mask=valid[:, None] & (groups < NG)[None, :], other=0).to(tl.uint32)
    if WRITE_SCALES:
        slu.store_mma_scales(Scales, (tile * 32 + lr)[:, None], groups[None, :], s.to(tl.uint8), REST_K,
                             valid[:, None] & (groups < NG)[None, :])
    if WRITE_MAX:
        maximum = tl.max(tl.max(s, axis=1), axis=0)
        maximum = tl.maximum(maximum, tl.where(count < 32, 56, 0))
        tl.store(Maxima + tile, maximum.to(tl.uint8))


def _nvfp4_emission(y, global_scale, *, offsets, padded_offsets, total_padded, padded_scale_output=False,
                    return_cta_scale_max=False):
    from . import _fp4 as fp4
    get_mma_scale_size = fp4.get_mma_scale_size
    nvfp4_quantize_normalized = fp4.nvfp4_quantize_normalized
    packed, scales, logical = nvfp4_quantize_normalized(y, global_scale, return_logical_scales=True)
    if not padded_scale_output and (not return_cta_scale_max):
        return (packed, scales)
    if total_padded is None:
        total_padded = int(padded_offsets[-1].item())
    starts, counts = build_padded_cta_metadata(offsets, padded_offsets, total_padded)
    out_scales = torch.full((get_mma_scale_size(total_padded, y.shape[1]), ), 56, device=y.device,
                            dtype=torch.uint8) if padded_scale_output else scales
    maxima = torch.empty(total_padded //
                         32, device=y.device, dtype=torch.uint8) if return_cta_scale_max else torch.empty(
                             0, device=y.device, dtype=torch.uint8)
    if total_padded:
        with torch.cuda.device(y.device):
            _nvfp4_scale_relayout[
                total_padded // 32,
            ](logical, starts, counts, out_scales, maxima, total_padded, triton.cdiv(y.shape[1] // 16, 4),
              NG=y.shape[1] // 16, BG=triton.next_power_of_2(y.shape[1] // 16), WRITE_SCALES=padded_scale_output,
              WRITE_MAX=return_cta_scale_max, num_warps=4)
    out = (packed, out_scales)
    return (*out, maxima) if return_cta_scale_max else out


class _LayernormMXFP8(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, n_heads, head_dim, offsets, padded_offsets, total_padded, starts, counts, gdpa):
        out = layernorm_mxfp8_quantize(x, weight, n_heads, head_dim, save_rstd=True, offsets=offsets,
                                       padded_offsets=padded_offsets, total_padded=total_padded, bias=bias,
                                       return_normalized=gdpa, cutlass_head_major_scales=gdpa,
                                       precomputed_src_starts=starts, precomputed_valid_counts=counts)
        if gdpa:
            routed, q, k, trans, rr, mu = out
        else:
            q, k, trans, rr, mu = out
            routed = x.clone()
        trans = trans.view(torch.uint8)
        if not gdpa:
            trans = trans.flatten()
        ctx.save_for_backward(x, weight, rr, mu,
                              bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype))
        ctx.has_bias = bias is not None
        ctx.mark_non_differentiable(q, k, trans)
        return (routed, q, k, trans)

    @staticmethod
    def backward(ctx, grad, _dq, _dk, _dt):
        x, w, rr, mu, bias = ctx.saved_tensors
        dx, dw, db = _backward_module().layernorm_backward(grad.contiguous(), x, w, rr, mu)
        return (dx, dw.to(w.dtype), db.to(bias.dtype) if ctx.has_bias else None, None, None, None, None, None, None,
                None, None)


def layernorm_mxfp8_with_backward(x, weight, bias, n_heads, head_dim):
    return _LayernormMXFP8.apply(x, weight, bias, n_heads, head_dim, None, None, None, None, None, False)


def layernorm_mxfp8_gdpa_with_backward(x, weight, bias, n_heads, head_dim, offsets, padded_offsets, cpu_offsets,
                                       total_padded, precomputed_src_starts=None, precomputed_valid_counts=None):
    return _LayernormMXFP8.apply(x, weight, bias, n_heads, head_dim, offsets, padded_offsets, total_padded,
                                 precomputed_src_starts, precomputed_valid_counts, True)


def layernorm_mxfp8_gdpa_precomputed_no_normalized(x, weight, bias, n_heads, head_dim, offsets, padded_offsets,
                                                   cpu_offsets, total_padded, precomputed_src_starts,
                                                   precomputed_valid_counts):
    q, k, trans, rstd, mean = layernorm_mxfp8_quantize(x, weight, n_heads, head_dim, save_rstd=True, offsets=offsets,
                                                       padded_offsets=padded_offsets, total_padded=total_padded,
                                                       bias=bias, cutlass_head_major_scales=True,
                                                       precomputed_src_starts=precomputed_src_starts,
                                                       precomputed_valid_counts=precomputed_valid_counts)
    return (q, k, trans.view(torch.uint8), rstd, mean)


@triton.jit
def _two_pass_quant(Y, Q, K, T, Starts, Counts, TILES, TOTAL_PADDED, NK, NT, N: tl.constexpr, HEAD_DIM: tl.constexpr,
                    JAGGED_OUTPUT: tl.constexpr):
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    lr = tl.arange(0, 32)
    for tile in range(tl.program_id(0), TILES, tl.num_programs(0)):
        src = tl.load(Starts + tile)
        valid = lr < tl.load(Counts + tile)
        pr = tile * 32 + lr
        rows = src.to(tl.int64) + lr
        y = _stage_load(Y, rows[:, None] * N + cols[None, :], valid[:, None] & (cols < N)[None, :], 32,
                        128).to(tl.float32)
        out_rows = rows if JAGGED_OUTPUT else pr.to(tl.int64)
        out_valid = valid if JAGGED_OUTPUT else pr < TOTAL_PADDED
        _emit_quant(y, Q, K, T, Y, Y, Y, out_rows, pr, out_valid, tile, cols, TOTAL_PADDED, NK, NT, N, 128, HEAD_DIM,
                    1.0 / 448.0, True, False, False, False, False, False, True, ZERO_SCALE_ONE=True)


def rmsnorm_mxfp8_quantize_3232_2pass(x, weight, cu_seqlens, padded_cu_seqlens, n_heads, head_dim, eps=1e-05,
                                      total_padded=None, save_rstd=False, jagged_output=False):
    if total_padded is None:
        total_padded = int(padded_cu_seqlens[-1].item())
    m, n = x.shape
    y, _, _, rr = norm_fwd(x, weight, norm='rms', eps=eps, out_dtype=torch.bfloat16, store_stats=save_rstd)
    starts, counts = build_padded_cta_metadata(cu_seqlens, padded_cu_seqlens, total_padded)
    out_m = m if jagged_output else total_padded
    q = torch.empty((out_m, n), device=x.device, dtype=torch.float8_e4m3fn)
    k = torch.full((slu.mma_scale_numel(out_m, n // 32), ), 1, device=x.device, dtype=torch.uint8)
    trans = torch.full(slu.mblock_scale_shape(total_padded, n), 1, device=x.device, dtype=torch.uint8)
    tiles = total_padded // 32
    if tiles:
        with torch.cuda.device(x.device):
            grid = min(tiles, torch.cuda.get_device_properties(x.device).multi_processor_count * 2)
            _two_pass_quant[grid, triton.cdiv(n, 128)](y, q, k, trans, starts, counts, tiles, total_padded,
                                                       triton.cdiv(n // 32, 4), triton.cdiv(total_padded // 32,
                                                                                            4), N=n, HEAD_DIM=head_dim,
                                                       JAGGED_OUTPUT=jagged_output, num_warps=4, enable_fp_fusion=False)
    out = (q, k, trans.view(torch.float8_e8m0fnu))
    return (*out, rr) if save_rstd else out
