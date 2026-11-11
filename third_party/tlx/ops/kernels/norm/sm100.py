from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice, tlx

from ._common import _read_bwd, _read_x, _row_offset, _stage_load
from ._common import pin as _pin

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def kernel_rmsnorm_fwd_single_cta(X, Y, W, Rstd_out, row_stride: tl.constexpr, M, N: tl.constexpr, eps,
                                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SHOULD_MASK_ROW: tl.constexpr,
                                  SHOULD_MASK_COL: tl.constexpr, B=None, Residual=None, Post=None, ResidualWeight=None,
                                  ResidualBias=None, ResidualMean=None, ResidualRstd=None, ResidualOut=None, Mean=None,
                                  IS_RMS: tl.constexpr = True, RS: tl.constexpr = 0, PS: tl.constexpr = 0,
                                  YS: tl.constexpr = None, ROS: tl.constexpr = 0, DIRECT: tl.constexpr = True,
                                  NW: tl.constexpr = 4, TPR: tl.constexpr = 32, V: tl.constexpr = 4,
                                  SINGLE_ITER: tl.constexpr = True, USE_LAYOUT: tl.constexpr = False,
                                  WIDE: tl.constexpr = False):
    BM: tl.constexpr = BLOCK_M
    BN: tl.constexpr = BLOCK_N
    XS: tl.constexpr = row_stride
    Y_STRIDE: tl.constexpr = row_stride if YS is None else YS
    COMPUTE_DTYPE = tl.float32
    if WIDE:
        pid = tl.program_id(0)
        np = tl.num_programs(0)
        c = tl.arange(0, BLOCK_N)
        for row in range(pid, M, np):
            total = tl.full((), 0.0, tl.float32)
            for base in range(0, N, BLOCK_N):
                cols = base + c
                mask = (cols < N)[None, :]
                x = _read_x(X, Residual, ResidualWeight, ResidualBias, ResidualMean, ResidualRstd, row, cols, mask, N,
                            BLOCK_N, row_stride, RS)
                if IS_RMS:
                    total += tl.sum(tl.reshape(x * x, [BLOCK_N]), 0)
                else:
                    total += tl.sum(tl.reshape(x, [BLOCK_N]), 0)
            if IS_RMS:
                mu = tl.full((), 0.0, tl.float32)
                var = total / N
            else:
                mu = total / N
                total_sq = tl.full((), 0.0, tl.float32)
                for base in range(0, N, BLOCK_N):
                    cols = base + c
                    mask = (cols < N)[None, :]
                    x = _read_x(X, Residual, ResidualWeight, ResidualBias, ResidualMean, ResidualRstd, row, cols, mask,
                                N, BLOCK_N, row_stride, RS)
                    centered = tl.where(mask, x - mu, 0.0)
                    total_sq += tl.sum(tl.reshape(centered * centered, [BLOCK_N]), 0)
                var = total_sq / N
            rr = libdevice.rsqrt(var + eps)
            if Rstd_out is not None:
                tl.store(Rstd_out + row, rr)
                if Mean is not None:
                    tl.store(Mean + row, mu)
            for base in range(0, N, BLOCK_N):
                cols = base + c
                mask = (cols < N)[None, :]
                idx = row.to(tl.int64) * N + cols[None, :]
                x = _read_x(X, Residual, ResidualWeight, ResidualBias, ResidualMean, ResidualRstd, row, cols, mask, N,
                            BLOCK_N, row_stride, RS)
                if ResidualOut is not None:
                    tl.store(ResidualOut + _row_offset(idx, ROS, N), x, mask=mask)
                y = (x - mu) * rr
                if W is not None:
                    y = y * tl.load(W + cols, mask=cols < N, other=0).to(tl.float32)[None, :]
                if B is not None:
                    y = y + tl.load(B + cols, mask=cols < N, other=0).to(tl.float32)[None, :]
                if Post is not None:
                    y = y + _stage_load(Post, idx, mask, 1, BLOCK_N, ROW_STRIDE=PS, N=N, DIRECT=True, TPR=128).to(
                        tl.float32)
                tl.store(Y + _row_offset(idx, Y_STRIDE, N), y, mask=mask)
    else:
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        col_offsets = tl.arange(0, BN)
        mask_col = col_offsets < N
        if W is not None:
            w = tl.load(W + col_offsets, mask=mask_col, other=0).to(COMPUTE_DTYPE)
        if B is not None:
            b = tl.load(B + col_offsets, mask=mask_col, other=0).to(COMPUTE_DTYPE)
        for iteration in range(1 if SINGLE_ITER else tl.cdiv(M - pid * BM, nprog * BM)):
            row_offsets = (pid + iteration * nprog).to(tl.int64) * BM + tl.arange(0, BM)
            mask_row = row_offsets < M
            rw_mask = mask_row[:, None] & mask_col[None, :]
            rw_offsets = row_offsets[:, None] * N + col_offsets[None, :]
            x = _read_x(X, Residual, ResidualWeight, ResidualBias, ResidualMean, ResidualRstd, row_offsets, col_offsets,
                        rw_mask, N, BN, XS, RS, BM=BM, DIRECT=DIRECT, NW=NW, TPR=TPR, V=V, USE_LAYOUT=USE_LAYOUT,
                        ROW_MASK=mask_row)
            if ResidualOut is not None:
                tl.store(ResidualOut + _row_offset(rw_offsets, ROS, N), x, mask=rw_mask)
            if IS_RMS:
                mean = tl.zeros([BM], COMPUTE_DTYPE)
                x_norm = x
            else:
                mean = tl.sum(x, axis=1) / N
                x_norm = tl.where(mask_col[None, :], x - mean[:, None], 0.0)
            x_sq = x_norm * x_norm
            sum_sq = tl.sum(x_sq, axis=1, keep_dims=True)
            var = sum_sq / N
            rstd = libdevice.rsqrt(var + eps)
            wdy = x_norm * rstd
            if W is not None:
                wdy = wdy * w[None, :]
            if B is not None:
                wdy = wdy + b[None, :]
            if Post is not None:
                wdy = wdy + _stage_load(Post, rw_offsets, rw_mask, BM, BN, ROW_STRIDE=PS, N=N, DIRECT=DIRECT, NW=NW,
                                        TPR=TPR, V=V).to(COMPUTE_DTYPE)
            y = tl.cast(wdy, Y.dtype.element_ty)
            tl.store(Y + _row_offset(rw_offsets, Y_STRIDE, N), y, mask=rw_mask)
            if Rstd_out is not None:
                rstd_1d = tl.reshape(rstd, (BM, ))
                tl.store(Rstd_out + row_offsets, rstd_1d, mask=mask_row)
                if Mean is not None:
                    tl.store(Mean + row_offsets, mean, mask=mask_row)


@triton.jit
def _load_norm_weight(W, cols, mask, BN: tl.constexpr):
    if W is not None:
        return tl.load(W + cols, mask=mask, other=0).to(tl.float32)
    return tl.full([BN], 1.0, tl.float32)


@triton.jit
def _norm_bwd_xhat(x, rr, mu, mask, FROM_NORMALIZED: tl.constexpr):
    if FROM_NORMALIZED:
        return tl.where(mask, x, 0.0)
    return tl.where(mask, x * rr if mu is None else (x - mu) * rr, 0.0)


@triton.jit
def _store_bwd_partial(P, offsets, value, mask, ACCUM: tl.constexpr = False):
    if P is not None:
        value = tl.reshape(value, offsets.shape)
        if ACCUM:
            value = tl.load(P + offsets, mask=mask, other=0) + value
        tl.store(P + offsets, value, mask=mask)


@triton.jit
def kernel_rmsnorm_bwd_persistent_single_cta(
        DX, DY, DW_PARTIAL, X, W, Rstd, row_stride: tl.constexpr, M, N: tl.constexpr, NUM_SMS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SHOULD_MASK_COL: tl.constexpr, Addend=None, DoutAddend=None,
        DxAccum=None, Mean=None, DbPart=None, IS_RMS: tl.constexpr = True, FROM_NORMALIZED: tl.constexpr = False,
        ROUND_DX: tl.constexpr = False, DS: tl.constexpr = None, AS: tl.constexpr = 0, DAS: tl.constexpr = 0,
        ACS: tl.constexpr = 0, DXS: tl.constexpr = None, DIRECT: tl.constexpr = True, NW: tl.constexpr = 4,
        TPR: tl.constexpr = 32, V: tl.constexpr = 4, SINGLE_ITER: tl.constexpr = False, STRIDED: tl.constexpr = False,
        USE_LAYOUT: tl.constexpr = False, WIDE: tl.constexpr = False):
    BM: tl.constexpr = BLOCK_M
    BN: tl.constexpr = BLOCK_N
    XS: tl.constexpr = row_stride
    DY_STRIDE: tl.constexpr = row_stride if DS is None else DS
    DX_STRIDE: tl.constexpr = row_stride if DXS is None else DXS
    COMPUTE_DTYPE = tl.float32
    dw_part = DW_PARTIAL if W is not None else None
    if WIDE:
        pid = tl.program_id(0)
        np = tl.num_programs(0)
        c = tl.arange(0, BN)
        for row in range(pid, M, np):
            rr = tl.load(Rstd + row)
            mu = tl.full((), 0.0, tl.float32) if IS_RMS or FROM_NORMALIZED else tl.load(Mean + row)
            dot = tl.full((), 0.0, tl.float32)
            avg = tl.full((), 0.0, tl.float32)
            for base in range(0, N, BN):
                cols = base + c
                mask = (cols < N)[None, :]
                x, dy = _read_bwd(X, DY, Addend, DoutAddend, row, cols, mask, N, BN, row_stride, DY_STRIDE, AS, DAS)
                xhat = _norm_bwd_xhat(x, rr, mu, mask, FROM_NORMALIZED)
                w = _load_norm_weight(W, cols, cols < N, BN)
                wdy = dy * w[None, :]
                dot += tl.sum(tl.reshape(xhat * wdy, [BN]), 0)
                avg += 0.0 if IS_RMS else tl.sum(tl.reshape(tl.where(mask, wdy, 0.0), [BN]), 0)
            dot = dot / N
            avg = avg / N
            for base in range(0, N, BN):
                cols = base + c
                mask = (cols < N)[None, :]
                idx = row.to(tl.int64) * N + cols[None, :]
                x, dy = _read_bwd(X, DY, Addend, DoutAddend, row, cols, mask, N, BN, row_stride, DY_STRIDE, AS, DAS)
                xhat = _norm_bwd_xhat(x, rr, mu, mask, FROM_NORMALIZED)
                w = _load_norm_weight(W, cols, cols < N, BN)
                dx = (dy * w[None, :] - avg - xhat * dot) * rr
                dx = dx.to(DX.dtype.element_ty if ROUND_DX else dx.dtype)
                dx = dx.to(tl.float32 if ROUND_DX else dx.dtype)
                if DxAccum is not None:
                    dx = dx + _stage_load(DxAccum, idx, mask, 1, BN, ROW_STRIDE=ACS, N=N, DIRECT=True, TPR=128).to(
                        tl.float32)
                tl.store(DX + _row_offset(idx, DX_STRIDE, N), dx, mask=mask)
                poff = pid.to(tl.int64) * N + cols
                _store_bwd_partial(dw_part, poff, dy * xhat, cols < N, ACCUM=True)
                _store_bwd_partial(DbPart, poff, dy, cols < N, ACCUM=True)
    else:
        pid = tl.program_id(0)
        nprog = NUM_SMS
        col_offsets = tl.arange(0, BN)
        mask_col = col_offsets < N
        w = _load_norm_weight(W, col_offsets, mask_col, BN)
        if STRIDED:
            blocks_per_sm = 1 if SINGLE_ITER else tl.cdiv(M - pid * BM, nprog * BM)
            start_block = pid
            block_step = nprog
        else:
            num_row_blocks = tl.cdiv(M, BM)
            blocks_per_sm = num_row_blocks // NUM_SMS
            extra = num_row_blocks % NUM_SMS
            if pid < extra:
                blocks_per_sm += 1
                start_block = pid * blocks_per_sm
            else:
                start_block = pid * blocks_per_sm + extra
            block_step = 1
        partial_dw = tl.zeros((BN, ), dtype=COMPUTE_DTYPE)
        partial_db = tl.zeros((BN, ), dtype=COMPUTE_DTYPE)
        for iteration in range(blocks_per_sm):
            row_offsets = (start_block + iteration * block_step).to(tl.int64) * BM + tl.arange(0, BM)
            mask_row = row_offsets < M
            rw_mask = mask_row[:, None] & mask_col[None, :]
            rw_offsets = row_offsets[:, None] * N + col_offsets[None, :]
            x, dy = _read_bwd(X, DY, Addend, DoutAddend, row_offsets, col_offsets, rw_mask, N, BN, XS, DY_STRIDE, AS,
                              DAS, BM=BM, DIRECT=DIRECT, NW=NW, TPR=TPR, V=V, USE_LAYOUT=USE_LAYOUT)
            rstd = tl.load(Rstd + row_offsets, mask=mask_row, other=0).to(COMPUTE_DTYPE)
            mean = tl.load(Mean + row_offsets, mask=mask_row,
                           other=0).to(COMPUTE_DTYPE)[:, None] if not FROM_NORMALIZED and (not IS_RMS) else None
            x_norm = _norm_bwd_xhat(x, rstd[:, None], mean, rw_mask, FROM_NORMALIZED)
            if W is not None:
                block_partial_dw = tl.where(rw_mask, dy * x_norm, 0.0)
                partial_dw += tl.sum(block_partial_dw, axis=0)
            if DbPart is not None:
                partial_db += tl.sum(tl.where(rw_mask, dy, 0.0), axis=0)
            wdy = w[None, :] * dy
            x_norm_wdy = tl.where(rw_mask, x_norm * wdy, 0.0)
            c = tl.sum(x_norm_wdy, axis=1, keep_dims=True) / N
            avg = None if IS_RMS else tl.sum(tl.where(rw_mask, wdy, 0.0), axis=1, keep_dims=True) / N
            dx = rstd[:, None] * (wdy - x_norm * c if IS_RMS else wdy - avg - x_norm * c)
            dx = dx.to(DX.dtype.element_ty if ROUND_DX else dx.dtype)
            dx = dx.to(COMPUTE_DTYPE if ROUND_DX else dx.dtype)
            if DxAccum is not None:
                dx = dx + _stage_load(DxAccum, rw_offsets, rw_mask, BM, BN, ROW_STRIDE=ACS, N=N, DIRECT=DIRECT, NW=NW,
                                      TPR=TPR, V=V).to(COMPUTE_DTYPE)
            dx = tl.cast(dx, DX.dtype.element_ty)
            tl.store(DX + _row_offset(rw_offsets, DX_STRIDE, N), dx, mask=rw_mask)
        _store_bwd_partial(dw_part, pid * N + col_offsets, partial_dw, mask_col)
        _store_bwd_partial(DbPart, pid * N + col_offsets, partial_db, mask_col)


@triton.jit
def _param_reduce(DW, FINAL_DW, DB, FINAL_DB, M: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    rows = tl.arange(0, BM)
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    for i in tl.static_range(2):
        partial = DW if i == 0 else DB
        final = FINAL_DW if i == 0 else FINAL_DB
        if final is not None:
            value = tl.load(partial + offsets, mask, 0)
            tl.store(final + cols, tl.sum(value, 0), cols < N)


@triton.jit
def _norm_fwd_384(X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor, B: torch.Tensor, Mean_out: torch.Tensor,
                  Rstd_out: torch.Tensor, row_stride: tl.constexpr, M: int, N: tl.constexpr, eps: float,
                  BLOCK_SIZE_M: tl.constexpr, NUM_PERSIST: tl.constexpr, IS_RMS: tl.constexpr = False,
                  USE_TMA: tl.constexpr = True, YS: tl.constexpr = None, NW: tl.constexpr = 4, TPR: tl.constexpr = 32,
                  V: tl.constexpr = 4, SINGLE_ITER: tl.constexpr = False) -> None:
    COMPUTE_DTYPE = tl.float32
    NUM_BUFFERS: tl.constexpr = 2
    CHUNK_N: tl.constexpr = 128
    CHUNKS: tl.constexpr = 3
    Y_STRIDE: tl.constexpr = row_stride if YS is None else YS
    CACHE: tl.constexpr = ''
    if USE_TMA:
        x_buffer = tlx.local_alloc((BLOCK_SIZE_M, CHUNK_N), tlx.dtype_of(X), NUM_BUFFERS * CHUNKS)
        producer_bars = tlx.alloc_barriers(num_barriers=NUM_BUFFERS)
    row_pid = tl.program_id(axis=0)
    num_row_blocks = tl.cdiv(M, BLOCK_SIZE_M)
    rows_per_sm = tl.cdiv(num_row_blocks - row_pid, NUM_PERSIST)
    start_row = row_pid.to(tl.int64) * BLOCK_SIZE_M
    row_step = NUM_PERSIST * BLOCK_SIZE_M
    end_row = start_row + rows_per_sm.to(tl.int64) * row_step
    if USE_TMA:
        x_desc = tl.make_tensor_descriptor(X, shape=[M, N], strides=[row_stride, 1],
                                           block_shape=[BLOCK_SIZE_M, CHUNK_N])
        expected_bytes: tl.constexpr = BLOCK_SIZE_M * CHUNK_N * CHUNKS * tlx.size_of(tlx.dtype_of(X))
    else:
        layout: tl.constexpr = tlx.layout(
            shape=((TPR, NW * 32 // TPR), (V, CHUNK_N // (TPR * V), BLOCK_SIZE_M // (NW * 32 // TPR))),
            stride=((V, CHUNK_N), (1, TPR * V, NW * 32 // TPR * CHUNK_N)))
    if USE_TMA and start_row < end_row:
        tlx.barrier_expect_bytes(producer_bars[0], expected_bytes)
        for j in tl.static_range(CHUNKS):
            tlx.async_descriptor_load(x_desc, x_buffer[j], [start_row.to(tl.int32), j * CHUNK_N], producer_bars[0])
    buffer_id = 0
    producer_phase = 0
    for iteration in range(1 if SINGLE_ITER else rows_per_sm):
        row = start_row + iteration.to(tl.int64) * row_step
        next_row = row + row_step
        if USE_TMA and next_row < end_row:
            next_id = buffer_id ^ 1
            tlx.barrier_expect_bytes(producer_bars[next_id], expected_bytes)
            for j in tl.static_range(CHUNKS):
                tlx.async_descriptor_load(x_desc, x_buffer[next_id * CHUNKS + j], [next_row.to(tl.int32), j * CHUNK_N],
                                          producer_bars[next_id])
        if USE_TMA:
            tlx.barrier_wait(producer_bars[buffer_id], producer_phase)
        row_offsets = (row + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
        row_mask = row_offsets < M
        c = tl.arange(0, CHUNK_N)
        values = ()
        for j in tl.static_range(CHUNKS):
            if USE_TMA:
                value = tlx.local_load(x_buffer[buffer_id * CHUNKS + j]).to(COMPUTE_DTYPE)
            else:
                offsets = row_offsets[:, None] * row_stride + c[None, :] + j * CHUNK_N
                value = _pin(tl.load(X + offsets, row_mask[:, None], 0), layout).to(COMPUTE_DTYPE)
            values += (value, )
        if USE_TMA:
            tl.debug_barrier()
        if IS_RMS:
            mean = tl.zeros((BLOCK_SIZE_M, 1), COMPUTE_DTYPE)
        else:
            summed = values[0]
            for j in tl.static_range(1, CHUNKS):
                summed = summed + values[j]
            mean = tl.sum(summed, axis=1, keep_dims=True) / N
        centered = ()
        for j in tl.static_range(CHUNKS):
            centered += (values[j] - mean, )
        summed_sq = centered[0] * centered[0]
        for j in tl.static_range(1, CHUNKS):
            summed_sq = summed_sq + centered[j] * centered[j]
        var = tl.sum(summed_sq, axis=1, keep_dims=True) / N
        rstd = libdevice.rsqrt(var + eps)
        if Rstd_out is not None:
            mean_1d = tl.reshape(mean, (BLOCK_SIZE_M, ))
            rstd_1d = tl.reshape(rstd, (BLOCK_SIZE_M, ))
            if Mean_out is not None:
                tl.store(Mean_out + row_offsets, mean_1d, mask=row_mask, cache_modifier=CACHE)
            tl.store(Rstd_out + row_offsets, rstd_1d, mask=row_mask, cache_modifier=CACHE)
        for j in tl.static_range(CHUNKS):
            cols = c + j * CHUNK_N
            y = centered[j] * rstd
            if W is not None:
                weight = tl.load(W + cols).to(COMPUTE_DTYPE)
                y = y * weight[None, :]
            if B is not None:
                bias = tl.load(B + cols).to(COMPUTE_DTYPE)
                y = y + bias[None, :]
            y_ptrs = Y + row_offsets[:, None] * Y_STRIDE + cols[None, :]
            y_mask = row_mask[:, None] & (cols < N)[None, :]
            tl.store(y_ptrs, y.to(y_ptrs.dtype.element_ty), mask=y_mask, cache_modifier=CACHE)
        buffer_id = buffer_id ^ 1
        if buffer_id == 0:
            producer_phase = producer_phase ^ 1


def _check_x(x):
    if x.ndim != 2 or x.shape[1] <= 0 or (not x.is_cuda) or (x.dtype not in _DTYPES):
        raise ValueError('expected CUDA FP16/BF16/FP32 [M,N], N>0')
    if x.stride(1) != 1 or x.stride(0) < x.shape[1]:
        raise ValueError('plain TLX requires contiguous feature dimension and nonoverlapping rows')
    return x.shape


def _geometry(x, norm='layer', backward=False, *inputs):
    m, n = x.shape
    bn = triton.next_power_of_2(n)
    nw = 8 if not backward and (n >= 16384 or (norm == 'rms' and n >= 4096)) else 4
    if backward and n == 16384:
        nw = 16 if norm == 'rms' else 8
    if backward:
        bm = max(1, min(16, 8192 // bn))
    elif n <= 512:
        bm = 1 if m <= 128 else 8
    elif n <= 2048:
        bm = 4
    else:
        bm = 2
    tpr = min(bn, nw * 32, max(16, nw * 32 // bm))
    bm = max(bm, nw * 32 // tpr)
    vec = min(4, bn // tpr)
    tiles = triton.cdiv(m, bm)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    stage_bytes = bm * bn * sum(t.element_size() for t in (x, *inputs) if t is not None)
    if n > 16384 or stage_bytes > 196608:
        return (1, 512, min(m, sms * 2), {'WIDE': True})
    np = min(tiles, sms * (1 if n == 16384 else 2)) if backward else min(tiles, sms *
                                                                         8) if n <= 512 and 128 < m < 65536 else tiles
    return (bm, bn, np, {
        'DIRECT': True, 'NW': nw, 'TPR': tpr, 'V': vec, 'SINGLE_ITER': np == tiles, 'USE_LAYOUT': True, 'WIDE': False
    })


def norm_fwd(x, weight=None, bias=None, *, norm='layer', eps=1e-06, residual=None, post_norm_addend=None,
             out_dtype=None, residual_dtype=None, store_stats=True, residual_norm_weight=None, residual_norm_bias=None,
             residual_norm_rstd=None, residual_norm_mean=None, out=None, mean_out=None, rstd_out=None,
             return_residual=None):
    m, n = _check_x(x)
    if norm not in ('layer', 'rms'):
        raise ValueError('norm must be layer or rms')
    rn = residual_norm_weight is not None
    if rn:
        if residual is None or norm != 'rms' or post_norm_addend is not None:
            raise ValueError('saved residual LayerNorm reconstruction requires RMS, residual and no post addend')
        if residual_norm_bias is None or residual_norm_rstd is None or residual_norm_mean is None or residual.dtype != torch.bfloat16 or residual_norm_weight.dtype != torch.bfloat16 or (
                residual_norm_bias.dtype != torch.bfloat16):
            raise ValueError('saved biased residual LayerNorm requires BF16 residual/weight/bias')
    elif any(t is not None for t in (residual_norm_bias, residual_norm_rstd, residual_norm_mean)):
        raise ValueError('residual norm parameters must be supplied together')
    dtype = x.dtype if out_dtype is None else out_dtype
    if dtype not in _DTYPES:
        raise ValueError('unsupported output dtype')
    out = torch.empty_like(x, dtype=dtype) if out is None else out
    if out.dtype != dtype:
        raise ValueError('out dtype must match out_dtype or input dtype')
    if residual is not None:
        residual_dtype = residual.dtype
    store_residual = residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype) if return_residual is None else bool(return_residual)
    residual_out = torch.empty_like(x, dtype=residual_dtype or x.dtype) if store_residual else None
    mean = mean_out if store_stats and norm == 'layer' else None
    rstd = rstd_out if store_stats else None
    if store_stats:
        if norm == 'layer' and mean is None:
            mean = torch.empty(m, device=x.device, dtype=torch.float32)
        if rstd is None:
            rstd = torch.empty(m, device=x.device, dtype=torch.float32)
    if m >= 4096 and n == 384 and (x.dtype == torch.bfloat16) and x.is_contiguous() and (
            x.data_ptr() % 16
            == 0) and out.is_contiguous() and (residual is None) and (post_norm_addend
                                                                      is None) and (not rn) and (not store_residual):
        use_tma = m >= 65536
        np = min(triton.cdiv(m, 32), torch.cuda.get_device_properties(x.device).multi_processor_count * 8)
        with torch.cuda.device(x.device):
            if use_tma:
                triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device='cuda'))
            kernel = _norm_fwd_384
            kernel[
                np,
            ](x, out, weight, bias, mean, rstd, n, m, n, eps, BLOCK_SIZE_M=32, NUM_PERSIST=np, IS_RMS=norm == 'rms',
              USE_TMA=use_tma, YS=n, NW=8, TPR=16, V=4, SINGLE_ITER=triton.cdiv(m, 32) <= np, num_warps=8,
              enable_fp_fusion=False)
        return (out, None, mean, rstd)
    if m:
        bm, bn, np, layout_kw = _geometry(x, norm, False, residual, post_norm_addend)
        kernel = kernel_rmsnorm_fwd_single_cta
        with torch.cuda.device(x.device):
            kernel[
                np,
            ](X=x, W=weight, B=bias, Residual=residual, Post=post_norm_addend, ResidualWeight=residual_norm_weight,
              ResidualBias=residual_norm_bias, ResidualMean=residual_norm_mean, ResidualRstd=residual_norm_rstd, Y=out,
              ResidualOut=residual_out, Mean=mean, Rstd_out=rstd, M=m, N=n, eps=eps, BLOCK_M=bm, BLOCK_N=bn,
              IS_RMS=norm == 'rms', row_stride=x.stride(0), RS=residual.stride(0) if residual is not None else n,
              PS=post_norm_addend.stride(0) if post_norm_addend is not None else n, YS=out.stride(0),
              ROS=residual_out.stride(0) if store_residual else n, num_warps=layout_kw.get('NW', 4),
              SHOULD_MASK_ROW=m % bm != 0, SHOULD_MASK_COL=n != bn, **layout_kw)
    return (out, residual_out, mean, rstd)


def _layernorm_direct(x, weight, eps=1e-06, bias=None):
    out, _, mean, rstd = norm_fwd(x, weight, bias, eps=eps)
    return (out, rstd, mean)


def _layernorm_add_direct(x, addend, weight, eps=1e-06, bias=None):
    out, _, mean, rstd = norm_fwd(x, weight, bias, eps=eps, residual=addend, return_residual=False)
    return (out, rstd, mean)


def rmsnorm_fwd(x, weight, bias=None, residual=None, out_dtype=None, residual_dtype=None, post_norm_addend=None,
                residual_norm_weight=None, residual_norm_bias=None, residual_norm_rstd=None, residual_norm_mean=None,
                eps=1e-06, store_rstd=False):
    out, resout, _, rr = norm_fwd(x, weight, bias, norm='rms', eps=eps, residual=residual,
                                  post_norm_addend=post_norm_addend, out_dtype=out_dtype, residual_dtype=residual_dtype,
                                  store_stats=store_rstd, residual_norm_weight=residual_norm_weight,
                                  residual_norm_bias=residual_norm_bias, residual_norm_rstd=residual_norm_rstd,
                                  residual_norm_mean=residual_norm_mean)
    return (out, resout, rr)


def _rmsnorm_direct(x, weight, bias=None, residual=None, post_norm_addend=None, eps=1e-05):
    return rmsnorm_fwd(x, weight, bias, residual, post_norm_addend=post_norm_addend, eps=eps, store_rstd=True)


def norm_bwd(x, weight, dout, rstd, *, mean=None, norm='rms', has_bias=False, addend=None, dout_addend=None,
             dx_accum=None, dx_out=None, from_normalized=False, dx_dtype=None, dw_dtype=None, db_dtype=None,
             round_dx_before_accum=False):
    m, n = _check_x(x)
    if norm not in ('layer', 'rms'):
        raise ValueError('norm must be layer or rms')
    if rstd is None or (norm == 'layer' and (not from_normalized) and (mean is None)):
        raise ValueError('normalization backward requires saved statistics')
    dtype = dx_dtype or x.dtype
    if dx_out is None:
        dx = torch.empty_like(x, dtype=dtype)
    else:
        if dx_out.dtype != dtype:
            raise ValueError('dx_out has wrong dtype')
        dx = dx_out
    pdtype = weight.dtype if weight is not None else x.dtype
    dw = torch.empty(n, device=x.device, dtype=dw_dtype or pdtype) if weight is not None else None
    db = torch.empty(n, device=x.device, dtype=db_dtype or pdtype) if has_bias else None
    if not m:
        if dw is not None:
            dw.zero_()
        if db is not None:
            db.zero_()
        return (dx, dw, db)
    bm, bn, np, layout_kw = _geometry(x, norm, True, dout, addend, dout_addend, dx_accum)
    allocate = torch.zeros if layout_kw['WIDE'] else torch.empty
    wp = allocate((np, n), device=x.device, dtype=torch.float32) if dw is not None else None
    bp = allocate((np, n), device=x.device, dtype=torch.float32) if db is not None else None
    kernel = kernel_rmsnorm_bwd_persistent_single_cta
    with torch.cuda.device(x.device):
        kernel[
            np,
        ](X=x, W=weight, DY=dout, Addend=addend, DoutAddend=dout_addend, DxAccum=dx_accum, Mean=mean, Rstd=rstd, DX=dx,
          DW_PARTIAL=wp, DbPart=bp, M=m, N=n, BLOCK_M=bm, BLOCK_N=bn, IS_RMS=norm == 'rms',
          FROM_NORMALIZED=from_normalized, ROUND_DX=round_dx_before_accum, row_stride=x.stride(0), DS=dout.stride(0),
          AS=addend.stride(0) if addend is not None else n, DAS=dout_addend.stride(0) if dout_addend is not None else n,
          ACS=dx_accum.stride(0) if dx_accum is not None else n, DXS=dx.stride(0), num_warps=layout_kw.get('NW', 4),
          NUM_SMS=np, SHOULD_MASK_COL=n != bn, STRIDED=not layout_kw['WIDE'], **layout_kw)
        if dw is not None or db is not None:
            _param_reduce[triton.cdiv(n, 32)](wp, dw, bp, db, np, n, triton.next_power_of_2(np), 32, num_warps=4)
    return (dx, dw, db)


def norm_backward_from_normalized(x_hat, weight, dout, rstd, *, norm='layer', has_bias=False, dx_dtype=None,
                                  dw_dtype=torch.float32, db_dtype=torch.float32, residual_grad=None,
                                  round_dx_before_accum=False):
    return norm_bwd(x_hat, weight, dout, rstd, norm=norm, has_bias=has_bias, from_normalized=True, dx_dtype=dx_dtype,
                    dw_dtype=dw_dtype, db_dtype=db_dtype, dx_accum=residual_grad,
                    round_dx_before_accum=round_dx_before_accum)


def _layernorm_backward(x, weight, dout, mean, rstd, bias=None, addend=None, dx_accum=None, dx_out=None,
                        dout_addend=None, *, round_dx_before_accum=False, return_fp32_param_grads=False):
    return norm_bwd(x, weight, dout, rstd, mean=mean, norm='layer', has_bias=bias is not None, addend=addend,
                    dout_addend=dout_addend, dx_accum=dx_accum, dx_out=dx_out,
                    round_dx_before_accum=round_dx_before_accum,
                    dw_dtype=torch.float32 if return_fp32_param_grads else None,
                    db_dtype=torch.float32 if return_fp32_param_grads else bias.dtype if bias is not None else None)


def _needs_grad(*values):
    return torch.is_grad_enabled() and any(v is not None and v.requires_grad for v in values)


class _LayerNormAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, addend, eps):
        out, _, mean, rr = norm_fwd(x, weight, bias, norm='layer', eps=eps, residual=addend, return_residual=False)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, weight, bias, addend, mean, rr)
        ctx.mark_non_differentiable(rr, mean)
        return (out, rr, mean)

    @staticmethod
    def backward(ctx, dout, _drr, _dmean):
        x, w, b, add, mean, rr = ctx.saved_tensors
        dx, dw, db = _layernorm_backward(x, w, dout.contiguous(), mean, rr, b, addend=add)
        return (dx, dw, db, dx if add is not None else None, None)


class _RMSNormAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, residual, post, eps, out_dtype, residual_dtype):
        out, resout, _, rr = norm_fwd(x, weight, bias, norm='rms', eps=eps, residual=residual, post_norm_addend=post,
                                      out_dtype=out_dtype, residual_dtype=residual_dtype, store_stats=True)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x if residual is None else resout, weight, rr)
        ctx.out_dtype = out.dtype
        ctx.x_dtype = x.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.residual_dtype = residual.dtype if residual is not None else None
        ctx.post_dtype = post.dtype if post is not None else None
        ctx.mark_non_differentiable(rr)
        return (out, resout, rr)

    @staticmethod
    def backward(ctx, dout, dresout, _drr):
        x, w, rr = ctx.saved_tensors
        effective_dout = dout if dout is not None else torch.zeros_like(x, dtype=ctx.out_dtype)
        dx, dw, db = norm_bwd(x, w, effective_dout.contiguous(), rr, norm='rms', has_bias=ctx.bias_dtype is not None,
                              dx_dtype=torch.float32, dx_accum=dresout.contiguous() if dresout is not None else None,
                              db_dtype=ctx.bias_dtype)
        dres = dx.to(ctx.residual_dtype) if ctx.residual_dtype is not None else None
        post = dout.to(ctx.post_dtype) if ctx.post_dtype is not None and dout is not None else None
        return (dx.to(ctx.x_dtype), dw, db, dres, post, None, None, None)


def layernorm(x, weight, eps=1e-06, bias=None):
    if _needs_grad(x, weight, bias):
        return _LayerNormAutograd.apply(x, weight, bias, None, eps)
    return _layernorm_direct(x, weight, eps, bias)


def layernorm_add(x, addend, weight, eps=1e-06, bias=None):
    if addend.dtype != x.dtype:
        raise ValueError('LayerNorm addend must match input dtype')
    if _needs_grad(x, addend, weight, bias):
        return _LayerNormAutograd.apply(x, weight, bias, addend, eps)
    return _layernorm_add_direct(x, addend, weight, eps, bias)


def rmsnorm(x, weight, bias=None, residual=None, post_norm_addend=None, eps=1e-05):
    if _needs_grad(x, weight, bias, residual, post_norm_addend):
        return _RMSNormAutograd.apply(x, weight, bias, residual, post_norm_addend, eps, None, None)
    return _rmsnorm_direct(x, weight, bias, residual, post_norm_addend, eps)
