"""Case 6 — LayerNorm forward, hand-written TLX TMA kernel.

Adapted from geo/kernel_library kernel_layernorm_fwd_persistent_tma_v2: a
persistent, double-buffered, TMA-prefetched LayerNorm. This is the hand-tuned
TMA reference for the case6 study (the target the modulo-generated kernel should
match).

Key trick for N=384: use BLOCK_N = next_pow2(N) = 512 and rely on TMA's OOB
zero-padding — cols >= N arrive as zero in the SMEM tile and contribute 0 to both
sum(x) and sum(x*x), so no in-kernel column mask is needed on x. w/b are masked.

Pipeline: while tile i is reduced+normalized, the TMA engine prefetches tile i+1
into the alternate SMEM buffer (producer/consumer mbarriers).
"""

from __future__ import annotations

import os

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from torch._inductor.runtime.triton_compat import libdevice


# To keep the comparison against the generated kernel apples-to-apples, we
# autotune ONLY the SMEM ring depth (the
# same degree of freedom the modulo scheduler chooses); LayerNorm is SMEM-only
# (no TMEM), and BLOCK_M/N, num_warps, and everything else stay fixed.
def _autotune_configs():
    # Measured best (do_bench on B200, square/representative shape). TLX_HW_BEST_CONFIG=1
    # uses it directly (fast path, no autotune search); otherwise the full depth grid
    # is swept. Mirrors the tutorial WS GEMM TLX_GEMM_USE_HEURISTIC pattern.
    if os.environ.get("TLX_HW_BEST_CONFIG") == "1":
        return [triton.Config({"NUM_BUFFERS": 2}, num_warps=4)]
    return [
        triton.Config({"NUM_BUFFERS": s}, num_warps=4)
        for s in (2, 3, 4, 5, 6)
    ]


@triton.autotune(configs=_autotune_configs(), key=["M", "N"])
@triton.jit
def layernorm_fwd_tma(
    X,  # [M, N]
    W,  # [N]
    B,  # [N]
    Y,  # [M, N]
    M,
    eps,
    row_stride: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # next_pow2(N), TMA tile width (OOB zero-padded)
    NUM_PERSIST: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    COMPUTE_DTYPE = tl.float32

    # Double-buffered SMEM staging for X tiles + producer/consumer barriers.
    x_buffer = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(X), NUM_BUFFERS)
    producer_bars = tlx.alloc_barriers(num_barriers=NUM_BUFFERS)
    consumer_bars = tlx.alloc_barriers(num_barriers=NUM_BUFFERS)

    # Persistent row-tile band for this CTA (balanced split).
    row_pid = tl.program_id(axis=0)
    num_row_blocks = tl.cdiv(M, BLOCK_M)
    rows_per_sm = num_row_blocks // NUM_PERSIST
    extra = num_row_blocks % NUM_PERSIST
    if row_pid < extra:
        rows_per_sm += 1
        start_off = 0
    else:
        start_off = extra * BLOCK_M
    start_row = start_off + row_pid * rows_per_sm * BLOCK_M
    end_row = start_row + rows_per_sm * BLOCK_M

    x_desc = tl.make_tensor_descriptor(X, shape=[M, N], strides=[row_stride, 1], block_shape=[BLOCK_M, BLOCK_N])
    expected_bytes: tl.constexpr = BLOCK_M * BLOCK_N * tlx.size_of(tlx.dtype_of(X))

    # Hoist w/b (constant across the CTA's rows); mask the OOB tail columns.
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(COMPUTE_DTYPE)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(COMPUTE_DTYPE)

    # Prefetch first tile.
    if start_row < end_row:
        tlx.barrier_expect_bytes(producer_bars[0], expected_bytes)
        tlx.async_descriptor_load(x_desc, x_buffer[0], [start_row, 0], producer_bars[0])

    buffer_id = 0
    producer_phase = 0
    consumer_phase = 1

    for row in tl.range(start_row, end_row, BLOCK_M):
        if row >= start_row + BLOCK_M:
            tlx.barrier_wait(consumer_bars[buffer_id], consumer_phase)

        next_row = row + BLOCK_M
        if next_row < end_row:
            next_id = buffer_id ^ 1
            tlx.barrier_expect_bytes(producer_bars[next_id], expected_bytes)
            tlx.async_descriptor_load(x_desc, x_buffer[next_id], [next_row, 0], producer_bars[next_id])

        tlx.barrier_wait(producer_bars[buffer_id], producer_phase)
        x = tlx.local_load(x_buffer[buffer_id]).to(COMPUTE_DTYPE)

        # Single-pass mean/var (OOB zero cols contribute 0 to both sums).
        sum_x = tl.sum(x, axis=1, keep_dims=True)
        sum_xx = tl.sum(x * x, axis=1, keep_dims=True)
        mean = sum_x / N
        var = sum_xx / N - mean * mean
        rstd = libdevice.rsqrt(var + eps)

        y = (x - mean) * rstd * w[None, :] + b[None, :]

        row_offsets = row + tl.arange(0, BLOCK_M)
        row_mask = row_offsets < M
        y_ptrs = Y + row_offsets[:, None] * row_stride + cols[None, :]
        y_mask = row_mask[:, None] & col_mask[None, :]
        tl.store(y_ptrs, y.to(y_ptrs.dtype.element_ty), mask=y_mask, cache_modifier=".cs")

        tlx.barrier_arrive(consumer_bars[buffer_id])
        buffer_id = buffer_id ^ 1
        if buffer_id == 0:
            producer_phase = producer_phase ^ 1
            consumer_phase = consumer_phase ^ 1
