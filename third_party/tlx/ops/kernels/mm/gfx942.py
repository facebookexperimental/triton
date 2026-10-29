"""Shared MI300X (gfx942/CDNA3) GEMM implementation for ``mm`` and ``addmm``.

A direct-load kernel serves both operations through a compact heuristic or
full autotune space. Short-M ``mm`` shapes use an intra-CTA K split instead.
"""

import functools
from typing import NamedTuple

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.inductor.reduce_k import _reduce_k_kernel

from ..._invariance import gemm_fixed_split_k


class _LocalSplitUPlan(NamedTuple):
    tile_m: int
    tile_n: int
    local_split_u: int
    wave_k: int
    k_width: int


# Debug/ablation toggle. Eligibility remains restricted by
# ``_precheck_local_split_u`` below.
ENABLE_LOCAL_SPLIT_U = True
ENABLE_GLOBAL_SPLIT_K = True

_MEASURED_LOCAL_SPLIT_U_PLANS = {
    # These cover the current gfx942_2 focus shapes and can be generalized after broader measurements.
    (7, 8192, 2048):
    _LocalSplitUPlan(tile_m=16, tile_n=32, local_split_u=16, wave_k=64, k_width=8),
    (7, 2048, 4096):
    _LocalSplitUPlan(tile_m=16, tile_n=16, local_split_u=16, wave_k=64, k_width=8),
}

# Full-space trials. These do not change heuristic dispatch until their
# decision-tree configuration is refreshed independently.
_FULL_LOCAL_SPLIT_U_PLANS = {
    **_MEASURED_LOCAL_SPLIT_U_PLANS,
    (128, 256, 6144):
    _LocalSplitUPlan(tile_m=16, tile_n=16, local_split_u=16, wave_k=64, k_width=8),
    (128, 256, 3072):
    _LocalSplitUPlan(tile_m=16, tile_n=16, local_split_u=16, wave_k=64, k_width=8),
}


@triton.jit
def _load_local_split_u_operands_gfx942(
    a_ptr,
    b_ptr,
    global_rows,
    global_cols,
    split_ids,
    rk,
    macro_k_base,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    WAVE_K: tl.constexpr,
    K_WIDTH: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
):
    split_k = macro_k_base + split_ids[:, None, None] * WAVE_K
    a_offsets = global_rows[None, :, None] * stride_am + (split_k + rk[None, None, :]) * stride_ak
    b_offsets = (split_k + rk[None, :, None]) * stride_bk + global_cols[None, None, :] * stride_bn
    a_offsets = tlx.require_layout(a_offsets, dot_a)
    b_offsets = tlx.require_layout(b_offsets, dot_b)
    a = tlx.buffer_load(a_ptr, a_offsets, contiguity=K_WIDTH)
    b = tlx.buffer_load(b_ptr, b_offsets, cache=".cg", contiguity=K_WIDTH)
    return a, b


@triton.jit
def _local_split_u_kernel_gfx942(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    K_WIDTH: tl.constexpr,
    WAVE_K: tl.constexpr,
    LOCAL_SPLIT_U: tl.constexpr,
):
    """Compute one output tile with one K partition per wave."""
    MACRO_K: tl.constexpr = WAVE_K * LOCAL_SPLIT_U
    tl.static_assert(K % MACRO_K == 0)

    pid = tl.program_id(0).to(tl.int32)
    grid_n = tl.cdiv(N, TILE_N)
    pid_n = pid % grid_n
    pid_m = pid // grid_n
    split_ids = tl.arange(0, LOCAL_SPLIT_U).to(tl.int32)
    rows = tl.arange(0, TILE_M).to(tl.int32)
    output_rows = pid_m * TILE_M + rows
    global_rows = tl.where(output_rows < M, output_rows, 0)
    local_cols = tl.arange(0, TILE_N).to(tl.int32)
    output_cols = pid_n * TILE_N + local_cols
    global_cols = tl.where(output_cols < N, output_cols, 0)
    rk = tl.arange(0, WAVE_K).to(tl.int32)

    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[LOCAL_SPLIT_U, 1, 1],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=K_WIDTH)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=K_WIDTH)
    acc = tlx.zeros((LOCAL_SPLIT_U, TILE_M, TILE_N), tl.float32, layout=mma)

    current_a, current_b = _load_local_split_u_operands_gfx942(
        a_ptr,
        b_ptr,
        global_rows,
        global_cols,
        split_ids,
        rk,
        0,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        WAVE_K,
        K_WIDTH,
        dot_a,
        dot_b,
    )
    for macro in tl.range(0, K // MACRO_K - 1, num_stages=1):
        next_a, next_b = _load_local_split_u_operands_gfx942(
            a_ptr,
            b_ptr,
            global_rows,
            global_cols,
            split_ids,
            rk,
            (macro + 1) * MACRO_K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            WAVE_K,
            K_WIDTH,
            dot_a,
            dot_b,
        )
        acc = tl.dot(current_a, current_b, acc, allow_tf32=False, out_dtype=tl.float32)
        current_a = next_a
        current_b = next_b
    acc = tl.dot(current_a, current_b, acc, allow_tf32=False, out_dtype=tl.float32)

    if LOCAL_SPLIT_U == 16 and TILE_N == 32:
        partial_layout: tl.constexpr = tlx.swizzled_layout(2, 2, 3, order=[2, 1, 0])
        partial_buffer = tlx.local_alloc(
            (LOCAL_SPLIT_U, TILE_M, TILE_N),
            tl.float32,
            1,
            layout=partial_layout,
        )
    else:
        partial_buffer = tlx.local_alloc((LOCAL_SPLIT_U, TILE_M, TILE_N), tl.float32, 1)
    partial_view = tlx.local_view(partial_buffer, 0)
    tlx.local_store(partial_view, acc)
    tl.debug_barrier()

    result = tl.reshape(
        tlx.local_load(tlx.local_slice(partial_view, [0, 0, 0], [1, TILE_M, TILE_N])),
        (TILE_M, TILE_N),
    )
    for split in tl.static_range(1, LOCAL_SPLIT_U):
        partial = tlx.local_load(tlx.local_slice(partial_view, [split, 0, 0], [1, TILE_M, TILE_N]))
        result += tl.reshape(partial, (TILE_M, TILE_N))

    output_ptrs = c_ptr + output_rows[:, None] * stride_cm + output_cols[None, :] * stride_cn
    tl.store(
        output_ptrs,
        result.to(c_ptr.dtype.element_ty),
        mask=(output_rows[:, None] < M) & (output_cols[None, :] < N),
    )


@triton.jit
def _load_lds_ring_operands_gfx942(
    a_ptr,
    b_ptr,
    reg_m,
    reg_n,
    offs_k,
    k,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    USE_I64_A_OFFSETS: tl.constexpr,
    USE_I64_B_OFFSETS: tl.constexpr,
    USE_BUFFER_A: tl.constexpr,
    USE_BUFFER_B: tl.constexpr,
):
    """Load a full K tile with the direct path's addressing contracts."""
    if USE_I64_A_OFFSETS:
        a_offsets = (reg_m.to(tl.int64)[:, None] * stride_am + (k + offs_k[None, :]).to(tl.int64) * stride_ak)
    else:
        a_offsets = reg_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
    if USE_I64_B_OFFSETS:
        b_offsets = ((k + offs_k[:, None]).to(tl.int64) * stride_bk + reg_n.to(tl.int64)[None, :] * stride_bn)
    else:
        b_offsets = (k + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn
    if USE_BUFFER_A:
        a = tlx.buffer_load(a_ptr, a_offsets, contiguity=1)
    else:
        a = tl.load(a_ptr + a_offsets)
    if USE_BUFFER_B:
        b = tlx.buffer_load(b_ptr, b_offsets, contiguity=1)
    else:
        b = tl.load(b_ptr + b_offsets)
    return a, b


@triton.jit
def _store_n_edge_panel_gfx942(c_ptr, acc, rows, base_n, N: tl.constexpr, WIDTH: tl.constexpr):
    """Keep each panel's scalar and aligned vector owners disjoint."""
    values = acc.to(tl.float16)
    shift = (rows * N) & 1
    cols = tl.arange(0, WIDTH).to(tl.int32)
    gather_cols = cols[None, :] + shift[:, None]
    shifted = tl.gather(values, tl.minimum(gather_cols, WIDTH - 1), axis=1)
    aligned_base = tl.multiple_of(rows * N + shift, 2)
    offsets = aligned_base[:, None] + base_n + cols[None, :]
    mask = (gather_cols < WIDTH) & (base_n + gather_cols < N)
    tl.store(c_ptr + offsets, shifted, mask=mask)
    first = tl.reshape(
        tl.gather(values, tl.full((256, 1), 0, tl.int32), axis=1),
        (256, ),
    )
    tl.store(c_ptr + rows * N + base_n, first, mask=(shift == 1) & (base_n < N))


@triton.jit
def _n_edge_panels_kernel_gfx942(a_ptr, b_ptr, c_ptr, pid_m, pid_n, N: tl.constexpr, K: tl.constexpr,
                                 SECOND_WIDTH: tl.constexpr = 64):
    """Share each A tile across two independently accumulated N panels."""
    tl.static_assert(SECOND_WIDTH == 64 or SECOND_WIDTH == 128)
    rows = pid_m * 256 + tl.arange(0, 256).to(tl.int32)
    base_n = pid_n * 256
    cols0 = base_n + tl.arange(0, 128).to(tl.int32)
    cols1 = base_n + 128 + tl.arange(0, SECOND_WIDTH).to(tl.int32)
    # Wrapped padding is read only from valid B columns and never stored.
    load_cols0 = cols0 % N
    load_cols1 = cols1 % N
    rk = tl.arange(0, 64).to(tl.int32)
    packed_a: tl.constexpr = tlx.layout(
        shape=((8, 8, 8), (8, 4)),
        stride=((8, 64, 512), (1, 4096)),
    )
    packed_b0: tl.constexpr = tlx.layout(
        shape=((8, 8, 8), (8, 2)),
        stride=((1024, 1, 8), (128, 64)),
    )
    packed_b1: tl.constexpr = tlx.layout(
        shape=((8, 8, 8), (8, SECOND_WIDTH // 64)),
        stride=((SECOND_WIDTH * 8, 1, 8), (SECOND_WIDTH, 64)),
    )
    acc0 = tl.zeros((256, 128), tl.float32)
    acc1 = tl.zeros((256, SECOND_WIDTH), tl.float32)
    k_main = (K // 64) * 64
    for k in range(0, k_main, 64):
        a_offsets = rows[:, None] * K + k + rk[None, :]
        b0_offsets = k + rk[:, None] + load_cols0[None, :] * K
        b1_offsets = k + rk[:, None] + load_cols1[None, :] * K
        a_offsets = tlx.require_layout(a_offsets, packed_a)
        b0_offsets = tlx.require_layout(b0_offsets, packed_b0)
        b1_offsets = tlx.require_layout(b1_offsets, packed_b1)
        a = tlx.buffer_load(a_ptr, a_offsets, contiguity=8)
        b0 = tlx.buffer_load(b_ptr, b0_offsets, contiguity=8)
        b1 = tlx.buffer_load(b_ptr, b1_offsets, contiguity=8)
        acc0 = tl.dot(a, b0, acc0, allow_tf32=False, out_dtype=tl.float32)
        acc1 = tl.dot(a, b1, acc1, allow_tf32=False, out_dtype=tl.float32)
    # Preserve the incumbent's K=16 zero-filled tail and increasing K order.
    tail_k = tl.arange(0, 16).to(tl.int32)
    tail_mask = tail_k < K - k_main
    a = tl.load(a_ptr + rows[:, None] * K + k_main + tail_k[None, :], mask=tail_mask[None, :], other=0.0)
    b0 = tl.load(b_ptr + k_main + tail_k[:, None] + load_cols0[None, :] * K, mask=tail_mask[:, None], other=0.0)
    b1 = tl.load(b_ptr + k_main + tail_k[:, None] + load_cols1[None, :] * K, mask=tail_mask[:, None], other=0.0)
    acc0 = tl.dot(a, b0, acc0, allow_tf32=False, out_dtype=tl.float32)
    acc1 = tl.dot(a, b1, acc1, allow_tf32=False, out_dtype=tl.float32)
    _store_n_edge_panel_gfx942(c_ptr, acc0, rows, base_n, N, 128)
    _store_n_edge_panel_gfx942(c_ptr, acc1, rows, base_n + 128, N, SECOND_WIDTH)


@triton.jit
def _direct_matmul_kernel_gfx942(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    USE_I64_A_OFFSETS: tl.constexpr = False,
    USE_I64_B_OFFSETS: tl.constexpr = False,
    USE_I64_C_OFFSETS: tl.constexpr = False,
    SPLIT_M_128_32: tl.constexpr = False,
    USE_LDS_RING: tl.constexpr = False,
    USE_PACKED_K_LOADS: tl.constexpr = False,
    SPLIT_M_128_64: tl.constexpr = False,
    ALIGN_FP16_STORES: tl.constexpr = False,
    PACK_FP16_STORES: tl.constexpr = False,
    USE_N_EDGE_PANELS: tl.constexpr = False,
    USE_N_INTERIOR_PANELS: tl.constexpr = False,
    USE_SINGLE_STAGE_POINTER_LOADS: tl.constexpr = False,
    USE_PIPELINED_POINTER_LOADS: tl.constexpr = False,
    TILE_ID=0,
    OVERRIDE_TILE_ID: tl.constexpr = False,
    USE_STREAMING_BF16_STORES: tl.constexpr = False,
):
    """Register-staged GEMM with per-operand cache and XCD policy."""
    pid = TILE_ID.to(tl.int32) if OVERRIDE_TILE_ID else tl.program_id(0).to(tl.int32)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    grid_mn = grid_m * grid_n

    # Stripe complete chunks over the eight XCDs.  Leave a short tail in its
    # original order so no remapped pid can escape the output-tile grid.
    if NUM_XCDS != 1:
        aligned = (grid_mn // (NUM_XCDS * XCD_CHUNK)) * (NUM_XCDS * XCD_CHUNK)
        if pid < aligned:
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            pid = ((local_pid // XCD_CHUNK) * NUM_XCDS * XCD_CHUNK + xcd * XCD_CHUNK + local_pid % XCD_CHUNK)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    EDGE_PANELS: tl.constexpr = (USE_N_EDGE_PANELS and M == 262144 and N == 1195 and K == 2373
                                 and a_ptr.dtype.element_ty == tl.float16 and b_ptr.dtype.element_ty == tl.float16
                                 and c_ptr.dtype.element_ty == tl.float16 and not ADD_BIAS and stride_am == K
                                 and stride_ak == 1 and stride_bk == 1 and stride_bn == K and stride_cm == N
                                 and stride_cn == 1 and not USE_I64_A_OFFSETS and not USE_I64_B_OFFSETS
                                 and not USE_I64_C_OFFSETS and BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 64
                                 and GROUP_M == 1 and NUM_XCDS == 8 and XCD_CHUNK == 8 and USE_PACKED_K_LOADS
                                 and ALIGN_FP16_STORES and not PACK_FP16_STORES and not SPLIT_M_128_32
                                 and not SPLIT_M_128_64 and not USE_LDS_RING)
    # The XCD permutation remains a bijection. Only the last N tile changes
    # its internal panels; the guarded shape has no partial M tiles.
    if EDGE_PANELS and pid_n == grid_n - 1:
        _n_edge_panels_kernel_gfx942(a_ptr, b_ptr, c_ptr, pid_m, pid_n, N, K)
    elif USE_N_INTERIOR_PANELS and EDGE_PANELS:
        # The guarded interior tile owns 256 valid columns and 256 valid rows.
        # Both panels consume the same immutable A tile in increasing K order;
        # their accumulators and output columns are disjoint. Packed B ownership
        # covers each 128-column panel exactly once. No local allocation, alias,
        # async operation or manually managed barrier is introduced: tl.dot's
        # compiler-managed conversions retain all cross-wave visibility fences.
        _n_edge_panels_kernel_gfx942(
            a_ptr,
            b_ptr,
            c_ptr,
            pid_m,
            pid_n,
            N,
            K,
            SECOND_WIDTH=128,
        )
    elif SPLIT_M_128_32 or SPLIT_M_128_64:
        # Two power-of-two row panels consume the same B registers before
        # the next K iteration. No asynchronous storage or alias is added.
        PANEL_M: tl.constexpr = 64 if SPLIT_M_128_64 else 32
        tl.static_assert(BLOCK_M == 128 + PANEL_M)
        if SPLIT_M_128_32:
            tl.static_assert(K % BLOCK_K == 0)
        base_m = pid_m * BLOCK_M
        base_n = pid_n * BLOCK_N
        offs_m0 = (base_m + tl.arange(0, 128).to(tl.int32)) % M
        offs_m1 = (base_m + 128 + tl.arange(0, PANEL_M).to(tl.int32)) % M
        offs_n = (base_n + tl.arange(0, BLOCK_N).to(tl.int32)) % N
        offs_k = tl.arange(0, BLOCK_K).to(tl.int32)

        acc0 = tl.zeros((128, BLOCK_N), tl.float32)
        acc1 = tl.zeros((PANEL_M, BLOCK_N), tl.float32)
        k_main = (K // BLOCK_K) * BLOCK_K
        for k in range(0, k_main, BLOCK_K):
            if USE_I64_B_OFFSETS:
                b_offsets = ((k + offs_k[:, None]).to(tl.int64) * stride_bk + offs_n.to(tl.int64)[None, :] * stride_bn)
            else:
                b_offsets = (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
            if USE_I64_A_OFFSETS:
                a0_offsets = (offs_m0.to(tl.int64)[:, None] * stride_am +
                              (k + offs_k[None, :]).to(tl.int64) * stride_ak)
                a1_offsets = (offs_m1.to(tl.int64)[:, None] * stride_am +
                              (k + offs_k[None, :]).to(tl.int64) * stride_ak)
            else:
                a0_offsets = offs_m0[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
                a1_offsets = offs_m1[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
            b_ptrs = b_ptr + b_offsets
            a0_ptrs = a_ptr + a0_offsets
            a1_ptrs = a_ptr + a1_offsets
            b = tl.load(b_ptrs)
            a0 = tl.load(a0_ptrs)
            a1 = tl.load(a1_ptrs)
            acc0 = tl.dot(a0, b, acc0, allow_tf32=False, out_dtype=tl.float32)
            acc1 = tl.dot(a1, b, acc1, allow_tf32=False, out_dtype=tl.float32)

        if K % BLOCK_K != 0:
            # Match the direct path's zero-filled tail and increasing K order.
            if a_ptr.dtype.element_ty == tl.float16 and K > 256 and BLOCK_K > 16 and K % BLOCK_K <= 16:
                tail_k = tl.arange(0, 16).to(tl.int32)
            else:
                tail_k = offs_k
            if USE_I64_B_OFFSETS:
                b_offsets = ((k_main + tail_k[:, None]).to(tl.int64) * stride_bk +
                             offs_n.to(tl.int64)[None, :] * stride_bn)
            else:
                b_offsets = (k_main + tail_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
            if USE_I64_A_OFFSETS:
                a0_offsets = (offs_m0.to(tl.int64)[:, None] * stride_am +
                              (k_main + tail_k[None, :]).to(tl.int64) * stride_ak)
                a1_offsets = (offs_m1.to(tl.int64)[:, None] * stride_am +
                              (k_main + tail_k[None, :]).to(tl.int64) * stride_ak)
            else:
                a0_offsets = offs_m0[:, None] * stride_am + (k_main + tail_k[None, :]) * stride_ak
                a1_offsets = offs_m1[:, None] * stride_am + (k_main + tail_k[None, :]) * stride_ak
            tail = tail_k < K - k_main
            b = tl.load(b_ptr + b_offsets, mask=tail[:, None], other=0.0)
            a0 = tl.load(a_ptr + a0_offsets, mask=tail[None, :], other=0.0)
            a1 = tl.load(a_ptr + a1_offsets, mask=tail[None, :], other=0.0)
            acc0 = tl.dot(a0, b, acc0, allow_tf32=False, out_dtype=tl.float32)
            acc1 = tl.dot(a1, b, acc1, allow_tf32=False, out_dtype=tl.float32)

        rows0 = base_m + tl.arange(0, 128).to(tl.int32)
        rows1 = base_m + 128 + tl.arange(0, PANEL_M).to(tl.int32)
        cols = base_n + tl.arange(0, BLOCK_N).to(tl.int32)
        idx_n = cols[None, :]
        idx_m0 = rows0[:, None]
        idx_m1 = rows1[:, None]
        mask0 = (idx_m0 < M) & (idx_n < N)
        mask1 = (idx_m1 < M) & (idx_n < N)
        if ADD_BIAS:
            bias0 = tl.load(
                bias_ptr + idx_m0 * stride_bias_m + idx_n * stride_bias_n,
                mask=mask0,
                eviction_policy="evict_last",
            )
            bias1 = tl.load(
                bias_ptr + idx_m1 * stride_bias_m + idx_n * stride_bias_n,
                mask=mask1,
                eviction_policy="evict_last",
            )
            acc0 += bias0.to(tl.float32)
            acc1 += bias1.to(tl.float32)
        if USE_I64_C_OFFSETS:
            c0_offsets = idx_m0.to(tl.int64) * stride_cm + idx_n.to(tl.int64) * stride_cn
            c1_offsets = idx_m1.to(tl.int64) * stride_cm + idx_n.to(tl.int64) * stride_cn
        else:
            c0_offsets = idx_m0 * stride_cm + idx_n * stride_cn
            c1_offsets = idx_m1 * stride_cm + idx_n * stride_cn
        tl.store(c_ptr + c0_offsets, acc0, mask=mask0)
        tl.store(c_ptr + c1_offsets, acc1, mask=mask1)
    else:
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)) % N
        offs_k = tl.arange(0, BLOCK_K).to(tl.int32)
        reg_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
        reg_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

        # Buffer descriptors use signed-i32 byte bounds. Keep generic pointer
        # loads for wide/negative strides and views that exceed the descriptor.
        # Include the final element's full byte width, not only its start.
        # The measured short-M family also uses smaller output tiles in the
        # same compiler-managed pointer pipeline. Operand ownership follows
        # reg_m/reg_n; the masked epilogue remains the unique output writer.
        SMALL_POINTER_TILE: tl.constexpr = (((M == 902 and ((N == 2048 and (K == 2048 or K == 4096)) or
                                                            (N == 4096 and (K == 1091 or K == 2048 or K == 6656)))) or
                                             (M == 440 and ((N == 2048 and (K == 2048 or K == 4096)) or
                                                            (N == 4096 and (K == 1093 or K == 2048 or K == 6400)))))
                                            and (BLOCK_M == 64 or BLOCK_M == 128) and (BLOCK_N == 64 or BLOCK_N == 128))
        # Keep the buffer policy for other layouts and element types.
        # Compiler-managed pointer loads for the four measured BF16 production tiles.
        BF16_POINTER_TILE: tl.constexpr = (tlx.dtype_of(a_ptr) == tl.bfloat16 and tlx.dtype_of(b_ptr) == tl.bfloat16
                                           and (((M, N, K) == (1024, 6144, 20480) and BLOCK_M == 128 and BLOCK_N == 128
                                                 and BLOCK_K == 64 and tlx.num_warps() == 4) or
                                                ((M, N, K) == (61440, 2048, 5120) and BLOCK_M == 128 and BLOCK_N == 256
                                                 and BLOCK_K == 32 and tlx.num_warps() == 4) or
                                                ((M, N, K) == (819200, 1024, 192) and BLOCK_M == 256 and BLOCK_N == 128
                                                 and BLOCK_K == 32 and tlx.num_warps() == 8) or
                                                ((M, N, K) == (4096, 1894, 242432) and BLOCK_M == 64 and BLOCK_N == 64
                                                 and BLOCK_K == 256 and tlx.num_warps() == 4)))
        PIPELINED_POINTERS: tl.constexpr = (
            USE_PIPELINED_POINTER_LOADS and stride_am == K and stride_ak == 1 and stride_bk == 1 and stride_bn == K
            and (BF16_POINTER_TILE or (tlx.dtype_of(a_ptr) == tl.float16 and tlx.dtype_of(b_ptr) == tl.float16 and
                                       ((BLOCK_M == 128 and BLOCK_N == 128) or SMALL_POINTER_TILE) and
                                       (BLOCK_K == 64 or (SMALL_POINTER_TILE and (BLOCK_K == 128 or BLOCK_K == 256))))))
        USE_BUFFER_A: tl.constexpr = (not USE_SINGLE_STAGE_POINTER_LOADS and not PIPELINED_POINTERS
                                      and not USE_I64_A_OFFSETS and 0 <= stride_am and stride_am <= 0x7fffffff
                                      and 0 <= stride_ak and stride_ak <= 0x7fffffff
                                      and ((M - 1) * stride_am + (K - 1) * stride_ak + 1) *
                                      (tlx.dtype_of(a_ptr).primitive_bitwidth // 8) <= 0x7ffffffe)
        USE_BUFFER_B: tl.constexpr = (not USE_SINGLE_STAGE_POINTER_LOADS and not PIPELINED_POINTERS
                                      and not USE_I64_B_OFFSETS and 0 <= stride_bk and stride_bk <= 0x7fffffff
                                      and 0 <= stride_bn and stride_bn <= 0x7fffffff
                                      and ((K - 1) * stride_bk + (N - 1) * stride_bn + 1) *
                                      (tlx.dtype_of(b_ptr).primitive_bitwidth // 8) <= 0x7ffffffe)
        PACKED_K_LOADS: tl.constexpr = (USE_PACKED_K_LOADS and USE_BUFFER_A and USE_BUFFER_B
                                        and tlx.dtype_of(a_ptr) == tl.float16 and stride_ak == 1 and stride_bk == 1
                                        and BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 64)
        SHORT_PACKED_K_LOADS: tl.constexpr = (USE_BUFFER_A and USE_BUFFER_B and tlx.dtype_of(a_ptr) == tl.float16
                                              and tlx.dtype_of(b_ptr) == tl.float16 and stride_ak == 1
                                              and stride_bk == 1 and BLOCK_M == 64 and BLOCK_N == 64 and BLOCK_K == 64
                                              and K >= 64 and K < 256 and tlx.num_warps() == 4)
        # Odd K leaves rows 2-byte aligned, but each thread's eight K elements
        # stay contiguous, and gfx942 buffer loads accept unaligned dwordx4.
        UNALIGNED_K_LOADS: tl.constexpr = (K % 8 != 0 and stride_ak == 1 and stride_bk == 1 and BLOCK_K % 8 == 0
                                           and tlx.dtype_of(a_ptr).primitive_bitwidth == 16
                                           and tlx.dtype_of(b_ptr).primitive_bitwidth == 16 and not PACKED_K_LOADS
                                           and not SHORT_PACKED_K_LOADS and not USE_SINGLE_STAGE_POINTER_LOADS
                                           and not USE_I64_A_OFFSETS and not USE_I64_B_OFFSETS
                                           and ((M - 1) * stride_am + K) * 2 <= 0x7ffffffe
                                           and (K + (N - 1) * stride_bn) * 2 <= 0x7ffffffe)
        if SHORT_PACKED_K_LOADS:
            packed_a: tl.constexpr = tlx.layout(
                shape=((8, 8, 4), (8, 2)),
                stride=((8, 64, 512), (1, 2048)),
            )
            packed_b: tl.constexpr = tlx.layout(
                shape=((8, 8, 4), (8, 2)),
                stride=((512, 1, 8), (64, 32)),
            )
        elif PACKED_K_LOADS:
            # Each of 512 threads owns four independent groups of eight K
            # elements. These are consecutive physical elements even when
            # the outer strides are odd. No alignment fact is asserted.
            # A: k = lane_k*8 + value_k; m = lane_m + warp*8 + value_m*64.
            packed_a: tl.constexpr = tlx.layout(
                shape=((8, 8, 8), (8, 4)),
                stride=((8, 64, 512), (1, 4096)),
            )
            # B: k = lane_k*8 + value_k; n = lane_n + warp*8 + value_n*64.
            packed_b: tl.constexpr = tlx.layout(
                shape=((8, 8, 8), (8, 4)),
                stride=((2048, 1, 8), (256, 64)),
            )
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        even_k = K % BLOCK_K == 0
        k_main = K if even_k else (K // BLOCK_K) * BLOCK_K
        if (USE_LDS_RING and a_ptr.dtype.element_ty == tl.float16 and BLOCK_M == 128 and BLOCK_N == 256
                and BLOCK_K == 32 and k_main >= BLOCK_K):
            # A synchronous LDS slot holds tile t while VGPR loads fetch t+1.
            # Shapes and logical B[K,N] ownership remain unchanged. Conservative
            # local ops let membar analysis order stores before all readers,
            # and all readers before refill. No async or warp phase shifts.
            smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, 1)
            smem_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, 1)
            a_slot = tlx.local_view(smem_a, 0)
            b_slot = tlx.local_view(smem_b, 0)
            first_a, first_b = _load_lds_ring_operands_gfx942(
                a_ptr,
                b_ptr,
                reg_m,
                reg_n,
                offs_k,
                0,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                USE_I64_A_OFFSETS,
                USE_I64_B_OFFSETS,
                USE_BUFFER_A,
                USE_BUFFER_B,
            )
            tlx.local_store(a_slot, first_a)
            tlx.local_store(b_slot, first_b)
            # Software staging owns prefetch; disable compiler staging.
            for tile in tl.range(1, k_main // BLOCK_K, num_stages=1):
                next_a, next_b = _load_lds_ring_operands_gfx942(
                    a_ptr,
                    b_ptr,
                    reg_m,
                    reg_n,
                    offs_k,
                    tile * BLOCK_K,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    USE_I64_A_OFFSETS,
                    USE_I64_B_OFFSETS,
                    USE_BUFFER_A,
                    USE_BUFFER_B,
                )
                a = tlx.local_load(a_slot)
                b = tlx.local_load(b_slot)
                acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
                # Consume tile t's registers before overwriting its LDS slot
                # with t+1. Membar analysis protects cross-wave read/refill.
                tlx.local_store(a_slot, next_a)
                tlx.local_store(b_slot, next_b)
            # Drain the final full tile exactly once, even for one full tile.
            a = tlx.local_load(a_slot)
            b = tlx.local_load(b_slot)
            acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
        else:
            LOOP_STAGES: tl.constexpr = 1 if USE_SINGLE_STAGE_POINTER_LOADS else None
            for k in tl.range(0, k_main, BLOCK_K, num_stages=LOOP_STAGES):
                if USE_I64_A_OFFSETS:
                    a_offsets = (reg_m.to(tl.int64)[:, None] * stride_am +
                                 (k + offs_k[None, :]).to(tl.int64) * stride_ak)
                else:
                    a_offsets = reg_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
                if USE_I64_B_OFFSETS:
                    b_offsets = ((k + offs_k[:, None]).to(tl.int64) * stride_bk +
                                 reg_n.to(tl.int64)[None, :] * stride_bn)
                else:
                    b_offsets = (k + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn
                if PACKED_K_LOADS or SHORT_PACKED_K_LOADS:
                    # Pin only the global-load ownership boundary; tl.dot
                    # retains its native consumer layout and synchronization.
                    a_offsets = tlx.require_layout(a_offsets, packed_a)
                    b_offsets = tlx.require_layout(b_offsets, packed_b)
                    a = tlx.buffer_load(a_ptr, a_offsets, contiguity=8)
                    b = tlx.buffer_load(b_ptr, b_offsets, contiguity=8)
                elif UNALIGNED_K_LOADS:
                    # The hints let coalescing give each thread eight K elements;
                    # the loads become buffer loads, which carry no alignment.
                    a_offsets = tl.max_contiguous(tl.multiple_of(a_offsets, [1, 8]), [1, 8])
                    b_offsets = tl.max_contiguous(tl.multiple_of(b_offsets, [8, 1]), [8, 1])
                    a = tl.load(a_ptr + a_offsets)
                    b = tl.load(b_ptr + b_offsets)
                elif USE_BUFFER_A:
                    a = tlx.buffer_load(a_ptr, a_offsets, contiguity=1)
                else:
                    a = tl.load(a_ptr + a_offsets)
                if not (PACKED_K_LOADS or SHORT_PACKED_K_LOADS or UNALIGNED_K_LOADS):
                    if USE_BUFFER_B:
                        b = tlx.buffer_load(b_ptr, b_offsets, contiguity=1)
                    else:
                        b = tl.load(b_ptr + b_offsets)
                acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
        if not even_k:
            # Small FP16 deep-K remainders need only one K=16 MFMA tile.
            # Masked lanes zero-fill the rest of that tile.
            if a_ptr.dtype.element_ty == tl.float16 and K > 256 and BLOCK_K > 16 and K % BLOCK_K <= 16:
                tail_k = tl.arange(0, 16).to(tl.int32)
            else:
                tail_k = offs_k
            if USE_I64_A_OFFSETS:
                a_offsets = (reg_m.to(tl.int64)[:, None] * stride_am +
                             (k_main + tail_k[None, :]).to(tl.int64) * stride_ak)
            else:
                a_offsets = reg_m[:, None] * stride_am + (k_main + tail_k[None, :]) * stride_ak
            if USE_I64_B_OFFSETS:
                b_offsets = ((k_main + tail_k[:, None]).to(tl.int64) * stride_bk +
                             reg_n.to(tl.int64)[None, :] * stride_bn)
            else:
                b_offsets = (k_main + tail_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn
            tail = tail_k < K - k_main
            # Masked buffer loads do not supply the explicit zero fill required
            # by the final dot tile. Keep the ragged tail on generic loads.
            a = tl.load(a_ptr + a_offsets, mask=tail[None, :], other=0.0)
            b = tl.load(b_ptr + b_offsets, mask=tail[:, None], other=0.0)
            acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
        idx_m = rows[:, None]
        idx_n = cols[None, :]
        mask = (idx_m < M) & (idx_n < N)
        if ADD_BIAS:
            bias = tl.load(
                bias_ptr + idx_m * stride_bias_m + idx_n * stride_bias_n,
                mask=mask,
                eviction_policy="evict_last",
            )
            acc += bias.to(tl.float32)
        if USE_I64_C_OFFSETS:
            c_offsets = idx_m.to(tl.int64) * stride_cm + idx_n.to(tl.int64) * stride_cn
        else:
            c_offsets = idx_m * stride_cm + idx_n * stride_cn
        if (PACK_FP16_STORES and c_ptr.dtype.element_ty == tl.float16 and stride_cn == 1 and stride_cm % 2 == 1
                and BLOCK_N == 256 and not USE_I64_C_OFFSETS):
            # Pack two already-rounded FP16 values into each physical dword.
            # A pair owns both elements only when both are inside this tile
            # and the output. Unpaired first/last elements have scalar owners.
            values = acc.to(tl.float16)
            shift = (rows * stride_cm) & 1
            pairs = tl.arange(0, BLOCK_N // 2).to(tl.int32)
            low_cols = 2 * pairs[None, :] + shift[:, None]
            high_cols = low_cols + 1
            low = tl.gather(values, tl.minimum(low_cols, BLOCK_N - 1), axis=1)
            high = tl.gather(values, tl.minimum(high_cols, BLOCK_N - 1), axis=1)
            words = (low.to(tl.uint16, bitcast=True).to(tl.uint32) |
                     (high.to(tl.uint16, bitcast=True).to(tl.uint32) << 16))
            aligned_base = tl.multiple_of(rows * stride_cm + shift, 2)
            word_offsets = (aligned_base[:, None] + pid_n * BLOCK_N) // 2 + pairs[None, :]
            pair_mask = ((rows[:, None] < M) & (high_cols < BLOCK_N) & (pid_n * BLOCK_N + high_cols < N))
            word_ptr = c_ptr.to(tl.pointer_type(tl.uint32))
            tl.store(word_ptr + word_offsets, words, mask=pair_mask)

            first = tl.reshape(
                tl.gather(values, tl.full((BLOCK_M, 1), 0, tl.int32), axis=1),
                (BLOCK_M, ),
            )
            tl.store(c_ptr + rows * stride_cm + pid_n * BLOCK_N, first,
                     mask=(rows < M) & (shift == 1) & (pid_n * BLOCK_N < N))
            valid_width = tl.minimum(BLOCK_N, N - pid_n * BLOCK_N)
            last_col = valid_width - 1
            last = tl.reshape(
                tl.gather(values,
                          tl.full((BLOCK_M, 1), 0, tl.int32) + tl.maximum(last_col, 0), axis=1),
                (BLOCK_M, ),
            )
            # The last element is scalar exactly when it starts an incomplete
            # pair. Exclude the column-zero scalar owner's singleton case.
            last_mask = ((rows < M) & (last_col >= shift) & (((last_col - shift) & 1) == 0))
            tl.store(c_ptr + rows * stride_cm + pid_n * BLOCK_N + last_col, last, mask=last_mask)
        elif (ALIGN_FP16_STORES and c_ptr.dtype.element_ty == tl.float16 and stride_cn == 1 and stride_cm % 2 == 1
              and BLOCK_N == 256 and not USE_I64_C_OFFSETS):
            # Narrow before the epilogue permutation. Odd rows skip column
            # zero in the vector path so every row base is half2 aligned.
            # The skipped value has one separate, masked scalar owner.
            values = acc.to(tl.float16)
            shift = (rows * stride_cm) & 1
            local_cols = tl.arange(0, BLOCK_N).to(tl.int32)
            gather_cols = local_cols[None, :] + shift[:, None]
            shifted = tl.gather(values, tl.minimum(gather_cols, BLOCK_N - 1), axis=1)
            aligned_base = tl.multiple_of(rows * stride_cm + shift, 2)
            aligned_offsets = aligned_base[:, None] + pid_n * BLOCK_N + local_cols[None, :]
            shifted_cols = pid_n * BLOCK_N + gather_cols
            shifted_mask = ((rows[:, None] < M) & (gather_cols < BLOCK_N) & (shifted_cols < N))
            tl.store(c_ptr + aligned_offsets, shifted, mask=shifted_mask)
            first = tl.reshape(
                tl.gather(values, tl.full((BLOCK_M, 1), 0, tl.int32), axis=1),
                (BLOCK_M, ),
            )
            first_offsets = rows * stride_cm + pid_n * BLOCK_N
            tl.store(c_ptr + first_offsets, first, mask=(rows < M) & (shift == 1) & (pid_n * BLOCK_N < N))
        else:
            # Match the vendor's streaming stores on the large
            # BF16 output. The same mask and logical elements are written.
            streaming_bf16: tl.constexpr = (USE_STREAMING_BF16_STORES and (M, N, K) == (819200, 1024, 192)
                                            and tlx.dtype_of(c_ptr) == tl.bfloat16)
            tl.store(c_ptr + c_offsets, acc, mask=mask, cache_modifier=".cs" if streaming_bf16 else "")


@triton.jit
def _global_split_matmul_kernel_gfx942(
    a_ptr,
    b_ptr,
    workspace_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SPLIT_ORDER_CHUNK: tl.constexpr = 1,
    LAUNCH_XCD_CHUNK: tl.constexpr = 0,
):
    # Balance contiguous K-block ranges: the first EXTRA_BLOCKS splits
    # receive one extra block, so ranges differ by at most BLOCK_K.
    # Each (output tile, split) still exclusively owns its fp32 partial.
    # Only the final range can contain a ragged block; its constexpr
    # length retains the helper's masked, zero-filled tail path.
    K_BLOCKS: tl.constexpr = tl.cdiv(K, BLOCK_K)
    BASE_BLOCKS: tl.constexpr = K_BLOCKS // SPLIT_K
    EXTRA_BLOCKS: tl.constexpr = K_BLOCKS % SPLIT_K
    SHORT_K: tl.constexpr = BASE_BLOCKS * BLOCK_K
    LONG_K: tl.constexpr = (BASE_BLOCKS + 1) * BLOCK_K
    FINAL_K: tl.constexpr = K - ((SPLIT_K - 1) * BASE_BLOCKS + EXTRA_BLOCKS) * BLOCK_K
    tl.static_assert(BASE_BLOCKS > 0)
    tl.static_assert(FINAL_K > 0)
    tl.static_assert(workspace_ptr.dtype.element_ty == tl.float32)
    # Use the existing single-stage pointer producer for N=512 and N=2560.
    # Slice addressing, fp32 partial ownership and reducer ordering stay
    # identical for full and final slices; ragged K loads still zero-fill.
    POINTER_LOADS: tl.constexpr = (4178 <= M and M <= 4942 and (N == 512 or N == 2560) and K == 18688
                                   and tlx.dtype_of(a_ptr) == tl.float16 and tlx.dtype_of(b_ptr) == tl.float16
                                   and stride_am == K and stride_ak == 1 and stride_bk == 1 and stride_bn == K
                                   and BLOCK_M == 128 and BLOCK_N == 128 and BLOCK_K == 128)
    TILES: tl.constexpr = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    flat = tl.program_id(0)
    # Permute complete eight-XCD launch groups; leave the short tail
    # unchanged so the padded launch remains a bijection.
    if LAUNCH_XCD_CHUNK > 0:
        total: tl.constexpr = tl.cdiv(TILES, SPLIT_ORDER_CHUNK) * SPLIT_ORDER_CHUNK * SPLIT_K
        aligned: tl.constexpr = (total // (8 * LAUNCH_XCD_CHUNK)) * (8 * LAUNCH_XCD_CHUNK)
        if flat < aligned:
            xcd = flat % 8
            local = flat // 8
            group = (local // LAUNCH_XCD_CHUNK) * (8 * LAUNCH_XCD_CHUNK)
            flat = group + xcd * LAUNCH_XCD_CHUNK + local % LAUNCH_XCD_CHUNK
    split = (flat // SPLIT_ORDER_CHUNK) % SPLIT_K
    tile = (flat // (SPLIT_ORDER_CHUNK * SPLIT_K)) * SPLIT_ORDER_CHUNK + flat % SPLIT_ORDER_CHUNK
    if tile < TILES:
        # prefix(split) = split * BASE_BLOCKS + min(split, EXTRA_BLOCKS).
        # Consecutive prefixes partition [0, K_BLOCKS), with no gaps/overlap.
        # Loads only read immutable A/B; all valid partial elements are stored
        # once before the current-stream reducer consumes the complete planes.
        start_k = (split * BASE_BLOCKS + tl.minimum(split, EXTRA_BLOCKS)) * BLOCK_K
        a_slice = a_ptr + start_k * stride_ak
        b_slice = b_ptr + start_k * stride_bk
        partial = workspace_ptr + split * M * N
        if split == SPLIT_K - 1:
            _direct_matmul_kernel_gfx942(
                a_slice,
                b_slice,
                partial,
                partial,
                M,
                N,
                FINAL_K,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                0,
                0,
                N,
                1,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                NUM_XCDS,
                XCD_CHUNK,
                False,
                USE_SINGLE_STAGE_POINTER_LOADS=POINTER_LOADS,
                TILE_ID=tile,
                OVERRIDE_TILE_ID=True,
            )
        elif split < EXTRA_BLOCKS:
            _direct_matmul_kernel_gfx942(
                a_slice,
                b_slice,
                partial,
                partial,
                M,
                N,
                LONG_K,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                0,
                0,
                N,
                1,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                NUM_XCDS,
                XCD_CHUNK,
                False,
                USE_SINGLE_STAGE_POINTER_LOADS=POINTER_LOADS,
                TILE_ID=tile,
                OVERRIDE_TILE_ID=True,
            )
        else:
            _direct_matmul_kernel_gfx942(
                a_slice,
                b_slice,
                partial,
                partial,
                M,
                N,
                SHORT_K,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                0,
                0,
                N,
                1,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                NUM_XCDS,
                XCD_CHUNK,
                False,
                USE_SINGLE_STAGE_POINTER_LOADS=POINTER_LOADS,
                TILE_ID=tile,
                OVERRIDE_TILE_ID=True,
            )


@triton.jit
def matmul_kernel_gfx942(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    USE_I64_A_OFFSETS: tl.constexpr = False,
    USE_I64_B_OFFSETS: tl.constexpr = False,
    USE_I64_C_OFFSETS: tl.constexpr = False,
    SPLIT_M_128_32: tl.constexpr = False,
    USE_LOCAL_SPLIT_U: tl.constexpr = False,
    LOCAL_SPLIT_U: tl.constexpr = 1,
    K_WIDTH: tl.constexpr = 1,
    USE_LDS_RING: tl.constexpr = False,
    USE_PACKED_K_LOADS: tl.constexpr = False,
    SPLIT_M_128_64: tl.constexpr = False,
    ALIGN_FP16_STORES: tl.constexpr = False,
    PACK_FP16_STORES: tl.constexpr = False,
    USE_N_EDGE_PANELS: tl.constexpr = False,
    USE_N_INTERIOR_PANELS: tl.constexpr = False,
    USE_PIPELINED_POINTER_LOADS: tl.constexpr = False,
    SPLIT_K: tl.constexpr = 1,
    SPLIT_ORDER_CHUNK: tl.constexpr = 1,
    LAUNCH_XCD_CHUNK: tl.constexpr = 0,
    USE_STREAMING_BF16_STORES: tl.constexpr = False,
):
    """Dispatch one autotune candidate to its selected GEMM implementation."""
    if SPLIT_K > 1:
        _global_split_matmul_kernel_gfx942(
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_M,
            NUM_XCDS,
            XCD_CHUNK,
            SPLIT_K,
            SPLIT_ORDER_CHUNK,
            LAUNCH_XCD_CHUNK,
        )
    elif USE_LOCAL_SPLIT_U:
        _local_split_u_kernel_gfx942(
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M,
            BLOCK_N,
            K_WIDTH,
            BLOCK_K,
            LOCAL_SPLIT_U,
        )
    else:
        _direct_matmul_kernel_gfx942(
            a_ptr,
            b_ptr,
            bias_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_bias_m,
            stride_bias_n,
            stride_cm,
            stride_cn,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_M,
            NUM_XCDS,
            XCD_CHUNK,
            ADD_BIAS,
            USE_I64_A_OFFSETS,
            USE_I64_B_OFFSETS,
            USE_I64_C_OFFSETS,
            SPLIT_M_128_32,
            USE_LDS_RING,
            USE_PACKED_K_LOADS,
            SPLIT_M_128_64,
            ALIGN_FP16_STORES,
            PACK_FP16_STORES,
            USE_N_EDGE_PANELS,
            USE_N_INTERIOR_PANELS,
            USE_PIPELINED_POINTER_LOADS=USE_PIPELINED_POINTER_LOADS,
            USE_STREAMING_BF16_STORES=USE_STREAMING_BF16_STORES,
        )


def _config(block_m, block_n, block_k, group_m, num_warps, *, waves_per_eu=0, kpack=1, split_m_128_32=False, split_k=1):
    # This overlaps register-staged global loads; it is not an explicit
    # two-buffer LDS allocation.
    meta = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "NUM_XCDS": 8,
        "XCD_CHUNK": 8,
        "waves_per_eu": waves_per_eu,
        "kpack": kpack,
    }
    if split_k > 1:
        meta["SPLIT_K"] = split_k
    if split_m_128_32:
        meta["SPLIT_M_128_32"] = True
    return triton.Config(meta, num_warps=num_warps, num_stages=2)


def _local_split_u_config(plan):
    return triton.Config(
        {
            "BLOCK_M": plan.tile_m,
            "BLOCK_N": plan.tile_n,
            "BLOCK_K": plan.wave_k,
            "GROUP_M": 1,
            "NUM_XCDS": 1,
            "XCD_CHUNK": 1,
            "USE_LOCAL_SPLIT_U": True,
            "LOCAL_SPLIT_U": plan.local_split_u,
            "K_WIDTH": plan.k_width,
            "waves_per_eu": 0,
            "kpack": 1,
        },
        num_warps=plan.local_split_u,
        num_stages=1,
    )


def _configs():
    """Curated ROCm search space plus every incumbent TLX configuration."""
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, waves_per_eu)
    candidates = [
        (16, 16, 256, 4, 4, 2),
        (32, 16, 256, 4, 4, 0),
        (32, 32, 16, 8, 4, 2),
        (32, 32, 128, 8, 4, 0),
        (32, 64, 64, 8, 4, 0),
        (64, 16, 128, 8, 4, 2),
        (64, 32, 32, 8, 4, 0),
        (64, 32, 64, 8, 4, 0),
        (64, 32, 64, 8, 8, 0),
        (64, 32, 128, 8, 4, 0),
        (64, 64, 16, 8, 4, 0),
        (64, 64, 64, 4, 4, 0),
        (64, 64, 128, 16, 8, 0),
        (64, 64, 256, 4, 8, 0),
        (64, 64, 256, 16, 8, 0),
        (64, 128, 32, 4, 4, 2),
        (64, 128, 32, 8, 8, 0),
        (64, 128, 64, 4, 8, 0),
        (64, 128, 128, 4, 8, 0),
        (128, 32, 32, 8, 4, 0),
        (128, 32, 64, 8, 4, 0),
        (128, 64, 32, 8, 4, 2),
        (128, 64, 64, 16, 4, 0),
        (128, 64, 128, 4, 8, 0),
        (128, 128, 32, 16, 4, 2),
        (128, 128, 32, 16, 8, 0),
        (128, 128, 32, 16, 8, 2),
        (128, 128, 64, 16, 4, 0),
        (128, 128, 64, 8, 8, 0),
        (128, 128, 128, 16, 8, 0),
        (128, 256, 32, 16, 4, 2),
        (128, 256, 64, 4, 8, 0),
        (256, 64, 64, 4, 8, 0),
        (256, 128, 32, 4, 4, 2),
        (256, 128, 32, 16, 8, 0),
        (256, 128, 64, 4, 8, 0),
        (256, 256, 64, 4, 8, 0),
        # Test adjacent N tiles for A reuse in tall, narrow-output GEMMs.
        (256, 256, 64, 1, 8, 0),
        # Preserve the original compact TLX search space as a strict subset.
        (64, 64, 128, 8, 8, 0),
        (128, 64, 64, 4, 8, 0),
        (64, 128, 64, 8, 8, 0),
        (128, 128, 32, 8, 4, 0),
        (256, 128, 32, 8, 8, 0),
        (128, 256, 32, 8, 8, 0),
        (256, 256, 64, 8, 8, 0),
    ]
    return [
        _config(
            block_m,
            block_n,
            block_k,
            group_m,
            num_warps,
            waves_per_eu=waves_per_eu,
        ) for block_m, block_n, block_k, group_m, num_warps, waves_per_eu in candidates
    ] + [
        _config(128, 128, 64, 16, 4, kpack=2),
        _config(64, 64, 256, 4, 8, kpack=2),
        # Match an incumbent tile; change only its operand staging mechanism.
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_M": 16,
                "NUM_XCDS": 8,
                "XCD_CHUNK": 8,
                "waves_per_eu": 2,
                "kpack": 1,
                "USE_LDS_RING": True,
            },
            num_warps=4,
            num_stages=1,
        ),
        # Preserve the winning tile and staging; test packed K load ownership.
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "GROUP_M": 1,
                "NUM_XCDS": 8,
                "XCD_CHUNK": 8,
                "waves_per_eu": 0,
                "kpack": 1,
                "USE_PACKED_K_LOADS": True,
            },
            num_warps=8,
            num_stages=2,
        )
    ]


CONFIGS = _configs


def _smoke_configs():
    return [_config(64, 64, 64, 4, 4), _config(128, 128, 32, 8, 4)]


SMOKE_CONFIGS = _smoke_configs


def heuristic_config(M, N, K, *, enable_global_split_k=True, enable_short_m_tuning=True):
    """Choose one configuration without runtime autotuning."""
    # Shared-B row panels reduce padding for the measured MLP projection.
    if (M, N, K) == (4662, 2560, 1280):
        config = _config(160, 256, 64, 4, 8, split_m_128_32=True)
        config.kwargs["XCD_CHUNK"] = 1
        return [config]
    # Short aligned reductions favor BK128, kpack2, and more output workgroups.
    if (enable_short_m_tuning and 384 < M <= 1024 and 2048 <= N <= 4096 and 2048 <= K <= 4096 and K % 128 == 0
            and (N <= 2048 or M <= 512) and (M > 512 or N >= 4096 or K >= 4096)):
        config = _config(64, 64, 128, 8, 4, kpack=2)
        config.kwargs["USE_PIPELINED_POINTER_LOADS"] = True
        return [config]
    # Tall shallow reductions benefit from streaming output stores.
    if M >= 65536 and 512 <= N <= 2048 and K <= 256:
        config = _config(256, 128, 32, 16, 8)
        config.kwargs.update(USE_STREAMING_BF16_STORES=True, USE_PIPELINED_POINTER_LOADS=True)
        return [config]
    # Compact deep reductions favor BK256 with four warps and pointer loads.
    if 2048 < M <= 4096 and 1024 <= N <= 2048 and K >= 16 * N and K % 256 == 0:
        config = _config(64, 64, 256, 4, 4)
        config.kwargs["USE_PIPELINED_POINTER_LOADS"] = True
        return [config]
    # Wide K-dominant row grids favor four warps and kpack2.
    if 768 < M <= 1024 and N >= 4096 and K >= 2 * N and K % 64 == 0:
        config = _config(128, 128, 64, 16, 4, kpack=2)
        config.kwargs["USE_PIPELINED_POINTER_LOADS"] = True
        return [config]
    # Tall aligned deep reductions favor rectangular tiles and pointer loads.
    if M >= 16384 and N >= 1024 and K > 256 and K >= 2 * N and K % 32 == 0:
        config = _config(128, 256, 32, 16, 4)
        config.kwargs["USE_PIPELINED_POINTER_LOADS"] = True
        return [config]
    # Shared-B row panels reduce padding for the measured unsplit-K MLP target.
    if (M, N, K) == (4662, 2560, 2560):
        config = _config(160, 256, 64, 4, 8, split_m_128_32=True)
        config.kwargs["XCD_CHUNK"] = 1
        return [config]
    # Offline paired cold-cache trials select the K step and output tile
    # within the existing compiler-managed pointer pipeline. Retain the
    # incumbent below for rows whose trials were inconclusive.
    short_m_pipeline_configs = {
        (902, 4096, 6656): (128, 128, 128, 8),
        (902, 4096, 2048): (128, 128, 128, 8),
        (902, 4096, 1091): (128, 128, 64, 8),
        (440, 2048, 4096): (64, 64, 128, 8),
        (440, 2048, 2048): (64, 64, 256, 8),
    }
    if enable_short_m_tuning and (M, N, K) in short_m_pipeline_configs:
        block_m, block_n, block_k, num_warps = short_m_pipeline_configs[M, N, K]
        config = _config(block_m, block_n, block_k, 8, num_warps)
        config.kwargs["USE_PIPELINED_POINTER_LOADS"] = True
        return [config]
    # Offline cold-cache trials choose output granularity for the short-M
    # family. Keep the square incumbent where smaller tiles were inconclusive.
    short_m_tiles = {
        (902, 2048, 4096): (64, 64, 4),
        (902, 4096, 7552): (128, 128, 8),
        (902, 4096, 6656): (128, 128, 8),
        (902, 4096, 2048): (128, 128, 8),
        (902, 2048, 2048): (64, 64, 4),
        (440, 2048, 4096): (64, 64, 4),
        (440, 4096, 6400): (64, 64, 4),
        (440, 4096, 2048): (64, 64, 4),
        (440, 2048, 2048): (64, 64, 8),
        (440, 4096, 1093): (64, 64, 4),
    }
    if (M, N, K) in short_m_tiles:
        block_m, block_n, num_warps = short_m_tiles[M, N, K]
        config = _config(block_m, block_n, 64, 8, num_warps)
        config.kwargs["USE_PIPELINED_POINTER_LOADS"] = True
        return [config]
    if enable_global_split_k and _global_split_k_shape(M, N, K):
        # Balance the narrow grids within two waves of the 304-CU device.
        # The wider grid favors fewer partial planes after 38 row tiles.
        row_tiles = triton.cdiv(M, 128)
        split_k = {256: 8, 512: 4, 1280: 3, 2560: 4}[N]
        if N in (256, 512):
            split_k = min(split_k, (2 * 304) // (row_tiles * triton.cdiv(N, 128)))
        elif N == 2560 and row_tiles > 38:
            split_k = 2
        config = _config(128, 128, 128, 16, 4, split_k=split_k)
        # Offline paired trials select the split-work order and B reuse group.
        # These options are confined to the guarded FP16 production family.
        group_m, num_xcds, chunk = 16, 8, 2
        if N == 256:
            num_xcds, chunk = 1, 4
        elif N == 512:
            if row_tiles <= 34:
                group_m, num_xcds, chunk = 4, 1, 1
            elif row_tiles > 38:
                group_m, num_xcds = 1, 1
        elif N == 1280:
            group_m, num_xcds = 8, 1
            if row_tiles > 38:
                group_m, chunk = 4, 1
        elif row_tiles <= 34:
            chunk = 32
        config.kwargs.update(
            GROUP_M=group_m,
            NUM_XCDS=num_xcds,
            SPLIT_ORDER_CHUNK=chunk,
            enable_sched_group_barrier_scheduler=True,
            sched_group_barrier_mfma_per_dwordx4=2,
        )
        if N == 512 and row_tiles > 34:
            config.kwargs.update(LAUNCH_XCD_CHUNK=8, NUM_XCDS=1, GROUP_M=16,
                                 SPLIT_ORDER_CHUNK=1 if row_tiles <= 35 else 2)
        return [config]
    # Offline cold-cache trials favor square tiles with a longer K step
    # for the eight shallow-K production rows. Bound this rule to the
    # production M range and exact (N, K) pairs; select one config only.
    if 4178 <= M <= 4942 and (N, K) in (
        (18688, 512),
        (2560, 2560),
        (8192, 2560),
        (18688, 256),
        (2560, 1280),
        (1280, 2560),
        (8192, 1280),
        (2048, 2560),
    ):
        config = _config(128, 128, 64, 16, 4)
        if (N, K) not in ((18688, 256), (8192, 1280)):
            config.kwargs["USE_PIPELINED_POINTER_LOADS"] = True
        return [config]
    # This measured large-N case uses the split-panel algorithm to share each B tile across 160 output rows.
    if (M, N, K) == (2048, 10240, 25408):
        return [_config(160, 512, 32, 8, 8, split_m_128_32=True)]
    # Very short M workloads use small direct tiles when they are not intercepted by the measured LocalSplitU path.
    if M <= 16:
        return [_config(32, 32, 128, 8, 4)]
    # Small-M, shallow-K workloads have enough N tiles to favor the efficient 128-square MFMA shape.
    if M <= 384 and K <= 2048:
        return [_config(128, 128, 128, 16, 8)]
    # Small-M workloads with both a narrow output and deep K favor a compact tile with a long K step.
    if M <= 384 and N <= 2048:
        return [_config(64, 64, 256, 4, 8)]
    # Small-M, wide-output reductions with deep K favor the measured 64-square tile with a K=256 step.
    if M <= 256 and N >= 4096 and K >= 4096:
        return [_config(64, 64, 256, 4, 8)]
    # Remaining small-M workloads favor a rectangular tile that limits masked rows while retaining N reuse.
    if M <= 384:
        return [_config(64, 128, 128, 4, 8)]
    # Moderately short M workloads expose enough row tiles for the high-reuse 128-square K=128 configuration.
    if M <= 768:
        return [_config(128, 128, 128, 16, 8)]
    # At M near 1024, K-dominant matrices favor square output tiles and a K=64 reduction step.
    if M <= 1024 and K >= 2 * N:
        return [_config(128, 128, 64, 8, 8)]
    # At M near 1024, strongly N-dominant matrices favor four-wave 128-square tiles with a short K step.
    if M <= 1024 and N >= 2 * K:
        return [_config(128, 128, 32, 8, 4)]
    # Remaining M-near-1024 matrices benefit from a wider N tile without enlarging the M tile.
    if M <= 1024:
        return [_config(128, 256, 64, 4, 8)]
    # Narrow-output, deep-reduction matrices need compact output tiles to expose sufficient parallelism.
    if M <= 4096 and N <= 2048 and K >= 16 * N:
        return [_config(64, 64, 256, 4, 8)]
    # M-near-2048 matrices with a strongly K-dominant aspect ratio favor four-wave 128-square tiles.
    if M <= 2048 and K >= 4 * N:
        return [_config(128, 128, 64, 16, 4)]
    # M-near-2048 matrices with a moderately K-dominant aspect ratio favor the larger K=128 step.
    if M <= 2048 and K >= 2 * N:
        return [_config(128, 128, 128, 16, 8)]
    # M-near-2048 matrices with shallow K favor wide 256-square output tiles.
    if M <= 2048 and K <= 1024:
        return [_config(256, 256, 64, 4, 8)]
    # Remaining M-near-2048 matrices have enough output work to amortize 256-square tiles.
    if M <= 2048:
        return [_config(256, 256, 64, 8, 8)]
    # M-near-4096 matrices with extremely deep K and modest N need compact tiles and a K=256 step.
    if M <= 4096 and K >= 16 * N:
        return [_config(64, 64, 256, 4, 8)]
    # Other M-near-4096 matrices favor a wide N tile and four-wave scheduling.
    if M <= 4096:
        return [_config(128, 256, 32, 16, 4, waves_per_eu=2)]
    # Tall deep-reduction workloads need output-width-specific choices because the measured 256-square default has substantial regret.
    if K > 256:
        # Outputs wider than 256 with K-dominant reductions use the measured rectangular winner.
        if N > 256 and K > N:
            return [_config(128, 256, 32, 16, 4, waves_per_eu=2)]
        # At most 32 output columns favor the measured 32-column tile without padding to a square output block.
        if N <= 32:
            return [_config(128, 32, 64, 8, 4)]
        # Up to 48 output columns favor the measured 128-row, 64-column tile with eight waves.
        if N <= 48:
            return [_config(128, 64, 64, 4, 8)]
        # Outputs between 49 and 64 columns favor the measured four-wave 64-square configuration.
        if N <= 64:
            return [_config(64, 64, 64, 4, 4)]
        # At most 128 output columns favor the measured four-wave square tile with a short K step.
        if N <= 128:
            return [_config(128, 128, 32, 8, 4)]
        # Outputs between 129 and 256 columns favor the measured 64-column tile across multiple output blocks.
        if N <= 256:
            return [_config(128, 64, 64, 4, 8)]
    # Very large M with shallow K favors a wide N tile to reduce the number of output workgroups.
    if M >= 1048576:
        return [_config(128, 256, 32, 16, 4, waves_per_eu=2)]
    # Tall matrices with a very shallow reduction favor a larger M tile and narrower N tile.
    if K <= 256:
        return [_config(256, 128, 32, 16, 8)]
    # Tall matrices with shallow K and N-dominant output favor four-wave rectangular tiles.
    if N >= 2 * K:
        return [_config(128, 256, 32, 16, 4, waves_per_eu=2)]
    # Tall matrices with narrow N favor square 256 tiles to maximize reuse across output rows.
    if N <= 256:
        return [_config(256, 256, 64, 8, 8)]
    return [_config(256, 256, 64, 4, 8)]


def _candidate_configs(shape, enable_local_split_u=False, enable_global_split_k=False):
    """Return the candidate universe, including a shape-specific incumbent."""
    configs = CONFIGS()
    if shape == (262144, 1195, 2373):
        # Append the two-panel decomposition without removing incumbents.
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": 192,
                    "BLOCK_N": 256,
                    "BLOCK_K": 64,
                    "GROUP_M": 1,
                    "NUM_XCDS": 8,
                    "XCD_CHUNK": 8,
                    "waves_per_eu": 0,
                    "kpack": 1,
                    "SPLIT_M_128_64": True,
                },
                num_warps=8,
                num_stages=2,
            ))
    if shape == (262144, 1195, 2373):
        # Retain all incumbents; test only the parity-aligned FP16 epilogue.
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": 256,
                    "BLOCK_N": 256,
                    "BLOCK_K": 64,
                    "GROUP_M": 1,
                    "NUM_XCDS": 8,
                    "XCD_CHUNK": 8,
                    "waves_per_eu": 0,
                    "kpack": 1,
                    "USE_PACKED_K_LOADS": True,
                    "ALIGN_FP16_STORES": True,
                },
                num_warps=8,
                num_stages=2,
            ))
    if shape == (262144, 1195, 2373):
        # Test explicit dword stores without changing incumbent load ownership.
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": 256,
                    "BLOCK_N": 256,
                    "BLOCK_K": 64,
                    "GROUP_M": 1,
                    "NUM_XCDS": 8,
                    "XCD_CHUNK": 8,
                    "waves_per_eu": 0,
                    "kpack": 1,
                    "USE_PACKED_K_LOADS": True,
                    "ALIGN_FP16_STORES": True,
                    "PACK_FP16_STORES": True,
                },
                num_warps=8,
                num_stages=2,
            ))
    if shape == (262144, 1195, 2373):
        # Retain the square incumbent; opt in only to reduced right-edge work.
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": 256,
                    "BLOCK_N": 256,
                    "BLOCK_K": 64,
                    "GROUP_M": 1,
                    "NUM_XCDS": 8,
                    "XCD_CHUNK": 8,
                    "waves_per_eu": 0,
                    "kpack": 1,
                    "USE_PACKED_K_LOADS": True,
                    "ALIGN_FP16_STORES": True,
                    "USE_N_EDGE_PANELS": True,
                },
                num_warps=8,
                num_stages=2,
            ))
    if shape == (262144, 1195, 2373):
        # Extend the shared-A N-panel decomposition to full interior tiles.
        # All existing configurations and the 128+64 right edge remain intact.
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": 256,
                    "BLOCK_N": 256,
                    "BLOCK_K": 64,
                    "GROUP_M": 1,
                    "NUM_XCDS": 8,
                    "XCD_CHUNK": 8,
                    "waves_per_eu": 0,
                    "kpack": 1,
                    "USE_PACKED_K_LOADS": True,
                    "ALIGN_FP16_STORES": True,
                    "USE_N_EDGE_PANELS": True,
                    "USE_N_INTERIOR_PANELS": True,
                },
                num_warps=8,
                num_stages=2,
            ))
    if enable_local_split_u:
        configs.append(_local_split_u_config(_FULL_LOCAL_SPLIT_U_PLANS[shape]))
    if enable_global_split_k and _global_split_k_shape(*shape):
        # Offline only: retain all unsplit candidates and expose bounded
        # split factors for the new decomposition.
        configs.extend(
            _config(128, 128, block_k, 16, 4, split_k=split) for block_k in (64, 128) for split in (2, 3, 4, 6, 8))
    incumbent = heuristic_config(*shape, enable_global_split_k=enable_global_split_k)[0]
    incumbent_key = (incumbent.kwargs, incumbent.num_warps, incumbent.num_stages)
    if not any((config.kwargs, config.num_warps, config.num_stages) == incumbent_key for config in configs):
        configs.append(incumbent)
    return configs


@functools.lru_cache(maxsize=None)
def _tuned(space, shape=None, enable_local_split_u=False):
    """Autotuned GEMM kernel per search space."""
    if space == "heuristic":
        configs = heuristic_config(*shape, enable_global_split_k=False)
    elif space == "full":
        configs = _candidate_configs(shape, enable_local_split_u)
    elif space == "smoke":
        configs = SMOKE_CONFIGS()
    else:
        raise ValueError(f"Unknown gfx942 MM search space: {space}")
    keys = ["M", "N", "K", "ADD_BIAS"]
    if space == "full":
        keys += ["stride_am", "stride_ak", "stride_bk", "stride_bn"]
    return triton.autotune(configs=configs, key=keys)(matmul_kernel_gfx942)


def _validate_operands(a, b, out):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected A[M, K] and B[K, N], got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("A and B must be on the same GPU")
    if a.dtype != b.dtype:
        raise ValueError("A and B must have the same dtype")
    M, K = a.shape
    N = b.shape[1]
    if out is not None:
        if out.shape != (M, N) or out.device != a.device or out.dtype != a.dtype or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous {a.dtype} tensor with shape ({M}, {N}) on A's device")
    return M, N, K


def _bias_strides(bias, M, N, a):
    if bias.device != a.device or bias.dtype != a.dtype:
        raise ValueError("input must match A's device and dtype")
    if bias.ndim == 1:
        if bias.shape[0] != N:
            raise ValueError(f"1-D addmm input must have shape ({N},), got {tuple(bias.shape)}")
        return 0, bias.stride(0)
    if bias.ndim == 2 and bias.shape[0] in (1, M) and bias.shape[1] in (1, N):
        return (0 if bias.shape[0] == 1 else bias.stride(0), 0 if bias.shape[1] == 1 else bias.stride(1))
    raise ValueError(f"addmm input with shape {tuple(bias.shape)} is not broadcastable to ({M}, {N})")


def _precheck_local_split_u(a, b, bias):
    """Enable LocalSplitU only for selected MM shapes and layouts."""
    shape = (a.shape[0], b.shape[1], a.shape[1])
    return (ENABLE_LOCAL_SPLIT_U and bias is None and a.dtype == torch.float16 and a.stride(1) == 1 and b.stride(0) == 1
            and shape in _FULL_LOCAL_SPLIT_U_PLANS)


def _launch_local_split_u(a, b, out, plan):
    M, K = a.shape
    N = b.shape[1]
    _local_split_u_kernel_gfx942[(triton.cdiv(M, plan.tile_m) * triton.cdiv(N, plan.tile_n), )](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        TILE_M=plan.tile_m,
        TILE_N=plan.tile_n,
        K_WIDTH=plan.k_width,
        WAVE_K=plan.wave_k,
        LOCAL_SPLIT_U=plan.local_split_u,
        num_warps=plan.local_split_u,
        num_stages=1,
        matrix_instr_nonkdim=16,
        waves_per_eu=0,
    )
    return out


def _full_config_key(a, b):
    return (a.device, a.dtype, tuple(a.shape), tuple(b.shape), tuple(a.stride()), tuple(b.stride()))


def _needs_i64_offsets(tensor):
    """Return whether this view can address beyond signed i32 byte offsets."""
    if any(stride < 0 for stride in tensor.stride()):
        return True
    max_element_offset = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return max_element_offset * tensor.element_size() > (1 << 31) - 1


def _global_split_k_shape(M, N, K):
    return ENABLE_GLOBAL_SPLIT_K and 4178 <= M <= 4942 and K == 18688 and N in (256, 512, 1280, 2560)


# Three K splits lift 384 BF16 output tiles (1.3 waves) to 1152 (3.8 waves).
_BF16_SPLIT_K_SHAPES = {(1024, 6144, 20480): 3}


def _bf16_split_k_config(M, N, K):
    return _config(128, 128, 128, 4, 4, split_k=_BF16_SPLIT_K_SHAPES[M, N, K])


def _precheck_global_split_k(a, b, bias, out):
    M, K = a.shape
    N = b.shape[1]
    dtype = a.dtype
    if dtype == torch.float16:
        shape_ok = _global_split_k_shape(M, N, K)
    else:
        shape_ok = dtype == torch.bfloat16 and ENABLE_GLOBAL_SPLIT_K and (M, N, K) in _BF16_SPLIT_K_SHAPES
    return (shape_ok and b.dtype == dtype and out.dtype == dtype and a.stride() == (K, 1) and b.stride() == (1, K)
            and out.is_contiguous()
            and (bias is None or (bias.dtype == dtype and bias.device == a.device and bias.ndim == 1
                                  and bias.shape[0] == N and bias.stride(0) == 1)))


_FULL_GLOBAL_SPLIT_K_CONFIGS = {}


def _launch_global_split_k_config(a, b, bias, out, workspace, config):
    M, K = a.shape
    N = b.shape[1]
    split_k = config.kwargs.get("SPLIT_K", 1)
    meta = dict(config.kwargs)
    meta.update(num_warps=config.num_warps, num_stages=config.num_stages)
    tiles = triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"])
    chunk = meta.get("SPLIT_ORDER_CHUNK", 1)
    grid = (triton.cdiv(tiles, chunk) * chunk * split_k, ) if split_k > 1 else (tiles, )
    target = workspace if split_k > 1 else out
    bias_arg = bias if bias is not None else out
    matmul_kernel_gfx942[grid](
        a,
        b,
        bias_arg,
        target,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        0,
        1,
        N,
        1,
        ADD_BIAS=bias is not None and split_k == 1,
        matrix_instr_nonkdim=16,
        **meta,
    )
    if split_k > 1:
        # Both launches use the current stream: all producers finish before
        # the reducer reads. Every valid workspace element is overwritten
        # each call; no zeroing or atomics, and bias/cast happen exactly once.
        _reduce_k_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
            workspace,
            out,
            bias_arg,
            M,
            N,
            SPLIT_K=split_k,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            OUTPUT_DTYPE=tl.float16 if out.dtype == torch.float16 else tl.bfloat16,
            HAS_BIAS=bias is not None,
            STRIDE_BIAS_M=0,
            STRIDE_BIAS_N=1,
        )
    return out


def _launch_global_split_k(a, b, bias, out, space):
    shape = (a.shape[0], b.shape[1], a.shape[1])
    bf16 = a.dtype == torch.bfloat16
    if space == "heuristic":
        config = _bf16_split_k_config(*shape) if bf16 else heuristic_config(*shape)[0]
        workspace = torch.empty((config.kwargs["SPLIT_K"] * shape[0], shape[1]), device=a.device, dtype=torch.float32)
    else:
        # Explicit offline full-space tuning measures the complete launch
        # pair. Production heuristic dispatch never enters this selection.
        key = (_full_config_key(a, b), tuple(bias.stride()) if bias is not None else None)
        config = _FULL_GLOBAL_SPLIT_K_CONFIGS.get(key)
        workspace = torch.empty((8 * shape[0], shape[1]), device=a.device, dtype=torch.float32)
        if config is None:
            from triton.testing import do_bench_cudagraph
            best_ms = float("inf")
            trials = _candidate_configs(shape, enable_global_split_k=True)
            if bf16:
                trials.append(_bf16_split_k_config(*shape))
            for trial in trials:

                def run():
                    return _launch_global_split_k_config(a, b, bias, out, workspace, trial)

                try:
                    run()
                    elapsed_ms = do_bench_cudagraph(run)
                except triton.OutOfResources:
                    continue
                if elapsed_ms < best_ms:
                    best_ms, config = elapsed_ms, trial
            if config is None:
                raise RuntimeError("No valid gfx942 full-space GEMM configuration")
            _FULL_GLOBAL_SPLIT_K_CONFIGS[key] = config
    return _launch_global_split_k_config(a, b, bias, out, workspace, config)


def _gemm(a, b, bias=None, *, out=None, space="heuristic", invariance=None):
    """Direct-load GEMM with measured LocalSplitU fast paths.

    TODO: let the user specify only LocalSplitU (so long as it's internally consistent).
    """
    # A fixed schedule prunes the LocalSplitU plans; the standard direct-load
    # configs reduce K sequentially.
    fixed = gemm_fixed_split_k(invariance)
    M, N, K = _validate_operands(a, b, out)
    bias_strides = _bias_strides(bias, M, N, a) if bias is not None else (0, 0)
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    if (space in ("heuristic", "full") and (_global_split_k_shape(M, N, K) or (M, N, K) in _BF16_SPLIT_K_SHAPES)
            and _precheck_global_split_k(a, b, bias, out)):
        return _launch_global_split_k(a, b, bias, out, space)

    enable_local_split_u = _precheck_local_split_u(a, b, bias) and fixed is None
    if space == "heuristic" and fixed is None and (M, N, K) in _MEASURED_LOCAL_SPLIT_U_PLANS:
        plan = _MEASURED_LOCAL_SPLIT_U_PLANS[(M, N, K)]
        return _launch_local_split_u(a, b, out, plan)

    def grid(meta):
        if meta.get("USE_LOCAL_SPLIT_U", False):
            return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )

    shape = (M, N, K) if space in ("heuristic", "full") else None
    kernel = _tuned(space, shape, space == "full" and enable_local_split_u)
    fast_configs = None
    fast_key = None
    launch_meta = {}
    if space == "full":
        fast_configs = getattr(kernel, "_tlx_fast_configs", None)
        if fast_configs is None:
            fast_configs = kernel._tlx_fast_configs = {}
        fast_key = (_full_config_key(a, b), bias is not None, bias_strides, fixed)
        cached = fast_configs.get(fast_key)
        if cached is not None and cached[1]:
            config = cached[0]
            if config.kwargs.get("USE_LOCAL_SPLIT_U", False):
                plan = _FULL_LOCAL_SPLIT_U_PLANS[(M, N, K)]
                return _launch_local_split_u(a, b, out, plan)
            # Reuse the measured launch configuration. The first two calls
            # still pass through Autotuner so instrumentation sees the winner.
            kernel.best_config = config
            launch_meta = dict(config.kwargs)
            launch_meta["num_warps"] = config.num_warps
            launch_meta["num_stages"] = config.num_stages

    launch_kernel = matmul_kernel_gfx942 if launch_meta else kernel

    bias_ptr = bias if bias is not None else out
    launch_kernel[grid](
        a,
        b,
        bias_ptr,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias_strides[0],
        bias_strides[1],
        out.stride(0),
        out.stride(1),
        ADD_BIAS=bias is not None,
        USE_I64_A_OFFSETS=_needs_i64_offsets(a),
        USE_I64_B_OFFSETS=_needs_i64_offsets(b),
        USE_I64_C_OFFSETS=_needs_i64_offsets(out),
        matrix_instr_nonkdim=16,
        **launch_meta,
    )
    if fast_configs is not None:
        # Keep one cached Autotuner launch so runtime instrumentation can
        # observe the winner before subsequent calls take the direct path.
        previous = fast_configs.get(fast_key)
        fast_configs[fast_key] = (kernel.best_config, previous is not None)
    return out


def mm(a, b, *, out=None, space="heuristic", invariance=None):
    """Compute ``a @ b`` using the gfx942 direct-load GEMM kernel."""
    return _gemm(a, b, out=out, space=space, invariance=invariance)


# Compatibility entry point used by the kernel-optimization agent.
matmul = mm

_THREADS_LSR_NK = frozenset((
    (2048, 4096),
    (2048, 2048),
    (4096, 2048),
    (4096, 7552),
    (4096, 6656),
    (4096, 6400),
    (4096, 1091),
    (4096, 1093),
))


def _template_config(tile, **extra):
    config = _config(*tile)
    config.kwargs.update(extra)
    return config


# Offline template sweeps, confirmed in tritonbench (>=3% over the shared policy).
# (18688, 512) gains 1.09x GEMM-only with 128x128x64 g4 w8 but loses 2% fused in
# the DCN epilogue, so it keeps the shared policy.
_TORCHTLX_PROD_TILES = {
    (2560, 2560): ((256, 256, 64, 8, 8), {"XCD_CHUNK": 4}),
    (2560, 1280): ((256, 256, 64, 16, 8), {"XCD_CHUNK": 4, "waves_per_eu": 2}),
    (1280, 2560): ((128, 128, 64, 8, 4), {"XCD_CHUNK": 4, "kpack": 2}),
    (8192, 1280): ((256, 256, 64, 4, 8), {"XCD_CHUNK": 4, "waves_per_eu": 2}),
}

# Exact bf16 focus shapes, confirmed against the shared policy on the template.
_TORCHTLX_BF16_TILES = {
    (61440, 2048, 5120): ((256, 256, 64, 4, 8), {"XCD_CHUNK": 16, "waves_per_eu": 2}),
    (1024, 6144, 20480): ((128, 128, 64, 4, 4), {"XCD_CHUNK": 4, "waves_per_eu": 2}),
    (819200, 1024, 192): ((256, 128, 32, 16, 8), {"XCD_CHUNK": 1}),
    (4096, 1894, 242432): ((64, 64, 256, 4, 8), {"XCD_CHUNK": 1}),
}


def _threads_lsr_config(M, N, K):
    # Threads LSR expert GEMMs: bands come from offline template sweeps over the
    # prod M range (100-1518); scheduling knobs are tuned at the median batch.
    # Odd K leaves rows 2-byte aligned, so those rows use packed K loads.
    odd_k = {"PACKED_K_LOADS": True}
    if M <= 256:
        return _template_config((64, 64, 256, 4, 8))
    if M <= 640 and N == 2048:
        if K == 4096:
            return _template_config((64, 64, 128, 8, 4), XCD_CHUNK=4, kpack=2)
        return _template_config((64, 64, 256, 1, 8), XCD_CHUNK=1, kpack=2, waves_per_eu=2)
    if M <= 640 and (N, K) == (4096, 6400):
        return _template_config((64, 128, 128, 4, 8), XCD_CHUNK=8, kpack=2)
    if M <= 640 and K % 64 != 0:
        return _template_config((64, 64, 128, 1, 4), kpack=2, **odd_k)
    if M <= 640 and K == 2048:
        return _template_config((64, 128, 128, 1, 8), XCD_CHUNK=4, kpack=2)
    if M <= 640:
        return _template_config((64, 128, 128, 4, 8))
    if M <= 1024 and K % 64 != 0:
        return _template_config((128, 64, 128, 4, 8), XCD_CHUNK=2, kpack=2, **odd_k)
    if M <= 1024 and (N, K) == (2048, 4096):
        return _template_config((64, 128, 128, 1, 8), XCD_CHUNK=2, kpack=2)
    if M <= 1024 and N == 2048:
        return _template_config((128, 64, 64, 4, 8))
    if M <= 1024 and K in (7552, 6656):
        return _template_config((128, 128, 128, 16, 8), XCD_CHUNK=4, waves_per_eu=2, kpack=2)
    if M <= 1024:
        return _template_config((128, 128, 128, 16, 8))
    if N == 4096 and M <= 1360:
        return _template_config((64, 128, 128, 4, 8))
    return _template_config((128, 128, 128, 16, 8))


def torchtlx_heuristic_config(M, N, K, *, dtype, out_dtype, a_strides, b_strides, alpha=1, beta=1):
    """Share the fixed shape policy with Inductor, preserving scaled epilogues."""
    enable = (dtype == torch.float16 and out_dtype == torch.float16 and a_strides == (K, 1) and b_strides == (1, K)
              and alpha == 1 and beta == 1)
    if enable and (M, N, K) == (4662, 2560, 2560):
        # Shared-B row panels reduce padding while preserving increasing K order.
        return [_template_config((320, 128, 64, 4, 8), NUM_XCDS=1, XCD_CHUNK=1, SPLIT_M_256_64=True)]
    if enable and 4178 <= M <= 4942 and (N, K) in _TORCHTLX_PROD_TILES:
        # The register-only template favors different tiles than the tlx.ops
        # buffer-load path on these production rows.
        tile, extra = _TORCHTLX_PROD_TILES[N, K]
        return [_template_config(tile, **extra)]
    if enable and M <= 1536 and (N, K) in _THREADS_LSR_NK:
        return [_threads_lsr_config(M, N, K)]
    bf16 = (dtype == torch.bfloat16 and out_dtype == torch.bfloat16 and a_strides == (K, 1) and b_strides == (1, K)
            and alpha == 1 and beta == 1)
    if bf16 and (M, N, K) in _TORCHTLX_BF16_TILES:
        tile, extra = _TORCHTLX_BF16_TILES[M, N, K]
        return [_template_config(tile, **extra)]
    return heuristic_config(M, N, K, enable_global_split_k=enable, enable_short_m_tuning=False)
