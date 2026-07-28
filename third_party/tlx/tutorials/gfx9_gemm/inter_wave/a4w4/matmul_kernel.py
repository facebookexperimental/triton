"""8-wave inter-wave warp-pipelined MXFP4 (a4w4) GEMM for gfx950 (CDNA4).

This is the 8-wave (WARPS_M=2, WARPS_N=4), 32x32x64-MFMA sibling of the
4-wave a4w4 kernel in `../../intra_wave/a4w4` -- the extra warps make the inter-wave
software pipeline (`tlx.warp_pipeline_stage`) actually active (two co-resident
wave groups run a full stage apart), which the 4-wave kernel cannot do.

Key ideas (same skeleton as the a16w16 8-wave inter_wave kernel):
  * 2x2 quadrant tiling: the 256x256 tile is split into four [128x128] quadrants
    (A top/bot x B left/right); each operand half-tile gets its own
    double-buffered LDS allocation so the four scaled-MFMAs stay independent.
  * Inter-wave software pipeline: 8 (mfma + mem) regions per 2x-unrolled step,
    `async_load_wait_group` hoisted before each MFMA cluster.
  * Combined B scale: instead of slicing the B scale into the 2x2
    quadrant grid (which at WARPS_N=4 gives each thread only 4 bytes -> byte-gather
    ds_read_u8 + v_perm), load the FULL [BLOCK_N, NG] = [256, 8] B scale as ONE
    buffer read with the hardware transpose (ds_read_b64_tr_b8, 8 bytes/thread),
    then split it into the two [128, 8] N-halves with a free register-bit split
    (`scale_b_comb_layout` == `scale_b_layout` + one extra register base [128,0]).

Inputs use the same ABI as the 4-wave a4w4 kernel:
  * A: packed e2m1, shape (M, K // 2), K-contiguous
  * B: packed e2m1, shape (N, K // 2), K-contiguous; computes A @ B.T
  * scales: e8m0 uint8, shapes (M, K // 32) and (N, K // 32), contiguous along M/N
  * C: bfloat16, shape (M, N)

The Shape:Stride register/blocked/accumulator layouts below encode the gfx950
32x32x64 scaled-MFMA / dot-operand / scale distributions, and are
round-trip-verified against the compiler's resolved linear layouts.
"""

import os

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

# Keep the LLVM post-RA machine scheduler from re-ordering the
# warp_pipeline_stage mem/MFMA interleave.
os.environ.setdefault("TRITON_DISABLE_POST_MISCHED", "1")

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 256
NUM_WARPS = 8
GROUP_SIZE_M = 4
NUM_XCDS = 8

# The 2x-unrolled pipeline prefetches 2 K-tiles and drains 2 in the epilogue, so
# the pipelined loop runs (K/BLOCK_K - 2)/2 trips. At exactly K == 4*BLOCK_K the
# loop runs a SINGLE trip, which the inter-wave `tritonamdgpu-warp-pipeline`
# transform mis-schedules on multi-tile grids (correct for grid_mn==1, races for
# grid_mn>1). Require >= 2 loop trips (K >= 6*BLOCK_K) so the steady state is
# always non-empty; every production a4w4 shape uses K >= 4096 anyway.
MIN_K = 6 * BLOCK_K
KERNEL_NAME = "a4w4_8wave"


@triton.jit
def _a4w4_8wave_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    workspace_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    GRID_MN: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    HALF_M: tl.constexpr = BLOCK_M // 2  # 128
    HALF_N: tl.constexpr = BLOCK_N // 2  # 128
    HALF_K: tl.constexpr = BLOCK_K // 2  # 128 (packed fp4)
    NG: tl.constexpr = BLOCK_K // SCALE_GROUP_SIZE  # 8 scale groups along K
    # Split-K: each program owns a contiguous K-slice of length KS (== K when
    # SPLIT_K==1). KS is constexpr, so every derived offset stays constexpr and
    # keeps its divisibility (no #linear -> #blocked collapse).
    KS: tl.constexpr = K // SPLIT_K
    KS_PACKED: tl.constexpr = KS // 2  # packed fp4 columns per split
    KS_SCALE: tl.constexpr = KS // SCALE_GROUP_SIZE  # scale groups per split

    # ---- fp4 tile global-load register layout (#linear, [128,128]) ----
    g_load_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((16, 32, 64, 128, 4096, 8192, 256, 512, 1024), (1, 2, 4, 8, 2048)),
    )
    # ---- A scale global-load blocked layout (#blocked, [128,8]) ----
    blocked_a_scales: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
        stride=((32, 64, 128, 256, 512, 1, 0, 2, 4), (8, 16)),
    )
    # ---- B scale global-load blocked layout (#blocked1, FULL [256,8]) ----
    blocked_b_scales: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
        stride=((32, 64, 128, 256, 512, 1024, 1, 2, 4), (8, 16)),
    )
    # ---- padded shared tile layout ([128,128] fp4, pad 16 @ 1024) ----
    shared_tile: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 16]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [1, 0],
            [32, 0],
            [64, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
        ],
        [HALF_M, HALF_K],
    )
    shared_scales: tl.constexpr = tlx.swizzled_layout(0, 0, 0, order=[0, 1])
    # ---- MFMA scale register layouts (get_mfma_scale_layout) ----
    scale_a_layout: tl.constexpr = tlx.layout(  # #linear1, [128,8]
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
        stride=((8, 16, 32, 64, 128, 1, 0, 0, 256), (2, 4, 512)),
    )
    scale_b_layout: tl.constexpr = tlx.layout(  # #linear5, per-quadrant [128,8]
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
        stride=((8, 16, 32, 64, 128, 1, 256, 512, 0), (2, 4)),
    )
    scale_b_comb_layout: tl.constexpr = tlx.layout(  # #linear2, combined [256,8]
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
        stride=((8, 16, 32, 64, 128, 1, 256, 512, 0), (2, 4, 1024)),
    )
    # ---- MFMA accumulator layout (#mma 32x32x64 [2,4], one [128,128] quadrant) ----
    accumulator_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((128, 256, 512, 1024, 2048, 4, 32, 64, 4096), (1, 2, 8, 16, 8192)),
    )
    # ---- store layout (#blocked2, [128,128] bf16 quadrant) ----
    store_layout_c: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2)),
        stride=((8, 16, 32, 64, 512, 1024, 0, 0, 2048), (1, 2, 4, 128, 256, 4096, 8192)),
    )

    # Grid is GRID_MN * SPLIT_K; peel the split id, keep the MN pid for the
    # XCD / GROUP_SIZE_M remap below.
    split_id = tl.program_id(0) // GRID_MN
    pid = tl.program_id(0) % GRID_MN
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m > 0)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

    # Four double-buffered operand half-tiles + A-scale halves + combined B scale.
    smem_a_top = tlx.local_alloc((HALF_M, HALF_K), tlx.dtype_of(a_ptr), 2, layout=shared_tile)
    smem_a_bot = tlx.local_alloc((HALF_M, HALF_K), tlx.dtype_of(a_ptr), 2, layout=shared_tile)
    smem_b_left = tlx.local_alloc((HALF_N, HALF_K), tlx.dtype_of(b_ptr), 2, layout=shared_tile)
    smem_b_right = tlx.local_alloc((HALF_N, HALF_K), tlx.dtype_of(b_ptr), 2, layout=shared_tile)
    smem_a_sc_t = tlx.local_alloc((HALF_M, NG), tlx.dtype_of(a_scales_ptr), 2, layout=shared_scales)
    smem_a_sc_b = tlx.local_alloc((HALF_M, NG), tlx.dtype_of(a_scales_ptr), 2, layout=shared_scales)
    smem_b_sc = tlx.local_alloc((BLOCK_N, NG), tlx.dtype_of(b_scales_ptr), 2, layout=shared_scales)

    # ---- fp4 tile load offsets ([128,128]) ----
    offs_am = tl.arange(0, HALF_M)
    offs_ak = tl.arange(0, HALF_K)
    a_tile_offsets = tlx.require_layout(offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak, g_load_layout)
    a_base = a_ptr + pid_m * BLOCK_M * stride_am

    offs_bn = tl.arange(0, HALF_N)
    offs_bk = tl.arange(0, HALF_K)
    b_tile_offsets = tlx.require_layout(offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk, g_load_layout)
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # ---- A scale load offsets ([128,8]) ----
    offs_ks_a = tl.arange(0, NG)
    offs_asm = (pid_m * BLOCK_M + tl.arange(0, HALF_M)) % M
    a_sc_offsets = tl.mul(offs_asm[:, None], stride_asm, sanitize_overflow=False) + tl.mul(
        offs_ks_a[None, :], stride_ask, sanitize_overflow=False)
    a_sc_offsets = tlx.require_layout(a_sc_offsets, blocked_a_scales)

    # ---- B scale load offsets: FULL [256,8] in one copy ----
    offs_ks_b = tl.arange(0, NG)
    offs_bsn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    b_sc_offsets = tl.mul(offs_bsn[:, None], stride_bsn, sanitize_overflow=False) + tl.mul(
        offs_ks_b[None, :], stride_bsk, sanitize_overflow=False)
    b_sc_offsets = tlx.require_layout(b_sc_offsets, blocked_b_scales)

    # Scalar (uniform) base-pointer deltas for the quadrant / K-buffer variants.
    a_half_m = HALF_M * stride_am  # a_top -> a_bot
    b_half_n = HALF_N * stride_bn  # b_left -> b_right
    a_k2 = HALF_K * stride_ak  # even -> odd (_next) K-step
    b_k2 = HALF_K * stride_bk
    a_sc_half_m = HALF_M * stride_asm
    a_sc_k = NG * stride_ask
    b_sc_k = NG * stride_bsk

    # Advance every base to this split's K-slice (no-op when SPLIT_K==1). The
    # A/B tiles are K-packed (KS_PACKED cols) and the scales are per-group
    # (KS_SCALE groups). All arith bases; buffer_load_to_local materializes the
    # fat pointer as it already does for the pid_m/pid_n base offsets.
    a_base += split_id * KS_PACKED * stride_ak
    b_base += split_id * KS_PACKED * stride_bk
    a_scales_ptr += split_id * KS_SCALE * stride_ask
    b_scales_ptr += split_id * KS_SCALE * stride_bsk

    acc_tl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_bl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_tr = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_br = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)

    iter_max: tl.constexpr = KS // BLOCK_K
    tl.assume(iter_max > 5)

    # ---- Prologue: prefetch K-steps 0,1 into buffers 0,1 (8 commits) ----
    tlx.buffer_load_to_local(smem_b_left[0], b_base, b_tile_offsets)
    tlx.buffer_load_to_local(smem_b_sc[0], b_scales_ptr, b_sc_offsets)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[0], a_base, a_tile_offsets)
    tlx.buffer_load_to_local(smem_a_sc_t[0], a_scales_ptr, a_sc_offsets)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[0], a_base + a_half_m, a_tile_offsets)
    tlx.buffer_load_to_local(smem_a_sc_b[0], a_scales_ptr + a_sc_half_m, a_sc_offsets)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[0], b_base + b_half_n, b_tile_offsets)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_b_left[1], b_base + b_k2, b_tile_offsets)
    tlx.buffer_load_to_local(smem_b_sc[1], b_scales_ptr + b_sc_k, b_sc_offsets)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[1], a_base + a_k2, a_tile_offsets)
    tlx.buffer_load_to_local(smem_a_sc_t[1], a_scales_ptr + a_sc_k, a_sc_offsets)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[1], a_base + a_half_m + a_k2, a_tile_offsets)
    tlx.buffer_load_to_local(smem_a_sc_b[1], a_scales_ptr + a_sc_half_m + a_sc_k, a_sc_offsets)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[1], b_base + b_half_n + b_k2, b_tile_offsets)
    tlx.async_load_commit_group()

    a_base += a_k2 * 2
    b_base += b_k2 * 2
    a_scales_ptr += a_sc_k * 2
    b_scales_ptr += b_sc_k * 2

    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
    a_top = tlx.local_load(smem_a_top[0], relaxed=True)
    a_sc_top = tlx.local_load(smem_a_sc_t[0], layout=scale_a_layout)
    b_sc_comb = tlx.local_load(smem_b_sc[0], layout=scale_b_comb_layout)
    b_sc_l, b_sc_r = tl.split(tl.trans(tl.reshape(b_sc_comb, 2, HALF_N, NG), 1, 2, 0))
    b_sc_left = tlx.require_layout(b_sc_l, scale_b_layout)
    b_sc_right = tlx.require_layout(b_sc_r, scale_b_layout)

    # ---- Main loop (2x unrolled): 8 (mfma + mem) regions ----
    for k in tl.range(0, iter_max - 2, 2, num_stages=1):
        # --- sub-iter 0 (buffer 0) ---
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_left, b_sc_left, "e2m1", acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
            a_sc_bot = tlx.local_load(smem_a_sc_b[0], layout=scale_a_layout)
            tlx.buffer_load_to_local(smem_b_left[0], b_base, b_tile_offsets)
            tlx.buffer_load_to_local(smem_b_sc[0], b_scales_ptr, b_sc_offsets)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_left, b_sc_left, "e2m1", acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
            tlx.buffer_load_to_local(smem_a_top[0], a_base, a_tile_offsets)
            tlx.buffer_load_to_local(smem_a_sc_t[0], a_scales_ptr, a_sc_offsets)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_right, b_sc_right, "e2m1", acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
            tlx.buffer_load_to_local(smem_a_bot[0], a_base + a_half_m, a_tile_offsets)
            tlx.buffer_load_to_local(smem_a_sc_b[0], a_scales_ptr + a_sc_half_m, a_sc_offsets)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_right, b_sc_right, "e2m1", acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[1], relaxed=True)
            a_sc_top = tlx.local_load(smem_a_sc_t[1], layout=scale_a_layout)
            b_sc_comb = tlx.local_load(smem_b_sc[1], layout=scale_b_comb_layout)
            b_sc_l, b_sc_r = tl.split(tl.trans(tl.reshape(b_sc_comb, 2, HALF_N, NG), 1, 2, 0))
            b_sc_left = tlx.require_layout(b_sc_l, scale_b_layout)
            b_sc_right = tlx.require_layout(b_sc_r, scale_b_layout)
            tlx.buffer_load_to_local(smem_b_right[0], b_base + b_half_n, b_tile_offsets)
            tlx.async_load_commit_group()

        # --- sub-iter 1 (buffer 1, base + one K-step) ---
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_left, b_sc_left, "e2m1", acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
            a_sc_bot = tlx.local_load(smem_a_sc_b[1], layout=scale_a_layout)
            tlx.buffer_load_to_local(smem_b_left[1], b_base + b_k2, b_tile_offsets)
            tlx.buffer_load_to_local(smem_b_sc[1], b_scales_ptr + b_sc_k, b_sc_offsets)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_left, b_sc_left, "e2m1", acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
            tlx.buffer_load_to_local(smem_a_top[1], a_base + a_k2, a_tile_offsets)
            tlx.buffer_load_to_local(smem_a_sc_t[1], a_scales_ptr + a_sc_k, a_sc_offsets)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_right, b_sc_right, "e2m1", acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
            tlx.buffer_load_to_local(smem_a_bot[1], a_base + a_half_m + a_k2, a_tile_offsets)
            tlx.buffer_load_to_local(smem_a_sc_b[1], a_scales_ptr + a_sc_half_m + a_sc_k, a_sc_offsets)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_right, b_sc_right, "e2m1", acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[0], relaxed=True)
            a_sc_top = tlx.local_load(smem_a_sc_t[0], layout=scale_a_layout)
            b_sc_comb = tlx.local_load(smem_b_sc[0], layout=scale_b_comb_layout)
            b_sc_l, b_sc_r = tl.split(tl.trans(tl.reshape(b_sc_comb, 2, HALF_N, NG), 1, 2, 0))
            b_sc_left = tlx.require_layout(b_sc_l, scale_b_layout)
            b_sc_right = tlx.require_layout(b_sc_r, scale_b_layout)
            tlx.buffer_load_to_local(smem_b_right[1], b_base + b_half_n + b_k2, b_tile_offsets)
            tlx.async_load_commit_group()
            a_base += a_k2 * 2
            b_base += b_k2 * 2
            a_scales_ptr += a_sc_k * 2
            b_scales_ptr += b_sc_k * 2

    # ---- Epilogue: last 2 K-steps, drain, 4-quadrant store ----
    # iter iter_max-2 (b_sc_left/right for this step were prefetched at loop tail)
    acc_tl = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_left, b_sc_left, "e2m1", acc_tl)
    tlx.async_load_wait_group(5)
    l_idx: tl.constexpr = 0  # (iter_max - 2) % 2, always 0 (iter_max even)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, l_idx), relaxed=True)
    a_sc_bot = tlx.local_load(tlx.local_view(smem_a_sc_b, l_idx), layout=scale_a_layout)

    acc_bl = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_left, b_sc_left, "e2m1", acc_bl)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b_right, l_idx)), relaxed=True)

    acc_tr = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_right, b_sc_right, "e2m1", acc_tr)
    tlx.async_load_wait_group(3)
    g_idx: tl.constexpr = 1  # 1 - l_idx
    b_left = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b_left, g_idx)), relaxed=True)

    acc_br = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_right, b_sc_right, "e2m1", acc_br)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(tlx.local_view(smem_a_top, g_idx), relaxed=True)
    a_sc_top = tlx.local_load(tlx.local_view(smem_a_sc_t, g_idx), layout=scale_a_layout)
    b_sc_comb = tlx.local_load(tlx.local_view(smem_b_sc, g_idx), layout=scale_b_comb_layout)
    b_sc_l, b_sc_r = tl.split(tl.trans(tl.reshape(b_sc_comb, 2, HALF_N, NG), 1, 2, 0))
    b_sc_left = tlx.require_layout(b_sc_l, scale_b_layout)
    b_sc_right = tlx.require_layout(b_sc_r, scale_b_layout)

    # iter iter_max-1: finish ALL four accumulators, then convert + store.
    acc_tl = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_left, b_sc_left, "e2m1", acc_tl)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, g_idx), relaxed=True)
    a_sc_bot = tlx.local_load(tlx.local_view(smem_a_sc_b, g_idx), layout=scale_a_layout)

    acc_bl = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_left, b_sc_left, "e2m1", acc_bl)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b_right, g_idx)), relaxed=True)

    acc_tr = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_right, b_sc_right, "e2m1", acc_tr)
    acc_br = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_right, b_sc_right, "e2m1", acc_br)

    # ---- 4-quadrant store ----
    if SPLIT_K == 1:
        # Direct coalesced store to C (bf16). c_quad_offsets is shared by all four
        # quadrants (same relative layout, different base).
        offs_cm = tl.arange(0, HALF_M)
        offs_cn = tl.arange(0, HALF_N)
        c_quad_offsets = tl.mul(stride_cm, offs_cm[:, None], sanitize_overflow=False) + tl.mul(
            stride_cn, offs_cn[None, :], sanitize_overflow=False)
        c_quad_offsets = tlx.require_layout(c_quad_offsets, store_layout_c)
        c_tl_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
        c_bl_base = c_tl_base + HALF_M * stride_cm
        c_tr_base = c_tl_base + HALF_N * stride_cn
        c_br_base = c_bl_base + HALF_N * stride_cn

        # require(accumulator) -> cast -> require(store): freeze the MFMA register
        # distribution, cast bf16 register-local in it, then require the coalesced
        # store layout. The compiler narrows before redistributing (no explicit
        # release_layout needed).
        et = c_ptr.dtype.element_ty
        acc_tl = tlx.require_layout(acc_tl, accumulator_layout)
        c_tl = tlx.require_layout(acc_tl.to(et), store_layout_c)
        tlx.buffer_store(c_tl, c_tl_base, c_quad_offsets)

        acc_bl = tlx.require_layout(acc_bl, accumulator_layout)
        c_bl = tlx.require_layout(acc_bl.to(et), store_layout_c)
        tlx.buffer_store(c_bl, c_bl_base, c_quad_offsets)

        acc_tr = tlx.require_layout(acc_tr, accumulator_layout)
        c_tr = tlx.require_layout(acc_tr.to(et), store_layout_c)
        tlx.buffer_store(c_tr, c_tr_base, c_quad_offsets)

        acc_br = tlx.require_layout(acc_br, accumulator_layout)
        c_br = tlx.require_layout(acc_br.to(et), store_layout_c)
        tlx.buffer_store(c_br, c_br_base, c_quad_offsets)
    else:
        # Split-K: write this split's fp32 partials into its workspace slice
        # (rows [split_id*M, split_id*M+M)); a separate fp32 reduce kernel sums
        # the SPLIT_K slabs into C, keeping the result bit-identical to a single
        # fp32-accumulated GEMM. Plain tl.store (fp32, no narrowing).
        rb = split_id * M
        offs_cm_t = rb + pid_m * BLOCK_M + tl.arange(0, HALF_M)
        offs_cm_b = offs_cm_t + HALF_M
        offs_cn_l = pid_n * BLOCK_N + tl.arange(0, HALF_N)
        offs_cn_r = offs_cn_l + HALF_N
        tl.store(workspace_ptr + offs_cm_t[:, None] * stride_cm + offs_cn_l[None, :] * stride_cn, acc_tl)
        tl.store(workspace_ptr + offs_cm_b[:, None] * stride_cm + offs_cn_l[None, :] * stride_cn, acc_bl)
        tl.store(workspace_ptr + offs_cm_t[:, None] * stride_cm + offs_cn_r[None, :] * stride_cn, acc_tr)
        tl.store(workspace_ptr + offs_cm_b[:, None] * stride_cm + offs_cn_r[None, :] * stride_cn, acc_br)


@triton.jit
def _reduce_k_kernel(workspace_ptr, c_ptr, M, N, SPLIT_K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                     BLOCK_SIZE_N: tl.constexpr, OUTPUT_DTYPE: tl.constexpr):
    # Sum the SPLIT_K fp32 partials (each a contiguous (M, N) slab in workspace)
    # into C with fp32 accumulation. Small tiles so small outputs still spawn many
    # CTAs (else the reduce is CTA-starved and dominates on skinny shapes).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base_offs = offs_m[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        partial = tl.load(workspace_ptr + base_offs + s * M * N, mask=mask, other=0.0)
        acc += partial.to(tl.float32)
    tl.store(c_ptr + base_offs, acc.to(OUTPUT_DTYPE), mask=mask)


NUM_CU = 256  # gfx950 (CDNA4) compute units
# Each split must be a whole number of BLOCK_K tiles and keep the pipelined loop
# at >= 2 trips (KS >= 6*BLOCK_K == MIN_K; see the module comment on MIN_K).
MIN_KTILES_PER_SPLIT = MIN_K // BLOCK_K  # == 6


def choose_split_k(M, N, K):
    """Largest SPLIT_K that fills more CUs while keeping each split a whole,
    512-aligned K-chunk of >= MIN_K. The fp32 reduce is a fixed ~4-5us tax, so
    split-K only pays off on *severely* under-filled grids; measured, grid_mn=8
    (K=8192) wins by ~15us but grid_mn=16 (K=4096) already loses to the reduce.
    Gate on grid_mn <= NUM_CU/32 and leave everything else at SPLIT_K=1."""
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    if grid_mn > NUM_CU // 32:
        return 1
    best = 1
    for sk in range(2, NUM_CU // grid_mn + 1):  # grid_mn*sk <= NUM_CU
        ks = K // sk
        if K % sk == 0 and ks >= MIN_K and ks % (2 * BLOCK_K) == 0:
            best = sk  # fill grows with sk, so the last valid sk wins
    return best


def _matmul_256tile(a, b, a_scales, b_scales, SPLIT_K=None):
    """256x256-tile inter-wave path -- the fast path for well-filled / large N."""
    assert a.dtype is torch.uint8
    assert b.dtype is torch.uint8
    assert a_scales.dtype is torch.uint8
    assert b_scales.dtype is torch.uint8
    assert a.is_cuda and b.is_cuda and a_scales.is_cuda and b_scales.is_cuda

    M = a.shape[0]
    K_packed = a.shape[1]
    K = K_packed * 2
    N = b.shape[0]

    assert b.shape[1] == K_packed, "B must have shape (N, K // 2)"
    assert a_scales.shape == (M, K // 32), "A scales must have shape (M, K // 32)"
    assert b_scales.shape == (N, K // 32), "B scales must have shape (N, K // 32)"
    assert a_scales.stride(0) == 1, "A scales must be contiguous along M"
    assert b_scales.stride(0) == 1, "B scales must be contiguous along N"

    assert M % BLOCK_M == 0, "M must be a multiple of 256"
    assert N % BLOCK_N == 0, "N must be a multiple of 256"
    assert K >= MIN_K and K % (2 * BLOCK_K) == 0, "K must be at least 1536 and a multiple of 512"

    if SPLIT_K is None:
        SPLIT_K = choose_split_k(M, N, K)
    KS = K // SPLIT_K
    assert K % SPLIT_K == 0, f"K={K} must be divisible by SPLIT_K={SPLIT_K}"
    assert KS >= MIN_K and KS % (2 * BLOCK_K) == 0, f"K/SPLIT_K={KS} must be >= {MIN_K} and a multiple of {2 * BLOCK_K}"

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    # fp32 workspace so the split-K result matches a single fp32-accumulated GEMM.
    workspace = torch.empty((SPLIT_K * M, N), device=a.device, dtype=torch.float32) if SPLIT_K > 1 else c
    _a4w4_8wave_kernel[(grid_mn * SPLIT_K, )](
        a,
        b,
        c,
        workspace,
        a_scales,
        b_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_XCDS=NUM_XCDS,
        GRID_MN=grid_mn,
        SPLIT_K=SPLIT_K,
        num_warps=NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=32,
        # Forbid AGPRs: f32 accumulators write VGPRs directly (packs tighter, no spills).
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
    )
    if SPLIT_K > 1:
        big = (M * N) >= (2048 * 2048)
        rbm, rbn, rw = (128, 128, 8) if big else (32, 32, 4)
        reduce_grid = (triton.cdiv(M, rbm), triton.cdiv(N, rbn))
        _reduce_k_kernel[reduce_grid](
            workspace,
            c,
            M,
            N,
            SPLIT_K=SPLIT_K,
            BLOCK_SIZE_M=rbm,
            BLOCK_SIZE_N=rbn,
            OUTPUT_DTYPE=tl.bfloat16,
            num_warps=rw,
        )
    return c


# ---------------------------------------------------------------------------
# Skinny-N path: a standalone 128x128 tile (8 warps [2,4], one 32x32x64 MFMA
# tile per warp) -- which is exactly ONE quadrant of the 256-tile 8-wave kernel, so
# it reuses that kernel's [128,128] Shape:Stride layouts verbatim. On
# occupancy-starved shapes the 256-tile leaves the machine idle (2048x256 -> 8
# workgroups); the 128-tile has 4x the MN-tiles and, with split-K, reaches ~full
# CU occupancy (2048x256x8192 -> 32 tiles x SK8 = 256 wg). It is a real TLX
# kernel (direct-to-LDS buffer_load_to_local + pinned layouts + double-buffered
# LDS), single accumulator, no warp_pipeline (so it is not bound by the
# single-trip MIN_K constraint and split-K can go as deep as occupancy needs).
# Measured 2048x256x8192 device-time: 16.0us (SK4) vs the 256-tile's 42.5us and
# the best hipBLASLt algo's 20.8us. Slower than the 256-tile on large shapes
# (tiny tiles, huge grid), so it is used only via the dispatcher below.
SKINNY_BLOCK_M = 128
SKINNY_BLOCK_N = 128


@triton.jit
def _a4w4_skinny_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    workspace_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    GRID_MN: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    NG: tl.constexpr = BLOCK_K // SCALE_GROUP_SIZE
    BKP: tl.constexpr = BLOCK_K // 2
    KS: tl.constexpr = K // SPLIT_K
    iter_max: tl.constexpr = KS // BLOCK_K

    # Reused the 256-tile kernel's [128,128]-quadrant layouts (a standalone 128x128 tile == one quadrant).
    g_load_layout: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
                                             stride=((16, 32, 64, 128, 4096, 8192, 256, 512, 1024), (1, 2, 4, 8, 2048)))
    blocked_scales: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
                                              stride=((32, 64, 128, 256, 512, 1, 0, 2, 4), (8, 16)))
    shared_tile: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 16]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0],
                       [4, 0], [8, 0], [16, 0]], [BLOCK_M, BKP])
    shared_scales: tl.constexpr = tlx.swizzled_layout(0, 0, 0, order=[0, 1])
    scale_a_layout: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
                                              stride=((8, 16, 32, 64, 128, 1, 0, 0, 256), (2, 4, 512)))
    scale_b_layout: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
                                              stride=((8, 16, 32, 64, 128, 1, 256, 512, 0), (2, 4)))
    accumulator_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((128, 256, 512, 1024, 2048, 4, 32, 64, 4096), (1, 2, 8, 16, 8192)))
    store_layout_c: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2)),
        stride=((8, 16, 32, 64, 512, 1024, 0, 0, 2048), (1, 2, 4, 128, 256, 4096, 8192)))

    split_id = tl.program_id(0) // GRID_MN
    pid = tl.program_id(0) % GRID_MN
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m > 0)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

    smem_a = tlx.local_alloc((BLOCK_M, BKP), tlx.dtype_of(a_ptr), 2, layout=shared_tile)
    smem_b = tlx.local_alloc((BLOCK_N, BKP), tlx.dtype_of(b_ptr), 2, layout=shared_tile)
    smem_asc = tlx.local_alloc((BLOCK_M, NG), tlx.dtype_of(a_scales_ptr), 2, layout=shared_scales)
    smem_bsc = tlx.local_alloc((BLOCK_N, NG), tlx.dtype_of(b_scales_ptr), 2, layout=shared_scales)

    offs_am = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BKP)
    a_off = tlx.require_layout(offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak, g_load_layout)
    a_base = a_ptr + pid_m * BLOCK_M * stride_am + split_id * (KS // 2) * stride_ak
    offs_bn = tl.arange(0, BLOCK_N)
    b_off = tlx.require_layout(offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk, g_load_layout)
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn + split_id * (KS // 2) * stride_bk
    offs_sg = tl.arange(0, NG)
    offs_asm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    asc_off = tlx.require_layout(
        tl.mul(offs_asm[:, None], stride_asm, sanitize_overflow=False) +
        tl.mul(offs_sg[None, :], stride_ask, sanitize_overflow=False), blocked_scales)
    a_scales_ptr += split_id * (KS // SCALE_GROUP_SIZE) * stride_ask
    offs_bsn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    bsc_off = tlx.require_layout(
        tl.mul(offs_bsn[:, None], stride_bsn, sanitize_overflow=False) +
        tl.mul(offs_sg[None, :], stride_bsk, sanitize_overflow=False), blocked_scales)
    b_scales_ptr += split_id * (KS // SCALE_GROUP_SIZE) * stride_bsk

    ak = BKP * stride_ak
    bk = BKP * stride_bk
    sck_a = NG * stride_ask
    sck_b = NG * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    tl.assume(iter_max > 1)

    # Prologue: buffers 0,1 (K-steps 0,1); one commit per buffer.
    tlx.buffer_load_to_local(smem_a[0], a_base, a_off)
    tlx.buffer_load_to_local(smem_asc[0], a_scales_ptr, asc_off)
    tlx.buffer_load_to_local(smem_b[0], b_base, b_off)
    tlx.buffer_load_to_local(smem_bsc[0], b_scales_ptr, bsc_off)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a[1], a_base + ak, a_off)
    tlx.buffer_load_to_local(smem_asc[1], a_scales_ptr + sck_a, asc_off)
    tlx.buffer_load_to_local(smem_b[1], b_base + bk, b_off)
    tlx.buffer_load_to_local(smem_bsc[1], b_scales_ptr + sck_b, bsc_off)
    tlx.async_load_commit_group()
    a_base += ak * 2
    b_base += bk * 2
    a_scales_ptr += sck_a * 2
    b_scales_ptr += sck_b * 2

    # Main loop (2x unrolled): consume a buffer, refill it one K-step ahead.
    for k in tl.range(0, iter_max - 2, 2, num_stages=1):
        tlx.async_load_wait_group(1)
        a = tlx.local_load(smem_a[0], relaxed=True)
        b = tlx.local_load(tlx.local_trans(smem_b[0]), relaxed=True)
        asc = tlx.local_load(smem_asc[0], layout=scale_a_layout)
        bsc = tlx.local_load(smem_bsc[0], layout=scale_b_layout)
        acc = tl.dot_scaled(a, asc, "e2m1", b, bsc, "e2m1", acc)
        tlx.buffer_load_to_local(smem_a[0], a_base, a_off)
        tlx.buffer_load_to_local(smem_asc[0], a_scales_ptr, asc_off)
        tlx.buffer_load_to_local(smem_b[0], b_base, b_off)
        tlx.buffer_load_to_local(smem_bsc[0], b_scales_ptr, bsc_off)
        tlx.async_load_commit_group()

        tlx.async_load_wait_group(1)
        a = tlx.local_load(smem_a[1], relaxed=True)
        b = tlx.local_load(tlx.local_trans(smem_b[1]), relaxed=True)
        asc = tlx.local_load(smem_asc[1], layout=scale_a_layout)
        bsc = tlx.local_load(smem_bsc[1], layout=scale_b_layout)
        acc = tl.dot_scaled(a, asc, "e2m1", b, bsc, "e2m1", acc)
        tlx.buffer_load_to_local(smem_a[1], a_base + ak, a_off)
        tlx.buffer_load_to_local(smem_asc[1], a_scales_ptr + sck_a, asc_off)
        tlx.buffer_load_to_local(smem_b[1], b_base + bk, b_off)
        tlx.buffer_load_to_local(smem_bsc[1], b_scales_ptr + sck_b, bsc_off)
        tlx.async_load_commit_group()
        a_base += ak * 2
        b_base += bk * 2
        a_scales_ptr += sck_a * 2
        b_scales_ptr += sck_b * 2

    # Epilogue: drain the two prefetched buffers (no refill).
    tlx.async_load_wait_group(1)
    a = tlx.local_load(tlx.local_view(smem_a, 0), relaxed=True)
    b = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b, 0)), relaxed=True)
    asc = tlx.local_load(tlx.local_view(smem_asc, 0), layout=scale_a_layout)
    bsc = tlx.local_load(tlx.local_view(smem_bsc, 0), layout=scale_b_layout)
    acc = tl.dot_scaled(a, asc, "e2m1", b, bsc, "e2m1", acc)

    tlx.async_load_wait_group(0)
    a = tlx.local_load(tlx.local_view(smem_a, 1), relaxed=True)
    b = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b, 1)), relaxed=True)
    asc = tlx.local_load(tlx.local_view(smem_asc, 1), layout=scale_a_layout)
    bsc = tlx.local_load(tlx.local_view(smem_bsc, 1), layout=scale_b_layout)
    acc = tl.dot_scaled(a, asc, "e2m1", b, bsc, "e2m1", acc)

    offs_cm = tl.arange(0, BLOCK_M)
    offs_cn = tl.arange(0, BLOCK_N)
    if SPLIT_K == 1:
        c_off = tl.mul(stride_cm, offs_cm[:, None], sanitize_overflow=False) + tl.mul(
            stride_cn, offs_cn[None, :], sanitize_overflow=False)
        c_off = tlx.require_layout(c_off, store_layout_c)
        c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
        et = c_ptr.dtype.element_ty
        acc = tlx.require_layout(acc, accumulator_layout)
        c = tlx.require_layout(acc.to(et), store_layout_c)
        tlx.buffer_store(c, c_base, c_off)
    else:
        rb = split_id * M
        rows = rb + pid_m * BLOCK_M + offs_cm
        cols = pid_n * BLOCK_N + offs_cn
        tl.store(workspace_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn, acc)


def choose_split_k_skinny(M, N, K):
    """SPLIT_K for the 128x128 skinny tile: fill the CUs with a whole,
    2*BLOCK_K-aligned K-chunk per split (no warp_pipeline -> no MIN_K limit)."""
    grid_mn = triton.cdiv(M, SKINNY_BLOCK_M) * triton.cdiv(N, SKINNY_BLOCK_N)
    best = 1
    for sk in range(2, NUM_CU // grid_mn + 1):
        ks = K // sk
        if K % sk == 0 and ks % BLOCK_K == 0 and ks >= 2 * BLOCK_K:
            best = sk
    return best


def _matmul_skinny(a, b, a_scales, b_scales, SPLIT_K=None):
    """128x128-tile TLX path for occupancy-starved (skinny-N) shapes. The 128
    tile only reaches ~32 workgroups alone, so it relies on split-K to fill the
    CUs (device-time 16us at SK8 vs 22.6us at SK1); the reduce adds a 2nd launch,
    a wall-clock cost that amortizes under CUDA-graph / pooled-workspace use."""
    M = a.shape[0]
    K = a.shape[1] * 2
    N = b.shape[0]
    BM, BN = SKINNY_BLOCK_M, SKINNY_BLOCK_N
    if SPLIT_K is None:
        SPLIT_K = choose_split_k_skinny(M, N, K)
    KS = K // SPLIT_K
    assert K % SPLIT_K == 0 and KS % BLOCK_K == 0
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid_mn = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    workspace = torch.empty((SPLIT_K * M, N), device=a.device, dtype=torch.float32) if SPLIT_K > 1 else c
    _a4w4_skinny_kernel[(grid_mn * SPLIT_K, )](
        a,
        b,
        c,
        workspace,
        a_scales,
        b_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_XCDS=NUM_XCDS,
        GRID_MN=grid_mn,
        SPLIT_K=SPLIT_K,
        num_warps=NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=32,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
    )
    if SPLIT_K > 1:
        rbm, rbn, rw = (32, 32, 4)
        _reduce_k_kernel[(triton.cdiv(M, rbm), triton.cdiv(N, rbn))](workspace, c, M, N, SPLIT_K=SPLIT_K,
                                                                     BLOCK_SIZE_M=rbm, BLOCK_SIZE_N=rbn,
                                                                     OUTPUT_DTYPE=tl.bfloat16, num_warps=rw)
    return c


def matmul(a, b, a_scales, b_scales):
    """A @ B.T for packed MXFP4 A/B. Dispatches on the 256-tile grid fill:
    severely occupancy-starved shapes (e.g. skinny-N) take the 128x128 + split-K
    TLX path; everything else takes the 256x256 inter-wave path."""
    M = a.shape[0]
    N = b.shape[0]
    grid_mn_256 = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    if grid_mn_256 <= NUM_CU // 32:
        return _matmul_skinny(a, b, a_scales, b_scales)
    return _matmul_256tile(a, b, a_scales, b_scales)
