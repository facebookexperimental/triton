"""8-wave inter-wave warp-pipelined MXFP4 (a4w4) GEMM for gfx950 (CDNA4).

This is the 8-wave (WARPS_M=2, WARPS_N=4), 16x16x128-MFMA sibling of the
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

The production entry point uses the same ABI as the 4-wave a4w4 kernel:
  * A: packed e2m1, shape (M, K // 2), K-contiguous
  * B: packed e2m1, shape (N, K // 2), K-contiguous; computes A @ B.T
  * scales: e8m0 uint8, shapes (M, K // 32) and (N, K // 32), contiguous along M/N
  * C: bfloat16, shape (M, N)

``matmul_preshuffled`` is a separate experimental ABI: both scale operands are
packed into their LDS-consumer order and retain the production DMA-to-LDS
pipeline.  The full 256-row A scale tile is read once, then split into its two
128-row MFMA fragments; this is intended to coalesce the two canonical
``ds_read_b64_tr_b8`` operations into one ``ds_read_b128``.  It remains outside
``matmul`` because its flat, prepacked scale buffers are a distinct caller ABI.

The Shape:Stride register/blocked/accumulator layouts below encode the gfx950
16x16x128 scaled-MFMA / dot-operand / scale distributions, and are
round-trip-verified against the compiler's resolved linear layouts.

Why 16x16x128 rather than the 32x32x64 this file used before: at the same tile and
MACs the two issue byte-identical memory traffic -- 54 ds_read, zero ds_write, 22
direct-to-LDS loads, 16 s_barrier. Only MFMA density changes: 64 MFMAs per body
become 128, halving memory ops per MFMA from 6.44 to 3.80, which is what gives the
scheduler enough compute to hide that traffic behind.

THE THREE CHANGES THAT GOT IT HERE
  D114296896 harness, ratio = AITER/ours, >1.0 beats AITER:

                                     2048x8192x4096   2048x8192x8192
    32x32x64 (this file, previous rev)        0.862            0.864
    -> 16x16x128, five layouts pinned         0.961            0.852
    -> + sched_barrier fences                 1.008            0.946
    -> + LDS pad 16 -> 32                     1.033            0.954

  1. 16x16x128 MFMA. Neutral alone; it exists to make change 2 pay.
  2. tlx.amd_sched_barrier(0) per warp-pipeline mem stage, pinning the ds_reads ahead
     of the global-load cluster. +4% on the 32x32x64 predecessor, but +10-15% here.
  3. LDS pad 16 -> 32; bank conflicts 98.1% -> 9.3%. Swept, not derived: pad 8 drives
     conflicts to 0.6% and is 3x SLOWER, because 8 bytes breaks the 16-byte alignment
     ds_read_b128 needs. The pad must be a power of two and >= 16.

The scale tiles now use a derived 8-byte-chunk XOR instead of padding. For
logical coordinates (row, group), A stores row ^ ((group & 4) << 2), while B
stores row ^ ((group & 1) << 4) ^ ((group & 4) << 3). The global offset layouts
encode the same maps, so direct-to-LDS writes the final physical image without
address arithmetic or reinterpretation. This reaches zero measured bank
conflicts without ds_bpermute, ds_permute, DPP, or permlane.

STRUCTURAL CEILING: 16 s_barrier per body, 8 carrying a mandatory full lgkmcnt(0) LDS
drain. The barrier is what forces the drain, not operand granularity, so the only way
down is to take an operand out of LDS entirely -- which also means chunked local_load
will not help here.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.tutorials.gfx9_gemm.intra_wave.a4w4.matmul_kernel import (
    matmul as intra_wave_matmul,
)
from triton.language.extra.tlx.tutorials.gfx9_gemm.skinny.a4w4 import is_skinny, skinny_matmul

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 256
NUM_WARPS = 8
GROUP_SIZE_M = 4
NUM_XCDS = 8

# Keep the LLVM post-RA machine scheduler from reordering this kernel's manual
# warp_pipeline_stage mem/MFMA interleave. The AMD function attribute selects
# LLVM's no-op post scheduler for this function only.
_A4W4_8WAVE_LLVM_FN_ATTRS = (
    ("amdgpu-agpr-alloc", "0,0"),
    ("amdgpu-post-sched-strategy", "nop"),
)

# The 2x-unrolled pipeline prefetches two K tiles and drains two in the
# epilogue, so it requires at least four tiles. The loop trip count stays a
# runtime scalar so the generic canonicalizer cannot flatten the one-trip
# K=1024 form and inflate its live ranges; K and KS remain constexpr to preserve
# the pinned operand layouts.
MIN_K = 4 * BLOCK_K
KERNEL_NAME = "a4w4_8wave"
PRESHUFFLED_KERNEL_NAME = "a4w4_8wave_preshuffled_scales"
INTRA_WAVE_MAX_K = 1536


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
    K_TILES,
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
    PRESHUFFLED_SCALES: tl.constexpr = False,
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
    SCALE_GROUP_BLOCKS: tl.constexpr = K // BLOCK_K

    # ---- fp4 tile global-load register layout (#linear, [128,128]) ----
    g_load_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((16, 32, 64, 128, 4096, 8192, 256, 512, 1024), (1, 2, 4, 8, 2048)),
    )
    # ---- padded shared tile layout ([128,128] fp4, pad 32 @ 1024) ----
    shared_tile: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
            [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0], [16, 0],
        ],
        [HALF_M, HALF_K],
    )
    # Map A (row, group) to (row ^ ((group & 4) << 2), group), and map B to
    # (row ^ ((group & 1) << 4) ^ ((group & 4) << 3), group). These are the
    # smallest row-bit changes that make the five ds_read_b64_tr_b8 bank bases
    # distinct: A toggles row bit 4 from group bit 2; B additionally toggles
    # row bit 4 from group bit 0 and row bit 5 from group bit 2. Row bits 0..2
    # remain unchanged, preserving each contiguous 8-byte global segment.
    shared_a_scales: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[
            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
            [0, 1], [0, 2], [16, 4],
        ],
        block_bases=[],
        alignment=16,
    )
    shared_b_scales: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[
            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0],
            [16, 1], [0, 2], [32, 4],
        ],
        block_bases=[],
        alignment=16,
    )
    # Experimental scale ABI.  Its physical low bits are exactly the value
    # bases consumed by each wave.  In particular the combined A tile puts
    # (group bit 2, row bits 5, 6, 7) in adjacent bytes, so one ordinary
    # 128-bit LDS transaction produces both 128-row A scale fragments.
    shared_scale_preshuffled: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[
            [1], [2], [4], [8], [16], [32], [64], [128], [256], [512], [1024],
        ],
        block_bases=[],
        alignment=16,
    )
    # Match each scale tile's physical shared offset order: two 4-byte value
    # bits, six lane bits, then warp bits. The XOR bases live entirely in the
    # warp mapping, so TLX resolves these to #ttg.generic_linear and gfx9 can
    # write the swizzled LDS image directly from ordinary logical offsets.
    scale_load_a_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
        stride=((32, 64, 128, 256, 512, 1, 2, 132, 0), (8, 16)),
    )
    scale_load_b_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
        stride=((32, 64, 128, 256, 512, 1024, 129, 2, 260), (8, 16)),
    )
    # Four byte values per thread stage each 256x8 preshuffled tile.  The value
    # bases are physical bits 0 and 1; the remaining physical bits are assigned
    # across the 512 CTA threads.
    scale_load_preshuffled_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
        stride=((4, 8, 16, 32, 64, 128, 256, 512, 1024), (1, 2)),
    )
    preshuffled_scale_shape: tl.constexpr = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    preshuffled_a_to_logical: tl.constexpr = (7, 8, 9, 4, 5, 6, 2, 3, 10, 0, 1)
    preshuffled_b_to_logical: tl.constexpr = (8, 9, 2, 4, 5, 6, 7, 3, 10, 0, 1)
    # ---- MFMA scale register layouts (get_mfma_scale_layout) ----
    scale_a_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
        stride=((8, 16, 32, 64, 1, 2, 0, 0, 128), (4, 256, 512)),
    )
    scale_a_comb_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
        stride=((8, 16, 32, 64, 1, 2, 0, 0, 128), (4, 256, 512, 1024)),
    )
    scale_b_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
        stride=((8, 16, 32, 64, 1, 2, 128, 256, 0), (4, 512)),
    )
    scale_b_comb_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
        stride=((8, 16, 32, 64, 1, 2, 128, 256, 0), (4, 512, 1024)),
    )
    # ---- MFMA accumulator layout (#mma 16x16x128 [2,4], one [128,128] quadrant) ----
    accumulator_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((128, 256, 512, 1024, 4, 8, 16, 32, 2048), (1, 2, 64, 4096, 8192)),
    )
    # ---- store layout (#blocked2, [128,128] bf16 quadrant) ----
    store_layout_c: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((8, 16, 32, 64, 128, 256, 512, 1024, 2048), (1, 2, 4, 4096, 8192)),
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

    # Four double-buffered operand half-tiles plus scale tiles.  The canonical
    # path keeps separate A halves; the experimental ABI stages one combined A
    # tile so its consumer can issue a single 128-bit read.
    smem_a_top = tlx.local_alloc((HALF_M, HALF_K), tlx.dtype_of(a_ptr), 2, layout=shared_tile)
    smem_a_bot = tlx.local_alloc((HALF_M, HALF_K), tlx.dtype_of(a_ptr), 2, layout=shared_tile)
    smem_b_left = tlx.local_alloc((HALF_N, HALF_K), tlx.dtype_of(b_ptr), 2, layout=shared_tile)
    smem_b_right = tlx.local_alloc((HALF_N, HALF_K), tlx.dtype_of(b_ptr), 2, layout=shared_tile)
    if PRESHUFFLED_SCALES:
        smem_a_sc = tlx.local_alloc(
            (BLOCK_M * NG, ), tlx.dtype_of(a_scales_ptr), 2, layout=shared_scale_preshuffled)
        smem_b_sc = tlx.local_alloc(
            (BLOCK_N * NG, ), tlx.dtype_of(b_scales_ptr), 2, layout=shared_scale_preshuffled)
    else:
        smem_a_sc_t = tlx.local_alloc((HALF_M, NG), tlx.dtype_of(a_scales_ptr), 2, layout=shared_a_scales)
        smem_a_sc_b = tlx.local_alloc((HALF_M, NG), tlx.dtype_of(a_scales_ptr), 2, layout=shared_a_scales)
        smem_b_sc = tlx.local_alloc((BLOCK_N, NG), tlx.dtype_of(b_scales_ptr), 2, layout=shared_b_scales)

    # ---- fp4 tile load offsets ([128,128]) ----
    offs_am = tl.arange(0, HALF_M)
    offs_ak = tl.arange(0, HALF_K)
    a_tile_offsets = tlx.require_layout(offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak, g_load_layout)
    a_base = a_ptr + pid_m * BLOCK_M * stride_am

    offs_bn = tl.arange(0, HALF_N)
    offs_bk = tl.arange(0, HALF_K)
    b_tile_offsets = tlx.require_layout(offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk, g_load_layout)
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # ---- A scale load offsets ----
    offs_ks_a = tl.arange(0, NG)
    if PRESHUFFLED_SCALES:
        a_sc_offsets = tlx.require_layout(tl.arange(0, BLOCK_M * NG), scale_load_preshuffled_layout)
    else:
        offs_asm = pid_m * BLOCK_M + tl.arange(0, HALF_M)
        a_sc_offsets = tl.mul(offs_asm[:, None], stride_asm, sanitize_overflow=False) + tl.mul(
            offs_ks_a[None, :], stride_ask, sanitize_overflow=False)
        a_sc_offsets = tlx.require_layout(a_sc_offsets, scale_load_a_layout)

    # ---- B scale load offsets: FULL [256,8] in one copy ----
    offs_ks_b = tl.arange(0, NG)
    if PRESHUFFLED_SCALES:
        b_sc_offsets = tlx.require_layout(tl.arange(0, BLOCK_N * NG), scale_load_preshuffled_layout)
    else:
        offs_bsn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        b_sc_offsets = tl.mul(offs_bsn[:, None], stride_bsn, sanitize_overflow=False) + tl.mul(
            offs_ks_b[None, :], stride_bsk, sanitize_overflow=False)
        b_sc_offsets = tlx.require_layout(b_sc_offsets, scale_load_b_layout)

    # Scalar (uniform) base-pointer deltas for the quadrant / K-buffer variants.
    a_half_m = HALF_M * stride_am  # a_top -> a_bot
    b_half_n = HALF_N * stride_bn  # b_left -> b_right
    a_k2 = HALF_K * stride_ak  # even -> odd (_next) K-step
    b_k2 = HALF_K * stride_bk
    if PRESHUFFLED_SCALES:
        a_sc_k: tl.constexpr = 2048
        b_sc_k: tl.constexpr = 2048
    else:
        a_sc_half_m = HALF_M * stride_asm
        a_sc_k = NG * stride_ask
        b_sc_k = NG * stride_bsk

    # Advance every base to this split's K-slice (no-op when SPLIT_K==1). The
    # A/B tiles are K-packed (KS_PACKED cols) and the scales are per-group
    # (KS_SCALE groups). All arith bases; buffer_load_to_local materializes the
    # fat pointer as it already does for the pid_m/pid_n base offsets.
    a_base += split_id * KS_PACKED * stride_ak
    b_base += split_id * KS_PACKED * stride_bk
    if PRESHUFFLED_SCALES:
        a_scales_ptr += pid_m * SCALE_GROUP_BLOCKS * 2048
        b_scales_ptr += pid_n * SCALE_GROUP_BLOCKS * 2048
        a_scales_ptr += split_id * (KS_SCALE // NG) * 2048
        b_scales_ptr += split_id * (KS_SCALE // NG) * 2048
    else:
        a_scales_ptr += split_id * KS_SCALE * stride_ask
        b_scales_ptr += split_id * KS_SCALE * stride_bsk

    acc_tl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_bl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_tr = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_br = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)

    # _matmul_256tile derives this trusted runtime value from the asserted KS.
    # The assumption keeps range/liveness analysis as precise as the old
    # constexpr bound while preserving the one-trip scf.for at K=1024.
    iter_max = K_TILES
    tl.assume(iter_max > 3)

    # ---- Prologue: prefetch K-steps 0,1 into buffers 0,1 (8 commits) ----
    tlx.buffer_load_to_local(smem_b_left[0], b_base, b_tile_offsets)
    tlx.buffer_load_to_local(smem_b_sc[0], b_scales_ptr, b_sc_offsets)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[0], a_base, a_tile_offsets)
    if PRESHUFFLED_SCALES:
        tlx.buffer_load_to_local(smem_a_sc[0], a_scales_ptr, a_sc_offsets)
    else:
        tlx.buffer_load_to_local(smem_a_sc_t[0], a_scales_ptr, a_sc_offsets)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[0], a_base + a_half_m, a_tile_offsets)
    if not PRESHUFFLED_SCALES:
        tlx.buffer_load_to_local(smem_a_sc_b[0], a_scales_ptr + a_sc_half_m, a_sc_offsets)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[0], b_base + b_half_n, b_tile_offsets)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_b_left[1], b_base + b_k2, b_tile_offsets)
    tlx.buffer_load_to_local(smem_b_sc[1], b_scales_ptr + b_sc_k, b_sc_offsets)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[1], a_base + a_k2, a_tile_offsets)
    if PRESHUFFLED_SCALES:
        tlx.buffer_load_to_local(smem_a_sc[1], a_scales_ptr + a_sc_k, a_sc_offsets)
    else:
        tlx.buffer_load_to_local(smem_a_sc_t[1], a_scales_ptr + a_sc_k, a_sc_offsets)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[1], a_base + a_half_m + a_k2, a_tile_offsets)
    if not PRESHUFFLED_SCALES:
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
    if PRESHUFFLED_SCALES:
        a_sc_view_init = tlx.local_reshape(smem_a_sc[0], preshuffled_scale_shape)
        a_sc_view_init = tlx.local_trans(a_sc_view_init, preshuffled_a_to_logical)
        a_sc_view_init = tlx.local_reshape(a_sc_view_init, [BLOCK_M, NG])
        a_sc_comb = tlx.local_load(a_sc_view_init, layout=scale_a_comb_layout, relaxed=True)
        a_sc_t, a_sc_b = tl.split(tl.trans(tl.reshape(a_sc_comb, 2, HALF_M, NG), 1, 2, 0))
        a_sc_top = tlx.require_layout(a_sc_t, scale_a_layout)
        a_sc_bot0 = tlx.require_layout(a_sc_b, scale_a_layout)
    else:
        a_sc_top = tlx.local_load(smem_a_sc_t[0], layout=scale_a_layout)
    if PRESHUFFLED_SCALES:
        b_sc_view_init = tlx.local_reshape(smem_b_sc[0], preshuffled_scale_shape)
        b_sc_view_init = tlx.local_trans(b_sc_view_init, preshuffled_b_to_logical)
        b_sc_view_init = tlx.local_reshape(b_sc_view_init, [BLOCK_N, NG])
        b_sc_comb = tlx.local_load(b_sc_view_init, layout=scale_b_comb_layout, relaxed=True)
    else:
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
            if PRESHUFFLED_SCALES:
                a_sc_bot = a_sc_bot0
            else:
                a_sc_bot = tlx.local_load(smem_a_sc_b[0], layout=scale_a_layout)
            tlx.amd_sched_barrier(0)  # keep ds_read(local_load) ahead of the global loads
            tlx.buffer_load_to_local(smem_b_left[0], b_base, b_tile_offsets)
            tlx.buffer_load_to_local(smem_b_sc[0], b_scales_ptr, b_sc_offsets)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_left, b_sc_left, "e2m1", acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
            tlx.amd_sched_barrier(0)  # keep ds_read(local_load) ahead of the global loads
            tlx.buffer_load_to_local(smem_a_top[0], a_base, a_tile_offsets)
            if PRESHUFFLED_SCALES:
                tlx.buffer_load_to_local(smem_a_sc[0], a_scales_ptr, a_sc_offsets)
            else:
                tlx.buffer_load_to_local(smem_a_sc_t[0], a_scales_ptr, a_sc_offsets)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_right, b_sc_right, "e2m1", acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
            tlx.amd_sched_barrier(0)  # keep ds_read(local_load) ahead of the global loads
            tlx.buffer_load_to_local(smem_a_bot[0], a_base + a_half_m, a_tile_offsets)
            if not PRESHUFFLED_SCALES:
                tlx.buffer_load_to_local(smem_a_sc_b[0], a_scales_ptr + a_sc_half_m, a_sc_offsets)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_right, b_sc_right, "e2m1", acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[1], relaxed=True)
            if PRESHUFFLED_SCALES:
                a_sc_view_1 = tlx.local_reshape(smem_a_sc[1], preshuffled_scale_shape)
                a_sc_view_1 = tlx.local_trans(a_sc_view_1, preshuffled_a_to_logical)
                a_sc_view_1 = tlx.local_reshape(a_sc_view_1, [BLOCK_M, NG])
                a_sc_comb = tlx.local_load(a_sc_view_1, layout=scale_a_comb_layout, relaxed=True)
                a_sc_t, a_sc_b = tl.split(tl.trans(tl.reshape(a_sc_comb, 2, HALF_M, NG), 1, 2, 0))
                a_sc_top = tlx.require_layout(a_sc_t, scale_a_layout)
                a_sc_bot1 = tlx.require_layout(a_sc_b, scale_a_layout)
            else:
                a_sc_top = tlx.local_load(smem_a_sc_t[1], layout=scale_a_layout)
            if PRESHUFFLED_SCALES:
                b_sc_view_1 = tlx.local_reshape(smem_b_sc[1], preshuffled_scale_shape)
                b_sc_view_1 = tlx.local_trans(b_sc_view_1, preshuffled_b_to_logical)
                b_sc_view_1 = tlx.local_reshape(b_sc_view_1, [BLOCK_N, NG])
                b_sc_comb = tlx.local_load(b_sc_view_1, layout=scale_b_comb_layout, relaxed=True)
            else:
                b_sc_comb = tlx.local_load(smem_b_sc[1], layout=scale_b_comb_layout)
            b_sc_l, b_sc_r = tl.split(tl.trans(tl.reshape(b_sc_comb, 2, HALF_N, NG), 1, 2, 0))
            b_sc_left = tlx.require_layout(b_sc_l, scale_b_layout)
            b_sc_right = tlx.require_layout(b_sc_r, scale_b_layout)
            tlx.amd_sched_barrier(0)  # keep ds_read(local_load) ahead of the global loads
            tlx.buffer_load_to_local(smem_b_right[0], b_base + b_half_n, b_tile_offsets)
            tlx.async_load_commit_group()

        # --- sub-iter 1 (buffer 1, base + one K-step) ---
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_left, b_sc_left, "e2m1", acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
            if PRESHUFFLED_SCALES:
                a_sc_bot = a_sc_bot1
            else:
                a_sc_bot = tlx.local_load(smem_a_sc_b[1], layout=scale_a_layout)
            tlx.amd_sched_barrier(0)  # keep ds_read(local_load) ahead of the global loads
            tlx.buffer_load_to_local(smem_b_left[1], b_base + b_k2, b_tile_offsets)
            tlx.buffer_load_to_local(smem_b_sc[1], b_scales_ptr + b_sc_k, b_sc_offsets)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_left, b_sc_left, "e2m1", acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
            tlx.amd_sched_barrier(0)  # keep ds_read(local_load) ahead of the global loads
            tlx.buffer_load_to_local(smem_a_top[1], a_base + a_k2, a_tile_offsets)
            if PRESHUFFLED_SCALES:
                tlx.buffer_load_to_local(smem_a_sc[1], a_scales_ptr + a_sc_k, a_sc_offsets)
            else:
                tlx.buffer_load_to_local(smem_a_sc_t[1], a_scales_ptr + a_sc_k, a_sc_offsets)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_right, b_sc_right, "e2m1", acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
            tlx.amd_sched_barrier(0)  # keep ds_read(local_load) ahead of the global loads
            tlx.buffer_load_to_local(smem_a_bot[1], a_base + a_half_m + a_k2, a_tile_offsets)
            if not PRESHUFFLED_SCALES:
                tlx.buffer_load_to_local(smem_a_sc_b[1], a_scales_ptr + a_sc_half_m + a_sc_k, a_sc_offsets)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_right, b_sc_right, "e2m1", acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[0], relaxed=True)
            if PRESHUFFLED_SCALES:
                a_sc_view_0 = tlx.local_reshape(smem_a_sc[0], preshuffled_scale_shape)
                a_sc_view_0 = tlx.local_trans(a_sc_view_0, preshuffled_a_to_logical)
                a_sc_view_0 = tlx.local_reshape(a_sc_view_0, [BLOCK_M, NG])
                a_sc_comb = tlx.local_load(a_sc_view_0, layout=scale_a_comb_layout, relaxed=True)
                a_sc_t, a_sc_b = tl.split(tl.trans(tl.reshape(a_sc_comb, 2, HALF_M, NG), 1, 2, 0))
                a_sc_top = tlx.require_layout(a_sc_t, scale_a_layout)
                a_sc_bot0 = tlx.require_layout(a_sc_b, scale_a_layout)
            else:
                a_sc_top = tlx.local_load(smem_a_sc_t[0], layout=scale_a_layout)
            if PRESHUFFLED_SCALES:
                b_sc_view_0 = tlx.local_reshape(smem_b_sc[0], preshuffled_scale_shape)
                b_sc_view_0 = tlx.local_trans(b_sc_view_0, preshuffled_b_to_logical)
                b_sc_view_0 = tlx.local_reshape(b_sc_view_0, [BLOCK_N, NG])
                b_sc_comb = tlx.local_load(b_sc_view_0, layout=scale_b_comb_layout, relaxed=True)
            else:
                b_sc_comb = tlx.local_load(smem_b_sc[0], layout=scale_b_comb_layout)
            b_sc_l, b_sc_r = tl.split(tl.trans(tl.reshape(b_sc_comb, 2, HALF_N, NG), 1, 2, 0))
            b_sc_left = tlx.require_layout(b_sc_l, scale_b_layout)
            b_sc_right = tlx.require_layout(b_sc_r, scale_b_layout)
            tlx.amd_sched_barrier(0)  # keep ds_read(local_load) ahead of the global loads
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
    if PRESHUFFLED_SCALES:
        a_sc_bot = a_sc_bot0
    else:
        a_sc_bot = tlx.local_load(smem_a_sc_b[l_idx], layout=scale_a_layout)

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
    if PRESHUFFLED_SCALES:
        a_sc_view_epilogue = tlx.local_reshape(smem_a_sc[g_idx], preshuffled_scale_shape)
        a_sc_view_epilogue = tlx.local_trans(a_sc_view_epilogue, preshuffled_a_to_logical)
        a_sc_view_epilogue = tlx.local_reshape(a_sc_view_epilogue, [BLOCK_M, NG])
        a_sc_comb = tlx.local_load(a_sc_view_epilogue, layout=scale_a_comb_layout, relaxed=True)
        a_sc_t, a_sc_b = tl.split(tl.trans(tl.reshape(a_sc_comb, 2, HALF_M, NG), 1, 2, 0))
        a_sc_top = tlx.require_layout(a_sc_t, scale_a_layout)
        a_sc_bot1 = tlx.require_layout(a_sc_b, scale_a_layout)
    else:
        a_sc_top = tlx.local_load(smem_a_sc_t[g_idx], layout=scale_a_layout)
    if PRESHUFFLED_SCALES:
        b_sc_view_epilogue = tlx.local_reshape(smem_b_sc[g_idx], preshuffled_scale_shape)
        b_sc_view_epilogue = tlx.local_trans(b_sc_view_epilogue, preshuffled_b_to_logical)
        b_sc_view_epilogue = tlx.local_reshape(b_sc_view_epilogue, [BLOCK_N, NG])
        b_sc_comb = tlx.local_load(b_sc_view_epilogue, layout=scale_b_comb_layout, relaxed=True)
    else:
        b_sc_comb = tlx.local_load(smem_b_sc[g_idx], layout=scale_b_comb_layout)
    b_sc_l, b_sc_r = tl.split(tl.trans(tl.reshape(b_sc_comb, 2, HALF_N, NG), 1, 2, 0))
    b_sc_left = tlx.require_layout(b_sc_l, scale_b_layout)
    b_sc_right = tlx.require_layout(b_sc_r, scale_b_layout)

    # iter iter_max-1: finish ALL four accumulators, then convert + store.
    acc_tl = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_left, b_sc_left, "e2m1", acc_tl)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, g_idx), relaxed=True)
    if PRESHUFFLED_SCALES:
        a_sc_bot = a_sc_bot1
    else:
        a_sc_bot = tlx.local_load(smem_a_sc_b[g_idx], layout=scale_a_layout)

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
# Each split must contain the two-tile prologue plus one two-tile main step.
MIN_KTILES_PER_SPLIT = MIN_K // BLOCK_K  # == 4


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


def _pad_mxfp4_scales(scales):
    assert scales.dtype is torch.uint8
    assert scales.is_cuda
    assert scales.ndim == 2
    rows, groups = scales.shape
    assert rows > 0 and groups > 0
    padded_rows = triton.cdiv(rows, 256) * 256
    padded_groups = triton.cdiv(groups, 8) * 8
    padded = torch.full(
        (padded_rows, padded_groups),
        0x7F,
        dtype=torch.uint8,
        device=scales.device,
    )
    padded[:rows, :groups] = scales
    return padded, padded_rows, padded_groups


def preshuffle_mxfp4_a_scales(scales):
    """Pack A scales so a combined 256x8 LDS tile is read with ``ds_read_b128``.

    The four physical low bits are group bit 2 and row bits 5, 6, and 7: the
    exact value bases of ``scale_a_comb_layout``. Padding uses the E8M0 identity
    value (0x7f).
    """
    padded, padded_rows, padded_groups = _pad_mxfp4_scales(scales)
    return (
        padded.reshape(
            padded_rows // 256, 2, 2, 2, 2, 2, 2, 2, 2,
            padded_groups // 8, 2, 2, 2)
        .permute(0, 9, 11, 12, 7, 8, 4, 5, 6, 1, 2, 3, 10)
        .contiguous()
        .reshape(-1)
    )


def preshuffle_mxfp4_b_scales(scales):
    """Pack B scales into the ordinary 64-bit LDS consumer order."""
    padded, padded_rows, padded_groups = _pad_mxfp4_scales(scales)
    return (
        padded.reshape(
            padded_rows // 256, 2, 2, 2, 2, 2, 2, 2, 2,
            padded_groups // 8, 2, 2, 2)
        .permute(0, 9, 11, 12, 3, 8, 4, 5, 6, 7, 1, 2, 10)
        .contiguous()
        .reshape(-1)
    )


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
    assert K >= MIN_K and K % (2 * BLOCK_K) == 0, \
        f"K must be at least {MIN_K} and a multiple of {2 * BLOCK_K}"

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
        KS // BLOCK_K,
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
        PRESHUFFLED_SCALES=False,
        num_warps=NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=16,
        # Keep f32 accumulators in VGPRs. Allowing AGPR allocation increases
        # private-segment spills and regresses this 8-wave ownership model.
        llvm_fn_attrs=_A4W4_8WAVE_LLVM_FN_ATTRS,
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


def matmul_preshuffled(a, b, a_scales, b_scales, SPLIT_K=None):
    """Experimental pure-DMA kernel for two preshuffled scale buffers.

    Both scale arguments are flat buffers returned by the corresponding
    :func:`preshuffle_mxfp4_a_scales` and
    :func:`preshuffle_mxfp4_b_scales` helpers.
    """
    assert a.dtype is torch.uint8
    assert b.dtype is torch.uint8
    assert a_scales.dtype is torch.uint8
    assert b_scales.dtype is torch.uint8
    assert a.is_cuda and b.is_cuda and a_scales.is_cuda and b_scales.is_cuda
    assert a_scales.ndim == 1 and b_scales.ndim == 1

    M = a.shape[0]
    K_packed = a.shape[1]
    K = K_packed * 2
    N = b.shape[0]
    groups = K // 32
    padded_groups = triton.cdiv(groups, 8) * 8
    assert b.shape == (N, K_packed), "B must have shape (N, K // 2)"
    assert a_scales.numel() == triton.cdiv(M, 256) * 256 * padded_groups
    assert b_scales.numel() == triton.cdiv(N, 256) * 256 * padded_groups
    assert M % BLOCK_M == 0, "M must be a multiple of 256"
    assert N % BLOCK_N == 0, "N must be a multiple of 256"
    assert K >= MIN_K and K % (2 * BLOCK_K) == 0, \
        f"K must be at least {MIN_K} and a multiple of {2 * BLOCK_K}"

    if SPLIT_K is None:
        SPLIT_K = choose_split_k(M, N, K)
    KS = K // SPLIT_K
    assert K % SPLIT_K == 0, f"K={K} must be divisible by SPLIT_K={SPLIT_K}"
    assert KS >= MIN_K and KS % (2 * BLOCK_K) == 0, \
        f"K/SPLIT_K={KS} must be >= {MIN_K} and a multiple of {2 * BLOCK_K}"

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
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
        KS // BLOCK_K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        0,
        0,
        0,
        0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_XCDS=NUM_XCDS,
        GRID_MN=grid_mn,
        SPLIT_K=SPLIT_K,
        PRESHUFFLED_SCALES=True,
        num_warps=NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=16,
        llvm_fn_attrs=_A4W4_8WAVE_LLVM_FN_ATTRS,
    )
    if SPLIT_K > 1:
        big = (M * N) >= (2048 * 2048)
        rbm, rbn, rw = (128, 128, 8) if big else (32, 32, 4)
        _reduce_k_kernel[(triton.cdiv(M, rbm), triton.cdiv(N, rbn))](
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


def select_matmul_path(M, N, K):
    """Select the measured gfx950 tile and wave-ownership strategy."""
    if is_skinny(M, N, K):
        return "skinny"
    if K <= INTRA_WAVE_MAX_K:
        return "intra_wave_256x256"
    return "inter_wave_256x256"


def matmul(a, b, a_scales, b_scales):
    """A @ B.T for packed MXFP4 A/B using measured gfx950 dispatch.

    * occupancy-starved grids use measured 32/64/128x128 tiles and bounded split-K;
    * well-filled K=1024/1536 grids use the lower-overhead 4-wave 256x256 path;
    * K >= 2048 well-filled grids use the 8-wave 256x256 inter-wave pipeline.
    """
    M = a.shape[0]
    K = a.shape[1] * 2
    N = b.shape[0]
    path = select_matmul_path(M, N, K)
    if path == "skinny":
        return skinny_matmul(a, b, a_scales, b_scales)
    if path == "intra_wave_256x256":
        return intra_wave_matmul(a, b, a_scales, b_scales)
    return _matmul_256tile(a, b, a_scales, b_scales)
