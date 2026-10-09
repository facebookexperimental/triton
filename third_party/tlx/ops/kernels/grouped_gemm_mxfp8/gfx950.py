"""gfx950 MXFP8 grouped GEMM for ``tlx.ops.grouped_gemm_mxfp8``.

Forward-only, persistent kernel for E4M3 operands, E8M0 per-32 microscales, and
BF16 output. It reuses the FP16 gfx950 grouped GEMM pipeline
(``grouped_gemm/gfx950.py``): a 256x256 output tile split into four 128x128
quadrants, one double-buffered LDS allocation per operand half-tile, and an
8-wave ``warp_pipeline_stage`` hot loop. ``tl.dot`` becomes ``tl.dot_scaled``
(CDNA4 16x16x128 scaled MFMA), and each half-tile's [128, 4] scale slice is
copied direct-to-LDS in the same commit group as its data.

Groups partition M. ``split_sizes`` stays on device; every program rescans it,
so the static persistent stride is graph-safe without a counter.

Unlike the FP16 kernel, this one keeps LLVM's default post-RA machine scheduler
(no ``TRITON_DISABLE_POST_MISCHED`` / nop strategy): with the scaled MFMAs and
scale reads it measured 1-8% faster across the TritonBench shapes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tlx.ops.kernels.grouped_gemm.gfx950 import (
    chiplet_transform_chunked,
    NUM_XCDS,
)
from triton.tlx.ops.kernels.grouped_gemm_mxfp8._scales import prepare_scales

_BLOCK_K = 128
# AMD buffer descriptors expose at most this many bytes from each scalar base.
_MAX_BUFFER_BYTES = (1 << 31) - 2

# [128, 4] E8M0 scale slice (flat r * 4 + c): each lane copies one row's 4
# contiguous bytes, so direct-to-LDS issues 32-bit copies. The two extra thread
# bits broadcast (stride 0) over the 512-thread workgroup.
_SCALE_COPY_LAYOUT = tlx.layout(shape=((128, 4), (4, )), stride=((4, 0), (1, )))

# K_MAJOR atoms (natural inputs on gfx950) store each 16-byte chunk as
# [k][row_group]: byte (r, c) of a 128-row block sits at
# (r % 32) * 16 + c * 4 + r // 32. The four row-group bytes one scaled MFMA packs
# into a register are then adjacent, so the atom is copied as flat [32, 16]
# bytes and read back through _scale_view with one ds_read_b32 instead of four
# ds_read_u8 plus v_perm.
_SCALE_FLAT_COPY_LAYOUT = tlx.layout(shape=((4, 32, 4), (4, )), stride=((4, 16, 0), (1, )))

# 16x16x128 scaled-MFMA scale register layouts (#mma warpsPerCTA=[2, 4]) for the
# [128, 4] A and B scale slices. Loading LDS scales straight into these avoids a
# convert_layout through shared scratch, which the two phase-shifted wave groups
# of the inter-wave pipeline would otherwise race on.
_SA_MFMA_LAYOUT = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
    stride=((4, 8, 16, 32, 1, 2, 0, 0, 64), (128, 256)),
)
_SB_MFMA_LAYOUT = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, )),
    stride=((4, 8, 16, 32, 1, 2, 64, 128, 0), (256, )),
)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@triton.jit
def _device_trap_if(condition):
    """Abort the kernel (s_trap 2) when the uniform scalar condition holds."""
    if condition:
        tl.inline_asm_elementwise(
            "s_trap 2\nv_mov_b32 $0, 0",
            "=v",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _scale_offsets(rows, k_chunks, K_MAJOR: tl.constexpr):
    """Byte offsets of the [rows, 4] scale slice for K-step 0 in the blocked layout."""
    cols = tl.arange(0, 4)
    base = (rows[:, None] // 128) * k_chunks * 512 + (rows[:, None] % 32) * 16
    if K_MAJOR:
        offsets = base + cols[None, :] * 4 + (rows[:, None] % 128) // 32
    else:
        offsets = base + ((rows[:, None] % 128) // 32) * 4 + cols[None, :]
    return offsets


@triton.jit
def _kmajor_scale_view(buf):
    read_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[32, 0], [64, 0], [0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        block_bases=[],
        alignment=4,
    )
    return tlx.local_reinterpret(buf, tl.uint8, [128, 4], layout=read_layout)


@triton.jit
def _load_scale(buf, layout: tl.constexpr, K_MAJOR: tl.constexpr):
    if K_MAJOR:
        value = tlx.local_load(_kmajor_scale_view(buf), layout=layout, relaxed=True)
    else:
        value = tlx.local_load(buf, layout=layout, relaxed=True)
    return value


@triton.jit
def _quad_tile(
    pid_m,
    pid_n,
    a_ptr,
    b_ptr,
    c_ptr,
    as_ptr,
    bs_ptr,
    m_start,
    m_size,
    N,
    K,
    k_chunks,
    smem_a_top,
    smem_a_bot,
    smem_b_left,
    smem_b_right,
    smem_sa_top,
    smem_sa_bot,
    smem_sb_left,
    smem_sb_right,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    opaque_zero,
    K_MAJOR: tl.constexpr,
):
    """One [256, 256] output tile as four 128x128 scaled-MFMA quadrants."""
    HALF_M: tl.constexpr = BLOCK_SIZE_M // 2
    HALF_N: tl.constexpr = BLOCK_SIZE_N // 2
    SC_STEP: tl.constexpr = 512  # one 128x4 scale atom per K-step

    # Group M is a multiple of 128, so each 128-row A half-tile is one whole
    # row block (wrapped to block 0 past the group end) and is addressed by a
    # scalar base plus one shared [128, BLOCK_K] offset tensor. Its scales are
    # exactly one 512-byte atom per K-step. Wrapped garbage rows/cols are dropped
    # by the masked store.
    offs_r = tl.arange(0, HALF_M)
    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)
    # opaque_zero (runtime 0) keeps these tile-invariant lane offsets inside the
    # tile. Hoisted above the persistent loop they spill, and the scratch reload
    # at each tile's prologue forces s_waitcnt vmcnt(0), which drains the previous
    # tile's output stores instead of overlapping them.
    tile_off = offs_r[:, None] * K + offs_k[None, :] + opaque_zero * 256
    if K_MAJOR:
        sc_off = tlx.require_layout(
            tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :] + opaque_zero * 256, _SCALE_FLAT_COPY_LAYOUT)
    else:
        sc_cols = tl.arange(0, 4)
        sc_off = tlx.require_layout(
            (offs_r[:, None] % 32) * 16 + (offs_r[:, None] // 32) * 4 + sc_cols[None, :] + opaque_zero * 256,
            _SCALE_COPY_LAYOUT)

    row_top = (pid_m.to(tl.int64) * BLOCK_SIZE_M) % m_size
    row_bot = (pid_m.to(tl.int64) * BLOCK_SIZE_M + HALF_M) % m_size
    a_top_base = a_ptr + row_top.to(tl.int64) * K
    a_bot_base = a_ptr + row_bot.to(tl.int64) * K
    a_top_off = tile_off
    a_bot_off = tile_off
    sa_top_base = as_ptr + ((m_start + row_top) // 128).to(tl.int64) * k_chunks * 512
    sa_bot_base = as_ptr + ((m_start + row_bot) // 128).to(tl.int64) * k_chunks * 512
    sa_top_off = sc_off
    sa_bot_off = sc_off

    if N_ALIGNED:
        col_left = (pid_n.to(tl.int64) * BLOCK_SIZE_N) % N
        col_right = (pid_n.to(tl.int64) * BLOCK_SIZE_N + HALF_N) % N
        b_left_base = b_ptr + col_left.to(tl.int64) * K
        b_right_base = b_ptr + col_right.to(tl.int64) * K
        b_left_off = tile_off
        b_right_off = tile_off
        sb_left_base = bs_ptr + (col_left // 128).to(tl.int64) * k_chunks * 512
        sb_right_base = bs_ptr + (col_right // 128).to(tl.int64) * k_chunks * 512
        sb_left_off = sc_off
        sb_right_off = sc_off
    else:
        # Rebase each ragged half at its aligned first column. Invalid columns
        # read that first column and are discarded by the output mask. Scales
        # already have padding to a whole 128-row atom.
        col_left = pid_n.to(tl.int64) * BLOCK_SIZE_N
        col_right = col_left + HALF_N
        col_left = tl.where(col_left < N, col_left, 0)
        col_right = tl.where(col_right < N, col_right, 0)
        b_left_base = b_ptr + col_left * K
        b_right_base = b_ptr + col_right * K
        left_rows = tl.where(offs_r < N - col_left, offs_r, 0)
        right_rows = tl.where(offs_r < N - col_right, offs_r, 0)
        b_left_off = left_rows[:, None] * K + offs_k[None, :]
        b_right_off = right_rows[:, None] * K + offs_k[None, :]
        sb_left_base = bs_ptr + (col_left // 128) * k_chunks * 512
        sb_right_base = bs_ptr + (col_right // 128) * k_chunks * 512
        sb_left_off = sc_off
        sb_right_off = sc_off

    kb1: tl.constexpr = BLOCK_SIZE_K
    ka = tl.zeros([], dtype=tl.int32)
    ks = tl.zeros([], dtype=tl.int32)

    acc_tl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_bl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_tr = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_br = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)

    n_full = K // BLOCK_SIZE_K
    n_pipe = (n_full // 2) * 2

    if n_full >= 2:
        # Prologue: K-steps 0, 1 into buffers 0, 1 (8 commits, scales ride along).
        tlx.buffer_load_to_local(smem_b_left[0], b_left_base, b_left_off + ka)
        tlx.buffer_load_to_local(smem_sb_left[0], sb_left_base, sb_left_off + ks)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_a_top[0], a_top_base, a_top_off + ka)
        tlx.buffer_load_to_local(smem_sa_top[0], sa_top_base, sa_top_off + ks)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_a_bot[0], a_bot_base, a_bot_off + ka)
        tlx.buffer_load_to_local(smem_sa_bot[0], sa_bot_base, sa_bot_off + ks)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_b_right[0], b_right_base, b_right_off + ka)
        tlx.buffer_load_to_local(smem_sb_right[0], sb_right_base, sb_right_off + ks)
        tlx.async_load_commit_group()

        tlx.buffer_load_to_local(smem_b_left[1], b_left_base, b_left_off + (ka + kb1))
        tlx.buffer_load_to_local(smem_sb_left[1], sb_left_base, sb_left_off + (ks + SC_STEP))
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_a_top[1], a_top_base, a_top_off + (ka + kb1))
        tlx.buffer_load_to_local(smem_sa_top[1], sa_top_base, sa_top_off + (ks + SC_STEP))
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_a_bot[1], a_bot_base, a_bot_off + (ka + kb1))
        tlx.buffer_load_to_local(smem_sa_bot[1], sa_bot_base, sa_bot_off + (ks + SC_STEP))
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_b_right[1], b_right_base, b_right_off + (ka + kb1))
        tlx.buffer_load_to_local(smem_sb_right[1], sb_right_base, sb_right_off + (ks + SC_STEP))
        tlx.async_load_commit_group()

        ka += BLOCK_SIZE_K * 2
        ks += SC_STEP * 2

        tlx.async_load_wait_group(6)
        b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
        sb_l = _load_scale(smem_sb_left[0], _SB_MFMA_LAYOUT, K_MAJOR)
        a_top = tlx.local_load(smem_a_top[0], relaxed=True)
        sa_t = _load_scale(smem_sa_top[0], _SA_MFMA_LAYOUT, K_MAJOR)

        for k in tl.range(0, n_pipe - 2, 2, num_stages=1):
            # sub-iter 0 (buffer 0)
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tl = tl.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
                sa_b = _load_scale(smem_sa_bot[0], _SA_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_b_left[0], b_left_base, b_left_off + ka)
                tlx.buffer_load_to_local(smem_sb_left[0], sb_left_base, sb_left_off + ks)
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_bl = tl.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
                sb_r = _load_scale(smem_sb_right[0], _SB_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_a_top[0], a_top_base, a_top_off + ka)
                tlx.buffer_load_to_local(smem_sa_top[0], sa_top_base, sa_top_off + ks)
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tr = tl.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
                sb_l = _load_scale(smem_sb_left[1], _SB_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_a_bot[0], a_bot_base, a_bot_off + ka)
                tlx.buffer_load_to_local(smem_sa_bot[0], sa_bot_base, sa_bot_off + ks)
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_br = tl.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_top = tlx.local_load(smem_a_top[1], relaxed=True)
                sa_t = _load_scale(smem_sa_top[1], _SA_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_b_right[0], b_right_base, b_right_off + ka)
                tlx.buffer_load_to_local(smem_sb_right[0], sb_right_base, sb_right_off + ks)
                tlx.async_load_commit_group()

            # sub-iter 1 (buffer 1)
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tl = tl.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
                sa_b = _load_scale(smem_sa_bot[1], _SA_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_b_left[1], b_left_base, b_left_off + (ka + kb1))
                tlx.buffer_load_to_local(smem_sb_left[1], sb_left_base, sb_left_off + (ks + SC_STEP))
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_bl = tl.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
                sb_r = _load_scale(smem_sb_right[1], _SB_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_a_top[1], a_top_base, a_top_off + (ka + kb1))
                tlx.buffer_load_to_local(smem_sa_top[1], sa_top_base, sa_top_off + (ks + SC_STEP))
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tr = tl.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
                sb_l = _load_scale(smem_sb_left[0], _SB_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_a_bot[1], a_bot_base, a_bot_off + (ka + kb1))
                tlx.buffer_load_to_local(smem_sa_bot[1], sa_bot_base, sa_bot_off + (ks + SC_STEP))
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_br = tl.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_top = tlx.local_load(smem_a_top[0], relaxed=True)
                sa_t = _load_scale(smem_sa_top[0], _SA_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_b_right[1], b_right_base, b_right_off + (ka + kb1))
                tlx.buffer_load_to_local(smem_sb_right[1], sb_right_base, sb_right_off + (ks + SC_STEP))
                tlx.async_load_commit_group()
                ka += BLOCK_SIZE_K * 2
                ks += SC_STEP * 2

        # Epilogue: last two pipelined K-steps, draining the LDS ring.
        acc_tl = tl.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
        tlx.async_load_wait_group(5)
        a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
        sa_b = _load_scale(smem_sa_bot[0], _SA_MFMA_LAYOUT, K_MAJOR)

        acc_bl = tl.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
        tlx.async_load_wait_group(4)
        b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
        sb_r = _load_scale(smem_sb_right[0], _SB_MFMA_LAYOUT, K_MAJOR)

        acc_tr = tl.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
        tlx.async_load_wait_group(3)
        b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
        sb_l = _load_scale(smem_sb_left[1], _SB_MFMA_LAYOUT, K_MAJOR)

        acc_br = tl.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
        tlx.async_load_wait_group(2)
        a_top = tlx.local_load(smem_a_top[1], relaxed=True)
        sa_t = _load_scale(smem_sa_top[1], _SA_MFMA_LAYOUT, K_MAJOR)

        acc_tl = tl.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
        tlx.async_load_wait_group(1)
        a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
        sa_b = _load_scale(smem_sa_bot[1], _SA_MFMA_LAYOUT, K_MAJOR)

        acc_bl = tl.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
        tlx.async_load_wait_group(0)
        b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
        sb_r = _load_scale(smem_sb_right[1], _SB_MFMA_LAYOUT, K_MAJOR)

        acc_tr = tl.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
        acc_br = tl.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)

    # Cold tail: an odd leftover whole K-tile, or all of K when K == 128.
    for kk in tl.range(n_pipe, n_full, num_stages=1):
        kd = kk * BLOCK_SIZE_K
        kss = kk * SC_STEP
        a_top_t = tl.load(a_top_base + a_top_off + kd)
        a_bot_t = tl.load(a_bot_base + a_bot_off + kd)
        b_left_t = tl.trans(tl.load(b_left_base + b_left_off + kd))
        b_right_t = tl.trans(tl.load(b_right_base + b_right_off + kd))
        if K_MAJOR:
            sc_tail = _scale_offsets(tl.arange(0, HALF_M), 0, True)
            sa_t_t = tl.load(sa_top_base + sc_tail + kss)
            sa_b_t = tl.load(sa_bot_base + sc_tail + kss)
            sb_l_t = tl.load(sb_left_base + sc_tail + kss)
            sb_r_t = tl.load(sb_right_base + sc_tail + kss)
        else:
            sa_t_t = tl.load(sa_top_base + sa_top_off + kss)
            sa_b_t = tl.load(sa_bot_base + sa_bot_off + kss)
            sb_l_t = tl.load(sb_left_base + sb_left_off + kss)
            sb_r_t = tl.load(sb_right_base + sb_right_off + kss)
        acc_tl = tl.dot_scaled(a_top_t, sa_t_t, "e4m3", b_left_t, sb_l_t, "e4m3", acc_tl)
        acc_bl = tl.dot_scaled(a_bot_t, sa_b_t, "e4m3", b_left_t, sb_l_t, "e4m3", acc_bl)
        acc_tr = tl.dot_scaled(a_top_t, sa_t_t, "e4m3", b_right_t, sb_r_t, "e4m3", acc_tr)
        acc_br = tl.dot_scaled(a_bot_t, sa_b_t, "e4m3", b_right_t, sb_r_t, "e4m3", acc_br)

    offs_cm_top = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, HALF_M)
    offs_cm_bot = offs_cm_top + HALF_M
    offs_cn_left = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, HALF_N)
    offs_cn_right = offs_cn_left + HALF_N
    tl.store(
        c_ptr + offs_cm_top[:, None] * N + offs_cn_left[None, :],
        acc_tl.to(tl.bfloat16),
        mask=(offs_cm_top[:, None] < m_size) & (offs_cn_left[None, :] < N),
    )
    tl.store(
        c_ptr + offs_cm_bot[:, None] * N + offs_cn_left[None, :],
        acc_bl.to(tl.bfloat16),
        mask=(offs_cm_bot[:, None] < m_size) & (offs_cn_left[None, :] < N),
    )
    tl.store(
        c_ptr + offs_cm_top[:, None] * N + offs_cn_right[None, :],
        acc_tr.to(tl.bfloat16),
        mask=(offs_cm_top[:, None] < m_size) & (offs_cn_right[None, :] < N),
    )
    tl.store(
        c_ptr + offs_cm_bot[:, None] * N + offs_cn_right[None, :],
        acc_br.to(tl.bfloat16),
        mask=(offs_cm_bot[:, None] < m_size) & (offs_cn_right[None, :] < N),
    )


@triton.jit
def _generic_tile(
    pid_m,
    pid_n,
    a_ptr,
    b_ptr,
    c_ptr,
    as_ptr,
    bs_ptr,
    m_start,
    m_size,
    N,
    K,
    k_chunks,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    K_MAJOR: tl.constexpr,
):
    """Compiler-pipelined tile for problems too small to fill the CUs at 256x256."""
    # Widen row indices before multiplying by data and scale strides.
    lm = (pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m_size
    nn = (pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)

    a_ptrs = a_ptr + lm[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] + nn[None, :] * K
    sa_ptrs = as_ptr + _scale_offsets(m_start + lm, k_chunks, K_MAJOR)
    sb_ptrs = bs_ptr + _scale_offsets(nn, k_chunks, K_MAJOR)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in tl.range(0, K // BLOCK_SIZE_K, num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        sa = tl.load(sa_ptrs)
        sb = tl.load(sb_ptrs)
        acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        sa_ptrs += 512
        sb_ptrs += 512

    offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(
        c_ptr + offs_cm[:, None] * N + offs_cn[None, :],
        acc.to(tl.bfloat16),
        mask=(offs_cm[:, None] < m_size) & (offs_cn[None, :] < N),
    )


@triton.jit
def _mxfp8_grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    stride_b_scale_g,
    split_sizes_ptr,
    G,
    M,
    N,
    K,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    TILE_MODE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    K_MAJOR: tl.constexpr,
):
    """Persistent XCD-remapped scheduler over the concatenated per-group tile space."""
    pid = chiplet_transform_chunked(tl.program_id(0), NUM_SM, NUM_XCDS, XCD_CHUNK).to(tl.int64)
    k_chunks = K // BLOCK_SIZE_K

    if TILE_MODE == 0:
        HALF_M: tl.constexpr = BLOCK_SIZE_M // 2
        HALF_N: tl.constexpr = BLOCK_SIZE_N // 2
        tl.static_assert(HALF_M == 128 and HALF_N == 128 and BLOCK_SIZE_K == 128,
                         "padded LDS bases are pinned for [128, 128] byte half-tiles")
        # Same byte geometry as the gfx950 a4w4 tutorial's packed-FP4 half-tile:
        # pad 32 bytes every 1024 to break ds_read bank conflicts.
        smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [[1024, 32]],
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
            [HALF_M, BLOCK_SIZE_K],
        )
        dt_a = tlx.dtype_of(a_ptr)
        dt_s = tlx.dtype_of(a_scale_ptr)
        smem_a_top = tlx.local_alloc((HALF_M, BLOCK_SIZE_K), dt_a, 2, layout=smem_layout)
        smem_a_bot = tlx.local_alloc((HALF_M, BLOCK_SIZE_K), dt_a, 2, layout=smem_layout)
        smem_b_left = tlx.local_alloc((HALF_N, BLOCK_SIZE_K), dt_a, 2, layout=smem_layout)
        smem_b_right = tlx.local_alloc((HALF_N, BLOCK_SIZE_K), dt_a, 2, layout=smem_layout)
        if K_MAJOR:
            smem_sa_top = tlx.local_alloc((32, 16), dt_s, 2)
            smem_sa_bot = tlx.local_alloc((32, 16), dt_s, 2)
            smem_sb_left = tlx.local_alloc((32, 16), dt_s, 2)
            smem_sb_right = tlx.local_alloc((32, 16), dt_s, 2)
        else:
            smem_sa_top = tlx.local_alloc((HALF_M, BLOCK_SIZE_K // 32), dt_s, 2)
            smem_sa_bot = tlx.local_alloc((HALF_M, BLOCK_SIZE_K // 32), dt_s, 2)
            smem_sb_left = tlx.local_alloc((HALF_N, BLOCK_SIZE_K // 32), dt_s, 2)
            smem_sb_right = tlx.local_alloc((HALF_N, BLOCK_SIZE_K // 32), dt_s, 2)

    tile_idx = pid
    last_problem_end = tl.full((), 0, tl.int64)
    running_m = tl.full((), 0, tl.int64)
    num_n_tiles = tl.cdiv(N.to(tl.int64), BLOCK_SIZE_N)

    for g in range(G):
        m_size = tl.load(split_sizes_ptr + g).to(tl.int64)
        _device_trap_if((m_size < 0) | (running_m % 128 != 0) | (running_m + m_size > M))
        m_start = running_m
        num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
        num_tiles = num_m_tiles * num_n_tiles

        a_grp = a_ptr + m_start.to(tl.int64) * K
        b_grp = b_ptr + g.to(tl.int64) * N * K
        c_grp = c_ptr + m_start.to(tl.int64) * N
        bs_grp = b_scale_ptr + g.to(tl.int64) * stride_b_scale_g

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            local = tile_idx - last_problem_end
            num_pid_in_group = GROUP_SIZE_M * num_n_tiles
            group_id = local // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((local % num_pid_in_group) % group_size_m)
            pid_n = (local % num_pid_in_group) // group_size_m

            if TILE_MODE == 0:
                _quad_tile(
                    pid_m,
                    pid_n,
                    a_grp,
                    b_grp,
                    c_grp,
                    a_scale_ptr,
                    bs_grp,
                    m_start,
                    m_size,
                    N,
                    K,
                    k_chunks,
                    smem_a_top,
                    smem_a_bot,
                    smem_b_left,
                    smem_b_right,
                    smem_sa_top,
                    smem_sa_bot,
                    smem_sb_left,
                    smem_sb_right,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    N_ALIGNED=N_ALIGNED,
                    opaque_zero=(tile_idx // 0x7FFFFFFFFFFFFFFF).to(tl.int32),
                    K_MAJOR=K_MAJOR,
                )
            else:
                _generic_tile(
                    pid_m,
                    pid_n,
                    a_grp,
                    b_grp,
                    c_grp,
                    a_scale_ptr,
                    bs_grp,
                    m_start,
                    m_size,
                    N,
                    K,
                    k_chunks,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    NUM_STAGES=NUM_STAGES,
                    K_MAJOR=K_MAJOR,
                )
            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles
        running_m = running_m + m_size

    _device_trap_if(running_m != M)


def _pick_config(gm: int, n: int, nsm: int, *, k: int, allow_quad: bool = True) -> dict[str, int]:
    """Use quadrants when tiles fill the machine and each half fits its buffer descriptor."""
    quad_tiles = _cdiv(gm, 256) * _cdiv(n, 256)
    if allow_quad and quad_tiles >= nsm and 128 * k <= _MAX_BUFFER_BYTES:
        return {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "GROUP_SIZE_M": 4,
            "XCD_CHUNK": 32 if quad_tiles >= 2 * nsm else 8,
            "TILE_MODE": 0,
            "NUM_STAGES": 1,
        }
    bn = 256 if _cdiv(gm, 128) * _cdiv(n, 256) >= nsm else 128
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": bn,
        "GROUP_SIZE_M": 8,
        "XCD_CHUNK": 8,
        "TILE_MODE": 1,
        "NUM_STAGES": 2,
    }


def grouped_gemm_mxfp8(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    split_sizes: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    num_sms: int | None = None,
    sf_layout: str = "natural",
    config: dict[str, int] | None = None,
) -> torch.Tensor:
    """Launch the forward-only gfx950 MXFP8 grouped GEMM backend."""
    if x.ndim != 2 or w.ndim not in (2, 3) or split_sizes.ndim != 1:
        raise ValueError("expected rank-2 x, rank-2/rank-3 w, and rank-1 split_sizes")
    if x.dtype != torch.float8_e4m3fn or w.dtype != torch.float8_e4m3fn:
        raise ValueError("gfx950 MXFP8 grouped GEMM supports E4M3 operands only")
    if x_scale.dtype != torch.float8_e8m0fnu or w_scale.dtype != torch.float8_e8m0fnu:
        raise ValueError("gfx950 MXFP8 grouped GEMM requires E8M0 scales")
    if split_sizes.dtype != torch.int32:
        raise ValueError("split_sizes must have dtype torch.int32")

    gm, k = x.shape
    g = split_sizes.shape[0]
    if w.ndim == 3:
        weight_groups, n, weight_k = w.shape
        if weight_groups != g:
            raise ValueError("w group count must match split_sizes")
    else:
        grouped_n, weight_k = w.shape
        if g == 0 or grouped_n % g != 0:
            raise ValueError("rank-2 w rows must be divisible by the group count")
        n = grouped_n // g
    if min(gm, g, n, k) <= 0 or weight_k != k:
        raise ValueError("grouped GEMM dimensions must be positive and K must match")
    if gm % 128 or k % _BLOCK_K:
        raise ValueError("GM and K must be divisible by 128")

    tensors = (x, x_scale, w, w_scale, split_sizes)
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("x, scales, w, and split_sizes must be contiguous")
    if x.device.type != "cuda" or any(tensor.device != x.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on x's device")

    device = x.device
    with torch.cuda.device(device):
        props = torch.cuda.get_device_properties(device)
        if not getattr(props, "gcnArchName", "").startswith("gfx950"):
            raise ValueError("gfx950 MXFP8 grouped GEMM requires a gfx950 device")
        if out is None:
            out = torch.empty((gm, n), device=device, dtype=torch.bfloat16)
        elif out.shape != (gm, n) or out.dtype != torch.bfloat16 or out.device != device or not out.is_contiguous():
            raise ValueError("out must be contiguous BF16 [GM, N] on x's device")

        # Natural scales are repacked anyway, so pack them in the K_MAJOR order the
        # gfx950 kernel reads with one ds_read_b32; blocked inputs stay zero-copy.
        k_major = sf_layout == "natural" and n % 128 == 0
        x_scale_5d, w_scale_5d = prepare_scales(x_scale, w_scale, gm=gm, g=g, n=n, k=k, sf_layout=sf_layout,
                                                k_major_chunks=k_major)
        launch_sms = props.multi_processor_count if num_sms is None else num_sms
        if launch_sms <= 0:
            raise ValueError("num_sms must be positive")
        cfg = _pick_config(gm, n, launch_sms, k=k)
        if config:
            cfg.update(config)
        # Direct-to-LDS needs a bounded byte span and provable pointer alignment.
        # Generic loads use full 64-bit addresses and support shifted storage.
        quad_safe = 128 * k <= _MAX_BUFFER_BYTES and all(tensor.data_ptr() % 16 == 0
                                                         for tensor in (x, w, x_scale_5d, w_scale_5d))
        if cfg["TILE_MODE"] == 0 and not quad_safe:
            cfg = _pick_config(gm, n, launch_sms, k=k, allow_quad=False)

        _mxfp8_grouped_gemm_kernel[(launch_sms, )](
            x,
            w,
            out,
            x_scale_5d,
            w_scale_5d,
            w_scale_5d.stride(0),
            split_sizes,
            g,
            gm,
            n,
            k,
            NUM_SM=launch_sms,
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=_BLOCK_K,
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
            NUM_XCDS=NUM_XCDS,
            XCD_CHUNK=cfg["XCD_CHUNK"],
            TILE_MODE=cfg["TILE_MODE"],
            NUM_STAGES=cfg["NUM_STAGES"],
            N_ALIGNED=n % 128 == 0,
            K_MAJOR=k_major,
            num_warps=8,
            num_stages=1,
            matrix_instr_nonkdim=16,
            llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
        )
    return out


__all__ = ["grouped_gemm_mxfp8"]
