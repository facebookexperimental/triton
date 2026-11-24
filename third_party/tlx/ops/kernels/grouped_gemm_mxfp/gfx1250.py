"""gfx1250 grouped MX GEMM (A8W8, A8W4 and W4A4) backend for ``tlx.ops``."""

from __future__ import annotations

from typing import NamedTuple

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tlx.ops.kernels.mm_mxfp.gfx1250 import (
    PIPELINED_COMPILE_OPTIONS,
    XCD_REMAP_MODES,
    mxgemm_load_a_operand,
    mxgemm_load_b_operand,
    mxgemm_position_load_descriptor,
    mxgemm_tdm_pipelined_kernel,
    mxgemm_tdm_persistent_kernel,
    operand_shared_layout,
    scale_shared_layout,
)

_BLOCK_N = 512
_BLOCK_K = 128


@triton.jit
def _build_grouped_tile_map(
    split_sizes,
    tile_meta0,
    tile_meta1,
    tile_count,
    NUM_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_TILES_PER_GROUP: tl.constexpr,
    GROUP_ALIGN: tl.constexpr,
    TOTAL_ALIGN: tl.constexpr,
):
    # Each group's tiles are padded to a multiple of GROUP_ALIGN with tiles of
    # no rows, and the map to a multiple of TOTAL_ALIGN with empty group-0 tiles.
    group = tl.program_id(0)
    offsets = tl.arange(0, GROUP_BLOCK)
    sizes = tl.load(split_sizes + offsets, mask=offsets < NUM_GROUPS, other=0)
    tiles = tl.cdiv(tl.cdiv(sizes, BLOCK_M), GROUP_ALIGN) * GROUP_ALIGN
    before = offsets < group
    row_base = tl.sum(tl.where(before, sizes, 0), axis=0).to(tl.int64)
    tile_base = tl.sum(tl.where(before, tiles, 0), axis=0).to(tl.int64)
    group_rows = tl.load(split_sizes + group).to(tl.int64)
    group_tiles = tl.cdiv(tl.cdiv(group_rows, BLOCK_M), GROUP_ALIGN) * GROUP_ALIGN

    local_tile = tl.arange(0, MAX_TILES_PER_GROUP)
    mask = local_tile < group_tiles
    meta0 = row_base | (group_rows << 32)
    meta1 = group.to(tl.int64) | (local_tile.to(tl.int64) << 32)
    tl.store(tile_meta0 + tile_base + local_tile, meta0, mask=mask)
    tl.store(tile_meta1 + tile_base + local_tile, meta1, mask=mask)
    if group == NUM_GROUPS - 1:
        end = tile_base + group_tiles
        count = tl.cdiv(end, TOTAL_ALIGN) * TOTAL_ALIGN
        pad = end + tl.arange(0, TOTAL_ALIGN)
        tl.store(tile_meta0 + pad, tl.zeros_like(pad), mask=pad < count)
        tl.store(tile_meta1 + pad, tl.zeros_like(pad), mask=pad < count)
        tl.store(tile_count, count)
        tl.store(tile_count + 1, row_base + group_rows)


@triton.jit
def _w4p_issue(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, tile, k, slot: tl.constexpr, pred,
               TILES_N: tl.constexpr, K_STEPS: tl.constexpr):
    expert = tile // TILES_N
    row = (tile * K_STEPS + k) * 4
    tlx.async_amd_descriptor_load_fused([
        (mxgemm_position_load_descriptor(a_desc, [expert * 64, k * 64], pred, True), tlx.local_view(a_buf, slot), 3),
        (mxgemm_position_load_descriptor(b_desc, [row, 0], pred, True), tlx.local_view(b_buf, slot), 12),
    ])
    tlx.async_amd_descriptor_load_fused([
        (mxgemm_position_load_descriptor(as_desc, [expert * 64, k * 4], pred, True), tlx.local_view(as_buf, slot), 3),
        (mxgemm_position_load_descriptor(bs_desc, [row, 0], pred, True), tlx.local_view(bs_buf, slot), 12),
    ])


@triton.jit
def _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, slot: tl.constexpr,
              ld_tile, ld_k, ld_pred, TILES_N: tl.constexpr, K_STEPS: tl.constexpr, PENDING: tl.constexpr = 4):
    a00, scale_a00 = mxgemm_load_a_operand(a_buf, as_buf, slot, 0, 0, 64, 128, 2, 32, 4, 0, 4, 3, 2, 1, False, True,
                                           W4A4_NATIVE=True)
    b00, scale_b00 = mxgemm_load_b_operand(b_buf, bs_buf, slot, 0, 0, 512, 128, 2, 32, 4, 4, 4, 3, 2, 1, True, True,
                                           B_PRESHUFFLE=True, W4A4_NATIVE=True, B_WMMA_ORDER=True)
    a10, scale_a10 = mxgemm_load_a_operand(a_buf, as_buf, slot, 1, 0, 64, 128, 2, 32, 4, 0, 4, 3, 2, 1, False, True,
                                           W4A4_NATIVE=True)
    b01, scale_b01 = mxgemm_load_b_operand(b_buf, bs_buf, slot, 0, 1, 512, 128, 2, 32, 4, 4, 4, 3, 2, 1, True, True,
                                           B_PRESHUFFLE=True, W4A4_NATIVE=True, B_WMMA_ORDER=True)
    c00 = tlx.dot_scaled(a00, scale_a00, "e2m1", b00, scale_b00, "e2m1", c00, tiles_per_warp=(1, 2))
    c01 = tlx.dot_scaled(a00, scale_a00, "e2m1", b01, scale_b01, "e2m1", c01, tiles_per_warp=(1, 2))
    _w4p_issue(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, ld_tile, ld_k, slot, ld_pred, TILES_N,
               K_STEPS)
    c10 = tlx.dot_scaled(a10, scale_a10, "e2m1", b00, scale_b00, "e2m1", c10, tiles_per_warp=(1, 2))
    c11 = tlx.dot_scaled(a10, scale_a10, "e2m1", b01, scale_b01, "e2m1", c11, tiles_per_warp=(1, 2))
    tlx.async_amd_descriptor_wait(PENDING)
    return c00, c01, c10, c11


@triton.jit
def mxgemm_w4_persistent_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, stride_am, stride_cm, stride_scale,
                                NUM_PROGRAMS: tl.constexpr, NUM_TILES: tl.constexpr, TILES_N: tl.constexpr,
                                M_TOTAL: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    K_STEPS: tl.constexpr = K // 128
    MAX_TILES: tl.constexpr = 3
    tl.static_assert(K_STEPS == 32 and NUM_PROGRAMS <= NUM_TILES and NUM_TILES <= MAX_TILES * NUM_PROGRAMS)
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M_TOTAL, K // 2], strides=[stride_am, tl.constexpr(1)],
                                       block_shape=[64, 64])
    as_desc = tl.make_tensor_descriptor(a_scale, shape=[M_TOTAL, K // 32], strides=[stride_scale,
                                                                                    tl.constexpr(1)],
                                        block_shape=[64, 16])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[NUM_TILES * K_STEPS * 4, 8192], strides=[8192,
                                                                                              tl.constexpr(1)],
                                       block_shape=[4, 8192])
    bs_desc = tl.make_tensor_descriptor(b_scale, shape=[NUM_TILES * K_STEPS * 4, 512], strides=[512,
                                                                                                tl.constexpr(1)],
                                        block_shape=[4, 512])
    a_buf = tlx.local_alloc((64, 64), tlx.dtype_of(a_ptr), 3, layout=operand_shared_layout([64, 64]))
    b_buf = tlx.local_alloc((4, 8192), tlx.dtype_of(b_ptr), 3, layout=scale_shared_layout((4, 8192)))
    as_buf = tlx.local_alloc((64, 16), tlx.dtype_of(a_scale), 3, layout=scale_shared_layout((64, 16)))
    bs_buf = tlx.local_alloc((4, 512), tlx.dtype_of(b_scale), 3, layout=scale_shared_layout((4, 512)))
    c_buf = tlx.local_alloc((32, 512), tlx.dtype_of(c_ptr), 1)
    c_last = tlx.local_alloc((32, 512), tlx.dtype_of(c_ptr), 1, reuse=b_buf)
    pid = tl.program_id(axis=0)
    for p in tl.static_range(3):
        _w4p_issue(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, pid, p, p, None, TILES_N, K_STEPS)
    tlx.async_amd_descriptor_wait(4)
    c_dtype: tl.constexpr = tlx.dtype_of(c_ptr)
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M_TOTAL, N], strides=[stride_cm, tl.constexpr(1)],
                                       block_shape=[32, 512])
    d10 = tl.zeros((32, 256), dtype=c_dtype)
    d11 = tl.zeros((32, 256), dtype=c_dtype)
    d_row = pid * 0
    d_col = pid * 0
    for i in tl.static_range(MAX_TILES):
        tile = pid + i * NUM_PROGRAMS
        if tile < NUM_TILES:
            PHASE: tl.constexpr = (i * K_STEPS) % 3
            nxt = tile + NUM_PROGRAMS
            if i + 1 < MAX_TILES:
                has_next = nxt < NUM_TILES
            else:
                has_next = tl.full((), False, tl.int1)
            c00 = tl.zeros((32, 256), dtype=tl.float32)
            c01 = tl.zeros((32, 256), dtype=tl.float32)
            c10 = tl.zeros((32, 256), dtype=tl.float32)
            c11 = tl.zeros((32, 256), dtype=tl.float32)
            c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                           bs_buf, PHASE, tile, 3, None, TILES_N, K_STEPS, 4 if i == 0 else 5)
            c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                           bs_buf, (PHASE + 1) % 3, tile, 4, None, TILES_N, K_STEPS)
            if i > 0:
                # The previous tile's first-half store precedes this tile's
                # step-0 load in TDM order, so step 1's wait retired it.
                tl.debug_barrier()
                c_view = tlx.local_view(c_buf, 0)
                tlx.local_store(tlx.local_slice(c_view, [0, 0], [32, 256]), d10)
                tlx.local_store(tlx.local_slice(c_view, [0, 256], [32, 256]), d11)
                tlx.async_amd_descriptor_store(c_desc, c_view, [d_row, d_col])
            c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                           bs_buf, (PHASE + 2) % 3, tile, 5, None, TILES_N, K_STEPS, 4 if i == 0 else 5)
            for base in tl.range(3, 27, 3):
                for r in tl.static_range(3):
                    c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf,
                                                   as_buf, bs_buf, (PHASE + r) % 3, tile, base + r + 3, None, TILES_N,
                                                   K_STEPS)
            c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                           bs_buf, PHASE, tile, 30, None, TILES_N, K_STEPS)
            c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                           bs_buf, (PHASE + 28) % 3, tile, 31, None, TILES_N, K_STEPS)
            for t in tl.static_range(3):
                c00, c01, c10, c11 = _w4p_step(c00, c01, c10, c11, a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf,
                                               as_buf, bs_buf, (PHASE + 29 + t) % 3, nxt, t, has_next, TILES_N, K_STEPS)
            row0 = (tile // TILES_N) * 64
            col0 = (tile % TILES_N) * 512
            c_view = tlx.local_view(c_buf, 0)
            tlx.local_store(tlx.local_slice(c_view, [0, 0], [32, 256]), c00.to(c_dtype))
            tlx.local_store(tlx.local_slice(c_view, [0, 256], [32, 256]), c01.to(c_dtype))
            tlx.async_amd_descriptor_store(c_desc, c_view, [row0, col0])
            if has_next:
                d10 = c10.to(c_dtype)
                d11 = c11.to(c_dtype)
                d_row = row0 + 32
                d_col = col0
            else:
                c_view2 = tlx.local_view(c_last, 0)
                tlx.local_store(tlx.local_slice(c_view2, [0, 0], [32, 256]), c10.to(c_dtype))
                tlx.local_store(tlx.local_slice(c_view2, [0, 256], [32, 256]), c11.to(c_dtype))
                tlx.async_amd_descriptor_store(c_desc, c_view2, [row0 + 32, col0])
    tlx.async_amd_descriptor_wait(0)


@triton.constexpr_function
def _stream_layout(block_shape, pad):
    # Pad every 256-byte row piece so the WMMA operand reads avoid LDS bank
    # conflicts (16 bytes for operands, 8 for scales).
    return tlx.padded_shared_layout_encoding.with_identity_for([[256, pad]], block_shape, [1, 0])


@triton.jit
def _stream_position(desc, offsets, pred, CLAMP: tl.constexpr = False):
    # CLAMP counts the descriptor's bounds from the new base, so a tile that
    # extends past the tensor's last row reads zeros there.
    return tlx.update_tensor_descriptor(desc, add_offsets=offsets, clamp_bounds=CLAMP, pred=pred)


@triton.jit
def _stream_tile(tile_meta0, tile_meta1, tile, m_tiles, TILES_N: tl.constexpr, BLOCK_M: tl.constexpr,
                 UNIFORM_ROWS: tl.constexpr):
    # (group, first row, valid rows, N tile) of an output tile. Tiles are
    # ordered M-tile-major, so the programs that run one M tile's N tiles at
    # the same time share its A rows through L2.
    m_tile = tile // TILES_N
    n_tile = tile % TILES_N
    if UNIFORM_ROWS > 0:
        group = m_tile // (UNIFORM_ROWS // BLOCK_M)
        row = m_tile * BLOCK_M
        rows = BLOCK_M + 0 * m_tile
    else:
        # Steps past a program's last tile are predicated off but still
        # position their descriptors: read entry 0 and use tile (0, 0) past
        # the map's m_tiles entries, which hold no tile.
        valid = m_tile < m_tiles
        entry = tl.where(valid, m_tile, 0)
        meta0 = tl.where(valid, tl.load(tile_meta0 + entry), 0)
        meta1 = tl.where(valid, tl.load(tile_meta1 + entry), 0)
        local = (meta1 >> 32).to(tl.int32)
        group = (meta1 & 0xFFFFFFFF).to(tl.int32)
        row = (meta0 & 0xFFFFFFFF).to(tl.int32) + local * BLOCK_M
        rows = tl.minimum((meta0 >> 32).to(tl.int32) - local * BLOCK_M, BLOCK_M)
    return group, row, rows, n_tile


@triton.jit
def _stream_issue(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, step, total, pid, tile_meta0,
                  tile_meta1, m_tiles, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, DB: tl.constexpr, K_ITERS: tl.constexpr, TILES_N: tl.constexpr,
                  UNIFORM_ROWS: tl.constexpr, NUM_BUFFERS: tl.constexpr, NUM_PROGRAMS: tl.constexpr):
    # Load K step `step` of this program's flattened (tile, K step) sequence.
    # Past the end the loads are predicated off; they still count towards
    # TENSORcnt, so the ring's waits stay uniform.
    pred = step < total
    tile = pid + (step // K_ITERS) * NUM_PROGRAMS
    k = step % K_ITERS
    group, row, _, n_tile = _stream_tile(tile_meta0, tile_meta1, tile, m_tiles, TILES_N, BLOCK_M, UNIFORM_ROWS)
    row_b = group * N + n_tile * BLOCK_N
    slot = step % NUM_BUFFERS
    BKS: tl.constexpr = BLOCK_K // 32
    # B and its scales are read once and bypass the caches; A, which every N
    # tile of its M tile reads, keeps the default policy.
    tlx.async_amd_descriptor_load_fused([
        (_stream_position(b_desc, [row_b, k * (BLOCK_K // DB)], pred), tlx.local_view(b_buf, slot), 3),
        (_stream_position(bs_desc, [row_b // 128, k * BKS * 128], pred), tlx.local_view(bs_buf, slot), 12),
    ], cache_modifier=".cv")
    # The last group's last A tile can extend past A's last row.
    tlx.async_amd_descriptor_load_fused([
        (_stream_position(a_desc, [row, k * BLOCK_K], pred, True), tlx.local_view(a_buf, slot), 3),
        (_stream_position(as_desc, [row, k * BKS], pred, True), tlx.local_view(as_buf, slot), 12),
    ])


@triton.jit(do_not_specialize=["gm"])
def mxgemm_grouped_stream_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale,
    b_scale,
    tile_meta0,
    tile_meta1,
    tile_count,
    gm,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    DTYPE_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    UNIFORM_ROWS: tl.constexpr,
):
    """Grouped A8W8 / A8W4 GEMM that streams the weights: out[rows of g] = x[rows of g] @ w[g].T.

    A [gm, K] e4m3, B [GROUPS * N, K] e4m3 or [GROUPS * N, K // 2] packed e2m1,
    A scales [gm, K // 32], B scales block128-ordered
    (preshuffle_grouped_weight_scale) [GROUPS * N // 128, K // 32 * 128];
    C [gm, N]. Each weight byte feeds one output tile, so the kernel is bound by
    reading B: persistent programs run their flattened (tile, K step) sequence
    through a NUM_BUFFERS-deep TDM ring and the next tile's first stages load
    while the current one finishes. With UNIFORM_ROWS every group has that many
    rows; otherwise the tile map from prepare_grouped_gemm_mxfp gives each
    BLOCK_M tile's group and rows, and tile_count[0] the number of M tiles.
    """
    tl.static_assert(DTYPE_B == "e4m3" or DTYPE_B == "e2m1")
    tl.static_assert(BLOCK_M % 16 == 0 and BLOCK_M <= 128)
    tl.static_assert(BLOCK_N % 128 == 0 and N % BLOCK_N == 0 and BLOCK_K % 128 == 0 and K % BLOCK_K == 0)
    tl.static_assert(NUM_BUFFERS >= 2 and K // BLOCK_K >= NUM_BUFFERS - 1)
    tl.static_assert(UNIFORM_ROWS == 0 or UNIFORM_ROWS % BLOCK_M == 0)
    DB: tl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    BKS: tl.constexpr = BLOCK_K // 32
    # TDM rows are at least 16 bytes: read 16 scale columns per stage and use
    # the stage's first BKS.
    BKS_LOAD: tl.constexpr = 16 if BKS < 16 else BKS
    K_ITERS: tl.constexpr = K // BLOCK_K
    TILES_N: tl.constexpr = N // BLOCK_N
    # Two fused TDM instructions per wave and K step.
    LOADS: tl.constexpr = 2
    # With BLOCK_N >= 4 * BLOCK_M the four waves sit side by side along N and
    # each covers the tile's rows and a quarter of its columns; otherwise the
    # compiler picks the wave layout and tiles.
    TILES_PER_WARP: tl.constexpr = (BLOCK_M // 16, BLOCK_N // 64) if BLOCK_N >= 4 * BLOCK_M else None

    pid = tl.program_id(0)
    if UNIFORM_ROWS > 0:
        m_tiles = gm // BLOCK_M
    else:
        m_tiles = tl.load(tile_count).to(tl.int32)
    num_tiles = m_tiles * TILES_N
    my_tiles = tl.where(pid < num_tiles, (num_tiles - 1 - pid) // NUM_PROGRAMS + 1, 0)
    total = my_tiles * K_ITERS
    if my_tiles == 0:
        return

    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[gm, K], strides=[K, tl.constexpr(1)],
                                       block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[GROUPS * N, K // DB], strides=[K // DB, tl.constexpr(1)],
                                       block_shape=[BLOCK_N, BLOCK_K // DB])
    as_desc = tl.make_tensor_descriptor(a_scale, shape=[gm, K // 32], strides=[K // 32, tl.constexpr(1)],
                                        block_shape=[BLOCK_M, BKS_LOAD])
    BS_SHAPE: tl.constexpr = [BLOCK_N // 128, BKS * 128]
    bs_desc = tl.make_tensor_descriptor(b_scale, shape=[GROUPS * N // 128, K // 32 * 128],
                                        strides=[K // 32 * 128, tl.constexpr(1)], block_shape=BS_SHAPE)
    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS,
                            layout=_stream_layout([BLOCK_M, BLOCK_K], 16))
    b_buf = tlx.local_alloc((BLOCK_N, BLOCK_K // DB), tlx.dtype_of(b_ptr), NUM_BUFFERS,
                            layout=_stream_layout([BLOCK_N, BLOCK_K // DB], 16))
    as_buf = tlx.local_alloc((BLOCK_M, BKS_LOAD), tlx.dtype_of(a_scale), NUM_BUFFERS,
                             layout=_stream_layout([BLOCK_M, BKS_LOAD], 8))
    bs_buf = tlx.local_alloc(BS_SHAPE, tlx.dtype_of(b_scale), NUM_BUFFERS, layout=_stream_layout(BS_SHAPE, 8))
    # Restage C through LDS so each wave stores whole rows.
    c_view = tlx.local_view(tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(c_ptr), 1), 0)

    for j in tl.static_range(NUM_BUFFERS - 1):
        _stream_issue(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, j, total, pid, tile_meta0,
                      tile_meta1, m_tiles, N, BLOCK_M, BLOCK_N, BLOCK_K, DB, K_ITERS, TILES_N, UNIFORM_ROWS,
                      NUM_BUFFERS, NUM_PROGRAMS)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    for t in range(my_tiles):
        tile = pid + t * NUM_PROGRAMS
        _, row, rows, n_tile = _stream_tile(tile_meta0, tile_meta1, tile, m_tiles, TILES_N, BLOCK_M, UNIFORM_ROWS)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(K_ITERS):
            step = t * K_ITERS + k
            _stream_issue(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, step + NUM_BUFFERS - 1, total,
                          pid, tile_meta0, tile_meta1, m_tiles, N, BLOCK_M, BLOCK_N, BLOCK_K, DB, K_ITERS, TILES_N,
                          UNIFORM_ROWS, NUM_BUFFERS, NUM_PROGRAMS)
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
            slot = step % NUM_BUFFERS
            a = tlx.local_load(tlx.local_view(a_buf, slot))
            b = tlx.local_load(tlx.local_trans(tlx.local_view(b_buf, slot)))
            sa = tlx.local_load(tlx.local_slice(tlx.local_view(as_buf, slot), [0, 0], [BLOCK_M, BKS]))
            sb_view = tlx.local_reshape(tlx.local_view(bs_buf, slot), [BLOCK_N // 128, BKS // 4, 32, 4, 4])
            sb = tlx.local_load(tlx.local_reshape(tlx.local_trans(sb_view, (0, 3, 2, 1, 4)), [BLOCK_N, BKS]))
            acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, DTYPE_B, acc, tiles_per_warp=TILES_PER_WARP)
        tlx.local_store(c_view, acc.to(tlx.dtype_of(c_ptr)))
        c = tlx.local_load(c_view)
        c_rows = c_ptr + (row + offs_m).to(tl.int64)[:, None] * N
        tl.store(c_rows + (n_tile * BLOCK_N + offs_n)[None, :], c, mask=(offs_m < rows)[:, None])
    tlx.async_amd_descriptor_wait(0)


def _pick_block_m(gm):
    return 16 if gm <= 4 else (32 if gm <= 256 else (64 if gm <= 4096 else 128))


def _streams(x_dtype, w_dtype):
    # E4M3 activations (A8W8 and A8W4) stream the weights through mxgemm_grouped_stream_kernel.
    return x_dtype == "e4m3" and (w_dtype == "e4m3" or _STREAM_A8W4)


def _stream_block_m(gm, groups):
    # One weight tile feeds BLOCK_M rows of one group: the smallest tile that
    # holds an average group keeps the A reads and WMMAs of padding rows low.
    rows = triton.cdiv(gm, max(groups, 1))
    return next((block_m for block_m in (16, 32, 64) if rows <= block_m), 128)


def _block_m(gm, groups, x_dtype, w_dtype):
    if _persistent_rows(gm, groups, x_dtype, w_dtype):
        return _PERSISTENT_BLOCK_M
    return _stream_block_m(gm, groups) if _streams(x_dtype, w_dtype) else _pick_block_m(gm)


class GroupedTileMap(NamedTuple):
    """Device tile map built by ``prepare_grouped_gemm_mxfp`` from one ``split_sizes`` tensor's values."""

    tile_meta0: torch.Tensor
    tile_meta1: torch.Tensor
    tile_count: torch.Tensor
    gm: int
    block_m: int
    # Tiles per group and in total are multiples of these (cluster rows and M groups).
    group_align: int = 1
    total_align: int = 1
    # (device, data_ptr, numel) of the split_sizes the map was built from; None skips the check.
    split_sizes_key: tuple | None = None


def prepare_grouped_gemm_mxfp(split_sizes, gm, x_dtype="e2m1", w_dtype="e2m1"):
    """Build a tile map for ``gm`` total rows from device ``split_sizes``, with no host readback.

    The map holds each tile's group and rows for the values ``split_sizes`` has when the map is built. It stays
    valid while those values do not change; after an in-place update, build a new map. The map uses the BLOCK_M
    that ``grouped_gemm_mxfp`` picks for ``gm`` rows in ``split_sizes.numel()`` groups of ``x_dtype`` x
    ``w_dtype``. It records ``gm``, BLOCK_M and which ``split_sizes`` tensor it came from, so the op rejects a map
    built for another row count or from another tensor; ``validate=True`` also compares its entries with the
    current ``split_sizes`` values.
    """
    groups = split_sizes.numel()
    block_m = _block_m(gm, groups, x_dtype, w_dtype)
    if block_m == _PERSISTENT_BLOCK_M:
        return _persistent_tile_map(split_sizes, gm)
    return _tile_map(split_sizes, gm, block_m)


def _tile_map(split_sizes, gm, block_m, group_align=1, total_align=1):
    groups = split_sizes.numel()
    max_m_tiles = _tile_map_entries(gm, groups, block_m, group_align, total_align)
    tile_meta0 = torch.empty(max_m_tiles, dtype=torch.int64, device=split_sizes.device)
    tile_meta1 = torch.empty_like(tile_meta0)
    tile_count = torch.zeros(2, dtype=torch.int64, device=split_sizes.device)
    if gm > 0:
        _build_grouped_tile_map[(groups, )](
            split_sizes,
            tile_meta0,
            tile_meta1,
            tile_count,
            NUM_GROUPS=groups,
            GROUP_BLOCK=triton.next_power_of_2(groups),
            BLOCK_M=block_m,
            MAX_TILES_PER_GROUP=triton.next_power_of_2(triton.cdiv(gm, block_m) + group_align - 1),
            GROUP_ALIGN=group_align,
            TOTAL_ALIGN=total_align,
        )
    return GroupedTileMap(tile_meta0, tile_meta1, tile_count, gm, block_m, group_align, total_align,
                          _split_sizes_key(split_sizes))


def _split_sizes_key(split_sizes):
    return (split_sizes.device, split_sizes.data_ptr(), split_sizes.numel())


def _expected_tile_map(sizes, block_m, group_align, total_align):
    # Host copy of _build_grouped_tile_map: the map entries, tile count and total rows for these group sizes.
    meta0, meta1, row = [], [], 0
    for group, rows in enumerate(sizes):
        tiles = triton.cdiv(triton.cdiv(rows, block_m), group_align) * group_align
        meta0 += [row | (rows << 32)] * tiles
        meta1 += [group | (local << 32) for local in range(tiles)]
        row += rows
    count = triton.cdiv(len(meta0), total_align) * total_align
    pad = count - len(meta0)
    return meta0 + [0] * pad, meta1 + [0] * pad, count, row


def _check_tile_map(tile_map, split_sizes, gm, validate):
    """Reject a tile map built from another split_sizes tensor; with validate, also one built from other values."""
    if tile_map.split_sizes_key is not None and tile_map.split_sizes_key != _split_sizes_key(split_sizes):
        raise ValueError("tile_map was built from a different split_sizes tensor; build it with "
                         "prepare_grouped_gemm_mxfp from the split_sizes passed here")
    if not validate:
        return
    if split_sizes.is_cuda and torch.cuda.is_current_stream_capturing():
        raise ValueError("validate=True reads split_sizes on the host and cannot run under graph capture")
    sizes = split_sizes.tolist()
    if tile_map.gm != gm or sum(sizes) != gm or min(sizes, default=0) < 0:
        # The row-count and split_sizes checks report these.
        return
    meta0, meta1, count, rows = _expected_tile_map(sizes, tile_map.block_m, tile_map.group_align, tile_map.total_align)
    if (tile_map.tile_count.tolist() != [count, rows] or tile_map.tile_meta0[:count].tolist() != meta0
            or tile_map.tile_meta1[:count].tolist() != meta1):
        raise ValueError("tile_map does not match the current split_sizes values; rebuild it with "
                         "prepare_grouped_gemm_mxfp after split_sizes changes")


def _tile_map_entries(gm, groups, block_m, group_align, total_align):
    # Each group adds at most one partial tile and group_align - 1 padding tiles.
    return triton.cdiv(gm, block_m) + groups * group_align + total_align - 1


@triton.jit
def _block128_words(src, dst, first_row, rows, QW: tl.constexpr, COLS: tl.constexpr):
    # Copy rows [first_row, first_row + rows) of [*, QW] int32 words of E8M0
    # scales (zeros past `rows`, at most 128) into the block128 block at dst:
    # word q of row a * 32 + b, four consecutive scales, goes to word
    # (q * 32 + b) * 4 + a, as in pack_scale. Reading whole rows and writing
    # whole 128-word runs makes the transpose go through LDS.
    r = tl.arange(0, 128)
    q = tl.program_id(1) * COLS + tl.arange(0, COLS)
    values = tl.load(src + (first_row + r).to(tl.int64)[:, None] * QW + q[None, :],
                     mask=(r < rows)[:, None] & (q < QW)[None, :], other=0)
    values = tl.reshape(tl.permute(tl.reshape(values, (4, 32, COLS)), (2, 1, 0)), (COLS, 128))
    tl.store(dst + q[:, None] * 128 + r[None, :], values, mask=(q < QW)[:, None])


@triton.jit
def _preshuffle_scale_kernel(src, dst, QW: tl.constexpr, COLS: tl.constexpr):
    block = tl.program_id(0)
    _block128_words(src, dst + block.to(tl.int64) * (128 * QW), block * 128, 128, QW, COLS)


def _scale_cols(qw):
    return min(32, triton.next_power_of_2(qw))


def preshuffle_grouped_weight_scale(w_scale, kstep_major=False):
    """Preshuffle E8M0 weight scales; kstep_major must match the payload's."""
    groups, n, k_scale = w_scale.shape
    if n % 128 or k_scale % 4:
        raise ValueError("Weight-scale N must be divisible by 128 and K/32 by 4")
    if w_scale.is_cuda and not kstep_major:
        packed = torch.empty((groups, n // 128, k_scale * 128), dtype=w_scale.dtype, device=w_scale.device)
        if packed.numel():
            qw = k_scale // 4
            with torch.cuda.device(w_scale.device):
                _preshuffle_scale_kernel[(groups * n // 128,
                                          triton.cdiv(qw, _scale_cols(qw)))](w_scale.contiguous().view(torch.int32),
                                                                             packed.view(torch.int32), QW=qw,
                                                                             COLS=_scale_cols(qw))
        return packed
    packed = w_scale.view(groups, n // 128, 4, 32, k_scale // 4, 4)
    packed = packed.permute(0, 1, 4, 3, 2, 5).contiguous()
    if kstep_major:
        if n % _BLOCK_N:
            raise ValueError(f"K-step-major weight scales need N divisible by {_BLOCK_N}")
        packed = packed.view(groups, n // _BLOCK_N, _BLOCK_N // 128, k_scale // 4, 512)
        packed = packed.permute(0, 1, 3, 2, 4).contiguous()
    return packed.view(groups, n // 128, k_scale * 128)


# (dk, dn) of each byte-offset bit of a [_BLOCK_N, _BLOCK_K // 2] B tile in W4A4
# WMMA register order; must match _b_wmma_order_layout() in the kernel.
_WMMA_ORDER_BASES = ((1, 0), (2, 0), (4, 0), (8, 0), (0, 1), (0, 2), (0, 4), (0, 8), (16, 0), (32, 0), (0, 16), (0, 32),
                     (0, 64), (0, 128), (0, 256))


def _wmma_order_index(device):
    offsets = torch.arange(_BLOCK_N * (_BLOCK_K // 2), device=device)
    k = torch.zeros_like(offsets)
    n = torch.zeros_like(offsets)
    for bit, (dk, dn) in enumerate(_WMMA_ORDER_BASES):
        set_bit = (offsets >> bit) & 1
        k |= set_bit * dk
        n |= set_bit * dn
    return n * (_BLOCK_K // 2) + k


def preshuffle_grouped_weight(w, kstep_major=False):
    """Preshuffle packed FP4 weights for B_PRESHUFFLE TDM loads.

    With kstep_major, each BLOCK_N-column tile stores its K steps back to back,
    each step in WMMA register order, so every K step's TDM transfer is one
    contiguous block.
    """
    groups, n, packed_k = w.shape
    if n % 128 or packed_k % 16:
        raise ValueError("Packed weight N must be divisible by 128 and K/2 by 16")
    if kstep_major:
        if n % _BLOCK_N or packed_k % (_BLOCK_K // 2):
            raise ValueError(f"K-step-major weights need N divisible by {_BLOCK_N} and K by {_BLOCK_K}")
        steps = packed_k // (_BLOCK_K // 2)
        tiles = w.view(groups, n // _BLOCK_N, _BLOCK_N, steps, _BLOCK_K // 2).permute(0, 1, 3, 2, 4)
        tiles = tiles.reshape(groups, n // _BLOCK_N, steps, _BLOCK_N * (_BLOCK_K // 2))
        return torch.gather(tiles, 3, _wmma_order_index(w.device).expand_as(tiles)).view_as(w)
    packed = w.view(groups, n // 128, 4, 32, packed_k // 16, 16)
    packed = packed.permute(0, 1, 4, 3, 2, 5).contiguous()
    return packed.view_as(w)


# LDS the streaming kernel may use per workgroup (one workgroup per CU, 320 KiB).
_STREAM_LDS_BUDGET = 316 * 1024
_STREAM_A8W4 = True


def _stream_config(n, k, w_dtype, block_m):
    """Pick (BLOCK_N, BLOCK_K, NUM_BUFFERS) for mxgemm_grouped_stream_kernel, or None if it does not apply.

    Deeper K stages halve the per-step barrier and wait overhead, so prefer
    BLOCK_K=256 with at least three stages, then fill the LDS with stages.
    With 256-column tiles, packed FP4 weights first try BLOCK_K=512 with at
    least two stages, so each K step moves as many weight bytes as A8W8's: with BLOCK_K=256 the
    per-step costs halve the weight bandwidth (gfx1250, 128 groups of 64
    rows with N = K = 8192: 11.9 TB/s with five stages, 14.0 TB/s with two
    BLOCK_K=512 stages; 12-18% faster for 8- to 64-row groups at K = 4096).
    """
    if n % 128 or k % 512:
        return None
    block_n = 256 if n % 256 == 0 else 128
    db = 2 if w_dtype == "e2m1" else 1

    def padded(rows, cols, pad):
        return rows * (cols + pad * triton.cdiv(cols, 256))

    candidates = ((512, 2), (256, 3), (128, 3)) if db == 2 and block_n == 256 else ((256, 3), (128, 3))
    for block_k, min_buffers in candidates:
        stage = (padded(block_m, block_k, 16) + padded(block_n, block_k // db, 16) + padded(block_m, 16, 8) +
                 padded(block_n // 128, block_k // 32 * 128, 8))
        buffers = min(6, (_STREAM_LDS_BUDGET - block_m * block_n * 2) // stage, k // block_k + 1)
        if buffers >= min_buffers:
            return block_n, block_k, buffers
    return None


def _grouped_stream(x, x_scale, w, w_scale, split_sizes, out, tile_map, w_dtype, uniform_rows, block_m, config,
                    num_programs=None):
    gm, k = x.shape
    groups, n = w.shape[0], w.shape[1]
    block_n, block_k, buffers = config
    if uniform_rows:
        tile_meta0 = tile_meta1 = tile_count = split_sizes
    else:
        if tile_map is None:
            tile_map = _tile_map(split_sizes, gm, block_m)
        tile_meta0, tile_meta1, tile_count = tile_map[:3]
    # One program per CU whatever the row count, so a new token count does
    # not recompile; programs without a tile return at once.
    programs = num_programs or torch.cuda.get_device_properties(x.device).multi_processor_count
    mxgemm_grouped_stream_kernel[(programs, )](x, w, out, x_scale, w_scale, tile_meta0, tile_meta1, tile_count, gm, N=n,
                                               K=k, GROUPS=groups, DTYPE_B=w_dtype, BLOCK_M=block_m, BLOCK_N=block_n,
                                               BLOCK_K=block_k, NUM_BUFFERS=buffers, NUM_PROGRAMS=programs,
                                               UNIFORM_ROWS=uniform_rows, num_warps=4, waves_per_eu=1)
    return out


# A8W8 and A8W4 groups that average more than this many rows reuse each
# weight tile across M tiles: the persistent mm_mxfp kernel runs them as
# 256x256 tiles. For A8W8 on gfx1250 (N = K = 8192) it beats streaming from
# 192-row groups (-33%); at 128-row groups streaming 128-row tiles is 20%
# faster. A8W4 uses the same threshold.
_STREAM_MAX_AVG_ROWS = 128
_PERSISTENT_BLOCK_M = 256
_PERSISTENT_BLOCK_N = 256
# BLOCK_K per weight dtype: A8W8 runs the BK128 schedules, A8W4 the BK256 one.
_PERSISTENT_BLOCK_K = {"e4m3": 128, "e2m1": 256}
# Static slot unroll pays off up to this K; beyond it the dynamic four-stage
# ring is faster (see mm_mxfp).
_PERSISTENT_STATIC_MAX_K = 16384
# 4x4 clusters share B across four M tiles of one group and A across four N
# tiles. Padding each group to whole cluster rows costs up to three empty
# tiles per group, so clusters are used from this average group size. On
# gfx1250 (N = K = 8192) they win 7-14% for ragged groups averaging 591, 788
# and 1040 rows and lose 7% at 500 rows and 25% for uniform 512-row groups.
# 2x8 clusters pad less but share B over only two tiles: they win only for
# uniform 512-row groups, which the op cannot tell from ragged ones.
_CLUSTER_MIN_AVG_ROWS = 576
_CLUSTER_ROWS = 4
_CLUSTER_COLS = 4
_CLUSTER_GROUP_M = 8


def _persistent_rows(gm, groups, x_dtype, w_dtype):
    return x_dtype == "e4m3" and w_dtype in _PERSISTENT_BLOCK_K and gm > _STREAM_MAX_AVG_ROWS * max(groups, 1)


def _persistent_shape(n, k, w_dtype):
    # A8W8 needs two BK128 steps, A8W4 its three-stage BK256 ring.
    block_k = _PERSISTENT_BLOCK_K[w_dtype]
    return n % _PERSISTENT_BLOCK_N == 0 and k % block_k == 0 and k >= (2 if w_dtype == "e4m3" else 3) * block_k


def _persistent_tile_map(split_sizes, gm):
    # Groups that average at least _CLUSTER_MIN_AVG_ROWS rows get a map padded for
    # 4x4 clusters: whole cluster rows in each group and whole M groups.
    if gm >= _CLUSTER_MIN_AVG_ROWS * split_sizes.numel():
        return _tile_map(split_sizes, gm, _PERSISTENT_BLOCK_M, _CLUSTER_ROWS, _CLUSTER_GROUP_M)
    return _tile_map(split_sizes, gm, _PERSISTENT_BLOCK_M)


@triton.jit
def _gather_tile_a_scale(x_scale, packed, tile_meta0, tile_meta1, tile_count, QW: tl.constexpr, BLOCK_M: tl.constexpr,
                         COLS: tl.constexpr):
    # Copy the natural [GM, 4 * QW] scales (as int32 words) of each map
    # entry's BLOCK_M rows into block128 order at packed row
    # entry * BLOCK_M // 128. Rows past the entry's group read as zero; the
    # GEMM does not store them.
    block = tl.program_id(0)
    entry = block // (BLOCK_M // 128)
    if entry < tl.load(tile_count):
        meta0 = tl.load(tile_meta0 + entry)
        local = (tl.load(tile_meta1 + entry) >> 32).to(tl.int32)
        skip = local * BLOCK_M + block % (BLOCK_M // 128) * 128
        _block128_words(x_scale, packed + block.to(tl.int64) * (128 * QW), (meta0 & 0xFFFFFFFF).to(tl.int32) + skip,
                        (meta0 >> 32).to(tl.int32) - skip, QW, COLS)


def _grouped_persistent(x, x_scale, w, w_scale, out, tile_map, num_programs=None):
    """A8W8 or A8W4 grouped GEMM on the persistent mm_mxfp kernel; ``w_scale`` is preshuffled, ``tile_map``
    BLOCK_M=256."""
    gm, k = x.shape
    groups, n = w.shape[0], w.shape[1]
    tile_meta0, tile_meta1, tile_count = tile_map[:3]
    entries = tile_meta0.numel()
    ks = k // 32
    rows_per_entry = _PERSISTENT_BLOCK_M // 128
    a_scale = torch.empty((entries * rows_per_entry, ks * 128), dtype=torch.uint8, device=x.device)
    qw = ks // 4
    _gather_tile_a_scale[(entries * rows_per_entry, triton.cdiv(qw,
                                                                _scale_cols(qw)))](x_scale.view(torch.int32),
                                                                                   a_scale.view(torch.int32),
                                                                                   tile_meta0, tile_meta1, tile_count,
                                                                                   QW=qw, BLOCK_M=_PERSISTENT_BLOCK_M,
                                                                                   COLS=_scale_cols(qw))
    w_dtype = "e4m3" if w.dtype == torch.float8_e4m3fn else "e2m1"
    block_k = _PERSISTENT_BLOCK_K[w_dtype]
    k_tiles = k // block_k
    static = w_dtype == "e4m3" and 4 < k_tiles and k <= _PERSISTENT_STATIC_MAX_K
    # One program per CU whatever the row count (see _grouped_stream).
    programs = num_programs or torch.cuda.get_device_properties(x.device).multi_processor_count
    # A map padded for clusters keeps every cluster in one group and every
    # cluster's members on the same number of tiles.
    cluster = (tile_map.group_align % _CLUSTER_ROWS == 0 and tile_map.total_align % _CLUSTER_GROUP_M == 0
               and n % (_CLUSTER_COLS * _PERSISTENT_BLOCK_N) == 0 and programs % (_CLUSTER_ROWS * _CLUSTER_COLS) == 0)
    cluster_size = _CLUSTER_ROWS * _CLUSTER_COLS if cluster else 1
    if w_dtype == "e4m3":
        schedule = dict(NUM_BUFFERS=min(4, k_tiles), CROSS_TILE_PREFETCH=True)
    else:
        # The A8W4 schedule of bench_gfx1250.py: three BK256 stages, C staged in
        # retired A stages, early C stores and split cluster barriers, plus
        # WMMA queuing (gfx1250, N = K = 8192: 2048-row groups -1.9%, ragged
        # ~1040-row groups -2.6%).
        schedule = dict(NUM_BUFFERS=3, CROSS_TILE_PREFETCH=False, EARLY_C_STORES=True, SPLIT_CLUSTER_BARRIER=cluster,
                        enable_wmma_queuing=True)
    with torch.cuda.device(x.device):
        mxgemm_tdm_persistent_kernel[(programs, )](
            x, w.view(groups * n, w.shape[2]), out, a_scale, w_scale.view(groups * n // 128, ks * 128), 0, n, k,
            x.stride(0), w.shape[2], out.stride(0), a_scale.stride(0), ks * 128, DTYPE_B=w_dtype,
            BLOCK_M=_PERSISTENT_BLOCK_M, BLOCK_N=_PERSISTENT_BLOCK_N, BLOCK_K=block_k, GROUP_SIZE_M=_CLUSTER_GROUP_M,
            NUM_PROGRAMS=programs, XCD_REMAP_MODE=XCD_REMAP_MODES["chunked"], NUM_XCDS=8, XCD_CHUNK=2,
            CLUSTER_SIZE=cluster_size, CLUSTER_MULTICAST=True, CLUSTER_BARRIER_INTERVAL=8 if cluster else 1,
            CLUSTER_LAYOUT=f"{_CLUSTER_ROWS}x{_CLUSTER_COLS}" if cluster else "linear", STATIC_SLOT_UNROLL=static,
            OUTPUT_C_SLOTS=2 if static else None, OVERLAP_PEEL=static, MERGED_HANDOFF_BARRIER=static,
            STATIC_CLUSTER_STEPS=static and cluster, NUM_GROUPS=groups, tile_meta0=tile_meta0, tile_meta1=tile_meta1,
            tile_count=tile_count, a_rows=gm, a_scale_rows=a_scale.shape[0], num_warps=4, waves_per_eu=1,
            ctas_per_cga=(cluster_size, 1, 1), **schedule)
    return out


_SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)
_X_STORAGE_DTYPES = {"e4m3": torch.float8_e4m3fn, "e2m1": torch.uint8}


def _check_split_sizes(split_sizes, gm, uniform_rows):
    if split_sizes.is_cuda and torch.cuda.is_current_stream_capturing():
        raise ValueError("validate=True reads split_sizes on the host and cannot run under graph capture")
    sizes = split_sizes.tolist()
    if any(rows < 0 for rows in sizes) or sum(sizes) != gm:
        raise ValueError(f"split_sizes must be nonnegative and sum to the {gm} rows of X; got sum {sum(sizes)}")
    if uniform_rows and any(rows != uniform_rows for rows in sizes):
        raise ValueError(f"uniform_grouped needs every group to have {uniform_rows} rows")


def _check_inputs(x, x_scale, w, w_scale, split_sizes, out, w_scale_preshuffled, x_dtype, w_dtype):
    tensors = (x, x_scale, w, w_scale, split_sizes)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("grouped_gemm_mxfp expects tensor inputs")
    if (x.ndim, x_scale.ndim, w.ndim, w_scale.ndim, split_sizes.ndim) != (2, 2, 3, 3, 1):
        raise ValueError("grouped_gemm_mxfp expects x/x_scale/w/w_scale/split_sizes ranks 2/2/3/3/1; got "
                         f"{x.ndim}/{x_scale.ndim}/{w.ndim}/{w_scale.ndim}/{split_sizes.ndim}")
    if (x_dtype, w_dtype) not in (("e4m3", "e2m1"), ("e2m1", "e2m1"), ("e4m3", "e4m3")):
        raise ValueError("grouped_gemm_mxfp supports (x_dtype, w_dtype) ('e4m3', 'e2m1'), ('e2m1', 'e2m1') and "
                         f"('e4m3', 'e4m3'); got x_dtype={x_dtype!r}, w_dtype={w_dtype!r}")
    w_storage = torch.float8_e4m3fn if w_dtype == "e4m3" else torch.uint8
    if x.dtype != _X_STORAGE_DTYPES[x_dtype] or w.dtype != w_storage:
        raise ValueError(f"grouped_gemm_mxfp expects x_dtype={x_dtype!r} x stored as {_X_STORAGE_DTYPES[x_dtype]} "
                         f"and w_dtype={w_dtype!r} w stored as {w_storage}; got x.dtype={x.dtype}, w.dtype={w.dtype}")
    if x_scale.dtype not in _SCALE_DTYPES or w_scale.dtype not in _SCALE_DTYPES:
        raise ValueError("grouped_gemm_mxfp expects E8M0 (or uint8) scales; "
                         f"got x_scale.dtype={x_scale.dtype}, w_scale.dtype={w_scale.dtype}")
    if split_sizes.dtype != torch.int32:
        raise ValueError(f"grouped_gemm_mxfp expects int32 split_sizes; got {split_sizes.dtype}")
    if not x.is_cuda or any(tensor.device != x.device for tensor in tensors[1:]):
        raise ValueError(f"grouped_gemm_mxfp expects all inputs on one GPU; got {[t.device for t in tensors]}")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("grouped_gemm_mxfp expects contiguous inputs")

    gm = x.shape[0]
    groups, n, packed_k = w.shape
    k = x.shape[1] if x_dtype == "e4m3" else 2 * x.shape[1]
    if (packed_k if w_dtype == "e4m3" else 2 * packed_k) != k or k % 32:
        raise ValueError(f"grouped_gemm_mxfp reduction dimensions must match and be divisible by 32; got "
                         f"x.shape={tuple(x.shape)} ({x.dtype}), w.shape={tuple(w.shape)}")
    if split_sizes.numel() != groups:
        raise ValueError(f"grouped_gemm_mxfp expects {groups} split_sizes; got {split_sizes.numel()}")
    if tuple(x_scale.shape) != (gm, k // 32):
        raise ValueError(f"grouped_gemm_mxfp expects x_scale shape {(gm, k // 32)}; got {tuple(x_scale.shape)}")
    w_scale_shape = (groups, n // 128, k // 32 * 128) if w_scale_preshuffled else (groups, n, k // 32)
    if (w_scale_preshuffled and n % 128) or tuple(w_scale.shape) != w_scale_shape:
        raise ValueError(f"grouped_gemm_mxfp expects {'preshuffled ' if w_scale_preshuffled else ''}w_scale shape "
                         f"{w_scale_shape}; got {tuple(w_scale.shape)}")
    if out is not None and (not isinstance(out, torch.Tensor) or out.dtype != torch.bfloat16
                            or tuple(out.shape) != (gm, n) or out.device != x.device or not out.is_contiguous()):
        raise ValueError(f"grouped_gemm_mxfp out must be a contiguous BF16 {(gm, n)} tensor on {x.device}")
    return gm, groups, n, k


def grouped_gemm_mxfp(
    x,
    x_scale,
    w,
    w_scale,
    split_sizes,
    *,
    x_dtype="e2m1",
    w_dtype="e2m1",
    out=None,
    tile_map=None,
    w_preshuffled=False,
    w_scale_preshuffled=False,
    uniform_grouped=False,
    w_kstep_major=False,
    validate=False,
):
    """Grouped MX GEMM on gfx1250: ``out[rows of group g] = x[rows of group g] @ w[g].T``.

    ``x_dtype`` and ``w_dtype`` name the MX formats: ``x`` is ``[GM, K]`` float8_e4m3fn for ``"e4m3"`` or
    ``[GM, K // 2]`` packed uint8 for ``"e2m1"``; ``w`` is ``[G, N, K // 2]`` packed E2M1 for ``"e2m1"`` or
    ``[G, N, K]`` float8_e4m3fn for ``"e4m3"`` (with ``x_dtype="e4m3"`` only). Scales are per-32 E8M0: ``x_scale [GM, K // 32]`` and ``w_scale [G, N, K // 32]``, or the
    ``preshuffle_grouped_weight_scale`` layout with ``w_scale_preshuffled=True``. Groups are contiguous along M;
    ``split_sizes`` is a device int32 ``[G]`` tensor and is not read on the host unless ``validate=True``, so the
    call is graph-capturable. Returns BF16 ``[GM, N]`` (``out`` if given).

    A8W8 and A8W4 groups that average more than 128 rows run 256x256 tiles on the persistent ``mm_mxfp`` kernel
    when N is divisible by 256 and K by 128 (K >= 256; A8W4: by 256, K >= 768); the A scales of each tile are
    reordered on every call. Groups averaging 576 rows or more get a tile map padded for 4x4 clusters, which
    share A and B through multicast when N is divisible by 1024. Other E4M3 activations (A8W8 and A8W4) stream
    the weights through ``mxgemm_grouped_stream_kernel`` when N is
    divisible by 128 and K by 512, which A8W8 requires. Without ``w_scale_preshuffled`` the weight scales are
    preshuffled on every call. A ``tile_map`` from ``prepare_grouped_gemm_mxfp(split_sizes, gm, x_dtype, w_dtype)``
    is used whenever given; otherwise one is built (for W4A4 and preshuffled A8W4 weights only for
    256 < GM <= 4096). A given ``tile_map`` must come from this ``split_sizes`` tensor and reflect its current
    values: the op rejects a map from another tensor or row count, but only ``validate=True`` detects values that
    changed after the map was built. ``uniform_grouped`` and ``w_kstep_major`` are caller promises that every group
    has 64 rows; ``validate=True`` checks ``split_sizes`` and a given ``tile_map`` on the host at the cost of a device
    sync.
    """
    gm, groups, n, k = _check_inputs(x, x_scale, w, w_scale, split_sizes, out, w_scale_preshuffled, x_dtype, w_dtype)
    if out is None:
        out = torch.empty((gm, n), dtype=torch.bfloat16, device=x.device)
    if gm == 0:
        return out
    if tile_map is not None:
        _check_tile_map(tile_map, split_sizes, gm, validate)
    x_scale_u8 = x_scale.view(torch.uint8)
    w_scale_u8 = w_scale.view(torch.uint8)
    stream_config = stream_block_m = None
    if _streams(x_dtype, w_dtype) and not w_preshuffled and not w_kstep_major:
        uniform_rows = 64 if uniform_grouped and gm == groups * 64 else 0
        if uniform_rows:
            stream_block_m = uniform_rows
        elif tile_map is not None:
            # Any tile map works; one from prepare_grouped_gemm_mxfp(split_sizes, gm, x_dtype, w_dtype) has the
            # tile height that suits this problem.
            if tile_map.gm != gm:
                raise ValueError(f"tile_map was built for {tile_map.gm} rows; x has {gm} rows. Rebuild it with "
                                 f"prepare_grouped_gemm_mxfp(split_sizes, {gm}, {x_dtype!r}, {w_dtype!r})")
            stream_block_m = tile_map.block_m
        else:
            stream_block_m = _block_m(gm, groups, x_dtype, w_dtype)
            if stream_block_m == _PERSISTENT_BLOCK_M and not _persistent_shape(n, k, w_dtype):
                stream_block_m = _stream_block_m(gm, groups)
        if stream_block_m == _PERSISTENT_BLOCK_M and _persistent_shape(n, k, w_dtype):
            if validate:
                _check_split_sizes(split_sizes, gm, 0)
            if tile_map is None:
                tile_map = _persistent_tile_map(split_sizes, gm)
            if not w_scale_preshuffled:
                w_scale_u8 = preshuffle_grouped_weight_scale(w_scale_u8)
            return _grouped_persistent(x, x_scale_u8, w, w_scale_u8, out, tile_map)
        if stream_block_m <= 128:
            stream_config = _stream_config(n, k, w_dtype, stream_block_m)
    if stream_config is not None:
        if validate:
            _check_split_sizes(split_sizes, gm, uniform_rows)
        if not w_scale_preshuffled:
            # The kernel reads block128-ordered weight scales; preshuffle them
            # offline (preshuffle_grouped_weight_scale) to skip this.
            w_scale_u8 = preshuffle_grouped_weight_scale(w_scale_u8)
        return _grouped_stream(x, x_scale_u8, w, w_scale_u8, split_sizes, out, tile_map, w_dtype, uniform_rows,
                               stream_block_m, stream_config)
    if w_dtype == "e4m3":
        raise ValueError("grouped_gemm_mxfp A8W8 needs unshuffled weights and N divisible by 128 and K by 512 "
                         f"(256 and 128 with 256-row tiles); got N={n}, K={k}, w_preshuffled={w_preshuffled}, "
                         f"tile rows {stream_block_m}")
    block_m = _pick_block_m(gm)
    block_n = _BLOCK_N
    schedule = "baseline" if gm <= 4 else "sliceMNK"
    tiles_per_warp = (1, 2) if gm <= 256 or x_dtype == "e2m1" else (2, 2)
    uniform_grouped = (uniform_grouped and x_dtype == "e2m1" and block_m == 64 and gm == groups * block_m)
    if validate:
        _check_split_sizes(split_sizes, gm, block_m if uniform_grouped else 0)
    if tile_map is not None and (tile_map.gm != gm or tile_map.block_m != block_m):
        raise ValueError(f"tile_map was built for {tile_map.gm} rows (BLOCK_M={tile_map.block_m}); x has {gm} rows "
                         f"(BLOCK_M={block_m}). Rebuild it with prepare_grouped_gemm_mxfp(split_sizes, {gm})")
    max_m_tiles = groups if uniform_grouped else triton.cdiv(gm, block_m) + min(groups, gm) - 1
    use_tile_map = not uniform_grouped and (tile_map is not None or 256 < gm <= 4096)
    if use_tile_map:
        if tile_map is None:
            tile_map = prepare_grouped_gemm_mxfp(split_sizes, gm)
        tile_meta0, tile_meta1, tile_count = tile_map[:3]
    else:
        tile_meta0 = tile_meta1 = tile_count = split_sizes
    grid = (max_m_tiles * triton.cdiv(n, block_n), )
    static_ring = uniform_grouped and groups == 32 and k == 4096 and w_preshuffled and w_scale_preshuffled
    if w_kstep_major and (not static_ring or n % block_n):
        raise ValueError("w_kstep_major needs the uniform 32-group K=4096 W4A4 static ring, K-step-major "
                         f"preshuffled weights and weight scales, and N divisible by {block_n}")
    if w_kstep_major:
        programs = 2 * torch.cuda.get_device_properties(x.device).multi_processor_count
        tiles = groups * (n // block_n)
        if tiles > 3 * programs:
            raise ValueError(f"w_kstep_major runs at most 3 tiles per program; N={n} needs {tiles} tiles but "
                             f"{x.device} runs {programs} programs")
        mxgemm_w4_persistent_kernel[(min(programs,
                                         tiles), )](x, w, out, x_scale_u8, w_scale_u8, x.stride(0), out.stride(0),
                                                    x_scale_u8.stride(0), NUM_PROGRAMS=min(programs, tiles),
                                                    NUM_TILES=tiles, TILES_N=n // block_n, M_TOTAL=gm, N=n, K=k,
                                                    num_warps=4, waves_per_eu=2, **PIPELINED_COMPILE_OPTIONS)
        return out
    mxgemm_tdm_pipelined_kernel[grid](
        x,
        w,
        out,
        x_scale_u8,
        w_scale_u8,
        gm,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(2),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        x_scale_u8.stride(0),
        w_scale_u8.stride(1),
        split_sizes,
        groups,
        gm,
        w.stride(0),
        w_scale_u8.stride(0),
        tile_meta0,
        tile_meta1,
        tile_count,
        DTYPE_A=x_dtype,
        DTYPE_B=w_dtype,
        SCALE_BLOCK=32,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=_BLOCK_K,
        GROUP_SIZE_M=4,
        TRANSPOSE_B=True,
        NUM_BUFFERS=3,
        SCALE_PRESHUFFLE=False,
        B_SCALE_PRESHUFFLE=w_scale_preshuffled,
        B_PRESHUFFLE=w_preshuffled,
        WITH_A_SCALE=True,
        SCHEDULE=schedule,
        TDM_FUSION="2way",
        L2_PREFETCH_DISTANCE=-1,
        TDM_SPLIT=False,
        TILES_PER_WARP=tiles_per_warp,
        GROUPED=True,
        GROUPED_TILE_MAP=use_tile_map,
        UNIFORM_GROUPED=uniform_grouped,
        STATIC_N=n if uniform_grouped else 0,
        STATIC_K=k if block_m >= 64 else 0,
        FLY_QUADRANT_W4=uniform_grouped,
        W4A4_INTRAWAVE=uniform_grouped,
        W4A4_RELAXED_INTRAWAVE=uniform_grouped,
        W4A4_STATIC_RING=static_ring,
        num_warps=4,
        waves_per_eu=2 if block_m == 64 else 1,
        **PIPELINED_COMPILE_OPTIONS,
    )
    return out


__all__ = [
    "GroupedTileMap",
    "grouped_gemm_mxfp",
    "prepare_grouped_gemm_mxfp",
    "preshuffle_grouped_weight",
    "preshuffle_grouped_weight_scale",
]
