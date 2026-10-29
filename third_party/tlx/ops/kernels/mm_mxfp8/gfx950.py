"""TLX gfx950 GEMM over E4M3 data with E8M0 per-32 microscales."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tlx.ops._catalog import InvalidInput
from triton.tlx.ops.kernels.mm_mxfp8._shapes import MODEL_LINEAR_SHAPES

_MODEL_LINEAR_SHAPES = frozenset(MODEL_LINEAR_SHAPES)

# Each copying lane owns four adjacent E8M0 bytes for dword-sized transfers.
# The K-major variant follows the packed byte order instead of natural rows.
_COPY_SCALE = tlx.layout(shape=((128, 4), (4, )), stride=((4, 0), (1, )))
_COPY_SCALE_WORD_4W = tlx.layout(shape=((128, 2), (1, )), stride=((1, 0), (0, )))
_COPY_SCALE_K_MAJOR = tlx.layout(shape=((4, 32, 4), (4, )), stride=((4, 16, 0), (1, )))
# Scaled MFMA consumes each row's four K-block scales in operand-lane order.
# A broadcasts across the four N waves; zero strides encode those broadcasts.
_SCALE_A = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
    stride=((4, 8, 16, 32, 1, 2, 0, 0, 64), (128, 256)),
)
# B's row scales follow the N coordinates of the [2, 4] MFMA wave grid.
# Its zero wave stride broadcasts the scales across the two M waves.
_SCALE_B = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, )),
    stride=((4, 8, 16, 32, 1, 2, 64, 128, 0), (256, )),
)
_SCALE_A_MI32 = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
    stride=((4, 8, 16, 32, 64, 1, 0, 128), (2, 256)),
)
_SCALE_B_MI32 = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
    stride=((4, 8, 16, 32, 64, 1, 128, 0), (2, 256)),
)
# Each lane owns eight adjacent BF16 columns at a time for 16-byte stores.
# The remaining register bits select rows rather than strided MFMA columns.
_STORE = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
    stride=((8, 16, 32, 64, 128, 256, 512, 1024, 2048), (1, 2, 4, 4096, 8192)),
)
_STORE_4W = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2)),
    stride=((8, 16, 32, 64, 128, 256, 512, 1024), (1, 2, 4, 2048, 4096, 8192)),
)


@triton.jit
def _load_scale(buf, layout: tl.constexpr, K_MAJOR: tl.constexpr):
    if K_MAJOR:
        # Reinterpret packed bytes as logical [row, K-block] scales without copying.
        # Four-byte alignment lets the MFMA operand layout use 32-bit LDS reads.
        read_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
            offset_bases=[[32, 0], [64, 0], [0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
            block_bases=[],
            alignment=4,
        )
        buf = tlx.local_reinterpret(buf, tl.uint8, [128, 4], layout=read_layout)
    return tlx.local_load(buf, layout=layout, relaxed=True)


@triton.jit
def _load_scale_words(buf, layout: tl.constexpr):
    read_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]],
        block_bases=[],
        alignment=4,
    )
    view = tlx.local_reinterpret(buf, tl.uint8, [128, 4], layout=read_layout)
    return tlx.local_load(view, layout=layout, relaxed=True)


@triton.jit
def _store_quadrant(acc, C, rows, cols, N: tl.constexpr):
    acc_layout: tl.constexpr = tlx.amd_mfma_layout(4, [16, 16, 128], True, [2, 4])
    acc = tlx.require_layout(acc, acc_layout)
    result = tlx.require_layout(acc.to(tl.bfloat16), _STORE, late_address_compute=True)
    tlx.buffer_store(result, C, rows[:, None] * N + cols[None, :])


@triton.jit
def _store_quadrant_mi32(acc, C, rows, cols, N: tl.constexpr):
    acc_layout: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], True, [2, 2])
    acc = tlx.require_layout(acc, acc_layout)
    result = tlx.require_layout(acc.to(tl.bfloat16), _STORE_4W, late_address_compute=True)
    tlx.buffer_store(result, C, rows[:, None] * N + cols[None, :])


@triton.jit
def _store_partial(acc, workspace, rows, cols, M: tl.constexpr, N: tl.constexpr, split_id):
    tl.store(workspace + (split_id * M + rows[:, None]) * N + cols[None, :], acc)


@triton.jit
def _reduce_split_k(workspace, C, M: tl.constexpr, N: tl.constexpr, SPLIT_K: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for split in tl.static_range(SPLIT_K):
        acc += tl.load(workspace + split * M * N + offsets)
    tl.store(C + offsets, acc.to(tl.bfloat16))


@triton.jit
def _gemm_skinny_mi32(A, B, C, SA, SB, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      GROUP_M: tl.constexpr):
    block: tl.constexpr = 128
    mma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], True, [2, 2])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    tile_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0],
         [4, 0], [8, 0], [16, 0]],
        [block, block],
    )
    a_s = tlx.local_alloc((block, block), tlx.dtype_of(A), 2, layout=tile_layout)
    b_s = tlx.local_alloc((block, block), tlx.dtype_of(B), 2, layout=tile_layout)
    sa_s = tlx.local_alloc((block, 1), tl.uint32, 2)
    sb_s = tlx.local_alloc((block, 1), tl.uint32, 2)

    grid_n: tl.constexpr = N // block
    pid = tl.program_id(0)
    first_m = pid // (GROUP_M * grid_n) * GROUP_M
    group_m = tl.minimum(M // block - first_m, GROUP_M)
    pm = first_m + pid % (GROUP_M * grid_n) % group_m
    pn = pid % (GROUP_M * grid_n) // group_m
    rows = tl.arange(0, block)
    cols = tl.arange(0, block)
    data_off = rows[:, None] * K + cols[None, :]
    scale_word_off = tlx.require_layout(((rows % 32) * 4 + rows // 32)[:, None], _COPY_SCALE_WORD_4W)
    a_base = A + pm * block * K
    b_base = B + pn * block * K
    sa_base = SA + pm * block * (K // 32)
    sb_base = SB + pn * block * (K // 32)

    for stage in tl.static_range(2):
        tlx.buffer_load_to_local(a_s[stage], a_base + stage * block, data_off)
        tlx.buffer_load_to_local(b_s[stage], b_base + stage * block, data_off)
        tlx.buffer_load_to_local(sa_s[stage], sa_base.to(tl.pointer_type(tl.uint32)) + stage * 128, scale_word_off)
        tlx.buffer_load_to_local(sb_s[stage], sb_base.to(tl.pointer_type(tl.uint32)) + stage * 128, scale_word_off)
        tlx.async_load_commit_group()
    a_base += 2 * block
    b_base += 2 * block
    sa_base += 2 * 512
    sb_base += 2 * 512

    acc = tlx.zeros((block, block), tl.float32, layout=mma)
    tlx.async_load_wait_group(1)
    tl.debug_barrier()
    a = tlx.local_load(a_s[0], layout=dot_a, relaxed=True)
    b = tlx.local_load(tlx.local_trans(b_s[0]), layout=dot_b, relaxed=True)
    sa = _load_scale_words(sa_s[0], _SCALE_A_MI32)
    sb = _load_scale_words(sb_s[0], _SCALE_B_MI32)
    for step in tl.range(0, K // block - 2, num_stages=1):
        pipe_stage = step % 2
        next_pipe_stage = 1 - pipe_stage
        tl.debug_barrier()
        tlx.buffer_load_to_local(a_s[pipe_stage], a_base + step * block, data_off)
        tlx.buffer_load_to_local(b_s[pipe_stage], b_base + step * block, data_off)
        tlx.buffer_load_to_local(
            sa_s[pipe_stage], sa_base.to(tl.pointer_type(tl.uint32)) + step * 128, scale_word_off)
        tlx.buffer_load_to_local(
            sb_s[pipe_stage], sb_base.to(tl.pointer_type(tl.uint32)) + step * 128, scale_word_off)
        tlx.async_load_commit_group()
        acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
        tlx.async_load_wait_group(1)
        tl.debug_barrier()
        a = tlx.local_load(a_s[next_pipe_stage], layout=dot_a, relaxed=True)
        b = tlx.local_load(tlx.local_trans(b_s[next_pipe_stage]), layout=dot_b, relaxed=True)
        sa = _load_scale_words(sa_s[next_pipe_stage], _SCALE_A_MI32)
        sb = _load_scale_words(sb_s[next_pipe_stage], _SCALE_B_MI32)
    acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    tlx.async_load_wait_group(0)
    tl.debug_barrier()
    a = tlx.local_load(a_s[1], layout=dot_a, relaxed=True)
    b = tlx.local_load(tlx.local_trans(b_s[1]), layout=dot_b, relaxed=True)
    sa = _load_scale_words(sa_s[1], _SCALE_A_MI32)
    sb = _load_scale_words(sb_s[1], _SCALE_B_MI32)
    acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)

    out_rows = pm * block + tl.arange(0, block)
    out_cols = pn * block + tl.arange(0, block)
    _store_quadrant_mi32(acc, C, out_rows, out_cols, N)


@triton.jit
def _gemm_skinny(A, B, C, WORKSPACE, SA, SB, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 K_TILES: tl.constexpr, SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr,
                 CUBLAS_BLOCKED: tl.constexpr, UNEVEN_SPLIT: tl.constexpr = False,
                 LOOP_UNROLL_FACTOR: tl.constexpr = 1):
    block: tl.constexpr = 128
    scale_groups: tl.constexpr = block // 32
    load_a: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((16, 32, 64, 128, 4096, 8192, 256, 512, 1024), (1, 2, 4, 8, 2048)),
    )
    load_b: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((16, 32, 64, 128, 4096, 8192, 256, 512, 1024), (1, 2, 4, 8, 2048)),
    )
    tile_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0],
         [4, 0], [8, 0], [16, 0]],
        [block, block],
    )

    grid_n: tl.constexpr = N // block
    grid_mn: tl.constexpr = M // block * grid_n
    split_id = tl.program_id(0) // grid_mn
    pid = tl.program_id(0) % grid_mn
    first_m = pid // (GROUP_M * grid_n) * GROUP_M
    group_m = tl.minimum(M // block - first_m, GROUP_M)
    pm = first_m + pid % (GROUP_M * grid_n) % group_m
    pn = pid % (GROUP_M * grid_n) // group_m

    split_k: tl.constexpr = K // SPLIT_K
    split_start = split_id * ((K_TILES + 1) * block if UNEVEN_SPLIT else split_k)
    rows = tl.arange(0, block)
    data_cols = tl.arange(0, block)
    groups = tl.arange(0, scale_groups)
    data_off_a = tlx.require_layout(rows[:, None] * K + data_cols[None, :], load_a)
    data_off_b = tlx.require_layout(rows[:, None] * K + data_cols[None, :], load_b)
    if CUBLAS_BLOCKED:
        scale_off = (rows[:, None] % 32) * 16 + (rows[:, None] // 32) * 4 + groups[None, :]
        scale_off_a = tlx.require_layout(scale_off, _COPY_SCALE)
        scale_off_b = scale_off_a
        scale_step: tl.constexpr = 512
        scale_split = (split_start // 128) * 512
    else:
        scale_off_a = tlx.require_layout(rows[:, None] * (K // 32) + groups[None, :], _COPY_SCALE)
        scale_off_b = scale_off_a
        scale_step: tl.constexpr = scale_groups
        scale_split = split_start // 32
    a_base = A + pm * block * K + split_start
    b_base = B + pn * block * K + split_start
    sa_base = SA + pm * block * (K // 32) + scale_split
    sb_base = SB + pn * block * (K // 32) + scale_split

    a_s = tlx.local_alloc((block, block), tlx.dtype_of(A), 2, layout=tile_layout)
    b_s = tlx.local_alloc((block, block), tlx.dtype_of(B), 2, layout=tile_layout)
    sa_s = tlx.local_alloc((block, scale_groups), tl.uint8, 2)
    sb_s = tlx.local_alloc((block, scale_groups), tl.uint8, 2)
    for stage in tl.static_range(2):
        tlx.buffer_load_to_local(a_s[stage], a_base + stage * block, data_off_a)
        tlx.buffer_load_to_local(b_s[stage], b_base + stage * block, data_off_b)
        tlx.buffer_load_to_local(sa_s[stage], sa_base + stage * scale_step, scale_off_a)
        tlx.buffer_load_to_local(sb_s[stage], sb_base + stage * scale_step, scale_off_b)
        tlx.async_load_commit_group()
    a_base += 2 * block
    b_base += 2 * block
    sa_base += 2 * scale_step
    sb_base += 2 * scale_step

    acc = tl.zeros((block, block), tl.float32)
    main_k_tiles: tl.constexpr = K_TILES - K_TILES % 2
    if UNEVEN_SPLIT:
        tlx.async_load_wait_group(1)
        tl.debug_barrier()
        a = tlx.local_load(a_s[0], relaxed=True)
        b = tlx.local_load(tlx.local_trans(b_s[0]), relaxed=True)
        sa = _load_scale(sa_s[0], _SCALE_A, False)
        sb = _load_scale(sb_s[0], _SCALE_B, False)
        for step in tl.range(0, main_k_tiles - 2, num_stages=1, loop_unroll_factor=LOOP_UNROLL_FACTOR):
            pipe_stage = step % 2
            next_pipe_stage = 1 - pipe_stage
            tl.debug_barrier()
            tlx.buffer_load_to_local(a_s[pipe_stage], a_base + step * block, data_off_a)
            tlx.buffer_load_to_local(b_s[pipe_stage], b_base + step * block, data_off_b)
            tlx.buffer_load_to_local(sa_s[pipe_stage], sa_base + step * scale_step, scale_off_a)
            tlx.buffer_load_to_local(sb_s[pipe_stage], sb_base + step * scale_step, scale_off_b)
            tlx.async_load_commit_group()
            acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
            tlx.async_load_wait_group(1)
            tl.debug_barrier()
            a = tlx.local_load(a_s[next_pipe_stage], relaxed=True)
            b = tlx.local_load(tlx.local_trans(b_s[next_pipe_stage]), relaxed=True)
            sa = _load_scale(sa_s[next_pipe_stage], _SCALE_A, False)
            sb = _load_scale(sb_s[next_pipe_stage], _SCALE_B, False)
        acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
        tlx.async_load_wait_group(0)
        tl.debug_barrier()
        a = tlx.local_load(a_s[1], relaxed=True)
        b = tlx.local_load(tlx.local_trans(b_s[1]), relaxed=True)
        sa = _load_scale(sa_s[1], _SCALE_A, False)
        sb = _load_scale(sb_s[1], _SCALE_B, False)
        acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    else:
        for _ in tl.range(0, main_k_tiles - 2, 2, num_stages=1):
            for stage in tl.static_range(2):
                tlx.async_load_wait_group(1)
                tl.debug_barrier()
                a = tlx.local_load(a_s[stage], relaxed=True)
                b = tlx.local_load(tlx.local_trans(b_s[stage]), relaxed=True)
                sa = _load_scale(sa_s[stage], _SCALE_A, False)
                sb = _load_scale(sb_s[stage], _SCALE_B, False)
                acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
                tl.debug_barrier()
                tlx.buffer_load_to_local(a_s[stage], a_base + stage * block, data_off_a)
                tlx.buffer_load_to_local(b_s[stage], b_base + stage * block, data_off_b)
                tlx.buffer_load_to_local(sa_s[stage], sa_base + stage * scale_step, scale_off_a)
                tlx.buffer_load_to_local(sb_s[stage], sb_base + stage * scale_step, scale_off_b)
                tlx.async_load_commit_group()
            a_base += 2 * block
            b_base += 2 * block
            sa_base += 2 * scale_step
            sb_base += 2 * scale_step
        for stage in tl.static_range(2):
            tlx.async_load_wait_group(1 - stage)
            tl.debug_barrier()
            a = tlx.local_load(a_s[stage], relaxed=True)
            b = tlx.local_load(tlx.local_trans(b_s[stage]), relaxed=True)
            sa = _load_scale(sa_s[stage], _SCALE_A, False)
            sb = _load_scale(sb_s[stage], _SCALE_B, False)
            acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)

    if K_TILES % 2 or UNEVEN_SPLIT:
        tail: tl.constexpr = (K_TILES if UNEVEN_SPLIT else K_TILES - 1) * block
        tail_scale: tl.constexpr = (K_TILES if UNEVEN_SPLIT else K_TILES - 1) * scale_step
        if not UNEVEN_SPLIT or split_id == 0:
            a_tail = A + pm * block * K + split_start + tail
            b_tail = B + pn * block * K + split_start + tail
            sa_tail = SA + pm * block * (K // 32) + scale_split + tail_scale
            sb_tail = SB + pn * block * (K // 32) + scale_split + tail_scale
            tlx.buffer_load_to_local(a_s[0], a_tail, data_off_a)
            tlx.buffer_load_to_local(b_s[0], b_tail, data_off_b)
            tlx.buffer_load_to_local(sa_s[0], sa_tail, scale_off_a)
            tlx.buffer_load_to_local(sb_s[0], sb_tail, scale_off_b)
            tlx.async_load_commit_group()
            tlx.async_load_wait_group(0)
            tl.debug_barrier()
            a = tlx.local_load(a_s[0], relaxed=True)
            b = tlx.local_load(tlx.local_trans(b_s[0]), relaxed=True)
            sa = _load_scale(sa_s[0], _SCALE_A, False)
            sb = _load_scale(sb_s[0], _SCALE_B, False)
            acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)

    cm = pm * block + tlx.rematerialized_range(0, block, 10, placement=K_TILES)
    cn = pn * block + tlx.rematerialized_range(0, block, 11, placement=K_TILES)
    if SPLIT_K == 1:
        _store_quadrant(acc, C, cm, cn, N)
    else:
        _store_partial(acc, WORKSPACE, cm, cn, M, N, split_id)


@triton.jit
def _gemm_interwave(A, B, C, WORKSPACE, SA, SB, M: tl.constexpr, N: tl.constexpr, K_TILES,
                    K: tl.constexpr, SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr,
                    CUBLAS_BLOCKED: tl.constexpr, MODEL_XCD: tl.constexpr,
                    K_MAJOR: tl.constexpr = False, UNEVEN_SPLIT: tl.constexpr = False,
                    LLVM_MFMA_SCHED: tl.constexpr = False, LOOP_UNROLL_FACTOR: tl.constexpr = 1):
    # Keep K bytes contiguous for direct copies, with MFMA-friendly row bases.
    # A 32-byte pad per 1024 E4M3 bytes spreads row strides across LDS banks.
    tile_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0],
         [16, 0]],
        [128, 128],
    )
    a_top_s = tlx.local_alloc((128, 128), tlx.dtype_of(A), 2, layout=tile_layout)
    a_bot_s = tlx.local_alloc((128, 128), tlx.dtype_of(A), 2, layout=tile_layout)
    b_left_s = tlx.local_alloc((128, 128), tlx.dtype_of(B), 2, layout=tile_layout)
    b_right_s = tlx.local_alloc((128, 128), tlx.dtype_of(B), 2, layout=tile_layout)
    if K_MAJOR:
        sa_top_s = tlx.local_alloc((32, 16), tl.uint8, 2)
        sa_bot_s = tlx.local_alloc((32, 16), tl.uint8, 2)
        sb_left_s = tlx.local_alloc((32, 16), tl.uint8, 2)
        sb_right_s = tlx.local_alloc((32, 16), tl.uint8, 2)
    else:
        sa_top_s = tlx.local_alloc((128, 4), tl.uint8, 2)
        sa_bot_s = tlx.local_alloc((128, 4), tl.uint8, 2)
        sb_left_s = tlx.local_alloc((128, 4), tl.uint8, 2)
        sb_right_s = tlx.local_alloc((128, 4), tl.uint8, 2)
    grid_mn: tl.constexpr = M // 256 * (N // 256)
    split_id = tl.program_id(0) // grid_mn
    pid = tl.program_id(0) % grid_mn
    if MODEL_XCD:
        pids_per_xcd: tl.constexpr = triton.cdiv(grid_mn, 8)
        tall_xcds: tl.constexpr = grid_mn % 8 or 8
        xcd = pid % 8
        local_pid = pid // 8
        pid = tl.where(
            xcd < tall_xcds,
            xcd * pids_per_xcd + local_pid,
            tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid,
        )
    if M == 8192 and N == 8192 and K == 8192:
        # On this 32x32 tile grid, 32 consecutive CTAs share one B panel after remapping.
        # The fixed 8x128 transpose is tuned for 8K^3; other shapes keep grouped order.
        pid = (pid % 8) * 128 + pid // 8
    first_m = pid // (GROUP_M * (N // 256)) * GROUP_M
    group_m = tl.minimum(M // 256 - first_m, GROUP_M)
    pm = first_m + pid % (GROUP_M * (N // 256)) % group_m
    pn = pid % (GROUP_M * (N // 256)) // group_m
    split_k: tl.constexpr = K // SPLIT_K
    split_start = split_id * ((K_TILES + 1) * 128 if UNEVEN_SPLIT else split_k)
    a_base = A + pm * 256 * K + split_start
    a_bot_base = a_base + 128 * K
    b_base = B + pn * 256 * K + split_start
    b_right_base = b_base + 128 * K
    rows = tl.arange(0, 128)
    groups = tl.arange(0, 4)
    data_off = rows[:, None] * K + tl.arange(0, 128)[None, :]
    if K_MAJOR:
        scale_off = tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]
        scale_off = tlx.require_layout(scale_off, _COPY_SCALE_K_MAJOR)
        scale_step: tl.constexpr = 512
    elif CUBLAS_BLOCKED:
        scale_off = (rows[:, None] % 32) * 16 + (rows[:, None] // 32) * 4 + groups[None, :]
        scale_off = tlx.require_layout(scale_off, _COPY_SCALE)
        scale_step: tl.constexpr = 512
    else:
        scale_off = rows[:, None] * (K // 32) + groups[None, :]
        scale_off = tlx.require_layout(scale_off, _COPY_SCALE)
        scale_step: tl.constexpr = 4
    scale_split = ((split_start // 128) * 512
                   if CUBLAS_BLOCKED or K_MAJOR else split_start // 32)
    sa_base = SA + pm * 256 * (K // 32) + scale_split
    sa_bot_base = sa_base + 128 * (K // 32)
    sb_base = SB + pn * 256 * (K // 32) + scale_split
    sb_right_base = sb_base + 128 * (K // 32)
    acc_tl = tl.zeros((128, 128), tl.float32)
    acc_bl = tl.zeros((128, 128), tl.float32)
    acc_tr = tl.zeros((128, 128), tl.float32)
    acc_br = tl.zeros((128, 128), tl.float32)
    main_k_tiles: tl.constexpr = K_TILES - K_TILES % 2

    for slot in tl.static_range(2):
        tlx.buffer_load_to_local(b_left_s[slot], b_base + slot * 128, data_off)
        tlx.buffer_load_to_local(sb_left_s[slot], sb_base + slot * scale_step, scale_off)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(a_top_s[slot], a_base + slot * 128, data_off)
        tlx.buffer_load_to_local(sa_top_s[slot], sa_base + slot * scale_step, scale_off)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(a_bot_s[slot], a_bot_base + slot * 128, data_off)
        tlx.buffer_load_to_local(sa_bot_s[slot], sa_bot_base + slot * scale_step, scale_off)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(b_right_s[slot], b_right_base + slot * 128, data_off)
        tlx.buffer_load_to_local(sb_right_s[slot], sb_right_base + slot * scale_step, scale_off)
        tlx.async_load_commit_group()
    a_base += 256
    a_bot_base += 256
    b_base += 256
    b_right_base += 256
    sa_base += 2 * scale_step
    sa_bot_base += 2 * scale_step
    sb_base += 2 * scale_step
    sb_right_base += 2 * scale_step
    # Groups count commits: 2 slots x 4 (data + scale) = 8; wait(6) completes slot 0's B-left/A-top.
    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(tlx.local_trans(b_left_s[0]), relaxed=True)
    sb_l = _load_scale(sb_left_s[0], _SCALE_B, K_MAJOR)
    a_top = tlx.local_load(a_top_s[0], relaxed=True)
    sa_t = _load_scale(sa_top_s[0], _SCALE_A, K_MAJOR)

    # Each refill adds one group; wait(5) completes the next data/scale group before its LDS read.
    for k in tl.range(0, main_k_tiles - 2, 2, num_stages=1, loop_unroll_factor=LOOP_UNROLL_FACTOR):
        for slot in tl.static_range(2):
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tl = tlx.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_bot = tlx.local_load(a_bot_s[slot], relaxed=True)
                sa_b = _load_scale(sa_bot_s[slot], _SCALE_A, K_MAJOR)
                if not LLVM_MFMA_SCHED:
                    tlx.amd_sched_barrier(0)
                tlx.buffer_load_to_local(b_left_s[slot], b_base + slot * 128, data_off)
                tlx.buffer_load_to_local(sb_left_s[slot], sb_base + slot * scale_step, scale_off)
                tlx.async_load_commit_group()
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_bl = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_right = tlx.local_load(tlx.local_trans(b_right_s[slot]), relaxed=True)
                sb_r = _load_scale(sb_right_s[slot], _SCALE_B, K_MAJOR)
                if not LLVM_MFMA_SCHED:
                    tlx.amd_sched_barrier(0)
                tlx.buffer_load_to_local(a_top_s[slot], a_base + slot * 128, data_off)
                tlx.buffer_load_to_local(sa_top_s[slot], sa_base + slot * scale_step, scale_off)
                tlx.async_load_commit_group()
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tr = tlx.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_left = tlx.local_load(tlx.local_trans(b_left_s[1 - slot]), relaxed=True)
                sb_l = _load_scale(sb_left_s[1 - slot], _SCALE_B, K_MAJOR)
                if not LLVM_MFMA_SCHED:
                    tlx.amd_sched_barrier(0)
                tlx.buffer_load_to_local(a_bot_s[slot], a_bot_base + slot * 128, data_off)
                tlx.buffer_load_to_local(sa_bot_s[slot], sa_bot_base + slot * scale_step, scale_off)
                tlx.async_load_commit_group()
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_br = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_top = tlx.local_load(a_top_s[1 - slot], relaxed=True)
                sa_t = _load_scale(sa_top_s[1 - slot], _SCALE_A, K_MAJOR)
                if not LLVM_MFMA_SCHED:
                    tlx.amd_sched_barrier(0)
                tlx.buffer_load_to_local(b_right_s[slot], b_right_base + slot * 128, data_off)
                tlx.buffer_load_to_local(sb_right_s[slot], sb_right_base + slot * scale_step, scale_off)
                tlx.async_load_commit_group()
                if slot == 1:
                    a_base += 256
                    a_bot_base += 256
                    b_base += 256
                    b_right_base += 256
                    sa_base += 2 * scale_step
                    sa_bot_base += 2 * scale_step
                    sb_base += 2 * scale_step
                    sb_right_base += 2 * scale_step

    # Without refills, wait(5), ..., wait(0) drains the six remaining groups in issue order.
    acc_tl = tlx.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
    tlx.async_load_wait_group(5)
    a_bot = tlx.local_load(a_bot_s[0], relaxed=True)
    sa_b = _load_scale(sa_bot_s[0], _SCALE_A, K_MAJOR)
    acc_bl = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_trans(b_right_s[0]), relaxed=True)
    sb_r = _load_scale(sb_right_s[0], _SCALE_B, K_MAJOR)
    acc_tr = tlx.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
    tlx.async_load_wait_group(3)
    b_left = tlx.local_load(tlx.local_trans(b_left_s[1]), relaxed=True)
    sb_l = _load_scale(sb_left_s[1], _SCALE_B, K_MAJOR)
    acc_br = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(a_top_s[1], relaxed=True)
    sa_t = _load_scale(sa_top_s[1], _SCALE_A, K_MAJOR)
    acc_tl = tlx.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(a_bot_s[1], relaxed=True)
    sa_b = _load_scale(sa_bot_s[1], _SCALE_A, K_MAJOR)
    acc_bl = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_trans(b_right_s[1]), relaxed=True)
    sb_r = _load_scale(sb_right_s[1], _SCALE_B, K_MAJOR)
    acc_tr = tlx.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
    acc_br = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
    if K_TILES % 2 or UNEVEN_SPLIT:
        tail: tl.constexpr = (K_TILES if UNEVEN_SPLIT else K_TILES - 1) * 128
        tail_scale: tl.constexpr = (K_TILES if UNEVEN_SPLIT else K_TILES - 1) * scale_step
        if not UNEVEN_SPLIT or split_id == 0:
            a_tail = A + pm * 256 * K + split_start + tail
            b_tail = B + pn * 256 * K + split_start + tail
            sa_tail = SA + pm * 256 * (K // 32) + scale_split + tail_scale
            sb_tail = SB + pn * 256 * (K // 32) + scale_split + tail_scale
            tlx.buffer_load_to_local(b_left_s[0], b_tail, data_off)
            tlx.buffer_load_to_local(sb_left_s[0], sb_tail, scale_off)
            tlx.async_load_commit_group()
            tlx.buffer_load_to_local(a_top_s[0], a_tail, data_off)
            tlx.buffer_load_to_local(sa_top_s[0], sa_tail, scale_off)
            tlx.async_load_commit_group()
            tlx.buffer_load_to_local(a_bot_s[0], a_tail + 128 * K, data_off)
            tlx.buffer_load_to_local(sa_bot_s[0], sa_tail + 128 * (K // 32), scale_off)
            tlx.async_load_commit_group()
            tlx.buffer_load_to_local(b_right_s[0], b_tail + 128 * K, data_off)
            tlx.buffer_load_to_local(sb_right_s[0], sb_tail + 128 * (K // 32), scale_off)
            tlx.async_load_commit_group()
            tlx.async_load_wait_group(0)
            b_left = tlx.local_load(tlx.local_trans(b_left_s[0]), relaxed=True)
            sb_l = _load_scale(sb_left_s[0], _SCALE_B, K_MAJOR)
            a_top = tlx.local_load(a_top_s[0], relaxed=True)
            sa_t = _load_scale(sa_top_s[0], _SCALE_A, K_MAJOR)
            a_bot = tlx.local_load(a_bot_s[0], relaxed=True)
            sa_b = _load_scale(sa_bot_s[0], _SCALE_A, K_MAJOR)
            b_right = tlx.local_load(tlx.local_trans(b_right_s[0]), relaxed=True)
            sb_r = _load_scale(sb_right_s[0], _SCALE_B, K_MAJOR)
            acc_tl = tlx.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
            acc_bl = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
            acc_tr = tlx.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
            acc_br = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
    cm = pm * 256 + tlx.rematerialized_range(0, 128, 10, placement=K_TILES)
    cn = pn * 256 + tlx.rematerialized_range(0, 128, 11, placement=K_TILES)
    if SPLIT_K == 1:
        _store_quadrant(acc_tl, C, cm, cn, N)
        _store_quadrant(acc_bl, C, cm + 128, cn, N)
        _store_quadrant(acc_tr, C, cm, cn + 128, N)
        _store_quadrant(acc_br, C, cm + 128, cn + 128, N)
    else:
        _store_partial(acc_tl, WORKSPACE, cm, cn, M, N, split_id)
        _store_partial(acc_bl, WORKSPACE, cm + 128, cn, M, N, split_id)
        _store_partial(acc_tr, WORKSPACE, cm, cn + 128, M, N, split_id)
        _store_partial(acc_br, WORKSPACE, cm + 128, cn + 128, M, N, split_id)


@triton.jit
def _gemm_generic(A, B, C, SA, SB, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, CUBLAS_BLOCKED: tl.constexpr):
    """Compute complete 128-aligned tiles; the backend rejects partial dimensions."""
    pid = tl.program_id(0)
    gn: tl.constexpr = N // BN
    first_m = pid // (GROUP_M * gn) * GROUP_M
    group_m = tl.minimum(M // BM - first_m, GROUP_M)
    pm = first_m + pid % (GROUP_M * gn) % group_m
    pn = pid % (GROUP_M * gn) // group_m
    ar = pm * BM + tl.arange(0, BM)
    br = pn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    sg = tl.arange(0, BK // 32)
    a_off = ar[:, None] * K + kk[None, :]
    b_off = kk[:, None] + br[None, :] * K
    if CUBLAS_BLOCKED:
        sa_off = (ar[:, None] // 128) * (K // 128) * 512 + (ar[:, None] % 32) * 16
        sa_off += (ar[:, None] % 128 // 32) * 4 + sg[None, :]
        sb_off = (br[:, None] // 128) * (K // 128) * 512 + (br[:, None] % 32) * 16
        sb_off += (br[:, None] % 128 // 32) * 4 + sg[None, :]
        scale_step: tl.constexpr = 512
    else:
        sa_off = ar[:, None] * (K // 32) + sg[None, :]
        sb_off = br[:, None] * (K // 32) + sg[None, :]
        scale_step: tl.constexpr = BK // 32
    acc = tl.zeros((BM, BN), tl.float32)
    for kt in tl.range(K // BK, num_stages=1):
        a = tlx.buffer_load(A + kt * BK, a_off)
        b = tlx.buffer_load(B + kt * BK, b_off)
        sa = tlx.buffer_load(SA + kt * scale_step, sa_off)
        sb = tlx.buffer_load(SB + kt * scale_step, sb_off)
        acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    tlx.buffer_store(acc.to(tl.bfloat16), C, ar[:, None] * N + br[None, :])


def _pack_k_major_scales(scale):
    """Return a contiguous K-major copy for 32-bit LDS scale reads.

    Uses one extra byte per scale while source and packed tensors coexist at
    launch. The copy is included in end-to-end GEMM timing.
    """
    rows, groups = scale.shape
    return (scale.view(torch.uint8).view(rows // 128, 4, 32, groups // 4, 4).permute(0, 3, 2, 4,
                                                                                     1).contiguous().view(rows, groups))


def _model_split_k(m, n, k, cu_count, block):
    if (m, n, k, block) == (2048, 6144, 22656, 256):
        return 1
    if (m, n, k, block) == (1024, 8192, 3072, 256):
        return 1
    if (m, n, k, block) == (1024, 3072, 15360, 256):
        return 5
    if (m, n, k, block) == (2048, 3072, 15360, 256):
        return 2
    if (m, n, k, block) == (1024, 1024, 24576, 256):
        return 12
    grid_mn = m // block * (n // block)
    k_tiles = k // 128
    target = triton.cdiv(cu_count, grid_mn)
    if target <= 1:
        return 1
    candidate = target + (1 if k >= 32768 else 0)
    for split_k in range(candidate, 1, -1):
        if k_tiles % split_k == 0:
            return split_k
    return 1


def _launch(a, a_scale, b, b_scale, out, *, sf_layout):
    m, k = a.shape
    n = b.shape[0]
    with torch.cuda.device(a.device):
        # GROUP_M=4 groups up to four M tiles per B panel in both launch paths.
        # Keeping the group small also bounds A-panel reuse distance along N.
        cu_count = torch.cuda.get_device_properties(a.device).multi_processor_count
        model_linear = (m, n, k) in _MODEL_LINEAR_SHAPES
        if (m, n, k) == (1024, 8192, 3072) and sf_layout == "cublas_blocked":
            return _gemm_skinny_mi32[(m // 128 * (n // 128), )](
                a, b, out, a_scale.view(torch.uint8), b_scale.view(torch.uint8), m, n, k, 1,
                num_warps=4, num_stages=1, matrix_instr_nonkdim=32,
                disable_unclustered_high_rp_reschedule=True, reverse_local_assignment=False)
        model_skinny = ((sf_layout == "natural" and model_linear and k >= 12288 and k % 256 == 0)
                        or (m, n, k) in {(1024, 8192, 3072), (1024, 3072, 12416)})
        if model_skinny:
            block = 128
            skinny_uneven_split = (m, n, k) == (1024, 3072, 12416)
            split_k = 2 if skinny_uneven_split else _model_split_k(m, n, k, cu_count, block)
            skinny_group_m = (1 if (m, n, k) == (1024, 8192, 3072) else
                              (2 if skinny_uneven_split else 4))
            workspace = (torch.empty((split_k * m, n), dtype=torch.float32, device=a.device)
                         if split_k > 1 else out)
            _gemm_skinny[(m // block * (n // block) * split_k, )](
                a, b, out, workspace, a_scale.view(torch.uint8), b_scale.view(torch.uint8), m, n, k,
                k // 256 if skinny_uneven_split else k // split_k // block, split_k, skinny_group_m,
                sf_layout == "cublas_blocked", UNEVEN_SPLIT=skinny_uneven_split,
                LOOP_UNROLL_FACTOR=8 if skinny_uneven_split else 1, num_warps=8, num_stages=1,
                matrix_instr_nonkdim=16, waves_per_eu=1 if skinny_uneven_split else 0,
                disable_unclustered_high_rp_reschedule=True, reverse_local_assignment=True,
                llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),
                               ("amdgpu-sched-strategy", "iterative-ilp" if skinny_uneven_split else "max-ilp")))
            if split_k > 1:
                reduce_m, reduce_n = 32, 32
                _reduce_split_k[(m // reduce_m, n // reduce_n)](
                    workspace, out, m, n, split_k, reduce_m, reduce_n, num_warps=4)
            return out
        interwave_shape = ((n <= 8192 and m <= 8192 and 2048 <= k <= 8192)
                           or model_linear
                           or (sf_layout == "natural" and m >= 16384 and n >= 1024 and 512 <= k <= 4096))
        interwave = (m % 256 == 0 and n % 256 == 0 and k % 128 == 0 and interwave_shape
                     and ((m // 256) * (n // 256) >= cu_count or model_linear)
                     and all(tensor.data_ptr() % 16 == 0 for tensor in (a, a_scale, b, b_scale, out)))
        if interwave:
            prepacked_k_major = sf_layout == "k_major"
            k_major = prepacked_k_major or (sf_layout == "natural"
                                            and (m == n == k == 8192 or (m >= 16384 and k >= 4096)))
            uneven_split = (m, n, k) in {
                (1024, 6144, 22656),
                (2048, 3072, 12416),
            }
            split_k = (2 if uneven_split else
                       (_model_split_k(m, n, k, cu_count, 256) if model_linear else 1))
            model_xcd = model_linear and (m, n, k) not in {
                (1024, 1024, 159744),
                (1024, 6144, 22656),
                (2048, 1024, 159744),
                (2048, 3072, 12416),
                (2048, 6144, 22656),
                (2048, 1024, 24576),
            }
            if (m, n, k) in {
                (1024, 6144, 22656),
                (1024, 32768, 6144),
                (1024, 64512, 1024),
                (2048, 3072, 12416),
                (2048, 3072, 15360),
            }:
                group_m = 2
            else:
                group_m = 8 if model_linear and n >= 153600 else 4
            use_fp16_workspace = split_k > 1 and (m, n, k) != (2048, 6144, 159744)
            workspace_dtype = torch.float16 if use_fp16_workspace else torch.float32
            workspace = (torch.empty((split_k * m, n), dtype=workspace_dtype, device=a.device)
                         if split_k > 1 else out)
            use_mfma_schedule = k == 1024 and (m, n) != (2048, 64512)
            sa = (_pack_k_major_scales(a_scale) if k_major and not prepacked_k_major else a_scale.view(torch.uint8))
            sb = (_pack_k_major_scales(b_scale) if k_major and not prepacked_k_major else b_scale.view(torch.uint8))
            _gemm_interwave[(m // 256 * (n // 256) * split_k, )](
                a, b, out, workspace, sa, sb, m, n, k // 256 if uneven_split else k // split_k // 128, k, split_k,
                group_m,
                sf_layout == "cublas_blocked", model_xcd, num_warps=8, num_stages=1, K_MAJOR=k_major,
                UNEVEN_SPLIT=uneven_split, LLVM_MFMA_SCHED=use_mfma_schedule,
                LOOP_UNROLL_FACTOR=2 if (m, n, k) == (1024, 6144, 39936) else 1,
                matrix_instr_nonkdim=16, disable_unclustered_high_rp_reschedule=not prepacked_k_major,
                reverse_local_assignment=k != 1024 or (m, n) == (2048, 64512),
                enable_sched_group_barrier_scheduler=k == 6144,
                schedule_hint="mfma-schedule" if use_mfma_schedule else "",
                llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ))
            if split_k > 1:
                block_m, block_n = 32, 32
                _reduce_split_k[(m // block_m, n // block_n)](
                    workspace, out, m, n, split_k, block_m, block_n, num_warps=4)
            return out
        return _gemm_generic[(triton.cdiv(m, 128) * triton.cdiv(n, 128), )](
            a, b, out, a_scale.view(torch.uint8), b_scale.view(torch.uint8), m, n, k, 128, 128, 128, 4,
            sf_layout == "cublas_blocked", num_warps=4, num_stages=1, matrix_instr_nonkdim=16,
            disable_unclustered_high_rp_reschedule=True, llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ))


def mm_mxfp8(a, a_scale, b, b_scale, *, out=None, sf_layout="natural", space="heuristic"):
    if space != "heuristic":
        raise InvalidInput(f"tlx.ops.mm_mxfp8 on gfx950 does not provide space={space!r}")
    m, k = a.shape
    n = b.shape[0]
    if min(m, n, k) <= 0 or m % 128 or n % 128 or k % 128:
        raise InvalidInput("tlx.ops.mm_mxfp8 on gfx950 requires positive M, N and K divisible by 128; "
                           f"got M={m}, N={n}, K={k}")
    if max(m * k, n * k, 2 * m * n) >= 2**31:
        raise InvalidInput(
            "gfx950 MXFP8 GEMM requires data and output byte spans addressable with signed 32-bit offsets")
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    _launch(a, a_scale, b, b_scale, out, sf_layout=sf_layout)
    return out
