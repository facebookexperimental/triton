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
_SCALE_A_MI32_4X1 = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, )),
    stride=((4, 8, 16, 32, 64, 1, 128, 256), (2, )),
)
_SCALE_B_MI32_4X1_128 = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
    stride=((4, 8, 16, 32, 64, 1, 0, 0), (2, 128, 256)),
)
_SCALE_B_MI32_4X1_64 = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
    stride=((4, 8, 16, 32, 64, 1, 0, 0), (2, 128)),
)
_COPY_SCALE_WORD_4W_64 = tlx.layout(shape=((64, 4), (1, )), stride=((1, 0), (0, )))
_COPY_DATA_4W_128 = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2)),
    stride=((16, 32, 64, 128, 4096, 8192, 256, 512), (1, 2, 4, 8, 1024, 2048)),
)
# Each lane owns eight adjacent BF16 columns at a time for 16-byte stores.
# The remaining register bits select rows rather than strided MFMA columns.
_STORE = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
    stride=((8, 16, 32, 64, 128, 256, 512, 1024, 2048), (1, 2, 4, 4096, 8192)),
)
# FP32 split partials: 32 lanes cover one 128-column row with four floats each.
_STORE_F32 = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
    stride=((4, 8, 16, 32, 64, 128, 256, 512, 1024), (1, 2, 2048, 4096, 8192)),
)
_STORE_4W = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2)),
    stride=((8, 16, 32, 64, 128, 256, 512, 1024), (1, 2, 4, 2048, 4096, 8192)),
)
_STORE_MI32_4X1_128 = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2)),
    stride=((8, 16, 32, 64, 128, 256, 4096, 8192), (1, 2, 4, 512, 1024, 2048)),
)
_STORE_MI32_4X1_64 = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
    stride=((8, 16, 32, 64, 128, 256, 2048, 4096), (1, 2, 4, 512, 1024)),
)
# FP32 partials keep four adjacent floats per lane so each row segment is whole lines.
_STORE_MI32_4X1_128_F32 = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2)),
    stride=((4, 8, 16, 32, 64, 128, 4096, 8192), (1, 2, 256, 512, 1024, 2048)),
)
_STORE_MI32_4X1_64_F32 = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
    stride=((4, 8, 16, 32, 64, 128, 2048, 4096), (1, 2, 256, 512, 1024)),
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
def _load_scale_words_64(buf, layout: tl.constexpr):
    read_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
        block_bases=[],
        alignment=4,
    )
    view = tlx.local_reinterpret(buf, tl.uint8, [64, 4], layout=read_layout)
    return tlx.local_load(view, layout=layout, relaxed=True)


@tl.core.builtin
def _dot_scaled_mi32_fragments(
    a_fragments,
    sa_fragments,
    b_fragments,
    sb_fragments,
    acc,
    n_fragments: tl.constexpr,
    _semantic=None,
):
    """Update a 128-row accumulator as native K64 by N32 fragments."""
    n_fragments = tl.core._unwrap_if_constexpr(n_fragments)
    values = list(acc)
    for k_half in range(2):
        for column in range(n_fragments):
            index = k_half * n_fragments + column
            values[column] = tlx.dot_scaled(
                a_fragments[k_half],
                sa_fragments[k_half],
                "e4m3",
                b_fragments[index],
                sb_fragments[index],
                "e4m3",
                values[column],
                _semantic=_semantic,
            )
    return tl.tuple(values)


@triton.jit
def _load_mi32_a_fragments(data_buf, scale_buf, dot_a: tl.constexpr):
    all_scales = tlx.require_layout(
        _load_scale_words(scale_buf, _SCALE_A_MI32_4X1),
        _SCALE_A_MI32_4X1,
        pin=False,
    )
    data = tl.tuple([])
    scales = tl.tuple([])
    for k_half in tl.static_range(2):
        data += tl.tuple(
            [tlx.local_load(
                tlx.local_slice(data_buf, [0, k_half * 64], [128, 64]),
                layout=dot_a,
                relaxed=True,
            )])
        scales += tl.tuple([tlx.extract_slice(
            all_scales,
            [128, 2],
            [0, k_half * 2],
        )])
    return data, scales


@triton.jit
def _load_mi32_b_fragments(
    data_buf,
    scale_buf,
    n_fragments: tl.constexpr,
    dot_b: tl.constexpr,
):
    if n_fragments == 4:
        scale_register_layout: tl.constexpr = _SCALE_B_MI32_4X1_128
        all_scales = _load_scale_words(scale_buf, scale_register_layout)
    else:
        scale_register_layout: tl.constexpr = _SCALE_B_MI32_4X1_64
        all_scales = _load_scale_words_64(scale_buf, scale_register_layout)
    all_scales = tlx.require_layout(
        all_scales,
        scale_register_layout,
        pin=False,
    )
    data = tl.tuple([])
    scales = tl.tuple([])
    for k_half in tl.static_range(2):
        for column in tl.static_range(n_fragments):
            panel = tlx.local_slice(
                data_buf,
                [column * 32, k_half * 64],
                [32, 64],
            )
            data += tl.tuple([tlx.local_load(tlx.local_trans(panel), layout=dot_b, relaxed=True)])
            scales += tl.tuple([tlx.extract_slice(
                all_scales,
                [32, 2],
                [column * 32, k_half * 2],
            )])
    return data, scales


@tl.core.builtin
def _join_mi32_fragments(values, n_fragments: tl.constexpr, _semantic=None):
    """Join N32 fragments through equal-size concatenations for the epilogue."""
    n_fragments = tl.core._unwrap_if_constexpr(n_fragments)
    values = list(values)
    if n_fragments == 2:
        return tl.cat(values[0], values[1], dim=1, _semantic=_semantic)
    low = tl.cat(values[0], values[1], dim=1, _semantic=_semantic)
    high = tl.cat(values[2], values[3], dim=1, _semantic=_semantic)
    return tl.cat(low, high, dim=1, _semantic=_semantic)


@triton.jit
def _blocked_scale_word_offsets(rows, K: tl.constexpr):
    return ((rows[:, None] // 128) * (K // 128) * 128 + (rows[:, None] % 32) * 4 + (rows[:, None] % 128) // 32)


@triton.jit
def _store_quadrant(acc, C, rows, cols, N: tl.constexpr, STREAMING: tl.constexpr = False):
    acc_layout: tl.constexpr = tlx.amd_mfma_layout(4, [16, 16, 128], True, [2, 4])
    acc = tlx.require_layout(acc, acc_layout)
    result = tlx.require_layout(acc.to(tl.bfloat16), _STORE, late_address_compute=True)
    tlx.buffer_store(
        result,
        C,
        rows[:, None] * N + cols[None, :],
        cache=".cs" if STREAMING else "",
    )


@triton.jit
def _store_quadrant_f32(acc, workspace, rows, cols, N: tl.constexpr):
    acc_layout: tl.constexpr = tlx.amd_mfma_layout(4, [16, 16, 128], True, [2, 4])
    acc = tlx.require_layout(acc, acc_layout)
    result = tlx.require_layout(acc, _STORE_F32, late_address_compute=True)
    tlx.buffer_store(result, workspace, rows[:, None] * N + cols[None, :])


@triton.jit
def _store_quadrant_mi32(acc, C, rows, cols, N: tl.constexpr, STREAMING: tl.constexpr = False):
    acc_layout: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], True, [2, 2])
    acc = tlx.require_layout(acc, acc_layout)
    result = tlx.require_layout(acc.to(tl.bfloat16), _STORE_4W, late_address_compute=True)
    tlx.buffer_store(
        result,
        C,
        rows[:, None] * N + cols[None, :],
        cache=".cs" if STREAMING else "",
    )


@triton.jit
def _store_mi32_4x1(acc, C, rows, cols, N: tl.constexpr, N64: tl.constexpr, STREAMING: tl.constexpr = False):
    acc_layout: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], True, [4, 1])
    acc = tlx.require_layout(acc, acc_layout)
    if tlx.dtype_of(C) == tl.float32:
        store_layout: tl.constexpr = _STORE_MI32_4X1_64_F32 if N64 else _STORE_MI32_4X1_128_F32
    else:
        store_layout: tl.constexpr = _STORE_MI32_4X1_64 if N64 else _STORE_MI32_4X1_128
    result = tlx.require_layout(acc.to(tlx.dtype_of(C)), store_layout, late_address_compute=True)
    tlx.buffer_store(
        result,
        C,
        rows[:, None] * N + cols[None, :],
        cache=".cs" if STREAMING else "",
    )


@triton.jit
def _store_partial(acc, workspace, rows, cols, M: tl.constexpr, N: tl.constexpr, split_id):
    tl.store(workspace + (split_id * M + rows[:, None]) * N + cols[None, :], acc)


@triton.jit
def _reduce_split_k(workspace, C, M: tl.constexpr, N: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr, STREAMING: tl.constexpr = False):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for split in tl.static_range(SPLIT_K):
        acc += tl.load(workspace + split * M * N + offsets)
    tl.store(
        C + offsets,
        acc.to(tl.bfloat16),
        cache_modifier=".cs" if STREAMING else "",
    )


@triton.jit
def _wait_for_mxfp8_partial(locks, slot, ready_value: tl.constexpr):
    while tl.load(locks + slot, cache_modifier=".cv", volatile=True) != ready_value:
        pass


@triton.jit
def _distributed_mxfp8_split_k_reduce(
    workspace,
    locks,
    C,
    raw_pid,
    split_id,
    pm,
    pn,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    ready_value: tl.constexpr = 3
    tl.debug_barrier()
    tl.store(
        locks + raw_pid + split_id * GRID_MN,
        ready_value,
        cache_modifier=".wt",
    )
    for peer in tl.static_range(SPLIT_K):
        _wait_for_mxfp8_partial(
            locks,
            raw_pid + peer * GRID_MN,
            ready_value,
        )
    rows_per_contributor: tl.constexpr = 256 // SPLIT_K
    rows = (pm * 256 + split_id * rows_per_contributor + tl.arange(0, rows_per_contributor))
    cols = pn * 256 + tl.arange(0, 256)
    offsets = rows[:, None] * N + cols[None, :]
    acc = tl.zeros((rows_per_contributor, 256), tl.float32)
    for peer in tl.static_range(SPLIT_K):
        acc += tl.load(workspace + peer * M * N + offsets)
    tl.store(C + offsets, acc.to(tl.bfloat16), cache_modifier=".cs")


@triton.jit
def _gemm_skinny_mi32(A, B, C, SA, SB, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP_M: tl.constexpr,
                      STREAMING_STORE: tl.constexpr = False, CK_MAPPING: tl.constexpr = False):
    block: tl.constexpr = 128
    mma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], True, [2, 2])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    tile_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0],
         [16, 0]],
        [block, block],
    )
    a_s = tlx.local_alloc((block, block), tlx.dtype_of(A), 2, layout=tile_layout)
    b_s = tlx.local_alloc((block, block), tlx.dtype_of(B), 2, layout=tile_layout)
    sa_s = tlx.local_alloc((block, 1), tl.uint32, 2)
    sb_s = tlx.local_alloc((block, 1), tl.uint32, 2)

    grid_n: tl.constexpr = N // block
    pid = tl.program_id(0)
    if CK_MAPPING:
        grid_m: tl.constexpr = M // block
        grid_mn: tl.constexpr = grid_m * grid_n
        pids_per_xcd: tl.constexpr = grid_mn // 8
        pid = pid % 8 * pids_per_xcd + pid // 8
        raw_m = pid // grid_n
        raw_n = pid % grid_n
        m00 = raw_m // 4
        local = raw_n + raw_m % 4 * grid_n
        pm = local % 4 + m00 * 4
        pn = local // 4
    else:
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
    for step in tl.range(0, K // block - 2, num_stages=1, loop_unroll_factor=2):
        pipe_stage = step % 2
        next_pipe_stage = 1 - pipe_stage
        tl.debug_barrier()
        tlx.buffer_load_to_local(a_s[pipe_stage], a_base + step * block, data_off)
        tlx.buffer_load_to_local(b_s[pipe_stage], b_base + step * block, data_off)
        tlx.buffer_load_to_local(sa_s[pipe_stage], sa_base.to(tl.pointer_type(tl.uint32)) + step * 128, scale_word_off)
        tlx.buffer_load_to_local(sb_s[pipe_stage], sb_base.to(tl.pointer_type(tl.uint32)) + step * 128, scale_word_off)
        tlx.async_load_commit_group()
        acc = tlx.amd_scheduled_mfma_scaled(a, b, acc, sa, sb, "e4m3", "e4m3", accumulator_role="transient")
        tlx.async_load_wait_group(1)
        tl.debug_barrier()
        a = tlx.local_load(a_s[next_pipe_stage], layout=dot_a, relaxed=True)
        b = tlx.local_load(tlx.local_trans(b_s[next_pipe_stage]), layout=dot_b, relaxed=True)
        sa = _load_scale_words(sa_s[next_pipe_stage], _SCALE_A_MI32)
        sb = _load_scale_words(sb_s[next_pipe_stage], _SCALE_B_MI32)
    acc = tlx.amd_scheduled_mfma_scaled(a, b, acc, sa, sb, "e4m3", "e4m3", accumulator_role="transient")
    tlx.async_load_wait_group(0)
    tl.debug_barrier()
    a = tlx.local_load(a_s[1], layout=dot_a, relaxed=True)
    b = tlx.local_load(tlx.local_trans(b_s[1]), layout=dot_b, relaxed=True)
    sa = _load_scale_words(sa_s[1], _SCALE_A_MI32)
    sb = _load_scale_words(sb_s[1], _SCALE_B_MI32)
    acc = tlx.amd_scheduled_mfma_scaled(a, b, acc, sa, sb, "e4m3", "e4m3", accumulator_role="transient")

    out_rows = pm * block + tl.arange(0, block)
    out_cols = pn * block + tl.arange(0, block)
    _store_quadrant_mi32(acc, C, out_rows, out_cols, N, STREAMING_STORE)


@triton.jit
def _gemm_skinny(A, B, C, WORKSPACE, SA, SB, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, K_TILES: tl.constexpr,
                 SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr, CUBLAS_BLOCKED: tl.constexpr,
                 UNEVEN_SPLIT: tl.constexpr = False, LOOP_UNROLL_FACTOR: tl.constexpr = 1):
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
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0],
         [16, 0]],
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
def _gemm_mi32_256x192(A, B, C, WORKSPACE, SA, SB, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       K_TILES: tl.constexpr, SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr, HAS_TAIL: tl.constexpr,
                       STREAMING_STORE: tl.constexpr = False):
    mma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], True, [4, 1])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    tile_layout_128: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0],
         [16, 0]],
        [128, 128],
    )
    tile_layout_64: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        [64, 128],
    )
    a_top_s = tlx.local_alloc((128, 128), tlx.dtype_of(A), 2, layout=tile_layout_128)
    a_bot_s = tlx.local_alloc((128, 128), tlx.dtype_of(A), 2, layout=tile_layout_128)
    b_left_s = tlx.local_alloc((128, 128), tlx.dtype_of(B), 2, layout=tile_layout_128)
    b_right_s = tlx.local_alloc((64, 128), tlx.dtype_of(B), 2, layout=tile_layout_64)
    sa_top_s = tlx.local_alloc((128, 1), tl.uint32, 2)
    sa_bot_s = tlx.local_alloc((128, 1), tl.uint32, 2)
    sb_left_s = tlx.local_alloc((128, 1), tl.uint32, 2)
    sb_right_s = tlx.local_alloc((64, 1), tl.uint32, 2)

    grid_n: tl.constexpr = N // 192
    grid_mn: tl.constexpr = M // 256 * grid_n
    split_id = tl.program_id(0) // grid_mn
    pid = tl.program_id(0) % grid_mn
    pids_per_xcd: tl.constexpr = triton.cdiv(grid_mn, 8)
    tall_xcds: tl.constexpr = grid_mn % 8 or 8
    xcd = pid % 8
    local_pid = pid // 8
    pid = tl.where(
        xcd < tall_xcds,
        xcd * pids_per_xcd + local_pid,
        tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid,
    )
    first_m = pid // (GROUP_M * grid_n) * GROUP_M
    group_m = tl.minimum(M // 256 - first_m, GROUP_M)
    pm = first_m + pid % (GROUP_M * grid_n) % group_m
    pn = pid % (GROUP_M * grid_n) // group_m
    tl.static_assert(K_TILES % 2 == 0)
    extra_splits: tl.constexpr = (K // 128) % SPLIT_K
    split_start = (split_id * K_TILES + tl.minimum(split_id, extra_splits)) * 128
    scale_start = split_start // 128

    rows128 = tl.arange(0, 128)
    rows64 = tl.arange(0, 64)
    kk = tl.arange(0, 128)
    a_top_rows = pm * 256 + rows128
    a_bot_rows = a_top_rows + 128
    wide_offset = tl.where(pn % 2 == 0, 0, 64)
    narrow_offset = tl.where(pn % 2 == 0, 128, 0)
    b_left_rows = pn * 192 + wide_offset + rows128
    b_right_rows = pn * 192 + narrow_offset + rows64
    data_off_128 = tlx.require_layout(rows128[:, None] * K + kk[None, :], _COPY_DATA_4W_128)
    data_off_64 = rows64[:, None] * K + kk[None, :]
    sa_top_off = tlx.require_layout(_blocked_scale_word_offsets(a_top_rows, K), _COPY_SCALE_WORD_4W)
    sa_bot_off = tlx.require_layout(_blocked_scale_word_offsets(a_bot_rows, K), _COPY_SCALE_WORD_4W)
    sb_left_off = tlx.require_layout(_blocked_scale_word_offsets(b_left_rows, K), _COPY_SCALE_WORD_4W)
    sb_right_off = tlx.require_layout(_blocked_scale_word_offsets(b_right_rows, K), _COPY_SCALE_WORD_4W_64)
    a_top_base = A + pm * 256 * K + split_start
    a_bot_base = a_top_base + 128 * K
    b_left_base = B + (pn * 192 + wide_offset) * K + split_start
    b_right_base = B + (pn * 192 + narrow_offset) * K + split_start
    sa_words = SA.to(tl.pointer_type(tl.uint32))
    sb_words = SB.to(tl.pointer_type(tl.uint32))

    for slot in tl.static_range(2):
        data_delta = slot * 128
        scale_delta = (scale_start + slot) * 128
        tlx.buffer_load_to_local(b_left_s[slot], b_left_base + data_delta, data_off_128)
        tlx.buffer_load_to_local(sb_left_s[slot], sb_words + scale_delta, sb_left_off)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(a_top_s[slot], a_top_base + data_delta, data_off_128)
        tlx.buffer_load_to_local(sa_top_s[slot], sa_words + scale_delta, sa_top_off)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(a_bot_s[slot], a_bot_base + data_delta, data_off_128)
        tlx.buffer_load_to_local(sa_bot_s[slot], sa_words + scale_delta, sa_bot_off)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(b_right_s[slot], b_right_base + data_delta, data_off_64)
        tlx.buffer_load_to_local(sb_right_s[slot], sb_words + scale_delta, sb_right_off)
        tlx.async_load_commit_group()
    a_top_base += 256
    a_bot_base += 256
    b_left_base += 256
    b_right_base += 256
    scale_start += 2

    acc_tl = tl.tuple([tlx.zeros((128, 32), tl.float32, layout=mma) for _ in range(4)])
    acc_bl = tl.tuple([tlx.zeros((128, 32), tl.float32, layout=mma) for _ in range(4)])
    acc_tr = tl.tuple([tlx.zeros((128, 32), tl.float32, layout=mma) for _ in range(2)])
    acc_br = tl.tuple([tlx.zeros((128, 32), tl.float32, layout=mma) for _ in range(2)])
    tlx.async_load_wait_group(6)
    b_left, sb_l = _load_mi32_b_fragments(b_left_s[0], sb_left_s[0], 4, dot_b)
    a_top, sa_t = _load_mi32_a_fragments(a_top_s[0], sa_top_s[0], dot_a)

    for _ in tl.range(0, K_TILES - 2, 2, num_stages=1):
        for slot in tl.static_range(2):
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma_tl", scope="intra_wave", pair=0):
                acc_tl = _dot_scaled_mi32_fragments(a_top, sa_t, b_left, sb_l, acc_tl, 4)
            with tlx.warp_pipeline_stage("mem_ab", scope="intra_wave", pair=0):
                a_bot, sa_b = _load_mi32_a_fragments(a_bot_s[slot], sa_bot_s[slot], dot_a)
                tlx.buffer_load_to_local(b_left_s[slot], b_left_base + slot * 128, data_off_128)
                tlx.buffer_load_to_local(sb_left_s[slot], sb_words + (scale_start + slot) * 128, sb_left_off)
                tlx.async_load_commit_group()
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma_bl", scope="intra_wave", pair=1):
                acc_bl = _dot_scaled_mi32_fragments(a_bot, sa_b, b_left, sb_l, acc_bl, 4)
            with tlx.warp_pipeline_stage("mem_br", scope="intra_wave", pair=1):
                b_right, sb_r = _load_mi32_b_fragments(b_right_s[slot], sb_right_s[slot], 2, dot_b)
                tlx.buffer_load_to_local(a_top_s[slot], a_top_base + slot * 128, data_off_128)
                tlx.buffer_load_to_local(sa_top_s[slot], sa_words + (scale_start + slot) * 128, sa_top_off)
                tlx.async_load_commit_group()
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma_tr", scope="intra_wave", pair=2):
                acc_tr = _dot_scaled_mi32_fragments(a_top, sa_t, b_right, sb_r, acc_tr, 2)
            with tlx.warp_pipeline_stage("mem_bl", scope="intra_wave", pair=2):
                b_left, sb_l = _load_mi32_b_fragments(b_left_s[1 - slot], sb_left_s[1 - slot], 4, dot_b)
                tlx.buffer_load_to_local(a_bot_s[slot], a_bot_base + slot * 128, data_off_128)
                tlx.buffer_load_to_local(sa_bot_s[slot], sa_words + (scale_start + slot) * 128, sa_bot_off)
                tlx.async_load_commit_group()
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma_br", scope="intra_wave", pair=3):
                acc_br = _dot_scaled_mi32_fragments(a_bot, sa_b, b_right, sb_r, acc_br, 2)
            with tlx.warp_pipeline_stage("mem_at", scope="intra_wave", pair=3):
                a_top, sa_t = _load_mi32_a_fragments(a_top_s[1 - slot], sa_top_s[1 - slot], dot_a)
                tlx.buffer_load_to_local(b_right_s[slot], b_right_base + slot * 128, data_off_64)
                tlx.buffer_load_to_local(sb_right_s[slot], sb_words + (scale_start + slot) * 128, sb_right_off)
                tlx.async_load_commit_group()
                if slot == 1:
                    a_top_base += 256
                    a_bot_base += 256
                    b_left_base += 256
                    b_right_base += 256
                    scale_start += 2

    acc_tl = _dot_scaled_mi32_fragments(a_top, sa_t, b_left, sb_l, acc_tl, 4)
    tlx.async_load_wait_group(5)
    a_bot, sa_b = _load_mi32_a_fragments(a_bot_s[0], sa_bot_s[0], dot_a)
    acc_bl = _dot_scaled_mi32_fragments(a_bot, sa_b, b_left, sb_l, acc_bl, 4)
    tlx.async_load_wait_group(4)
    b_right, sb_r = _load_mi32_b_fragments(b_right_s[0], sb_right_s[0], 2, dot_b)
    acc_tr = _dot_scaled_mi32_fragments(a_top, sa_t, b_right, sb_r, acc_tr, 2)
    tlx.async_load_wait_group(3)
    b_left, sb_l = _load_mi32_b_fragments(b_left_s[1], sb_left_s[1], 4, dot_b)
    acc_br = _dot_scaled_mi32_fragments(a_bot, sa_b, b_right, sb_r, acc_br, 2)
    tlx.async_load_wait_group(2)
    a_top, sa_t = _load_mi32_a_fragments(a_top_s[1], sa_top_s[1], dot_a)
    acc_tl = _dot_scaled_mi32_fragments(a_top, sa_t, b_left, sb_l, acc_tl, 4)
    tlx.async_load_wait_group(1)
    a_bot, sa_b = _load_mi32_a_fragments(a_bot_s[1], sa_bot_s[1], dot_a)
    acc_bl = _dot_scaled_mi32_fragments(a_bot, sa_b, b_left, sb_l, acc_bl, 4)
    tlx.async_load_wait_group(0)
    b_right, sb_r = _load_mi32_b_fragments(b_right_s[1], sb_right_s[1], 2, dot_b)
    acc_tr = _dot_scaled_mi32_fragments(a_top, sa_t, b_right, sb_r, acc_tr, 2)
    acc_br = _dot_scaled_mi32_fragments(a_bot, sa_b, b_right, sb_r, acc_br, 2)

    if HAS_TAIL and (SPLIT_K == 1 or split_id < extra_splits):
        tail = split_start + K_TILES * 128
        scale_tile = tail // 128
        a_top_tail = A + pm * 256 * K + tail
        a_bot_tail = a_top_tail + 128 * K
        b_left_tail = B + (pn * 192 + wide_offset) * K + tail
        b_right_tail = B + (pn * 192 + narrow_offset) * K + tail
        tlx.buffer_load_to_local(b_left_s[0], b_left_tail, data_off_128)
        tlx.buffer_load_to_local(sb_left_s[0], sb_words + scale_tile * 128, sb_left_off)
        tlx.buffer_load_to_local(a_top_s[0], a_top_tail, data_off_128)
        tlx.buffer_load_to_local(sa_top_s[0], sa_words + scale_tile * 128, sa_top_off)
        tlx.buffer_load_to_local(a_bot_s[0], a_bot_tail, data_off_128)
        tlx.buffer_load_to_local(sa_bot_s[0], sa_words + scale_tile * 128, sa_bot_off)
        tlx.buffer_load_to_local(b_right_s[0], b_right_tail, data_off_64)
        tlx.buffer_load_to_local(sb_right_s[0], sb_words + scale_tile * 128, sb_right_off)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(0)
        b_left, sb_l = _load_mi32_b_fragments(b_left_s[0], sb_left_s[0], 4, dot_b)
        a_top, sa_t = _load_mi32_a_fragments(a_top_s[0], sa_top_s[0], dot_a)
        a_bot, sa_b = _load_mi32_a_fragments(a_bot_s[0], sa_bot_s[0], dot_a)
        b_right, sb_r = _load_mi32_b_fragments(b_right_s[0], sb_right_s[0], 2, dot_b)
        acc_tl = _dot_scaled_mi32_fragments(a_top, sa_t, b_left, sb_l, acc_tl, 4)
        acc_bl = _dot_scaled_mi32_fragments(a_bot, sa_b, b_left, sb_l, acc_bl, 4)
        acc_tr = _dot_scaled_mi32_fragments(a_top, sa_t, b_right, sb_r, acc_tr, 2)
        acc_br = _dot_scaled_mi32_fragments(a_bot, sa_b, b_right, sb_r, acc_br, 2)

    out_rows = pm * 256 + tl.arange(0, 128)
    out_cols_left = pn * 192 + wide_offset + tl.arange(0, 128)
    out_cols_right = pn * 192 + narrow_offset + tl.arange(0, 64)
    if SPLIT_K == 1:
        target = C
    else:
        target = WORKSPACE + split_id * M * N
    acc_tl = _join_mi32_fragments(acc_tl, 4)
    acc_bl = _join_mi32_fragments(acc_bl, 4)
    acc_tr = _join_mi32_fragments(acc_tr, 2)
    acc_br = _join_mi32_fragments(acc_br, 2)
    _store_mi32_4x1(acc_tl, target, out_rows, out_cols_left, N, False, STREAMING_STORE and SPLIT_K == 1)
    if K == 1024 and N == 36864:
        _store_mi32_4x1(acc_tr, target, out_rows, out_cols_right, N, True, STREAMING_STORE and SPLIT_K == 1)
        _store_mi32_4x1(acc_bl, target, out_rows + 128, out_cols_left, N, False, STREAMING_STORE and SPLIT_K == 1)
    else:
        _store_mi32_4x1(acc_bl, target, out_rows + 128, out_cols_left, N, False, STREAMING_STORE and SPLIT_K == 1)
        _store_mi32_4x1(acc_tr, target, out_rows, out_cols_right, N, True, STREAMING_STORE and SPLIT_K == 1)
    _store_mi32_4x1(acc_br, target, out_rows + 128, out_cols_right, N, True, STREAMING_STORE and SPLIT_K == 1)


@triton.jit
def _accumulate_interwave_tail(
    A,
    B,
    SA,
    SB,
    a_top_s,
    a_bot_s,
    b_left_s,
    b_right_s,
    sa_top_s,
    sa_bot_s,
    sb_left_s,
    sb_right_s,
    acc_tl,
    acc_bl,
    acc_tr,
    acc_br,
    pm,
    pn,
    split_start,
    scale_split,
    data_off,
    scale_off,
    K: tl.constexpr,
    TAIL: tl.constexpr,
    TAIL_SCALE: tl.constexpr,
    K_MAJOR: tl.constexpr,
):
    a_tail = A + pm * 256 * K + split_start + TAIL
    b_tail = B + pn * 256 * K + split_start + TAIL
    sa_tail = SA + pm * 256 * (K // 32) + scale_split + TAIL_SCALE
    sb_tail = SB + pn * 256 * (K // 32) + scale_split + TAIL_SCALE
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
    acc_tl = tlx.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
    a_bot = tlx.local_load(a_bot_s[0], relaxed=True)
    sa_b = _load_scale(sa_bot_s[0], _SCALE_A, K_MAJOR)
    acc_bl = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
    b_right = tlx.local_load(tlx.local_trans(b_right_s[0]), relaxed=True)
    sb_r = _load_scale(sb_right_s[0], _SCALE_B, K_MAJOR)
    acc_tr = tlx.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
    acc_br = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


@triton.jit
def _publish_interwave_k64(
    dst,
    src,
    stage: tl.constexpr,
    K: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    reduction = tl.arange(0, 64)
    outer = tl.arange(0, 128)
    if TRANSPOSED:
        offsets = reduction[:, None] + outer[None, :] * K
        offsets = tl.max_contiguous(tl.multiple_of(offsets, (16, 1)), (16, 1))
    else:
        offsets = outer[:, None] * K + reduction[None, :]
        offsets = tl.max_contiguous(tl.multiple_of(offsets, (1, 16)), (1, 16))
    for chunk in tl.static_range(2):
        tlx.buffer_load_to_local(
            tlx.local_view(dst[chunk], stage),
            src + chunk * 64,
            offsets,
            contiguity=16,
        )


@triton.jit
def _load_interwave_a_k64(src, stage: tl.constexpr, dot_a: tl.constexpr):
    low = tlx.local_load(tlx.local_view(src[0], stage), layout=dot_a, relaxed=True)
    high = tlx.local_load(tlx.local_view(src[1], stage), layout=dot_a, relaxed=True)
    return tlx.amd_concat((low, high), (128, 128))


@triton.jit
def _load_interwave_b_k64(src, stage: tl.constexpr, dot_b: tl.constexpr):
    low = tlx.local_load(tlx.local_view(src[0], stage), layout=dot_b, relaxed=True)
    high = tlx.local_load(tlx.local_view(src[1], stage), layout=dot_b, relaxed=True)
    return tlx.amd_concat((low, high), (128, 128))


@triton.jit
def _gemm_interwave(A, B, C, WORKSPACE, LOCKS, SA, SB, M: tl.constexpr, N: tl.constexpr, K_TILES: tl.constexpr,
                    K: tl.constexpr, SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr, CUBLAS_BLOCKED: tl.constexpr,
                    MODEL_XCD: tl.constexpr, K_MAJOR: tl.constexpr = False, UNEVEN_SPLIT: tl.constexpr = False,
                    LLVM_MFMA_SCHED: tl.constexpr = False, LOOP_UNROLL_FACTOR: tl.constexpr = 1,
                    STREAMING_STORE: tl.constexpr = False, WINDOWED_K64: tl.constexpr = False,
                    FUSED_SPLIT_REDUCE: tl.constexpr = False):
    mma: tl.constexpr = tlx.amd_mfma_layout(4, [16, 16, 128], True, [2, 4])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    # Keep K bytes contiguous for direct copies, with MFMA-friendly row bases.
    # A 32-byte pad per 1024 E4M3 bytes spreads row strides across LDS banks.
    tile_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0],
         [16, 0]],
        [128, 128],
    )
    if WINDOWED_K64:
        a_window_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [(1024, 32)],
            [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0],
             [8, 0]],
            [128, 64],
        )
        b_window_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [(1024, 32)],
            [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4],
             [0, 8]],
            [64, 128],
        )
        a_top_s = tl.tuple([tlx.local_alloc((128, 64), tlx.dtype_of(A), 2, layout=a_window_layout) for _ in range(2)])
        a_bot_s = tl.tuple([tlx.local_alloc((128, 64), tlx.dtype_of(A), 2, layout=a_window_layout) for _ in range(2)])
        b_left_s = tl.tuple([tlx.local_alloc((64, 128), tlx.dtype_of(B), 2, layout=b_window_layout) for _ in range(2)])
        b_right_s = tl.tuple([tlx.local_alloc((64, 128), tlx.dtype_of(B), 2, layout=b_window_layout) for _ in range(2)])
    else:
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
    program = tl.program_id(0)
    split_id = program // grid_mn
    raw_pid = program % grid_mn
    pid = raw_pid
    if FUSED_SPLIT_REDUCE:
        tl.store(LOCKS + program, 0, cache_modifier=".wt")
        tl.debug_barrier()
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
    extra_splits: tl.constexpr = (K // 128) % SPLIT_K
    split_start = ((split_id * K_TILES + tl.minimum(split_id, extra_splits)) * 128 if UNEVEN_SPLIT else split_id *
                   split_k)
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
    scale_split = ((split_start // 128) * 512 if CUBLAS_BLOCKED or K_MAJOR else split_start // 32)
    sa_base = SA + pm * 256 * (K // 32) + scale_split
    sa_bot_base = sa_base + 128 * (K // 32)
    sb_base = SB + pn * 256 * (K // 32) + scale_split
    sb_right_base = sb_base + 128 * (K // 32)
    acc_tl = tl.zeros((128, 128), tl.float32)
    acc_bl = tl.zeros((128, 128), tl.float32)
    acc_tr = tl.zeros((128, 128), tl.float32)
    acc_br = tl.zeros((128, 128), tl.float32)
    main_k_tiles: tl.constexpr = K_TILES - K_TILES % 2
    fused_publish: tl.constexpr = (M == 1024 and N == 36864 and K == 1024 and SPLIT_K == 1)

    for slot in tl.static_range(2):
        if WINDOWED_K64:
            _publish_interwave_k64(b_left_s, b_base + slot * 128, slot, K, True)
        else:
            tlx.buffer_load_to_local(b_left_s[slot], b_base + slot * 128, data_off, cache_modifier=".cg" if
                                     (M, N, K) == (2048, 6144, 67584) else "")
        tlx.buffer_load_to_local(sb_left_s[slot], sb_base + slot * scale_step, scale_off)
        if not fused_publish:
            tlx.async_load_commit_group()
        if WINDOWED_K64:
            _publish_interwave_k64(a_top_s, a_base + slot * 128, slot, K, False)
        else:
            tlx.buffer_load_to_local(a_top_s[slot], a_base + slot * 128, data_off)
        tlx.buffer_load_to_local(sa_top_s[slot], sa_base + slot * scale_step, scale_off)
        if not fused_publish:
            tlx.async_load_commit_group()
        if WINDOWED_K64:
            _publish_interwave_k64(a_bot_s, a_bot_base + slot * 128, slot, K, False)
        else:
            tlx.buffer_load_to_local(a_bot_s[slot], a_bot_base + slot * 128, data_off)
        tlx.buffer_load_to_local(sa_bot_s[slot], sa_bot_base + slot * scale_step, scale_off)
        if not fused_publish:
            tlx.async_load_commit_group()
        if WINDOWED_K64:
            _publish_interwave_k64(b_right_s, b_right_base + slot * 128, slot, K, True)
        else:
            tlx.buffer_load_to_local(b_right_s[slot], b_right_base + slot * 128, data_off, cache_modifier=".cg" if
                                     (M, N, K) == (2048, 6144, 67584) else "")
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
    if fused_publish:
        prime_wait = tlx.async_load_wait_group(1)
        b_left = tlx.local_load(tlx.local_trans(b_left_s[0]), token=prime_wait, relaxed=True)
        sb_l = _load_scale(sb_left_s[0], _SCALE_B, K_MAJOR)
        a_top = tlx.local_load(a_top_s[0], relaxed=True)
        sa_t = _load_scale(sa_top_s[0], _SCALE_A, K_MAJOR)
        for _ in tl.range(0, main_k_tiles - 2, 2, num_stages=1):
            for slot in tl.static_range(2):
                with tlx.warp_pipeline_stage("fused_read_bottom", scope="intra_wave", pair=0):
                    a_bot = tlx.local_load(a_bot_s[slot], relaxed=True)
                    sa_b = _load_scale(sa_bot_s[slot], _SCALE_A, K_MAJOR)
                with tlx.warp_pipeline_stage("fused_compute_top_left", scope="intra_wave", pair=0):
                    acc_tl = tlx.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
                with tlx.warp_pipeline_stage("fused_read_right", scope="intra_wave", pair=1):
                    b_right = tlx.local_load(tlx.local_trans(b_right_s[slot]), relaxed=True)
                    sb_r = _load_scale(sb_right_s[slot], _SCALE_B, K_MAJOR)
                with tlx.warp_pipeline_stage("fused_compute_bottom_left", scope="intra_wave", pair=1):
                    acc_bl = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
                next_wait = tlx.async_load_wait_group(0)
                with tlx.warp_pipeline_stage("fused_publish_and_read_next", scope="intra_wave", pair=2):
                    next_b_left = tlx.local_load(tlx.local_trans(b_left_s[1 - slot]), token=next_wait, relaxed=True)
                    next_sb_l = _load_scale(sb_left_s[1 - slot], _SCALE_B, K_MAJOR)
                    next_a_top = tlx.local_load(a_top_s[1 - slot], relaxed=True)
                    next_sa_t = _load_scale(sa_top_s[1 - slot], _SCALE_A, K_MAJOR)
                    tlx.buffer_load_to_local(b_left_s[slot], b_base + slot * 128, data_off)
                    tlx.buffer_load_to_local(sb_left_s[slot], sb_base + slot * scale_step, scale_off)
                    tlx.buffer_load_to_local(a_top_s[slot], a_base + slot * 128, data_off)
                    tlx.buffer_load_to_local(sa_top_s[slot], sa_base + slot * scale_step, scale_off)
                    tlx.buffer_load_to_local(a_bot_s[slot], a_bot_base + slot * 128, data_off)
                    tlx.buffer_load_to_local(sa_bot_s[slot], sa_bot_base + slot * scale_step, scale_off)
                    tlx.buffer_load_to_local(b_right_s[slot], b_right_base + slot * 128, data_off)
                    tlx.buffer_load_to_local(sb_right_s[slot], sb_right_base + slot * scale_step, scale_off)
                    tlx.async_load_commit_group()
                with tlx.warp_pipeline_stage("fused_compute_right", scope="intra_wave", pair=2):
                    acc_tr = tlx.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
                    acc_br = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
                b_left = next_b_left
                sb_l = next_sb_l
                a_top = next_a_top
                sa_t = next_sa_t
                if slot == 1:
                    a_base += 256
                    a_bot_base += 256
                    b_base += 256
                    b_right_base += 256
                    sa_base += 2 * scale_step
                    sa_bot_base += 2 * scale_step
                    sb_base += 2 * scale_step
                    sb_right_base += 2 * scale_step

        cm = pm * 256 + tlx.rematerialized_range(0, 128, 10, placement=K_TILES)
        cn = pn * 256 + tlx.rematerialized_range(0, 128, 11, placement=K_TILES)
        with tlx.warp_pipeline_stage("fused_tail_read_bottom", scope="intra_wave", pair=3):
            a_bot = tlx.local_load(a_bot_s[0], relaxed=True)
            sa_b = _load_scale(sa_bot_s[0], _SCALE_A, K_MAJOR)
        with tlx.warp_pipeline_stage("fused_tail_compute_top_left", scope="intra_wave", pair=3):
            acc_tl = tlx.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
        with tlx.warp_pipeline_stage("fused_tail_read_right", scope="intra_wave", pair=4):
            b_right = tlx.local_load(tlx.local_trans(b_right_s[0]), relaxed=True)
            sb_r = _load_scale(sb_right_s[0], _SCALE_B, K_MAJOR)
        with tlx.warp_pipeline_stage("fused_tail_compute_bottom_left", scope="intra_wave", pair=4):
            acc_bl = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
        final_wait = tlx.async_load_wait_group(0)
        with tlx.warp_pipeline_stage("fused_tail_read_final", scope="intra_wave", pair=5):
            final_b_left = tlx.local_load(tlx.local_trans(b_left_s[1]), token=final_wait, relaxed=True)
            final_sb_l = _load_scale(sb_left_s[1], _SCALE_B, K_MAJOR)
            final_a_top = tlx.local_load(a_top_s[1], relaxed=True)
            final_sa_t = _load_scale(sa_top_s[1], _SCALE_A, K_MAJOR)
        with tlx.warp_pipeline_stage("fused_tail_compute_right", scope="intra_wave", pair=5):
            acc_tr = tlx.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
            acc_br = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)

        acc_tl = tlx.dot_scaled(final_a_top, final_sa_t, "e4m3", final_b_left, final_sb_l, "e4m3", acc_tl)
        _store_quadrant(acc_tl, C, cm, cn, N, STREAMING_STORE)
        final_a_bot = tlx.local_load(a_bot_s[1], relaxed=True)
        final_sa_b = _load_scale(sa_bot_s[1], _SCALE_A, K_MAJOR)
        acc_bl = tlx.dot_scaled(final_a_bot, final_sa_b, "e4m3", final_b_left, final_sb_l, "e4m3", acc_bl)
        _store_quadrant(acc_bl, C, cm + 128, cn, N, STREAMING_STORE)
        final_b_right = tlx.local_load(tlx.local_trans(b_right_s[1]), relaxed=True)
        final_sb_r = _load_scale(sb_right_s[1], _SCALE_B, K_MAJOR)
        acc_tr = tlx.dot_scaled(final_a_top, final_sa_t, "e4m3", final_b_right, final_sb_r, "e4m3", acc_tr)
        _store_quadrant(acc_tr, C, cm, cn + 128, N, STREAMING_STORE)
        acc_br = tlx.dot_scaled(final_a_bot, final_sa_b, "e4m3", final_b_right, final_sb_r, "e4m3", acc_br)
        _store_quadrant(acc_br, C, cm + 128, cn + 128, N, STREAMING_STORE)
        return
    # Groups count commits: 2 slots x 4 (data + scale) = 8; wait(6) completes slot 0's B-left/A-top.
    tlx.async_load_wait_group(6)
    if WINDOWED_K64:
        b_left = _load_interwave_b_k64(b_left_s, 0, dot_b)
    else:
        b_left = tlx.local_load(tlx.local_trans(b_left_s[0]), relaxed=True)
    sb_l = _load_scale(sb_left_s[0], _SCALE_B, K_MAJOR)
    if WINDOWED_K64:
        a_top = _load_interwave_a_k64(a_top_s, 0, dot_a)
    else:
        a_top = tlx.local_load(a_top_s[0], relaxed=True)
    sa_t = _load_scale(sa_top_s[0], _SCALE_A, K_MAJOR)

    # Each refill adds one group; wait(5) completes the next data/scale group before its LDS read.
    for k in tl.range(0, main_k_tiles - 2, 2, num_stages=1, loop_unroll_factor=LOOP_UNROLL_FACTOR):
        for slot in tl.static_range(2):
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tl = tlx.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                if WINDOWED_K64:
                    a_bot = _load_interwave_a_k64(a_bot_s, slot, dot_a)
                else:
                    a_bot = tlx.local_load(a_bot_s[slot], relaxed=True)
                sa_b = _load_scale(sa_bot_s[slot], _SCALE_A, K_MAJOR)
                if not LLVM_MFMA_SCHED:
                    tlx.amd_sched_barrier(0)
                if WINDOWED_K64:
                    _publish_interwave_k64(b_left_s, b_base + slot * 128, slot, K, True)
                else:
                    tlx.buffer_load_to_local(b_left_s[slot], b_base + slot * 128, data_off, cache_modifier=".cg" if
                                             (M, N, K) == (2048, 6144, 67584) else "")
                tlx.buffer_load_to_local(sb_left_s[slot], sb_base + slot * scale_step, scale_off)
                tlx.async_load_commit_group()
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_bl = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                if WINDOWED_K64:
                    b_right = _load_interwave_b_k64(b_right_s, slot, dot_b)
                else:
                    b_right = tlx.local_load(tlx.local_trans(b_right_s[slot]), relaxed=True)
                sb_r = _load_scale(sb_right_s[slot], _SCALE_B, K_MAJOR)
                if not LLVM_MFMA_SCHED:
                    tlx.amd_sched_barrier(0)
                if WINDOWED_K64:
                    _publish_interwave_k64(a_top_s, a_base + slot * 128, slot, K, False)
                else:
                    tlx.buffer_load_to_local(a_top_s[slot], a_base + slot * 128, data_off)
                tlx.buffer_load_to_local(sa_top_s[slot], sa_base + slot * scale_step, scale_off)
                tlx.async_load_commit_group()
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tr = tlx.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
            with tlx.warp_pipeline_stage("mem", priority=1):
                if WINDOWED_K64:
                    b_left = _load_interwave_b_k64(b_left_s, 1 - slot, dot_b)
                else:
                    b_left = tlx.local_load(tlx.local_trans(b_left_s[1 - slot]), relaxed=True)
                sb_l = _load_scale(sb_left_s[1 - slot], _SCALE_B, K_MAJOR)
                if not LLVM_MFMA_SCHED:
                    tlx.amd_sched_barrier(0)
                if WINDOWED_K64:
                    _publish_interwave_k64(a_bot_s, a_bot_base + slot * 128, slot, K, False)
                else:
                    tlx.buffer_load_to_local(a_bot_s[slot], a_bot_base + slot * 128, data_off)
                tlx.buffer_load_to_local(sa_bot_s[slot], sa_bot_base + slot * scale_step, scale_off)
                tlx.async_load_commit_group()
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_br = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
            with tlx.warp_pipeline_stage("mem", priority=1):
                if WINDOWED_K64:
                    a_top = _load_interwave_a_k64(a_top_s, 1 - slot, dot_a)
                else:
                    a_top = tlx.local_load(a_top_s[1 - slot], relaxed=True)
                sa_t = _load_scale(sa_top_s[1 - slot], _SCALE_A, K_MAJOR)
                if not LLVM_MFMA_SCHED:
                    tlx.amd_sched_barrier(0)
                if WINDOWED_K64:
                    _publish_interwave_k64(b_right_s, b_right_base + slot * 128, slot, K, True)
                else:
                    tlx.buffer_load_to_local(b_right_s[slot], b_right_base + slot * 128, data_off,
                                             cache_modifier=".cg" if (M, N, K) == (2048, 6144, 67584) else "")
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
    if WINDOWED_K64:
        a_bot = _load_interwave_a_k64(a_bot_s, 0, dot_a)
    else:
        a_bot = tlx.local_load(a_bot_s[0], relaxed=True)
    sa_b = _load_scale(sa_bot_s[0], _SCALE_A, K_MAJOR)
    acc_bl = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
    tlx.async_load_wait_group(4)
    if WINDOWED_K64:
        b_right = _load_interwave_b_k64(b_right_s, 0, dot_b)
    else:
        b_right = tlx.local_load(tlx.local_trans(b_right_s[0]), relaxed=True)
    sb_r = _load_scale(sb_right_s[0], _SCALE_B, K_MAJOR)
    acc_tr = tlx.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
    tlx.async_load_wait_group(3)
    if WINDOWED_K64:
        b_left = _load_interwave_b_k64(b_left_s, 1, dot_b)
    else:
        b_left = tlx.local_load(tlx.local_trans(b_left_s[1]), relaxed=True)
    sb_l = _load_scale(sb_left_s[1], _SCALE_B, K_MAJOR)
    acc_br = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
    tlx.async_load_wait_group(2)
    if WINDOWED_K64:
        a_top = _load_interwave_a_k64(a_top_s, 1, dot_a)
    else:
        a_top = tlx.local_load(a_top_s[1], relaxed=True)
    sa_t = _load_scale(sa_top_s[1], _SCALE_A, K_MAJOR)
    acc_tl = tlx.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
    if M == 1024 and N == 36864 and K == 1024 and SPLIT_K == 1:
        early_cm = pm * 256 + tlx.rematerialized_range(0, 128, 10, placement=K_TILES)
        early_cn = pn * 256 + tlx.rematerialized_range(0, 128, 11, placement=K_TILES)
        _store_quadrant(acc_tl, C, early_cm, early_cn, N, STREAMING_STORE)
    tlx.async_load_wait_group(1)
    if WINDOWED_K64:
        a_bot = _load_interwave_a_k64(a_bot_s, 1, dot_a)
    else:
        a_bot = tlx.local_load(a_bot_s[1], relaxed=True)
    sa_b = _load_scale(sa_bot_s[1], _SCALE_A, K_MAJOR)
    acc_bl = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
    if M == 1024 and N == 36864 and K == 1024 and SPLIT_K == 1:
        _store_quadrant(acc_bl, C, early_cm + 128, early_cn, N, STREAMING_STORE)
    tlx.async_load_wait_group(0)
    if WINDOWED_K64:
        b_right = _load_interwave_b_k64(b_right_s, 1, dot_b)
    else:
        b_right = tlx.local_load(tlx.local_trans(b_right_s[1]), relaxed=True)
    sb_r = _load_scale(sb_right_s[1], _SCALE_B, K_MAJOR)
    acc_tr = tlx.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
    if M == 1024 and N == 36864 and K == 1024 and SPLIT_K == 1:
        _store_quadrant(acc_tr, C, early_cm, early_cn + 128, N, STREAMING_STORE)
    acc_br = tlx.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
    if M == 1024 and N == 36864 and K == 1024 and SPLIT_K == 1:
        _store_quadrant(acc_br, C, early_cm + 128, early_cn + 128, N, STREAMING_STORE)
        return
    if K_TILES % 2:
        acc_tl, acc_bl, acc_tr, acc_br = _accumulate_interwave_tail(A, B, SA, SB, a_top_s, a_bot_s, b_left_s, b_right_s,
                                                                    sa_top_s, sa_bot_s, sb_left_s, sb_right_s, acc_tl,
                                                                    acc_bl, acc_tr, acc_br, pm, pn, split_start,
                                                                    scale_split, data_off, scale_off, K,
                                                                    (K_TILES - 1) * 128, (K_TILES - 1) * scale_step,
                                                                    K_MAJOR)
    if UNEVEN_SPLIT:
        if split_id < extra_splits:
            acc_tl, acc_bl, acc_tr, acc_br = _accumulate_interwave_tail(A, B, SA, SB, a_top_s, a_bot_s, b_left_s,
                                                                        b_right_s, sa_top_s, sa_bot_s, sb_left_s,
                                                                        sb_right_s, acc_tl, acc_bl, acc_tr, acc_br, pm,
                                                                        pn, split_start, scale_split, data_off,
                                                                        scale_off, K, K_TILES * 128,
                                                                        K_TILES * scale_step, K_MAJOR)
    cm = pm * 256 + tlx.rematerialized_range(0, 128, 10, placement=K_TILES)
    cn = pn * 256 + tlx.rematerialized_range(0, 128, 11, placement=K_TILES)
    if SPLIT_K == 1:
        _store_quadrant(acc_tl, C, cm, cn, N, STREAMING_STORE)
        if ((M == 1024 and (N == 64512 or N == 153600 or N == 159744) and K == 1024)
                or (M == 2048 and N == 24576 and K == 6144)):
            _store_quadrant(acc_tr, C, cm, cn + 128, N, STREAMING_STORE)
            _store_quadrant(acc_bl, C, cm + 128, cn, N, STREAMING_STORE)
        else:
            _store_quadrant(acc_bl, C, cm + 128, cn, N, STREAMING_STORE)
            _store_quadrant(acc_tr, C, cm, cn + 128, N, STREAMING_STORE)
        _store_quadrant(acc_br, C, cm + 128, cn + 128, N, STREAMING_STORE)
    else:
        partial = WORKSPACE + split_id * M * N
        _store_quadrant_f32(acc_tl, partial, cm, cn, N)
        _store_quadrant_f32(acc_bl, partial, cm + 128, cn, N)
        _store_quadrant_f32(acc_tr, partial, cm, cn + 128, N)
        _store_quadrant_f32(acc_br, partial, cm + 128, cn + 128, N)
        if FUSED_SPLIT_REDUCE:
            _distributed_mxfp8_split_k_reduce(
                WORKSPACE,
                LOCKS,
                C,
                raw_pid,
                split_id,
                pm,
                pn,
                M,
                N,
                SPLIT_K,
                grid_mn,
            )


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


@triton.jit
def _gemm_m_masked(A, B, C, SA, SB, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, CUBLAS_BLOCKED: tl.constexpr):
    block: tl.constexpr = 128
    grid_n: tl.constexpr = triton.cdiv(N, block)
    grid_m: tl.constexpr = triton.cdiv(M, block)
    pid = tl.program_id(0)
    first_m = pid // (4 * grid_n) * 4
    group_m = tl.minimum(grid_m - first_m, 4)
    pm = first_m + pid % (4 * grid_n) % group_m
    pn = pid % (4 * grid_n) // group_m
    rows = pm * block + tl.arange(0, block)
    cols = pn * block + tl.arange(0, block)
    kk = tl.arange(0, block)
    groups = tl.arange(0, block // 32)
    row_mask = rows < M
    col_mask = cols < N
    a_off = rows[:, None] * K + kk[None, :]
    b_off = kk[:, None] + cols[None, :] * K
    if CUBLAS_BLOCKED:
        sa_off = ((rows[:, None] // 128) * (K // 128) * 512 + (rows[:, None] % 32) * 16 +
                  (rows[:, None] % 128 // 32) * 4 + groups[None, :])
        sb_off = ((cols[:, None] // 128) * (K // 128) * 512 + (cols[:, None] % 32) * 16 +
                  (cols[:, None] % 128 // 32) * 4 + groups[None, :])
        scale_step: tl.constexpr = 512
    else:
        sa_off = rows[:, None] * (K // 32) + groups[None, :]
        sb_off = cols[:, None] * (K // 32) + groups[None, :]
        scale_step: tl.constexpr = block // 32
    acc = tl.zeros((block, block), tl.float32)
    for kt in tl.range(K // block, num_stages=1):
        a = tlx.buffer_load(A + kt * block, a_off, mask=row_mask[:, None], other=0.0)
        b = tlx.buffer_load(
            B + kt * block,
            b_off,
            mask=col_mask[None, :],
            other=0.0,
        )
        sa = tlx.buffer_load(SA + kt * scale_step, sa_off, mask=row_mask[:, None], other=127)
        sb = tlx.buffer_load(
            SB + kt * scale_step,
            sb_off,
            mask=col_mask[:, None],
            other=127,
        )
        acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    offsets = rows[:, None] * N + cols[None, :]
    tlx.buffer_store(
        acc.to(tl.bfloat16),
        C,
        offsets,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _reduce_split_k_masked(workspace, C, M: tl.constexpr, N: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr,
                           BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    mask = rows[:, None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for split in tl.static_range(SPLIT_K):
        acc += tl.load(
            workspace + split * M * N + offsets,
            mask=mask,
            other=0.0,
        )
    tl.store(C + offsets, acc.to(tl.bfloat16), mask=mask)


def _pack_k_major_scales(scale):
    """Return a contiguous K-major copy for 32-bit LDS scale reads.

    Uses one extra byte per scale while source and packed tensors coexist at
    launch. The copy is included in end-to-end GEMM timing.
    """
    rows, groups = scale.shape
    return (scale.view(torch.uint8).view(rows // 128, 4, 32, groups // 4, 4).permute(0, 3, 2, 4,
                                                                                     1).contiguous().view(rows, groups))


@triton.jit
def _pack_cublas_blocked_scales_kernel(A, B, OA, OB, A_ROWS: tl.constexpr, B_ROWS: tl.constexpr, GROUPS: tl.constexpr,
                                       A_PAD: tl.constexpr, B_PAD: tl.constexpr, ATOMS_PER_CTA: tl.constexpr):
    panels: tl.constexpr = GROUPS // 4
    panel_pairs: tl.constexpr = triton.cdiv(panels, 2)
    a_atoms: tl.constexpr = A_PAD // 128
    a_programs: tl.constexpr = a_atoms * panel_pairs
    word_id = tl.arange(0, 128 * ATOMS_PER_CTA)
    pid = tl.program_id(0) * ATOMS_PER_CTA + word_id // 128
    word = word_id % 128
    lane_row = word // 4
    row_group = word % 4

    a_pid = pid
    a_atom = a_pid // panel_pairs
    a_panel = a_pid % panel_pairs * 2
    a_row = a_atom * 128 + row_group * 32 + lane_row
    a_valid = (pid < a_programs) & (a_row < A_ROWS)
    a_pointer = A.to(tl.pointer_type(tl.uint32)) + a_row * panels + a_panel
    a_has_pair = a_panel + 1 < panels
    a_pair = tl.load(
        a_pointer.to(tl.pointer_type(tl.uint64)),
        mask=a_valid & a_has_pair,
        other=0x7F7F7F7F7F7F7F7F,
    )
    a_tail = tl.load(
        a_pointer,
        mask=a_valid & ~a_has_pair,
        other=0x7F7F7F7F,
    )
    a_value = tl.where(a_has_pair, a_pair, a_tail.to(tl.uint64))
    a_output = a_atom * panels + a_panel
    tl.store(
        OA.to(tl.pointer_type(tl.uint32)) + a_output * 128 + word,
        a_value.to(tl.uint32),
        mask=pid < a_programs,
    )
    tl.store(
        OA.to(tl.pointer_type(tl.uint32)) + (a_output + 1) * 128 + word,
        (a_value >> 32).to(tl.uint32),
        mask=(pid < a_programs) & a_has_pair,
    )

    b_pid = pid - a_programs
    b_atom = b_pid // panel_pairs
    b_panel = b_pid % panel_pairs * 2
    b_row = b_atom * 128 + row_group * 32 + lane_row
    b_programs: tl.constexpr = B_PAD // 128 * panel_pairs
    b_valid = (b_pid >= 0) & (b_pid < b_programs) & (b_row < B_ROWS)
    b_pointer = B.to(tl.pointer_type(tl.uint32)) + b_row * panels + b_panel
    b_has_pair = b_panel + 1 < panels
    b_pair = tl.load(
        b_pointer.to(tl.pointer_type(tl.uint64)),
        mask=b_valid & b_has_pair,
        other=0x7F7F7F7F7F7F7F7F,
    )
    b_tail = tl.load(
        b_pointer,
        mask=b_valid & ~b_has_pair,
        other=0x7F7F7F7F,
    )
    b_value = tl.where(b_has_pair, b_pair, b_tail.to(tl.uint64))
    b_output = b_atom * panels + b_panel
    tl.store(
        OB.to(tl.pointer_type(tl.uint32)) + b_output * 128 + word,
        b_value.to(tl.uint32),
        mask=(b_pid >= 0) & (b_pid < b_programs),
    )
    tl.store(
        OB.to(tl.pointer_type(tl.uint32)) + (b_output + 1) * 128 + word,
        (b_value >> 32).to(tl.uint32),
        mask=(b_pid >= 0) & (b_pid < b_programs) & b_has_pair,
    )


def _pack_cublas_blocked_scale_pair(a_scale, b_scale):
    a_rows, groups = a_scale.shape
    b_rows = b_scale.shape[0]
    a_pad = (a_rows + 127) // 128 * 128
    b_pad = (b_rows + 127) // 128 * 128
    a_out = torch.empty(a_pad * groups, dtype=torch.uint8, device=a_scale.device)
    b_out = torch.empty(b_pad * groups, dtype=torch.uint8, device=b_scale.device)
    panel_pairs = (groups + 7) // 8
    atoms_per_cta = 16
    _pack_cublas_blocked_scales_kernel[(triton.cdiv((a_pad + b_pad) // 128 * panel_pairs, atoms_per_cta), )](
        a_scale.view(torch.uint8),
        b_scale.view(torch.uint8),
        a_out,
        b_out,
        a_rows,
        b_rows,
        groups,
        a_pad,
        b_pad,
        atoms_per_cta,
        num_warps=4,
    )
    return (
        a_out.view(torch.float8_e8m0fnu).reshape(-1, 512),
        b_out.view(torch.float8_e8m0fnu).reshape(-1, 512),
    )


def _model_split_k(m, n, k, cu_count, block):
    if (m, n, k, block) == (1024, 1024, 159744, 256):
        return 13
    if (m, n, k, block) == (2048, 6144, 39936, 256):
        return 1
    if (m, n, k, block) == (2048, 6144, 22656, 256):
        return 1
    if (m, n, k, block) == (2048, 6144, 159744, 256):
        return 1
    if (m, n, k, block) == (2048, 6144, 67584, 256):
        return 1
    if (m, n, k, block) == (1024, 8192, 3072, 256):
        return 1
    if (m, n, k, block) == (1024, 3072, 15360, 256):
        return 5
    if (m, n, k, block) == (2048, 3072, 15360, 256):
        return 2
    if (m, n, k, block) == (1024, 1024, 24576, 256):
        return 16
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


def _allocate_split_workspace(a, split_k, m, n):
    assert split_k > 1
    return torch.empty((split_k * m, n), dtype=torch.float32, device=a.device)


def _supports_native_tiles(a, a_scale, b, b_scale, out, sf_layout):
    return (sf_layout in {"cublas_blocked", "mfma_preshuffled"} and a.shape in {(1024, 24576), (2048, 24576)}
            and b.shape == (1024, 24576) and all(tensor.data_ptr() % 16 == 0 for tensor in (a, b, out))
            and all(tensor.data_ptr() % 8 == 0 for tensor in (a_scale, b_scale)))


def _launch(a, a_scale, b, b_scale, out, *, sf_layout):
    m, k = a.shape
    n = b.shape[0]
    with torch.cuda.device(a.device):
        if _supports_native_tiles(a, a_scale, b, b_scale, out, sf_layout):
            split_k = 8 if m == 1024 else 4
            native_kernel = _gemm_native_split8 if m == 1024 else _gemm_native_split4_wide
            workspace = _allocate_split_workspace(a, split_k, m, n)
            scales_shuffled = sf_layout == "mfma_preshuffled"
            native_kernel[(m // 128 * (n // 256) * split_k, )](
                a,
                b,
                a_scale.view(torch.uint8),
                b_scale.view(torch.uint8),
                out,
                workspace,
                m,
                n,
                k,
                128,
                256,
                2,
                split_k,
                A_SHUFFLED=scales_shuffled,
                B_SHUFFLED=scales_shuffled,
                num_warps=4,
                num_stages=1,
                matrix_instr_nonkdim=32,
                enable_sched_group_barrier_scheduler=True,
                sched_group_barrier_mfma_per_dwordx4=4,
                disable_unclustered_high_rp_reschedule=True,
                reverse_local_assignment=m == 2048,
                sink_insts_to_avoid_spills=True,
                mfma_vgpr_form=False,
            )
            reduce_n = 256 if m == 2048 else 128
            _reduce_split_k[(m // 32, n // reduce_n)](workspace, out, m, n, split_k, 32, reduce_n, num_warps=8)
            return out
        if ((m, n, k) in {(1024, 1024, 24576), (2048, 1024, 24576)} and sf_layout in {"natural", "cublas_blocked"}
                and (any(tensor.data_ptr() % 16 for tensor in (a, b, out)) or any(tensor.data_ptr() % 8
                                                                                  for tensor in (a_scale, b_scale)))):
            _gemm_m_masked[(m // 128 * (n // 128), )](
                a,
                b,
                out,
                a_scale.view(torch.uint8),
                b_scale.view(torch.uint8),
                m,
                n,
                k,
                sf_layout == "cublas_blocked",
                num_warps=4,
                num_stages=1,
                matrix_instr_nonkdim=16,
                disable_unclustered_high_rp_reschedule=True,
                llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
            )
            return out
        if ((m, n, k) == (1024, 1024, 159744) and sf_layout == "cublas_blocked"
                and all(tensor.data_ptr() % 16 == 0 for tensor in (a, b, out))
                and all(tensor.data_ptr() % 8 == 0 for tensor in (a_scale, b_scale))):
            split_k = 12
            workspace = _allocate_split_workspace(a, split_k, m, n)
            _gemm_native_256x256_split16[(m // 256 * (n // 256) * split_k, )](
                a,
                b,
                a_scale.view(torch.uint8),
                b_scale.view(torch.uint8),
                out,
                workspace,
                m,
                n,
                k,
                256,
                256,
                2,
                split_k,
                BALANCED_G2L_CADENCE=True,
                num_warps=4,
                num_stages=1,
                matrix_instr_nonkdim=32,
                enable_sched_group_barrier_scheduler=False,
                sched_group_barrier_mfma_per_dwordx4=4,
                disable_unclustered_high_rp_reschedule=True,
                reverse_local_assignment=True,
                sink_insts_to_avoid_spills=True,
                regclass_priority_trumps_globalness=True,
                mfma_vgpr_form=False,
            )
            _reduce_split_k[(m // 32, n // 128)](workspace, out, m, n, split_k, 32, 128, num_warps=8)
            return out
        if ((m, n, k) in {
            (1024, 32768, 6144),
            (2048, 32768, 6144),
            (2048, 8192, 3072),
            (2048, 15360, 3584),
            (2048, 23040, 1792),
            (2048, 22464, 1024),
            (2048, 14976, 1024),
        } and (sf_layout == "cublas_blocked" or
               ((m, n, k) == (2048, 32768, 6144) and sf_layout == "mfma_preshuffled"))):
            _gemm_native_256x256_split16[(m // 256 * triton.cdiv(n, 256), )](
                a,
                b,
                a_scale.view(torch.uint8),
                b_scale.view(torch.uint8),
                out,
                out,
                m,
                n,
                k,
                256,
                256,
                2,
                1,
                SCALES_SHUFFLED=sf_layout == "mfma_preshuffled",
                BALANCED_G2L_CADENCE=(m, n, k) in {
                    (1024, 32768, 6144),
                    (2048, 32768, 6144),
                    (2048, 8192, 3072),
                    (2048, 15360, 3584),
                },
                STREAMING_STORE=(m, n, k) in {
                    (2048, 23040, 1792),
                    (2048, 22464, 1024),
                },
                num_warps=4,
                num_stages=1,
                matrix_instr_nonkdim=32,
                enable_sched_group_barrier_scheduler=(m, n, k) != (2048, 32768, 6144),
                sched_group_barrier_mfma_per_dwordx4=4,
                disable_unclustered_high_rp_reschedule=True,
                reverse_local_assignment=True,
                sink_insts_to_avoid_spills=True,
                regclass_priority_trumps_globalness=True,
                mfma_vgpr_form=False,
            )
            return out
        if ((m, n, k) == (2048, 3584, 25600) and sf_layout == "cublas_blocked"):
            split_k = 2
            workspace = _allocate_split_workspace(a, split_k, m, n)
            _gemm_native_256x256_split16[(m // 256 * (n // 256) * split_k, )](
                a,
                b,
                a_scale.view(torch.uint8),
                b_scale.view(torch.uint8),
                out,
                workspace,
                m,
                n,
                k,
                256,
                256,
                2,
                split_k,
                BALANCED_G2L_CADENCE=True,
                num_warps=4,
                num_stages=1,
                matrix_instr_nonkdim=32,
                enable_sched_group_barrier_scheduler=True,
                sched_group_barrier_mfma_per_dwordx4=4,
                disable_unclustered_high_rp_reschedule=True,
                reverse_local_assignment=True,
                sink_insts_to_avoid_spills=True,
                regclass_priority_trumps_globalness=True,
                mfma_vgpr_form=False,
            )
            _reduce_split_k[(m // 32, n // 256)](workspace, out, m, n, split_k, 32, 256, num_warps=8)
            return out
        # GROUP_M=4 groups up to four M tiles per B panel in both launch paths.
        # Keeping the group small also bounds A-panel reuse distance along N.
        cu_count = torch.cuda.get_device_properties(a.device).multi_processor_count
        model_linear = (m, n, k) in _MODEL_LINEAR_SHAPES
        mi32_256x192_shape = (m, n, k) in {
            (1024, 3072, 12416),
            (1024, 24576, 6144),
            (1024, 6144, 22656),
            (1024, 6144, 39936),
            (1024, 6144, 67584),
            (1024, 6144, 159744),
            (1024, 3072, 15360),
            (1024, 36864, 1024),
            (2048, 3072, 12416),
            (2048, 3072, 15360),
            (2048, 6144, 22656),
            (2048, 6144, 39936),
            (2048, 6144, 67584),
            (2048, 6144, 159744),
            (2048, 36864, 1024),
            (2048, 15360, 3584),
            (2048, 3072, 18048),
            (2048, 3584, 25600),
        }
        mfma_preshuffled_256x192 = ((m, n, k) in {
            (1024, 6144, 22656),
            (1024, 6144, 67584),
            (1024, 6144, 159744),
            (2048, 3072, 15360),
            (2048, 6144, 22656),
            (2048, 6144, 39936),
            (2048, 6144, 67584),
            (2048, 6144, 159744),
            (2048, 15360, 3584),
            (2048, 3072, 18048),
            (2048, 3584, 25600),
        } and sf_layout == "mfma_preshuffled")
        native_128x192 = ((m, n, k) == (1024, 36864, 1024) and sf_layout == "mfma_preshuffled") or ((m, n, k) in {
            (2048, 23040, 1792),
            (2048, 22464, 1024),
            (2048, 14976, 1024),
        } and sf_layout in {"cublas_blocked", "mfma_preshuffled"}) or (
            (m, n, k) == (2048, 36864, 1024) and sf_layout == "mfma_preshuffled") or (
                (m, n, k) == (2048, 3072, 12416) and sf_layout in {"cublas_blocked", "mfma_preshuffled"})
        if native_128x192:
            scales_shuffled = sf_layout == "mfma_preshuffled"
            if ((m, n, k) == (2048, 3072, 12416) and sf_layout == "cublas_blocked"):
                a_scale, b_scale = _pack_mfma_scale_pair_from_blocked(a_scale, b_scale, m, n, k, 2, 3)
                scales_shuffled = True
            return _gemm_native_cadence_128x192[(m // 128 * (n // 192), )](
                a,
                b,
                a_scale.view(torch.uint8),
                b_scale.view(torch.uint8),
                out,
                m,
                n,
                k,
                (4 if (m, n, k) == (2048, 3072, 12416) else (5 if m == 1024 else 12)),
                False,
                True,
                A_SHUFFLED=scales_shuffled,
                B_SHUFFLED=scales_shuffled,
                num_warps=4,
                num_stages=1,
                matrix_instr_nonkdim=32,
                enable_sched_group_barrier_scheduler=(m == 1024 or (m, n, k) == (2048, 3072, 12416)),
                sched_group_barrier_mfma_per_dwordx4=4,
                disable_unclustered_high_rp_reschedule=True,
                reverse_local_assignment=False,
                sink_insts_to_avoid_spills=True,
                mfma_vgpr_form=False,
            )
        if mi32_256x192_shape and (sf_layout == "cublas_blocked" or mfma_preshuffled_256x192):
            if (m, n, k) in {
                (1024, 6144, 22656),
                (1024, 6144, 39936),
                (1024, 6144, 67584),
                (1024, 6144, 159744),
                (1024, 3072, 15360),
                (2048, 3072, 12416),
                (2048, 3072, 15360),
                (2048, 3072, 18048),
                (2048, 3584, 25600),
            }:
                split_k = 4 if (m, n, k) in {
                    (1024, 3072, 15360),
                    (2048, 3584, 25600),
                } else 2
                workspace = _allocate_split_workspace(a, split_k, m, n)
                _gemm_native_cadence_split2[(m // 256 * triton.cdiv(n, 192) * split_k, )](
                    a,
                    b,
                    a_scale.view(torch.uint8),
                    b_scale.view(torch.uint8),
                    out,
                    workspace,
                    m,
                    n,
                    k,
                    256,
                    192,
                    2,
                    split_k,
                    A_SHUFFLED=mfma_preshuffled_256x192,
                    B_SHUFFLED=mfma_preshuffled_256x192,
                    num_warps=4,
                    num_stages=1,
                    matrix_instr_nonkdim=32,
                    enable_sched_group_barrier_scheduler=(m, n, k) not in {
                        (1024, 6144, 22592),
                        (2048, 3072, 12416),
                    } and not mfma_preshuffled_256x192,
                    sched_group_barrier_mfma_per_dwordx4=(2 if (m, n, k) == (1024, 6144, 67584) else 4),
                    disable_unclustered_high_rp_reschedule=True,
                    reverse_local_assignment=(mfma_preshuffled_256x192 and (m, n, k) in {
                        (1024, 6144, 67584),
                        (1024, 6144, 159744),
                    }),
                    sink_insts_to_avoid_spills=True,
                    mfma_vgpr_form=False,
                )
                if (m, n) == (1024, 6144):
                    _reduce_split_k[(m // 16, n // 64)](workspace, out, m, n, split_k, 16, 64, num_warps=4)
                else:
                    _reduce_split_k[(m // 32, n // 128)](workspace, out, m, n, split_k, 32, 128, num_warps=8)
                return out
            if (m, n, k) in {
                (1024, 36864, 1024),
                (1024, 24576, 6144),
                (2048, 6144, 22656),
                (2048, 6144, 39936),
                (2048, 6144, 67584),
                (2048, 6144, 159744),
                (2048, 15360, 3584),
            }:
                sa = a_scale.view(torch.uint8)
                sb = b_scale.view(torch.uint8)
                a_shuffled = mfma_preshuffled_256x192
                b_shuffled = mfma_preshuffled_256x192
                return _gemm_native_cadence_256x192[(m // 256 * (n // 192), )](
                    a,
                    b,
                    sa,
                    sb,
                    out,
                    out,
                    m,
                    n,
                    k,
                    256,
                    192,
                    2,
                    1,
                    GROUP_M=(8 if (m, n, k) in {
                        (1024, 24576, 6144),
                        (2048, 6144, 159744),
                    } else 4),
                    A_SHUFFLED=a_shuffled,
                    B_SHUFFLED=b_shuffled,
                    num_warps=4,
                    num_stages=1,
                    matrix_instr_nonkdim=32,
                    enable_sched_group_barrier_scheduler=not (mfma_preshuffled_256x192 and
                                                              (m, n, k) == (2048, 6144, 67584)),
                    sched_group_barrier_mfma_per_dwordx4=4,
                    disable_unclustered_high_rp_reschedule=True,
                    reverse_local_assignment=False,
                    sink_insts_to_avoid_spills=True,
                    mfma_vgpr_form=False,
                    schedule_hint="mfma-schedule" if (m, n, k) == (2048, 6144, 39936) else "",
                )
            split_k = (1 if k in {1024, 6144} or (m, k) == (2048, 39936) else (4 if (m, k) == (1024, 12416) else
                                                                               (1 if (m, k) == (2048, 22656) else 2)))
            k_tiles = k // 128 // split_k
            if split_k == 1 and k_tiles % 2:
                k_tiles -= 1
            workspace = (_allocate_split_workspace(a, split_k, m, n) if split_k > 1 else out)
            _gemm_mi32_256x192[(m // 256 * (n // 192) * split_k, )](
                a, b, out, workspace, a_scale.view(torch.uint8), b_scale.view(torch.uint8), m, n, k, k_tiles, split_k,
                (8 if (m, n, k) == (2048, 36864, 1024) else (2 if k in {6144, 12416} else 4)),
                HAS_TAIL=k_tiles * split_k != k // 128, STREAMING_STORE=(k == 1024 or (m, n, k) in {
                    (1024, 24576, 6144),
                }), num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=(0 if (m, n, k) in {
                    (1024, 36864, 1024),
                    (2048, 36864, 1024),
                } else 1), disable_unclustered_high_rp_reschedule=(m, n, k) in {
                    (2048, 3072, 12416),
                }, sink_insts_to_avoid_spills=(k == 1024 or (m, n, k) in {
                    (2048, 3072, 12416),
                    (2048, 6144, 22656),
                }), reverse_local_assignment=(k == 1024 or (m, n, k) == (1024, 24576, 6144)),
                enable_sched_group_barrier_scheduler=True,
                sched_group_barrier_mfma_per_dwordx4=(2 if k in {6144, 22656} else 4),
                schedule_hint=("" if (m, n, k) in {
                    (1024, 36864, 1024),
                    (2048, 36864, 1024),
                } else "mfma-schedule"), llvm_fn_attrs=(("amdgpu-sched-strategy", "iterative-minreg"), ) if
                (m, n, k) == (1024, 6144, 39936) else "")
            if split_k > 1:
                _reduce_split_k[(m // 32, n // 128)](workspace, out, m, n, split_k, 32, 128, num_warps=8)
            return out
        if ((m, n, k) == (1024, 8192, 3072) and sf_layout == "cublas_blocked"
                and all(tensor.data_ptr() % 16 == 0 for tensor in (a, b, out))
                and all(tensor.data_ptr() % 8 == 0 for tensor in (a_scale, b_scale))):
            return _gemm_native_split8[(m // 128 * (n // 256), )](
                a,
                b,
                a_scale.view(torch.uint8),
                b_scale.view(torch.uint8),
                out,
                out,
                m,
                n,
                k,
                128,
                256,
                2,
                1,
                A_SHUFFLED=sf_layout == "mfma_preshuffled",
                B_SHUFFLED=sf_layout == "mfma_preshuffled",
                num_warps=4,
                num_stages=1,
                matrix_instr_nonkdim=32,
                enable_sched_group_barrier_scheduler=True,
                sched_group_barrier_mfma_per_dwordx4=4,
                disable_unclustered_high_rp_reschedule=True,
                reverse_local_assignment=False,
                sink_insts_to_avoid_spills=True,
                mfma_vgpr_form=False,
            )
        if ((m, n, k) == (1024, 8192, 3072) and sf_layout == "mfma_preshuffled"
                and all(tensor.data_ptr() % 16 == 0 for tensor in (a, b, out))
                and all(tensor.data_ptr() % 8 == 0 for tensor in (a_scale, b_scale))):
            return _gemm_native_128x128_s1[(m // 128 * (n // 128), )](
                a,
                b,
                a_scale.view(torch.uint8),
                b_scale.view(torch.uint8),
                out,
                out,
                m,
                n,
                k,
                128,
                128,
                2,
                1,
                SCALES_SHUFFLED=True,
                num_warps=4,
                num_stages=1,
                matrix_instr_nonkdim=32,
                enable_sched_group_barrier_scheduler=True,
                sched_group_barrier_mfma_per_dwordx4=4,
                disable_unclustered_high_rp_reschedule=True,
                reverse_local_assignment=False,
                sink_insts_to_avoid_spills=True,
                mfma_vgpr_form=False,
            )
        if (m, n, k) == (1024, 8192, 3072) and sf_layout == "cublas_blocked":
            return _gemm_skinny_mi32[(m // 128 * (n // 128), )](
                a, b, out, a_scale.view(torch.uint8), b_scale.view(torch.uint8), m, n, k, 1, STREAMING_STORE=True,
                CK_MAPPING=True, num_warps=4, num_stages=1, matrix_instr_nonkdim=32,
                disable_unclustered_high_rp_reschedule=True, reverse_local_assignment=True,
                sink_insts_to_avoid_spills=True, enable_sched_group_barrier_scheduler=True,
                sched_group_barrier_mfma_per_dwordx4=4)
        model_skinny = ((sf_layout == "natural" and model_linear and k >= 12288 and k % 256 == 0) or (m, n, k) in {
            (1024, 1024, 24576),
            (1024, 8192, 3072),
            (1024, 3072, 12416),
        })
        if model_skinny:
            block = 128
            skinny_uneven_split = (m, n, k) == (1024, 3072, 12416)
            split_k = 2 if skinny_uneven_split else _model_split_k(m, n, k, cu_count, block)
            skinny_group_m = (1 if (m, n, k) == (1024, 8192, 3072) else (2 if skinny_uneven_split else 4))
            workspace = (_allocate_split_workspace(a, split_k, m, n) if split_k > 1 else out)
            _gemm_skinny[(m // block * (n // block) * split_k, )](
                a, b, out, workspace, a_scale.view(torch.uint8), b_scale.view(torch.uint8), m, n, k,
                k // 256 if skinny_uneven_split else k // split_k // block, split_k, skinny_group_m,
                sf_layout == "cublas_blocked", UNEVEN_SPLIT=skinny_uneven_split,
                LOOP_UNROLL_FACTOR=8 if skinny_uneven_split else 1, num_warps=8, num_stages=1, matrix_instr_nonkdim=16,
                waves_per_eu=(1 if skinny_uneven_split or (m, n, k) == (1024, 1024, 24576) else 0),
                disable_unclustered_high_rp_reschedule=(m, n, k) != (1024, 1024, 24576), reverse_local_assignment=(m, n,
                                                                                                                   k)
                != (1024, 1024, 24576), sink_insts_to_avoid_spills=(m, n, k) == (1024, 1024, 24576),
                regclass_priority_trumps_globalness=(m, n, k) == (1024, 1024, 24576), llvm_fn_attrs=((
                    ("amdgpu-agpr-alloc", "0,0"),
                    ("amdgpu-sched-strategy", "max-ilp"),
                    ("amdgpu-post-sched-strategy", "nop"),
                ) if (m, n, k) == (1024, 1024, 24576) else (
                    ("amdgpu-agpr-alloc", "0,0"),
                    ("amdgpu-sched-strategy", "iterative-ilp" if skinny_uneven_split else "max-ilp"),
                )))
            if split_k > 1:
                _reduce_split_k[(m // 32, n // 128)](workspace, out, m, n, split_k, 32, 128, num_warps=8)
            return out
        interwave_shape = ((n <= 8192 and m <= 8192 and 2048 <= k <= 8192) or model_linear
                           or (sf_layout == "natural" and m >= 16384 and n >= 1024 and 512 <= k <= 4096))
        interwave = (m % 256 == 0 and n % 256 == 0 and k % 128 == 0 and interwave_shape
                     and ((m // 256) * (n // 256) >= cu_count or model_linear)
                     and all(tensor.data_ptr() % 16 == 0 for tensor in (a, a_scale, b, b_scale, out)))
        if interwave:
            prepacked_k_major = sf_layout == "k_major"
            k_major = prepacked_k_major or (sf_layout == "natural" and (m == n == k == 8192 or
                                                                        (m >= 16384 and k >= 4096)))
            uneven_split = (m, n, k) in {
                (1024, 6144, 22656),
                (1024, 6144, 39936),
                (1024, 6144, 67584),
                (1024, 6144, 159744),
                (2048, 3072, 12416),
            }
            split_k = (5 if (m, n, k) in {
                (1024, 6144, 39936),
                (1024, 6144, 67584),
                (1024, 6144, 159744),
            } else (2 if uneven_split else (_model_split_k(m, n, k, cu_count, 256) if model_linear else 1)))
            model_xcd = model_linear and (m, n, k) not in {
                (1024, 1024, 159744),
                (1024, 6144, 22656),
                (2048, 1024, 159744),
                (2048, 3072, 12416),
                (2048, 1024, 24576),
            }
            if (m, n, k) == (2048, 6144, 22656):
                group_m = 2
            elif (m, n, k) == (2048, 3072, 15360):
                group_m = 3
            elif (m, n, k) in {
                (1024, 6144, 22656),
                (1024, 32768, 6144),
                (2048, 3072, 12416),
            }:
                group_m = 2
            elif (m, n, k) in {
                (1024, 64512, 1024),
                (1024, 153600, 1024),
                (1024, 159744, 1024),
            }:
                group_m = 3
            elif (m, n, k) in {
                (2048, 64512, 1024),
                (2048, 153600, 1024),
                (2048, 159744, 1024),
            }:
                group_m = 7
            else:
                group_m = 8 if model_linear and n >= 153600 else 4
            workspace = (_allocate_split_workspace(a, split_k, m, n) if split_k > 1 else out)
            fused_split_reduce = (m, n, k) == (1024, 1024, 24576)
            locks = (torch.empty(
                (m // 256 * (n // 256) * split_k, ),
                dtype=torch.int32,
                device=a.device,
            ) if fused_split_reduce else out)
            use_mfma_schedule = k == 1024 and (m, n) not in {
                (1024, 36864),
                (1024, 64512),
                (2048, 64512),
            }
            sa = (_pack_k_major_scales(a_scale) if k_major and not prepacked_k_major else a_scale.view(torch.uint8))
            sb = (_pack_k_major_scales(b_scale) if k_major and not prepacked_k_major else b_scale.view(torch.uint8))
            _gemm_interwave[(m // 256 * (n // 256) * split_k, )](
                a, b, out, workspace, locks, sa, sb, m, n, k // 128 // split_k, k, split_k, group_m,
                sf_layout == "cublas_blocked", model_xcd, num_warps=8, num_stages=1, K_MAJOR=k_major,
                UNEVEN_SPLIT=uneven_split, LLVM_MFMA_SCHED=use_mfma_schedule,
                LOOP_UNROLL_FACTOR=(3 if (m, n, k) == (1024, 6144, 39936) else (2 if (m, n, k) in {
                    (1024, 6144, 22656),
                    (2048, 6144, 22656),
                } else 1)), WINDOWED_K64=(m, n, k) == (2048, 1024, 24576), FUSED_SPLIT_REDUCE=fused_split_reduce,
                STREAMING_STORE=(m, n, k) in {
                    (1024, 153600, 1024),
                    (1024, 159744, 1024),
                    (1024, 24576, 6144),
                    (1024, 32768, 6144),
                    (1024, 64512, 1024),
                    (2048, 24576, 6144),
                    (2048, 8192, 3072),
                    (2048, 64512, 1024),
                    (2048, 153600, 1024),
                    (2048, 159744, 1024),
                }, matrix_instr_nonkdim=16, waves_per_eu=(1 if (m, n, k) in {
                    (1024, 1024, 24576),
                    (1024, 36864, 1024),
                    (1024, 64512, 1024),
                    (2048, 1024, 24576),
                    (2048, 24576, 6144),
                    (2048, 6144, 22656),
                } else 0), disable_unclustered_high_rp_reschedule=(not prepacked_k_major
                                                                   and (m, n, k) != (1024, 64512, 1024)),
                reverse_local_assignment=((k != 1024 or (m, n) == (2048, 64512)) and (m, n, k) not in {
                    (1024, 24576, 6144),
                    (1024, 32768, 6144),
                    (1024, 1024, 24576),
                    (2048, 1024, 24576),
                    (2048, 24576, 6144),
                    (2048, 8192, 3072),
                    (2048, 32768, 6144),
                    (2048, 6144, 22656),
                }), sink_insts_to_avoid_spills=(m, n, k) == (2048, 1024, 24576),
                enable_sched_group_barrier_scheduler=(k == 6144 or (m, n, k) in {
                    (1024, 64512, 1024),
                    (2048, 8192, 3072),
                }), schedule_hint="mfma-schedule" if use_mfma_schedule else "",
                regclass_priority_trumps_globalness=(m, n, k) == (2048, 1024, 24576), llvm_fn_attrs=((
                    ("amdgpu-agpr-alloc", "0,0"),
                    ("amdgpu-sched-strategy", "iterative-ilp"),
                ) if (m, n, k) == (2048, 3072, 15360) else (("amdgpu-agpr-alloc", "0,0"), )))
            if split_k > 1 and not fused_split_reduce:
                _reduce_split_k[(m // 32, n // 128)](workspace, out, m, n, split_k, 32, 128, STREAMING=(m, n, k) in {
                    (1024, 6144, 22656),
                    (1024, 6144, 39936),
                }, num_warps=8)
            return out
        return _gemm_generic[(triton.cdiv(m, 128) * triton.cdiv(n, 128), )](
            a, b, out, a_scale.view(torch.uint8), b_scale.view(torch.uint8), m, n, k, 128, 128, 128, 4,
            sf_layout == "cublas_blocked", num_warps=4, num_stages=1, matrix_instr_nonkdim=16,
            disable_unclustered_high_rp_reschedule=True, llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ))


def mm_mxfp8(a, a_scale, b, b_scale, *, out=None, sf_layout="natural", space="heuristic"):
    if space not in ("heuristic", "full"):
        raise InvalidInput(f"tlx.ops.mm_mxfp8 on gfx950 does not provide space={space!r}")
    m, k = a.shape
    n = b.shape[0]
    if min(m, n, k) <= 0 or k % 128:
        raise InvalidInput("tlx.ops.mm_mxfp8 on gfx950 requires positive M and N, with K divisible by 128; "
                           f"got M={m}, N={n}, K={k}")
    if max(m * k, n * k, 2 * m * n) >= 2**31:
        raise InvalidInput(
            "gfx950 MXFP8 GEMM requires data and output byte spans addressable with signed 32-bit offsets")
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    if (m <= 128 and sf_layout == "natural" and k >= 24576 and not (m in {85, 128} and (n, k) == (3584, 25600))):
        a_scale, b_scale = _pack_cublas_blocked_scale_pair(a_scale, b_scale)
        sf_layout = "cublas_blocked"
    if (m % 128 == 0 and n % 128 == 0 and not (m <= 128 and n % 32 == 0)):
        _launch(a, a_scale, b, b_scale, out, sf_layout=sf_layout)
        return out
    if ((m, n, k) == (2048, 22464, 1024) and sf_layout in {"cublas_blocked", "mfma_preshuffled"}):
        _launch(a, a_scale, b, b_scale, out, sf_layout=sf_layout)
        return out
    if (m <= 128 and n % 32 == 0 and sf_layout in {"natural", "cublas_blocked", "mfma_preshuffled"}):
        split_k = 5 if n <= 3584 and k >= 24576 else 8 if n <= 3072 and k >= 18048 else 1
        short_block_m = 64
        if sf_layout == "natural" and (n, k) in {
            (23040, 1792),
            (22464, 1024),
        }:
            short_block_m = 96 if m == 85 else 128 if m == 128 else 64
        workspace = _allocate_split_workspace(a, split_k, m, n) if split_k > 1 else out
        _gemm_native_short_m[(triton.cdiv(m, short_block_m) * triton.cdiv(n, 128) * split_k, )](
            a,
            b,
            a_scale.view(torch.uint8),
            b_scale.view(torch.uint8),
            out,
            workspace,
            m,
            n,
            k,
            split_k,
            short_block_m,
            sf_layout == "mfma_preshuffled",
            sf_layout == "cublas_blocked",
            num_warps=4,
            num_stages=1,
            matrix_instr_nonkdim=32,
            enable_sched_group_barrier_scheduler=False,
            sched_group_barrier_mfma_per_dwordx4=4,
            disable_unclustered_high_rp_reschedule=False,
            reverse_local_assignment=True,
            sink_insts_to_avoid_spills=True,
            mfma_vgpr_form=False,
        )
        if split_k > 1:
            _reduce_split_k_masked[(triton.cdiv(m, 32), triton.cdiv(n, 128))](workspace, out, m, n, split_k, 32, 128,
                                                                              num_warps=8)
        return out
    _gemm_m_masked[(triton.cdiv(m, 128) * triton.cdiv(n, 128), )](
        a,
        b,
        out,
        a_scale.view(torch.uint8),
        b_scale.view(torch.uint8),
        m,
        n,
        k,
        sf_layout == "cublas_blocked",
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
        disable_unclustered_high_rp_reschedule=True,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
    )
    return out


# -----------------------------------------------------------------------------
# Grouped Tiles
# -----------------------------------------------------------------------------

# [128, 4] E8M0 scale slice (flat r * 4 + c): each lane copies one row's 4
# contiguous bytes, so direct-to-LDS issues 32-bit copies. The two extra thread
# bits broadcast (stride 0) over the 512-thread workgroup.
_grouped_SCALE_COPY_LAYOUT = tlx.layout(shape=((128, 4), (4, )), stride=((4, 0), (1, )))

# K_MAJOR atoms (natural inputs on gfx950) store each 16-byte chunk as
# [k][row_group]: byte (r, c) of a 128-row block sits at
# (r % 32) * 16 + c * 4 + r // 32. The four row-group bytes one scaled MFMA packs
# into a register are then adjacent, so the atom is copied as flat [32, 16]
# bytes and read back through _scale_view with one ds_read_b32 instead of four
# ds_read_u8 plus v_perm.
_grouped_SCALE_FLAT_COPY_LAYOUT = tlx.layout(shape=((4, 32, 4), (4, )), stride=((4, 16, 0), (1, )))

# 16x16x128 scaled-MFMA scale register layouts (#mma warpsPerCTA=[2, 4]) for the
# [128, 4] A and B scale slices. Loading LDS scales straight into these avoids a
# convert_layout through shared scratch, which the two phase-shifted wave groups
# of the inter-wave pipeline would otherwise race on.
_grouped_SA_MFMA_LAYOUT = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
    stride=((4, 8, 16, 32, 1, 2, 0, 0, 64), (128, 256)),
)
_grouped_SB_MFMA_LAYOUT = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, )),
    stride=((4, 8, 16, 32, 1, 2, 64, 128, 0), (256, )),
)


@triton.jit
def _grouped_scale_offsets(rows, k_chunks, K_MAJOR: tl.constexpr):
    """Byte offsets of the [rows, 4] scale slice for K-step 0 in the blocked layout."""
    cols = tl.arange(0, 4)
    base = (rows[:, None] // 128) * k_chunks * 512 + (rows[:, None] % 32) * 16
    if K_MAJOR:
        offsets = base + cols[None, :] * 4 + (rows[:, None] % 128) // 32
    else:
        offsets = base + ((rows[:, None] % 128) // 32) * 4 + cols[None, :]
    return offsets


@triton.jit
def _grouped_kmajor_scale_view(buf):
    read_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[32, 0], [64, 0], [0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        block_bases=[],
        alignment=4,
    )
    return tlx.local_reinterpret(buf, tl.uint8, [128, 4], layout=read_layout)


@triton.jit
def _grouped_load_scale(buf, layout: tl.constexpr, K_MAJOR: tl.constexpr):
    if K_MAJOR:
        value = tlx.local_load(_grouped_kmajor_scale_view(buf), layout=layout, relaxed=True)
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

    # M is a multiple of 128, so each 128-row A half-tile is one whole
    # row block (wrapped to block 0 past the group end) and is addressed by a
    # scalar base plus one shared [128, BLOCK_K] offset tensor. Its scales are
    # exactly one 512-byte atom per K-step. Wrapped garbage rows/cols are dropped
    # by the masked store.
    offs_r = tl.arange(0, HALF_M)
    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)
    # For the persistent grouped caller, opaque_zero (runtime 0) keeps these
    # tile-invariant lane offsets inside the tile. Hoisting them above the loop
    # would spill and force a scratch reload that drains prior output stores.
    tile_off = offs_r[:, None] * K + offs_k[None, :] + opaque_zero * 256
    if K_MAJOR:
        sc_off = tlx.require_layout(
            tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :] + opaque_zero * 256,
            _grouped_SCALE_FLAT_COPY_LAYOUT)
    else:
        sc_cols = tl.arange(0, 4)
        sc_off = tlx.require_layout(
            (offs_r[:, None] % 32) * 16 + (offs_r[:, None] // 32) * 4 + sc_cols[None, :] + opaque_zero * 256,
            _grouped_SCALE_COPY_LAYOUT)

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
        sb_l = _grouped_load_scale(smem_sb_left[0], _grouped_SB_MFMA_LAYOUT, K_MAJOR)
        a_top = tlx.local_load(smem_a_top[0], relaxed=True)
        sa_t = _grouped_load_scale(smem_sa_top[0], _grouped_SA_MFMA_LAYOUT, K_MAJOR)

        for k in tl.range(0, n_pipe - 2, 2, num_stages=1):
            # sub-iter 0 (buffer 0)
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tl = tl.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
                sa_b = _grouped_load_scale(smem_sa_bot[0], _grouped_SA_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_b_left[0], b_left_base, b_left_off + ka)
                tlx.buffer_load_to_local(smem_sb_left[0], sb_left_base, sb_left_off + ks)
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_bl = tl.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
                sb_r = _grouped_load_scale(smem_sb_right[0], _grouped_SB_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_a_top[0], a_top_base, a_top_off + ka)
                tlx.buffer_load_to_local(smem_sa_top[0], sa_top_base, sa_top_off + ks)
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tr = tl.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
                sb_l = _grouped_load_scale(smem_sb_left[1], _grouped_SB_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_a_bot[0], a_bot_base, a_bot_off + ka)
                tlx.buffer_load_to_local(smem_sa_bot[0], sa_bot_base, sa_bot_off + ks)
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_br = tl.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_top = tlx.local_load(smem_a_top[1], relaxed=True)
                sa_t = _grouped_load_scale(smem_sa_top[1], _grouped_SA_MFMA_LAYOUT, K_MAJOR)
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
                sa_b = _grouped_load_scale(smem_sa_bot[1], _grouped_SA_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_b_left[1], b_left_base, b_left_off + (ka + kb1))
                tlx.buffer_load_to_local(smem_sb_left[1], sb_left_base, sb_left_off + (ks + SC_STEP))
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_bl = tl.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
                sb_r = _grouped_load_scale(smem_sb_right[1], _grouped_SB_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_a_top[1], a_top_base, a_top_off + (ka + kb1))
                tlx.buffer_load_to_local(smem_sa_top[1], sa_top_base, sa_top_off + (ks + SC_STEP))
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tr = tl.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
                sb_l = _grouped_load_scale(smem_sb_left[0], _grouped_SB_MFMA_LAYOUT, K_MAJOR)
                tlx.amd_sched_barrier(0)  # keep ds_reads ahead of the global loads
                tlx.buffer_load_to_local(smem_a_bot[1], a_bot_base, a_bot_off + (ka + kb1))
                tlx.buffer_load_to_local(smem_sa_bot[1], sa_bot_base, sa_bot_off + (ks + SC_STEP))
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_br = tl.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_top = tlx.local_load(smem_a_top[0], relaxed=True)
                sa_t = _grouped_load_scale(smem_sa_top[0], _grouped_SA_MFMA_LAYOUT, K_MAJOR)
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
        sa_b = _grouped_load_scale(smem_sa_bot[0], _grouped_SA_MFMA_LAYOUT, K_MAJOR)

        acc_bl = tl.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
        tlx.async_load_wait_group(4)
        b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
        sb_r = _grouped_load_scale(smem_sb_right[0], _grouped_SB_MFMA_LAYOUT, K_MAJOR)

        acc_tr = tl.dot_scaled(a_top, sa_t, "e4m3", b_right, sb_r, "e4m3", acc_tr)
        tlx.async_load_wait_group(3)
        b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
        sb_l = _grouped_load_scale(smem_sb_left[1], _grouped_SB_MFMA_LAYOUT, K_MAJOR)

        acc_br = tl.dot_scaled(a_bot, sa_b, "e4m3", b_right, sb_r, "e4m3", acc_br)
        tlx.async_load_wait_group(2)
        a_top = tlx.local_load(smem_a_top[1], relaxed=True)
        sa_t = _grouped_load_scale(smem_sa_top[1], _grouped_SA_MFMA_LAYOUT, K_MAJOR)

        acc_tl = tl.dot_scaled(a_top, sa_t, "e4m3", b_left, sb_l, "e4m3", acc_tl)
        tlx.async_load_wait_group(1)
        a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
        sa_b = _grouped_load_scale(smem_sa_bot[1], _grouped_SA_MFMA_LAYOUT, K_MAJOR)

        acc_bl = tl.dot_scaled(a_bot, sa_b, "e4m3", b_left, sb_l, "e4m3", acc_bl)
        tlx.async_load_wait_group(0)
        b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
        sb_r = _grouped_load_scale(smem_sb_right[1], _grouped_SB_MFMA_LAYOUT, K_MAJOR)

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
            sc_tail = _grouped_scale_offsets(tl.arange(0, HALF_M), 0, True)
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
    sa_ptrs = as_ptr + _grouped_scale_offsets(m_start + lm, k_chunks, K_MAJOR)
    sb_ptrs = bs_ptr + _grouped_scale_offsets(nn, k_chunks, K_MAJOR)

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


# -----------------------------------------------------------------------------
# Native Cadence 256X192
# -----------------------------------------------------------------------------


@triton.jit
def _n256x192_packed_scale_pair(source, block, panel, IS_A: tl.constexpr, PAIR: tl.constexpr, LIMIT: tl.constexpr,
                                K: tl.constexpr):
    word_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((1, 2, 4, 8, 16, 0, 32, 64), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    lane_row = tl.arange(0, 32)[None, :, None]
    wave_rows = wave // 2 * 128 if IS_A else wave % 2 * 64
    rows = block + wave_rows + PAIR * 64 + lane_row
    word_offsets = ((rows // 128) * (K // 128) * 128 + rows % 32 * 4 + rows % 128 // 32)
    offsets = tlx.require_layout(word_offsets // 2, word_layout)
    mask = tlx.require_layout((rows < LIMIT) & (panel < K // 128), word_layout)
    words = tlx.buffer_load(source.to(tl.pointer_type(tl.uint64)) + panel * 64, offsets, mask=mask, other=0)
    return tlx.require_layout(words, word_layout)


@triton.jit
def _n256x192_packed_scale_a_pairs(source, block, panel, LIMIT: tl.constexpr, K: tl.constexpr):
    tl.static_assert(LIMIT % 256 == 0)
    word_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
        stride=((4, 8, 16, 32, 64, 0, 128, 256), (1, 2)),
    )
    wave = tl.arange(0, 4)[:, None, None]
    lane_row = tl.arange(0, 32)[None, :, None]
    words_in_atom = tl.arange(0, 4)[None, None, :]
    rows = block + wave // 2 * 128 + lane_row
    word_offsets = ((rows // 128) * (K // 128) * 128 + rows % 32 * 4 + words_in_atom)
    offsets = tlx.require_layout(word_offsets, word_layout)
    mask = tlx.require_layout(
        tl.broadcast_to((rows < LIMIT) & (panel < K // 128), [4, 32, 4]),
        word_layout,
    )
    words = tlx.buffer_load(
        source.to(tl.pointer_type(tl.uint32)) + panel * 128,
        offsets,
        mask=mask,
        other=0,
        contiguity=4,
    )
    words = tlx.require_layout(words, word_layout)
    pieces = tl.tuple([tlx.extract_slice(words, [4, 32, 1], [0, 0, index]) for index in range(4)])
    return tl.tuple([
        pieces[0].to(tl.uint64) | (pieces[1].to(tl.uint64) << 32),
        pieces[2].to(tl.uint64) | (pieces[3].to(tl.uint64) << 32),
    ])


@triton.jit
def _n256x192_packed_scale_a_shuffled(source, pm, panel, K: tl.constexpr):
    scale_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    row = tl.arange(0, 32)[None, :, None]
    group = tl.arange(0, 2)[None, None, :]
    lane = row + group * 32
    base_group = (pm * 2 + wave // 2) * 2
    row_stride: tl.constexpr = K // 128 * 64
    k_offset = panel * 64
    return tl.tuple([
        tlx.buffer_load(
            source.to(tl.pointer_type(tl.uint32)),
            tlx.require_layout(
                (base_group + scale_group) * row_stride + k_offset + lane,
                scale_layout,
            ),
            contiguity=1,
        ) for scale_group in range(2)
    ])


@triton.jit
def _n256x192_packed_scale_b_shuffled(source, pn, panel, K: tl.constexpr):
    scale_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    row = tl.arange(0, 32)[None, :, None]
    group = tl.arange(0, 2)[None, None, :]
    lane = row + group * 32
    base_group = (pn * 2 + wave % 2) * 2
    row_stride: tl.constexpr = K // 128 * 64
    k_offset = panel * 64
    return tl.tuple([
        tlx.buffer_load(
            source.to(tl.pointer_type(tl.uint32)),
            tlx.require_layout(
                (base_group + scale_group) * row_stride + k_offset + lane,
                scale_layout,
            ),
            contiguity=1,
        ) for scale_group in range(2)
    ])


@tl.core.builtin
def _n256x192_packed_scale_permute(low, high, control, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        [high, low, control],
        {(tl.uint32, tl.uint32, tl.uint32): ("llvm.amdgcn.perm", tl.uint32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _n256x192_packed_scale_word(low, high):
    layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    low = tlx.release_layout(low, relaxed=True)
    high = tlx.release_layout(high, relaxed=True)
    low = tlx.require_layout(tl.broadcast_to(low, [4, 32, 2]), layout)
    high = tlx.require_layout(tl.broadcast_to(high, [4, 32, 2]), layout)
    group = tlx.require_layout(tl.broadcast_to(tl.arange(0, 2)[None, None, :], [4, 32, 2]), layout)
    control = (group.to(tl.uint32) * 0x01010101 + 0x06020400).to(tl.uint32)
    return tlx.require_layout(_n256x192_packed_scale_permute(low, high, control), layout)


@triton.jit
def _pack_mfma_scale_operand_from_blocked(source, output, rows: tl.constexpr, panels: tl.constexpr,
                                          repeat: tl.constexpr, atom, lane):
    blocks_per_chunk: tl.constexpr = triton.cdiv(repeat, 2)
    chunks: tl.constexpr = triton.cdiv(rows, repeat * 32)
    atoms: tl.constexpr = chunks * blocks_per_chunk * panels
    block = atom // panels
    panel = atom % panels
    pair = block % blocks_per_chunk
    low_row = block // blocks_per_chunk * repeat * 32 + pair * 64 + lane
    high_row = low_row + 32
    low_offset = (low_row // 128 * panels * 128 + panel * 128 + low_row % 32 * 4 + low_row % 128 // 32)
    high_offset = (high_row // 128 * panels * 128 + panel * 128 + high_row % 32 * 4 + high_row % 128 // 32)
    low = tl.load(
        source.to(tl.pointer_type(tl.uint32)) + low_offset,
        mask=(atom < atoms) & (low_row < rows),
        other=0x7F7F7F7F,
    )
    high = tl.load(
        source.to(tl.pointer_type(tl.uint32)) + high_offset,
        mask=(atom < atoms) & (pair * 2 + 1 < repeat) & (high_row < rows),
        other=0x7F7F7F7F,
    )
    output_offset = atom * 64 + lane
    tl.store(
        output.to(tl.pointer_type(tl.uint32)) + output_offset,
        _n256x192_packed_scale_permute(low, high, tl.full(low.shape, 0x06020400, tl.uint32)),
        mask=atom < atoms,
    )
    tl.store(
        output.to(tl.pointer_type(tl.uint32)) + output_offset + 32,
        _n256x192_packed_scale_permute(low, high, tl.full(low.shape, 0x07030501, tl.uint32)),
        mask=atom < atoms,
    )


@triton.jit
def _pack_mfma_scale_pair_from_blocked_kernel(A, B, OA, OB, A_ROWS: tl.constexpr, B_ROWS: tl.constexpr,
                                              PANELS: tl.constexpr, A_REPEAT: tl.constexpr, B_REPEAT: tl.constexpr,
                                              ATOMS_PER_CTA: tl.constexpr):
    word_id = tl.arange(0, 32 * ATOMS_PER_CTA)
    atom = tl.program_id(0) * ATOMS_PER_CTA + word_id // 32
    lane = word_id % 32
    _pack_mfma_scale_operand_from_blocked(A, OA, A_ROWS, PANELS, A_REPEAT, atom, lane)
    _pack_mfma_scale_operand_from_blocked(B, OB, B_ROWS, PANELS, B_REPEAT, atom, lane)


def _pack_mfma_scale_pair_from_blocked(a_scale, b_scale, m, n, k, a_repeat, b_repeat):
    panels = k // 128
    a_blocks = triton.cdiv(m, a_repeat * 32) * triton.cdiv(a_repeat, 2)
    b_blocks = triton.cdiv(n, b_repeat * 32) * triton.cdiv(b_repeat, 2)
    a_out = torch.empty(a_blocks * panels * 256, dtype=torch.uint8, device=a_scale.device)
    b_out = torch.empty(b_blocks * panels * 256, dtype=torch.uint8, device=b_scale.device)
    atoms_per_cta = 16
    atoms = max(a_blocks, b_blocks) * panels
    _pack_mfma_scale_pair_from_blocked_kernel[(triton.cdiv(atoms,
                                                           atoms_per_cta), )](a_scale.view(torch.uint8),
                                                                              b_scale.view(torch.uint8), a_out, b_out,
                                                                              m, n, panels, a_repeat, b_repeat,
                                                                              atoms_per_cta, num_warps=4)
    return (
        a_out.view(torch.float8_e8m0fnu).reshape(-1, 512),
        b_out.view(torch.float8_e8m0fnu).reshape(-1, 512),
    )


@triton.jit
def _n256x192_packed_scale_a_finish(a_pairs):
    return tl.tuple([_n256x192_packed_scale_word(pair.to(tl.uint32), (pair >> 32).to(tl.uint32)) for pair in a_pairs])


@triton.jit
def _n256x192_packed_scale_b_finish(b_pairs):
    b0_low = b_pairs[0].to(tl.uint32)
    b0_high = (b_pairs[0] >> 32).to(tl.uint32)
    b1_low = b_pairs[1].to(tl.uint32)
    b1_high = (b_pairs[1] >> 32).to(tl.uint32)
    n_wave = tlx.assume_uniform(tlx.thread_id(0) // 64 % 2)
    first_low = tl.where(n_wave == 0, b0_low, b0_high)
    first_high = tl.where(n_wave == 0, b0_high, b1_low)
    last = tl.where(n_wave == 0, b1_low, b1_high)
    return tl.tuple([
        _n256x192_packed_scale_word(first_low, first_high),
        _n256x192_packed_scale_word(last, last),
    ])


@triton.jit
def _n256x192_packed_scale_initial(SA, SB, pm, pn, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                   A_SHUFFLED: tl.constexpr = False, B_SHUFFLED: tl.constexpr = False):
    a_words = tl.tuple([])
    b_words = tl.tuple([])
    if A_SHUFFLED:
        a_words = _n256x192_packed_scale_a_shuffled(SA, pm, 0, K)
        a_pairs = tl.tuple([])
    elif M % 256 == 0:
        a_pairs = _n256x192_packed_scale_a_pairs(SA, pm * 256, 0, M, K)
    else:
        a_pairs = tl.tuple([_n256x192_packed_scale_pair(SA, pm * 256, 0, True, pair, M, K) for pair in range(2)])
    if B_SHUFFLED:
        b_words = _n256x192_packed_scale_b_shuffled(SB, pn, 0, K)
        b_pairs = tl.tuple([])
    else:
        b_pairs = tl.tuple([_n256x192_packed_scale_pair(SB, pn * 192, 0, False, pair, N, K) for pair in range(2)])
    packed_a = tl.tuple([]) if A_SHUFFLED else _n256x192_packed_scale_a_finish(a_pairs)
    packed_b = tl.tuple([]) if B_SHUFFLED else _n256x192_packed_scale_b_finish(b_pairs)
    return (
        a_words if A_SHUFFLED else packed_a,
        b_words if B_SHUFFLED else packed_b,
    )


@triton.jit
def _n256x192_fly_layout_copy(source, row_base, panel, destination, ROWS: tl.constexpr, LIMIT: tl.constexpr,
                              K: tl.constexpr):
    copy_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2) + ((2, ) if ROWS >= 64 else
                                                         ()) + ((2, ) if ROWS >= 128 else
                                                                ()) + ((2, ) if ROWS >= 256 else ())),
        stride=((16, 32, 64, 128, 256, 512, 1024, 2048), (1, 2, 4, 8) + ((4096, ) if ROWS >= 64 else
                                                                         ()) + ((8192, ) if ROWS >= 128 else
                                                                                ()) + ((16384, ) if ROWS >= 256 else
                                                                                       ())),
    )
    indices = tl.arange(0, ROWS * 128)
    rows = row_base + indices // 128
    # Row bits 1:3 swizzle column bits 4:6 in the physical LDS copy.
    kk = (indices % 128) ^ ((indices & 0x700) >> 4)
    offsets = tlx.require_layout(tl.max_contiguous(tl.multiple_of(rows * K + kk, 16), 16), copy_layout)
    mask = tlx.require_layout((rows < LIMIT) & (panel < K // 128), copy_layout)
    other = tlx.require_layout(tl.zeros([ROWS * 128], tlx.dtype_of(source)), copy_layout)
    flat_shared: tl.constexpr = tlx.shared_linear_layout_encoding(
        [[1 << bit] for bit in range(12 + (ROWS >= 64) + (ROWS >= 128) + (ROWS >= 256))], [], alignment=16)
    destination = tlx.local_reinterpret(destination, tlx.dtype_of(source), [ROWS * 128], layout=flat_shared)
    tlx.buffer_load_to_local(destination, source + panel * 128, offsets, mask=mask, other=other)


@triton.jit
def _n256x192_fly_layout_publish(A, B, pm, pn, panel, a_stage, b_stage, M: tl.constexpr, N: tl.constexpr,
                                 K: tl.constexpr):
    _n256x192_fly_layout_copy(A, pm * 256, panel, a_stage[0], 256, M, K)
    for plane in tl.static_range(3):
        _n256x192_fly_layout_copy(B, pn * 192 + plane * 64, panel, b_stage[plane], 64, N, K)
    tlx.async_load_commit_group()


@triton.jit
def _n256x192_fly_layout_read_a(a_stage, tile: tl.constexpr, half: tl.constexpr, dot_a: tl.constexpr):
    wave_m = tlx.assume_uniform(tlx.thread_id(0) // 128)
    panel = tlx.local_slice(a_stage[0], [0, wave_m * 128 + tile * 32, half * 64], [1, 32, 64])
    return tlx.local_load(panel, layout=dot_a, relaxed=True)


@triton.jit
def _n256x192_fly_layout_read_b(b_stage, tile: tl.constexpr, half: tl.constexpr, dot_b: tl.constexpr):
    wave_n = tlx.assume_uniform(tlx.thread_id(0) // 64 % 2)
    first_row = wave_n * 96 + tile * 32
    plane = tlx.local_view(b_stage, first_row // 64)
    plane = tlx.local_trans(plane, (0, 2, 1))
    panel = tlx.local_slice(plane, [0, half * 64, first_row % 64], [1, 64, 32])
    return tlx.local_load(panel, layout=dot_b, relaxed=True)


@tl.core.builtin
def _n256x192_fly_layout_replace(values, index: tl.constexpr, value, _semantic=None):
    index = tl.core._unwrap_if_constexpr(index)
    items = list(values)
    items[index] = value
    return tl.tuple(items)


@tl.core.builtin
def _n256x192_fly_layout_dot_one(acc, a, b, sa, sb, half: tl.constexpr, mi: tl.constexpr, ni: tl.constexpr,
                                 mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr, _semantic=None):
    half, mi, ni = (tl.core._unwrap_if_constexpr(value) for value in (half, mi, ni))
    values = list(acc)
    a = tlx.require_layout(a, dot_a, pin=False, _semantic=_semantic)
    b = tlx.require_layout(b, dot_b, pin=False, _semantic=_semantic)
    scale_layout = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    sa = tlx.require_layout(sa, scale_layout, pin=False, _semantic=_semantic)
    sb = tlx.require_layout(sb, scale_layout, pin=False, _semantic=_semantic)
    c = tlx.require_layout(values[mi * 3 + ni], mma, pin=False, _semantic=_semantic)
    values[mi * 3 + ni] = tlx.amd_scheduled_mfma_scaled(
        a,
        b,
        c,
        sa,
        sb,
        "e4m3",
        "e4m3",
        accumulator_role="persistent",
        accumulator_register_class="agpr",
        a_scale_selector=half * 2 + mi % 2,
        b_scale_selector=half * 2 + ni % 2,
        _semantic=_semantic,
    )
    return tl.tuple(values)


@triton.jit
def _n256x192_cadence_publish_chunk(A, B, pm, pn, panel, a_stage, b_stage, CHUNK: tl.constexpr, M: tl.constexpr,
                                    N: tl.constexpr, K: tl.constexpr):
    if CHUNK < 8:
        destination = tlx.local_slice(a_stage[0], [0, CHUNK * 32, 0], [1, 32, 128])
        _n256x192_fly_layout_copy(A, pm * 256 + CHUNK * 32, panel, destination, 32, M, K)
    else:
        column: tl.constexpr = CHUNK - 8
        destination = tlx.local_slice(b_stage[column // 2], [0, column % 2 * 32, 0], [1, 32, 128])
        _n256x192_fly_layout_copy(B, pn * 192 + column * 32, panel, destination, 32, N, K)


@triton.jit
def _n256x192_cadence_dot_at(acc, a, b, sa, sb, INDEX: tl.constexpr, mma: tl.constexpr, dot_a: tl.constexpr,
                             dot_b: tl.constexpr):
    half: tl.constexpr = INDEX // 12
    ni: tl.constexpr = INDEX % 12 // 4
    mi: tl.constexpr = INDEX % 4
    return _n256x192_fly_layout_dot_one(acc, a[half * 4 + mi], b[half * 3 + ni], sa[mi // 2], sb[ni // 2], half, mi, ni,
                                        mma, dot_a, dot_b)


@triton.jit
def _n256x192_cadence_read_at(a_stage, b_stage, a, b, INDEX: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr):
    is_a: tl.constexpr = (False, False, True, True, True, False, True, False, False, True, True, True, False,
                          True)[INDEX]
    half: tl.constexpr = (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1)[INDEX]
    tile: tl.constexpr = (0, 1, 0, 1, 2, 2, 3, 0, 1, 0, 1, 2, 2, 3)[INDEX]
    if is_a:
        a = _n256x192_fly_layout_replace(a, half * 4 + tile, _n256x192_fly_layout_read_a(a_stage, tile, half, dot_a))
    else:
        b = _n256x192_fly_layout_replace(b, half * 3 + tile, _n256x192_fly_layout_read_b(b_stage, tile, half, dot_b))
    return a, b


@triton.jit
def _n256x192_cadence_compute(A, B, SA, SB, pm, pn, panel, a_current, b_current, a_next, b_next, a, b, sa, sb, acc,
                              mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr, M: tl.constexpr,
                              N: tl.constexpr, K: tl.constexpr, LOAD_FUTURE: tl.constexpr, LOAD_NEXT: tl.constexpr,
                              A_SHUFFLED: tl.constexpr = False, B_SHUFFLED: tl.constexpr = False):
    a_pairs = tl.tuple([])
    b_pairs = tl.tuple([])
    a_words = sa
    b_words = sb
    if LOAD_NEXT:
        for index in tl.static_range(5):
            acc = _n256x192_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)
            if index < 2:
                if A_SHUFFLED:
                    if index == 0:
                        tlx.amd_sched_barrier(0)
                        a_words = _n256x192_packed_scale_a_shuffled(SA, pm, panel + 1, K)
                        tlx.amd_sched_barrier(0)
                elif M % 256 == 0:
                    if index == 0:
                        tlx.amd_sched_barrier(0)
                        a_pairs = _n256x192_packed_scale_a_pairs(SA, pm * 256, panel + 1, M, K)
                        tlx.amd_sched_barrier(0)
                else:
                    tlx.amd_sched_barrier(0)
                    a_pairs += tl.tuple([_n256x192_packed_scale_pair(SA, pm * 256, panel + 1, True, index, M, K)])
                    tlx.amd_sched_barrier(0)
            elif index < 4:
                if B_SHUFFLED:
                    if index == 2:
                        tlx.amd_sched_barrier(0)
                        b_words = _n256x192_packed_scale_b_shuffled(SB, pn, panel + 1, K)
                        tlx.amd_sched_barrier(0)
                else:
                    tlx.amd_sched_barrier(0)
                    b_pairs += tl.tuple([_n256x192_packed_scale_pair(SB, pn * 192, panel + 1, False, index - 2, N, K)])
                    tlx.amd_sched_barrier(0)
    else:
        for index in tl.static_range(5):
            acc = _n256x192_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)

    for chunk in tl.static_range(12):
        if LOAD_FUTURE:
            with tlx.warp_pipeline_stage("cadence_g2l", scope="intra_wave", pair=chunk):
                _n256x192_cadence_publish_chunk(A, B, pm, pn, panel + 2, a_current, b_current, chunk, M, N, K)
            with tlx.warp_pipeline_stage("cadence_mfma", scope="intra_wave", pair=chunk):
                acc = _n256x192_cadence_dot_at(acc, a, b, sa, sb, chunk + 5, mma, dot_a, dot_b)
        else:
            acc = _n256x192_cadence_dot_at(acc, a, b, sa, sb, chunk + 5, mma, dot_a, dot_b)
    if LOAD_FUTURE:
        with tlx.warp_pipeline_stage("cadence_g2l_last", scope="intra_wave", pair=12):
            _n256x192_cadence_publish_chunk(A, B, pm, pn, panel + 2, a_current, b_current, 12, M, N, K)
            _n256x192_cadence_publish_chunk(A, B, pm, pn, panel + 2, a_current, b_current, 13, M, N, K)
            tlx.async_load_commit_group()
        with tlx.warp_pipeline_stage("cadence_mfma_last", scope="intra_wave", pair=12):
            acc = _n256x192_cadence_dot_at(acc, a, b, sa, sb, 17, mma, dot_a, dot_b)
    else:
        acc = _n256x192_cadence_dot_at(acc, a, b, sa, sb, 17, mma, dot_a, dot_b)

    next_a, next_b = a, b
    if LOAD_NEXT:
        tlx.async_load_wait_group(1 if LOAD_FUTURE else 0)
        for group in tl.static_range(7):
            if group > 0:
                tlx.amd_sched_barrier(0)
                acc = _n256x192_cadence_dot_at(acc, a, b, sa, sb, 17 + group, mma, dot_a, dot_b)
                tlx.amd_sched_barrier(0)
            next_a, next_b = _n256x192_cadence_read_at(a_next, b_next, next_a, next_b, group * 2, dot_a, dot_b)
            next_a, next_b = _n256x192_cadence_read_at(a_next, b_next, next_a, next_b, group * 2 + 1, dot_a, dot_b)
        packed_a = (tl.tuple([]) if A_SHUFFLED else _n256x192_packed_scale_a_finish(a_pairs))
        packed_b = (tl.tuple([]) if B_SHUFFLED else _n256x192_packed_scale_b_finish(b_pairs))
        next_sa = a_words if A_SHUFFLED else packed_a
        next_sb = b_words if B_SHUFFLED else packed_b
    else:
        for index in tl.static_range(18, 24):
            acc = _n256x192_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)
        tlx.async_load_wait_group(0)
        next_sa, next_sb = sa, sb
    return acc, next_a, next_b, next_sa, next_sb


@triton.jit
def _n256x192_fly_layout_step(A, B, SA, SB, pm, pn, panel, a_current, b_current, a_next, b_next, operands_a, operands_b,
                              words_a, words_b, acc, mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr,
                              M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, LOAD_FUTURE: tl.constexpr,
                              LOAD_NEXT: tl.constexpr, A_SHUFFLED: tl.constexpr = False,
                              B_SHUFFLED: tl.constexpr = False):
    tlx.workgroup_barrier()
    return _n256x192_cadence_compute(A, B, SA, SB, pm, pn, panel, a_current, b_current, a_next, b_next, operands_a,
                                     operands_b, words_a, words_b, acc, mma, dot_a, dot_b, M, N, K, LOAD_FUTURE,
                                     LOAD_NEXT, A_SHUFFLED, B_SHUFFLED)


@triton.jit
def _gemm_native_cadence_256x192(A, B, SA, SB, C, WORKSPACE, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                 BM: tl.constexpr, BN: tl.constexpr, WARPS_M: tl.constexpr, SPLIT_K: tl.constexpr,
                                 GROUP_M: tl.constexpr = 4, A_SHUFFLED: tl.constexpr = False,
                                 B_SHUFFLED: tl.constexpr = False):
    tl.static_assert(BM == 256 and BN == 192 and WARPS_M == 2 and SPLIT_K == 1)
    tl.static_assert(K % 128 == 0 and M % 128 == 0 and N % 192 == 0)
    gm: tl.constexpr = triton.cdiv(M, 256)
    gn: tl.constexpr = triton.cdiv(N, 192)
    total: tl.constexpr = gm * gn
    pid = tl.program_id(0)
    per_xcd: tl.constexpr = triton.cdiv(total, 8)
    tall_xcds: tl.constexpr = total % 8 or 8
    xcd, local_pid = pid % 8, pid // 8
    pid = tl.where(xcd < tall_xcds, xcd * per_xcd + local_pid,
                   tall_xcds * per_xcd + (xcd - tall_xcds) * (per_xcd - 1) + local_pid)
    first_m = pid // (GROUP_M * gn) * GROUP_M
    group_m = tl.minimum(gm - first_m, GROUP_M)
    pm = first_m + pid % (GROUP_M * gn) % group_m
    pn = pid % (GROUP_M * gn) // group_m
    tl.assume((pm >= 0) & (pm < gm))
    tl.assume((pn >= 0) & (pn < gn))
    mma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], False, [4, 1, 1])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    a_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 16], [4, 32],
                             [8, 64], [16, 0], [32, 0], [64, 0], [128, 0]]
    b_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 16], [4, 32],
                             [8, 64], [16, 0], [32, 0]]
    a_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in a_bases], [],
                                                               alignment=16)
    b_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in b_bases], [],
                                                               alignment=16)
    a0 = tlx.local_alloc((1, 256, 128), tlx.dtype_of(A), 1, layout=a_shared)
    b0 = tlx.local_alloc((1, 64, 128), tlx.dtype_of(B), 3, layout=b_shared)
    a1 = tlx.local_alloc((1, 256, 128), tlx.dtype_of(A), 1, layout=a_shared)
    b1 = tlx.local_alloc((1, 64, 128), tlx.dtype_of(B), 3, layout=b_shared)
    a_stages = tl.tuple([a0, a1])
    b_stages = tl.tuple([b0, b1])
    _n256x192_fly_layout_publish(A, B, pm, pn, 0, a0, b0, M, N, K)
    words_a, words_b = _n256x192_packed_scale_initial(SA, SB, pm, pn, M, N, K, A_SHUFFLED, B_SHUFFLED)
    tlx.async_load_wait_group(0)
    tlx.workgroup_barrier()
    operands_a = tl.tuple([_n256x192_fly_layout_read_a(a0, index % 4, index // 4, dot_a) for index in range(8)])
    operands_b = tl.tuple([_n256x192_fly_layout_read_b(b0, index % 3, index // 3, dot_b) for index in range(6)])
    if K // 128 > 1:
        _n256x192_fly_layout_publish(A, B, pm, pn, 1, a1, b1, M, N, K)
    acc = tl.tuple([tlx.zeros((4, 32, 32), tl.float32, layout=mma) for _ in range(12)])
    loop_end: tl.constexpr = max(((K // 128 - 2) // 2) * 2, 0)
    if loop_end > 0:
        for stage in tl.static_range(2):
            acc, operands_a, operands_b, words_a, words_b = _n256x192_fly_layout_step(
                A, B, SA, SB, pm, pn, stage, a_stages[stage], b_stages[stage], a_stages[1 - stage], b_stages[1 - stage],
                operands_a, operands_b, words_a, words_b, acc, mma, dot_a, dot_b, M, N, K, True, True, A_SHUFFLED,
                B_SHUFFLED)
        for panel in tl.range(2, loop_end, 2, num_stages=1):
            for stage in tl.static_range(2):
                acc, operands_a, operands_b, words_a, words_b = _n256x192_fly_layout_step(
                    A, B, SA, SB, pm, pn, panel + stage, a_stages[stage], b_stages[stage], a_stages[1 - stage],
                    b_stages[1 - stage], operands_a, operands_b, words_a, words_b, acc, mma, dot_a, dot_b, M, N, K,
                    True, True, A_SHUFFLED, B_SHUFFLED)
    for panel in tl.static_range(loop_end, K // 128):
        acc, operands_a, operands_b, words_a, words_b = _n256x192_fly_layout_step(
            A, B, SA, SB, pm, pn, panel, a_stages[panel % 2], b_stages[panel % 2], a_stages[1 - panel % 2],
            b_stages[1 - panel % 2], operands_a, operands_b, words_a, words_b, acc, mma, dot_a, dot_b, M, N, K,
            panel + 2 < K // 128, panel + 1 < K // 128, A_SHUFFLED, B_SHUFFLED)
    acc = tlx.amd_mfma_commit(acc)
    wave = tl.arange(0, 4)[:, None, None]
    rows32 = tl.arange(0, 32)[None, :, None]
    cols32 = tl.arange(0, 32)[None, None, :]
    if M == 2048 and N == 6144 and K == 22656:
        for mi in tl.static_range(4):
            for ni in tl.static_range(3):
                rows = pm * 256 + wave // 2 * 128 + mi * 32 + rows32
                cols = pn * 192 + wave % 2 * 96 + ni * 32 + cols32
                value = acc[mi * 3 + ni].to(tlx.dtype_of(C))
                offsets = tlx.require_layout(rows * N + cols, mma, late_address_compute=True)
                mask = tlx.require_layout((rows < M) & (cols < N), mma)
                tlx.buffer_store(value, C, offsets, mask=mask)
    else:
        for ni in tl.static_range(3):
            for mi in tl.static_range(4):
                rows = pm * 256 + wave // 2 * 128 + mi * 32 + rows32
                cols = pn * 192 + wave % 2 * 96 + ni * 32 + cols32
                value = acc[mi * 3 + ni].to(tlx.dtype_of(C))
                offsets = tlx.require_layout(rows * N + cols, mma, late_address_compute=True)
                mask = tlx.require_layout((rows < M) & (cols < N), mma)
                tlx.buffer_store(value, C, offsets, mask=mask)


# -----------------------------------------------------------------------------
# Native Split4
# -----------------------------------------------------------------------------


@triton.jit
def _ns4_packed_scale_pair(source, block, panel, IS_A: tl.constexpr, PAIR: tl.constexpr, LIMIT: tl.constexpr,
                           K: tl.constexpr):
    word_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((1, 2, 4, 8, 16, 0, 32, 64), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    lane_row = tl.arange(0, 32)[None, :, None]
    wave_rows = wave // 2 * 64 if IS_A else wave % 2 * 64
    rows = block + wave_rows + PAIR * 64 + lane_row
    word_offsets = ((rows // 128) * (K // 128) * 128 + rows % 32 * 4 + rows % 128 // 32)
    offsets = tlx.require_layout(word_offsets // 2, word_layout)
    mask = tlx.require_layout((rows < LIMIT) & (panel < K // 128), word_layout)
    words = tlx.buffer_load(source.to(tl.pointer_type(tl.uint64)) + panel * 64, offsets, mask=mask, other=0)
    return tlx.require_layout(words, word_layout)


@tl.core.builtin
def _ns4_packed_scale_permute(low, high, control, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        [high, low, control],
        {(tl.uint32, tl.uint32, tl.uint32): ("llvm.amdgcn.perm", tl.uint32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _ns4_packed_scale_word(low, high):
    layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    low = tlx.release_layout(low, relaxed=True)
    high = tlx.release_layout(high, relaxed=True)
    low = tlx.require_layout(tl.broadcast_to(low, [4, 32, 2]), layout)
    high = tlx.require_layout(tl.broadcast_to(high, [4, 32, 2]), layout)
    group = tlx.require_layout(tl.broadcast_to(tl.arange(0, 2)[None, None, :], [4, 32, 2]), layout)
    control = (group.to(tl.uint32) * 0x01010101 + 0x06020400).to(tl.uint32)
    return tlx.require_layout(_ns4_packed_scale_permute(low, high, control), layout)


@triton.jit
def _ns4_packed_scale_finish(a_pairs, b_pairs):
    a_words = tl.tuple([_ns4_packed_scale_word(pair.to(tl.uint32), (pair >> 32).to(tl.uint32)) for pair in a_pairs])
    b_words = tl.tuple([_ns4_packed_scale_word(pair.to(tl.uint32), (pair >> 32).to(tl.uint32)) for pair in b_pairs])
    return a_words, b_words


@triton.jit
def _ns4_preshuffled_scale(source, block, panel, IS_A: tl.constexpr, K: tl.constexpr):
    scale_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    row = tl.arange(0, 32)[None, :, None]
    group = tl.arange(0, 2)[None, None, :]
    lane = row + group * 32
    wave_group = wave // 2 if IS_A else wave % 2
    base_group = block * 2 + wave_group
    row_stride: tl.constexpr = K // 128 * 64
    k_offset = panel * 64
    return tl.tuple([
        tlx.buffer_load(
            source.to(tl.pointer_type(tl.uint32)),
            tlx.require_layout(
                base_group * row_stride + k_offset + lane,
                scale_layout,
            ),
            contiguity=1,
        )
    ])


@triton.jit
def _ns4_packed_scale_initial(SA, SB, pm, pn, first_panel, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                              SCALES_SHUFFLED: tl.constexpr = False):
    if SCALES_SHUFFLED:
        return (
            _ns4_preshuffled_scale(SA, pm, first_panel, True, K),
            _ns4_preshuffled_scale(SB, pn, first_panel, False, K),
        )
    a_pairs = tl.tuple([_ns4_packed_scale_pair(SA, pm * 128, first_panel, True, pair, M, K) for pair in range(1)])
    b_pairs = tl.tuple([_ns4_packed_scale_pair(SB, pn * 128, first_panel, False, pair, N, K) for pair in range(1)])
    return _ns4_packed_scale_finish(a_pairs, b_pairs)


@triton.jit
def _ns4_fly_layout_copy(source, row_base, panel, destination, ROWS: tl.constexpr, LIMIT: tl.constexpr,
                         K: tl.constexpr):
    copy_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2) + ((2, ) if ROWS >= 64 else
                                                         ()) + ((2, ) if ROWS >= 128 else
                                                                ()) + ((2, ) if ROWS >= 256 else ())),
        stride=((16, 32, 64, 128, 256, 512, 1024, 2048), (1, 2, 4, 8) + ((4096, ) if ROWS >= 64 else
                                                                         ()) + ((8192, ) if ROWS >= 128 else
                                                                                ()) + ((16384, ) if ROWS >= 256 else
                                                                                       ())),
    )
    indices = tl.arange(0, ROWS * 128)
    rows = row_base + indices // 128
    # Row bits 1:3 swizzle column bits 4:6 in the physical LDS copy.
    kk = (indices % 128) ^ ((indices & 0x700) >> 4)
    offsets = tlx.require_layout(tl.max_contiguous(tl.multiple_of(rows * K + kk, 16), 16), copy_layout)
    mask = tlx.require_layout((rows < LIMIT) & (panel < K // 128), copy_layout)
    other = tlx.require_layout(tl.zeros([ROWS * 128], tlx.dtype_of(source)), copy_layout)
    flat_shared: tl.constexpr = tlx.shared_linear_layout_encoding(
        [[1 << bit] for bit in range(12 + (ROWS >= 64) + (ROWS >= 128) + (ROWS >= 256))], [], alignment=16)
    destination = tlx.local_reinterpret(destination, tlx.dtype_of(source), [ROWS * 128], layout=flat_shared)
    tlx.buffer_load_to_local(destination, source + panel * 128, offsets, mask=mask, other=other)


@triton.jit
def _ns4_fly_layout_publish(A, B, pm, pn, panel, a_stage, b_stage, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                            B_CONSOLIDATED: tl.constexpr = False):
    _ns4_fly_layout_copy(A, pm * 128, panel, a_stage[0], 128, M, K)
    for plane in tl.static_range(2):
        if B_CONSOLIDATED:
            destination = tlx.local_slice(b_stage[0], [0, plane * 64, 0], [1, 64, 128])
        else:
            destination = b_stage[plane]
        _ns4_fly_layout_copy(B, pn * 128 + plane * 64, panel, destination, 64, N, K)
    tlx.async_load_commit_group()


@triton.jit
def _ns4_fly_layout_read_a(a_stage, tile: tl.constexpr, half: tl.constexpr, dot_a: tl.constexpr):
    wave_m = tlx.assume_uniform(tlx.thread_id(0) // 128)
    panel = tlx.local_slice(a_stage[0], [0, wave_m * 64 + tile * 32, half * 64], [1, 32, 64])
    return tlx.local_load(panel, layout=dot_a, relaxed=True)


@triton.jit
def _ns4_fly_layout_read_b(b_stage, tile: tl.constexpr, half: tl.constexpr, dot_b: tl.constexpr,
                           B_CONSOLIDATED: tl.constexpr = False):
    wave_n = tlx.assume_uniform(tlx.thread_id(0) // 64 % 2)
    first_row = wave_n * 64 + tile * 32
    if B_CONSOLIDATED:
        plane = tlx.local_view(b_stage, 0)
    else:
        plane = tlx.local_view(b_stage, first_row // 64)
    plane = tlx.local_trans(plane, (0, 2, 1))
    first_column = first_row if B_CONSOLIDATED else first_row % 64
    panel = tlx.local_slice(plane, [0, half * 64, first_column], [1, 64, 32])
    return tlx.local_load(panel, layout=dot_b, relaxed=True)


@tl.core.builtin
def _ns4_fly_layout_replace(values, index: tl.constexpr, value, _semantic=None):
    index = tl.core._unwrap_if_constexpr(index)
    items = list(values)
    items[index] = value
    return tl.tuple(items)


@tl.core.builtin
def _ns4_fly_layout_dot_one(acc, a, b, sa, sb, half: tl.constexpr, mi: tl.constexpr, ni: tl.constexpr,
                            mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr, _semantic=None):
    half, mi, ni = (tl.core._unwrap_if_constexpr(value) for value in (half, mi, ni))
    values = list(acc)
    a = tlx.require_layout(a, dot_a, pin=False, _semantic=_semantic)
    b = tlx.require_layout(b, dot_b, pin=False, _semantic=_semantic)
    scale_layout = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    sa = tlx.require_layout(sa, scale_layout, pin=False, _semantic=_semantic)
    sb = tlx.require_layout(sb, scale_layout, pin=False, _semantic=_semantic)
    c = tlx.require_layout(values[mi * 2 + ni], mma, pin=False, _semantic=_semantic)
    values[mi * 2 + ni] = tlx.amd_scheduled_mfma_scaled(
        a,
        b,
        c,
        sa,
        sb,
        "e4m3",
        "e4m3",
        accumulator_role="persistent",
        accumulator_register_class="agpr",
        a_scale_selector=half * 2 + mi % 2,
        b_scale_selector=half * 2 + ni % 2,
        _semantic=_semantic,
    )
    return tl.tuple(values)


@triton.jit
def _ns4_cadence_publish_chunk(A, B, pm, pn, panel, a_stage, b_stage, CHUNK: tl.constexpr, M: tl.constexpr,
                               N: tl.constexpr, K: tl.constexpr, B_CONSOLIDATED: tl.constexpr = False):
    if CHUNK < 4:
        destination = tlx.local_slice(a_stage[0], [0, CHUNK * 32, 0], [1, 32, 128])
        _ns4_fly_layout_copy(A, pm * 128 + CHUNK * 32, panel, destination, 32, M, K)
    else:
        column: tl.constexpr = CHUNK - 4
        if B_CONSOLIDATED:
            destination = tlx.local_slice(b_stage[0], [0, column * 32, 0], [1, 32, 128])
        else:
            destination = tlx.local_slice(b_stage[column // 2], [0, column % 2 * 32, 0], [1, 32, 128])
        _ns4_fly_layout_copy(B, pn * 128 + column * 32, panel, destination, 32, N, K)


@triton.jit
def _ns4_cadence_dot_at(acc, a, b, sa, sb, INDEX: tl.constexpr, mma: tl.constexpr, dot_a: tl.constexpr,
                        dot_b: tl.constexpr):
    half: tl.constexpr = INDEX // 4
    ni: tl.constexpr = INDEX % 4 // 2
    mi: tl.constexpr = INDEX % 2
    return _ns4_fly_layout_dot_one(acc, a[half * 2 + mi], b[half * 2 + ni], sa[mi // 2], sb[ni // 2], half, mi, ni, mma,
                                   dot_a, dot_b)


@triton.jit
def _ns4_cadence_read_at(a_stage, b_stage, a, b, INDEX: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr,
                         B_CONSOLIDATED: tl.constexpr = False):
    is_a: tl.constexpr = (False, True, True, False, False, True, True, False)[INDEX]
    half: tl.constexpr = (0, 0, 0, 0, 1, 1, 1, 1)[INDEX]
    tile: tl.constexpr = (0, 0, 1, 1, 0, 0, 1, 1)[INDEX]
    if is_a:
        a = _ns4_fly_layout_replace(a, half * 2 + tile, _ns4_fly_layout_read_a(a_stage, tile, half, dot_a))
    else:
        b = _ns4_fly_layout_replace(b, half * 2 + tile,
                                    _ns4_fly_layout_read_b(b_stage, tile, half, dot_b, B_CONSOLIDATED))
    return a, b


@triton.jit
def _ns4_cadence_compute(A, B, SA, SB, pm, pn, panel, a_current, b_current, a_next, b_next, a, b, sa, sb, acc,
                         mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                         K: tl.constexpr, LOAD_FUTURE: tl.constexpr, LOAD_NEXT: tl.constexpr,
                         B_CONSOLIDATED: tl.constexpr = False, SCALES_SHUFFLED: tl.constexpr = False):
    a_pairs = tl.tuple([])
    b_pairs = tl.tuple([])
    next_sa = sa
    next_sb = sb
    if LOAD_NEXT:
        for index in tl.static_range(1):
            acc = _ns4_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)
            tlx.amd_sched_barrier(0)
            if SCALES_SHUFFLED:
                next_sa = _ns4_preshuffled_scale(SA, pm, panel + 1, True, K)
            else:
                a_pairs += tl.tuple([_ns4_packed_scale_pair(SA, pm * 128, panel + 1, True, 0, M, K)])
            tlx.amd_sched_barrier(0)
    else:
        for index in tl.static_range(1):
            acc = _ns4_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)

    for group in tl.static_range(4):
        if LOAD_FUTURE:
            with tlx.warp_pipeline_stage("cadence_g2l", scope="intra_wave", pair=group):
                _ns4_cadence_publish_chunk(A, B, pm, pn, panel + 2, a_current, b_current, group * 2, M, N, K,
                                           B_CONSOLIDATED)
                _ns4_cadence_publish_chunk(A, B, pm, pn, panel + 2, a_current, b_current, group * 2 + 1, M, N, K,
                                           B_CONSOLIDATED)
                if group == 3:
                    tlx.async_load_commit_group()
            with tlx.warp_pipeline_stage("cadence_mfma", scope="intra_wave", pair=group):
                acc = _ns4_cadence_dot_at(acc, a, b, sa, sb, group + 1, mma, dot_a, dot_b)
        else:
            acc = _ns4_cadence_dot_at(acc, a, b, sa, sb, group + 1, mma, dot_a, dot_b)
        if LOAD_NEXT and group == 0:
            tlx.amd_sched_barrier(0)
            if SCALES_SHUFFLED:
                next_sb = _ns4_preshuffled_scale(SB, pn, panel + 1, False, K)
            else:
                b_pairs += tl.tuple([_ns4_packed_scale_pair(SB, pn * 128, panel + 1, False, 0, N, K)])
            tlx.amd_sched_barrier(0)

    next_a, next_b = a, b
    if LOAD_NEXT:
        tlx.async_load_wait_group(1 if LOAD_FUTURE else 0)
        for group in tl.static_range(4):
            if group > 0:
                tlx.amd_sched_barrier(0)
                acc = _ns4_cadence_dot_at(acc, a, b, sa, sb, 4 + group, mma, dot_a, dot_b)
                tlx.amd_sched_barrier(0)
            next_a, next_b = _ns4_cadence_read_at(a_next, b_next, next_a, next_b, group * 2, dot_a, dot_b,
                                                  B_CONSOLIDATED)
            next_a, next_b = _ns4_cadence_read_at(a_next, b_next, next_a, next_b, group * 2 + 1, dot_a, dot_b,
                                                  B_CONSOLIDATED)
        if not SCALES_SHUFFLED:
            next_sa, next_sb = _ns4_packed_scale_finish(a_pairs, b_pairs)
    else:
        for index in tl.static_range(5, 8):
            acc = _ns4_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)
        tlx.async_load_wait_group(0)
        next_sa, next_sb = sa, sb
    return acc, next_a, next_b, next_sa, next_sb


@triton.jit
def _ns4_fly_layout_step(A, B, SA, SB, pm, pn, panel, a_current, b_current, a_next, b_next, operands_a, operands_b,
                         words_a, words_b, acc, mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr,
                         M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, LOAD_FUTURE: tl.constexpr,
                         LOAD_NEXT: tl.constexpr, B_CONSOLIDATED: tl.constexpr = False,
                         SCALES_SHUFFLED: tl.constexpr = False):
    tlx.workgroup_barrier()
    return _ns4_cadence_compute(A, B, SA, SB, pm, pn, panel, a_current, b_current, a_next, b_next, operands_a,
                                operands_b, words_a, words_b, acc, mma, dot_a, dot_b, M, N, K, LOAD_FUTURE, LOAD_NEXT,
                                B_CONSOLIDATED, SCALES_SHUFFLED)


# -----------------------------------------------------------------------------
# Native Split8
# -----------------------------------------------------------------------------


@triton.jit
def _ns8_packed_scale_pair(source, block, panel, IS_A: tl.constexpr, PAIR: tl.constexpr, LIMIT: tl.constexpr,
                           K: tl.constexpr):
    word_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((1, 2, 4, 8, 16, 0, 32, 64), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    lane_row = tl.arange(0, 32)[None, :, None]
    wave_rows = wave // 2 * 64 if IS_A else wave % 2 * 128
    rows = block + wave_rows + PAIR * 64 + lane_row
    word_offsets = ((rows // 128) * (K // 128) * 128 + rows % 32 * 4 + rows % 128 // 32)
    offsets = tlx.require_layout(word_offsets // 2, word_layout)
    mask = tlx.require_layout((rows < LIMIT) & (panel < K // 128), word_layout)
    words = tlx.buffer_load(source.to(tl.pointer_type(tl.uint64)) + panel * 64, offsets, mask=mask, other=0)
    return tlx.require_layout(words, word_layout)


@triton.jit
def _ns8_packed_scale_shuffled(source, tile, panel, IS_A: tl.constexpr, GROUPS: tl.constexpr, K: tl.constexpr):
    scale_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    row = tl.arange(0, 32)[None, :, None]
    group = tl.arange(0, 2)[None, None, :]
    lane = row + group * 32
    wave_group = wave // 2 if IS_A else wave % 2
    base_group = (tile * 2 + wave_group) * GROUPS
    row_stride: tl.constexpr = K // 128 * 64
    k_offset = panel * 64
    return tl.tuple([
        tlx.buffer_load(
            source.to(tl.pointer_type(tl.uint32)),
            tlx.require_layout(
                (base_group + scale_group) * row_stride + k_offset + lane,
                scale_layout,
            ),
            contiguity=1,
        ) for scale_group in range(GROUPS)
    ])


@triton.jit
def _ns8_packed_scale_initial(SA, SB, pm, pn, first_panel, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                              A_SHUFFLED: tl.constexpr = False, B_SHUFFLED: tl.constexpr = False):
    if A_SHUFFLED:
        a_words = _ns8_packed_scale_shuffled(SA, pm, first_panel, True, 1, K)
    else:
        a_pairs = tl.tuple([_ns8_packed_scale_pair(SA, pm * 128, first_panel, True, pair, M, K) for pair in range(1)])
        a_words = _ns4_packed_scale_finish(a_pairs, tl.tuple([]))[0]
    if B_SHUFFLED:
        b_words = _ns8_packed_scale_shuffled(SB, pn, first_panel, False, 2, K)
    else:
        b_pairs = tl.tuple([_ns8_packed_scale_pair(SB, pn * 256, first_panel, False, pair, N, K) for pair in range(2)])
        b_words = _ns4_packed_scale_finish(tl.tuple([]), b_pairs)[1]
    return a_words, b_words


@triton.jit
def _ns8_fly_layout_publish(A, B, pm, pn, panel, a_stage, b_stage, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    _ns4_fly_layout_copy(A, pm * 128, panel, a_stage[0], 128, M, K)
    for plane in tl.static_range(4):
        _ns4_fly_layout_copy(B, pn * 256 + plane * 64, panel, b_stage[plane], 64, N, K)
    tlx.async_load_commit_group()


@triton.jit
def _ns8_fly_layout_read_b(b_stage, tile: tl.constexpr, half: tl.constexpr, dot_b: tl.constexpr):
    wave_n = tlx.assume_uniform(tlx.thread_id(0) // 64 % 2)
    first_row = wave_n * 128 + tile * 32
    plane = tlx.local_view(b_stage, first_row // 64)
    plane = tlx.local_trans(plane, (0, 2, 1))
    panel = tlx.local_slice(plane, [0, half * 64, first_row % 64], [1, 64, 32])
    return tlx.local_load(panel, layout=dot_b, relaxed=True)


@tl.core.builtin
def _ns8_fly_layout_dot_one(acc, a, b, sa, sb, half: tl.constexpr, mi: tl.constexpr, ni: tl.constexpr,
                            mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr, _semantic=None):
    half, mi, ni = (tl.core._unwrap_if_constexpr(value) for value in (half, mi, ni))
    values = list(acc)
    a = tlx.require_layout(a, dot_a, pin=False, _semantic=_semantic)
    b = tlx.require_layout(b, dot_b, pin=False, _semantic=_semantic)
    scale_layout = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    sa = tlx.require_layout(sa, scale_layout, pin=False, _semantic=_semantic)
    sb = tlx.require_layout(sb, scale_layout, pin=False, _semantic=_semantic)
    c = tlx.require_layout(values[mi * 4 + ni], mma, pin=False, _semantic=_semantic)
    values[mi * 4 + ni] = tlx.amd_scheduled_mfma_scaled(
        a,
        b,
        c,
        sa,
        sb,
        "e4m3",
        "e4m3",
        accumulator_role="persistent",
        accumulator_register_class="agpr",
        a_scale_selector=half * 2 + mi % 2,
        b_scale_selector=half * 2 + ni % 2,
        _semantic=_semantic,
    )
    return tl.tuple(values)


@triton.jit
def _ns8_cadence_publish_chunk(A, B, pm, pn, panel, a_stage, b_stage, CHUNK: tl.constexpr, M: tl.constexpr,
                               N: tl.constexpr, K: tl.constexpr):
    if CHUNK < 4:
        destination = tlx.local_slice(a_stage[0], [0, CHUNK * 32, 0], [1, 32, 128])
        _ns4_fly_layout_copy(A, pm * 128 + CHUNK * 32, panel, destination, 32, M, K)
    else:
        column: tl.constexpr = CHUNK - 4
        destination = tlx.local_slice(b_stage[column // 2], [0, column % 2 * 32, 0], [1, 32, 128])
        _ns4_fly_layout_copy(B, pn * 256 + column * 32, panel, destination, 32, N, K)


@triton.jit
def _ns8_cadence_dot_at(acc, a, b, sa, sb, INDEX: tl.constexpr, mma: tl.constexpr, dot_a: tl.constexpr,
                        dot_b: tl.constexpr):
    half: tl.constexpr = INDEX // 8
    ni: tl.constexpr = INDEX % 8 // 2
    mi: tl.constexpr = INDEX % 2
    return _ns8_fly_layout_dot_one(acc, a[half * 2 + mi], b[half * 4 + ni], sa[mi // 2], sb[ni // 2], half, mi, ni, mma,
                                   dot_a, dot_b)


@triton.jit
def _ns8_cadence_read_at(a_stage, b_stage, a, b, INDEX: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr):
    is_a: tl.constexpr = (False, True, True, False, False, False, False, False, True, False, True, False)[INDEX]
    half: tl.constexpr = (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)[INDEX]
    tile: tl.constexpr = (0, 0, 1, 1, 2, 3, 0, 1, 0, 2, 1, 3)[INDEX]
    if is_a:
        a = _ns4_fly_layout_replace(a, half * 2 + tile, _ns4_fly_layout_read_a(a_stage, tile, half, dot_a))
    else:
        b = _ns4_fly_layout_replace(b, half * 4 + tile, _ns8_fly_layout_read_b(b_stage, tile, half, dot_b))
    return a, b


@triton.jit
def _ns8_cadence_compute(A, B, SA, SB, pm, pn, panel, a_current, b_current, a_next, b_next, a, b, sa, sb, acc,
                         mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                         K: tl.constexpr, LOAD_FUTURE: tl.constexpr, LOAD_NEXT: tl.constexpr,
                         A_SHUFFLED: tl.constexpr = False, B_SHUFFLED: tl.constexpr = False):
    a_pairs = tl.tuple([])
    b_pairs = tl.tuple([])
    a_words = sa
    b_words = sb
    if LOAD_NEXT:
        for index in tl.static_range(5):
            acc = _ns8_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)
            if index == 0:
                tlx.amd_sched_barrier(0)
                if A_SHUFFLED:
                    a_words = _ns8_packed_scale_shuffled(SA, pm, panel + 1, True, 1, K)
                else:
                    a_pairs += tl.tuple([_ns8_packed_scale_pair(SA, pm * 128, panel + 1, True, 0, M, K)])
                tlx.amd_sched_barrier(0)
            elif index < 3:
                if B_SHUFFLED:
                    if index == 1:
                        tlx.amd_sched_barrier(0)
                        b_words = _ns8_packed_scale_shuffled(SB, pn, panel + 1, False, 2, K)
                        tlx.amd_sched_barrier(0)
                else:
                    tlx.amd_sched_barrier(0)
                    b_pairs += tl.tuple([_ns8_packed_scale_pair(SB, pn * 256, panel + 1, False, index - 1, N, K)])
                    tlx.amd_sched_barrier(0)
    else:
        for index in tl.static_range(5):
            acc = _ns8_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)

    for group in tl.static_range(6):
        if LOAD_FUTURE:
            with tlx.warp_pipeline_stage("cadence_g2l", scope="intra_wave", pair=group):
                _ns8_cadence_publish_chunk(A, B, pm, pn, panel + 2, a_current, b_current, group * 2, M, N, K)
                _ns8_cadence_publish_chunk(A, B, pm, pn, panel + 2, a_current, b_current, group * 2 + 1, M, N, K)
                if group == 5:
                    tlx.async_load_commit_group()
            with tlx.warp_pipeline_stage("cadence_mfma", scope="intra_wave", pair=group):
                acc = _ns8_cadence_dot_at(acc, a, b, sa, sb, group + 5, mma, dot_a, dot_b)
        else:
            acc = _ns8_cadence_dot_at(acc, a, b, sa, sb, group + 5, mma, dot_a, dot_b)

    next_a, next_b = a, b
    if LOAD_NEXT:
        tlx.async_load_wait_group(1 if LOAD_FUTURE else 0)
        for group in tl.static_range(6):
            if group > 0:
                tlx.amd_sched_barrier(0)
                acc = _ns8_cadence_dot_at(acc, a, b, sa, sb, 10 + group, mma, dot_a, dot_b)
                tlx.amd_sched_barrier(0)
            next_a, next_b = _ns8_cadence_read_at(a_next, b_next, next_a, next_b, group * 2, dot_a, dot_b)
            next_a, next_b = _ns8_cadence_read_at(a_next, b_next, next_a, next_b, group * 2 + 1, dot_a, dot_b)
        packed_a = (tl.tuple([]) if A_SHUFFLED else _ns4_packed_scale_finish(a_pairs, tl.tuple([]))[0])
        packed_b = (tl.tuple([]) if B_SHUFFLED else _ns4_packed_scale_finish(tl.tuple([]), b_pairs)[1])
        next_sa = a_words if A_SHUFFLED else packed_a
        next_sb = b_words if B_SHUFFLED else packed_b
    else:
        for index in tl.static_range(11, 16):
            acc = _ns8_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)
        tlx.async_load_wait_group(0)
        next_sa, next_sb = sa, sb
    return acc, next_a, next_b, next_sa, next_sb


@triton.jit
def _ns8_fly_layout_step(A, B, SA, SB, pm, pn, panel, a_current, b_current, a_next, b_next, operands_a, operands_b,
                         words_a, words_b, acc, mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr,
                         M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, LOAD_FUTURE: tl.constexpr,
                         LOAD_NEXT: tl.constexpr, A_SHUFFLED: tl.constexpr = False, B_SHUFFLED: tl.constexpr = False):
    tlx.workgroup_barrier()
    return _ns8_cadence_compute(A, B, SA, SB, pm, pn, panel, a_current, b_current, a_next, b_next, operands_a,
                                operands_b, words_a, words_b, acc, mma, dot_a, dot_b, M, N, K, LOAD_FUTURE, LOAD_NEXT,
                                A_SHUFFLED, B_SHUFFLED)


@triton.jit
def _ns8_split_major_tile(raw_pid, gm: tl.constexpr, gn: tl.constexpr, SPLIT_K: tl.constexpr):
    # Workgroups are dispatched round-robin across the eight XCDs. Giving each
    # XCD a whole K split lets its L2 serve every A/B panel to all tiles of
    # that split instead of streaming several unrelated K ranges.
    total: tl.constexpr = gm * gn
    xcds_per_split: tl.constexpr = 8 // SPLIT_K
    tiles_per_xcd: tl.constexpr = total // xcds_per_split
    tl.static_assert(8 % SPLIT_K == 0 and total % xcds_per_split == 0)
    xcd = raw_pid % 8
    split_id = xcd // xcds_per_split
    tile = (xcd % xcds_per_split) * tiles_per_xcd + raw_pid // 8
    tl.assume((split_id >= 0) & (split_id < SPLIT_K))
    return tile % gm, tile // gm, split_id


@triton.jit
def _gemm_native_split8(A, B, SA, SB, C, WORKSPACE, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                        BN: tl.constexpr, WARPS_M: tl.constexpr, SPLIT_K: tl.constexpr,
                        A_SHUFFLED: tl.constexpr = False, B_SHUFFLED: tl.constexpr = False):
    tl.static_assert(BM == 128 and BN == 256 and WARPS_M == 2 and (SPLIT_K == 1 or SPLIT_K == 8))
    tl.static_assert(K % (128 * SPLIT_K) == 0 and M % 128 == 0 and N % 128 == 0)
    piece_panels: tl.constexpr = K // (128 * SPLIT_K)
    tl.static_assert(piece_panels >= 2 and piece_panels % 2 == 0)
    gm: tl.constexpr = triton.cdiv(M, 128)
    gn: tl.constexpr = triton.cdiv(N, 256)
    raw_pid = tl.program_id(0)
    pm, pn, split_id = _ns8_split_major_tile(raw_pid, gm, gn, SPLIT_K)
    first_panel = split_id * piece_panels
    tl.assume((pm >= 0) & (pm < gm))
    tl.assume((pn >= 0) & (pn < gn))
    mma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], False, [4, 1, 1])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    a_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 16], [4, 32],
                             [8, 64], [16, 0], [32, 0], [64, 0]]
    b_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 16], [4, 32],
                             [8, 64], [16, 0], [32, 0]]
    a_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in a_bases], [],
                                                               alignment=16)
    b_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in b_bases], [],
                                                               alignment=16)
    a0 = tlx.local_alloc((1, 128, 128), tlx.dtype_of(A), 1, layout=a_shared)
    b0 = tlx.local_alloc((1, 64, 128), tlx.dtype_of(B), 4, layout=b_shared)
    a1 = tlx.local_alloc((1, 128, 128), tlx.dtype_of(A), 1, layout=a_shared)
    b1 = tlx.local_alloc((1, 64, 128), tlx.dtype_of(B), 4, layout=b_shared)
    a_stages = tl.tuple([a0, a1])
    b_stages = tl.tuple([b0, b1])
    _ns8_fly_layout_publish(A, B, pm, pn, first_panel, a0, b0, M, N, K)
    words_a, words_b = _ns8_packed_scale_initial(SA, SB, pm, pn, first_panel, M, N, K, A_SHUFFLED, B_SHUFFLED)
    tlx.async_load_wait_group(0)
    tlx.workgroup_barrier()
    operands_a = tl.tuple([_ns4_fly_layout_read_a(a0, index % 2, index // 2, dot_a) for index in range(4)])
    operands_b = tl.tuple([_ns8_fly_layout_read_b(b0, index % 4, index // 4, dot_b) for index in range(8)])
    if piece_panels > 1:
        _ns8_fly_layout_publish(A, B, pm, pn, first_panel + 1, a1, b1, M, N, K)
    acc = tl.tuple([tlx.zeros((4, 32, 32), tl.float32, layout=mma) for _ in range(8)])
    loop_end: tl.constexpr = max(((piece_panels - 2) // 2) * 2, 0)
    if loop_end > 0:
        for stage in tl.static_range(2):
            acc, operands_a, operands_b, words_a, words_b = _ns8_fly_layout_step(
                A, B, SA, SB, pm, pn, first_panel + stage, a_stages[stage], b_stages[stage], a_stages[1 - stage],
                b_stages[1 - stage], operands_a, operands_b, words_a, words_b, acc, mma, dot_a, dot_b, M, N, K, True,
                True, A_SHUFFLED, B_SHUFFLED)
        for panel in tl.range(2, loop_end, 2, num_stages=1):
            for stage in tl.static_range(2):
                acc, operands_a, operands_b, words_a, words_b = _ns8_fly_layout_step(
                    A, B, SA, SB, pm, pn, first_panel + panel + stage, a_stages[stage], b_stages[stage],
                    a_stages[1 - stage], b_stages[1 - stage], operands_a, operands_b, words_a, words_b, acc, mma, dot_a,
                    dot_b, M, N, K, True, True, A_SHUFFLED, B_SHUFFLED)
    for panel in tl.static_range(loop_end, piece_panels):
        acc, operands_a, operands_b, words_a, words_b = _ns8_fly_layout_step(
            A, B, SA, SB, pm, pn, first_panel + panel, a_stages[panel % 2], b_stages[panel % 2],
            a_stages[1 - panel % 2], b_stages[1 - panel % 2], operands_a, operands_b, words_a, words_b, acc, mma, dot_a,
            dot_b, M, N, K, panel + 2 < piece_panels, panel + 1 < piece_panels, A_SHUFFLED, B_SHUFFLED)
    acc = tlx.amd_mfma_commit(acc)
    target = C if SPLIT_K == 1 else WORKSPACE + split_id * M * N
    wave = tl.arange(0, 4)[:, None, None]
    rows32 = tl.arange(0, 32)[None, :, None]
    cols32 = tl.arange(0, 32)[None, None, :]
    store_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
        stride=((4, 8, 16, 32, 64, 128, 1024, 2048), (1, 2, 256, 512)),
    )
    for ni in tl.static_range(4):
        for mi in tl.static_range(2):
            rows = pm * 128 + wave // 2 * 64 + mi * 32 + rows32
            cols = pn * 256 + wave % 2 * 128 + ni * 32 + cols32
            value = tlx.require_layout(acc[mi * 4 + ni].to(tlx.dtype_of(target)), store_layout,
                                       late_address_compute=True)
            offsets = tlx.require_layout(rows * N + cols, store_layout)
            mask = tlx.require_layout((rows < M) & (cols < N), store_layout)
            tlx.buffer_store(value, target, offsets, mask=mask)


# -----------------------------------------------------------------------------
# Native Cadence 128X128
# -----------------------------------------------------------------------------


@triton.jit
def _n128x128_pack_bf16_pairs(values):
    return tl.inline_asm_elementwise(
        "v_cvt_pk_bf16_f32 $0, $1, $2",
        "=v,v,v",
        [values],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _gemm_native_128x128_s1(A, B, SA, SB, C, WORKSPACE, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                            BM: tl.constexpr, BN: tl.constexpr, WARPS_M: tl.constexpr, SPLIT_K: tl.constexpr,
                            SCALES_SHUFFLED: tl.constexpr = False):
    tl.static_assert(BM == 128 and BN == 128 and WARPS_M == 2 and SPLIT_K == 1)
    tl.static_assert(K % 128 == 0 and M % 128 == 0 and N % 128 == 0)
    piece_panels: tl.constexpr = K // (128 * SPLIT_K)
    tl.static_assert(piece_panels >= 2 and piece_panels % 2 == 0)
    gm: tl.constexpr = triton.cdiv(M, 128)
    gn: tl.constexpr = triton.cdiv(N, 128)
    total: tl.constexpr = gm * gn
    raw_pid = tl.program_id(0)
    split_id = raw_pid // total
    pid = raw_pid % total
    tl.assume((split_id >= 0) & (split_id < SPLIT_K))
    first_panel = split_id * piece_panels
    per_xcd: tl.constexpr = triton.cdiv(total, 8)
    tall_xcds: tl.constexpr = total % 8 or 8
    xcd, local_pid = pid % 8, pid // 8
    pid = tl.where(xcd < tall_xcds, xcd * per_xcd + local_pid,
                   tall_xcds * per_xcd + (xcd - tall_xcds) * (per_xcd - 1) + local_pid)
    first_m = pid // (4 * gn) * 4
    group_m = tl.minimum(gm - first_m, 4)
    pm = first_m + pid % (4 * gn) % group_m
    pn = pid % (4 * gn) // group_m
    tl.assume((pm >= 0) & (pm < gm))
    tl.assume((pn >= 0) & (pn < gn))
    mma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], False, [4, 1, 1])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    a_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 16], [4, 32],
                             [8, 64], [16, 0], [32, 0], [64, 0]]
    b_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 16], [4, 32],
                             [8, 64], [16, 0], [32, 0], [64, 0]]
    a_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in a_bases], [],
                                                               alignment=16)
    b_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in b_bases], [],
                                                               alignment=16)
    output_shared: tl.constexpr = tlx.shared_linear_layout_encoding(
        [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0],
         [0, 16, 0], [0, 32, 0], [1, 0, 0], [2, 0, 0]], [], alignment=16)
    shared_storage = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
    a0 = tlx.local_alloc((1, 128, 128), tlx.dtype_of(A), 1, reuse=shared_storage, layout=a_shared)
    b0 = tlx.local_alloc((1, 128, 128), tlx.dtype_of(B), 1, reuse=shared_storage, layout=b_shared)
    a1 = tlx.local_alloc((1, 128, 128), tlx.dtype_of(A), 1, reuse=shared_storage, layout=a_shared)
    b1 = tlx.local_alloc((1, 128, 128), tlx.dtype_of(B), 1, reuse=shared_storage, layout=b_shared)
    output_scratch = tlx.local_alloc((4, 64, 64), tl.bfloat16, 1, reuse=shared_storage, layout=output_shared)
    shared_storage.set_buffer_overlap(
        tlx.reuse_group(
            tlx.reuse_group(
                a0,
                b0,
                a1,
                b1,
                group_type=tlx.reuse_group_type.distinct,
            ),
            output_scratch,
            group_type=tlx.reuse_group_type.shared,
        ))
    a_stages = tl.tuple([a0, a1])
    b_stages = tl.tuple([b0, b1])
    _ns4_fly_layout_publish(A, B, pm, pn, first_panel, a0, b0, M, N, K, True)
    words_a, words_b = _ns4_packed_scale_initial(SA, SB, pm, pn, first_panel, M, N, K, SCALES_SHUFFLED)
    tlx.async_load_wait_group(0)
    tlx.workgroup_barrier()
    operands_a = tl.tuple([_ns4_fly_layout_read_a(a0, index % 2, index // 2, dot_a) for index in range(4)])
    operands_b = tl.tuple([_ns4_fly_layout_read_b(b0, index % 2, index // 2, dot_b, True) for index in range(4)])
    if piece_panels > 1:
        _ns4_fly_layout_publish(A, B, pm, pn, first_panel + 1, a1, b1, M, N, K, True)
    acc = tl.tuple([tlx.zeros((4, 32, 32), tl.float32, layout=mma) for _ in range(4)])
    for panel in tl.static_range(piece_panels):
        acc, operands_a, operands_b, words_a, words_b = _ns4_fly_layout_step(
            A, B, SA, SB, pm, pn, first_panel + panel, a_stages[panel % 2], b_stages[panel % 2],
            a_stages[1 - panel % 2], b_stages[1 - panel % 2], operands_a, operands_b, words_a, words_b, acc, mma, dot_a,
            dot_b, M, N, K, panel + 2 < piece_panels, panel + 1 < piece_panels, True, SCALES_SHUFFLED)
    acc = tlx.amd_mfma_commit(acc)
    target = C
    store_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2)),
        stride=((8, 16, 32, 64, 128, 256, 4096, 8192), (1, 2, 4, 512, 1024, 2048)),
    )
    for index in tl.static_range(4):
        output_fragment = tlx.local_slice(tlx.local_view(output_scratch, 0), [0, index // 2 * 32, index % 2 * 32],
                                          [4, 32, 32])
        tlx.local_store(output_fragment, _n128x128_pack_bf16_pairs(acc[index]))
    tlx.workgroup_barrier()
    value = tlx.local_load(tlx.local_view(output_scratch, 0), layout=store_layout)
    value = tlx.require_layout(value, store_layout)
    wave = tl.arange(0, 4)[:, None, None]
    rows = pm * 128 + wave // 2 * 64 + tl.arange(0, 64)[None, :, None]
    cols = pn * 128 + wave % 2 * 64 + tl.arange(0, 64)[None, None, :]
    offsets = tlx.require_layout(rows * N + cols, store_layout)
    mask = tlx.require_layout((rows < M) & (cols < N), store_layout)
    tlx.buffer_store(value, target, offsets, mask=mask, cache=".cs", contiguity=8)


# -----------------------------------------------------------------------------
# Native Cadence 128X192
# -----------------------------------------------------------------------------


@triton.jit
def _n128x192_packed_scale_a_128_shuffled(source, pm, panel, K: tl.constexpr):
    scale_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    row = tl.arange(0, 32)[None, :, None]
    group = tl.arange(0, 2)[None, None, :]
    lane = row + group * 32
    base_group = pm * 2 + wave // 2
    row_stride: tl.constexpr = K // 128 * 64
    k_offset = panel * 64
    return tl.tuple([
        tlx.buffer_load(
            source.to(tl.pointer_type(tl.uint32)),
            tlx.require_layout(
                base_group * row_stride + k_offset + lane,
                scale_layout,
            ),
            contiguity=1,
        )
    ])


@triton.jit
def _n128x192_packed_scale_a_128(source, block, panel, LIMIT: tl.constexpr, K: tl.constexpr):
    word_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((1, 2, 4, 8, 16, 0, 32, 64), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    lane_row = tl.arange(0, 32)[None, :, None]
    rows = block + wave // 2 * 64 + lane_row
    word_offsets = ((rows // 128) * (K // 128) * 128 + rows % 32 * 4 + rows % 128 // 32)
    offsets = tlx.require_layout(word_offsets // 2, word_layout)
    mask = tlx.require_layout((rows < LIMIT) & (panel < K // 128), word_layout)
    words = tlx.buffer_load(
        source.to(tl.pointer_type(tl.uint64)) + panel * 64,
        offsets,
        mask=mask,
        other=0,
    )
    return tlx.require_layout(words, word_layout)


@triton.jit
def _n128x192_packed_scales_128(SA, SB, pm, pn, panel, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                A_SHUFFLED: tl.constexpr = False, B_SHUFFLED: tl.constexpr = False):
    if A_SHUFFLED:
        a_words = _n128x192_packed_scale_a_128_shuffled(SA, pm, panel, K)
    else:
        a_pairs = tl.tuple([_n128x192_packed_scale_a_128(SA, pm * 128, panel, M, K)])
        a_words = _n256x192_packed_scale_a_finish(a_pairs)
    if B_SHUFFLED:
        b_words = _n256x192_packed_scale_b_shuffled(SB, pn, panel, K)
    else:
        b_pairs = tl.tuple([_n256x192_packed_scale_pair(SB, pn * 192, panel, False, pair, N, K) for pair in range(2)])
        b_words = _n256x192_packed_scale_b_finish(b_pairs)
    return a_words, b_words


@triton.jit
def _n128x192_publish_128x192(A, B, pm, pn, panel, a_stage, b_stage, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    _n256x192_fly_layout_copy(A, pm * 128, panel, a_stage[0], 128, M, K)
    for plane in tl.static_range(3):
        _n256x192_fly_layout_copy(B, pn * 192 + plane * 64, panel, b_stage[plane], 64, N, K)
    tlx.async_load_commit_group()


@triton.jit
def _n128x192_publish_chunk_128x192(A, B, pm, pn, panel, a_stage, b_stage, chunk: tl.constexpr, M: tl.constexpr,
                                    N: tl.constexpr, K: tl.constexpr):
    if chunk < 4:
        destination = tlx.local_slice(a_stage[0], [0, chunk * 32, 0], [1, 32, 128])
        _n256x192_fly_layout_copy(A, pm * 128 + chunk * 32, panel, destination, 32, M, K)
    else:
        column: tl.constexpr = chunk - 4
        destination = tlx.local_slice(
            b_stage[column // 2],
            [0, column % 2 * 32, 0],
            [1, 32, 128],
        )
        _n256x192_fly_layout_copy(B, pn * 192 + column * 32, panel, destination, 32, N, K)


@triton.jit
def _n128x192_read_a_128(a_stage, tile: tl.constexpr, half: tl.constexpr, dot_a: tl.constexpr):
    wave_m = tlx.assume_uniform(tlx.thread_id(0) // 128)
    panel = tlx.local_slice(
        a_stage[0],
        [0, wave_m * 64 + tile * 32, half * 64],
        [1, 32, 64],
    )
    return tlx.local_load(panel, layout=dot_a, relaxed=True)


@tl.core.builtin
def _n128x192_dot_one_128(acc, a, b, sa, sb, half: tl.constexpr, mi: tl.constexpr, ni: tl.constexpr, mma: tl.constexpr,
                          dot_a: tl.constexpr, dot_b: tl.constexpr, _semantic=None):
    half, mi, ni = (tl.core._unwrap_if_constexpr(value) for value in (half, mi, ni))
    values = list(acc)
    a = tlx.require_layout(a, dot_a, pin=False, _semantic=_semantic)
    b = tlx.require_layout(b, dot_b, pin=False, _semantic=_semantic)
    scale_layout = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    sa = tlx.require_layout(sa, scale_layout, pin=False, _semantic=_semantic)
    sb = tlx.require_layout(sb, scale_layout, pin=False, _semantic=_semantic)
    index = mi * 3 + ni
    c = tlx.require_layout(values[index], mma, pin=False, _semantic=_semantic)
    values[index] = tlx.amd_scheduled_mfma_scaled(
        a,
        b,
        c,
        sa,
        sb,
        "e4m3",
        "e4m3",
        accumulator_role="persistent",
        accumulator_register_class="agpr",
        a_scale_selector=half * 2 + mi % 2,
        b_scale_selector=half * 2 + ni % 2,
        _semantic=_semantic,
    )
    return tl.tuple(values)


@triton.jit
def _n128x192_dot_at_128(acc, a, b, sa, sb, index: tl.constexpr, mma: tl.constexpr, dot_a: tl.constexpr,
                         dot_b: tl.constexpr):
    half: tl.constexpr = index // 6
    ni: tl.constexpr = index % 6 // 2
    mi: tl.constexpr = index % 2
    return _n128x192_dot_one_128(
        acc,
        a[half * 2 + mi],
        b[half * 3 + ni],
        sa[0],
        sb[ni // 2],
        half,
        mi,
        ni,
        mma,
        dot_a,
        dot_b,
    )


@triton.jit
def _n128x192_read_at_128(a_stage, b_stage, a, b, index: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr):
    is_a: tl.constexpr = (
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
    )[index]
    half: tl.constexpr = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)[index]
    tile: tl.constexpr = (0, 1, 0, 1, 2, 0, 0, 1, 1, 2)[index]
    if is_a:
        a = _n256x192_fly_layout_replace(
            a,
            half * 2 + tile,
            _n128x192_read_a_128(a_stage, tile, half, dot_a),
        )
    else:
        b = _n256x192_fly_layout_replace(
            b,
            half * 3 + tile,
            _n256x192_fly_layout_read_b(b_stage, tile, half, dot_b),
        )
    return a, b


@triton.jit
def _n128x192_step_128x192(A, B, SA, SB, pm, pn, panel, current_a, current_b, next_a, next_b, operands_a, operands_b,
                           scales_a, scales_b, acc, mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr,
                           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, load_future: tl.constexpr,
                           load_next: tl.constexpr, A_SHUFFLED: tl.constexpr = False, B_SHUFFLED: tl.constexpr = False):
    tlx.workgroup_barrier()
    if load_next:
        if A_SHUFFLED and B_SHUFFLED:
            for index in tl.static_range(2):
                acc = _n128x192_dot_at_128(acc, operands_a, operands_b, scales_a, scales_b, index, mma, dot_a, dot_b)
            acc = _n128x192_dot_at_128(acc, operands_a, operands_b, scales_a, scales_b, 2, mma, dot_a, dot_b)
            tlx.amd_sched_barrier(0)
            next_scales_a, next_scales_b = _n128x192_packed_scales_128(SA, SB, pm, pn, panel + 1, M, N, K, True, True)
            tlx.amd_sched_barrier(0)
        else:
            acc = _n128x192_dot_at_128(acc, operands_a, operands_b, scales_a, scales_b, 0, mma, dot_a, dot_b)
            tlx.amd_sched_barrier(0)
            next_a_pair = _n128x192_packed_scale_a_128(SA, pm * 128, panel + 1, M, K)
            tlx.amd_sched_barrier(0)
            acc = _n128x192_dot_at_128(acc, operands_a, operands_b, scales_a, scales_b, 1, mma, dot_a, dot_b)
            tlx.amd_sched_barrier(0)
            next_b0 = _n256x192_packed_scale_pair(SB, pn * 192, panel + 1, False, 0, N, K)
            tlx.amd_sched_barrier(0)
            acc = _n128x192_dot_at_128(acc, operands_a, operands_b, scales_a, scales_b, 2, mma, dot_a, dot_b)
            tlx.amd_sched_barrier(0)
            next_b1 = _n256x192_packed_scale_pair(SB, pn * 192, panel + 1, False, 1, N, K)
            tlx.amd_sched_barrier(0)
            next_scales_a, next_scales_b = scales_a, scales_b
    else:
        for index in tl.static_range(3):
            acc = _n128x192_dot_at_128(acc, operands_a, operands_b, scales_a, scales_b, index, mma, dot_a, dot_b)
        next_scales_a, next_scales_b = scales_a, scales_b

    for group in tl.static_range(4):
        if load_future:
            with tlx.warp_pipeline_stage("cadence128_g2l", scope="intra_wave", pair=group):
                if group == 0:
                    _n128x192_publish_chunk_128x192(A, B, pm, pn, panel + 2, current_a, current_b, 0, M, N, K)
                    _n128x192_publish_chunk_128x192(A, B, pm, pn, panel + 2, current_a, current_b, 1, M, N, K)
                    _n128x192_publish_chunk_128x192(A, B, pm, pn, panel + 2, current_a, current_b, 2, M, N, K)
                elif group == 1:
                    _n128x192_publish_chunk_128x192(A, B, pm, pn, panel + 2, current_a, current_b, 3, M, N, K)
                    _n128x192_publish_chunk_128x192(A, B, pm, pn, panel + 2, current_a, current_b, 4, M, N, K)
                    _n128x192_publish_chunk_128x192(A, B, pm, pn, panel + 2, current_a, current_b, 5, M, N, K)
                elif group == 2:
                    _n128x192_publish_chunk_128x192(A, B, pm, pn, panel + 2, current_a, current_b, 6, M, N, K)
                    _n128x192_publish_chunk_128x192(A, B, pm, pn, panel + 2, current_a, current_b, 7, M, N, K)
                else:
                    _n128x192_publish_chunk_128x192(A, B, pm, pn, panel + 2, current_a, current_b, 8, M, N, K)
                    _n128x192_publish_chunk_128x192(A, B, pm, pn, panel + 2, current_a, current_b, 9, M, N, K)
                    tlx.async_load_commit_group()
            with tlx.warp_pipeline_stage("cadence128_mfma", scope="intra_wave", pair=group):
                acc = _n128x192_dot_at_128(acc, operands_a, operands_b, scales_a, scales_b, group + 3, mma, dot_a,
                                           dot_b)
        else:
            acc = _n128x192_dot_at_128(acc, operands_a, operands_b, scales_a, scales_b, group + 3, mma, dot_a, dot_b)
        if load_next and not A_SHUFFLED and not B_SHUFFLED:
            if group == 0:
                next_scales_a = _n256x192_packed_scale_a_finish(tl.tuple([next_a_pair]))
            elif group == 1:
                next_scales_b = _n256x192_packed_scale_b_finish(tl.tuple([next_b0, next_b1]))

    next_operands_a, next_operands_b = operands_a, operands_b
    if load_next:
        tlx.async_load_wait_group(1 if load_future else 0)
        for group in tl.static_range(5):
            acc = _n128x192_dot_at_128(acc, operands_a, operands_b, scales_a, scales_b, group + 7, mma, dot_a, dot_b)
            tlx.amd_sched_barrier(0)
            next_operands_a, next_operands_b = _n128x192_read_at_128(next_a, next_b, next_operands_a, next_operands_b,
                                                                     group * 2, dot_a, dot_b)
            next_operands_a, next_operands_b = _n128x192_read_at_128(next_a, next_b, next_operands_a, next_operands_b,
                                                                     group * 2 + 1, dot_a, dot_b)
            if group + 1 < 5:
                tlx.amd_sched_barrier(0)
    else:
        for index in tl.static_range(7, 12):
            acc = _n128x192_dot_at_128(acc, operands_a, operands_b, scales_a, scales_b, index, mma, dot_a, dot_b)
        tlx.async_load_wait_group(0)
    return (
        acc,
        next_operands_a,
        next_operands_b,
        next_scales_a,
        next_scales_b,
    )


@triton.jit
def _n128x192_store_acc_128x192(value, C, pm, pn, mi: tl.constexpr, ni: tl.constexpr, N: tl.constexpr,
                                STREAMING_STORE: tl.constexpr):
    index: tl.constexpr = mi * 3 + ni
    wave = tl.arange(0, 4)[:, None, None]
    rows32 = tlx.rematerialized_range(0, 32, 20, placement=index)[None, :, None]
    cols32 = tlx.rematerialized_range(0, 32, 21, placement=index)[None, None, :]
    rows = pm * 128 + wave // 2 * 64 + mi * 32 + rows32
    cols = pn * 192 + wave % 2 * 96 + ni * 32 + cols32
    offsets = tlx.require_layout(
        rows * N + cols,
        tlx.amd_mfma_layout(4, [32, 32, 64], False, [4, 1, 1]),
        late_address_compute=True,
    )
    tlx.buffer_store(
        value.to(tlx.dtype_of(C)),
        C,
        offsets,
        cache=".cs" if STREAMING_STORE else "",
    )


@triton.jit
def _gemm_native_cadence_128x192(A, B, SA, SB, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                 GROUP_M: tl.constexpr, STREAMING_STORE: tl.constexpr = False,
                                 MODEL_XCD: tl.constexpr = True, A_SHUFFLED: tl.constexpr = False,
                                 B_SHUFFLED: tl.constexpr = False):
    tl.static_assert(K % 128 == 0 and M % 128 == 0 and N % 192 == 0)
    gm: tl.constexpr = M // 128
    gn: tl.constexpr = N // 192
    total: tl.constexpr = gm * gn
    pid = tl.program_id(0)
    if MODEL_XCD:
        per_xcd: tl.constexpr = triton.cdiv(total, 8)
        tall_xcds: tl.constexpr = total % 8 or 8
        xcd = pid % 8
        local_pid = pid // 8
        pid = tl.where(
            xcd < tall_xcds,
            xcd * per_xcd + local_pid,
            tall_xcds * per_xcd + (xcd - tall_xcds) * (per_xcd - 1) + local_pid,
        )
    first_m = pid // (GROUP_M * gn) * GROUP_M
    group_m = tl.minimum(gm - first_m, GROUP_M)
    pm = first_m + pid % (GROUP_M * gn) % group_m
    pn = pid % (GROUP_M * gn) // group_m
    mma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], False, [4, 1, 1])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    a_bases: tl.constexpr = [
        [0, 1],
        [0, 2],
        [0, 4],
        [0, 8],
        [0, 16],
        [0, 32],
        [0, 64],
        [1, 0],
        [2, 16],
        [4, 32],
        [8, 64],
        [16, 0],
        [32, 0],
        [64, 0],
    ]
    b_bases: tl.constexpr = [
        [0, 1],
        [0, 2],
        [0, 4],
        [0, 8],
        [0, 16],
        [0, 32],
        [0, 64],
        [1, 0],
        [2, 16],
        [4, 32],
        [8, 64],
        [16, 0],
        [32, 0],
    ]
    a_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in a_bases], [],
                                                               alignment=16)
    b_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in b_bases], [],
                                                               alignment=16)
    a0 = tlx.local_alloc((1, 128, 128), tlx.dtype_of(A), 1, layout=a_shared)
    b0 = tlx.local_alloc((1, 64, 128), tlx.dtype_of(B), 3, layout=b_shared)
    a1 = tlx.local_alloc((1, 128, 128), tlx.dtype_of(A), 1, layout=a_shared)
    b1 = tlx.local_alloc((1, 64, 128), tlx.dtype_of(B), 3, layout=b_shared)
    a_stages = tl.tuple([a0, a1])
    b_stages = tl.tuple([b0, b1])
    _n128x192_publish_128x192(A, B, pm, pn, 0, a0, b0, M, N, K)
    scales_a, scales_b = _n128x192_packed_scales_128(SA, SB, pm, pn, 0, M, N, K, A_SHUFFLED, B_SHUFFLED)
    if K // 128 > 1:
        _n128x192_publish_128x192(A, B, pm, pn, 1, a1, b1, M, N, K)
    tlx.async_load_wait_group(1 if K // 128 > 1 else 0)
    tlx.workgroup_barrier()
    operands_a = tl.tuple([_n128x192_read_a_128(a0, index % 2, index // 2, dot_a) for index in range(4)])
    operands_b = tl.tuple([_n256x192_fly_layout_read_b(b0, index % 3, index // 3, dot_b) for index in range(6)])
    acc = tl.tuple([tlx.zeros((4, 32, 32), tl.float32, layout=mma) for _ in range(6)])
    loop_end: tl.constexpr = max(((K // 128 - 2) // 2) * 2, 0)
    if loop_end > 0:
        for stage in tl.static_range(2):
            acc, operands_a, operands_b, scales_a, scales_b = _n128x192_step_128x192(
                A, B, SA, SB, pm, pn, stage, a_stages[stage], b_stages[stage], a_stages[1 - stage], b_stages[1 - stage],
                operands_a, operands_b, scales_a, scales_b, acc, mma, dot_a, dot_b, M, N, K, True, True, A_SHUFFLED,
                B_SHUFFLED)
        for panel in tl.range(2, loop_end, 2, num_stages=1):
            for stage in tl.static_range(2):
                acc, operands_a, operands_b, scales_a, scales_b = _n128x192_step_128x192(
                    A, B, SA, SB, pm, pn, panel + stage, a_stages[stage], b_stages[stage], a_stages[1 - stage],
                    b_stages[1 - stage], operands_a, operands_b, scales_a, scales_b, acc, mma, dot_a, dot_b, M, N, K,
                    True, True, A_SHUFFLED, B_SHUFFLED)
    for panel in tl.static_range(loop_end, K // 128):
        acc, operands_a, operands_b, scales_a, scales_b = _n128x192_step_128x192(
            A, B, SA, SB, pm, pn, panel, a_stages[panel % 2], b_stages[panel % 2], a_stages[1 - panel % 2],
            b_stages[1 - panel % 2], operands_a, operands_b, scales_a, scales_b, acc, mma, dot_a, dot_b, M, N, K,
            panel + 2 < K // 128, panel + 1 < K // 128, A_SHUFFLED, B_SHUFFLED)
    acc = tlx.amd_mfma_commit(acc)
    for ni in tl.static_range(3):
        for mi in tl.static_range(2):
            _n128x192_store_acc_128x192(acc[mi * 3 + ni], C, pm, pn, mi, ni, N, STREAMING_STORE)


# -----------------------------------------------------------------------------
# Native Cadence 256X256
# -----------------------------------------------------------------------------


@triton.jit
def _n256x256_packed_scale_pairs(
    source,
    block,
    panel,
    IS_A: tl.constexpr,
    LIMIT: tl.constexpr,
    K: tl.constexpr,
):
    word_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
        stride=((4, 8, 16, 32, 64, 0, 128, 256), (1, 2)),
    )
    wave = tl.arange(0, 4)[:, None, None]
    lane_row = tl.arange(0, 32)[None, :, None]
    words_in_atom = tl.arange(0, 4)[None, None, :]
    wave_rows = wave // 2 * 128 if IS_A else wave % 2 * 128
    rows = block + wave_rows + lane_row
    word_offsets = ((rows // 128) * (K // 128) * 128 + rows % 32 * 4 + words_in_atom)
    words = tlx.buffer_load(
        source.to(tl.pointer_type(tl.uint32)) + panel * 128,
        tlx.require_layout(word_offsets, word_layout),
        mask=tlx.require_layout(
            tl.broadcast_to((rows < LIMIT) & (panel < K // 128), [4, 32, 4]),
            word_layout,
        ),
        other=0,
        contiguity=4,
    )
    pieces = tl.tuple([tlx.extract_slice(words, [4, 32, 1], [0, 0, index]) for index in range(4)])
    return tl.tuple([
        pieces[0].to(tl.uint64) | (pieces[1].to(tl.uint64) << 32),
        pieces[2].to(tl.uint64) | (pieces[3].to(tl.uint64) << 32),
    ])


@tl.core.builtin
def _n256x256_packed_scale_permute(low, high, control, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        [high, low, control],
        {(tl.uint32, tl.uint32, tl.uint32): ("llvm.amdgcn.perm", tl.uint32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _n256x256_packed_scale_word(low, high):
    layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    low = tlx.release_layout(low, relaxed=True)
    high = tlx.release_layout(high, relaxed=True)
    low = tlx.require_layout(tl.broadcast_to(low, [4, 32, 2]), layout)
    high = tlx.require_layout(tl.broadcast_to(high, [4, 32, 2]), layout)
    group = tlx.require_layout(tl.broadcast_to(tl.arange(0, 2)[None, None, :], [4, 32, 2]), layout)
    control = (group.to(tl.uint32) * 0x01010101 + 0x06020400).to(tl.uint32)
    return tlx.require_layout(_n256x256_packed_scale_permute(low, high, control), layout)


@triton.jit
def _n256x256_packed_scale_finish(a_pairs, b_pairs):
    a_words = tl.tuple(
        [_n256x256_packed_scale_word(pair.to(tl.uint32), (pair >> 32).to(tl.uint32)) for pair in a_pairs])
    b_words = tl.tuple(
        [_n256x256_packed_scale_word(pair.to(tl.uint32), (pair >> 32).to(tl.uint32)) for pair in b_pairs])
    return a_words, b_words


@triton.jit
def _n256x256_packed_scale_initial(SA, SB, pm, pn, first_panel, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                   SCALES_SHUFFLED: tl.constexpr = False):
    if SCALES_SHUFFLED:
        return (
            _n256x192_packed_scale_a_shuffled(SA, pm, first_panel, K),
            _n256x192_packed_scale_b_shuffled(SB, pn, first_panel, K),
        )
    a_pairs = _n256x256_packed_scale_pairs(SA, pm * 256, first_panel, True, M, K)
    b_pairs = _n256x256_packed_scale_pairs(SB, pn * 256, first_panel, False, N, K)
    return _n256x256_packed_scale_finish(a_pairs, b_pairs)


@triton.jit
def _n256x256_fly_layout_copy(source, row_base, panel, destination, ROWS: tl.constexpr, LIMIT: tl.constexpr,
                              K: tl.constexpr):
    copy_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2) + ((2, ) if ROWS >= 64 else
                                                         ()) + ((2, ) if ROWS >= 128 else
                                                                ()) + ((2, ) if ROWS >= 256 else ())),
        stride=((16, 32, 64, 128, 256, 512, 1024, 2048), (1, 2, 4, 8) + ((4096, ) if ROWS >= 64 else
                                                                         ()) + ((8192, ) if ROWS >= 128 else
                                                                                ()) + ((16384, ) if ROWS >= 256 else
                                                                                       ())),
    )
    indices = tl.arange(0, ROWS * 128)
    rows = row_base + indices // 128
    # Row bits 1:3 swizzle column bits 4:6 in the physical LDS copy.
    kk = (indices % 128) ^ ((indices & 0x700) >> 4)
    offsets = tlx.require_layout(tl.max_contiguous(tl.multiple_of(rows * K + kk, 16), 16), copy_layout)
    mask = tlx.require_layout((rows < LIMIT) & (panel < K // 128), copy_layout)
    other = tlx.require_layout(tl.zeros([ROWS * 128], tlx.dtype_of(source)), copy_layout)
    flat_shared: tl.constexpr = tlx.shared_linear_layout_encoding(
        [[1 << bit] for bit in range(12 + (ROWS >= 64) + (ROWS >= 128) + (ROWS >= 256))], [], alignment=16)
    destination = tlx.local_reinterpret(destination, tlx.dtype_of(source), [ROWS * 128], layout=flat_shared)
    tlx.buffer_load_to_local(destination, source + panel * 128, offsets, mask=mask, other=other)


@triton.jit
def _n256x256_fly_layout_publish(A, B, pm, pn, panel, a_stage, b_stage, M: tl.constexpr, N: tl.constexpr,
                                 K: tl.constexpr):
    _n256x256_fly_layout_copy(A, pm * 256, panel, a_stage[0], 256, M, K)
    for plane in tl.static_range(4):
        _n256x256_fly_layout_copy(B, pn * 256 + plane * 64, panel, b_stage[plane], 64, N, K)
    tlx.async_load_commit_group()


@triton.jit
def _n256x256_fly_layout_read_a(a_stage, tile: tl.constexpr, half: tl.constexpr, dot_a: tl.constexpr):
    wave_m = tlx.assume_uniform(tlx.thread_id(0) // 128)
    panel = tlx.local_slice(a_stage[0], [0, wave_m * 128 + tile * 32, half * 64], [1, 32, 64])
    return tlx.local_load(panel, layout=dot_a, relaxed=True)


@triton.jit
def _n256x256_fly_layout_read_b(b_stage, tile: tl.constexpr, half: tl.constexpr, dot_b: tl.constexpr):
    wave_n = tlx.assume_uniform(tlx.thread_id(0) // 64 % 2)
    first_row = wave_n * 128 + tile * 32
    plane = tlx.local_view(b_stage, first_row // 64)
    plane = tlx.local_trans(plane, (0, 2, 1))
    panel = tlx.local_slice(plane, [0, half * 64, first_row % 64], [1, 64, 32])
    return tlx.local_load(panel, layout=dot_b, relaxed=True)


@tl.core.builtin
def _n256x256_fly_layout_replace(values, index: tl.constexpr, value, _semantic=None):
    index = tl.core._unwrap_if_constexpr(index)
    items = list(values)
    items[index] = value
    return tl.tuple(items)


@tl.core.builtin
def _n256x256_fly_layout_dot_one(acc, a, b, sa, sb, half: tl.constexpr, mi: tl.constexpr, ni: tl.constexpr,
                                 mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr, _semantic=None):
    half, mi, ni = (tl.core._unwrap_if_constexpr(value) for value in (half, mi, ni))
    values = list(acc)
    a = tlx.require_layout(a, dot_a, pin=False, _semantic=_semantic)
    b = tlx.require_layout(b, dot_b, pin=False, _semantic=_semantic)
    scale_layout = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    sa = tlx.require_layout(sa, scale_layout, pin=False, _semantic=_semantic)
    sb = tlx.require_layout(sb, scale_layout, pin=False, _semantic=_semantic)
    c = tlx.require_layout(values[mi * 4 + ni], mma, pin=False, _semantic=_semantic)
    values[mi * 4 + ni] = tlx.amd_scheduled_mfma_scaled(
        a,
        b,
        c,
        sa,
        sb,
        "e4m3",
        "e4m3",
        accumulator_role="persistent",
        accumulator_register_class="agpr",
        a_scale_selector=half * 2 + mi % 2,
        b_scale_selector=half * 2 + ni % 2,
        _semantic=_semantic,
    )
    return tl.tuple(values)


@triton.jit
def _n256x256_cadence_publish_chunk(A, B, pm, pn, panel, a_stage, b_stage, CHUNK: tl.constexpr, M: tl.constexpr,
                                    N: tl.constexpr, K: tl.constexpr):
    if CHUNK < 8:
        destination = tlx.local_slice(a_stage[0], [0, CHUNK * 32, 0], [1, 32, 128])
        _n256x256_fly_layout_copy(A, pm * 256 + CHUNK * 32, panel, destination, 32, M, K)
    else:
        column: tl.constexpr = CHUNK - 8
        destination = tlx.local_slice(b_stage[column // 2], [0, column % 2 * 32, 0], [1, 32, 128])
        _n256x256_fly_layout_copy(B, pn * 256 + column * 32, panel, destination, 32, N, K)


@triton.jit
def _n256x256_cadence_dot_at(acc, a, b, sa, sb, INDEX: tl.constexpr, mma: tl.constexpr, dot_a: tl.constexpr,
                             dot_b: tl.constexpr):
    half: tl.constexpr = INDEX // 16
    ni: tl.constexpr = INDEX % 16 // 4
    mi: tl.constexpr = INDEX % 4
    return _n256x256_fly_layout_dot_one(acc, a[half * 4 + mi], b[half * 4 + ni], sa[mi // 2], sb[ni // 2], half, mi, ni,
                                        mma, dot_a, dot_b)


@triton.jit
def _n256x256_cadence_read_at(a_stage, b_stage, a, b, INDEX: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr):
    is_a: tl.constexpr = (False, True, True, True, True, False, False, False, False, False, False, True, True, True,
                          True, False)[INDEX]
    half: tl.constexpr = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)[INDEX]
    tile: tl.constexpr = (0, 0, 1, 2, 3, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 3)[INDEX]
    if is_a:
        a = _n256x256_fly_layout_replace(a, half * 4 + tile, _n256x256_fly_layout_read_a(a_stage, tile, half, dot_a))
    else:
        b = _n256x256_fly_layout_replace(b, half * 4 + tile, _n256x256_fly_layout_read_b(b_stage, tile, half, dot_b))
    return a, b


@triton.jit
def _n256x256_cadence_compute(A, B, SA, SB, pm, pn, panel, a_current, b_current, a_next, b_next, a, b, sa, sb, acc,
                              mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr, M: tl.constexpr,
                              N: tl.constexpr, K: tl.constexpr, LOAD_FUTURE: tl.constexpr, LOAD_NEXT: tl.constexpr,
                              SCALES_SHUFFLED: tl.constexpr = False, BALANCED_G2L_CADENCE: tl.constexpr = False):
    a_pairs = tl.tuple([])
    b_pairs = tl.tuple([])
    next_sa = sa
    next_sb = sb
    scale_mfmas: tl.constexpr = 4 if BALANCED_G2L_CADENCE else 8
    if LOAD_NEXT:
        for index in tl.static_range(scale_mfmas):
            acc = _n256x256_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)
            if index == 0:
                tlx.amd_sched_barrier(0)
                if SCALES_SHUFFLED:
                    next_sa = _n256x192_packed_scale_a_shuffled(SA, pm, panel + 1, K)
                else:
                    a_pairs = _n256x256_packed_scale_pairs(SA, pm * 256, panel + 1, True, M, K)
                tlx.amd_sched_barrier(0)
            elif index == 2:
                tlx.amd_sched_barrier(0)
                if SCALES_SHUFFLED:
                    next_sb = _n256x192_packed_scale_b_shuffled(SB, pn, panel + 1, K)
                else:
                    b_pairs = _n256x256_packed_scale_pairs(SB, pn * 256, panel + 1, False, N, K)
                tlx.amd_sched_barrier(0)
    else:
        for index in tl.static_range(scale_mfmas):
            acc = _n256x256_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)

    g2l_mfmas: tl.constexpr = 16 if BALANCED_G2L_CADENCE else 8
    for group in tl.static_range(g2l_mfmas):
        if LOAD_FUTURE:
            with tlx.warp_pipeline_stage("cadence_g2l", scope="intra_wave", pair=group):
                if BALANCED_G2L_CADENCE:
                    _n256x256_cadence_publish_chunk(A, B, pm, pn, panel + 2, a_current, b_current, group, M, N, K)
                else:
                    _n256x256_cadence_publish_chunk(A, B, pm, pn, panel + 2, a_current, b_current, group * 2, M, N, K)
                    _n256x256_cadence_publish_chunk(A, B, pm, pn, panel + 2, a_current, b_current, group * 2 + 1, M, N,
                                                    K)
                if group == g2l_mfmas - 1:
                    tlx.async_load_commit_group()
            with tlx.warp_pipeline_stage("cadence_mfma", scope="intra_wave", pair=group):
                acc = _n256x256_cadence_dot_at(acc, a, b, sa, sb, group + scale_mfmas, mma, dot_a, dot_b)
        else:
            acc = _n256x256_cadence_dot_at(acc, a, b, sa, sb, group + scale_mfmas, mma, dot_a, dot_b)

    next_a, next_b = a, b
    if LOAD_NEXT:
        tlx.async_load_wait_group(1 if LOAD_FUTURE else 0)
        refill_begin: tl.constexpr = 20 if BALANCED_G2L_CADENCE else 16
        for index in tl.static_range(refill_begin, 32):
            acc = _n256x256_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)
            if BALANCED_G2L_CADENCE and (index <= 23 or index >= 27):
                tlx.amd_sched_barrier(0)
                if index == 20:
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 0, dot_a, dot_b)
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 1, dot_a, dot_b)
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 8, dot_a, dot_b)
                elif index == 21:
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 2, dot_a, dot_b)
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 3, dot_a, dot_b)
                elif index == 22:
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 4, dot_a, dot_b)
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 5, dot_a, dot_b)
                elif index == 23:
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 6, dot_a, dot_b)
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 7, dot_a, dot_b)
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 9, dot_a, dot_b)
                elif index >= 27:
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, index - 17, dot_a, dot_b)
                    if index == 31:
                        next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 15, dot_a, dot_b)
                if index < 31:
                    tlx.amd_sched_barrier(0)
            elif not BALANCED_G2L_CADENCE and (index < 20 or index == 20 or index == 23 or index >= 27):
                tlx.amd_sched_barrier(0)
                if index < 20:
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, (index - 16) * 2, dot_a,
                                                               dot_b)
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, (index - 16) * 2 + 1,
                                                               dot_a, dot_b)
                elif index == 20:
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 8, dot_a, dot_b)
                elif index == 23:
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 9, dot_a, dot_b)
                elif index >= 27:
                    next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, index - 17, dot_a, dot_b)
                    if index == 31:
                        next_a, next_b = _n256x256_cadence_read_at(a_next, b_next, next_a, next_b, 15, dot_a, dot_b)
                if index < 31:
                    tlx.amd_sched_barrier(0)
        if not SCALES_SHUFFLED:
            next_sa, next_sb = _n256x256_packed_scale_finish(a_pairs, b_pairs)
    else:
        refill_begin: tl.constexpr = 20 if BALANCED_G2L_CADENCE else 16
        for index in tl.static_range(refill_begin, 32):
            acc = _n256x256_cadence_dot_at(acc, a, b, sa, sb, index, mma, dot_a, dot_b)
        tlx.async_load_wait_group(0)
        next_sa, next_sb = sa, sb
    return acc, next_a, next_b, next_sa, next_sb


@triton.jit
def _n256x256_fly_layout_step(A, B, SA, SB, pm, pn, panel, a_current, b_current, a_next, b_next, operands_a, operands_b,
                              words_a, words_b, acc, mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr,
                              M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, LOAD_FUTURE: tl.constexpr,
                              LOAD_NEXT: tl.constexpr, SCALES_SHUFFLED: tl.constexpr = False,
                              BALANCED_G2L_CADENCE: tl.constexpr = False):
    tlx.workgroup_barrier()
    return _n256x256_cadence_compute(A, B, SA, SB, pm, pn, panel, a_current, b_current, a_next, b_next, operands_a,
                                     operands_b, words_a, words_b, acc, mma, dot_a, dot_b, M, N, K, LOAD_FUTURE,
                                     LOAD_NEXT, SCALES_SHUFFLED, BALANCED_G2L_CADENCE)


@triton.jit
def _gemm_native_256x256_split16(A, B, SA, SB, C, WORKSPACE, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                 BM: tl.constexpr, BN: tl.constexpr, WARPS_M: tl.constexpr, SPLIT_K: tl.constexpr,
                                 SCALES_SHUFFLED: tl.constexpr = False, BALANCED_G2L_CADENCE: tl.constexpr = False,
                                 STREAMING_STORE: tl.constexpr = False):
    tl.static_assert(BM == 256 and BN == 256 and WARPS_M == 2
                     and (SPLIT_K == 1 or SPLIT_K == 2 or SPLIT_K == 12 or SPLIT_K == 16))
    tl.static_assert(K % (128 * SPLIT_K) == 0 and M % 256 == 0)
    piece_panels: tl.constexpr = K // (128 * SPLIT_K)
    tl.static_assert(piece_panels >= 2 and piece_panels % 2 == 0)
    gm: tl.constexpr = triton.cdiv(M, 256)
    gn: tl.constexpr = triton.cdiv(N, 256)
    total: tl.constexpr = gm * gn
    raw_pid = tl.program_id(0)
    split_id = raw_pid // total
    pid = raw_pid % total
    tl.assume((split_id >= 0) & (split_id < SPLIT_K))
    first_panel = split_id * piece_panels
    per_xcd: tl.constexpr = triton.cdiv(total, 8)
    tall_xcds: tl.constexpr = total % 8 or 8
    xcd, local_pid = pid % 8, pid // 8
    pid = tl.where(xcd < tall_xcds, xcd * per_xcd + local_pid,
                   tall_xcds * per_xcd + (xcd - tall_xcds) * (per_xcd - 1) + local_pid)
    first_m = pid // (4 * gn) * 4
    group_m = tl.minimum(gm - first_m, 4)
    pm = first_m + pid % (4 * gn) % group_m
    pn = pid % (4 * gn) // group_m
    tl.assume((pm >= 0) & (pm < gm))
    tl.assume((pn >= 0) & (pn < gn))
    mma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], False, [4, 1, 1])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    a_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 16], [4, 32],
                             [8, 64], [16, 0], [32, 0], [64, 0], [128, 0]]
    b_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 16], [4, 32],
                             [8, 64], [16, 0], [32, 0]]
    a_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in a_bases], [],
                                                               alignment=16)
    b_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in b_bases], [],
                                                               alignment=16)
    a0 = tlx.local_alloc((1, 256, 128), tlx.dtype_of(A), 1, layout=a_shared)
    b0 = tlx.local_alloc((1, 64, 128), tlx.dtype_of(B), 4, layout=b_shared)
    a1 = tlx.local_alloc((1, 256, 128), tlx.dtype_of(A), 1, layout=a_shared)
    b1 = tlx.local_alloc((1, 64, 128), tlx.dtype_of(B), 4, layout=b_shared)
    a_stages = tl.tuple([a0, a1])
    b_stages = tl.tuple([b0, b1])
    _n256x256_fly_layout_publish(A, B, pm, pn, first_panel, a0, b0, M, N, K)
    words_a, words_b = _n256x256_packed_scale_initial(SA, SB, pm, pn, first_panel, M, N, K, SCALES_SHUFFLED)
    tlx.async_load_wait_group(0)
    tlx.workgroup_barrier()
    operands_a = tl.tuple([_n256x256_fly_layout_read_a(a0, index % 4, index // 4, dot_a) for index in range(8)])
    operands_b = tl.tuple([_n256x256_fly_layout_read_b(b0, index % 4, index // 4, dot_b) for index in range(8)])
    if piece_panels > 1:
        _n256x256_fly_layout_publish(A, B, pm, pn, first_panel + 1, a1, b1, M, N, K)
    acc = tl.tuple([tlx.zeros((4, 32, 32), tl.float32, layout=mma) for _ in range(16)])
    loop_end: tl.constexpr = max(((piece_panels - 2) // 2) * 2, 0)
    if loop_end > 0:
        for stage in tl.static_range(2):
            acc, operands_a, operands_b, words_a, words_b = _n256x256_fly_layout_step(
                A, B, SA, SB, pm, pn, first_panel + stage, a_stages[stage], b_stages[stage], a_stages[1 - stage],
                b_stages[1 - stage], operands_a, operands_b, words_a, words_b, acc, mma, dot_a, dot_b, M, N, K, True,
                True, SCALES_SHUFFLED, BALANCED_G2L_CADENCE)
        for panel in tl.range(2, loop_end, 2, num_stages=1):
            for stage in tl.static_range(2):
                acc, operands_a, operands_b, words_a, words_b = _n256x256_fly_layout_step(
                    A, B, SA, SB, pm, pn, first_panel + panel + stage, a_stages[stage], b_stages[stage],
                    a_stages[1 - stage], b_stages[1 - stage], operands_a, operands_b, words_a, words_b, acc, mma, dot_a,
                    dot_b, M, N, K, True, True, SCALES_SHUFFLED, BALANCED_G2L_CADENCE)
    for panel in tl.static_range(loop_end, piece_panels):
        acc, operands_a, operands_b, words_a, words_b = _n256x256_fly_layout_step(
            A, B, SA, SB, pm, pn, first_panel + panel, a_stages[panel % 2], b_stages[panel % 2],
            a_stages[1 - panel % 2], b_stages[1 - panel % 2], operands_a, operands_b, words_a, words_b, acc, mma, dot_a,
            dot_b, M, N, K, panel + 2 < piece_panels, panel + 1 < piece_panels, SCALES_SHUFFLED, BALANCED_G2L_CADENCE)
    acc = tlx.amd_mfma_commit(acc)
    target = C if SPLIT_K == 1 else WORKSPACE + split_id * M * N
    wave = tl.arange(0, 4)[:, None, None]
    rows32 = tl.arange(0, 32)[None, :, None]
    cols32 = tl.arange(0, 32)[None, None, :]
    store_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
        stride=((4, 8, 16, 32, 64, 128, 1024, 2048), (1, 2, 256, 512)),
    )
    for ni in tl.static_range(4):
        for mi in tl.static_range(4):
            rows = pm * 256 + wave // 2 * 128 + mi * 32 + rows32
            cols = pn * 256 + wave % 2 * 128 + ni * 32 + cols32
            value = tlx.require_layout(acc[mi * 4 + ni].to(tlx.dtype_of(target)), store_layout,
                                       late_address_compute=True)
            offsets = tlx.require_layout(rows * N + cols, store_layout)
            mask = tlx.require_layout((rows < M) & (cols < N), store_layout)
            tlx.buffer_store(
                value,
                target,
                offsets,
                mask=mask,
                cache=".cs" if STREAMING_STORE else "",
            )


# -----------------------------------------------------------------------------
# Native Cadence Split2
# -----------------------------------------------------------------------------


@triton.jit
def _ns2_packed_scale_initial_split2(SA, SB, pm, pn, first_panel, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                     A_SHUFFLED: tl.constexpr = False, B_SHUFFLED: tl.constexpr = False):
    if A_SHUFFLED:
        a_words = _n256x192_packed_scale_a_shuffled(SA, pm, first_panel, K)
    elif M % 256 == 0:
        a_pairs = _n256x192_packed_scale_a_pairs(SA, pm * 256, first_panel, M, K)
    else:
        a_pairs = tl.tuple(
            [_n256x192_packed_scale_pair(SA, pm * 256, first_panel, True, pair, M, K) for pair in range(2)])
    if B_SHUFFLED:
        b_words = _n256x192_packed_scale_b_shuffled(SB, pn, first_panel, K)
    else:
        b_pairs = tl.tuple(
            [_n256x192_packed_scale_pair(SB, pn * 192, first_panel, False, pair, N, K) for pair in range(2)])
    packed_a = (tl.tuple([]) if A_SHUFFLED else _n256x192_packed_scale_a_finish(a_pairs))
    packed_b = (tl.tuple([]) if B_SHUFFLED else _n256x192_packed_scale_b_finish(b_pairs))
    return (
        a_words if A_SHUFFLED else packed_a,
        b_words if B_SHUFFLED else packed_b,
    )


@triton.jit
def _gemm_native_cadence_split2(A, B, SA, SB, C, WORKSPACE, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                BM: tl.constexpr, BN: tl.constexpr, WARPS_M: tl.constexpr, SPLIT_K: tl.constexpr,
                                A_SHUFFLED: tl.constexpr = False, B_SHUFFLED: tl.constexpr = False):
    tl.static_assert(BM == 256 and BN == 192 and WARPS_M == 2 and (SPLIT_K == 2 or SPLIT_K == 4))
    tl.static_assert(K % 128 == 0 and M % 256 == 0 and N % 128 == 0)
    total_panels: tl.constexpr = K // 128
    piece_panels: tl.constexpr = total_panels // SPLIT_K
    extra_splits: tl.constexpr = total_panels % SPLIT_K
    tl.static_assert(piece_panels >= 2 and piece_panels % 2 == 0)
    gm: tl.constexpr = triton.cdiv(M, 256)
    gn: tl.constexpr = triton.cdiv(N, 192)
    total: tl.constexpr = gm * gn
    raw_pid = tl.program_id(0)
    split_id = raw_pid // total
    pid = raw_pid % total
    tl.assume((split_id >= 0) & (split_id < SPLIT_K))
    first_panel = split_id * piece_panels + tl.minimum(split_id, extra_splits)
    per_xcd: tl.constexpr = triton.cdiv(total, 8)
    tall_xcds: tl.constexpr = total % 8 or 8
    xcd, local_pid = pid % 8, pid // 8
    pid = tl.where(xcd < tall_xcds, xcd * per_xcd + local_pid,
                   tall_xcds * per_xcd + (xcd - tall_xcds) * (per_xcd - 1) + local_pid)
    first_m = pid // (4 * gn) * 4
    group_m = tl.minimum(gm - first_m, 4)
    pm = first_m + pid % (4 * gn) % group_m
    pn = pid % (4 * gn) // group_m
    tl.assume((pm >= 0) & (pm < gm))
    tl.assume((pn >= 0) & (pn < gn))
    mma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], False, [4, 1, 1])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    a_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 16], [4, 32],
                             [8, 64], [16, 0], [32, 0], [64, 0], [128, 0]]
    b_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 16], [4, 32],
                             [8, 64], [16, 0], [32, 0]]
    a_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in a_bases], [],
                                                               alignment=16)
    b_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in b_bases], [],
                                                               alignment=16)
    a0 = tlx.local_alloc((1, 256, 128), tlx.dtype_of(A), 1, layout=a_shared)
    b0 = tlx.local_alloc((1, 64, 128), tlx.dtype_of(B), 3, layout=b_shared)
    a1 = tlx.local_alloc((1, 256, 128), tlx.dtype_of(A), 1, layout=a_shared)
    b1 = tlx.local_alloc((1, 64, 128), tlx.dtype_of(B), 3, layout=b_shared)
    a_stages = tl.tuple([a0, a1])
    b_stages = tl.tuple([b0, b1])
    _n256x192_fly_layout_publish(A, B, pm, pn, first_panel, a0, b0, M, N, K)
    words_a, words_b = _ns2_packed_scale_initial_split2(SA, SB, pm, pn, first_panel, M, N, K, A_SHUFFLED, B_SHUFFLED)
    tlx.async_load_wait_group(0)
    tlx.workgroup_barrier()
    operands_a = tl.tuple([_n256x192_fly_layout_read_a(a0, index % 4, index // 4, dot_a) for index in range(8)])
    operands_b = tl.tuple([_n256x192_fly_layout_read_b(b0, index % 3, index // 3, dot_b) for index in range(6)])
    if piece_panels > 1:
        _n256x192_fly_layout_publish(A, B, pm, pn, first_panel + 1, a1, b1, M, N, K)
    acc = tl.tuple([tlx.zeros((4, 32, 32), tl.float32, layout=mma) for _ in range(12)])
    loop_end: tl.constexpr = max(((piece_panels - 2) // 2) * 2, 0)
    if loop_end > 0:
        for stage in tl.static_range(2):
            acc, operands_a, operands_b, words_a, words_b = _n256x192_fly_layout_step(
                A, B, SA, SB, pm, pn, first_panel + stage, a_stages[stage], b_stages[stage], a_stages[1 - stage],
                b_stages[1 - stage], operands_a, operands_b, words_a, words_b, acc, mma, dot_a, dot_b, M, N, K, True,
                True, A_SHUFFLED, B_SHUFFLED)
        for panel in tl.range(2, loop_end, 2, num_stages=1):
            for stage in tl.static_range(2):
                acc, operands_a, operands_b, words_a, words_b = _n256x192_fly_layout_step(
                    A, B, SA, SB, pm, pn, first_panel + panel + stage, a_stages[stage], b_stages[stage],
                    a_stages[1 - stage], b_stages[1 - stage], operands_a, operands_b, words_a, words_b, acc, mma, dot_a,
                    dot_b, M, N, K, True, True, A_SHUFFLED, B_SHUFFLED)
    for panel in tl.static_range(loop_end, piece_panels):
        acc, operands_a, operands_b, words_a, words_b = _n256x192_fly_layout_step(
            A, B, SA, SB, pm, pn, first_panel + panel, a_stages[panel % 2], b_stages[panel % 2],
            a_stages[1 - panel % 2], b_stages[1 - panel % 2], operands_a, operands_b, words_a, words_b, acc, mma, dot_a,
            dot_b, M, N, K, panel + 2 < piece_panels, panel + 1 < piece_panels, A_SHUFFLED, B_SHUFFLED)
    if extra_splits > 0:
        if split_id < extra_splits:
            tail_panel = first_panel + piece_panels
            _n256x192_fly_layout_publish(A, B, pm, pn, tail_panel, a0, b0, M, N, K)
            tail_words_a, tail_words_b = _ns2_packed_scale_initial_split2(SA, SB, pm, pn, tail_panel, M, N, K,
                                                                          A_SHUFFLED, B_SHUFFLED)
            tlx.async_load_wait_group(0)
            tlx.workgroup_barrier()
            tail_a = tl.tuple([_n256x192_fly_layout_read_a(a0, index % 4, index // 4, dot_a) for index in range(8)])
            tail_b = tl.tuple([_n256x192_fly_layout_read_b(b0, index % 3, index // 3, dot_b) for index in range(6)])
            for index in tl.static_range(24):
                acc = _n256x192_cadence_dot_at(acc, tail_a, tail_b, tail_words_a, tail_words_b, index, mma, dot_a,
                                               dot_b)
    acc = tlx.amd_mfma_commit(acc)
    wave = tl.arange(0, 4)[:, None, None]
    rows32 = tl.arange(0, 32)[None, :, None]
    cols32 = tl.arange(0, 32)[None, None, :]
    store_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
        stride=((4, 8, 16, 32, 64, 128, 1024, 2048), (1, 2, 256, 512)),
    )
    for ni in tl.static_range(3):
        for mi in tl.static_range(4):
            rows = pm * 256 + wave // 2 * 128 + mi * 32 + rows32
            cols = pn * 192 + wave % 2 * 96 + ni * 32 + cols32
            value = tlx.require_layout(acc[mi * 3 + ni].to(tlx.dtype_of(WORKSPACE)), store_layout,
                                       late_address_compute=True)
            offsets = tlx.require_layout(rows * N + cols, store_layout)
            mask = tlx.require_layout((rows < M) & (cols < N), store_layout)
            tlx.buffer_store(value, WORKSPACE + split_id * M * N, offsets, mask=mask)


# -----------------------------------------------------------------------------
# Native Short M
# -----------------------------------------------------------------------------


@triton.jit
def _nshort_short_a_scale_pair(source, block, panel, LIMIT: tl.constexpr, K: tl.constexpr):
    word_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((1, 2, 4, 8, 16, 0, 32, 64), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    lane_row = tl.arange(0, 32)[None, :, None]
    rows = block + wave * 0 + lane_row
    word_offsets = ((rows // 128) * (K // 128) * 128 + rows % 32 * 4 + rows % 128 // 32)
    offsets = tlx.require_layout(word_offsets // 2, word_layout)
    mask = tlx.require_layout((rows < LIMIT) & (panel < K // 128), word_layout)
    return tlx.buffer_load(
        source.to(tl.pointer_type(tl.uint64)) + panel * 64,
        offsets,
        mask=mask,
        other=0,
    )


@triton.jit
def _nshort_short_b_scale_pair(source, block, panel, LIMIT: tl.constexpr, K: tl.constexpr):
    word_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((1, 2, 4, 8, 16, 0, 32, 64), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    lane_row = tl.arange(0, 32)[None, :, None]
    rows = block + wave // 2 * 64 + lane_row
    word_offsets = ((rows // 128) * (K // 128) * 128 + rows % 32 * 4 + rows % 128 // 32)
    return tlx.buffer_load(
        source.to(tl.pointer_type(tl.uint64)) + panel * 64,
        tlx.require_layout(word_offsets // 2, word_layout),
        mask=tlx.require_layout((rows < LIMIT) & (panel < K // 128), word_layout),
        other=0,
    )


@triton.jit
def _nshort_short_natural_a_scale_word(source, block, panel, LIMIT: tl.constexpr, K: tl.constexpr):
    word_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((1, 2, 4, 8, 16, 0, 32, 64), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    row = tl.arange(0, 32)[None, :, None]
    low_rows = block + wave * 0 + row
    high_rows = low_rows + 32
    low = tlx.buffer_load(
        source.to(tl.pointer_type(tl.uint32)),
        tlx.require_layout(low_rows * (K // 128) + panel, word_layout),
        mask=tlx.require_layout((low_rows < LIMIT) & (panel < K // 128), word_layout),
        other=0,
    )
    high = tlx.buffer_load(
        source.to(tl.pointer_type(tl.uint32)),
        tlx.require_layout(high_rows * (K // 128) + panel, word_layout),
        mask=tlx.require_layout((high_rows < LIMIT) & (panel < K // 128), word_layout),
        other=0,
    )
    return _ns4_packed_scale_word(low, high)


@triton.jit
def _nshort_short_natural_dword_pair(
    source,
    rows,
    panel_pair,
    LIMIT: tl.constexpr,
    K: tl.constexpr,
):
    word_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((1, 2, 4, 8, 16, 0, 32, 64), (0, )),
    )
    row_mask = tlx.require_layout(rows < LIMIT, word_layout)
    if K % 256 == 0:
        return tlx.buffer_load(
            source.to(tl.pointer_type(tl.uint64)),
            tlx.require_layout(rows * (K // 256) + panel_pair, word_layout),
            mask=row_mask,
            other=0,
        )
    dword_offsets = tlx.require_layout(rows * (K // 128) + panel_pair * 2, word_layout)
    has_pair = panel_pair * 2 + 1 < K // 128
    pointers = (source.to(tl.pointer_type(tl.uint32)) + dword_offsets).to(tl.pointer_type(tl.uint64))
    wide = tl.load(
        pointers,
        mask=row_mask & has_pair,
        other=0,
    )
    tail = tlx.buffer_load(
        source.to(tl.pointer_type(tl.uint32)),
        dword_offsets,
        mask=row_mask & ~has_pair & (panel_pair * 2 < K // 128),
        other=0,
    )
    return tlx.require_layout(tl.where(has_pair, wide, tail.to(tl.uint64)), word_layout)


@triton.jit
def _nshort_short_natural_scale_word_pair(
    SA,
    SB,
    pm,
    pn,
    panel_pair,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    scale_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    row = tl.arange(0, 32)[None, :, None]
    a_even = tl.tuple([])
    a_odd = tl.tuple([])
    for pair in tl.static_range(triton.cdiv(BLOCK_M, 64)):
        low_rows = pm * BLOCK_M + pair * 64 + wave * 0 + row
        high_rows = low_rows + 32
        low = _nshort_short_natural_dword_pair(SA, low_rows, panel_pair, M, K)
        high = _nshort_short_natural_dword_pair(SA, high_rows, panel_pair, M, K)
        a_even += tl.tuple([_ns4_packed_scale_word(low.to(tl.uint32), high.to(tl.uint32))])
        a_odd += tl.tuple([_ns4_packed_scale_word((low >> 32).to(tl.uint32), (high >> 32).to(tl.uint32))])
    b_low_rows = pn * 128 + wave // 2 * 64 + row
    b_high_rows = b_low_rows + 32
    b_low = _nshort_short_natural_dword_pair(SB, b_low_rows, panel_pair, N, K)
    b_high = _nshort_short_natural_dword_pair(SB, b_high_rows, panel_pair, N, K)
    b_even = _ns4_packed_scale_word(b_low.to(tl.uint32), b_high.to(tl.uint32))
    b_even_swapped = _ns4_packed_scale_word(b_high.to(tl.uint32), b_low.to(tl.uint32))
    b_odd = _ns4_packed_scale_word((b_low >> 32).to(tl.uint32), (b_high >> 32).to(tl.uint32))
    b_odd_swapped = _ns4_packed_scale_word((b_high >> 32).to(tl.uint32), (b_low >> 32).to(tl.uint32))
    odd_wave = tlx.require_layout(tl.broadcast_to(wave % 2 == 1, [4, 32, 2]), scale_layout)
    return (
        a_even,
        tl.where(odd_wave, b_even_swapped, b_even),
        a_odd,
        tl.where(odd_wave, b_odd_swapped, b_odd),
    )


@triton.jit
def _nshort_short_natural_dword_quad(
    source,
    rows,
    first_panel,
    LIMIT: tl.constexpr,
    K: tl.constexpr,
):
    word_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
        stride=((4, 8, 16, 32, 64, 0, 128, 256), (1, 2)),
    )
    panel = tl.arange(0, 4)[None, None, :]
    offsets = tlx.require_layout(rows * (K // 128) + first_panel + panel, word_layout)
    mask = tlx.require_layout((rows < LIMIT) & (first_panel + panel < K // 128), word_layout)
    words = tlx.buffer_load(
        source.to(tl.pointer_type(tl.uint32)),
        offsets,
        mask=mask,
        other=0,
        contiguity=4,
    )
    return tl.tuple([tlx.extract_slice(words, [4, 32, 1], [0, 0, index]) for index in range(4)])


@triton.jit
def _nshort_short_natural_scale_word_quad(
    SA,
    SB,
    pm,
    pn,
    first_panel,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    scale_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    row = tl.arange(0, 32)[None, :, None]
    a0 = tl.tuple([])
    a1 = tl.tuple([])
    a2 = tl.tuple([])
    a3 = tl.tuple([])
    for pair in tl.static_range(triton.cdiv(BLOCK_M, 64)):
        low_rows = pm * BLOCK_M + pair * 64 + wave * 0 + row
        high_rows = low_rows + 32
        low = _nshort_short_natural_dword_quad(SA, low_rows, first_panel, M, K)
        high = _nshort_short_natural_dword_quad(SA, high_rows, first_panel, M, K)
        a0 += tl.tuple([_ns4_packed_scale_word(low[0], high[0])])
        a1 += tl.tuple([_ns4_packed_scale_word(low[1], high[1])])
        a2 += tl.tuple([_ns4_packed_scale_word(low[2], high[2])])
        a3 += tl.tuple([_ns4_packed_scale_word(low[3], high[3])])
    b_low_rows = pn * 128 + wave // 2 * 64 + row
    b_high_rows = b_low_rows + 32
    b_low = _nshort_short_natural_dword_quad(SB, b_low_rows, first_panel, N, K)
    b_high = _nshort_short_natural_dword_quad(SB, b_high_rows, first_panel, N, K)
    odd_wave = tlx.require_layout(tl.broadcast_to(wave % 2 == 1, [4, 32, 2]), scale_layout)
    b0 = _ns4_packed_scale_word(b_low[0], b_high[0])
    b1 = _ns4_packed_scale_word(b_low[1], b_high[1])
    b2 = _ns4_packed_scale_word(b_low[2], b_high[2])
    b3 = _ns4_packed_scale_word(b_low[3], b_high[3])
    return (
        a0,
        tl.where(odd_wave, _ns4_packed_scale_word(b_high[0], b_low[0]), b0),
        a1,
        tl.where(odd_wave, _ns4_packed_scale_word(b_high[1], b_low[1]), b1),
        a2,
        tl.where(odd_wave, _ns4_packed_scale_word(b_high[2], b_low[2]), b2),
        a3,
        tl.where(odd_wave, _ns4_packed_scale_word(b_high[3], b_low[3]), b3),
    )


@triton.jit
def _nshort_short_scale_words(SA, SB, pm, pn, panel, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                              BLOCK_M: tl.constexpr, SHUFFLED: tl.constexpr = False,
                              CUBLAS_BLOCKED: tl.constexpr = False):
    scale_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    wave = tl.arange(0, 4)[:, None, None]
    row = tl.arange(0, 32)[None, :, None]
    group = tl.arange(0, 2)[None, None, :]
    lane = row + group * 32
    if SHUFFLED:
        row_stride: tl.constexpr = K // 128 * 64
        k_offset = panel * 64
        a_words = tlx.buffer_load(
            SA.to(tl.pointer_type(tl.uint32)),
            tlx.require_layout(
                pm * row_stride + k_offset + wave * 0 + lane,
                scale_layout,
            ),
            mask=tlx.require_layout(
                tl.full([4, 32, 2], panel < K // 128, tl.int1),
                scale_layout,
            ),
            other=0,
            contiguity=1,
        )
        b_words = tlx.buffer_load(
            SB.to(tl.pointer_type(tl.uint32)),
            tlx.require_layout(
                (pn * 4 + wave) * row_stride + k_offset + lane,
                scale_layout,
            ),
            mask=tlx.require_layout(
                tl.broadcast_to(
                    (pn * 128 + wave * 32 < N) & (panel < K // 128),
                    [4, 32, 2],
                ),
                scale_layout,
            ),
            other=0,
            contiguity=1,
        )
        return tl.tuple([a_words]), b_words
    if CUBLAS_BLOCKED:
        a_words = tl.tuple([
            _ns4_packed_scale_finish(
                tl.tuple([_nshort_short_a_scale_pair(SA, pm * BLOCK_M + pair * 64, panel, M, K)]),
                tl.tuple([]),
            )[0][0] for pair in range(triton.cdiv(BLOCK_M, 64))
        ])
        b_pair = _nshort_short_b_scale_pair(SB, pn * 128, panel, N, K)
        b_low = b_pair.to(tl.uint32)
        b_high = (b_pair >> 32).to(tl.uint32)
        b_words = _ns4_packed_scale_word(b_low, b_high)
        b_words_swapped = _ns4_packed_scale_word(b_high, b_low)
        odd_wave = tlx.require_layout(tl.broadcast_to(wave % 2 == 1, [4, 32, 2]), scale_layout)
        b_words = tl.where(odd_wave, b_words_swapped, b_words)
    else:
        word_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
            stride=((1, 2, 4, 8, 16, 0, 32, 64), (0, )),
        )
        b_low_rows = pn * 128 + wave // 2 * 64 + row
        b_high_rows = b_low_rows + 32
        a_words = tl.tuple([
            _nshort_short_natural_a_scale_word(SA, pm * BLOCK_M + pair * 64, panel, M, K)
            for pair in range(triton.cdiv(BLOCK_M, 64))
        ])
        b_low = tlx.buffer_load(
            SB.to(tl.pointer_type(tl.uint32)),
            tlx.require_layout(b_low_rows * (K // 128) + panel, word_layout),
            mask=tlx.require_layout((b_low_rows < N) & (panel < K // 128), word_layout),
            other=0,
        )
        b_high = tlx.buffer_load(
            SB.to(tl.pointer_type(tl.uint32)),
            tlx.require_layout(b_high_rows * (K // 128) + panel, word_layout),
            mask=tlx.require_layout((b_high_rows < N) & (panel < K // 128), word_layout),
            other=0,
        )
        b_words = _ns4_packed_scale_word(b_low, b_high)
        b_words_swapped = _ns4_packed_scale_word(b_high, b_low)
        odd_wave = tlx.require_layout(tl.broadcast_to(wave % 2 == 1, [4, 32, 2]), scale_layout)
        b_words = tl.where(odd_wave, b_words_swapped, b_words)
    return a_words, b_words


@triton.jit
def _nshort_short_publish(A, B, pm, pn, panel, a_stage_low, a_stage_high, b_stage, M: tl.constexpr, N: tl.constexpr,
                          K: tl.constexpr, BLOCK_M: tl.constexpr):
    _ns4_fly_layout_copy(A, pm * BLOCK_M, panel, a_stage_low[0], 64, M, K)
    if BLOCK_M > 64:
        _ns4_fly_layout_copy(A, pm * BLOCK_M + 64, panel, a_stage_high[0], 64, M, K)
    _ns4_fly_layout_copy(B, pn * 128, panel, b_stage[0], 128, N, K)
    tlx.async_load_commit_group()


@triton.jit
def _nshort_short_read_a(a_stage_low, a_stage_high, tile: tl.constexpr, half: tl.constexpr, dot_a: tl.constexpr):
    stage = a_stage_low if tile < 2 else a_stage_high
    panel = tlx.local_slice(stage[0], [0, tile % 2 * 32, half * 64], [1, 32, 64])
    return tlx.local_load(panel, layout=dot_a, relaxed=True)


@triton.jit
def _nshort_short_read_b(b_stage, half: tl.constexpr, dot_b: tl.constexpr):
    wave_n = tlx.assume_uniform(tlx.thread_id(0) // 64)
    plane = tlx.local_trans(tlx.local_view(b_stage, 0), (0, 2, 1))
    panel = tlx.local_slice(plane, [0, half * 64, wave_n * 32], [1, 64, 32])
    return tlx.local_load(panel, layout=dot_b, relaxed=True)


@tl.core.builtin
def _nshort_short_dot(acc, a, b, sa_words, sb, half: tl.constexpr, mi: tl.constexpr, mma: tl.constexpr,
                      dot_a: tl.constexpr, dot_b: tl.constexpr, _semantic=None):
    half, mi = (tl.core._unwrap_if_constexpr(value) for value in (half, mi))
    values = list(acc)
    m_tiles = len(values)
    a_value = tlx.require_layout(a[half * m_tiles + mi], dot_a, pin=False, _semantic=_semantic)
    b_value = tlx.require_layout(b[half], dot_b, pin=False, _semantic=_semantic)
    scale_layout = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (1, )),
        stride=((2, 4, 8, 16, 32, 1, 64, 128), (0, )),
    )
    sa = tlx.require_layout(sa_words[mi // 2], scale_layout, pin=False, _semantic=_semantic)
    sb = tlx.require_layout(sb, scale_layout, pin=False, _semantic=_semantic)
    c = tlx.require_layout(values[mi], mma, pin=False, _semantic=_semantic)
    values[mi] = tlx.amd_scheduled_mfma_scaled(
        a_value,
        b_value,
        c,
        sa,
        sb,
        "e4m3",
        "e4m3",
        accumulator_role="persistent",
        accumulator_register_class="agpr",
        a_scale_selector=half * 2 + mi % 2,
        b_scale_selector=half * 2,
        _semantic=_semantic,
    )
    return tl.tuple(values)


@triton.jit
def _gemm_native_short_m(A, B, SA, SB, C, WORKSPACE, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, SHUFFLED: tl.constexpr = False,
                         CUBLAS_BLOCKED: tl.constexpr = False):
    tl.static_assert(M <= 128 and N % 32 == 0)
    tl.static_assert(K % 128 == 0)
    panels: tl.constexpr = triton.cdiv(K // 128, SPLIT_K)
    tl.static_assert(BLOCK_M == 64 or BLOCK_M == 96 or BLOCK_M == 128)
    m_tiles: tl.constexpr = BLOCK_M // 32
    gm: tl.constexpr = triton.cdiv(M, BLOCK_M)
    gn: tl.constexpr = triton.cdiv(N, 128)
    total: tl.constexpr = gm * gn
    raw_pid = tl.program_id(0)
    split_id = raw_pid // total
    pid = raw_pid % total
    pm = pid % gm
    pn = pid // gm
    first_panel = split_id * panels
    mma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], False, [4, 1, 1])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    a_bases: tl.constexpr = [
        [0, 1],
        [0, 2],
        [0, 4],
        [0, 8],
        [0, 16],
        [0, 32],
        [0, 64],
        [1, 0],
        [2, 16],
        [4, 32],
        [8, 64],
        [16, 0],
        [32, 0],
    ]
    b_bases: tl.constexpr = [
        [0, 1],
        [0, 2],
        [0, 4],
        [0, 8],
        [0, 16],
        [0, 32],
        [0, 64],
        [1, 0],
        [2, 16],
        [4, 32],
        [8, 64],
        [16, 0],
        [32, 0],
        [64, 0],
    ]
    a_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in a_bases], [],
                                                               alignment=16)
    b_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in b_bases], [],
                                                               alignment=16)
    a0_low = tlx.local_alloc((1, 64, 128), tlx.dtype_of(A), 1, layout=a_shared)
    a0_high = (tlx.local_alloc((1, 64, 128), tlx.dtype_of(A), 1, layout=a_shared) if BLOCK_M > 64 else a0_low)
    b0 = tlx.local_alloc((1, 128, 128), tlx.dtype_of(B), 1, layout=b_shared)
    a1_low = tlx.local_alloc((1, 64, 128), tlx.dtype_of(A), 1, layout=a_shared)
    a1_high = (tlx.local_alloc((1, 64, 128), tlx.dtype_of(A), 1, layout=a_shared) if BLOCK_M > 64 else a1_low)
    b1 = tlx.local_alloc((1, 128, 128), tlx.dtype_of(B), 1, layout=b_shared)
    target = C if SPLIT_K == 1 else WORKSPACE + split_id * M * N
    a_low_stages = tl.tuple([a0_low, a1_low])
    a_high_stages = tl.tuple([a0_high, a1_high])
    b_stages = tl.tuple([b0, b1])
    _nshort_short_publish(A, B, pm, pn, first_panel, a0_low, a0_high, b0, M, N, K, BLOCK_M)
    tlx.async_load_wait_group(0)
    tlx.workgroup_barrier()
    a = tl.tuple([
        _nshort_short_read_a(a0_low, a0_high, index % m_tiles, index // m_tiles, dot_a) for index in range(2 * m_tiles)
    ])
    b = tl.tuple([_nshort_short_read_b(b0, half, dot_b) for half in range(2)])
    if panels > 1:
        _nshort_short_publish(A, B, pm, pn, first_panel + 1, a1_low, a1_high, b1, M, N, K, BLOCK_M)
    acc = tl.tuple([tlx.zeros((4, 32, 32), tl.float32, layout=mma) for _ in range(m_tiles)])
    cache_natural_scales: tl.constexpr = (not SHUFFLED and not CUBLAS_BLOCKED
                                          and (K == 1024 or K == 1792 or K == 3584 or K == 18048))
    quad_natural_scales: tl.constexpr = (not SHUFFLED and not CUBLAS_BLOCKED and K == 25600)
    if cache_natural_scales:
        cached_sa = tl.tuple([])
        cached_sb = tl.tuple([])
        if K == 3584:
            for scale_quad in tl.static_range(triton.cdiv(panels, 4)):
                s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b = (_nshort_short_natural_scale_word_quad(
                    SA, SB, pm, pn, first_panel + scale_quad * 4, M, N, K, BLOCK_M))
                cached_sa += tl.tuple([s0a, s1a, s2a, s3a])
                cached_sb += tl.tuple([s0b, s1b, s2b, s3b])
        else:
            for scale_pair in tl.static_range(panels // 2):
                even_sa, even_sb, odd_sa, odd_sb = (_nshort_short_natural_scale_word_pair(
                    SA, SB, pm, pn, first_panel // 2 + scale_pair, M, N, K, BLOCK_M))
                cached_sa += tl.tuple([even_sa, odd_sa])
                cached_sb += tl.tuple([even_sb, odd_sb])
        sa, sb = cached_sa[0], cached_sb[0]
    elif quad_natural_scales:
        q0a, q0b, q1a, q1b, q2a, q2b, q3a, q3b = (_nshort_short_natural_scale_word_quad(
            SA, SB, pm, pn, first_panel, M, N, K, BLOCK_M))
        quad_sa = tl.tuple([q0a, q1a, q2a, q3a])
        quad_sb = tl.tuple([q0b, q1b, q2b, q3b])
        sa, sb = quad_sa[0], quad_sb[0]
    else:
        sa, sb = _nshort_short_scale_words(SA, SB, pm, pn, first_panel, M, N, K, BLOCK_M, SHUFFLED, CUBLAS_BLOCKED)
    for panel in tl.static_range(panels):
        for mi in tl.static_range(m_tiles):
            acc = _nshort_short_dot(acc, a, b, sa, sb, 0, mi, mma, dot_a, dot_b)
        if panel + 1 < panels:
            if cache_natural_scales:
                next_sa = cached_sa[panel + 1]
                next_sb = cached_sb[panel + 1]
            elif quad_natural_scales:
                if (panel + 1) % 4 == 0:
                    q0a, q0b, q1a, q1b, q2a, q2b, q3a, q3b = (_nshort_short_natural_scale_word_quad(
                        SA, SB, pm, pn, first_panel + panel + 1, M, N, K, BLOCK_M))
                    quad_sa = tl.tuple([q0a, q1a, q2a, q3a])
                    quad_sb = tl.tuple([q0b, q1b, q2b, q3b])
                next_sa = quad_sa[(panel + 1) % 4]
                next_sb = quad_sb[(panel + 1) % 4]
            else:
                next_sa, next_sb = _nshort_short_scale_words(SA, SB, pm, pn, first_panel + panel + 1, M, N, K, BLOCK_M,
                                                             SHUFFLED, CUBLAS_BLOCKED)
        for mi in tl.static_range(m_tiles):
            acc = _nshort_short_dot(acc, a, b, sa, sb, 1, mi, mma, dot_a, dot_b)
        if panel + 2 < panels:
            _nshort_short_publish(A, B, pm, pn, first_panel + panel + 2, a_low_stages[panel % 2],
                                  a_high_stages[panel % 2], b_stages[panel % 2], M, N, K, BLOCK_M)
        if panel + 1 < panels:
            tlx.async_load_wait_group(1 if panel + 2 < panels else 0)
            tlx.workgroup_barrier()
            a = tl.tuple([
                _nshort_short_read_a(a_low_stages[1 - panel % 2], a_high_stages[1 - panel % 2], index % m_tiles,
                                     index // m_tiles, dot_a) for index in range(2 * m_tiles)
            ])
            b = tl.tuple([_nshort_short_read_b(b_stages[1 - panel % 2], half, dot_b) for half in range(2)])
            sa, sb = next_sa, next_sb
    acc = tlx.amd_mfma_commit(acc)
    wave = tl.arange(0, 4)[:, None, None]
    rows32 = tl.arange(0, 32)[None, :, None]
    cols32 = tl.arange(0, 32)[None, None, :]
    for mi in tl.static_range(m_tiles):
        rows = pm * BLOCK_M + mi * 32 + rows32
        cols = pn * 128 + wave * 32 + cols32
        offsets = tlx.require_layout(tl.broadcast_to(rows * N + cols, [4, 32, 32]), mma)
        mask = tlx.require_layout(tl.broadcast_to((rows < M) & (cols < N), [4, 32, 32]), mma)
        value = tlx.require_layout(acc[mi].to(tlx.dtype_of(target)), mma)
        tlx.buffer_store(value, target, offsets, mask=mask)


# -----------------------------------------------------------------------------
# Native Split4 Wide
# -----------------------------------------------------------------------------


@triton.jit
def _gemm_native_split4_wide(A, B, SA, SB, C, WORKSPACE, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr, WARPS_M: tl.constexpr, SPLIT_K: tl.constexpr,
                             A_SHUFFLED: tl.constexpr = False, B_SHUFFLED: tl.constexpr = False):
    tl.static_assert(BM == 128 and BN == 256 and WARPS_M == 2 and SPLIT_K == 4)
    tl.static_assert(K % (128 * SPLIT_K) == 0 and M % 128 == 0 and N % 128 == 0)
    piece_panels: tl.constexpr = K // (128 * SPLIT_K)
    tl.static_assert(piece_panels >= 2 and piece_panels % 2 == 0)
    gm: tl.constexpr = triton.cdiv(M, 128)
    gn: tl.constexpr = triton.cdiv(N, 256)
    raw_pid = tl.program_id(0)
    pm, pn, split_id = _ns8_split_major_tile(raw_pid, gm, gn, SPLIT_K)
    first_panel = split_id * piece_panels
    tl.assume((pm >= 0) & (pm < gm))
    tl.assume((pn >= 0) & (pn < gn))
    mma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], False, [4, 1, 1])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    a_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 16], [4, 32],
                             [8, 64], [16, 0], [32, 0], [64, 0]]
    b_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 16], [4, 32],
                             [8, 64], [16, 0], [32, 0]]
    a_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in a_bases], [],
                                                               alignment=16)
    b_shared: tl.constexpr = tlx.shared_linear_layout_encoding([[0, pair[0], pair[1]] for pair in b_bases], [],
                                                               alignment=16)
    a0 = tlx.local_alloc((1, 128, 128), tlx.dtype_of(A), 1, layout=a_shared)
    b0 = tlx.local_alloc((1, 64, 128), tlx.dtype_of(B), 4, layout=b_shared)
    a1 = tlx.local_alloc((1, 128, 128), tlx.dtype_of(A), 1, layout=a_shared)
    b1 = tlx.local_alloc((1, 64, 128), tlx.dtype_of(B), 4, layout=b_shared)
    a_stages = tl.tuple([a0, a1])
    b_stages = tl.tuple([b0, b1])
    _ns8_fly_layout_publish(A, B, pm, pn, first_panel, a0, b0, M, N, K)
    words_a, words_b = _ns8_packed_scale_initial(SA, SB, pm, pn, first_panel, M, N, K, A_SHUFFLED, B_SHUFFLED)
    tlx.async_load_wait_group(0)
    tlx.workgroup_barrier()
    operands_a = tl.tuple([_ns4_fly_layout_read_a(a0, index % 2, index // 2, dot_a) for index in range(4)])
    operands_b = tl.tuple([_ns8_fly_layout_read_b(b0, index % 4, index // 4, dot_b) for index in range(8)])
    if piece_panels > 1:
        _ns8_fly_layout_publish(A, B, pm, pn, first_panel + 1, a1, b1, M, N, K)
    acc = tl.tuple([tlx.zeros((4, 32, 32), tl.float32, layout=mma) for _ in range(8)])
    loop_end: tl.constexpr = max(((piece_panels - 2) // 2) * 2, 0)
    if loop_end > 0:
        for stage in tl.static_range(2):
            acc, operands_a, operands_b, words_a, words_b = _ns8_fly_layout_step(
                A, B, SA, SB, pm, pn, first_panel + stage, a_stages[stage], b_stages[stage], a_stages[1 - stage],
                b_stages[1 - stage], operands_a, operands_b, words_a, words_b, acc, mma, dot_a, dot_b, M, N, K, True,
                True, A_SHUFFLED, B_SHUFFLED)
        for panel in tl.range(2, loop_end, 2, num_stages=1):
            for stage in tl.static_range(2):
                acc, operands_a, operands_b, words_a, words_b = _ns8_fly_layout_step(
                    A, B, SA, SB, pm, pn, first_panel + panel + stage, a_stages[stage], b_stages[stage],
                    a_stages[1 - stage], b_stages[1 - stage], operands_a, operands_b, words_a, words_b, acc, mma, dot_a,
                    dot_b, M, N, K, True, True, A_SHUFFLED, B_SHUFFLED)
    for panel in tl.static_range(loop_end, piece_panels):
        acc, operands_a, operands_b, words_a, words_b = _ns8_fly_layout_step(
            A, B, SA, SB, pm, pn, first_panel + panel, a_stages[panel % 2], b_stages[panel % 2],
            a_stages[1 - panel % 2], b_stages[1 - panel % 2], operands_a, operands_b, words_a, words_b, acc, mma, dot_a,
            dot_b, M, N, K, panel + 2 < piece_panels, panel + 1 < piece_panels, A_SHUFFLED, B_SHUFFLED)
    acc = tlx.amd_mfma_commit(acc)
    target = WORKSPACE + split_id * M * N
    wave = tl.arange(0, 4)[:, None, None]
    rows32 = tl.arange(0, 32)[None, :, None]
    cols32 = tl.arange(0, 32)[None, None, :]
    store_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
        stride=((4, 8, 16, 32, 64, 128, 1024, 2048), (1, 2, 256, 512)),
    )
    for ni in tl.static_range(4):
        for mi in tl.static_range(2):
            rows = pm * 128 + wave // 2 * 64 + mi * 32 + rows32
            cols = pn * 256 + wave % 2 * 128 + ni * 32 + cols32
            value = tlx.require_layout(acc[mi * 4 + ni].to(tlx.dtype_of(target)), store_layout,
                                       late_address_compute=True)
            offsets = tlx.require_layout(rows * N + cols, store_layout)
            mask = tlx.require_layout((rows < M) & (cols < N), store_layout)
            tlx.buffer_store(value, target, offsets, mask=mask)
