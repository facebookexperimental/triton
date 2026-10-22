from __future__ import annotations

import torch
import triton
import triton.language as tl

MXFP8_BLOCK_SIZE = 32
E8M0_NEUTRAL_SCALE = 127
E8M0_PAD_SCALE = 1
SF_ROW_TILE = 128
SF_COL_TILE = 4
SF_BLOCK_BYTES = 512
SF_BLOCK_ROWS_2D = 32
SF_BLOCK_COLS_2D = 16
SCALE_LAYOUT_ROWMAJOR = 0
SCALE_LAYOUT_MMA = 1
SCALE_LAYOUT_CUBLAS = 2
SCALE_LAYOUT_DUAL = 3
SCALE_LAYOUT_NAMES = {
    SCALE_LAYOUT_ROWMAJOR: 'rowmajor', SCALE_LAYOUT_MMA: 'mma', SCALE_LAYOUT_CUBLAS: 'cublas', SCALE_LAYOUT_DUAL: 'dual'
}
SCALE_LAYOUT_IDS = {
    'rowmajor': SCALE_LAYOUT_ROWMAJOR, 'row': SCALE_LAYOUT_ROWMAJOR, 'mma': SCALE_LAYOUT_MMA, 'cublas':
    SCALE_LAYOUT_CUBLAS, 'dual': SCALE_LAYOUT_DUAL
}


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def num_scale_cols(dim: int) -> int:
    """Number of E8M0 scale columns covering ``dim`` elements."""
    return ceil_div(dim, MXFP8_BLOCK_SIZE)


def num_row_blocks(rows: int) -> int:
    """``rest_m`` -- number of 128-row scale tiles."""
    return ceil_div(rows, SF_ROW_TILE)


def num_col_blocks(scale_cols: int) -> int:
    """``rest_k`` / ``n_col_blocks`` -- number of 4-column scale tiles."""
    return ceil_div(scale_cols, SF_COL_TILE)


def rowmajor_scale_shape(rows: int, scale_cols: int) -> tuple[int, int]:
    return (rows, scale_cols)


def mma_scale_numel(rows: int, scale_cols: int) -> int:
    """Byte count of the flat MMA-tiled scale buffer."""
    return num_row_blocks(rows) * num_col_blocks(scale_cols) * SF_BLOCK_BYTES


def cublas_blocked_scale_shape(rows: int, scale_cols: int) -> tuple[int, int]:
    """2D shape of the cuBLAS blocked scale tensor."""
    return (num_row_blocks(rows) * SF_BLOCK_ROWS_2D, num_col_blocks(scale_cols) * SF_BLOCK_COLS_2D)


def mblock_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    """cuBLAS blocked shape of the transposed (M-block) scales of a [rows, cols] tile."""
    return cublas_blocked_scale_shape(cols, num_scale_cols(rows))


def scale_shape_for_layout(layout: int, rows: int, scale_cols: int) -> tuple[int, ...]:
    """Allocation shape of a single-target scale tensor in ``layout``."""
    if layout == SCALE_LAYOUT_ROWMAJOR:
        return rowmajor_scale_shape(rows, scale_cols)
    if layout == SCALE_LAYOUT_MMA:
        return (mma_scale_numel(rows, scale_cols), )
    if layout == SCALE_LAYOUT_CUBLAS:
        return cublas_blocked_scale_shape(rows, scale_cols)
    raise ValueError(
        f'scale_shape_for_layout: unsupported layout {layout}; use alloc_dual_scales / mblock_scale_shape for SCALE_LAYOUT_DUAL'
    )


def alloc_scales(layout: int, rows: int, scale_cols: int, device: torch.device | str,
                 fill: int | None = E8M0_NEUTRAL_SCALE, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    """Allocate a scale tensor in ``layout``."""
    shape = scale_shape_for_layout(layout, rows, scale_cols)
    if fill is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return torch.full(shape, fill, dtype=dtype, device=device)


@triton.jit
def rowmajor_scale_offset(row, scale_col, stride_row, stride_col):
    """Element offset of ``(row, scale_col)`` in a strided row-major scale tensor."""
    return row * stride_row + scale_col * stride_col


@triton.jit
def mma_scale_offset(row, scale_col, rest_k):
    """Byte offset of ``(row, scale_col)`` in the flat MMA-tiled scale buffer."""
    atom_m0 = row % 32
    atom_m1 = row // 32 % 4
    rest_m_idx = row // 128
    atom_k = scale_col % 4
    rest_k_idx = scale_col // 4
    return (rest_m_idx * rest_k + rest_k_idx) * 512 + atom_m0 * 16 + atom_m1 * 4 + atom_k


@triton.jit
def cublas_blocked_scale_offset(row, scale_col, n_col_blocks):
    """Byte offset of ``(row, scale_col)`` in the cuBLAS blocked scale tensor."""
    row_block = row // 128
    col_block = scale_col // 4
    inner_row = row % 32
    sub_row = row // 32 % 4
    local_col = scale_col % 4
    return (row_block * n_col_blocks + col_block) * 512 + inner_row * 16 + sub_row * 4 + local_col


@triton.jit
def mxfp8_scale_offset(row, scale_col, blocks_k, stride_row, LAYOUT: tl.constexpr):
    """Constexpr-dispatched offset for any single-target scale layout."""
    if LAYOUT == 0:
        offset = rowmajor_scale_offset(row, scale_col, stride_row, 1)
    elif LAYOUT == 1:
        offset = mma_scale_offset(row, scale_col, blocks_k)
    elif LAYOUT == 2:
        offset = cublas_blocked_scale_offset(row, scale_col, blocks_k)
    else:
        tl.static_assert(False, 'mxfp8_scale_offset: LAYOUT must be ROWMAJOR, MMA or CUBLAS')
    return offset


@triton.jit
def store_rowmajor_scales(SF, row, scale_col, vals, stride_row, stride_col, mask):
    tl.store(SF + rowmajor_scale_offset(row, scale_col, stride_row, stride_col), vals, mask=mask)


@triton.jit
def store_mma_scales(SF, row, scale_col, vals, rest_k, mask):
    tl.store(SF + mma_scale_offset(row, scale_col, rest_k), vals, mask=mask)


@triton.jit
def store_cublas_blocked_scales(SF, row, scale_col, vals, n_col_blocks, mask):
    tl.store(SF + cublas_blocked_scale_offset(row, scale_col, n_col_blocks), vals, mask=mask)


@triton.jit
def store_scales(SF, row, scale_col, vals, mask, stride_row, stride_col, n_col_blocks, LAYOUT: tl.constexpr):
    """Constexpr-dispatched single-target scale store."""
    if LAYOUT == 0:
        store_rowmajor_scales(SF, row, scale_col, vals, stride_row, stride_col, mask)
    elif LAYOUT == 1:
        store_mma_scales(SF, row, scale_col, vals, n_col_blocks, mask)
    elif LAYOUT == 2:
        store_cublas_blocked_scales(SF, row, scale_col, vals, n_col_blocks, mask)
    else:
        tl.static_assert(False, 'store_scales: LAYOUT must be ROWMAJOR, MMA or CUBLAS')
