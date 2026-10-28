"""Host-side E8M0 scale packing shared by the MXFP8 grouped GEMM backends.

Both backends consume scales in the cuBLAS-blocked 128x4 byte layout, viewed as
``[batch, row_chunks, k_chunks, 2, 256]``: one 512-byte atom covers 128 rows x 4
scale columns, and natural ``(row, col)`` lands at byte
``(row % 32) * 16 + (row % 128 // 32) * 4 + col % 4`` within its atom.
"""

from __future__ import annotations

import torch

VEC_SIZE = 32


def _cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _as_uint8_scale(scale: torch.Tensor) -> torch.Tensor:
    return scale.view(torch.uint8)


def _swizzle_scale_to_5d(
    scale: torch.Tensor,
    *,
    batch: int,
    rows: int,
    outer_chunks: int,
    k_groups: int,
    k_chunks: int,
    k_major_chunks: bool = False,
) -> torch.Tensor:
    """Pack natural E8M0 scales using the cuBLAS 128x4 byte swizzle."""
    scale = _as_uint8_scale(scale).reshape(batch, rows, k_groups)
    padded_rows = outer_chunks * 128
    padded_cols = k_chunks * 4
    if rows != padded_rows or k_groups != padded_cols:
        padded = scale.new_zeros((batch, padded_rows, padded_cols))
        padded[:, :rows, :k_groups] = scale
        scale = padded

    # row = row_group * 32 + row_lane. Flattening the last three axes after
    # this permutation gives dest = row_lane * 16 + row_group * 4 + col, or
    # row_lane * 16 + col * 4 + row_group with k_major_chunks.
    order = (0, 1, 4, 3, 5, 2) if k_major_chunks else (0, 1, 4, 3, 2, 5)
    swizzled = scale.view(batch, outer_chunks, 4, 32, k_chunks, 4).permute(*order).contiguous()
    return swizzled.view(batch, outer_chunks, k_chunks, 2, 256)


def _view_blocked_scale(
    scale: torch.Tensor,
    shape: tuple[int, int, int, int, int],
) -> torch.Tensor:
    required = 1
    for extent in shape:
        required *= extent
    flat = _as_uint8_scale(scale).view(-1)
    if flat.numel() < required:
        raise ValueError(f"cublas_blocked scale has {flat.numel()} bytes; expected at least {required}")
    return flat[:required].view(shape)


def prepare_scales(
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    gm: int,
    g: int,
    n: int,
    k: int,
    sf_layout: str,
    k_major_chunks: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x_scale_5d, w_scale_5d)`` uint8 views in the blocked layout.

    ``k_major_chunks`` packs natural scales with every 16-byte chunk of each atom
    stored as [k][row_group] instead of [row_group][k]; it requires
    ``sf_layout == "natural"``.
    """
    if k_major_chunks and sf_layout != "natural":
        raise ValueError("k_major_chunks requires natural scales")
    k_groups = k // VEC_SIZE
    k_chunks = _cdiv(k_groups, 4)
    m_chunks = _cdiv(gm, 128)
    n_chunks = _cdiv(n, 128)

    if sf_layout == "natural":
        if x_scale.ndim != 2 or x_scale.shape[0] < gm or x_scale.shape[1] < k_groups:
            raise ValueError("natural x_scale must be a 2D tensor covering [GM, K // 32]")
        x_scale_5d = _swizzle_scale_to_5d(
            x_scale[:gm, :k_groups],
            batch=1,
            rows=gm,
            outer_chunks=m_chunks,
            k_groups=k_groups,
            k_chunks=k_chunks,
            k_major_chunks=k_major_chunks,
        )
        if w_scale.numel() != g * n * k_groups:
            raise ValueError("natural w_scale must contain exactly G * N * (K // 32) scales")
        w_scale_5d = _swizzle_scale_to_5d(
            w_scale,
            batch=g,
            rows=n,
            outer_chunks=n_chunks,
            k_groups=k_groups,
            k_chunks=k_chunks,
            k_major_chunks=k_major_chunks,
        )
    elif sf_layout == "cublas_blocked":
        x_scale_5d = _view_blocked_scale(
            x_scale,
            (1, m_chunks, k_chunks, 2, 256),
        )
        w_scale_5d = _view_blocked_scale(
            w_scale,
            (g, n_chunks, k_chunks, 2, 256),
        )
    else:
        raise ValueError(f"unsupported sf_layout {sf_layout!r}; expected 'natural' or 'cublas_blocked'")
    return x_scale_5d, w_scale_5d
