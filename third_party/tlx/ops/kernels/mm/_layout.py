from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class DescriptorLayout:
    source: Any
    row_major: bool
    row_stride: int


def _descriptor_properties(rows: int, cols: int, strides: tuple[int, int]) -> tuple[bool, int] | None:
    stride_row, stride_col = strides
    if stride_col == 1 and stride_row >= cols:
        return True, stride_row
    if stride_row == 1 and stride_col >= rows:
        return False, stride_col
    return None


def shape_has_tma_compatible_strides(M, N, K, a_strides, b_strides, element_size: int) -> bool:
    """Whether a captured MM shape can use aligned row-major TMA descriptors."""

    # ``operand`` realizes a broadcast-looking stride on a singleton row as a
    # normal contiguous tensor. Its recorded stride addresses no second row.
    if M == 1 and a_strides == (0, 1):
        a_strides = (K, 1)
    if K == 1 and b_strides == (0, 1):
        b_strides = (N, 1)

    a_layout = _descriptor_properties(M, K, a_strides)
    b_layout = _descriptor_properties(K, N, b_strides)
    if a_layout is None or b_layout is None:
        return False
    row_strides = (a_layout[1], b_layout[1], N)
    return all(stride * element_size % 16 == 0 for stride in row_strides)


def descriptor_layout(tensor, name: str) -> DescriptorLayout:
    """Normalize a supported 2D tensor view for a row-major TMA descriptor."""
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")

    rows, cols = tensor.shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{name} must have positive dimensions, got {tuple(tensor.shape)}")
    tensor_type = type(tensor).__name__
    if tensor_type not in ("FakeTensor", "FunctionalTensor") and tensor.data_ptr() % 16 != 0:
        raise ValueError(f"{name} base pointer must be 16-byte aligned")
    properties = _descriptor_properties(rows, cols, tensor.stride())
    if properties is None:
        raise ValueError(f"{name} has unsupported shape/strides {tuple(tensor.shape)}/{tuple(tensor.stride())}; "
                         "expected row-major or column-major storage without broadcast or overlap")
    row_major, row_stride = properties
    if row_major:
        source = tensor
    else:
        source = tensor.T

    if row_stride * tensor.element_size() % 16 != 0:
        raise ValueError(f"{name} descriptor row stride {row_stride} elements is not 16-byte aligned "
                         f"for {tensor.dtype}")
    return DescriptorLayout(source=source, row_major=row_major, row_stride=row_stride)
