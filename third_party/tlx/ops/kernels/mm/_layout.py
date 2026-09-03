from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class DescriptorLayout:
    source: Any
    row_major: bool
    row_stride: int


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
    stride_row, stride_col = tensor.stride()
    if stride_col == 1 and stride_row >= cols:
        source = tensor
        row_major = True
    elif stride_row == 1 and stride_col >= rows:
        source = tensor.T
        row_major = False
    else:
        raise ValueError(f"{name} has unsupported shape/strides {tuple(tensor.shape)}/{tuple(tensor.stride())}; "
                         "expected row-major or column-major storage without broadcast or overlap")

    row_stride = source.stride(0)
    if row_stride * tensor.element_size() % 16 != 0:
        raise ValueError(f"{name} descriptor row stride {row_stride} elements is not 16-byte aligned "
                         f"for {tensor.dtype}")
    return DescriptorLayout(source=source, row_major=row_major, row_stride=row_stride)
