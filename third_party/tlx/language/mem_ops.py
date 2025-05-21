from typing import Optional
import triton.language.core as tl
from triton.language.semantic import (
    _convert_elem_to_ir_value,
    _str_to_load_cache_modifier,
    _str_to_eviction_policy,
)

from . import types as tlx
from typing import Optional


@tl.builtin
def local_alloc(
    shape: tuple,
    dtype: tl.dtype,
    num: tl.constexpr,
    layout: Optional[tlx.shared_layout_encoding] = None,
    _builder=None,
) -> tlx.buffered_tensor:
    """
    Allocates buffer in shared memory and return a view of the buffer.
    """
    unwrapped_shape = [tl._constexpr_to_value(dim) for dim in shape]
    unwrapped_num = tl._constexpr_to_value(num)
    full_shape = [unwrapped_num] + unwrapped_shape
    dtype = tl._constexpr_to_value(dtype)
    elem_type = dtype.to_ir(_builder)
    block_type = tl.block_type(dtype, full_shape)
    if layout is None:
        layout = tlx.swizzled_shared_layout_encoding.make_default(rank=len(shape))
        layout_handle = _builder.make_swizzled_shared_encoding_attr(
            layout.vectorSize,
            layout.perPhase,
            layout.maxPhase,
            layout.order,
            layout.numCTAsPerCGA,
            layout.numCTASplit,
            layout.numCTAOrder,
        )
    else:
        raise NotImplementedError("User-specified layout encoding not yet implemented.")
    return tlx.buffered_tensor(
        _builder.create_local_alloc(full_shape, elem_type, layout_handle),
        block_type,
        layout,
    )


@tl.builtin
def local_view(
    local_allocated_buffers: tlx.buffered_tensor,
    buffer_idx: int,
    _builder=None,
) -> tlx.buffered_tensor:
    """
    Returns a subview of the buffer.
    """
    buffer_idx = _convert_elem_to_ir_value(_builder, buffer_idx, require_i64=False)
    buffer_type = local_allocated_buffers.type
    # A subview of a one-dimensional buffer is still one-dimensional.
    view_shape = (
        buffer_type.shape[1:] if len(buffer_type.shape) > 1 else buffer_type.shape
    )
    view_type = tl.block_type(buffer_type.element_ty, view_shape)
    layout = local_allocated_buffers.layout
    return tlx.buffered_tensor(
        _builder.create_memdesc_subview(local_allocated_buffers.handle, buffer_idx),
        view_type,
        layout,
    )


@tl.builtin
def async_load(
    src: tl.tensor,
    result: tlx.buffered_tensor,
    mask: Optional[tl.tensor] = None,
    other: Optional[tl.tensor] = None,
    cache_modifier: str = "",
    eviction_policy: str = "",
    is_volatile: bool = False,
    _builder=None,
) -> tlx.buffered_tensor:
    """
    Loads buffer from global to local memory asynchronously.
    """
    cache = _str_to_load_cache_modifier(cache_modifier)
    eviction = _str_to_eviction_policy(eviction_policy)
    return tlx.buffered_tensor(
        _builder.create_async_load(
            src.handle,
            result.handle,
            mask.handle if mask else None,
            other.handle if other else None,
            cache,
            eviction,
            is_volatile,
        ),
        result.type,
        result.layout,
    )
