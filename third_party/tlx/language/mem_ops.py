from typing import Optional
import triton.language.core as tl
from triton.language.semantic import (
    _convert_elem_to_ir_value,
    _str_to_load_cache_modifier,
    _str_to_eviction_policy,
)

from . import types as tlx


@tl.builtin
def local_alloc(
    shape: tuple,
    dtype: tl.dtype,
    num: tl.constexpr,
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
    return tlx.buffered_tensor(
        _builder.create_local_alloc(full_shape, elem_type), dtype
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
    return tlx.buffered_tensor(
        _builder.create_memdesc_subview(local_allocated_buffers.handle, buffer_idx),
        local_allocated_buffers.type,
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
    )
