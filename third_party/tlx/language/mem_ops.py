from typing import Optional
import triton.language.core as tl
from triton.language.semantic import (
    _convert_elem_to_ir_value,
    _convert_to_ir_values,
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
    unwrapped_shape = [tl._unwrap_if_constexpr(dim) for dim in shape]
    unwrapped_num = tl._unwrap_if_constexpr(num)
    full_shape = [unwrapped_num] + unwrapped_shape
    dtype = tl._unwrap_if_constexpr(dtype)
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
) -> None:
    """
    Loads buffer from global to local memory asynchronously.
    """
    cache = _str_to_load_cache_modifier(cache_modifier)
    eviction = _str_to_eviction_policy(eviction_policy)
    _builder.create_async_load(
        src.handle,
        result.handle,
        mask.handle if mask else None,
        other.handle if other else None,
        cache,
        eviction,
        is_volatile,
    )


@tl.builtin
def async_descriptor_load(
    desc: tl.tensor_descriptor_base,
    result: tlx.buffered_tensor,
    offsets : list[tl.tensor],
    barrier: tlx.mbarriers,
    cache_modifier: str = "",
    eviction_policy: str = "",
    _builder=None,
) -> None:
    assert isinstance(desc, tl.tensor_descriptor_base)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"

    # TODO: require buffer layout to be NVMMASharedEncodingAttr
    # layout_handle = _builder.make_nv_mma_shared_shared_encoding_attr(...)
    # result = _builder.create_require_layout(result.handle, layout_handle)
    offsets = _convert_to_ir_values(_builder, offsets, require_i64=False)
    cache = _str_to_load_cache_modifier(cache_modifier)
    eviction = _str_to_eviction_policy(eviction_policy)
    _builder.create_async_TMA_load(
        desc.handle,
        offsets,
        barrier.handle,
        result.handle,
        cache,
        eviction,
        False)
