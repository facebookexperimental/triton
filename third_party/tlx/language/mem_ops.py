import triton.language.core as tl
from triton.language.semantic import (
    _convert_elem_to_ir_value,
    _convert_to_ir_values,
    _str_to_load_cache_modifier,
    _str_to_eviction_policy,
    _prepare_legacy_load,
)

from . import types as tlx
from .utility import cuda_parse_arch
from .mma_ops import require_nv_mma_shared_layout
from typing import Optional, Tuple


def _assert_blackwell_for_tmem(arch):
    capability = int(cuda_parse_arch(arch))
    assert capability >= 100, "tmem is only available on Blackwell"


def _create_tmem_compatible_tensor_layout_encoding(
    _builder,
    tensor: tlx.buffered_tensor,
):
    num_warps = _builder.options.num_warps
    assert num_warps > 0, "tmem load requires num_warps > 0"
    num_ctas = _builder.options.num_ctas
    assert num_ctas > 0, "tmem load requires num_ctas > 0"
    threads_per_warp = 32  # hard-coded for now
    return _builder.make_default_tmem_compatible_tensor_layout_encoding(list(tensor.shape),
                                                                        tensor.dtype.to_ir(_builder), num_warps,
                                                                        threads_per_warp, num_ctas)


@tl.builtin
def local_alloc(
    shape: tuple,
    dtype: tl.dtype,
    num: tl.constexpr,
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    layout: Optional[tlx.shared_layout_encoding] = None,
    _builder=None,
) -> tlx.buffered_tensor:
    """
    Allocates buffer in shared memory and return a view of the buffer.
    """
    if storage == tlx.storage_kind.tmem:
        _assert_blackwell_for_tmem(_builder.options.arch)

    unwrapped_shape = [tl._unwrap_if_constexpr(dim) for dim in shape]
    unwrapped_num = tl._unwrap_if_constexpr(num)
    full_shape = [unwrapped_num] + unwrapped_shape
    dtype = tl._unwrap_if_constexpr(dtype)
    elem_type = dtype.to_ir(_builder)
    block_type = tl.block_type(dtype, full_shape)
    if layout is None:
        if storage == tlx.storage_kind.smem:
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
            layout = tlx.tensor_memory_layout_encoding.make_default(shape)
            layout_handle = _builder.make_tensor_memory_encoding_attr(
                layout.blockM,
                layout.blockN,
                layout.unpacked,
                layout.CTASplitM,
                layout.CTASplitN,
            )
    else:
        raise NotImplementedError("User-specified layout encoding not yet implemented.")

    if storage == tlx.storage_kind.smem:
        tensor_handle = _builder.create_local_alloc(full_shape, elem_type, layout_handle)
    else:
        tensor_handle = _builder.create_tmem_alloc(full_shape, elem_type, layout_handle)

    return tlx.buffered_tensor(
        tensor_handle,
        block_type,
        storage,
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
    view_shape = (buffer_type.shape[1:] if len(buffer_type.shape) > 1 else buffer_type.shape)
    view_type = tl.block_type(buffer_type.element_ty, view_shape)
    return tlx.buffered_tensor(
        _builder.create_memdesc_subview(local_allocated_buffers.handle, buffer_idx),
        view_type,
        local_allocated_buffers.storage,
        local_allocated_buffers.layout,
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
) -> tlx.async_token:
    """
    Loads buffer from global to local memory asynchronously.
    """
    if src.type.is_ptr() and src.type.element_ty.is_block():
        # Load by a block pointer: `pointer_type<block_type<>>`
        # unsupported for now
        raise NotImplementedError("async_load by block pointer is not supported yet")
    else:
        # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
        _, mask, other = _prepare_legacy_load(src, mask, other, None, None, _builder)

    cache = _str_to_load_cache_modifier(cache_modifier)
    eviction = _str_to_eviction_policy(eviction_policy)
    return tlx.async_token(
        _builder.create_async_load(
            src.handle,
            result.handle,
            mask.handle if mask else None,
            other.handle if other else None,
            cache,
            eviction,
            is_volatile,
        ))


@tl.builtin
def async_load_commit_group(
    tokens: list[tlx.async_token] = [],
    _builder=None,
) -> tlx.async_token:
    """
    Commits all prior initiated but uncommitted async_load ops an async group.
    Each token represents a tracked async load operation.
    """
    handles = [t.handle for t in tokens]
    return tlx.async_token(_builder.create_async_commit_group(handles))


@tl.builtin
def async_load_wait_group(
    pendings: tl.constexpr,
    tokens: list[tlx.async_token] = [],
    _builder=None,
) -> tlx.async_token:
    """
    Wait for completion of prior asynchronous copy operations.
    Each token represents a tracked async commit group operation.
    """
    pendings = tl._unwrap_if_constexpr(pendings)
    handles = [t.handle for t in tokens]
    return tlx.async_token(_builder.create_async_wait(handles, pendings))


@tl.builtin
def local_load(
    src: tlx.buffered_tensor,
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    token: tlx.async_token = None,
    _builder=None,
) -> tl.tensor:
    """
    Loads buffer from local or tensor memory into a distributed tensor.
    """
    if storage == tlx.storage_kind.tmem:
        _assert_blackwell_for_tmem(_builder.options.arch)
        tmem_compatible_layout_encoding = _create_tmem_compatible_tensor_layout_encoding(_builder, src)
        load_handle = _builder.create_tmem_load(src.handle, tmem_compatible_layout_encoding,
                                                token.handle if token else None)
        output = _builder.create_release_layout(load_handle)
        return tl.tensor(output, src.type)

    return tl.tensor(_builder.create_local_load(src.handle, token.handle if token else None), src.type)


@tl.builtin
def local_store(
    dst: tlx.buffered_tensor,
    src: tl.tensor,
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    _builder=None,
) -> tl.tensor:
    """
    Store a distributed tensor into a buffer in local or tensor memory.
    """
    if storage == tlx.storage_kind.tmem:
        _assert_blackwell_for_tmem(_builder.options.arch)
        tmem_compatible_layout_encoding = _create_tmem_compatible_tensor_layout_encoding(_builder, dst)
        src_handle = _builder.create_require_layout(src.handle, tmem_compatible_layout_encoding)
        return tl.tensor(_builder.create_tmem_store(dst.handle, src_handle), tl.void)

    return tl.tensor(_builder.create_local_store(dst.handle, src.handle), tl.void)


@tl.builtin
def local_trans(input: tlx.buffered_tensor, dims: Tuple[int] = (1, 0), _builder=None) -> tlx.buffered_tensor:
    """
        Permutes the dimensions of a tensor.

        If the parameter :code:`dims` is not specified, the function defaults to a (1,0) permutation,
        effectively transposing a 2D tensor.

        :param input: The input tensor.
        :param dims: The desired ordering of dimensions.  For example,
            :code:`(2, 1, 0)` reverses the order dims in a 3D tensor.
    """
    if len(input.shape) != len(dims):
        raise ValueError("permute dims must have the same length as input shape")
    if sorted(tl._unwrap_if_constexpr(d) for d in dims) != list(range(len(dims))):
        raise ValueError(f"permute dims must be a permutation of 0, 1, ..., n-1, but were {dims}")

    permuted_handle = _builder.create_memdesc_trans(input.handle, dims)
    return input.make_permute(permuted_handle, dims)


@tl.builtin
def async_descriptor_load(
    desc: tl.tensor_descriptor_base,
    result: tlx.buffered_tensor,
    offsets: list[tl.tensor],
    barrier: tlx.mbarriers,
    cache_modifier: str = "",
    eviction_policy: str = "",
    _builder=None,
) -> None:
    assert isinstance(desc, tl.tensor_descriptor_base)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"

    # Requires a row-major layout for TMA only supports continuous memory access
    result_handle = require_nv_mma_shared_layout(result, [1, 0], _builder)
    offsets = _convert_to_ir_values(_builder, offsets, require_i64=False)
    cache = _str_to_load_cache_modifier(cache_modifier)
    eviction = _str_to_eviction_policy(eviction_policy)
    _builder.create_async_TMA_load(desc.handle, offsets, barrier.handle, result_handle, cache, eviction, False)


@tl.builtin
def async_descriptor_store(
    desc: tl.tensor_descriptor_base,
    source: tlx.buffered_tensor,
    offsets: list[tl.tensor],
    _builder=None,
) -> None:
    assert isinstance(desc, tl.tensor_descriptor_base)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"

    # Requires a row-major layout for TMA only supports continuous memory access
    source_handle = require_nv_mma_shared_layout(source, [1, 0], _builder)
    offsets = _convert_to_ir_values(_builder, offsets, require_i64=False)
    _builder.create_async_TMA_store(desc.handle, offsets, source_handle)
