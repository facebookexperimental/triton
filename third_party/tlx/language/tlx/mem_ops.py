import triton.language.core as tl

from . import types as tlx
from .utility import cuda_parse_arch
from .mma_ops import require_nv_mma_shared_layout
from typing import Optional, Tuple, overload


def _assert_blackwell_for_tmem(arch):
    capability = int(cuda_parse_arch(arch))
    assert capability >= 100, "tmem is only available on Blackwell"


def _create_tmem_compatible_tensor_layout_encoding(
    builder,
    tensor: tlx.buffered_tensor,
):
    module_num_warps = builder.options.num_warps
    assert module_num_warps > 0, "tmem load requires num_warps > 0"
    num_ctas = builder.options.num_ctas
    assert num_ctas > 0, "tmem load requires num_ctas > 0"
    threads_per_warp = 32
    return builder.make_default_tmem_compatible_tensor_layout_encoding(list(tensor.shape), tensor.dtype.to_ir(builder),
                                                                       module_num_warps, threads_per_warp, num_ctas)


@tl.builtin
def local_alloc(
    shape: tuple,
    dtype: tl.dtype,
    num: tl.constexpr,
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    reuse: Optional[tlx.buffered_tensor] = None,
    layout: Optional[tlx.shared_layout_encoding] = None,
    _semantic=None,
) -> tlx.buffered_tensor:
    """
    Allocates buffer in shared memory and return a view of the buffer.
    """
    if storage == tlx.storage_kind.tmem:
        _assert_blackwell_for_tmem(_semantic.builder.options.arch)

    if not isinstance(num, tl.constexpr):
        user_error = """
`num` must be a constexpr without introducing any `ast.Assign` nodes,
otherwise its value will be wrapped as `tensor.handle`.
For example, following will fail because `num` will be promoted to tl.tensor by semantics.py
in visit_Assign
    num = tl.constexpr(2)
    local_alloc(..., num=num)

To bypass, rewrite it to `local_alloc(..., num=tl.constexpr(2))` or `local_alloc(..., num=2)`
        """
        raise ValueError(user_error)

    unwrapped_shape = [tl._unwrap_if_constexpr(dim) for dim in shape]
    unwrapped_num = tl._unwrap_if_constexpr(num)
    full_shape = [unwrapped_num] + unwrapped_shape
    dtype = tl._unwrap_if_constexpr(dtype)
    elem_type = dtype.to_ir(_semantic.builder)
    if layout is None:
        if storage == tlx.storage_kind.smem:
            if len(shape) == 1:
                layout = tlx.swizzled_shared_layout_encoding.make_default(rank=len(shape))
                layout_handle = _semantic.builder.make_swizzled_shared_encoding_attr(
                    layout.vectorSize,
                    layout.perPhase,
                    layout.maxPhase,
                    layout.order,
                    layout.numCTAsPerCGA,
                    layout.numCTASplit,
                    layout.numCTAOrder,
                )
            else:
                layout = tlx.nv_mma_shared_layout_encoding.make_default(shape, dtype)
                layout_handle = _semantic.builder.make_nv_mma_shared_encoding_attr(
                    [int(x) for x in layout.shape],
                    layout.order,
                    layout.elemType.to_ir(_semantic.builder),
                    layout.numCTAsPerCGA,
                    layout.numCTASplit,
                    layout.numCTAOrder,
                    layout.fp4Padded,
                )
        else:
            layout = tlx.tensor_memory_layout_encoding.make_default(shape)
            layout_handle = _semantic.builder.make_tensor_memory_encoding_attr(
                layout.blockM,
                layout.blockN,
                layout.unpacked,
                layout.CTASplitM,
                layout.CTASplitN,
            )
    else:
        raise NotImplementedError("User-specified layout encoding not yet implemented.")

    alias_handle = None
    if reuse:
        # reuse tensor has to be a buffered tensor
        if not isinstance(reuse, tlx.buffered_tensor):
            raise ValueError("reuse tensor has to be a buffered tensor")
        # verify that the reuse tensor has the same storage
        if reuse.type.storage != storage:
            raise ValueError("reuse tensor has different storage")
        alias_handle = reuse.handle

    if storage == tlx.storage_kind.smem:
        tensor_handle = _semantic.builder.create_local_alloc(full_shape, elem_type, layout_handle, alias_handle)
    else:
        tensor_handle = _semantic.builder.create_tmem_alloc(full_shape, elem_type, layout_handle, alias_handle)

    return tlx.buffered_tensor(tensor_handle, dtype, unwrapped_shape, unwrapped_num, storage, layout, _semantic)


# overload declarations just to make linter happy
@overload
def local_view(
    local_allocated_buffers: tlx.buffered_tensor,
    buffer_idx: int,
    _semantic=None,
) -> tlx.buffered_tensor:
    ...


@overload
def local_view(
    local_allocated_buffers: tlx.mbarrier,
    buffer_idx: int,
    _semantic=None,
) -> tlx.mbarrier:
    ...


@tl.builtin
def local_view(
    local_allocated_buffers: tlx.buffered_tensor | tlx.mbarrier,
    buffer_idx: int,
    _semantic=None,
) -> tlx.buffered_tensor | tlx.mbarrier:
    """
    Returns a subview of the buffer.
    """
    buffer_idx = _semantic._convert_elem_to_ir_value(buffer_idx, require_i64=False)
    view_handle = _semantic.builder.create_memdesc_subview(local_allocated_buffers.handle, buffer_idx)
    if isinstance(local_allocated_buffers, tlx.mbarrier):
        return tlx.mbarrier(view_handle, 0, local_allocated_buffers.type.layout)
    else:
        return tlx.buffered_tensor(
            view_handle,
            local_allocated_buffers.type.scalar,
            local_allocated_buffers.shape,
            0,
            local_allocated_buffers.type.storage,
            local_allocated_buffers.type.layout,
        )


def _buffered_tensor_getitem(self, buffer_idx):
    return local_view(self, buffer_idx, _semantic=self.type.semantic)


tlx.buffered_tensor.__getitem__ = _buffered_tensor_getitem
tlx.mbarrier.__getitem__ = _buffered_tensor_getitem


@tl.builtin
def subslice(
    local_allocated_buffer: tlx.buffered_tensor,
    offset: int,
    size: int,
    _semantic=None,
) -> tlx.buffered_tensor:
    """
    Returns a subslice of the buffer (in TMEM). The source has to be 128xN and the slicing is
    along the innermost dimension.

    :param local_allocated_buffer: the source buffer
    :param offset: the start offset of the subslice, in terms of number of elements
    :param size: the size of the subslice, in terms of number of elements
    """
    # this is for TMEM subslice
    assert local_allocated_buffer.type.storage == tlx.storage_kind.tmem, "subslice is only supported for tmem"
    assert isinstance(local_allocated_buffer.type, tl.block_type), "subslice src is not block type"
    subslice_shape = [dim for dim in local_allocated_buffer.type.shape[:-1]] + [size]
    return tlx.buffered_tensor(
        _semantic.builder.create_tmem_subslice(local_allocated_buffer.handle, offset, size),
        local_allocated_buffer.type.element_ty,
        subslice_shape,
        local_allocated_buffer.type.num,
        local_allocated_buffer.type.storage,
        local_allocated_buffer.type.layout,
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
    _semantic=None,
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
        _, src, mask, other = _semantic._prepare_legacy_load(src, mask, other, None, None)

    cache = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction = _semantic._str_to_eviction_policy(eviction_policy)
    return tlx.async_token(
        _semantic.builder.create_async_load(
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
    _semantic=None,
) -> tlx.async_token:
    """
    Commits all prior initiated but uncommitted async_load ops an async group.
    Each token represents a tracked async load operation.
    """
    handles = [t.handle for t in tokens]
    return tlx.async_token(_semantic.builder.create_async_commit_group(handles))


@tl.builtin
def async_load_wait_group(
    pendings: tl.constexpr,
    tokens: list[tlx.async_token] = [],
    _semantic=None,
) -> tlx.async_token:
    """
    Wait for completion of prior asynchronous copy operations.
    Each token represents a tracked async commit group operation.
    """
    pendings = tl._unwrap_if_constexpr(pendings)
    handles = [t.handle for t in tokens]
    return tlx.async_token(_semantic.builder.create_async_wait(handles, pendings))


@tl.builtin
def local_load(
    src: tlx.buffered_tensor,
    token: tlx.async_token = None,
    _semantic=None,
) -> tl.tensor:
    """
    Loads buffer from local or tensor memory into a distributed tensor.
    """
    block_type = tl.block_type(src.type.element_ty, src.type.shape)
    storage = src.type.storage
    if storage == tlx.storage_kind.tmem:
        _assert_blackwell_for_tmem(_semantic.builder.options.arch)
        tmem_compatible_layout_encoding = _create_tmem_compatible_tensor_layout_encoding(_semantic.builder, src)
        load_handle = _semantic.builder.create_tmem_load(src.handle, tmem_compatible_layout_encoding,
                                                         token.handle if token else None)
        output = _semantic.builder.create_release_layout(load_handle)
        return tl.tensor(output, block_type)
    else:
        output = _semantic.builder.create_local_load(src.handle, token.handle if token else None)
        return tl.tensor(output, block_type)


@tl.builtin
def local_store(
    dst: tlx.buffered_tensor,
    src: tl.tensor,
    _semantic=None,
) -> tl.tensor:
    """
    Store a distributed tensor into a buffer in local or tensor memory.
    """
    storage = dst.type.storage
    if storage == tlx.storage_kind.tmem:
        _assert_blackwell_for_tmem(_semantic.builder.options.arch)
        tmem_compatible_layout_encoding = _create_tmem_compatible_tensor_layout_encoding(_semantic.builder, dst)
        src_handle = _semantic.builder.create_require_layout(src.handle, tmem_compatible_layout_encoding)
        return tl.tensor(_semantic.builder.create_tmem_store(dst.handle, src_handle), tl.void)

    return tl.tensor(_semantic.builder.create_local_store(dst.handle, src.handle), tl.void)


@tl.builtin
def local_trans(input: tlx.buffered_tensor, dims: Tuple[int] = (1, 0), _semantic=None) -> tlx.buffered_tensor:
    """
        Permutes the dimensions of a tensor.

        If the parameter :code:`dims` is not specified, the function defaults to a (1,0) permutation,
        effectively transposing a 2D tensor.

        :param input: The input tensor.
        :param dims: The desired ordering of dimensions.  For example,
            :code:`(2, 1, 0)` reverses the order dims in a 3D tensor.
    """
    if len(input.type.shape) != len(dims):
        raise ValueError("permute dims must have the same length as input shape")
    if sorted(tl._unwrap_if_constexpr(d) for d in dims) != list(range(len(dims))):
        raise ValueError(f"permute dims must be a permutation of 0, 1, ..., n-1, but were {dims}")

    permuted_handle = _semantic.builder.create_memdesc_trans(input.handle, dims)
    return input.make_permute(permuted_handle, dims)


@tl.builtin
def local_reinterpret(src: tlx.buffered_tensor, dtype: tl.dtype, shape: list[tl.constexpr] = None,
                      _semantic=None) -> tlx.buffered_tensor:
    """
        Reinterpret the dtype and shape of a buffered tensor. Layout is preserved.
    """
    if shape is None:
        shape = src.type.shape
    else:
        assert isinstance(
            src, tlx.buffered_tensor
        ) and src.type.storage == tlx.storage_kind.smem, "TLX local_reinterpret with reshaping only supports SMEM"

    reinterpreted_value_handle = _semantic.builder.create_memdesc_reinterpret(src.handle,
                                                                              dtype.to_ir(_semantic.builder), shape)
    return tlx.buffered_tensor(
        reinterpreted_value_handle,
        dtype,
        shape,
        src.type.num,
        src.type.storage,
        src.type.layout,
    )


@tl.builtin
def async_descriptor_load(
    desc: tl.tensor_descriptor_base,
    result: tlx.buffered_tensor,
    offsets: list[tl.tensor],
    barrier: tlx.mbarrier,
    pred: tl.tensor = None,
    cache_modifier: str = "",
    eviction_policy: str = "",
    _semantic=None,
) -> None:
    assert isinstance(desc, tl.tensor_descriptor_base)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"
    result_handle = require_nv_mma_shared_layout(result, _semantic.builder)
    offsets = _semantic._convert_to_ir_values(offsets, require_i64=False)
    cache = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction = _semantic._str_to_eviction_policy(eviction_policy)
    if pred is None:
        pred_handle = _semantic.builder.get_int1(True)
    else:
        pred_handle = pred.handle
    _semantic.builder.create_async_TMA_load(desc.handle, offsets, barrier.handle, pred_handle, result_handle, cache,
                                            eviction, False)


@tl.builtin
def async_descriptor_store(
    desc: tl.tensor_descriptor_base,
    source: tlx.buffered_tensor,
    offsets: list[tl.tensor],
    _semantic=None,
) -> None:
    assert isinstance(desc, tl.tensor_descriptor_base)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"
    source_handle = require_nv_mma_shared_layout(source, _semantic.builder)
    offsets = _semantic._convert_to_ir_values(offsets, require_i64=False)
    _semantic.builder.create_async_TMA_store(desc.handle, offsets, source_handle)


@tl.builtin
def async_descriptor_store_wait(
    pendings: tl.constexpr,
    _semantic=None,
) -> None:
    """
    Wait for completion of prior asynchronous TMA store operations.
    """
    pendings = tl._unwrap_if_constexpr(pendings)
    _semantic.builder.create_async_TMA_store_wait(pendings)


@tl.builtin
def fence_async_shared(
    _semantic=None,
) -> None:
    """
    Order memory operations that go through the shared memory.
    """
    _semantic.builder.create_fence_async_shared(False)
