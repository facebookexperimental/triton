"""TLX Plugin memory ops: local_alloc, local_view, local_store, local_load.

Ported from triton-tlx/third_party/tlx/language/tlx/mem_ops.py.
Only includes the SMEM paths (default allocation, no alias/storageAlias).

These ops call plugin custom ops registered by TLXLocalAllocPlugin.cpp:
  - tlx_local_alloc(type_carrier, *shape_dims) -> MemDesc Value
  - tlx_local_view(alloc, buffer_idx) -> MemDesc Value
  - tlx_local_store(dst, src) -> void
  - tlx_local_load(subView) -> tensor Value
"""

from typing import Optional

import triton.language.core as tl

from . import types as tlx


# Map triton dtype to builder method name for creating a type-carrier constant
_DTYPE_TO_BUILDER_METHOD = {
    tl.float16: "get_fp16",
    tl.bfloat16: "get_bf16",
    tl.float32: "get_fp32",
    tl.float64: "get_fp64",
    tl.int8: "get_int8",
    tl.int16: "get_int16",
    tl.int32: "get_int32",
    tl.int64: "get_int64",
    tl.uint8: "get_uint8",
    tl.uint16: "get_uint16",
    tl.uint32: "get_uint32",
    tl.uint64: "get_uint64",
}


@tl.builtin
def local_alloc(
    shape: tuple,
    dtype: tl.dtype,
    num: tl.constexpr,
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    layout: Optional[tlx.shared_layout_encoding] = None,
    _semantic=None,
) -> tlx.buffered_tensor:
    """
    Allocate buffers in shared memory and return a buffered_tensor.

    Args:
        shape: Shape of each buffer (excluding the num dimension).
        dtype: Data type of the buffer elements.
        num: Number of buffers to allocate (compile-time constant).
        storage: Storage kind (only smem supported in plugin).
        layout: Optional memory layout encoding (ignored; defaults computed
                by the C++ plugin based on shape rank).

    Returns:
        A buffered_tensor representing the allocated buffers.
    """
    if storage != tlx.storage_kind.smem:
        raise NotImplementedError(
            "TLX plugin only supports smem storage for local_alloc. "
            "tmem requires a Dialect Plugin."
        )

    if not isinstance(num, tl.constexpr):
        raise ValueError(
            "`num` must be a constexpr. Use `local_alloc(..., num=tl.constexpr(2))` "
            "or `local_alloc(..., num=2)`"
        )

    unwrapped_shape = [tl._unwrap_if_constexpr(dim) for dim in shape]
    unwrapped_num = tl._unwrap_if_constexpr(num)
    full_shape = [unwrapped_num] + unwrapped_shape
    dtype = tl._unwrap_if_constexpr(dtype)

    # Create a type-carrier scalar constant of the desired element type.
    # The C++ plugin extracts the MLIR Type from this value.
    builder_method = _DTYPE_TO_BUILDER_METHOD.get(dtype)
    if builder_method is None:
        raise ValueError(f"Unsupported dtype for local_alloc: {dtype}")

    if builder_method.startswith("get_fp") or builder_method.startswith("get_bf"):
        type_carrier = getattr(_semantic.builder, builder_method)(0.0)
    else:
        type_carrier = getattr(_semantic.builder, builder_method)(0)

    # Create i32 constants for each dimension of the full shape
    shape_values = [_semantic.builder.get_int32(int(dim)) for dim in full_shape]

    # Call the composite custom op: tlx_local_alloc(type_carrier, *shape_dims)
    # This creates the encoding attribute and LocalAllocOp internally.
    args = [type_carrier] + shape_values
    tensor_handle = _semantic.builder.tlx_local_alloc(args)

    # Construct the Python-level layout object for metadata tracking
    if len(unwrapped_shape) == 1:
        py_layout = tlx.swizzled_shared_layout_encoding.make_default(
            rank=len(unwrapped_shape)
        )
    else:
        py_layout = tlx.nv_mma_shared_layout_encoding.make_default(
            unwrapped_shape, dtype
        )

    return tlx.buffered_tensor(
        tensor_handle, dtype, unwrapped_shape, unwrapped_num, storage, py_layout
    )


@tl.builtin
def local_view(
    local_allocated_buffers: tlx.buffered_tensor,
    buffer_idx: int,
    _semantic=None,
) -> tlx.buffered_tensor:
    """
    Returns a subview of the buffer at the given index.
    """
    buffer_idx = _semantic._convert_elem_to_ir_value(buffer_idx, require_i64=False)
    view_handle = _semantic.builder.tlx_local_view(
        [local_allocated_buffers.handle, buffer_idx]
    )

    original_shape = local_allocated_buffers.shape
    if local_allocated_buffers.type.num == 0:
        if len(original_shape) == 1:
            new_shape = [1]
        else:
            new_shape = original_shape[1:]
    else:
        new_shape = original_shape

    return tlx.buffered_tensor(
        view_handle,
        local_allocated_buffers.dtype,
        new_shape,
        0,
        local_allocated_buffers.type.storage,
        local_allocated_buffers.type.layout,
    )


@tl.builtin
def local_load(
    src: tlx.buffered_tensor,
    _semantic=None,
) -> tl.tensor:
    """
    Load from SMEM buffer into a register tensor.
    """
    block_type = tl.block_type(src.type.element_ty, src.type.shape)
    output = _semantic.builder.tlx_local_load([src.handle])
    return tl.tensor(output, block_type)


@tl.builtin
def local_store(
    dst: tlx.buffered_tensor,
    src: tl.tensor,
    _semantic=None,
) -> tl.tensor:
    """
    Store a register tensor into an SMEM buffer.
    """
    _semantic.builder.tlx_local_store([dst.handle, src.handle])
    return tl.tensor(src.handle, tl.void)
