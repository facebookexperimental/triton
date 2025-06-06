import triton.language.core as tl

from . import types as tlx


# async dot signature needs to be close to tl.dot as much as possible
@tl.builtin
def async_dot(
    # A: Union[tl.tensor, tlx.buffered_tensor],
    A: tlx.buffered_tensor,
    B: tlx.buffered_tensor,
    C: tl.tensor,
    input_precision: str = "tf32",
    out_dtype=tl.float32,
    _builder=None,
) -> tl.tensor:
    """
    Performs a warp-group matrix multiply-accumulate operation of two blocks and return the matrix product.

    This maps directly to NVIDIA Hopper’s wgmma.mma_async instructions, enabling high-throughput matrix multiplication
    across multiple warps within a warpgroup.

    The operation computes:
        D = A @ B + C

    Where:

        A: A matrix tile held in registers or shared memory

        B: A matrix tile loaded from shared memory

        C is an accumulator tile in registers

        D is the output tile in registers

    input_precision can be one of: tf32, tf32x3, ieee.
    """
    import pdb; pdb.set_trace()
    out_dtype = tl._unwrap_if_constexpr(out_dtype)
    M = C.type.shape[-2]
    N = C.type.shape[-1]
    ret_ty = tl.block_type(out_dtype, [M, N])

    input_precision = input_precision.upper()

    # Emit layout conversion for mismatched layouts
    # if isinstance(A, tlx.buffered_tensor):

    layout_A = A.layout
    # if layout_A is not tlx.nv_mma_shared_layout_encoding:
    layout = tlx.nv_mma_shared_layout_encoding(shape=A.shape, order=[1,0], elemType=A.dtype, numCTAsPerCGA=[1, 1], numCTASplit=[1,1], numCTAOrder=[1,1], fp4Padded=False) # more like a dataclass
    shape = [int(x) for x in layout.shape]
    elem_type = layout.elemType.to_ir(_builder)
    layout_handle = _builder.make_nv_mma_shared_shared_encoding_attr(
        shape,
        layout.order,
        elem_type,
        layout.numCTAsPerCGA,
        layout.numCTASplit,
        layout.numCTAOrder,
        layout.fp4Padded,
    )

    A = _builder.create_require_layout(A.handle, layout_handle)

    # else:
    #     # promote reigster tensor to mma layout
    #     raise NotImplementedError("Assign register mma layout not yet implemented.")

    layout_B = B.layout
    # if layout_B is not tlx.nv_mma_shared_layout_encoding:
        # mma_layout = _builder.make_nv_mma_shared_shared_encoding_attr()
        # B = _builder.create_convert_layout(B, mma_layout)

    layout = tlx.nv_mma_shared_layout_encoding(shape=B.shape, order=[1,0], elemType=B.dtype, numCTAsPerCGA=[1, 1], numCTASplit=[1,1], numCTAOrder=[1,1], fp4Padded=False)
    shape = [int(x) for x in layout.shape]
    elem_type = layout.elemType.to_ir(_builder)
    layout_handle = _builder.make_nv_mma_shared_shared_encoding_attr(
        shape,
        layout.order,
        elem_type,
        layout.numCTAsPerCGA,
        layout.numCTASplit,
        layout.numCTAOrder,
        layout.fp4Padded,
    )
    B = _builder.create_require_layout(B.handle, layout_handle)

    # import pdb; pdb.set_trace()
    # tl.foo()

    if input_precision is None:
        input_precision = _builder.options.default_dot_input_precision

    # input_precision = _str_to_dot_input_precision(input_precision, builder)

    # return C
    return tl.tensor(
        # _builder.create_warp_group_dot(
        _builder.create_dot(
            # A, B, C.handle, input_precision, True
            A, B, C.handle, input_precision, 0
        ),
        ret_ty,
    )
