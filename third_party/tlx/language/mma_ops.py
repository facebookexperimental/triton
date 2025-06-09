import triton.language.core as tl
from triton import knobs
import triton.language.semantic as semantic

from . import types as tlx


def require_mma_layout(x: tlx.buffered_tensor, order, _builder=None):
    layout_A = x.layout
    if layout_A is not tlx.nv_mma_shared_layout_encoding:
        # create datastruct to wrap layout encoding attributes
        # TODO. why do we need this class object?
        layout = tlx.nv_mma_shared_layout_encoding(shape=x.shape, order=order, elemType=x.dtype, numCTAsPerCGA=[1, 1], numCTASplit=[1,1], numCTAOrder=[1,1], fp4Padded=False)

    layout_handle = _builder.make_nv_mma_shared_shared_encoding_attr(
        [int(x) for x in layout.shape],
        layout.order,
        layout.elemType.to_ir(_builder),
        layout.numCTAsPerCGA,
        layout.numCTASplit,
        layout.numCTAOrder,
        layout.fp4Padded,
    )
    return _builder.create_require_layout(x.handle, layout_handle)


# async dot signature needs to be close to tl.dot as much as possible
@tl.builtin
def async_dot(
    # A: Union[tl.tensor, tlx.buffered_tensor],
    input: tlx.buffered_tensor,
    other: tlx.buffered_tensor,
    dummy_output: tl.tensor,
    acc=None, # tl.tensor,
    input_precision=None,
    allow_tf32=None,
    max_num_imprecise_acc=None,
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

    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    if input_precision is None:
        supports_tf32 = _builder and "tf32" in _builder.options.allowed_dot_input_precisions
        input_precision = knobs.language.fp32_default or (
            "tf32" if (
                supports_tf32 and (allow_tf32 or allow_tf32 is None)) else "ieee"
            )

    input_precision = tl._unwrap_if_constexpr(input_precision)
    out_dtype = tl._unwrap_if_constexpr(out_dtype)
    max_num_imprecise_acc = tl._unwrap_if_constexpr(max_num_imprecise_acc)

    # Perform dot_precheck shared by tl.dot
    (input, other, acc_handle, input_precision, max_num_imprecise_acc, ret_ty) = semantic.dot_precheck(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype, _builder)

    # return tl.tensor(input, ret_ty)
    # out_dtype = tl._unwrap_if_constexpr(out_dtype)
    # M = C.type.shape[-2]
    # N = C.type.shape[-1]
    #
    input = require_mma_layout(input, [1,0], _builder)
    other = require_mma_layout(other, [0,1], _builder)
    # return dummy_output

    # if acc is None:
    #     acc_handle = _builder.create_splat(ret_ty.to_ir(_builder), _0)
    # else:
    #     assert False # TODO. fix this later
        # acc_handle = acc.handle
        # import pdb; pdb.set_trace()
        # assert acc.type == ret_ty, f"acc type {acc.type} does not match ret_ty {ret_ty}"

    # TODO. placeholder to test prior logics
    # return acc_handle

    import pdb; pdb.set_trace()
    _builder.create_fence_async_shared()
    return dummy_output
    return tl.tensor(
        _builder.create_warp_group_dot(
            input, other, acc_handle, input_precision, max_num_imprecise_acc, True
        ),
        ret_ty
    )
    # ret_ty = tl.block_type(out_dtype, [M, N])

    # input_precision = input_precision.upper()

    # Emit layout conversion for mismatched layouts
    # if isinstance(A, tlx.buffered_tensor):

    # layout_A = input.layout
    # if layout_A is not tlx.nv_mma_shared_layout_encoding:
    #     # create datastruct to wrap layout encoding attributes
    #     layout = tlx.nv_mma_shared_layout_encoding(shape=A.shape, order=[1,0], elemType=A.dtype, numCTAsPerCGA=[1, 1], numCTASplit=[1,1], numCTAOrder=[1,1], fp4Padded=False)
    # layout_handle = _builder.make_nv_mma_shared_shared_encoding_attr(
    #     [int(x) for x in layout.shape],
    #     layout.order,
    #     layout.elemType.to_ir(_builder),
    #     layout.numCTAsPerCGA,
    #     layout.numCTASplit,
    #     layout.numCTAOrder,
    #     layout.fp4Padded,
    # )
    # input = _builder.create_require_layout(input.handle, layout_handle)
    #
    # else:
    #     # promote reigster tensor to mma layout
    #     raise NotImplementedError("Assign register mma layout not yet implemented.")

    # layout_B = B.layout
    # if layout_B is not tlx.nv_mma_shared_layout_encoding:
        # mma_layout = _builder.make_nv_mma_shared_shared_encoding_attr()
        # B = _builder.create_convert_layout(B, mma_layout)

    # layout = tlx.nv_mma_shared_layout_encoding(shape=B.shape, order=[1,0], elemType=B.dtype, numCTAsPerCGA=[1, 1], numCTASplit=[1,1], numCTAOrder=[1,1], fp4Padded=False)
    # shape = [int(x) for x in layout.shape]
    # elem_type = layout.elemType.to_ir(_builder)
    # layout_handle = _builder.make_nv_mma_shared_shared_encoding_attr(
    #     shape,
    #     layout.order,
    #     elem_type,
    #     layout.numCTAsPerCGA,
    #     layout.numCTASplit,
    #     layout.numCTAOrder,
    #     layout.fp4Padded,
    # )
    # B = _builder.create_require_layout(B.handle, layout_handle)
    #
    # import pdb; pdb.set_trace()
    # tl.foo()

    # if input_precision is None:
    #     input_precision = _builder.options.default_dot_input_precision

    # input_precision = _str_to_dot_input_precision(input_precision, builder)

    # return C
    # return tl.tensor(
        # _builder.create_warp_group_dot(
        # _builder.create_dot(
            # A, B, C.handle, input_precision, True
            # A, B, C.handle, input_precision, 0
        # ),
    #     ret_ty,
    # )
