import triton.language.core as tl
from triton import knobs
import triton.language.semantic as semantic

from . import types as tlx
from .utility import cuda_parse_arch


def require_nv_mma_shared_layout(x: tlx.buffered_tensor, order, _builder=None):
    if not isinstance(x.layout, tlx.nv_mma_shared_layout_encoding):
        # TODO. why do we need this class object?
        layout = tlx.nv_mma_shared_layout_encoding(shape=x.shape, order=order, elemType=x.dtype, numCTAsPerCGA=[1, 1],
                                                   numCTASplit=[1, 1], numCTAOrder=[1, 1], fp4Padded=False)

    layout_handle = _builder.make_nv_mma_shared_encoding_attr(
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
    input: tlx.buffered_tensor,
    other: tlx.buffered_tensor,
    acc=None,  # tl.tensor,
    mmav5: bool = False,
    mBarrier=None,
    input_precision=None,
    allow_tf32=None,
    max_num_imprecise_acc=None,
    col_input=0,
    col_other=0,
    out_dtype=tl.float32,
    _builder=None,
) -> tl.tensor:
    """
    Performs a warp-group matrix multiply-accumulate operation of two blocks and return the matrix product.

    This maps directly to NVIDIA Hopper’s wgmma.mma_async instructions, enabling high-throughput matrix multiplication
    across multiple warps within a warpgroup, or Blackwell's tcgen05.mma instruction.

    The operation computes:
        D = A @ B + C

    Where:

        A: A matrix tile held in registers or shared memory

        B: A matrix tile loaded from shared memory

        C is an accumulator tile in registers

        D is the output tile in registers

    input_precision can be one of: tf32, tf32x3, ieee.
    """

    # Perform dot_precheck shared by tl.dot
    (input, other, acc_handle, input_precision, max_num_imprecise_acc,
     ret_ty) = semantic.dot_precheck(input, other, acc, input_precision, allow_tf32, max_num_imprecise_acc, out_dtype,
                                     _builder)

    # TODO. batched dot is not supported yet
    input = require_nv_mma_shared_layout(input, [0, 1] if col_input else [1, 0], _builder)
    other = require_nv_mma_shared_layout(other, [0, 1] if col_other else [1, 0], _builder)

    if mmav5:
        assert int(cuda_parse_arch(_builder.options.arch)) >= 100, "mmav5 is only supported on Blackwell and above"
        output = _builder.create_tcgen5_dot(input, other, acc.handle, mBarrier.handle if mBarrier else None)
        return tl.async_token(output)

    acc = _builder.create_require_layout(acc_handle, _builder.make_nv_mma_encoding_attr())

    _builder.create_fence_async_shared()

    output = _builder.create_warp_group_dot(input, other, acc, input_precision, max_num_imprecise_acc, True)

    # Release the mma layout for the output to conform to what the user expects
    output = _builder.create_release_layout(output)
    return tl.tensor(output, ret_ty)


@tl.builtin
def async_dot_wait(
    pendings: tl.constexpr,
    inp: tl.tensor,
    _builder=None,
) -> tl.tensor:
    """
    Wait for completion of prior asynchronous dot operations.
    Each input must be the tensors corresponding to the async dot ops that we're
    waiting on.
    """
    pendings = tl._unwrap_if_constexpr(pendings)
    return tl.tensor(_builder.create_warp_group_dot_wait([inp.handle], pendings)[0], inp.type)
