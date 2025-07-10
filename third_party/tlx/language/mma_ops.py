import triton.language.core as tl
import triton.language.semantic as semantic

from . import types as tlx
from .utility import cuda_parse_arch


def require_nv_mma_shared_layout(x: tlx.buffered_tensor, _builder=None):
    assert isinstance(x.layout, tlx.shared_layout_encoding), "input must be a shared tensor"
    if isinstance(x.layout, tlx.swizzled_shared_layout_encoding):
        layout = tlx.nv_mma_shared_layout_encoding(shape=x.shape, order=x.layout.order, elemType=x.dtype,
                                                   numCTAsPerCGA=[1, 1], numCTASplit=[1, 1], numCTAOrder=[1, 1],
                                                   fp4Padded=False)
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
    else:
        assert isinstance(x.layout, tlx.nv_mma_shared_layout_encoding), "input must be a shared mma tensor"
        return x.handle


def require_dot_operand_layout(opnd: tl.tensor, opIdx, parent_layout, _builder=None):
    layout_handle = _builder.make_dot_operand_encoding_attr(opnd.handle, opIdx, parent_layout)
    return _builder.create_require_layout(opnd.handle, layout_handle)


# async dot signature needs to be close to tl.dot as much as possible
@tl.builtin
def async_dot(
    A: tlx.buffered_tensor | tl.tensor,
    B: tlx.buffered_tensor,
    acc: tlx.buffered_tensor | tl.tensor | None = None,
    mBarrier=None,
    input_precision=None,
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
    (A, B, acc_handle, input_precision, max_num_imprecise_acc,
     ret_ty) = semantic.dot_precheck(A, B, acc, input_precision, None, None, out_dtype, _builder)

    assert A.shape[0] >= 64, "M must be at least 64"
    assert A.shape[1] >= 16, "K must be at least 16"
    assert B.shape[1] >= 32, "N must be at least 32"

    cuda_compute_capability = int(cuda_parse_arch(_builder.options.arch))
    version = 5 if cuda_compute_capability >= 100 else 3

    # TODO. batched dot is not supported yet
    if isinstance(A, tlx.buffered_tensor) and A.storage == tlx.storage_kind.smem:
        A_handle = require_nv_mma_shared_layout(A, _builder)
    if isinstance(A, tlx.buffered_tensor) and A.storage == tlx.storage_kind.tmem:
        # require A's TMEM layout to have `unpacked` set to False
        assert isinstance(A.layout, tlx.tensor_memory_layout_encoding), "input must be a TMEM tensor"
        if A.layout.unpacked:
            layout_handle = _builder.make_tensor_memory_encoding_attr(
                A.layout.blockM,
                A.layout.blockN,
                False,  # unpacked == False
                A.layout.CTASplitM,
                A.layout.CTASplitN,
            )
            A_handle = _builder.create_require_layout(A.handle, layout_handle)
    elif isinstance(A, tl.tensor):
        assert cuda_compute_capability < 100, "register operand is not supported on Blackwell"
        A_handle = A.handle
    else:
        # Registers do not need mma shared layout
        A_handle = A.handle

    B_handle = require_nv_mma_shared_layout(B, _builder)

    if version == 5:
        assert isinstance(A, tlx.buffered_tensor), "input must be a buffered tensor"
        output = _builder.create_tcgen5_dot(A_handle, B_handle, acc.handle, mBarrier.handle if mBarrier else None)
        return tlx.async_token(output)
    else:
        mma_layout = _builder.make_nv_mma_encoding_attr(A_handle, acc_handle, version, 0, _builder.options.num_warps)
        acc = _builder.create_require_layout(acc_handle, mma_layout)
        if isinstance(A, tl.tensor):
            A_handle = require_dot_operand_layout(A, 0, mma_layout, _builder)
        output = _builder.create_warp_group_dot(A_handle, B_handle, acc, input_precision, max_num_imprecise_acc, True)
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
