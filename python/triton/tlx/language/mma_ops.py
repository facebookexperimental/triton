from typing import Union

import triton.language.core as tl

from . import types as tlx


def _str_to_dot_input_precision(input_precision, builder):
    assert (
        input_precision.lower() in builder.options.allowed_dot_input_precisions
    ), f"input_precision must be one of {builder.options.allowed_dot_input_precisions}. Got {input_precision}"
    input_precision = input_precision.upper()
    if input_precision == "TF32X3":
        input_precision = "TF32x3"
    return getattr(ir.INPUT_PRECISION, input_precision)


@tl.builtin
def async_dot(
    A: Union[tl.tensor, tlx.buffered_tensor],
    B: tlx.buffered_tensor,
    C: tl.tensor,
    input_precision: str = "tf32",
    out_dtype=tl.float32,
    _builder=None,
):
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
    out_dtype = tl._constexpr_to_value(out_dtype)
    return tl.tensor(
        _builder.create_warp_group_dot(A.handle, B.handle, C.handle, input_precision),
        out_dtype,
    )
