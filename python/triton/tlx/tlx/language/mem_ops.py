import triton.language.core as tl

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
        _builder.create_local_alloc(full_shape, elem_type),
    )
