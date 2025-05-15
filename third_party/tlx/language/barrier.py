import triton.language.core as tl

from . import types as tlx

@tl.builtin
def alloc_barriers(
    num_barriers: tl.constexpr,
    # dtype: tl.dtype = tl.uint64,
    _builder=None,
) -> tlx.mbarriers:
    """
    Allocates buffer in shared memory and return a view of the buffer.
    """
    # elem_type = dtype.to_ir(_builder)
    return tlx.mbarriers(_builder.create_alloc_barriers(num_barriers), )
