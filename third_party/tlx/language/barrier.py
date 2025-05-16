import triton.language.core as tl

from . import types as tlx


@tl.builtin
def alloc_barriers(
    # arrive_counts: tuple,
    num_barriers: tl.constexpr,
    _builder=None,
) -> tlx.mbarriers:
    """
    Allocates buffer in shared memory and initialize mbarriers with arrive_counts.
    
    Input: arrive_counts: tuple of size num_barriers. Each element is the number of threads that need to arrive at the barrier before it can be released.
    """
    # arrive_counts_list = [tl._constexpr_to_value(x) for x in arrive_counts]
    return tlx.mbarriers(_builder.create_alloc_barriers(num_barriers), )

# @tl.builtin
# def inval_barriers(
#     barriers: tlx.mbarriers,
#     idx: tl.constexpr,
#     # dtype: tl.dtype = tl.uint64,
#     _builder=None,
# ):
#     """
#     Allocates buffer in shared memory and return a view of the buffer.
#     """
#     # elem_type = dtype.to_ir(_builder)
#     return tlx.mbarriers(_builder.create_alloc_barriers(num_barriers), )
