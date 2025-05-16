import triton.language.core as tl

from . import types as tlx


@tl.builtin
def alloc_barriers(
    num_barriers: tl.constexpr,
    arrive_count: tl.constexpr,
    _builder=None,
) -> tlx.mbarriers:
    """
    Allocates buffer in shared memory and initialize mbarriers with arrive_counts.
    
    Input: 
    - `num_barriers`: The number of barriers to allocate.
    - `arrive_counts`: The number of threads that need to arrive at the barrier before it can be released.
    """
    return tlx.mbarriers(_builder.create_alloc_barriers(tl._constexpr_to_value(num_barriers), tl._constexpr_to_value(arrive_count)), )
