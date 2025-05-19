import triton.language.core as tl

from . import types as tlx


@tl.builtin
def alloc_barriers(
    num_barriers: tl.constexpr,
    arrive_count: tl.constexpr,
    _builder=None,
) -> tlx.mbarrier:
    """
    Allocates buffer in shared memory and initialize mbarriers with arrive_counts.

    Input:
    - `num_barriers`: The number of barriers to allocate.
    - `arrive_counts`: The number of threads that need to arrive at the barrier before it can be released.
    """
    return tlx.mbarrier(_builder.create_alloc_barriers(num_barriers.value, arrive_count.value), )

@tl.builtin
def barrier_expect_bytes(
    bar: tlx.buffered_tensor,
    size: tl.constexpr,
    _builder=None,
) -> None:
    """
    Signal a barrier of an expected number of bytes to be copied 
    """

    # TODO. add validator logics
    _builder.create_barrier_expect(bar.handle, size.value)
