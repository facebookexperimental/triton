import triton.language.core as tl

from . import types as tlx


@tl.builtin
def alloc_barriers(
        num_barriers: tl.constexpr,
        arrive_count: tl.constexpr = tl.constexpr(1),
        _builder=None,
) -> tlx.mbarriers:
    """
    Allocates buffer in shared memory and initialize mbarriers with arrive_counts.

    Input:
    - `num_barriers`: The number of barriers to allocate.
    - `arrive_counts`: The number of threads that need to arrive at the barrier before it can be released.
    """
    return tlx.mbarriers(_builder.create_alloc_barriers(num_barriers.value, arrive_count.value), )


@tl.builtin
def barrier_expect_bytes(
    bar: tlx.mbarriers,
    size: tl.constexpr,
    _builder=None,
) -> None:
    """
    Signal a barrier of an expected number of bytes to be copied
    """

    # TODO. add validator logics
    _builder.create_barrier_expect(bar.handle, size.value)


@tl.builtin
def barrier_wait(
    bar: tlx.buffered_tensor,
    phase,
    _builder=None,
) -> None:
    """
    Wait until the mbarrier phase completes
    """

    # TODO. add validator logics

    # TODO. Need to improve phase typing so that both of following usages would work:
    # Case 1.
    #   tlx.barrier_wait(b, 0)
    # Case 2.
    #   p=0
    #   tlx.barrier_wait(b, p)
    #  Now we just ensure case 2 works to support use case in WS-GEMM
    assert type(
        phase
    ) == tl.tensor, "Users are suggested passing by `phase=<variable>` instead of `phase=<const>` (such as phase=0)"
    _builder.create_barrier_wait(bar.handle, phase.handle)


@tl.builtin
def barrier_arrive(
        bar: tlx.buffered_tensor,
        arrive_count: tl.constexpr = tl.constexpr(1),
        _builder=None,
) -> None:
    """
    Perform the arrive operation on an mbarrier
    """

    # TODO. add validator logics
    _builder.create_barrier_arrive(bar.handle, arrive_count.value)
