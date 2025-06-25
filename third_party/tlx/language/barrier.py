import triton.language.core as tl

from triton.language.semantic import _convert_elem_to_ir_value

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
    base_barrier = tlx.mbarrier(_builder.create_alloc_barriers(num_barriers.value, arrive_count.value))
    return tlx.mbarriers(base_barrier, num_barriers)


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

    if isinstance(phase, tl.tensor):
        _builder.create_barrier_wait(bar.handle, phase.handle)
    elif isinstance(phase, tl.constexpr):
        _builder.create_barrier_wait(bar.handle, _convert_elem_to_ir_value(_builder, phase.value, require_i64=False))
    else:
        raise RuntimeError(f"`phase` is in type {type(phase)} (must be either `tl.tensor` or `tl.constexpr`)")


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
