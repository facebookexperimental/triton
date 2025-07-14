import triton.language.core as tl

from . import types as tlx


@tl.builtin
def alloc_barriers(
        num_barriers: tl.constexpr,
        arrive_count: tl.constexpr = tl.constexpr(1),
        _semantic=None,
) -> tlx.mbarriers:
    """
    Allocates buffer in shared memory and initialize mbarriers with arrive_counts.

    Input:
    - `num_barriers`: The number of barriers to allocate.
    - `arrive_counts`: The number of threads that need to arrive at the barrier before it can be released.
    """
    base_barrier = tlx.mbarrier(_semantic.builder.create_alloc_barriers(num_barriers.value, arrive_count.value))
    return tlx.mbarriers(base_barrier, num_barriers)


@tl.builtin
def barrier_expect_bytes(
    bar: tlx.mbarrier,
    size: tl.constexpr,
    _semantic=None,
) -> None:
    """
    Signal a barrier of an expected number of bytes to be copied
    """

    # TODO. add validator logics
    _semantic.builder.create_barrier_expect(bar.handle, size.value)


@tl.builtin
def barrier_wait(
    bar: tlx.buffered_tensor,
    phase,
    _semantic=None,
) -> None:
    """
    Wait until the mbarrier phase completes
    """

    # TODO. add validator logics

    if isinstance(phase, tl.tensor):
        _semantic.builder.create_barrier_wait(bar.handle, phase.handle)
    elif isinstance(phase, tl.constexpr):
        _semantic.builder.create_barrier_wait(bar.handle, _semantic._convert_elem_to_ir_value(phase.value, require_i64=False))
    else:
        raise RuntimeError(f"`phase` is in type {type(phase)} (must be either `tl.tensor` or `tl.constexpr`)")


@tl.builtin
def barrier_arrive(
        bar: tlx.buffered_tensor,
        arrive_count: tl.constexpr = tl.constexpr(1),
        _semantic=None,
) -> None:
    """
    Perform the arrive operation on an mbarrier
    """

    # TODO. add validator logics
    _semantic.builder.create_barrier_arrive(bar.handle, arrive_count.value)


@tl.builtin
def named_barrier_wait(
    bar: int,
    arrive_count: int,
    _builder=None,
) -> None:
    """
    Wait until `arrive_count` threads have reached the specified named mbarrier phase.

    Arguments:
        bar (tl.constexpr): Identifier for the named barrier (e.g. from a buffer view).
        count (tl.constexpr): Number of threads arriving at the barrier.
    """

    bar_handle = _convert_elem_to_ir_value(_builder, bar, require_i64=False)
    arrive_count_handle = _convert_elem_to_ir_value(_builder, arrive_count, require_i64=False)
    _builder.create_named_barrier_wait(bar_handle, arrive_count_handle)


@tl.builtin
def named_barrier_arrive(
    bar: tl.constexpr,
    arrive_count: tl.constexpr,
    _builder=None,
) -> None:
    """
    Signal arrival at a named mbarrier with the given thread count.

    Arguments:
        bar (tl.constexpr): Identifier for the named barrier (e.g. from a buffer view).
        count (tl.constexpr): Number of threads arriving at the barrier.
    """
    bar_handle = _convert_elem_to_ir_value(_builder, bar, require_i64=False)
    arrive_count_handle = _convert_elem_to_ir_value(_builder, arrive_count, require_i64=False)
    _builder.create_named_barrier_arrive(bar_handle, arrive_count_handle)
