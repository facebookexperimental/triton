import triton.language.core as tl

from . import types as tlx
from .utility import is_hip


@tl.builtin
def alloc_barriers(
        num_barriers: tl.constexpr,
        arrive_count: tl.constexpr = tl.constexpr(1),
        _semantic=None,
) -> tlx.mbarrier:
    """
    Allocates buffer in shared memory and initialize mbarriers with arrive_counts.

    Input:
    - `num_barriers`: The number of barriers to allocate.
    - `arrive_counts`: The number of threads that need to arrive at the barrier before it can be released.
    """
         
    layout = tlx.swizzled_shared_layout_encoding.make_default(rank=1)
    layout_handle = _semantic.builder.make_swizzled_shared_encoding_attr(
        layout.vectorSize,
        layout.perPhase,
        layout.maxPhase,
        layout.order,
        layout.numCTAsPerCGA,
        layout.numCTASplit,
        layout.numCTAOrder,
    )
    return tlx.mbarrier(_semantic.builder.create_alloc_barriers(num_barriers.value, arrive_count.value, layout_handle),
                        num_barriers, layout, _semantic)


@tl.builtin
def barrier_expect_bytes(
    bar: tlx.mbarrier,
    size: tl.constexpr,
    pred: tl.tensor = None,
    _semantic=None,
) -> None:
    """
    Signal a barrier of an expected number of bytes to be copied
    """

    # TODO. add validator logics
    if pred is None:
        pred_handle = _semantic.builder.get_int1(True)
    else:
        pred_handle = pred.handle
    _semantic.builder.create_barrier_expect(bar.handle, size.value, pred_handle)


@tl.builtin
def barrier_wait(
    bar: tlx.buffered_tensor,
    phase,
    pred: tl.tensor = None,
    _semantic=None,
) -> None:
    """
    Wait until the mbarrier phase completes
    """

    # TODO. add validator logics
    if pred is None:
        pred_handle = _semantic.builder.get_int1(True)
    else:
        pred_handle = pred.handle
    if isinstance(phase, tl.tensor):
        _semantic.builder.create_barrier_wait(bar.handle, phase.handle, pred_handle)
    elif isinstance(phase, tl.constexpr):
        _semantic.builder.create_barrier_wait(bar.handle,
                                              _semantic._convert_elem_to_ir_value(phase.value, require_i64=False),
                                              pred_handle)
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

    assert arrive_count.value == 1 or not is_hip(), "AMD backend currently only supports arrive_count == 1"

    # TODO. add validator logics
    _semantic.builder.create_barrier_arrive(bar.handle, arrive_count.value)


@tl.builtin
def named_barrier_wait(
    bar: int,
    arrive_count: int,
    _semantic=None,
) -> None:
    """
    Wait until `arrive_count` threads have reached the specified named mbarrier phase.

    Arguments:
        bar (tl.constexpr): Identifier for the named barrier (e.g. from a buffer view).
        count (tl.constexpr): Number of threads arriving at the barrier.
    """

    bar_handle = _semantic._convert_elem_to_ir_value(bar, require_i64=False)
    arrive_count_handle = _semantic._convert_elem_to_ir_value(arrive_count, require_i64=False)
    _semantic.builder.create_named_barrier_wait(bar_handle, arrive_count_handle)


@tl.builtin
def named_barrier_arrive(
    bar: tl.constexpr,
    arrive_count: tl.constexpr,
    _semantic=None,
) -> None:
    """
    Signal arrival at a named mbarrier with the given thread count.

    Arguments:
        bar (tl.constexpr): Identifier for the named barrier (e.g. from a buffer view).
        count (tl.constexpr): Number of threads arriving at the barrier.
    """
    bar_handle = _semantic._convert_elem_to_ir_value(bar, require_i64=False)
    arrive_count_handle = _semantic._convert_elem_to_ir_value(arrive_count, require_i64=False)
    _semantic.builder.create_named_barrier_arrive(bar_handle, arrive_count_handle)
