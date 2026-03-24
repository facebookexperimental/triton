"""TLX Plugin barrier ops: alloc_barriers.

Ported from triton-tlx/third_party/tlx/language/tlx/barrier.py.
Only includes alloc_barriers (and alloc_warp_barrier convenience wrapper).

These ops call plugin custom ops registered by TLXLocalAllocPlugin.cpp:
  - tlx_alloc_barriers(numBarriers, arriveCount) -> MemDesc Value
"""

import triton.language.core as tl

from . import types as tlx


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
    - `arrive_count`: The number of threads that need to arrive at the barrier
                      before it can be released.
    """

    num_barriers_val = tl._unwrap_if_constexpr(num_barriers)
    arrive_count_val = tl._unwrap_if_constexpr(arrive_count)

    # Pass numBarriers and arriveCount as i32 constants to the C++ plugin
    num_barriers_ir = _semantic.builder.get_int32(int(num_barriers_val))
    arrive_count_ir = _semantic.builder.get_int32(int(arrive_count_val))

    args = [num_barriers_ir, arrive_count_ir]
    handle = _semantic.builder.tlx_alloc_barriers(args)

    layout = tlx.swizzled_shared_layout_encoding.make_default(rank=1)
    return tlx.mbarrier(handle, num_barriers_val, layout)


@tl.builtin
def alloc_warp_barrier(
    num_barriers: tl.constexpr,
    num_warps: tl.constexpr = tl.constexpr(1),
    num_arrivals: tl.constexpr = tl.constexpr(1),
    _semantic=None,
) -> tlx.mbarrier:
    """
    Allocates warp barriers where all threads arrive independently.

    Unlike alloc_barriers (where a single leader thread signals the arrive after
    a warp sync), warp barriers expect every thread to arrive individually. This
    removes the need for thread synchronization before the arrive, reducing
    unnecessary syncs and improving performance when there is warp divergence.

    Input:
    - `num_barriers`: The number of barriers to allocate.
    - `num_warps`: The number of warps whose threads will arrive at the barrier.
    - `num_arrivals`: The number of times barrier_arrive is called per phase.
                      The total arrive count is num_warps * 32 * num_arrivals.
    """

    num_barriers_val = tl._unwrap_if_constexpr(num_barriers)
    num_warps_val = tl._unwrap_if_constexpr(num_warps)
    num_arrivals_val = tl._unwrap_if_constexpr(num_arrivals)
    arrive_count = num_warps_val * 32 * num_arrivals_val

    num_barriers_ir = _semantic.builder.get_int32(int(num_barriers_val))
    arrive_count_ir = _semantic.builder.get_int32(int(arrive_count))

    args = [num_barriers_ir, arrive_count_ir]
    handle = _semantic.builder.tlx_alloc_barriers(args)

    layout = tlx.swizzled_shared_layout_encoding.make_default(rank=1)
    return tlx.mbarrier(handle, num_barriers_val, layout, is_warp_barrier=True)
