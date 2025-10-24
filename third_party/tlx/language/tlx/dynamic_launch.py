import triton.language.core as tl

from . import types as tlx
from .barrier import alloc_barriers, barrier_expect_bytes, barrier_wait

# Blackwell-only


@tl.builtin
def _alloc_clc_responses(
    num_responses: int,
    _semantic=None,
) -> tlx.clc_response:
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
    return tlx.clc_response(
        _semantic.builder.create_alloc_clc_responses(num_responses, layout_handle),
        num_responses,
        layout,
        _semantic,
    )


@tl.builtin
def _clc_issue(
    clc_response_addr: tlx.clc_response,
    barrier: tlx.mbarrier,
    _semantic=None,
):
    # Issue async `clusterlaunchcontrol.try_cancel` request for
    # CTA ID of available cluster
    assert isinstance(clc_response_addr, tlx.clc_response)

    return _semantic.builder.clc_issue(clc_response_addr.handle, barrier.handle)


@tl.builtin
def _clc_query(
    clc_response_addr: tlx.clc_response,
    _semantic=None,
):
    # Extract CTA ID from CLC response
    assert isinstance(clc_response_addr, tlx.clc_response)

    x = _semantic.builder.clc_query(clc_response_addr.handle, )
    return _semantic.tensor(x, tl.int32)


@tl.builtin
def clc_create_scheduler(num_stages: tl.constexpr, _semantic=None) -> tlx.CLCPipeliner:
    return tlx.CLCPipeliner(
        clc_mbars=alloc_barriers(num_barriers=num_stages, _semantic=_semantic),
        clc_responses=_alloc_clc_responses(num_responses=num_stages, _semantic=_semantic),
        # phase=_semantic._convert_elem_to_ir_value(0, require_i64=False),
        _semantic=_semantic,
    )


@tl.builtin
def clc_fetch_next_worker(scheduler: tlx.CLCPipeliner, phase, _semantic=None):
    # Issue async clc.try_cancel for the next available CTA
    barrier_expect_bytes(scheduler.clc_mbars[0], tl.constexpr(16), _semantic=_semantic)  # CLC response is 16-byte
    _clc_issue(scheduler.clc_responses[0], scheduler.clc_mbars[0], _semantic=_semantic)

    # Wait for clc.try_cancel finishes
    barrier_wait(scheduler.clc_mbars[0], phase, _semantic=_semantic)

    # Extract CTA ID from CLC response
    return _clc_query(scheduler.clc_responses[0], _semantic=_semantic)
