import triton.language.core as tl

from . import types as tlx
from .mem_ops import local_view, remote_view
from .barrier import alloc_barriers, barrier_expect_bytes, barrier_wait, barrier_arrive

# Blackwell-only


@tl.builtin
def _alloc_clc_responses(
    num_responses: tl.constexpr,
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
    )


@tl.builtin
def _clc_issue(
    clc_response_addr: tlx.clc_response,
    barrier: tlx.mbarrier,
    _semantic=None,
):
    # Issue async `clusterlaunchcontrol.try_cancel` request for
    # CTA ID of available cluster
    return _semantic.builder.clc_issue(clc_response_addr.handle, barrier.handle)


@tl.builtin
def _clc_query(
    clc_response_addr: tlx.clc_response,
    _semantic=None,
):
    # Extract CTA ID from CLC response
    x = _semantic.builder.clc_query(clc_response_addr.handle, )
    return _semantic.tensor(x, tl.int32)


@tl.builtin
def clc_create_context(num_stages: tl.tensor, num_consumers, _semantic=None) -> tlx.CLCPipelineContext:
    return tlx.CLCPipelineContext(
        clc_mbars_empty=alloc_barriers(num_barriers=num_stages, arrive_count=num_consumers, _semantic=_semantic),
        clc_mbars_full=alloc_barriers(num_barriers=num_stages, _semantic=_semantic),
        clc_responses=_alloc_clc_responses(num_responses=num_stages, _semantic=_semantic),
    )


@tl.builtin
def clc_producer(context, k, p_producer, pred_cta0: tl.tensor = None, _semantic=None):
    bar_empty = local_view(context._clc_mbars_empty, k, _semantic=_semantic)
    bar_full = local_view(context._clc_mbars_full, k, _semantic=_semantic)
    response = local_view(context._clc_responses, k, _semantic=_semantic)

    if pred_cta0 is not None:
        bar_empty = remote_view(bar_empty, 0, _semantic=_semantic)
        response = remote_view(response, 0, _semantic=_semantic)

    # acquire
    barrier_wait(bar_empty, p_producer, pred_cta0, _semantic=_semantic)

    # commit
    barrier_expect_bytes(bar_full, tl.constexpr(16), pred_cta0, _semantic=_semantic)

    _clc_issue(
        response,
        bar_full,
        _semantic=_semantic,
    )


@tl.builtin
def clc_consumer(context, k, p_consumer, pred_cta0: tl.tensor = None, _semantic=None):
    bar_empty = local_view(context._clc_mbars_empty, k, _semantic=_semantic)
    bar_full = local_view(context._clc_mbars_full, k, _semantic=_semantic)
    response = local_view(context._clc_responses, k, _semantic=_semantic)

    if pred_cta0 is not None:
        bar_empty = remote_view(bar_empty, 0, _semantic=_semantic)
        bar_full = remote_view(bar_full, 0, _semantic=_semantic)
        response = remote_view(response, 0, _semantic=_semantic)

    # wait
    barrier_wait(bar_full, p_consumer, pred_cta0, _semantic=_semantic)

    # extract
    stolen_tile_id = _clc_query(response, _semantic=_semantic)

    # release
    barrier_arrive(bar_empty, _semantic=_semantic)

    # return
    return stolen_tile_id
