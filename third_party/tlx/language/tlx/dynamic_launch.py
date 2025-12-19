import triton.language.core as tl

from . import types as tlx
from .barrier import alloc_barriers, barrier_arrive, barrier_expect_bytes, barrier_wait
from .mem_ops import local_view

# Blackwell-only


@tl.builtin
def _alloc_clc_responses(
    num_responses: tl.constexpr,
    _semantic=None,
) -> tlx.clc_response:
    # Use placeholder layout to be resolved after inlining
    layout = tlx.DummySMemLayoutEncoding([1], tl.int64)
    layout_handle = layout.to_ir(_semantic.builder)
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
def clc_create_context(num_stages: tl.tensor, num_consumers, _semantic=None) -> tlx.CLCPipelineContext:
    return tlx.CLCPipelineContext(
        clc_mbars_empty=alloc_barriers(num_barriers=num_stages, arrive_count=num_consumers, _semantic=_semantic),
        clc_mbars_full=alloc_barriers(num_barriers=num_stages, _semantic=_semantic),
        clc_responses=_alloc_clc_responses(num_responses=num_stages, _semantic=_semantic),
    )


@tl.builtin
def clc_producer(context, k, p_producer, _semantic=None):
    # acquire
    barrier_wait(
        local_view(context._clc_mbars_empty, k, _semantic=_semantic),
        p_producer,
        _semantic=_semantic,
    )

    # commit
    barrier_expect_bytes(
        local_view(context._clc_mbars_full, k, _semantic=_semantic),
        tl.constexpr(16),
        _semantic=_semantic,
    )
    _clc_issue(
        local_view(context._clc_responses, k, _semantic=_semantic),
        local_view(context._clc_mbars_full, k, _semantic=_semantic),
        _semantic=_semantic,
    )


@tl.builtin
def clc_consumer(context, k, p_consumer, _semantic=None):
    # wait
    barrier_wait(
        local_view(context._clc_mbars_full, k, _semantic=_semantic),
        p_consumer,
        _semantic=_semantic,
    )

    # extract
    stolen_tile_id = _clc_query(local_view(context._clc_responses, k, _semantic=_semantic), _semantic=_semantic)

    # release
    barrier_arrive(
        local_view(context._clc_mbars_empty, k, _semantic=_semantic),
        _semantic=_semantic,
    )

    # return
    return stolen_tile_id
