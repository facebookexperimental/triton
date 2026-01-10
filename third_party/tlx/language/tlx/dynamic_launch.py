from typing import Optional
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
    # Issue an async `clusterlaunchcontrol.try_cancel` request to obtain
    # the CTA ID of an available cluster.
    assert isinstance(clc_response_addr, tlx.clc_response)
    return _semantic.builder.clc_issue(clc_response_addr.handle, barrier.handle)


@tl.builtin
def _clc_query(
    clc_response_addr: tlx.clc_response,
    _semantic=None,
):
    # Extract the CTA ID from the CLC response.
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
def clc_producer(context, k, p_producer, multi_ctas: bool = False, pred_cta0: Optional[bool] = None, _semantic=None):
    """
    Issue a CLC try_cancel request from the first CTA in the cluster.

    This function is lowered to the following PTX instruction:
        clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128

    The `.multicast::cluster::all` qualifier indicates that the response is
    asynchronously written via weak async-proxy writes to the corresponding
    local shared memory address of each CTA in the requesting cluster. Write
    completion for a particular CTA is signaled through a complete-tx operation
    on the mbarrier object in that CTA's shared memory.

    Consequently, each CTA maintains its own `bar_full` and `clc_response`.
    Although `try_cancel` is executed only on CTA-0, other CTAs in the same
    cluster can access the CLC response from their own shared memory once their
    respective `bar_full` barrier signals completion.
    """
    bar_empty = local_view(context._clc_mbars_empty, k, _semantic=_semantic)
    bar_full = local_view(context._clc_mbars_full, k, _semantic=_semantic)
    response = local_view(context._clc_responses, k, _semantic=_semantic)

    if multi_ctas:
        assert pred_cta0 is not None, "pred_cta0 must be provided when two_ctas is True"
        bar_empty = remote_view(bar_empty, 0, _semantic=_semantic)
    else:
        assert pred_cta0 is None, "pred_cta0 must be None when two_ctas is False"

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
def clc_consumer(context, k, p_consumer, multi_ctas: bool = False, pred_cta0: Optional[bool] = None, _semantic=None):
    """
    Decode the tile ID from a CLC response.

    Returns the tile ID if the response was written successfully, otherwise -1.

    This function is lowered to the following PTX instructions:
        clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_response;
        @p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128

    All CTAs in the cluster will decode the same tile ID. Typically, the result
    (if not -1) should be offset by `tlx.cluster_cta_rank()` within the kernel.

    Note: We may encapsulate this offset step in the TLX frontend after
    evaluating additional use cases.
    """
    bar_empty = local_view(context._clc_mbars_empty, k, _semantic=_semantic)
    bar_full = local_view(context._clc_mbars_full, k, _semantic=_semantic)
    response = local_view(context._clc_responses, k, _semantic=_semantic)

    if multi_ctas:
        assert pred_cta0 is not None, "pred_cta0 must be provided when two_ctas is True"
        bar_empty = remote_view(bar_empty, 0, _semantic=_semantic)
    else:
        assert pred_cta0 is None, "pred_cta0 must be None when two_ctas is False"

    # wait
    barrier_wait(bar_full, p_consumer, pred_cta0, _semantic=_semantic)

    # extract
    stolen_tile_id = _clc_query(response, _semantic=_semantic)

    # release
    barrier_arrive(bar_empty, _semantic=_semantic)

    # return
    return stolen_tile_id
