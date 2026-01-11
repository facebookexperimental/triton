from typing import Optional
import triton.language.core as tl

from . import types as tlx
from .mem_ops import local_view
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
    """
    Extract tile ID from CLC response.

    Returns the tile ID decoded from the CLC response buffer, or -1 if no work
    is available. In multi-CTA mode, callers should offset by cluster_cta_rank()
    to get unique tile assignments (CTA 0 gets tile N, CTA 1 gets tile N+1, etc.).
    """
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

    Multi-CTA Synchronization ("Arrive Remote, Wait Local"):
    ---------------------------------------------------------
    - WAIT: Only CTA 0 waits on its LOCAL bar_empty (predicated by pred_cta0).
            Other CTAs skip the wait since they will signal CTA 0's barrier.
    - EXPECT: Only CTA 0 sets barrier_expect_bytes (predicated by pred_cta0).
    - ISSUE: CLC try_cancel is issued; hardware multicasts response to all CTAs.

    Key constraint: barrier_wait must use LOCAL mbarrier only (per NVIDIA spec).
    Remote signaling is done via barrier_arrive with remote_cta_rank parameter.

    PTX instruction generated:
        clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128
    """
    bar_empty = local_view(context._clc_mbars_empty, k, _semantic=_semantic)
    bar_full = local_view(context._clc_mbars_full, k, _semantic=_semantic)
    response = local_view(context._clc_responses, k, _semantic=_semantic)

    if multi_ctas:
        assert pred_cta0 is not None, "pred_cta0 must be provided when multi_ctas is True"
    else:
        assert pred_cta0 is None, "pred_cta0 must be None when multi_ctas is False"

    # Only CTA 0 waits on its LOCAL bar_empty (arrive remote, wait local)
    barrier_wait(bar_empty, p_producer, pred_cta0, _semantic=_semantic)

    # Only CTA 0 sets barrier_expect_bytes
    barrier_expect_bytes(bar_full, tl.constexpr(16), pred_cta0, _semantic=_semantic)

    # CLC issue - hardware handles multicast to all CTAs
    _clc_issue(
        response,
        bar_full,
        _semantic=_semantic,
    )


@tl.builtin
def clc_consumer(context, k, p_consumer, multi_ctas: bool = False, pred_cta0: Optional[bool] = None, _semantic=None):
    """
    Decode the tile ID from a CLC response and signal completion.

    Multi-CTA Synchronization ("Arrive Remote, Wait Local"):
    ---------------------------------------------------------
    - WAIT: Only CTA 0 waits on its LOCAL bar_full (predicated by pred_cta0).
            CLC multicasts response to all CTAs, but only CTA 0 needs to wait.
    - QUERY: Extract tile_id from response. Caller should offset by cluster_cta_rank().
    - SIGNAL: All CTAs signal CTA 0's bar_empty via remote_cta_rank=0.
              This is valid because we can arrive at remote mbar, but not wait on it.

    Returns the tile ID if successful, otherwise -1.

    PTX instructions generated:
        clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_response;
        @p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
    """
    bar_empty = local_view(context._clc_mbars_empty, k, _semantic=_semantic)
    bar_full = local_view(context._clc_mbars_full, k, _semantic=_semantic)
    response = local_view(context._clc_responses, k, _semantic=_semantic)

    if multi_ctas:
        assert pred_cta0 is not None, "pred_cta0 must be provided when multi_ctas is True"
    else:
        assert pred_cta0 is None, "pred_cta0 must be None when multi_ctas is False"

    # Only CTA 0 waits on its LOCAL bar_full
    barrier_wait(bar_full, p_consumer, pred_cta0, _semantic=_semantic)

    # Extract tile_id (caller should offset by cluster_cta_rank() in multi-CTA mode)
    stolen_tile_id = _clc_query(response, _semantic=_semantic)

    # Signal completion: all CTAs signal CTA 0's bar_empty
    if multi_ctas:
        # Arrive at CTA 0's bar_empty via remote_cta_rank=0
        # (barrier_arrive handles remote_view internally)
        barrier_arrive(bar_empty, tl.constexpr(1), 0, _semantic=_semantic)
    else:
        barrier_arrive(bar_empty, _semantic=_semantic)

    return stolen_tile_id
