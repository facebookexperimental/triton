"""The MoE routing front end on gfx950: FP32 router logits, sigmoid top-k, and the expert-aligned slot order that a
grouped expert GEMM consumes, in three launches with no logits round trip.

The steps are the kernels of ``router_gemm``, ``moe_routing`` and ``tlx.ops.moe``'s slot sort:

1. ``router_gemm``'s GEMV (up to ``GEMV_MAX_TOKENS`` tokens) writes the logits, or its split-K MFMA kernel writes the
   per-split partials and skips its reduction launch;
2. ``moe_routing``'s kernel adds the partials in split order as it loads them (the same additions as
   ``router_gemm``'s reduction), then picks the top-k;
3. ``tlx.ops.moe``'s align maps the ids to the local experts and sorts the slots into ``BLOCK_M``-padded blocks.

Every output is bitwise equal to that chain of ops. A serving stack that runs those steps as router GEMM, reduction,
top-k, local-id map and align launches saves two launches per MoE layer.

A single launch, with each program counting itself off on a device-scope atomic and the last program doing the top-k
and the sort, is slower on MI350X: the atomic reduction and the one-program tail cost far more than the launches they
save (at 16 tokens 38.6 us against 12.8 us for the separate ops).
"""

from __future__ import annotations

from ...moe import gfx950 as moe
from ...moe import routing_gfx950 as routing
from ...router_gemm import gfx950 as router


def moe_router_topk_align(x, weight, correction_bias, topk, num_local, local_start, block_m, router_multiplier,
                          route_scale, eps):
    T = x.shape[0]
    if T <= router.GEMV_MAX_TOKENS:
        logits, splits = router.router_gemm(x, weight, None), 1
    else:
        logits, splits = router.router_gemm_partials(x, weight)
    weights, ids = routing.moe_routing(logits, correction_bias, topk, router_multiplier, route_scale, eps, None, None,
                                       splits=splits)
    sorted_ids, expert_ids, num_post = moe._align(ids, num_local, local_start, block_m)
    return weights, ids, sorted_ids, expert_ids, num_post
