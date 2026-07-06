"""Warp-partition machinery: turn a node->warp-group assignment into the
``warp_groups[]`` + per-node ``warp_group`` + **re-synthesized**
``cross_wg_barriers[]`` that the sched2tlx emitter consumes.

This is the piece that makes changing the partition safe. SemIR derives every
mbarrier from ``cross_wg_barriers`` alone, so when the solver moves an op to a
different warp group we must rebuild that list to match, or the kernel races /
faults. Rules mirror what modulo's Pass B emits:

  * one mbarrier channel per DDG *data* edge (u->v) whose endpoints land in
    different warp groups AND that ferries a buffer (u produces buffer b, v
    consumes b);
  * the MMA -> epilogue TMEM accumulator hand-off is EXCLUDED (the emitter
    special-cases it as ``acc_tmem`` via tcgen05_commit, not a cross_wg_barrier);
  * depth = the paired buffer's count, expect_bytes = its size_bytes.

We keep the emitter's supported topology: the "default"/epilogue group (owner of
the store / tmem_load) is preserved, and TC ops stay co-resident with the MMA.
"""

from __future__ import annotations

from typing import Any

from .ddg_model import LoopModel


def _pipelines_for_group(loop: LoopModel, node_wg: dict[int, int],
                         wg_id: int) -> list[str]:
    """Ordered, de-duplicated HW-pipeline labels of the ops in a group.
    Must literally contain 'TC'/'TMA' where present (drives emitter role + regs)."""
    seen: list[str] = []
    for n in loop.nodes:
        if node_wg.get(n.id) == wg_id and n.pipeline not in seen:
            seen.append(n.pipeline)
    # Emitter expects at least one label; NONE-only groups keep "NONE".
    return seen or ["NONE"]


def _num_warps_for_group(loop: LoopModel, node_wg: dict[int, int],
                         wg_id: int) -> int:
    """max(min_warps) over the group's ops, snapped to {1,2,4,8} (modulo's rule)."""
    mw = 1
    for n in loop.nodes:
        if node_wg.get(n.id) == wg_id:
            mw = max(mw, n.min_warps)
    for c in (1, 2, 4, 8):
        if mw <= c:
            return c
    return 8


def rebuild_warp_groups(loop: LoopModel, node_wg: dict[int, int]) -> list[dict[str, Any]]:
    ids = sorted({wg for wg in node_wg.values() if wg >= 0})
    return [
        {"id": wg, "pipelines": _pipelines_for_group(loop, node_wg, wg),
         "num_warps": _num_warps_for_group(loop, node_wg, wg)}
        for wg in ids
    ]


def _is_tmem_handoff(loop: LoopModel, producer_id: int, buf_id: int) -> bool:
    """The MMA -> epilogue accumulator hand-off (TMEM). Excluded from barriers."""
    buf = loop.buffer_by_id(buf_id)
    if buf is not None and buf.kind == "tmem":
        return True
    prod = loop.node_by_id(producer_id)
    return prod.op_kind in ("ttng.tc_gen5_mma", "ttng.tc_gen5_mma_scaled")


def synthesize_barriers(loop: LoopModel, node_wg: dict[int, int],
                        buffer_count: dict[int, int]) -> list[dict[str, Any]]:
    """Rebuild cross_wg_barriers from the DDG for a given partition.

    For each buffer that crosses a warp-group boundary (its producer and a
    consumer are in different groups), emit one mbarrier channel.
    """
    barriers: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for buf in loop.buffers:
        if buf.kind not in ("smem",):  # TMEM handoff excluded; barriers are SMEM
            continue
        # producer: node with produces_buffer == buf.id
        producers = [n for n in loop.nodes if n.produces_buffer == buf.id]
        consumers = [n for n in loop.nodes if buf.id in n.consumes_buffers]
        if not producers or not consumers:
            continue
        prod = producers[0]
        pwg = node_wg.get(prod.id, -1)
        for cons in consumers:
            cwg = node_wg.get(cons.id, -1)
            if pwg < 0 or cwg < 0 or pwg == cwg:
                continue
            if _is_tmem_handoff(loop, prod.id, buf.id):
                continue
            key = (prod.id, cons.id, buf.id)
            if key in seen:
                continue
            seen.add(key)
            barriers.append({
                "producer_node": prod.id,
                "consumer_node": cons.id,
                "producer_wg": pwg,
                "consumer_wg": cwg,
                "kind": "mbarrier",
                "depth": buffer_count.get(buf.id, buf.count),
                "paired_buffer_id": buf.id,
                "expect_bytes": buf.size_bytes,
            })
    return barriers
