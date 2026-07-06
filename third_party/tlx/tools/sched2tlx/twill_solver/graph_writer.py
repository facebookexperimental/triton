"""Stamp a solved ``Solution`` back into ``schedule_graph.json``.

Strategy: deep-copy the original raw JSON and mutate ONLY the fields the emitter
consumes — per-node ``warp_group`` + ``schedule.{cycle,stage,cluster}``, loop
``warp_groups[]``, ``buffers[].count`` (+ paired barrier count), and
``cross_wg_barriers[]``. Everything else (ops table, kernel, bounds, edges)
passes through verbatim, guaranteeing round-trip fidelity for anything the
solver did not change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ddg_model import Model
from .solution import LoopSolution, Solution


class ContractError(RuntimeError):
    """A solved graph violates an emitter structural invariant."""


def build(model: Model, solution: Solution) -> dict[str, Any]:
    raw = model.raw_copy()
    raw["eager_smem_release"] = solution.eager_smem_release
    for L in raw.get("loops", []):
        lid = L["loop_id"]
        ls = solution.loops.get(lid)
        if ls is None:
            continue
        _apply_loop(L, ls)
    _validate(raw)
    return raw


def _apply_loop(raw_loop: dict[str, Any], ls: LoopSolution) -> None:
    raw_loop["warp_groups"] = [dict(wg) for wg in ls.warp_groups]

    sched = raw_loop["schedule_loop"]
    sched["II"] = ls.II
    if ls.node_stage:
        sched["max_stage"] = max(ls.node_stage.values())

    for b in sched.get("buffers", []):
        if b["id"] in ls.buffer_count:
            b["count"] = ls.buffer_count[b["id"]]

    g = sched["graph"]
    for n in g["nodes"]:
        nid = n["id"]
        if nid in ls.node_wg:
            n["warp_group"] = ls.node_wg[nid]
        sc = n.setdefault("schedule", {})
        if nid in ls.node_cycle:
            sc["cycle"] = ls.node_cycle[nid]
        if nid in ls.node_stage:
            sc["stage"] = ls.node_stage[nid]
        if nid in ls.node_cluster:
            sc["cluster"] = ls.node_cluster[nid]

    # Barrier depth follows the paired data buffer's (possibly new) count.
    bufcount = ls.buffer_count
    new_barriers = []
    for cb in ls.cross_wg_barriers:
        cb = dict(cb)
        pb = cb.get("paired_buffer_id")
        if pb is not None and pb in bufcount:
            cb["depth"] = bufcount[pb]
        new_barriers.append(cb)
    g["cross_wg_barriers"] = new_barriers


def _validate(raw: dict[str, Any]) -> None:
    for L in raw.get("loops", []):
        wg_ids = {wg["id"] for wg in L.get("warp_groups", [])}
        g = L["schedule_loop"]["graph"]
        for n in g["nodes"]:
            wg = n.get("warp_group", -1)
            # -1 (unassigned) / -2 (replicated infra) are legal sentinels.
            if wg >= 0 and wg not in wg_ids:
                raise ContractError(
                    f"loop {L['loop_id']} node {n['id']} warp_group={wg} "
                    f"not in warp_groups {sorted(wg_ids)}")
        for b in L["schedule_loop"].get("buffers", []):
            if b["count"] < 1:
                raise ContractError(
                    f"loop {L['loop_id']} buffer {b['id']} count<1")
        for cb in g.get("cross_wg_barriers", []):
            for side in ("producer_wg", "consumer_wg"):
                if cb[side] not in wg_ids:
                    raise ContractError(
                        f"loop {L['loop_id']} barrier {side}={cb[side]} "
                        f"not in warp_groups {sorted(wg_ids)}")
        # Heterogeneous ring-depth guard: the emitter indexes every SMEM ring in
        # a warp group with one shared counter `_it % max(count)`, so all SMEM
        # buffers a group consumes must share the same count — otherwise the
        # shallower ring is indexed out of range (illegal access). Group the
        # paired SMEM buffers by consumer_wg and require a single count each.
        bufs = {b["id"]: b for b in L["schedule_loop"].get("buffers", [])}
        per_wg: dict[int, set[int]] = {}
        for cb in g.get("cross_wg_barriers", []):
            pb = cb.get("paired_buffer_id")
            b = bufs.get(pb) if pb is not None else None
            if b is not None and b.get("kind") == "smem" and b["count"] > 1:
                per_wg.setdefault(cb["consumer_wg"], set()).add(b["count"])
        for wg, counts in per_wg.items():
            if len(counts) > 1:
                raise ContractError(
                    f"loop {L['loop_id']} warp_group {wg} consumes SMEM rings "
                    f"with mixed counts {sorted(counts)} — the emitter's single "
                    f"per-WG ring index would go out of range")


def write(model: Model, solution: Solution, out_path: str | Path) -> dict[str, Any]:
    raw = build(model, solution)
    with open(out_path, "w") as f:
        json.dump(raw, f, indent=2)
    return raw
