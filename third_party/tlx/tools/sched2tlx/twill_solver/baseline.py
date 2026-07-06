"""Extract the committed modulo schedule as a feasible ``Solution``.

This is both the warm-start hint for CP-SAT and the no-regression floor: any
candidate the solver emits must be cost-<= this baseline, otherwise we emit the
baseline verbatim (guaranteeing the generated kernel is never worse than what
modulo produced).
"""

from __future__ import annotations

from .ddg_model import Model
from .solution import LoopSolution, Solution


def extract(model: Model) -> Solution:
    sol = Solution()
    for loop in model.loops:
        node_cycle: dict[int, int] = {}
        node_stage: dict[int, int] = {}
        node_cluster: dict[int, int] = {}
        node_wg: dict[int, int] = {}
        # Re-read the committed per-node schedule straight from the raw graph so
        # we capture the exact cycle/stage/cluster the emitter would have seen.
        raw_loop = _raw_loop(model, loop.loop_id)
        raw_nodes = raw_loop["schedule_loop"]["graph"]["nodes"]
        for rn in raw_nodes:
            nid = rn["id"]
            s = rn.get("schedule", {})
            node_cycle[nid] = s.get("cycle", 0)
            node_stage[nid] = s.get("stage", 0)
            node_cluster[nid] = s.get("cluster", 0)
            node_wg[nid] = rn.get("warp_group", -1)

        buffer_count = {b.id: b.count for b in loop.buffers}

        sol.loops[loop.loop_id] = LoopSolution(
            loop_id=loop.loop_id,
            II=loop.baseline_ii,
            node_cycle=node_cycle,
            node_stage=node_stage,
            node_cluster=node_cluster,
            node_wg=node_wg,
            warp_groups=[dict(wg) for wg in loop.warp_groups],
            buffer_count=buffer_count,
            cross_wg_barriers=[dict(b) for b in loop.cross_wg_barriers],
        )
    return sol


def _raw_loop(model: Model, loop_id: int) -> dict:
    for L in model.raw.get("loops", []):
        if L["loop_id"] == loop_id:
            return L
    raise KeyError(f"loop {loop_id} not in raw schedule graph")
