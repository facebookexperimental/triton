"""Algorithm 1: monotone search over (I, L) wrapping the modulo-ILP seed and
the joint SMT system.

Fidelity notes: the paper increments I from 1; every I below
max(ResMII, RecMII) provably has no modulo schedule, so we start at the bound
(identical result, fewer no-op ILP calls).  Each inner failure increments L
while ceil(L/I) is unchanged, exactly as Algorithm 1 line 9.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .errors import SolverUnknownError

if TYPE_CHECKING:
    from .ddg import Problem
    from .joint_smt import JointSolution


def _log(attempts, entry):
    attempts.append(entry)
    print("[attempt]", entry, flush=True)


@dataclass
class SearchResult:
    solution: JointSolution | None
    attempts: list[dict]
    wall_s: float
    status: str


def _in_no_cross_warp_domain(prob, edge, domain: str) -> bool:
    if not prob.edge_has_ws_semantics(edge) or edge.signal_only:
        return False
    return domain == "all-ws" or prob.regs.get(edge.src, 0) > 0


def run_search(prob: Problem, num_warps: int | None = None,
               max_groups: int | None = None,
               allow_cross_warp: bool = True,
               no_cross_warp_domain: str = "reg-data",
               structural_precheck: bool = True,
               max_ii_span: int | None = None,
               ilp_seconds: float = 300.0, smt_seconds: float | None = None,
               max_wall_s: float = 3600.0, minimize_groups: bool = False,
               prefix_lane_masks: bool = False,
               full_group_lane_masks: bool = False,
               spill_smem_footprint: bool = True,
               reg_carried_same_lane: bool = True,
               max_probes_per_ii: int | None = None,
               modulo_solver: Callable | None = None,
               joint_solver: Callable | None = None) -> SearchResult:
    """Per Algorithm 1 the inner loop probes the FULL L window while
    ceil(L/I) is unchanged (it is finite: the window closes when the ceiling
    grows).  max_probes_per_ii=None is the paper-faithful default; an integer
    is an explicit resource cap, like max_ii_span / max_wall_s (plan doc
    sec 12 resource-limit class).  full_group_lane_masks is an optional
    downstream-emitter capability constraint, not a paper objective.
    `no_cross_warp_domain="reg-data"` is the paper's register-transport
    interpretation; `"all-ws"` additionally forbids buffer-mediated data
    edges. `structural_precheck=False` preserves the contradiction in the SMT
    attempt log instead of returning before Algorithm 1.
    """
    if no_cross_warp_domain not in ("reg-data", "all-ws"):
        raise ValueError(
            "no-cross-warp domain must be 'reg-data' or 'all-ws'"
        )
    if modulo_solver is None:
        from .modulo_ilp import solve_modulo
        modulo_solver = solve_modulo
    if joint_solver is None:
        from .joint_smt import solve_joint
        joint_solver = solve_joint

    t0 = time.time()
    attempts: list[dict] = []
    if not allow_cross_warp and structural_precheck:
        conflict = next(
            (
                edge
                for edge in prob.edges
                if _in_no_cross_warp_domain(
                    prob, edge, no_cross_warp_domain
                )
                and ((edge.src in prob.variable_latency) !=
                     (edge.dst in prob.variable_latency))
            ),
            None,
        )
        if conflict is not None:
            _log(
                attempts,
                {
                    "stage": "structural",
                    "result": "unsat",
                    "reason": "VARIABLELATENCY conflicts with no-cross-warp",
                    "edge": conflict.index,
                },
            )
            return SearchResult(None, attempts, time.time() - t0, "unsat")
    ii = prob.min_ii()
    ii_attempts = 0
    while max_ii_span is None or ii_attempts < max_ii_span:
        if time.time() - t0 > max_wall_s:
            return SearchResult(
                None, attempts, time.time() - t0, "resource_limited"
            )
        try:
            m = modulo_solver(prob, ii, max_seconds=ilp_seconds)
        except SolverUnknownError as error:
            _log(attempts, {"ii": ii, "stage": "modulo", "result": "unknown",
                            "reason": str(error)})
            return SearchResult(None, attempts, time.time() - t0, "unknown")
        if m is None:
            _log(attempts, {"ii": ii, "stage": "modulo", "result": "unsat"})
            ii += 1
            ii_attempts += 1
            continue
        length = m.length
        copies = -(-length // ii)
        probes = 0
        while -(-length // ii) == copies:
            if max_probes_per_ii is not None and probes >= max_probes_per_ii:
                return SearchResult(
                    None, attempts, time.time() - t0, "resource_limited"
                )
            if time.time() - t0 > max_wall_s:
                return SearchResult(
                    None, attempts, time.time() - t0, "resource_limited"
                )
            t1 = time.time()
            sol, verdict = joint_solver(
                prob,
                ii,
                length,
                num_warps=num_warps,
                max_groups=max_groups,
                allow_cross_warp=allow_cross_warp,
                no_cross_warp_domain=no_cross_warp_domain,
                timeout_s=smt_seconds,
                prefix_lane_masks=prefix_lane_masks,
                full_group_lane_masks=full_group_lane_masks,
                spill_smem_footprint=spill_smem_footprint,
                reg_carried_same_lane=reg_carried_same_lane,
            )
            _log(attempts, {"ii": ii, "L": length, "stage": "joint",
                            "result": verdict,
                            "seconds": round(time.time() - t1, 1)})
            if verdict == "unknown":
                return SearchResult(None, attempts, time.time() - t0, "unknown")
            if sol is not None:
                if minimize_groups and max_groups is None:
                    sol = _minimize_groups(
                        prob,
                        sol,
                        num_warps,
                        allow_cross_warp,
                        no_cross_warp_domain,
                        smt_seconds,
                        prefix_lane_masks,
                        full_group_lane_masks,
                        spill_smem_footprint,
                        reg_carried_same_lane,
                        attempts,
                        joint_solver,
                    )
                return SearchResult(sol, attempts, time.time() - t0, "sat")
            length += 1
            probes += 1
        ii += 1
        ii_attempts += 1
    return SearchResult(None, attempts, time.time() - t0, "resource_limited")


def _minimize_groups(
    prob,
    sol,
    num_warps,
    allow_cross_warp,
    no_cross_warp_domain,
    smt_seconds,
    prefix_lane_masks,
    full_group_lane_masks,
    spill_smem_footprint,
    reg_carried_same_lane,
    attempts,
    joint_solver,
):
    """Optional non-paper secondary objective over emitter group labels."""
    best = sol
    used = len(set(sol.warp.values()))
    for groups in range(used - 1, 0, -1):
        t1 = time.time()
        trial, verdict = joint_solver(
            prob,
            sol.ii,
            sol.length,
            num_warps=num_warps,
            max_groups=groups,
            allow_cross_warp=allow_cross_warp,
            no_cross_warp_domain=no_cross_warp_domain,
            timeout_s=smt_seconds,
            prefix_lane_masks=prefix_lane_masks,
            full_group_lane_masks=full_group_lane_masks,
            spill_smem_footprint=spill_smem_footprint,
            reg_carried_same_lane=reg_carried_same_lane,
        )
        _log(attempts, {"ii": sol.ii, "L": sol.length, "stage": "min-groups",
                        "groups": groups, "result": verdict,
                        "seconds": round(time.time() - t1, 1)})
        if verdict == "unknown":
            best.stats["group_minimality"] = "unknown"
            break
        if verdict == "unsat":
            best.stats["group_minimality"] = "proven"
            break
        best = trial
    return best
