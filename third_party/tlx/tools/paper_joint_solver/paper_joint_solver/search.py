"""Algorithm 1: monotone search over (I, L) wrapping the modulo-ILP seed and
the joint SMT system.

Fidelity notes: the paper increments I from 1; every I below
max(ResMII, RecMII) provably has no modulo schedule, so we start at the bound
(identical result, fewer no-op ILP calls).  Each inner failure increments L
while ceil(L/I) is unchanged, exactly as Algorithm 1 line 9.
"""

import time
from dataclasses import dataclass

from .ddg import Problem
from .joint_smt import JointSolution, solve_joint
from .modulo_ilp import solve_modulo


def _log(attempts, entry):
    attempts.append(entry)
    print("[attempt]", entry, flush=True)


@dataclass
class SearchResult:
    solution: JointSolution | None
    attempts: list[dict]
    wall_s: float


def run_search(prob: Problem, num_warps: int | None = None,
               allow_cross_warp: bool = True, max_ii_span: int = 8,
               ilp_seconds: float = 300.0, smt_seconds: float | None = None,
               max_wall_s: float = 3600.0, minimize_warps: bool = True,
               max_probes_per_ii: int = 6) -> SearchResult:
    t0 = time.time()
    attempts: list[dict] = []
    start_ii = prob.min_ii()
    for ii in range(start_ii, start_ii + max_ii_span):
        if time.time() - t0 > max_wall_s:
            break
        m = solve_modulo(prob, ii, max_seconds=ilp_seconds)
        if m is None:
            _log(attempts, {"ii": ii, "stage": "modulo", "result": "fail"})
            continue
        length = m.length
        copies = -(-length // ii)
        probes = 0
        while -(-length // ii) == copies and probes < max_probes_per_ii:
            if time.time() - t0 > max_wall_s:
                break
            t1 = time.time()
            sol, verdict = solve_joint(prob, ii, length, num_warps=num_warps,
                                       allow_cross_warp=allow_cross_warp,
                                       timeout_s=smt_seconds)
            _log(attempts, {"ii": ii, "L": length, "stage": "joint",
                            "result": verdict,
                            "seconds": round(time.time() - t1, 1)})
            if sol is not None:
                if minimize_warps and num_warps is None:
                    sol = _minimize_warps(prob, sol, allow_cross_warp,
                                          smt_seconds, attempts)
                return SearchResult(sol, attempts, time.time() - t0)
            length += 1
            probes += 1
    return SearchResult(None, attempts, time.time() - t0)


def _minimize_warps(prob, sol, allow_cross_warp, smt_seconds, attempts):
    """SAT at (I, L) established: shrink the warp budget until UNSAT and keep
    the smallest satisfiable strategy.  (I, L) minimization keeps precedence;
    the first UNSAT here is also the measured 'reduced warps' ablation."""
    best = sol
    used = len(set(sol.warp.values()))
    for w in range(used, 1, -1):
        t1 = time.time()
        trial, verdict = solve_joint(prob, sol.ii, sol.length, num_warps=w,
                                     allow_cross_warp=allow_cross_warp,
                                     timeout_s=smt_seconds)
        _log(attempts, {"ii": sol.ii, "L": sol.length, "stage": "min-warps",
                        "W": w, "result": verdict,
                        "seconds": round(time.time() - t1, 1)})
        if trial is None:
            break
        best = trial
    return best
