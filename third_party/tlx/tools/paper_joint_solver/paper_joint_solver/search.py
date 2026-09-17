"""Algorithm 1: monotone search over (I, L) wrapping the modulo-ILP seed and
the joint SMT system.

Each inner failure increments L while ceil(L/I) is unchanged, exactly as
Algorithm 1 line 9.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

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


def run_search(prob: Problem, num_warps_override: int | None = None,
               allow_cross_warp: bool = True,
               modulo_solver: Callable | None = None,
               joint_solver: Callable | None = None) -> SearchResult:
    """Algorithm 1, verbatim: ascend I from 1; one cold, uncapped ILP per I;
    probe the full L window while ceil(L/I) is unchanged; return on first sat.
    num_warps_override / allow_cross_warp are the paper's own experiment inputs
    (its ablation section), not model switches; the model itself has none.
    Termination is not guaranteed for structurally-UNSAT models; bounding is
    the harness's job (watchdog / probe protocol), never this function's.
    """
    if modulo_solver is None:
        from .modulo_ilp import solve_modulo
        modulo_solver = solve_modulo
    if joint_solver is None:
        from .joint_smt import solve_joint
        joint_solver = solve_joint

    t0 = time.time()
    attempts: list[dict] = []
    ii = 0
    while True:                                   # Algorithm 1 line 3
        ii += 1                                   # line 4
        m = modulo_solver(prob, ii)               # line 5
        if m is None:                             # lines 6-7
            _log(attempts, {"ii": ii, "stage": "modulo", "result": "unsat"})
            continue
        length = m.length                         # line 8: L <- LEN(M)
        copies = -(-length // ii)                 # ceil(LEN(M)/I)
        while -(-length // ii) == copies:         # line 9
            t1 = time.time()
            sol, verdict = joint_solver(          # line 10
                prob, ii, length,
                num_warps_override=num_warps_override,
                allow_cross_warp=allow_cross_warp,
            )
            _log(attempts, {"ii": ii, "L": length, "stage": "joint",
                            "result": verdict,
                            "seconds": round(time.time() - t1, 1)})
            if verdict not in ("sat", "unsat"):
                raise RuntimeError(
                    f"joint solver returned non-verdict {verdict!r}")
            if sol is not None:                   # line 14
                return SearchResult(sol, attempts, time.time() - t0, "sat")
            length += 1                           # lines 11-13
