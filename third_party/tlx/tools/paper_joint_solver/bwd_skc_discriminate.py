"""Which pinned constraint blocks the bwd dedicated-issuer family?

Relax one element of the (colocate, separate) template at a time and probe
(II=95..105, a few L) — the relaxation that flips SAT names the binding
constraint.  Verdicts only; solutions are discarded.
"""

import resource
import time

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.joint_smt import solve_joint

resource.setrlimit(resource.RLIMIT_AS, (150 * 2**30, 150 * 2**30))

DDG = "../sched2tlx/examples/case4_FA_bwd/ddg_hd128.json"
MMA = [11, 21, 22, 31, 36]
VL = [0, 2, 7, 9]
COMPUTE = [17, 25, 26]
RED = [32, 35]
SEP_FULL = [(11, 0), (11, 17), (11, 32), (0, 17), (0, 32), (17, 32)]

VARIANTS = {
    "full-template": ([MMA, VL, COMPUTE, RED], SEP_FULL),
    "no-sep-compute-vs-reduction": ([MMA, VL, COMPUTE, RED],
                                    [s for s in SEP_FULL if s != (17, 32)]),
    "only-mma-colocate": ([MMA], [(11, 0), (11, 17), (11, 32)]),
    "mma-minus-dv-dot": ([[11, 21, 31, 36], VL, COMPUTE, RED], SEP_FULL),
    "mma-minus-dq-dot": ([[11, 21, 22, 31], VL, COMPUTE, RED], SEP_FULL),
    "w5": ([MMA, VL, COMPUTE, RED], SEP_FULL),  # extra free warp
}

prob = load_problem(DDG)
for name, (coloc, sep) in VARIANTS.items():
    W = 5 if name == "w5" else 4
    for II, L in [(95, 285), (99, 297), (105, 315)]:
        t0 = time.time()
        _, verdict = solve_joint(prob, II, L, num_warps=W,
                                 allow_cross_warp=True, timeout_s=600,
                                 colocate=coloc, separate=sep)
        print(f"{name:28s} II={II} L={L} W={W} -> {verdict} "
              f"({time.time() - t0:.1f}s)", flush=True)
        if verdict == "sat":
            break
