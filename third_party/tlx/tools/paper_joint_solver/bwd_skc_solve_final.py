"""Solve and save the bwd skeleton-family solution (only-mma-colocate form).

The dedicated-issuer pin is the R1 structure the skeleton realizes; compute
chain placement is left to the solver (its spreading is audited at bind time
as protocol overrides, like R1's TMEM relocations).
"""
import json
import resource
import time

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.joint_smt import solve_joint

resource.setrlimit(resource.RLIMIT_AS, (150 * 2**30, 150 * 2**30))

DDG = "../sched2tlx/examples/case4_FA_bwd/ddg_hd128.json"
COLOC = [[11, 21, 22, 31, 36]]
SEP = [(11, 0), (11, 17), (11, 32)]

prob = load_problem(DDG)
best = None
L = 285
while L >= 273:
    t0 = time.time()
    sol, verdict = solve_joint(prob, 95, L, num_warps=4, allow_cross_warp=True,
                               timeout_s=900, colocate=COLOC, separate=SEP)
    print(f"II=95 L={L} -> {verdict} ({time.time() - t0:.1f}s)", flush=True)
    if verdict != "sat":
        break
    best = sol
    L -= 2
if best:
    out = {"ii": best.ii, "length": best.length, "copies": best.copies,
           "cycles": {str(v): t for v, t in best.cycles.items()},
           "warp": {str(v): w for v, w in best.warp.items()},
           "stats": {**best.stats, "probe": "skc-bwd-dedicated-issuer",
                     "colocate": COLOC, "separate": SEP}}
    json.dump(out, open("bwd_skc_solution.json", "w"), indent=1)
    print(f"wrote bwd_skc_solution.json II={best.ii} L={best.length}",
          flush=True)
