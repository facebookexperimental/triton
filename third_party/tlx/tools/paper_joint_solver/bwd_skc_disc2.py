"""Which pair inside the compute colocate [17,25,26] is infeasible?"""
import resource
import time

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.joint_smt import solve_joint

resource.setrlimit(resource.RLIMIT_AS, (150 * 2**30, 150 * 2**30))
MMA = [11, 21, 22, 31, 36]
SEP = [(11, 0), (11, 17), (11, 32)]
prob = load_problem("../sched2tlx/examples/case4_FA_bwd/ddg_hd128.json")
for name, extra in [("17+25", [[17, 25]]), ("17+26", [[17, 26]]),
                    ("25+26", [[25, 26]]), ("17+25+26", [[17, 25, 26]])]:
    for II, L in [(95, 285), (105, 315)]:
        t0 = time.time()
        _, v = solve_joint(prob, II, L, num_warps=4, allow_cross_warp=True,
                           timeout_s=600, colocate=[MMA] + extra, separate=SEP)
        print(f"compute-pin {name:9s} II={II} L={L} -> {v} "
              f"({time.time() - t0:.1f}s)", flush=True)
        if v == "sat":
            break
