"""Skeleton-family probe for the bwd case (SKC §5 closed loop).

The free v6 optimum spreads the 5 MMAs over three warps — not expressible by
the fa_bwd_dkdv_ws skeleton (dedicated 1-warp MMA issuer).  Re-solve at the
same (II, W) with the skeleton's role template pinned via colocate/separate,
walking L upward until SAT (same methodology as the fwd FA4-exact probe).

Node ids (ddg_hd128, v6 numbering):
  MMA set      {11, 21, 22, 31, 36}   all tc_gen5 dots
  TMA loads    {0, 2}                 Q(blk), dO(blk) descriptor loads
  compute      {7, 9, 17, 25, 26}     M/D loads + exp2 + dpT read + dS sub
  reduction    {32, 35}               dq tmem_load + descriptor_reduce
"""

import json
import resource
import sys
import time

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.joint_smt import solve_joint

resource.setrlimit(resource.RLIMIT_AS, (150 * 2**30, 150 * 2**30))

DDG = "../sched2tlx/examples/case4_FA_bwd/ddg_hd128.json"
OUT = "bwd_skc_solution.json"

# M/D tt.loads are variable-latency in the model, so VARIABLELATENCY forces
# them onto the designated VL warp with the descriptor loads (the skeleton's
# in-compute tl.load of M/D is an implementation-layer override, audited at
# bind time).
COLOCATE = [[11, 21, 22, 31, 36], [0, 2, 7, 9], [17, 25, 26], [32, 35]]
SEPARATE = [(11, 0), (11, 17), (11, 32), (0, 17), (0, 32), (17, 32)]

prob = load_problem(DDG)

# The 5 dots' occupancy sums to exactly 95, so the dedicated-issuer family
# has zero slack at the free optimum's II — walk (II, L) ascending per
# Algorithm 1 until the pinned template turns SAT.


def probe(II, L):
    t0 = time.time()
    sol, verdict = solve_joint(prob, II, L, num_warps=4, allow_cross_warp=True,
                               timeout_s=900, colocate=COLOCATE,
                               separate=SEPARATE)
    print(f"II={II} L={L} W=4 -> {verdict} ({time.time() - t0:.1f}s)",
          flush=True)
    return sol, verdict


found = None
for II in range(99, 137, 4):
    for L in range(297, 3 * II + 24, 6):
        sol, verdict = probe(II, L)
        if verdict == "sat":
            found = (II, L, sol)
            break
    if found:
        break

if found:
    II, L, sol = found
    # tighten L: descend while still SAT
    while L > 273:
        s2, v2 = probe(II, L - 1)
        if v2 != "sat":
            break
        L, sol = L - 1, s2
    verdict = "sat"
    if verdict == "sat":
        out = {"ii": sol.ii, "length": sol.length, "copies": sol.copies,
               "cycles": {str(v): t for v, t in sol.cycles.items()},
               "warp": {str(v): w for v, w in sol.warp.items()},
               "stats": {**sol.stats, "probe": "skc-bwd-skeleton-family",
                         "colocate": COLOCATE, "separate": SEPARATE}}
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"wrote {OUT} (II={sol.ii} L={sol.length})", flush=True)
else:
    print("NO SAT in II<=110", flush=True)
    sys.exit(1)
