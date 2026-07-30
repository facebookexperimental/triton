"""Re-verify historical verdicts after paper-fidelity model changes.

Sequential (memory-gated).  Prints a PASS/CHANGE line per check against the
v6 record; artifacts to refit/.  A CHANGE is expected after the P1 resource
model correction and must not be relabeled as a regression without analysis.
"""
import json
import resource
import time
from pathlib import Path

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.joint_smt import solve_joint
from paper_joint_solver.search import run_search

resource.setrlimit(resource.RLIMIT_AS, (150 * 2**30, 150 * 2**30))
OUT = Path("refit")
OUT.mkdir(exist_ok=True)
EX = Path("../sched2tlx/examples")
report = []


def check(name, got, want):
    ok = got == want
    report.append((name, got, want, ok))
    print(f"[{'PASS' if ok else 'CHANGE'}] {name}: got {got}, v6 {want}", flush=True)


def save(name, res):
    if res.solution:
        s = res.solution
        json.dump({"ii": s.ii, "length": s.length, "copies": s.copies,
                   "cycles": {str(k): v for k, v in s.cycles.items()},
                   "warp": {str(k): v for k, v in s.warp.items()},
                   "group_widths": {str(k): v for k, v in s.group_widths.items()},
                   "lane_masks": {str(k): list(v)
                                  for k, v in s.lane_masks.items()},
                   "stats": s.stats},
                  open(OUT / f"{name}.json", "w"), indent=1)


# 1) legacy subtiled fwd search point (pre-fidelity model)
prob = load_problem(
    EX / "case3_FA_fp16_subtiled" / "ddg.json",
    baseline_graph=EX / "case3_FA_fp16_subtiled" / "schedule_graph.json",
)
t0 = time.time()
res = run_search(prob)
save("subtiled", res)
s = res.solution
check("subtiled (II,L,groups)", (s.ii, s.length, len(set(s.warp.values()))) if s else None, (66, 146, 4))
print(f"  wall {time.time()-t0:.0f}s", flush=True)

# 2) Structural group probes; these are not the paper's physical-warp ablation.
_, v = solve_joint(prob, 66, 146, max_groups=3, allow_cross_warp=True, timeout_s=600)
check("subtiled groups=3 verdict", v, "unsat")
# FA4-exact: pin the recorded v6 partition (colocate per group, separate reps)
fa4 = json.load(open("subtiled_fa4exact_solution.json"))
groups = {}
for vs, w in fa4["warp"].items():
    groups.setdefault(w, []).append(int(vs))
coloc = [sorted(g) for g in groups.values() if len(g) > 1]
reps = {w: min(g) for w, g in groups.items()}
sep = [(reps[a], reps[b]) for a in sorted(reps) for b in sorted(reps) if a < b]
sol, v = solve_joint(prob, fa4["ii"], fa4["length"], max_groups=5,
                     allow_cross_warp=True, timeout_s=900,
                     colocate=coloc, separate=sep)
check(f"FA4-exact partition SAT at ({fa4['ii']},{fa4['length']})", v, "sat")

# 3) no-subtile fwd at ZLP-min II=60: full L window must be 16/16 unsat
prob_f = load_problem(
    EX / "case3_FA_fp16" / "ddg.json",
    u=150,
    baseline_graph=EX / "case3_FA_fp16" / "schedule_graph.json",
)
n_unsat = 0
L0 = None
from paper_joint_solver.modulo_ilp import solve_modulo
m = solve_modulo(prob_f, 60, max_seconds=300)
L0 = m.length if m else None
if L0 is not None:
    copies = -(-L0 // 60)
    L, n_pts = L0, 0
    while -(-L // 60) == copies and n_pts < 16:
        _, v = solve_joint(prob_f, 60, L, allow_cross_warp=True, timeout_s=600)
        n_unsat += (v == "unsat")
        n_pts += 1
        L += 1
    check("no-subtile II=60 unsat count", (n_unsat, n_pts), (16, 16))
else:
    check("no-subtile II=60 seed", None, "modulo seed")

# 4) Legacy bwd search (v6: II=95, L=273, four groups).
prob_b = load_problem(
    EX / "case4_FA_bwd" / "ddg_hd128.json",
    baseline_graph=EX / "case4_FA_bwd" / "schedule_graph_hd128.json",
)
t0 = time.time()
res_b = run_search(prob_b)
save("bwd", res_b)
s = res_b.solution
check("bwd (II,L,groups)", (s.ii, s.length, len(set(s.warp.values()))) if s else None, (95, 273, 4))
print(f"  wall {time.time()-t0:.0f}s", flush=True)
_, v = solve_joint(prob_b, 95, 273, max_groups=3, allow_cross_warp=True, timeout_s=600)
check("bwd groups=3 verdict", v, "unsat")

print("\n==== SUMMARY ====")
for name, got, want, ok in report:
    print(f"{'PASS' if ok else 'CHANGE':6s} {name}: {got} (v6: {want})")
