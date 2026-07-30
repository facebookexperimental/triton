"""De-symmetrization double-check: re-solve a found solution's exact
(I*, L*, group-count) point with symmetry breaking DISABLED and verify the
verdict matches (SAT), plus re-verify the structural group-count boundary.

The usage-ordering constraint is solution-preserving up to warp relabeling;
this check validates that claim empirically on the reported points.

Usage: python desym_check.py <ddg.json> <solution.json> [timeout_s]
"""

import json
import sys

sys.path.insert(0, ".")

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.joint_smt import solve_joint


def main():
    ddg_path, sol_path = sys.argv[1], sys.argv[2]
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0
    prob = load_problem(ddg_path)
    sol = json.loads(open(sol_path).read())
    ii, length = sol["ii"], sol["length"]
    w_star = len(set(sol["warp"].values()))
    checks = []
    for label, w, expect in ((f"SAT@groups={w_star}", w_star, "sat"),
                             (f"UNSAT@groups={w_star - 1}", w_star - 1, "unsat")):
        _, verdict = solve_joint(prob, ii, length, max_groups=w,
                                 timeout_s=timeout, symmetry_breaking=False)
        ok = verdict == expect
        checks.append(ok)
        print(f"desym {label} (II={ii}, L={length}): got {verdict} "
              f"-> {'MATCH' if ok else 'MISMATCH'}", flush=True)
    print("DESYM-CHECK", "PASS" if all(checks) else "FAIL", flush=True)


if __name__ == "__main__":
    main()
