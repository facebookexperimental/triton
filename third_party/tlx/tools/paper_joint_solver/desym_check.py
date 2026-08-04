"""De-symmetrization double-check: re-solve a found solution's exact
(I*, L*, group-count) point with symmetry breaking DISABLED and verify the
verdict matches (SAT), plus re-verify a proven group-count boundary.

The usage-ordering constraint is solution-preserving up to warp relabeling;
this check validates that claim empirically on the reported points.

Usage: python desym_check.py <ddg.json> <solution.json> [timeout_s]
       --baseline-graph <schedule_graph.json>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from paper_joint_solver.joint_smt import solve_joint
from paper_joint_solver.schedule_plan import load_schedule_context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recheck a solution with symmetry breaking disabled"
    )
    parser.add_argument("ddg")
    parser.add_argument("solution")
    parser.add_argument("timeout_s", nargs="?", type=float, default=1800.0)
    parser.add_argument(
        "--baseline-graph",
        required=True,
        help="schedule_graph.json carrying emitter-infrastructure ownership",
    )
    args = parser.parse_args(argv)
    if args.timeout_s <= 0:
        parser.error("timeout_s must be positive")

    sol = json.loads(Path(args.solution).read_text())
    prob, plan = load_schedule_context(
        args.solution,
        args.ddg,
        args.baseline_graph,
    )
    ii, length = plan.ii, plan.length
    w_star = len(plan.group_widths)
    probes = [(f"SAT@groups={w_star}", w_star, "sat")]
    if sol.get("stats", {}).get("group_minimality") == "proven" and w_star > 1:
        probes.append((f"UNSAT@groups={w_star - 1}", w_star - 1, "unsat"))
    checks = []
    for label, w, expect in probes:
        _, verdict = solve_joint(
            prob,
            ii,
            length,
            max_groups=w,
            timeout_s=args.timeout_s,
            symmetry_breaking=False,
        )
        ok = verdict == expect
        checks.append(ok)
        print(
            f"desym {label} (II={ii}, L={length}): got {verdict} "
            f"-> {'MATCH' if ok else 'MISMATCH'}",
            flush=True,
        )
    if len(probes) == 1:
        print("desym group-boundary probe skipped: minimality was not proven")
    passed = all(checks)
    print("DESYM-CHECK", "PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
