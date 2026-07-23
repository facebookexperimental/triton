"""CLI: run the joint solver on a ddg.json and emit solution artifacts.

  python -m paper_joint_solver <ddg.json> -o solution.json \
      [--baseline-graph schedule_graph.json --graph-out rewritten.json] \
      [--num-warps N] [--no-cross-warp] [--reg-budget REGS] \
      [--viz out.dot] [--ilp-seconds S] [--smt-seconds S]

--num-warps / --no-cross-warp are the paper's UNSAT-ablation switches;
--reg-budget is the reduced-register ("-LR") run.
"""

import argparse
import json
import sys
from pathlib import Path

from .ddg import load_problem
from .machine import MachineModel
from .search import run_search


def main(argv=None):
    ap = argparse.ArgumentParser(prog="paper_joint_solver")
    ap.add_argument("ddg")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--baseline-graph")
    ap.add_argument("--graph-out")
    ap.add_argument("--num-warps", type=int, default=None)
    ap.add_argument("--no-cross-warp", action="store_true")
    ap.add_argument("--reg-budget", type=int, default=None)
    ap.add_argument("--viz")
    ap.add_argument("--ilp-seconds", type=float, default=300.0)
    ap.add_argument("--smt-seconds", type=float, default=300.0)
    ap.add_argument("--max-wall-s", type=float, default=3600.0)
    args = ap.parse_args(argv)

    machine = MachineModel()
    if args.reg_budget is not None:
        machine.regs_per_warpgroup = args.reg_budget
    prob = load_problem(args.ddg, machine=machine)
    result = run_search(prob, num_warps=args.num_warps,
                        allow_cross_warp=not args.no_cross_warp,
                        ilp_seconds=args.ilp_seconds,
                        smt_seconds=args.smt_seconds,
                        max_wall_s=args.max_wall_s)
    payload = {
        "attempts": result.attempts,
        "wall_s": round(result.wall_s, 1),
        "satisfiable": result.solution is not None,
    }
    if result.solution:
        s = result.solution
        payload.update({"ii": s.ii, "length": s.length, "copies": s.copies,
                        "cycles": s.cycles, "warp": s.warp, "stats": s.stats})
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"[paper_joint_solver] {'SAT' if result.solution else 'no solution'}"
          f" after {result.wall_s:.1f}s -> {args.out}")

    if result.solution and args.baseline_graph and args.graph_out:
        from .graph_writer import rewrite_schedule_graph
        rewrite_schedule_graph(args.baseline_graph, args.graph_out,
                               ii=result.solution.ii,
                               cycles=result.solution.cycles,
                               warp=result.solution.warp,
                               length=result.solution.length)
        print(f"[paper_joint_solver] wrote {args.graph_out}")
    if result.solution and args.viz:
        from .viz import render
        render(prob, result.solution.warp, args.viz)
        print(f"[paper_joint_solver] wrote {args.viz}")
    return 0 if result.solution else 2


if __name__ == "__main__":
    sys.exit(main())
