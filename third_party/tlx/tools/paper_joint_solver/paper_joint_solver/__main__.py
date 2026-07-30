"""CLI: run the joint solver on a ddg.json and emit solution artifacts.

  python -m paper_joint_solver <ddg.json> -o solution.json \
      [--baseline-graph schedule_graph.json --ir-out pipelined_ir.json \
       --handoff-manifest-out manual_cuda_handoff.json] \
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
from .normalize import DEFAULT_U
from .schedule_plan import SOLUTION_SCHEMA, solution_provenance
from .search import run_search


def main(argv=None):
    ap = argparse.ArgumentParser(prog="paper_joint_solver")
    ap.add_argument("ddg")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--baseline-graph")
    ap.add_argument(
        "--ir-out",
        help="emit the paper's software-pipelined, warp-annotated IR",
    )
    ap.add_argument(
        "--handoff-manifest-out",
        help="record the remaining expert CUDA C++ lowering contract",
    )
    ap.add_argument(
        "--num-warps",
        type=int,
        default=None,
        help="physical warp budget (paper's reduced-warp ablation)",
    )
    ap.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="optional emitter partition-label cap; not a paper objective",
    )
    ap.add_argument(
        "--warp-fixed-overhead",
        type=int,
        default=None,
        help="physical warps for default/function-scope work absent from DDG",
    )
    ap.add_argument("--minimize-groups", action="store_true")
    ap.add_argument("--no-cross-warp", action="store_true")
    ap.add_argument(
        "--reg-budget",
        type=int,
        default=None,
        help="32-bit register words available to each physical warp",
    )
    ap.add_argument(
        "--reg-fixed-overhead",
        type=int,
        default=0,
        help="CTA register words used outside the scheduled loop DDG",
    )
    ap.add_argument(
        "--smem-fixed-overhead",
        type=int,
        default=0,
        help="function-scope SMEM bytes absent from the loop DDG",
    )
    ap.add_argument(
        "--tmem-fixed-cols",
        type=int,
        default=0,
        help="function-scope TMEM columns absent from the loop DDG",
    )
    ap.add_argument("--viz")
    ap.add_argument(
        "--normalization-u",
        type=int,
        default=DEFAULT_U,
        help="paper section 5.2 normalization budget",
    )
    ap.add_argument("--ilp-seconds", type=float, default=300.0)
    ap.add_argument("--smt-seconds", type=float, default=300.0)
    ap.add_argument("--max-wall-s", type=float, default=3600.0)
    ap.add_argument(
        "--max-ii-span",
        type=int,
        default=None,
        help="optional resource cap on initiation intervals searched",
    )
    args = ap.parse_args(argv)

    machine = MachineModel()
    if args.ir_out and not args.baseline_graph:
        ap.error("--ir-out requires --baseline-graph")
    if bool(args.ir_out) != bool(args.handoff_manifest_out):
        ap.error("--ir-out and --handoff-manifest-out must be specified together")
    if args.max_ii_span is not None and args.max_ii_span < 1:
        ap.error("--max-ii-span must be positive")
    if args.normalization_u < 1:
        ap.error("--normalization-u must be positive")
    if args.num_warps is not None:
        if not 0 < args.num_warps <= machine.num_warps:
            ap.error(
                f"--num-warps must be in [1, {machine.num_warps}]"
            )
        machine.num_warps = args.num_warps
    warp_fixed_overhead = args.warp_fixed_overhead or 0
    if not 0 <= warp_fixed_overhead < machine.num_warps:
        ap.error(
            f"--warp-fixed-overhead must be in [0, {machine.num_warps - 1}]"
        )
    machine.warp_fixed_overhead = warp_fixed_overhead
    if args.reg_budget is not None:
        if not 0 < args.reg_budget <= machine.regs_per_warp:
            ap.error(
                f"--reg-budget must be in [1, {machine.regs_per_warp}]"
            )
        machine.regs_per_warp = args.reg_budget
    if not 0 <= args.reg_fixed_overhead < machine.regs_per_sm:
        ap.error(
            f"--reg-fixed-overhead must be in [0, {machine.regs_per_sm - 1}]"
        )
    machine.regs_fixed_overhead = args.reg_fixed_overhead
    if not 0 <= args.smem_fixed_overhead <= machine.smem_bytes:
        ap.error(
            f"--smem-fixed-overhead must be in [0, {machine.smem_bytes}]"
        )
    if not 0 <= args.tmem_fixed_cols <= machine.tmem_cols:
        ap.error(f"--tmem-fixed-cols must be in [0, {machine.tmem_cols}]")
    machine.smem_fixed_overhead = args.smem_fixed_overhead
    machine.tmem_fixed_cols = args.tmem_fixed_cols
    prob = load_problem(
        args.ddg,
        machine=machine,
        u=args.normalization_u,
        baseline_graph=args.baseline_graph,
    )
    result = run_search(prob, num_warps=args.num_warps,
                        max_groups=args.max_groups,
                        allow_cross_warp=not args.no_cross_warp,
                        ilp_seconds=args.ilp_seconds,
                        smt_seconds=args.smt_seconds,
                        max_wall_s=args.max_wall_s,
                        max_ii_span=args.max_ii_span,
                        minimize_groups=args.minimize_groups)
    payload = {
        "schema_version": SOLUTION_SCHEMA,
        "provenance": solution_provenance(
            args.ddg,
            args.baseline_graph,
            machine,
            args.normalization_u,
        ),
        "attempts": result.attempts,
        "wall_s": round(result.wall_s, 1),
        "status": result.status,
        "satisfiable": result.solution is not None,
    }
    if result.solution:
        s = result.solution
        payload.update({"ii": s.ii, "length": s.length, "copies": s.copies,
                        "horizon": s.horizon,
                        "cycles": s.cycles, "warp": s.warp,
                        "group_widths": s.group_widths,
                        "lane_masks": s.lane_masks, "stats": s.stats})
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"[paper_joint_solver] {result.status}"
          f" after {result.wall_s:.1f}s -> {args.out}")

    if result.solution and args.ir_out:
        from .pipelined_ir import prepare_manual_cuda_handoff

        handoff = prepare_manual_cuda_handoff(
            solution_path=args.out,
            ddg_path=args.ddg,
            baseline_graph_path=args.baseline_graph,
            ir_out=args.ir_out,
            manifest_out=args.handoff_manifest_out,
        )
        print(f"[paper_joint_solver] wrote {handoff.ir_path}")
        print(f"[paper_joint_solver] wrote {handoff.manifest_path}")
    if result.solution and args.viz:
        from .viz import render
        render(prob, result.solution.warp, args.viz)
        print(f"[paper_joint_solver] wrote {args.viz}")
    return 0 if result.solution else 2


if __name__ == "__main__":
    sys.exit(main())
