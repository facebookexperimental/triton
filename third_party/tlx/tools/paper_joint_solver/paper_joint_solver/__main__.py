"""CLI: run the joint solver on a curated ddg.json and emit solution artifacts.

  python -m paper_joint_solver <curated_ddg.json> -o solution.json \
      [--num-warps-override N] [--no-cross-warp] [--reg-budget REGS] \
      [--normalization-u U] \
      [--baseline-graph GRAPH.json --ir-out IR.json \
       --handoff-manifest-out HANDOFF.json] \
      [--viz out.dot]

--num-warps-override / --no-cross-warp are the paper's UNSAT-ablation
switches; --reg-budget is the reduced-register ("-LR") run.  There are no
model switches: the solver builds the paper's system and nothing else.

--ir-out / --handoff-manifest-out are the implementation path: the paper's
own workflow hands the pipelined IR to an expert for a manual implementation,
so these two artifacts are the deliverable, not an input to another solver.
"""

import argparse
import json
import sys
from pathlib import Path

from .ddg import load_problem
from .machine import MachineModel
from .normalize import DEFAULT_U
from .schedule_plan import (
    PAPER_MODEL_PROFILE,
    PAPER_SOLUTION_SCHEMA,
    experiment_inputs,
    solution_provenance,
)
from .search import run_search


def add_experiment_inputs(ap):
    """The paper's own experiment inputs, shared by the solve and probe CLIs."""
    ap.add_argument(
        "--num-warps-override",
        type=int,
        default=None,
        help="physical warp budget (paper's reduced-warp ablation)",
    )
    ap.add_argument("--no-cross-warp", action="store_true")
    ap.add_argument(
        "--reg-budget",
        type=int,
        default=None,
        help="32-bit register words available to each physical warp",
    )
    ap.add_argument(
        "--normalization-u",
        type=int,
        default=DEFAULT_U,
        help="paper section 5.2 normalization budget",
    )


def configure_machine(ap, args, machine):
    """Apply the shared experiment inputs to the machine description."""
    if args.normalization_u < 1:
        ap.error("--normalization-u must be positive")
    if args.num_warps_override is not None:
        if not 0 < args.num_warps_override <= machine.num_warps:
            ap.error(
                f"--num-warps-override must be in [1, {machine.num_warps}]"
            )
        machine.num_warps = args.num_warps_override
    if args.reg_budget is not None:
        if not 0 < args.reg_budget <= machine.regs_per_warp:
            ap.error(
                f"--reg-budget must be in [1, {machine.regs_per_warp}]"
            )
        machine.regs_per_warp = args.reg_budget


def recorded_experiment_inputs(args) -> dict:
    return experiment_inputs(
        allow_cross_warp=not args.no_cross_warp,
        num_warps_override=args.num_warps_override,
        reg_budget=args.reg_budget,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="paper_joint_solver")
    ap.add_argument("curated_ddg", help="curated-ddg-0.2 artifact (see curate_ddg)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument(
        "--baseline-graph",
        help=(
            "baseline schedule graph; an IR-emission input only, legal solely "
            "together with --ir-out/--handoff-manifest-out"
        ),
    )
    ap.add_argument(
        "--ir-out",
        help="emit the paper's software-pipelined, warp-annotated IR",
    )
    ap.add_argument(
        "--handoff-manifest-out",
        help="record the remaining expert CUDA C++ lowering contract",
    )
    add_experiment_inputs(ap)
    ap.add_argument("--viz")
    args = ap.parse_args(argv)

    machine = MachineModel()
    # The baseline graph is only ever an IR-construction input: the IR needs
    # its op table to resolve op_ref.  All three must appear together.
    ir_group = (args.baseline_graph, args.ir_out, args.handoff_manifest_out)
    if any(value is not None for value in ir_group) and not all(
        value is not None for value in ir_group
    ):
        ap.error(
            "--baseline-graph, --ir-out and --handoff-manifest-out "
            "must be specified together"
        )
    configure_machine(ap, args, machine)
    prob = load_problem(
        args.curated_ddg,
        machine=machine,
        u=args.normalization_u,
    )
    normalization = prob.normalization_stats()
    print(f"[normalization] {json.dumps(normalization, sort_keys=True)}")
    print("[normalization] index\traw\tscaled")
    for index, (raw, scaled) in enumerate(prob.normalization_cost_pairs):
        print(f"[normalization] {index}\t{raw}\t{scaled}")
    result = run_search(prob, num_warps_override=args.num_warps_override,
                        allow_cross_warp=not args.no_cross_warp)
    payload = {
        "schema_version": PAPER_SOLUTION_SCHEMA,
        "provenance": solution_provenance(
            args.curated_ddg,
            machine,
            args.normalization_u,
            model=PAPER_MODEL_PROFILE,
            inputs=recorded_experiment_inputs(args),
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
                        "cycles": s.cycles, "warp_sets": s.warp_sets,
                        "stats": s.stats})
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"[paper_joint_solver] {result.status}"
          f" after {result.wall_s:.1f}s -> {args.out}")

    if result.solution and args.ir_out:
        from .pipelined_ir import prepare_manual_cuda_handoff

        handoff = prepare_manual_cuda_handoff(
            solution_path=args.out,
            ddg_path=args.curated_ddg,
            baseline_graph_path=args.baseline_graph,
            ir_out=args.ir_out,
            manifest_out=args.handoff_manifest_out,
        )
        print(f"[paper_joint_solver] wrote {handoff.ir_path}")
        print(f"[paper_joint_solver] wrote {handoff.manifest_path}")
    if result.solution and args.viz:
        from .viz import render
        render(prob, result.solution.warp_sets, args.viz)
        print(f"[paper_joint_solver] wrote {args.viz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
