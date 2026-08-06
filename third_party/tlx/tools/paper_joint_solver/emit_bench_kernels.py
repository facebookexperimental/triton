#!/usr/bin/env python3
"""Materialize the three v8 joint-solver kernels used by bench_bars.

The workflow is intentionally append-never. It validates all three solver
artifacts before creating a run record, stages every graph and emitted kernel
inside that record, and publishes the six benchmark inputs only after every
case succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from paper_joint_solver.graph_writer import (
    normalized_timing_model,
    rewrite_schedule_graph,
    validate,
)
from paper_joint_solver.machine import MachineModel
from paper_joint_solver.schedule_plan import (
    EMITTER_MODEL_PROFILE,
    ScheduleArtifactError,
    load_schedule_context,
    machine_manifest,
)


ROOT = Path(__file__).resolve().parent
TOOLS = ROOT.parent
SCHED2TLX = TOOLS / "sched2tlx"
EXAMPLES = SCHED2TLX / "examples"
WORKFLOW_SCHEMA = "jos-bench-emission-v1"
NORMALIZATION_U = 300
EMISSION_SOURCE_FILES = (
    Path(__file__).resolve(),
    ROOT / "paper_joint_solver" / "graph_writer.py",
    ROOT / "paper_joint_solver" / "machine.py",
    ROOT / "paper_joint_solver" / "resource_model.py",
    SCHED2TLX / "sched2tlx" / "__init__.py",
    SCHED2TLX / "sched2tlx" / "__main__.py",
    SCHED2TLX / "sched2tlx" / "cli.py",
    SCHED2TLX / "sched2tlx" / "emitter.py",
    SCHED2TLX / "sched2tlx" / "schedule_graph.py",
    SCHED2TLX / "sched2tlx" / "semaphore_ir.py",
)


class WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaseSpec:
    name: str
    solution: Path
    ddg: Path
    baseline: Path
    graph_output: Path
    kernel_output: Path
    regs_per_warp: int
    warp_fixed_overhead: int
    require_tmem_evidence: bool = False


def _cases(solution_dir: Path) -> tuple[CaseSpec, ...]:
    fwd = EXAMPLES / "case3_FA_fp16_subtiled"
    bwd = EXAMPLES / "case4_FA_bwd_subtiled"
    return (
        CaseSpec(
            name="fwd_subtiled_emitter_v8",
            solution=solution_dir / "fwd_subtiled_emitter_v8.json",
            ddg=fwd / "ddg.json",
            baseline=fwd / "schedule_graph.json",
            graph_output=fwd / "schedule_graph_jos_v8.json",
            kernel_output=fwd / "generated_jos_v8.py",
            regs_per_warp=8160,
            warp_fixed_overhead=4,
        ),
        CaseSpec(
            name="bwd_emitter_v8",
            solution=solution_dir / "bwd_emitter_v8.json",
            ddg=bwd / "ddg.json",
            baseline=bwd / "schedule_graph.json",
            graph_output=bwd / "schedule_graph_jos_v8.json",
            kernel_output=bwd / "generated_jos_v8.py",
            regs_per_warp=8160,
            warp_fixed_overhead=0,
            require_tmem_evidence=True,
        ),
        CaseSpec(
            name="bwd_lr4096_emitter_v8",
            solution=solution_dir / "bwd_lr4096_emitter_v8.json",
            ddg=bwd / "ddg.json",
            baseline=bwd / "schedule_graph.json",
            graph_output=bwd / "schedule_graph_jos_lr4096_v8.json",
            kernel_output=bwd / "generated_jos_lr4096_v8.py",
            regs_per_warp=4096,
            warp_fixed_overhead=0,
            require_tmem_evidence=True,
        ),
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _emission_source_manifest() -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    sources = {}
    for path in EMISSION_SOURCE_FILES:
        name = str(path.relative_to(TOOLS))
        content = path.read_bytes()
        sources[name] = hashlib.sha256(content).hexdigest()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(content)
    return digest.hexdigest(), sources


def _expected_machine(regs_per_warp: int, warp_fixed_overhead: int) -> dict:
    machine = MachineModel()
    machine.regs_per_warp = regs_per_warp
    machine.warp_fixed_overhead = warp_fixed_overhead
    return machine_manifest(machine)


def _load_emitter_plan(
    *,
    name: str,
    solution: Path,
    ddg: Path,
    baseline: Path,
    regs_per_warp: int,
    warp_fixed_overhead: int,
    require_tmem_evidence: bool,
):
    problem, plan = load_schedule_context(
        solution,
        ddg,
        baseline,
        model_profile=EMITTER_MODEL_PROFILE,
    )
    if plan.normalization_u != NORMALIZATION_U:
        raise WorkflowError(
            f"{name}: normalization_u={plan.normalization_u}, "
            f"expected {NORMALIZATION_U}"
        )
    expected_machine = _expected_machine(regs_per_warp, warp_fixed_overhead)
    if plan.machine != expected_machine:
        raise WorkflowError(
            f"{name}: machine manifest is not the production configuration"
        )

    payload = json.loads(solution.read_text())
    attempts = payload.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise WorkflowError(f"{name}: missing search-attempt evidence")
    final_attempt = attempts[-1]
    if not isinstance(final_attempt, dict) or any(
        (
            final_attempt.get("stage") != "joint",
            final_attempt.get("result") != "sat",
            final_attempt.get("ii") != plan.ii,
            final_attempt.get("L") != plan.length,
        )
    ):
        raise WorkflowError(
            f"{name}: final search attempt does not prove the "
            "emitter-constrained optimum"
        )
    if require_tmem_evidence:
        stats = payload.get("stats")
        evidence = (
            stats.get("tmem_object_count") if isinstance(stats, dict) else None,
            stats.get("peak_tmem_cols") if isinstance(stats, dict) else None,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in evidence
        ):
            raise WorkflowError(
                f"{name}: missing positive TMEM-object/peak-liveness evidence"
            )
    return problem, plan


def _emit_graph(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    if _lexists(output):
        raise WorkflowError(f"refusing to overwrite existing target: {output}")
    if not output.parent.is_dir():
        raise WorkflowError(f"output parent does not exist: {output.parent}")

    solution = Path(args.solution).resolve()
    ddg = Path(args.ddg).resolve()
    baseline = Path(args.baseline).resolve()
    problem, plan = _load_emitter_plan(
        name=args.case_name,
        solution=solution,
        ddg=ddg,
        baseline=baseline,
        regs_per_warp=args.expected_regs_per_warp,
        warp_fixed_overhead=args.expected_warp_fixed_overhead,
        require_tmem_evidence=args.require_tmem_evidence,
    )
    machine = plan.machine
    try:
        rewrite_schedule_graph(
            baseline,
            output,
            ii=plan.ii,
            cycles=plan.cycles,
            warp=plan.warp,
            length=plan.length,
            group_widths=plan.group_widths,
            lane_masks=plan.lane_masks,
            warp_fixed_overhead=machine["warp_fixed_overhead"],
            smem_fixed_overhead=machine["smem_fixed_overhead"],
            tmem_fixed_cols=machine["tmem_fixed_cols"],
            dependence_edges=plan.edges,
            normalized_timing=normalized_timing_model(problem),
        )
        problems = validate(output)
        if problems:
            raise WorkflowError(
                f"{args.case_name}: graph validation failed: " + "; ".join(problems)
            )
    except Exception:
        if _lexists(output):
            output.unlink()
        raise

    print(
        json.dumps(
            {
                "case": args.case_name,
                "solution_sha256": plan.solution_sha256,
                "ii": plan.ii,
                "length": plan.length,
                "graph": str(output),
                "graph_sha256": _sha256(output),
                "validation": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


def _write_command(path: Path, command: list[str], cwd: Path) -> None:
    with path.open("x") as stream:
        stream.write(f"cd {shlex.quote(str(cwd))}\n")
        stream.write("env -u LD_LIBRARY_PATH ")
        stream.write(shlex.join(command))
        stream.write("\n")


def _run_logged(
    command: list[str], command_path: Path, log_path: Path, cwd: Path
) -> None:
    _write_command(command_path, command, cwd)
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    with log_path.open("x") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise WorkflowError(
            f"command returned {completed.returncode}; see {command_path} "
            f"and {log_path}"
        )


def _publish_exclusive(source: Path, target: Path) -> None:
    created = False
    try:
        with target.open("xb") as dst:
            created = True
            with source.open("rb") as src:
                shutil.copyfileobj(src, dst)
    except FileExistsError as error:
        raise WorkflowError(
            f"refusing to overwrite existing target: {target}"
        ) from error
    except Exception:
        if created and _lexists(target):
            target.unlink()
        raise
    try:
        target.chmod(0o644)
    except Exception:
        if created and _lexists(target):
            target.unlink()
        raise


def _manifest_case(spec: CaseSpec, staged: Path) -> dict:
    return {
        "solution": str(spec.solution),
        "solution_sha256": _sha256(spec.solution),
        "ddg": str(spec.ddg),
        "ddg_sha256": _sha256(spec.ddg),
        "baseline_graph": str(spec.baseline),
        "baseline_graph_sha256": _sha256(spec.baseline),
        "staged_graph": str(staged / spec.graph_output.name),
        "graph_output": str(spec.graph_output),
        "graph_sha256": _sha256(staged / spec.graph_output.name),
        "staged_kernel": str(staged / spec.kernel_output.name),
        "kernel_output": str(spec.kernel_output),
        "kernel_sha256": _sha256(staged / spec.kernel_output.name),
    }


def _write_manifest(record_dir: Path, payload: dict) -> None:
    path = record_dir / "manifest.json"
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _preflight(cases: tuple[CaseSpec, ...], record_dir: Path) -> None:
    if _lexists(record_dir):
        raise WorkflowError(
            f"refusing to reuse existing record directory: {record_dir}"
        )
    if not record_dir.parent.is_dir():
        raise WorkflowError(
            f"record-directory parent does not exist: {record_dir.parent}"
        )
    for spec in cases:
        for required in (spec.solution, spec.ddg, spec.baseline):
            if not required.is_file():
                raise WorkflowError(f"required input does not exist: {required}")
        for target in (spec.graph_output, spec.kernel_output):
            if _lexists(target):
                raise WorkflowError(
                    f"refusing to overwrite existing target: {target}"
                )


def _run_workflow(args: argparse.Namespace) -> int:
    solution_dir = Path(args.solution_dir).resolve()
    record_dir = (
        Path(args.record_dir).resolve()
        if args.record_dir
        else solution_dir / "bench_emit_emitter_v8"
    )
    python = Path(args.python).resolve()
    cases = _cases(solution_dir)
    _preflight(cases, record_dir)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise WorkflowError(f"Python interpreter is not executable: {python}")
    emission_sources_sha256, emission_sources = _emission_source_manifest()

    # Validate every emitter-constrained artifact before leaving any output.
    for spec in cases:
        _load_emitter_plan(
            name=spec.name,
            solution=spec.solution,
            ddg=spec.ddg,
            baseline=spec.baseline,
            regs_per_warp=spec.regs_per_warp,
            warp_fixed_overhead=spec.warp_fixed_overhead,
            require_tmem_evidence=spec.require_tmem_evidence,
        )

    record_dir.mkdir(mode=0o755)
    staged_root = record_dir / "staged"
    staged_root.mkdir(mode=0o755)
    published: list[Path] = []
    try:
        for spec in cases:
            staged = staged_root / spec.name
            staged.mkdir(mode=0o755)
            staged_graph = staged / spec.graph_output.name
            staged_kernel = staged / spec.kernel_output.name
            graph_command = [
                str(python),
                str(Path(__file__).resolve()),
                "emit-graph",
                "--case-name",
                spec.name,
                "--solution",
                str(spec.solution),
                "--ddg",
                str(spec.ddg),
                "--baseline-graph",
                str(spec.baseline),
                "--output",
                str(staged_graph),
                "--expected-regs-per-warp",
                str(spec.regs_per_warp),
                "--expected-warp-fixed-overhead",
                str(spec.warp_fixed_overhead),
            ]
            if spec.require_tmem_evidence:
                graph_command.append("--require-tmem-evidence")
            _run_logged(
                graph_command,
                record_dir / f"{spec.name}.graph.command",
                record_dir / f"{spec.name}.graph.log",
                ROOT,
            )

            emit_command = [
                str(python),
                "-m",
                "sched2tlx",
                str(staged_graph),
                "-o",
                str(staged_kernel),
            ]
            _run_logged(
                emit_command,
                record_dir / f"{spec.name}.sched2tlx.command",
                record_dir / f"{spec.name}.sched2tlx.log",
                SCHED2TLX,
            )
            if not staged_kernel.is_file() or staged_kernel.stat().st_size == 0:
                raise WorkflowError(
                    f"{spec.name}: sched2tlx did not produce a non-empty kernel"
                )

        for spec in cases:
            staged = staged_root / spec.name
            for source, target in (
                (staged / spec.graph_output.name, spec.graph_output),
                (staged / spec.kernel_output.name, spec.kernel_output),
            ):
                _publish_exclusive(source, target)
                published.append(target)

        _write_manifest(
            record_dir,
            {
                "schema_version": WORKFLOW_SCHEMA,
                "status": "success",
                "python": str(python),
                "workflow": str(Path(__file__).resolve()),
                "workflow_sha256": _sha256(Path(__file__).resolve()),
                "emission_sources_sha256": emission_sources_sha256,
                "emission_sources": emission_sources,
                "selection_constraints": {
                    "classification": "non-paper-emitter-compatibility",
                    "prefix_lane_masks": True,
                    "full_group_lane_masks": True,
                },
                "cases": {
                    spec.name: _manifest_case(
                        spec, staged_root / spec.name
                    )
                    for spec in cases
                },
            },
        )
    except Exception as error:
        for target in reversed(published):
            if _lexists(target):
                target.unlink()
        if not _lexists(record_dir / "manifest.json"):
            _write_manifest(
                record_dir,
                {
                    "schema_version": WORKFLOW_SCHEMA,
                    "status": "failed",
                    "error": str(error),
                },
            )
        raise

    print(f"published v8 benchmark kernels; record: {record_dir}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and materialize the three emitter-constrained v8 JOS "
            "benchmark kernels"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the append-never three-case workflow")
    run.add_argument(
        "--solution-dir",
        default=str(ROOT / "solutions"),
        help="directory containing the three emitter-mode v8 solution JSON files",
    )
    run.add_argument(
        "--record-dir",
        help=(
            "new command/log directory "
            "(default: SOLUTION_DIR/bench_emit_emitter_v8)"
        ),
    )
    run.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for graph validation and sched2tlx",
    )
    run.set_defaults(handler=_run_workflow)

    graph = commands.add_parser(
        "emit-graph", help="strict single-case graph emitter used by the workflow"
    )
    graph.add_argument("--case-name", required=True)
    graph.add_argument("--solution", required=True)
    graph.add_argument("--ddg", required=True)
    graph.add_argument("--baseline-graph", required=True, dest="baseline")
    graph.add_argument("--output", required=True)
    graph.add_argument("--expected-regs-per-warp", type=int, required=True)
    graph.add_argument("--expected-warp-fixed-overhead", type=int, required=True)
    graph.add_argument("--require-tmem-evidence", action="store_true")
    graph.set_defaults(handler=_emit_graph)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, ScheduleArtifactError, WorkflowError) as error:
        print(f"emit_bench_kernels: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
