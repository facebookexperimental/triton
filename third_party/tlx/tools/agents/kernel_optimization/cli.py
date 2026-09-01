from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .harness import HarnessExecutionError, SubprocessHarness
from .models import (
    InputCase,
    KernelOptimizationRequest,
    KernelOptimizationResult,
    KernelTarget,
    OptimizationBudget,
    passes_protected_cases,
    to_json_value,
)
from .optimizer import KernelOptimizer
from .providers import CodexCandidateProvider, MockLLMProvider
from .vcs import commit_winner, failed_auto_commit, prepare_auto_commit


def _load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a Triton or TLX kernel with a deterministic harness."
    )
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--reference-kernel", type=Path, default=None, help="Optional reference kernel source used as correctness oracle (harness verify can compare candidate vs reference).")
    parser.add_argument("--harness", type=Path, default=None)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument(
        "--arch",
        default=None,
        help="Target arch under harnesses/<arch>/targets/<kernel> (e.g. blackwell, hopper, host). Defaults to first available.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=2)
    parser.add_argument("--max-candidate-seconds", type=float, default=600.0)
    parser.add_argument("--max-total-seconds", type=float, default=3600.0)
    parser.add_argument("--min-speedup", type=float, default=1.01)
    parser.add_argument("--max-cv", type=float, default=0.10)
    parser.add_argument("--benchmark-repetitions", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--commit-winner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Commit a successfully revalidated winner to the kernel's repository "
            "(default: enabled; use --no-commit-winner to disable)."
        ),
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit subject; the TLX agent attribution is always added to the body.",
    )
    parser.add_argument(
        "--vcs",
        choices=["auto", "git", "hg"],
        default="auto",
        help="Version control for --commit-winner; auto detects from --kernel.",
    )
    parser.add_argument(
        "--provider",
        choices=["codex", "mock"],
        default="codex",
        help="Candidate provider: codex (default) or mock (deterministic CI stub).",
    )
    parser.add_argument(
        "--harness-mode",
        choices=["subprocess", "standalone"],
        default="subprocess",
        help="subprocess (default, isolated) or standalone (in-process, for debugging).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect harness profile() for baseline and candidates (default: profile is always collected).",
    )
    parser.add_argument(
        "--diagnostic-proton-intra-kernel",
        action="store_true",
        default=False,
        help=(
            "Collect diagnostic-only per-warp proton_intra_kernel traces for the "
            "baseline and final winner only."
        ),
    )
    parser.add_argument(
        "--budget",
        type=Path,
        default=None,
        help="Optional JSON file that overrides --max-* / --min-speedup / --max-cv flags.",
    )
    return parser.parse_args(arguments)


def _budget_from_args(args: argparse.Namespace) -> OptimizationBudget:
    if args.budget is not None:
        payload = _load_json(args.budget)
        return OptimizationBudget(
            max_rounds=int(payload.get("max_rounds", args.max_rounds)),
            candidates_per_round=int(
                payload.get("candidates_per_round", args.candidates_per_round)
            ),
            max_candidate_seconds=float(
                payload.get("max_candidate_seconds", args.max_candidate_seconds)
            ),
            max_total_seconds=float(
                payload.get("max_total_seconds", args.max_total_seconds)
            ),
            min_speedup=float(payload.get("min_speedup", args.min_speedup)),
            max_cv=float(payload.get("max_cv", args.max_cv)),
            benchmark_repetitions=int(
                payload.get("benchmark_repetitions", args.benchmark_repetitions)
            ),
        )
    return OptimizationBudget(
        max_rounds=args.max_rounds,
        candidates_per_round=args.candidates_per_round,
        max_candidate_seconds=args.max_candidate_seconds,
        max_total_seconds=args.max_total_seconds,
        min_speedup=args.min_speedup,
        max_cv=args.max_cv,
        benchmark_repetitions=args.benchmark_repetitions,
    )


def _resolve_harness_paths(kernel: Path, harness: Path | None, cases: Path | None, target: Path | None, arch: str | None) -> tuple[Path, Path, Path]:
    # Kernel-only invocation: infer harness/cases/target from
    # harnesses/<arch>/targets/<stem>/
    # e.g. --kernel gemm.py -> harnesses/blackwell/targets/gemm/{harness.py,cases.json,target.json}
    # Harness must be colocated with cases (target-specific), so both are resolved together.
    base = Path(__file__).resolve().parent / "harnesses"
    stem = kernel.stem  # gemm, vector_add, etc.
    if base.exists() and (harness is None or cases is None or target is None):
        archs = sorted(
            p.name
            for p in base.iterdir()
            if p.is_dir() and (p / "targets" / stem).is_dir()
        )
        chosen = arch or (archs[0] if archs else None)
        if chosen is None:
            chosen = "blackwell" if harness is None else None
        if chosen is not None:
            tdir = base / chosen / "targets" / stem
            if harness is None and (tdir / "harness.py").exists():
                harness = tdir / "harness.py"
            if cases is None and (tdir / "cases.json").exists():
                cases = tdir / "cases.json"
            if target is None and (tdir / "target.json").exists():
                target = tdir / "target.json"
    if harness is None or cases is None or target is None:
        missing = [n for n, v in [("harness", harness), ("cases", cases), ("target", target)] if v is None]
        raise SystemExit(f"missing required {'/'.join(missing)}; pass them explicitly or use a kernel with harnesses/<arch>/targets/<name>/")
    return harness, cases, target


def _expected_cuda_major(arch: str) -> int | None:
    normalized = arch.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"hopper", "h100", "sm90", "sm_90"}:
        return 9
    if normalized in {"blackwell", "b200", "gb200", "sm100", "sm_100"}:
        return 10
    return None


def _probe_cuda_compute_capability(device: str | None) -> tuple[int, int]:
    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "CUDA target validation requires torch to be importable"
        ) from error
    if not torch.cuda.is_available():
        raise SystemExit("CUDA target selected, but no CUDA device is available")
    torch_device = torch.device(device or "cuda")
    if torch_device.type != "cuda":
        raise SystemExit(f"CUDA target selected, but target device is {device!r}")
    device_index = torch_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return torch.cuda.get_device_capability(device_index)


def _validate_host_matches_target(
    target: KernelTarget,
    arch: str | None,
    capability_probe: Callable[[str | None], tuple[int, int]] = _probe_cuda_compute_capability,
) -> None:
    if target.backend != "cuda":
        return
    expected_major = _expected_cuda_major(arch or target.architecture)
    if expected_major is None:
        return
    previous_environment: dict[str, str | None] = {}
    try:
        for key, value in target.environment.items():
            previous_environment[key] = os.environ.get(key)
            os.environ[key] = value
        actual_major, actual_minor = capability_probe(target.device)
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if actual_major != expected_major:
        expected = f"sm_{expected_major}x"
        actual = f"sm_{actual_major}{actual_minor}"
        raise SystemExit(
            f"--arch {arch or target.architecture} expects {expected}, "
            f"but {target.device or 'cuda'} is {actual}"
        )


def _commit_body(result: KernelOptimizationResult) -> str:
    baseline_by_id = {case.case_id: case for case in result.baseline.cases}
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for winner in result.final.cases:
        baseline = baseline_by_id.get(winner.case_id)
        baseline_timing = baseline.timing if baseline else None
        winner_timing = winner.timing
        if baseline_timing is not None and winner_timing is not None:
            speedup = baseline_timing.median_us / winner_timing.median_us
            baseline_us = f"{baseline_timing.median_us:.2f}"
            winner_us = f"{winner_timing.median_us:.2f}"
            speedup_text = f"{speedup:.4f}x"
            baseline_cv = f"{100.0 * baseline_timing.coefficient_of_variation:.2f}%"
            winner_cv = f"{100.0 * winner_timing.coefficient_of_variation:.2f}%"
        else:
            baseline_us = winner_us = speedup_text = baseline_cv = winner_cv = "n/a"
        rows.append(
            (
                winner.case_id,
                baseline_us,
                winner_us,
                speedup_text,
                baseline_cv,
                winner_cv,
                "pass" if winner.verification.passed else "fail",
            )
        )

    headers = ("Case", "Baseline us", "Winner us", "Speedup", "Base CV", "Winner CV", "Correct")
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()

    table = [format_row(headers), format_row(tuple("-" * width for width in widths))]
    table.extend(format_row(row) for row in rows)
    validation = (
        "Performance:\n"
        f"Final revalidation for {result.winner_experiment_id}:\n"
        + "\n".join(table)
        + f"\nWeighted aggregate speedup: {result.final.aggregate_speedup:.4f}x."
    )
    summary = result.winner_commit_summary.strip()
    return f"{summary}\n\n{validation}" if summary else validation


def _report_commit(commit_result: object) -> None:
    result = commit_result
    parts = [
        "[tlx-agent] commit",
        f"status={'committed' if result.success else 'failed'}",
        f"vcs={result.vcs or 'unknown'}",
    ]
    if result.commit_revision:
        parts.append(f"id={result.commit_revision}")
    if result.repo_root:
        parts.append(f"repo={result.repo_root}")
    if result.target_relpath:
        parts.append(f"file={result.target_relpath}")
    if result.subject:
        parts.append(f"subject={json.dumps(result.subject)}")
    parts.append(f"attribution={json.dumps(result.attribution)}")
    if result.diagnostics:
        parts.append(f"diagnostics={json.dumps(result.diagnostics)}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def main() -> int:
    args = _parse_args()
    harness_path, cases_path, target_path = _resolve_harness_paths(args.kernel, args.harness, args.cases, args.target, args.arch)
    case_payloads = _load_json(cases_path)
    target_payload = _load_json(target_path)
    cases = tuple(
        InputCase(
            case_id=str(case["case_id"]),
            parameters=case.get("parameters", {}),
            weight=float(case.get("weight", 1.0)),
            protected=bool(case.get("protected", True)),
        )
        for case in case_payloads
    )
    target = KernelTarget(
        backend=str(target_payload["backend"]),
        architecture=str(target_payload["architecture"]),
        device=target_payload.get("device"),
        environment=target_payload.get("environment", {}),
        optimization_guidance=str(target_payload.get("optimization_guidance", "")),
    )
    _validate_host_matches_target(target, args.arch)
    budget = _budget_from_args(args)
    # CLI always evaluates via the optimizer's SubprocessHarness. The legacy
    # --harness-mode flag is kept for compatibility and documented as such;
    # standalone evaluation is available programmatically via StandaloneHarness.
    if args.harness_mode != "subprocess":
        import sys as _sys

        print(
            "warning: --harness-mode standalone is only available via the Python API "
            "(StandaloneHarness); CLI still uses subprocess isolation.",
            file=_sys.stderr,
        )
    provider = (
        MockLLMProvider()
        if args.provider == "mock"
        else CodexCandidateProvider(
            model=args.model, timeout_seconds=budget.max_candidate_seconds
        )
    )
    kernel_path = args.kernel.resolve()
    kernel_source = kernel_path.read_text()
    fallback_commit_subject = f"Optimize {kernel_path.name} with TLX agent"
    commit_snapshot = None
    if args.commit_winner:
        try:
            commit_snapshot = prepare_auto_commit(kernel_path, kernel_source, args.vcs)
        except Exception as error:  # noqa: BLE001
            commit_result = failed_auto_commit(
                None, args.commit_message or fallback_commit_subject, error
            )
            _report_commit(commit_result)
            print(json.dumps(to_json_value(commit_result), indent=2, sort_keys=True))
            return 3
    reference_source = args.reference_kernel.read_text() if args.reference_kernel else None
    request = KernelOptimizationRequest(
        kernel_source=kernel_source,
        reference_kernel_source=reference_source,
        harness_path=harness_path,
        cases=cases,
        target=target,
        budget=budget,
        output_dir=args.output_dir,
        diagnostic_proton_intra_kernel=args.diagnostic_proton_intra_kernel,
    )
    result = KernelOptimizer(provider).optimize(request)
    exit_code = 0 if result.success else 2
    if args.commit_winner and result.success:
        assert commit_snapshot is not None
        commit_subject = (
            args.commit_message
            or result.winner_commit_title
            or fallback_commit_subject
        )

        def validate_committed_source(committed_source: str) -> None:
            harness = SubprocessHarness(harness_path, budget.max_candidate_seconds)
            validation = harness.evaluate(
                committed_source,
                cases,
                target,
                budget.benchmark_repetitions,
            )
            args.output_dir.joinpath("commit_revalidation.json").write_text(
                json.dumps(to_json_value(validation), indent=2, sort_keys=True) + "\n"
            )
            if not passes_protected_cases(validation, cases):
                raise HarnessExecutionError(
                    "merged commit source failed one or more protected correctness cases"
                )

        try:
            commit_result = commit_winner(
                commit_snapshot,
                result.best_kernel,
                commit_subject,
                body=_commit_body(result),
                validate_committed_source=validate_committed_source,
            )
        except Exception as error:  # noqa: BLE001
            commit_result = failed_auto_commit(commit_snapshot, commit_subject, error)
            exit_code = 3
        result = replace(result, auto_commit=commit_result)
        _report_commit(commit_result)
        args.output_dir.joinpath("auto_commit.json").write_text(
            json.dumps(to_json_value(commit_result), indent=2, sort_keys=True) + "\n"
        )
        args.output_dir.joinpath("result.json").write_text(
            json.dumps(to_json_value(result), indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(to_json_value(result), indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
