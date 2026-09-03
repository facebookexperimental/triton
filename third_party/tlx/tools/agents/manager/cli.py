from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from ..build_agent.harness import HarnessExecutionError, SubprocessHarness
from ..build_agent.vcs import (
    AutoCommitSession,
    commit_promotion,
    commit_rollback,
    failed_auto_commit,
    prepare_auto_commit,
)
from ..worker.artifacts import load_prior_run_evidence
from ..worker.providers import CodexCandidateProvider, MockLLMProvider
from .models import (
    AutoCommitResult,
    ExperimentSummary,
    InputCase,
    KernelOptimizationRequest,
    KernelOptimizationResult,
    KernelTarget,
    OptimizationBudget,
    PerformanceSummary,
    passes_protected_cases,
    to_json_value,
)
from .optimizer import KernelOptimizer


def _load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a Triton or TLX kernel with a deterministic harness.")
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument(
        "--reference-kernel", type=Path, default=None, help=
        "Optional reference kernel source used as correctness oracle (harness verify can compare candidate vs reference)."
    )
    parser.add_argument("--harness", type=Path, default=None)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument(
        "--arch",
        default=None,
        help=
        "Target arch under validator/targets/<arch>/<kernel> (e.g. blackwell, hopper, host). Defaults to first available.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prior-run",
        type=Path,
        default=None,
        help=(
            "Read-only path to a prior TLX Agent output directory or its "
            "experiments.json; imports evidence and source hashes without "
            "adopting the prior winner."
        ),
    )
    parser.add_argument(
        "--result-format",
        choices=["full", "summary", "none"],
        default="full",
        help=("Final stdout payload: full JSON (default), a compact JSON summary, or nothing. "
              "Full results are always persisted under --output-dir."),
    )
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
        help=("Commit a successfully revalidated winner to the kernel's repository "
              "(default: enabled; use --no-commit-winner to disable)."),
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
        help=("Collect diagnostic-only per-warp proton_intra_kernel traces for the "
              "baseline and final winner only."),
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
            candidates_per_round=int(payload.get("candidates_per_round", args.candidates_per_round)),
            max_candidate_seconds=float(payload.get("max_candidate_seconds", args.max_candidate_seconds)),
            max_total_seconds=float(payload.get("max_total_seconds", args.max_total_seconds)),
            min_speedup=float(payload.get("min_speedup", args.min_speedup)),
            max_cv=float(payload.get("max_cv", args.max_cv)),
            benchmark_repetitions=int(payload.get("benchmark_repetitions", args.benchmark_repetitions)),
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


def _resolve_harness_paths(kernel: Path, harness: Path | None, cases: Path | None, target: Path | None,
                           arch: str | None) -> tuple[Path, Path, Path]:
    # Kernel-only invocation: infer harness/cases/target from
    # validator/targets/<arch>/<stem>/
    # e.g. --kernel gemm.py -> validator/targets/blackwell/gemm/{harness.py,cases.json,target.json}
    # Harness must be colocated with cases (target-specific), so both are resolved together.
    base = Path(__file__).resolve().parents[1] / "validator" / "targets"
    stem = kernel.stem  # gemm, vector_add, etc.
    if base.exists() and (harness is None or cases is None or target is None):
        archs = sorted(p.name for p in base.iterdir() if p.is_dir() and (p / stem).is_dir())
        chosen = arch or (archs[0] if archs else None)
        if chosen is None:
            chosen = "blackwell" if harness is None else None
        if chosen is not None:
            tdir = base / chosen / stem
            if harness is None and (tdir / "harness.py").exists():
                harness = tdir / "harness.py"
            if cases is None and (tdir / "cases.json").exists():
                cases = tdir / "cases.json"
            if target is None and (tdir / "target.json").exists():
                target = tdir / "target.json"
    if harness is None or cases is None or target is None:
        missing = [n for n, v in [("harness", harness), ("cases", cases), ("target", target)] if v is None]
        raise SystemExit(
            f"missing required {'/'.join(missing)}; pass them explicitly or use a kernel with validator/targets/<arch>/<name>/"
        )
    return harness, cases, target


# TL strategy and curated knowledge live in the frozen bundle, not a global doc
# tree, so a run's prompt is reproducible from its recorded bundle hashes. Per
# `references/input-contract.md` the provider receives only their resolved text.
# Backstop against a runaway document, not a budget to fill -- the prompt also
# carries the preamble, reference kernel and profiles. ~2x the shipped gfx942
# bundle. Truncation is announced so a fragment is never read as complete.
MAX_GUIDANCE_BYTES = 16384


def _resolve_guidance(target_path: Path, inline: str) -> str:
    """Bundle knowledge files plus ``target.json``'s inline string, widest first."""
    target_dir = target_path.resolve().parent
    arch = target_dir.parent.name
    kernel = target_dir.name
    # validator/targets/<arch>/<kernel>/target.json -> agents/
    agents_dir = target_dir.parents[3]
    sections: list[str] = []
    for path in (
        agents_dir / "tl" / "strategies" / arch / f"{kernel}.md",
        agents_dir / "knowledge_keeper" / "knowledge" / arch / "architecture.md",
        agents_dir / "knowledge_keeper" / "knowledge" / arch / "targets" / f"{kernel}.md",
    ):
        if path.is_file():
            text = path.read_text().strip()
            if text:
                sections.append(f"--- {path.name} ({path.parent.name}) ---\n{text}")
    if inline.strip():
        sections.append(inline.strip())
    guidance = "\n\n".join(sections)
    if len(guidance) > MAX_GUIDANCE_BYTES:
        guidance = (guidance[:MAX_GUIDANCE_BYTES] +
                    f"\n\n[truncated at {MAX_GUIDANCE_BYTES} bytes; read the bundle files for the rest]")
    return guidance


def _expected_cuda_major(arch: str) -> int | None:
    normalized = arch.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"hopper", "h100", "sm90", "sm_90"}:
        return 9
    if normalized in {"blackwell", "b200", "gb200", "sm100", "sm_100"}:
        return 10
    return None


def _expected_gcn_arch(arch: str) -> str | None:
    normalized = arch.lower().replace("-", "_").replace(" ", "_")
    return {
        "gfx942": "gfx942",
        "mi300": "gfx942",
        "mi300x": "gfx942",
        "cdna3": "gfx942",
        "gfx950": "gfx950",
        "mi350": "gfx950",
        "mi355": "gfx950",
        "cdna4": "gfx950",
        "gfx1250": "gfx1250",
    }.get(normalized)


def _probe_gcn_arch(device: str | None) -> str:
    try:
        import torch
    except ImportError as error:
        raise SystemExit("HIP target validation requires torch to be importable") from error
    if not torch.cuda.is_available():
        raise SystemExit("HIP target selected, but no ROCm device is available")
    if not getattr(torch.version, "hip", None):
        raise SystemExit("HIP target selected, but this torch is not a ROCm build")
    torch_device = torch.device(device or "cuda")
    index = torch_device.index
    if index is None:
        index = torch.cuda.current_device()
    # e.g. "gfx942:sramecc+:xnack-" -- the feature suffix is not part of the target.
    return torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]


def _probe_cuda_compute_capability(device: str | None) -> tuple[int, int]:
    try:
        import torch
    except ImportError as error:
        raise SystemExit("CUDA target validation requires torch to be importable") from error
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
    gcn_probe: Callable[[str | None], str] = _probe_gcn_arch,
) -> None:
    if target.backend not in {"cuda", "hip"}:
        return
    requested = arch or target.architecture
    if target.backend == "cuda":
        expected: object | None = _expected_cuda_major(requested)
    else:
        expected = _expected_gcn_arch(requested)
    if expected is None:
        return
    previous_environment: dict[str, str | None] = {}
    try:
        for key, value in target.environment.items():
            previous_environment[key] = os.environ.get(key)
            os.environ[key] = value
        if target.backend == "cuda":
            actual_major, actual_minor = capability_probe(target.device)
            actual: str = f"sm_{actual_major}{actual_minor}"
            matched = actual_major == expected
            expected_label = f"sm_{expected}x"
        else:
            actual = gcn_probe(target.device)
            matched = actual == expected
            expected_label = str(expected)
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if not matched:
        raise SystemExit(f"--arch {requested} expects {expected_label}, "
                         f"but {target.device or target.backend} is {actual}")


def _performance_commit_body(
    baseline_summary: PerformanceSummary,
    comparison: PerformanceSummary,
    experiment_id: str,
    commit_summary: str,
    heading: str,
) -> str:
    baseline_by_id = {case.case_id: case for case in baseline_summary.cases}
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for winner in comparison.cases:
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
        f"{heading} for {experiment_id}:\n"
        + "\n".join(table)
        + f"\nWeighted aggregate speedup: {comparison.aggregate_speedup:.4f}x."
    )
    summary = commit_summary.strip()
    return f"{summary}\n\n{validation}" if summary else validation


def _commit_body(result: KernelOptimizationResult) -> str:
    return _performance_commit_body(
        result.baseline,
        result.final,
        result.winner_experiment_id,
        result.winner_commit_summary,
        "Final revalidation",
    )


class _PromotionAutoCommitter:
    def __init__(
        self,
        session: AutoCommitSession,
        harness_path: Path,
        cases: tuple[InputCase, ...],
        target: KernelTarget,
        budget: OptimizationBudget,
        output_dir: Path,
        fallback_subject: str,
        override_subject: str | None,
    ) -> None:
        self._session = session
        self._harness_path = harness_path
        self._cases = cases
        self._target = target
        self._budget = budget
        self._output_dir = output_dir
        self._fallback_subject = fallback_subject
        self._override_subject = override_subject

    def _validate(self, committed_source: str, experiment_id: str) -> None:
        validation = SubprocessHarness(
            self._harness_path, self._budget.max_candidate_seconds
        ).evaluate(
            committed_source,
            self._cases,
            self._target,
            self._budget.benchmark_repetitions,
        )
        self._output_dir.joinpath(
            "experiments", experiment_id, "commit_revalidation.json"
        ).write_text(json.dumps(to_json_value(validation), indent=2, sort_keys=True) + "\n")
        if not passes_protected_cases(validation, self._cases):
            raise HarnessExecutionError(
                "merged promotion source failed one or more protected correctness cases"
            )

    def commit_promotion(
        self,
        experiment: ExperimentSummary,
        source: str,
        baseline: PerformanceSummary,
        performance: PerformanceSummary,
    ) -> AutoCommitResult:
        subject = self._override_subject or experiment.commit_title or self._fallback_subject
        try:
            result = commit_promotion(
                self._session,
                source,
                subject,
                _performance_commit_body(
                    baseline,
                    performance,
                    experiment.experiment_id,
                    experiment.commit_summary,
                    "Promotion evaluation",
                ),
                validate_committed_source=lambda committed: self._validate(
                    committed, experiment.experiment_id
                ),
            )
        except Exception as error:  # noqa: BLE001
            result = failed_auto_commit(self._session.snapshot, subject, error)
        _report_commit(result)
        self._output_dir.joinpath("promotion_commits.json").write_text(
            json.dumps(
                to_json_value(tuple(self._session.promotion_commits)),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return result

    def rollback_to_baseline(self, diagnostics: str) -> AutoCommitResult:
        subject = f"Revert TLX agent promotions after failed final revalidation"
        try:
            result = commit_rollback(self._session, subject, diagnostics)
        except Exception as error:  # noqa: BLE001
            result = failed_auto_commit(self._session.snapshot, subject, error)
        _report_commit(result)
        self._output_dir.joinpath("rollback_commit.json").write_text(
            json.dumps(to_json_value(result), indent=2, sort_keys=True) + "\n"
        )
        return result


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


def _print_result(result: Any, output_dir: Path, result_format: str) -> None:
    if result_format == "none":
        return
    if result_format == "full":
        print(json.dumps(to_json_value(result), indent=2, sort_keys=True))
        return
    summary = {
        "artifacts_dir": str(output_dir),
        "result_json": str(output_dir / "result.json"),
        "stopping_reason": result.stopping_reason,
        "success": result.success,
        "final_speedup": result.final.aggregate_speedup,
        "experiments": len(result.experiments),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    args = _parse_args()
    harness_path, cases_path, target_path = _resolve_harness_paths(args.kernel, args.harness, args.cases, args.target,
                                                                   args.arch)
    case_payloads = _load_json(cases_path)
    target_payload = _load_json(target_path)
    cases = tuple(
        InputCase(
            case_id=str(case["case_id"]),
            parameters=case.get("parameters", {}),
            weight=float(case.get("weight", 1.0)),
            protected=bool(case.get("protected", True)),
        ) for case in case_payloads)
    target = KernelTarget(
        backend=str(target_payload["backend"]),
        architecture=str(target_payload["architecture"]),
        device=target_payload.get("device"),
        environment=target_payload.get("environment", {}),
        optimization_guidance=_resolve_guidance(target_path, str(target_payload.get("optimization_guidance", ""))),
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
    provider = (MockLLMProvider() if args.provider == "mock" else CodexCandidateProvider(
        model=args.model, timeout_seconds=budget.max_candidate_seconds))
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
    prior_run_evidence = None
    if args.prior_run is not None:
        try:
            prior_run_evidence = load_prior_run_evidence(args.prior_run)
        except ValueError as error:
            raise SystemExit(f"--prior-run is invalid: {error}") from error
        print(
            "[tlx-agent] prior-run "
            f"path={json.dumps(str(prior_run_evidence.run_path))} "
            f"experiments={len(prior_run_evidence.experiments)} "
            f"source_hashes={len(prior_run_evidence.source_hashes)} "
            f"warnings={len(prior_run_evidence.warnings)}",
            file=sys.stderr,
            flush=True,
        )
        for warning in prior_run_evidence.warnings:
            print(
                f"[tlx-agent] prior-run warning={json.dumps(warning)}",
                file=sys.stderr,
                flush=True,
            )
    request = KernelOptimizationRequest(
        kernel_source=kernel_source,
        reference_kernel_source=reference_source,
        harness_path=harness_path,
        cases=cases,
        target=target,
        budget=budget,
        output_dir=args.output_dir,
        diagnostic_proton_intra_kernel=args.diagnostic_proton_intra_kernel,
        prior_run_evidence=prior_run_evidence,
    )
    promotion_committer = None
    if commit_snapshot is not None:
        promotion_committer = _PromotionAutoCommitter(
            AutoCommitSession.create(commit_snapshot),
            harness_path,
            cases,
            target,
            budget,
            args.output_dir,
            fallback_commit_subject,
            args.commit_message,
        )
    result = KernelOptimizer(provider).optimize(request, promotion_committer)
    exit_code = 0 if result.success else 2
    if result.auto_commit is not None:
        args.output_dir.joinpath("auto_commit.json").write_text(
            json.dumps(to_json_value(result.auto_commit), indent=2, sort_keys=True) + "\n"
        )
    if result.stopping_reason in {"promotion_commit_failed", "rollback_commit_failed"}:
        exit_code = 3
    _print_result(result, args.output_dir, args.result_format)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
