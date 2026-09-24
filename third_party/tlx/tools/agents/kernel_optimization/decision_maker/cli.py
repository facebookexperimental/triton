from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

from .artifacts import load_prior_run_evidence
from .harness import HarnessExecutionError, SubprocessHarness
from ..contracts import (
    AutoCommitResult,
    DecisionStatus,
    ExperimentKind,
    ExperimentSummary,
    InputCase,
    KernelOptimizationRequest,
    KernelOptimizationResult,
    KernelTarget,
    OptimizationBudget,
    PerformanceSummary,
    to_json_value,
)
from ..optimizer.agent import CodexCandidateProvider, MockLLMProvider
from .orchestrator import KernelOptimizer
from .policy import passes_protected_cases
from .targets import expected_cuda_major, resolve_target_paths
from .vcs import (
    AutoCommitSession,
    commit_promotion,
    commit_rollback,
    failed_auto_commit,
    prepare_auto_commit,
)


def _load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a Triton or TLX kernel with a deterministic harness."
    )
    parser.add_argument(
        "--kernel",
        type=Path,
        default=None,
        help="kernel source for standalone authoring; inferred when --op and --arch are given",
    )
    parser.add_argument(
        "--task",
        choices=("authoring", "tuning"),
        default=None,
        help=(
            "top-level TLX-agent task; inferred as tuning when --op/--suite "
            "are provided, otherwise authoring"
        ),
    )
    parser.add_argument(
        "--objective",
        choices=("kernel", "heuristic-policy"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--op", default=None, help="tlx.ops name for an inferred production kernel")
    parser.add_argument("--suite", default=None, help="production shape suite for tuning")
    parser.add_argument("--search-rounds", type=int, default=2)
    parser.add_argument("--heuristic-rounds", type=int, default=5)
    parser.add_argument(
        "--reference-kernel",
        type=Path,
        default=None,
        help=(
            "Optional reference kernel source used as correctness oracle "
            "(harness verify can compare candidate vs reference)."
        ),
    )
    parser.add_argument("--harness", type=Path, default=None)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument(
        "--target-name",
        default=None,
        help="Registered target operation name; defaults to the kernel filename stem.",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help=(
            "Target architecture registered under decision_maker/targets "
            "(e.g. blackwell, hopper, host). Defaults to the first match."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="artifact directory; tuning defaults to /tmp/tlx-agent-<arch>-<op>-<suite>",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="physical GPU index, or auto to select the least-used matching GPU (default)",
    )
    parser.add_argument(
        "--govern",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply clock/power and GPU-local NUMA controls (default: enabled)",
    )
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


def _resolve_harness_paths(
    kernel: Path,
    harness: Path | None,
    cases: Path | None,
    target: Path | None,
    arch: str | None,
    target_name: str | None = None,
) -> tuple[Path, Path, Path]:
    return resolve_target_paths(kernel, harness, cases, target, arch, target_name)


def _expected_cuda_major(arch: str) -> int | None:
    return expected_cuda_major(arch)


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


def _probe_rocm_architecture(device: str | None) -> str:
    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "HIP target validation requires torch to be importable"
        ) from error
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise SystemExit("HIP target selected, but no ROCm device is available")
    torch_device = torch.device(device or "cuda")
    if torch_device.type != "cuda":
        raise SystemExit(f"HIP target selected, but target device is {device!r}")
    device_index = torch_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    architecture = str(getattr(properties, "gcnArchName", ""))
    return architecture or str(torch.cuda.get_device_name(device_index))


def _validate_host_matches_target(
    target: KernelTarget,
    arch: str | None,
    capability_probe: Callable[
        [str | None], tuple[int, int]
    ] = _probe_cuda_compute_capability,
    rocm_arch_probe: Callable[[str | None], str] = _probe_rocm_architecture,
) -> None:
    previous_environment: dict[str, str | None] = {}
    try:
        for key, value in target.environment.items():
            previous_environment[key] = os.environ.get(key)
            os.environ[key] = value
        backend = target.backend.lower()
        if backend == "cuda":
            expected_major = _expected_cuda_major(arch or target.architecture)
            if expected_major is None:
                return
            actual_major, actual_minor = capability_probe(target.device)
            if actual_major != expected_major:
                expected = f"sm_{expected_major}x"
                actual = f"sm_{actual_major}{actual_minor}"
                raise SystemExit(
                    f"--arch {arch or target.architecture} expects {expected}, "
                    f"but {target.device or 'cuda'} is {actual}"
                )
        elif backend in {"amd", "hip", "rocm"}:
            expected_match = re.search(
                r"gfx[0-9a-f]+", (arch or target.architecture).lower()
            )
            if expected_match is None:
                return
            expected = expected_match.group(0)
            actual = rocm_arch_probe(target.device)
            actual_match = re.search(r"gfx[0-9a-f]+", actual.lower())
            if actual_match is None or expected != actual_match.group(0):
                raise SystemExit(
                    f"--arch {arch or target.architecture} expects {expected}, "
                    f"but {target.device or 'cuda'} is {actual}"
                )
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
        subject = "Revert TLX agent promotions after failed final revalidation"
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


def _result_exit_code(result: KernelOptimizationResult) -> int:
    if result.stopping_reason in {"promotion_commit_failed", "rollback_commit_failed"}:
        return 3
    if result.decision is not None and result.decision.status is DecisionStatus.NEEDS_HUMAN:
        return 4
    return 0 if result.success else 2


def _repository_root() -> Path:
    repository = next(
        (parent for parent in Path(__file__).resolve().parents
         if parent.joinpath("third_party", "tlx", "ops").is_dir()),
        None,
    )
    if repository is None:
        raise SystemExit("could not locate the Triton repository")
    return repository


def _resolve_task(args: argparse.Namespace) -> str:
    if args.task is not None and args.objective is not None:
        raise SystemExit("use --task; do not combine it with deprecated --objective")
    if args.objective is not None:
        task = {"kernel": "authoring", "heuristic-policy": "tuning"}[args.objective]
        print(
            f"warning: --objective {args.objective} is deprecated; use --task {task}",
            file=sys.stderr,
        )
        return task
    if args.task is not None:
        return args.task
    return "tuning" if args.op is not None or args.suite is not None else "authoring"


def _run_task(args: argparse.Namespace, *, environment_ready: bool = False) -> int:
    task = _resolve_task(args)
    args.task = task
    args.objective = None
    if task == "tuning":
        if not args.op or not args.arch or not args.suite:
            raise SystemExit("--task tuning requires --op, --arch, and --suite")
        if args.kernel is not None:
            raise SystemExit("--kernel is inferred for tuning; remove the redundant argument")
    tuning_epilogue = task == "authoring" and (args.op is not None or args.suite is not None)
    if tuning_epilogue and (not args.op or not args.arch or not args.suite):
        raise SystemExit("authoring's tuning epilogue requires --op, --arch, and --suite")
    if args.output_dir is None:
        if task != "tuning" or not args.op or not args.arch or not args.suite:
            raise SystemExit("--output-dir is required for --task authoring")
        args.output_dir = Path(f"/tmp/tlx-agent-{args.arch}-{args.op}-{args.suite}")

    repository = _repository_root() if task == "tuning" or tuning_epilogue else None
    if repository is not None and not environment_ready:
        from .benchmark_environment import governed_benchmark_device

        with governed_benchmark_device(
            repository,
            args.arch,
            args.device,
            govern=args.govern,
        ):
            return _run_task(args, environment_ready=True)

    budget = _budget_from_args(args)
    provider = (
        MockLLMProvider()
        if args.provider == "mock"
        else CodexCandidateProvider(
            model=args.model, timeout_seconds=budget.max_candidate_seconds
        )
    )
    if task == "tuning":
        from .tuning import run_tuning

        assert repository is not None
        exit_code, result = run_tuning(
            repository=repository,
            op=args.op,
            arch=args.arch,
            suite=args.suite,
            output_dir=args.output_dir,
            provider=provider,
            budget=budget,
            search_rounds=args.search_rounds,
            heuristic_rounds=args.heuristic_rounds,
            commit=args.commit_winner,
            commit_message=args.commit_message,
            vcs=args.vcs,
        )
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return exit_code

    if tuning_epilogue:
        from .tuning import _tuning_target, infer_kernel_path, production_cases

        assert repository is not None
        production_kernel = infer_kernel_path(repository, args.op, args.arch).resolve()
        if args.kernel is None:
            args.kernel = production_kernel
        elif args.kernel.resolve() != production_kernel:
            raise SystemExit(
                f"authoring's tuning epilogue targets {production_kernel}, but --kernel is {args.kernel.resolve()}")
    if args.kernel is None:
        raise SystemExit("--kernel is required for standalone authoring")
    if tuning_epilogue and args.harness is None and args.cases is None and args.target is None:
        harness_path = Path(__file__).with_name("tuning_harnesses") / f"{args.op}.py"
        cases = production_cases(args.op, args.suite)
        target = _tuning_target(args.op, args.arch)
    else:
        harness_path, cases_path, target_path = _resolve_harness_paths(
            args.kernel,
            args.harness,
            args.cases,
            args.target,
            args.arch,
            args.target_name,
        )
        case_payloads = _load_json(cases_path)
        target_payload = _load_json(target_path)
        optimization_skills = target_payload.get("optimization_skills", [])
        if not isinstance(optimization_skills, list):
            raise ValueError("target optimization_skills must be a list")
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
            optimization_guidance=str(target_payload.get("optimization_guidance", "")),
            optimization_skills=tuple(optimization_skills),
            evaluation_policy=target_payload.get("evaluation_policy", {}),
            supported_experiment_kinds=tuple(
                ExperimentKind(value)
                for value in target_payload.get(
                    "supported_experiment_kinds",
                    (ExperimentKind.PROMOTABLE.value, ExperimentKind.HUMAN_REVIEW.value),
                )
            ),
        )
    _validate_host_matches_target(target, args.arch)
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
    exit_code = _result_exit_code(result)
    if result.auto_commit is not None:
        args.output_dir.joinpath("auto_commit.json").write_text(
            json.dumps(to_json_value(result.auto_commit), indent=2, sort_keys=True) + "\n"
        )
    if exit_code != 0 or not tuning_epilogue:
        print(json.dumps(to_json_value(result), indent=2, sort_keys=True))
        return exit_code

    from .tuning import run_tuning

    assert repository is not None
    tuning_exit_code, tuning_result = run_tuning(
        repository=repository,
        op=args.op,
        arch=args.arch,
        suite=args.suite,
        output_dir=args.output_dir / "tuning",
        provider=provider,
        budget=budget,
        search_rounds=args.search_rounds,
        heuristic_rounds=args.heuristic_rounds,
        commit=args.commit_winner,
        commit_message=None,
        vcs=args.vcs,
        initial_source=result.best_kernel,
    )
    task_result = {
        "task": "authoring",
        "authoring": to_json_value(result),
        "epilogue": tuning_result,
    }
    args.output_dir.joinpath("task_result.json").write_text(
        json.dumps(task_result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(task_result, indent=2, sort_keys=True))
    return tuning_exit_code


def main() -> int:
    return _run_task(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
