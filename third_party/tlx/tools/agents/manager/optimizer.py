from __future__ import annotations

import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from ..build_agent.harness import HarnessExecutionError, SubprocessHarness
from ..profiler.profiling import (
    ProfileRequest,
    compact_profile_summary,
    extract_ncu_duration_us,
    ncu_regression_diagnostic,
)
from ..worker.artifacts import ArtifactStore, CandidateArtifactPaths
from ..worker.providers import CandidateContext, CandidateProvider, CodexCandidateProvider
from ..worker.source import source_digest
from .models import (
    AutoCommitResult,
    ExperimentSummary,
    InputCase,
    KernelOptimizationRequest,
    KernelOptimizationResult,
    PerformanceSummary,
    is_promotable,
    passes_protected_cases,
    per_case_speedups,
    weighted_geometric_speedup,
)

_PROFILE_TOOLS = ("proton_launch", "native_profiler")
_DIAGNOSTIC_PROFILE_TOOLS = ("proton_intra_kernel",)
_DIAGNOSTIC_PROFILE_KEY = "diagnostic_proton_intra_kernel"
_NEAR_THRESHOLD_WINDOW = 0.01


class PromotionCommitter(Protocol):
    def commit_promotion(
        self,
        experiment: ExperimentSummary,
        source: str,
        baseline: PerformanceSummary,
        performance: PerformanceSummary,
    ) -> AutoCommitResult: ...

    def rollback_to_baseline(self, diagnostics: str) -> AutoCommitResult: ...


def _profile_request(
    artifacts_dir: Path,
    experiment_id: str,
    *,
    level: str,
    reason: str,
) -> ProfileRequest:
    return ProfileRequest(
        level=level,
        tools=_PROFILE_TOOLS,
        experiment_id=experiment_id,
        artifacts_dir=artifacts_dir / "experiments" / experiment_id / "profile_artifacts",
        reason=reason,
    )


def _diagnostic_profile_request(
    artifacts_dir: Path,
    experiment_id: str,
    *,
    reason: str,
) -> ProfileRequest:
    return ProfileRequest(
        level="deep",
        tools=_DIAGNOSTIC_PROFILE_TOOLS,
        experiment_id=experiment_id,
        artifacts_dir=artifacts_dir
        / "experiments"
        / experiment_id
        / "diagnostic_profile_artifacts",
        reason=reason,
        diagnostic_only=True,
        granularity="warp",
    )


def _profiles_by_case(performance: PerformanceSummary) -> dict[str, dict[str, Any]]:
    return {evaluation.case_id: dict(evaluation.profile) for evaluation in performance.cases}


def _diagnostic_error_profiles(
    cases: tuple[InputCase, ...], error: Exception
) -> dict[str, dict[str, Any]]:
    return {
        case.case_id: {"error": f"{type(error).__name__}: {error}"}
        for case in cases
    }


def _collect_diagnostic_profiles(
    harness: SubprocessHarness,
    source: str,
    request: KernelOptimizationRequest,
    artifacts_dir: Path,
    experiment_id: str,
    *,
    reason: str,
) -> dict[str, dict[str, Any]]:
    try:
        performance = harness.evaluate(
            source,
            request.cases,
            request.target,
            request.budget.benchmark_repetitions,
            profile=_diagnostic_profile_request(
                artifacts_dir,
                experiment_id,
                reason=reason,
            ),
        )
    except Exception as error:  # noqa: BLE001
        return _diagnostic_error_profiles(request.cases, error)
    return _profiles_by_case(performance)


def _merge_diagnostic_profiles(
    performance: PerformanceSummary,
    diagnostic_profiles: Mapping[str, Mapping[str, Any]],
) -> PerformanceSummary:
    if not diagnostic_profiles:
        return performance
    merged_cases = []
    for evaluation in performance.cases:
        diagnostic_profile = diagnostic_profiles.get(evaluation.case_id)
        if not diagnostic_profile:
            merged_cases.append(evaluation)
            continue
        profile = dict(evaluation.profile)
        profile[_DIAGNOSTIC_PROFILE_KEY] = compact_profile_summary(diagnostic_profile)
        merged_cases.append(replace(evaluation, profile=profile))
    return replace(performance, cases=tuple(merged_cases))


def _report_candidate_summary(experiment_id: str, proposal: object) -> None:
    fields = (
        ("hypothesis", getattr(proposal, "hypothesis", "")),
        ("evidence", getattr(proposal, "evidence", "")),
        ("change", getattr(proposal, "summary", "")),
        ("predicted_signal", getattr(proposal, "expected_effect", "")),
        ("risk", getattr(proposal, "risk", "")),
        ("falsifier", getattr(proposal, "falsifier", "")),
        ("scope", getattr(proposal, "change_scope", "kernel-python")),
    )
    details = " ".join(
        f"{name}={value!r}" for name, value in fields if value
    ) or "change='candidate source edited'"
    print(
        f"[tlx-agent] TL_FINDING id={experiment_id} {details} recipient=worker",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[tlx-agent] WORKER_REPORT worker={experiment_id} {details} recipient=TL",
        file=sys.stderr,
        flush=True,
    )


def _correctness_counts(performance: PerformanceSummary) -> tuple[int, int]:
    return (
        sum(case.verification.passed for case in performance.cases),
        len(performance.cases),
    )


def _report_correctness_callback(
    experiment_id: str,
    performance: PerformanceSummary,
) -> None:
    passed, total = _correctness_counts(performance)
    status = "PASS" if passed == total else "FAIL"
    print(
        f"[tlx-agent] CORRECTNESS_CALLBACK worker={experiment_id} status={status} "
        f"shapes={passed}/{total} recipient=worker,TL",
        file=sys.stderr,
        flush=True,
    )


def _report_worker_final(
    experiment_id: str,
    proposal: object,
    source_path: Path,
    correctness: PerformanceSummary,
) -> None:
    passed, total = _correctness_counts(correctness)
    status = "PASS" if passed == total else "FAIL"
    scope = getattr(proposal, "change_scope", "kernel-python") or "kernel-python"
    print(
        f"[tlx-agent] FINAL worker={experiment_id} scope={scope} "
        f"files=kernel-python:1,compiler:0 artifact={source_path} "
        f"correctness_callback={status}({passed}/{total}) recipient=TL",
        file=sys.stderr,
        flush=True,
    )


def _report_tl_perf_request(experiment_id: str, case_count: int) -> None:
    print(
        f"[tlx-agent] TL_PERF_REQUEST worker={experiment_id} shapes=ALL({case_count}) "
        "recipient=performance-validator",
        file=sys.stderr,
        flush=True,
    )


def _report_perf_callback(
    experiment_id: str,
    performance: PerformanceSummary | None,
    *,
    expected_cases: int,
    status: str,
) -> None:
    if performance is None:
        shape_text = f"0/{expected_cases}"
        correct_text = f"0/{expected_cases}"
    else:
        passed, total = _correctness_counts(performance)
        shape_text = f"{len(performance.cases)}/{expected_cases}"
        correct_text = f"{passed}/{total}"
    print(
        f"[tlx-agent] PERF_CALLBACK worker={experiment_id} status={status} "
        f"shapes={shape_text} correctness={correct_text} recipient=TL",
        file=sys.stderr,
        flush=True,
    )


def _report_performance(
    experiment_id: str,
    status: str,
    performance: PerformanceSummary | None,
    *,
    baseline: PerformanceSummary | None = None,
    cases: tuple[InputCase, ...] = (),
    diagnostics: str = "",
) -> None:
    parts = [f"[tlx-agent] {experiment_id} status={status}"]
    if performance is not None:
        parts.append(f"aggregate_speedup={performance.aggregate_speedup:.4f}x")
        speedups = (
            per_case_speedups(baseline, performance, cases)
            if baseline is not None
            else {}
        )
        for evaluation in performance.cases:
            case_parts = [
                evaluation.case_id,
                "correct" if evaluation.verification.passed else "incorrect",
            ]
            if evaluation.timing is not None:
                timing = evaluation.timing
                case_parts.extend(
                    (
                        f"median={timing.median_us:.3f}us",
                        f"p95={timing.p95_us:.3f}us",
                        f"cv={timing.coefficient_of_variation:.4f}",
                    )
                )
            speedup = speedups.get(evaluation.case_id)
            if speedup is not None:
                case_parts.append(f"speedup={speedup:.4f}x")
            case_parts.extend(_profile_log_parts(evaluation.profile))
            parts.append("case=" + ",".join(case_parts))
    if diagnostics:
        parts.append(f"diagnostics={diagnostics}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def _profile_log_parts(profile: Mapping[str, Any]) -> list[str]:
    if not profile:
        return ["ncu=unavailable"]
    compact = compact_profile_summary(profile)
    parts: list[str] = []
    proton_totals = _find_mapping_with_keys(
        compact,
        frozenset({"wrapper_us", "main_kernel_us", "non_main_kernel_us"}),
    ) or _find_mapping_with_keys(
        profile,
        frozenset({"wrapper_us", "main_kernel_us", "non_main_kernel_us"}),
    )
    if proton_totals is not None:
        for label, key in (
            ("proton.wrapper_us", "wrapper_us"),
            ("proton.main_kernel_us", "main_kernel_us"),
            ("proton.non_main_kernel_us", "non_main_kernel_us"),
        ):
            value = _coerce_float(proton_totals.get(key))
            if value is not None:
                parts.append(f"{label}={value:.3f}")
    ncu_duration = extract_ncu_duration_us(compact)
    if ncu_duration is None:
        ncu_duration = extract_ncu_duration_us(profile)
    if ncu_duration is not None:
        parts.append(f"ncu.duration_us={ncu_duration:.3f}")
    else:
        parts.append("ncu=unavailable")
    diagnostic = compact.get(_DIAGNOSTIC_PROFILE_KEY)
    if not isinstance(diagnostic, Mapping):
        diagnostic = profile.get(_DIAGNOSTIC_PROFILE_KEY)
    if isinstance(diagnostic, Mapping):
        intra = diagnostic.get(_DIAGNOSTIC_PROFILE_KEY)
        if not isinstance(intra, Mapping):
            intra = diagnostic
        valid = intra.get("valid")
        if isinstance(valid, bool):
            parts.append(f"proton.intra.valid={str(valid).lower()}")
        selected_cta = intra.get("selected_cta")
        if selected_cta is not None:
            parts.append(f"proton.intra.cta={selected_cta}")
        coordinates = intra.get("logical_coordinates")
        if isinstance(coordinates, Mapping):
            coordinate_text = "/".join(
                f"{key}:{coordinates[key]}"
                for key in (
                    "start_n",
                    "logical_block",
                    "curr_m",
                    "mma_producer_j",
                    "load_input_j",
                )
                if key in coordinates
            )
            if coordinate_text:
                parts.append(f"proton.intra.tile={coordinate_text}")
        waits = intra.get("dominant_waits")
        if isinstance(waits, list) and waits and isinstance(waits[0], Mapping):
            wait_name = waits[0].get("name")
            wait_duration = _coerce_float(waits[0].get("duration"))
            if wait_name and wait_duration is not None:
                parts.append(
                    f"proton.intra.dominant_wait={wait_name}:{wait_duration:.3f}us"
                )
        trace_path = intra.get("trace_path")
        if trace_path:
            parts.append(f"proton.intra.trace={trace_path}")
    error = compact.get("error")
    if error:
        parts.append(f"profile.error={error}")
    artifact = compact.get("artifact")
    if artifact:
        parts.append(f"profile.artifact={artifact}")
    return parts


def _rejection_feedback(
    experiment_id: str,
    proposal: object,
    performance: PerformanceSummary,
    baseline: PerformanceSummary,
    cases: tuple[InputCase, ...],
    decision: str,
) -> str:
    parts = [
        f"{experiment_id}: rejected",
        f"hypothesis={getattr(proposal, 'hypothesis', '')!r}",
        f"change={getattr(proposal, 'summary', '')!r}",
        f"decision={decision}",
        f"aggregate_speedup={performance.aggregate_speedup:.4f}x",
    ]
    speedups = per_case_speedups(baseline, performance, cases)
    for evaluation in performance.cases:
        case_parts = [
            evaluation.case_id,
            "correct" if evaluation.verification.passed else "incorrect",
        ]
        if evaluation.timing is not None:
            case_parts.extend(
                (
                    f"median={evaluation.timing.median_us:.3f}us",
                    f"cv={evaluation.timing.coefficient_of_variation:.4f}",
                )
            )
        speedup = speedups.get(evaluation.case_id)
        if speedup is not None:
            case_parts.append(f"speedup={speedup:.4f}x")
        case_parts.extend(_profile_log_parts(evaluation.profile))
        parts.append("case=" + ",".join(case_parts))
    return " ".join(parts)[:4000]


def _find_mapping_with_keys(
    value: Any,
    keys: frozenset[str],
) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        if keys.issubset({str(key) for key in value.keys()}):
            return value
        for item in value.values():
            match = _find_mapping_with_keys(item, keys)
            if match is not None:
                return match
    if isinstance(value, list | tuple):
        for item in value:
            match = _find_mapping_with_keys(item, keys)
            if match is not None:
                return match
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except ValueError:
        return None


def _is_correct_and_stable(
    performance: PerformanceSummary,
    budget: object,
    cases: tuple[InputCase, ...],
) -> bool:
    case_by_id = {case.case_id: case for case in cases}
    for evaluation in performance.cases:
        case = case_by_id.get(evaluation.case_id)
        protected = case.protected if case is not None else True
        if not evaluation.verification.passed:
            if protected:
                return False
            continue
        if evaluation.timing is None:
            return False
        if evaluation.timing.coefficient_of_variation > budget.max_cv:
            return False
    return True


def _is_near_threshold(speedup: float, threshold: float) -> bool:
    return threshold - _NEAR_THRESHOLD_WINDOW <= speedup <= threshold + _NEAR_THRESHOLD_WINDOW


def _rejection_decision(
    performance: PerformanceSummary,
    *,
    speedup: float,
    best_speedup: float,
    min_speedup: float,
    max_cv: float,
    ncu_diagnostics: str,
    correctness_diagnostics: str,
) -> str:
    if ncu_diagnostics:
        return ncu_diagnostics
    if correctness_diagnostics:
        return correctness_diagnostics
    if speedup < min_speedup:
        return f"speedup below {min_speedup:.4f}x threshold"
    unstable = [
        evaluation.case_id
        for evaluation in performance.cases
        if evaluation.verification.passed
        and (
            evaluation.timing is None
            or evaluation.timing.coefficient_of_variation > max_cv
        )
    ]
    if unstable:
        return (
            f"unstable or missing timing above max_cv={max_cv:.4f}: "
            + ",".join(unstable)
        )
    if speedup <= best_speedup:
        return f"did not beat incumbent aggregate_speedup={best_speedup:.4f}x"
    return "candidate did not satisfy the promotion policy"


def _ncu_regression_diagnostics(
    baseline: PerformanceSummary,
    candidate: PerformanceSummary,
) -> str:
    baseline_by_case = {evaluation.case_id: evaluation for evaluation in baseline.cases}
    diagnostics: list[str] = []
    for evaluation in candidate.cases:
        baseline_evaluation = baseline_by_case.get(evaluation.case_id)
        if baseline_evaluation is None:
            continue
        diagnostic = ncu_regression_diagnostic(
            baseline_evaluation.profile,
            evaluation.profile,
        )
        if diagnostic:
            diagnostics.append(f"{evaluation.case_id}: {diagnostic}")
    return "; ".join(diagnostics)


def _report_candidate_artifacts(
    experiment_id: str, artifacts: CandidateArtifactPaths
) -> None:
    print(
        " ".join(
            [
                f"[tlx-agent] {experiment_id} status=artifacts",
                f"source={str(artifacts.source_path.resolve())!r}",
                f"incremental_patch={str(artifacts.incremental_patch_path.resolve())!r}",
                f"cumulative_patch={str(artifacts.cumulative_patch_path.resolve())!r}",
            ]
        ),
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[tlx-agent] {experiment_id} incremental-diff-begin",
        file=sys.stderr,
    )
    patch = artifacts.incremental_patch_path.read_text()
    if patch:
        sys.stderr.write(patch)
        if not patch.endswith("\n"):
            sys.stderr.write("\n")
    else:
        print("(no source changes)", file=sys.stderr)
    print(
        f"[tlx-agent] {experiment_id} incremental-diff-end",
        file=sys.stderr,
        flush=True,
    )


class KernelOptimizer:
    def __init__(self, provider: CandidateProvider | None = None) -> None:
        self._provider = provider or CodexCandidateProvider()

    def optimize(
        self,
        request: KernelOptimizationRequest,
        promotion_committer: PromotionCommitter | None = None,
    ) -> KernelOptimizationResult:
        artifacts_dir = request.output_dir or Path(
            tempfile.mkdtemp(prefix="tlx-kernel-agent-")
        )
        store = ArtifactStore(artifacts_dir)
        # Persist reference kernel if provided so harness can load it as oracle
        if request.reference_kernel_source:
            ref_path = artifacts_dir / "reference_kernel.py"
            ref_path.write_text(request.reference_kernel_source)
            # Expose to harness workers via env var
            import os
            os.environ["TLX_REFERENCE_KERNEL_PATH"] = str(ref_path)
        harness = SubprocessHarness(
            request.harness_path, request.budget.max_candidate_seconds
        )
        start_time = time.monotonic()
        baseline_source_path = store.write_source("baseline", request.kernel_source)
        baseline = harness.evaluate(
            request.kernel_source,
            request.cases,
            request.target,
            request.budget.benchmark_repetitions,
            profile=_profile_request(
                artifacts_dir,
                "baseline",
                level="deep",
                reason="baseline",
            ),
        )
        if not passes_protected_cases(baseline, request.cases):
            raise HarnessExecutionError(
                "baseline failed one or more protected correctness cases"
            )
        baseline = replace(baseline, aggregate_speedup=1.0)
        if request.diagnostic_proton_intra_kernel:
            baseline_diagnostics = _collect_diagnostic_profiles(
                harness,
                request.kernel_source,
                request,
                artifacts_dir,
                "baseline",
                reason="baseline_diagnostic",
            )
            baseline = _merge_diagnostic_profiles(baseline, baseline_diagnostics)
        baseline_profiles = _profiles_by_case(baseline)
        baseline_profile_path = store.write_profile("baseline", baseline_profiles)
        store.write_aggregated_profile("baseline_profile", baseline_profiles)
        experiments = [
            ExperimentSummary(
                experiment_id="baseline",
                round_index=0,
                parent_id=None,
                status="baseline",
                source_path=baseline_source_path,
                performance=baseline,
                profile_path=baseline_profile_path,
            )
        ]
        store.write_json("experiments/baseline/result.json", baseline)
        _report_performance("baseline", "baseline", baseline)
        _report_perf_callback(
            "baseline",
            baseline,
            expected_cases=len(request.cases),
            status="BASELINE",
        )

        best_source = request.kernel_source
        best_performance = baseline
        best_experiment_id = "baseline"
        best_commit_title = ""
        best_commit_summary = ""
        best_profiles = baseline_profiles
        diagnostics: list[str] = []
        promotion_commits: list[AutoCommitResult] = []
        rollback_commit: AutoCommitResult | None = None
        last_auto_commit: AutoCommitResult | None = None
        prior_source_hashes = set(
            request.prior_run_evidence.source_hashes
            if request.prior_run_evidence is not None
            else ()
        )
        seen_sources = {source_digest(request.kernel_source), *prior_source_hashes}
        stopping_reason = "round_budget_exhausted"
        exhausted = False

        for round_index in range(1, request.budget.max_rounds + 1):
            if time.monotonic() - start_time >= request.budget.max_total_seconds:
                stopping_reason = "time_budget_exhausted"
                break
            promoted_this_round = False
            for candidate_index in range(request.budget.candidates_per_round):
                if time.monotonic() - start_time >= request.budget.max_total_seconds:
                    stopping_reason = "time_budget_exhausted"
                    exhausted = True
                    break
                experiment_id = f"r{round_index:03d}-c{candidate_index:03d}"
                parent_id = best_experiment_id
                parent_source = best_source
                proposal = None
                candidate_artifacts = None
                try:
                    _report_performance(experiment_id, "generating", None)
                    proposal = self._provider.propose(
                        request,
                        CandidateContext(
                            round_index=round_index,
                            candidate_index=candidate_index,
                            current_source=parent_source,
                            current_performance=best_performance,
                            previous_diagnostics=tuple(diagnostics),
                        ),
                    )
                    _report_candidate_summary(experiment_id, proposal)
                    candidate_artifacts = store.write_candidate_artifacts(
                        experiment_id,
                        source=proposal.source,
                        parent_source=parent_source,
                        parent_id=parent_id,
                        baseline_source=request.kernel_source,
                    )
                    _report_candidate_artifacts(experiment_id, candidate_artifacts)
                    digest = source_digest(proposal.source)
                    if digest in seen_sources:
                        origin = (
                            "a prior run experiment"
                            if digest in prior_source_hashes
                            else "an earlier experiment"
                        )
                        raise ValueError(f"candidate source duplicates {origin}")
                    seen_sources.add(digest)
                    _report_performance(experiment_id, "correctness", None)
                    correctness = harness.evaluate(
                        proposal.source,
                        request.cases,
                        request.target,
                        request.budget.benchmark_repetitions,
                        benchmark=False,
                    )
                    _report_correctness_callback(experiment_id, correctness)
                    _report_worker_final(
                        experiment_id,
                        proposal,
                        candidate_artifacts.source_path,
                        correctness,
                    )
                    if not passes_protected_cases(correctness, request.cases):
                        performance = replace(correctness, aggregate_speedup=0.0)
                        speedup = 0.0
                        _report_perf_callback(
                            experiment_id,
                            None,
                            expected_cases=len(request.cases),
                            status="BLOCKED_BY_CORRECTNESS",
                        )
                    else:
                        _report_tl_perf_request(experiment_id, len(request.cases))
                        performance = harness.evaluate(
                            proposal.source,
                            request.cases,
                            request.target,
                            request.budget.benchmark_repetitions,
                            profile=_profile_request(
                                artifacts_dir,
                                experiment_id,
                                level="summary",
                                reason="candidate",
                            ),
                        )
                        speedup = weighted_geometric_speedup(
                            baseline, performance, request.cases
                        )
                        performance = replace(performance, aggregate_speedup=speedup)
                        _report_perf_callback(
                            experiment_id,
                            performance,
                            expected_cases=len(request.cases),
                            status="MEASURED",
                        )
                    if _is_correct_and_stable(
                        performance,
                        request.budget,
                        request.cases,
                    ) and _is_near_threshold(speedup, request.budget.min_speedup):
                        _report_performance(
                            experiment_id,
                            "near-threshold-profiling",
                            performance,
                            baseline=baseline,
                            cases=request.cases,
                            diagnostics="reason=near_threshold",
                        )
                        performance = harness.evaluate(
                            proposal.source,
                            request.cases,
                            request.target,
                            request.budget.benchmark_repetitions,
                            profile=_profile_request(
                                artifacts_dir,
                                experiment_id,
                                level="deep",
                                reason="near_threshold",
                            ),
                        )
                        speedup = weighted_geometric_speedup(
                            baseline, performance, request.cases
                        )
                        performance = replace(performance, aggregate_speedup=speedup)
                        _report_perf_callback(
                            experiment_id,
                            performance,
                            expected_cases=len(request.cases),
                            status="REMEASURED",
                        )
                    # Persist per-case profiles for this candidate regardless of promotion.
                    perf_profiles = _profiles_by_case(performance)
                    profile_path = store.write_profile(experiment_id, perf_profiles)
                    ncu_diagnostics = _ncu_regression_diagnostics(baseline, performance)
                    status = (
                        "promoted"
                        if not ncu_diagnostics
                        and is_promotable(performance, request.budget, request.cases)
                        and speedup > best_performance.aggregate_speedup
                        else "rejected"
                    )
                    experiment = ExperimentSummary(
                        experiment_id=experiment_id,
                        round_index=round_index,
                        parent_id=parent_id,
                        status=status,
                        source_path=candidate_artifacts.source_path,
                        incremental_patch_path=candidate_artifacts.incremental_patch_path,
                        cumulative_patch_path=candidate_artifacts.cumulative_patch_path,
                        performance=performance,
                        mutation_summary=proposal.summary,
                        hypothesis=proposal.hypothesis,
                        evidence=proposal.evidence,
                        expected_effect=proposal.expected_effect,
                        risk=proposal.risk,
                        commit_title=proposal.commit_title,
                        commit_summary=proposal.commit_summary,
                        falsifier=proposal.falsifier,
                        change_scope=proposal.change_scope,
                        profile_path=profile_path,
                    )
                    rejection_diagnostics = "; ".join(
                        f"{evaluation.case_id}: {evaluation.verification.diagnostics}"
                        for evaluation in performance.cases
                        if not evaluation.verification.passed
                        and evaluation.verification.diagnostics
                    )
                    decision = (
                        "correct, stable, and beat the incumbent"
                        if status == "promoted"
                        else _rejection_decision(
                            performance,
                            speedup=speedup,
                            best_speedup=best_performance.aggregate_speedup,
                            min_speedup=request.budget.min_speedup,
                            max_cv=request.budget.max_cv,
                            ncu_diagnostics=ncu_diagnostics,
                            correctness_diagnostics=rejection_diagnostics,
                        )
                    )
                    if status == "rejected":
                        diagnostics.append(
                            _rejection_feedback(
                                experiment_id,
                                proposal,
                                performance,
                                baseline,
                                request.cases,
                                decision,
                            )
                        )
                    _report_performance(
                        experiment_id,
                        status,
                        performance,
                        baseline=baseline,
                        cases=request.cases,
                        diagnostics=f"decision={decision}",
                    )
                    if status == "promoted" and promotion_committer is not None:
                        commit_result = promotion_committer.commit_promotion(
                            experiment,
                            proposal.source,
                            baseline,
                            performance,
                        )
                        experiment = replace(experiment, auto_commit=commit_result)
                        last_auto_commit = commit_result
                        if commit_result.success:
                            promotion_commits.append(commit_result)
                        else:
                            experiment = replace(
                                experiment,
                                status="failed",
                                diagnostics=commit_result.diagnostics,
                            )
                            stopping_reason = "promotion_commit_failed"
                            exhausted = True
                    if status == "promoted" and not exhausted:
                        best_source = proposal.source
                        best_performance = performance
                        best_experiment_id = experiment_id
                        best_commit_title = proposal.commit_title
                        best_commit_summary = proposal.commit_summary
                        best_profiles = perf_profiles
                        promoted_this_round = True
                except Exception as error:  # noqa: BLE001
                    message = f"{experiment_id}: {type(error).__name__}: {error}"
                    diagnostics.append(message)
                    source_path = (
                        candidate_artifacts.source_path
                        if candidate_artifacts is not None
                        else store.write_source(experiment_id, "")
                    )
                    experiment = ExperimentSummary(
                        experiment_id=experiment_id,
                        round_index=round_index,
                        parent_id=parent_id,
                        status="failed",
                        source_path=source_path,
                        incremental_patch_path=(
                            candidate_artifacts.incremental_patch_path
                            if candidate_artifacts is not None
                            else None
                        ),
                        cumulative_patch_path=(
                            candidate_artifacts.cumulative_patch_path
                            if candidate_artifacts is not None
                            else None
                        ),
                        diagnostics=message,
                    )
                    _report_performance(
                        experiment_id,
                        "failed",
                        None,
                        diagnostics=message,
                    )
                experiments.append(experiment)
                store.write_json(
                    f"experiments/{experiment_id}/result.json", experiment
                )
            if exhausted:
                break
            if not promoted_this_round:
                stopping_reason = "round_budget_exhausted"

        final_profile = harness.evaluate(
            best_source,
            request.cases,
            request.target,
            request.budget.benchmark_repetitions,
            profile=_profile_request(
                artifacts_dir,
                "final",
                level="deep",
                reason="final",
            ),
        )
        final_profile = replace(
            final_profile,
            aggregate_speedup=weighted_geometric_speedup(
                baseline, final_profile, request.cases
            ),
        )
        final_ncu_diagnostics = _ncu_regression_diagnostics(baseline, final_profile)
        if best_experiment_id != "baseline" and (
            final_ncu_diagnostics
            or not is_promotable(final_profile, request.budget, request.cases)
        ):
            if final_ncu_diagnostics:
                diagnostics.append(f"final: rejected: {final_ncu_diagnostics}")
            best_source = request.kernel_source
            final_profile = baseline
            final_profiles = baseline_profiles
            best_profiles = baseline_profiles
            best_experiment_id = "baseline"
            best_commit_title = ""
            best_commit_summary = ""
            stopping_reason = "finalist_revalidation_failed"
            if promotion_committer is not None and promotion_commits:
                rollback_commit = promotion_committer.rollback_to_baseline(
                    "; ".join(diagnostics) or "final revalidation failed"
                )
                last_auto_commit = rollback_commit
                if not rollback_commit.success:
                    stopping_reason = "rollback_commit_failed"
        elif best_experiment_id != "baseline":
            if request.diagnostic_proton_intra_kernel:
                final_diagnostics = _collect_diagnostic_profiles(
                    harness,
                    best_source,
                    request,
                    artifacts_dir,
                    "final",
                    reason="final_winner_diagnostic",
                )
                final_profile = _merge_diagnostic_profiles(
                    final_profile, final_diagnostics
                )
            final_profiles = _profiles_by_case(final_profile)
            best_profiles = final_profiles
        else:
            final_profile = baseline
            final_profiles = baseline_profiles
        store.write_profile("final", final_profiles)
        _report_performance(
            "final",
            "revalidated" if best_experiment_id != "baseline" else "baseline",
            final_profile,
            baseline=baseline,
            cases=request.cases,
            diagnostics=final_ncu_diagnostics,
        )
        store.write_aggregated_profile("best_profile", best_profiles)
        # Doc-compatible alias: experiments.json mirrors the experiments list.
        store.write_json("experiments.json", tuple(experiments))
        store.write_best(best_source)
        result = KernelOptimizationResult(
            success=best_experiment_id != "baseline",
            best_kernel=best_source,
            baseline=baseline,
            final=final_profile,
            experiments=tuple(experiments),
            artifacts_dir=artifacts_dir,
            stopping_reason=stopping_reason,
            winner_experiment_id=best_experiment_id,
            winner_commit_title=best_commit_title,
            winner_commit_summary=best_commit_summary,
            promotion_commits=tuple(promotion_commits),
            rollback_commit=rollback_commit,
            auto_commit=last_auto_commit,
        )
        store.write_json("result.json", result)
        return result
