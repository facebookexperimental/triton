from __future__ import annotations

import math

from ..contracts import (
    BlastRadius,
    CandidateSubmission,
    CaseEvaluation,
    ChangeScope,
    Decision,
    DecisionStatus,
    ExperimentKind,
    InputCase,
    KernelTarget,
    OptimizationBudget,
    PerformanceSummary,
)
from .profiling import (
    extract_native_profiler_duration_us,
    native_profiler_regression_diagnostic,
)

NEAR_THRESHOLD_WINDOW = 0.01
ABLATION_KINDS = frozenset(
    {
        ExperimentKind.PTX_ABLATION,
        ExperimentKind.AMDGCN_ABLATION,
        ExperimentKind.IR_OVERRIDE,
    }
)


def preflight_decision(
    submission: CandidateSubmission,
    target: KernelTarget,
) -> Decision | None:
    """Validate authority boundaries before any candidate execution."""

    kind = submission.experiment_kind
    if kind not in target.supported_experiment_kinds:
        raise ValueError(
            f"target does not support experiment kind {kind.value!r}; supported kinds: "
            + ", ".join(item.value for item in target.supported_experiment_kinds)
        )
    if submission.changes:
        declared_scopes = frozenset(change.scope for change in submission.changes)
        if declared_scopes != submission.change_scopes:
            raise ValueError("change_scopes must exactly match the scopes in changes")
    if not submission.change_scopes:
        raise ValueError("submission must declare at least one change scope")

    backend = target.backend.strip().lower()
    if kind is ExperimentKind.PTX_ABLATION and backend not in {"cuda", "nvidia"}:
        raise ValueError("ptx_ablation requires an NVIDIA target")
    if kind is ExperimentKind.AMDGCN_ABLATION and backend not in {"amd", "hip", "rocm"}:
        raise ValueError("amdgcn_ablation requires an AMD target")
    if kind in ABLATION_KINDS:
        if ChangeScope.COMPILER not in submission.change_scopes:
            raise ValueError("compiler ablations must declare compiler scope")
        if not submission.experiment_payload:
            raise ValueError("ablation submissions require experiment_payload")
    elif kind is ExperimentKind.PROMOTABLE and submission.experiment_payload:
        raise ValueError("promotable submissions must not include experiment_payload")

    if kind is ExperimentKind.HUMAN_REVIEW:
        if not submission.experiment_payload:
            raise ValueError("human_review requires experiment_payload")
        return Decision(
            status=DecisionStatus.NEEDS_HUMAN,
            rationale=submission.summary or "optimizer requested human review",
            feedback="autonomous execution stopped before evaluation",
        )
    if kind is ExperimentKind.PROMOTABLE and (
        ChangeScope.COMPILER in submission.change_scopes
        or submission.blast_radius is not BlastRadius.LOCAL
    ):
        return Decision(
            status=DecisionStatus.NEEDS_HUMAN,
            rationale=(
                "promotable compiler or non-local changes require a human-reviewed "
                "implementation workflow"
            ),
            feedback="resubmit as an ablation or request human review",
        )
    return None


def evaluated_decision(
    submission: CandidateSubmission,
    performance: PerformanceSummary,
    budget: OptimizationBudget,
    cases: tuple[InputCase, ...],
    *,
    best_speedup: float,
    profiler_diagnostics: str = "",
    target: KernelTarget | None = None,
) -> Decision:
    """Return the authoritative action for a completed experiment."""

    if submission.experiment_kind in ABLATION_KINDS:
        return Decision(
            status=DecisionStatus.RECORD_SIGNAL,
            rationale=(
                f"{submission.experiment_kind.value} measured "
                f"{performance.aggregate_speedup:.4f}x aggregate speedup"
            ),
            feedback=profiler_diagnostics,
            evaluation=performance,
        )
    if submission.experiment_kind is not ExperimentKind.PROMOTABLE:
        raise ValueError(
            f"experiment kind {submission.experiment_kind.value!r} cannot be evaluated"
        )
    if (
        not profiler_diagnostics
        and is_promotable(performance, budget, cases, target)
        and performance.aggregate_speedup > best_speedup
    ):
        return Decision(
            status=DecisionStatus.PROMOTE,
            rationale="correct and exceeded speedup threshold",
            evaluation=performance,
        )
    rationale = (
        profiler_diagnostics
        or _verification_diagnostics(performance)
        or f"speedup below {budget.min_speedup:.4f}x threshold"
    )
    return Decision(
        status=DecisionStatus.RETRY,
        rationale=rationale,
        evaluation=performance,
    )


def _verification_diagnostics(performance: PerformanceSummary) -> str:
    return "; ".join(
        f"{evaluation.case_id}: {evaluation.verification.diagnostics}"
        for evaluation in performance.cases
        if not evaluation.verification.passed
        and evaluation.verification.diagnostics
    )


def weighted_geometric_speedup(
    baseline: PerformanceSummary,
    candidate: PerformanceSummary,
    cases: tuple[InputCase, ...],
) -> float:
    baseline_by_id = {result.case_id: result for result in baseline.cases}
    candidate_by_id = {result.case_id: result for result in candidate.cases}
    weighted_logs = 0.0
    total_weight = 0.0
    for case in cases:
        before = baseline_by_id.get(case.case_id)
        after = candidate_by_id.get(case.case_id)
        if before is None or after is None:
            raise ValueError(f"missing evaluation for case {case.case_id}")
        if not after.verification.passed:
            if case.protected:
                return 0.0
            continue
        if before.timing is None or after.timing is None:
            raise ValueError(f"missing timing for case {case.case_id}")
        speedup = before.timing.median_us / after.timing.median_us
        weighted_logs += case.weight * math.log(speedup)
        total_weight += case.weight
    if total_weight == 0:
        return 0.0
    return math.exp(weighted_logs / total_weight)


def per_case_speedups(
    baseline: PerformanceSummary,
    candidate: PerformanceSummary,
    cases: tuple[InputCase, ...],
) -> dict[str, float | None]:
    """Per-case speedup (baseline median / candidate median), None for failed/skipped."""

    baseline_by_id = {result.case_id: result for result in baseline.cases}
    candidate_by_id = {result.case_id: result for result in candidate.cases}
    result: dict[str, float | None] = {}
    for case in cases:
        before = baseline_by_id.get(case.case_id)
        after = candidate_by_id.get(case.case_id)
        if before is None or after is None:
            result[case.case_id] = None
            continue
        if not after.verification.passed:
            result[case.case_id] = None
            continue
        if before.timing is None or after.timing is None:
            result[case.case_id] = None
            continue
        result[case.case_id] = before.timing.median_us / after.timing.median_us
    return result


def passes_protected_cases(
    summary: PerformanceSummary, cases: tuple[InputCase, ...]
) -> bool:
    protected_by_id = {case.case_id: case.protected for case in cases}
    return all(
        evaluation.verification.passed
        or not protected_by_id.get(evaluation.case_id, True)
        for evaluation in summary.cases
    )


def is_promotable(
    summary: PerformanceSummary,
    budget: OptimizationBudget,
    cases: tuple[InputCase, ...] | None = None,
    target: KernelTarget | None = None,
) -> bool:
    case_by_id = {case.case_id: case for case in cases or ()}
    if summary.aggregate_speedup < budget.min_speedup:
        return False
    for evaluation in summary.cases:
        protected = case_by_id.get(evaluation.case_id)
        is_protected = protected.protected if protected is not None else True
        if not evaluation.verification.passed:
            if is_protected:
                return False
            continue
        if (
            evaluation.timing is None
            or evaluation.timing.coefficient_of_variation > budget.max_cv
        ):
            return False
    return _passes_evaluation_policy(summary, tuple(cases or ()), target)


def _passes_evaluation_policy(
    summary: PerformanceSummary,
    cases: tuple[InputCase, ...],
    target: KernelTarget | None,
) -> bool:
    policy = dict(target.evaluation_policy) if target is not None else {}
    minimum_full_configs = policy.get("minimum_full_config_count")
    if minimum_full_configs is not None:
        counts = [
            int(evaluation.verification.metrics.get("full_config_count", 0))
            for evaluation in summary.cases
            if evaluation.verification.passed
        ]
        if not counts or max(counts) < int(minimum_full_configs):
            return False
    if policy.get("kind") != "full_space_parity":
        return True

    per_case_min = float(policy.get("per_case_min", 0.95))
    aggregate_min = float(policy.get("aggregate_min", 0.98))
    max_configs = int(policy.get("max_heuristic_configs", 1))
    weights = {case.case_id: case.weight for case in cases}
    weighted_logs = 0.0
    total_weight = 0.0
    for evaluation in summary.cases:
        metrics = evaluation.verification.metrics
        parity = metrics.get("full_space_parity")
        config_count = metrics.get("heuristic_config_count")
        stable = bool(metrics.get("parity_stable", True))
        if config_count is None or int(config_count) > max_configs:
            return False
        if parity is None:
            return False
        parity = float(parity)
        if not math.isfinite(parity) or parity <= 0:
            return False
        if stable and parity < per_case_min:
            return False
        if stable:
            weight = float(weights.get(evaluation.case_id, 1.0))
            weighted_logs += weight * math.log(parity)
            total_weight += weight
    if total_weight == 0:
        return False
    return math.exp(weighted_logs / total_weight) >= aggregate_min


def is_correct_and_stable(
    performance: PerformanceSummary,
    budget: OptimizationBudget,
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


def is_near_threshold(speedup: float, threshold: float) -> bool:
    return (
        threshold - NEAR_THRESHOLD_WINDOW
        <= speedup
        <= threshold + NEAR_THRESHOLD_WINDOW
    )


def profiler_regression_diagnostics(
    baseline: PerformanceSummary,
    candidate: PerformanceSummary,
) -> str:
    baseline_by_case = {evaluation.case_id: evaluation for evaluation in baseline.cases}
    diagnostics: list[str] = []
    for evaluation in candidate.cases:
        baseline_evaluation = baseline_by_case.get(evaluation.case_id)
        if baseline_evaluation is None:
            continue
        if _benchmark_uses_native_profiler(
            baseline_evaluation
        ) and _benchmark_uses_native_profiler(evaluation):
            continue
        diagnostic = native_profiler_regression_diagnostic(
            baseline_evaluation.profile,
            evaluation.profile,
        )
        if diagnostic:
            diagnostics.append(f"{evaluation.case_id}: {diagnostic}")
    return "; ".join(diagnostics)


def _benchmark_uses_native_profiler(evaluation: CaseEvaluation) -> bool:
    tool, duration = extract_native_profiler_duration_us(evaluation.profile)
    return (
        tool is not None
        and duration is not None
        and evaluation.timing is not None
        and tool.lower() in evaluation.timing.cache_policy.lower()
    )
