from __future__ import annotations

import math

from ..contracts import InputCase, OptimizationBudget, PerformanceSummary
from .profiling import ncu_regression_diagnostic

NEAR_THRESHOLD_WINDOW = 0.01


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
    return True


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
        diagnostic = ncu_regression_diagnostic(
            baseline_evaluation.profile,
            evaluation.profile,
        )
        if diagnostic:
            diagnostics.append(f"{evaluation.case_id}: {diagnostic}")
    return "; ".join(diagnostics)
