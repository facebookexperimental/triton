from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..contracts import KernelOptimizationRequest, PerformanceSummary
from .harness import SubprocessHarness
from .profiling import ProfileRequest, compact_profile_summary

PROFILE_TOOLS = ("proton_launch", "native_profiler")
DIAGNOSTIC_PROFILE_TOOLS = ("proton_intra_kernel",)
DIAGNOSTIC_PROFILE_KEY = "diagnostic_proton_intra_kernel"


def profile_request(
    artifacts_dir: Path,
    experiment_id: str,
    *,
    level: str,
    reason: str,
) -> ProfileRequest:
    return ProfileRequest(
        level=level,
        tools=PROFILE_TOOLS,
        experiment_id=experiment_id,
        artifacts_dir=artifacts_dir / "experiments" / experiment_id / "profile_artifacts",
        reason=reason,
    )


def diagnostic_profile_request(
    artifacts_dir: Path,
    experiment_id: str,
    *,
    reason: str,
) -> ProfileRequest:
    return ProfileRequest(
        level="deep",
        tools=DIAGNOSTIC_PROFILE_TOOLS,
        experiment_id=experiment_id,
        artifacts_dir=artifacts_dir
        / "experiments"
        / experiment_id
        / "diagnostic_profile_artifacts",
        reason=reason,
        diagnostic_only=True,
        granularity="warp",
    )


def profiles_by_case(performance: PerformanceSummary) -> dict[str, dict[str, Any]]:
    return {evaluation.case_id: dict(evaluation.profile) for evaluation in performance.cases}


def collect_diagnostic_profiles(
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
            profile=diagnostic_profile_request(
                artifacts_dir,
                experiment_id,
                reason=reason,
            ),
        )
    except Exception as error:  # noqa: BLE001
        return {
            case.case_id: {"error": f"{type(error).__name__}: {error}"}
            for case in request.cases
        }
    return profiles_by_case(performance)


def merge_diagnostic_profiles(
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
        profile[DIAGNOSTIC_PROFILE_KEY] = compact_profile_summary(diagnostic_profile)
        merged_cases.append(replace(evaluation, profile=profile))
    return replace(performance, cases=tuple(merged_cases))
