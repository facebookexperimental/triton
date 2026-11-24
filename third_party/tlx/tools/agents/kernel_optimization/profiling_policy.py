from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .profiling import ProfileRequest, affected_region_upper_bound

_PROFILE_TOOLS = ("proton_launch", "native_profiler")
_INTRA_KERNEL_TOOL = "proton_intra_kernel"
_ROLE_PASS = "role"
_FULL_DIAGNOSTIC_PASSES = (_ROLE_PASS, "coarse", "wait", "compute")
_NEAR_THRESHOLD_WINDOW = 0.01
_FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ProfileDecision:
    request: ProfileRequest | None
    reason: str


def adaptive_candidate_profile_decision(
    correct_and_stable: bool,
    speedup: float,
    min_speedup: float,
    artifacts_dir: Path,
    experiment_id: str,
    source_digest: str,
) -> ProfileDecision:
    if not correct_and_stable:
        return ProfileDecision(None, "incorrect_or_unstable")
    if speedup < min_speedup - _NEAR_THRESHOLD_WINDOW - _FLOAT_TOLERANCE:
        return ProfileDecision(None, "clear_loser")
    policy_reason = (
        "near_threshold"
        if abs(speedup - min_speedup) <= _NEAR_THRESHOLD_WINDOW + _FLOAT_TOLERANCE
        else "candidate"
    )
    return ProfileDecision(
        ProfileRequest(
            level="deep" if policy_reason == "near_threshold" else "summary",
            tools=_PROFILE_TOOLS,
            experiment_id=experiment_id,
            artifacts_dir=artifacts_dir
            / "experiments"
            / experiment_id
            / "profile_artifacts",
            reason="candidate_profile",
            source_digest=source_digest,
            policy_reason=policy_reason,
        ),
        policy_reason,
    )


def targeted_diagnostic_decision(
    enabled: bool,
    hint_pass: str,
    expected_regions: Iterable[str],
    capabilities: Mapping[str, Iterable[str]],
    baseline_profile: Mapping[str, Any],
    min_speedup: float,
    remaining_pass_budget: int,
    artifacts_dir: Path,
    experiment_id: str,
    source_digest: str,
) -> ProfileDecision:
    if not enabled:
        return ProfileDecision(None, "disabled")
    if hint_pass == _ROLE_PASS:
        return ProfileDecision(None, "role_only_hint")
    supported_regions = capabilities.get(hint_pass)
    if supported_regions is None:
        return ProfileDecision(None, "unsupported_pass")
    regions = tuple(str(region) for region in expected_regions)
    if not set(regions).issubset({str(region) for region in supported_regions}):
        return ProfileDecision(None, "unsupported_regions")
    passes = (_ROLE_PASS, str(hint_pass))
    if remaining_pass_budget < len(passes):
        return ProfileDecision(None, "pass_budget_exhausted")
    upper_bound = affected_region_upper_bound(baseline_profile, regions)
    if upper_bound is None:
        return ProfileDecision(None, "unknown_upper_bound")
    required = 2 * (min_speedup - 1)
    if upper_bound < required:
        return ProfileDecision(None, "low_upper_bound")
    policy_reason = "targeted_diagnostic"
    return ProfileDecision(
        _diagnostic_request(
            artifacts_dir,
            experiment_id,
            source_digest,
            reason="targeted_diagnostic",
            policy_reason=policy_reason,
            passes=passes,
        ),
        policy_reason,
    )


def full_diagnostic_request(
    artifacts_dir: Path,
    experiment_id: str,
    source_digest: str,
    *,
    reason: str,
    pass_budget: int | None = None,
    passes: Iterable[str] = _FULL_DIAGNOSTIC_PASSES,
) -> ProfileRequest | None:
    requested_passes = _dedupe_passes(passes)
    if not requested_passes:
        return None
    if _ROLE_PASS not in requested_passes:
        requested_passes = (_ROLE_PASS,) + requested_passes
    if pass_budget is not None:
        if pass_budget <= 0:
            return None
        if len(requested_passes) > pass_budget:
            return None
    return _diagnostic_request(
        artifacts_dir,
        experiment_id,
        source_digest,
        reason=reason,
        policy_reason="full_diagnostic",
        passes=requested_passes,
    )


def _diagnostic_request(
    artifacts_dir: Path,
    experiment_id: str,
    source_digest: str,
    *,
    reason: str,
    policy_reason: str,
    passes: Iterable[str],
) -> ProfileRequest:
    return ProfileRequest(
        level="deep",
        tools=(_INTRA_KERNEL_TOOL,),
        passes=_dedupe_passes(passes),
        experiment_id=experiment_id,
        artifacts_dir=artifacts_dir
        / "experiments"
        / experiment_id
        / "diagnostic_profile_artifacts",
        reason=reason,
        source_digest=source_digest,
        policy_reason=policy_reason,
        diagnostic_only=True,
        granularity="warp",
    )


def _dedupe_passes(passes: Iterable[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for pass_name in passes:
        normalized = str(pass_name)
        if normalized not in deduped:
            deduped.append(normalized)
    return tuple(deduped)
