from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from ..contracts import InputCase, PerformanceSummary
from .artifacts import CandidateArtifactPaths
from .evaluation import DIAGNOSTIC_PROFILE_KEY
from .policy import per_case_speedups
from .profiling import compact_profile_summary, extract_ncu_duration_us


def report_candidate_summary(experiment_id: str, proposal: object) -> None:
    fields = (
        ("hypothesis", getattr(proposal, "hypothesis", "")),
        ("evidence", getattr(proposal, "evidence", "")),
        ("change", getattr(proposal, "summary", "")),
        ("expected", getattr(proposal, "expected_effect", "")),
        ("risk", getattr(proposal, "risk", "")),
    )
    details = " ".join(
        f"{name}={value!r}" for name, value in fields if value
    ) or "change='candidate source edited'"
    print(f"[tlx-agent] {experiment_id} {details}", file=sys.stderr, flush=True)


def report_performance(
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
            case_parts.extend(profile_log_parts(evaluation.profile))
            parts.append("case=" + ",".join(case_parts))
    if diagnostics:
        parts.append(f"diagnostics={diagnostics}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def profile_log_parts(profile: Mapping[str, Any]) -> list[str]:
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
    diagnostic = compact.get(DIAGNOSTIC_PROFILE_KEY)
    if not isinstance(diagnostic, Mapping):
        diagnostic = profile.get(DIAGNOSTIC_PROFILE_KEY)
    if isinstance(diagnostic, Mapping):
        intra = diagnostic.get(DIAGNOSTIC_PROFILE_KEY)
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


def rejection_feedback(
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
        case_parts.extend(profile_log_parts(evaluation.profile))
        parts.append("case=" + ",".join(case_parts))
    return " ".join(parts)[:4000]


def report_candidate_artifacts(
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
    print(f"[tlx-agent] {experiment_id} incremental-diff-begin", file=sys.stderr)
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
