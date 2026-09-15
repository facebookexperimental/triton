from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from .artifacts import ArtifactStore, CandidateArtifactPaths
from .harness import HarnessExecutionError, SubprocessHarness
from .models import (
    AutoCommitResult,
    CaseEvaluation,
    DiagnosticArtifact,
    DiagnosticEvidence,
    DiagnosticResult,
    ExperimentSummary,
    InputCase,
    JsonValue,
    KernelOptimizationRequest,
    KernelOptimizationResult,
    PerformanceSummary,
    ResearchEvidence,
    is_promotable,
    passes_protected_cases,
    per_case_speedups,
    weighted_geometric_speedup,
)
from .profiling import (
    ProfileRequest,
    compact_profile_summary,
    diagnostic_capabilities,
    extract_native_profiler_duration_us,
    is_profile_fresh,
    is_valid_intra_kernel_evidence,
    native_profiler_regression_diagnostic,
)
from .profiling_policy import (
    adaptive_candidate_profile_decision,
    full_diagnostic_request,
    targeted_diagnostic_decision,
)
from .providers import (
    AgentDiagnosticRequest,
    AgentSourceResearchRequest,
    CandidateContext,
    CandidateProposal,
    CandidateProvider,
    CodexCandidateProvider,
    CodexDiagnosticInstrumentationProvider,
    DiagnosticInstrumentationContext,
    DiagnosticInstrumentationProvider,
)
from .source_research import (
    CodexSourceResearchProvider,
    SourceResearchContext,
    SourceResearchProvider,
)
from .source import (
    source_digest,
    validate_candidate_source,
    validate_diagnostic_instrumentation_source,
)

_PROFILE_TOOLS = ("proton_launch", "native_profiler")
_DIAGNOSTIC_PROFILE_TOOLS = ("proton_intra_kernel",)
_DIAGNOSTIC_PROFILE_KEY = "diagnostic_proton_intra_kernel"
_NEAR_THRESHOLD_WINDOW = 0.01
_DIAGNOSTIC_SCHEMA_VERSION = "agent_diagnostic_v1"
_DIAGNOSTIC_SUMMARY_OMIT_KEYS = frozenset(
    {
        "artifact",
        "artifacts",
        "command",
        "commands",
        "environment",
        "instrumentation_mapping",
        "instrumentation_mapping_path",
        "instrumented_source",
        "instrumented_source_path",
        "kernel_source",
        "raw",
        "raw_metrics",
        "stderr",
        "stdout",
        "trace_events",
        "trace_path",
    }
)


class _GlobalDeadlineExceeded(RuntimeError):
    pass


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
    source_digest: str = "",
    policy_reason: str = "legacy",
) -> ProfileRequest:
    return ProfileRequest(
        level=level,
        tools=_PROFILE_TOOLS,
        experiment_id=experiment_id,
        artifacts_dir=artifacts_dir
        / "experiments"
        / experiment_id
        / "profile_artifacts",
        reason=reason,
        source_digest=source_digest,
        policy_reason=policy_reason,
    )


def _diagnostic_profile_request(
    artifacts_dir: Path,
    experiment_id: str,
    *,
    reason: str,
    source_digest: str = "",
    passes: tuple[str, ...] = (),
) -> ProfileRequest:
    return ProfileRequest(
        level="deep",
        tools=_DIAGNOSTIC_PROFILE_TOOLS,
        passes=passes,
        experiment_id=experiment_id,
        artifacts_dir=artifacts_dir
        / "experiments"
        / experiment_id
        / "diagnostic_profile_artifacts",
        reason=reason,
        source_digest=source_digest,
        policy_reason="legacy_diagnostic",
        diagnostic_only=True,
        granularity="warp",
    )


def _profiles_by_case(performance: PerformanceSummary) -> dict[str, dict[str, Any]]:
    return {
        evaluation.case_id: dict(evaluation.profile) for evaluation in performance.cases
    }


def _diagnostic_error_profiles(
    cases: tuple[InputCase, ...], error: Exception
) -> dict[str, dict[str, Any]]:
    return {
        case.case_id: {"error": f"{type(error).__name__}: {error}"} for case in cases
    }


def _collect_diagnostic_profiles(
    harness: SubprocessHarness,
    source: str,
    request: KernelOptimizationRequest,
    artifacts_dir: Path,
    experiment_id: str,
    *,
    reason: str,
    instrumentation_provider: DiagnosticInstrumentationProvider | None = None,
    profile_request: ProfileRequest | None = None,
    cases: tuple[InputCase, ...] | None = None,
    pass_capabilities: Mapping[str, tuple[str, ...]] | None = None,
    instrumentation_rationale: str = "",
    previous_diagnostics: tuple[str, ...] = (),
    check_deadline: Callable[[], None] | None = None,
    collection_metadata: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    selected_cases = cases or request.cases
    try:
        diagnostic_request = profile_request or _diagnostic_profile_request(
            artifacts_dir,
            experiment_id,
            reason=reason,
            source_digest=source_digest(source),
        )
        if instrumentation_provider is None:
            performance = harness.profile_only(
                source,
                selected_cases,
                request.target,
                diagnostic_request,
            )
            return _profiles_by_case(performance)

        uninstrumented_digest = source_digest(source)
        if diagnostic_request.source_digest != uninstrumented_digest:
            raise ValueError("diagnostic request source digest is stale")
        capabilities = pass_capabilities or {
            pass_name: () for pass_name in diagnostic_request.passes
        }
        if check_deadline is not None:
            check_deadline()
        proposal = instrumentation_provider.instrument(
            request,
            DiagnosticInstrumentationContext(
                current_source=source,
                requested_passes=diagnostic_request.passes,
                pass_capabilities=capabilities,
                case_id=",".join(case.case_id for case in selected_cases),
                rationale=instrumentation_rationale or reason,
                previous_diagnostics=previous_diagnostics,
            ),
        )
        if check_deadline is not None:
            check_deadline()
        validate_diagnostic_instrumentation_source(
            proposal.source,
            source,
            proposal.mapping,
            pass_capabilities=capabilities,
        )
        mapping_text = json.dumps(proposal.mapping, indent=2, sort_keys=True) + "\n"
        mapping_root = diagnostic_request.artifacts_dir
        if mapping_root is None:
            mapping_root = (
                artifacts_dir
                / "experiments"
                / experiment_id
                / "diagnostic_profile_artifacts"
            )
        mapping_root.mkdir(parents=True, exist_ok=True)
        mapping_path = (mapping_root / "instrumentation_mapping.json").resolve()
        mapping_path.write_text(mapping_text)
        mapping_digest = hashlib.sha256(mapping_text.encode()).hexdigest()
        instrumented_digest = source_digest(proposal.source)
        if collection_metadata is not None:
            collection_metadata.update(
                {
                    "instrumented_source_digest": instrumented_digest,
                    "instrumentation_mapping_digest": mapping_digest,
                }
            )
        instrumented_request = replace(
            diagnostic_request,
            source_digest=instrumented_digest,
        )
        if check_deadline is not None:
            check_deadline()
        performance = harness.profile_only(
            proposal.source,
            selected_cases,
            request.target,
            instrumented_request,
        )
        if check_deadline is not None:
            check_deadline()
    except _GlobalDeadlineExceeded:
        raise
    except Exception as error:  # noqa: BLE001
        print(
            f"[tlx-agent] {experiment_id} diagnostic instrumentation unavailable: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return _diagnostic_error_profiles(selected_cases, error)
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
    hypothesis_kind = getattr(proposal, "hypothesis_kind", "unknown")
    fields = (
        ("hypothesis", getattr(proposal, "hypothesis", "")),
        (
            "hypothesis_kind",
            hypothesis_kind if hypothesis_kind != "unknown" else "",
        ),
        ("escalation", getattr(proposal, "escalation_reason", "")),
        ("research", getattr(proposal, "research_evidence_ids", ())),
        ("evidence", getattr(proposal, "evidence", "")),
        ("change", getattr(proposal, "summary", "")),
        ("expected", getattr(proposal, "expected_effect", "")),
        ("risk", getattr(proposal, "risk", "")),
    )
    details = (
        " ".join(f"{name}={value!r}" for name, value in fields if value)
        or "change='candidate source edited'"
    )
    print(f"[tlx-agent] {experiment_id} {details}", file=sys.stderr, flush=True)


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


def _diagnostic_intra_kernel_profile(
    profile: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    compact = compact_profile_summary(profile)
    diagnostic = compact.get(_DIAGNOSTIC_PROFILE_KEY)
    if not isinstance(diagnostic, Mapping):
        diagnostic = profile.get(_DIAGNOSTIC_PROFILE_KEY)
    if not isinstance(diagnostic, Mapping):
        return None
    intra = diagnostic.get(_DIAGNOSTIC_PROFILE_KEY)
    if isinstance(intra, Mapping):
        return intra
    summary = diagnostic.get("summary")
    diagnostic_keys = {
        "valid",
        "selected_cta",
        "logical_coordinates",
        "task_spans",
        "top_waits",
        "dominant_waits",
        "key_overlaps",
        "missing_scopes",
    }
    if isinstance(summary, Mapping) and diagnostic_keys.intersection(summary):
        return summary
    return diagnostic


def _named_duration(value: object, *, default_name: str = "unknown") -> str | None:
    if not isinstance(value, Mapping):
        return None
    name = next(
        (
            str(value[key])
            for key in ("name", "task", "scope", "pair", "tasks", "scopes")
            if value.get(key) not in (None, "")
        ),
        default_name,
    )
    if name == default_name:
        endpoints = [
            str(value[key])
            for key in ("source", "target", "producer", "consumer")
            if value.get(key) not in (None, "")
        ]
        if endpoints:
            name = "->".join(endpoints)
    duration = next(
        (
            parsed
            for key in ("duration_us", "duration", "overlap_us")
            if (parsed := _coerce_float(value.get(key))) is not None
        ),
        None,
    )
    return f"{name}:{duration:.3f}us" if duration is not None else name


def _named_duration_list(value: object, *, limit: int = 3) -> str:
    if isinstance(value, Mapping):
        entries = [
            {"name": name, "duration_us": duration}
            for name, duration in value.items()
        ]
    elif isinstance(value, list | tuple):
        entries = list(value)
    else:
        return ""
    formatted = [
        text
        for entry in entries[:limit]
        if (text := _named_duration(entry)) is not None
    ]
    return ";".join(formatted)


def _profile_log_parts(profile: Mapping[str, Any]) -> list[str]:
    if not profile:
        return ["native_profiler=unavailable"]
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
    profiler_name, profiler_duration = extract_native_profiler_duration_us(compact)
    if profiler_duration is None:
        profiler_name, profiler_duration = extract_native_profiler_duration_us(profile)
    if profiler_name is not None and profiler_duration is not None:
        parts.append(f"{profiler_name.lower()}.duration_us={profiler_duration:.3f}")
    else:
        parts.append("native_profiler=unavailable")
    att = compact.get("fb_att")
    if isinstance(att, Mapping):
        valid = att.get("valid")
        if isinstance(valid, bool):
            parts.append(f"fb_att.valid={str(valid).lower()}")
        att_artifacts = att.get("artifacts")
        if isinstance(att_artifacts, Mapping):
            ui_directories = att_artifacts.get("ui_directories")
            if isinstance(ui_directories, list) and ui_directories:
                parts.append(f"fb_att.ui={ui_directories[0]}")
        if att.get("error"):
            parts.append(f"fb_att.error={att['error']}")
    intra = _diagnostic_intra_kernel_profile(profile)
    if intra is not None:
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
        task_spans = _named_duration_list(intra.get("task_spans"), limit=5)
        if task_spans:
            parts.append(f"proton.intra.task_spans={task_spans}")
        waits = intra.get("top_waits", intra.get("dominant_waits"))
        top_waits = _named_duration_list(waits)
        if top_waits:
            parts.append(f"proton.intra.top_waits={top_waits}")
        if isinstance(waits, list | tuple) and waits:
            dominant_wait = _named_duration(waits[0])
            if dominant_wait:
                parts.append(f"proton.intra.dominant_wait={dominant_wait}")
        key_overlaps = _named_duration_list(intra.get("key_overlaps"))
        if key_overlaps:
            parts.append(f"proton.intra.key_overlaps={key_overlaps}")
        missing_scopes = intra.get("missing_scopes")
        if isinstance(missing_scopes, list | tuple) and missing_scopes:
            parts.append(
                "proton.intra.missing_scopes="
                + ";".join(str(scope) for scope in missing_scopes[:5])
            )
        diagnostic_error = intra.get("error")
        if diagnostic_error:
            parts.append(f"proton.intra.error={diagnostic_error}")
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


def _baseline_diagnostic_evidence(
    performance: PerformanceSummary,
) -> tuple[str, ...]:
    evidence = []
    for evaluation in performance.cases:
        parts = [
            part
            for part in _profile_log_parts(evaluation.profile)
            if part.startswith("proton.intra.")
            and not part.startswith("proton.intra.trace=")
        ]
        if parts:
            evidence.append(f"- {evaluation.case_id}: " + ", ".join(parts))
    return tuple(evidence)


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
        f"hypothesis_kind={getattr(proposal, 'hypothesis_kind', 'unknown')!r}",
        f"escalation_reason={getattr(proposal, 'escalation_reason', '')!r}",
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
    return (
        threshold - _NEAR_THRESHOLD_WINDOW
        <= speedup
        <= threshold + _NEAR_THRESHOLD_WINDOW
    )


def _targeted_diagnostic_case(
    request: KernelOptimizationRequest,
    requested_case_ids: tuple[str, ...],
) -> InputCase:
    requested = set(requested_case_ids)
    eligible = [case for case in request.cases if not requested or case.case_id in requested]
    if not eligible:
        eligible = list(request.cases)
    return max(eligible, key=lambda case: (case.protected, case.weight))


def _collect_targeted_diagnostic(
    harness: SubprocessHarness,
    source: str,
    performance: PerformanceSummary,
    proposal: CandidateProposal,
    request: KernelOptimizationRequest,
    baseline: PerformanceSummary,
    baseline_digest: str,
    artifacts_dir: Path,
    experiment_id: str,
    diagnostic_passes_used: int,
) -> tuple[PerformanceSummary, int, str]:
    if not proposal.profiling_hint.pass_name:
        return performance, diagnostic_passes_used, ""
    selected_case = _targeted_diagnostic_case(
        request, proposal.profiling_hint.case_ids
    )
    baseline_evaluation = next(
        evaluation
        for evaluation in baseline.cases
        if evaluation.case_id == selected_case.case_id
    )
    baseline_diagnostic = baseline_evaluation.profile
    capabilities = (
        diagnostic_capabilities(baseline_diagnostic)
        if is_profile_fresh(
            baseline_diagnostic,
            baseline_digest,
            allow_legacy=False,
        )
        and is_valid_intra_kernel_evidence(baseline_diagnostic)
        else {}
    )
    remaining_pass_budget = max(
        0,
        request.budget.max_diagnostic_proton_passes
        - diagnostic_passes_used
        - 4,
    )
    decision = targeted_diagnostic_decision(
        True,
        proposal.profiling_hint.pass_name,
        proposal.profiling_hint.expected_regions,
        capabilities,
        baseline_diagnostic,
        request.budget.min_speedup,
        remaining_pass_budget,
        artifacts_dir,
        experiment_id,
        source_digest(source),
    )
    if decision.request is None:
        return performance, diagnostic_passes_used, decision.reason
    profiles = _collect_diagnostic_profiles(
        harness,
        source,
        request,
        artifacts_dir,
        experiment_id,
        reason="targeted_diagnostic",
        profile_request=decision.request,
        cases=(selected_case,),
    )
    return (
        _merge_diagnostic_profiles(performance, profiles),
        diagnostic_passes_used + len(decision.request.passes),
        decision.reason,
    )


def _native_profiler_regression_diagnostics(
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


def _check_global_deadline(start_time: float, max_total_seconds: float) -> None:
    if time.monotonic() - start_time >= max_total_seconds:
        raise _GlobalDeadlineExceeded("global optimization deadline exhausted")


def _target_identity(request: KernelOptimizationRequest) -> str:
    payload = {
        "backend": request.target.backend,
        "architecture": request.target.architecture,
        "device": request.target.device,
        "environment": dict(sorted(request.target.environment.items())),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"target-v1:{digest}"


def _canonical_agent_diagnostic_key(
    action: AgentDiagnosticRequest,
    target_identity: str,
) -> str:
    payload = {
        "schema": _DIAGNOSTIC_SCHEMA_VERSION,
        "source_digest": action.source_digest,
        "target_identity": target_identity,
        "tool": "ncu" if action.tool == "ncu" else "proton_intra_kernel",
        "level": action.ncu_level or "deep",
        "focus": sorted(_resolved_ncu_focus(action.ncu_focus)),
        "passes": sorted(action.proton_passes),
        "case_ids": sorted(action.case_ids),
        "expected_regions": sorted(action.expected_regions),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _resolved_ncu_focus(focus: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("cache_behavior" if item == "cache" else item for item in focus)


def build_diagnostic_ncu_profile_request(
    *,
    level: str | None,
    focus: tuple[str, ...],
    case_ids: tuple[str, ...],
    allowed_case_ids: Iterable[str] | Mapping[str, object],
    experiment_id: str,
    artifacts_dir: Path,
    reason: str,
    source_digest: str,
) -> ProfileRequest:
    allowed = (
        set(allowed_case_ids.keys())
        if isinstance(allowed_case_ids, Mapping)
        else set(allowed_case_ids)
    )
    unknown_cases = sorted(set(case_ids) - allowed)
    if unknown_cases:
        raise ValueError("unknown diagnostic case IDs: " + ", ".join(unknown_cases))
    if any(not item for item in focus):
        raise ValueError("NCU focus entries must be non-empty strings")
    return ProfileRequest(
        level=level or "deep",
        tools=("native_profiler",),
        experiment_id=experiment_id,
        artifacts_dir=artifacts_dir,
        reason=reason,
        source_digest=source_digest,
        policy_reason="agent_diagnostic",
    )


def _compact_diagnostic_value(value: object, depth: int = 0) -> JsonValue:
    if depth >= 8:
        return "<omitted>"
    if isinstance(value, Mapping):
        compact: dict[str, JsonValue] = {}
        for key, item in sorted(value.items(), key=lambda entry: str(entry[0])):
            name = str(key)
            if name in _DIAGNOSTIC_SUMMARY_OMIT_KEYS:
                continue
            compact[name] = _compact_diagnostic_value(item, depth + 1)
            if len(compact) >= 64:
                break
        return compact
    if isinstance(value, list | tuple):
        return [_compact_diagnostic_value(item, depth + 1) for item in value[:64]]
    if value is None or isinstance(value, bool | int | float):
        return value
    return str(value)[:512]


def _find_diagnostic_value(value: object, key: str) -> object | None:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for item in value.values():
            found = _find_diagnostic_value(item, key)
            if found is not None:
                return found
    elif isinstance(value, list | tuple):
        for item in value:
            found = _find_diagnostic_value(item, key)
            if found is not None:
                return found
    return None


def _diagnostic_result(
    tool: str,
    profiles: Mapping[str, Mapping[str, Any]],
    artifacts: tuple[DiagnosticArtifact, ...],
) -> DiagnosticResult:
    compact = {
        case_id: _compact_diagnostic_value(compact_profile_summary(dict(profile)))
        for case_id, profile in sorted(profiles.items())
    }
    failures: list[str] = []
    if not profiles:
        failures.append("diagnostic profile returned no case evidence")
    for case_id, profile in profiles.items():
        error = _find_diagnostic_value(profile, "error")
        if error:
            failures.append(f"{case_id}: {error}")
            continue
        if tool == "ncu":
            if _find_diagnostic_value(profile, "available") is not True:
                reason = _find_diagnostic_value(profile, "unavailable_reason")
                failures.append(f"{case_id}: {reason or 'NCU evidence unavailable'}")
        elif not is_valid_intra_kernel_evidence(profile):
            failures.append(f"{case_id}: Proton evidence unavailable")
    available = bool(profiles) and not failures
    schema = _find_diagnostic_value(profiles, "schema")
    parser_schema = _find_diagnostic_value(profiles, "parser_schema_version")
    return DiagnosticResult(
        available=available,
        summary=compact,
        failure="; ".join(failures)[:2000],
        artifacts=artifacts,
        tool_schema_version=str(schema or "")[:120],
        parser_schema_version=str(parser_schema or "")[:120],
    )


def _diagnostic_artifacts(
    root: Path,
    store: ArtifactStore,
) -> tuple[DiagnosticArtifact, ...]:
    if not root.is_dir():
        return ()
    artifacts: list[DiagnosticArtifact] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            reference = str(path.relative_to(store.root.resolve()))
        except (OSError, ValueError):
            continue
        artifacts.append(DiagnosticArtifact(reference=reference, sha256=digest))
        if len(artifacts) >= 32:
            break
    return tuple(artifacts)


def _benchmark_uses_native_profiler(evaluation: CaseEvaluation) -> bool:
    tool, duration = extract_native_profiler_duration_us(evaluation.profile)
    return (
        tool is not None
        and duration is not None
        and evaluation.timing is not None
        and tool.lower() in evaluation.timing.cache_policy.lower()
    )


def _diagnostic_feedback(evidence: DiagnosticEvidence) -> str:
    summary = json.dumps(
        evidence.result.summary, sort_keys=True, separators=(",", ":")
    )
    state = (
        "available"
        if evidence.result.available
        else evidence.result.failure or "unavailable"
    )
    return (
        f"{evidence.action_id}: diagnostic {evidence.tool} status={evidence.status} "
        f"result={state!r} summary={summary[:2500]}"
    )[:4000]


@dataclass
class _DiagnosticLedger:
    ncu_collections_used: int = 0
    proton_passes_used: int = 0
    cache: dict[str, DiagnosticEvidence] = field(default_factory=dict)
    history: list[DiagnosticEvidence] = field(default_factory=list)


def _record_diagnostic_evidence(
    store: ArtifactStore,
    ledger: _DiagnosticLedger,
    evidence: DiagnosticEvidence,
) -> None:
    ledger.history.append(evidence)
    store.write_diagnostic_evidence(evidence.action_id, evidence)
    store.write_diagnostic_history(tuple(ledger.history))


@dataclass
class _ResearchLedger:
    actions_used: int = 0
    history: list[ResearchEvidence] = field(default_factory=list)


def _research_feedback(evidence: ResearchEvidence) -> str:
    findings = json.dumps(evidence.findings, separators=(",", ":"))
    limitations = json.dumps(evidence.limitations, separators=(",", ":"))
    return (
        f"{evidence.action_id}: source research status={evidence.status} "
        f"findings={findings[:2500]} limitations={limitations[:1000]}"
    )[:4000]


def _record_research_evidence(
    store: ArtifactStore,
    ledger: _ResearchLedger,
    evidence: ResearchEvidence,
) -> None:
    ledger.history.append(evidence)
    store.write_research_evidence(evidence.action_id, evidence)
    store.write_research_history(tuple(ledger.history))


def _failed_research_evidence(
    action_id: str,
    action: AgentSourceResearchRequest,
    status: str,
    limitation: str,
    duration: float = 0.0,
) -> ResearchEvidence:
    return ResearchEvidence(
        action_id=action_id,
        status=status,
        source_digest=action.source_digest,
        question=action.question,
        rationale=action.rationale,
        limitations=(limitation[:2000],),
        collection_duration_seconds=duration,
    )


def _failed_diagnostic_evidence(
    action_id: str,
    action: AgentDiagnosticRequest,
    target_identity: str,
    canonical_key: str,
    status: str,
    failure: str,
    duration: float = 0.0,
) -> DiagnosticEvidence:
    return DiagnosticEvidence(
        action_id=action_id,
        status=status,
        source_digest=action.source_digest,
        target_identity=target_identity,
        tool=action.tool,
        case_ids=tuple(sorted(action.case_ids)),
        canonical_key=canonical_key,
        question=action.question,
        rationale=action.rationale,
        level=action.ncu_level or "deep",
        focus=tuple(sorted(_resolved_ncu_focus(action.ncu_focus))),
        passes=tuple(sorted(action.proton_passes)),
        expected_regions=tuple(sorted(action.expected_regions)),
        result=DiagnosticResult(failure=failure[:2000]),
        collection_duration_seconds=duration,
    )


def _execute_agent_diagnostic(
    harness: SubprocessHarness,
    instrumentation_provider: DiagnosticInstrumentationProvider | None,
    source: str,
    request: KernelOptimizationRequest,
    selected_cases: tuple[InputCase, ...],
    action: AgentDiagnosticRequest,
    action_id: str,
    canonical_key: str,
    target_identity: str,
    store: ArtifactStore,
    check_deadline: Callable[[], None],
) -> DiagnosticEvidence:
    action_root = (store.root / "diagnostics" / "actions" / action_id).resolve()
    profile_root = action_root / "profile_artifacts"
    started = time.monotonic()
    metadata: dict[str, str] = {}
    try:
        check_deadline()
        if action.tool == "ncu":
            profile_request = build_diagnostic_ncu_profile_request(
                level=action.ncu_level,
                focus=_resolved_ncu_focus(action.ncu_focus),
                case_ids=action.case_ids,
                allowed_case_ids=(case.case_id for case in request.cases),
                experiment_id=action_id,
                artifacts_dir=profile_root,
                reason=action.question,
                source_digest=action.source_digest,
            )
            check_deadline()
            performance = harness.evaluate(
                source,
                selected_cases,
                request.target,
                request.budget.benchmark_repetitions,
                profile=profile_request,
            )
            check_deadline()
            if {evaluation.case_id for evaluation in performance.cases} != {
                case.case_id for case in selected_cases
            }:
                raise ValueError("diagnostic profile returned unexpected case IDs")
            profiles = _profiles_by_case(performance)
        else:
            profile_request = ProfileRequest(
                level="deep",
                tools=_DIAGNOSTIC_PROFILE_TOOLS,
                passes=tuple(action.proton_passes),
                experiment_id=action_id,
                artifacts_dir=profile_root,
                reason=action.question,
                source_digest=action.source_digest,
                policy_reason="agent_diagnostic",
                diagnostic_only=True,
                granularity="warp",
            )
            profiles = _collect_diagnostic_profiles(
                harness,
                source,
                request,
                store.root,
                action_id,
                reason="agent_diagnostic",
                instrumentation_provider=instrumentation_provider,
                profile_request=profile_request,
                cases=selected_cases,
                pass_capabilities={
                    pass_name: tuple(action.expected_regions)
                    for pass_name in action.proton_passes
                },
                instrumentation_rationale=action.rationale,
                check_deadline=check_deadline,
                collection_metadata=metadata,
            )
        result = _diagnostic_result(
            action.tool,
            profiles,
            _diagnostic_artifacts(profile_root, store),
        )
        status = "succeeded" if result.available else "failed"
        return DiagnosticEvidence(
            action_id=action_id,
            status=status,
            source_digest=action.source_digest,
            target_identity=target_identity,
            tool=action.tool,
            case_ids=tuple(sorted(action.case_ids)),
            canonical_key=canonical_key,
            question=action.question,
            rationale=action.rationale,
            level=action.ncu_level or "deep",
            focus=tuple(sorted(_resolved_ncu_focus(action.ncu_focus))),
            passes=tuple(sorted(action.proton_passes)),
            expected_regions=tuple(sorted(action.expected_regions)),
            result=result,
            collection_duration_seconds=time.monotonic() - started,
            instrumented_source_digest=metadata.get(
                "instrumented_source_digest", ""
            ),
            instrumentation_mapping_digest=metadata.get(
                "instrumentation_mapping_digest", ""
            ),
        )
    except _GlobalDeadlineExceeded:
        raise
    except Exception as error:  # noqa: BLE001
        return _failed_diagnostic_evidence(
            action_id,
            action,
            target_identity,
            canonical_key,
            "failed",
            f"{type(error).__name__}: {error}",
            time.monotonic() - started,
        )


def _propose_candidate_for_slot(
    provider: CandidateProvider,
    harness: SubprocessHarness,
    instrumentation_provider: DiagnosticInstrumentationProvider | None,
    research_provider: SourceResearchProvider | None,
    request: KernelOptimizationRequest,
    store: ArtifactStore,
    ledger: _DiagnosticLedger,
    research_ledger: _ResearchLedger,
    *,
    round_index: int,
    candidate_index: int,
    parent_source: str,
    current_performance: PerformanceSummary,
    previous_diagnostics: tuple[str, ...],
    baseline_diagnostic_evidence: tuple[str, ...],
    local_search_failure_streak: int,
    start_time: float,
) -> CandidateProposal | None:
    current_digest = source_digest(parent_source)
    target_identity = _target_identity(request)
    case_by_id = {case.case_id: case for case in request.cases}
    slot_evidence: list[DiagnosticEvidence] = []
    slot_research_evidence: list[ResearchEvidence] = []
    slot_feedback: list[str] = []
    diagnostic_actions = 0
    research_actions = 0

    def check_deadline() -> None:
        _check_global_deadline(start_time, request.budget.max_total_seconds)

    for action_index in range(request.budget.max_agent_actions_per_candidate):
        check_deadline()
        try:
            action = provider.propose(
                request,
                CandidateContext(
                    round_index=round_index,
                    candidate_index=candidate_index,
                    current_source=parent_source,
                    current_performance=current_performance,
                    previous_diagnostics=(
                        previous_diagnostics + tuple(slot_feedback)
                    ),
                    baseline_diagnostic_evidence=baseline_diagnostic_evidence,
                    action_index=action_index,
                    current_source_digest=current_digest,
                    remaining_agent_actions=(
                        request.budget.max_agent_actions_per_candidate - action_index
                    ),
                    remaining_diagnostic_actions=max(
                        0,
                        request.budget.max_diagnostic_actions_per_candidate
                        - diagnostic_actions,
                    ),
                    remaining_ncu_collections=max(
                        0,
                        request.budget.max_diagnostic_ncu_collections
                        - ledger.ncu_collections_used,
                    ),
                    remaining_proton_passes=max(
                        0,
                        request.budget.max_diagnostic_proton_passes
                        - ledger.proton_passes_used,
                    ),
                    remaining_source_research_actions=(
                        max(
                            0,
                            min(
                                request.budget.max_source_research_actions_per_candidate
                                - research_actions,
                                request.budget.max_source_research_actions_total
                                - research_ledger.actions_used,
                            ),
                        )
                        if research_provider is not None
                        and request.repository_root is not None
                        and request.kernel_path is not None
                        else 0
                    ),
                    diagnostic_evidence=tuple(slot_evidence),
                    research_evidence=tuple(slot_research_evidence),
                    local_search_failure_streak=local_search_failure_streak,
                    source_research_saturation_threshold=(
                        request.budget.source_research_saturation_threshold
                    ),
                ),
            )
        except Exception as error:  # noqa: BLE001
            if slot_evidence or slot_research_evidence:
                print(
                    f"[tlx-agent] r{round_index:03d}-c{candidate_index:03d} "
                    "candidate provider failed after evidence collection: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            raise
        check_deadline()
        if isinstance(action, CandidateProposal):
            known_research_ids = {
                evidence.action_id for evidence in slot_research_evidence
            }
            unknown_research_ids = sorted(
                set(action.research_evidence_ids) - known_research_ids
            )
            if unknown_research_ids:
                raise ValueError(
                    "candidate cites unknown source research evidence IDs: "
                    + ", ".join(unknown_research_ids)
                )
            return action
        if action_index + 1 >= request.budget.max_agent_actions_per_candidate:
            raise ValueError(
                "the final agent action in a candidate slot must propose a candidate"
            )
        action_id = (
            f"r{round_index:03d}-c{candidate_index:03d}-a{action_index:02d}"
        )
        if isinstance(action, AgentSourceResearchRequest):
            store.write_research_request(action_id, action)
            failure = ""
            if (
                research_actions
                >= request.budget.max_source_research_actions_per_candidate
            ):
                failure = "source research action limit exhausted for candidate slot"
            elif (
                research_ledger.actions_used
                >= request.budget.max_source_research_actions_total
            ):
                failure = "source research run budget exhausted"
            elif action.source_digest != current_digest:
                failure = "source research request source digest is stale"
            elif research_provider is None:
                failure = "source research provider is unavailable"
            elif request.repository_root is None or request.kernel_path is None:
                failure = "source research repository identity is unavailable"

            if failure:
                research_evidence = _failed_research_evidence(
                    action_id, action, "denied", failure
                )
            else:
                assert research_provider is not None
                assert request.repository_root is not None
                assert request.kernel_path is not None
                research_actions += 1
                research_ledger.actions_used += 1
                started = time.monotonic()
                try:
                    remaining_total_seconds = (
                        request.budget.max_total_seconds
                        - (time.monotonic() - start_time)
                    )
                    if remaining_total_seconds <= 0:
                        raise _GlobalDeadlineExceeded(
                            "global optimization deadline exhausted"
                        )
                    research_evidence = research_provider.research(
                        action,
                        SourceResearchContext(
                            action_id=action_id,
                            repository_root=request.repository_root,
                            kernel_path=request.kernel_path,
                            current_source=parent_source,
                            target=request.target,
                            timeout_seconds=min(
                                request.budget.max_candidate_seconds,
                                remaining_total_seconds,
                            ),
                        ),
                    )
                    check_deadline()
                except _GlobalDeadlineExceeded:
                    raise
                except Exception as error:  # noqa: BLE001
                    research_evidence = _failed_research_evidence(
                        action_id,
                        action,
                        "failed",
                        f"{type(error).__name__}: {error}",
                        time.monotonic() - started,
                    )
            _record_research_evidence(store, research_ledger, research_evidence)
            slot_research_evidence.append(research_evidence)
            slot_feedback.append(_research_feedback(research_evidence))
            continue

        if not isinstance(action, AgentDiagnosticRequest):
            raise TypeError("candidate provider returned an unsupported action")

        diagnostic_actions += 1
        store.write_diagnostic_request(action_id, action)
        canonical_key = _canonical_agent_diagnostic_key(action, target_identity)
        failure = ""
        if diagnostic_actions > request.budget.max_diagnostic_actions_per_candidate:
            failure = "diagnostic action limit exhausted for candidate slot"
        elif action.source_digest != current_digest:
            failure = "diagnostic request source digest is stale"
        else:
            unknown_cases = sorted(set(action.case_ids) - case_by_id.keys())
            if unknown_cases:
                failure = "unknown diagnostic case IDs: " + ", ".join(unknown_cases)
            elif action.tool == "ncu":
                try:
                    build_diagnostic_ncu_profile_request(
                        level=action.ncu_level,
                        focus=_resolved_ncu_focus(action.ncu_focus),
                        case_ids=action.case_ids,
                        allowed_case_ids=case_by_id,
                        experiment_id=action_id,
                        artifacts_dir=(
                            store.root
                            / "diagnostics"
                            / "actions"
                            / action_id
                            / "profile_artifacts"
                        ).resolve(),
                        reason=action.question,
                        source_digest=action.source_digest,
                    )
                except ValueError as error:
                    failure = str(error)

        cached = ledger.cache.get(canonical_key)
        if not failure and cached is not None:
            evidence = replace(cached, action_id=action_id, status="cached")
        elif failure:
            evidence = _failed_diagnostic_evidence(
                action_id,
                action,
                target_identity,
                canonical_key,
                "denied",
                failure,
            )
        elif action.tool == "ncu" and (
            ledger.ncu_collections_used
            >= request.budget.max_diagnostic_ncu_collections
        ):
            evidence = _failed_diagnostic_evidence(
                action_id,
                action,
                target_identity,
                canonical_key,
                "denied",
                "NCU diagnostic budget exhausted",
            )
        elif action.tool == "proton" and (
            ledger.proton_passes_used + len(action.proton_passes)
            > request.budget.max_diagnostic_proton_passes
        ):
            evidence = _failed_diagnostic_evidence(
                action_id,
                action,
                target_identity,
                canonical_key,
                "denied",
                "Proton diagnostic pass budget exhausted",
            )
        else:
            if action.tool == "ncu":
                ledger.ncu_collections_used += 1
            else:
                ledger.proton_passes_used += len(action.proton_passes)
            selected_cases = tuple(case_by_id[case_id] for case_id in action.case_ids)
            try:
                evidence = _execute_agent_diagnostic(
                    harness,
                    instrumentation_provider,
                    parent_source,
                    request,
                    selected_cases,
                    action,
                    action_id,
                    canonical_key,
                    target_identity,
                    store,
                    check_deadline,
                )
            except _GlobalDeadlineExceeded as error:
                evidence = _failed_diagnostic_evidence(
                    action_id,
                    action,
                    target_identity,
                    canonical_key,
                    "failed",
                    str(error),
                )
                ledger.cache[canonical_key] = evidence
                _record_diagnostic_evidence(store, ledger, evidence)
                raise
            ledger.cache[canonical_key] = evidence

        _record_diagnostic_evidence(store, ledger, evidence)
        slot_evidence.append(evidence)
        slot_feedback.append(_diagnostic_feedback(evidence))
    return None


def _post_promotion_diagnostic_comparison(
    proposal: CandidateProposal,
    source_digest_value: str,
) -> None:
    del proposal, source_digest_value
    # Candidate validation intents can attach comparison collection here later.


class KernelOptimizer:
    def __init__(
        self,
        provider: CandidateProvider | None = None,
        *,
        diagnostic_instrumentation_provider: (
            DiagnosticInstrumentationProvider | None
        ) = None,
        source_research_provider: SourceResearchProvider | None = None,
    ) -> None:
        self._provider = provider or CodexCandidateProvider()
        if diagnostic_instrumentation_provider is not None:
            self._diagnostic_instrumentation_provider = (
                diagnostic_instrumentation_provider
            )
        elif isinstance(self._provider, CodexCandidateProvider):
            self._diagnostic_instrumentation_provider = (
                CodexDiagnosticInstrumentationProvider(
                    model=self._provider.model,
                    timeout_seconds=self._provider.timeout_seconds,
                )
            )
        else:
            self._diagnostic_instrumentation_provider = None
        if source_research_provider is not None:
            self._source_research_provider = source_research_provider
        elif isinstance(self._provider, CodexCandidateProvider):
            self._source_research_provider = CodexSourceResearchProvider(
                model=self._provider.model,
                timeout_seconds=self._provider.timeout_seconds,
            )
        else:
            self._source_research_provider = None

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
        baseline_digest = source_digest(request.kernel_source)
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
                source_digest=baseline_digest,
                policy_reason=request.profiling_policy,
            ),
        )
        if not passes_protected_cases(baseline, request.cases):
            raise HarnessExecutionError(
                "baseline failed one or more protected correctness cases"
            )
        baseline = replace(baseline, aggregate_speedup=1.0)
        diagnostic_passes_used = 0
        use_diagnostic_proton = request.use_diagnostic_proton_intra_kernel
        if use_diagnostic_proton:
            baseline_diagnostic_request = full_diagnostic_request(
                artifacts_dir,
                "baseline",
                baseline_digest,
                reason="baseline_diagnostic",
                pass_budget=request.budget.max_diagnostic_proton_passes,
            )
            if baseline_diagnostic_request is not None:
                baseline_diagnostics = _collect_diagnostic_profiles(
                    harness,
                    request.kernel_source,
                    request,
                    artifacts_dir,
                    "baseline",
                    reason="baseline_diagnostic",
                    profile_request=baseline_diagnostic_request,
                )
                baseline = _merge_diagnostic_profiles(baseline, baseline_diagnostics)
                diagnostic_passes_used += len(baseline_diagnostic_request.passes)
        baseline_diagnostic_evidence = _baseline_diagnostic_evidence(baseline)
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

        best_source = request.kernel_source
        best_performance = baseline
        best_experiment_id = "baseline"
        best_commit_title = ""
        best_commit_summary = ""
        best_profiles = baseline_profiles
        diagnostics: list[str] = []
        diagnostic_ledger = _DiagnosticLedger()
        research_ledger = _ResearchLedger()
        local_search_failure_streak = 0
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
                    proposal = _propose_candidate_for_slot(
                        self._provider,
                        harness,
                        self._diagnostic_instrumentation_provider,
                        self._source_research_provider,
                        request,
                        store,
                        diagnostic_ledger,
                        research_ledger,
                        round_index=round_index,
                        candidate_index=candidate_index,
                        parent_source=parent_source,
                        current_performance=best_performance,
                        previous_diagnostics=tuple(diagnostics),
                        baseline_diagnostic_evidence=baseline_diagnostic_evidence,
                        local_search_failure_streak=local_search_failure_streak,
                        start_time=start_time,
                    )
                    if proposal is None:
                        raise RuntimeError(
                            "candidate slot exhausted without a candidate proposal"
                        )
                    validate_candidate_source(proposal.source, parent_source)
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
                    _report_performance(experiment_id, "evaluating", None)
                    candidate_digest = source_digest(proposal.source)
                    if request.profiling_policy == "legacy":
                        candidate_profile_request = _profile_request(
                            artifacts_dir,
                            experiment_id,
                            level="summary",
                            reason="candidate",
                            source_digest=candidate_digest,
                        )
                    else:
                        candidate_profile_request = None
                    performance = harness.evaluate(
                        proposal.source,
                        request.cases,
                        request.target,
                        request.budget.benchmark_repetitions,
                        profile=candidate_profile_request,
                    )
                    speedup = weighted_geometric_speedup(
                        baseline, performance, request.cases
                    )
                    performance = replace(performance, aggregate_speedup=speedup)
                    if request.profiling_policy == "adaptive":
                        profile_decision = adaptive_candidate_profile_decision(
                            _is_correct_and_stable(
                                performance,
                                request.budget,
                                request.cases,
                            ),
                            speedup,
                            request.budget.min_speedup,
                            artifacts_dir,
                            experiment_id,
                            candidate_digest,
                        )
                        if profile_decision.request is not None:
                            _report_performance(
                                experiment_id,
                                "adaptive-profiling",
                                performance,
                                baseline=baseline,
                                cases=request.cases,
                                diagnostics=f"reason={profile_decision.reason}",
                            )
                            performance = harness.evaluate(
                                proposal.source,
                                request.cases,
                                request.target,
                                request.budget.benchmark_repetitions,
                                profile=profile_decision.request,
                            )
                            speedup = weighted_geometric_speedup(
                                baseline, performance, request.cases
                            )
                            performance = replace(
                                performance, aggregate_speedup=speedup
                            )
                    elif _is_correct_and_stable(
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
                                source_digest=candidate_digest,
                            ),
                        )
                        speedup = weighted_geometric_speedup(
                            baseline, performance, request.cases
                        )
                        performance = replace(performance, aggregate_speedup=speedup)
                    # Persist per-case profiles for this candidate regardless of promotion.
                    perf_profiles = _profiles_by_case(performance)
                    profile_path = store.write_profile(experiment_id, perf_profiles)
                    profiler_diagnostics = _native_profiler_regression_diagnostics(
                        baseline, performance
                    )
                    status = (
                        "promoted"
                        if not profiler_diagnostics
                        and is_promotable(performance, request.budget, request.cases)
                        and speedup > best_performance.aggregate_speedup
                        else "rejected"
                    )
                    targeted_profile_reason = ""
                    if (
                        status == "promoted"
                        and not request.auto_test
                        and request.profiling_policy == "adaptive"
                        and use_diagnostic_proton
                    ):
                        (
                            performance,
                            diagnostic_passes_used,
                            targeted_profile_reason,
                        ) = _collect_targeted_diagnostic(
                            harness,
                            proposal.source,
                            performance,
                            proposal,
                            request,
                            baseline,
                            baseline_digest,
                            artifacts_dir,
                            experiment_id,
                            diagnostic_passes_used,
                        )
                        perf_profiles = _profiles_by_case(performance)
                        profile_path = store.write_profile(
                            experiment_id, perf_profiles
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
                        hypothesis_kind=proposal.hypothesis_kind,
                        escalation_reason=proposal.escalation_reason,
                        research_evidence_ids=proposal.research_evidence_ids,
                        evidence=proposal.evidence,
                        expected_effect=proposal.expected_effect,
                        risk=proposal.risk,
                        commit_title=proposal.commit_title,
                        commit_summary=proposal.commit_summary,
                        profile_path=profile_path,
                    )
                    rejection_diagnostics = "; ".join(
                        f"{evaluation.case_id}: {evaluation.verification.diagnostics}"
                        for evaluation in performance.cases
                        if not evaluation.verification.passed
                        and evaluation.verification.diagnostics
                    )
                    decision = (
                        "correct and exceeded speedup threshold"
                        if status == "promoted"
                        else profiler_diagnostics
                        or rejection_diagnostics
                        or f"speedup below {request.budget.min_speedup:.4f}x threshold"
                    )
                    if status == "rejected":
                        collected_research_ids = {
                            evidence.action_id
                            for evidence in research_ledger.history
                            if evidence.status == "collected"
                        }
                        used_collected_research = bool(
                            collected_research_ids.intersection(
                                proposal.research_evidence_ids
                            )
                        )
                        if used_collected_research:
                            local_search_failure_streak = 0
                        else:
                            local_search_failure_streak += 1
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
                    else:
                        local_search_failure_streak = 0
                    _report_performance(
                        experiment_id,
                        status,
                        performance,
                        baseline=baseline,
                        cases=request.cases,
                        diagnostics=(
                            f"decision={decision}"
                            + (
                                f" targeted_proton={targeted_profile_reason}"
                                if targeted_profile_reason
                                else ""
                            )
                        ),
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
                    # Protocol and infrastructure failures do not establish that a
                    # source hypothesis was rejected by authoritative evaluation.
                    # Preserve the local-search streak until a measured result exists.
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
                        mutation_summary=proposal.summary if proposal is not None else "",
                        hypothesis=proposal.hypothesis if proposal is not None else "",
                        hypothesis_kind=(
                            proposal.hypothesis_kind
                            if proposal is not None
                            else "unknown"
                        ),
                        escalation_reason=(
                            proposal.escalation_reason if proposal is not None else ""
                        ),
                        research_evidence_ids=(
                            proposal.research_evidence_ids if proposal is not None else ()
                        ),
                    )
                    _report_performance(
                        experiment_id,
                        "failed",
                        None,
                        diagnostics=message,
                    )
                experiments.append(experiment)
                store.write_json(f"experiments/{experiment_id}/result.json", experiment)
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
                source_digest=source_digest(best_source),
                policy_reason=request.profiling_policy,
            ),
        )
        final_profile = replace(
            final_profile,
            aggregate_speedup=weighted_geometric_speedup(
                baseline, final_profile, request.cases
            ),
        )
        final_profiler_diagnostics = _native_profiler_regression_diagnostics(
            baseline, final_profile
        )
        if best_experiment_id != "baseline" and (
            final_profiler_diagnostics
            or not is_promotable(final_profile, request.budget, request.cases)
        ):
            finalist_revalidation = final_profile
            if use_diagnostic_proton:
                rejected_request = full_diagnostic_request(
                    artifacts_dir,
                    "finalist_revalidation",
                    source_digest(best_source),
                    reason="rejected_finalist_diagnostic",
                    pass_budget=max(
                        0,
                        request.budget.max_diagnostic_proton_passes
                        - diagnostic_passes_used,
                    ),
                )
                if rejected_request is not None:
                    rejected_final_diagnostics = _collect_diagnostic_profiles(
                        harness,
                        best_source,
                        request,
                        artifacts_dir,
                        "finalist_revalidation",
                        reason="rejected_finalist_diagnostic",
                        profile_request=rejected_request,
                    )
                    finalist_revalidation = _merge_diagnostic_profiles(
                        finalist_revalidation,
                        rejected_final_diagnostics,
                    )
            store.write_profile(
                "finalist_revalidation",
                _profiles_by_case(finalist_revalidation),
            )
            finalist_rejection = (
                final_profiler_diagnostics
                or "final performance did not pass correctness, speedup, or CV gates"
            )
            diagnostics.append(f"final: rejected: {finalist_rejection}")
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
            if use_diagnostic_proton:
                final_diagnostic_request = full_diagnostic_request(
                    artifacts_dir,
                    "final",
                    source_digest(best_source),
                    reason="final_winner_diagnostic",
                    pass_budget=max(
                        0,
                        request.budget.max_diagnostic_proton_passes
                        - diagnostic_passes_used,
                    ),
                )
                if final_diagnostic_request is not None:
                    final_diagnostics = _collect_diagnostic_profiles(
                        harness,
                        best_source,
                        request,
                        artifacts_dir,
                        "final",
                        reason="final_winner_diagnostic",
                        profile_request=final_diagnostic_request,
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
            diagnostics=final_profiler_diagnostics,
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
