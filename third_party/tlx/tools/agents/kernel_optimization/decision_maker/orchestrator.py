from __future__ import annotations

import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .artifacts import ArtifactStore
from .evaluation import (
    collect_diagnostic_profiles as _collect_diagnostic_profiles,
    merge_diagnostic_profiles as _merge_diagnostic_profiles,
    profile_request as _profile_request,
    profiles_by_case as _profiles_by_case,
)
from .harness import HarnessExecutionError, SubprocessHarness
from ..contracts import (
    AutoCommitResult,
    BlastRadius,
    CandidateSubmission,
    Decision,
    DecisionStatus,
    ExperimentKind,
    ExperimentSummary,
    JsonValue,
    KernelOptimizationRequest,
    KernelOptimizationResult,
    PerformanceSummary,
)
from .policy import (
    ABLATION_KINDS,
    evaluated_decision,
    is_acceptable_winner,
    is_correct_and_stable as _is_correct_and_stable,
    is_near_threshold as _is_near_threshold,
    is_promotable,
    passes_protected_cases,
    profiler_regression_diagnostics as _profiler_regression_diagnostics,
    preflight_decision,
    weighted_geometric_speedup,
)
from .reporting import (
    profile_log_parts as _profile_log_parts,  # noqa: F401 - compatibility export
    rejection_feedback as _rejection_feedback,
    report_candidate_artifacts as _report_candidate_artifacts,
    report_candidate_summary as _report_candidate_summary,
    report_performance as _report_performance,
)
from ..optimizer.agent import CandidateContext, CandidateProvider, CodexCandidateProvider
from ..optimizer.source import source_digest


def _experiment_request(proposal: CandidateSubmission) -> dict[str, JsonValue]:
    return {
        "kind": proposal.experiment_kind.value,
        "change_scopes": sorted(scope.value for scope in proposal.change_scopes),
        "blast_radius": proposal.blast_radius.value,
        "payload": dict(proposal.experiment_payload),
    }

class PromotionCommitter(Protocol):
    def commit_promotion(
        self,
        experiment: ExperimentSummary,
        source: str,
        baseline: PerformanceSummary,
        performance: PerformanceSummary,
    ) -> AutoCommitResult: ...

    def rollback_to_baseline(self, diagnostics: str) -> AutoCommitResult: ...


class DecisionMaker:
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
        seen_experiments: set[str] = set()
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
                experiment_payload_path = None
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
                    experiment_payload_path = store.write_experiment_payload(
                        experiment_id,
                        proposal.experiment_payload,
                    )
                    if (
                        proposal.experiment_kind is ExperimentKind.HUMAN_REVIEW
                        and proposal.source != parent_source
                    ):
                        raise ValueError("human_review must not modify candidate source")
                    preflight = preflight_decision(proposal, request.target)
                    if preflight is not None:
                        experiment = ExperimentSummary(
                            experiment_id=experiment_id,
                            round_index=round_index,
                            parent_id=parent_id,
                            status="needs_human",
                            source_path=candidate_artifacts.source_path,
                            decision=preflight,
                            experiment_kind=proposal.experiment_kind,
                            change_scopes=tuple(
                                sorted(proposal.change_scopes, key=lambda scope: scope.value)
                            ),
                            blast_radius=proposal.blast_radius,
                            experiment_payload_path=experiment_payload_path,
                            incremental_patch_path=candidate_artifacts.incremental_patch_path,
                            cumulative_patch_path=candidate_artifacts.cumulative_patch_path,
                            mutation_summary=proposal.summary,
                            hypothesis=proposal.hypothesis,
                            evidence=proposal.evidence,
                            expected_effect=proposal.expected_effect,
                            risk=proposal.risk,
                        )
                        _report_performance(
                            experiment_id,
                            "needs_human",
                            None,
                            diagnostics=f"decision={preflight.rationale}",
                        )
                        diagnostics.append(
                            f"{experiment_id}: human review required: {preflight.rationale}"
                        )
                        stopping_reason = "human_review_requested"
                        exhausted = True
                    else:
                        digest = source_digest(proposal.source)
                        if proposal.experiment_kind is ExperimentKind.PROMOTABLE:
                            if digest in seen_sources:
                                origin = (
                                    "a prior run experiment"
                                    if digest in prior_source_hashes
                                    else "an earlier experiment"
                                )
                                raise ValueError(f"candidate source duplicates {origin}")
                            seen_sources.add(digest)
                        else:
                            fingerprint = source_digest(
                                proposal.experiment_kind.value
                                + "\0"
                                + proposal.source
                                + "\0"
                                + json.dumps(
                                    proposal.experiment_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                            )
                            if fingerprint in seen_experiments:
                                raise ValueError("experiment duplicates an earlier ablation")
                            seen_experiments.add(fingerprint)

                        is_ablation = proposal.experiment_kind in ABLATION_KINDS
                        _report_performance(experiment_id, "evaluating", None)
                        performance = harness.evaluate(
                            proposal.source,
                            request.cases,
                            request.target,
                            request.budget.benchmark_repetitions,
                            profile=_profile_request(
                                artifacts_dir,
                                experiment_id,
                                level="deep" if is_ablation else "summary",
                                reason=(
                                    proposal.experiment_kind.value
                                    if is_ablation
                                    else "candidate"
                                ),
                            ),
                            experiment=(
                                _experiment_request(proposal) if is_ablation else None
                            ),
                        )
                        speedup = weighted_geometric_speedup(
                            baseline, performance, request.cases
                        )
                        performance = replace(performance, aggregate_speedup=speedup)
                        if (
                            not is_ablation
                            and not bool(
                                request.target.evaluation_policy.get(
                                    "profile_complete_per_measurement", False
                                )
                            )
                            and _is_correct_and_stable(
                                performance,
                                request.budget,
                                request.cases,
                            )
                            and _is_near_threshold(
                                speedup, request.budget.min_speedup
                            )
                        ):
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
                            performance = replace(
                                performance, aggregate_speedup=speedup
                            )

                        perf_profiles = _profiles_by_case(performance)
                        profile_path = store.write_profile(experiment_id, perf_profiles)
                        profiler_diagnostics = _profiler_regression_diagnostics(
                            baseline, performance
                        )
                        decision = evaluated_decision(
                            proposal,
                            performance,
                            request.budget,
                            request.cases,
                            best_speedup=best_performance.aggregate_speedup,
                            best_performance=best_performance,
                            profiler_diagnostics=profiler_diagnostics,
                            target=request.target,
                        )
                        status = {
                            DecisionStatus.PROMOTE: "promoted",
                            DecisionStatus.RETRY: "rejected",
                            DecisionStatus.RECORD_SIGNAL: "signal_recorded",
                        }[decision.status]
                        experiment = ExperimentSummary(
                            experiment_id=experiment_id,
                            round_index=round_index,
                            parent_id=parent_id,
                            status=status,
                            source_path=candidate_artifacts.source_path,
                            decision=decision,
                            experiment_kind=proposal.experiment_kind,
                            change_scopes=tuple(
                                sorted(proposal.change_scopes, key=lambda scope: scope.value)
                            ),
                            blast_radius=proposal.blast_radius,
                            experiment_payload_path=experiment_payload_path,
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
                            profile_path=profile_path,
                        )
                        if decision.status is DecisionStatus.RETRY:
                            diagnostics.append(
                                _rejection_feedback(
                                    experiment_id,
                                    proposal,
                                    performance,
                                    baseline,
                                    request.cases,
                                    decision.rationale,
                                )
                            )
                        elif decision.status is DecisionStatus.RECORD_SIGNAL:
                            diagnostics.append(
                                f"{experiment_id}: {decision.rationale}"
                                + (
                                    f"; {decision.feedback}"
                                    if decision.feedback
                                    else ""
                                )
                            )
                        _report_performance(
                            experiment_id,
                            status,
                            performance,
                            baseline=baseline,
                            cases=request.cases,
                            diagnostics=f"decision={decision.rationale}",
                        )
                        if (
                            decision.status is DecisionStatus.PROMOTE
                            and promotion_committer is not None
                            and is_promotable(
                                performance,
                                request.budget,
                                request.cases,
                                request.target,
                            )
                        ):
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
                        if (
                            decision.status is DecisionStatus.PROMOTE
                            and not exhausted
                        ):
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
                        experiment_kind=(
                            proposal.experiment_kind
                            if proposal is not None
                            else ExperimentKind.PROMOTABLE
                        ),
                        change_scopes=(
                            tuple(
                                sorted(
                                    proposal.change_scopes,
                                    key=lambda scope: scope.value,
                                )
                            )
                            if proposal is not None
                            else ()
                        ),
                        blast_radius=(
                            proposal.blast_radius
                            if proposal is not None
                            else BlastRadius.LOCAL
                        ),
                        experiment_payload_path=experiment_payload_path,
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

        final_profiler_diagnostics = ""
        if best_experiment_id != "baseline":
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
            final_profiler_diagnostics = _profiler_regression_diagnostics(
                baseline, final_profile
            )
        else:
            final_profile = baseline

        if best_experiment_id != "baseline" and (
            final_profiler_diagnostics
            or not is_acceptable_winner(
                final_profile,
                baseline,
                request.budget,
                request.cases,
                request.target,
            )
        ):
            if final_profiler_diagnostics:
                diagnostics.append(f"final: rejected: {final_profiler_diagnostics}")
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
            decision=(
                experiments[-1].decision
                if stopping_reason == "human_review_requested"
                else Decision(
                    status=DecisionStatus.STOP,
                    rationale=stopping_reason,
                    evaluation=final_profile,
                )
            ),
            winner_experiment_id=best_experiment_id,
            winner_commit_title=best_commit_title,
            winner_commit_summary=best_commit_summary,
            promotion_commits=tuple(promotion_commits),
            rollback_commit=rollback_commit,
            auto_commit=last_auto_commit,
        )
        store.write_json("result.json", result)
        return result


# Compatibility with the original public API and CLI implementation.
KernelOptimizer = DecisionMaker
