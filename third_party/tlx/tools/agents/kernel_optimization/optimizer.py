from __future__ import annotations

import tempfile
import time
from dataclasses import replace
from pathlib import Path

from .artifacts import ArtifactStore
from .harness import HarnessExecutionError, SubprocessHarness
from .models import (
    ExperimentSummary,
    KernelOptimizationRequest,
    KernelOptimizationResult,
    is_promotable,
    passes_protected_cases,
    weighted_geometric_speedup,
)
from .providers import CandidateContext, CandidateProvider, CodexCandidateProvider
from .source import source_digest


class KernelOptimizer:
    def __init__(self, provider: CandidateProvider | None = None) -> None:
        self._provider = provider or CodexCandidateProvider()

    def optimize(self, request: KernelOptimizationRequest) -> KernelOptimizationResult:
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
            profile=True,
        )
        if not passes_protected_cases(baseline, request.cases):
            raise HarnessExecutionError(
                "baseline failed one or more protected correctness cases"
            )
        baseline = replace(baseline, aggregate_speedup=1.0)
        baseline_profiles = {
            evaluation.case_id: dict(evaluation.profile) for evaluation in baseline.cases
        }
        store.write_profile("baseline", baseline_profiles)
        store.write_aggregated_profile("baseline_profile", baseline_profiles)
        experiments = [
            ExperimentSummary(
                experiment_id="baseline",
                round_index=0,
                parent_id=None,
                status="baseline",
                source_path=baseline_source_path,
                performance=baseline,
            )
        ]
        store.write_json("experiments/baseline/result.json", baseline)

        best_source = request.kernel_source
        best_performance = baseline
        best_experiment_id = "baseline"
        best_profiles = baseline_profiles
        diagnostics: list[str] = []
        seen_sources = {source_digest(request.kernel_source)}
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
                proposal = None
                try:
                    proposal = self._provider.propose(
                        request,
                        CandidateContext(
                            round_index=round_index,
                            candidate_index=candidate_index,
                            current_source=best_source,
                            current_performance=best_performance,
                            previous_diagnostics=tuple(diagnostics),
                        ),
                    )
                    digest = source_digest(proposal.source)
                    if digest in seen_sources:
                        raise ValueError("candidate source duplicates an earlier experiment")
                    seen_sources.add(digest)
                    source_path = store.write_source(experiment_id, proposal.source)
                    performance = harness.evaluate(
                        proposal.source,
                        request.cases,
                        request.target,
                        request.budget.benchmark_repetitions,
                        profile=True,
                    )
                    # Persist per-case profiles for this candidate regardless of promotion.
                    perf_profiles = {
                        evaluation.case_id: dict(evaluation.profile)
                        for evaluation in performance.cases
                    }
                    store.write_profile(experiment_id, perf_profiles)
                    speedup = weighted_geometric_speedup(
                        baseline, performance, request.cases
                    )
                    performance = replace(
                        performance, aggregate_speedup=speedup
                    )
                    status = (
                        "promoted"
                        if is_promotable(performance, request.budget, request.cases)
                        and speedup > best_performance.aggregate_speedup
                        else "rejected"
                    )
                    experiment = ExperimentSummary(
                        experiment_id=experiment_id,
                        round_index=round_index,
                        parent_id=best_experiment_id,
                        status=status,
                        source_path=source_path,
                        performance=performance,
                        mutation_summary=proposal.summary,
                    )
                    if status == "promoted":
                        best_source = proposal.source
                        best_performance = performance
                        best_experiment_id = experiment_id
                        best_profiles = perf_profiles
                        promoted_this_round = True
                except Exception as error:  # noqa: BLE001
                    message = f"{experiment_id}: {type(error).__name__}: {error}"
                    diagnostics.append(message)
                    source_path = store.write_source(
                        experiment_id, proposal.source if proposal is not None else ""
                    )
                    experiment = ExperimentSummary(
                        experiment_id=experiment_id,
                        round_index=round_index,
                        parent_id=best_experiment_id,
                        status="failed",
                        source_path=source_path,
                        diagnostics=message,
                    )
                experiments.append(experiment)
                store.write_json(
                    f"experiments/{experiment_id}/result.json", experiment
                )
            if exhausted:
                break
            if not promoted_this_round:
                stopping_reason = "no_promotable_candidate"
                break

        final_profile = harness.evaluate(
            best_source,
            request.cases,
            request.target,
            request.budget.benchmark_repetitions,
            profile=True,
        )
        final_profile = replace(
            final_profile,
            aggregate_speedup=weighted_geometric_speedup(
                baseline, final_profile, request.cases
            ),
        )
        final_profiles = {
            evaluation.case_id: dict(evaluation.profile) for evaluation in final_profile.cases
        }
        if best_experiment_id != "baseline" and not is_promotable(
            final_profile, request.budget, request.cases
        ):
            best_source = request.kernel_source
            final_profile = baseline
            final_profiles = baseline_profiles
            best_experiment_id = "baseline"
            stopping_reason = "finalist_revalidation_failed"
        else:
            best_profiles = final_profiles
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
        )
        store.write_json("result.json", result)
        return result
