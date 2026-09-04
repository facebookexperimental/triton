from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

VALID_STRATEGIES: frozenset[str] = frozenset({"best_first", "beam"})


@dataclass(frozen=True)
class InputCase:
    case_id: str
    parameters: Mapping[str, JsonValue]
    weight: float = 1.0
    protected: bool = True

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("case weight must be finite and positive")


@dataclass(frozen=True)
class KernelTarget:
    backend: str
    architecture: str
    device: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    optimization_guidance: str = ""


@dataclass(frozen=True)
class OptimizationBudget:
    max_rounds: int = 5
    candidates_per_round: int = 2
    max_candidate_seconds: float = 600.0
    max_total_seconds: float = 3600.0
    min_speedup: float = 1.01
    max_cv: float = 0.10
    benchmark_repetitions: int = 10

    def __post_init__(self) -> None:
        if self.max_rounds <= 0 or self.candidates_per_round <= 0:
            raise ValueError("round and candidate budgets must be positive")
        if self.max_candidate_seconds <= 0 or self.max_total_seconds <= 0:
            raise ValueError("time budgets must be positive")
        if self.min_speedup < 1:
            raise ValueError("min_speedup must be at least 1")
        if self.max_cv < 0:
            raise ValueError("max_cv must not be negative")
        if self.benchmark_repetitions <= 0:
            raise ValueError("benchmark_repetitions must be positive")


@dataclass(frozen=True)
class PriorExperimentEvidence:
    experiment_id: str
    status: str
    hypothesis: str = ""
    change: str = ""
    aggregate_speedup: float | None = None
    diagnostics: str = ""


@dataclass(frozen=True)
class PriorRunEvidence:
    run_path: Path
    experiments_path: Path
    source_hashes: tuple[str, ...] = ()
    experiments: tuple[PriorExperimentEvidence, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class KernelOptimizationRequest:
    kernel_source: str
    harness_path: Path
    cases: tuple[InputCase, ...]
    target: KernelTarget
    budget: OptimizationBudget = OptimizationBudget()
    strategy: str = "best_first"
    reference_kernel_source: str | None = None
    output_dir: Path | None = None
    diagnostic_proton_intra_kernel: bool = False
    prior_run_evidence: PriorRunEvidence | None = None

    def __post_init__(self) -> None:
        if not self.kernel_source.strip():
            raise ValueError("kernel_source must not be empty")
        if not self.cases:
            raise ValueError("at least one input case is required")
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(VALID_STRATEGIES)}")


@dataclass(frozen=True)
class BuildResult:
    success: bool
    artifact: JsonValue = None
    diagnostics: str = ""


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    diagnostics: str = ""
    metrics: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class TimingSamples:
    samples_us: tuple[float, ...]
    warmup_count: int = 0
    cache_policy: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.samples_us:
            raise ValueError("timing samples must not be empty")
        if any(not math.isfinite(sample) or sample <= 0 for sample in self.samples_us):
            raise ValueError("timing samples must be finite and positive")

    @property
    def median_us(self) -> float:
        return statistics.median(self.samples_us)

    @property
    def p50_us(self) -> float:
        return self.median_us

    @property
    def p95_us(self) -> float:
        if len(self.samples_us) == 1:
            return self.samples_us[0]
        sorted_samples = sorted(self.samples_us)
        # Linear interpolation for the 95th percentile.
        rank = 0.95 * (len(sorted_samples) - 1)
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        if lower == upper:
            return sorted_samples[lower]
        weight = rank - lower
        return sorted_samples[lower] * (1 - weight) + sorted_samples[upper] * weight

    @property
    def mean_us(self) -> float:
        return statistics.fmean(self.samples_us)

    @property
    def stdev_us(self) -> float:
        if len(self.samples_us) == 1:
            return 0.0
        return statistics.stdev(self.samples_us)

    @property
    def coefficient_of_variation(self) -> float:
        mean = self.mean_us
        if len(self.samples_us) == 1:
            return 0.0
        return self.stdev_us / mean if mean != 0 else 0.0


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    verification: VerificationResult
    timing: TimingSamples | None = None
    profile: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class PerformanceSummary:
    cases: tuple[CaseEvaluation, ...]
    aggregate_speedup: float = 1.0

    @property
    def correct(self) -> bool:
        return all(case.verification.passed for case in self.cases)


@dataclass(frozen=True)
class ExperimentSummary:
    experiment_id: str
    round_index: int
    parent_id: str | None
    status: str
    source_path: Path
    incremental_patch_path: Path | None = None
    cumulative_patch_path: Path | None = None
    performance: PerformanceSummary | None = None
    diagnostics: str = ""
    mutation_summary: str = ""
    hypothesis: str = ""
    evidence: str = ""
    expected_effect: str = ""
    risk: str = ""
    commit_title: str = ""
    commit_summary: str = ""
    falsifier: str = ""
    change_scope: str = ""
    profile_path: Path | None = None
    auto_commit: AutoCommitResult | None = None


@dataclass(frozen=True)
class AutoCommitResult:
    requested: bool
    success: bool
    vcs: str | None = None
    repo_root: Path | None = None
    target_path: Path | None = None
    target_relpath: str | None = None
    base_revision: str | None = None
    commit_revision: str | None = None
    subject: str | None = None
    attribution: str = "Kernel optimization agent authored"
    dirty_target_at_start: bool = False
    diagnostics: str = ""


@dataclass(frozen=True)
class KernelOptimizationResult:
    success: bool
    best_kernel: str
    baseline: PerformanceSummary
    final: PerformanceSummary
    experiments: tuple[ExperimentSummary, ...]
    artifacts_dir: Path
    stopping_reason: str
    winner_experiment_id: str = "baseline"
    winner_commit_title: str = ""
    winner_commit_summary: str = ""
    promotion_commits: tuple[AutoCommitResult, ...] = ()
    rollback_commit: AutoCommitResult | None = None
    auto_commit: AutoCommitResult | None = None


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


def to_json_value(value: Any) -> JsonValue:
    if hasattr(value, "__dataclass_fields__"):
        return to_json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")
