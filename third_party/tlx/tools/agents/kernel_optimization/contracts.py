from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

VALID_STRATEGIES: frozenset[str] = frozenset({"best_first", "beam"})
VALID_HYPOTHESIS_KINDS: frozenset[str] = frozenset(
    {
        "algorithmic",
        "compiler_layout",
        "memory_layout",
        "parameter",
        "pipeline",
        "synchronization",
        "topology",
        "unknown",
    }
)


class ChangeScope(str, Enum):
    CONFIG = "config"
    KERNEL = "kernel"
    COMPILER = "compiler"


class ExperimentKind(str, Enum):
    PROMOTABLE = "promotable"
    PTX_ABLATION = "ptx_ablation"
    AMDGCN_ABLATION = "amdgcn_ablation"
    IR_OVERRIDE = "ir_override"
    HUMAN_REVIEW = "human_review"


class BlastRadius(str, Enum):
    LOCAL = "local"
    MULTI_FILE = "multi_file"
    COMPILER = "compiler"
    CROSS_PLATFORM = "cross_platform"


class DecisionStatus(str, Enum):
    PROMOTE = "promote"
    RETRY = "retry"
    RECORD_SIGNAL = "record_signal"
    STOP = "stop"
    NEEDS_HUMAN = "needs_human"


@dataclass(frozen=True)
class CandidateChange:
    scope: ChangeScope
    summary: str
    files: tuple[str, ...] = ()


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
    optimization_skills: tuple[str, ...] = ()
    supported_experiment_kinds: tuple[ExperimentKind, ...] = (
        ExperimentKind.PROMOTABLE,
        ExperimentKind.HUMAN_REVIEW,
    )
    evaluation_policy: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_policy, Mapping):
            raise ValueError("evaluation_policy must be a mapping")
        raw_skills = self.optimization_skills
        if not isinstance(raw_skills, (list, tuple)):
            raise ValueError("optimization_skills must be a sequence of names")
        normalized: list[str] = []
        for raw_name in raw_skills:
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("optimization_skills must contain non-empty strings")
            name = raw_name.strip().lower()
            if name not in normalized:
                normalized.append(name)
        object.__setattr__(self, "optimization_skills", tuple(normalized))
        if not self.supported_experiment_kinds:
            raise ValueError("target must support at least one experiment kind")
        if any(
            not isinstance(kind, ExperimentKind)
            for kind in self.supported_experiment_kinds
        ):
            raise ValueError("supported_experiment_kinds must contain ExperimentKind values")
        if len(set(self.supported_experiment_kinds)) != len(
            self.supported_experiment_kinds
        ):
            raise ValueError("supported_experiment_kinds must not contain duplicates")


@dataclass(frozen=True)
class OptimizationBudget:
    max_rounds: int = 5
    candidates_per_round: int = 2
    max_candidate_seconds: float = 600.0
    max_total_seconds: float = 3600.0
    min_speedup: float = 1.01
    max_cv: float = 0.10
    benchmark_repetitions: int = 10
    max_diagnostic_proton_passes: int = 10
    max_diagnostic_ncu_collections: int = 2
    max_agent_actions_per_candidate: int = 3
    max_diagnostic_actions_per_candidate: int = 2
    max_source_research_actions_per_candidate: int = 1
    max_source_research_actions_total: int = 4
    source_research_saturation_threshold: int = 1

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
        if self.max_diagnostic_proton_passes < 0:
            raise ValueError("max_diagnostic_proton_passes must not be negative")
        if self.max_diagnostic_ncu_collections < 0:
            raise ValueError("max_diagnostic_ncu_collections must not be negative")
        if self.max_agent_actions_per_candidate <= 0:
            raise ValueError("max_agent_actions_per_candidate must be positive")
        if self.max_diagnostic_actions_per_candidate < 0:
            raise ValueError("max_diagnostic_actions_per_candidate must not be negative")
        if self.max_source_research_actions_per_candidate < 0:
            raise ValueError(
                "max_source_research_actions_per_candidate must not be negative"
            )
        if self.max_source_research_actions_total < 0:
            raise ValueError("max_source_research_actions_total must not be negative")
        if self.source_research_saturation_threshold <= 0:
            raise ValueError("source_research_saturation_threshold must be positive")


@dataclass(frozen=True)
class DiagnosticArtifact:
    reference: str
    sha256: str = ""


@dataclass(frozen=True)
class DiagnosticResult:
    available: bool = False
    summary: Mapping[str, JsonValue] = field(default_factory=dict)
    failure: str = ""
    artifacts: tuple[DiagnosticArtifact, ...] = ()
    tool_schema_version: str = ""
    parser_schema_version: str = ""


@dataclass(frozen=True)
class DiagnosticEvidence:
    action_id: str
    status: str
    source_digest: str
    target_identity: str
    tool: str
    case_ids: tuple[str, ...]
    canonical_key: str
    question: str = ""
    rationale: str = ""
    level: str = ""
    focus: tuple[str, ...] = ()
    passes: tuple[str, ...] = ()
    expected_regions: tuple[str, ...] = ()
    result: DiagnosticResult = field(default_factory=DiagnosticResult)
    collection_duration_seconds: float = 0.0
    instrumented_source_digest: str = ""
    instrumentation_mapping_digest: str = ""


@dataclass(frozen=True)
class SourceExcerpt:
    path: str
    start_line: int
    end_line: int
    symbol: str = ""
    text: str = ""


@dataclass(frozen=True)
class ResearchEvidence:
    action_id: str
    status: str
    source_digest: str
    question: str
    rationale: str
    findings: tuple[str, ...] = ()
    excerpts: tuple[SourceExcerpt, ...] = ()
    inspected_paths: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    collection_duration_seconds: float = 0.0


@dataclass(frozen=True)
class DiagnosticMetricComparison:
    name: str
    before: float | None = None
    after: float | None = None
    relative_change: float | None = None


@dataclass(frozen=True)
class DiagnosticComparison:
    comparison_id: str
    verdict: str
    tool: str
    before_action_id: str
    after_action_id: str
    before_source_digest: str
    after_source_digest: str
    case_ids: tuple[str, ...]
    metrics: tuple[DiagnosticMetricComparison, ...] = ()
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class PriorExperimentEvidence:
    experiment_id: str
    status: str
    hypothesis: str = ""
    change: str = ""
    aggregate_speedup: float | None = None
    diagnostics: str = ""
    experiment_kind: str = ""
    decision_status: str = ""


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
    diagnostic_proton_intra_kernel: bool | None = None
    profiling_policy: str = "adaptive"
    prior_run_evidence: PriorRunEvidence | None = None
    auto_test: bool = False
    kernel_path: Path | None = None
    repository_root: Path | None = None

    def __post_init__(self) -> None:
        if not self.kernel_source.strip():
            raise ValueError("kernel_source must not be empty")
        if not self.cases:
            raise ValueError("at least one input case is required")
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(VALID_STRATEGIES)}")
        if self.profiling_policy not in {"adaptive", "legacy"}:
            raise ValueError("profiling_policy must be 'adaptive' or 'legacy'")
        if self.repository_root is not None and not self.repository_root.is_dir():
            raise ValueError("repository_root must be an existing directory")
        if self.kernel_path is not None and not self.kernel_path.is_file():
            raise ValueError("kernel_path must be an existing file")

    @property
    def use_diagnostic_proton_intra_kernel(self) -> bool:
        if self.diagnostic_proton_intra_kernel is not None:
            return self.diagnostic_proton_intra_kernel
        return (
            self.profiling_policy == "adaptive"
            and self.target.backend.strip().lower() in {"cuda", "nvidia"}
        )


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
class CandidateSubmission:
    source: str
    summary: str = ""
    hypothesis: str = ""
    evidence: str = ""
    expected_effect: str = ""
    risk: str = ""
    commit_title: str = ""
    commit_summary: str = ""
    change_scopes: frozenset[ChangeScope] = frozenset({ChangeScope.KERNEL})
    changes: tuple[CandidateChange, ...] = ()
    experiment_kind: ExperimentKind = ExperimentKind.PROMOTABLE
    blast_radius: BlastRadius = BlastRadius.LOCAL
    experiment_payload: Mapping[str, JsonValue] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_kind, ExperimentKind):
            raise ValueError("experiment_kind must be an ExperimentKind")
        if not isinstance(self.blast_radius, BlastRadius):
            raise ValueError("blast_radius must be a BlastRadius")
        if not self.change_scopes or any(
            not isinstance(scope, ChangeScope) for scope in self.change_scopes
        ):
            raise ValueError("change_scopes must contain at least one ChangeScope")
        if self.summary and self.rationale and self.summary != self.rationale:
            raise ValueError("summary and rationale must match when both are provided")
        if self.rationale and not self.summary:
            object.__setattr__(self, "summary", self.rationale)
        elif self.summary and not self.rationale:
            object.__setattr__(self, "rationale", self.summary)


@dataclass(frozen=True)
class Decision:
    status: DecisionStatus
    rationale: str
    feedback: str = ""
    evaluation: PerformanceSummary | None = None


@dataclass(frozen=True)
class ExperimentSummary:
    experiment_id: str
    round_index: int
    parent_id: str | None
    status: str
    source_path: Path
    decision: Decision | None = None
    experiment_kind: ExperimentKind = ExperimentKind.PROMOTABLE
    change_scopes: tuple[ChangeScope, ...] = ()
    blast_radius: BlastRadius = BlastRadius.LOCAL
    experiment_payload_path: Path | None = None
    incremental_patch_path: Path | None = None
    cumulative_patch_path: Path | None = None
    performance: PerformanceSummary | None = None
    diagnostics: str = ""
    mutation_summary: str = ""
    hypothesis: str = ""
    hypothesis_kind: str = "unknown"
    escalation_reason: str = ""
    research_evidence_ids: tuple[str, ...] = ()
    evidence: str = ""
    expected_effect: str = ""
    risk: str = ""
    commit_title: str = ""
    commit_summary: str = ""
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
    attribution: str = "TLX agent authored"
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
    decision: Decision | None = None
    winner_experiment_id: str = "baseline"
    winner_commit_title: str = ""
    winner_commit_summary: str = ""
    promotion_commits: tuple[AutoCommitResult, ...] = ()
    rollback_commit: AutoCommitResult | None = None
    auto_commit: AutoCommitResult | None = None


def to_json_value(value: Any) -> JsonValue:
    if hasattr(value, "__dataclass_fields__"):
        return to_json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")
