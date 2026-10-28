from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

from ..contracts import (
    BlastRadius,
    CandidateChange,
    CaseEvaluation,
    ChangeScope,
    DiagnosticEvidence,
    ExperimentKind,
    JsonValue,
    KernelOptimizationRequest,
    PerformanceSummary,
    ResearchEvidence,
    VALID_HYPOTHESIS_KINDS,
)
from ..decision_maker.profiling import (
    compact_profile_summary,
    format_intra_kernel_evidence,
    is_profile_fresh,
    is_valid_intra_kernel_evidence,
)
from .knowledge import load_knowledge
from .source import (
    source_digest,
    validate_replacement_source,
)
from .strategy import OPTIMIZATION_STRATEGY

_SKILLS_ROOT = Path(__file__).resolve().parent / "skills"
_LAYOUT_CONVERSION_SKILL = _SKILLS_ROOT / "common/layout-conversion-efficiency.md"
_ARCHITECTURE_DISCOVERY_SKILL = _SKILLS_ROOT / "common/architecture-discovery.md"
_NVIDIA_TARGET_SKILLS = _SKILLS_ROOT / "targets/nvidia"
_ASYNC_TMA_OUTPUT_SKILL = _NVIDIA_TARGET_SKILLS / "async-tma-output-publication.md"
_NVIDIA_WARP_BARRIER_SKILL = (
    _NVIDIA_TARGET_SKILLS / "nvidia-warp-barrier-efficiency.md"
)
_BLACKWELL_CLC_SKILL = _NVIDIA_TARGET_SKILLS / "blackwell-persistent-clc-scheduling.md"
_NVIDIA_PERSISTENT_PIPELINE_SKILL = (
    _NVIDIA_TARGET_SKILLS / "nvidia-persistent-pipeline-efficiency.md"
)
_AMD_TARGET_SKILLS = _SKILLS_ROOT / "targets/amd"
_AMD_GENERAL_SKILL = _AMD_TARGET_SKILLS / "amd-kernel-optimization.md"
_AMD_ATTENTION_SKILL = _AMD_TARGET_SKILLS / "amd-attention-optimization.md"
_AMD_ATTENTION_REFERENCE = (
    _AMD_TARGET_SKILLS / "references/attention-variant-study.md"
)
_AMD_LIVE_RANGE_SKILL = _AMD_TARGET_SKILLS / "amd-ir-live-range-analysis.md"
_AMD_LIVE_RANGE_REFERENCE = (
    _AMD_TARGET_SKILLS / "references/live-range-interpretation.md"
)
_AMD_OPTIONAL_SKILLS = {
    "optimize-amd-tlx-attention": (
        _AMD_ATTENTION_SKILL,
        _AMD_ATTENTION_REFERENCE,
    ),
    "analyze-amd-ir-live-ranges": (
        _AMD_LIVE_RANGE_SKILL,
        _AMD_LIVE_RANGE_REFERENCE,
    ),
}
_AMD_BACKENDS = frozenset({"amd", "hip", "rocm"})
_BLACKWELL_ARCHITECTURES = frozenset(
    {"blackwell", "sm100", "sm_100", "b200", "b200a", "gb200", "gb300"}
)
_HOPPER_ARCHITECTURES = frozenset({"hopper", "h100", "sm90", "sm_90"})
_PERSISTENT_PIPELINE_ARCHITECTURES = _BLACKWELL_ARCHITECTURES | _HOPPER_ARCHITECTURES
_AGENT_ACTION_SCHEMA_VERSION = 1
_MAX_DIAGNOSTIC_CASES = 16
_MAX_DIAGNOSTIC_FOCUS = 4
_MAX_DIAGNOSTIC_PASSES = 4
_MAX_EXPECTED_REGIONS = 16
_MAX_RESEARCH_TERMS = 8
_MAX_RESEARCH_GOALS = 8
_MAX_AGENT_TEXT = 240
_AGENT_CANDIDATE_KEYS = frozenset({"schema_version", "action"})
_AGENT_DIAGNOSTIC_COMMON_KEYS = frozenset(
    {
        "schema_version",
        "action",
        "tool",
        "source_digest",
        "case_ids",
        "question",
        "rationale",
    }
)
_AGENT_NCU_KEYS = _AGENT_DIAGNOSTIC_COMMON_KEYS | {"level", "focus"}
_AGENT_PROTON_KEYS = _AGENT_DIAGNOSTIC_COMMON_KEYS | {
    "passes",
    "expected_regions",
}
_AGENT_RESEARCH_KEYS = frozenset(
    {
        "schema_version",
        "action",
        "source_digest",
        "question",
        "rationale",
        "search_terms",
        "goals",
    }
)
_AGENT_EXECUTION_CONTROL_FIELDS = frozenset(
    {
        "artifacts_dir",
        "command",
        "commands",
        "diagnostic_only",
        "environment",
        "env",
        "experiment_id",
        "granularity",
        "instrumentation_id",
        "instrumentation_mapping_digest",
        "instrumentation_mapping_path",
        "instrumented_source_digest",
        "kernel_filter",
        "kernel_name",
        "launch_count",
        "metric_names",
        "metrics",
        "ncu_binary",
        "policy_reason",
        "reason",
        "timeout",
        "timeout_s",
        "tools",
    }
)
_AGENT_SOURCE_MUTATION_FIELDS = frozenset(
    {
        "candidate_source",
        "diff",
        "patch",
        "replacement_source",
        "source",
        "source_code",
        "source_text",
    }
)
VALID_NCU_FOCUS: frozenset[str] = frozenset(
    {
        "cache",
        "compute_throughput",
        "memory_throughput",
        "occupancy",
        "register_pressure",
        "shared_memory_bank_conflicts",
        "tensor_core_utilization",
        "tma_efficiency",
        "warp_stalls",
    }
)
VALID_PROTON_PASSES: frozenset[str] = frozenset(
    {"role", "coarse", "wait", "compute"}
)
INSTRUMENTATION_MAPPING_SCHEMA_VERSION = 1


TLX_PROMPT_PREAMBLE = """You are optimizing one Triton or TLX kernel against an external deterministic harness.
The candidate is a complete replacement source file. Preserve every public entry point,
algorithmic contract, supported workload, and synchronization invariant required by the
harness and target guidance.

Evidence-driven optimization workflow:
1. Keep measurement scopes separate. The public benchmark, individual kernel profiles,
   Proton launch attribution, and diagnostic intra-kernel traces may cover different work.
   Do not subtract unrelated measurements or infer task overlap from a launch timeline.
2. Treat lower end-to-end benchmark latency with passing correctness as the promotion goal.
   Use target profiler duration, utilization, traffic, occupancy, registers, and stalls as
   explanatory evidence rather than standalone optimization targets.
3. Choose exactly one testable hypothesis and one coherent change. Map measured evidence to
   the narrowest relevant subsystem, and use failed hypotheses as exclusions in later rounds.
4. For warp-specialized or asynchronous kernels, change barriers, buffer counts, aliases,
   task scheduling, or visibility only with an explicit producer/consumer and lifetime proof.
5. Treat changes inside the noise floor as inconclusive. Do not repeat a configuration when
   benchmark and profile evidence show that it did not affect the targeted bottleneck.

TLX API guidance:
- Treat `.claude/skills/tlx-api-reference/SKILL.md` in the target repository as the primary
  TLX API reference when present.
- Reuse APIs, synchronization patterns, and architecture-specific examples already used by
  the supplied source and nearby code. Do not invent APIs or transplant incompatible target
  patterns.

The complete current source is available as `candidate.py` in your writable working
directory. `original.py` is an immutable copy of the same source. Choose exactly one action
and write it to `agent_action.json` using one of the exact schemas below.
Do not modify any other file. Keep the final response to one short plain-text summary.
do not print source code or a patch.
"""


def _validate_agent_text(value: str, field_name: str) -> None:
    if not value or value != value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{field_name} must be a non-empty single line")
    if len(value) > _MAX_AGENT_TEXT:
        raise ValueError(f"{field_name} must be at most {_MAX_AGENT_TEXT} characters")


@dataclass(frozen=True)
class AgentDiagnosticRequest:
    tool: Literal["ncu", "proton"]
    source_digest: str
    case_ids: tuple[str, ...]
    question: str
    rationale: str
    ncu_level: Literal["", "summary", "deep"] = ""
    ncu_focus: tuple[str, ...] = ()
    proton_passes: tuple[str, ...] = ()
    expected_regions: tuple[str, ...] = ()
    action: Literal["diagnostic"] = field(default="diagnostic", init=False)

    def __post_init__(self) -> None:
        if self.tool not in {"ncu", "proton"}:
            raise ValueError("diagnostic tool must be 'ncu' or 'proton'")
        if re.fullmatch(r"[0-9a-f]{64}", self.source_digest) is None:
            raise ValueError("diagnostic source_digest must be a lowercase SHA-256 digest")
        if not self.case_ids or len(self.case_ids) > _MAX_DIAGNOSTIC_CASES:
            raise ValueError("diagnostic case_ids must select 1 to 16 cases")
        if len(set(self.case_ids)) != len(self.case_ids) or any(
            not case_id
            or case_id != case_id.strip()
            or "\n" in case_id
            or "\r" in case_id
            or len(case_id) > _MAX_AGENT_TEXT
            for case_id in self.case_ids
        ):
            raise ValueError("diagnostic case_ids must be unique bounded non-empty strings")
        _validate_agent_text(self.question, "diagnostic question")
        _validate_agent_text(self.rationale, "diagnostic rationale")
        if self.tool == "ncu":
            self._validate_ncu()
        else:
            self._validate_proton()

    def _validate_ncu(self) -> None:
        if self.ncu_level not in {"summary", "deep"}:
            raise ValueError("NCU level must be 'summary' or 'deep'")
        if len(self.ncu_focus) > _MAX_DIAGNOSTIC_FOCUS:
            raise ValueError("NCU focus must contain at most 4 categories")
        if len(set(self.ncu_focus)) != len(self.ncu_focus):
            raise ValueError("NCU focus categories must be unique")
        unknown = set(self.ncu_focus) - VALID_NCU_FOCUS
        if unknown:
            raise ValueError(f"unknown NCU focus categories: {sorted(unknown)}")
        if self.ncu_level == "summary" and self.ncu_focus:
            raise ValueError("summary NCU requests must not specify focus categories")
        if self.ncu_level == "deep" and not self.ncu_focus:
            raise ValueError("deep NCU requests require at least one focus category")
        if self.proton_passes or self.expected_regions:
            raise ValueError("NCU requests must not contain Proton fields")

    def _validate_proton(self) -> None:
        if self.ncu_level or self.ncu_focus:
            raise ValueError("Proton requests must not contain NCU fields")
        if not self.proton_passes or len(self.proton_passes) > _MAX_DIAGNOSTIC_PASSES:
            raise ValueError("Proton passes must select 1 to 4 passes")
        if len(set(self.proton_passes)) != len(self.proton_passes):
            raise ValueError("Proton passes must be unique")
        unknown = set(self.proton_passes) - VALID_PROTON_PASSES
        if unknown:
            raise ValueError(f"unknown Proton passes: {sorted(unknown)}")
        if len(self.expected_regions) > _MAX_EXPECTED_REGIONS:
            raise ValueError("expected_regions must contain at most 16 regions")
        if len(set(self.expected_regions)) != len(self.expected_regions) or any(
            not region or region != region.strip() or len(region) > 120
            for region in self.expected_regions
        ):
            raise ValueError("expected_regions must be unique bounded non-empty strings")


@dataclass(frozen=True)
class AgentSourceResearchRequest:
    source_digest: str
    question: str
    rationale: str
    search_terms: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()
    action: Literal["source_research"] = field(default="source_research", init=False)

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.source_digest) is None:
            raise ValueError("source research source_digest must be a lowercase SHA-256 digest")
        _validate_agent_text(self.question, "source research question")
        _validate_agent_text(self.rationale, "source research rationale")
        self._validate_terms(self.search_terms, "search_terms", _MAX_RESEARCH_TERMS)
        self._validate_terms(self.goals, "goals", _MAX_RESEARCH_GOALS)

    @staticmethod
    def _validate_terms(values: tuple[str, ...], name: str, limit: int) -> None:
        if len(values) > limit:
            raise ValueError(f"source research {name} must contain at most {limit} items")
        if len(set(values)) != len(values) or any(
            not value
            or value != value.strip()
            or "\n" in value
            or "\r" in value
            or len(value) > _MAX_AGENT_TEXT
            for value in values
        ):
            raise ValueError(
                f"source research {name} must be unique bounded non-empty strings"
            )


def _validate_agent_action_keys(
    payload: Mapping[str, object],
    expected: frozenset[str],
    action_name: str,
) -> None:
    actual = set(payload)
    controlled = sorted(actual & _AGENT_EXECUTION_CONTROL_FIELDS)
    if controlled:
        raise ValueError(
            "agent action cannot control internal execution fields: "
            + ", ".join(controlled)
        )
    source_fields = sorted(actual & _AGENT_SOURCE_MUTATION_FIELDS)
    if source_fields:
        raise ValueError(
            "agent action cannot contain source mutation fields: "
            + ", ".join(source_fields)
        )
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(
            f"{action_name} agent action is missing keys: " + ", ".join(missing)
        )
    if unknown:
        raise ValueError(
            f"{action_name} agent action has unknown keys: " + ", ".join(unknown)
        )


def _agent_string(payload: Mapping[str, object], field_name: str) -> str:
    value = payload[field_name]
    if not isinstance(value, str):
        raise ValueError(f"agent action field {field_name!r} must be a string")
    return value


def _agent_string_array(
    payload: Mapping[str, object], field_name: str
) -> tuple[str, ...]:
    value = payload[field_name]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"agent action field {field_name!r} must be a string array")
    return tuple(value)


def _parse_agent_action(
    payload: object,
    *,
    current_source_digest: str,
    allowed_case_ids: tuple[str, ...] | None = None,
) -> Literal["candidate"] | AgentDiagnosticRequest | AgentSourceResearchRequest:
    if re.fullmatch(r"[0-9a-f]{64}", current_source_digest) is None:
        raise ValueError("current_source_digest must be a lowercase SHA-256 digest")
    if not isinstance(payload, dict):
        raise ValueError("agent action must be a JSON object")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError("agent action keys must be strings")
    if type(payload.get("schema_version")) is not int:
        raise ValueError("agent action schema_version must be an integer")
    if payload["schema_version"] != _AGENT_ACTION_SCHEMA_VERSION:
        raise ValueError(
            "agent action schema_version must be "
            f"{_AGENT_ACTION_SCHEMA_VERSION}"
        )
    action = payload.get("action")
    if action == "candidate":
        _validate_agent_action_keys(payload, _AGENT_CANDIDATE_KEYS, "candidate")
        return "candidate"
    if action == "source_research":
        _validate_agent_action_keys(
            payload, _AGENT_RESEARCH_KEYS, "source research"
        )
        declared_digest = _agent_string(payload, "source_digest")
        if declared_digest != current_source_digest:
            raise ValueError("source research agent action source_digest is stale")
        return AgentSourceResearchRequest(
            source_digest=declared_digest,
            question=_agent_string(payload, "question"),
            rationale=_agent_string(payload, "rationale"),
            search_terms=_agent_string_array(payload, "search_terms"),
            goals=_agent_string_array(payload, "goals"),
        )
    if action != "diagnostic":
        raise ValueError(
            "agent action must be 'candidate', 'diagnostic', or 'source_research'"
        )

    tool = payload.get("tool")
    if tool == "ncu":
        _validate_agent_action_keys(payload, _AGENT_NCU_KEYS, "NCU diagnostic")
    elif tool == "proton":
        _validate_agent_action_keys(payload, _AGENT_PROTON_KEYS, "Proton diagnostic")
    else:
        raise ValueError("diagnostic agent action tool must be 'ncu' or 'proton'")

    declared_digest = _agent_string(payload, "source_digest")
    if declared_digest != current_source_digest:
        raise ValueError("diagnostic agent action source_digest is stale")
    case_ids = _agent_string_array(payload, "case_ids")
    if allowed_case_ids is not None:
        unknown_cases = sorted(set(case_ids) - set(allowed_case_ids))
        if unknown_cases:
            raise ValueError(
                "diagnostic agent action contains unknown case_ids: "
                + ", ".join(unknown_cases)
            )
    question = _agent_string(payload, "question")
    rationale = _agent_string(payload, "rationale")
    if tool == "ncu":
        level = _agent_string(payload, "level")
        if level not in {"summary", "deep"}:
            raise ValueError("NCU level must be 'summary' or 'deep'")
        return AgentDiagnosticRequest(
            tool="ncu",
            source_digest=declared_digest,
            case_ids=case_ids,
            question=question,
            rationale=rationale,
            ncu_level=level,
            ncu_focus=_agent_string_array(payload, "focus"),
        )
    return AgentDiagnosticRequest(
        tool="proton",
        source_digest=declared_digest,
        case_ids=case_ids,
        question=question,
        rationale=rationale,
        proton_passes=_agent_string_array(payload, "passes"),
        expected_regions=_agent_string_array(payload, "expected_regions"),
    )


def _read_agent_action(
    path: Path,
    *,
    current_source_digest: str,
    allowed_case_ids: tuple[str, ...] | None = None,
) -> Literal["candidate"] | AgentDiagnosticRequest | AgentSourceResearchRequest:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError("agent_action.json was not created") from error
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"agent action is not valid JSON: {error}") from error
    return _parse_agent_action(
        payload,
        current_source_digest=current_source_digest,
        allowed_case_ids=allowed_case_ids,
    )


@dataclass(frozen=True)
class CandidateProfilingHint:
    pass_name: str = ""
    expected_regions: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class CandidateProposal:
    source: str
    summary: str = ""
    hypothesis: str = ""
    hypothesis_kind: str = "unknown"
    escalation_reason: str = ""
    research_evidence_ids: tuple[str, ...] = ()
    evidence: str = ""
    expected_effect: str = ""
    risk: str = ""
    commit_title: str = ""
    commit_summary: str = ""
    change_scopes: frozenset[ChangeScope] = field(
        default_factory=lambda: frozenset({ChangeScope.KERNEL})
    )
    changes: tuple[CandidateChange, ...] = ()
    experiment_kind: ExperimentKind = ExperimentKind.PROMOTABLE
    blast_radius: BlastRadius = BlastRadius.LOCAL
    experiment_payload: Mapping[str, JsonValue] = field(default_factory=dict)
    rationale: str = ""
    profiling_hint: CandidateProfilingHint = field(
        default_factory=CandidateProfilingHint
    )
    action: Literal["candidate"] = field(default="candidate", init=False)

    def __post_init__(self) -> None:
        if self.hypothesis_kind not in VALID_HYPOTHESIS_KINDS:
            raise ValueError(
                "hypothesis_kind must be one of "
                + ", ".join(sorted(VALID_HYPOTHESIS_KINDS))
            )
        if self.escalation_reason:
            _validate_agent_text(self.escalation_reason, "escalation_reason")
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


CandidateAction: TypeAlias = (
    CandidateProposal | AgentDiagnosticRequest | AgentSourceResearchRequest
)


@dataclass(frozen=True)
class CandidateContext:
    round_index: int
    candidate_index: int
    current_source: str
    current_performance: PerformanceSummary
    previous_diagnostics: tuple[str, ...]
    baseline_diagnostic_evidence: tuple[str, ...] = ()
    action_index: int = 0
    current_source_digest: str = ""
    remaining_agent_actions: int = 0
    remaining_diagnostic_actions: int = 0
    remaining_ncu_collections: int = 0
    remaining_proton_passes: int = 0
    remaining_source_research_actions: int = 0
    diagnostic_evidence: tuple[DiagnosticEvidence, ...] = ()
    research_evidence: tuple[ResearchEvidence, ...] = ()
    local_search_failure_streak: int = 0
    source_research_saturation_threshold: int = 1


class CandidateProvider(Protocol):
    def propose(
        self,
        request: KernelOptimizationRequest,
        context: CandidateContext,
    ) -> CandidateAction: ...


def _validated_current_source_digest(context: CandidateContext) -> str:
    computed = source_digest(context.current_source)
    declared = context.current_source_digest
    if not declared:
        return computed
    if re.fullmatch(r"[0-9a-f]{64}", declared) is None:
        raise ValueError("current_source_digest must be a lowercase SHA-256 digest")
    if declared != computed:
        raise ValueError("current_source_digest does not match current_source")
    return declared


@dataclass(frozen=True)
class DiagnosticInstrumentationContext:
    current_source: str
    requested_passes: tuple[str, ...] = ()
    pass_capabilities: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    case_id: str = ""
    rationale: str = ""
    previous_diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiagnosticInstrumentationProposal:
    source: str
    mapping: Mapping[str, object]
    summary: str = ""


class DiagnosticInstrumentationProvider(Protocol):
    def instrument(
        self,
        request: KernelOptimizationRequest,
        context: DiagnosticInstrumentationContext,
    ) -> DiagnosticInstrumentationProposal: ...


Optimizer = CandidateProvider


@dataclass
class FixedCandidateProvider:
    candidates: list[CandidateProposal]

    def propose(
        self,
        request: KernelOptimizationRequest,
        context: CandidateContext,
    ) -> CandidateProposal:
        del request, context
        if not self.candidates:
            raise RuntimeError("fixed candidate provider is exhausted")
        return self.candidates.pop(0)


@dataclass(frozen=True)
class MockLLMProvider:
    """Deterministic stub for CI — replays canned candidates without a live LLM."""

    canned: tuple[CandidateProposal, ...] = ()
    fallback_source: str | None = None

    def propose(
        self,
        request: KernelOptimizationRequest,
        context: CandidateContext,
    ) -> CandidateProposal:
        del request
        index = (context.round_index - 1) * 10 + context.candidate_index
        if index < len(self.canned):
            return self.canned[index]
        if self.fallback_source is not None:
            return CandidateProposal(source=self.fallback_source, summary="mock-fallback")
        # Default: echo current source so the harness re-evaluates it (dedup will
        # turn the second echo into a deterministic failure rather than a hang).
        return CandidateProposal(source=context.current_source, summary="mock-echo")


_METADATA_SCHEMA_VERSION = 3
_LEGACY_METADATA_SCHEMA_VERSION = 2
_INSTRUMENTATION_PROMPT_LIMIT = 6000
_COMMIT_SUMMARY_RE = re.compile(
    r"\AChange summary:[ \t]*\n?(?P<change>.+?)\n\nWhy:[ \t]*\n?(?P<why>.+)\Z",
    re.DOTALL,
)
_GENERIC_COMMIT_TITLES = frozenset(
    {
        "improve performance",
        "optimize candidate",
        "optimize kernel",
        "optimize performance",
        "update kernel",
    }
)
_COMMIT_METADATA_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "in",
        "of",
        "on",
        "the",
        "to",
        "use",
        "with",
    }
)

_SHORT_METADATA_FIELDS = (
    "hypothesis",
    "evidence",
    "change",
    "expected_effect",
    "risk",
)


def _clean_short_metadata(value: object) -> str:
    return " ".join(str(value or "").split())[:240]


def _clean_commit_title(value: object) -> str:
    return _clean_short_metadata(value).rstrip(".")[:80].strip()


def _clean_commit_summary(value: object) -> str:
    text = str(value or "").replace("\x00", "")
    paragraphs = [" ".join(part.split()) for part in text.split("\n\n")]
    return "\n\n".join(part for part in paragraphs if part).strip()


def _top_level_nodes(source: str) -> dict[str, ast.AST]:
    nodes: dict[str, ast.AST] = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    nodes[target.id] = node
    return nodes


def _commit_words(text: str) -> frozenset[str]:
    words = set()
    for token in re.findall(
        r"[A-Za-z][A-Za-z0-9]*", text.casefold().replace("_", " ")
    ):
        word = token[:-1] if token.endswith("s") and len(token) > 4 else token
        if word not in _COMMIT_METADATA_STOP_WORDS:
            words.add(word)
    return frozenset(words)


def _changed_top_level_names(before: str, after: str) -> tuple[str, ...]:
    before_nodes = _top_level_nodes(before)
    after_nodes = _top_level_nodes(after)
    names = set(before_nodes) | set(after_nodes)

    def node_dump(node: ast.AST | None) -> str | None:
        return ast.dump(node, include_attributes=False) if node is not None else None

    return tuple(
        sorted(
            name
            for name in names
            if node_dump(before_nodes.get(name)) != node_dump(after_nodes.get(name))
        )
    )


def _read_candidate_metadata(
    path: Path,
    *,
    source: str,
    original_source: str,
) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError("candidate_metadata.json was not created") from error
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"candidate metadata is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("candidate metadata must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version not in {_LEGACY_METADATA_SCHEMA_VERSION, _METADATA_SCHEMA_VERSION}:
        raise ValueError(
            "candidate metadata schema_version must be "
            f"{_LEGACY_METADATA_SCHEMA_VERSION} or {_METADATA_SCHEMA_VERSION}"
        )
    legacy_schema = schema_version == _LEGACY_METADATA_SCHEMA_VERSION

    metadata: dict[str, object] = {}
    for field_name in _SHORT_METADATA_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"candidate metadata field {field_name!r} must be non-empty")
        metadata[field_name] = value.strip()
    for field_name in _SHORT_METADATA_FIELDS:
        text = str(metadata[field_name])
        if "\n" in text or len(text) > 240:
            raise ValueError(f"candidate metadata field {field_name!r} must be one line under 240 characters")
        metadata[field_name] = _clean_short_metadata(text)

    if not legacy_schema and "experiment_kind" not in payload:
        raise ValueError("experiment_kind must name a supported experiment kind")
    try:
        experiment_kind = ExperimentKind(
            payload.get("experiment_kind", ExperimentKind.PROMOTABLE.value)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("experiment_kind must name a supported experiment kind") from error
    if not legacy_schema and "blast_radius" not in payload:
        raise ValueError("blast_radius must name a supported blast radius")
    try:
        blast_radius = BlastRadius(payload.get("blast_radius", BlastRadius.LOCAL.value))
    except (TypeError, ValueError) as error:
        raise ValueError("blast_radius must name a supported blast radius") from error

    raw_scopes = payload.get("change_scopes")
    if raw_scopes is None and legacy_schema:
        raw_scopes = [ChangeScope.KERNEL.value]
    if not isinstance(raw_scopes, list) or not raw_scopes:
        raise ValueError("change_scopes must be a non-empty list")
    try:
        change_scopes = frozenset(ChangeScope(scope) for scope in raw_scopes)
    except (TypeError, ValueError) as error:
        raise ValueError("change_scopes contains an unsupported scope") from error
    if len(change_scopes) != len(raw_scopes):
        raise ValueError("change_scopes must not contain duplicates")

    raw_changes = payload.get("changes")
    if raw_changes is None and legacy_schema:
        default_scope = next(iter(sorted(change_scopes, key=lambda scope: scope.value)))
        raw_changes = [
            {
                "scope": default_scope.value,
                "summary": str(metadata["change"]),
                "files": [],
            }
        ]
    if not isinstance(raw_changes, list) or not raw_changes:
        raise ValueError("changes must be a non-empty list")
    changes = []
    for index, raw_change in enumerate(raw_changes):
        if not isinstance(raw_change, dict):
            raise ValueError(f"changes[{index}] must be an object")
        try:
            scope = ChangeScope(raw_change["scope"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"changes[{index}].scope is invalid") from error
        summary = raw_change.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"changes[{index}].summary must be non-empty")
        files = raw_change.get("files", [])
        if not isinstance(files, list) or not all(
            isinstance(file, str) and file.strip() for file in files
        ):
            raise ValueError(f"changes[{index}].files must be a string list")
        changes.append(
            CandidateChange(
                scope=scope,
                summary=_clean_short_metadata(summary),
                files=tuple(file.strip() for file in files),
            )
        )
    if frozenset(change.scope for change in changes) != change_scopes:
        raise ValueError("change_scopes must exactly match the scopes in changes")

    experiment_payload = payload.get("experiment_payload")
    if experiment_payload is None and legacy_schema:
        experiment_payload = {}
    if not isinstance(experiment_payload, dict):
        raise ValueError("experiment_payload must be an object")
    if experiment_kind is ExperimentKind.PROMOTABLE and experiment_payload:
        raise ValueError("promotable submissions must not include experiment_payload")
    if experiment_kind is not ExperimentKind.PROMOTABLE and not experiment_payload:
        raise ValueError("non-promotable submissions require experiment_payload")

    metadata.update(
        experiment_kind=experiment_kind,
        blast_radius=blast_radius,
        change_scopes=change_scopes,
        changes=tuple(changes),
        experiment_payload=experiment_payload,
    )

    hypothesis_kind = payload.get("hypothesis_kind", "unknown")
    if hypothesis_kind not in VALID_HYPOTHESIS_KINDS:
        raise ValueError(
            "candidate metadata hypothesis_kind must be one of "
            + ", ".join(sorted(VALID_HYPOTHESIS_KINDS))
        )
    metadata["hypothesis_kind"] = hypothesis_kind
    escalation_reason = payload.get("escalation_reason", "")
    if not isinstance(escalation_reason, str):
        raise ValueError("candidate metadata escalation_reason must be a string")
    escalation_reason = escalation_reason.strip()
    if escalation_reason:
        _validate_agent_text(
            escalation_reason,
            "candidate metadata escalation_reason",
        )
    metadata["escalation_reason"] = escalation_reason
    research_evidence_ids = payload.get("research_evidence_ids", [])
    if not isinstance(research_evidence_ids, list) or not all(
        isinstance(item, str) for item in research_evidence_ids
    ):
        raise ValueError("candidate metadata research_evidence_ids must be a string array")
    metadata["research_evidence_ids"] = tuple(
        dict.fromkeys(item.strip() for item in research_evidence_ids if item.strip())
    )

    if experiment_kind is ExperimentKind.HUMAN_REVIEW:
        if source != original_source:
            raise ValueError("human_review must not modify candidate.py")
        if payload.get("commit_title", "") or payload.get("commit_summary", ""):
            raise ValueError("human_review must not provide commit metadata")
        metadata["commit_title"] = ""
        metadata["commit_summary"] = ""
        title = ""
    elif experiment_kind is not ExperimentKind.PROMOTABLE:
        if payload.get("commit_title", "") or payload.get("commit_summary", ""):
            raise ValueError("ablation submissions must not provide commit metadata")
        metadata["commit_title"] = ""
        metadata["commit_summary"] = ""
        title = ""
    else:
        title_value = payload.get("commit_title")
        if not isinstance(title_value, str) or not title_value.strip():
            raise ValueError("candidate metadata field 'commit_title' must be non-empty")
        title = title_value.strip()
        summary_value = payload.get("commit_summary")
        if not isinstance(summary_value, str) or not summary_value.strip():
            raise ValueError("candidate metadata field 'commit_summary' must be non-empty")
        metadata["commit_summary"] = summary_value.strip()
    if experiment_kind is ExperimentKind.PROMOTABLE:
        if "\n" in title or len(title) >= 80:
            raise ValueError("commit_title must be one line under 80 characters")
        title = _clean_commit_title(title)
        if title.casefold() in _GENERIC_COMMIT_TITLES:
            raise ValueError("commit_title is too generic to describe the candidate diff")
        metadata["commit_title"] = title

        summary = str(metadata["commit_summary"])
        if len(summary) >= 4000:
            raise ValueError("commit_summary must be under 4000 characters")
        match = _COMMIT_SUMMARY_RE.fullmatch(summary)
        if match is None:
            raise ValueError(
                "commit_summary must contain exactly 'Change summary:' and 'Why:' sections"
            )
        if "Performance:" in summary or "tlx agent authored" in summary.casefold():
            raise ValueError("commit_summary contains content reserved for the external harness")
        changed_names = _changed_top_level_names(original_source, source)
        if not changed_names:
            raise ValueError("candidate source does not change a top-level scope")
        if not any(name in match.group("change") for name in changed_names):
            names = ", ".join(changed_names[:8])
            raise ValueError(
                "commit_summary must name at least one changed top-level scope: " + names
            )
        if not (_commit_words(title) & _commit_words(match.group("change"))):
            raise ValueError("commit_title does not describe the commit_summary change")
        change_summary = " ".join(match.group("change").split())
        why = " ".join(match.group("why").split())
        metadata["commit_summary"] = (
            f"Change summary:\n{change_summary}\n\nWhy:\n{why}"
        )

    digest = payload.get("source_sha256")
    expected_digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != expected_digest:
        raise ValueError("candidate metadata source_sha256 does not match candidate.py")
    metadata["source_sha256"] = expected_digest

    hint_payload = payload.get("profiling_hint")
    if hint_payload is None:
        metadata["profiling_hint"] = CandidateProfilingHint()
    else:
        if not isinstance(hint_payload, dict):
            raise ValueError("profiling_hint must be a JSON object")
        pass_name = hint_payload.get("pass_name", "")
        expected_regions = hint_payload.get("expected_regions", [])
        case_ids = hint_payload.get("case_ids", [])
        rationale = hint_payload.get("rationale", "")
        if not isinstance(pass_name, str) or not isinstance(rationale, str):
            raise ValueError("profiling_hint pass_name and rationale must be strings")
        if not isinstance(expected_regions, list) or not all(
            isinstance(region, str) for region in expected_regions
        ):
            raise ValueError("profiling_hint expected_regions must be a string array")
        if not isinstance(case_ids, list) or not all(
            isinstance(case_id, str) for case_id in case_ids
        ):
            raise ValueError("profiling_hint case_ids must be a string array")
        if "\n" in rationale or len(rationale) > 240:
            raise ValueError("profiling_hint rationale must be one line under 240 characters")
        metadata["profiling_hint"] = CandidateProfilingHint(
            pass_name=pass_name.strip(),
            expected_regions=tuple(region.strip() for region in expected_regions if region.strip()),
            case_ids=tuple(case_id.strip() for case_id in case_ids if case_id.strip()),
            rationale=_clean_short_metadata(rationale),
        )
    return metadata


def _read_instrumentation_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError("instrumentation_mapping.json was not created") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"instrumentation mapping is not valid JSON: {error}") from error
    except OSError as error:
        raise ValueError(f"unable to read instrumentation mapping: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("instrumentation mapping must be a JSON object")
    return payload


def _instrumentation_summary(mapping: Mapping[str, object]) -> str:
    passes = mapping.get("passes")
    if not isinstance(passes, Mapping):
        return "Codex-generated diagnostic Proton instrumentation"
    pass_names = ", ".join(str(name) for name in sorted(passes, key=str))
    return f"Codex-generated diagnostic Proton instrumentation for passes: {pass_names}"


@dataclass(frozen=True)
class CodexCandidateProvider:
    model: str | None = None
    timeout_seconds: float = 300.0

    # Candidate generation is source-in/source-out. The model must never mutate
    # the live checkout; harness workers materialize and evaluate returned source.

    def propose(
        self,
        request: KernelOptimizationRequest,
        context: CandidateContext,
    ) -> CandidateAction:
        current_digest = _validated_current_source_digest(context)
        prompt = _build_prompt(request, context)
        try:
            with tempfile.TemporaryDirectory(prefix="tlx-agent-candidate-") as directory:
                workspace = Path(directory)
                candidate_path = workspace / "candidate.py"
                original_path = workspace / "original.py"
                output_path = workspace / "last-message.txt"
                metadata_path = workspace / "candidate_metadata.json"
                action_path = workspace / "agent_action.json"
                candidate_path.write_text(context.current_source)
                original_path.write_text(context.current_source)
                command = [
                    "codex",
                    "exec",
                    "--skip-git-repo-check",
                    "--sandbox",
                    "workspace-write",
                    "--cd",
                    str(workspace),
                    "--output-last-message",
                    str(output_path),
                ]
                if self.model:
                    command.extend(("--model", self.model))
                command.append("-")
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
                if completed.returncode != 0:
                    diagnostics = completed.stderr.strip().splitlines()
                    raise RuntimeError(
                        f"candidate generator exited with code {completed.returncode}: "
                        + " | ".join(diagnostics[-8:])
                    )
                source = candidate_path.read_text()
                if original_path.read_text() != context.current_source:
                    raise RuntimeError("candidate generator modified immutable original.py")
                action = _read_agent_action(
                    action_path,
                    current_source_digest=current_digest,
                    allowed_case_ids=tuple(case.case_id for case in request.cases),
                )
                if not isinstance(action, str):
                    if source != context.current_source:
                        raise ValueError(
                            "non-candidate agent actions must leave candidate.py unchanged"
                        )
                    return action
                metadata = _read_candidate_metadata(
                    metadata_path,
                    source=source,
                    original_source=context.current_source,
                )
        except FileNotFoundError as error:
            raise RuntimeError(
                "codex binary not found; install it or use --provider mock"
            ) from error
        if not source.strip():
            raise RuntimeError("candidate generator returned empty source")
        validate_replacement_source(source, context.current_source)
        return CandidateProposal(
            source=source,
            summary=str(metadata["change"]) or "Codex-edited candidate",
            hypothesis=str(metadata["hypothesis"]),
            hypothesis_kind=str(metadata["hypothesis_kind"]),
            escalation_reason=str(metadata["escalation_reason"]),
            research_evidence_ids=tuple(metadata["research_evidence_ids"]),
            evidence=str(metadata["evidence"]),
            expected_effect=str(metadata["expected_effect"]),
            risk=str(metadata["risk"]),
            commit_title=str(metadata["commit_title"]),
            commit_summary=str(metadata["commit_summary"]),
            change_scopes=metadata["change_scopes"],
            experiment_kind=metadata["experiment_kind"],
            blast_radius=metadata["blast_radius"],
            changes=metadata["changes"],
            experiment_payload=metadata["experiment_payload"],
            profiling_hint=metadata["profiling_hint"],
        )


def _prior_run_prompt_block(request: KernelOptimizationRequest) -> str:
    prior = request.prior_run_evidence
    if prior is None or not prior.experiments:
        return ""
    lines = []
    for experiment in prior.experiments:
        speedup = (
            f"{experiment.aggregate_speedup:.4f}x"
            if experiment.aggregate_speedup is not None
            else "unavailable"
        )
        lines.append(
            f"- {experiment.experiment_id}: status={experiment.status}, "
            f"kind={experiment.experiment_kind or 'unknown'}, "
            f"decision={experiment.decision_status or 'unknown'}, "
            f"speedup={speedup}, hypothesis={json.dumps(experiment.hypothesis)}, "
            f"change={json.dumps(experiment.change)}, "
            f"diagnostics={json.dumps(experiment.diagnostics)}"
        )
    evidence = "\n".join(lines)
    return (
        "\nPrior run evidence, read-only:\n"
        "Do not automatically adopt a prior winner. Do not repeat exact prior "
        "candidates or semantically equivalent rejected changes; use these "
        "results only to choose a new evidence-backed hypothesis.\n"
        f"{evidence[:8000]}\n"
    )


_PROMPT_OMIT_PROFILE_KEYS = frozenset(
    {
        "artifact",
        "artifacts",
        "command",
        "commands",
        "instrumentation_mapping_path",
        "instrumented_source_path",
        "profile_metadata",
        "stderr",
        "stdout",
        "trace_path",
    }
)


def _profile_prompt_summary(profile: dict[str, object]) -> dict[str, object]:
    compact = compact_profile_summary(profile)
    compact.pop("diagnostic_proton_intra_kernel", None)
    safe = _prompt_safe_profile_value(compact)
    assert isinstance(safe, dict)
    return safe


def _prompt_safe_profile_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _prompt_safe_profile_value(item)
            for key, item in value.items()
            if str(key) not in _PROMPT_OMIT_PROFILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_prompt_safe_profile_value(item) for item in value]
    return value


def _diagnostic_evidence_prompt_block(context: CandidateContext) -> str:
    evidence = "\n".join(context.baseline_diagnostic_evidence) or "None captured."
    limit = 4000
    if len(evidence) > limit:
        evidence = evidence[: limit - 16].rstrip() + "\n<truncated>"
    return f"""
Diagnostic-only intra-kernel evidence from the frozen baseline:
Instrumentation perturbs source, compiler decisions, and timing. Use this summary only to
form optimization hypotheses; diagnostic timing cannot drive promotion. Raw trace events
are intentionally omitted.
{evidence}
"""


def _agent_diagnostic_history_prompt_block(context: CandidateContext) -> str:
    payload = [
        {
            "action_id": evidence.action_id,
            "status": evidence.status,
            "source_digest": evidence.source_digest,
            "target_identity": evidence.target_identity,
            "tool": evidence.tool,
            "case_ids": list(evidence.case_ids),
            "question": evidence.question,
            "rationale": evidence.rationale,
            "level": evidence.level,
            "focus": list(evidence.focus),
            "passes": list(evidence.passes),
            "expected_regions": list(evidence.expected_regions),
            "result": {
                "available": evidence.result.available,
                "summary": _prompt_safe_profile_value(dict(evidence.result.summary)),
                "failure": evidence.result.failure,
                "tool_schema_version": evidence.result.tool_schema_version,
                "parser_schema_version": evidence.result.parser_schema_version,
            },
        }
        for evidence in context.diagnostic_evidence
    ]
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    limit = 8000
    if len(rendered) > limit:
        rendered = rendered[: limit - 16].rstrip() + "\n<truncated>"
    return f"""
Prior agent-requested diagnostic evidence for the current source:
{rendered if payload else "None captured."}
"""


def _agent_research_history_prompt_block(context: CandidateContext) -> str:
    payload = [
        {
            "action_id": evidence.action_id,
            "status": evidence.status,
            "source_digest": evidence.source_digest,
            "question": evidence.question,
            "rationale": evidence.rationale,
            "findings": list(evidence.findings),
            "excerpts": [
                {
                    "path": excerpt.path,
                    "start_line": excerpt.start_line,
                    "end_line": excerpt.end_line,
                    "symbol": excerpt.symbol,
                    "text": excerpt.text,
                }
                for excerpt in evidence.excerpts
            ],
            "inspected_paths": list(evidence.inspected_paths),
            "limitations": list(evidence.limitations),
        }
        for evidence in context.research_evidence
    ]
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    limit = 12000
    if len(rendered) > limit:
        rendered = rendered[: limit - 16].rstrip() + "\n<truncated>"
    return f"""
Read-only repository research evidence for the current candidate slot:
This evidence may support a hypothesis, but it cannot affect correctness, timing, or
promotion by itself. Preserve source provenance and account for reported limitations.
{rendered if payload else "None captured."}
"""


def _agent_action_prompt_block(
    context: CandidateContext, current_digest: str
) -> str:
    candidate_action = json.dumps(
        {"schema_version": _AGENT_ACTION_SCHEMA_VERSION, "action": "candidate"},
        indent=2,
    )
    ncu_action = json.dumps(
        {
            "schema_version": _AGENT_ACTION_SCHEMA_VERSION,
            "action": "diagnostic",
            "tool": "ncu",
            "source_digest": current_digest,
            "case_ids": ["case_id"],
            "question": "one-line question the counters will answer",
            "rationale": "one-line reason this evidence is needed before editing",
            "level": "summary",
            "focus": [],
        },
        indent=2,
    )
    proton_action = json.dumps(
        {
            "schema_version": _AGENT_ACTION_SCHEMA_VERSION,
            "action": "diagnostic",
            "tool": "proton",
            "source_digest": current_digest,
            "case_ids": ["case_id"],
            "question": "one-line source-phase or scheduling question",
            "rationale": "one-line reason this evidence is needed before editing",
            "passes": ["role"],
            "expected_regions": ["region_name"],
        },
        indent=2,
    )
    research_action = json.dumps(
        {
            "schema_version": _AGENT_ACTION_SCHEMA_VERSION,
            "action": "source_research",
            "source_digest": current_digest,
            "question": "one-line repository question needed before choosing an edit",
            "rationale": "one-line reason current source and measurements are insufficient",
            "search_terms": ["bounded search term"],
            "goals": ["transferable implementation pattern or invariant to identify"],
        },
        indent=2,
    )
    metadata = json.dumps(
        {
            "schema_version": _METADATA_SCHEMA_VERSION,
            "hypothesis": "one line under 240 characters",
            "hypothesis_kind": "parameter | memory_layout | pipeline | topology | synchronization | compiler_layout | algorithmic | unknown",
            "escalation_reason": "optional one-line reason for moving beyond local tuning",
            "research_evidence_ids": ["source research action IDs used by this hypothesis"],
            "evidence": "one line under 240 characters",
            "change": "one line under 240 characters",
            "expected_effect": "one line under 240 characters",
            "risk": "one line under 240 characters",
            "commit_title": "imperative title under 80 characters",
            "commit_summary": (
                "Change summary:\nName the changed top-level scope and preserved "
                "invariants.\n\nWhy:\nExplain the measured evidence and rationale."
            ),
            "experiment_kind": "promotable | ptx_ablation | amdgcn_ablation | ir_override | human_review",
            "blast_radius": "local | multi_file | compiler | cross_platform",
            "change_scopes": ["kernel"],
            "changes": [{"scope": "kernel", "summary": "bounded change", "files": []}],
            "experiment_payload": {},
            "source_sha256": "lowercase SHA-256 of final candidate.py bytes",
        },
        indent=2,
    )
    saturation_reached = (
        context.local_search_failure_streak
        >= context.source_research_saturation_threshold
    )
    if (
        saturation_reached
        and context.remaining_source_research_actions > 0
        and not context.research_evidence
    ):
        saturation_guidance = (
            "LOCAL SEARCH IS SATURATED: at least "
            f"{context.source_research_saturation_threshold} rejected candidate "
            "without research evidence has reached the configured threshold. Prefer one focused "
            "source_research action before another current-file-only edit. Proceed "
            "directly only when existing evidence supports a coherent non-parameter "
            "hypothesis, and record that reason in escalation_reason."
        )
    elif saturation_reached and context.remaining_source_research_actions <= 0:
        saturation_guidance = (
            "LOCAL SEARCH IS SATURATED, but source research is unavailable. "
            "Do not repeat current-file-only tuning; use existing evidence for one coherent "
            "non-parameter hypothesis."
        )
    else:
        saturation_guidance = (
            "Local search has not reached the configured source-research "
            "saturation threshold."
        )
    return f"""
Agent action protocol (schema version {_AGENT_ACTION_SCHEMA_VERSION}):
- Prefer a direct candidate whenever the supplied measurements and diagnostic evidence are
  sufficient for one testable optimization. Do not request diagnostics merely because a
  diagnostic budget remains.
- Request NCU for hardware counters: throughput, occupancy, registers, warp stalls, cache
  behavior, shared-memory bank conflicts, tensor-core utilization, or TMA efficiency. Use
  `level: "summary"` with an empty `focus`; use `level: "deep"` with 1-4 values from
  {json.dumps(sorted(VALID_NCU_FOCUS))} for a specific counter hypothesis.
- Request Proton for named source phases, warp-role timing, waits, barriers, or overlap. Use
  1-4 passes from {json.dumps(sorted(VALID_PROTON_PASSES))}; expected regions must name the
  source regions needed to answer the question. Proton timing does not replace NCU counters.
- Request a diagnostic only when a concrete hardware or source-phase evidence gap blocks
  choosing a source edit and the relevant remaining budget below is sufficient.
- Request source research only for a specific repository-level question that cannot be
  answered from the current source, measurements, diagnostics, or prior research. Use it to
  discover transferable implementation patterns, APIs, work decompositions, pipelines,
  topologies, layouts, and synchronization invariants. Do not require references to match the
  target dtype, quantization mode, framework, or operator name; structurally analogous kernels
  can be stronger evidence when their semantic differences are recorded as limitations. Do not
  request research merely because budget remains.
- Once the configured unresearched-candidate failure threshold is reached, do not continue
  current-file-only tuning by default. Follow the saturation guidance below.
- Never include commands, metrics, paths, timeouts, environment variables, profiler binary
  names, raw source, patches, or other execution controls in `agent_action.json`; the
  external harness owns execution.

Current source digest: {current_digest}
Action index: {context.action_index}
Remaining agent actions: {context.remaining_agent_actions}
Remaining diagnostic actions: {context.remaining_diagnostic_actions}
Remaining NCU collections: {context.remaining_ncu_collections}
Remaining Proton passes: {context.remaining_proton_passes}
Remaining source research actions: {context.remaining_source_research_actions}
Consecutive rejected candidates without research evidence: {context.local_search_failure_streak}
Source research saturation threshold: {context.source_research_saturation_threshold}
Saturation guidance: {saturation_guidance}

For a candidate, use exactly this `agent_action.json` schema:
```json
{candidate_action}
```
Then edit `candidate.py` as the complete replacement source and write
`candidate_metadata.json` using this schema:
```json
{metadata}
```
You may also add the existing optional `profiling_hint` object with `pass_name`, string
arrays `expected_regions` and `case_ids`, and a one-line `rationale`. Compare `original.py`
to final `candidate.py`; base metadata only on that final unified diff. The first five
metadata strings must each be one line under 240 characters. `commit_title` must be an
imperative line under 80 characters that precisely describes the actual change. Avoid
performance claims, attribution, or vague titles such as "Optimize kernel".
`commit_summary` must be under 4000 characters and contain exactly the labeled
`Change summary:` and `Why:` sections. Name at least one changed top-level function, class,
or module variable exactly, state preserved invariants or fallback paths, and do not include
`Performance:` or `TLX agent authored`. The external harness adds authoritative performance.

For an NCU diagnostic, use exactly this `agent_action.json` schema:
```json
{ncu_action}
```
For a Proton diagnostic, use exactly this `agent_action.json` schema:
```json
{proton_action}
```
For read-only repository research, use exactly this `agent_action.json` schema:
```json
{research_action}
```
For any diagnostic or source research action, leave `candidate.py` byte-for-byte unchanged
and do not write or modify `candidate_metadata.json`. The external orchestrator validates and
executes the request. When only one agent action remains, propose a candidate rather than
requesting evidence that cannot be consumed in the same slot.
"""


def _diagnostic_pass_capabilities(
    context: DiagnosticInstrumentationContext,
) -> dict[str, list[str]]:
    if context.pass_capabilities:
        return {
            str(pass_name): [str(region) for region in regions]
            for pass_name, regions in sorted(context.pass_capabilities.items())
        }
    pass_names = context.requested_passes or ("role", "coarse", "wait", "compute")
    return {str(pass_name): [] for pass_name in pass_names}


def _diagnostic_request_prompt_block(context: DiagnosticInstrumentationContext) -> str:
    capabilities = _diagnostic_pass_capabilities(context)
    diagnostics = "\n".join(context.previous_diagnostics[-5:]) or "None"
    rationale = context.rationale.strip() or "Collect diagnostic-only source-level Proton timing."
    return f"""
Diagnostic instrumentation request:
- case_id: {context.case_id or "unspecified"}
- rationale: {rationale[:_INSTRUMENTATION_PROMPT_LIMIT]}
- requested_passes: {json.dumps(list(context.requested_passes))}
- pass_capabilities: {json.dumps(capabilities, sort_keys=True)}
- previous_diagnostics:
{diagnostics[:_INSTRUMENTATION_PROMPT_LIMIT]}
"""


def _build_diagnostic_instrumentation_prompt(
    request: KernelOptimizationRequest,
    context: DiagnosticInstrumentationContext,
) -> str:
    case_lines = "\n".join(
        f"- {case.case_id}: parameters={dict(case.parameters)}, weight={case.weight}"
        for case in request.cases
    )
    capability_block = _diagnostic_request_prompt_block(context)
    return f"""You are generating diagnostic-only Triton Proton instrumentation for one existing Triton/TLX Python source file.
This is source-level instrumentation only. Do not use Triton-MPP, do not run profilers, do not benchmark, and do not modify any file except `instrumented.py` and `instrumentation_mapping.json` in this temporary directory.

Files:
- `instrumented.py` is a writable copy of the current source. Leave it as the complete instrumented Python source.
- `original.py` is immutable read-only context. Do not modify it.
- `instrumentation_mapping.json` must be strict JSON matching the schema below.

Instrumentation rules:
1. Preserve every public entry point, function/class signature, algorithm, synchronization operation, memory operation, API call, and computation. Do not optimize or refactor.
2. Add only `import triton.profiler.language as pl`, one module-level `pl.enable_semantic("triton")`, optional `_tlx_agent_proton_*` boolean predicate helper assignments using existing values or `tl.program_id`, and balanced `pl.enter_scope` / `pl.exit_scope` calls inside existing kernel functions.
3. Use literal scope names and a nonconstant predicate that selects the requested CTA plus a tile or loop iteration when one is available. Every `pl.enter_scope("name", predicate=...)` must have a matching `pl.exit_scope("name", predicate=...)` in the same lexical region and nesting order.
4. Prefer predicated scopes over control-flow wrappers. If you need a guard block, it may contain only Proton scope calls or `_tlx_agent_proton_*` helper assignments, never original computation.
5. Do not add imports other than `triton.profiler.language as pl`. Do not call `proton`, `subprocess`, `os`, `open`, `eval`, `exec`, or any profiling/analysis tool.
6. Scope selected source regions that answer the requested diagnostic passes. Common roles are load, compute/mma, wait/barrier, store/epilogue, and whole loop/tile regions. Keep the number of scopes small and names stable.

`instrumentation_mapping.json` schema:
```json
{{
  "schema_version": {INSTRUMENTATION_MAPPING_SCHEMA_VERSION},
  "diagnostic_only": true,
  "instrumentation": {{
    "backend": "instrumentation",
    "data": "trace",
    "granularity": "warp",
    "triton_semantic": true
  }},
  "expected_kernel": "nonempty regex matching the kernel name",
  "selected_cta": 0,
  "tasks": {{
    "load": {{"scope": "scope_name_present_in_source", "warps": [0]}}
  }},
  "required_scopes": ["scope_name_present_in_source"],
  "scope_kinds": {{"scope_name_present_in_source": "work_or_wait_or_other"}},
  "passes": {{
    "role": {{
      "expected_regions": ["region_name_from_pass_capabilities_when_available"],
      "scope_names": ["scope_name_present_in_source"]
    }}
  }},
  "limitations": ["one-line conservative limitation if useful"]
}}
```

Strict mapping constraints:
- Use only requested diagnostic pass names and pass capability region names supplied below when a pass has nonempty capabilities.
- `tasks` must be nonempty, task scopes must be unique, and warp ownership must not overlap.
- Every task scope, required scope, `scope_kinds` key, and pass `scope_names` entry must exactly match a literal Proton scope name in `instrumented.py`.
- Include `scope_kinds`, not `wait_scopes`, unless every listed scope is a wait scope.
- Keep all names ASCII, short, and stable. Do not include raw traces or benchmark claims.

Target: backend={request.target.backend}, architecture={request.target.architecture}
Cases:
{case_lines}
{capability_block}
After editing, compare `original.py` to `instrumented.py` yourself and ensure the only semantic difference is diagnostic Proton instrumentation. Keep your final response short; the harness reads the files.
"""


def _build_prompt(
    request: KernelOptimizationRequest,
    context: CandidateContext,
) -> str:
    case_lines = "\n".join(
        f"- {case.case_id}: parameters={dict(case.parameters)}, weight={case.weight}"
        for case in request.cases
    )
    performance_lines = "\n".join(
        f"- {case.case_id}: median_us="
        f"{case.timing.median_us if case.timing else 'unavailable'}, "
        f"p95_us={case.timing.p95_us if case.timing else 'unavailable'}, "
        f"cv={case.timing.coefficient_of_variation if case.timing else 'unavailable'}, "
        f"{_verification_metrics_prompt(case)}"
        f"profile={_profile_prompt_summary(dict(case.profile))}"
        for case in context.current_performance.cases
    )
    diagnostics = "\n".join(context.previous_diagnostics[-5:]) or "None"
    intra_kernel_lines = []
    current_digest = _validated_current_source_digest(context)
    for case in context.current_performance.cases:
        if (
            "diagnostic_proton_intra_kernel" in case.profile
            and is_profile_fresh(
                case.profile,
                current_digest,
                allow_legacy=request.profiling_policy == "legacy",
            )
            and is_valid_intra_kernel_evidence(case.profile)
        ):
            intra_kernel_lines.append(
                f"- {case.case_id}:\n{format_intra_kernel_evidence(case.profile)}"
            )
    intra_kernel_block = (
        "\nDiagnostic intra-kernel Proton evidence (instrumented, hypothesis-only):\n"
        + "\n".join(intra_kernel_lines)
        + "\n"
        if intra_kernel_lines
        else ""
    )
    performance_lines += intra_kernel_block
    reference_block = ""
    if getattr(request, "reference_kernel_source", None):
        reference_block = f"\nReference kernel (oracle, do not copy verbatim — use for correctness/performance comparison):\n```python\n{request.reference_kernel_source[:4000]}\n```\n"
    target_knowledge = load_knowledge(request.target)
    target_knowledge_block = (
        f"\nTrusted built-in target optimization knowledge:\n{target_knowledge}\n"
        if target_knowledge
        else ""
    )
    guidance = request.target.optimization_guidance.strip()
    guidance_block = (
        f"\nFrozen target-specific optimization guidance:\n{guidance}\n"
        if guidance
        else ""
    )
    prior_run_block = _prior_run_prompt_block(request)
    supported_kinds = ", ".join(
        kind.value for kind in request.target.supported_experiment_kinds
    )
    evaluation_policy_block = (
        "Evaluation policy: "
        + json.dumps(dict(request.target.evaluation_policy), sort_keys=True)
        + "\n"
        if request.target.evaluation_policy
        else ""
    )
    diagnostic_evidence_block = _diagnostic_evidence_prompt_block(context)
    diagnostic_history_block = _agent_diagnostic_history_prompt_block(context)
    research_history_block = _agent_research_history_prompt_block(context)
    action_protocol_block = _agent_action_prompt_block(context, current_digest)
    return f"""{TLX_PROMPT_PREAMBLE}
Optimization strategy:
{OPTIMIZATION_STRATEGY}
{target_knowledge_block}{guidance_block}{reference_block}{prior_run_block}
You are choosing the next action for the closed loop `build -> verify -> benchmark -> profile -> propose -> repeat`.
Do not return source or a diff, and do not claim correctness or performance; an external
deterministic harness reads the protocol files and decides both. Preserve every public entry
point expected by the harness. A candidate must make one coherent optimization that can be
diagnosed if it fails.
{action_protocol_block}

Target: backend={request.target.backend}, architecture={request.target.architecture}
Target-supported experiment kinds: {supported_kinds}
{evaluation_policy_block}Round: {context.round_index}, candidate: {context.candidate_index}
Cases:
{case_lines}

Current measurements:
{performance_lines}
{diagnostic_evidence_block}
{diagnostic_history_block}
{research_history_block}
Recent failed-candidate diagnostics:
{diagnostics}

Current source: read `candidate.py` in the working directory and follow the selected
action's mutation rules exactly.
"""


def _verification_metrics_prompt(case: CaseEvaluation) -> str:
    metrics = dict(case.verification.metrics)
    return f"metrics={metrics}, " if metrics else ""
