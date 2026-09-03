from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from ..manager.models import KernelOptimizationRequest, KernelTarget, PerformanceSummary
from ..profiler.profiling import compact_profile_summary
from .source import validate_replacement_source

_STRATEGIES_ROOT = Path(__file__).resolve().parents[1] / "tl" / "strategies"
_LAYOUT_CONVERSION_SKILL = _STRATEGIES_ROOT / "common/layout-conversion-efficiency.md"
_NVIDIA_TARGET_SKILLS = _STRATEGIES_ROOT / "nvidia"
_ASYNC_TMA_OUTPUT_SKILL = _NVIDIA_TARGET_SKILLS / "async-tma-output-publication.md"
_BLACKWELL_CLC_SKILL = _NVIDIA_TARGET_SKILLS / "blackwell-persistent-clc-scheduling.md"
_BLACKWELL_PIPELINE_SKILL = (
    _NVIDIA_TARGET_SKILLS / "blackwell-persistent-pipeline-efficiency.md"
)
_BLACKWELL_ARCHITECTURES = frozenset(
    {"blackwell", "sm100", "sm_100", "b200", "b200a", "gb200", "gb300"}
)


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
directory. Edit that file directly and leave it as the complete replacement source. Also
write `candidate_metadata.json` with integer `schema_version` set to 1 and these string
fields: `hypothesis`, `evidence`, `change`, `expected_effect`, `risk`, `commit_title`,
and `commit_summary`. The first five fields must each be one line and under 240 characters.
`commit_title` must be an imperative, one-line title under 80 characters that describes the
actual source change, without performance claims or attribution. `commit_summary` must be
under 4000 characters and contain exactly two clearly labeled
sections: `Change summary:` explains what changed, its affected scope, and preserved
invariants or fallback paths; `Why:` explains the measured evidence and optimization
rationale. Do not include a commit subject, a `Performance:` section, `TLX agent authored`,
or any unverified performance or correctness claim. The external harness adds a formatted
`Performance:` section with authoritative numbers after final revalidation.
Do not modify any other file. Keep the final response to one short plain-text summary;
do not print source code or a patch.
"""


@dataclass(frozen=True)
class CandidateProposal:
    source: str
    summary: str = ""
    hypothesis: str = ""
    evidence: str = ""
    expected_effect: str = ""
    risk: str = ""
    commit_title: str = ""
    commit_summary: str = ""


@dataclass(frozen=True)
class CandidateContext:
    round_index: int
    candidate_index: int
    current_source: str
    current_performance: PerformanceSummary
    previous_diagnostics: tuple[str, ...]


class CandidateProvider(Protocol):
    def propose(
        self,
        request: KernelOptimizationRequest,
        context: CandidateContext,
    ) -> CandidateProposal: ...


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
    return "\n\n".join(part for part in paragraphs if part)[:4000].strip()


def _fallback_commit_summary(metadata: dict[str, str]) -> str:
    change = metadata.get("change", "") or (
        "Apply the source change selected by TLX Agent against the frozen harness."
    )
    risk = metadata.get("risk", "")
    if risk:
        change = f"{change} Preserved behavior and risk: {risk}"
    why = " ".join(
        part
        for part in (
            metadata.get("hypothesis", ""),
            f"Evidence: {metadata['evidence']}" if metadata.get("evidence") else "",
            (
                f"Expected effect: {metadata['expected_effect']}"
                if metadata.get("expected_effect")
                else ""
            ),
        )
        if part
    ) or "TLX Agent selected this candidate for external correctness and performance evaluation."
    return _clean_commit_summary(f"Change summary:\n{change}\n\nWhy:\n{why}")


def _read_candidate_metadata(path: Path) -> dict[str, str]:
    metadata = {field: "" for field in _SHORT_METADATA_FIELDS}
    metadata["commit_title"] = ""
    metadata["commit_summary"] = ""
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            payload = {}
        if isinstance(payload, dict):
            for field in _SHORT_METADATA_FIELDS:
                metadata[field] = _clean_short_metadata(payload.get(field, ""))
            metadata["commit_title"] = _clean_commit_title(
                payload.get("commit_title", "")
            )
            metadata["commit_summary"] = _clean_commit_summary(
                payload.get("commit_summary", "")
            )
    if not metadata["commit_title"]:
        metadata["commit_title"] = _clean_commit_title(metadata.get("change", ""))
    if not metadata["commit_summary"]:
        metadata["commit_summary"] = _fallback_commit_summary(metadata)
    return metadata


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
    ) -> CandidateProposal:
        prompt = _build_prompt(request, context)
        try:
            with tempfile.TemporaryDirectory(prefix="tlx-agent-candidate-") as directory:
                workspace = Path(directory)
                candidate_path = workspace / "candidate.py"
                output_path = workspace / "last-message.txt"
                metadata_path = workspace / "candidate_metadata.json"
                candidate_path.write_text(context.current_source)
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
                metadata = _read_candidate_metadata(metadata_path)
        except FileNotFoundError as error:
            raise RuntimeError(
                "codex binary not found; install it or use --provider mock"
            ) from error
        if not source.strip():
            raise RuntimeError("candidate generator returned empty source")
        validate_replacement_source(source, context.current_source)
        return CandidateProposal(
            source=source,
            summary=metadata["change"] or "Codex-edited candidate",
            hypothesis=metadata["hypothesis"],
            evidence=metadata["evidence"],
            expected_effect=metadata["expected_effect"],
            risk=metadata["risk"],
            commit_title=metadata["commit_title"],
            commit_summary=metadata["commit_summary"],
        )


@lru_cache(maxsize=None)
def _read_target_skill(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(f"unable to read built-in target skill {path}: {error}") from error


def _target_skill_paths(target: KernelTarget) -> tuple[Path, ...]:
    skills = [_LAYOUT_CONVERSION_SKILL]
    backend = target.backend.strip().lower()
    if backend not in {"cuda", "nvidia"}:
        return tuple(skills)
    architecture = target.architecture.strip().lower()
    skills.append(_ASYNC_TMA_OUTPUT_SKILL)
    if architecture in _BLACKWELL_ARCHITECTURES:
        skills.extend((_BLACKWELL_CLC_SKILL, _BLACKWELL_PIPELINE_SKILL))
    return tuple(skills)


def _target_skill_guidance(target: KernelTarget) -> str:
    return "\n\n".join(_read_target_skill(path) for path in _target_skill_paths(target))


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
        f"profile={compact_profile_summary(case.profile)}"
        for case in context.current_performance.cases
    )
    diagnostics = "\n".join(context.previous_diagnostics[-5:]) or "None"
    reference_block = ""
    if getattr(request, "reference_kernel_source", None):
        reference_block = f"\nReference kernel (oracle, do not copy verbatim — use for correctness/performance comparison):\n```python\n{request.reference_kernel_source[:4000]}\n```\n"
    target_skills = _target_skill_guidance(request.target)
    target_skills_block = (
        f"\nTrusted built-in target optimization skills:\n{target_skills}\n"
        if target_skills
        else ""
    )
    guidance = request.target.optimization_guidance.strip()
    guidance_block = (
        f"\nFrozen target-specific optimization guidance:\n{guidance}\n"
        if guidance
        else ""
    )
    prior_run_block = _prior_run_prompt_block(request)
    return f"""{TLX_PROMPT_PREAMBLE}{target_skills_block}{guidance_block}{reference_block}{prior_run_block}
You are proposing one candidate for the closed loop `build -> verify -> benchmark -> profile -> propose -> repeat`.
Edit `candidate.py` directly. Do not return source or a diff, and do not claim correctness
or performance; an external deterministic harness reads the file and decides both.
Preserve the public entry points expected by the harness. Make one coherent optimization
that can be diagnosed if it fails.

Target: backend={request.target.backend}, architecture={request.target.architecture}
Round: {context.round_index}, candidate: {context.candidate_index}
Cases:
{case_lines}

Current measurements:
{performance_lines}

Recent failed-candidate diagnostics:
{diagnostics}

Current source: read and edit `candidate.py` in the working directory.
"""
