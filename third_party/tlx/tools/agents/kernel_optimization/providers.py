from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from .models import KernelOptimizationRequest, KernelTarget, PerformanceSummary
from .profiling import compact_profile_summary
from .source import validate_replacement_source

_SKILLS_ROOT = Path(__file__).resolve().parent / "skills"
_LAYOUT_CONVERSION_SKILL = _SKILLS_ROOT / "common/layout-conversion-efficiency.md"
_NVIDIA_TARGET_SKILLS = _SKILLS_ROOT / "targets/nvidia"
_ASYNC_TMA_OUTPUT_SKILL = _NVIDIA_TARGET_SKILLS / "async-tma-output-publication.md"
_NVIDIA_WARP_BARRIER_SKILL = (
    _NVIDIA_TARGET_SKILLS / "nvidia-warp-barrier-efficiency.md"
)
_BLACKWELL_CLC_SKILL = _NVIDIA_TARGET_SKILLS / "blackwell-persistent-clc-scheduling.md"
_NVIDIA_PERSISTENT_PIPELINE_SKILL = (
    _NVIDIA_TARGET_SKILLS / "nvidia-persistent-pipeline-efficiency.md"
)
_BLACKWELL_ARCHITECTURES = frozenset(
    {"blackwell", "sm100", "sm_100", "b200", "b200a", "gb200", "gb300"}
)
_HOPPER_ARCHITECTURES = frozenset({"hopper", "h100", "sm90", "sm_90"})
_PERSISTENT_PIPELINE_ARCHITECTURES = _BLACKWELL_ARCHITECTURES | _HOPPER_ARCHITECTURES


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
write `candidate_metadata.json` with integer `schema_version` set to 2 and these string
fields: `hypothesis`, `evidence`, `change`, `expected_effect`, `risk`, `commit_title`,
`commit_summary`, and `source_sha256`. `original.py` is an immutable copy of the source you
started from. After the final edit, inspect the final unified diff from `original.py` to `candidate.py`
and base all metadata only on that diff. Set `source_sha256` to the lowercase SHA-256 digest
of the final `candidate.py` bytes so stale metadata from an earlier edit is rejected.
The first five fields must each be one line and under 240 characters. `commit_title` must be
an imperative, one-line title under 80 characters that precisely describes the actual
source change, without performance claims, vague labels such as "Optimize kernel", or
attribution. `commit_summary` must be under 4000 characters and contain exactly two clearly
labeled sections: `Change summary:` explains what changed, names at least one changed
top-level function, class, or module variable exactly as spelled in the source, and states
preserved invariants or fallback paths; `Why:` explains the measured evidence and
optimization rationale. Do not describe edits absent from the final diff. Do not include a
commit subject, a `Performance:` section, `TLX agent authored`, or any unverified
performance or correctness claim. The external harness adds a formatted `Performance:`
section with authoritative numbers after final revalidation.
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


_METADATA_SCHEMA_VERSION = 2
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
) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError("candidate_metadata.json was not created") from error
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"candidate metadata is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("candidate metadata must be a JSON object")
    if payload.get("schema_version") != _METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"candidate metadata schema_version must be {_METADATA_SCHEMA_VERSION}"
        )

    metadata: dict[str, str] = {}
    for field in (*_SHORT_METADATA_FIELDS, "commit_title", "commit_summary"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"candidate metadata field {field!r} must be non-empty")
        metadata[field] = value.strip()
    for field in _SHORT_METADATA_FIELDS:
        if "\n" in metadata[field] or len(metadata[field]) > 240:
            raise ValueError(f"candidate metadata field {field!r} must be one line under 240 characters")
        metadata[field] = _clean_short_metadata(metadata[field])

    title = metadata["commit_title"]
    if "\n" in title or len(title) >= 80:
        raise ValueError("commit_title must be one line under 80 characters")
    title = _clean_commit_title(title)
    if title.casefold() in _GENERIC_COMMIT_TITLES:
        raise ValueError("commit_title is too generic to describe the candidate diff")
    metadata["commit_title"] = title

    summary = metadata["commit_summary"]
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
                original_path = workspace / "original.py"
                output_path = workspace / "last-message.txt"
                metadata_path = workspace / "candidate_metadata.json"
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
    skills.extend((_ASYNC_TMA_OUTPUT_SKILL, _NVIDIA_WARP_BARRIER_SKILL))
    if architecture in _BLACKWELL_ARCHITECTURES:
        skills.append(_BLACKWELL_CLC_SKILL)
    if architecture in _PERSISTENT_PIPELINE_ARCHITECTURES:
        skills.append(_NVIDIA_PERSISTENT_PIPELINE_SKILL)
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
