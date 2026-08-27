from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import KernelOptimizationRequest, PerformanceSummary
from .profiling import compact_profile_summary
from .source import validate_replacement_source

# third_party/tlx/doc/kernel_opt/<arch>.md -- one human-curated file per
# architecture. See that directory's README for the contract; the short version
# is that a human writes it and the agent only ever reads it.
_KNOWLEDGE_DIR = Path(
    os.environ.get(
        "TLX_KERNEL_OPT_KNOWLEDGE_DIR",
        Path(__file__).resolve().parents[3] / "doc" / "kernel_opt",
    ))

# target.architecture as written in target.json -> knowledge file stem.
_ARCH_TO_KNOWLEDGE = {
    "gfx942": "gfx942",
    "gfx950": "gfx950",
    "mi300x": "gfx942",
    "mi355x": "gfx950",
}


def knowledge_for(architecture: str) -> str | None:
    """Curated prior for ``architecture``, or None when none has been written."""
    stem = _ARCH_TO_KNOWLEDGE.get(architecture.strip().lower())
    if stem is None:
        return None
    path = _KNOWLEDGE_DIR / f"{stem}.md"
    if not path.is_file():
        return None
    return path.read_text()


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
write `candidate_metadata.json` containing exactly these short string fields: `hypothesis`,
`evidence`, `change`, `expected_effect`, and `risk`. Each field must be one line and under
240 characters. Do not modify any other file. Keep the final response to one short plain-text
summary; do not print source code or a patch.
"""


@dataclass(frozen=True)
class CandidateProposal:
    source: str
    summary: str = ""
    hypothesis: str = ""
    evidence: str = ""
    expected_effect: str = ""
    risk: str = ""


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
    ) -> CandidateProposal:
        ...


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


def _read_candidate_metadata(path: Path) -> dict[str, str]:
    fields = ("hypothesis", "evidence", "change", "expected_effect", "risk")
    fallback = {field: "" for field in fields}
    if not path.exists():
        return fallback
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    return {
        field: " ".join(str(payload.get(field, "")).split())[:240]
        for field in fields
    }


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
    guidance = request.target.optimization_guidance.strip()
    guidance_block = (
        f"\nFrozen target-specific optimization guidance:\n{guidance}\n"
        if guidance
        else ""
    )
    # Appended to the generic preamble rather than replacing it: the preamble is
    # arch-agnostic workflow, this is arch-specific fact. Its two sections are
    # not equally trustworthy, so the prompt says which is which -- an untested
    # hypothesis presented as measurement is worse than no prior at all.
    knowledge = knowledge_for(request.target.architecture)
    knowledge_block = (
        f"\nHuman-curated knowledge base for {request.target.architecture}. Treat its\n"
        "'measured on' section as established fact and its 'ported from' section as\n"
        "untested hypotheses. Prefer an optimization it supports over one it does not,\n"
        f"and say which entry you are acting on.\n\n{knowledge}\n"
        if knowledge
        else ""
    )
    return f"""{TLX_PROMPT_PREAMBLE}{knowledge_block}{guidance_block}{reference_block}
You are proposing one candidate for the closed loop `build -> verify -> benchmark -> profile -> propose -> repeat`.
Edit `candidate.py` directly. Do not return source or a diff, and do not claim correctness
or performance; an external deterministic harness reads the file and decides both.
Preserve the public entry points expected by the harness. Make one coherent optimization
that can be diagnosed if it fails.

Base your proposal on the profile data below, not on general intuition. State which
measurement motivates the change; if the profile does not support any change you are
confident in, say so and make the smallest well-motivated one.

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
