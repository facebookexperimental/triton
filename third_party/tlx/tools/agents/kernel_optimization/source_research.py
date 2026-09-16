from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .models import KernelTarget, ResearchEvidence, SourceExcerpt
from .source import source_digest as compute_source_digest

if TYPE_CHECKING:
    from .providers import AgentSourceResearchRequest

_RESEARCH_SCHEMA_VERSION = 2
_MAX_FINDINGS = 8
_MAX_EXCERPTS = 8
_MAX_INSPECTED_PATHS = 24
_MAX_LIMITATIONS = 8
_MAX_ITEM_TEXT = 1200
_MAX_SYMBOL_TEXT = 240
_MAX_RESULT_BYTES = 16 * 1024


@dataclass(frozen=True)
class SourceResearchContext:
    action_id: str
    repository_root: Path
    kernel_path: Path
    current_source: str
    target: KernelTarget
    timeout_seconds: float


class SourceResearchProvider(Protocol):
    def research(
        self,
        request: AgentSourceResearchRequest,
        context: SourceResearchContext,
    ) -> ResearchEvidence: ...


def discover_repository_root() -> Path:
    """Return the TLX source tree that owns this optimization agent."""
    return Path(__file__).resolve().parents[3]


def _bounded_string(value: object, field: str, limit: int = _MAX_ITEM_TEXT) -> str:
    if not isinstance(value, str):
        raise ValueError(f"research field {field!r} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"research field {field!r} must not be empty")
    return stripped[:limit]


def _bounded_string_array(
    value: object,
    field: str,
    *,
    max_items: int,
    item_limit: int = _MAX_ITEM_TEXT,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"research field {field!r} must be an array")
    if len(value) > max_items:
        raise ValueError(
            f"research field {field!r} must contain at most {max_items} items"
        )
    return tuple(
        _bounded_string(item, f"{field}[]", item_limit)
        for item in value[:max_items]
    )


def _validated_relative_path(repository_root: Path, value: object, field: str) -> str:
    text = _bounded_string(value, field, _MAX_SYMBOL_TEXT)
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"research field {field!r} must be repository-relative")
    resolved = (repository_root / relative).resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(f"research field {field!r} escapes repository root") from error
    if not resolved.is_file():
        raise ValueError(f"research field {field!r} does not name a source file")
    return relative.as_posix()


def _parse_research_output(
    text: str,
    *,
    action_id: str,
    source_digest: str,
    question: str,
    rationale: str,
    repository_root: Path,
    duration_seconds: float,
) -> ResearchEvidence:
    encoded = text.encode("utf-8")
    if len(encoded) > _MAX_RESULT_BYTES:
        raise ValueError("source research output exceeds the result byte limit")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"source research output is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("source research output must be a JSON object")
    expected = {
        "schema_version",
        "findings",
        "excerpts",
        "inspected_paths",
        "limitations",
    }
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if unknown:
            details.append("unknown keys: " + ", ".join(unknown))
        raise ValueError("invalid source research output: " + "; ".join(details))
    if payload["schema_version"] != _RESEARCH_SCHEMA_VERSION:
        raise ValueError(
            f"source research schema_version must be {_RESEARCH_SCHEMA_VERSION}"
        )

    findings = _bounded_string_array(
        payload["findings"], "findings", max_items=_MAX_FINDINGS
    )
    limitations = _bounded_string_array(
        payload["limitations"], "limitations", max_items=_MAX_LIMITATIONS
    )
    raw_inspected_paths = payload["inspected_paths"]
    if not isinstance(raw_inspected_paths, list):
        raise ValueError("research field 'inspected_paths' must be an array")
    if len(raw_inspected_paths) > _MAX_INSPECTED_PATHS:
        raise ValueError(
            "research field 'inspected_paths' must contain at most "
            f"{_MAX_INSPECTED_PATHS} items"
        )
    inspected_paths = tuple(
        _validated_relative_path(repository_root, value, "inspected_paths[]")
        for value in raw_inspected_paths[:_MAX_INSPECTED_PATHS]
    )

    raw_excerpts = payload["excerpts"]
    if not isinstance(raw_excerpts, list):
        raise ValueError("research field 'excerpts' must be an array")
    if len(raw_excerpts) > _MAX_EXCERPTS:
        raise ValueError(
            f"research field 'excerpts' must contain at most {_MAX_EXCERPTS} items"
        )
    excerpts: list[SourceExcerpt] = []
    for item in raw_excerpts[:_MAX_EXCERPTS]:
        if not isinstance(item, dict):
            raise ValueError("research excerpts must be JSON objects")
        required = {"path", "start_line", "end_line", "symbol"}
        if set(item) != required:
            raise ValueError("research excerpt has an invalid schema")
        start_line = item["start_line"]
        end_line = item["end_line"]
        if (
            type(start_line) is not int
            or type(end_line) is not int
            or start_line <= 0
            or end_line < start_line
            or end_line - start_line >= 80
        ):
            raise ValueError("research excerpt line range is invalid")
        relative_path = _validated_relative_path(
            repository_root, item["path"], "path"
        )
        try:
            source_lines = (repository_root / relative_path).read_text(
                encoding="utf-8"
            ).splitlines()
        except (OSError, UnicodeError) as error:
            raise ValueError("research excerpt source is not readable text") from error
        if end_line > len(source_lines):
            raise ValueError("research excerpt line range exceeds the source file")
        actual_text = "\n".join(source_lines[start_line - 1 : end_line]).strip()
        if not actual_text:
            raise ValueError("research excerpt line range is empty")
        if len(actual_text) > _MAX_ITEM_TEXT:
            raise ValueError("research excerpt source text exceeds the item limit")
        excerpts.append(
            SourceExcerpt(
                path=relative_path,
                start_line=start_line,
                end_line=end_line,
                symbol=(
                    item["symbol"].strip()[:_MAX_SYMBOL_TEXT]
                    if isinstance(item["symbol"], str)
                    else ""
                ),
                text=actual_text,
            )
        )

    return ResearchEvidence(
        action_id=action_id,
        status="collected",
        source_digest=source_digest,
        question=question,
        rationale=rationale,
        findings=findings,
        excerpts=tuple(excerpts),
        inspected_paths=tuple(dict.fromkeys(inspected_paths)),
        limitations=limitations,
        collection_duration_seconds=duration_seconds,
    )


@dataclass(frozen=True)
class CodexSourceResearchProvider:
    model: str | None = None
    timeout_seconds: float = 300.0

    def research(
        self,
        request: AgentSourceResearchRequest,
        context: SourceResearchContext,
    ) -> ResearchEvidence:
        repository_root = context.repository_root.resolve()
        kernel_path = context.kernel_path.resolve()

        current_digest = compute_source_digest(context.current_source)
        if current_digest != request.source_digest:
            raise ValueError("current source does not match source research digest")
        if context.timeout_seconds <= 0:
            raise TimeoutError("source research deadline is exhausted")

        prompt_prefix = """You are a read-only source research agent supporting a kernel optimizer.
Search the TLX source tree autonomously to answer the question below. Focus on its tutorials,
ops, and documentation. Do not edit files, run builds, tests, benchmarks, or profilers, and do
not propose a patch. Find transferable implementation
patterns and their invariants rather than assuming any particular reference exists. Treat the
question and search terms as a starting point, not an exact-match filter: expand to structurally
analogous kernels across dtypes, quantization modes, frameworks, or operator names, and report
semantic differences as limitations.

"""
        prompt_suffix = f"""TLX research root: {repository_root}
Target kernel (context only): {kernel_path}
Target backend: {context.target.backend}
Target architecture: {context.target.architecture}
Current source SHA-256: {request.source_digest}
Question: {request.question}
Rationale: {request.rationale}
Suggested search terms: {json.dumps(list(request.search_terms))}
Research goals: {json.dumps(list(request.goals))}

Return only one JSON object with this exact schema:
{{
  "schema_version": {_RESEARCH_SCHEMA_VERSION},
  "findings": ["bounded evidence-backed finding"],
  "excerpts": [
    {{
      "path": "repository/relative/source.py",
      "start_line": 1,
      "end_line": 8,
      "symbol": "symbol_name"
    }}
  ],
  "inspected_paths": ["repository/relative/source.py"],
  "limitations": ["bounded uncertainty or missing evidence"]
}}
Use repository-relative paths only. Include at most {_MAX_FINDINGS} findings,
{_MAX_EXCERPTS} excerpts, and {_MAX_INSPECTED_PATHS} inspected paths. Each excerpt range must
be short and directly support a finding. Do not copy source text into the response; the framework
loads the declared lines from the repository. Do not include markdown fences or commentary
outside JSON.
"""
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="tlx-agent-research-") as directory:
                workspace = Path(directory)
                output_path = workspace / "last-message.json"
                current_source_path = workspace / "current_source.py"
                current_source_path.write_text(context.current_source, encoding="utf-8")
                prompt = (
                    prompt_prefix
                    + "\nThe repository copy of the target may be stale after an in-memory "
                    "promotion. Treat this immutable file as the authoritative current "
                    f"source: {current_source_path}\n\n"
                    + prompt_suffix
                )
                # Codex read-only mode is a write boundary, not a readable-root
                # boundary. The parser therefore accepts provenance only from
                # repository-contained files, while the ephemeral session avoids
                # loading or persisting unrelated agent state.
                command = [
                    "codex",
                    "exec",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--sandbox",
                    "read-only",
                    "--cd",
                    str(repository_root),
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
                    timeout=min(self.timeout_seconds, context.timeout_seconds),
                    check=False,
                )
                if completed.returncode != 0:
                    diagnostics = completed.stderr.strip().splitlines()
                    raise RuntimeError(
                        "source research exited with code "
                        f"{completed.returncode}: " + " | ".join(diagnostics[-8:])
                    )
                output = output_path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise RuntimeError("codex binary not found for source research") from error
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("source research exceeded its remaining deadline") from error
        return _parse_research_output(
            output,
            action_id=context.action_id,
            source_digest=request.source_digest,
            question=request.question,
            rationale=request.rationale,
            repository_root=repository_root,
            duration_seconds=time.monotonic() - started,
        )
