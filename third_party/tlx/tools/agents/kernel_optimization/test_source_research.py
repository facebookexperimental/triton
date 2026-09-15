from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from .models import KernelTarget
from .providers import AgentSourceResearchRequest
from .source_research import (
    CodexSourceResearchProvider,
    SourceResearchContext,
    _parse_research_output,
    discover_repository_root,
)


class SourceResearchTest(unittest.TestCase):
    def _repository(self, root: Path) -> tuple[Path, Path]:
        repository = root / "repo"
        repository.mkdir()
        (repository / ".git").mkdir()
        kernel = repository / "kernels" / "target.py"
        kernel.parent.mkdir()
        kernel.write_text("def kernel():\n    return 1\n")
        reference = repository / "kernels" / "reference.py"
        reference.write_text("def reference():\n    return 2\n")
        return repository, kernel

    def test_discovers_agent_tlx_root(self) -> None:
        repository = discover_repository_root()

        self.assertEqual(repository, Path(__file__).resolve().parents[3])
        self.assertTrue((repository / "tutorials").is_dir())
        self.assertTrue((repository / "ops").is_dir())

    def test_parses_bounded_repo_relative_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self._repository(Path(directory))
            payload = {
                "schema_version": 2,
                "findings": ["The reference splits work across two stages."],
                "excerpts": [
                    {
                        "path": "kernels/reference.py",
                        "start_line": 1,
                        "end_line": 2,
                        "symbol": "reference",
                    }
                ],
                "inspected_paths": ["kernels/reference.py"],
                "limitations": [],
            }
            evidence = _parse_research_output(
                json.dumps(payload),
                action_id="r001-c000-a00",
                source_digest="a" * 64,
                question="How is work decomposed?",
                rationale="The current source has no comparable topology.",
                repository_root=repository,
                duration_seconds=0.25,
            )

            self.assertEqual(evidence.status, "collected")
            self.assertEqual(evidence.excerpts[0].path, "kernels/reference.py")
            self.assertEqual(
                evidence.excerpts[0].text,
                "def reference():\n    return 2",
            )
            self.assertEqual(evidence.collection_duration_seconds, 0.25)

    def test_rejects_model_supplied_excerpt_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self._repository(Path(directory))
            payload = {
                "schema_version": 2,
                "findings": ["finding"],
                "excerpts": [
                    {
                        "path": "kernels/reference.py",
                        "start_line": 1,
                        "end_line": 1,
                        "symbol": "reference",
                        "text": "model-supplied source is forbidden",
                    }
                ],
                "inspected_paths": ["kernels/reference.py"],
                "limitations": [],
            }
            with self.assertRaisesRegex(ValueError, "invalid schema"):
                _parse_research_output(
                    json.dumps(payload),
                    action_id="r001-c000-a00",
                    source_digest="a" * 64,
                    question="question",
                    rationale="rationale",
                    repository_root=repository,
                    duration_seconds=0.0,
                )

    def test_rejects_research_arrays_over_their_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self._repository(Path(directory))
            base = {
                "schema_version": 2,
                "findings": ["finding"],
                "excerpts": [],
                "inspected_paths": [],
                "limitations": [],
            }
            oversized = {
                "findings": ["finding"] * 9,
                "excerpts": [{}] * 9,
                "inspected_paths": ["kernels/reference.py"] * 25,
                "limitations": ["limitation"] * 9,
            }
            for field, value in oversized.items():
                with self.subTest(field=field):
                    payload = dict(base)
                    payload[field] = value
                    with self.assertRaisesRegex(ValueError, "at most"):
                        _parse_research_output(
                            json.dumps(payload),
                            action_id="r001-c000-a00",
                            source_digest="a" * 64,
                            question="question",
                            rationale="rationale",
                            repository_root=repository,
                            duration_seconds=0.0,
                        )

    def test_rejects_absolute_or_escaping_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self._repository(Path(directory))
            for path in ("/etc/passwd", "../outside.py"):
                payload = {
                    "schema_version": 2,
                    "findings": ["finding"],
                    "excerpts": [],
                    "inspected_paths": [path],
                    "limitations": [],
                }
                with self.subTest(path=path), self.assertRaisesRegex(
                    ValueError, "repository-relative"
                ):
                    _parse_research_output(
                        json.dumps(payload),
                        action_id="r001-c000-a00",
                        source_digest="a" * 64,
                        question="question",
                        rationale="rationale",
                        repository_root=repository,
                        duration_seconds=0.0,
                    )

    def test_codex_provider_uses_read_only_repository_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, _ = self._repository(root)
            kernel = root / "external" / "target.py"
            kernel.parent.mkdir()
            kernel.write_text("def kernel():\n    return 1\n")

            def write_result(command: list[str], **kwargs: object) -> Mock:
                prompt = kwargs["input"]
                assert isinstance(prompt, str)
                source_line = next(
                    line for line in prompt.splitlines() if "source: " in line
                )
                source_path = source_line.split("source: ", 1)[1]
                self.assertEqual(
                    Path(source_path).read_text(),
                    "def kernel():\n    return 1\n",
                )
                self.assertIn("starting point, not an exact-match filter", prompt)
                self.assertIn("structurally", prompt)
                self.assertIn("analogous kernels across dtypes", prompt)
                self.assertIn(f"TLX research root: {repository.resolve()}", prompt)
                self.assertIn(
                    f"Target kernel (context only): {kernel.resolve()}", prompt
                )
                self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
                self.assertIn("--ephemeral", command)
                self.assertIn("--ignore-user-config", command)
                self.assertIn("--ignore-rules", command)
                self.assertEqual(
                    Path(command[command.index("--cd") + 1]), repository
                )
                output = Path(command[command.index("--output-last-message") + 1])
                output.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "findings": ["A reference implementation exists."],
                            "excerpts": [],
                            "inspected_paths": ["kernels/reference.py"],
                            "limitations": [],
                        }
                    )
                )
                return Mock(returncode=0, stderr="")

            current_source = "def kernel():\n    return 1\n"
            request = AgentSourceResearchRequest(
                source_digest=hashlib.sha256(current_source.encode()).hexdigest(),
                question="How do similar kernels decompose work?",
                rationale="Local parameter candidates failed.",
                search_terms=("work decomposition",),
                goals=("Find transferable topology patterns",),
            )
            context = SourceResearchContext(
                action_id="r001-c000-a00",
                repository_root=repository,
                kernel_path=kernel,
                current_source=current_source,
                target=KernelTarget("cuda", "blackwell"),
                timeout_seconds=30.0,
            )
            with patch(
                "third_party.tlx.tools.agents.kernel_optimization.source_research.subprocess.run",
                side_effect=write_result,
            ):
                evidence = CodexSourceResearchProvider().research(request, context)

            self.assertEqual(evidence.status, "collected")
            self.assertEqual(
                evidence.inspected_paths, ("kernels/reference.py",)
            )

    def test_codex_provider_clamps_to_context_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, kernel = self._repository(Path(directory))
            current_source = kernel.read_text()
            request = AgentSourceResearchRequest(
                source_digest=hashlib.sha256(current_source.encode()).hexdigest(),
                question="How do similar kernels decompose work?",
                rationale="Local parameter candidates failed.",
            )
            context = SourceResearchContext(
                action_id="r001-c000-a00",
                repository_root=repository,
                kernel_path=kernel,
                current_source=current_source,
                target=KernelTarget("cuda", "blackwell"),
                timeout_seconds=0.25,
            )
            with patch(
                "third_party.tlx.tools.agents.kernel_optimization.source_research.subprocess.run",
                side_effect=subprocess.TimeoutExpired("codex", 0.25),
            ) as run:
                with self.assertRaisesRegex(TimeoutError, "remaining deadline"):
                    CodexSourceResearchProvider(timeout_seconds=300).research(
                        request, context
                    )

            self.assertEqual(run.call_args.kwargs["timeout"], 0.25)


if __name__ == "__main__":
    unittest.main()
