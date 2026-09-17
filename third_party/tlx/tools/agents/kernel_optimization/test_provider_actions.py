from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from .models import (
    DiagnosticEvidence,
    DiagnosticResult,
    InputCase,
    KernelOptimizationRequest,
    KernelTarget,
    OptimizationBudget,
    PerformanceSummary,
    ResearchEvidence,
    SourceExcerpt,
)
from .providers import (
    AgentDiagnosticRequest,
    AgentSourceResearchRequest,
    CandidateContext,
    CandidateProposal,
    CodexCandidateProvider,
    _build_prompt,
    _parse_agent_action,
    _read_candidate_metadata,
)
from .source import source_digest


_CURRENT_SOURCE = "def kernel():\n    return 1\n"
_CURRENT_DIGEST = source_digest(_CURRENT_SOURCE)


def _request() -> KernelOptimizationRequest:
    return KernelOptimizationRequest(
        kernel_source=_CURRENT_SOURCE,
        harness_path=Path(__file__),
        cases=(InputCase("primary", {}), InputCase("secondary", {})),
        target=KernelTarget("cuda", "blackwell"),
    )


def _context(
    *,
    current_source_digest: str = _CURRENT_DIGEST,
    diagnostic_evidence: tuple[DiagnosticEvidence, ...] = (),
    research_evidence: tuple[ResearchEvidence, ...] = (),
    local_search_failure_streak: int = 0,
    source_research_saturation_threshold: int = 1,
    remaining_source_research_actions: int = 1,
) -> CandidateContext:
    return CandidateContext(
        round_index=2,
        candidate_index=1,
        current_source=_CURRENT_SOURCE,
        current_performance=PerformanceSummary(cases=()),
        previous_diagnostics=(),
        action_index=3,
        current_source_digest=current_source_digest,
        remaining_agent_actions=2,
        remaining_diagnostic_actions=1,
        remaining_ncu_collections=1,
        remaining_proton_passes=3,
        remaining_source_research_actions=remaining_source_research_actions,
        diagnostic_evidence=diagnostic_evidence,
        research_evidence=research_evidence,
        local_search_failure_streak=local_search_failure_streak,
        source_research_saturation_threshold=source_research_saturation_threshold,
    )


def _ncu_action(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "action": "diagnostic",
        "tool": "ncu",
        "source_digest": _CURRENT_DIGEST,
        "case_ids": ["primary"],
        "question": "Are warp stalls limiting throughput?",
        "rationale": "The summary lacks counter evidence for the scheduling hypothesis.",
        "level": "deep",
        "focus": ["warp_stalls", "occupancy"],
    }
    payload.update(overrides)
    return payload


def _research_action(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "action": "source_research",
        "source_digest": _CURRENT_DIGEST,
        "question": "How do analogous kernels increase operand reuse?",
        "rationale": "Repeated local tuning did not improve the measured bottleneck.",
        "search_terms": ["operand reuse", "pipeline topology"],
        "goals": ["Identify a transferable work decomposition"],
    }
    payload.update(overrides)
    return payload


def _proton_action(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "action": "diagnostic",
        "tool": "proton",
        "source_digest": _CURRENT_DIGEST,
        "case_ids": ["primary", "secondary"],
        "question": "Do load and MMA phases overlap?",
        "rationale": "Phase attribution is required before changing the pipeline.",
        "passes": ["role", "wait"],
        "expected_regions": ["load", "mma"],
    }
    payload.update(overrides)
    return payload


class AgentActionParserTest(unittest.TestCase):
    def test_candidate_proposal_rejects_invalid_escalation_reason(self) -> None:
        for value in ("two\rlines", "x" * 241, " padded"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "escalation_reason"
            ):
                CandidateProposal(_CURRENT_SOURCE, escalation_reason=value)

    def test_candidate_metadata_rejects_overlong_escalation_reason(self) -> None:
        candidate = _CURRENT_SOURCE + "# optimized\n"
        payload = {
            "schema_version": 2,
            "hypothesis": "Reduce repeated work.",
            "evidence": "The profile identifies repeated work.",
            "change": "Remove repeated work.",
            "expected_effect": "Reduce instruction count.",
            "risk": "Preserve output semantics.",
            "commit_title": "Remove repeated work",
            "commit_summary": "Change summary:\nRemove repeated work.\n\nWhy:\nReduce instructions.",
            "source_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
            "hypothesis_kind": "pipeline",
            "escalation_reason": "x" * 241,
            "research_evidence_ids": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate_metadata.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "at most 240"):
                _read_candidate_metadata(
                    path,
                    source=candidate,
                    original_source=_CURRENT_SOURCE,
                )

    def test_research_saturation_threshold_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            OptimizationBudget(source_research_saturation_threshold=0)

    def test_parses_exact_candidate_action(self) -> None:
        action = _parse_agent_action(
            {"schema_version": 1, "action": "candidate"},
            current_source_digest=_CURRENT_DIGEST,
        )

        self.assertEqual(action, "candidate")

    def test_parses_exact_ncu_action(self) -> None:
        action = _parse_agent_action(
            _ncu_action(),
            current_source_digest=_CURRENT_DIGEST,
            allowed_case_ids=("primary", "secondary"),
        )

        self.assertIsInstance(action, AgentDiagnosticRequest)
        self.assertEqual(action.tool, "ncu")
        self.assertEqual(action.ncu_level, "deep")
        self.assertEqual(action.ncu_focus, ("warp_stalls", "occupancy"))
        self.assertEqual(action.proton_passes, ())

    def test_parses_exact_source_research_action(self) -> None:
        action = _parse_agent_action(
            _research_action(),
            current_source_digest=_CURRENT_DIGEST,
        )

        self.assertIsInstance(action, AgentSourceResearchRequest)
        self.assertEqual(action.search_terms, ("operand reuse", "pipeline topology"))
        self.assertEqual(
            action.goals, ("Identify a transferable work decomposition",)
        )

    def test_parses_exact_proton_action(self) -> None:
        action = _parse_agent_action(
            _proton_action(),
            current_source_digest=_CURRENT_DIGEST,
            allowed_case_ids=("primary", "secondary"),
        )

        self.assertIsInstance(action, AgentDiagnosticRequest)
        self.assertEqual(action.tool, "proton")
        self.assertEqual(action.proton_passes, ("role", "wait"))
        self.assertEqual(action.expected_regions, ("load", "mma"))
        self.assertEqual(action.ncu_level, "")

    def test_rejects_noninteger_or_unknown_schema_version(self) -> None:
        for version in (True, 2, "1"):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    _parse_agent_action(
                        {"schema_version": version, "action": "candidate"},
                        current_source_digest=_CURRENT_DIGEST,
                    )

    def test_rejects_unknown_internal_and_source_fields(self) -> None:
        cases = (
            ({"schema_version": 1, "action": "candidate", "extra": 1}, "unknown keys"),
            (dict(_ncu_action(), metrics=["raw_metric"]), "execution fields"),
            (dict(_proton_action(), source="mutated source"), "source mutation"),
            (dict(_research_action(), command="rg pattern"), "execution fields"),
        )
        for payload, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    _parse_agent_action(
                        payload,
                        current_source_digest=_CURRENT_DIGEST,
                    )

    def test_rejects_stale_digest_unknown_case_and_cross_tool_fields(self) -> None:
        cases = (
            (_ncu_action(source_digest="0" * 64), "source_digest is stale"),
            (_research_action(source_digest="0" * 64), "source_digest is stale"),
            (_ncu_action(case_ids=["unknown"]), "unknown case_ids"),
            (dict(_ncu_action(), passes=["role"]), "unknown keys: passes"),
            (dict(_proton_action(), level="deep"), "unknown keys: level"),
        )
        for payload, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    _parse_agent_action(
                        payload,
                        current_source_digest=_CURRENT_DIGEST,
                        allowed_case_ids=("primary", "secondary"),
                    )

    def test_enforces_ncu_level_and_proton_pass_rules(self) -> None:
        cases = (
            (_ncu_action(level="summary"), "summary NCU requests"),
            (_ncu_action(focus=[]), "deep NCU requests"),
            (_proton_action(passes=[]), "Proton passes"),
            (_proton_action(passes=["unknown"]), "unknown Proton passes"),
        )
        for payload, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    _parse_agent_action(
                        payload,
                        current_source_digest=_CURRENT_DIGEST,
                    )


class CodexProviderActionTest(unittest.TestCase):
    def test_returns_source_research_without_candidate_metadata(self) -> None:
        def write_action(command: list[str], **kwargs: object) -> Mock:
            del kwargs
            workspace = Path(command[command.index("--cd") + 1])
            (workspace / "agent_action.json").write_text(
                json.dumps(_research_action())
            )
            return Mock(returncode=0, stderr="")

        with patch(
            "third_party.tlx.tools.agents.kernel_optimization.providers.subprocess.run",
            side_effect=write_action,
        ):
            action = CodexCandidateProvider().propose(_request(), _context())

        self.assertIsInstance(action, AgentSourceResearchRequest)
        self.assertEqual(action.goals, ("Identify a transferable work decomposition",))

    def test_returns_diagnostic_without_candidate_metadata(self) -> None:
        def write_action(command: list[str], **kwargs: object) -> Mock:
            del kwargs
            workspace = Path(command[command.index("--cd") + 1])
            (workspace / "agent_action.json").write_text(json.dumps(_ncu_action()))
            return Mock(returncode=0, stderr="")

        with patch(
            "third_party.tlx.tools.agents.kernel_optimization.providers.subprocess.run",
            side_effect=write_action,
        ):
            action = CodexCandidateProvider().propose(_request(), _context())

        self.assertIsInstance(action, AgentDiagnosticRequest)
        self.assertEqual(action.ncu_focus, ("warp_stalls", "occupancy"))

    def test_rejects_diagnostic_that_mutates_candidate_source(self) -> None:
        def mutate_and_request(command: list[str], **kwargs: object) -> Mock:
            del kwargs
            workspace = Path(command[command.index("--cd") + 1])
            (workspace / "candidate.py").write_text(
                "def kernel():\n    return 2\n"
            )
            (workspace / "agent_action.json").write_text(json.dumps(_ncu_action()))
            return Mock(returncode=0, stderr="")

        with patch(
            "third_party.tlx.tools.agents.kernel_optimization.providers.subprocess.run",
            side_effect=mutate_and_request,
        ):
            with self.assertRaisesRegex(ValueError, "leave candidate.py unchanged"):
                CodexCandidateProvider().propose(_request(), _context())

    def test_candidate_action_uses_existing_source_and_metadata_path(self) -> None:
        candidate = "def kernel():\n    return 2\n"

        def write_candidate(command: list[str], **kwargs: object) -> Mock:
            del kwargs
            workspace = Path(command[command.index("--cd") + 1])
            (workspace / "candidate.py").write_text(candidate)
            (workspace / "agent_action.json").write_text(
                json.dumps({"schema_version": 1, "action": "candidate"})
            )
            metadata = {
                "schema_version": 2,
                "hypothesis": "Remove repeated work.",
                "hypothesis_kind": "pipeline",
                "escalation_reason": "Local parameter tuning did not reduce the repeated work.",
                "research_evidence_ids": ["research-1"],
                "evidence": "The profile attributes time to the repeated operation.",
                "change": "Fold the repeated operation in kernel.",
                "expected_effect": "Reduce instruction count.",
                "risk": "Preserve the existing return contract.",
                "commit_title": "Fold repeated work in kernel",
                "commit_summary": (
                    "Change summary:\nUpdate kernel to fold repeated work while preserving "
                    "its return contract.\n\nWhy:\nThe profile attributes time to the "
                    "repeated operation."
                ),
                "source_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
            }
            (workspace / "candidate_metadata.json").write_text(json.dumps(metadata))
            return Mock(returncode=0, stderr="")

        with patch(
            "third_party.tlx.tools.agents.kernel_optimization.providers.subprocess.run",
            side_effect=write_candidate,
        ):
            action = CodexCandidateProvider().propose(_request(), _context())

        self.assertIsInstance(action, CandidateProposal)
        self.assertEqual(action.source, candidate)
        self.assertEqual(action.commit_title, "Fold repeated work in kernel")
        self.assertEqual(action.hypothesis_kind, "pipeline")
        self.assertEqual(action.research_evidence_ids, ("research-1",))


class AgentActionPromptTest(unittest.TestCase):
    def test_prompt_contains_exact_schemas_selection_rules_and_budgets(self) -> None:
        evidence = DiagnosticEvidence(
            action_id="diag-1",
            status="collected",
            source_digest=_CURRENT_DIGEST,
            target_identity="cuda:blackwell",
            tool="ncu",
            case_ids=("primary",),
            canonical_key="key",
            question="Are warp stalls limiting throughput?",
            result=DiagnosticResult(
                available=True,
                summary={"barrier_pct": 12.5},
            ),
        )
        research = ResearchEvidence(
            action_id="research-1",
            status="collected",
            source_digest=_CURRENT_DIGEST,
            question="How do analogous kernels increase operand reuse?",
            rationale="Local tuning was exhausted.",
            findings=("A nearby pipeline reuses one operand across two output tiles.",),
            excerpts=(
                SourceExcerpt(
                    path="kernels/reference.py",
                    start_line=10,
                    end_line=12,
                    symbol="reference_kernel",
                    text="for stage in range(2):\n    consume(shared_operand)",
                ),
            ),
        )
        prompt = _build_prompt(
            _request(),
            _context(
                diagnostic_evidence=(evidence,),
                research_evidence=(research,),
            ),
        )

        self.assertIn(f"Current source digest: {_CURRENT_DIGEST}", prompt)
        self.assertIn('"action": "candidate"', prompt)
        self.assertIn('"tool": "ncu"', prompt)
        self.assertIn('"tool": "proton"', prompt)
        self.assertIn('"action": "source_research"', prompt)
        self.assertIn("Prefer a direct candidate whenever", prompt)
        self.assertIn("Request NCU for hardware counters", prompt)
        self.assertIn("Request Proton for named source phases", prompt)
        self.assertIn("Do not require references to match", prompt)
        self.assertIn("target dtype, quantization mode, framework", prompt)
        self.assertIn("leave `candidate.py` byte-for-byte unchanged", prompt)
        self.assertIn("Remaining NCU collections: 1", prompt)
        self.assertIn("Remaining Proton passes: 3", prompt)
        self.assertIn("Remaining source research actions: 1", prompt)
        self.assertIn('"barrier_pct": 12.5', prompt)
        self.assertIn("A nearby pipeline reuses one operand", prompt)
        self.assertIn("kernels/reference.py", prompt)
        self.assertNotIn("artifacts_dir", prompt)

    def test_one_unresearched_rejection_triggers_research_saturation(self) -> None:
        prompt = _build_prompt(
            _request(),
            _context(local_search_failure_streak=1),
        )

        self.assertIn("Source research saturation threshold: 1", prompt)
        self.assertIn("LOCAL SEARCH IS SATURATED", prompt)
        self.assertIn("Prefer one focused source_research action", prompt)
        self.assertIn("Proceed directly only when existing evidence supports", prompt)

    def test_configured_higher_threshold_does_not_trigger_early(self) -> None:
        prompt = _build_prompt(
            _request(),
            _context(
                local_search_failure_streak=1,
                source_research_saturation_threshold=2,
            ),
        )

        self.assertIn("Source research saturation threshold: 2", prompt)
        self.assertNotIn("LOCAL SEARCH IS SATURATED", prompt)
        self.assertIn("has not reached", prompt)

    def test_saturated_prompt_handles_exhausted_research_budget(self) -> None:
        prompt = _build_prompt(
            _request(),
            _context(
                local_search_failure_streak=1,
                remaining_source_research_actions=0,
            ),
        )

        self.assertIn("source research is unavailable", prompt)
        self.assertIn("Do not repeat current-file-only tuning", prompt)

    def test_prompt_rejects_mismatched_context_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match current_source"):
            _build_prompt(
                _request(),
                _context(current_source_digest="0" * 64),
            )
