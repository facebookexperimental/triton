from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, patch

from . import optimizer as optimizer_module
from .artifacts import load_prior_run_evidence
from .cli import (
    MAX_GUIDANCE_BYTES,
    _commit_body,
    _parse_args,
    _resolve_guidance,
    _resolve_harness_paths,
    _validate_host_matches_target,
)
from .harness import StandaloneHarness, SubprocessHarness
from .harnesses.gfx942 import att
from .models import (
    AutoCommitResult,
    CaseEvaluation,
    InputCase,
    KernelOptimizationRequest,
    KernelTarget,
    OptimizationBudget,
    PerformanceSummary,
    PriorExperimentEvidence,
    PriorRunEvidence,
    TimingSamples,
    VerificationResult,
    is_promotable,
    per_case_speedups,
    weighted_geometric_speedup,
)
from .optimizer import KernelOptimizer, _profile_log_parts
from .profiling import ProfileRequest
from .providers import (
    CandidateContext,
    CandidateProposal,
    FixedCandidateProvider,
    MockLLMProvider,
    _build_prompt,
    _read_candidate_metadata,
)
from .source import (
    apply_candidate_diff,
    extract_python_source,
    source_digest,
    validate_kernel_source,
    validate_replacement_source,
)


class ScoringTest(unittest.TestCase):

    def test_weighted_geometric_speedup(self) -> None:
        cases = (
            InputCase("large", {}, weight=3.0),
            InputCase("small", {}, weight=1.0),
        )
        baseline = _performance(("large", 100.0), ("small", 40.0))
        candidate = _performance(("large", 50.0), ("small", 80.0))
        self.assertAlmostEqual(
            weighted_geometric_speedup(baseline, candidate, cases),
            (2.0**3 * 0.5)**0.25,
        )

    def test_per_case_speedups(self) -> None:
        cases = (InputCase("a", {}), InputCase("b", {}))
        baseline = _performance(("a", 100.0), ("b", 40.0))
        candidate = _performance(("a", 50.0), ("b", 80.0))
        result = per_case_speedups(baseline, candidate, cases)
        self.assertAlmostEqual(result["a"], 2.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(result["b"], 0.5)  # type: ignore[arg-type]

    def test_per_case_speedups_none_for_failed_case(self) -> None:
        cases = (InputCase("a", {}), InputCase("b", {}, protected=False))
        baseline = _performance(("a", 100.0), ("b", 40.0))
        candidate = PerformanceSummary(cases=(
            CaseEvaluation(
                case_id="a",
                verification=VerificationResult(True),
                timing=TimingSamples((50.0, 50.0, 50.0)),
            ),
            CaseEvaluation(
                case_id="b",
                verification=VerificationResult(False),
            ),
        ))
        result = per_case_speedups(baseline, candidate, cases)
        self.assertEqual(result["b"], None)

    def test_timing_samples_p50_p95(self) -> None:
        timing = TimingSamples((10.0, 20.0, 30.0, 40.0, 50.0))
        self.assertAlmostEqual(timing.p50_us, timing.median_us)
        # p95 should be near the high end.
        self.assertGreater(timing.p95_us, timing.median_us)

    def test_extracts_and_validates_python_source(self) -> None:
        self.assertEqual(
            extract_python_source("Explanation\n```python\nVALUE = 1\n```"),
            "VALUE = 1\n",
        )
        self.assertEqual(
            extract_python_source(
                "```python\nVALUE = 1\nOTHER = 2\n```\n"
                "```python\nVALUE = 3\n```\n```python\nif:\n```"
            ),
            "VALUE = 1\nOTHER = 2\n",
        )
        with self.assertRaisesRegex(ValueError, "not valid Python"):
            extract_python_source("```python\nif:\n```")

    def test_codex_prompt_has_generic_and_target_guidance(self) -> None:
        target_guidance = (
            "Preserve the producer/consumer barrier phases. "
            "Do not retry BLOCK_SIZE=256 because baseline profiling rejected it."
        )
        request = KernelOptimizationRequest(
            kernel_source="VALUE = 1\n",
            harness_path=Path(__file__),
            cases=(InputCase("target", {"mode": "bwd"}),),
            target=KernelTarget(
                "cuda", "blackwell", optimization_guidance=target_guidance
            ),
            budget=OptimizationBudget(),
            output_dir=Path("/tmp/tlx-agent-test"),
        )
        prompt = _build_prompt(
            request,
            CandidateContext(
                round_index=1,
                candidate_index=0,
                current_source=request.kernel_source,
                current_performance=_performance(("target", 100.0)),
                previous_diagnostics=(),
            ),
        )
        self.assertIn("Evidence-driven optimization workflow", prompt)
        self.assertIn("Keep measurement scopes separate", prompt)
        self.assertIn("exactly one testable hypothesis", prompt)
        self.assertIn(".claude/skills/tlx-api-reference/SKILL.md", prompt)
        self.assertIn("Frozen target-specific optimization guidance", prompt)
        self.assertIn(target_guidance, prompt)
        self.assertNotIn("MXFP8", prompt)
        self.assertNotIn("DQ_REDUCE_NCOL", prompt)
        self.assertIn("edit `candidate.py`", prompt)
        self.assertIn("Do not modify any other file", prompt)
        self.assertIn("do not print source code or a patch", prompt)
        self.assertIn("candidate_metadata.json", prompt)
        self.assertIn("schema_version", prompt)
        self.assertIn("commit_title", prompt)
        self.assertIn("actual source change", prompt)
        self.assertIn("commit_summary", prompt)
        self.assertIn("Change summary:", prompt)
        self.assertIn("Why:", prompt)
        self.assertIn("Performance:", prompt)
        self.assertIn("external harness adds", prompt)
        self.assertIn("expected_effect", prompt)
        self.assertIn("Trusted built-in target optimization skills", prompt)
        self.assertIn("# TLX Layout Conversion Efficiency", prompt)
        self.assertIn("# NVIDIA Async TMA Output Publication", prompt)
        self.assertIn("# Blackwell Persistent CLC Scheduling", prompt)
        self.assertIn("# Blackwell Persistent Pipeline Efficiency", prompt)
        self.assertIn("optimize data movement and scheduling", prompt)
        self.assertNotIn("# NVIDIA Target Profiling With NCU", prompt)
        self.assertLess(
            prompt.index("# TLX Layout Conversion Efficiency"),
            prompt.index("# NVIDIA Async TMA Output Publication"),
        )
        self.assertLess(
            prompt.index("Trusted built-in target optimization skills"),
            prompt.index("Frozen target-specific optimization guidance"),
        )

    def test_codex_prompt_selects_nvidia_non_blackwell_skill(self) -> None:
        request = KernelOptimizationRequest(
            kernel_source="VALUE = 1\n",
            harness_path=Path(__file__),
            cases=(InputCase("target", {}),),
            target=KernelTarget("nvidia", "hopper"),
            output_dir=Path("/tmp/tlx-agent-test"),
        )
        prompt = _build_prompt(
            request,
            CandidateContext(
                1,
                0,
                request.kernel_source,
                _performance(("target", 100.0)),
                (),
            ),
        )
        self.assertIn("# TLX Layout Conversion Efficiency", prompt)
        self.assertIn("# NVIDIA Async TMA Output Publication", prompt)
        self.assertLess(
            prompt.index("# TLX Layout Conversion Efficiency"),
            prompt.index("# NVIDIA Async TMA Output Publication"),
        )
        self.assertNotIn("# Blackwell Persistent CLC Scheduling", prompt)
        self.assertNotIn("# Blackwell Persistent Pipeline Efficiency", prompt)
        self.assertNotIn("# NVIDIA Target Profiling With NCU", prompt)

    def test_codex_prompt_selects_common_skill_only_for_amd(self) -> None:
        guidance = "Preserve runtime scale behavior."
        request = KernelOptimizationRequest(
            kernel_source="VALUE = 1\n",
            harness_path=Path(__file__),
            cases=(InputCase("target", {}),),
            target=KernelTarget(
                "amd",
                "gfx950",
                optimization_guidance=guidance,
            ),
            output_dir=Path("/tmp/tlx-agent-test"),
        )
        prompt = _build_prompt(
            request,
            CandidateContext(
                1,
                0,
                request.kernel_source,
                _performance(("target", 100.0)),
                (),
            ),
        )
        self.assertIn(guidance, prompt)
        self.assertIn("Trusted built-in target optimization skills", prompt)
        self.assertIn("# TLX Layout Conversion Efficiency", prompt)
        self.assertNotIn("# NVIDIA Async TMA Output Publication", prompt)
        self.assertNotIn("# Blackwell Persistent CLC Scheduling", prompt)
        self.assertNotIn("# Blackwell Persistent Pipeline Efficiency", prompt)
        self.assertNotIn("# NVIDIA Target Profiling With NCU", prompt)
        self.assertLess(
            prompt.index("# TLX Layout Conversion Efficiency"),
            prompt.index("Frozen target-specific optimization guidance"),
        )

    def test_prompt_includes_prior_run_evidence_without_source(self) -> None:
        prior_source = "SECRET_PRIOR_SOURCE = 1\n"
        request = KernelOptimizationRequest(
            kernel_source="VALUE = 1\n",
            harness_path=Path(__file__),
            cases=(InputCase("target", {}),),
            target=KernelTarget("amd", "gfx950"),
            output_dir=Path("/tmp/tlx-agent-test"),
            prior_run_evidence=PriorRunEvidence(
                run_path=Path("/tmp/prior"),
                experiments_path=Path("/tmp/prior/experiments.json"),
                source_hashes=(source_digest(prior_source),),
                experiments=(
                    PriorExperimentEvidence(
                        experiment_id="r001-c000",
                        status="rejected",
                        hypothesis="wider tiles improve reuse",
                        change="double the tile width",
                        aggregate_speedup=0.95,
                        diagnostics="speedup below threshold",
                    ),
                ),
            ),
        )
        prompt = _build_prompt(
            request,
            CandidateContext(
                1,
                0,
                request.kernel_source,
                _performance(("target", 100.0)),
                (),
            ),
        )
        self.assertIn("Prior run evidence, read-only", prompt)
        self.assertIn("Do not automatically adopt a prior winner", prompt)
        self.assertIn("wider tiles improve reuse", prompt)
        self.assertIn("speedup=0.9500x", prompt)
        self.assertNotIn(prior_source.strip(), prompt)

    def test_candidate_metadata_reads_summary_and_builds_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate_metadata.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "hypothesis": "  reduce   work ",
                        "change": "Fold a constant scale.",
                        "commit_title": "Fold half scale into dS encoding",
                        "commit_summary": "Change summary:\nFold the scale into the encoded exponent.\n\nWhy:\nReduce repeated arithmetic while preserving the generic fallback.",
                    }
                )
            )
            metadata = _read_candidate_metadata(path)
            self.assertEqual(metadata["hypothesis"], "reduce work")
            self.assertEqual(metadata["commit_title"], "Fold half scale into dS encoding")
            self.assertIn("encoded exponent", metadata["commit_summary"])
            self.assertIn("generic fallback", metadata["commit_summary"])

            path.write_text(json.dumps({"change": "Fold a constant scale."}))
            fallback = _read_candidate_metadata(path)
            self.assertEqual(fallback["commit_title"], "Fold a constant scale")
            self.assertIn("Change summary:", fallback["commit_summary"])
            self.assertIn("Fold a constant scale.", fallback["commit_summary"])
            self.assertIn("Why:", fallback["commit_summary"])

    def test_codex_prompt_compacts_profile_and_preserves_scope_boundaries(self) -> None:
        request = KernelOptimizationRequest(
            kernel_source="VALUE = 1\n",
            harness_path=Path(__file__),
            cases=(InputCase("target", {"mode": "bwd"}),),
            target=KernelTarget("cuda", "blackwell"),
            budget=OptimizationBudget(),
            output_dir=Path("/tmp/tlx-agent-test"),
        )
        performance = PerformanceSummary(
            cases=(
                CaseEvaluation(
                    case_id="target",
                    verification=VerificationResult(True),
                    timing=TimingSamples((100.0, 100.0, 100.0)),
                    profile={
                        "level": "deep",
                        "raw": "x" * 5000,
                        "proton": {
                            "totals": {
                                "wrapper_us": 1.0,
                                "main_kernel_us": 2.0,
                                "non_main_kernel_us": 3.0,
                            }
                        },
                    },
                ),
            )
        )
        prompt = _build_prompt(
            request,
            CandidateContext(1, 0, request.kernel_source, performance, ()),
        )
        self.assertIn("infer task overlap from a launch timeline", prompt)
        self.assertIn("diagnostic intra-kernel traces", prompt)
        self.assertIn("wrapper_us", prompt)
        self.assertNotIn("xxxxx", prompt)

    def test_applies_candidate_unified_diff(self) -> None:
        current = "VALUE = 1\n\ndef kernel():\n    return VALUE\n"
        output = """```diff
--- candidate.py
+++ candidate.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
```
"""
        self.assertEqual(
            apply_candidate_diff(output, current),
            "VALUE = 2\n\ndef kernel():\n    return VALUE\n",
        )

    def test_rejects_response_without_candidate_diff(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate.py unified diff"):
            apply_candidate_diff("```python\nVALUE = 2\n```", "VALUE = 1\n")

    def test_replacement_source_requires_all_top_level_symbols(self) -> None:
        current = "def kernel():\n    return 1\n\ndef wrapper():\n    return kernel()\n"
        with self.assertRaisesRegex(ValueError, "missing top-level symbols: wrapper"):
            validate_replacement_source("def kernel():\n    return 2\n", current)

    def test_replacement_source_rejects_unexpectedly_short_source(self) -> None:
        current = "def kernel():\n    return 1\n" + ("VALUE = 1\n" * 100)
        with self.assertRaisesRegex(ValueError, "unexpectedly short"):
            validate_replacement_source("def kernel():\n    return 2\n", current)

    def test_validate_kernel_source_rejects_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_kernel_source("   \n")

    def test_kernel_optimization_request_disables_diagnostic_proton_by_default(self) -> None:
        request = KernelOptimizationRequest(
            kernel_source="VALUE = 1\n",
            harness_path=Path(__file__),
            cases=(InputCase("a", {}),),
            target=KernelTarget("fake", "fake"),
        )
        self.assertFalse(request.diagnostic_proton_intra_kernel)

    def test_failed_protected_case_is_not_promotable(self) -> None:
        summary = PerformanceSummary(
            cases=(CaseEvaluation(
                case_id="case",
                verification=VerificationResult(False),
                timing=None,
            ), ),
            aggregate_speedup=2.0,
        )
        self.assertFalse(is_promotable(summary, OptimizationBudget()))

    def test_failed_unprotected_case_does_not_block_promotion(self) -> None:
        cases = (
            InputCase("required", {}, protected=True),
            InputCase("diagnostic", {}, protected=False),
        )
        baseline = _performance(("required", 100.0), ("diagnostic", 100.0))
        candidate = PerformanceSummary(cases=(
            CaseEvaluation(
                case_id="required",
                verification=VerificationResult(True),
                timing=TimingSamples((50.0, 50.0, 50.0)),
            ),
            CaseEvaluation(
                case_id="diagnostic",
                verification=VerificationResult(False),
            ),
        ))
        speedup = weighted_geometric_speedup(baseline, candidate, cases)
        candidate = PerformanceSummary(candidate.cases, speedup)
        self.assertEqual(speedup, 2.0)
        self.assertTrue(is_promotable(candidate, OptimizationBudget(), cases))

    def test_noisy_timing_is_not_promotable(self) -> None:
        summary = PerformanceSummary(
            cases=(CaseEvaluation(
                case_id="case",
                verification=VerificationResult(True),
                timing=TimingSamples((1.0, 10.0, 1.0)),
            ), ),
            aggregate_speedup=2.0,
        )
        self.assertFalse(is_promotable(summary, OptimizationBudget(max_cv=0.1)))

    def test_source_digest_stable(self) -> None:
        self.assertEqual(source_digest("VALUE = 1\n"), source_digest("VALUE = 1\n  \n"))


class PriorRunEvidenceTest(unittest.TestCase):
    def test_loads_sources_and_sanitized_evidence_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "VALUE = 2\nCORRECT = True\n"
            source_path = root / "experiments" / "r001-c000" / "kernel.py"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(source)
            payload = [
                {
                    "experiment_id": "r001-c000",
                    "status": "rejected",
                    "source_path": "/etc/passwd",
                    "hypothesis": "  remove   a conversion  ",
                    "mutation_summary": "pin consumer layout",
                    "performance": {"aggregate_speedup": 0.99},
                    "diagnostics": "below threshold",
                }
            ]
            (root / "experiments.json").write_text(json.dumps(payload))
            before = sorted(path.relative_to(root) for path in root.rglob("*"))

            prior = load_prior_run_evidence(root)

            after = sorted(path.relative_to(root) for path in root.rglob("*"))
            self.assertEqual(before, after)
            self.assertEqual(prior.source_hashes, (source_digest(source),))
            self.assertEqual(prior.experiments[0].hypothesis, "remove a conversion")
            self.assertEqual(prior.experiments[0].aggregate_speedup, 0.99)

    def test_rejects_unsafe_experiment_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments.json").write_text(
                json.dumps([{"experiment_id": "../escape", "status": "failed"}])
            )
            with self.assertRaisesRegex(ValueError, "unsafe experiment_id"):
                load_prior_run_evidence(root)

    def test_rejects_non_list_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiments.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "must contain a list"):
                load_prior_run_evidence(path)


class CliTest(unittest.TestCase):
    def test_commit_winner_is_enabled_by_default(self) -> None:
        args = _parse_args(["--kernel", "kernel.py", "--output-dir", "/tmp/out"])
        self.assertTrue(args.commit_winner)

    def test_prior_run_arg_parses(self) -> None:
        args = _parse_args(
            [
                "--kernel",
                "kernel.py",
                "--output-dir",
                "/tmp/out",
                "--prior-run",
                "/tmp/prior",
            ]
        )
        self.assertEqual(args.prior_run, Path("/tmp/prior"))

    def test_commit_winner_can_be_disabled(self) -> None:
        args = _parse_args(
            [
                "--kernel",
                "kernel.py",
                "--output-dir",
                "/tmp/out",
                "--no-commit-winner",
            ]
        )
        self.assertFalse(args.commit_winner)

    def test_diagnostic_proton_intra_kernel_is_disabled_by_default(self) -> None:
        args = _parse_args(["--kernel", "kernel.py", "--output-dir", "/tmp/out"])
        self.assertFalse(args.diagnostic_proton_intra_kernel)

    def test_diagnostic_proton_intra_kernel_can_be_enabled(self) -> None:
        args = _parse_args(
            [
                "--kernel",
                "kernel.py",
                "--output-dir",
                "/tmp/out",
                "--diagnostic-proton-intra-kernel",
            ]
        )
        self.assertTrue(args.diagnostic_proton_intra_kernel)


class HarnessTest(unittest.TestCase):

    def test_cuda_arch_validation_rejects_mismatched_host(self) -> None:
        target = KernelTarget("cuda", "B200", device="cuda:0")
        with self.assertRaisesRegex(SystemExit, "expects sm_10x.*is sm_90"):
            _validate_host_matches_target(
                target,
                "blackwell",
                capability_probe=lambda device: (9, 0),
            )

    def test_cuda_arch_validation_accepts_matching_host(self) -> None:
        target = KernelTarget("cuda", "B200", device="cuda:0")
        _validate_host_matches_target(
            target,
            "blackwell",
            capability_probe=lambda device: (10, 0),
        )

    def test_host_arch_validation_does_not_probe_cuda(self) -> None:
        target = KernelTarget("cpu", "host", device="cpu")

        def fail_probe(device: str | None) -> tuple[int, int]:
            raise AssertionError("CPU target should not probe CUDA")

        _validate_host_matches_target(target, "host", capability_probe=fail_probe)

    def test_hip_arch_validation_accepts_matching_host(self) -> None:
        target = KernelTarget("hip", "gfx942", device="cuda:0")
        _validate_host_matches_target(
            target,
            "gfx942",
            gcn_probe=lambda device: "gfx942",
        )

    def test_hip_arch_validation_rejects_mismatched_host(self) -> None:
        target = KernelTarget("hip", "gfx942", device="cuda:0")
        with self.assertRaisesRegex(SystemExit, "expects gfx942.*is gfx950"):
            _validate_host_matches_target(
                target,
                "gfx942",
                gcn_probe=lambda device: "gfx950",
            )

    def _bundle(self, root: Path, arch_knowledge: str | None, target_guidance: str | None) -> Path:
        target_dir = root / "harnesses" / "arch" / "targets" / "kernel"
        target_dir.mkdir(parents=True)
        if arch_knowledge is not None:
            (target_dir.parent.parent / "knowledge.md").write_text(arch_knowledge)
        if target_guidance is not None:
            (target_dir / "optimization_guidance.md").write_text(target_guidance)
        return target_dir / "target.json"

    def test_guidance_concatenates_arch_then_target_then_inline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = self._bundle(Path(directory), "ARCH FACTS", "TARGET FACTS")
            guidance = _resolve_guidance(target, "INLINE")
        # Widest scope first: a target note refines the arch note, not vice versa.
        self.assertLess(guidance.index("ARCH FACTS"), guidance.index("TARGET FACTS"))
        self.assertLess(guidance.index("TARGET FACTS"), guidance.index("INLINE"))

    def test_guidance_tolerates_each_source_being_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(_resolve_guidance(self._bundle(root / "a", None, None), ""), "")
            self.assertIn("ONLY ARCH", _resolve_guidance(self._bundle(root / "b", "ONLY ARCH", None), ""))
            self.assertIn("ONLY TARGET", _resolve_guidance(self._bundle(root / "c", None, "ONLY TARGET"), ""))
            self.assertEqual(_resolve_guidance(self._bundle(root / "d", "  \n ", None), " "), "")

    def test_guidance_truncation_is_announced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = self._bundle(Path(directory), "x" * (MAX_GUIDANCE_BYTES * 2), None)
            guidance = _resolve_guidance(target, "")
        self.assertIn("truncated", guidance)
        self.assertLess(len(guidance), MAX_GUIDANCE_BYTES + 200)

    def test_shipped_gfx942_bundle_supplies_guidance(self) -> None:
        target = (Path(__file__).with_name("harnesses") / "gfx942" / "targets" / "gfx942" / "target.json")
        guidance = _resolve_guidance(target, "")
        self.assertIn("knowledge.md", guidance)
        self.assertIn("optimization_guidance.md", guidance)
        # A shipped bundle must never reach the agent already cut off.
        self.assertNotIn("truncated", guidance)

    def test_resolves_arch_first_harness_layout(self) -> None:
        kernel = Path("gemm.py")
        harness, cases, target = _resolve_harness_paths(kernel, None, None, None, "hopper")
        self.assertEqual(
            harness,
            Path(__file__).with_name("harnesses") / "hopper" / "targets" / "gemm" / "harness.py",
        )
        self.assertEqual(cases.name, "cases.json")
        self.assertEqual(target.name, "target.json")

    def test_default_arch_only_uses_arches_with_matching_target(self) -> None:
        kernel = Path("vector_add.py")
        harness, cases, target = _resolve_harness_paths(kernel, None, None, None, None)
        self.assertEqual(
            harness,
            Path(__file__).with_name("harnesses") / "host" / "targets" / "vector_add" / "harness.py",
        )
        self.assertEqual(cases.name, "cases.json")
        self.assertEqual(target.name, "target.json")

    def test_standalone_harness_evaluates_fake_kernel(self) -> None:
        harness = StandaloneHarness(Path(__file__).with_name("testdata") / "fake_harness.py")
        cases = (InputCase("a", {"scale": 1.0}), InputCase("b", {"scale": 2.0}))
        target = KernelTarget("fake", "fake")
        performance = harness.evaluate(
            "LATENCY_US = 50\nCORRECT = True\n",
            cases,
            target,
            benchmark_repetitions=3,
            profile=True,
        )
        self.assertEqual(len(performance.cases), 2)
        self.assertTrue(all(case.verification.passed for case in performance.cases))
        self.assertEqual(performance.cases[0].profile.get("bottleneck"), "synthetic")

    def test_subprocess_harness_evaluates_legacy_profile_signature(self) -> None:
        harness = SubprocessHarness(
            Path(__file__).with_name("testdata") / "fake_harness.py",
            timeout_seconds=30.0,
        )
        performance = harness.evaluate(
            "LATENCY_US = 50\nCORRECT = True\n",
            (InputCase("a", {"scale": 1.0}),),
            KernelTarget("fake", "fake"),
            benchmark_repetitions=2,
            profile=True,
        )
        self.assertEqual(performance.cases[0].profile.get("bottleneck"), "synthetic")

    def test_profile_request_passed_after_benchmark_with_case_artifact_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness_path = Path(tmp) / "three_arg_harness.py"
            harness_path.write_text(
                "from pathlib import Path\n"
                "def build(kernel_source, target):\n"
                "    return {'success': True, 'artifact': {'events': []}}\n"
                "def verify(artifact, case):\n"
                "    artifact['events'].append('verify')\n"
                "    return {'passed': True}\n"
                "def benchmark(artifact, case, repetitions):\n"
                "    artifact['events'].append('benchmark')\n"
                "    return {'samples_us': [10.0] * repetitions}\n"
                "def profile(artifact, case, request):\n"
                "    artifact['events'].append('profile')\n"
                "    assert artifact['events'] == ['verify', 'benchmark', 'profile']\n"
                "    path = Path(request['artifacts_dir'])\n"
                "    assert path.is_absolute()\n"
                "    assert path.exists()\n"
                "    return {'request': request, 'events': artifact['events']}\n"
            )
            artifacts_dir = Path(tmp) / "profiles"
            performance = StandaloneHarness(harness_path).evaluate(
                "source",
                (InputCase("shape/128?case", {}),),
                KernelTarget("cuda", "blackwell"),
                benchmark_repetitions=2,
                profile=ProfileRequest(
                    level="deep",
                    tools=("native_profiler",),
                    experiment_id="r001-c000",
                    artifacts_dir=artifacts_dir,
                    reason="unit test",
                ),
            )
            profile = performance.cases[0].profile
            request = profile["request"]
            self.assertEqual(request["level"], "deep")
            self.assertEqual(request["tools"], ["ncu"])
            per_case_dir = Path(str(request["artifacts_dir"]))
            self.assertEqual(per_case_dir.parent, artifacts_dir)
            self.assertTrue(per_case_dir.exists())
            self.assertNotIn("/", per_case_dir.name)
            self.assertEqual(profile["events"], ["verify", "benchmark", "profile"])

    def test_subprocess_harness_passes_three_arg_profile_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness_path = Path(tmp) / "three_arg_subprocess_harness.py"
            harness_path.write_text(
                "from pathlib import Path\n"
                "def build(kernel_source, target):\n"
                "    return {'success': True, 'artifact': {}}\n"
                "def verify(artifact, case):\n"
                "    return {'passed': True}\n"
                "def benchmark(artifact, case, repetitions):\n"
                "    return {'samples_us': [11.0] * repetitions}\n"
                "def profile(artifact, case, request):\n"
                "    return {'case_id': case['case_id'], 'request': request, 'exists': Path(request['artifacts_dir']).exists()}\n"
            )
            artifacts_dir = Path(tmp) / "profiles"
            performance = SubprocessHarness(harness_path, timeout_seconds=30.0).evaluate(
                "source",
                (InputCase("case a", {}),),
                KernelTarget("cuda", "blackwell"),
                benchmark_repetitions=2,
                profile={
                    "level": "summary",
                    "tools": ["proton_launch", "native_profiler"],
                    "experiment_id": "baseline",
                    "artifacts_dir": str(artifacts_dir),
                    "reason": "unit test",
                },
            )
            profile = performance.cases[0].profile
            self.assertEqual(profile["case_id"], "case a")
            self.assertTrue(profile["exists"])
            request = profile["request"]
            self.assertEqual(request["tools"], ["proton_launch", "ncu"])
            self.assertEqual(Path(str(request["artifacts_dir"])).parent, artifacts_dir)

    def test_large_profile_spills_to_raw_profile_when_artifacts_dir_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness_path = Path(tmp) / "big_profile_harness.py"
            harness_path.write_text(
                "def build(kernel_source, target):\n"
                "    return {'success': True, 'artifact': {}}\n"
                "def verify(artifact, case):\n"
                "    return {'passed': True}\n"
                "def benchmark(artifact, case, repetitions):\n"
                "    return {'samples_us': [12.0] * repetitions}\n"
                "def profile(artifact, case, request):\n"
                "    return {'big': 'x' * 2000000}\n"
            )
            artifacts_dir = Path(tmp) / "profiles"
            performance = StandaloneHarness(harness_path).evaluate(
                "source",
                (InputCase("a", {}),),
                KernelTarget("fake", "fake"),
                benchmark_repetitions=2,
                profile=ProfileRequest(artifacts_dir=artifacts_dir),
            )
            profile = performance.cases[0].profile
            self.assertEqual(profile["truncated"], True)
            artifact = Path(str(profile["artifact"]))
            self.assertTrue(artifact.is_absolute())
            self.assertTrue(artifact.exists())
            self.assertEqual(artifact.name, "raw_profile.json")
            self.assertGreater(profile["size_bytes"], 1_000_000)

    def test_standalone_harness_reports_build_error(self) -> None:
        harness = StandaloneHarness(Path(__file__).with_name("testdata") / "fake_harness.py")
        cases = (InputCase("a", {}), )
        target = KernelTarget("fake", "fake")
        with self.assertRaisesRegex(Exception, "missing fake kernel controls"):
            harness.evaluate(
                "VALUE = 1\n",
                cases,
                target,
                benchmark_repetitions=2,
            )

    def test_mock_provider_replays_canned_candidates(self) -> None:
        provider = MockLLMProvider(canned=(
            CandidateProposal("LATENCY_US = 80\nCORRECT = True\n", "first"),
            CandidateProposal("LATENCY_US = 60\nCORRECT = True\n", "second"),
        ))
        request = KernelOptimizationRequest(
            kernel_source="LATENCY_US = 100\nCORRECT = True\n",
            harness_path=Path(__file__).with_name("testdata") / "fake_harness.py",
            cases=(InputCase("a", {"scale": 1.0}), ),
            target=KernelTarget("fake", "fake"),
        )
        from .models import PerformanceSummary as PS

        ctx = provider.propose.__code__  # touch to avoid unused
        del ctx
        # Simulate two sequential proposals via the same provider instance.
        dummy_perf = PS(cases=())
        from .providers import CandidateContext

        first = provider.propose(
            request,
            CandidateContext(1, 0, request.kernel_source, dummy_perf, ()),
        )
        second = provider.propose(
            request,
            CandidateContext(1, 1, request.kernel_source, dummy_perf, ()),
        )
        self.assertEqual(first.source, "LATENCY_US = 80\nCORRECT = True\n")
        self.assertEqual(second.source, "LATENCY_US = 60\nCORRECT = True\n")

    def test_guidance_is_appended_to_the_generic_preamble(self) -> None:
        # Append, not replace: an earlier version substituted one for the other
        # and silently dropped the workflow rules.
        request = KernelOptimizationRequest(
            kernel_source="VALUE = 1\n",
            harness_path=Path(__file__),
            cases=(InputCase("a", {}), ),
            target=KernelTarget("hip", "gfx942", optimization_guidance="Ring depth is not the lever."),
        )
        dummy_perf = PerformanceSummary(cases=(), aggregate_speedup=1.0)
        prompt = _build_prompt(request, CandidateContext(0, 0, request.kernel_source, dummy_perf, ()))
        self.assertIn("Evidence-driven optimization workflow", prompt)
        self.assertIn("Ring depth is not the lever.", prompt)

    def test_prompt_has_no_guidance_block_when_the_bundle_supplies_none(self) -> None:
        request = KernelOptimizationRequest(
            kernel_source="VALUE = 1\n",
            harness_path=Path(__file__),
            cases=(InputCase("a", {}), ),
            target=KernelTarget("cuda", "sm100"),
        )
        dummy_perf = PerformanceSummary(cases=(), aggregate_speedup=1.0)
        prompt = _build_prompt(request, CandidateContext(0, 0, request.kernel_source, dummy_perf, ()))
        self.assertIn("Evidence-driven optimization workflow", prompt)
        self.assertNotIn("Frozen target-specific optimization guidance", prompt)

    def test_provider_never_reads_knowledge_from_outside_the_bundle(self) -> None:
        # input-contract.md: "Do not put such knowledge into the generic
        # candidate provider."
        import inspect

        from . import providers

        source = inspect.getsource(providers)
        self.assertNotIn("knowledge_for", source)
        self.assertNotIn("kernel_opt", source)


class CommitBodyTest(unittest.TestCase):
    def test_includes_winner_summary_and_per_case_perf(self) -> None:
        baseline = _performance(("large", 100.0), ("small", 50.0))
        final = PerformanceSummary(
            cases=_performance(("large", 80.0), ("small", 40.0)).cases,
            aggregate_speedup=1.25,
        )
        from .models import KernelOptimizationResult

        result = KernelOptimizationResult(
            success=True,
            best_kernel="source",
            baseline=baseline,
            final=final,
            experiments=(),
            artifacts_dir=Path("/tmp/result"),
            stopping_reason="round_budget_exhausted",
            winner_experiment_id="r001-c000",
            winner_commit_summary="Fold the scale into the encoded exponent.",
        )
        body = _commit_body(result)
        self.assertIn("Fold the scale into the encoded exponent.", body)
        self.assertIn("Performance:", body)
        self.assertIn("large", body)
        self.assertIn("100.00", body)
        self.assertIn("80.00", body)
        self.assertIn("1.2500x", body)
        self.assertIn("Weighted aggregate speedup: 1.2500x", body)
        self.assertIn("Correct", body)


class KernelOptimizerTest(unittest.TestCase):
    def test_prior_source_is_rejected_without_adopting_prior_winner(self) -> None:
        baseline_source = "LATENCY_US = 100\nCORRECT = True\n"
        prior_source = "LATENCY_US = 80\nCORRECT = True\n"
        provider = FixedCandidateProvider([CandidateProposal(prior_source, "duplicate")])
        prior = PriorRunEvidence(
            run_path=Path("/tmp/prior"),
            experiments_path=Path("/tmp/prior/experiments.json"),
            source_hashes=(source_digest(prior_source),),
            experiments=(
                PriorExperimentEvidence(
                    experiment_id="r001-c000",
                    status="promoted",
                    change="prior winner",
                    aggregate_speedup=1.25,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = KernelOptimizer(provider).optimize(
                KernelOptimizationRequest(
                    kernel_source=baseline_source,
                    harness_path=Path(__file__).with_name("testdata")
                    / "fake_harness.py",
                    cases=(InputCase("a", {"scale": 1.0}),),
                    target=KernelTarget("fake", "fake"),
                    budget=OptimizationBudget(
                        max_rounds=1,
                        candidates_per_round=1,
                        min_speedup=1.01,
                        benchmark_repetitions=2,
                    ),
                    output_dir=Path(directory),
                    prior_run_evidence=prior,
                )
            )
            experiment = result.experiments[-1]
            cached_source = experiment.source_path.read_text()
            incremental_patch_exists = experiment.incremental_patch_path.is_file()
            cumulative_patch_exists = experiment.cumulative_patch_path.is_file()

        self.assertFalse(result.success)
        self.assertEqual(result.winner_experiment_id, "baseline")
        self.assertEqual(result.best_kernel, baseline_source)
        self.assertEqual(experiment.status, "failed")
        self.assertIn(
            "candidate source duplicates a prior run experiment",
            experiment.diagnostics,
        )
        self.assertEqual(cached_source, prior_source)
        self.assertTrue(incremental_patch_exists)
        self.assertTrue(cumulative_patch_exists)

    def test_candidate_patches_track_promoted_parent_and_baseline(self) -> None:
        baseline_source = "LATENCY_US = 100\nCORRECT = True\n"
        first_source = "LATENCY_US = 80\nCORRECT = True\n"
        second_source = "LATENCY_US = 60\nCORRECT = True\n"
        provider = FixedCandidateProvider(
            [
                CandidateProposal(first_source, "first promotion"),
                CandidateProposal(second_source, "second promotion"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = KernelOptimizer(provider).optimize(
                KernelOptimizationRequest(
                    kernel_source=baseline_source,
                    harness_path=Path(__file__).with_name("testdata")
                    / "fake_harness.py",
                    cases=(InputCase("a", {"scale": 1.0}),),
                    target=KernelTarget("fake", "fake"),
                    budget=OptimizationBudget(
                        max_rounds=1,
                        candidates_per_round=2,
                        min_speedup=1.01,
                        benchmark_repetitions=2,
                    ),
                    output_dir=Path(directory),
                )
            )

            first, second = result.experiments[1:]
            self.assertEqual(second.parent_id, first.experiment_id)
            incremental = second.incremental_patch_path.read_text()
            cumulative = second.cumulative_patch_path.read_text()
            self.assertIn("-LATENCY_US = 80", incremental)
            self.assertIn("+LATENCY_US = 60", incremental)
            self.assertIn("-LATENCY_US = 100", cumulative)
            self.assertIn("+LATENCY_US = 60", cumulative)

    def test_failed_promotion_commit_does_not_advance_best(self) -> None:
        class FailingCommitter:
            def commit_promotion(self, experiment, source, baseline, performance):
                del experiment, source, baseline, performance
                return AutoCommitResult(
                    requested=True,
                    success=False,
                    diagnostics="checkpoint failed",
                )

            def rollback_to_baseline(self, diagnostics):
                raise AssertionError(f"unexpected rollback: {diagnostics}")

        baseline_source = "LATENCY_US = 100\nCORRECT = True\n"
        candidate_source = "LATENCY_US = 80\nCORRECT = True\n"
        with tempfile.TemporaryDirectory() as directory:
            result = KernelOptimizer(
                FixedCandidateProvider(
                    [CandidateProposal(candidate_source, "faster")]
                )
            ).optimize(
                KernelOptimizationRequest(
                    kernel_source=baseline_source,
                    harness_path=Path(__file__).with_name("testdata")
                    / "fake_harness.py",
                    cases=(InputCase("a", {"scale": 1.0}),),
                    target=KernelTarget("fake", "fake"),
                    budget=OptimizationBudget(
                        max_rounds=1,
                        candidates_per_round=1,
                        min_speedup=1.01,
                        benchmark_repetitions=2,
                    ),
                    output_dir=Path(directory),
                ),
                FailingCommitter(),
            )
            experiment = result.experiments[-1]
            self.assertTrue(experiment.source_path.is_file())
            self.assertTrue(experiment.incremental_patch_path.is_file())
            self.assertTrue(experiment.cumulative_patch_path.is_file())

        self.assertFalse(result.success)
        self.assertEqual(result.best_kernel, baseline_source)
        self.assertEqual(result.stopping_reason, "promotion_commit_failed")
        self.assertEqual(result.promotion_commits, ())
        self.assertIsNone(result.rollback_commit)
        self.assertFalse(result.auto_commit.success)
        self.assertEqual(experiment.status, "failed")

    def test_reports_performance_after_each_evaluation(self) -> None:
        provider = FixedCandidateProvider(
            [
                CandidateProposal("LATENCY_US = 120\nCORRECT = True\n", "slower"),
                CandidateProposal(
                    "LATENCY_US = 80\nCORRECT = True\n",
                    "faster",
                    commit_title="Use faster latency path",
                    commit_summary=(
                        "Change summary:\nUse the faster path.\n\n"
                        "Why:\nReduce measured latency."
                    ),
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = KernelOptimizer(provider).optimize(
                    KernelOptimizationRequest(
                        kernel_source="LATENCY_US = 100\nCORRECT = True\n",
                        harness_path=Path(__file__).with_name("testdata")
                        / "fake_harness.py",
                        cases=(InputCase("a", {"scale": 1.0}),),
                        target=KernelTarget("fake", "fake"),
                        budget=OptimizationBudget(
                            max_rounds=1,
                            candidates_per_round=2,
                            min_speedup=1.01,
                            benchmark_repetitions=3,
                        ),
                        output_dir=Path(directory),
                    )
                )

            output = stderr.getvalue()
            self.assertIn("[tlx-agent] baseline status=baseline", output)
            self.assertIn("median=100.000us", output)
            self.assertIn("[tlx-agent] r001-c000 status=artifacts", output)
            self.assertIn("incremental_patch=", output)
            self.assertIn("cumulative_patch=", output)
            self.assertIn("[tlx-agent] r001-c000 incremental-diff-begin", output)
            self.assertIn("-LATENCY_US = 100", output)
            self.assertIn("+LATENCY_US = 120", output)
            self.assertIn("[tlx-agent] r001-c000 incremental-diff-end", output)
            self.assertIn("[tlx-agent] r001-c000 status=rejected", output)
            self.assertIn("speedup=0.8333x", output)
            self.assertIn("[tlx-agent] r001-c001 change='faster'", output)
            self.assertIn("[tlx-agent] r001-c001 status=promoted", output)
            self.assertIn("speedup=1.2500x", output)
            self.assertIn("decision=correct and exceeded speedup threshold", output)
            self.assertIn("ncu=unavailable", output)
            self.assertIn("[tlx-agent] final status=revalidated", output)
            self.assertEqual(result.winner_experiment_id, "r001-c001")
            self.assertEqual(result.winner_commit_title, "Use faster latency path")
            self.assertIn("Use the faster path.", result.winner_commit_summary)
            self.assertTrue(result.success)

    def test_duplicate_candidate_patch_is_logged_before_rejection(self) -> None:
        source = "LATENCY_US = 100\nCORRECT = True\n"
        with tempfile.TemporaryDirectory() as directory:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = KernelOptimizer(
                    FixedCandidateProvider([CandidateProposal(source, "duplicate")])
                ).optimize(
                    KernelOptimizationRequest(
                        kernel_source=source,
                        harness_path=Path(__file__).with_name("testdata")
                        / "fake_harness.py",
                        cases=(InputCase("a", {"scale": 1.0}),),
                        target=KernelTarget("fake", "fake"),
                        budget=OptimizationBudget(
                            max_rounds=1,
                            candidates_per_round=1,
                            min_speedup=1.01,
                            benchmark_repetitions=2,
                        ),
                        output_dir=Path(directory),
                    )
                )

            output = stderr.getvalue()
            artifacts_at = output.index("r001-c000 status=artifacts")
            failed_at = output.index("r001-c000 status=failed")
            self.assertLess(artifacts_at, failed_at)
            self.assertIn("r001-c000 incremental-diff-begin", output)
            self.assertIn("(no source changes)", output)
            self.assertIn("r001-c000 incremental-diff-end", output)
            self.assertEqual(result.experiments[-1].status, "failed")

    def test_rejected_candidate_evidence_reaches_next_proposal(self) -> None:
        contexts: list[CandidateContext] = []
        proposals = [
            CandidateProposal(
                "LATENCY_US = 120\nCORRECT = True\n",
                summary="increase tile size",
                hypothesis="larger tiles improve reuse",
            ),
            CandidateProposal(
                "LATENCY_US = 80\nCORRECT = True\n",
                summary="reduce tile size",
            ),
        ]

        class RecordingProvider:
            def propose(
                self,
                request: KernelOptimizationRequest,
                context: CandidateContext,
            ) -> CandidateProposal:
                del request
                contexts.append(context)
                return proposals.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            result = KernelOptimizer(RecordingProvider()).optimize(
                KernelOptimizationRequest(
                    kernel_source="LATENCY_US = 100\nCORRECT = True\n",
                    harness_path=Path(__file__).with_name("testdata")
                    / "fake_harness.py",
                    cases=(InputCase("a", {"scale": 1.0}),),
                    target=KernelTarget("fake", "fake"),
                    budget=OptimizationBudget(
                        max_rounds=1,
                        candidates_per_round=2,
                        min_speedup=1.01,
                        benchmark_repetitions=3,
                    ),
                    output_dir=Path(directory),
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(len(contexts), 2)
        feedback = contexts[1].previous_diagnostics[-1]
        self.assertIn("r001-c000: rejected", feedback)
        self.assertIn("hypothesis='larger tiles improve reuse'", feedback)
        self.assertIn("change='increase tile size'", feedback)
        self.assertIn("decision=speedup below 1.0100x threshold", feedback)
        self.assertIn("aggregate_speedup=0.8333x", feedback)
        self.assertIn("median=120.000us", feedback)
        self.assertIn("cv=0.0000", feedback)
        self.assertIn("speedup=0.8333x", feedback)
        self.assertIn("ncu=unavailable", feedback)

    def test_continues_after_round_without_promotion(self) -> None:
        provider = FixedCandidateProvider(
            [
                CandidateProposal("LATENCY_US = 120\nCORRECT = True\n", "slower"),
                CandidateProposal("LATENCY_US = 80\nCORRECT = True\n", "faster"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = KernelOptimizer(provider).optimize(
                KernelOptimizationRequest(
                    kernel_source="LATENCY_US = 100\nCORRECT = True\n",
                    harness_path=Path(__file__).with_name("testdata")
                    / "fake_harness.py",
                    cases=(InputCase("a", {"scale": 1.0}),),
                    target=KernelTarget("fake", "fake"),
                    budget=OptimizationBudget(
                        max_rounds=2,
                        candidates_per_round=1,
                        min_speedup=1.01,
                        benchmark_repetitions=3,
                    ),
                    output_dir=Path(directory),
                )
            )
            self.assertTrue(result.success)
            self.assertEqual(result.best_kernel, "LATENCY_US = 80\nCORRECT = True\n")
            self.assertEqual(result.experiments[1].status, "rejected")
            self.assertEqual(result.experiments[2].status, "promoted")

    def test_rejects_incorrect_candidate_and_promotes_faster_candidate(self) -> None:
        provider = FixedCandidateProvider([
            CandidateProposal("LATENCY_US = 1\nCORRECT = False\n", "wrong"),
            CandidateProposal("LATENCY_US = 80\nCORRECT = True\n", "faster"),
            CandidateProposal("LATENCY_US = 90\nCORRECT = True\n", "slower"),
            CandidateProposal("LATENCY_US = 70\nCORRECT = True\n", "faster again"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            result = KernelOptimizer(provider).optimize(
                KernelOptimizationRequest(
                    kernel_source="LATENCY_US = 100\nCORRECT = True\n",
                    harness_path=Path(__file__).with_name("testdata") / "fake_harness.py",
                    cases=(
                        InputCase("a", {"scale": 1.0}, weight=2.0),
                        InputCase("b", {"scale": 2.0}),
                    ),
                    target=KernelTarget("fake", "fake"),
                    budget=OptimizationBudget(
                        max_rounds=2,
                        candidates_per_round=2,
                        min_speedup=1.01,
                        benchmark_repetitions=3,
                    ),
                    output_dir=Path(directory),
                ))
            self.assertTrue(result.success)
            self.assertEqual(result.best_kernel, "LATENCY_US = 70\nCORRECT = True\n")
            self.assertAlmostEqual(result.final.aggregate_speedup, 100 / 70)
            self.assertEqual(result.experiments[1].status, "rejected")
            self.assertEqual(result.experiments[2].status, "promoted")
            self.assertTrue((Path(directory) / "best_kernel.py").exists())
            self.assertTrue((Path(directory) / "result.json").exists())
            # Profile artifacts written by the optimizer.
            self.assertTrue((Path(directory) / "baseline_profile.json").exists())
            self.assertTrue((Path(directory) / "best_profile.json").exists())

    def test_failed_final_revalidation_restores_baseline_best_profile(self) -> None:
        baseline_source = "LATENCY_US = 100\nCORRECT = True\n"
        candidate_source = "LATENCY_US = 80\nCORRECT = True\n"
        baseline = PerformanceSummary(
            cases=(
                CaseEvaluation(
                    case_id="a",
                    verification=VerificationResult(True),
                    timing=TimingSamples((100.0, 100.0)),
                    profile={"marker": "baseline"},
                ),
            )
        )
        candidate = PerformanceSummary(
            cases=(
                CaseEvaluation(
                    case_id="a",
                    verification=VerificationResult(True),
                    timing=TimingSamples((80.0, 80.0)),
                    profile={"marker": "candidate"},
                ),
            )
        )
        rejected_final = PerformanceSummary(
            cases=(
                CaseEvaluation(
                    case_id="a",
                    verification=VerificationResult(True),
                    timing=TimingSamples((120.0, 120.0)),
                    profile={"marker": "rejected-final"},
                ),
            )
        )
        harness = Mock()
        harness.evaluate.side_effect = [baseline, candidate, rejected_final]
        provider = FixedCandidateProvider(
            [CandidateProposal(candidate_source, "faster before final revalidation")]
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch.object(
                optimizer_module,
                "SubprocessHarness",
                return_value=harness,
            ):
                result = KernelOptimizer(provider).optimize(
                    KernelOptimizationRequest(
                        kernel_source=baseline_source,
                        harness_path=output_dir / "unused_harness.py",
                        cases=(InputCase("a", {}),),
                        target=KernelTarget("fake", "fake"),
                        budget=OptimizationBudget(
                            max_rounds=1,
                            candidates_per_round=1,
                            min_speedup=1.01,
                            benchmark_repetitions=2,
                        ),
                        output_dir=output_dir,
                    )
                )
            baseline_profile = _read_json(output_dir / "baseline_profile.json")
            best_profile = _read_json(output_dir / "best_profile.json")
            final_profile = _read_json(output_dir / "experiments/final/profile.json")

            self.assertFalse(result.success)
            self.assertEqual(result.stopping_reason, "finalist_revalidation_failed")
            self.assertEqual(result.best_kernel, baseline_source)
            self.assertEqual(result.final.cases[0].profile["marker"], "baseline")
            self.assertEqual((output_dir / "best_kernel.py").read_text(), baseline_source)
            self.assertEqual(best_profile, baseline_profile)
            self.assertEqual(final_profile, baseline_profile)

    def test_optimizer_uses_profile_policy_and_records_profile_paths(self) -> None:
        provider = FixedCandidateProvider(
            [CandidateProposal("LATENCY_US = 80\nCORRECT = True\nNCU_US = 90\n", "faster")]
        )
        with tempfile.TemporaryDirectory() as tmp:
            harness_path = _write_policy_harness(Path(tmp))
            output_dir = Path(tmp) / "out"
            result = KernelOptimizer(provider).optimize(
                KernelOptimizationRequest(
                    kernel_source="LATENCY_US = 100\nCORRECT = True\nNCU_US = 100\n",
                    harness_path=harness_path,
                    cases=(InputCase("case/a", {"scale": 1.0}),),
                    target=KernelTarget("fake", "fake"),
                    budget=OptimizationBudget(
                        max_rounds=1,
                        candidates_per_round=1,
                        min_speedup=1.01,
                        benchmark_repetitions=2,
                    ),
                    output_dir=output_dir,
                )
            )

        baseline_request = result.baseline.cases[0].profile["request"]
        candidate_request = result.experiments[1].performance.cases[0].profile["request"]  # type: ignore[union-attr]
        final_request = result.final.cases[0].profile["request"]
        self.assertEqual(baseline_request["level"], "deep")
        expected_tools = ["proton_launch", "native_profiler"]
        self.assertEqual(baseline_request["tools"], expected_tools)
        self.assertEqual(baseline_request["reason"], "baseline")
        self.assertEqual(
            Path(str(baseline_request["artifacts_dir"])).parent,
            output_dir / "experiments" / "baseline" / "profile_artifacts",
        )
        self.assertEqual(candidate_request["level"], "summary")
        self.assertEqual(candidate_request["tools"], expected_tools)
        self.assertEqual(candidate_request["reason"], "candidate")
        self.assertEqual(
            Path(str(candidate_request["artifacts_dir"])).parent,
            output_dir / "experiments" / "r001-c000" / "profile_artifacts",
        )
        self.assertEqual(final_request["level"], "deep")
        self.assertEqual(final_request["tools"], expected_tools)
        self.assertEqual(final_request["reason"], "final")
        self.assertEqual(
            Path(str(final_request["artifacts_dir"])).parent,
            output_dir / "experiments" / "final" / "profile_artifacts",
        )
        self.assertEqual(
            result.experiments[0].profile_path,
            output_dir / "experiments" / "baseline" / "profile.json",
        )
        self.assertEqual(
            result.experiments[1].profile_path,
            output_dir / "experiments" / "r001-c000" / "profile.json",
        )
        self.assertTrue(result.success)

    def test_near_threshold_candidate_gets_deep_profile_rerun(self) -> None:
        provider = FixedCandidateProvider(
            [CandidateProposal("LATENCY_US = 100\nCORRECT = True\nTAG = 1\n", "same")]
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = KernelOptimizer(provider).optimize(
                KernelOptimizationRequest(
                    kernel_source="LATENCY_US = 100\nCORRECT = True\n",
                    harness_path=_write_policy_harness(Path(tmp)),
                    cases=(InputCase("a", {}),),
                    target=KernelTarget("fake", "fake"),
                    budget=OptimizationBudget(
                        max_rounds=1,
                        candidates_per_round=1,
                        min_speedup=1.01,
                        benchmark_repetitions=2,
                    ),
                    output_dir=Path(tmp) / "out",
                )
            )
        candidate_request = result.experiments[1].performance.cases[0].profile["request"]  # type: ignore[union-attr]
        self.assertEqual(candidate_request["level"], "deep")
        self.assertEqual(
            candidate_request["tools"],
            ["proton_launch", "native_profiler"],
        )
        self.assertEqual(candidate_request["reason"], "near_threshold")
        self.assertFalse(result.success)

    def test_ncu_regression_diagnostic_vetoes_candidate(self) -> None:
        provider = FixedCandidateProvider(
            [CandidateProposal("LATENCY_US = 80\nCORRECT = True\nNCU_US = 102\n", "fast-wrapper")]
        )
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = KernelOptimizer(provider).optimize(
                    KernelOptimizationRequest(
                        kernel_source="LATENCY_US = 100\nCORRECT = True\nNCU_US = 100\n",
                        harness_path=_write_policy_harness(Path(tmp)),
                        cases=(InputCase("a", {}),),
                        target=KernelTarget("fake", "fake"),
                        budget=OptimizationBudget(
                            max_rounds=1,
                            candidates_per_round=1,
                            min_speedup=1.01,
                            benchmark_repetitions=2,
                        ),
                        output_dir=Path(tmp) / "out",
                    )
                )
        self.assertFalse(result.success)
        self.assertEqual(result.experiments[1].status, "rejected")
        self.assertIn("NCU duration regressed", stderr.getvalue())

    def test_missing_ncu_does_not_veto_candidate(self) -> None:
        provider = FixedCandidateProvider(
            [CandidateProposal("LATENCY_US = 80\nCORRECT = True\n", "faster")]
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = KernelOptimizer(provider).optimize(
                KernelOptimizationRequest(
                    kernel_source="LATENCY_US = 100\nCORRECT = True\n",
                    harness_path=_write_policy_harness(Path(tmp)),
                    cases=(InputCase("a", {}),),
                    target=KernelTarget("fake", "fake"),
                    budget=OptimizationBudget(
                        max_rounds=1,
                        candidates_per_round=1,
                        min_speedup=1.01,
                        benchmark_repetitions=2,
                    ),
                    output_dir=Path(tmp) / "out",
                )
            )
        self.assertTrue(result.success)
        self.assertEqual(result.experiments[1].status, "promoted")

    def test_diagnostic_proton_log_includes_capture_summary(self) -> None:
        trace_path = "/tmp/selected_cta.chrome_trace"
        parts = _profile_log_parts(
            {
                "diagnostic_proton_intra_kernel": {
                    "diagnostic_proton_intra_kernel": {
                        "valid": True,
                        "selected_cta": 6,
                        "logical_coordinates": {
                            "start_n": 3840,
                            "logical_block": 31,
                            "curr_m": 3968,
                            "mma_producer_j": 32,
                            "load_input_j": 31,
                        },
                        "dominant_waits": [
                            {"name": "reduction_wait_dq", "duration": 2.852}
                        ],
                        "trace_path": trace_path,
                    }
                }
            }
        )
        self.assertIn("proton.intra.valid=true", parts)
        self.assertIn("proton.intra.cta=6", parts)
        self.assertIn(
            "proton.intra.tile=start_n:3840/logical_block:31/curr_m:3968/mma_producer_j:32/load_input_j:31",
            parts,
        )
        self.assertIn(
            "proton.intra.dominant_wait=reduction_wait_dq:2.852us", parts
        )
        self.assertIn(f"proton.intra.trace={trace_path}", parts)

    def test_diagnostic_proton_profiles_baseline_and_successful_final_only(self) -> None:
        provider = FixedCandidateProvider(
            [
                CandidateProposal("LATENCY_US = 120\nCORRECT = True\nNCU_US = 100\n", "slower"),
                CandidateProposal("LATENCY_US = 80\nCORRECT = True\nNCU_US = 90\n", "faster"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            result = KernelOptimizer(provider).optimize(
                KernelOptimizationRequest(
                    kernel_source="LATENCY_US = 100\nCORRECT = True\nNCU_US = 100\n",
                    harness_path=_write_policy_harness(root),
                    cases=(InputCase("a", {}),),
                    target=KernelTarget("fake", "fake"),
                    budget=OptimizationBudget(
                        max_rounds=1,
                        candidates_per_round=2,
                        min_speedup=1.01,
                        benchmark_repetitions=2,
                    ),
                    output_dir=output_dir,
                    diagnostic_proton_intra_kernel=True,
                )
            )
            requests = _read_profile_requests(root)
            baseline_profile = _read_json(output_dir / "baseline_profile.json")
            best_profile = _read_json(output_dir / "best_profile.json")

        diagnostic_requests = [
            request
            for request in requests
            if request["tools"] == ["proton_intra_kernel"]
        ]
        self.assertEqual(
            [(request["experiment_id"], request["reason"]) for request in diagnostic_requests],
            [("baseline", "baseline_diagnostic"), ("final", "final_winner_diagnostic")],
        )
        candidate_diagnostics = [
            request
            for request in diagnostic_requests
            if request["experiment_id"].startswith("r")
        ]
        self.assertEqual(candidate_diagnostics, [])
        self.assertTrue(result.success)
        self.assertEqual(result.experiments[1].status, "rejected")
        self.assertEqual(result.experiments[2].status, "promoted")
        self.assertEqual(result.final.aggregate_speedup, 1.25)
        self.assertIn("diagnostic_proton_intra_kernel", result.baseline.cases[0].profile)
        self.assertIn("diagnostic_proton_intra_kernel", result.final.cases[0].profile)
        self.assertNotIn(
            "diagnostic_proton_intra_kernel",
            result.experiments[1].performance.cases[0].profile,  # type: ignore[union-attr]
        )
        baseline_diag = baseline_profile["a"]["diagnostic_proton_intra_kernel"]
        self.assertEqual(baseline_diag["tools"], ["proton_intra_kernel"])
        self.assertEqual(baseline_diag["artifacts"]["trace"], "/tmp/proton.trace")
        self.assertNotIn("trace_events", baseline_diag)
        self.assertEqual(best_profile["a"]["diagnostic_proton_intra_kernel"]["summary"]["granularity"], "warp")

    def test_diagnostic_proton_does_not_duplicate_final_when_baseline_wins(self) -> None:
        provider = FixedCandidateProvider(
            [CandidateProposal("LATENCY_US = 120\nCORRECT = True\nNCU_US = 100\n", "slower")]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = KernelOptimizer(provider).optimize(
                KernelOptimizationRequest(
                    kernel_source="LATENCY_US = 100\nCORRECT = True\nNCU_US = 100\n",
                    harness_path=_write_policy_harness(root),
                    cases=(InputCase("a", {}),),
                    target=KernelTarget("fake", "fake"),
                    budget=OptimizationBudget(
                        max_rounds=1,
                        candidates_per_round=1,
                        min_speedup=1.01,
                        benchmark_repetitions=2,
                    ),
                    output_dir=root / "out",
                    diagnostic_proton_intra_kernel=True,
                )
            )
            requests = _read_profile_requests(root)

        diagnostic_requests = [
            request
            for request in requests
            if request["tools"] == ["proton_intra_kernel"]
        ]
        self.assertFalse(result.success)
        self.assertEqual(
            [(request["experiment_id"], request["reason"]) for request in diagnostic_requests],
            [("baseline", "baseline_diagnostic")],
        )
        self.assertIn("diagnostic_proton_intra_kernel", result.final.cases[0].profile)

    def test_diagnostic_proton_is_promotion_neutral(self) -> None:
        provider = FixedCandidateProvider(
            [CandidateProposal("LATENCY_US = 80\nCORRECT = True\nNCU_US = 90\n", "faster")]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = KernelOptimizer(provider).optimize(
                KernelOptimizationRequest(
                    kernel_source="LATENCY_US = 100\nCORRECT = True\nNCU_US = 100\n",
                    harness_path=_write_policy_harness(root),
                    cases=(InputCase("a", {}),),
                    target=KernelTarget("fake", "fake"),
                    budget=OptimizationBudget(
                        max_rounds=1,
                        candidates_per_round=1,
                        min_speedup=1.01,
                        benchmark_repetitions=2,
                    ),
                    output_dir=root / "out",
                    diagnostic_proton_intra_kernel=True,
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.experiments[1].status, "promoted")
        self.assertEqual(result.final.aggregate_speedup, 1.25)
        self.assertEqual(
            result.final.cases[0].profile["diagnostic_proton_intra_kernel"]["ncu"]["summary"]["duration_us"],
            999999.0,
        )

    def test_profile_payload_caps_large_values(self) -> None:
        # Use a harness that returns a >1MB profile to exercise spill logic.
        provider = FixedCandidateProvider([CandidateProposal("LATENCY_US = 100\nCORRECT = True\n", "same")])
        with tempfile.TemporaryDirectory() as tmp:
            harness_path = Path(tmp) / "big_profile_harness.py"
            harness_path.write_text("def build(kernel_source, target):\n"
                                    "    return {'success': True, 'artifact': {}}\n"
                                    "def verify(artifact, case):\n"
                                    "    return {'passed': True}\n"
                                    "def benchmark(artifact, case, repetitions):\n"
                                    "    return {'samples_us': [10.0]*repetitions}\n"
                                    "def profile(artifact, case):\n"
                                    "    return {'big': 'x' * 2000000}\n")
            with tempfile.TemporaryDirectory() as directory:
                result = KernelOptimizer(provider).optimize(
                    KernelOptimizationRequest(
                        kernel_source="LATENCY_US = 100\nCORRECT = True\n",
                        harness_path=harness_path,
                        cases=(InputCase("a", {}), ),
                        target=KernelTarget("fake", "fake"),
                        budget=OptimizationBudget(
                            max_rounds=1,
                            candidates_per_round=1,
                            benchmark_repetitions=2,
                        ),
                        output_dir=Path(directory),
                    ))
                # Optimizer should complete without error even with oversized profile.
        self.assertIsNotNone(result)


class Gfx942AttTest(unittest.TestCase):

    def test_parser_rejects_header_only_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory) / "stats_dispatch.csv"
            stats.write_text('"CodeObj","Vaddr","Instruction","Hitcount","Latency",'
                             '"Stall","Idle","Source"\n')
            result = att._summarize_att(Path(directory))
        self.assertIn("header but no instruction rows", result["parse_error"])

    def test_parser_accepts_real_att_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory) / "stats_dispatch.csv"
            stats.write_text('"CodeObj","Vaddr","Instruction","Hitcount","Latency",'
                             '"Stall","Idle","Source"\n'
                             '1,100,"s_barrier",2,20,12,1,"kernel.py:10"\n'
                             '1,104,"v_mfma_f32_16x16x16_f16 v[0:3]",4,30,8,0,'
                             '"kernel.py:11"\n')
            result = att._summarize_att(Path(directory))
        self.assertEqual(result["instruction_rows"], 2)
        self.assertEqual(result["totals"]["stall"], 20.0)
        self.assertEqual(
            [row["opcode"] for row in result["stall_by_opcode"]],
            ["barrier", "mfma"],
        )

    def test_att_command_selects_one_based_warm_dispatch(self) -> None:
        completed = __import__("subprocess").CompletedProcess([], 0, "", "")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                att, "_rocprofv3",
                return_value="/opt/rocprofv3"), mock.patch.object(att.subprocess, "run", return_value=completed) as run:
            att._run(
                mode="att",
                kernel_path=Path("kernel.py"),
                case={"case_id": "case", "parameters": {}},
                output_dir=Path(directory),
                entry_point="matmul",
                kernel_regex=".*matmul.*",
                warmup=3,
                pin_config=None,
                timeout_s=1,
            )
        command = run.call_args.args[0]
        self.assertIn("--kernel-trace", command)
        self.assertEqual(command[command.index("--kernel-iteration-range") + 1], "[4]")
        self.assertEqual(command[command.index("--att-target-cu") + 1], "0")

    def test_collect_falls_back_to_counters_when_the_att_run_fails(self) -> None:
        # Advertised --att can still fail at run time; without the fallback the
        # round would have no profile at all.
        calls: list[str] = []

        def fake_run(*, mode, **_):
            calls.append(mode)
            return {"mode": mode, "error": "decoder missing"} if mode == "att" else {"mode": mode, "counters": {}}

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                att, "capability", return_value={"att_available": True}), mock.patch.object(att, "_run", fake_run):
            result = att.collect(kernel_path=Path("k.py"), case={"case_id": "c"}, output_dir=Path(directory))
        self.assertEqual(calls, ["att", "counters"])
        self.assertEqual(result["mode"], "counters")
        self.assertEqual(result["att_unavailable_reason"], "decoder missing")

    def test_rocprofv3_override(self) -> None:
        with mock.patch.dict("os.environ", {"TLX_ROCPROFV3": "/opt/rocm-dev/bin/rocprofv3"}), mock.patch.object(
                att.shutil, "which", return_value="/opt/rocm-dev/bin/rocprofv3") as which:
            self.assertEqual(att._rocprofv3(), "/opt/rocm-dev/bin/rocprofv3")
        which.assert_called_once_with("/opt/rocm-dev/bin/rocprofv3")


def _write_policy_harness(directory: Path) -> Path:
    harness_path = directory / "policy_harness.py"
    log_path = directory / "profile_requests.jsonl"
    harness_path.write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "import re\n"
        f"LOG_PATH = {str(log_path)!r}\n"
        "def build(kernel_source, target):\n"
        "    del target\n"
        "    latency = re.search(r'LATENCY_US\\s*=\\s*([0-9.]+)', kernel_source)\n"
        "    correct = re.search(r'CORRECT\\s*=\\s*(True|False)', kernel_source)\n"
        "    ncu = re.search(r'NCU_US\\s*=\\s*([0-9.]+)', kernel_source)\n"
        "    if latency is None or correct is None:\n"
        "        return {'success': False, 'diagnostics': 'missing fake kernel controls'}\n"
        "    artifact = {\n"
        "        'latency_us': float(latency.group(1)),\n"
        "        'correct': correct.group(1) == 'True',\n"
        "        'ncu_us': float(ncu.group(1)) if ncu else None,\n"
        "    }\n"
        "    return {'success': True, 'artifact': artifact}\n"
        "def verify(artifact, case):\n"
        "    del case\n"
        "    return {'passed': artifact['correct'], 'diagnostics': '' if artifact['correct'] else 'wrong'}\n"
        "def benchmark(artifact, case, repetitions):\n"
        "    scale = float(case.get('parameters', {}).get('scale', 1.0))\n"
        "    return {'samples_us': [artifact['latency_us'] * scale] * repetitions}\n"
        "def profile(artifact, case, request):\n"
        "    del case\n"
        "    with open(LOG_PATH, 'a') as stream:\n"
        "        stream.write(json.dumps(request, sort_keys=True) + '\\n')\n"
        "    tools = request.get('tools', []) if request else []\n"
        "    if tools == ['proton_intra_kernel']:\n"
        "        return {\n"
        "            'level': request['level'],\n"
        "            'tools': tools,\n"
        "            'request': request,\n"
        "            'summary': {'active_warps': 8, 'granularity': request.get('granularity')},\n"
        "            'ncu': {'summary': {'duration_us': 999999.0}},\n"
        "            'artifacts': {'trace': '/tmp/proton.trace'},\n"
        "            'trace_events': [{'raw': 'event'}],\n"
        "            'raw_profile': 'raw trace blob',\n"
        "        }\n"
        "    proton = {'totals': {'wrapper_us': 1.0, 'main_kernel_us': 2.0, 'non_main_kernel_us': 3.0}}\n"
        "    payload = {'request': request, 'proton': proton}\n"
        "    if artifact['ncu_us'] is not None:\n"
        "        payload['ncu'] = {'summary': {'duration_us': artifact['ncu_us']}}\n"
        "    return payload\n"
    )
    return harness_path


def _read_profile_requests(directory: Path) -> list[dict[str, object]]:
    path = directory / "profile_requests.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict)
    return payload


def _performance(*values: tuple[str, float]) -> PerformanceSummary:
    return PerformanceSummary(cases=tuple(
        CaseEvaluation(
            case_id=case_id,
            verification=VerificationResult(True),
            timing=TimingSamples((latency, latency, latency)),
        ) for case_id, latency in values))


if __name__ == "__main__":
    unittest.main()
