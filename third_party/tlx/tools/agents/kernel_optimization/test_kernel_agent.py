from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .harness import StandaloneHarness
from .models import (
    CaseEvaluation,
    InputCase,
    KernelOptimizationRequest,
    KernelTarget,
    OptimizationBudget,
    PerformanceSummary,
    TimingSamples,
    VerificationResult,
    is_promotable,
    per_case_speedups,
    weighted_geometric_speedup,
)
from .optimizer import KernelOptimizer
from .providers import CandidateProposal, FixedCandidateProvider, MockLLMProvider
from .source import extract_python_source, source_digest, validate_kernel_source


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
            (2.0**3 * 0.5) ** 0.25,
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
        candidate = PerformanceSummary(
            cases=(
                CaseEvaluation(
                    case_id="a",
                    verification=VerificationResult(True),
                    timing=TimingSamples((50.0, 50.0, 50.0)),
                ),
                CaseEvaluation(
                    case_id="b",
                    verification=VerificationResult(False),
                ),
            )
        )
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
        with self.assertRaisesRegex(ValueError, "not valid Python"):
            extract_python_source("```python\nif:\n```")

    def test_validate_kernel_source_rejects_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_kernel_source("   \n")

    def test_failed_protected_case_is_not_promotable(self) -> None:
        summary = PerformanceSummary(
            cases=(
                CaseEvaluation(
                    case_id="case",
                    verification=VerificationResult(False),
                    timing=None,
                ),
            ),
            aggregate_speedup=2.0,
        )
        self.assertFalse(is_promotable(summary, OptimizationBudget()))

    def test_failed_unprotected_case_does_not_block_promotion(self) -> None:
        cases = (
            InputCase("required", {}, protected=True),
            InputCase("diagnostic", {}, protected=False),
        )
        baseline = _performance(("required", 100.0), ("diagnostic", 100.0))
        candidate = PerformanceSummary(
            cases=(
                CaseEvaluation(
                    case_id="required",
                    verification=VerificationResult(True),
                    timing=TimingSamples((50.0, 50.0, 50.0)),
                ),
                CaseEvaluation(
                    case_id="diagnostic",
                    verification=VerificationResult(False),
                ),
            )
        )
        speedup = weighted_geometric_speedup(baseline, candidate, cases)
        candidate = PerformanceSummary(candidate.cases, speedup)
        self.assertEqual(speedup, 2.0)
        self.assertTrue(is_promotable(candidate, OptimizationBudget(), cases))

    def test_noisy_timing_is_not_promotable(self) -> None:
        summary = PerformanceSummary(
            cases=(
                CaseEvaluation(
                    case_id="case",
                    verification=VerificationResult(True),
                    timing=TimingSamples((1.0, 10.0, 1.0)),
                ),
            ),
            aggregate_speedup=2.0,
        )
        self.assertFalse(is_promotable(summary, OptimizationBudget(max_cv=0.1)))

    def test_source_digest_stable(self) -> None:
        self.assertEqual(source_digest("VALUE = 1\n"), source_digest("VALUE = 1\n  \n"))


class HarnessTest(unittest.TestCase):
    def test_standalone_harness_evaluates_fake_kernel(self) -> None:
        harness = StandaloneHarness(
            Path(__file__).with_name("testdata") / "fake_harness.py"
        )
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

    def test_standalone_harness_reports_build_error(self) -> None:
        harness = StandaloneHarness(
            Path(__file__).with_name("testdata") / "fake_harness.py"
        )
        cases = (InputCase("a", {}),)
        target = KernelTarget("fake", "fake")
        with self.assertRaisesRegex(Exception, "missing fake kernel controls"):
            harness.evaluate(
                "VALUE = 1\n",
                cases,
                target,
                benchmark_repetitions=2,
            )

    def test_mock_provider_replays_canned_candidates(self) -> None:
        provider = MockLLMProvider(
            canned=(
                CandidateProposal("LATENCY_US = 80\nCORRECT = True\n", "first"),
                CandidateProposal("LATENCY_US = 60\nCORRECT = True\n", "second"),
            )
        )
        request = KernelOptimizationRequest(
            kernel_source="LATENCY_US = 100\nCORRECT = True\n",
            harness_path=Path(__file__).with_name("testdata") / "fake_harness.py",
            cases=(InputCase("a", {"scale": 1.0}),),
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


class KernelOptimizerTest(unittest.TestCase):
    def test_rejects_incorrect_candidate_and_promotes_faster_candidate(self) -> None:
        provider = FixedCandidateProvider(
            [
                CandidateProposal("LATENCY_US = 1\nCORRECT = False\n", "wrong"),
                CandidateProposal("LATENCY_US = 80\nCORRECT = True\n", "faster"),
                CandidateProposal("LATENCY_US = 90\nCORRECT = True\n", "slower"),
                CandidateProposal("LATENCY_US = 70\nCORRECT = True\n", "faster again"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = KernelOptimizer(provider).optimize(
                KernelOptimizationRequest(
                    kernel_source="LATENCY_US = 100\nCORRECT = True\n",
                    harness_path=Path(__file__).with_name("testdata")
                    / "fake_harness.py",
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
                )
            )
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

    def test_profile_payload_caps_large_values(self) -> None:
        # Use a harness that returns a >1MB profile to exercise spill logic.
        provider = FixedCandidateProvider(
            [CandidateProposal("LATENCY_US = 100\nCORRECT = True\n", "same")]
        )
        with tempfile.TemporaryDirectory() as tmp:
            harness_path = Path(tmp) / "big_profile_harness.py"
            harness_path.write_text(
                "def build(kernel_source, target):\n"
                "    return {'success': True, 'artifact': {}}\n"
                "def verify(artifact, case):\n"
                "    return {'passed': True}\n"
                "def benchmark(artifact, case, repetitions):\n"
                "    return {'samples_us': [10.0]*repetitions}\n"
                "def profile(artifact, case):\n"
                "    return {'big': 'x' * 2000000}\n"
            )
            with tempfile.TemporaryDirectory() as directory:
                result = KernelOptimizer(provider).optimize(
                    KernelOptimizationRequest(
                        kernel_source="LATENCY_US = 100\nCORRECT = True\n",
                        harness_path=harness_path,
                        cases=(InputCase("a", {}),),
                        target=KernelTarget("fake", "fake"),
                        budget=OptimizationBudget(
                            max_rounds=1,
                            candidates_per_round=1,
                            benchmark_repetitions=2,
                        ),
                        output_dir=Path(directory),
                    )
                )
                # Optimizer should complete without error even with oversized profile.
                self.assertIsNotNone(result)


def _performance(*values: tuple[str, float]) -> PerformanceSummary:
    return PerformanceSummary(
        cases=tuple(
            CaseEvaluation(
                case_id=case_id,
                verification=VerificationResult(True),
                timing=TimingSamples((latency, latency, latency)),
            )
            for case_id, latency in values
        )
    )


if __name__ == "__main__":
    unittest.main()
