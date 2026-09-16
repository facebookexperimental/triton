from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from unittest import mock
from unittest.mock import patch

from . import profiling
from . import rocm_profiler as rocm_profiler_module
from .amd_att import collect_fb_att
from .profiling import (
    ProfileRequest,
    affected_region_upper_bound,
    annotate_profile,
    compact_profile_summary,
    diagnostic_capabilities,
    export_ncu_report_details,
    extract_native_profiler_duration_us,
    extract_ncu_duration_us,
    is_profile_fresh,
    is_valid_intra_kernel_evidence,
    native_profiler_regression_diagnostic,
    ncu_regression_diagnostic,
    normalize_ncu_metrics,
    normalize_profile_request,
    parse_ncu_csv,
    parse_ncu_query_metrics,
    parse_proton_intra_kernel_trace,
    parse_proton_launch_attribution,
    per_case_profile_request,
    resolve_profile_request_for_target,
    resolve_profile_tools,
    select_ncu_metric_names,
)
from .profiling_policy import (
    adaptive_candidate_profile_decision,
    full_diagnostic_request,
    targeted_diagnostic_decision,
)
from .rocm_profiler import (
    collect_rocprofv3,
    filter_extreme_timing_outliers,
    find_rocprofv3,
    parse_rocprof_counter_collection,
    parse_rocprof_kernel_trace,
)


def _diagnostic_profile(
    *,
    valid: bool = True,
    validation_errors: object = (),
    passes: dict[str, object] | None = None,
    dominant_phases: list[dict[str, object]] | None = None,
    dominant_waits: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if passes is None:
        passes = {
            "role": {
                "valid": True,
                "expected_regions": ["consumer_loop"],
                "dropped_events": 0,
            },
            "wait": {
                "valid": True,
                "expected_regions": ["tma_wait", "barrier_wait"],
                "drop_warnings": {"dropped_events": 0},
                "validation": {"details": [{"dropped": 0}]},
            },
        }
    return {
        "diagnostic_proton_intra_kernel": {
            "valid": valid,
            "dominant_phases": dominant_phases or [],
            "dominant_waits": dominant_waits or [],
            "validation": {"errors": validation_errors, "passes": passes},
        }
    }


class ProfileRequestTest(unittest.TestCase):
    def test_bool_and_mapping_normalization(self) -> None:
        self.assertIsNone(normalize_profile_request(False))
        request = normalize_profile_request(True)
        self.assertIsInstance(request, ProfileRequest)
        assert request is not None
        self.assertEqual(request.level, "summary")

        with tempfile.TemporaryDirectory() as directory:
            mapped = normalize_profile_request(
                {
                    "level": "deep",
                    "tools": ["native_profiler"],
                    "experiment_id": "baseline",
                    "artifacts_dir": directory,
                    "reason": "baseline profile",
                }
            )
            assert mapped is not None
            self.assertEqual(mapped.level, "deep")
            self.assertEqual(mapped.tools, ("native_profiler",))
            self.assertEqual(mapped.artifacts_dir, Path(directory))

    def test_validates_request_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "summary.*deep"):
            ProfileRequest(level="diagnostic")
        with self.assertRaisesRegex(ValueError, "absolute"):
            ProfileRequest(artifacts_dir=Path("relative"))
        with self.assertRaisesRegex(TypeError, "mapping must be a mapping"):
            ProfileRequest.from_mapping({"instrumentation_mapping": []})
        with self.assertRaisesRegex(ValueError, "mapping requires proton_intra_kernel"):
            ProfileRequest(instrumentation_mapping={"scope": "task"})
        with self.assertRaisesRegex(ValueError, "digest requires a mapping"):
            ProfileRequest(instrumentation_mapping_digest="digest")
        with self.assertRaisesRegex(ValueError, "mapping digest mismatch"):
            ProfileRequest(
                tools=("proton_intra_kernel",),
                diagnostic_only=True,
                instrumentation_mapping={"scope": "task"},
                instrumentation_mapping_digest="wrong-digest",
            )
        with self.assertRaisesRegex(ValueError, "diagnostic_only"):
            ProfileRequest(tools=("proton_intra_kernel",))
        with self.assertRaisesRegex(ValueError, "warp"):
            ProfileRequest(
                tools=("proton_intra_kernel",),
                diagnostic_only=True,
                granularity="warp_group",
            )
        request = ProfileRequest(
            tools=("proton_intra_kernel",),
            diagnostic_only=True,
            granularity="warp",
        )
        self.assertEqual(request.granularity, "warp")

    def test_resolves_native_profiler_and_deduplicates_aliases(self) -> None:
        self.assertEqual(
            resolve_profile_tools(
                ("proton_launch", "native_profiler", "ncu"),
                native_profiler="ncu",
            ),
            ("proton_launch", "ncu"),
        )
        self.assertEqual(
            resolve_profile_tools((), native_profiler="ncu", default=("ncu",)),
            ("ncu",),
        )
        self.assertEqual(
            resolve_profile_tools(("native_profiler",)),
            ("native_profiler",),
        )

    def test_resolves_native_profiler_for_target_backend(self) -> None:
        payload = ProfileRequest(
            tools=("proton_launch", "native_profiler", "ncu"),
        ).to_json()
        cuda = resolve_profile_request_for_target(payload, {"backend": "cuda"})
        assert cuda is not None
        self.assertEqual(cuda["tools"], ["proton_launch", "ncu"])

        amd = resolve_profile_request_for_target(payload, {"backend": "hip"})
        assert amd is not None
        self.assertEqual(
            amd["tools"],
            ["rocprofv3", "ncu"],
        )

        hip = resolve_profile_request_for_target(payload, {"backend": "hip"})
        assert hip is not None
        self.assertEqual(hip["tools"], ["rocprofv3", "ncu"])

        deep = resolve_profile_request_for_target(
            ProfileRequest(
                level="deep",
                tools=("proton_launch", "native_profiler"),
            ).to_json(),
            {"backend": "hip"},
        )
        assert deep is not None
        self.assertEqual(deep["tools"], ["fb_att", "rocprofv3"])

    def test_per_case_profile_request_expands_absolute_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mapping = {"expected_kernel": "^kernel$"}
            mapping_digest = _instrumentation_mapping_digest(mapping)
            payload = ProfileRequest(
                tools=("proton_intra_kernel",),
                artifacts_dir=Path(directory),
                instrumentation_mapping=mapping,
                instrumentation_mapping_digest=mapping_digest,
                diagnostic_only=True,
            ).to_json()
            request = per_case_profile_request(payload, "shape/128?case")
            assert request is not None
            artifacts_dir = Path(str(request["artifacts_dir"]))
            self.assertTrue(artifacts_dir.is_absolute())
            self.assertTrue(artifacts_dir.exists())
            self.assertEqual(artifacts_dir.parent, Path(directory))
            self.assertNotIn("/", artifacts_dir.name)
            self.assertEqual(request["instrumentation_mapping"], mapping)
            self.assertEqual(
                request["instrumentation_mapping_digest"], mapping_digest
            )


class ProfileMetadataTest(unittest.TestCase):
    def test_profile_request_json_round_trip_preserves_diagnostic_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mapping = {"expected_kernel": "^kernel$"}
            mapping_digest = _instrumentation_mapping_digest(mapping)
            request = ProfileRequest(
                level="deep",
                tools=("proton_intra_kernel",),
                passes=("role", "wait"),
                experiment_id="exp-1",
                artifacts_dir=Path(directory),
                instrumentation_mapping=mapping,
                instrumentation_mapping_digest=mapping_digest,
                reason="collect diagnostic profile",
                source_digest="digest-123",
                policy_reason="targeted_diagnostic",
                diagnostic_only=True,
                granularity="warp",
            )

            payload = json.loads(json.dumps(request.to_json()))
            restored = ProfileRequest.from_mapping(payload)

        self.assertEqual(restored.level, "deep")
        self.assertEqual(restored.tools, ("proton_intra_kernel",))
        self.assertEqual(restored.passes, ("role", "wait"))
        self.assertEqual(restored.experiment_id, "exp-1")
        self.assertEqual(restored.reason, "collect diagnostic profile")
        self.assertEqual(restored.source_digest, "digest-123")
        self.assertEqual(restored.policy_reason, "targeted_diagnostic")
        self.assertTrue(restored.diagnostic_only)
        self.assertEqual(restored.granularity, "warp")
        self.assertEqual(restored.artifacts_dir, Path(directory))
        self.assertEqual(restored.instrumentation_mapping, mapping)
        self.assertEqual(
            restored.instrumentation_mapping_digest,
            mapping_digest,
        )

    def test_rejects_passes_without_intra_kernel_tool(self) -> None:
        with self.assertRaisesRegex(ValueError, "passes require proton_intra_kernel"):
            ProfileRequest(tools=("native_profiler",), passes=("role",))

    def test_annotates_profile_and_checks_freshness(self) -> None:
        request = ProfileRequest(
            tools=("proton_launch", "native_profiler"),
            experiment_id="exp-1",
            reason="candidate_profile",
            source_digest="digest-123",
            policy_reason="candidate",
        )

        annotated = annotate_profile({"summary": {"duration_us": 12.5}}, request, "case/a")

        self.assertEqual(annotated["summary"], {"duration_us": 12.5})
        self.assertEqual(
            annotated["profile_metadata"],
            {
                "source_digest": "digest-123",
                "case_id": "case/a",
                "experiment_id": "exp-1",
                "tools": ["proton_launch", "native_profiler"],
                "passes": [],
                "reason": "candidate_profile",
                "policy_reason": "candidate",
                "diagnostic_only": False,
                "granularity": None,
            },
        )
        self.assertTrue(is_profile_fresh(annotated, "digest-123"))
        self.assertFalse(is_profile_fresh(annotated, "digest-456"))
        self.assertTrue(is_profile_fresh({"summary": {}}, "digest-123"))
        self.assertFalse(
            is_profile_fresh({"summary": {}}, "digest-123", allow_legacy=False)
        )


class IntraKernelEvidenceTest(unittest.TestCase):
    def test_valid_intra_kernel_evidence_accepts_zero_drop_passes(self) -> None:
        self.assertTrue(is_valid_intra_kernel_evidence(_diagnostic_profile()))

    def test_valid_intra_kernel_evidence_rejects_invalid_errors_and_drops(
        self,
    ) -> None:
        cases = {
            "invalid_diagnostic": _diagnostic_profile(valid=False),
            "validation_error": _diagnostic_profile(validation_errors=("bad trace",)),
            "invalid_pass": _diagnostic_profile(
                passes={"wait": {"valid": False, "dropped_events": 0}}
            ),
            "record_drop": _diagnostic_profile(
                passes={"wait": {"valid": True, "dropped_events": 1}}
            ),
            "drop_warning_details": _diagnostic_profile(
                passes={
                    "wait": {
                        "valid": True,
                        "drop_warnings": {"details": [{"dropped": 1}]},
                    }
                }
            ),
            "nested_validation_drop": _diagnostic_profile(
                passes={
                    "wait": {
                        "valid": True,
                        "validation": {"details": [{"dropped": 1}]},
                    }
                }
            ),
        }
        for name, profile in cases.items():
            with self.subTest(name=name):
                self.assertFalse(is_valid_intra_kernel_evidence(profile))

    def test_diagnostic_capabilities_extracts_expected_regions(self) -> None:
        self.assertEqual(
            diagnostic_capabilities(_diagnostic_profile()),
            {
                "role": ("consumer_loop",),
                "wait": ("tma_wait", "barrier_wait"),
            },
        )

    def test_affected_region_upper_bound_uses_dominant_cycles(self) -> None:
        profile = _diagnostic_profile(
            dominant_phases=[
                {"name": "consumer_loop", "mean_cycles": 100.0},
                {"name": "tma_load", "mean_cycles": 30.0},
                {"name": "tma_wait", "mean_cycles": 40.0},
            ],
            dominant_waits=[
                {"name": "tma_wait", "mean_cycles": 45.0},
                {"name": "barrier_wait", "mean_cycles": 80.0},
            ],
        )

        self.assertEqual(
            affected_region_upper_bound(profile, ("tma_load", "tma_wait")), 0.75
        )
        self.assertEqual(
            affected_region_upper_bound(
                profile,
                ("tma_wait", "tma_wait", "barrier_wait"),
            ),
            1.0,
        )
        self.assertIsNone(affected_region_upper_bound(profile, "unknown_region"))
        self.assertIsNone(
            affected_region_upper_bound(
                _diagnostic_profile(
                    dominant_phases=[{"name": "tma_load", "mean_cycles": 30.0}]
                ),
                "tma_load",
            )
        )


class ProfilingPolicyTest(unittest.TestCase):
    def test_adaptive_candidate_profile_decision_skips_bad_candidates(self) -> None:
        root = Path("/tmp/kernel-profile-tests")

        unstable = adaptive_candidate_profile_decision(
            False,
            1.2,
            1.05,
            root,
            "exp-1",
            "digest-123",
        )
        self.assertIsNone(unstable.request)
        self.assertEqual(unstable.reason, "incorrect_or_unstable")

        loser = adaptive_candidate_profile_decision(
            True,
            1.039,
            1.05,
            root,
            "exp-1",
            "digest-123",
        )
        self.assertIsNone(loser.request)
        self.assertEqual(loser.reason, "clear_loser")

    def test_adaptive_candidate_profile_decision_requests_profile_by_window(
        self,
    ) -> None:
        root = Path("/tmp/kernel-profile-tests")

        near = adaptive_candidate_profile_decision(
            True,
            1.05,
            1.05,
            root,
            "exp-1",
            "digest-123",
        )
        assert near.request is not None
        self.assertEqual(near.reason, "near_threshold")
        self.assertEqual(near.request.level, "deep")
        self.assertEqual(near.request.tools, ("proton_launch", "native_profiler"))
        self.assertEqual(near.request.source_digest, "digest-123")
        self.assertEqual(near.request.policy_reason, "near_threshold")
        self.assertEqual(
            near.request.artifacts_dir,
            root / "experiments" / "exp-1" / "profile_artifacts",
        )

        candidate = adaptive_candidate_profile_decision(
            True,
            1.07,
            1.05,
            root,
            "exp-2",
            "digest-456",
        )
        assert candidate.request is not None
        self.assertEqual(candidate.reason, "candidate")
        self.assertEqual(candidate.request.level, "summary")
        self.assertEqual(candidate.request.source_digest, "digest-456")
        self.assertEqual(candidate.request.policy_reason, "candidate")

    def test_targeted_diagnostic_decision_rejects_invalid_requests(self) -> None:
        root = Path("/tmp/kernel-profile-tests")
        capabilities = {"wait": ("tma_wait",)}
        baseline = _diagnostic_profile(
            dominant_phases=[{"name": "consumer_loop", "mean_cycles": 100.0}],
            dominant_waits=[{"name": "tma_wait", "mean_cycles": 20.0}],
        )
        low_impact = _diagnostic_profile(
            dominant_phases=[{"name": "consumer_loop", "mean_cycles": 100.0}],
            dominant_waits=[{"name": "tma_wait", "mean_cycles": 5.0}],
        )

        cases = (
            (
                "disabled",
                False,
                "wait",
                ("tma_wait",),
                capabilities,
                baseline,
                2,
            ),
            (
                "unsupported_pass",
                True,
                "compute",
                ("mma",),
                capabilities,
                baseline,
                2,
            ),
            (
                "unsupported_regions",
                True,
                "wait",
                ("tma_wait", "barrier_wait"),
                capabilities,
                baseline,
                2,
            ),
            (
                "pass_budget_exhausted",
                True,
                "wait",
                ("tma_wait",),
                capabilities,
                baseline,
                1,
            ),
            (
                "low_upper_bound",
                True,
                "wait",
                ("tma_wait",),
                capabilities,
                low_impact,
                2,
            ),
        )
        for reason, enabled, hint_pass, regions, caps, profile, budget in cases:
            with self.subTest(reason=reason):
                decision = targeted_diagnostic_decision(
                    enabled,
                    hint_pass,
                    regions,
                    caps,
                    profile,
                    1.1,
                    budget,
                    root,
                    "exp-1",
                    "digest-123",
                )
                self.assertIsNone(decision.request)
                self.assertEqual(decision.reason, reason)

    def test_targeted_diagnostic_decision_requests_role_and_hint_pass(self) -> None:
        root = Path("/tmp/kernel-profile-tests")
        baseline = _diagnostic_profile(
            dominant_phases=[{"name": "consumer_loop", "mean_cycles": 100.0}],
            dominant_waits=[{"name": "tma_wait", "mean_cycles": 25.0}],
        )

        decision = targeted_diagnostic_decision(
            True,
            "wait",
            ("tma_wait",),
            {"wait": ("tma_wait", "barrier_wait")},
            baseline,
            1.1,
            2,
            root,
            "exp-1",
            "digest-123",
        )

        assert decision.request is not None
        self.assertEqual(decision.reason, "targeted_diagnostic")
        self.assertEqual(decision.request.level, "deep")
        self.assertEqual(decision.request.tools, ("proton_intra_kernel",))
        self.assertEqual(decision.request.passes, ("role", "wait"))
        self.assertEqual(decision.request.reason, "targeted_diagnostic")
        self.assertEqual(decision.request.source_digest, "digest-123")
        self.assertEqual(decision.request.policy_reason, "targeted_diagnostic")
        self.assertTrue(decision.request.diagnostic_only)
        self.assertEqual(decision.request.granularity, "warp")
        self.assertEqual(
            decision.request.artifacts_dir,
            root / "experiments" / "exp-1" / "diagnostic_profile_artifacts",
        )

    def test_full_diagnostic_request_enforces_budget(self) -> None:
        root = Path("/tmp/kernel-profile-tests")

        self.assertIsNone(
            full_diagnostic_request(
                root,
                "exp-1",
                "digest-123",
                reason="investigate",
                pass_budget=0,
            )
        )
        self.assertIsNone(
            full_diagnostic_request(
                root,
                "exp-1",
                "digest-123",
                reason="investigate",
                pass_budget=1,
                passes=("wait",),
            )
        )
        self.assertIsNone(
            full_diagnostic_request(
                root,
                "exp-1",
                "digest-123",
                reason="investigate",
                passes=(),
            )
        )

        targeted = full_diagnostic_request(
            root,
            "exp-1",
            "digest-123",
            reason="investigate",
            pass_budget=2,
            passes=("wait", "wait"),
        )
        assert targeted is not None
        self.assertEqual(targeted.passes, ("role", "wait"))
        self.assertEqual(targeted.tools, ("proton_intra_kernel",))
        self.assertTrue(targeted.diagnostic_only)

        full = full_diagnostic_request(
            root,
            "exp-1",
            "digest-123",
            reason="investigate",
            pass_budget=4,
        )
        assert full is not None
        self.assertEqual(full.passes, ("role", "coarse", "wait", "compute"))
        self.assertEqual(full.policy_reason, "full_diagnostic")


class ProfileParsingTest(unittest.TestCase):
    def _write_proton_trace(self, root: Path, events: list[object]) -> Path:
        path = root / "profile.chrome_trace"
        path.write_text(json.dumps({"traceEvents": events}))
        return path

    def _proton_mapping(self) -> dict[str, object]:
        return {
            "expected_kernel": r"^my_kernel_v\d+$",
            "tasks": {
                "compute": {"warps": ["2-3"], "scope": "compute_scope"},
                "load": {"warps": [0], "scope": "load_scope"},
                "mma": {"warps": [1], "scope": "mma_scope"},
                "store": {"warps": [[4, 5]], "scope": "store_scope"},
            },
            "required_scopes": ["wait_input", "wait_output"],
            "scope_kinds": {
                "compute_scope": "work",
                "load_scope": "work",
                "mma_scope": "work",
                "store_scope": "work",
                "wait_input": "wait",
                "wait_output": "wait",
            },
        }

    def _proton_event(
        self,
        *,
        name: str,
        cta: int,
        warp: int,
        ts: float,
        dur: float,
        kernel: str = "my_kernel_v1",
        phase: str = "X",
    ) -> dict[str, object]:
        return {
            "name": name,
            "cat": kernel,
            "ph": phase,
            "pid": f"{kernel} Core11 CTA{cta}",
            "tid": f"warp {warp} (line 0)",
            "ts": ts,
            "dur": dur,
            "args": {"raw_secret": "must-not-leak"},
        }

    def test_parse_proton_intra_kernel_selects_complete_cta_and_summarizes_tasks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = [
                self._proton_event(
                    name="load_scope", cta=1, warp=0, ts=0.0, dur=10.0
                ),
                self._proton_event(
                    name="mma_scope", cta=1, warp=1, ts=2.0, dur=8.0
                ),
                self._proton_event(
                    name="compute_scope", cta=1, warp=2, ts=4.0, dur=6.0
                ),
            ]
            complete = [
                self._proton_event(
                    name="load_scope", cta=2, warp=0, ts=0.0, dur=10.0
                ),
                self._proton_event(
                    name="mma_scope", cta=2, warp=1, ts=2.0, dur=8.0
                ),
                self._proton_event(
                    name="compute_scope", cta=2, warp=2, ts=4.0, dur=6.0
                ),
                self._proton_event(
                    name="compute_scope", cta=2, warp=3, ts=5.0, dur=4.0
                ),
                self._proton_event(
                    name="store_scope", cta=2, warp=4, ts=9.0, dur=3.0
                ),
                self._proton_event(
                    name="store_scope", cta=2, warp=5, ts=10.0, dur=1.0
                ),
                self._proton_event(
                    name="wait_input", cta=2, warp=0, ts=1.0, dur=2.0
                ),
                self._proton_event(
                    name="wait_input", cta=2, warp=0, ts=4.0, dur=1.0
                ),
                self._proton_event(
                    name="wait_output", cta=2, warp=4, ts=10.0, dur=1.5
                ),
                self._proton_event(
                    name="ignored_instant", cta=2, warp=0, ts=0.0, dur=99.0, phase="i"
                ),
                self._proton_event(
                    name="other_kernel_scope",
                    cta=0,
                    warp=0,
                    ts=0.0,
                    dur=99.0,
                    kernel="other_kernel",
                ),
            ]
            trace = self._write_proton_trace(root, incomplete + complete)

            summary = parse_proton_intra_kernel_trace(trace, self._proton_mapping())

        self.assertEqual(summary["schema"], "proton_intra_kernel_v1")
        self.assertTrue(summary["diagnostic_only"])
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["selected_kernel"], "my_kernel_v1")
        self.assertEqual(summary["selected_cta"], 2)
        self.assertEqual(
            summary["task_mappings"]["compute"],
            {"scope": "compute_scope", "warps": [2, 3]},
        )
        self.assertEqual(
            summary["task_spans"],
            {"compute": 6.0, "load": 10.0, "mma": 8.0, "store": 3.0},
        )
        self.assertEqual(
            summary["scope_stats"]["wait_input"],
            {"kind": "wait", "count": 2, "total_us": 3.0, "max_us": 2.0},
        )
        self.assertEqual(summary["dominant_waits"][0]["name"], "wait_input")
        load_mma = next(
            overlap
            for overlap in summary["task_overlaps"]
            if overlap["tasks"] == ["load", "mma"]
        )
        self.assertEqual(load_mma["overlap_us"], 8.0)
        self.assertEqual(load_mma["left_ratio"], 0.8)
        self.assertEqual(load_mma["right_ratio"], 1.0)
        serialized = json.dumps(summary)
        self.assertNotIn("traceEvents", serialized)
        self.assertNotIn("raw_secret", serialized)
        self.assertNotIn("must-not-leak", serialized)

    def test_parse_proton_intra_kernel_validates_paths_and_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute"):
            parse_proton_intra_kernel_trace("relative.trace", {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                parse_proton_intra_kernel_trace(root / "missing.trace", {})
            empty = root / "empty.trace"
            empty.write_text("")
            with self.assertRaisesRegex(ValueError, "empty"):
                parse_proton_intra_kernel_trace(empty, {})

            trace = self._write_proton_trace(root, [])
            missing = parse_proton_intra_kernel_trace(trace, {})
            malformed = parse_proton_intra_kernel_trace(
                trace,
                {
                    "expected_kernel": "[",
                    "tasks": {},
                    "required_scopes": [],
                    "wait_scopes": [],
                },
            )

        self.assertFalse(missing["valid"])
        self.assertTrue(any("expected_kernel" in item for item in missing["diagnostics"]))
        self.assertFalse(malformed["valid"])
        self.assertTrue(any("invalid mapping" in item for item in malformed["diagnostics"]))

    def test_parse_proton_intra_kernel_reports_ownership_and_required_coverage(
        self,
    ) -> None:
        mapping = self._proton_mapping()
        mapping["selected_cta"] = 7
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                self._proton_event(
                    name="load_scope", cta=7, warp=1, ts=0.0, dur=1.0
                ),
                self._proton_event(
                    name="mma_scope", cta=7, warp=1, ts=0.0, dur=1.0
                ),
                self._proton_event(
                    name="compute_scope", cta=7, warp=2, ts=0.0, dur=1.0
                ),
                self._proton_event(
                    name="store_scope", cta=7, warp=4, ts=0.0, dur=1.0
                ),
                self._proton_event(
                    name="wait_input", cta=7, warp=0, ts=0.0, dur=1.0
                ),
            ]
            trace = self._write_proton_trace(root, events)
            summary = parse_proton_intra_kernel_trace(trace, mapping)

        self.assertEqual(summary["selected_cta"], 7)
        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("load_scope" in item and "expected load" in item for item in summary["diagnostics"])
        )
        self.assertTrue(
            any("wait_output" in item for item in summary["diagnostics"])
        )

    def test_parse_proton_intra_kernel_enforces_output_bounds(self) -> None:
        mapping = self._proton_mapping()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self._write_proton_trace(
                root,
                [
                    self._proton_event(
                        name="load_scope", cta=2, warp=0, ts=index, dur=1.0
                    )
                    for index in range(3)
                ],
            )
            with mock.patch.object(profiling, "_PROTON_MAX_EVENTS", 2):
                summary = parse_proton_intra_kernel_trace(trace, mapping)

            oversized_mapping = self._proton_mapping()
            oversized_mapping["tasks"] = {
                f"task_{index}": {"warps": [index], "scope": f"scope_{index}"}
                for index in range(profiling._PROTON_MAX_TASKS + 1)
            }
            oversized = parse_proton_intra_kernel_trace(trace, oversized_mapping)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["task_spans"], {})
        self.assertTrue(any("limit is 2" in item for item in summary["diagnostics"]))
        self.assertFalse(oversized["valid"])
        self.assertTrue(any("tasks exceeds" in item for item in oversized["diagnostics"]))

    def test_parse_proton_tree_into_launch_attribution_summary(self) -> None:
        proton = {
            "name": "root",
            "children": [
                {"name": "python wrapper", "time_us": 5.0, "count": 2},
                {"name": "main kernel launch", "duration_ns": 7000, "count": 1},
                {"name": "helper kernel", "duration_us": 3.0, "count": 1},
                {
                    "name": "nested",
                    "children": [{"name": "dispatch wrapper", "time_ms": 0.002}],
                },
            ],
        }
        summary = parse_proton_launch_attribution(proton)
        self.assertEqual(summary["schema"], "launch_attribution_only")
        totals = summary["totals"]
        self.assertAlmostEqual(totals["wrapper_us"], 7.0)
        self.assertAlmostEqual(totals["main_kernel_us"], 7.0)
        self.assertAlmostEqual(totals["non_main_kernel_us"], 3.0)
        self.assertEqual(totals["count"], 5)

    def test_parse_proton_hatchet_tree_with_explicit_main_scope(self) -> None:
        proton = [
            {
                "children": [
                    {
                        "children": [],
                        "frame": {
                            "name": "_attn_bwd_preprocess",
                            "type": "function",
                        },
                        "metrics": {
                            "count": 2,
                            "device_id": "0",
                            "device_type": "CUDA",
                            "time (ns)": 160448,
                        },
                    },
                    {
                        "children": [],
                        "frame": {
                            "name": "_attn_bwd_mxf8_ws",
                            "type": "function",
                        },
                        "metrics": {
                            "count": 3,
                            "device_id": "0",
                            "device_type": "CUDA",
                            "time (ns)": 9322456,
                        },
                    },
                ],
                "frame": {"name": "ROOT", "type": "function"},
                "metrics": {"count": 0, "time (ns)": 0},
            },
            {"CUDA": {"0": {"arch": "100"}}},
        ]

        summary = parse_proton_launch_attribution(
            proton, main_scope="_attn_bwd_mxf8_ws"
        )

        self.assertEqual(
            summary["leaves"],
            [
                {
                    "name": "_attn_bwd_preprocess",
                    "time_us": 160.448,
                    "count": 2,
                },
                {"name": "_attn_bwd_mxf8_ws", "time_us": 9322.456, "count": 3},
            ],
        )
        totals = summary["totals"]
        self.assertAlmostEqual(totals["leaf_time_us"], 9482.904)
        self.assertAlmostEqual(totals["main_kernel_us"], 9322.456)
        self.assertAlmostEqual(totals["non_main_kernel_us"], 160.448)
        self.assertEqual(totals["wrapper_us"], 0.0)
        self.assertEqual(totals["count"], 5)

    def test_parse_ncu_csv_and_metric_selection(self) -> None:
        csv_text = (
            "ID,Metric Name,Metric Unit,Metric Value\n"
            '0,gpu__time_duration.sum,ns,"2,000"\n'
            "1,sm__throughput.avg.pct_of_peak_sustained_elapsed,%,75.5\n"
        )
        metrics = parse_ncu_csv(csv_text)
        self.assertEqual(
            metrics["gpu__time_duration.sum"], {"value": 2000.0, "unit": "ns"}
        )
        selected = select_ncu_metric_names(metrics.keys(), "summary")
        self.assertEqual(selected["metrics"]["duration_us"], "gpu__time_duration.sum")
        self.assertIsNone(selected["metrics"]["dram_throughput_pct"])
        self.assertTrue(selected["diagnostics"])

        normalized = normalize_ncu_metrics(metrics, "summary")
        self.assertAlmostEqual(normalized["summary"]["duration_us"], 2.0)
        self.assertAlmostEqual(normalized["summary"]["sm_throughput_pct"], 75.5)
        self.assertIsNone(normalized["summary"]["dram_throughput_pct"])

    def test_normalizes_local_memory_load_bytes(self) -> None:
        metric = "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum"
        selected = select_ncu_metric_names((metric,), "deep")
        self.assertEqual(selected["metrics"]["local_load_bytes"], metric)

        normalized = normalize_ncu_metrics(
            {metric: {"value": 4096.0, "unit": "byte"}},
            "deep",
        )
        self.assertEqual(normalized["registers"]["local_load_bytes"], 4096.0)

    def test_exports_and_normalizes_ncu_report_details(self) -> None:
        details_csv = (
            'ID,Metric Name,Metric Unit,Metric Value\n'
            '0,gpu__time_duration.sum,ms,17.18\n'
            '0,sm__throughput.avg.pct_of_peak_sustained_elapsed,%,39.83\n'
            '0,dram__bytes_read.sum,Gbyte,1.52\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "profile.ncu-rep"
            report.write_bytes(b"report")
            artifacts_dir = root / "export"
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=details_csv, stderr="warning\n"
            )
            with mock.patch("subprocess.run", return_value=completed) as run:
                result = export_ncu_report_details(
                    report,
                    artifacts_dir=artifacts_dir,
                    level="deep",
                    ncu_binary="/opt/ncu",
                )

            run.assert_called_once_with(
                [
                    "/opt/ncu",
                    "--import",
                    str(report),
                    "--page",
                    "details",
                    "--csv",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=120.0,
            )
            self.assertTrue(result["success"])
            self.assertEqual(result["ncu"]["summary"]["duration_us"], 17180.0)
            self.assertEqual(
                result["ncu"]["summary"]["sm_throughput_pct"], 39.83
            )
            self.assertEqual(
                result["ncu"]["memory"]["dram_read_bytes"],
                1_520_000_000.0,
            )
            artifacts = result["artifacts"]
            self.assertTrue(all(Path(path).is_absolute() for path in artifacts.values()))
            self.assertEqual(Path(artifacts["ncu_details_csv"]).read_text(), details_csv)
            self.assertEqual(
                Path(artifacts["ncu_import_stderr"]).read_text(), "warning\n"
            )

    def test_preserves_failed_ncu_report_import_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "profile.ncu-rep"
            report.write_bytes(b"report")
            completed = subprocess.CompletedProcess(
                args=[], returncode=2, stdout="status\n", stderr="bad metric\n"
            )
            with mock.patch("subprocess.run", return_value=completed):
                result = export_ncu_report_details(
                    report,
                    artifacts_dir=root / "export",
                )

            self.assertFalse(result["success"])
            self.assertTrue(
                any("exit code 2" in item for item in result["diagnostics"])
            )
            self.assertEqual(
                Path(result["artifacts"]["ncu_details_csv"]).read_text(),
                "status\n",
            )
            self.assertEqual(
                Path(result["artifacts"]["ncu_import_stderr"]).read_text(),
                "bad metric\n",
            )
            self.assertIsNone(result["ncu"]["summary"]["duration_us"])

    def test_handles_missing_and_timed_out_ncu_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.ncu-rep"
            with mock.patch("subprocess.run") as run:
                result = export_ncu_report_details(
                    missing,
                    artifacts_dir=root / "missing-export",
                )
            run.assert_not_called()
            self.assertFalse(result["success"])
            self.assertTrue(
                any("does not exist" in item for item in result["diagnostics"])
            )

            report = root / "profile.ncu-rep"
            report.write_bytes(b"report")
            timeout = subprocess.TimeoutExpired(
                cmd=["ncu"], timeout=0.1, output=b"partial", stderr=b"timeout"
            )
            with mock.patch("subprocess.run", side_effect=timeout):
                result = export_ncu_report_details(
                    report,
                    artifacts_dir=root / "timeout-export",
                    timeout_s=0.1,
                )
            self.assertFalse(result["success"])
            self.assertTrue(
                any("timed out" in item for item in result["diagnostics"])
            )
            self.assertEqual(
                Path(result["artifacts"]["ncu_details_csv"]).read_text(),
                "partial",
            )
            self.assertEqual(
                Path(result["artifacts"]["ncu_import_stderr"]).read_text(),
                "timeout",
            )

    def test_rejects_relative_ncu_export_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "report path must be absolute"):
            export_ncu_report_details(
                "profile.ncu-rep", artifacts_dir="/tmp/ncu-export"
            )
        with self.assertRaisesRegex(ValueError, "artifacts directory must be absolute"):
            export_ncu_report_details(
                "/tmp/profile.ncu-rep", artifacts_dir="relative"
            )

    def test_parse_ncu_query_metrics_accepts_csv_and_plain_text(self) -> None:
        csv_metrics = parse_ncu_query_metrics(
            "Metric Name,Description\n"
            "gpu__time_duration.sum,Duration\n"
            "sm__throughput.avg.pct_of_peak_sustained_elapsed,SM\n"
        )
        self.assertIn("gpu__time_duration.sum", csv_metrics)
        text_metrics = parse_ncu_query_metrics(
            "gpu__time_duration.sum\n"
            "sm__throughput.avg.pct_of_peak_sustained_elapsed some description\n"
        )
        self.assertIn("gpu__time_duration.sum", text_metrics)

    def test_parse_ncu_query_metrics_skips_command_preamble(self) -> None:
        metrics = parse_ncu_query_metrics(
            "$ /usr/local/cuda-12.8/bin/ncu --query-metrics --csv\n"
            "\n"
            '"Metric name","Metric type","Metric unit",'
            '"Metric description","NVIDIA B200"\n'
            '"gpu__time_duration","Counter","ns","Duration","1"\n'
            '"sm__throughput","Throughput","%","SM throughput","1"\n'
        )

        self.assertEqual(metrics, {"gpu__time_duration", "sm__throughput"})

    def test_selects_b200_base_and_scoped_metric_names(self) -> None:
        supported = {
            "gpu__time_duration",
            "sm__throughput",
            "TriageCompute.sm__throughput",
            "FBSP.TriageCompute.dram__throughput",
            "smsp__warp_issue_stalled_barrier_per_warp_active",
            "l1tex__t_bytes_pipe_lsu_mem_local_op_ld",
            "dram__bytes_read",
        }

        selected = select_ncu_metric_names(supported, "deep")

        self.assertEqual(
            selected["metrics"]["duration_us"], "gpu__time_duration.sum"
        )
        self.assertEqual(
            selected["metrics"]["sm_throughput_pct"],
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        )
        self.assertEqual(
            selected["metrics"]["dram_throughput_pct"],
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        )
        self.assertEqual(
            selected["metrics"]["barrier_pct"],
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        )
        self.assertEqual(
            selected["metrics"]["local_load_bytes"],
            "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
        )
        self.assertEqual(
            selected["metrics"]["dram_read_bytes"], "dram__bytes_read.sum"
        )
        self.assertIsNone(selected["metrics"]["registers_per_thread"])
        self.assertIsNone(selected["metrics"]["tensor_activity_pct"])

    def test_normalizes_b200_base_metrics(self) -> None:
        normalized = normalize_ncu_metrics(
            {
                "gpu__time_duration": {"value": 125000.0, "unit": "ns"},
                "sm__throughput": {"value": 75.0, "unit": "%"},
                "dram__throughput": {"value": 50.0, "unit": "%"},
                "dram__bytes_read": {"value": 1.52, "unit": "Gbyte"},
                "smsp__warp_issue_stalled_wait_per_warp_active": {
                    "value": 12.5,
                    "unit": "%",
                },
            },
            "deep",
        )

        self.assertEqual(normalized["summary"]["duration_us"], 125.0)
        self.assertEqual(normalized["summary"]["sm_throughput_pct"], 75.0)
        self.assertEqual(normalized["summary"]["dram_throughput_pct"], 50.0)
        self.assertEqual(
            normalized["memory"]["dram_read_bytes"], 1_520_000_000.0
        )
        self.assertEqual(normalized["stalls"]["async_wait_pct"], 12.5)
        self.assertIsNone(normalized["compute"]["tensor_activity_pct"])

    def test_does_not_match_ncu_sibling_derivatives(self) -> None:
        selected = select_ncu_metric_names(
            {
                "sm__throughput.avg.pct_of_peak_sustained_active",
                "dram__bytes_read.avg",
            },
            "deep",
        )
        self.assertIsNone(selected["metrics"]["sm_throughput_pct"])
        self.assertIsNone(selected["metrics"]["dram_read_bytes"])

        normalized = normalize_ncu_metrics(
            {
                "gpu__time_duration.max": {"value": 250000.0, "unit": "ns"},
                "gpu__time_duration.min": {"value": 125000.0, "unit": "ns"},
            },
            "summary",
        )
        self.assertIsNone(normalized["summary"]["duration_us"])
        self.assertTrue(
            any(
                "missing NCU metric for duration_us" in diagnostic
                for diagnostic in normalized["diagnostics"]
            )
        )

    def test_duration_extraction_and_regression_diagnostic(self) -> None:
        old_flat = {"ncu": {"gpu__time_duration.sum": {"value": 1000, "unit": "ns"}}}
        normalized = {"ncu": {"summary": {"duration_us": 1.02}}}
        self.assertAlmostEqual(extract_ncu_duration_us(old_flat), 1.0)
        self.assertAlmostEqual(extract_ncu_duration_us(normalized), 1.02)
        self.assertIn("regressed", ncu_regression_diagnostic(old_flat, normalized))
        self.assertEqual(ncu_regression_diagnostic(normalized, old_flat), "")

        rocprof_baseline = {"rocprofv3": {"summary": {"duration_us": 10.0}}}
        rocprof_candidate = {"rocprofv3": {"summary": {"duration_us": 10.2}}}
        self.assertEqual(
            extract_native_profiler_duration_us(rocprof_baseline),
            ("rocprofv3", 10.0),
        )
        self.assertIn(
            "rocprofv3 duration regressed",
            native_profiler_regression_diagnostic(rocprof_baseline, rocprof_candidate),
        )
        self.assertEqual(
            native_profiler_regression_diagnostic(old_flat, rocprof_candidate),
            "",
        )
        self.assertEqual(
            extract_native_profiler_duration_us(
                {"rocprofv3": {"summary": {"duration_us": "invalid"}}}
            ),
            (None, None),
        )

    def test_parses_rocprof_kernel_trace_and_resources(self) -> None:
        trace = (
            "Kernel_Name,Start_Timestamp,End_Timestamp,Workgroup_Size,"
            "LDS_Block_Size,VGPR_Count,Accum_VGPR_Count,SGPR_Count\n"
            "helper,0,1000,64,0,8,0,16\n"
            "gemm,1000,5000,256,32768,64,16,32\n"
            "gemm,5000,9200,256,32768,64,16,32\n"
            "gemm,9200,13000,256,32768,64,16,32\n"
        )
        profile = parse_rocprof_kernel_trace(trace, sample_count=2)
        self.assertEqual(profile["summary"]["dominant_kernel"], "gemm")
        self.assertEqual(profile["summary"]["duration_us"], 4.0)
        self.assertEqual(profile["summary"]["scope"], "dominant_kernel")
        self.assertEqual(profile["kernels"][0]["dispatches"], 3)
        self.assertEqual(profile["kernels"][0]["resources"]["lds_bytes"], 32768.0)

    def test_parses_rocprof_counter_collection(self) -> None:
        counters = (
            "Kernel_Name,Counter_Name,Counter_Value\n"
            "gemm,SQ_WAVES,100\n"
            "gemm,SQ_WAVES,120\n"
            "gemm,SQ_INSTS_MFMA,40\n"
            "helper,SQ_WAVES,2\n"
        )
        parsed = parse_rocprof_counter_collection(counters)
        self.assertEqual(parsed["gemm"]["SQ_WAVES"], 110.0)
        self.assertEqual(parsed["gemm"]["SQ_INSTS_MFMA"], 40.0)
        self.assertEqual(parsed["helper"]["SQ_WAVES"], 2.0)

    def test_filters_only_extreme_rocprof_timing_outliers(self) -> None:
        samples = [100.0, 101.0, 99.0, 102.0, 98.0, 1000.0]

        self.assertEqual(
            filter_extreme_timing_outliers(samples),
            [100.0, 101.0, 99.0, 102.0, 98.0],
        )
        self.assertEqual(
            filter_extreme_timing_outliers([100.0, 100.0, 100.0, 100.0]),
            [100.0, 100.0, 100.0, 100.0],
        )

    def test_finds_newest_internal_rocprof_by_numeric_version(self) -> None:
        candidates = (
            Path("/usr/local/fbcode/platform010/lib/rocm-6.9/bin/rocprofv3"),
            Path("/usr/local/fbcode/platform010/lib/rocm-6.10/bin/rocprofv3"),
        )

        with (
            patch.object(
                rocm_profiler_module.shutil,
                "which",
                return_value=None,
            ),
            patch.object(Path, "is_dir", autospec=True, return_value=True),
            patch.object(Path, "glob", autospec=True, return_value=iter(candidates)),
            patch.object(
                Path,
                "is_file",
                autospec=True,
                side_effect=lambda path: path in candidates,
            ),
            patch.object(
                rocm_profiler_module.os,
                "access",
                return_value=True,
            ),
        ):
            profiler = find_rocprofv3({"PATH": ""})

        self.assertEqual(profiler, candidates[1])

    def test_collects_rocprof_trace_and_counters_from_external_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiler = root / "rocprofv3"
            profiler.write_text(
                "#!/usr/bin/python3\n"
                "import os\n"
                "import sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                "assert os.environ['LD_PRELOAD'] == '/usr/lib64/libzstd.so.1'\n"
                "assert args[args.index('--mangled-kernels') + 1] == 'true'\n"
                "output_dir = Path(args[args.index('--output-directory') + 1])\n"
                "output_file = args[args.index('--output-file') + 1]\n"
                "output_dir.mkdir(parents=True, exist_ok=True)\n"
                "(Path.cwd() / '.rocprofv3').mkdir()\n"
                "if '--kernel-trace' in args:\n"
                "    (output_dir / f'{output_file}_kernel_trace.csv').write_text(\n"
                "        'Kernel_Name,Start_Timestamp,End_Timestamp,VGPR_Count\\n'\n"
                "        'gemm,0,4000,64\\n'\n"
                "        'gemm,4000,8200,64\\n'\n"
                "        'gemm,8200,12000,64\\n'\n"
                "    )\n"
                "else:\n"
                "    counters = args[args.index('--pmc') + 1].split(',')\n"
                "    rows = ''.join(f'gemm,{counter},100\\n' for counter in counters)\n"
                "    (output_dir / f'{output_file}_counter_collection.csv').write_text(\n"
                "        'Kernel_Name,Counter_Name,Counter_Value\\n' + rows\n"
                "    )\n"
                "print('fake rocprofv3')\n"
            )
            profiler.chmod(0o755)

            profile = collect_rocprofv3(
                ("/usr/bin/python3", "-c", "pass"),
                root / "artifacts",
                environment={
                    "TLX_ROCPROFV3": str(profiler),
                    "LD_PRELOAD": (
                        "/tmp/librocprofiler-sdk-tool.so:/usr/lib64/libzstd.so.1"
                    ),
                },
            )

            self.assertEqual(profile["tool"], "rocprofv3")
            self.assertEqual(profile["summary"]["dominant_kernel"], "gemm")
            self.assertEqual(profile["summary"]["duration_us"], 4.0)
            self.assertEqual(profile["counters"]["MfmaUtil"], 100.0)
            self.assertEqual(profile["kernels"][0]["samples_us"], [4.0, 4.2, 3.8])
            self.assertTrue(Path(profile["artifacts"]["kernel_trace_csv"]).exists())
            self.assertTrue(
                (
                    Path(profile["artifacts"]["kernel_trace_csv"]).parent / ".rocprofv3"
                ).is_dir()
            )

    def test_collect_rocprof_returns_error_for_unreadable_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiler = root / "rocprofv3"
            profiler.write_text(
                "#!/usr/bin/python3\n"
                "import sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                "output_dir = Path(args[args.index('--output-directory') + 1])\n"
                "output_file = args[args.index('--output-file') + 1]\n"
                "output_dir.mkdir(parents=True, exist_ok=True)\n"
                "(output_dir / f'{output_file}_kernel_trace.csv').write_bytes(b'\\xff')\n"
            )
            profiler.chmod(0o755)

            profile = collect_rocprofv3(
                ("/usr/bin/python3", "-c", "pass"),
                root / "artifacts",
                level="timing",
                environment={"TLX_ROCPROFV3": str(profiler)},
            )

            self.assertIn("failed to read rocprofv3 kernel trace", profile["error"])
            self.assertTrue(Path(profile["artifacts"]["command"]).exists())

    def test_gfx950_benchmark_surfaces_rocprof_error(self) -> None:
        harness_path = (
            Path(__file__).with_name("harnesses")
            / "gfx950"
            / "targets"
            / "gemm"
            / "harness.py"
        )
        spec = importlib.util.spec_from_file_location(
            "test_gfx950_harness", harness_path
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertIsNotNone(spec.loader)
        assert spec.loader is not None
        harness = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": ModuleType("torch")}):
            spec.loader.exec_module(harness)

        with (
            patch.object(
                harness,
                "collect_rocprofv3",
                return_value={
                    "error": "rocprofv3 was not found; set TLX_ROCPROFV3 or PATH"
                },
            ),
            self.assertRaisesRegex(RuntimeError, "set TLX_ROCPROFV3 or PATH"),
        ):
            harness.benchmark(
                (None, None, Path("/tmp/candidate.py"), "cuda"),
                {"case_id": "square"},
                repetitions=3,
            )

    def test_collects_fb_att_without_rocprof_separator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiler = root / "fb_att"
            profiler.write_text(
                "#!/usr/bin/python3\n"
                "import sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                "assert '--' not in args\n"
                "assert args[args.index('--att-perfcounter-ctrl') + 1] == '3'\n"
                "output_dir = Path(args[args.index('--fb-output-directory') + 1])\n"
                "ui_dir = output_dir / 'fbrocrof_1' / 'gemm_0_ui'\n"
                "ui_dir.mkdir(parents=True)\n"
                "(ui_dir / 'wstates0.json').write_text('{}')\n"
            )
            profiler.chmod(0o755)

            profile = collect_fb_att(
                ("/usr/bin/python3", "-c", "pass"),
                root / "artifacts",
                kernel_filter="gemm",
                counters=("SQ_LDS_BANK_CONFLICT",),
                environment={"TLX_FB_ATT": str(profiler)},
            )

            self.assertTrue(profile["valid"])
            self.assertEqual(profile["kernel_filter"], "gemm")
            self.assertEqual(profile["iteration_range"], "[4-4]")
            self.assertEqual(profile["perfcounter_control"], 3)
            self.assertEqual(len(profile["artifacts"]["ui_directories"]), 1)
            self.assertTrue(Path(profile["artifacts"]["wstates"][0]).exists())

    def test_compact_profile_summary_omits_raw_blobs(self) -> None:
        compact = compact_profile_summary(
            {
                "level": "summary",
                "summary": {"duration_us": 1.0},
                "raw": "x" * 5000,
                "ncu": {
                    "raw_metrics": {"huge": "blob"},
                    "summary": {"duration_us": 1.0},
                },
                "rocprofv3": {
                    "summary": {"duration_us": 2.0},
                    "raw_metrics": {"huge": "blob"},
                },
                "fb_att": {
                    "valid": True,
                    "artifacts": {"ui_directories": ["/tmp/gemm_0_ui"]},
                },
                "native_profiler": {
                    "raw": "x" * 5000,
                    "summary": {"duration_us": 1.0},
                },
                "diagnostic_proton_intra_kernel": {
                    "summary": {"active_warps": 8},
                    "trace_events": [{"raw": "event"}],
                    "artifacts": {"trace": "/tmp/proton.trace"},
                },
                "artifacts": {"csv": "/tmp/profile.csv"},
            }
        )
        self.assertEqual(compact["level"], "summary")
        self.assertNotIn("raw", compact)
        self.assertNotIn("raw_metrics", compact["ncu"])
        self.assertEqual(compact["rocprofv3"]["summary"]["duration_us"], 2.0)
        self.assertNotIn("raw_metrics", compact["rocprofv3"])
        self.assertTrue(compact["fb_att"]["valid"])
        self.assertIn("native_profiler", compact)
        self.assertNotIn("raw", compact["native_profiler"])
        diagnostic = compact["diagnostic_proton_intra_kernel"]
        self.assertNotIn("trace_events", diagnostic)
        self.assertEqual(diagnostic["artifacts"]["trace"], "/tmp/proton.trace")
        self.assertEqual(compact["artifacts"]["csv"], "/tmp/profile.csv")


def _instrumentation_mapping_digest(mapping: Mapping[str, object]) -> str:
    mapping_text = json.dumps(mapping, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(mapping_text.encode()).hexdigest()


if __name__ == "__main__":
    unittest.main()
