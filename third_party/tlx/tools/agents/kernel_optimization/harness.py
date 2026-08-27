from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Protocol, runtime_checkable

from .models import (
    CaseEvaluation,
    InputCase,
    JsonValue,
    KernelTarget,
    PerformanceSummary,
    TimingSamples,
    VerificationResult,
    to_json_value,
)


class HarnessExecutionError(RuntimeError):
    pass


class BuildError(HarnessExecutionError):
    pass


class HarnessTimeoutError(HarnessExecutionError):
    pass


@runtime_checkable
class KernelHarness(Protocol):
    """Deterministic harness that owns build / verify / benchmark / profile.

    The optimizer never decides correctness or performance — only the harness does.
    `profile` is optional and may be omitted by harness implementations.
    """

    def build(self, kernel_source: str, target: Mapping[str, JsonValue]) -> Mapping[str, JsonValue] | object: ...

    def verify(
        self, build_artifact: object, case: Mapping[str, JsonValue]
    ) -> bool | Mapping[str, JsonValue]: ...

    def benchmark(
        self, build_artifact: object, case: Mapping[str, JsonValue], repetitions: int
    ) -> list[float] | Mapping[str, JsonValue]: ...

    def profile(
        self, build_artifact: object, case: Mapping[str, JsonValue]
    ) -> Mapping[str, JsonValue]: ...


# ---------------------------------------------------------------------------
# Subprocess isolation (default)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubprocessHarness:
    harness_path: Path
    timeout_seconds: float

    def evaluate(
        self,
        kernel_source: str,
        cases: tuple[InputCase, ...],
        target: KernelTarget,
        benchmark_repetitions: int,
        profile: bool = False,
    ) -> PerformanceSummary:
        request = {
            "kernel_source": kernel_source,
            "cases": to_json_value(cases),
            "target": to_json_value(target),
            "benchmark_repetitions": benchmark_repetitions,
            "profile": profile,
        }
        worker_path = Path(__file__).with_name("worker.py")
        environment = os.environ.copy()
        environment.update(target.environment)
        response_file = tempfile.NamedTemporaryFile(
            prefix="tlx-kernel-agent-response-", suffix=".json", delete=False
        )
        response_path = Path(response_file.name)
        response_file.close()
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(worker_path),
                    "--harness",
                    str(self.harness_path.resolve()),
                    "--response",
                    str(response_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(
                    json.dumps(request), timeout=self.timeout_seconds
                )
            except subprocess.TimeoutExpired as error:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                response_path.unlink(missing_ok=True)
                raise HarnessTimeoutError(
                    f"harness timed out after {self.timeout_seconds:.1f}s"
                ) from error
            completed = subprocess.CompletedProcess(
                process.args, process.returncode, stdout, stderr
            )
        except OSError as error:
            response_path.unlink(missing_ok=True)
            raise HarnessExecutionError(f"could not start harness: {error}") from error
        if completed.returncode != 0:
            response_path.unlink(missing_ok=True)
            diagnostics = completed.stderr.strip() or completed.stdout.strip()
            raise HarnessExecutionError(
                f"harness exited with code {completed.returncode}: {diagnostics}"
            )
        try:
            payload = json.loads(response_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise HarnessExecutionError(
                "harness did not write a valid response; "
                f"stdout={completed.stdout[:500]!r}, stderr={completed.stderr[:500]!r}"
            ) from error
        finally:
            response_path.unlink(missing_ok=True)
        if not payload.get("build", {}).get("success", False):
            diagnostics = payload.get("build", {}).get("diagnostics", "build failed")
            raise BuildError(str(diagnostics))
        return PerformanceSummary(
            cases=tuple(_case_evaluation_from_json(case) for case in payload["cases"])
        )


# ---------------------------------------------------------------------------
# In-process harness (debug / unit-test path, no subprocess isolation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandaloneHarness:
    harness_path: Path
    timeout_seconds: float = 600.0

    def evaluate(
        self,
        kernel_source: str,
        cases: tuple[InputCase, ...],
        target: KernelTarget,
        benchmark_repetitions: int,
        profile: bool = False,
    ) -> PerformanceSummary:
        harness = _load_harness(self.harness_path)
        if not hasattr(harness, "build") or not hasattr(harness, "verify") or not hasattr(harness, "benchmark"):
            raise HarnessExecutionError(
                "harness must define build(), verify(), and benchmark()"
            )
        target_dict: Mapping[str, JsonValue] = to_json_value(target)  # type: ignore[assignment]
        # Reuse worker normalization helpers for consistency.
        from .worker import _normalize_build, _normalize_timing, _normalize_verification

        build_result = harness.build(kernel_source, dict(target_dict))  # type: ignore[arg-type]
        success, artifact, diagnostics = _normalize_build(build_result)
        if not success:
            raise BuildError(str(diagnostics))
        evaluations: list[CaseEvaluation] = []
        for case in cases:
            case_dict: Mapping[str, JsonValue] = to_json_value(case)  # type: ignore[assignment]
            # Harness receives the same dict shape as via the subprocess path.
            verification_raw = harness.verify(artifact, dict(case_dict))  # type: ignore[arg-type]
            verification_norm = _normalize_verification(verification_raw)
            verification = VerificationResult(
                passed=bool(verification_norm["passed"]),
                diagnostics=str(verification_norm.get("diagnostics", "")),
                metrics=dict(verification_norm.get("metrics", {})),
            )
            timing: TimingSamples | None = None
            profile_payload: Mapping[str, JsonValue] = {}
            if verification.passed:
                benchmark_raw = harness.benchmark(artifact, dict(case_dict), benchmark_repetitions)  # type: ignore[arg-type]
                timing_norm = _normalize_timing(benchmark_raw)
                timing = TimingSamples(
                    samples_us=tuple(float(s) for s in timing_norm["samples_us"]),
                    warmup_count=int(timing_norm.get("warmup_count", 0)),
                    cache_policy=str(timing_norm.get("cache_policy", "unspecified")),
                )
                if profile and hasattr(harness, "profile"):
                    try:
                        raw_profile = harness.profile(artifact, dict(case_dict))  # type: ignore[arg-type]
                        if not isinstance(raw_profile, Mapping):
                            raise TypeError("profile() must return a mapping")
                        # Cap inline size to keep PerformanceSummary bounded.
                        serialized = json.dumps(dict(raw_profile))
                        if len(serialized) > 1_000_000:
                            profile_payload = {
                                "error": "profile payload exceeded inline limit (1MB)",
                                "size_bytes": len(serialized),
                                "truncated_keys": list(raw_profile.keys()),
                            }
                        else:
                            profile_payload = dict(raw_profile)
                    except Exception as error:  # noqa: BLE001
                        profile_payload = {"error": f"{type(error).__name__}: {error}"}
            evaluations.append(
                CaseEvaluation(
                    case_id=case.case_id,
                    verification=verification,
                    timing=timing,
                    profile=profile_payload,
                )
            )
        return PerformanceSummary(cases=tuple(evaluations))


def _load_harness(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("tlx_kernel_agent_user_harness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load harness from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case_evaluation_from_json(payload: Mapping[str, JsonValue]) -> CaseEvaluation:
    verification_payload = _require_mapping(payload, "verification")
    timing_payload = payload.get("timing")
    timing = None
    if isinstance(timing_payload, Mapping):
        samples = timing_payload.get("samples_us")
        if not isinstance(samples, list):
            raise HarnessExecutionError("timing.samples_us must be a list")
        timing = TimingSamples(
            samples_us=tuple(float(sample) for sample in samples),
            warmup_count=int(timing_payload.get("warmup_count", 0)),
            cache_policy=str(timing_payload.get("cache_policy", "unspecified")),
        )
    profile_payload = payload.get("profile", {})
    if not isinstance(profile_payload, Mapping):
        raise HarnessExecutionError("profile must be a JSON object")
    metrics = verification_payload.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise HarnessExecutionError("verification.metrics must be a JSON object")
    return CaseEvaluation(
        case_id=str(payload["case_id"]),
        verification=VerificationResult(
            passed=bool(verification_payload.get("passed", False)),
            diagnostics=str(verification_payload.get("diagnostics", "")),
            metrics=dict(metrics),
        ),
        timing=timing,
        profile=dict(profile_payload),
    )


def _require_mapping(
    payload: Mapping[str, JsonValue], key: str
) -> Mapping[str, JsonValue]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise HarnessExecutionError(f"{key} must be a JSON object")
    return value
