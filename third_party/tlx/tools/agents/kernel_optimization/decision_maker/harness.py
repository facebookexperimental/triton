from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Queue
from types import ModuleType
from typing import Any, Mapping, Protocol, runtime_checkable

from ..contracts import (
    CaseEvaluation,
    InputCase,
    JsonValue,
    KernelTarget,
    PerformanceSummary,
    TimingSamples,
    VerificationResult,
    to_json_value,
)
from .profiling import (
    ProfileRequest,
    compact_profile_output,
    invoke_profile,
    per_case_profile_request,
    profile_request_to_json,
    resolve_profile_request_for_target,
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
        self,
        build_artifact: object,
        case: Mapping[str, JsonValue],
        request: Mapping[str, JsonValue] | None = None,
    ) -> Mapping[str, JsonValue]: ...


@runtime_checkable
class ExperimentHarness(KernelHarness, Protocol):
    """Harness extension for explicitly supported non-promotable experiments."""

    def build_experiment(
        self,
        kernel_source: str,
        target: Mapping[str, JsonValue],
        experiment: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue] | object: ...


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
        profile: bool | ProfileRequest | Mapping[str, Any] = False,
        experiment: Mapping[str, JsonValue] | None = None,
    ) -> PerformanceSummary:
        raw_pool = target.environment.get("TLX_AGENT_DEVICE_POOL")
        if raw_pool and cases:
            try:
                device_pool = json.loads(raw_pool)
            except json.JSONDecodeError as error:
                raise HarnessExecutionError("TLX_AGENT_DEVICE_POOL is not valid JSON") from error
            if not isinstance(device_pool, list) or not device_pool:
                raise HarnessExecutionError("TLX_AGENT_DEVICE_POOL must be a non-empty list")
            return self._evaluate_isolated(
                kernel_source,
                cases,
                target,
                benchmark_repetitions,
                profile,
                experiment,
                tuple(device_pool),
            )
        return self._evaluate_subprocess(
            kernel_source,
            cases,
            target,
            benchmark_repetitions,
            profile,
            experiment,
        )

    def _evaluate_isolated(
        self,
        kernel_source: str,
        cases: tuple[InputCase, ...],
        target: KernelTarget,
        benchmark_repetitions: int,
        profile: bool | ProfileRequest | Mapping[str, Any],
        experiment: Mapping[str, JsonValue] | None,
        device_pool: tuple[Mapping[str, str], ...],
    ) -> PerformanceSummary:
        checkpoint_root_raw = target.environment.get("TLX_AGENT_CHECKPOINT_DIR")
        checkpoint_root = Path(checkpoint_root_raw) if checkpoint_root_raw else None
        if checkpoint_root is not None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
        common_environment = {
            key: value
            for key, value in target.environment.items()
            if key not in {"TLX_AGENT_DEVICE_POOL", "TLX_AGENT_CHECKPOINT_DIR"}
        }
        source_digest = hashlib.sha256(kernel_source.encode()).hexdigest()

        def checkpoint_path(case: InputCase) -> Path | None:
            if checkpoint_root is None:
                return None
            digest = hashlib.sha256(case.case_id.encode()).hexdigest()[:16]
            return checkpoint_root / f"{digest}.json"

        results: dict[str, CaseEvaluation] = {}
        pending: list[InputCase] = []
        for case in cases:
            path = checkpoint_path(case)
            if path is not None and path.is_file():
                try:
                    saved = json.loads(path.read_text())
                    if saved.get("source_digest") == source_digest and saved.get("case_id") == case.case_id:
                        results[case.case_id] = _case_evaluation_from_json(saved["evaluation"])
                        continue
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass
            pending.append(case)

        work_queue: Queue[InputCase] = Queue()
        for case in pending:
            work_queue.put(case)

        def run_worker(worker_index: int) -> list[tuple[str, CaseEvaluation]]:
            completed: list[tuple[str, CaseEvaluation]] = []
            device_environment = device_pool[worker_index]
            while True:
                try:
                    case = work_queue.get_nowait()
                except Empty:
                    break
                case_target = replace(
                    target,
                    environment={**common_environment, **device_environment},
                )
                summary = self._evaluate_subprocess(
                    kernel_source,
                    (case,),
                    case_target,
                    benchmark_repetitions,
                    profile,
                    experiment,
                )
                evaluation = summary.cases[0]
                path = checkpoint_path(case)
                if path is not None:
                    payload = {
                        "case_id": case.case_id,
                        "source_digest": source_digest,
                        "evaluation": to_json_value(evaluation),
                    }
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                    temporary.replace(path)
                completed.append((case.case_id, evaluation))
                print(
                    f"[tlx-agent] gpu{device_environment.get('TLX_AGENT_PHYSICAL_GPU', '?')}: "
                    f"completed {case.case_id}",
                    file=sys.stderr,
                    flush=True,
                )
                work_queue.task_done()
            return completed

        if pending:
            workers = min(len(device_pool), len(pending))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(run_worker, index) for index in range(workers)]
                for future in as_completed(futures):
                    for case_id, evaluation in future.result():
                        results[case_id] = evaluation
        return PerformanceSummary(cases=tuple(results[case.case_id] for case in cases))

    def _evaluate_subprocess(
        self,
        kernel_source: str,
        cases: tuple[InputCase, ...],
        target: KernelTarget,
        benchmark_repetitions: int,
        profile: bool | ProfileRequest | Mapping[str, Any] = False,
        experiment: Mapping[str, JsonValue] | None = None,
    ) -> PerformanceSummary:
        request = {
            "kernel_source": kernel_source,
            "cases": to_json_value(cases),
            "target": to_json_value(target),
            "benchmark_repetitions": benchmark_repetitions,
            "profile": resolve_profile_request_for_target(
                profile_request_to_json(profile),
                to_json_value(target),  # type: ignore[arg-type]
            ),
            "experiment": dict(experiment) if experiment is not None else None,
        }
        worker_path = Path(__file__).with_name("runner.py")
        environment = os.environ.copy()
        environment.update(target.environment)
        cpu_list = environment.pop("TLX_AGENT_NUMA_CPUS", "")

        def set_affinity() -> None:
            if cpu_list and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {int(cpu) for cpu in cpu_list.split(",")})
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
                preexec_fn=set_affinity if cpu_list else None,
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
        profile: bool | ProfileRequest | Mapping[str, Any] = False,
        experiment: Mapping[str, JsonValue] | None = None,
    ) -> PerformanceSummary:
        harness = _load_harness(self.harness_path)
        if not hasattr(harness, "build") or not hasattr(harness, "verify") or not hasattr(harness, "benchmark"):
            raise HarnessExecutionError(
                "harness must define build(), verify(), and benchmark()"
            )
        target_dict: Mapping[str, JsonValue] = to_json_value(target)  # type: ignore[assignment]
        # Reuse worker normalization helpers for consistency.
        from .runner import _normalize_build, _normalize_timing, _normalize_verification

        profile_payload_root = resolve_profile_request_for_target(
            profile_request_to_json(profile),
            target_dict,
        )
        if experiment is None:
            build_result = harness.build(kernel_source, dict(target_dict))  # type: ignore[arg-type]
        else:
            build_experiment = getattr(harness, "build_experiment", None)
            if build_experiment is None:
                raise HarnessExecutionError(
                    "non-promotable experiments require harness.build_experiment()"
                )
            build_result = build_experiment(
                kernel_source,
                dict(target_dict),
                dict(experiment),
            )
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
                if isinstance(benchmark_raw, Mapping):
                    verification = VerificationResult(
                        passed=verification.passed,
                        diagnostics=verification.diagnostics,
                        metrics={
                            **verification.metrics,
                            **dict(benchmark_raw.get("metrics", {})),
                        },
                    )
                timing = TimingSamples(
                    samples_us=tuple(float(s) for s in timing_norm["samples_us"]),
                    warmup_count=int(timing_norm.get("warmup_count", 0)),
                    cache_policy=str(timing_norm.get("cache_policy", "unspecified")),
                )
                if profile_payload_root and hasattr(harness, "profile"):
                    try:
                        profile_request = per_case_profile_request(
                            profile_payload_root, case.case_id
                        )
                        raw_profile = invoke_profile(
                            harness.profile, artifact, dict(case_dict), profile_request
                        )
                        profile_payload = compact_profile_output(raw_profile, profile_request)
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
