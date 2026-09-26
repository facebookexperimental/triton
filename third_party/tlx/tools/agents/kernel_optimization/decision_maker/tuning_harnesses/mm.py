from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import triton


def _repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.joinpath("third_party", "tlx", "ops").is_dir():
            return parent
    raise RuntimeError("could not locate the Triton repository")


_REPO_ROOT = _repository_root()
_AGENT_ROOT = _REPO_ROOT / "third_party/tlx/tools/agents/kernel_optimization"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from policy_source import frozen_source_digest, validate_branch_comments  # noqa: E402


def _target_setting(target: Mapping[str, Any], name: str, default: str = "") -> str:
    environment = target.get("environment", {})
    return str(environment.get(name, default)) if isinstance(environment, Mapping) else default


def build(kernel_source: str, target: dict[str, Any]) -> dict[str, Any]:
    editable = tuple(filter(None, _target_setting(target, "TLX_AGENT_EDITABLE_SYMBOLS").split(",")))
    expected_digest = _target_setting(target, "TLX_AGENT_FROZEN_SOURCE_DIGEST")
    if expected_digest and frozen_source_digest(kernel_source, editable) != expected_digest:
        return {
            "success": False,
            "diagnostics": f"candidate changed source outside the allowed symbols: {', '.join(editable)}",
        }
    heuristic_symbol = _target_setting(target, "TLX_AGENT_HEURISTIC_SYMBOL", "heuristic_config")
    if _target_setting(target, "TLX_AGENT_TUNING_PHASE") == "heuristic":
        baseline_sha = _target_setting(target, "TLX_AGENT_PHASE_BASELINE_SHA256")
        is_phase_baseline = hashlib.sha256(kernel_source.encode()).hexdigest() == baseline_sha
        if not is_phase_baseline:
            valid, diagnostic = validate_branch_comments(kernel_source, heuristic_symbol)
            if not valid:
                return {"success": False, "diagnostics": diagnostic}

    directory = tempfile.TemporaryDirectory(prefix="tlx-agent-mm-tuning-")
    source_path = Path(directory.name) / "candidate.py"
    source_path.write_text(kernel_source)
    arch = str(target["architecture"])
    try:
        module = _load_module(source_path, arch)
        op = _install_candidate(module, arch)
    except Exception as error:  # noqa: BLE001
        directory.cleanup()
        return {"success": False, "diagnostics": f"candidate import failed: {type(error).__name__}: {error}"}
    return {
        "success": True,
        "artifact": {
            "directory": directory,
            "module": module,
            "op": op,
            "device": str(target.get("device") or "cuda"),
            "phase": _target_setting(target, "TLX_AGENT_TUNING_PHASE", "search_space"),
            "stable_cv_max": float(target.get("evaluation_policy", {}).get("stable_cv_max", 0.03)),
            "records": {},
        },
    }


def verify(artifact: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    a, b = _inputs(case, artifact["device"])
    try:
        expected = torch.matmul(a, b)
        tolerance = 8e-3 if a.dtype == torch.bfloat16 else 1e-3
        spaces = ("heuristic", "full") if artifact["phase"] == "search_space" else ("heuristic", )
        for space in spaces:
            actual = artifact["op"](a, b, space=space)
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=tolerance)
    except Exception as error:  # noqa: BLE001
        return {"passed": False, "diagnostics": f"{type(error).__name__}: {error}"}
    return {"passed": True}


def benchmark(artifact: dict[str, Any], case: dict[str, Any], repetitions: int) -> dict[str, Any]:
    a, b = _inputs(case, artifact["device"])
    full = lambda: artifact["op"](a, b, space="full")  # noqa: E731
    full_tuners: dict[str, Any] = {}
    with _record_tuners(full_tuners):
        full()
    torch.cuda.synchronize()
    full_samples = _measure(full, repetitions)
    config_rows = _config_rows(full_tuners)
    full_median = statistics.median(full_samples)

    metrics: dict[str, Any] = {
        "full_config_count": sum(len(getattr(tuner, "configs", ())) for tuner in full_tuners.values()),
        "full_best_config": _best_configs(full_tuners),
        "full_median_us": full_median,
        "top_full_configs": config_rows[:8],
    }
    if artifact["phase"] == "search_space":
        samples = full_samples
    else:
        heuristic_tuners: dict[str, Any] = {}
        heuristic = lambda: artifact["op"](a, b, space="heuristic")  # noqa: E731
        with _record_tuners(heuristic_tuners):
            heuristic()
        torch.cuda.synchronize()
        heuristic_samples = _measure(heuristic, repetitions)
        heuristic_median = statistics.median(heuristic_samples)
        full_cv = _coefficient_of_variation(full_samples)
        heuristic_cv = _coefficient_of_variation(heuristic_samples)
        metrics.update({
            "full_space_parity":
            full_median / heuristic_median,
            "heuristic_config_count":
            max(
                1,
                sum(len(getattr(tuner, "configs", ())) for tuner in heuristic_tuners.values()),
            ),
            "heuristic_median_us":
            heuristic_median,
            "full_cv":
            full_cv,
            "heuristic_cv":
            heuristic_cv,
            "parity_stable":
            max(full_cv, heuristic_cv) <= artifact["stable_cv_max"],
        })
        samples = heuristic_samples

    artifact["records"][case["case_id"]] = {
        **metrics,
        "full_config_timings": config_rows,
    }
    return {
        "samples_us": samples,
        "warmup_count": 5,
        "cache_policy": f"{artifact['phase']}_triton_do_bench",
        "metrics": metrics,
    }


def profile(
    artifact: dict[str, Any],
    case: dict[str, Any],
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = dict(artifact["records"].get(case["case_id"], {}))
    config_timings = record.pop("full_config_timings", [])
    artifacts_dir = (request or {}).get("artifacts_dir")
    if artifacts_dir:
        path = Path(str(artifacts_dir)) / "full_config_timings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config_timings, indent=2, sort_keys=True) + "\n")
        record["full_config_timings_artifact"] = str(path)
    return record


def _measure(fn, repetitions: int) -> list[float]:
    rep_ms = max(20, int(repetitions))
    samples_ms = triton.testing.do_bench(fn, warmup=5, rep=rep_ms, return_mode="all")
    return [float(sample) * 1000.0 for sample in samples_ms]


def _coefficient_of_variation(samples: Sequence[float]) -> float:
    return statistics.stdev(samples) / statistics.fmean(samples) if len(samples) > 1 else 0.0


def _format_config(config) -> str:
    if config is None:
        return ""
    kwargs = " ".join(f"{key}={value}" for key, value in config.kwargs.items())
    return f"{kwargs} num_warps={config.num_warps} num_stages={config.num_stages}"


def _best_configs(tuners: Mapping[str, Any]) -> str:
    return " | ".join(f"{name}: {_format_config(tuner.best_config)}" for name, tuner in tuners.items()
                      if getattr(tuner, "best_config", None) is not None)


def _config_rows(tuners: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, tuner in tuners.items():
        for config, timings in getattr(tuner, "configs_timings", {}).items():
            median_ms = float(timings[0])
            if math.isfinite(median_ms):
                rows.append({
                    "kernel": name,
                    "config": _format_config(config),
                    "median_us": median_ms * 1000.0,
                })
    return sorted(rows, key=lambda row: row["median_us"])


@contextlib.contextmanager
def _record_tuners(into: dict[str, Any]):
    from triton.runtime.autotuner import Autotuner

    original = Autotuner.run

    def run(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        into[self.base_fn.__name__] = self
        return result

    Autotuner.run = run
    try:
        yield
    finally:
        Autotuner.run = original


def _load_module(path: Path, arch: str) -> ModuleType:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    name = f"triton.tlx.ops.kernels.mm._tlx_agent_{arch}_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_candidate(module: ModuleType, arch: str):
    import triton.tlx.ops as ops
    from triton.tlx.ops import _catalog

    candidate = getattr(module, "mm", None)
    if not callable(candidate):
        raise TypeError("candidate does not export mm()")
    spec = _catalog._BY_KEY.get(("mm", arch))
    if spec is None:
        raise RuntimeError(f"tlx.ops.mm has no catalog entry for {arch}")

    def call(*args, **kwargs):
        original = ops.impl_for

        def impl_for(op: str, *impl_args, **impl_kwargs):
            return (candidate, spec) if op == "mm" else original(op, *impl_args, **impl_kwargs)

        ops.impl_for = impl_for
        try:
            return ops.mm(*args, **kwargs)
        finally:
            ops.impl_for = original

    return call


def _inputs(case: Mapping[str, Any], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    parameters = case["parameters"]
    m, n, k = (int(parameters[name]) for name in ("m", "n", "k"))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[str(parameters["dtype"])]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(parameters.get("seed", 0)))
    a = _strided_random((m, k), tuple(parameters["a_strides"]), dtype, device, generator)
    b = _strided_random((k, n), tuple(parameters["b_strides"]), dtype, device, generator)
    return a, b


def _strided_random(shape, strides, dtype, device, generator):
    storage_size = 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
    storage = torch.randn(storage_size, device=device, dtype=dtype, generator=generator)
    return torch.as_strided(storage, shape, strides)
