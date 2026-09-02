from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import triton


def build(kernel_source: str, target: dict[str, Any]) -> dict[str, Any]:
    if target["backend"] != "cuda":
        return {"success": False, "diagnostics": "example harness requires CUDA"}
    directory = tempfile.TemporaryDirectory(prefix="tlx-agent-candidate-")
    source_path = Path(directory.name) / "candidate.py"
    source_path.write_text(kernel_source)
    try:
        module = _load_module(source_path)
    except Exception as error:
        directory.cleanup()
        return {
            "success": False,
            "diagnostics": f"candidate import failed: {type(error).__name__}: {error}",
        }
    if not callable(getattr(module, "matmul", None)):
        directory.cleanup()
        return {"success": False, "diagnostics": "candidate must define matmul(a, b)"}
    return {"success": True, "artifact": (directory, module)}


def verify(artifact: tuple[Any, ModuleType], case: dict[str, Any]) -> dict[str, Any]:
    _, module = artifact
    a, b = _inputs(case)
    try:
        actual = module.matmul(a, b)
        expected = torch.matmul(a, b)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    except Exception as error:
        return {
            "passed": False,
            "diagnostics": f"{type(error).__name__}: {error}",
        }
    return {"passed": True}


def benchmark(
    artifact: tuple[Any, ModuleType], case: dict[str, Any], repetitions: int
) -> dict[str, Any]:
    _, module = artifact
    a, b = _inputs(case)
    module.matmul(a, b)
    samples = [
        float(triton.testing.do_bench(lambda: module.matmul(a, b), warmup=25, rep=100))
        * 1000.0
        for _ in range(repetitions)
    ]
    return {
        "samples_us": samples,
        "warmup_count": 25,
        "cache_policy": "warm",
    }


def profile(artifact: tuple[Any, ModuleType], case: dict[str, Any]) -> dict[str, Any]:
    _, module = artifact
    a, b = _inputs(case)
    # Optional Triton Proton path: harness can set TRITON_PROTON=1 to emit a trace.
    # The agent itself never interprets the trace — it just persists whatever dict
    # the harness returns (capped at 1MB inline, spilled to artifacts/ otherwise).
    proton_trace: str | None = None
    if os.environ.get("TRITON_PROTON"):
        try:
            import triton.profiler as proton  # type: ignore[import-untyped]

            proton.start("tlx_matmul")  # type: ignore[attr-defined]
            module.matmul(a, b)
            proton_trace = "proton_trace_collected"
            try:
                proton.deactivate()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        except Exception as error:  # noqa: BLE001
            proton_trace = f"proton_error: {type(error).__name__}: {error}"

    latency_ms = float(
        triton.testing.do_bench(lambda: module.matmul(a, b), warmup=25, rep=100)
    )
    m, k = a.shape
    _, n = b.shape
    tflops = 2.0 * m * n * k / (latency_ms * 1e-3) / 1e12
    result: dict[str, Any] = {"latency_ms": latency_ms, "tflops": tflops}
    if proton_trace is not None:
        result["proton_trace"] = proton_trace
    return result


def _load_module(path: Path) -> ModuleType:
    module_name = f"tlx_kernel_agent_candidate_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _inputs(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    parameters = case["parameters"]
    device = parameters.get("device", "cuda")
    dtype = getattr(torch, parameters.get("dtype", "float16"))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(parameters.get("seed", 0)))
    a = torch.randn(
        (int(parameters["m"]), int(parameters["k"])),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    b = torch.randn(
        (int(parameters["k"]), int(parameters["n"])),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return a, b
