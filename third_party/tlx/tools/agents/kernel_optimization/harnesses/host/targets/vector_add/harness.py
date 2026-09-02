# Minimal standalone harness for smoke tests without a real GPU.
# Candidate must export `vector_add(a, b)` operating on torch.Tensor.

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import triton


def build(kernel_source: str, target: dict[str, Any]) -> dict[str, Any]:
    del target
    directory = tempfile.TemporaryDirectory(prefix="tlx-agent-vector-add-")
    source_path = Path(directory.name) / "candidate.py"
    source_path.write_text(kernel_source)
    try:
        module = _load_module(source_path)
    except Exception as error:  # noqa: BLE001
        directory.cleanup()
        return {
            "success": False,
            "diagnostics": f"candidate import failed: {type(error).__name__}: {error}",
        }
    if not callable(getattr(module, "vector_add", None)):
        directory.cleanup()
        return {"success": False, "diagnostics": "candidate must define vector_add(a, b)"}
    return {"success": True, "artifact": (directory, module)}


def verify(artifact: tuple[Any, ModuleType], case: dict[str, Any]) -> dict[str, Any]:
    _, module = artifact
    a, b = _inputs(case)
    try:
        actual = module.vector_add(a, b)
        expected = a + b
        torch.testing.assert_close(actual, expected)
    except Exception as error:  # noqa: BLE001
        return {"passed": False, "diagnostics": f"{type(error).__name__}: {error}"}
    return {"passed": True}


def _try_cuda_benchmark(
    module: ModuleType, a: torch.Tensor, b: torch.Tensor, repetitions: int
) -> dict[str, Any] | None:
    """Try a real CUDA benchmark; return None if unavailable or fails."""
    if not torch.cuda.is_available():
        return None
    try:
        # Probe CUDA availability (busy GPU throws on get_empty_cache_for_benchmark).
        module.vector_add(a, b)
        samples = [
            float(triton.testing.do_bench(lambda: module.vector_add(a, b), warmup=25, rep=100)) * 1000.0
            for _ in range(repetitions)
        ]
        return {"samples_us": samples, "warmup_count": 25, "cache_policy": "warm"}
    except Exception:  # noqa: BLE001
        return None


def benchmark(
    artifact: tuple[Any, ModuleType], case: dict[str, Any], repetitions: int
) -> dict[str, Any]:
    _, module = artifact
    a, b = _inputs(case)
    cuda_result = _try_cuda_benchmark(module, a, b, repetitions)
    if cuda_result is not None:
        return cuda_result
    # CPU / synthetic fallback — deterministic so tests pass without a GPU.
    base_latency = float(getattr(module, "LATENCY_US", 100.0))
    scale = float(case["parameters"].get("scale", 1.0))
    samples = [base_latency * scale] * repetitions
    return {"samples_us": samples, "warmup_count": 0, "cache_policy": "synthetic-cpu"}


def profile(artifact: tuple[Any, ModuleType], case: dict[str, Any]) -> dict[str, Any]:
    _, module = artifact
    a, b = _inputs(case)
    cuda_result = _try_cuda_benchmark(module, a, b, 1)
    if cuda_result is not None:
        latency_ms = cuda_result["samples_us"][0] / 1000.0
    else:
        latency_ms = float(getattr(module, "LATENCY_US", 100.0)) / 1000.0
    n = int(case["parameters"].get("n", a.numel()))
    return {"latency_ms": latency_ms, "n": n, "device": str(a.device)}


def _load_module(path: Path) -> ModuleType:
    module_name = f"tlx_kernel_agent_vector_add_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _inputs(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    parameters = case["parameters"]
    n = int(parameters.get("n", 1024))
    dtype = getattr(torch, parameters.get("dtype", "float32"))
    # Always use CPU tensors for the synthetic path; move to CUDA if available and requested.
    device_str = str(parameters.get("device", "cpu"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    seed = int(parameters.get("seed", 0))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    a = torch.randn((n,), dtype=dtype, generator=generator).to(device)
    b = torch.randn((n,), dtype=dtype, generator=generator).to(device)
    return a, b
