"""Optimization harness for the TLX MI300X (gfx942 / CDNA3) GEMM tutorial.

Differs from the blackwell/hopper GEMM harnesses in three ways that are all
forced by the part rather than by taste:

**Timing method.** `do_bench` is not used. Its burst-and-sleep pattern was
measured at 14-20% burst-to-burst variation on this part, which would swamp the
1.01 default promotion threshold and make promotion a coin flip. This harness
reuses `gfx942_perf_harness.measure_samples` -- warm continuously until clocks
settle, then one timed window -- so the samples the optimizer computes its
median and CV over are the real per-call distribution.

**Pinned gate, autotuned report.** The kernel searches 42 configs per shape. The
gate runs pinned to the tile the baseline autotuner chose, so a candidate is
judged on its structural change and not on which tile it happened to win with;
`profile()` separately runs the search unpinned and reports that number, which
is what a user would actually get.

**ATT profiling.** `profile()` shells rocprofv3 rather than profiling in
process, because ATT requires the application to be launched underneath it. See
`../../att.py`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

# Set before the first kernel launch, matching the standalone perf script. The
# AMD backend never emits `asm["launch_metadata"]`, so the C dispatcher can
# never be built and every launch would warn; post-misched would re-order a
# hand-scheduled hot loop.
os.environ.setdefault("TRITON_DISABLE_POST_MISCHED", "1")
os.environ.setdefault("TRITON_USE_C_DISPATCHER", "0")

# `harnesses/gfx942/`, which holds the shared att/pinning modules. The harness
# is loaded by file path with no package context, so this is how they resolve.
_ARCH_DIR = Path(__file__).resolve().parents[2]
if str(_ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ARCH_DIR))

import att  # noqa: E402
import pinning  # noqa: E402

from triton.language.extra.tlx.tutorials.testing.gfx942_perf_harness import (  # noqa: E402
    measure_samples, )

ENTRY_POINT = "matmul"

# Enough samples for a meaningful median and CV without bloating the response
# JSON; a 2 s window at these shapes produces thousands.
MAX_SAMPLES = 2000


def build(kernel_source: str, target: dict[str, Any]) -> dict[str, Any]:
    if target["backend"] != "hip":
        return {
            "success": False,
            "diagnostics": f"gfx942 harness requires backend 'hip', got {target['backend']!r}",
        }
    directory = tempfile.TemporaryDirectory(prefix="tlx-gfx942-candidate-")
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
    if not callable(getattr(module, ENTRY_POINT, None)):
        directory.cleanup()
        return {
            "success": False,
            "diagnostics": f"candidate must define {ENTRY_POINT}(a, b)",
        }
    # Captured before anything pins, because pinning mutates the autotuner in
    # place and `verify` runs first. Snapshotting later would capture the pinned
    # single config and make `profile`'s "autotuned" number a second pinned run.
    return {
        "success": True,
        "artifact": (directory, module, str(source_path), pinning.snapshot(module)),
    }


def verify(artifact: tuple[Any, ...], case: dict[str, Any]) -> dict[str, Any]:
    _, module, source_path, original_configs = artifact
    a, b = _inputs(case)
    report = _apply_pin(module, case)
    try:
        expected = torch.matmul(a, b)
        pinned = getattr(module, ENTRY_POINT)(a, b)
        torch.testing.assert_close(pinned, expected)
    except Exception as error:  # noqa: BLE001
        return {
            "passed": False,
            "diagnostics": f"pinned correctness: {type(error).__name__}: {error}",
            "metrics": {"pin": report},
        }
    finally:
        pinning.restore(module, original_configs)

    # The performance gate is deliberately pinned and disables post-misched,
    # but users and the repository correctness suite execute the autotuner in a
    # default environment. Validate that path in a fresh process so the harness'
    # performance settings and compiled-kernel cache cannot hide a bad config.
    default_check = _verify_default_environment(Path(source_path), case)
    if not default_check["passed"]:
        return {
            "passed": False,
            "diagnostics": f"autotuned correctness: {default_check['diagnostics']}",
            "metrics": {"pin": report, "autotuned_verified": False},
        }

    metrics: dict[str, Any] = {"pin": report, "autotuned_verified": True}
    reference = _reference_module()
    if reference is not None:
        # An oracle beyond torch: if the trusted kernel and the candidate agree
        # more tightly than either agrees with torch, a torch-only check can hide
        # a real regression inside the tolerance band.
        try:
            reference_output = getattr(reference, ENTRY_POINT)(a, b)
            metrics["max_abs_diff_vs_reference"] = float((pinned.float() - reference_output.float()).abs().max())
        except Exception as error:  # noqa: BLE001
            metrics["reference_error"] = f"{type(error).__name__}: {error}"
    return {"passed": True, "metrics": metrics}


def _verify_default_environment(kernel_path: Path, case: dict[str, Any], timeout_s: float = 300.0) -> dict[str, Any]:
    child = Path(__file__).with_name("verify_child.py")
    environment = os.environ.copy()
    environment.pop("TRITON_DISABLE_POST_MISCHED", None)
    environment.pop("TRITON_USE_C_DISPATCHER", None)
    environment["TLX_VERIFY_KERNEL"] = str(kernel_path)
    environment["TLX_VERIFY_CASE"] = json.dumps(case)
    environment["TLX_VERIFY_ENTRY"] = ENTRY_POINT
    try:
        completed = subprocess.run(
            [sys.executable, str(child)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "diagnostics": f"default-environment check timed out after {timeout_s:.0f}s",
        }
    if completed.returncode != 0:
        return {
            "passed": False,
            "diagnostics": (completed.stderr or completed.stdout)[-4000:],
        }
    return {"passed": True}


def benchmark(artifact: tuple[Any, ...], case: dict[str, Any], repetitions: int) -> dict[str, Any]:
    del repetitions  # the settled-clock window sizes itself; see module docstring
    _, module, _, _ = artifact
    a, b = _inputs(case)
    _apply_pin(module, case)
    call = getattr(module, ENTRY_POINT)
    samples_ms = measure_samples(lambda: call(a, b))
    stride = max(1, len(samples_ms) // MAX_SAMPLES)
    return {
        "samples_us": [value * 1000.0 for value in samples_ms[::stride]],
        "warmup_count": 0,  # continuous warmup, not a fixed iteration count
        "cache_policy": "warm",
    }


def profile(artifact: tuple[Any, ...], case: dict[str, Any]) -> dict[str, Any]:
    _, module, source_path, original_configs = artifact
    a, b = _inputs(case)
    call = getattr(module, ENTRY_POINT)
    m, k = a.shape
    _, n = b.shape
    flops = 2.0 * m * n * k
    pin_config = _pin_config(case)

    result: dict[str, Any] = {}
    try:
        # 1. Gate conditions: pinned to one tile.
        _apply_pin(module, case)
        pinned_ms = _median_ms(lambda: call(a, b))
        result["pinned"] = {
            "config": pin_config,
            "latency_ms": pinned_ms,
            "tflops": flops * 1e-12 / (pinned_ms * 1e-3),
        }

        # 2. What a user would actually get: full autotune search.
        pinning.restore(module, original_configs)
        autotuned_ms = _median_ms(lambda: call(a, b))
        result["autotuned"] = {
            "latency_ms": autotuned_ms,
            "tflops": flops * 1e-12 / (autotuned_ms * 1e-3),
        }
    finally:
        pinning.restore(module, original_configs)

    # 3. Per-instruction thread trace, in a fresh process under rocprofv3.
    trace_root = Path(os.environ.get("TLX_ATT_OUTPUT_ROOT", tempfile.gettempdir())) / "tlx-att" / str(case["case_id"])
    result["att"] = att.collect(
        kernel_path=Path(source_path),
        case=case,
        output_dir=trace_root,
        entry_point=ENTRY_POINT,
        kernel_regex=os.environ.get("TLX_ATT_KERNEL_REGEX", ".*matmul.*"),
        pin_config=pin_config,
    )
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _median_ms(fn) -> float:
    samples = sorted(measure_samples(fn))
    return samples[len(samples) // 2]


def _pin_config(case: dict[str, Any]) -> dict[str, Any] | None:
    config = case.get("parameters", {}).get("pin_config")
    return dict(config) if config else None


def _apply_pin(module: ModuleType, case: dict[str, Any]) -> dict[str, Any]:
    return pinning.pin(module, _pin_config(case))


_REFERENCE: ModuleType | None | bool = False


def _reference_module() -> ModuleType | None:
    """Load the optional oracle kernel once, if the CLI supplied one."""
    global _REFERENCE
    if _REFERENCE is not False:
        return _REFERENCE  # type: ignore[return-value]
    path = os.environ.get("TLX_REFERENCE_KERNEL_PATH")
    if not path or not Path(path).is_file():
        _REFERENCE = None
        return None
    try:
        _REFERENCE = _load_module(Path(path), name="tlx_gfx942_reference")
    except Exception:  # noqa: BLE001
        _REFERENCE = None
    return _REFERENCE  # type: ignore[return-value]


def _load_module(path: Path, name: str | None = None) -> ModuleType:
    module_name = name or f"tlx_gfx942_candidate_{path.parent.name}"
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
    torch.manual_seed(int(parameters.get("seed", 0)))
    # Match testing/test_correctness.py's GEMM input distribution and default
    # assert_close tolerance. Large unscaled random inputs plus atol/rtol=1e-2
    # let schedule changes through the agent gate that the repository's real
    # correctness test rejects.
    a = (torch.randn(
        (int(parameters["m"]), int(parameters["k"])),
        device=device,
        dtype=dtype,
    ) + 1) / int(parameters["k"])
    b = (torch.randn(
        (int(parameters["k"]), int(parameters["n"])),
        device=device,
        dtype=dtype,
    ) + 1) / int(parameters["k"])
    return a, b
