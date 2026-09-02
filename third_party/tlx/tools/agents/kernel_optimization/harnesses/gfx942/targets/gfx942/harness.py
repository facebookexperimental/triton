"""Optimization harness for `third_party/tlx/ops/kernels/mm/gfx942.py`.

Three departures from the blackwell/hopper GEMM harnesses, all forced by the part:

- **No `do_bench`.** Its burst-and-sleep pattern does not let clocks settle here;
  the spread swamps the 1.01 promotion threshold. Uses
  `gfx942_perf_harness.measure_samples` instead.
- **No autotune search to control for.** `matmul` reaches `mm`'s default
  `space="heuristic"`, one config per (M, N, K), so the gate is pinned by
  construction. Pointing this at a searching kernel reintroduces tile selection
  as a confound.
- **ATT profiling out of process** (`../../att.py`): rocprofv3 must launch the
  application, so `profile()` cannot trace itself.
"""

from __future__ import annotations

import functools
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

# Loaded by file path with no package context, so the shared modules in
# `harnesses/gfx942/` need an explicit path entry.
_ARCH_DIR = Path(__file__).resolve().parents[2]
if str(_ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ARCH_DIR))

import att  # noqa: E402
from inputs import make_inputs  # noqa: E402
from loader import load_candidate  # noqa: E402

# Sets TRITON_DISABLE_POST_MISCHED / TRITON_USE_C_DISPATCHER on import, covering
# a directly-imported harness; `target.json` covers the worker.
from triton.language.extra.tlx.tutorials.testing.gfx942_perf_harness import (  # noqa: E402
    measure_samples, )

ENTRY_POINT = "matmul"

# Caps the response JSON; the perf harness window runs to 4000 iterations.
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
        module = load_candidate(source_path)
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
    return {"success": True, "artifact": (directory, module, str(source_path))}


def verify(artifact: tuple[Any, ...], case: dict[str, Any]) -> dict[str, Any]:
    _, module, source_path = artifact
    a, b = make_inputs(case)
    try:
        expected = torch.matmul(a, b)
        actual = getattr(module, ENTRY_POINT)(a, b)
        torch.testing.assert_close(actual, expected)
    except Exception as error:  # noqa: BLE001
        return {
            "passed": False,
            "diagnostics": f"in-process correctness: {type(error).__name__}: {error}",
        }

    # Users run without the gate's performance env, so re-check there.
    default_check = _verify_default_environment(Path(source_path), case)
    if not default_check["passed"]:
        return {
            "passed": False,
            "diagnostics": f"default-environment correctness: {default_check['diagnostics']}",
            "metrics": {"default_environment_verified": False},
        }

    metrics: dict[str, Any] = {"default_environment_verified": True}
    reference = _reference_module()
    if reference is not None:
        # Tighter than torch: a regression can hide inside the assert_close band.
        try:
            reference_output = getattr(reference, ENTRY_POINT)(a, b)
            metrics["max_abs_diff_vs_reference"] = float((actual.float() - reference_output.float()).abs().max())
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
        # Not a correctness verdict -- a contended host is not a wrong answer.
        return {
            "passed":
            False,
            "timed_out":
            True,
            "diagnostics": (f"default-environment check did not finish within {timeout_s:.0f}s; "
                            "this is an infrastructure result, not a correctness failure"),
        }
    if completed.returncode != 0:
        return {"passed": False, "diagnostics": (completed.stderr or completed.stdout)[-4000:]}
    return {"passed": True}


def benchmark(artifact: tuple[Any, ...], case: dict[str, Any], repetitions: int) -> dict[str, Any]:
    del repetitions  # the settled-clock window sizes itself
    _, module, _ = artifact
    a, b = make_inputs(case)
    call = getattr(module, ENTRY_POINT)
    samples_ms = measure_samples(lambda: call(a, b))
    stride = max(1, len(samples_ms) // MAX_SAMPLES)
    return {
        "samples_us": [value * 1000.0 for value in samples_ms[::stride]],
        "warmup_count": 0,  # continuous warmup, not a fixed iteration count
        "cache_policy": "warm",
    }


def profile(artifact: tuple[Any, ...], case: dict[str, Any]) -> dict[str, Any]:
    _, module, source_path = artifact
    a, b = make_inputs(case)
    call = getattr(module, ENTRY_POINT)
    m, k = a.shape
    _, n = b.shape
    flops = 2.0 * m * n * k

    latency_ms = statistics.median(measure_samples(lambda: call(a, b)))
    result: dict[str, Any] = {
        "latency_ms": latency_ms,
        "tflops": flops * 1e-12 / (latency_ms * 1e-3),
    }

    # Per-instruction thread trace, in a fresh process under rocprofv3.
    trace_root = Path(os.environ.get("TLX_ATT_OUTPUT_ROOT", tempfile.gettempdir())) / "tlx-att" / str(case["case_id"])
    result["att"] = att.collect(
        kernel_path=Path(source_path),
        case=case,
        output_dir=trace_root,
        entry_point=ENTRY_POINT,
        kernel_regex=os.environ.get("TLX_ATT_KERNEL_REGEX", ".*matmul.*"),
    )
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _reference_module() -> ModuleType | None:
    """Load the optional oracle kernel once, if the CLI supplied one."""
    path = os.environ.get("TLX_REFERENCE_KERNEL_PATH")
    if not path or not Path(path).is_file():
        return None
    try:
        return load_candidate(Path(path), suffix="reference")
    except Exception:  # noqa: BLE001
        return None
