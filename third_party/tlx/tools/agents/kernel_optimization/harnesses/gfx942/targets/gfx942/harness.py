"""Optimization harness for the TLX MI300X (gfx942 / CDNA3) GEMM op.

Targets `third_party/tlx/ops/kernels/mm/gfx942.py`, whose `matmul` alias reaches
`mm`'s default `space="heuristic"`.

Differs from the blackwell/hopper GEMM harnesses in three ways that are all
forced by the part rather than by taste:

**Timing method.** `do_bench` is not used. Its burst-and-sleep pattern was
measured at 14-20% burst-to-burst variation on this part, which would swamp the
1.01 default promotion threshold and make promotion a coin flip. This harness
reuses `gfx942_perf_harness.measure_samples` -- warm continuously until clocks
settle, then one timed window -- so the samples the optimizer computes its
median and CV over are the real per-call distribution.

**No autotune search to control for.** `space="heuristic"` resolves to a single
config per (M, N, K), so the gate is pinned by construction: a candidate is
judged on its structural change and cannot win by landing on a luckier tile.
Nothing here filters the autotuner, and `profile()` reports one latency rather
than a pinned/unpinned pair. Pointing this harness at a kernel that *does*
search would reintroduce tile selection as a confound.

**ATT profiling.** `profile()` shells rocprofv3 rather than profiling in
process, because ATT requires the application to be launched underneath it. See
`../../att.py`.
"""

from __future__ import annotations

import functools
import importlib.util
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

# `harnesses/gfx942/`, which holds the shared att/inputs modules. The harness is
# loaded by file path with no package context, so this is how they resolve.
_ARCH_DIR = Path(__file__).resolve().parents[2]
if str(_ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ARCH_DIR))

import att  # noqa: E402
from inputs import make_inputs  # noqa: E402

# Importing this sets TRITON_DISABLE_POST_MISCHED / TRITON_USE_C_DISPATCHER
# before the first kernel launch, which is also what `target.json` puts in the
# worker's environment. Both are needed: `target.json` covers the worker, this
# covers a harness imported directly.
from triton.language.extra.tlx.tutorials.testing.gfx942_perf_harness import (  # noqa: E402
    measure_samples, )

ENTRY_POINT = "matmul"

# Enough samples for a meaningful median and CV without bloating the response
# JSON; the perf harness caps its window at 4000 iterations.
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

    # The gate disables post-misched and the C dispatcher, but users and the
    # repository correctness suite execute this kernel in a default environment.
    # Validate that path in a fresh process so the harness' performance settings
    # cannot hide a config that only works under them.
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
        # An oracle beyond torch: if the trusted kernel and the candidate agree
        # more tightly than either agrees with torch, a torch-only check can hide
        # a real regression inside the tolerance band.
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
        # Not a correctness verdict: a slow or contended host must not be
        # reported as a wrong answer. The caller sees `timed_out` and can
        # distinguish it from a real mismatch.
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
    del repetitions  # the settled-clock window sizes itself; see module docstring
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
        return _load_module(Path(path), name="tlx_gfx942_reference")
    except Exception:  # noqa: BLE001
        return None


def _load_module(path: Path, name: str | None = None) -> ModuleType:
    module_name = name or f"tlx_gfx942_candidate_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
