"""Smoke tests for the compilation-pipeline tutorials.

Each tutorial is a runnable script that compiles a kernel, prints a per-stage IR
transform, and asserts its own runtime correctness. We run each as a subprocess
(the files are digit-prefixed, so they are scripts, not importable modules) and
require a clean exit. The examples self-skip when hardware is missing, so this
passes on any CUDA box and the Blackwell-only one no-ops off Blackwell.
"""
import os
import subprocess
import sys

import pytest

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch always present in this env
    _HAS_CUDA = False

_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tutorials", "compilation-pipeline")
_EXAMPLES = [
    "01_read_ttir.py",
    "02_layout_assignment.py",
    "03_coalesce_vectorization.py",
    "04_remove_layout_conversions.py",
    "05_software_pipelining.py",
    "06_warp_specialization.py",
    "07_reduction_lowering.py",
    "08_reduction_order_numerics.py",
    "09_dot_to_mma_lowering.py",
    "10_mma_precision_numerics.py",
]

# Tutorials that demonstrate a single named compiler pass via the dump_passes /
# pass_diff hook print a ">>> pass-diff: <PassName>" line. Asserting on it proves
# the examples close the loop on *specific passes* (not just the final IR) —
# 08/10 are numerics demos that diff whole configs instead.
_PASS_DIFF_EXAMPLES = {
    "01_read_ttir.py",
    "02_layout_assignment.py",
    "03_coalesce_vectorization.py",
    "04_remove_layout_conversions.py",
    "05_software_pipelining.py",
    "06_warp_specialization.py",
    "07_reduction_lowering.py",
    "09_dot_to_mma_lowering.py",
}


@pytest.mark.skipif(not _HAS_CUDA, reason="compilation-pipeline examples need a CUDA GPU")
@pytest.mark.parametrize("script", _EXAMPLES)
def test_example_runs(script):
    path = os.path.abspath(os.path.join(_DIR, script))
    assert os.path.exists(path), path
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    skipped = "skipped" in proc.stdout
    assert "[OK]" in proc.stdout or skipped, proc.stdout[-2000:]
    # The per-pass hook must actually run (unless the example self-skipped on this
    # hardware): a relevant pass name is printed as a ">>> pass-diff:" line.
    if script in _PASS_DIFF_EXAMPLES and not skipped:
        assert ">>> pass-diff:" in proc.stdout, proc.stdout[-2000:]
