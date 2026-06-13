"""Smoke tests for the layout/numerics examples (project: bitwise equivalence).

Each example compiles a kernel, prints the IR/PTX evidence for a bit-changing
pass, and asserts the runtime bit behavior itself (e.g. that two layouts differ,
or that inner_tree makes them equal). We run each as a subprocess (digit/letter
prefixed scripts, not importable modules) and require a clean exit. Blackwell-only
examples self-skip elsewhere, so this passes on any CUDA box.
"""
import os
import subprocess
import sys

import pytest

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAS_CUDA = False

_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "layout_numerics")
_EXAMPLES = [
    "b01_reduction_tree_from_layout.py",
    "b02_reduction_ordering_inner_tree.py",
    "b03_mma_precision.py",
    "b04_f32_dot_tc.py",
    "b05_fma_contraction.py",
    "b06_elementwise_math_precision.py",
]


@pytest.mark.skipif(not _HAS_CUDA, reason="layout/numerics examples need a CUDA GPU")
@pytest.mark.parametrize("script", _EXAMPLES)
def test_example_runs(script):
    path = os.path.abspath(os.path.join(_DIR, script))
    assert os.path.exists(path), path
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    assert "[OK]" in proc.stdout or "skipped" in proc.stdout, proc.stdout[-2000:]
