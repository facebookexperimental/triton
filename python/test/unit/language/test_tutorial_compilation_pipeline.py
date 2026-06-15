"""Smoke tests for the compilation-pipeline tutorials.

Each tutorial is a runnable script that compiles a kernel, prints a per-stage IR
transform, and asserts its own runtime correctness. We run each as a subprocess
(the files are digit-prefixed, so they are scripts, not importable modules) and
require a clean exit. The examples self-skip when hardware is missing, so this
passes on any CUDA box and the Blackwell-only one no-ops off Blackwell.

The tutorials live in a *different* source tree than this test
(``python/tutorials/`` vs ``python/test/``). A hardcoded ``../../../`` relative
path resolves under the OSS checkout but not under Buck, where each target gets
its own link-tree. We instead anchor on the importable ``triton`` package
(``tutorials/`` is a sibling of ``triton/`` under ``python/`` in both layouts)
and fall back to walking up from this file; if the scripts are not packaged with
this test target at all, the cases self-skip rather than hard-fail.
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


def _find_examples_dir():
    """Locate ``tutorials/compilation-pipeline`` in OSS or Buck layouts.

    Returns the directory (verified to contain ``_ir_utils.py``) or ``None`` if
    it cannot be found, in which case the tests self-skip.
    """
    rel = os.path.join("tutorials", "compilation-pipeline")
    candidates = []
    # Most robust: anchor on the installed triton package. `tutorials/` sits next
    # to `triton/` under `python/` in both the OSS tree and the Buck link-tree.
    try:
        import triton
        py_root = os.path.dirname(os.path.dirname(os.path.abspath(triton.__file__)))
        candidates.append(os.path.join(py_root, rel))
    except Exception:  # pragma: no cover - triton always importable in this env
        pass
    # OSS source layout: three levels up from this test file.
    candidates.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", rel))
    # Fallback: walk up from this file, trying both `<d>/python/<rel>` and `<d>/<rel>`.
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        candidates.append(os.path.join(d, "python", rel))
        candidates.append(os.path.join(d, rel))
        d = os.path.dirname(d)
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.exists(os.path.join(c, "_ir_utils.py")):
            return c
    return None


_DIR = _find_examples_dir()
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
    if _DIR is None:
        pytest.skip("compilation-pipeline tutorials are not packaged with this test target")
    path = os.path.join(_DIR, script)
    if not os.path.exists(path):
        pytest.skip(f"{script} is not packaged with this test target")
    # Make the sibling `_ir_utils` import resolve in the child interpreter. When a
    # plain `python script.py` runs, CPython prepends the script's dir to sys.path,
    # but a Buck par launched via sys.executable does not, so set it explicitly.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_DIR, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=600, env=env)
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    skipped = "skipped" in proc.stdout
    assert "[OK]" in proc.stdout or skipped, proc.stdout[-2000:]
    # The per-pass hook must actually run (unless the example self-skipped on this
    # hardware): a relevant pass name is printed as a ">>> pass-diff:" line.
    if script in _PASS_DIFF_EXAMPLES and not skipped:
        assert ">>> pass-diff:" in proc.stdout, proc.stdout[-2000:]
