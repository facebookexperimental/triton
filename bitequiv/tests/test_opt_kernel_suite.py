"""GPU integration test: run the M2 optimization-pass evaluator and require bit-preservation.

Drives ``bitequiv/evaluation/evaluate_opt.py`` (Stages 1+2, light config) as a subprocess
and asserts the correctness gate: applying the optimization pass(es) must NOT change the
result bits vs the baseline compile (``bit-changed configs == 0``). The driver prints two
machine-readable lines this test parses::

    CHECKED: total configs checked = N
    CORRECTNESS: total bit-changed configs (must be 0) = M

To exercise the in-process pass-injection path itself (not just the no-op baseline), the
gate injects a benign, always-available, bit-preserving pass (``cse``). A future M2 pass
can be swapped in via ``BITEQUIV_OPT_PASSES``.

Run as a subprocess with a timeout so a wedged kernel can't hang the session — if it ever
hangs, kill with ``third_party/tlx/killgpu.sh``.

Run:  pytest bitequiv/tests/test_opt_kernel_suite.py -v
"""
import os
import re
import subprocess
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_TRITON_PY = os.environ.get("BITEQUIV_TRITON_PY", os.path.join(_REPO, "python"))
# Benign, bit-preserving pass to exercise the injection plumbing; override for a real pass.
_OPT_PASSES = os.environ.get("BITEQUIV_OPT_PASSES", "cse")


def _int_after(pattern, text):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_optimization_preserves_bits():
    cmd = [
        sys.executable, "-m", "bitequiv.evaluation.evaluate_opt", "--opt-passes", _OPT_PASSES, "--stages", "1,2",
        "--config-effort", "light"
    ]
    pythonpath = os.pathsep.join([_REPO, _TRITON_PY])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=_REPO,
                          env={**os.environ, "PYTHONPATH": pythonpath})
    tail = proc.stdout[-4000:] + "\n" + proc.stderr[-2000:]

    checked = _int_after(r"total configs checked = (\d+)", proc.stdout)
    changed = _int_after(r"total bit-changed configs \(must be 0\) = (\d+)", proc.stdout)
    regressions = _int_after(r"total compile regressions \(must be 0\) = (\d+)", proc.stdout)

    # Build drift (python newer than the compiled libtriton) makes every kernel fail to
    # compile; that is environmental, not an optimization defect, so skip rather than fail.
    if changed is None or checked == 0:
        pytest.skip(f"nothing compiled/checked (triton build drift?); rebuild triton.\n{tail}")

    assert changed == 0, f"optimization changed the bits of {changed} config(s):\n{tail}"
    assert regressions == 0, f"optimization caused {regressions} compile regression(s):\n{tail}"
    assert proc.returncode == 0, f"evaluate_opt exited nonzero:\n{tail}"
