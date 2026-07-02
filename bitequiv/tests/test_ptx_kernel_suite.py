"""GPU integration test: run the bitequiv evaluation framework and require soundness.

Drives ``bitequiv/evaluation/evaluate.py`` (Stage 2, light config + fast fuzzer) as a
subprocess and asserts the soundness gate: the checker must never merge two configs the
fuzzer proved bit-different (``over-merges == 0``). The driver prints a machine-readable
``SOUNDNESS: total over-merges (must be 0) = N`` line that this test parses.

Run as a subprocess with a timeout so a wedged kernel can't hang the session — if it ever
hangs, kill with ``third_party/tlx/killgpu.sh``.

The PTX checker is pure Python; it only needs a working triton to *compile* kernels. We
point the subprocess at this checkout's own build (``_REPO/python``); override with
``BITEQUIV_TRITON_PY``.

Run:  pytest bitequiv/tests/test_ptx_kernel_suite.py -v
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


def _over_merges(stdout):
    """The gated number from 'SOUNDNESS: total over-merges (must be 0) = N' (None if absent)."""
    m = re.search(r"total over-merges \(must be 0\) = (\d+)", stdout)
    return int(m.group(1)) if m else None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_evaluation_framework_sound():
    cmd = [
        sys.executable, "-m", "bitequiv.evaluation.evaluate", "--stages", "1,2", "--config-effort", "light",
        "--fuzzer-effort", "fast"
    ]
    # PYTHONPATH carries both the repo root (so `bitequiv` imports) and the triton build.
    pythonpath = os.pathsep.join([_REPO, _TRITON_PY])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=_REPO,
                          env={**os.environ, "PYTHONPATH": pythonpath})
    tail = proc.stdout[-4000:] + "\n" + proc.stderr[-2000:]

    over_merges = _over_merges(proc.stdout)
    # Build drift (python newer than the compiled libtriton -> every kernel fails to compile)
    # is environmental, not a checker defect; skip rather than fail when nothing compiled.
    if over_merges is None and "0/" in proc.stdout and "compiled (build drift?)" in proc.stdout:
        pytest.skip(f"triton build drift: kernels fail to compile; rebuild triton.\n{tail}")

    assert over_merges is not None, f"evaluation did not report a soundness count:\n{tail}"
    assert over_merges == 0, f"checker over-merged {over_merges} pair(s) the fuzzer separated:\n{tail}"
    assert proc.returncode == 0, f"evaluation exited nonzero:\n{tail}"
