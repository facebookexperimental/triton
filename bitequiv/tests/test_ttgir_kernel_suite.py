"""GPU integration test: run the evaluation framework against the TTGIR checker.

Drives ``bitequiv/evaluation/evaluate.py`` (Stage 2, light config + fast fuzzer) as a
subprocess, but with ``--checker bitequiv.ttgir_reduction:ttgir_reduction_descriptor
--artifact ttgir --allow-unsound`` — i.e. it measures the **TTGIR** reduction-order
checker instead of the default PTX one.

Unlike the PTX checker, the TTGIR checker is *expectedly not sound*: it sees the data
layout only and is provably blind to **FMA contraction** (decided below TTGIR, gated by
``enable_fp_fusion``). So on the mul-fed kernels (``dot``, ``welford``) it over-merges
configs whose bits differ — exactly the gap the PTX checker exists to close. We
therefore do NOT assert ``over-merges == 0`` here (that is the PTX test's job, see
``test_ptx_kernel_suite.py``). This test asserts the framework drives the TTGIR checker
end to end with ``--allow-unsound`` and emits the honest report; the over-merge count it
reports is the documented finding, not a failure.

Run as a subprocess with a timeout so a wedged kernel can't hang the session — if it ever
hangs, kill with ``third_party/tlx/killgpu.sh``.

The TTGIR checker calls into the built ``libtriton.bitequiv`` (the MLIR-native C++
analysis), so the subprocess must point at a freshly built triton (``_REPO/python``);
override with ``BITEQUIV_TRITON_PY``.

Run:  pytest bitequiv/tests/test_ttgir_kernel_suite.py -v
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
    """The count from 'SOUNDNESS: total over-merges (...) = N' (None if absent).

    The parenthetical differs by mode ('must be 0' vs '0; --allow-unsound'), so match
    any parenthetical."""
    m = re.search(r"total over-merges \([^)]*\) = (\d+)", stdout)
    return int(m.group(1)) if m else None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_ttgir_checker_reports():
    cmd = [
        sys.executable, "-m", "bitequiv.evaluation.evaluate", "--checker",
        "bitequiv.ttgir_reduction:ttgir_reduction_descriptor", "--artifact", "ttgir", "--allow-unsound", "--stages",
        "1,2", "--config-effort", "light", "--fuzzer-effort", "fast"
    ]
    # PYTHONPATH carries both the repo root (so `bitequiv` imports) and the triton build.
    pythonpath = os.pathsep.join([_REPO, _TRITON_PY])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=_REPO,
                          env={**os.environ, "PYTHONPATH": pythonpath})
    tail = proc.stdout[-4000:] + "\n" + proc.stderr[-2000:]

    over_merges = _over_merges(proc.stdout)
    # Build drift (python newer than the compiled libtriton -> every kernel fails to
    # compile) is environmental, not a checker defect; skip rather than fail.
    if over_merges is None and "0/" in proc.stdout and "compiled (build drift?)" in proc.stdout:
        pytest.skip(f"triton build drift: kernels fail to compile; rebuild triton.\n{tail}")

    # We actually exercised the TTGIR path, not the default PTX one.
    assert "artifact:       ttgir" in proc.stdout, f"framework did not run the TTGIR artifact:\n{tail}"
    # The framework produced a Stage 2 precision report with a soundness line ...
    assert "== Stage 2: precision ==" in proc.stdout, f"Stage 2 did not run:\n{tail}"
    assert over_merges is not None, f"evaluation did not report a soundness count:\n{tail}"
    # ... and --allow-unsound lets an expectedly-unsound checker exit 0 (report, not fail).
    assert proc.returncode == 0, f"evaluation exited nonzero despite --allow-unsound:\n{tail}"
