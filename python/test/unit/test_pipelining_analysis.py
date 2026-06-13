"""End-to-end test for the PipeliningAnalysis diagnostic pass."""
import os
from contextlib import contextmanager

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda, is_hip


@contextmanager
def pipelining_analysis_context():
    """Enable pipelining analysis diagnostics."""
    old_diag = os.environ.get("MLIR_ENABLE_DIAGNOSTICS", "")
    old_analysis = os.environ.get("TRITON_ENABLE_EXTRA_ANALYSIS_PASSES", "")
    try:
        os.environ["MLIR_ENABLE_DIAGNOSTICS"] = "remarks"
        os.environ["TRITON_ENABLE_EXTRA_ANALYSIS_PASSES"] = "1"
        yield
    finally:
        os.environ["MLIR_ENABLE_DIAGNOSTICS"] = old_diag
        os.environ["TRITON_ENABLE_EXTRA_ANALYSIS_PASSES"] = old_analysis


def test_outer_loop_remark(capfd, fresh_triton_cache):
    """An outer loop containing a nested loop should emit 'outer loop' remark."""
    if is_hip():
        pytest.skip("CUDA specific test")
    if not is_cuda():
        pytest.skip("Requires CUDA GPU")

    @triton.jit
    def kernel(in_ptr, out_ptr, N: tl.constexpr):
        offs = tl.arange(0, N)
        acc = tl.zeros((N, ), dtype=tl.float32)
        for i in tl.range(0, 4, num_stages=3):
            for j in tl.range(0, 16):
                x = tl.load(in_ptr + offs + (i * 16 + j) * N)
                acc += x
        tl.store(out_ptr + offs, acc)

    inp = torch.randn(64 * 64, device="cuda")
    out = torch.empty(64, device="cuda")

    with pipelining_analysis_context():
        kernel[(1, )](inp, out, N=64)

    _, err = capfd.readouterr()
    assert "outer loop" in err, (f"Expected 'outer loop' remark, got:\n{err}")
