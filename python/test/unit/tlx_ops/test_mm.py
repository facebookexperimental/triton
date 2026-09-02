"""L1 correctness for ``tlx.ops.mm``.

The shape list lives in ``triton.tlx.ops.kernels.mm._shapes`` and is shared
with the perf suite (``python/test/tlx_benchmark/bench_mm.py``), so a shape
disabled here is automatically not benchmarked either. One shape is currently
commented out there for a NUM_CTAS=2 bug -- see that module's docstring for the
measurement table and the control case.
"""

import time

import pytest
import torch

from triton._internal_testing import is_blackwell
from triton.tlx.ops.kernels.mm._shapes import SHAPES, operand

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.mm is sm100-only today")

torch.manual_seed(0)

ARCH = "sm100"

# The heuristic space builds exactly one config per shape, so exceeding this is
# a compile-time regression rather than a slow test.
MAX_SECONDS_PER_CASE = 60

REL_PRECISION = {torch.float16: 1e-3, torch.bfloat16: 8e-3}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("M, N, K, a_row_major, b_row_major", SHAPES)
def test_mm(M, N, K, a_row_major, b_row_major, dtype):
    from triton.tlx.ops import mm as tlx_mm

    a = operand(M, K, dtype, a_row_major)
    b = operand(K, N, dtype, b_row_major)
    assert a.is_contiguous() == a_row_major
    assert b.is_contiguous() == b_row_major

    torch.cuda.synchronize()
    started = time.perf_counter()
    out = tlx_mm(a, b, arch=ARCH, space="heuristic")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert elapsed < MAX_SECONDS_PER_CASE, (f"mm({M}x{N}x{K}, {dtype}) took {elapsed:.1f}s, "
                                            f"over the {MAX_SECONDS_PER_CASE}s budget")

    ref = torch.matmul(a, b)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)
