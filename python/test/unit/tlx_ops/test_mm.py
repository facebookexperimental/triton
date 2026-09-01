"""L1 correctness for ``tlx.ops.mm``.

Runs :data:`ALL` -- every architecture's shapes, not just this GPU's. The lists
live in ``triton.tlx.ops.kernels.mm._shapes`` and are shared with the perf suite
(``python/test/tlx_benchmark/bench_mm.py``), so a shape disabled here is
automatically not benchmarked either. One shape is currently commented out there
for a NUM_CTAS=2 bug -- see that module's docstring for the measurement table
and the control case.

Running the union is the point rather than an accident: another architecture's
real-world shapes are geometries nobody chose with *this* kernel in mind, which
is what makes them good at finding assumptions it did not know it had. The cost
is that not every shape is admissible everywhere -- sm100 needs 16-byte-aligned
TMA strides and gfx942 does not -- so a shape the op declines is reported as a
skip with the reason, never as a pass.
"""

import time

import pytest
import torch

from triton._internal_testing import is_blackwell, is_hip_cdna3
from triton.tlx.ops import InvalidInput, UnsupportedOp
from triton.tlx.ops.kernels.mm._shapes import ALL, operand


def _arch():
    """The catalog key for the GPU under test, or None if mm has no entry."""
    if is_blackwell():
        return "sm100"
    if is_hip_cdna3():
        return "gfx942"
    return None


ARCH = _arch()

pytestmark = pytest.mark.skipif(ARCH is None, reason="tlx.ops.mm has no implementation for this GPU")

torch.manual_seed(0)

# The heuristic space builds exactly one config per shape, so exceeding this is
# a compile-time regression rather than a slow test.
MAX_SECONDS_PER_CASE = 60

REL_PRECISION = {torch.float16: 1e-3, torch.bfloat16: 8e-3}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("M, N, K, a_row_major, b_row_major", ALL)
def test_mm(M, N, K, a_row_major, b_row_major, dtype):
    from triton.tlx.ops import mm as tlx_mm

    a = operand(M, K, dtype, a_row_major)
    b = operand(K, N, dtype, b_row_major)
    assert a.is_contiguous() == a_row_major
    assert b.is_contiguous() == b_row_major

    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        out = tlx_mm(a, b, arch=ARCH, space="heuristic")
    except (InvalidInput, UnsupportedOp) as declined:
        # An arch refusing a shape from another arch's list is expected and is
        # not a failure. Skipping rather than passing keeps it visible: a shape
        # silently not covered here is a shape nobody is checking.
        pytest.skip(f"{ARCH} declines this shape: {declined}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert elapsed < MAX_SECONDS_PER_CASE, (f"mm({M}x{N}x{K}, {dtype}) took {elapsed:.1f}s, "
                                            f"over the {MAX_SECONDS_PER_CASE}s budget")

    ref = torch.matmul(a, b)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)

    # These are multi-gigabyte operands at the top of the real-world lists; without
    # this the union OOMs partway through rather than at a diagnosable point.
    del a, b, out, ref
    torch.cuda.empty_cache()
