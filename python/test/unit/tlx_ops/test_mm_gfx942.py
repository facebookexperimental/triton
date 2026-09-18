"""gfx942 L1 correctness for ``tlx.ops.mm``.

Runs the synthetic list plus the gfx942 focus list -- the same set the perf
suite measures. Shapes live in ``triton.tlx.ops.kernels.mm._shapes``, so one
disabled for a correctness bug is automatically not benchmarked either.

A shape the op declines is reported as a skip with the reason, never as a
pass.
"""

import pytest
from triton._internal_testing import is_hip_cdna3
from triton.tlx.ops.kernels.mm._shapes import GFX942_FOCUS

from mm_test_utils import run_mm_case, shapes

pytestmark = pytest.mark.skipif(not is_hip_cdna3(), reason="Requires gfx942")

ARCH = "gfx942"


@pytest.mark.parametrize("M, N, K, a_strides, b_strides, dtype_name", shapes(GFX942_FOCUS))
def test_mm(M, N, K, a_strides, b_strides, dtype_name):
    run_mm_case(ARCH, M, N, K, a_strides, b_strides, dtype_name)
