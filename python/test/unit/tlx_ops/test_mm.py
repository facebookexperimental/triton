"""L1 correctness for ``tlx.ops.mm``.

Black-box: everything here goes through the public ``tlx.ops.mm``. The split-K
workspace layout is covered white-box in ``test_mm_splitk.py``.
"""

import time

import pytest
import torch

from triton._internal_testing import is_blackwell

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.mm is sm100-only today")

torch.manual_seed(0)

ARCH = "sm100"

# The heuristic space builds exactly one config per shape, so exceeding this is
# a compile-time regression rather than a slow test.
MAX_SECONDS_PER_CASE = 60

#  Hard agnostic testsuite ``[M, N, K, opA_row_major, opB_row_major]``
SHAPES = [
    # Square, both row-major -- the baseline path, small and large.
    [256, 256, 256, True, True],
    [1024, 1024, 1024, True, True],
    # Rectangular.
    [2048, 512, 1024, True, True],
    # Column-major B: descriptor sees B.T, so the constraint lands on K.
    [512, 4096, 1024, True, False],
    # Column-major A: descriptor sees A.T, so the constraint lands on M.
    [1024, 2048, 512, False, True],
    # Both column-major.
    [2048, 2048, 2048, False, False],
    # M not a multiple of any plausible block size -- masked edge tile.
    [136, 256, 128, True, True],
    # Non-power-of-two in M and N together.
    [1000, 1000, 1024, True, True],
    # K-heavy: few output tiles, long reduction. Split-K territory, the one
    # path that runs a second kernel.
    [256, 256, 16384, True, True],
    # Tall-skinny: most of the grid idle.
    [64, 4096, 4096, True, True],
]

REL_PRECISION = {torch.float16: 1e-3, torch.bfloat16: 8e-3}


def _operand(rows, cols, dtype, row_major):
    t = torch.randn((rows, cols), device="cuda", dtype=dtype)
    return t if row_major else t.T.contiguous().T


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("M, N, K, a_row_major, b_row_major", SHAPES)
def test_mm(M, N, K, a_row_major, b_row_major, dtype):
    from triton.tlx.ops import mm as tlx_mm

    a = _operand(M, K, dtype, a_row_major)
    b = _operand(K, N, dtype, b_row_major)
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
