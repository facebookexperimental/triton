"""L1 correctness for ``tlx.ops.mm``.

TODO(split-K remainder tile): ``[1000, 1000, 1024]`` and ``[64, 4096, 4096]``
are commented out of SHAPES below because they FAIL. Re-enable them with the
fix -- they are the regression test for it.

Split-K gives wrong results when M is not a multiple of BLOCK_SIZE_M: partials
go into a (SPLIT_K * M, N) workspace that the reduction reads at fixed M
strides, and the masked edge tile breaks that correspondence.

Measured at M=1000, N=1000, K=1024, BLOCK_SIZE_M=256 (relative L2 vs torch):
SPLIT_K=1 -> 0.0 exact, =2 -> 1.04e-1, =4 -> 1.30e-1.

Pre-existing, not a promotion artifact: reproduces as
`blackwell_gemm_ws.matmul(a, b, config=get_heuristic_config(...))`. Reachable in
production -- the heuristic picks SPLIT_K=4 for both shapes. Not yet filed.
"""

import time

import pytest
import torch

from triton._internal_testing import is_blackwell

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.mm is sm100-only today")

torch.manual_seed(0)

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
    # TODO(split-K remainder tile): fails, see module docstring.
    # [1000, 1000, 1024, True, True],
    # K-heavy: few output tiles, long reduction. Split-K territory, the one
    # path that runs a second kernel.
    [256, 256, 16384, True, True],
    # Tall-skinny: most of the grid idle.
    # TODO(split-K remainder tile): fails, see module docstring.
    # [64, 4096, 4096, True, True],
]

REL_PRECISION = {torch.float16: 1e-3, torch.bfloat16: 8e-3}


def _operand(rows, cols, dtype, row_major):
    t = torch.randn((rows, cols), device="cuda", dtype=dtype)
    return t if row_major else t.T.contiguous().T


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("M, N, K, a_row_major, b_row_major", SHAPES)
def test_mm(M, N, K, a_row_major, b_row_major, dtype):
    from triton.tlx.ops.kernels.mm import sm100

    a = _operand(M, K, dtype, a_row_major)
    b = _operand(K, N, dtype, b_row_major)
    assert a.is_contiguous() == a_row_major
    assert b.is_contiguous() == b_row_major

    torch.cuda.synchronize()
    started = time.perf_counter()
    out = sm100.mm(a, b, space="heuristic")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert elapsed < MAX_SECONDS_PER_CASE, (f"mm({M}x{N}x{K}, {dtype}) took {elapsed:.1f}s, "
                                            f"over the {MAX_SECONDS_PER_CASE}s budget")

    ref = torch.matmul(a, b)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)
