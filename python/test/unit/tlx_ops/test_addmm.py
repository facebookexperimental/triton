"""Correctness coverage for the gfx942 ``tlx.ops.addmm`` primitive."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna3
from triton.tlx.ops.kernels.mm.gfx942_tuned import TUNED_CONFIGS

pytestmark = pytest.mark.skipif(not is_hip_cdna3(), reason="tlx.ops.addmm requires gfx942")


@pytest.mark.parametrize("shape", TUNED_CONFIGS, ids=lambda shape: "x".join(map(str, shape)))
def test_tuned_mm_addmm(shape):
    from triton.tlx.ops import addmm as tlx_addmm
    from triton.tlx.ops import mm as tlx_mm

    M, N, K = shape
    torch.manual_seed(M + N + K)
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    b = weight.T
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    if shape == (2048, 25408, 10240):
        actual = tlx_mm(a, b, out=out)
        expected = torch.mm(a, b)
    else:
        bias = torch.randn((N, ), device="cuda", dtype=torch.bfloat16)
        actual = tlx_addmm(bias, a, b, out=out)
        expected = torch.addmm(bias, a, b)

    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)

    del a, weight, b, out, actual, expected
    torch.cuda.empty_cache()


@pytest.mark.parametrize("bias_shape", [(37, ), (1, 37), (23, 1), (23, 37)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_generic_addmm_broadcast(bias_shape, dtype):
    from triton.tlx.ops import addmm as tlx_addmm

    torch.manual_seed(0)
    a = torch.randn((23, 19), device="cuda", dtype=dtype)
    b = torch.randn((19, 37), device="cuda", dtype=dtype)
    bias = torch.randn(bias_shape, device="cuda", dtype=dtype)
    actual = tlx_addmm(bias, a, b, space="heuristic")
    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
