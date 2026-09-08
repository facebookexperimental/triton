"""Correctness coverage for the frozen gfx942 ``tlx.ops.linear`` shapes."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna3
from triton.tlx.ops.kernels.linear.gfx942 import LINEAR_CONFIGS

pytestmark = pytest.mark.skipif(not is_hip_cdna3(), reason="tlx.ops.linear requires gfx942")


@pytest.mark.parametrize("shape", LINEAR_CONFIGS, ids=lambda shape: "x".join(map(str, shape)))
def test_linear(shape):
    from triton.tlx.ops import linear as tlx_linear

    M, N, K = shape
    config = LINEAR_CONFIGS[shape]
    torch.manual_seed(M + N + K)
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((N, ), device="cuda", dtype=torch.bfloat16) if config["bias"] else None
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    actual = tlx_linear(a, weight, bias, out=out)
    expected = torch.nn.functional.linear(a, weight, bias)

    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)

    del a, weight, bias, out, actual, expected
    torch.cuda.empty_cache()
