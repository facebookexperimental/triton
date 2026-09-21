import pytest
import torch

from triton._internal_testing import is_hip_cdna3
from triton.tlx.ops.kernels.addmm._shapes import CORRECTNESS_SHAPES, FOCUS, inputs
from triton.tlx.ops.kernels.mm.gfx942 import TUNED_CONFIGS

pytestmark = pytest.mark.skipif(not is_hip_cdna3(), reason="tlx.ops.addmm requires gfx942")


def test_focus_suite_covers_addmm_tuned_configs():
    focus_shapes = {(shape.m, shape.n, shape.k) for shape in FOCUS.resolved_shapes("gfx942_1")}
    assert focus_shapes == set(TUNED_CONFIGS) - {(2048, 25408, 10240)}


@pytest.mark.parametrize("m,n,k,a_strides,b_strides,dtype_name", CORRECTNESS_SHAPES)
def test_addmm(m, n, k, a_strides, b_strides, dtype_name):
    from triton.tlx.ops import addmm as tlx_addmm

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    torch.manual_seed(m + n + k)
    bias, a, b = inputs((m, n, k, a_strides, b_strides, dtype_name), dtype)
    out = torch.empty((m, n), device="cuda", dtype=dtype)
    actual = tlx_addmm(bias, a, b, out=out)
    expected = torch.addmm(bias, a, b)

    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)

    del bias, a, b, out, actual, expected
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
