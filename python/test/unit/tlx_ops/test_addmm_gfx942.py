import pytest
import torch

from triton._internal_testing import is_hip_cdna3
from triton.tlx.ops.kernels.addmm._shapes import CORRECTNESS_SHAPES, FOCUS, inputs

pytestmark = pytest.mark.skipif(not is_hip_cdna3(), reason="tlx.ops.addmm requires gfx942")


@pytest.mark.parametrize("m,n,k,a_strides,b_strides,bias_strides,dtype_name", CORRECTNESS_SHAPES)
def test_addmm(m, n, k, a_strides, b_strides, bias_strides, dtype_name):
    from triton.tlx.ops import addmm as tlx_addmm

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    torch.manual_seed(m + n + k)
    bias, a, b = inputs((m, n, k, a_strides, b_strides, bias_strides, dtype_name), dtype)
    out = torch.empty((m, n), device="cuda", dtype=dtype)
    actual = tlx_addmm(bias, a, b, out=out)
    expected = torch.addmm(bias, a, b)

    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)

    del bias, a, b, out, actual, expected
    torch.cuda.empty_cache()


@pytest.mark.parametrize("shape", FOCUS.resolved_shapes("gfx942_3"), ids=lambda shape: f"{shape.m}-{shape.n}-{shape.k}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_short_m_pipeline_repeated_outputs(shape, dtype):
    from triton.tlx.ops import addmm as tlx_addmm

    torch.manual_seed(shape.m + shape.n + shape.k)
    bias, a, b = inputs(shape, dtype)
    out = torch.empty((shape.m, shape.n), device="cuda", dtype=dtype)
    for _ in range(2):
        out.fill_(float("nan"))
        assert tlx_addmm(bias, a, b, out=out, space="heuristic") is out
        torch.testing.assert_close(out, torch.addmm(bias, a, b), atol=0.05, rtol=0.05)
        a.neg_()
        bias.add_(1)

    # Every output must be overwritten and the bias applied once, even when
    # the same output tensor is reused after a nonzero GEMM.
    a.zero_()
    bias.fill_(2)
    out.fill_(float("nan"))
    assert tlx_addmm(bias, a, b, out=out, space="heuristic") is out
    torch.testing.assert_close(out, bias.expand_as(out), atol=1e-3, rtol=0)


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


@pytest.mark.parametrize("m", [4178, 4434, 4942])
@pytest.mark.parametrize("n", [256, 512, 1280, 2560])
def test_deep_k_batch_boundaries(m, n):
    from triton.tlx.ops import addmm as tlx_addmm

    k = 18688
    torch.manual_seed(m + n + k)
    bias, a, b = inputs((m, n, k, (k, 1), (1, k), (1, ), "fp16"), torch.float16)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    actual = tlx_addmm(bias, a, b, out=out, space="heuristic")
    assert actual is out
    torch.testing.assert_close(actual, torch.addmm(bias, a, b), atol=0.05, rtol=0.05)


def test_deep_k_bias_applied_once_on_repeated_calls():
    from triton.tlx.ops import addmm as tlx_addmm

    m, n, k = 4178, 256, 18688
    a = torch.zeros((m, k), device="cuda", dtype=torch.float16)
    b = torch.ones((n, k), device="cuda", dtype=torch.float16).t()
    bias = torch.arange(n, device="cuda", dtype=torch.float16)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    for value in (0.0, 1.0 / 1024, -1.0 / 1024):
        a.fill_(value)
        actual = tlx_addmm(bias, a, b, out=out, space="heuristic")
        assert actual is out
        torch.testing.assert_close(actual, torch.addmm(bias, a, b), atol=1e-3, rtol=0)


@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
def test_deep_k_rejects_mismatched_bias_dtype(bias_dtype):
    from triton.tlx.ops.kernels.mm import gfx942

    m, n, k = 4178, 256, 18688
    a = torch.empty((m, k), device="cuda", dtype=torch.float16)
    b = torch.empty((n, k), device="cuda", dtype=torch.float16).t()
    bias = torch.empty((n, ), device="cuda", dtype=bias_dtype)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    assert not gfx942._precheck_global_split_k(a, b, bias, out)
    with pytest.raises(ValueError, match="input must match A's device and dtype"):
        gfx942._gemm(a, b, bias, out=out)


def test_deep_k_rejects_cpu_bias():
    from triton.tlx.ops.kernels.mm import gfx942

    m, n, k = 4178, 256, 18688
    a = torch.empty((m, k), device="cuda", dtype=torch.float16)
    b = torch.empty((n, k), device="cuda", dtype=torch.float16).t()
    bias = torch.empty((n, ), device="cpu", dtype=torch.float16)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    assert not gfx942._precheck_global_split_k(a, b, bias, out)
    with pytest.raises(ValueError, match="input must match A's device and dtype"):
        gfx942._gemm(a, b, bias, out=out)
