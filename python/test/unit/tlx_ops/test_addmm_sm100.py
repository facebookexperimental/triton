import pytest
import torch

from triton._internal_testing import is_blackwell
from triton.tlx.ops.kernels.addmm._shapes import CORRECTNESS_SHAPES, inputs

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.addmm requires sm100")


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


@pytest.mark.parametrize("bias_2d", [False, True])
def test_addmm_paired_handoff_smoke(bias_2d):
    from triton.tlx.ops import addmm as tlx_addmm
    from triton.tlx.ops.kernels.addmm.sm100 import _ws_smoke_configs

    for cfg in (*_ws_smoke_configs(True), *_ws_smoke_configs(False)):
        assert cfg.kwargs["EPILOGUE_SUBTILE"] % 2 == 0
        assert cfg.kwargs["NUM_EPILOGUE_SMEM_BUFFERS"] % 2 == 0

    torch.manual_seed(0)
    m, n, k = 512, 256, 128
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(((m, n) if bias_2d else (n, )), device="cuda", dtype=torch.bfloat16)
    actual = tlx_addmm(bias, a, b, space="smoke")
    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("bias_2d", [False, True])
def test_addmm_clc_tuned_paths(bias_2d):
    from triton.tlx.ops import addmm as tlx_addmm
    from triton.tlx.ops.kernels.addmm import sm100

    # The cost-model heuristic always launches CLC on 16-bit dtypes ...
    kw_2cta = sm100._ws_heuristic_config(8192, 2048, 512, True)[0].kwargs
    kw_narrow = sm100._ws_heuristic_config(512, 32, 64, True)[0].kwargs
    assert kw_2cta["USE_CLC"] is True
    assert kw_narrow["USE_CLC"] is True
    # ... the 2-CTA pick fires past the two-wave crossover ...
    assert kw_2cta["NUM_CTAS"] == 2

    # End to end through the tuned paths: collaborative 2-CTA above the
    # crossover, and the narrow BK=128 family (BK > K here: TMA zero-fills
    # the overhang) on the skinny-N shape.
    torch.manual_seed(0)
    for m, n, k in [(8192, 2048, 512), (512, 32, 64)]:
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(((m, n) if bias_2d else (n, )), device="cuda", dtype=torch.bfloat16)
        actual = tlx_addmm(bias, a, b, space="heuristic")
        expected = torch.addmm(bias, a, b)
        torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
