import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.addmm._shapes import CORRECTNESS_SHAPES, inputs

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="tlx.ops.addmm requires gfx950")


@pytest.mark.parametrize("m,n,k,a_strides,b_strides,bias_strides,dtype_name", CORRECTNESS_SHAPES)
def test_addmm(m, n, k, a_strides, b_strides, bias_strides, dtype_name):
    from triton.tlx.ops import addmm as tlx_addmm

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    torch.manual_seed(m + n + k)
    entry = (m, n, k, a_strides, b_strides, bias_strides, dtype_name)
    bias, a, b = inputs(entry, dtype)
    out = torch.empty((m, n), device="cuda", dtype=dtype)
    actual = tlx_addmm(bias, a, b, out=out)
    expected = torch.addmm(bias, a, b)

    assert actual is out
    precision = {torch.float16: 1e-3, torch.bfloat16: 8e-3}[dtype]
    atol = max(0.05, precision * expected.abs().max().item())
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.05)

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


@pytest.mark.parametrize(
    "expected_path,m,n,k,dtype,b_row_major",
    [
        ("cached_b_lds", 819200, 1024, 192, torch.bfloat16, False),
        ("short_k_register", 512, 512, 64, torch.bfloat16, False),
        ("row_major_direct", 2252800, 256, 256, torch.float16, True),
        ("hybrid_n160", 208, 106496, 6144, torch.float16, False),
        ("streamk", 4096, 4352, 8192, torch.float16, False),
        ("ragged_n", 4096, 320, 4096, torch.float16, False),
        ("m192n256", 800, 76800, 512, torch.float16, False),
        ("transposed_wave_grid", 128, 70770, 192, torch.float16, False),
        ("transposed_register", 128, 1536, 2688, torch.float16, False),
        ("fragmented_register", 196608, 80, 80, torch.float16, False),
        ("wave_grid", 128, 9216, 7392, torch.float16, False),
        ("persistent", 1024, 6144, 4096, torch.float16, False),
        ("range_register", 24, 1024, 768, torch.bfloat16, False),
        ("register", 4096, 4096, 2048, torch.float16, False),
        ("lds", 1024, 6144, 20480, torch.bfloat16, False),
        ("inter_wave", 4096, 192, 2048, torch.float16, False),
        ("local_split_u", 7, 8192, 2048, torch.float16, False),
    ],
)
def test_mm_family_fused_bias(expected_path, m, n, k, dtype, b_row_major):
    from triton.tlx.ops.kernels.mm import gfx950 as mm_gfx950

    torch.manual_seed(m + n + k)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    if b_row_major:
        b = torch.randn((k, n), device="cuda", dtype=dtype)
    else:
        b = torch.randn((n, k), device="cuda", dtype=dtype).T
    bias = torch.randn((n, ), device="cuda", dtype=dtype)
    if expected_path == "row_major_direct":
        dispatch = expected_path, mm_gfx950._ROW_MAJOR_DIRECT_PLANS[(m, n, k)]
    elif expected_path == "ragged_n":
        dispatch = expected_path, None
    elif expected_path == "register":
        dispatch = expected_path, mm_gfx950._register_plan_for_shape(m, n, k, dtype)
    else:
        dispatch = mm_gfx950._dispatch_for(a, b)
        assert dispatch is not None and dispatch[0] == expected_path

    actual = torch.empty((m, n), device="cuda", dtype=dtype)
    mm_gfx950._launch_dispatch(a, b, actual, dispatch, bias=bias.reshape(1, n).expand(m, n))
    expected = torch.addmm(bias, a, b)
    precision = 1e-3 if dtype == torch.float16 else 8e-3
    atol = max(0.05, precision * expected.abs().max().item())
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.05)


@pytest.mark.parametrize(
    "entry",
    [
        (3072, 1024, 2048, (2048, 1), (1, 2048), (1, ), "fp16"),
        (16, 512, 80, (80, 1), (1, 80), (1, ), "bf16"),
    ],
)
def test_full_space_warppipe(entry):
    from triton.tlx.ops import addmm as tlx_addmm

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[entry[-1]]
    bias, a, b = inputs(entry, dtype)
    actual = tlx_addmm(bias, a, b, space="full")
    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
