"""Correctness and public-contract coverage for gfx950 MXFP8 GEMM."""

from __future__ import annotations

import pytest
import torch

from triton.tlx.ops import InvalidInput, UnsupportedBackward, mm_mxfp8
from triton.tlx.ops.kernels.mm_mxfp8._shapes import MODEL_LINEAR_SHAPES, PRODUCTION_SHAPES


def _is_gfx950():
    return (torch.cuda.is_available() and torch.version.hip is not None
            and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx950")


requires_gfx950 = pytest.mark.skipif(not _is_gfx950(), reason="Requires gfx950")


def _inputs(m, n, k, *, seed=71, fractional=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if fractional:
        a = torch.randn((m, k), generator=generator, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn((n, k), generator=generator, device="cuda").to(torch.float8_e4m3fn)
    else:
        a = torch.randint(-4, 5, (m, k), generator=generator, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randint(-4, 5, (n, k), generator=generator, device="cuda").to(torch.float8_e4m3fn)
    a_scale = torch.randint(124, 128, (m, k // 32), generator=generator, device="cuda",
                            dtype=torch.uint8).view(torch.float8_e8m0fnu)
    b_scale = torch.randint(124, 128, (n, k // 32), generator=generator, device="cuda",
                            dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return a, a_scale, b, b_scale


def _dequantize(data, scale, dtype=torch.float32):
    raw = scale.view(torch.uint8)
    decoded = torch.exp2(raw.to(dtype) - 127)
    decoded = torch.where(raw == 255, float("nan"), decoded)
    return data.to(dtype) * decoded.repeat_interleave(32, dim=-1)


def _reference(a, a_scale, b, b_scale, *, dtype=torch.float32):
    return (_dequantize(a, a_scale, dtype) @ _dequantize(b, b_scale, dtype).T).bfloat16()


def _blocked(scale):
    rows, groups = scale.shape
    return (scale.view(torch.uint8).view(rows // 128, 4, 32, groups // 4,
                                         4).permute(0, 3, 2, 1,
                                                    4).contiguous().view(torch.float8_e8m0fnu).reshape(-1, 512))


def _call(inputs, sf_layout, *, out=None):
    a, a_scale, b, b_scale = inputs
    if sf_layout == "cublas_blocked":
        a_scale, b_scale = _blocked(a_scale), _blocked(b_scale)
    return mm_mxfp8(a, a_scale, b, b_scale, sf_layout=sf_layout, out=out)


def test_mm_mxfp8_gfx950_catalog():
    from triton.tlx import ops
    from triton.tlx.ops._catalog import CATALOG, has_impl

    spec, = (spec for spec in CATALOG if (spec.op, spec.arch) == ("mm_mxfp8", "gfx950"))
    assert has_impl("mm_mxfp8", "gfx950")
    assert not has_impl("mm_mxfp8", "gfx942")
    assert spec.dtypes == frozenset({"float8_e4m3fn"})
    assert not spec.supports_backward
    assert spec.default_space == "heuristic"
    assert not hasattr(ops, "mm_a8w8")


def test_mm_mxfp8_gfx950_uses_tlx_frontend():
    from triton.tlx.ops.kernels.mm_mxfp8.gfx950 import _gemm_generic, _gemm_interwave

    assert not _gemm_generic.is_gluon()
    assert not _gemm_interwave.is_gluon()


def test_mm_mxfp8_gfx950_model_linear_shapes():
    assert len(MODEL_LINEAR_SHAPES) == 30
    assert {m for m, _, _ in MODEL_LINEAR_SHAPES} == {1024, 2048}
    assert len({(n, k) for _, n, k in MODEL_LINEAR_SHAPES}) == 15
    assert all(m % 128 == n % 128 == k % 128 == 0 for m, n, k in MODEL_LINEAR_SHAPES)


def test_mm_mxfp8_gfx950_model_split_k():
    from triton.tlx.ops.kernels.mm_mxfp8.gfx950 import _model_split_k

    assert _model_split_k(1024, 6144, 159744, 256, 256) == 4
    assert _model_split_k(1024, 6144, 22656, 256, 256) == 3
    assert _model_split_k(1024, 3072, 15360, 256, 256) == 5
    assert _model_split_k(2048, 3072, 15360, 256, 256) == 2
    assert _model_split_k(1024, 32768, 6144, 256, 256) == 1


@pytest.mark.parametrize("shape", [(128, 128, 128), (256, 384, 384), (384, 256, 640), (128, 512, 8192), (512, 512, 512),
                                   (1280, 3328, 8192), (4096, 4096, 2048), (3840, 4608, 2048), (7936, 7936, 7936),
                                   (8192, 8192, 8192)])
@pytest.mark.parametrize("sf_layout", ["natural", "cublas_blocked"])
@requires_gfx950
def test_mm_mxfp8_gfx950_correctness(shape, sf_layout):
    inputs = _inputs(*shape)
    expected = _reference(*inputs)
    out = torch.empty_like(expected)
    assert _call(inputs, sf_layout, out=out) is out
    assert out.is_contiguous()
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [
    *PRODUCTION_SHAPES,
    (61440, 1024, 4096), (122880, 1024, 4096), (16384, 1024, 512),
    (16384, 1024, 4096), (16256, 1024, 1024), (16512, 1024, 4096),
])
@requires_gfx950
def test_mm_mxfp8_gfx950_production_shapes(shape):
    inputs = _inputs(*shape, seed=91)
    expected = _reference(*inputs)
    out = torch.full_like(expected, float("nan"))
    assert mm_mxfp8(*inputs, out=out) is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@requires_gfx950
def test_mm_mxfp8_gfx950_per_32_scales():
    a = torch.ones((128, 256), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.ones_like(a)
    raw_a = torch.arange(8, device="cuda", dtype=torch.uint8)[None, :].expand(128, -1).contiguous() + 120
    raw_b = torch.full_like(raw_a, 127)
    a_scale, b_scale = raw_a.view(torch.float8_e8m0fnu), raw_b.view(torch.float8_e8m0fnu)
    actual = mm_mxfp8(a, a_scale, b, b_scale)
    torch.testing.assert_close(actual, _reference(a, a_scale, b, b_scale), rtol=0, atol=0)
    assert actual[0, 0].item() == 63.75


@pytest.mark.parametrize("shape", [(128, 256, 256), (4096, 4096, 2048), (8192, 8192, 8192),
                                   (16384, 1024, 512), (16384, 1024, 4096)])
@pytest.mark.parametrize("sf_layout", ["natural", "cublas_blocked"])
@requires_gfx950
def test_mm_mxfp8_gfx950_scale_dynamic_range(shape, sf_layout):
    m, n, k = shape
    a = torch.ones((m, k), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.ones((n, k), device="cuda", dtype=torch.float8_e4m3fn)
    exponents = torch.tensor([0, 1, 64, 127, 191, 253, 254, 127], device="cuda", dtype=torch.uint8)
    raw_a = exponents.repeat(k // 256)[None, :].expand(m, -1).contiguous()
    raw_b = (254 - exponents).repeat(k // 256)[None, :].expand(n, -1).contiguous()
    inputs = a, raw_a.view(torch.float8_e8m0fnu), b, raw_b.view(torch.float8_e8m0fnu)
    actual = _call(inputs, sf_layout)
    torch.testing.assert_close(actual, torch.full_like(actual, k), rtol=0, atol=0)


@requires_gfx950
def test_mm_mxfp8_gfx950_blocked_scales_are_bitwise_equal():
    inputs = _inputs(512, 384, 768, seed=83)
    assert torch.equal(_call(inputs, "natural"), _call(inputs, "cublas_blocked"))


@pytest.mark.parametrize("shape", [(256, 384, 640), (8192, 8192, 8192)])
@requires_gfx950
def test_mm_mxfp8_gfx950_fractional_inputs(shape):
    inputs = _inputs(*shape, fractional=True, seed=83)
    expected = _reference(*inputs, dtype=torch.float64)
    actual = mm_mxfp8(*inputs)
    torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=0.01)
    relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
        expected.float())
    assert relative_l2.item() < 1e-3


@pytest.mark.parametrize("shape", [(65536, 8192, 1024), (65536, 1024, 4096),
                                   (65536, 4096, 1024), (264192, 1024, 512)])
@requires_gfx950
def test_mm_mxfp8_gfx950_production_fractional_inputs(shape):
    inputs = _inputs(*shape, fractional=True, seed=97)
    m = shape[0]
    rows = torch.tensor([0, 7, 127, 128, 255, 256, m // 2 - 1, m // 2, m - 256, m - 1], device="cuda")
    a = inputs[0].view(torch.uint8)[rows].view(torch.float8_e4m3fn)
    a_scale = inputs[1].view(torch.uint8)[rows].view(torch.float8_e8m0fnu)
    expected = _reference(a, a_scale, *inputs[2:], dtype=torch.float64)
    actual = mm_mxfp8(*inputs)[rows]
    torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=0.01)
    relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(expected.float())
    assert relative_l2.item() < 1e-3


@pytest.mark.parametrize("shape", [(512, 512, 512), (4096, 4096, 2048), (8192, 8192, 8192),
                                   (16384, 1024, 512), (65536, 1024, 4096), (65536, 4096, 1024)])
@requires_gfx950
def test_mm_mxfp8_gfx950_repeated_launches(shape):
    inputs = _inputs(*shape, seed=83)
    expected = _reference(*inputs)
    out = torch.empty_like(expected)
    checks = torch.empty(500, device=out.device, dtype=torch.bool)
    for iteration in range(500):
        out.fill_(float("nan"))
        mm_mxfp8(*inputs, out=out)
        checks[iteration] = torch.all(out == expected)
    assert checks.all().item()


@pytest.mark.parametrize("shape", [(256, 256, 256), (4096, 4096, 2048), (8192, 8192, 8192),
                                   (16384, 1024, 512), (16384, 1024, 4096)])
@pytest.mark.parametrize("sf_layout", ["natural", "cublas_blocked"])
@requires_gfx950
def test_mm_mxfp8_gfx950_nan_scale(shape, sf_layout):
    inputs = _inputs(*shape)
    inputs[1].view(torch.uint8)[7, 3] = 255
    inputs[3].view(torch.uint8)[-1, -1] = 255
    actual = _call(inputs, sf_layout)
    expected = _reference(*inputs)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("shape", [(256, 384, 256), (16384, 1024, 512), (16384, 1024, 4096)])
@requires_gfx950
def test_mm_mxfp8_gfx950_unaligned_storage(shape):
    a, a_scale, b, b_scale = _inputs(*shape)
    tensors = []
    for tensor in (a, a_scale, b, b_scale):
        storage = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype)
        shifted = storage[1:].view_as(tensor)
        shifted.copy_(tensor)
        tensors.append(shifted)
    m, n, _ = shape
    out = torch.empty(m * n + 1, device=a.device, dtype=torch.bfloat16)[1:].view(m, n)
    assert mm_mxfp8(*tensors, out=out) is out
    torch.testing.assert_close(out, _reference(a, a_scale, b, b_scale), rtol=0, atol=0)


@requires_gfx950
def test_mm_mxfp8_gfx950_rejects_ordinary_scales_and_full_space():
    a, a_scale, b, b_scale = _inputs(128, 128, 128)
    with pytest.raises(InvalidInput, match="E8M0"):
        mm_mxfp8(a, torch.ones(128, device=a.device), b, torch.ones(128, device=a.device))
    with pytest.raises(InvalidInput, match="exact shapes"):
        mm_mxfp8(a, a_scale[:, :1].contiguous(), b, b_scale)
    with pytest.raises(InvalidInput, match="space='full'"):
        mm_mxfp8(a, a_scale, b, b_scale, space="full")


@requires_gfx950
def test_mm_mxfp8_gfx950_backward_policy():
    inputs = _inputs(128, 128, 128)
    out = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(UnsupportedBackward, match="backward on gfx950"):
        mm_mxfp8(*inputs, out=out)
    with torch.no_grad():
        mm_mxfp8(*inputs, out=out)
    torch.testing.assert_close(out, _reference(*inputs), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2033, 256, 256), (256, 2033, 256), (256, 256, 160), (256, 256, 64),
                                   (0, 256, 256), (256, 0, 256), (256, 256, 0)])
@pytest.mark.parametrize("entrypoint", ["public", "gfx950"])
def test_mm_mxfp8_gfx950_rejects_partial_dimensions(shape, entrypoint):
    from triton.tlx.ops.kernels.mm_mxfp8.gfx950 import mm_mxfp8 as backend

    m, n, k = shape
    a = torch.empty((m, k), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty((n, k), device="meta", dtype=torch.float8_e4m3fn)
    a_scale = torch.empty((m, k // 32), device="meta", dtype=torch.float8_e8m0fnu)
    b_scale = torch.empty((n, k // 32), device="meta", dtype=torch.float8_e8m0fnu)
    fn = mm_mxfp8 if entrypoint == "public" else backend
    with pytest.raises(InvalidInput, match="positive M, N and K divisible by 128"):
        fn(a, a_scale, b, b_scale)


@pytest.mark.parametrize("shape", [(65536, 65536, 32768), (32768, 32768, 128)])
def test_mm_mxfp8_gfx950_rejects_offset_overflow(shape):
    from triton.tlx.ops.kernels.mm_mxfp8.gfx950 import mm_mxfp8 as backend

    m, n, k = shape
    a = torch.empty((m, k), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty((n, k), device="meta", dtype=torch.float8_e4m3fn)
    scale = torch.empty((m, k // 32), device="meta", dtype=torch.float8_e8m0fnu)
    with pytest.raises(InvalidInput, match="32-bit offsets"):
        backend(a, scale, b, scale)
