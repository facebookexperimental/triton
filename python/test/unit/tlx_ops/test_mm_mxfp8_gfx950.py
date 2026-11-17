"""gfx950 public-API coverage for ``tlx.ops.mm_mxfp8``."""

from __future__ import annotations

import pytest
import torch

from triton._internal_testing import is_hip_cdna4, swizzle_scale_to_5d

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")


def _inputs(m, n, k, *, sf_layout):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(43)
    a = torch.randint(-2, 3, (m, k), device="cuda", generator=generator).to(torch.float8_e4m3fn)
    b = torch.randint(-2, 3, (n, k), device="cuda", generator=generator).to(torch.float8_e4m3fn)
    a_scale_bytes = torch.randint(125, 128, (m, k // 32), device="cuda", dtype=torch.uint8, generator=generator)
    b_scale_bytes = torch.randint(125, 128, (n, k // 32), device="cuda", dtype=torch.uint8, generator=generator)
    if sf_layout == "cublas_blocked":
        a_padded = torch.nn.functional.pad(a_scale_bytes, (0, 0, 0, -m % 128))
        b_padded = torch.nn.functional.pad(b_scale_bytes, (0, 0, 0, -n % 128))
        a_scale = swizzle_scale_to_5d(a_padded[None], (m + 127) // 128, k // 128).view(torch.float8_e8m0fnu)
        b_scale = swizzle_scale_to_5d(b_padded[None], (n + 127) // 128, k // 128).view(torch.float8_e8m0fnu)
    else:
        a_scale = a_scale_bytes.view(torch.float8_e8m0fnu)
        b_scale = b_scale_bytes.view(torch.float8_e8m0fnu)
    return (a, a_scale, b, b_scale), (a_scale_bytes, b_scale_bytes)


def _check_sample(out, inputs, scale_bytes):
    a, _, b, _ = inputs
    a_scales, b_scales = scale_bytes
    rows = min(128, a.shape[0])
    cols = min(128, b.shape[0])
    a_dequant = a[:rows].float() * torch.exp2(a_scales[:rows].float() - 127).repeat_interleave(32, dim=1)
    b_dequant = b[:cols].float() * torch.exp2(b_scales[:cols].float() - 127).repeat_interleave(32, dim=1)
    reference = a_dequant @ b_dequant.T
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:rows, :cols].float(), reference, rtol=0.03, atol=0.5)


def _reference_slice(inputs, scale_bytes, row_slice, col_slice):
    a, _, b, _ = inputs
    a_scales, b_scales = scale_bytes
    a_dequant = a[row_slice].float() * torch.exp2(a_scales[row_slice].float() - 127).repeat_interleave(32, dim=1)
    b_dequant = b[col_slice].float() * torch.exp2(b_scales[col_slice].float() - 127).repeat_interleave(32, dim=1)
    return a_dequant @ b_dequant.T


def test_mm_mxfp8_gfx950_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, has_impl, impl_for
    from triton.tlx.ops.kernels.mm_mxfp8.gfx950 import mm_mxfp8

    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("mm_mxfp8", "gfx950")]
    assert len(specs) == 1
    assert specs[0].dtypes == frozenset({"float8_e4m3fn"})
    assert not specs[0].supports_backward
    assert has_impl("mm_mxfp8", "gfx950")
    implementation, _ = impl_for("mm_mxfp8", arch="gfx950")
    assert implementation is mm_mxfp8


@pytest.mark.parametrize("shape", [(128, 128, 128), (256, 384, 512), (1024, 1024, 1024), (2048, 2048, 2048),
                                   (4096, 4096, 4096)])
@pytest.mark.parametrize("sf_layout", ["natural", "cublas_blocked"])
def test_mm_mxfp8_gfx950_public_numerics(shape, sf_layout):
    from triton.tlx.ops import mm_mxfp8

    inputs, scale_bytes = _inputs(*shape, sf_layout=sf_layout)
    out = torch.empty((shape[0], shape[1]), dtype=torch.bfloat16, device="cuda")
    assert mm_mxfp8(*inputs, out=out, sf_layout=sf_layout, space="heuristic") is out
    _check_sample(out, inputs, scale_bytes)


@pytest.mark.parametrize("shape", [(1, 1, 128), (65, 73, 128), (128, 129, 128), (129, 128, 128), (129, 257, 256),
                                   (255, 130, 128)])
@pytest.mark.parametrize("sf_layout", ["natural", "cublas_blocked"])
def test_mm_mxfp8_gfx950_ragged_mn(shape, sf_layout):
    from triton.tlx.ops import mm_mxfp8

    inputs, scale_bytes = _inputs(*shape, sf_layout=sf_layout)
    out = mm_mxfp8(*inputs, sf_layout=sf_layout)
    reference = _reference_slice(inputs, scale_bytes, slice(None), slice(None))
    assert out.shape == shape[:2]
    torch.testing.assert_close(out.float(), reference, rtol=0.03, atol=0.5)


@pytest.mark.parametrize("sf_layout", ["natural", "cublas_blocked"])
def test_mm_mxfp8_gfx950_quad_odd_k_tail(sf_layout):
    from triton.tlx.ops import mm_mxfp8

    inputs, scale_bytes = _inputs(4096, 4096, 4224, sf_layout=sf_layout)
    # Only the 33rd K tile contributes; skipping or misaddressing the cold tail
    # must produce the wrong output rather than being masked by the first 32.
    inputs[0][:, :4096] = 0
    inputs[2][:, :4096] = 0
    out = mm_mxfp8(*inputs, sf_layout=sf_layout)
    for rows, cols in ((slice(0, 128), slice(0, 128)), (slice(-128, None), slice(-128, None))):
        reference = _reference_slice(inputs, scale_bytes, rows, cols)
        torch.testing.assert_close(out[rows, cols].float(), reference, rtol=0.03, atol=0.5)


def test_mm_mxfp8_gfx950_default_space_and_blocked_parity():
    from triton.tlx.ops import mm_mxfp8

    natural, scale_bytes = _inputs(256, 384, 512, sf_layout="natural")
    a, a_scale, b, b_scale = natural
    blocked_a = swizzle_scale_to_5d(a_scale.view(torch.uint8).view(1, 256, 16), 2, 4).view(torch.float8_e8m0fnu)
    blocked_b = swizzle_scale_to_5d(b_scale.view(torch.uint8).view(1, 384, 16), 3, 4).view(torch.float8_e8m0fnu)
    natural_out = mm_mxfp8(*natural)
    blocked_out = mm_mxfp8(a, blocked_a, b, blocked_b, sf_layout="cublas_blocked")
    _check_sample(natural_out, natural, scale_bytes)
    assert torch.equal(natural_out, blocked_out)


def test_mm_mxfp8_gfx950_rejects_invalid_scales_and_output():
    from triton.tlx.ops import InvalidInput, mm_mxfp8

    inputs, _ = _inputs(128, 256, 128, sf_layout="natural")
    a, a_scale, b, b_scale = inputs
    with pytest.raises(InvalidInput, match="natural scales must have exact shapes"):
        mm_mxfp8(a, a_scale[:, :-1].contiguous(), b, b_scale)
    with pytest.raises(InvalidInput, match="expects E8M0"):
        mm_mxfp8(a, a_scale.view(torch.uint8), b, b_scale)
    with pytest.raises(InvalidInput, match="out must be contiguous BF16"):
        mm_mxfp8(*inputs, out=torch.empty((128, 256), dtype=torch.float16, device="cuda"))


def test_mm_mxfp8_gfx950_unaligned_operand_uses_generic_fallback():
    from triton.tlx.ops import mm_mxfp8

    inputs, scale_bytes = _inputs(4096, 4096, 4096, sf_layout="natural")
    a, a_scale, b, b_scale = inputs
    backing = torch.empty(a.numel() + 1, dtype=a.dtype, device=a.device)
    unaligned_a = backing[1:].view_as(a)
    unaligned_a.copy_(a)
    assert unaligned_a.data_ptr() % 16 != 0
    out = mm_mxfp8(unaligned_a, a_scale, b, b_scale, space="heuristic")
    _check_sample(out, (unaligned_a, a_scale, b, b_scale), scale_bytes)


def _mfma_preshuffled(scale, repeat):
    rows, groups = scale.shape
    logical_row_chunk = repeat * 32
    physical_row_chunk = ((repeat + 1) // 2) * 64
    padded_rows = (rows + logical_row_chunk - 1) // logical_row_chunk * physical_row_chunk
    padded = torch.full((padded_rows, groups), 127, dtype=torch.uint8, device=scale.device)
    logical_rows = torch.arange(rows, device=scale.device)
    physical_rows = (logical_rows // logical_row_chunk * physical_row_chunk + logical_rows % logical_row_chunk)
    padded[physical_rows] = scale
    return (padded.view(padded_rows // 64, 2, 32, groups // 4, 2, 2).permute(0, 3, 5, 2, 4,
                                                                             1).contiguous().view(padded_rows, groups))


@pytest.mark.parametrize(
    "shape,a_repeat,b_repeat",
    [
        ((1024, 36864, 1024), 2, 3),
        ((1024, 6144, 67584), 4, 3),
        ((2048, 1024, 24576), 2, 4),
    ],
)
def test_mm_mxfp8_gfx950_mfma_preshuffled_scales_are_bitwise_equal(shape, a_repeat, b_repeat):
    from triton.tlx.ops import mm_mxfp8

    blocked_inputs, _ = _inputs(*shape, sf_layout="cublas_blocked")
    expected = mm_mxfp8(*blocked_inputs, sf_layout="cublas_blocked")
    natural_inputs, scale_bytes = _inputs(*shape, sf_layout="natural")
    a, _, b, _ = natural_inputs
    scale_a, scale_b = scale_bytes
    actual = mm_mxfp8(
        a,
        _mfma_preshuffled(scale_a, a_repeat),
        b,
        _mfma_preshuffled(scale_b, b_repeat),
        sf_layout="mfma_preshuffled",
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "shape,a_repeat,b_repeat",
    [
        ((85, 22464, 1024), 2, 1),
        ((85, 3584, 25600), 2, 1),
        ((85, 3072, 18048), 2, 1),
        ((2048, 22464, 1024), 2, 3),
        ((2048, 15360, 3584), 4, 3),
        ((2048, 3584, 25600), 4, 3),
        ((2048, 3072, 18048), 4, 3),
        ((1024, 8192, 3072), 2, 2),
        ((2048, 3072, 15360), 4, 3),
        ((1024, 6144, 22656), 4, 3),
        ((2048, 6144, 22656), 4, 3),
        ((2048, 3072, 12416), 2, 3),
        ((2048, 32768, 6144), 4, 4),
    ],
)
def test_mm_mxfp8_gfx950_production_preshuffled_scales(shape, a_repeat, b_repeat):
    from triton.tlx.ops import mm_mxfp8

    inputs, scale_bytes = _inputs(*shape, sf_layout="natural")
    a, _, b, _ = inputs
    scale_a, scale_b = scale_bytes
    out = mm_mxfp8(
        a,
        _mfma_preshuffled(scale_a, a_repeat),
        b,
        _mfma_preshuffled(scale_b, b_repeat),
        sf_layout="mfma_preshuffled",
    )
    for cols in (slice(0, 128), slice(-128, None)):
        reference = _reference_slice(inputs, scale_bytes, slice(None), cols)
        torch.testing.assert_close(out[:, cols].float(), reference, rtol=0.03, atol=0.5)


def test_mm_mxfp8_gfx950_long_k_split12():
    from triton.tlx.ops import mm_mxfp8

    inputs, scale_bytes = _inputs(1024, 1024, 159744, sf_layout="cublas_blocked")
    out = mm_mxfp8(*inputs, sf_layout="cublas_blocked")
    _check_sample(out, inputs, scale_bytes)


def test_mm_mxfp8_gfx950_compact_blocked_route():
    from triton.tlx.ops import mm_mxfp8

    inputs, scale_bytes = _inputs(1024, 36864, 1024, sf_layout="cublas_blocked")
    out = mm_mxfp8(*inputs, sf_layout="cublas_blocked")
    _check_sample(out, inputs, scale_bytes)


@pytest.mark.parametrize(
    "shape",
    [
        (1024, 32768, 6144),
        (2048, 8192, 3072),
    ],
)
def test_mm_mxfp8_gfx950_balanced_route(shape):
    from triton.tlx.ops import mm_mxfp8

    inputs, scale_bytes = _inputs(*shape, sf_layout="cublas_blocked")
    out = mm_mxfp8(*inputs, sf_layout="cublas_blocked")
    _check_sample(out, inputs, scale_bytes)


@pytest.mark.parametrize(
    "shape",
    [
        (85, 23040, 1792),
        (85, 22464, 1024),
        (128, 23040, 1792),
        (128, 22464, 1024),
        (85, 3584, 25600),
        (128, 3584, 25600),
        (85, 3072, 18048),
        (128, 15360, 3584),
        (2048, 23040, 1792),
        (2048, 22464, 1024),
        (2048, 14976, 1024),
        (2048, 3584, 25600),
    ],
)
@pytest.mark.parametrize("sf_layout", ["natural", "cublas_blocked"])
def test_mm_mxfp8_gfx950_production_scale_contracts(shape, sf_layout):
    from triton.tlx.ops import mm_mxfp8

    inputs, scale_bytes = _inputs(*shape, sf_layout=sf_layout)
    out = mm_mxfp8(*inputs, sf_layout=sf_layout)
    for cols in (slice(0, 128), slice(-128, None)):
        reference = _reference_slice(inputs, scale_bytes, slice(None), cols)
        torch.testing.assert_close(out[:, cols].float(), reference, rtol=0.03, atol=0.5)
