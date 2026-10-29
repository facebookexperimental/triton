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
        a_scale = swizzle_scale_to_5d(a_scale_bytes.view(1, m, k // 32), m // 128,
                                      k // 128).view(torch.float8_e8m0fnu)
        b_scale = swizzle_scale_to_5d(b_scale_bytes.view(1, n, k // 32), n // 128,
                                      k // 128).view(torch.float8_e8m0fnu)
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


@pytest.mark.parametrize("shape", [(128, 128, 128), (256, 384, 512), (1024, 1024, 1024),
                                    (2048, 2048, 2048), (4096, 4096, 4096)])
@pytest.mark.parametrize("sf_layout", ["natural", "cublas_blocked"])
def test_mm_mxfp8_gfx950_public_numerics(shape, sf_layout):
    from triton.tlx.ops import mm_mxfp8

    inputs, scale_bytes = _inputs(*shape, sf_layout=sf_layout)
    out = torch.empty((shape[0], shape[1]), dtype=torch.bfloat16, device="cuda")
    assert mm_mxfp8(*inputs, out=out, sf_layout=sf_layout, space="heuristic") is out
    _check_sample(out, inputs, scale_bytes)


def test_mm_mxfp8_gfx950_default_space_and_blocked_parity():
    from triton.tlx.ops import mm_mxfp8

    natural, scale_bytes = _inputs(256, 384, 512, sf_layout="natural")
    a, a_scale, b, b_scale = natural
    blocked_a = swizzle_scale_to_5d(a_scale.view(torch.uint8).view(1, 256, 16), 2,
                                    4).view(torch.float8_e8m0fnu)
    blocked_b = swizzle_scale_to_5d(b_scale.view(torch.uint8).view(1, 384, 16), 3,
                                    4).view(torch.float8_e8m0fnu)
    natural_out = mm_mxfp8(*natural)
    blocked_out = mm_mxfp8(a, blocked_a, b, blocked_b, sf_layout="cublas_blocked")
    _check_sample(natural_out, natural, scale_bytes)
    assert torch.equal(natural_out, blocked_out)


def test_mm_mxfp8_gfx950_square_config_and_unaligned_fallback():
    from triton.tlx.ops.kernels.mm_mxfp8.gfx950 import _pick_regular_config

    assert _pick_regular_config(1024, 1024, 1024, quad_safe=True) == (64, 64, 3, 4, False)
    assert _pick_regular_config(2048, 2048, 2048, quad_safe=True) == (64, 128, 2, 4, False)
    assert _pick_regular_config(4096, 4096, 4096, quad_safe=True) == (256, 256, 1, 8, True)
    assert _pick_regular_config(4096, 4096, 4096, quad_safe=False) == (128, 128, 2, 4, False)


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
