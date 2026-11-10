"""gfx1250 public-API coverage for ``tlx.ops.mm_mxfp8``."""

from __future__ import annotations

import pytest
import torch

from triton._internal_testing import is_hip_gfx1250, swizzle_scale_to_5d

requires_gfx1250 = pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")


def _inputs(m, n, k, *, sf_layout):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(43)
    a = torch.randint(-2, 3, (m, k), device="cuda", generator=generator).to(torch.float8_e4m3fn)
    b = torch.randint(-2, 3, (n, k), device="cuda", generator=generator).to(torch.float8_e4m3fn)
    a_scale_bytes = torch.randint(125, 128, (m, k // 32), device="cuda", dtype=torch.uint8, generator=generator)
    b_scale_bytes = torch.randint(125, 128, (n, k // 32), device="cuda", dtype=torch.uint8, generator=generator)
    if sf_layout == "cublas_blocked":
        a_scale = swizzle_scale_to_5d(a_scale_bytes[None], m // 128, k // 128).view(torch.float8_e8m0fnu)
        b_scale = swizzle_scale_to_5d(b_scale_bytes[None], n // 128, k // 128).view(torch.float8_e8m0fnu)
    else:
        a_scale = a_scale_bytes.view(torch.float8_e8m0fnu)
        b_scale = b_scale_bytes.view(torch.float8_e8m0fnu)
    return (a, a_scale, b, b_scale), (a_scale_bytes, b_scale_bytes)


def _reference(inputs, scale_bytes):
    a, _, b, _ = inputs
    a_scales, b_scales = scale_bytes
    a_dequant = a.float() * torch.exp2(a_scales.float() - 127).repeat_interleave(32, dim=1)
    b_dequant = b.float() * torch.exp2(b_scales.float() - 127).repeat_interleave(32, dim=1)
    return a_dequant @ b_dequant.T


def test_mm_mxfp8_gfx1250_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, has_impl

    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("mm_mxfp8", "gfx1250")]
    assert len(specs) == 1
    assert specs[0].impl == "triton.tlx.ops.kernels.mm_mxfp8.gfx1250:mm_mxfp8"
    assert specs[0].dtypes == frozenset({"float8_e4m3fn"})
    assert not specs[0].supports_backward
    assert has_impl("mm_mxfp8", "gfx1250")


@pytest.mark.parametrize(
    "m,n,k,expected",
    [
        # Benchmarked defaults: static merged loop up to K=16384, dynamic beyond.
        (8192, 8192, 8192, dict(NUM_BUFFERS=4, CLUSTER_SIZE=16, STATIC_SLOT_UNROLL=True, OVERLAP_PEEL=True)),
        (8192, 8192, 16384, dict(NUM_BUFFERS=4, CLUSTER_SIZE=16, STATIC_SLOT_UNROLL=True, OVERLAP_PEEL=True)),
        (16384, 16384, 32768, dict(NUM_BUFFERS=4, CLUSTER_SIZE=16, STATIC_SLOT_UNROLL=False, OVERLAP_PEEL=False)),
        # Clusters need whole 2048-row M groups and 1024-column N groups.
        (512, 512, 768, dict(NUM_BUFFERS=4, CLUSTER_SIZE=1, STATIC_SLOT_UNROLL=True, OVERLAP_PEEL=True)),
        # Four or fewer K tiles cannot overlap the next tile's first step.
        (8192, 8192, 512, dict(NUM_BUFFERS=4, CLUSTER_SIZE=16, STATIC_SLOT_UNROLL=False, OVERLAP_PEEL=False)),
        (512, 512, 384, dict(NUM_BUFFERS=3, CLUSTER_SIZE=1, STATIC_SLOT_UNROLL=False)),
        (256, 256, 256, dict(NUM_BUFFERS=2, CLUSTER_SIZE=1, NUM_PROGRAMS=1)),
    ],
)
def test_mm_mxfp8_gfx1250_schedule(m, n, k, expected):
    from triton.tlx.ops.kernels.mm_mxfp8.gfx1250 import _schedule

    cfg = _schedule(m, n, k, 256)
    assert {key: cfg[key] for key in expected} == expected
    if cfg["CLUSTER_SIZE"] > 1:
        tiles = (m // 256) * (n // 256)
        assert cfg["NUM_PROGRAMS"] % 16 == 0 and tiles % cfg["NUM_PROGRAMS"] == 0


def test_mm_mxfp8_gfx1250_cluster_needs_even_tile_split():
    from triton.tlx.ops.kernels.mm_mxfp8.gfx1250 import _schedule

    # 96 tiles over 64 programs would give cluster members different tile counts.
    cfg = _schedule(2048, 3072, 1024, 64)
    assert cfg["NUM_PROGRAMS"] == 64 and cfg["CLUSTER_SIZE"] == 1
    assert _schedule(2048, 4096, 1024, 64)["CLUSTER_SIZE"] == 16


@requires_gfx1250
@pytest.mark.parametrize(
    "shape",
    [(256, 256, 256),  # two-buffer ring
     (512, 768, 384),  # three-buffer ring
     (512, 512, 512),  # dynamic four-buffer loop
     (512, 256, 768),  # static merged loop with overlapped first step
     (2048, 1024, 640),  # 4x4 multicast clusters
     ],
)
@pytest.mark.parametrize("sf_layout", ["natural", "cublas_blocked"])
def test_mm_mxfp8_gfx1250_public_numerics(shape, sf_layout):
    from triton.tlx.ops import mm_mxfp8

    inputs, scale_bytes = _inputs(*shape, sf_layout=sf_layout)
    out = torch.empty(shape[:2], dtype=torch.bfloat16, device="cuda")
    assert mm_mxfp8(*inputs, out=out, sf_layout=sf_layout, space="heuristic") is out
    torch.testing.assert_close(out.float(), _reference(inputs, scale_bytes), rtol=0.01, atol=0.5)


@requires_gfx1250
def test_mm_mxfp8_gfx1250_default_space_and_blocked_parity():
    from triton.tlx.ops import mm_mxfp8

    natural, _ = _inputs(512, 256, 768, sf_layout="natural")
    blocked, _ = _inputs(512, 256, 768, sf_layout="cublas_blocked")
    out = mm_mxfp8(*natural)
    assert out.dtype == torch.bfloat16 and out.shape == (512, 256)
    torch.testing.assert_close(out, mm_mxfp8(*natural, space="heuristic"), rtol=0, atol=0)
    torch.testing.assert_close(out, mm_mxfp8(*blocked, sf_layout="cublas_blocked"), rtol=0, atol=0)


@requires_gfx1250
@pytest.mark.parametrize("shape", [(384, 256, 256), (256, 128, 256), (256, 256, 128)])
def test_mm_mxfp8_gfx1250_rejects_unsupported_shapes(shape):
    from triton.tlx.ops import InvalidInput, mm_mxfp8

    inputs, _ = _inputs(*shape, sf_layout="natural")
    with pytest.raises(InvalidInput, match="divisible by 256 and K >= 256"):
        mm_mxfp8(*inputs)


@requires_gfx1250
def test_mm_mxfp8_gfx1250_rejects_misaligned_base():
    from triton.tlx.ops import InvalidInput, mm_mxfp8

    (a, a_scale, b, b_scale), _ = _inputs(256, 256, 256, sf_layout="natural")
    shifted = torch.empty(a.numel() + 8, dtype=a.dtype, device=a.device)[8:].view_as(a)
    shifted.copy_(a)
    with pytest.raises(InvalidInput):
        mm_mxfp8(shifted, a_scale, b, b_scale)


@requires_gfx1250
def test_mm_mxfp8_gfx1250_rejects_mfma_preshuffled_scales():
    from triton.tlx.ops import InvalidInput, mm_mxfp8

    (a, _, b, _), _ = _inputs(256, 256, 256, sf_layout="natural")
    # Sizes the public wrapper accepts for the gfx950-only packed layout.
    a_scale = torch.zeros(2 * 128 * 8, dtype=torch.uint8, device=a.device)
    b_scale = torch.zeros(3 * 128 * 8, dtype=torch.uint8, device=a.device)
    with pytest.raises(InvalidInput, match="sf_layout"):
        mm_mxfp8(a, a_scale, b, b_scale, sf_layout="mfma_preshuffled")
