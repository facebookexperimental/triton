"""gfx950 coverage for ``tlx.ops.mm_mxfp6``.

Catalog and validation coverage runs anywhere; kernel numerics require
a gfx950 (CDNA4) device.
"""

from __future__ import annotations

import pytest
import torch
from triton._internal_testing import is_hip_cdna4


def test_mm_mxfp6_gfx950_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, has_impl

    # Held until the gfx950 numerics fix lands: no catalog entry, so
    # MI350 callers get UnsupportedOp instead of silent wrong results.
    specs = [spec for spec in CATALOG if spec.op == "mm_mxfp6" and spec.arch == "gfx950"]
    assert len(specs) == 0
    assert not has_impl("mm_mxfp6", "gfx950")


def test_mm_mxfp6_gfx950_catalog_resolves_implementation():
    from triton.tlx.ops._catalog import UnsupportedOp, impl_for

    with pytest.raises(UnsupportedOp, match="no implementation for arch='gfx950'"):
        impl_for("mm_mxfp6", arch="gfx950")


def test_mm_mxfp6_gfx950_rejects_blocked_scales_and_bad_space():
    from triton.tlx.ops.kernels.mm_mxfp6.gfx950 import mm_mxfp6

    a = torch.zeros((128, 128), dtype=torch.uint8)
    b = torch.zeros((128, 128), dtype=torch.uint8)
    scale = torch.zeros((128, 4), dtype=torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="only supports sf_layout='natural'"):
        mm_mxfp6(a, scale, b, scale, sf_layout="cublas_blocked")
    with pytest.raises(ValueError, match="unsupported space"):
        mm_mxfp6(a, scale, b, scale, space="bogus")


def _packed_ones(rows, K, code, device):
    # Dense-pack `code` (decodes to 1.0) and pad the row to logical width.
    quads = torch.full((rows, K // 4, 4), code, dtype=torch.int64, device=device)
    v0, v1, v2, v3 = (quads[..., i] for i in range(4))
    triples = torch.stack([v0 | ((v1 & 3) << 6), ((v1 >> 2) & 15) | ((v2 & 15) << 4), ((v2 >> 4) & 3) | (v3 << 2)],
                          dim=-1).reshape(rows, -1)
    packed = torch.zeros((rows, K), dtype=torch.uint8, device=device)
    packed[:, :triples.shape[-1]] = triples.to(torch.uint8)
    return packed


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
@pytest.mark.parametrize("space", ["heuristic", "full"])
def test_mm_mxfp6_gfx950_ones(space):
    from triton.tlx import ops
    from triton.tlx.ops._catalog import UnsupportedOp

    device = "cuda"
    M = N = K = 128
    # E2M3 code 8 and E3M2 code 12 both decode to 1.0; E8M0 127 is 1.0.
    a = _packed_ones(M, K, 8, device)
    b = _packed_ones(N, K, 12, device)
    scales_a = torch.full((M, K // 32), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
    scales_b = torch.full((N, K // 32), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
    # Held until the gfx950 numerics fix lands (see catalog note).
    with pytest.raises(UnsupportedOp):
        ops.mm_mxfp6(a, scales_a, b, scales_b, space=space)
