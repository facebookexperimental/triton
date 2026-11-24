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

    specs = [spec for spec in CATALOG if spec.op == "mm_mxfp6" and spec.arch == "gfx950"]
    assert len(specs) == 1
    assert specs[0].impl == "triton.tlx.ops.kernels.mm_mxfp6.gfx950:mm_mxfp6"
    assert has_impl("mm_mxfp6", "gfx950")


def test_mm_mxfp6_gfx950_catalog_resolves_implementation():
    from triton.tlx.ops._catalog import impl_for

    impl, spec = impl_for("mm_mxfp6", arch="gfx950")
    assert impl.__name__ == "mm_mxfp6"
    assert spec.arch == "gfx950"


def test_mm_mxfp6_gfx950_rejects_blocked_scales_and_bad_space():
    from triton.tlx.ops.kernels.mm_mxfp6.gfx950 import mm_mxfp6

    a = torch.zeros((128, 128), dtype=torch.uint8)
    b = torch.zeros((128, 128), dtype=torch.uint8)
    scale = torch.zeros((128, 4), dtype=torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="only supports sf_layout='natural'"):
        mm_mxfp6(a, scale, b, scale, sf_layout="cublas_blocked")
    with pytest.raises(ValueError, match="unsupported space"):
        mm_mxfp6(a, scale, b, scale, space="bogus")


def _pack_fp6_dense(codes):
    rows, K = codes.shape
    quads = codes.reshape(rows, K // 4, 4).to(torch.int64)
    v0, v1, v2, v3 = (quads[..., i] for i in range(4))
    triples = torch.stack([v0 | ((v1 & 3) << 6), ((v1 >> 2) & 15) | ((v2 & 15) << 4), ((v2 >> 4) & 3) | (v3 << 2)],
                          dim=-1).reshape(rows, -1)
    packed = torch.zeros((rows, K), dtype=torch.uint8, device=codes.device)
    packed[:, :triples.shape[-1]] = triples.to(torch.uint8)
    return packed


def _packed_ones(rows, K, code, device):
    return _pack_fp6_dense(torch.full((rows, K), code, dtype=torch.uint8, device=device))


def _decode_fp6(codes, ebits, bias):
    mbits = 5 - ebits
    sign = (codes >> 5) & 1
    exponent = (codes >> mbits) & ((1 << ebits) - 1)
    mantissa = (codes & ((1 << mbits) - 1)).to(torch.float64) / 2**mbits
    normal = (1.0 + mantissa) * torch.exp2(exponent.to(torch.float64) - bias)
    denormal = mantissa * 2.0**(1 - bias)
    magnitude = torch.where(exponent == 0, denormal, normal)
    return torch.where(sign == 1, -magnitude, magnitude)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
@pytest.mark.parametrize("space", ["heuristic", "full"])
@pytest.mark.parametrize("a_format", ["e2m3", "e3m2"])
@pytest.mark.parametrize("b_format", ["e2m3", "e3m2"])
def test_mm_mxfp6_gfx950_ones(space, a_format, b_format):
    from triton.tlx.ops.kernels.mm_mxfp6.gfx950 import mm_mxfp6

    device = "cuda"
    M = N = K = 128
    # E2M3 code 8 and E3M2 code 12 both decode to 1.0; E8M0 127 is 1.0.
    a = _packed_ones(M, K, 8 if a_format == "e2m3" else 12, device)
    b = _packed_ones(N, K, 8 if b_format == "e2m3" else 12, device)
    scales_a = torch.full((M, K // 32), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
    scales_b = torch.full((N, K // 32), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
    out = mm_mxfp6(a, scales_a, b, scales_b, space=space, a_format=a_format, b_format=b_format)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), torch.full((M, N), float(K), device=device))


@pytest.mark.parametrize("split_k", [0, 3, 32])
def test_mm_mxfp6_gfx950_rejects_invalid_split_k(split_k):
    from triton.tlx.ops.kernels.mm_mxfp6.gfx950 import mm_mxfp6

    a = torch.zeros((128, 128), dtype=torch.uint8)
    scales = torch.zeros((128, 4), dtype=torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="split_k must"):
        mm_mxfp6(a, scales, a, scales, split_k=split_k)
    with pytest.raises(ValueError, match="not exceed"):
        mm_mxfp6(a, scales, a, scales, split_k=2)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
@pytest.mark.parametrize("M,N,K,split_k", [
    (256, 256, 192, 1),
    (256, 256, 512, 1),
    (256, 256, 640, 1),
    (256, 256, 192, 2),
    (256, 256, 640, 4),
    (256, 256, 896, 2),
    (256, 256, 6144, 4),
    (256, 256, 16128, 8),
    (128, 256, 640, 4),
    (256, 128, 896, 2),
])
@pytest.mark.parametrize("space", ["heuristic", "full"])
@pytest.mark.parametrize("a_format", ["e2m3", "e3m2"])
@pytest.mark.parametrize("b_format", ["e2m3", "e3m2"])
def test_mm_mxfp6_gfx950_split_k_random(M, N, K, split_k, space, a_format, b_format):
    from triton.tlx.ops.kernels.mm_mxfp6.gfx950 import mm_mxfp6

    if space == "full" and K % 128:
        pytest.skip("full configs require K=128 tiles")
    generator = torch.Generator(device="cuda").manual_seed(0)
    a_codes = torch.randint(0, 64, (M, K), dtype=torch.uint8, device="cuda", generator=generator)
    b_codes = torch.randint(0, 64, (N, K), dtype=torch.uint8, device="cuda", generator=generator)
    a, b = _pack_fp6_dense(a_codes), _pack_fp6_dense(b_codes)
    sa = torch.randint(124, 129, (M, K // 32), dtype=torch.uint8, device="cuda", generator=generator)
    sb = torch.randint(124, 129, (N, K // 32), dtype=torch.uint8, device="cuda", generator=generator)
    formats = {"e2m3": (2, 1), "e3m2": (3, 3)}
    reference_a = _decode_fp6(a_codes, *formats[a_format]) * torch.exp2(sa.double() - 127).repeat_interleave(32, dim=1)
    reference_b = _decode_fp6(b_codes, *formats[b_format]) * torch.exp2(sb.double() - 127).repeat_interleave(32, dim=1)
    reference = reference_a @ reference_b.T
    out = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    result = mm_mxfp6(a, sa.view(torch.float8_e8m0fnu), b, sb.view(torch.float8_e8m0fnu), out=out, space=space,
                      split_k=split_k, a_format=a_format, b_format=b_format)
    assert result is out
    # Native MFMA can lose small terms while aligning products with widely
    # differing exponents. Bound cancellation error by the sum of magnitudes.
    tolerance = 1e-1 + 1e-2 * reference.abs() + 2**-14 * (reference_a.abs() @ reference_b.abs().T)
    error = (result.double() - reference).abs()
    assert torch.all(error <= tolerance), f"maximum error/tolerance: {(error / tolerance).max().item()}"
    b.zero_()
    mm_mxfp6(a, sa.view(torch.float8_e8m0fnu), b, sb.view(torch.float8_e8m0fnu), out=out, space=space, split_k=split_k,
             a_format=a_format, b_format=b_format)
    assert torch.count_nonzero(out) == 0


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
@pytest.mark.parametrize("a_format", ["e2m3", "e3m2"])
@pytest.mark.parametrize("b_format", ["e2m3", "e3m2"])
@pytest.mark.parametrize("scale", [124, 127, 128])
def test_mm_mxfp6_gfx950_all_code_products(a_format, b_format, scale):
    from triton.tlx.ops.kernels.mm_mxfp6.gfx950 import mm_mxfp6

    M = K = 128
    codes = (torch.arange(M, device="cuda") % 64).to(torch.uint8)[:, None].expand(M, K).contiguous()
    a, b = _pack_fp6_dense(codes), _pack_fp6_dense(codes)
    scales = torch.full((M, K // 32), scale, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    formats = {"e2m3": (2, 1), "e3m2": (3, 3)}
    reference = (_decode_fp6(codes, *formats[a_format]) @ _decode_fp6(codes, *formats[b_format]).T) * 2**(2 *
                                                                                                          (scale - 127))
    result = mm_mxfp6(a, scales, b, scales, space="heuristic", a_format=a_format, b_format=b_format)
    torch.testing.assert_close(result, reference.to(torch.bfloat16), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
@pytest.mark.parametrize("a_format", ["e2m3", "e3m2"])
@pytest.mark.parametrize("b_format", ["e2m3", "e3m2"])
@pytest.mark.parametrize("mma", [16, 32])
@pytest.mark.parametrize("split_k", [1, 4])
def test_mm_mxfp6_gfx950_split_k_native_mfma(a_format, b_format, mma, split_k):
    from triton.tlx.ops.kernels.mm_mxfp6.gfx950 import _gemm_mxfp6_native_kernel, _reduce_split_k

    M = N = 128
    K = 640
    a = _packed_ones(M, K, 8 if a_format == "e2m3" else 12, "cuda")
    b = _packed_ones(N, K, 8 if b_format == "e2m3" else 12, "cuda")
    sa = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    sb = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    target = torch.empty((split_k, M, N), dtype=torch.float32, device="cuda") if split_k > 1 else out
    kernel = _gemm_mxfp6_native_kernel[(1, split_k)](a, b, target, sa, sb, M, N, K, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128,
                                                     BLOCK_SIZE_K=128, GROUP_SIZE_M=8, SPLIT_K=split_k,
                                                     A_FORMAT=a_format, B_FORMAT=b_format, num_warps=4,
                                                     matrix_instr_nonkdim=mma)
    instructions = [line for line in kernel.asm["amdgcn"].splitlines() if "v_mfma" in line]
    assert instructions
    # The transposed MFMA layout places B in cbsz and A in blgp.
    cbsz = 2 if b_format == "e2m3" else 3
    blgp = 2 if a_format == "e2m3" else 3
    shape = "16x16x128" if mma == 16 else "32x32x64"
    assert all(f"v_mfma_scale_f32_{shape}_f8f6f4" in line and f"cbsz:{cbsz}" in line and f"blgp:{blgp}" in line
               for line in instructions)
    if split_k > 1:
        _reduce_split_k[(16, )](target, out, M * N, split_k, 1024)
    torch.testing.assert_close(out, torch.full_like(out, K), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
def test_mm_mxfp6_gfx950_split_k_large_workspace():
    from triton.tlx.ops.kernels.mm_mxfp6.gfx950 import mm_mxfp6

    M, N, K = 32768, 8192, 2048
    a = _packed_ones(M, K, 8, "cuda")
    b = _packed_ones(N, K, 12, "cuda")
    sa = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    sb = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    out = mm_mxfp6(a, sa, b, sb, space="heuristic", split_k=16)
    assert torch.all(out == K).item()


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
@pytest.mark.parametrize("M, N", [(128, 128), (128, 256), (256, 128), (256, 256), (384, 384)])
def test_mm_mxfp6_gfx950_full_ones_preserves_output_guard(M, N):
    from triton.tlx import ops
    from triton.tlx.ops.kernels.mm_mxfp6.gfx950 import _kernel_full

    _kernel_full.cache.clear()
    device = "cuda"
    K = 128
    a = _packed_ones(M, K, 8, device)
    b = _packed_ones(N, K, 12, device)
    scales_a = torch.full((M, K // 32), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
    scales_b = torch.full((N, K // 32), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
    backing = torch.full((M + 256, N), -17, dtype=torch.bfloat16, device=device)
    out = backing[:M]
    assert ops.mm_mxfp6(a, scales_a, b, scales_b, out=out, space="full") is out
    torch.testing.assert_close(out.float(), torch.full((M, N), float(K), device=device))
    torch.testing.assert_close(backing[M:], torch.full_like(backing[M:], -17), rtol=0, atol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
@pytest.mark.parametrize("M, N, K", [
    (128, 128, 128),
    (128, 384, 256),
    (384, 128, 384),
    (384, 384, 1024),
    (1152, 128, 128),
    (13, 29, 64),
    (13, 29, 192),
    (13, 29, 320),
])
def test_mm_mxfp6_gfx950_all_configs_partial_tiles(M, N, K):
    import triton
    from triton.tlx.ops.kernels.mm_mxfp6.gfx950 import _full_configs, _gemm_mxfp6_native_kernel

    generator = torch.Generator().manual_seed(0)
    a_codes = torch.randint(16, (M, K), dtype=torch.uint8, generator=generator)
    b_codes = torch.randint(16, (N, K), dtype=torch.uint8, generator=generator)
    a_codes |= torch.randint(2, (M, K), dtype=torch.uint8, generator=generator) << 5
    b_codes |= torch.randint(2, (N, K), dtype=torch.uint8, generator=generator) << 5
    a_scale_bytes = torch.randint(126, 129, (M, K // 32), dtype=torch.uint8, generator=generator)
    b_scale_bytes = torch.randint(126, 129, (N, K // 32), dtype=torch.uint8, generator=generator)
    a_ref = _decode_fp6(a_codes, 2, 1) * torch.exp2(a_scale_bytes.to(torch.float64) - 127).repeat_interleave(32, dim=1)
    b_ref = _decode_fp6(b_codes, 3, 3) * torch.exp2(b_scale_bytes.to(torch.float64) - 127).repeat_interleave(32, dim=1)
    expected = (a_ref @ b_ref.T).to(device="cuda", dtype=torch.bfloat16)
    a = _pack_fp6_dense(a_codes).cuda()
    b = _pack_fp6_dense(b_codes).cuda()
    a_scale = a_scale_bytes.cuda().view(torch.float8_e8m0fnu)
    b_scale = b_scale_bytes.cuda().view(torch.float8_e8m0fnu)

    for config in _full_configs():
        backing = torch.full((M + 256, N), -17, dtype=torch.bfloat16, device="cuda")
        out = backing[:M]
        grid = (triton.cdiv(M, config.kwargs["BLOCK_SIZE_M"]) * triton.cdiv(N, config.kwargs["BLOCK_SIZE_N"]), )
        _gemm_mxfp6_native_kernel[grid](a, b, out, a_scale, b_scale, M, N, K, **config.all_kwargs())
        torch.testing.assert_close(out, expected, rtol=0, atol=0, msg=lambda message: f"{config}\n{message}")
        torch.testing.assert_close(backing[M:], torch.full_like(backing[M:], -17), rtol=0, atol=0,
                                   msg=lambda message: f"{config}\n{message}")


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
@pytest.mark.parametrize("K", [64, 128, 192, 256, 320, 384, 1024])
def test_mm_mxfp6_gfx950_native_ones(K):
    from triton.tlx.ops.kernels.mm_mxfp6.gfx950 import mm_mxfp6

    device = "cuda"
    M = N = 128
    a = _packed_ones(M, K, 8, device)
    b = _packed_ones(N, K, 12, device)
    scales_a = torch.full((M, K // 32), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
    scales_b = torch.full((N, K // 32), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
    out = mm_mxfp6(a, scales_a, b, scales_b, space="heuristic")
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), torch.full((M, N), float(K), device=device))
