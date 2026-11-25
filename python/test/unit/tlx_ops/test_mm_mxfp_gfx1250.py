"""gfx1250 public-API coverage for ``tlx.ops.mm_mxfp``."""

from __future__ import annotations

import pytest
import torch

from triton._internal_testing import is_hip_cdna5, swizzle_scale_to_5d

requires_gfx1250 = pytest.mark.skipif(not is_hip_cdna5(), reason="Requires CDNA5")


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


def test_mm_mxfp_gfx1250_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, has_impl

    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("mm_mxfp", "gfx1250")]
    assert len(specs) == 1
    assert specs[0].impl == "triton.tlx.ops.kernels.mm_mxfp.gfx1250:mm_mxfp"
    assert specs[0].dtypes == frozenset({"float8_e4m3fn"})
    assert not specs[0].supports_backward
    assert has_impl("mm_mxfp", "gfx1250")


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
def test_mm_mxfp_gfx1250_schedule(m, n, k, expected):
    from triton.tlx.ops.kernels.mm_mxfp.gfx1250 import _schedule

    cfg = _schedule(m, n, k, 256)
    assert {key: cfg[key] for key in expected} == expected
    if cfg["CLUSTER_SIZE"] > 1:
        tiles = (m // 256) * (n // 256)
        assert cfg["NUM_PROGRAMS"] % 16 == 0 and tiles % cfg["NUM_PROGRAMS"] == 0


def test_mm_mxfp_gfx1250_cluster_needs_even_tile_split():
    from triton.tlx.ops.kernels.mm_mxfp.gfx1250 import _schedule

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
def test_mm_mxfp_gfx1250_public_numerics(shape, sf_layout):
    from triton.tlx.ops import mm_mxfp

    inputs, scale_bytes = _inputs(*shape, sf_layout=sf_layout)
    out = torch.empty(shape[:2], dtype=torch.bfloat16, device="cuda")
    assert mm_mxfp(*inputs, out=out, sf_layout=sf_layout, space="heuristic") is out
    torch.testing.assert_close(out.float(), _reference(inputs, scale_bytes), rtol=0.01, atol=0.5)


@requires_gfx1250
def test_mm_mxfp_gfx1250_default_space_and_blocked_parity():
    from triton.tlx.ops import mm_mxfp

    natural, _ = _inputs(512, 256, 768, sf_layout="natural")
    blocked, _ = _inputs(512, 256, 768, sf_layout="cublas_blocked")
    out = mm_mxfp(*natural)
    assert out.dtype == torch.bfloat16 and out.shape == (512, 256)
    torch.testing.assert_close(out, mm_mxfp(*natural, space="heuristic"), rtol=0, atol=0)
    torch.testing.assert_close(out, mm_mxfp(*blocked, sf_layout="cublas_blocked"), rtol=0, atol=0)


@requires_gfx1250
@pytest.mark.parametrize("shape", [(384, 256, 256), (256, 128, 256), (256, 256, 128)])
def test_mm_mxfp_gfx1250_rejects_unsupported_shapes(shape):
    from triton.tlx.ops import InvalidInput, mm_mxfp

    inputs, _ = _inputs(*shape, sf_layout="natural")
    with pytest.raises(InvalidInput, match="divisible by 256 and K >= 256"):
        mm_mxfp(*inputs)


@requires_gfx1250
def test_mm_mxfp_gfx1250_rejects_misaligned_base():
    from triton.tlx.ops import InvalidInput, mm_mxfp

    (a, a_scale, b, b_scale), _ = _inputs(256, 256, 256, sf_layout="natural")
    shifted = torch.empty(a.numel() + 8, dtype=a.dtype, device=a.device)[8:].view_as(a)
    shifted.copy_(a)
    with pytest.raises(InvalidInput):
        mm_mxfp(shifted, a_scale, b, b_scale)


@requires_gfx1250
def test_mm_mxfp_gfx1250_rejects_mfma_preshuffled_scales():
    from triton.tlx.ops import InvalidInput, mm_mxfp

    (a, _, b, _), _ = _inputs(256, 256, 256, sf_layout="natural")
    # Sizes the public wrapper accepts for the gfx950-only packed layout.
    a_scale = torch.zeros(2 * 128 * 8, dtype=torch.uint8, device=a.device)
    b_scale = torch.zeros(3 * 128 * 8, dtype=torch.uint8, device=a.device)
    with pytest.raises(InvalidInput, match="sf_layout"):
        mm_mxfp(a, a_scale, b, b_scale, sf_layout="mfma_preshuffled")


@pytest.mark.parametrize("formats", [{"a_dtype": "e2m1"}, {"b_dtype": "e2m1"}, {"a_dtype": "e5m2"}])
def test_mm_mxfp_rejects_unsupported_formats(formats):
    from triton.tlx.ops import UnsupportedOp, mm_mxfp

    a = torch.empty((256, 256), dtype=torch.float8_e4m3fn)
    scale = torch.empty((256, 8), dtype=torch.float8_e8m0fnu)
    with pytest.raises(UnsupportedOp, match="a_dtype='e4m3' and b_dtype='e4m3'"):
        mm_mxfp(a, scale, a, scale, **formats)


# The non-persistent TDM-pipelined kernel and the persistent kernel's launcher
# options, through the config-dictionary ``matmul`` entry point.
@requires_gfx1250
@pytest.mark.parametrize("transpose_b", [False])
def test_mxgemm_pipelined_matmul(transpose_b):
    from triton.tlx.ops.kernels.mm_mxfp import gfx1250
    from triton.tools.mxfp import MXScaleTensor

    torch.manual_seed(0)
    m = n = 256
    k = 512
    a = torch.randint(20, 40, (m, k), dtype=torch.uint8).view(torch.float8_e5m2)
    b = torch.randint(20, 40, (k, n), dtype=torch.uint8).view(torch.float8_e5m2)
    a_scale = MXScaleTensor(size=(m, k // 32)).random(high=32.0).data
    b_scale = MXScaleTensor(size=(n, k // 32)).random(high=32.0).data
    ref = gfx1250.torch_gemm_mxfp(a, b, a_scale, b_scale, 32, m, n, k)

    b_d = b.T.contiguous() if transpose_b else b.contiguous()
    out = gfx1250.matmul(a.cuda(), b_d.cuda(),
                         gfx1250.pack_scale(a_scale).cuda(),
                         gfx1250.pack_scale(b_scale).cuda(), config={"TRANSPOSE_B": transpose_b})
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=2e-2)


def _check_persistent(dtype_b, buffers, k_iters, cross_tile_prefetch, block_k, m=768, n=512, extra_config=None):
    from triton.tlx.ops.kernels.mm_mxfp import gfx1250
    from triton.tools.mxfp import MXFP4Tensor

    # Three output tiles per program, including a short final M group. Exercise
    # both an empty steady loop and a K count that rotates the ring's phase.
    torch.manual_seed(123)
    k = block_k * k_iters
    a = (torch.randn(m, k) * 0.5).to(torch.float8_e4m3fn)
    if dtype_b == "e2m1":
        b_mx = MXFP4Tensor(size=(n, k)).random()
        b, b_ref = b_mx.to_packed_tensor(dim=1), b_mx.to(torch.float32)
    else:
        b = (torch.randn(n, k) * 0.5).to(torch.float8_e4m3fn)
        b_ref = b.float()
    a_scale = torch.randint(125, 130, (m, k // 32), dtype=torch.uint8)
    b_scale = torch.randint(125, 130, (n, k // 32), dtype=torch.uint8)
    a_ref = a.float() * gfx1250.fp8e8m0_to_float32(a_scale).repeat_interleave(32, dim=1)
    b_ref *= gfx1250.fp8e8m0_to_float32(b_scale).repeat_interleave(32, dim=1)
    ref = a_ref @ b_ref.T
    config = dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=block_k, NUM_BUFFERS=buffers, DTYPE_A="e4m3", DTYPE_B=dtype_b,
                  TRANSPOSE_B=True, SCHEDULE="sliceMNK", TDM_FUSION="partial", PERSISTENT=True, NUM_PROGRAMS=2,
                  GROUP_SIZE_M=2, CROSS_TILE_PREFETCH=cross_tile_prefetch)
    config.update(extra_config or {})
    out = gfx1250.matmul(a.cuda(), b.cuda(),
                         gfx1250.pack_scale(a_scale).cuda(),
                         gfx1250.pack_scale(b_scale).cuda(), config=config)
    torch.testing.assert_close(out.cpu(), ref, atol=2e-3, rtol=1e-4)


@requires_gfx1250
@pytest.mark.parametrize("dtype_b,block_k,buffers,k_iters,cross_tile_prefetch", [
    ("e4m3", 128, 2, 3, True),
    ("e4m3", 128, 3, 4, True),
    ("e4m3", 128, 3, 5, True),
    ("e4m3", 128, 3, 8, False),
    ("e4m3", 128, 4, 4, True),
    ("e4m3", 128, 4, 5, True),
    ("e4m3", 256, 2, 2, False),
    ("e4m3", 256, 2, 3, False),
    ("e2m1", 256, 2, 3, False),
    ("e2m1", 256, 3, 3, True),
    ("e2m1", 256, 3, 4, False),
])
def test_mxgemm_persistent_short_k_stages(dtype_b, block_k, buffers, k_iters, cross_tile_prefetch):
    _check_persistent(dtype_b, buffers, k_iters, cross_tile_prefetch, block_k)


@requires_gfx1250
@pytest.mark.parametrize("dtype_b,block_k", [("e4m3", 128), ("e2m1", 256)])
@pytest.mark.parametrize("mode", ["none", "balanced", "chunked"])
def test_mxgemm_persistent_xcd_remap(dtype_b, block_k, mode):
    # Seventeen programs exercise both uneven XCD ranges and chunked tails;
    # nine M tiles with group-M=8 also exercise the final short M group.
    _check_persistent(dtype_b, 3, 5, block_k == 128, block_k, m=2304, n=768,
                      extra_config=dict(NUM_PROGRAMS=17, GROUP_SIZE_M=8, XCD_REMAP=mode))


@requires_gfx1250
@pytest.mark.parametrize("dtype_b,block_k", [("e4m3", 128), ("e2m1", 256)])
@pytest.mark.parametrize("k_iters", [3, 5])
@pytest.mark.parametrize("cluster_size,multicast,group_m", [(2, True, 8), (4, True, 4), (4, True, 8), (4, False, 4)])
@pytest.mark.parametrize("remap_mode", ["none", "chunked"])
def test_mxgemm_persistent_multicast(dtype_b, block_k, k_iters, cluster_size, multicast, group_m, remap_mode):
    # Distinct data/scales catch wrong recipients. Two tiles/program exercise
    # cross-tile prefetch (BK128) and reusing A's LDS for C (BK256). K=3 has no
    # steady loop; K=5 rotates the three-slot ring at the tile boundary.
    _check_persistent(
        dtype_b, 3, k_iters, block_k == 128, block_k, m=2048, n=1024,
        extra_config=dict(NUM_PROGRAMS=16, GROUP_SIZE_M=group_m, XCD_REMAP=remap_mode, CLUSTER_SIZE=cluster_size,
                          CLUSTER_MULTICAST=multicast))


@requires_gfx1250
@pytest.mark.parametrize("dtype_b,block_k", [("e4m3", 128), ("e2m1", 256)])
@pytest.mark.parametrize("k_iters", [3, 10])
@pytest.mark.parametrize("interval,remap_mode,group_m,multicast", [
    (0, "none", 8, True),
    (0, "chunked", 4, True),
    (0, "none", 8, False),
    (4, "none", 4, True),
    (3, "chunked", 4, True),
])
def test_mxgemm_persistent_cluster_barrier_interval(dtype_b, block_k, k_iters, interval, remap_mode, group_m,
                                                    multicast):
    _check_persistent(
        dtype_b, 3, k_iters, block_k == 128, block_k, m=2048, n=1024,
        extra_config=dict(NUM_PROGRAMS=16, GROUP_SIZE_M=group_m, XCD_REMAP=remap_mode, CLUSTER_SIZE=4,
                          CLUSTER_MULTICAST=multicast, CLUSTER_BARRIER_INTERVAL=interval))
