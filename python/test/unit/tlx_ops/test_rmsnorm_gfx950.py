"""Public ``tlx.ops.rmsnorm`` and ``tlx.ops.norm_fwd`` coverage for gfx950."""

import pytest
import torch
import torch.nn.functional as F

from triton._internal_testing import is_hip_cdna4
from triton.tlx import ops
from triton.tlx.ops import InvalidInput, UnsupportedBackward, UnsupportedOp
from triton.tlx.ops._catalog import CATALOG
from triton.tlx.ops.kernels.norm import _shapes as norm_shapes
from triton.tlx.ops.kernels.norm import gfx950 as impl

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
RMS_SHAPES = tuple(s for s in norm_shapes.CORRECTNESS_SHAPES
                   if (s.family, s.norm, s.direction) == ("plain", "rms", "fwd"))


def _inputs(tokens, features=4096, seed=0):
    """BF16 ``x`` and a weight around one."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(tokens, features, device="cuda", generator=gen) * 2).to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(features, device="cuda", generator=gen)).to(torch.bfloat16)
    return x, weight


def _reference(x, weight, eps):
    """The kernel's math in FP32: ``x * rsqrt(mean(x * x) + eps) * weight``, rounded once to the input dtype."""
    xf = x.float()
    rstd = torch.rsqrt(xf.square().mean(1) + eps)
    y = xf * rstd[:, None]
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype), rstd


def _check(got, x, weight, eps):
    y, residual_out, rstd = got
    ref_y, ref_rstd = _reference(x, weight, eps)
    assert residual_out is None
    assert y.dtype == x.dtype and y.shape == x.shape and y.is_contiguous()
    torch.testing.assert_close(rstd, ref_rstd, atol=0, rtol=1e-5)
    # Only the summation order differs from the reference. In FP32 that moves a whole row by an ulp of rstd; a 16-bit
    # output rounds it away almost everywhere.
    torch.testing.assert_close(y.float(), ref_y.float(), atol=1e-6, rtol=1e-2 if x.dtype != torch.float32 else 1e-5)
    if x.dtype != torch.float32:
        assert (y != ref_y).float().mean() < 1e-2


@pytest.mark.parametrize("op", ["rmsnorm", "norm_fwd"])
def test_catalog_entry(op):
    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == (op, "gfx950")]
    assert len(specs) == 1
    spec, = specs
    assert spec.impl == f"triton.tlx.ops.kernels.norm.gfx950:{op}"
    assert spec.dtypes == frozenset({"float16", "bfloat16", "float32"})
    assert not spec.supports_backward
    assert op in ops.__all__
    assert {s.arch for s in CATALOG if s.op == op and s.variant == "row_norm"} == {"sm90", "sm100", "gfx950"}


@requires_gfx950
@pytest.mark.parametrize("shape", RMS_SHAPES, ids=norm_shapes.label)
def test_matches_torch_registered_shapes(shape):
    x, w = _inputs(shape.m, shape.n)
    got = ops.rmsnorm(x, w, eps=1e-5)
    _check(got, x, w, 1e-5)
    ref = F.rms_norm(x, (shape.n, ), w, 1e-5)
    torch.testing.assert_close(got[0].float(), ref.float(), atol=2e-2, rtol=1e-2)
    assert (got[0] != ref).float().mean() < 1e-3


@requires_gfx950
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", [1, 7, 128, 1000, 4097, 16384])
@pytest.mark.parametrize("weight_dtype", ["same", "fp32", None])
def test_dtypes_widths_and_weights(dtype, N, weight_dtype):
    gen = torch.Generator(device="cuda").manual_seed(N)
    x = (torch.randn(37, N, device="cuda", generator=gen) * 3).to(DTYPES[dtype])
    w = None
    if weight_dtype is not None:
        w_dtype = x.dtype if weight_dtype == "same" else torch.float32
        w = (1 + 0.1 * torch.randn(N, device="cuda", generator=gen)).to(w_dtype)
    _check(ops.rmsnorm(x, w, eps=1e-6), x, w, 1e-6)


def test_tlx_kernel_choice():
    assert impl.use_tlx_kernel(4096, torch.bfloat16) and impl.use_tlx_kernel(4096, torch.float16)
    assert not impl.use_tlx_kernel(4096, torch.float32) and not impl.use_tlx_kernel(8192, torch.bfloat16)
    assert not impl.use_tlx_kernel(2048, torch.bfloat16) and not impl.use_tlx_kernel(4097, torch.bfloat16)


@requires_gfx950
@pytest.mark.parametrize("N", impl.TLX_FEATURES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("weight_dtype", ["same", "fp32", None])
@pytest.mark.parametrize("T", [1, 37, 3000])
def test_tlx_kernel_is_bitwise_equal_to_the_8_wave_plain_kernel(N, dtype, weight_dtype, T):
    gen = torch.Generator(device="cuda").manual_seed(N + T)
    x = (torch.randn(T, N, device="cuda", generator=gen) * 3).to(DTYPES[dtype])
    w = None
    if weight_dtype is not None:
        w = (
            1 +
            0.1 * torch.randn(N, device="cuda", generator=gen)).to(x.dtype if weight_dtype == "same" else torch.float32)
    y, _, rstd = ops.rmsnorm(x, w, eps=1e-6)
    plain_y, plain_rstd = torch.empty_like(x), torch.empty(T, device="cuda")
    impl._rmsnorm_kernel[(T, )](x, w, plain_y, plain_rstd, x.stride(0), 1e-6, N, N, w is not None, num_warps=8)
    assert torch.equal(y, plain_y) and torch.equal(rstd, plain_rstd)


@requires_gfx950
@pytest.mark.parametrize("dtype", DTYPES)
def test_row_strided_input(dtype):
    base = torch.randn(70, 4096 + 64, device="cuda").to(DTYPES[dtype])
    x = base[:, :4096]
    w = torch.randn(4096, device="cuda").to(DTYPES[dtype])
    _check(ops.rmsnorm(x, w, eps=1e-6), x.contiguous(), w, 1e-6)


@requires_gfx950
def test_rows_do_not_depend_on_the_batch():
    x, w = _inputs(5000)
    y, _, rstd = ops.rmsnorm(x, w)
    for lo, hi in ((0, 1), (17, 18), (100, 4500), (4999, 5000)):
        ys, _, rs = ops.rmsnorm(x[lo:hi], w)
        assert torch.equal(ys, y[lo:hi]) and torch.equal(rs, rstd[lo:hi])


@requires_gfx950
def test_zero_rows_and_eps():
    x = torch.zeros(3, 4096, device="cuda", dtype=torch.bfloat16)
    x[1] = 1.0
    y, _, rstd = ops.rmsnorm(x, None, eps=1e-5)
    assert torch.isfinite(y.float()).all() and (y[0] == 0).all()
    torch.testing.assert_close(rstd[0], torch.tensor(1e-5, device="cuda").rsqrt(), atol=0, rtol=1e-6)
    torch.testing.assert_close(y[1].float(), torch.full((4096, ), 1 / (1 + 1e-5)**0.5, device="cuda"), atol=4e-3,
                               rtol=0)
    y, _, _ = ops.rmsnorm(x[1:2], None, eps=0.0)
    assert (y == 1).all()


@requires_gfx950
@pytest.mark.parametrize("N", [4096, 16384])
def test_64bit_row_offsets(N):
    """Row offsets past 2**31 elements, in both the strided input and the contiguous output."""
    M = (1 << 31) // N + 8
    x = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    x[-8:] = torch.randn(8, N, device="cuda").to(torch.bfloat16)
    x[:-8] = 1
    y, _, rstd = ops.rmsnorm(x, None)
    tail = x[-8:].clone()
    _check((y[-8:], None, rstd[-8:]), tail, None, 1e-5)
    assert torch.equal(y[-8:], ops.rmsnorm(tail, None)[0])


@requires_gfx950
def test_empty_batch():
    x = torch.empty(0, 4096, device="cuda", dtype=torch.bfloat16)
    y, residual_out, rstd = ops.rmsnorm(x, torch.ones(4096, device="cuda", dtype=torch.bfloat16))
    assert y.shape == (0, 4096) and rstd.shape == (0, ) and residual_out is None


@requires_gfx950
@pytest.mark.parametrize("T", [9, 2000])
def test_graph_capture_replays_with_new_inputs(T):
    x, w = _inputs(T)
    static = x.clone()
    ops.rmsnorm(static, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y, _, rstd = ops.rmsnorm(static, w)
    static.copy_(_inputs(T, seed=3)[0])
    graph.replay()
    _check((y, None, rstd), static, w, 1e-5)


@requires_gfx950
def test_torch_compile():
    x, w = _inputs(33)
    compiled = torch.compile(lambda a, b: ops.rmsnorm(a, b, eps=1e-5)[0])
    assert torch.equal(compiled(x, w), ops.rmsnorm(x, w, eps=1e-5)[0])


@requires_gfx950
def test_unsupported_features_and_invalid_inputs():
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    for kwargs in (dict(residual=x.clone()), dict(bias=w.clone()), dict(post_norm_addend=x.clone())):
        with pytest.raises(UnsupportedOp):
            ops.rmsnorm(x, w, **kwargs)
    with pytest.raises(UnsupportedOp):
        ops.rmsnorm(torch.randn(2, impl.MAX_FEATURES + 1, device="cuda", dtype=torch.bfloat16), None)
    for args in ((x[0], w), (x.t(), torch.ones(4, device="cuda", dtype=torch.bfloat16)), (x, w[:128]),
                 (x, torch.ones(512, device="cuda", dtype=torch.bfloat16)[::2]), (x[:, :0], None)):
        with pytest.raises(InvalidInput):
            ops.rmsnorm(*args)
    with pytest.raises(InvalidInput):
        ops.rmsnorm(x, w, eps=-1.0)
    with pytest.raises(UnsupportedBackward):
        ops.rmsnorm(x.requires_grad_(), w)


@requires_gfx950
@pytest.mark.parametrize("N", [1000, 4096])
@pytest.mark.parametrize("weight", [True, False])
def test_norm_fwd_is_rmsnorm(N, weight):
    x, w = _inputs(37, N)
    w = w if weight else None
    y, residual_out, rstd = ops.rmsnorm(x, w, eps=1e-6)
    out, residual, mean, stats = ops.norm_fwd(x, w, norm="rms", eps=1e-6)
    assert torch.equal(out, y) and torch.equal(stats, rstd)
    assert residual is None and mean is None
    out, _, _, stats = ops.norm_fwd(x, w, norm="rms", eps=1e-6, store_stats=False, out_dtype=x.dtype)
    assert torch.equal(out, y) and stats is None


@requires_gfx950
def test_norm_fwd_unsupported_features():
    x, w = _inputs(4, 256)
    with pytest.raises(UnsupportedOp):
        ops.norm_fwd(x, w)
    for kwargs in (dict(residual=x.clone()), dict(bias=w.clone()), dict(post_norm_addend=x.clone()),
                   dict(out=torch.empty_like(x)), dict(rstd_out=torch.empty(4, device="cuda")),
                   dict(out_dtype=torch.float32), dict(return_residual=True), dict(residual_dtype=torch.float32)):
        with pytest.raises(UnsupportedOp):
            ops.norm_fwd(x, w, norm="rms", **kwargs)
