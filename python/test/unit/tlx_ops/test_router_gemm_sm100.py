"""Public ``tlx.ops.router_gemm`` coverage for sm100."""

import pytest
import torch

from triton._internal_testing import is_blackwell
from triton.tlx.ops import InvalidInput, UnsupportedBackward, router_gemm
from triton.tlx.ops._catalog import CATALOG
from triton.tlx.ops.kernels.router_gemm import sm100 as impl
from triton.tlx.ops.kernels.router_gemm._shapes import (CORRECTNESS_SHAPES, RouterGemmShape, inputs, label, reference)

requires_sm100 = pytest.mark.skipif(not is_blackwell(), reason="Requires sm100")
# Token counts across the GEMV and every tile bucket, with and without split K and with partial row tiles.
TOKENS = (1, 2, 5, 9, 12, 13, 16, 17, 33, 64, 100, 129, 300, 513, 1000, 2049, 4100, 9000)


def _check(got, x, weight, steps=64):
    """Within ``steps`` fp32 rounding steps of the sum of absolute products of the exact result."""
    ref = reference(x, weight)
    bound = steps * 2.0**-24 * (x.double().abs() @ weight.double().abs().T)
    assert got.dtype == torch.float32 and got.shape == ref.shape
    assert ((got.double() - ref).abs() <= bound).all()


def _exact_inputs(T, E, K, seed=0):
    """Rows with four +-1 entries and weights that are multiples of 2**-22 below 0.5: every partial sum of the products
    is an fp32 number, so an exact-product fp32 accumulation returns the fp64 result bit for bit, while dropping the
    weight bits past the 16th (keeping two bf16 parts) does not."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.zeros(T, K, device="cuda")
    cols = torch.rand(T, K, device="cuda", generator=gen).argsort(dim=1)[:, :4]
    signs = torch.randint(0, 2, (T, 4), device="cuda", generator=gen).float() * 2 - 1
    x.scatter_(1, cols, signs)
    m = torch.randint(-(1 << 21) + 1, 1 << 21, (E, K), device="cuda", generator=gen)
    return x.to(torch.bfloat16), m.double().mul(2.0**-22).float()


def test_catalog_entry():
    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("router_gemm", "sm100")]
    assert len(specs) == 1
    spec, = specs
    assert spec.dtypes == frozenset({"bfloat16"})
    assert not spec.supports_backward
    assert spec.accepts({"E": 256, "K": 8192}) and spec.accepts({"E": 16, "K": 5120})
    assert not spec.accepts({"E": 250, "K": 8192}) and not spec.accepts({"E": 256, "K": 8000})


def test_kernel_choice():
    assert impl.GEMV_MAX_TOKENS == 12
    for T in (13, 16, 64, 1024, 16384):
        block_m, block_n, block_k, splits, _, _ = impl.mfma_config(T, 256, 8192)
        assert 256 % block_n == 0 and 8192 % (splits * block_k) == 0
    # K = 5120 keeps a split count whose partition is a whole number of K blocks.
    for T in (9, 100, 5000):
        _, _, block_k, splits, _, _ = impl.mfma_config(T, 256, 5120)
        assert 5120 % (splits * block_k) == 0
    for T in range(1, impl.GEMV_MAX_TOKENS + 1):
        block_n, block_k, _ = impl.gemv_config(T, 256, 5120)
        assert 256 % block_n == 0 and 5120 % block_k == 0


@requires_sm100
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=label)
def test_registered_shapes(shape):
    x, weight = inputs(shape)
    # The tensor cores truncate as they accumulate, so the error grows with the K each program covers; the configs cap
    # it at 2048, which stays within 16 steps.
    _check(router_gemm(x, weight), x, weight, steps=16)


@requires_sm100
@pytest.mark.parametrize("T", TOKENS)
@pytest.mark.parametrize("E,K", [(256, 8192), (128, 5120), (16, 1024)])
def test_products_are_exact(T, E, K):
    x, weight = _exact_inputs(T, E, K, seed=T)
    got = router_gemm(x, weight)
    assert torch.equal(got.double(), reference(x, weight))


@requires_sm100
@pytest.mark.parametrize("T", TOKENS)
def test_token_counts(T):
    x, weight = inputs(RouterGemmShape(T, 256, 8192), seed=T)
    _check(router_gemm(x, weight), x, weight)


@requires_sm100
def test_top8_selection_matches_fp64():
    x, weight = inputs(RouterGemmShape(4096, 256, 8192), seed=7)
    bias = torch.randn(256, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)) * 0.05
    ref = (reference(x, weight) + bias.double()).topk(8, dim=1).indices.sort(dim=1).values
    for got in (router_gemm(x, weight), router_gemm(x[:12], weight)):
        ids = (got + bias).topk(8, dim=1).indices.sort(dim=1).values
        assert (ids == ref[:got.shape[0]]).all(dim=1).float().mean() >= 0.999


@requires_sm100
@pytest.mark.parametrize("T", [3, 40, 600])
def test_row_strided_x_and_out(T):
    x, weight = inputs(RouterGemmShape(T, 256, 8192), seed=T)
    wide = torch.zeros(T, 8192 + 64, device="cuda", dtype=torch.bfloat16)
    wide[:, 32:32 + 8192] = x
    out = torch.full((T, 256), float("nan"), device="cuda")
    got = router_gemm(wide[:, 32:32 + 8192], weight, out=out)
    assert got.data_ptr() == out.data_ptr()
    assert torch.equal(got, router_gemm(x, weight))


@requires_sm100
def test_64bit_row_offsets():
    K = 8192
    T = (1 << 31) // K + 2
    x = torch.zeros(T, K, device="cuda", dtype=torch.bfloat16)
    x[-1] = 1.0
    weight = torch.zeros(256, K, device="cuda")
    weight[:, 0] = torch.arange(256, device="cuda", dtype=torch.float32)
    got = router_gemm(x, weight)
    assert torch.equal(got[-1], weight[:, 0]) and not got[0].any()
    for rows in (1, 5, 600):
        tail = x[-rows:]
        assert torch.equal(router_gemm(tail, weight)[-1], weight[:, 0])


@requires_sm100
def test_zero_tokens():
    x = torch.empty(0, 8192, device="cuda", dtype=torch.bfloat16)
    assert router_gemm(x, torch.zeros(256, 8192, device="cuda")).shape == (0, 256)


@requires_sm100
@pytest.mark.parametrize("T", [4, 64, 2048])
def test_graph_capture_replays_with_new_inputs(T):
    x, weight = inputs(RouterGemmShape(T, 256, 8192))
    static = x.clone()
    router_gemm(static, weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = router_gemm(static, weight)
    static.copy_(inputs(RouterGemmShape(T, 256, 8192), seed=3)[0])
    weight.mul_(0.5)
    graph.replay()
    _check(got, static, weight)


@requires_sm100
def test_torch_compile():
    x, weight = inputs(RouterGemmShape(33, 256, 8192))
    compiled = torch.compile(lambda a, b: router_gemm(a, b))
    assert torch.equal(compiled(x, weight), router_gemm(x, weight))


@requires_sm100
def test_invalid_inputs():
    x, weight = inputs(RouterGemmShape(4, 256, 8192))
    bad = (
        (x.float(), weight),
        (x, weight.bfloat16()),
        (x[0], weight),
        (x, weight[:, :4096]),
        (x, weight.t().contiguous().t()),
        (x.t().contiguous().t(), weight),
        (x[:, :8000], weight[:, :8000].contiguous()),
        (x, weight[:250].contiguous()),
    )
    for args in bad:
        with pytest.raises(InvalidInput):
            router_gemm(*args)
    with pytest.raises(InvalidInput):
        router_gemm(x, weight, out=torch.empty(4, 256, device="cuda", dtype=torch.float16))
    with pytest.raises(InvalidInput):
        router_gemm(x, weight, out=torch.empty(256, 4, device="cuda").t())
    storage = torch.zeros(256 * 8192, device="cuda")
    with pytest.raises(InvalidInput):
        router_gemm(x, storage.view(256, 8192), out=storage[:4 * 256].view(4, 256))
    with pytest.raises(InvalidInput):
        router_gemm(x, weight.cpu())
    with pytest.raises(UnsupportedBackward):
        router_gemm(x, weight, out=torch.empty(4, 256, device="cuda", requires_grad=True))
    with pytest.raises(UnsupportedBackward):
        router_gemm(x, weight.requires_grad_())
