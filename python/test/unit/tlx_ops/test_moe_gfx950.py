"""Public ``tlx.ops.moe`` coverage for gfx950."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import InvalidInput, UnsupportedBackward
from triton.tlx.ops import moe as tlx_moe
from triton.tlx.ops._catalog import CATALOG
from triton.tlx.ops.kernels.moe._shapes import CORRECTNESS_SHAPES, inputs, label

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")

SMALL = dict(K=256, I=256, H=384, E=8, E_total=32, topk=4)


def _weights(dims, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    K, I, H, E = dims["K"], dims["I"], dims["H"], dims["E"]
    w13 = (torch.randn(E, 2 * I, K, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    norm = (torch.randn(H, device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    return w13, w2, norm


def _routing(T, dims, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, dims["K"], device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, dims["E_total"], device="cuda", generator=gen).argsort(dim=1)[:, :dims["topk"]]
    weights = torch.rand(T, dims["topk"], device="cuda", generator=gen)
    return x, weights / weights.sum(dim=1, keepdim=True), ids.to(torch.int32)


def _reference(x, weights, ids, w13, w2, expert_start, norm, eps, gain):
    """Torch reference with the op's BF16 rounding points."""
    T = x.shape[0]
    E, two_i, _ = w13.shape
    I = two_i // 2
    out = torch.zeros(T, w2.shape[1], device=x.device, dtype=torch.float32)
    if norm is not None:
        scale = (norm.float() + gain).to(norm.dtype).float()
    local = ids.long() - expert_start
    for e in range(E):
        tok, slot = torch.nonzero(local == e, as_tuple=True)
        if tok.numel() == 0:
            continue
        h = x[tok].float() @ w13[e].float().T
        inter = (torch.nn.functional.silu(h[:, :I]) * h[:, I:]).to(torch.bfloat16)
        y = (inter.float() @ w2[e].float().T).to(torch.bfloat16).float()
        wgt = weights[tok, slot].float()
        if norm is not None:
            y = (y * scale).to(torch.bfloat16).float()
            wgt = wgt * torch.rsqrt(y.square().mean(dim=1) + eps)
        out.index_add_(0, tok, wgt[:, None] * y)
    return out.to(torch.bfloat16)


def _check(got, ref):
    torch.testing.assert_close(got.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_catalog_entry():
    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("moe", "gfx950")]
    assert len(specs) == 1
    spec, = specs
    assert spec.dtypes == frozenset({"bfloat16"})
    assert not spec.supports_backward
    assert spec.accepts(dict(K=8192, I=4096, H=8192))
    assert not spec.accepts(dict(K=8192, I=4096, H=8000))
    assert not spec.accepts(dict(K=0, I=4096, H=8192))
    for shape in CORRECTNESS_SHAPES:
        assert spec.accepts(dict(K=shape.K, I=shape.I, H=shape.H)), label(shape)


def test_moe_is_exported():
    import triton.tlx.ops as ops
    assert "moe" in ops.__all__


@requires_gfx950
@pytest.mark.parametrize("T", [1, 4, 11, 16, 27, 35, 64, 300, 4090])
@pytest.mark.parametrize("with_norm", [True, False])
@pytest.mark.parametrize("expert_start", [0, 8])
def test_matches_reference_small(T, with_norm, expert_start):
    w13, w2, norm = _weights(SMALL)
    x, weights, ids = _routing(T, SMALL, seed=T)
    norm = norm if with_norm else None
    got = tlx_moe(x, weights, ids, w13, w2, expert_start=expert_start, post_norm_weight=norm, post_norm_eps=1e-6,
                  post_norm_gain=1.0)
    _check(got, _reference(x, weights, ids, w13, w2, expert_start, norm, 1e-6, 1.0))


# Every registered shape, including the Muse Spark Open decode and prefill suites the nightly perf run times.
@requires_gfx950
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=label)
def test_matches_reference_registered_shapes(shape):
    x, weights, ids, w13, w2, norm = inputs(shape, seed=shape.tokens)
    got = tlx_moe(x, weights, ids, w13, w2, expert_start=shape.expert_start, post_norm_weight=norm, post_norm_eps=1e-6,
                  post_norm_gain=1.0)
    _check(got, _reference(x, weights, ids, w13, w2, shape.expert_start, norm, 1e-6, 1.0))


@requires_gfx950
@pytest.mark.parametrize("T", [3, 100])
def test_no_local_routes_is_zero(T):
    w13, w2, norm = _weights(SMALL)
    x, weights, ids = _routing(T, SMALL, seed=7)
    ids = (ids % SMALL["E"]) + SMALL["E"]
    got = tlx_moe(x, weights, ids, w13, w2, expert_start=0, post_norm_weight=norm)
    assert torch.count_nonzero(got) == 0


@requires_gfx950
@pytest.mark.parametrize("T", [5, 200])
def test_all_routes_on_one_expert(T):
    w13, w2, _ = _weights(SMALL)
    x, weights, _ = _routing(T, SMALL, seed=8)
    ids = torch.full((T, SMALL["topk"]), 3, device="cuda", dtype=torch.int64)
    got = tlx_moe(x, weights, ids, w13, w2)
    _check(got, _reference(x, weights, ids, w13, w2, 0, None, 0.0, 0.0))


# 2400 / 16000 / 32000 slots: the BLOCK_N=256 register tiers and the quadrant tier, with I and H not multiples of 256.
@requires_gfx950
@pytest.mark.parametrize("T", [300, 2000, 4000])
def test_n_not_multiple_of_256(T):
    dims = dict(K=256, I=384, H=384, E=8, E_total=32, topk=8)
    w13, w2, norm = _weights(dims)
    x, weights, ids = _routing(T, dims, seed=20 + T)
    got = tlx_moe(x, weights, ids, w13, w2, post_norm_weight=norm)
    _check(got, _reference(x, weights, ids, w13, w2, 0, norm, 1e-6, 0.0))


@requires_gfx950
def test_strided_weights_match_contiguous():
    # 32000 slots with N a multiple of 256 would take the quadrant path for contiguous weights.
    dims = dict(K=256, I=256, H=512, E=8, E_total=32, topk=8)
    w13, w2, norm = _weights(dims)
    x, weights, ids = _routing(4000, dims, seed=30)
    w13_s = torch.zeros(*w13.shape[:2], 2 * w13.shape[2], device="cuda", dtype=w13.dtype)[..., ::2]
    w2_s = torch.zeros(*w2.shape[:2], 2 * w2.shape[2], device="cuda", dtype=w2.dtype)[..., ::2]
    w13_s.copy_(w13)
    w2_s.copy_(w2)
    norm_s = torch.zeros(2 * norm.numel(), device="cuda", dtype=norm.dtype)[::2]
    norm_s.copy_(norm)
    assert w13_s.stride(2) == 2 and norm_s.stride(0) == 2
    got = tlx_moe(x, weights, ids, w13_s, w2_s, post_norm_weight=norm_s)
    assert torch.equal(got, tlx_moe(x, weights, ids, w13, w2, post_norm_weight=norm))


# The register engine reads x through its strides; the quad engine (4000 tokens here) copies x only when its rows
# are not unit-stride and 16-byte aligned.
@requires_gfx950
@pytest.mark.parametrize("T", [3, 300, 4000])
@pytest.mark.parametrize("layout", ["row_pad8", "row_pad4", "col_stride2", "transposed"])
def test_strided_x_matches_contiguous(T, layout):
    dims = dict(K=256, I=256, H=512, E=8, E_total=32, topk=8)
    w13, w2, norm = _weights(dims)
    x, weights, ids = _routing(T, dims, seed=31)
    K = dims["K"]
    if layout == "row_pad8":
        x_s = torch.zeros(T, K + 8, device="cuda", dtype=x.dtype)[:, :K]
    elif layout == "row_pad4":
        x_s = torch.zeros(T, K + 4, device="cuda", dtype=x.dtype)[:, :K]
    elif layout == "col_stride2":
        x_s = torch.zeros(T, 2 * K, device="cuda", dtype=x.dtype)[:, ::2]
    else:
        x_s = torch.zeros(K, T, device="cuda", dtype=x.dtype).T
    x_s.copy_(x)
    assert not x_s.is_contiguous()
    got = tlx_moe(x_s, weights, ids, w13, w2, post_norm_weight=norm)
    assert torch.equal(got, tlx_moe(x, weights, ids, w13, w2, post_norm_weight=norm))


@requires_gfx950
@pytest.mark.parametrize("T", [3, 300])
def test_int64_ids_outside_int32_are_not_local(T):
    w13, w2, _ = _weights(SMALL)
    x, weights, ids = _routing(T, SMALL, seed=40)
    ids = ids.long()
    non_local = ids.clone()
    non_local[:, :2] = SMALL["E_total"] - 1
    ids[:, 0] = 2**32 + 3
    ids[:, 1] = -(2**32) + 5
    got = tlx_moe(x, weights, ids, w13, w2)
    assert torch.equal(got, tlx_moe(x, weights, non_local, w13, w2))


# Largest finalize tiles that fit registers: 2**16 elements, as topk-heavy and H-heavy shapes.
@requires_gfx950
@pytest.mark.parametrize("topk,H", [(16, 4096), (2, 32768)])
@pytest.mark.parametrize("T", [3, 40])
def test_largest_finalize_tile(topk, H, T):
    dims = dict(K=128, I=128, H=H, E=8, E_total=32, topk=topk)
    w13, w2, norm = _weights(dims)
    x, weights, ids = _routing(T, dims, seed=50 + T)
    got = tlx_moe(x, weights, ids, w13, w2, post_norm_weight=norm)
    _check(got, _reference(x, weights, ids, w13, w2, 0, norm, 1e-6, 0.0))


@requires_gfx950
def test_empty_batch():
    w13, w2, _ = _weights(SMALL)
    x, weights, ids = _routing(0, SMALL, seed=9)
    assert tlx_moe(x, weights, ids, w13, w2).shape == (0, SMALL["H"])


@requires_gfx950
@pytest.mark.parametrize("T", [2, 40])
def test_graph_capture_replays_with_new_routing(T):
    w13, w2, norm = _weights(SMALL)
    x, weights, ids = _routing(T, SMALL, seed=10)
    out = torch.empty(T, SMALL["H"], device="cuda", dtype=torch.bfloat16)

    def run():
        tlx_moe(x, weights, ids, w13, w2, expert_start=8, post_norm_weight=norm, out=out)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for seed in (11, 12, 13):
        new_x, new_weights, new_ids = _routing(T, SMALL, seed=seed)
        x.copy_(new_x)
        weights.copy_(new_weights)
        ids.copy_(new_ids)
        graph.replay()
        torch.cuda.synchronize()
        _check(out, _reference(new_x, new_weights, new_ids, w13, w2, 8, norm, 1e-6, 0.0))


@requires_gfx950
def test_out_is_filled_and_returned():
    w13, w2, _ = _weights(SMALL)
    x, weights, ids = _routing(6, SMALL, seed=14)
    out = torch.empty(6, SMALL["H"], device="cuda", dtype=torch.bfloat16)
    assert tlx_moe(x, weights, ids, w13, w2, out=out) is out
    _check(out, _reference(x, weights, ids, w13, w2, 0, None, 0.0, 0.0))


@requires_gfx950
def test_rejects_invalid_inputs():
    w13, w2, norm = _weights(SMALL)
    x, weights, ids = _routing(4, SMALL, seed=15)
    with pytest.raises(InvalidInput):
        tlx_moe(x.half(), weights, ids, w13.half(), w2.half())
    with pytest.raises(InvalidInput):
        tlx_moe(x, weights, ids, w13, w2, space="full")
    with pytest.raises(InvalidInput):
        tlx_moe(x, weights, ids.float(), w13, w2)
    with pytest.raises(InvalidInput):
        tlx_moe(x, weights[:, :2], ids, w13, w2)
    with pytest.raises(InvalidInput):
        tlx_moe(x, weights, ids, w13, w2[:, :, :128])
    with pytest.raises(InvalidInput):
        tlx_moe(x, weights, ids, w13, w2, post_norm_weight=norm[:-1])
    with pytest.raises(InvalidInput):
        tlx_moe(x, weights, ids, w13, w2, expert_start=-1)
    with pytest.raises(InvalidInput):
        tlx_moe(x, weights, ids, w13, w2, out=w2.view(-1)[:4 * SMALL["H"]].view(4, SMALL["H"]))
    odd = dict(SMALL, H=320)
    w13_odd, w2_odd, _ = _weights(odd)
    with pytest.raises(InvalidInput):
        tlx_moe(x, weights, ids, w13_odd, w2_odd)
    with pytest.raises(InvalidInput):
        tlx_moe(x[:, :0], weights, ids, w13[:, :, :0], w2)
    wide = dict(SMALL, K=128, I=128, H=8192)
    w13_wide, w2_wide, _ = _weights(wide)
    wide_ids = ids.repeat(1, 4)  # next_pow2(16) * next_pow2(8192) = 2**17 finalize elements
    with pytest.raises(InvalidInput):
        tlx_moe(x[:, :128], wide_ids.float(), wide_ids, w13_wide, w2_wide)
    many = dict(SMALL, K=128, I=128, H=128, E=513)
    w13_many, w2_many, _ = _weights(many)
    with pytest.raises(InvalidInput):
        tlx_moe(x[:, :128], weights, ids, w13_many, w2_many)
    with pytest.raises(UnsupportedBackward):
        tlx_moe(x.requires_grad_(), weights, ids, w13, w2)
