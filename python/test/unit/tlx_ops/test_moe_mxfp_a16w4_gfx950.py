"""gfx950 MXFP4 A16W4 routed MoE kernel coverage (BF16 activations, MXFP4 weights)."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.moe.gfx950 import _align
from triton.tlx.ops.kernels.moe_mxfp import gfx950_a16w4_decode as decode
from triton.tlx.ops.kernels.moe_mxfp.gfx950_a16w4 import (DECODE_MAX_TOKENS, _launch, moe_mxfp4_a16w4, pick_config,
                                                          pick_down_config)

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _dequant(q, s):
    lut = torch.tensor(_E2M1, device=q.device)
    v = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], -1).flatten(-2)
    return v * torch.exp2(s.float() - 127).repeat_interleave(32, -1)


def _random_weights(E, N, K, gen):
    """Random MXFP4 weights in the natural layout: any byte is a valid E2M1 pair; scales keep values O(0.1)."""
    q = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda", generator=gen)
    s = torch.randint(121, 125, (E, N, K // 32), dtype=torch.uint8, device="cuda", generator=gen)
    return q, s


def _reference(x, wts, ids, w13, s13, w2, s2, start, norm, eps, gain):
    """Op rounding points: BF16 x, fp32 GEMM, BF16 SiLU*up, fp32 GEMM, BF16 expert output."""
    T = x.shape[0]
    E, two_i, _ = w13.shape
    I = two_i // 2
    out = torch.zeros(T, w2.shape[1], device=x.device)
    for e in range(E):
        tt, jj = torch.nonzero(ids == e + start, as_tuple=True)
        if not len(tt):
            continue
        gu = x[tt].float() @ _dequant(w13[e], s13[e]).T
        h = (torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]).to(torch.bfloat16)
        y = (h.float() @ _dequant(w2[e], s2[e]).T).to(torch.bfloat16).float()
        if norm is not None:
            y = (y * (norm.float() + gain).to(norm.dtype).float()).to(torch.bfloat16).float()
            y = y * torch.rsqrt((y * y).mean(-1, keepdim=True) + eps)
        out.index_add_(0, tt, y * wts[tt, jj, None])
    return out.to(torch.bfloat16)


@requires_gfx950
@pytest.mark.parametrize("T", [1, 5, 40, 300, 600, 1024, 1100])
@pytest.mark.parametrize("config", ["auto", "m32", "m128"])
@pytest.mark.parametrize("start,use_norm", [(0, False), (0, True), (16, True)])
def test_moe_mxfp4_a16w4_matches_reference(T, config, start, use_norm):
    K, I, H, E, E_total, topk = 1024, 1024, 256, 8, 32, 8
    gen = torch.Generator(device="cuda").manual_seed(0)
    w13, s13 = _random_weights(E, 2 * I, K, gen)
    w2, s2 = _random_weights(E, H, I, gen)
    norm = (torch.randn(H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16) if use_norm else None
    x = torch.randn(T, K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, E_total, device="cuda", generator=gen).argsort(dim=1)[:, :topk].to(torch.int32)
    wts = torch.softmax(torch.randn(T, topk, device="cuda", generator=gen), -1)
    cfg = {
        "auto": None,
        "m32": dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, GROUP_M=1, num_warps=4, num_stages=2),
        "m128": dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=4, num_warps=8, num_stages=2),
    }[config]
    got = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, expert_start=start, post_norm_weight=norm, post_norm_eps=1e-8,
                          post_norm_gain=1.0, cfg=cfg)
    ref = _reference(x, wts, ids, w13, s13, w2, s2, start, norm, 1e-8, 1.0)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert rel < 4e-3, rel


@requires_gfx950
def test_moe_mxfp4_a16w4_no_local_routes():
    K, I, H, E, topk = 1024, 1024, 256, 8, 8
    gen = torch.Generator(device="cuda").manual_seed(1)
    w13, s13 = _random_weights(E, 2 * I, K, gen)
    w2, s2 = _random_weights(E, H, I, gen)
    x = torch.randn(9, K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = (torch.arange(9 * topk, device="cuda", dtype=torch.int32).reshape(9, topk) % 8) + 100
    wts = torch.full((9, topk), 1.0 / topk, device="cuda")
    got = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, expert_start=0)
    assert torch.equal(got, torch.zeros_like(got))


_K, _I, _H, _E, _E_TOTAL, _TOPK = 1024, 1024, 256, 8, 32, 8


def _problem(T, gen, use_norm=False):
    w13, s13 = _random_weights(_E, 2 * _I, _K, gen)
    w2, s2 = _random_weights(_E, _H, _I, gen)
    norm = (torch.randn(_H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16) if use_norm else None
    x = torch.randn(T, _K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, _E_TOTAL, device="cuda", generator=gen).argsort(dim=1)[:, :_TOPK].to(torch.int32)
    wts = torch.softmax(torch.randn(T, _TOPK, device="cuda", generator=gen), -1)
    return x, wts, ids, (w13, s13, w2, s2), norm


def _run(x, wts, ids, weights, norm, cfg=None):
    return moe_mxfp4_a16w4(x, wts, ids, *weights, post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0,
                           cfg=cfg)


def _assert_close(got, x, wts, ids, weights, norm):
    ref = _reference(x, wts, ids, *weights, 0, norm, 1e-8, 1.0)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert rel < 4e-3, rel


@requires_gfx950
@pytest.mark.parametrize("num_valid", [64, 2048, 4096, 8192, 8193])
def test_moe_mxfp4_a16w4_every_config_tier(num_valid):
    """Each pick_config tier (keyed by its upper cutoff) on the same routing, so no tier goes untested."""
    gen = torch.Generator(device="cuda").manual_seed(2)
    x, wts, ids, weights, norm = _problem(300, gen, use_norm=True)
    _assert_close(_run(x, wts, ids, weights, norm, pick_config(num_valid)), x, wts, ids, weights, norm)


@requires_gfx950
@pytest.mark.parametrize("T", [5, 1100])
def test_moe_mxfp4_a16w4_strided_activations(T):
    """A view with a non-unit column stride reads its logical columns (both the _align and _align_quad paths)."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    x, wts, ids, weights, norm = _problem(T, gen)
    xs = torch.empty(T, 2 * _K, device="cuda", dtype=torch.bfloat16)[:, ::2]
    xs.copy_(x)
    assert xs.stride(1) == 2
    _assert_close(_run(xs, wts, ids, weights, norm), x, wts, ids, weights, norm)


@requires_gfx950
@pytest.mark.parametrize("T", [1, 40, 1100])
def test_moe_mxfp4_a16w4_graph_replay(T):
    """Capture once, then replay with routings whose local-row counts differ (num_post changes): uniform, every slot
    on two local experts, and no local slots."""
    gen = torch.Generator(device="cuda").manual_seed(4)
    x, wts, ids, weights, norm = _problem(T, gen)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(x, wts, ids, weights, norm)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _run(x, wts, ids, weights, norm)

    r = torch.arange(T, device="cuda")
    skewed = torch.stack([r % 2] + [_E + (r + j) % (_E_TOTAL - _E) for j in range(1, _TOPK)], 1).to(torch.int32)
    for new_ids in (_problem(T, gen)[2], skewed, ids + _E_TOTAL):
        ids.copy_(new_ids)
        wts.copy_(torch.softmax(torch.randn(T, _TOPK, device="cuda", generator=gen), -1))
        graph.replay()
        torch.cuda.synchronize()
        if (new_ids >= _E).all():
            assert torch.equal(out, torch.zeros_like(out))
        else:
            _assert_close(out, x, wts, ids, weights, norm)


@requires_gfx950
@pytest.mark.parametrize("T", [1, 2, 7, 16, DECODE_MAX_TOKENS])
@pytest.mark.parametrize("start,use_norm", [(0, False), (16, True)])
def test_moe_mxfp4_a16w4_decode_engine_matches_dot_scaled(T, start, use_norm):
    """Up to DECODE_MAX_TOKENS the op runs the register-weight decode engine; it must give the bits of the
    tl.dot_scaled engine (forced here through ``cfg``) on the same inputs."""
    gen = torch.Generator(device="cuda").manual_seed(5)
    x, wts, ids, weights, norm = _problem(T, gen, use_norm)
    kw = dict(expert_start=start, post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0)
    got = moe_mxfp4_a16w4(x, wts, ids, *weights, **kw)
    want = moe_mxfp4_a16w4(x, wts, ids, *weights, cfg=pick_config(T * _TOPK), **kw)
    assert torch.equal(got, want)


@requires_gfx950
@pytest.mark.parametrize("T", [1, 16])
def test_moe_mxfp4_a16w4_decode_engine_model_shape(T):
    """The decode engine at the served model's per-rank shape (H = I = K = 4096), bitwise against tl.dot_scaled."""
    K = I = H = 4096
    E, E_total = 4, 32
    gen = torch.Generator(device="cuda").manual_seed(6)
    w13, s13 = _random_weights(E, 2 * I, K, gen)
    w2, s2 = _random_weights(E, H, I, gen)
    norm = (torch.randn(H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    x = torch.randn(T, K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, E_total, device="cuda", generator=gen).argsort(dim=1)[:, :_TOPK].to(torch.int32)
    ids[:, 0] = torch.arange(T, device="cuda") % E
    wts = torch.softmax(torch.randn(T, _TOPK, device="cuda", generator=gen), -1)
    kw = dict(post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0)
    got = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, **kw)
    want = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, cfg=pick_config(T * _TOPK), **kw)
    assert torch.equal(got, want)
    _assert_close(got, x, wts, ids, (w13, s13, w2, s2), norm)


@requires_gfx950
@pytest.mark.parametrize("T", [2, 4, 8])
def test_moe_mxfp4_a16w4_decode_engine_expert_tp_shape(T):
    """The decode engine on an expert-TP8 slice of the served model (I = 512, every expert local), whose configs
    differ from the full-width ones, bitwise against tl.dot_scaled."""
    K = H = 4096
    I, E = 512, 32
    assert decode.pick_config(T, I) != decode.pick_config(T)
    gen = torch.Generator(device="cuda").manual_seed(10)
    w13, s13 = _random_weights(E, 2 * I, K, gen)
    w2, s2 = _random_weights(E, H, I, gen)
    norm = (torch.randn(H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    x = torch.randn(T, K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, E, device="cuda", generator=gen).argsort(dim=1)[:, :_TOPK].to(torch.int32)
    wts = torch.softmax(torch.randn(T, _TOPK, device="cuda", generator=gen), -1)
    kw = dict(post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0)
    got = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, **kw)
    want = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, cfg=pick_config(T * _TOPK), **kw)
    assert torch.equal(got, want)
    _assert_close(got, x, wts, ids, (w13, s13, w2, s2), norm)


def _padded(t, experts=False, rows=0, cols=0):
    """``t``'s values as a view into a larger zero tensor: every other expert, and/or ``rows`` extra rows and ``cols``
    extra bytes per row after the real ones."""
    E, N, C = t.shape
    big = torch.zeros(2 * E if experts else E, N + rows, C + cols, dtype=t.dtype, device=t.device)
    view = big[::2 if experts else 1, :N, :C]
    view.copy_(t)
    return view


def _strided_views(weights, kind):
    """The same weight values as views whose strides are not those of a contiguous tensor: every other expert of a
    larger tensor, rows of wider rows in extra-tall experts (as an expert-TP slice of a full weight would be), or a w2
    row stride that is not a multiple of four bytes."""
    w13, s13, w2, s2 = weights
    if kind == "experts":
        views = tuple(_padded(t, experts=True) for t in weights)
    elif kind == "rows":
        views = (_padded(w13, rows=64, cols=128), _padded(s13, rows=64, cols=8), _padded(w2,
                                                                                         cols=128), _padded(s2, cols=8))
    else:
        views = (w13, s13, _padded(w2, cols=2), s2)
        assert views[2].stride(1) % 4 == 2
    for v, t in zip(views, weights):
        assert torch.equal(v, t)
    return views


def _spy_decode(monkeypatch):
    """Records the weights of every decode-engine launch."""
    calls = []
    launch = decode.launch

    def spy(*args, **kwargs):
        calls.append(args[1])
        return launch(*args, **kwargs)

    monkeypatch.setattr(decode, "launch", spy)
    return calls


@requires_gfx950
@pytest.mark.parametrize("T", [1, 5, 16, DECODE_MAX_TOKENS])
@pytest.mark.parametrize("kind", ["experts", "rows", "unaligned"])
def test_moe_mxfp4_a16w4_strided_weights(T, kind, monkeypatch):
    """Weight and scale views with expert and row strides of their own (the op accepts any layout that is contiguous
    along K) give the bits of contiguous weights. The decode engine reads them through their strides; a w2 row stride
    that is not a whole number of 32-bit words falls back to the tl.dot_scaled engine."""
    calls = _spy_decode(monkeypatch)
    gen = torch.Generator(device="cuda").manual_seed(11)
    x, wts, ids, weights, norm = _problem(T, gen, use_norm=True)
    # Every token hits a local expert other than 0, whose offset would not depend on the expert stride.
    e = torch.arange(T, device="cuda", dtype=torch.int32) % (_E - 1) + 1
    ids[:, 0] = torch.where((ids == e[:, None]).any(1), ids[:, 0], e)
    views = _strided_views(weights, kind)
    assert not all(v.is_contiguous() for v in views)
    got = _run(x, wts, ids, views, norm)
    assert len(calls) == (0 if kind == "unaligned" else 2)
    assert torch.equal(got, _run(x, wts, ids, weights, norm, pick_config(T * _TOPK)))
    _assert_close(got, x, wts, ids, weights, norm)


@requires_gfx950
@pytest.mark.parametrize("T", [1, 4, 16])
def test_moe_mxfp4_a16w4_strided_weights_expert_tp_shape(T, monkeypatch):
    """An expert-TP8 slice (I = 512, every expert local) taken as a view of wider weights, at the slice tiles."""
    calls = _spy_decode(monkeypatch)
    K = H = 4096
    I, E = 512, 32
    gen = torch.Generator(device="cuda").manual_seed(12)
    w13, s13 = _random_weights(E, 2 * I, K, gen)
    w2, s2 = _random_weights(E, H, I, gen)
    weights = (w13, s13, w2, s2)
    views = _strided_views(weights, "rows")
    norm = (torch.randn(H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    x = torch.randn(T, K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, E, device="cuda", generator=gen).argsort(dim=1)[:, :_TOPK].to(torch.int32)
    wts = torch.softmax(torch.randn(T, _TOPK, device="cuda", generator=gen), -1)
    got = _run(x, wts, ids, views, norm)
    assert len(calls) == 2
    assert torch.equal(got, _run(x, wts, ids, weights, norm, pick_config(T * _TOPK)))
    _assert_close(got, x, wts, ids, weights, norm)


@requires_gfx950
def test_moe_mxfp4_a16w4_prefill_down_tile_same_bits():
    """Past 8192 slots the down GEMM runs 256-wide N tiles; the output must equal the single-config run."""
    gen = torch.Generator(device="cuda").manual_seed(9)
    x, wts, ids, weights, norm = _problem(1100, gen, use_norm=True)
    assert pick_down_config(1100 * _TOPK, pick_config(1100 * _TOPK)) != pick_config(1100 * _TOPK)
    got = _run(x, wts, ids, weights, norm)
    assert torch.equal(got, _run(x, wts, ids, weights, norm, pick_config(1100 * _TOPK)))
    _assert_close(got, x, wts, ids, weights, norm)


@requires_gfx950
def test_moe_mxfp4_a16w4_decode_engine_gate_up_bits():
    """The gate/up stage of both engines on 4M intermediate values (every token on all eight experts), compared
    before the down GEMM can hide a difference: SiLU's 1-ulp rounding choices show up only every few 1e6 values."""
    T, E, I, K, topk = DECODE_MAX_TOKENS, 8, 8192, 512, 8
    gen = torch.Generator(device="cuda").manual_seed(8)
    w13, s13 = _random_weights(E, 2 * I, K, gen)
    x = torch.randn(T, K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, E, device="cuda", generator=gen).argsort(dim=1).to(torch.int32)
    num_valid = T * topk
    sorted_ids, expert_ids, num_post = _align(ids, E, 0, 16)
    ours = torch.empty(num_valid, I, device="cuda", dtype=torch.bfloat16)
    c1, _ = decode.pick_config(T)
    decode.launch(x, w13, s13, ours, sorted_ids, expert_ids, num_post, num_valid, I, K, topk, True, c1, 16)
    theirs = torch.empty(sorted_ids.numel(), I, device="cuda", dtype=torch.bfloat16)
    _launch(x, w13, s13, theirs, sorted_ids, expert_ids, num_post, num_valid, I, K, topk, True,
            dict(pick_config(num_valid), BLOCK_M=16))
    rows = torch.nonzero(sorted_ids < num_valid).flatten()
    assert rows.numel() == num_valid
    assert torch.equal(ours[sorted_ids[rows].long()], theirs[rows])


@requires_gfx950
@pytest.mark.parametrize("T", [DECODE_MAX_TOKENS, DECODE_MAX_TOKENS + 1])
def test_moe_mxfp4_a16w4_engine_switch(T):
    """Both sides of the token-count switch against the reference."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    x, wts, ids, weights, norm = _problem(T, gen, use_norm=True)
    _assert_close(_run(x, wts, ids, weights, norm), x, wts, ids, weights, norm)


def test_moe_mxfp4_a16w4_rejects_bad_inputs():
    E, I, H, K, T = 2, 256, 256, 512, 4
    x = torch.zeros(T, K, dtype=torch.bfloat16)
    ids = torch.zeros(T, 2, dtype=torch.int32)
    wts = torch.zeros(T, 2)
    w13, s13 = torch.zeros(E, 2 * I, K // 2, dtype=torch.uint8), torch.zeros(E, 2 * I, K // 32, dtype=torch.uint8)
    w2, s2 = torch.zeros(E, H, I // 2, dtype=torch.uint8), torch.zeros(E, H, I // 32, dtype=torch.uint8)
    with pytest.raises(ValueError, match="x must be bf16"):
        moe_mxfp4_a16w4(x.float(), wts, ids, w13, s13, w2, s2)
    with pytest.raises(ValueError, match="topk_ids"):
        moe_mxfp4_a16w4(x, wts[:, :1], ids, w13, s13, w2, s2)
    with pytest.raises(ValueError, match="w13 must be uint8"):
        moe_mxfp4_a16w4(x, wts, ids, w13.float(), s13, w2, s2)
    with pytest.raises(ValueError, match="contiguous along its last"):
        moe_mxfp4_a16w4(x, wts, ids, w13.transpose(1, 2).contiguous().transpose(1, 2), s13, w2, s2)
    with pytest.raises(ValueError, match="do not match K"):
        moe_mxfp4_a16w4(x, wts, ids, w13, s13[:, :, :8], w2, s2)
    with pytest.raises(ValueError, match="do not match E"):
        moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2[:, :, :64], s2)
    with pytest.raises(ValueError, match="not a multiple of BLOCK_N"):
        moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, cfg=dict(BLOCK_M=16, BLOCK_N=512, BLOCK_K=256, GROUP_M=1,
                                                                num_warps=4, num_stages=2))
