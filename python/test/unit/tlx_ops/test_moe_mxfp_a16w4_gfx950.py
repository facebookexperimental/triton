"""gfx950 MXFP4 A16W4 routed MoE kernel coverage (BF16 activations, MXFP4 weights)."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.moe_mxfp.gfx950_a16w4 import moe_mxfp4_a16w4, pick_config

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
