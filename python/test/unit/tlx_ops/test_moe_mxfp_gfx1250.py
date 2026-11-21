"""gfx1250 MXFP4 A16W4 routed MoE coverage (BF16 activations, MXFP4 weights) at the ``moe_mxfp`` focus shapes."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna5
from triton.tlx.ops.kernels.moe_mxfp._shapes import CORRECTNESS_SHAPES, label
from triton.tlx.ops.kernels.moe_mxfp.gfx1250 import moe_mxfp4_a16w4, pick_config

requires_gfx1250 = pytest.mark.skipif(not is_hip_cdna5(), reason="Requires CDNA5")

_SHAPE = CORRECTNESS_SHAPES[0]
_K, _I, _H, _E, _E_TOTAL, _TOPK = (_SHAPE.K, _SHAPE.I, _SHAPE.H, _SHAPE.local_experts, _SHAPE.num_experts, _SHAPE.topk)
assert all((s.K, s.I, s.H, s.local_experts, s.num_experts, s.topk) == (_K, _I, _H, _E, _E_TOTAL, _TOPK)
           for s in CORRECTNESS_SHAPES)
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _dequant(q, s):
    lut = torch.tensor(_E2M1, device=q.device)
    v = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], -1).flatten(-2)
    return v * torch.exp2(s.float() - 127).repeat_interleave(32, -1)


def _random_weights(E, N, K, gen):
    """Random MXFP4 weights in the natural layout: any byte is a valid E2M1 pair; scales keep values O(0.1)."""
    q = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda", generator=gen)
    s = torch.randint(118, 122, (E, N, K // 32), dtype=torch.uint8, device="cuda", generator=gen)
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


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator(device="cuda").manual_seed(0)
    w13, s13 = _random_weights(_E, 2 * _I, _K, gen)
    w2, s2 = _random_weights(_E, _H, _I, gen)
    norm = (torch.randn(_H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    return w13, s13, w2, s2, norm


def _routing(T, gen):
    x = torch.randn(T, _K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, _E_TOTAL, device="cuda", generator=gen).argsort(dim=1)[:, :_TOPK].to(torch.int32)
    wts = torch.softmax(torch.randn(T, _TOPK, device="cuda", generator=gen), -1)
    return x, wts, ids


def _check(weights, T, start, use_norm, seed):
    w13, s13, w2, s2, norm = weights
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x, wts, ids = _routing(T, gen)
    norm = norm if use_norm else None
    got = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, expert_start=start, post_norm_weight=norm, post_norm_eps=1e-8,
                          post_norm_gain=1.0)
    ref = _reference(x, wts, ids, w13, s13, w2, s2, start, norm, 1e-8, 1.0)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert rel < 4e-3, rel


@requires_gfx1250
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=label)
@pytest.mark.parametrize("use_norm", [True, False])
def test_moe_mxfp4_a16w4_gfx1250_matches_reference(weights, shape, use_norm):
    _check(weights, shape.tokens, shape.expert_start, use_norm, shape.tokens)


@requires_gfx1250
@pytest.mark.parametrize("start", [0, 64])
def test_moe_mxfp4_a16w4_gfx1250_expert_start(weights, start):
    """Other EP ranks' expert windows, including the first."""
    _check(weights, 40, start, True, start)


@requires_gfx1250
@pytest.mark.parametrize("block_m", [16, 32, 64])
def test_moe_mxfp4_a16w4_gfx1250_every_block_m(weights, block_m):
    """Every BLOCK_M on one 1024-token routing, so ragged expert tails hit each tile height."""
    w13, s13, w2, s2, norm = weights
    gen = torch.Generator(device="cuda").manual_seed(7)
    x, wts, ids = _routing(1024, gen)
    cfg = dict(pick_config(1024), BLOCK_M=block_m)
    got = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0,
                          cfg=cfg)
    ref = _reference(x, wts, ids, w13, s13, w2, s2, 0, norm, 1e-8, 1.0)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert rel < 4e-3, rel


@requires_gfx1250
def test_moe_mxfp4_a16w4_gfx1250_no_local_routes(weights):
    w13, s13, w2, s2, _ = weights
    ids = (torch.arange(9 * _TOPK, device="cuda", dtype=torch.int32).reshape(9, _TOPK) % 8) + 100
    x = torch.randn(9, _K, device="cuda").to(torch.bfloat16)
    wts = torch.full((9, _TOPK), 1.0 / _TOPK, device="cuda")
    got = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, expert_start=0)
    assert torch.equal(got, torch.zeros_like(got))


@requires_gfx1250
def test_moe_mxfp4_a16w4_gfx1250_one_hot_expert(weights):
    """All tokens routed to one local expert (the largest group a rank can see)."""
    w13, s13, w2, s2, norm = weights
    gen = torch.Generator(device="cuda").manual_seed(3)
    T = 300
    x = torch.randn(T, _K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.full((T, _TOPK), 200, device="cuda", dtype=torch.int32)
    ids[:, 0] = 5
    wts = torch.softmax(torch.randn(T, _TOPK, device="cuda", generator=gen), -1)
    got = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0)
    ref = _reference(x, wts, ids, w13, s13, w2, s2, 0, norm, 1e-8, 1.0)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert rel < 4e-3, rel
