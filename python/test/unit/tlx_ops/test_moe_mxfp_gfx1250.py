"""gfx1250 MXFP4 A16W4 routed MoE coverage (BF16 activations, MXFP4 weights) at the ``moe_mxfp`` focus shapes."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna5
from triton.tlx.ops.kernels.moe_mxfp._shapes import CORRECTNESS_SHAPES, FOCUS, SYNTHETIC, label
from triton.tlx.ops.kernels.moe_mxfp.gfx1250 import moe_mxfp4_a16w4, pick_config

requires_gfx1250 = pytest.mark.skipif(not is_hip_cdna5(), reason="Requires CDNA5")

# The targeted tests below run the Muse-Spark-Open layer.
_MUSE = FOCUS.all_shapes()[0]
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
    """Random weights per (K, I, H, local experts), made once per module."""
    cache = {}

    def get(shape):
        key = (shape.K, shape.I, shape.H, shape.local_experts)
        if key not in cache:
            gen = torch.Generator(device="cuda").manual_seed(0)
            w13, s13 = _random_weights(shape.local_experts, 2 * shape.I, shape.K, gen)
            w2, s2 = _random_weights(shape.local_experts, shape.H, shape.I, gen)
            norm = (torch.randn(shape.H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
            cache[key] = w13, s13, w2, s2, norm
        return cache[key]

    return get


def _routing(T, shape, gen):
    x = torch.randn(T, shape.K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, shape.num_experts, device="cuda", generator=gen).argsort(dim=1)[:, :shape.topk].to(torch.int32)
    wts = torch.softmax(torch.randn(T, shape.topk, device="cuda", generator=gen), -1)
    return x, wts, ids


def _check(weights, shape, T, start, use_norm, seed):
    w13, s13, w2, s2, norm = weights(shape)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x, wts, ids = _routing(T, shape, gen)
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
    _check(weights, shape, shape.tokens, shape.expert_start, use_norm, shape.tokens)


@requires_gfx1250
@pytest.mark.parametrize("start", [0, 64])
def test_moe_mxfp4_a16w4_gfx1250_expert_start(weights, start):
    """Other EP ranks' expert windows, including the first."""
    _check(weights, _MUSE, 40, start, True, start)


@requires_gfx1250
@pytest.mark.parametrize("block_m", [16, 32, 64, 128])
def test_moe_mxfp4_a16w4_gfx1250_every_block_m(weights, block_m):
    """Every BLOCK_M in both stages on one 1024-token routing, so ragged expert tails hit each tile height."""
    w13, s13, w2, s2, norm = weights(_MUSE)
    gen = torch.Generator(device="cuda").manual_seed(7)
    x, wts, ids = _routing(1024, _MUSE, gen)
    cfg = dict(pick_config(1024), BLOCK_M=block_m, stage2=dict(BLOCK_M=block_m, BLOCK_K=64))
    got = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0,
                          cfg=cfg)
    ref = _reference(x, wts, ids, w13, s13, w2, s2, 0, norm, 1e-8, 1.0)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert rel < 4e-3, rel


@requires_gfx1250
def test_moe_mxfp4_a16w4_gfx1250_ring_deeper_than_k(weights):
    """A TDM ring deeper than the K loop (512 / 128 = 4 steps, 6 stages): the prologue's extra stages are predicated
    off."""
    shape = SYNTHETIC[1]
    w13, s13, w2, s2, norm = weights(shape)
    x, wts, ids = _routing(shape.tokens, shape, torch.Generator(device="cuda").manual_seed(5))
    cfg = dict(pick_config(shape.tokens), NUM_BUFFERS=6)
    got = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, expert_start=shape.expert_start, post_norm_weight=norm,
                          post_norm_eps=1e-8, post_norm_gain=1.0, cfg=cfg)
    ref = _reference(x, wts, ids, w13, s13, w2, s2, shape.expert_start, norm, 1e-8, 1.0)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert rel < 4e-3, rel


@requires_gfx1250
def test_moe_mxfp4_a16w4_gfx1250_no_local_routes(weights):
    w13, s13, w2, s2, _ = weights(_MUSE)
    topk = _MUSE.topk
    ids = (torch.arange(9 * topk, device="cuda", dtype=torch.int32).reshape(9, topk) % 8) + 100
    x = torch.randn(9, _MUSE.K, device="cuda").to(torch.bfloat16)
    wts = torch.full((9, topk), 1.0 / topk, device="cuda")
    got = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, expert_start=0)
    assert torch.equal(got, torch.zeros_like(got))


@requires_gfx1250
def test_moe_mxfp4_a16w4_gfx1250_one_hot_expert(weights):
    """All tokens routed to one local expert (the largest group a rank can see)."""
    w13, s13, w2, s2, norm = weights(_MUSE)
    gen = torch.Generator(device="cuda").manual_seed(3)
    T = 300
    x = torch.randn(T, _MUSE.K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.full((T, _MUSE.topk), 200, device="cuda", dtype=torch.int32)
    ids[:, 0] = 5
    wts = torch.softmax(torch.randn(T, _MUSE.topk, device="cuda", generator=gen), -1)
    got = moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0)
    ref = _reference(x, wts, ids, w13, s13, w2, s2, 0, norm, 1e-8, 1.0)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert rel < 4e-3, rel


@requires_gfx1250
def test_moe_mxfp4_a16w4_gfx1250_rejects_unaligned_weights(weights):
    small = SYNTHETIC[0]
    w13, s13, w2, s2, _ = weights(small)
    x, wts, ids = _routing(4, small, torch.Generator(device="cuda").manual_seed(0))
    raw = torch.empty(w13.numel() + 2, dtype=torch.uint8, device="cuda")
    shifted = raw[2:].view(w13.shape)
    shifted.copy_(w13)
    with pytest.raises(ValueError, match="multiples of 4 bytes"):
        moe_mxfp4_a16w4(x, wts, ids, shifted, s13, w2, s2)


@requires_gfx1250
def test_moe_mxfp4_a16w4_gfx1250_rejects_bad_post_norm_weight(weights):
    small = SYNTHETIC[0]
    w13, s13, w2, s2, norm = weights(small)
    x, wts, ids = _routing(4, small, torch.Generator(device="cuda").manual_seed(0))
    for bad in (norm[:-1], norm.to(torch.int32), norm.cpu()):
        with pytest.raises(ValueError, match="post_norm_weight"):
            moe_mxfp4_a16w4(x, wts, ids, w13, s13, w2, s2, post_norm_weight=bad)
