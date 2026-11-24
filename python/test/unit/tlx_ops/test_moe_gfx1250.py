"""gfx1250 BF16 routed MoE coverage at the ``moe`` gfx1250 focus shapes (Muse-Spark-Open TP4/EP4 boundary layers)."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna5
from triton.tlx.ops.kernels.moe._shapes import GFX1250_CORRECTNESS_SHAPES, label
from triton.tlx.ops.kernels.moe.gfx1250 import moe, pick_config

requires_gfx1250 = pytest.mark.skipif(not is_hip_cdna5(), reason="Requires CDNA5")

_SHAPE = GFX1250_CORRECTNESS_SHAPES[0]
_K, _I, _H, _E, _E_TOTAL, _TOPK = (_SHAPE.K, _SHAPE.I, _SHAPE.H, _SHAPE.local_experts, _SHAPE.num_experts, _SHAPE.topk)
assert all((s.K, s.I, s.H, s.local_experts, s.num_experts, s.topk) == (_K, _I, _H, _E, _E_TOTAL, _TOPK)
           for s in GFX1250_CORRECTNESS_SHAPES)


def _reference(x, wts, ids, w13, w2, start, norm, eps, gain):
    """Op rounding points: BF16 x, fp32 GEMM, BF16 SiLU*up, fp32 GEMM, BF16 expert output."""
    T = x.shape[0]
    E, two_i, _ = w13.shape
    I = two_i // 2
    out = torch.zeros(T, w2.shape[1], device=x.device)
    for e in range(E):
        tt, jj = torch.nonzero(ids == e + start, as_tuple=True)
        if not len(tt):
            continue
        gu = x[tt].float() @ w13[e].float().T
        h = (torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]).to(torch.bfloat16)
        y = (h.float() @ w2[e].float().T).to(torch.bfloat16).float()
        if norm is not None:
            y = (y * (norm.float() + gain).to(norm.dtype).float()).to(torch.bfloat16).float()
            y = y * torch.rsqrt((y * y).mean(-1, keepdim=True) + eps)
        out.index_add_(0, tt, y * wts[tt, jj, None])
    return out.to(torch.bfloat16)


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator(device="cuda").manual_seed(0)
    w13 = (torch.randn(_E, 2 * _I, _K, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(_E, _H, _I, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    norm = (torch.randn(_H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    return w13, w2, norm


def _routing(T, gen):
    x = torch.randn(T, _K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, _E_TOTAL, device="cuda", generator=gen).argsort(dim=1)[:, :_TOPK].to(torch.int32)
    wts = torch.softmax(torch.randn(T, _TOPK, device="cuda", generator=gen), -1)
    return x, wts, ids


def _assert_close(got, ref):
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert rel < 4e-3, rel


def _check(weights, T, start, use_norm, seed, cfg=None):
    w13, w2, norm = weights
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x, wts, ids = _routing(T, gen)
    norm = norm if use_norm else None
    got = moe(x, wts, ids, w13, w2, expert_start=start, post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0,
              cfg=cfg)
    _assert_close(got, _reference(x, wts, ids, w13, w2, start, norm, 1e-8, 1.0))


def test_moe_gfx1250_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, has_impl

    spec, = [spec for spec in CATALOG if (spec.op, spec.arch) == ("moe", "gfx1250")]
    assert spec.impl == "triton.tlx.ops.kernels.moe.gfx1250:moe"
    assert spec.dtypes == frozenset({"bfloat16"})
    assert not spec.supports_backward
    assert has_impl("moe", "gfx1250")
    assert spec.accepts(dict(K=_K, I=_I, H=_H)) and spec.accepts(dict(K=128, I=384, H=640))
    assert not spec.accepts(dict(K=_K, I=_I, H=4000)) and not spec.accepts(dict(K=0, I=_I, H=_H))


@requires_gfx1250
@pytest.mark.parametrize("shape", GFX1250_CORRECTNESS_SHAPES, ids=label)
@pytest.mark.parametrize("use_norm", [True, False])
def test_moe_gfx1250_matches_reference(weights, shape, use_norm):
    _check(weights, shape.tokens, shape.expert_start, use_norm, shape.tokens)


@requires_gfx1250
@pytest.mark.parametrize("start", [0, 64])
def test_moe_gfx1250_expert_start(weights, start):
    """Other EP ranks' expert windows, including the first."""
    _check(weights, 40, start, True, start)


@requires_gfx1250
@pytest.mark.parametrize("T", [64, 512, 1536, 6144, 6145])
def test_moe_gfx1250_tier_boundaries(weights, T):
    """Both sides of every pick_config tier boundary."""
    _check(weights, T, 0, True, T)


@requires_gfx1250
@pytest.mark.parametrize("block_m", [16, 32, 64, 128, 256])
def test_moe_gfx1250_every_block_m(weights, block_m):
    """Every BLOCK_M with both short tails on one 1024-token routing, so ragged expert tails hit each tile height.
    A 2-deep ring keeps the BLOCK_M = 256 tile within LDS."""
    cfg = dict(pick_config(1024), BLOCK_M=block_m, NUM_BUFFERS=2, HALF_TAIL=True, QUARTER_TAIL=True)
    _check(weights, 1024, 0, True, 7, cfg)


@requires_gfx1250
def test_moe_gfx1250_no_local_routes(weights):
    w13, w2, _ = weights
    ids = (torch.arange(9 * _TOPK, device="cuda", dtype=torch.int32).reshape(9, _TOPK) % 8) + 100
    x = torch.randn(9, _K, device="cuda").to(torch.bfloat16)
    wts = torch.full((9, _TOPK), 1.0 / _TOPK, device="cuda")
    got = moe(x, wts, ids, w13, w2, expert_start=0)
    assert torch.equal(got, torch.zeros_like(got))


@requires_gfx1250
def test_moe_gfx1250_one_hot_expert(weights):
    """All tokens routed to one local expert (the largest group a rank can see)."""
    w13, w2, norm = weights
    gen = torch.Generator(device="cuda").manual_seed(3)
    T = 300
    x = torch.randn(T, _K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.full((T, _TOPK), 200, device="cuda", dtype=torch.int32)
    ids[:, 0] = 5
    wts = torch.softmax(torch.randn(T, _TOPK, device="cuda", generator=gen), -1)
    got = moe(x, wts, ids, w13, w2, post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0)
    _assert_close(got, _reference(x, wts, ids, w13, w2, 0, norm, 1e-8, 1.0))


@requires_gfx1250
@pytest.mark.parametrize("T", [40, 512, 1024, 2048, 8192])
@pytest.mark.parametrize("K,I,H", [(128, 128, 128), (256, 256, 384), (640, 640, 640), (768, 384, 640)])
def test_moe_gfx1250_untuned_dims(T, K, I, H):
    """Multiples of 128 that a pick_config tier's BLOCK_K, ring or 256-wide N tile, or the 512-wide row copy, do not
    divide; H > K takes a separate stage-2 output buffer."""
    gen = torch.Generator(device="cuda").manual_seed(T + K)
    E, start = 8, 8
    w13 = (torch.randn(E, 2 * I, K, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    x = torch.randn(T, K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, 4 * E, device="cuda", generator=gen).argsort(dim=1)[:, :4].to(torch.int32)
    wts = torch.softmax(torch.randn(T, 4, device="cuda", generator=gen), -1)
    got = moe(x, wts, ids, w13, w2, expert_start=start)
    _assert_close(got, _reference(x, wts, ids, w13, w2, start, None, 1e-6, 0.0))


@pytest.mark.parametrize("name", ["w13", "w2"])
def test_moe_gfx1250_rejects_misaligned_weights(name):
    E, I, K, H, T = 2, 128, 128, 128, 4
    w = {"w13": (E, 2 * I, K), "w2": (E, H, I)}
    shifted = torch.zeros(E * 2 * I * K + 1, dtype=torch.bfloat16)[1:]
    ws = {
        n: shifted[:E * shape[1] * shape[2]].view(shape) if n == name else torch.zeros(shape, dtype=torch.bfloat16)
        for n, shape in w.items()
    }
    x = torch.zeros(T, K, dtype=torch.bfloat16)
    ids = torch.zeros(T, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match=f"{name} is TDM-loaded"):
        moe(x, torch.ones(T, 2), ids, ws["w13"], ws["w2"])


def test_moe_gfx1250_rejects_bad_post_norm_weight():
    E, I, K, H, T = 2, 128, 128, 128, 4
    w13, w2 = torch.zeros(E, 2 * I, K, dtype=torch.bfloat16), torch.zeros(E, H, I, dtype=torch.bfloat16)
    x, ids = torch.zeros(T, K, dtype=torch.bfloat16), torch.zeros(T, 2, dtype=torch.int32)
    for bad in (torch.ones(H - 1, dtype=torch.bfloat16), torch.ones(H, dtype=torch.int32)):
        with pytest.raises(ValueError, match="post_norm_weight"):
            moe(x, torch.ones(T, 2), ids, w13, w2, post_norm_weight=bad)


@requires_gfx1250
def test_moe_gfx1250_public_op(weights):
    """``tlx.ops.moe`` dispatches to this kernel on gfx1250 and fills ``out``."""
    import triton.tlx.ops as ops

    w13, w2, norm = weights
    gen = torch.Generator(device="cuda").manual_seed(11)
    x, wts, ids = _routing(40, gen)
    out = torch.empty(40, _H, device="cuda", dtype=torch.bfloat16)
    got = ops.moe(x, wts, ids, w13, w2, expert_start=64, post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0,
                  out=out)
    assert got is out
    _assert_close(out, _reference(x, wts, ids, w13, w2, 64, norm, 1e-8, 1.0))
