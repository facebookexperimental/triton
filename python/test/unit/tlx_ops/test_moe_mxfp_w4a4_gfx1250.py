"""gfx1250 MXFP4 W4A4 routed MoE coverage (MXFP4 activations, MXFP4 weights) at the ``moe_mxfp`` focus shapes."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna5
from triton.tlx.ops.kernels.moe_mxfp._shapes import FOCUS, label
from triton.tlx.ops.kernels.moe_mxfp.gfx1250_w4a4 import moe_mxfp4_w4a4, pick_config

requires_gfx1250 = pytest.mark.skipif(not is_hip_cdna5(), reason="Requires CDNA5")

_SHAPES = FOCUS.all_shapes()
_SHAPE = _SHAPES[0]
_K, _I, _H, _E, _E_TOTAL, _TOPK = (_SHAPE.K, _SHAPE.I, _SHAPE.H, _SHAPE.local_experts, _SHAPE.num_experts, _SHAPE.topk)
assert all((s.K, s.I, s.H, s.local_experts, s.num_experts, s.topk) == (_K, _I, _H, _E, _E_TOTAL, _TOPK) for s in _SHAPES)
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
# E2M1 magnitudes and the round-to-nearest-even decision points between them (ties go to the even code).
_E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_UP = [(0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False)]


def _dequant(q, s):
    lut = torch.tensor(_E2M1, device=q.device)
    v = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], -1).flatten(-2)
    return v * torch.exp2(s.float() - 127).repeat_interleave(32, -1)


def _fake_quant(v):
    """MXFP4 quantize-dequantize of fp32 v [..., n] along the last dim (AITER / Quark even mode)."""
    vb = v.float().reshape(*v.shape[:-1], -1, 32)
    amax = vb.abs().amax(-1)
    e = ((amax.view(torch.int32) + 0x200000) & 0x7F800000) >> 23
    unb = (e - 129).clamp(-127, 127).float()
    q = vb * torch.exp2(-unb)[..., None]
    m = q.abs()
    code = torch.zeros_like(m, dtype=torch.long)
    for t, inclusive in _UP:
        code += (m >= t if inclusive else m > t).long()
    mag = torch.tensor(_E2M1_POS, device=v.device)[code]
    return (torch.where(q < 0, -mag, mag) * torch.exp2(unb)[..., None]).reshape(v.shape)


def _random_weights(E, N, K, gen):
    """Random MXFP4 weights in the natural layout: any byte is a valid E2M1 pair; scales keep values O(0.1)."""
    q = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda", generator=gen)
    s = torch.randint(118, 122, (E, N, K // 32), dtype=torch.uint8, device="cuda", generator=gen)
    return q, s


def _reference(x, wts, ids, w13, s13, w2, s2, start, norm, eps, gain):
    """Op rounding points: MXFP4 x, fp32 GEMM, BF16 gate / up, fp32 SiLU * up, MXFP4, fp32 GEMM, BF16 expert output."""
    T = x.shape[0]
    E, two_i, _ = w13.shape
    I = two_i // 2
    xq = _fake_quant(x.float())
    out = torch.zeros(T, w2.shape[1], device=x.device)
    for e in range(E):
        tt, jj = torch.nonzero(ids == e + start, as_tuple=True)
        if not len(tt):
            continue
        gu = (xq[tt] @ _dequant(w13[e], s13[e]).T).to(torch.bfloat16).float()
        g, u = gu[:, :I], gu[:, I:]
        h = _fake_quant(g / (1.0 + torch.exp2(g * -1.4426950408889634)) * u)
        y = (h @ _dequant(w2[e], s2[e]).T).to(torch.bfloat16).float()
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


def _assert_close(got, ref):
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert rel < 4e-3, rel


def _check(weights, T, start, use_norm, seed, cfg=None):
    w13, s13, w2, s2, norm = weights
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x, wts, ids = _routing(T, gen)
    norm = norm if use_norm else None
    got = moe_mxfp4_w4a4(x, wts, ids, w13, s13, w2, s2, expert_start=start, post_norm_weight=norm, post_norm_eps=1e-8,
                         post_norm_gain=1.0, cfg=cfg)
    _assert_close(got, _reference(x, wts, ids, w13, s13, w2, s2, start, norm, 1e-8, 1.0))


def test_fake_quant_rounding():
    """The reference quantizer: E8M0 scale from the amax rounded to two mantissa bits, then E2M1 round to nearest
    even with saturation."""
    v = torch.zeros(32)
    v[:12] = torch.tensor([6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.3, 2.9, 5.9, -6.0])
    want = torch.tensor([6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -0.5, 3.0, 6.0, -6.0])
    assert torch.equal(_fake_quant(v)[:12], want)
    # amax 7.0 rounds up to 8 -> scale 2, so 7.0 / 2 = 3.5 rounds (to even) to 4 -> 8.
    assert _fake_quant(torch.full((32, ), 7.0))[0] == 8.0


@pytest.mark.parametrize("K, I, out_shape, norm_shape, match", [
    (384, 512, None, None, "K=384 is not a multiple of 512"),
    (512, 640, None, None, "I=640 is not a multiple of 512"),
    (512, 512, (1, 256), None, "out must be"),
    (512, 512, None, (256, ), "post_norm_weight must be"),
])
def test_moe_mxfp4_w4a4_rejects_bad_inputs(K, I, out_shape, norm_shape, match):
    """Validation runs before any launch, so CPU tensors exercise it."""
    E, H, topk = 2, 512, 2
    u8 = lambda *shape: torch.zeros(shape, dtype=torch.uint8)  # noqa: E731
    x = torch.zeros(1, K, dtype=torch.bfloat16)
    ids = torch.zeros(1, topk, dtype=torch.int32)
    wts = torch.ones(1, topk)
    out = torch.empty(out_shape, dtype=torch.bfloat16) if out_shape else None
    norm = torch.ones(norm_shape) if norm_shape else None
    with pytest.raises(ValueError, match=match):
        moe_mxfp4_w4a4(x, wts, ids, u8(E, 2 * I, K // 2), u8(E, 2 * I, K // 32), u8(E, H, I // 2), u8(E, H, I // 32),
                       post_norm_weight=norm, out=out)


@requires_gfx1250
@pytest.mark.parametrize("shape", _SHAPES, ids=label)
@pytest.mark.parametrize("use_norm", [True, False])
def test_moe_mxfp4_w4a4_gfx1250_matches_reference(weights, shape, use_norm):
    _check(weights, shape.tokens, shape.expert_start, use_norm, shape.tokens)


@requires_gfx1250
@pytest.mark.parametrize("start", [0, 64])
def test_moe_mxfp4_w4a4_gfx1250_expert_start(weights, start):
    """Other EP ranks' expert windows, including the first."""
    _check(weights, 40, start, True, start)


@requires_gfx1250
@pytest.mark.parametrize("block_m", [32, 64, 128])
def test_moe_mxfp4_w4a4_gfx1250_every_block_m(weights, block_m):
    """Every BLOCK_M on one 1024-token routing, so ragged expert tails hit each tile height (and the half-height
    tail tile for BLOCK_M >= 64)."""
    _check(weights, 1024, 0, True, 7, cfg=dict(pick_config(1024), BLOCK_M=block_m))


@requires_gfx1250
def test_moe_mxfp4_w4a4_gfx1250_no_local_routes(weights):
    w13, s13, w2, s2, _ = weights
    ids = (torch.arange(9 * _TOPK, device="cuda", dtype=torch.int32).reshape(9, _TOPK) % 8) + 100
    x = torch.randn(9, _K, device="cuda").to(torch.bfloat16)
    wts = torch.full((9, _TOPK), 1.0 / _TOPK, device="cuda")
    got = moe_mxfp4_w4a4(x, wts, ids, w13, s13, w2, s2, expert_start=0)
    assert torch.equal(got, torch.zeros_like(got))


@requires_gfx1250
def test_moe_mxfp4_w4a4_gfx1250_one_hot_expert(weights):
    """All tokens routed to one local expert (the largest group a rank can see)."""
    w13, s13, w2, s2, norm = weights
    gen = torch.Generator(device="cuda").manual_seed(3)
    T = 300
    x = torch.randn(T, _K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.full((T, _TOPK), 200, device="cuda", dtype=torch.int32)
    ids[:, 0] = 5
    wts = torch.softmax(torch.randn(T, _TOPK, device="cuda", generator=gen), -1)
    got = moe_mxfp4_w4a4(x, wts, ids, w13, s13, w2, s2, post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0)
    _assert_close(got, _reference(x, wts, ids, w13, s13, w2, s2, 0, norm, 1e-8, 1.0))
