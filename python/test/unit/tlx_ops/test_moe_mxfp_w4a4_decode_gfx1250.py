"""gfx1250 small-batch MXFP4 W4A4 routed MoE (decode) against a torch W4A4 reference with the op's rounding points."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna5
from triton.tlx.ops.kernels.moe_mxfp.gfx1250_w4a4_decode import moe_mxfp4_w4a4_decode

requires_gfx1250 = pytest.mark.skipif(not is_hip_cdna5(), reason="Requires CDNA5")

# (K, I, H, local experts, total experts, topk): a small synthetic layer and Muse-Spark-Open at EP4.
_SMALL = (512, 512, 512, 8, 32, 4)
_MUSE = (4096, 4096, 4096, 64, 256, 8)
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
_E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Round-to-nearest-even decision points between E2M1 magnitudes (True: a tie rounds up to the even code).
_UP = [(0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False)]


def _dequant(q, s):
    lut = torch.tensor(_E2M1, device=q.device)
    v = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], -1).flatten(-2)
    return v * torch.exp2(s.float() - 127).repeat_interleave(32, -1)


def _fake_quant(v):
    """MXFP4 quantize-dequantize along the last dim (AITER / Quark even mode)."""
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


def _reference(x, wts, ids, w13, s13, w2, s2, start, norm, eps, gain):
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
def layers():
    """Random natural-layout MXFP4 weights per layer shape: any byte is a valid E2M1 pair; scales keep values O(0.1)."""
    cache = {}

    def get(shape):
        if shape not in cache:
            K, I, H, E, _, _ = shape
            gen = torch.Generator(device="cuda").manual_seed(0)
            q = lambda n, k: torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device="cuda", generator=gen
                                           )  # noqa: E731
            s = lambda n, k: torch.randint(118, 122, (E, n, k // 32), dtype=torch.uint8, device="cuda", generator=gen
                                           )  # noqa: E731
            norm = (torch.randn(H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
            cache[shape] = q(2 * I, K), s(2 * I, K), q(H, I), s(H, I), norm
        return cache[shape]

    return get


def _routing(T, shape, seed):
    K, _, _, _, e_total, topk = shape
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, e_total, device="cuda", generator=gen).argsort(dim=1)[:, :topk].to(torch.int32)
    wts = torch.softmax(torch.randn(T, topk, device="cuda", generator=gen), -1)
    return x, wts, ids


def _run(layers, shape, T, start, use_norm, seed):
    w13, s13, w2, s2, norm = layers(shape)
    norm = norm if use_norm else None
    x, wts, ids = _routing(T, shape, seed)
    got = moe_mxfp4_w4a4_decode(x, wts, ids, w13, s13, w2, s2, expert_start=start, post_norm_weight=norm,
                                post_norm_eps=1e-8, post_norm_gain=1.0)
    return got, _reference(x, wts, ids, w13, s13, w2, s2, start, norm, 1e-8, 1.0)


def _assert_close(got, ref):
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert rel < 4e-3, rel


@requires_gfx1250
@pytest.mark.parametrize("use_norm", [True, False])
@pytest.mark.parametrize("shape, T, start", [(_SMALL, 1, 0), (_SMALL, 2, 8), (_SMALL, 8, 24), (_MUSE, 1, 0),
                                             (_MUSE, 1, 64), (_MUSE, 1, 192), (_MUSE, 2, 0)])
def test_moe_mxfp4_w4a4_decode_matches_reference(layers, shape, T, start, use_norm):
    got, ref = _run(layers, shape, T, start, use_norm, seed=T + start)
    _assert_close(got, ref)


@requires_gfx1250
def test_moe_mxfp4_w4a4_decode_repeat_calls(layers):
    """The cached workspace and the per-token tickets are reused: back-to-back calls give identical results."""
    first, _ = _run(layers, _MUSE, 1, 0, True, seed=11)
    for _ in range(3):
        again, _ = _run(layers, _MUSE, 1, 0, True, seed=11)
        assert torch.equal(first, again)


@requires_gfx1250
def test_moe_mxfp4_w4a4_decode_no_local_routes(layers):
    w13, s13, w2, s2, norm = layers(_SMALL)
    topk = _SMALL[5]
    x = torch.randn(3, _SMALL[0], device="cuda").to(torch.bfloat16)
    ids = torch.full((3, topk), 30, dtype=torch.int32, device="cuda")
    wts = torch.full((3, topk), 1.0 / topk, device="cuda")
    out = torch.full((3, _SMALL[2]), 7.0, dtype=torch.bfloat16, device="cuda")
    moe_mxfp4_w4a4_decode(x, wts, ids, w13, s13, w2, s2, expert_start=0, post_norm_weight=norm, out=out)
    assert torch.equal(out, torch.zeros_like(out))


@requires_gfx1250
def test_moe_mxfp4_w4a4_decode_rejects_bad_inputs(layers):
    w13, s13, w2, s2, _ = layers(_SMALL)
    x, wts, ids = _routing(2, _SMALL, 0)
    with pytest.raises(ValueError, match="contiguous"):
        moe_mxfp4_w4a4_decode(x, wts.t().contiguous().t(), ids.t().contiguous().t(), w13, s13, w2, s2)
    with pytest.raises(ValueError, match="BF16"):
        moe_mxfp4_w4a4_decode(x.float(), wts, ids, w13, s13, w2, s2)
    H = w2.shape[1]
    for out in (torch.empty(2, H // 2, dtype=torch.bfloat16, device="cuda"),
                torch.empty(2, 2 * H, dtype=torch.bfloat16, device="cuda")[:, :H], torch.empty(2, H, device="cuda")):
        with pytest.raises(ValueError, match="out must be"):
            moe_mxfp4_w4a4_decode(x, wts, ids, w13, s13, w2, s2, out=out)
    for norm in (torch.ones(H, dtype=torch.int32, device="cuda"), torch.ones(H)):
        with pytest.raises(ValueError, match="post_norm_weight must be"):
            moe_mxfp4_w4a4_decode(x, wts, ids, w13, s13, w2, s2, post_norm_weight=norm)
