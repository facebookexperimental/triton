"""gfx950 MXFP4 W4A4 routed MoE kernel coverage."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.moe_mxfp4.gfx950 import moe_mxfp4, prepare_w13, prepare_w2, quantize_rows

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _quant_ref(x):
    """OCP MX E2M1 with round-up-amax E8M0 scales and RNE (the AITER/Quark convention), in torch."""
    M, K = x.shape
    g = x.float().reshape(M, K // 32, 32)
    bits = g.abs().amax(-1).view(torch.int32).to(torch.int64) + 0x200000
    biased = (bits >> 23) & 0xFF
    e = torch.where(biased == 255, torch.full_like(biased, 127), (biased - 129).clamp(min=-127))
    scale = (e + 127).to(torch.uint8)
    q = g * torch.exp2(-e.float())[..., None]
    mag = q.abs()
    grid = torch.tensor(_E2M1[:8], device=x.device)
    # Nearest grid point; a tie is between neighbours, so prefer the even index (even mantissa bit).
    d = (mag[..., None] - grid).abs()
    cand = (d == d.min(-1, keepdim=True).values).float()
    code = (cand * (1 + (torch.arange(8, device=x.device) % 2 == 0).float())).argmax(-1)
    code = torch.where(mag >= 6.0, torch.full_like(code, 7), code)
    code = code | ((q.view(torch.int32) < 0).long() << 3)
    code = code.reshape(M, K).to(torch.uint8)
    return code[:, 0::2] | (code[:, 1::2] << 4), scale.reshape(M, K // 32)


def _dequant(q, s):
    lut = torch.tensor(_E2M1, device=q.device)
    v = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], -1).flatten(-2)
    return v * torch.exp2(s.float() - 127).repeat_interleave(32, -1)


def _quant_weights(E, N, K, gen):
    w = (torch.randn(E * N, K, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    q, s = _quant_ref(w)
    return q.reshape(E, N, K // 2).contiguous(), s.reshape(E, N, K // 32).contiguous()


def _reference(x, wts, ids, w13, s13, w2, s2, start, norm, eps, gain):
    """Op rounding points: MXFP4 x, fp32 GEMM, BF16 SiLU*up, MXFP4 requant, BF16 expert output."""
    T = x.shape[0]
    E, two_i, _ = w13.shape
    I = two_i // 2
    xd = _dequant(*_quant_ref(x))
    out = torch.zeros(T, w2.shape[1], device=x.device)
    for e in range(E):
        tt, jj = torch.nonzero(ids == e + start, as_tuple=True)
        if not len(tt):
            continue
        gu = xd[tt] @ _dequant(w13[e], s13[e]).T
        h = (torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]).to(torch.bfloat16)
        y = (_dequant(*_quant_ref(h)) @ _dequant(w2[e], s2[e]).T).to(torch.bfloat16).float()
        if norm is not None:
            y = (y * (norm.float() + gain).to(norm.dtype).float()).to(torch.bfloat16).float()
            y = y * torch.rsqrt((y * y).mean(-1, keepdim=True) + eps)
        out.index_add_(0, tt, y * wts[tt, jj, None])
    return out.to(torch.bfloat16)


@requires_gfx950
@pytest.mark.parametrize("M,K", [(7, 256), (300, 4096), (1, 8192)])
def test_quantize_rows_matches_reference(M, K):
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16) * 3
    x[0, :32] = 0
    x[-1, 5] = 1e-30
    q, s = quantize_rows(x)
    rq, rs = _quant_ref(x)
    assert torch.equal(s, rs)
    assert torch.equal(q, rq)
    if K % 512 == 0:
        pq, ps = quantize_rows(x, packed=True)
        assert torch.equal(pq, rq)
        # [K/512, M, g%4, g//4%2, g//8%2] -> [M, K/32]
        assert torch.equal(ps.view(K // 512, M, 4, 2, 2).permute(1, 0, 4, 3, 2).reshape(M, K // 32), rs)


@requires_gfx950
@pytest.mark.parametrize("T", [1, 5, 40, 300])
@pytest.mark.parametrize("engine", ["auto", "small32", "small64", "quad128", "quad256"])
@pytest.mark.parametrize("start,use_norm", [(0, False), (0, True), (16, True)])
def test_moe_mxfp4_matches_reference(T, engine, start, use_norm):
    K, I, H, E, E_total, topk = 1024, 1024, 256, 8, 32, 8
    gen = torch.Generator(device="cuda").manual_seed(0)
    w13, s13 = _quant_weights(E, 2 * I, K, gen)
    w2, s2 = _quant_weights(E, H, I, gen)
    norm = (torch.randn(H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16) if use_norm else None
    x = torch.randn(T, K, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.rand(T, E_total, device="cuda", generator=gen).argsort(dim=1)[:, :topk].to(torch.int32)
    wts = torch.softmax(torch.randn(T, topk, device="cuda", generator=gen), -1)
    p13, ps13 = prepare_w13(w13, s13)
    p2, ps2 = prepare_w2(w2, s2)
    cfg = {
        "auto": None,
        "small32": dict(BLOCK_M=32, WN=8, A_LDS=False),
        "small64": dict(BLOCK_M=64, WN=4, A_LDS=True),
        "quad128": dict(BLOCK_M=128, GROUP_M=1, QUAD=True),
        "quad256": dict(BLOCK_M=256, GROUP_M=1, QUAD=True),
    }[engine]
    got = moe_mxfp4(x, wts, ids, p13, ps13, p2, ps2, expert_start=start, post_norm_weight=norm, post_norm_eps=1e-8,
                    post_norm_gain=1.0, cfg=cfg)
    ref = _reference(x, wts, ids, w13, s13, w2, s2, start, norm, 1e-8, 1.0)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    # The fused 128-row path sums the top-k expert outputs with BF16 atomics, rounding after each add.
    fused = engine == "quad128" and norm is None
    assert rel < (6e-3 if fused else 1e-3), rel
