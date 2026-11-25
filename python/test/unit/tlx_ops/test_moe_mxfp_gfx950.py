"""gfx950 MXFP4 W4A4 routed MoE kernel coverage."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.moe_mxfp.gfx950 import (_shuffle_weights, moe_mxfp4, pick_config, prepare_w13,
                                                    prepare_w13_scales, prepare_w2, prepare_w2_scales, quantize_rows)

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


_CFG = {
    "auto": None,
    "small32": dict(BLOCK_M=32, WN=8, A_LDS=False),
    "small64": dict(BLOCK_M=64, WN=4, A_LDS=True),
    "small64s": dict(BLOCK_M=64, WN=4, A_LDS=True, SORTED=True),
    "small96s": dict(BLOCK_M=96, WN=4, A_LDS=True, SORTED=True, WPE=2),
    "quad128": dict(BLOCK_M=128, GROUP_M=1, QUAD=True),
    "quad128g4": dict(BLOCK_M=128, GROUP_M=4, QUAD=True),
    "quad256": dict(BLOCK_M=256, GROUP_M=1, QUAD=True),
}
_K, _I, _H, _E_TOTAL, _TOPK = 1024, 1024, 256, 32, 8


def _weights(E, gen, use_norm=False):
    w13, s13 = _quant_weights(E, 2 * _I, _K, gen)
    w2, s2 = _quant_weights(E, _H, _I, gen)
    norm = (torch.randn(_H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16) if use_norm else None
    return w13, s13, w2, s2, norm


def _routing(T, gen):
    ids = torch.rand(T, _E_TOTAL, device="cuda", generator=gen).argsort(dim=1)[:, :_TOPK].to(torch.int32)
    wts = torch.softmax(torch.randn(T, _TOPK, device="cuda", generator=gen), -1)
    return ids, wts


def _engine_cfg(engine, T):
    return _CFG[engine] if engine != "auto" else pick_config(T * _TOPK, _E_TOTAL)


def _check(got, x, wts, ids, weights, start, engine):
    w13, s13, w2, s2, norm = weights
    ref = _reference(x, wts, ids, w13, s13, w2, s2, start, norm, 1e-8, 1.0)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    # The fused 128-row path sums the top-k expert outputs with BF16 atomics, rounding after each add.
    cfg = _engine_cfg(engine, x.shape[0])
    fused = cfg.get("QUAD", False) and cfg["BLOCK_M"] == 128 and norm is None
    assert rel < (6e-3 if fused else 1e-3), rel


def _run(x, wts, ids, weights, prepared, start, engine):
    norm = weights[4]
    return moe_mxfp4(x, wts, ids, *prepared, expert_start=start, post_norm_weight=norm, post_norm_eps=1e-8,
                     post_norm_gain=1.0, cfg=_CFG[engine], num_experts=_E_TOTAL)


def _prepare(weights):
    w13, s13, w2, s2, _ = weights
    return (*prepare_w13(w13, s13), *prepare_w2(w2, s2))


@requires_gfx950
@pytest.mark.parametrize("T", [1, 5, 40, 300])
@pytest.mark.parametrize("engine",
                         ["auto", "small32", "small64", "small64s", "small96s", "quad128", "quad128g4", "quad256"])
@pytest.mark.parametrize("start,use_norm", [(0, False), (0, True), (16, True)])
def test_moe_mxfp4_matches_reference(T, engine, start, use_norm):
    gen = torch.Generator(device="cuda").manual_seed(0)
    weights = _weights(8, gen, use_norm)
    x = torch.randn(T, _K, device="cuda", generator=gen).to(torch.bfloat16)
    ids, wts = _routing(T, gen)
    _check(_run(x, wts, ids, weights, _prepare(weights), start, engine), x, wts, ids, weights, start, engine)


@requires_gfx950
@pytest.mark.parametrize("T", [1, 40, 300])
@pytest.mark.parametrize("engine",
                         ["auto", "small32", "small64", "small64s", "small96s", "quad128", "quad128g4", "quad256"])
@pytest.mark.parametrize("start,use_norm", [(0, False), (16, True)])
def test_moe_mxfp4_natural_weights_match_prepared(T, engine, start, use_norm):
    """Natural-layout weights load the same bytes into each lane, so the result is bitwise the prepared path's."""
    gen = torch.Generator(device="cuda").manual_seed(1)
    weights = _weights(8, gen, use_norm)
    w13, s13, w2, s2, norm = weights
    x = torch.randn(T, _K, device="cuda", generator=gen).to(torch.bfloat16)
    ids, wts = _routing(T, gen)
    # The fused 128-row weighted sum uses BF16 atomics, whose order varies run to run.
    cfg = dict(_engine_cfg(engine, T), FUSE_WSUM=False)
    kw = dict(expert_start=start, post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0, cfg=cfg,
              num_experts=_E_TOTAL)
    want = moe_mxfp4(x, wts, ids, *_prepare(weights), **kw)
    got = moe_mxfp4(x, wts, ids, w13, prepare_w13_scales(w13, s13), w2, prepare_w2_scales(w2, s2), natural_weights=True,
                    **kw)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@requires_gfx950
@pytest.mark.parametrize("T", [1, 40, 300])
@pytest.mark.parametrize("engine",
                         ["auto", "small32", "small64", "small64s", "small96s", "quad128", "quad128g4", "quad256"])
def test_moe_mxfp4_gate_up_stacked_match_prepared(T, engine):
    """Pre-shuffled w13 that keeps [gate; up] stacked (AITER's ``shuffle_weight(w13, (16, 16))`` bytes) reads the
    same bytes into each lane as the interleaved prepared layout, so the result is bitwise the same."""
    gen = torch.Generator(device="cuda").manual_seed(2)
    weights = _weights(8, gen, use_norm=True)
    w13, s13, w2, s2, norm = weights
    x = torch.randn(T, _K, device="cuda", generator=gen).to(torch.bfloat16)
    ids, wts = _routing(T, gen)
    cfg = dict(_engine_cfg(engine, T), FUSE_WSUM=False)
    kw = dict(expert_start=0, post_norm_weight=norm, post_norm_eps=1e-8, post_norm_gain=1.0, cfg=cfg,
              num_experts=_E_TOTAL)
    want = moe_mxfp4(x, wts, ids, *_prepare(weights), **kw)
    got = moe_mxfp4(x, wts, ids, _shuffle_weights(w13), prepare_w13_scales(w13, s13), _shuffle_weights(w2),
                    prepare_w2_scales(w2, s2), gate_up_stacked=True, **kw)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("num_valid,num_experts,engine",
                         [(128, 32, "small32"),  # 4 rows per expert
                          (512, 32, "small32"),  # 16
                          (1024, 32, "quad128g4"),  # 32
                          (2048, 32, "quad128g4"),  # 64
                          (4096, 32, "small64s"),  # 128
                          (8192, 32, "small96s"),  # 256
                          (12288, 32, "small96s"),  # 384
                          (16384, 32, "small96s"),  # 512
                          (16384, 64, "small96s"),  # 256: the same slots over twice the experts pick the smaller tile
                          (65536, 256, "small96s"),  # 256 under expert parallelism (ids over all 256 experts)
                          ])
def test_moe_mxfp4_pick_config_keys_on_rows_per_expert(num_valid, num_experts, engine):
    assert pick_config(num_valid, num_experts) == _CFG[engine]


@requires_gfx950
@pytest.mark.parametrize("engine", ["small32", "small64", "quad128", "quad256"])
def test_moe_mxfp4_strided_activations(engine):
    """A view with a non-unit column stride quantizes its logical columns, not adjacent storage."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    weights = _weights(8, gen)
    x = torch.randn(300, 2 * _K, device="cuda", generator=gen).to(torch.bfloat16)[:, ::2]
    assert x.stride(1) == 2
    ids, wts = _routing(300, gen)
    _check(_run(x, wts, ids, weights, _prepare(weights), 0, engine), x.contiguous(), wts, ids, weights, 0, engine)


@requires_gfx950
def test_quantize_rows_strided():
    x = (torch.randn(40, 2 * _K, device="cuda") * 3).to(torch.bfloat16)[:, 1::2]
    for packed in (False, True):
        q, s = quantize_rows(x, packed=packed)
        rq, rs = quantize_rows(x.contiguous(), packed=packed)
        assert torch.equal(q, rq) and torch.equal(s, rs)


@requires_gfx950
@pytest.mark.parametrize("engine", ["small32", "quad128", "quad256"])
@pytest.mark.parametrize("E,start", [(6, 0), (12, 16)])
def test_moe_mxfp4_non_power_of_two_experts(engine, E, start):
    gen = torch.Generator(device="cuda").manual_seed(0)
    weights = _weights(E, gen)
    x = torch.randn(300, _K, device="cuda", generator=gen).to(torch.bfloat16)
    ids, wts = _routing(300, gen)
    _check(_run(x, wts, ids, weights, _prepare(weights), start, engine), x, wts, ids, weights, start, engine)


@requires_gfx950
@pytest.mark.parametrize("engine", ["small32", "small64", "quad128", "quad256"])
def test_moe_mxfp4_graph_replay(engine):
    """Capture once, then replay with routings whose local-row counts differ (num_post changes): uniform, every slot
    on two local experts, and no local slots at all."""
    T, E = 300, 8
    gen = torch.Generator(device="cuda").manual_seed(0)
    weights = _weights(E, gen)
    prepared = _prepare(weights)
    x = torch.randn(T, _K, device="cuda", generator=gen).to(torch.bfloat16)
    ids, wts = _routing(T, gen)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(x, wts, ids, weights, prepared, 0, engine)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _run(x, wts, ids, weights, prepared, 0, engine)

    skewed = torch.stack([torch.arange(T, device="cuda") % 2] +
                         [8 + (torch.arange(T, device="cuda") + j) % 24 for j in range(1, _TOPK)], 1).to(torch.int32)
    for new_ids in (_routing(T, gen)[0], skewed, ids + _E_TOTAL):
        ids.copy_(new_ids)
        wts.copy_(_routing(T, gen)[1])
        graph.replay()
        torch.cuda.synchronize()
        if (new_ids >= E).all():
            assert not out.float().abs().max().item()
        else:
            _check(out, x, wts, ids, weights, 0, engine)


def test_prepare_weights_rejects_bad_shapes():
    w = torch.zeros(2, 512, 512, dtype=torch.uint8)
    s = torch.zeros(2, 512, 32, dtype=torch.uint8)
    prepare_w13(w, s)
    prepare_w2(w, s)
    with pytest.raises(ValueError, match="uint8"):
        prepare_w2(w.to(torch.bfloat16), s)
    with pytest.raises(ValueError, match="do not match"):
        prepare_w2(w, s[:, :, :16])
    with pytest.raises(ValueError, match="multiple of 256"):
        prepare_w13(w[:, :384], s[:, :384])
    with pytest.raises(ValueError, match="multiple of 512"):
        prepare_w2(w[:, :, :128], s[:, :, :8])
