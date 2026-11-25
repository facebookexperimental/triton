"""Public ``tlx.ops.moe`` coverage for gfx950 BF16 and gfx1250 FP8."""

import pytest
import torch
import triton

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import InvalidInput, UnsupportedBackward
from triton.tlx.ops import moe as tlx_moe
from triton.tlx.ops._catalog import CATALOG
from triton.tlx.ops.kernels.moe._shapes import CORRECTNESS_SHAPES, GFX1250_1, inputs, label
from triton.tlx.ops.kernels.moe import gfx950 as _gfx950

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")

SMALL = dict(K=256, I=256, H=384, E=8, E_total=32, topk=4)


@pytest.mark.parametrize("rows,wide", [(160, False), (161, True), (384, True), (385, False)])
def test_wide_a_ownership_is_config_scoped(rows, wide):
    gate, down = _gfx950.pick_config(rows)
    assert gate.get("WIDE_A_OWNERSHIP", False) is wide
    assert down.get("WIDE_A_OWNERSHIP", False) is wide
    if wide:
        assert gate["num_warps"] == down["num_warps"] == 2


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
@pytest.mark.parametrize("T", [3, 27, 300, 4000])
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


def _is_gfx1250():
    return (torch.version.hip is not None and torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1250")


requires_gfx1250 = pytest.mark.skipif(not _is_gfx1250(), reason="Requires gfx1250")
FP8_SCALES = (0.5, 0.25, 0.0625, 0.125)


def _fp8_inputs(m, *, d=256, i=256, h=256, e=4, topk=2, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed + m)

    def payload(shape):
        return (torch.randn(shape, generator=gen, device="cuda") * .5).to(torch.float8_e4m3fn)

    x = payload((m, d))
    w13 = payload((e, 2 * i, d))
    w2 = payload((e, h, i))
    weights, ids = torch.randn(m, e, generator=gen, device="cuda").topk(topk, dim=1)
    return x, weights.softmax(1), ids.to(torch.int32), w13, w2


def _fp8_reference(x, weights, ids, w13, w2, scales=FP8_SCALES, expert_start=0, round_division=True):
    """Independent native FP8 GEMMs; preserve the API's BF16 and slot-order boundaries."""
    m, topk = ids.shape
    e, two_i, _ = w13.shape
    i, h = two_i // 2, w2.shape[1]
    sx, sw13, sh, sw2 = scales
    expert_output = torch.zeros(m, topk, h, device=x.device)

    def mm(a, w, sa, sb):
        padded = torch.zeros(triton.cdiv(len(a), 16) * 16, a.shape[1], device=a.device, dtype=a.dtype)
        padded[:len(a)] = a
        return torch._scaled_mm(padded, w.T, scale_a=torch.tensor(sa, device=a.device),
                                scale_b=torch.tensor(sb, device=a.device), out_dtype=torch.float32,
                                use_fast_accum=False)[:len(a)]

    for expert in range(e):
        token, slot = torch.where(ids.long() - expert_start == expert)
        if not token.numel():
            continue
        gu = mm(x.float()[token].to(x.dtype), w13[expert], sx, sw13)
        hidden = torch.nn.functional.silu(gu[:, :i]) * gu[:, i:]
        # The kernel uses gl.div_rn. Scalar torch division can multiply by a
        # rounded reciprocal, changing an FP8 tie for non-power-of-two scales.
        # Evaluate the quotient in FP64, then round to the declared FP32 point.
        hidden = ((hidden.double() / sh).float() if round_division else hidden / sh).to(x.dtype)
        y = mm(hidden, w2[expert], sh, sw2).to(torch.bfloat16).float()
        expert_output[token, slot] = y
    output = torch.zeros(m, h, device=x.device)
    for slot in range(topk):
        local = (ids[:, slot] >= expert_start) & (ids[:, slot] < expert_start + e)
        weight = torch.where(local, weights[:, slot], 0)
        output = output + expert_output[:, slot] * weight[:, None]
    return output.to(torch.bfloat16)


def test_catalog_entry_fp8_gfx1250():
    spec, = [s for s in CATALOG if (s.op, s.arch) == ("moe", "gfx1250")]
    assert spec.dtypes == frozenset({"float8_e4m3fn"})
    assert not spec.supports_backward
    assert spec.accepts(dict(K=256, I=768, H=1024))
    assert not spec.accepts(dict(K=128, I=256, H=256))
    assert not spec.accepts(dict(K=256, I=256, H=8448))


@requires_gfx1250
@pytest.mark.parametrize(
    "m", [0, 1, 17, 63, 64, 65, 127, 128, 129, 511, 512, 513, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192])
@pytest.mark.parametrize("topk", [2, 4])
def test_fp8_public_matches_reference(m, topk):
    x, weights, ids, w13, w2 = _fp8_inputs(m, topk=topk)
    out = torch.full((m, w2.shape[1]), float("nan"), device="cuda", dtype=torch.bfloat16)
    got = tlx_moe(x, weights, ids, w13, w2, fp8_scales=FP8_SCALES, out=out)
    assert got is out
    assert torch.equal(got, _fp8_reference(x, weights, ids, w13, w2))


@requires_gfx1250
@pytest.mark.parametrize("routing", ["remote", "mixed", "duplicate", "balanced", "int64_sentinels"])
@pytest.mark.parametrize("expert_start", [0, 8])
def test_fp8_public_routing(routing, expert_start):
    x, weights, ids, w13, w2 = _fp8_inputs(129)
    ids = ids.long() + expert_start
    if routing == "remote":
        ids.fill_(expert_start + 4)
    elif routing == "mixed":
        ids[::2] = -1
        ids[1::2, 0] = expert_start + 4
    elif routing == "duplicate":
        ids.fill_(expert_start)
    elif routing == "balanced":
        ids.copy_(torch.arange(ids.numel(), device="cuda").reshape_as(ids) % 4 + expert_start)
    else:
        ids[::3] = -(1 << 63)
        ids[1::3] = (1 << 32) + expert_start
        ids[2::3, 0] = (1 << 63) - 1
    got = tlx_moe(x, weights, ids, w13, w2, expert_start=expert_start, fp8_scales=FP8_SCALES)
    assert torch.equal(got, _fp8_reference(x, weights, ids, w13, w2, expert_start=expert_start))


@requires_gfx1250
@pytest.mark.parametrize("dims", [(512, 768, 1024, 8), (1024, 2048, 1024, 8), (8192, 2048, 8192, 8)])
def test_fp8_public_dimensions(dims):
    d, i, h, e = dims
    x, weights, ids, w13, w2 = _fp8_inputs(128, d=d, i=i, h=h, e=e)
    scales = (0.5, 0.25, 1.0, 0.125)
    got = tlx_moe(x, weights, ids, w13, w2, fp8_scales=scales)
    assert torch.equal(got, _fp8_reference(x, weights, ids, w13, w2, scales))


@requires_gfx1250
@pytest.mark.parametrize("sh", [.0625, .09375])
@pytest.mark.parametrize("m", [129, 4096, 8192, 16384])
def test_fp8_public_graph_changed_inputs_and_weights(sh, m):
    values = _fp8_inputs(m)
    x, weights, ids, w13, w2 = values
    scales = (.5, .25, sh, .125)
    out = torch.empty(m, 256, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        tlx_moe(*values, fp8_scales=scales, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = tlx_moe(*values, fp8_scales=scales, out=out)
    for seed in [101, 509, 701, 907]:
        for dst, src in zip(values, _fp8_inputs(m, seed=seed)):
            dst.copy_(src)
        if seed == 509:
            ids[::2] = -1
        elif seed == 701:
            x.view(torch.uint8).zero_()
            ids.fill_(-1)
        out.fill_(float("nan"))
        graph.replay()
        assert result is out
        assert torch.equal(out, _fp8_reference(*values, scales))


@requires_gfx1250
def test_fp8_public_rejects_invalid_inputs():
    values = _fp8_inputs(17)
    x, weights, ids, w13, w2 = values
    for scales in [
            None, (1., ), (1., 1., 0., 1.), (1., True, 1., 1.), (1., 1., float("inf"), 1.), (1., 1., float("nan"), 1.)
    ]:
        with pytest.raises(InvalidInput):
            tlx_moe(*values, fp8_scales=scales)
    with pytest.raises(InvalidInput):
        tlx_moe(*values, fp8_scales=FP8_SCALES, post_norm_weight=torch.ones(256, device="cuda"))
    with pytest.raises(InvalidInput):
        tlx_moe(x, weights.half(), ids, w13, w2, fp8_scales=FP8_SCALES)
    with pytest.raises(InvalidInput):
        tlx_moe(x, weights[:, :1].contiguous(), ids[:, :1].contiguous(), w13, w2, fp8_scales=FP8_SCALES)
    with pytest.raises(InvalidInput):
        tlx_moe(x.T, weights, ids, w13, w2, fp8_scales=FP8_SCALES)
    with pytest.raises(InvalidInput):
        tlx_moe(*values, fp8_scales=FP8_SCALES, space="full")
    with pytest.raises(InvalidInput):
        tlx_moe(*values, fp8_scales=FP8_SCALES, out=torch.empty_like(x))
    alias = w13.view(torch.bfloat16).reshape(-1)[:17 * 256].view(17, 256)
    with pytest.raises(InvalidInput):
        tlx_moe(*values, fp8_scales=FP8_SCALES, out=alias)
    with pytest.raises(InvalidInput):
        tlx_moe(x, weights, ids, w13.transpose(1, 2).contiguous().transpose(1, 2), w2, fp8_scales=FP8_SCALES)
    with pytest.raises(UnsupportedBackward):
        tlx_moe(x, weights.requires_grad_(), ids, w13, w2, fp8_scales=FP8_SCALES)


@requires_gfx1250
@pytest.mark.parametrize("experts", [1, 3, 16])
def test_fp8_public_expert_count_boundaries(experts):
    x, weights, ids, w13, w2 = _fp8_inputs(65, e=max(2, experts))
    ids.remainder_(experts)
    got = tlx_moe(x, weights, ids, w13[:experts], w2[:experts], fp8_scales=FP8_SCALES)
    assert torch.equal(got, _fp8_reference(x, weights, ids, w13[:experts], w2[:experts]))


@requires_gfx1250
@pytest.mark.parametrize("shape", GFX1250_1.shapes, ids=label)
def test_fp8_public_registered_shapes(shape):
    x, weights, ids, w13, w2, norm = inputs(shape, seed=shape.tokens)
    assert norm is None
    got = tlx_moe(x, weights, ids, w13, w2, fp8_scales=FP8_SCALES)
    assert torch.equal(got, _fp8_reference(x, weights, ids, w13, w2))


@requires_gfx1250
@pytest.mark.parametrize("m,topk,experts", [(4095, 2, 4), (4096, 2, 4), (4097, 2, 4), (8191, 2, 4), (8192, 2, 4),
                                            (8193, 2, 4), (16384, 2, 4), (2047, 4, 8), (2048, 4, 8), (2049, 4, 8),
                                            (4096, 2, 1), (4096, 2, 3), (8192, 2, 8), (8192, 2, 16)])
@pytest.mark.parametrize("pattern", ["random", "remote", "mixed", "duplicate"])
def test_fp8_large_routing_metadata(m, topk, experts, pattern):
    """Stable grouping, full gather padding and tile maps against a CPU specification."""
    from triton.tlx.ops.kernels.moe import gfx1250
    gen = torch.Generator().manual_seed(31 + m + experts)
    ids = torch.randint(experts, (m, topk), generator=gen, dtype=torch.int64)
    if pattern == "remote":
        ids.fill_(-1)
    elif pattern == "mixed":
        ids[::2] = (1 << 32) + 8
        ids[1::3] = -(1 << 63)
    elif pattern == "duplicate":
        ids.zero_()
    # Nonnegative local IDs shift by 8; sentinels must retain their original meaning.
    local = (ids >= 0) & (ids < experts)
    ids[local] += 8
    r = ids.numel()
    pad, cap = triton.cdiv(r, 64) * 64 + 64, triton.cdiv(r, 64) + experts
    expected_gather = torch.zeros(pad, dtype=torch.int32)
    expected_inverse = torch.full((r, ), -1, dtype=torch.int32)
    expected_tiles = torch.full((cap + 1, 4), -1, dtype=torch.int32)
    start = tile = 0
    for e in range(experts):
        selected = torch.where(ids.flatten() == e + 8)[0]
        count = len(selected)
        expected_gather[start:start + count] = (selected // topk).int()
        expected_inverse[selected] = torch.arange(start, start + count, dtype=torch.int32)
        for first in range(0, count, 64):
            expected_tiles[tile] = torch.tensor([e, start, count, first])
            tile += 1
        start += count
    expected_tiles[cap, 0] = start
    ids = ids.cuda()
    gather = torch.full((pad, ), -777, device="cuda", dtype=torch.int32)
    inverse = torch.full((r, ), -777, device="cuda", dtype=torch.int32)
    tiles = torch.full((cap + 1, 4), -777, device="cuda", dtype=torch.int32)
    gfx1250._chunked_routes(ids, gather, inverse, tiles, experts, 8, chunk_size=128 if r <= 16384 else 256)
    assert torch.equal(gather.cpu(), expected_gather)
    assert torch.equal(inverse.cpu(), expected_inverse)
    actual_tiles = tiles.cpu()
    assert torch.equal(actual_tiles[:tile], expected_tiles[:tile])
    assert torch.equal(actual_tiles[tile:, 0], expected_tiles[tile:, 0])
