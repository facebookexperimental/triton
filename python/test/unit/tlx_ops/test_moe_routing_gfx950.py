"""Public ``tlx.ops.moe_routing`` coverage for gfx950."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import InvalidInput, UnsupportedBackward
from triton.tlx.ops import moe as tlx_moe
from triton.tlx.ops import moe_routing
from triton.tlx.ops._catalog import CATALOG
from triton.tlx.ops.kernels.moe import routing_gfx950
from triton.tlx.ops.kernels.moe._routing_shapes import (CORRECTNESS_SHAPES, ROUTE_EPS, ROUTER_MULTIPLIER, inputs, label,
                                                        reference)

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")

ROUTER = dict(router_multiplier=ROUTER_MULTIPLIER, route_scale=1.0, eps=ROUTE_EPS)


def _separated(T, E, seed=0):
    """Logits whose biased scores differ by >= 1e-4 within a row, so a one-ulp sigmoid difference cannot reorder
    them."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bias = torch.zeros(E, device="cuda")
    ranks = torch.rand(T, E, device="cuda", generator=gen).argsort(dim=1).float()
    scores = 0.02 + 0.96 * ranks / max(E - 1, 1)
    return torch.logit(scores.double()).float(), bias


def _check(got, ref):
    (w, ids), (rw, rids) = got, ref
    torch.testing.assert_close(ids, rids, rtol=0, atol=0)
    torch.testing.assert_close(w, rw, rtol=2e-6, atol=1e-7)


def test_catalog_entry():
    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("moe_routing", "gfx950")]
    assert len(specs) == 1
    spec, = specs
    assert spec.dtypes == frozenset({"float32"})
    assert not spec.supports_backward


def test_moe_routing_is_exported():
    import triton.tlx.ops as ops
    assert "moe_routing" in ops.__all__


@pytest.mark.parametrize("tokens,experts,num_cus,expected", [
    (3072, 256, 256, 1),
    (3073, 256, 256, 4),
    (1536, 256, 128, 1),
    (1537, 256, 128, 4),
    (1, 16, 256, 4),
    (1, 8, 256, 8),
    (1, 64, 256, 1),
])
def test_layout_choice_scales_with_cu_count(tokens, experts, num_cus, expected):
    assert routing_gfx950.pick_config(tokens, experts, num_cus) == expected


@requires_gfx950
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=label)
def test_matches_reference_registered_shapes(shape):
    logits, bias = inputs(shape)
    got = moe_routing(logits, bias, shape.topk, **ROUTER)
    w, ids = got
    rw, rids = reference(logits, bias, shape.topk, **ROUTER)
    # Random logits can tie to within one ulp; compare only rows whose top-(k+1) biased scores are separated.
    biased = torch.sigmoid(logits * ROUTER_MULTIPLIER) + bias
    top = biased.topk(min(shape.topk + 1, shape.experts), dim=1).values
    clear = (top[:, :-1] - top[:, 1:]).min(dim=1).values > 1e-5
    assert clear.float().mean() > 0.5
    _check((w[clear], ids[clear]), (rw[clear], rids[clear]))


@requires_gfx950
@pytest.mark.parametrize("T", [1, 5, 300, 3073, 5000])
@pytest.mark.parametrize("E,topk", [(8, 1), (8, 8), (16, 2), (64, 6), (160, 8), (256, 8), (384, 16), (1024, 32)])
def test_matches_reference_separated_scores(T, E, topk):
    logits, bias = _separated(T, E, seed=T + E)
    _check(moe_routing(logits, bias, topk, **ROUTER), reference(logits, bias, topk, **ROUTER))


@requires_gfx950
@pytest.mark.parametrize("T", [7, 4000])
def test_bias_steers_selection_but_not_weights(T):
    logits, _ = _separated(T, 256, seed=1)
    gen = torch.Generator(device="cuda").manual_seed(2)
    bias = torch.rand(256, device="cuda", generator=gen)
    w, ids = moe_routing(logits, bias, 8, route_scale=2.5, eps=1e-3)
    _check((w, ids), reference(logits, bias, 8, route_scale=2.5, eps=1e-3))
    _, unbiased_ids = moe_routing(logits, torch.zeros_like(bias), 8)
    assert not torch.equal(ids, unbiased_ids)
    scores = torch.sigmoid(logits).gather(1, ids.long())
    torch.testing.assert_close(w, scores * 2.5 / (scores.sum(dim=1, keepdim=True) + 1e-3), rtol=2e-6, atol=0)


@requires_gfx950
@pytest.mark.parametrize("T", [7, 4000])
@pytest.mark.parametrize("shift", [-2.0, -0.5])
def test_negative_biased_scores(T, shift):
    """A bias that pushes every (shift -2) or about half (shift -0.5) of the selection keys below zero. It is the
    same for every expert, so the separated scores stay separated."""
    logits, _ = _separated(T, 256, seed=4)
    bias = torch.full((256, ), shift, device="cuda")
    _check(moe_routing(logits, bias, 8, **ROUTER), reference(logits, bias, 8, **ROUTER))


@requires_gfx950
@pytest.mark.parametrize("T", [3, 4000])
def test_ties_pick_lower_expert_id(T):
    E = 256
    gen = torch.Generator(device="cuda").manual_seed(3)
    logits = torch.randn(T, E, device="cuda", generator=gen) - 10.0
    # Saturated sigmoids: twelve experts score exactly 1.0.
    hot = torch.rand(T, E, device="cuda", generator=gen).argsort(dim=1)[:, :12]
    logits.scatter_(1, hot, 1000.0)
    w, ids = moe_routing(logits, torch.zeros(E, device="cuda"), 8)
    expected = hot.sort(dim=1).values[:, :8].to(torch.int32)
    assert torch.equal(ids, expected)
    assert torch.equal(w, torch.full_like(w, 0.125))


@requires_gfx950
@pytest.mark.parametrize("T", [3, 4000])
def test_nan_logits_are_never_picked(T):
    logits, bias = _separated(T, 256, seed=7)
    gen = torch.Generator(device="cuda").manual_seed(8)
    nan = torch.rand(T, 256, device="cuda", generator=gen).argsort(dim=1)[:, :16]
    logits.scatter_(1, nan, float("nan"))
    w, ids = moe_routing(logits, bias, 8, **ROUTER)
    assert not (ids[:, :, None] == nan[:, None, :]).any()
    assert torch.isfinite(w).all()
    clean = logits.clone()
    clean.scatter_(1, nan, -float("inf"))
    _check((w, ids), reference(clean, bias, 8, **ROUTER))


@requires_gfx950
@pytest.mark.parametrize("E", [8, 256])
def test_rows_with_fewer_non_nan_experts_than_topk(E):
    """Slots past the last non-NaN expert get id -1 and weight 0; the found experts normalize among themselves."""
    topk = 8
    counts = [0, 1, topk // 2, topk - 1, topk, E]
    logits, bias = _separated(len(counts), E, seed=11)
    gen = torch.Generator(device="cuda").manual_seed(12)
    for row, n in enumerate(counts):
        logits[row, torch.randperm(E, device="cuda", generator=gen)[n:]] = float("nan")
    w, ids = moe_routing(logits, bias, topk, **ROUTER)
    # NaN as -inf ranks below every finite score here (zero bias), so the reference's first slots are the same.
    rw, rids = reference(torch.where(logits.isnan(), -float("inf"), logits), bias, topk, **ROUTER)
    found = torch.arange(topk, device="cuda")[None, :] < torch.tensor(counts, device="cuda")[:, None]
    _check((w, ids), (torch.where(found, rw, 0.0), torch.where(found, rids, -1)))
    for row in ids:
        picked = row[row >= 0]
        assert picked.unique().numel() == picked.numel()

    one = torch.full((1, 8), float("nan"), device="cuda")
    one[0, 0] = 0.0
    w, ids = moe_routing(one, torch.zeros(8, device="cuda"), 8)
    assert ids.tolist() == [[0] + [-1] * 7]
    assert w.tolist() == [[1.0] + [0.0] * 7]


@requires_gfx950
@pytest.mark.parametrize("T", [5, 4000])
def test_layouts_are_bitwise_identical(T, monkeypatch):
    logits, bias = inputs(type(CORRECTNESS_SHAPES[0])(T, 256, 8))
    results = []
    for rows_per_wave in (1, 2, 4, 8):
        monkeypatch.setattr(routing_gfx950, "pick_config", lambda t, e, c, r=rows_per_wave: r)
        results.append(moe_routing(logits, bias, 8, **ROUTER))
    for w, ids in results[1:]:
        assert torch.equal(ids, results[0][1])
        assert torch.equal(w.view(torch.int32), results[0][0].view(torch.int32))


@requires_gfx950
def test_rows_do_not_depend_on_the_batch():
    logits, bias = inputs(type(CORRECTNESS_SHAPES[0])(6000, 256, 8))
    w, ids = moe_routing(logits, bias, 8, **ROUTER)
    for start, stop in ((0, 1), (5, 17), (100, 3172), (2000, 6000)):
        ws, ids_s = moe_routing(logits[start:stop], bias, 8, **ROUTER)
        assert torch.equal(ids_s, ids[start:stop])
        assert torch.equal(ws.view(torch.int32), w[start:stop].view(torch.int32))


@requires_gfx950
@pytest.mark.parametrize("T", [1, 6, 3500])
@pytest.mark.parametrize("valid", [0, 1, 5, 10**9])
def test_padded_rows_route_nowhere(T, valid):
    logits, bias = inputs(type(CORRECTNESS_SHAPES[0])(T, 256, 8))
    n = torch.tensor([valid], device="cuda", dtype=torch.int32)
    w, ids = moe_routing(logits, bias, 8, num_valid_tokens=n, **ROUTER)
    rw, rids = moe_routing(logits, bias, 8, **ROUTER)
    real = min(valid, T)
    assert torch.equal(ids[:real], rids[:real]) and torch.equal(w[:real], rw[:real])
    assert (ids[real:] == -1).all() and (w[real:] == 0).all()


@requires_gfx950
@pytest.mark.parametrize("T", [3, 4000])
def test_strided_inputs_and_outputs(T):
    logits, bias = inputs(type(CORRECTNESS_SHAPES[0])(T, 256, 8))
    rw, rids = moe_routing(logits, bias, 8, **ROUTER)
    padded = torch.full((T, 320), float("nan"), device="cuda")
    padded[:, :256] = logits
    transposed = logits.t().contiguous().t()
    bias_strided = torch.stack([bias, torch.full_like(bias, float("nan"))], dim=1)[:, 0]
    w_buf = torch.full((T + 2, 16), -7.0, device="cuda")
    id_buf = torch.full((T + 2, 16), -7, device="cuda", dtype=torch.int32)
    for x, b in ((padded[:, :256], bias), (transposed, bias), (logits, bias_strided)):
        w, ids = moe_routing(x, b, 8, out=(w_buf[:T, :8], id_buf[:T, :8]), **ROUTER)
        assert w.data_ptr() == w_buf.data_ptr() and ids.data_ptr() == id_buf.data_ptr()
        assert torch.equal(ids, rids) and torch.equal(w, rw)
    assert (w_buf[:, 8:] == -7).all() and (w_buf[T:] == -7).all() and (id_buf[:, 8:] == -7).all()


@requires_gfx950
@pytest.mark.parametrize("T", [37, 4000])
def test_column_offsets_past_int32(T):
    """Expert strides below 2**31 whose last column lies past 2**31 elements, for the logits and the bias."""
    E = 16
    stride = (1 << 31) // (E - 1) + 1
    logits, bias = _separated(T, E)
    bias = torch.linspace(-0.01, 0.01, E, device="cuda")
    rw, rids = moe_routing(logits, bias, 4, **ROUTER)
    storage = torch.empty((E - 1) * stride + T + 1, device="cuda")
    far_logits = storage.as_strided((T, E), (1, stride))
    far_bias = storage.as_strided((E, ), (stride, ), T)
    far_logits.copy_(logits)
    far_bias.copy_(bias)
    for x, b in ((far_logits, bias), (logits, far_bias), (far_logits, far_bias)):
        w, ids = moe_routing(x, b, 4, **ROUTER)
        assert torch.equal(ids, rids) and torch.equal(w, rw)


@requires_gfx950
def test_bf16_bias_matches_its_fp32_cast():
    logits, bias = inputs(type(CORRECTNESS_SHAPES[0])(300, 256, 8))
    bias = bias.to(torch.bfloat16)
    w, ids = moe_routing(logits, bias, 8, **ROUTER)
    rw, rids = moe_routing(logits, bias.float(), 8, **ROUTER)
    assert torch.equal(ids, rids) and torch.equal(w, rw)


@requires_gfx950
def test_empty_batch():
    w, ids = moe_routing(torch.empty(0, 256, device="cuda"), torch.zeros(256, device="cuda"), 8)
    assert w.shape == (0, 8) and ids.shape == (0, 8)


@requires_gfx950
@pytest.mark.parametrize("T", [2, 4000])
def test_graph_capture_replays_with_new_logits(T):
    logits, bias = inputs(type(CORRECTNESS_SHAPES[0])(T, 256, 8))
    n = torch.tensor([T], device="cuda", dtype=torch.int32)
    out = (torch.empty(T, 8, device="cuda"), torch.empty(T, 8, device="cuda", dtype=torch.int32))
    moe_routing(logits, bias, 8, num_valid_tokens=n, out=out, **ROUTER)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_routing(logits, bias, 8, num_valid_tokens=n, out=out, **ROUTER)
    new_logits, _ = inputs(type(CORRECTNESS_SHAPES[0])(T, 256, 8), seed=5)
    logits.copy_(new_logits)
    n.fill_(T // 2)
    graph.replay()
    rw, rids = moe_routing(new_logits, bias, 8, **ROUTER)
    assert torch.equal(out[1][:T // 2], rids[:T // 2]) and torch.equal(out[0][:T // 2], rw[:T // 2])
    assert (out[1][T // 2:] == -1).all()


@requires_gfx950
@pytest.mark.parametrize("dynamic", [False, True])
def test_torch_compile_matches_eager(dynamic):
    logits, bias = inputs(type(CORRECTNESS_SHAPES[0])(40, 256, 8))

    def route(x, b):
        return moe_routing(x, b, 8, **ROUTER)

    # Catalog dispatch imports the kernel module, which dynamo does not trace, so the op runs at a graph break.
    compiled = torch.compile(route, dynamic=dynamic)
    w, ids = compiled(logits, bias)
    rw, rids = route(logits, bias)
    assert torch.equal(ids, rids) and torch.equal(w, rw)
    if dynamic:
        logits2, _ = inputs(type(CORRECTNESS_SHAPES[0])(3500, 256, 8), seed=1)
        w2, ids2 = compiled(logits2, bias)
        rw2, rids2 = route(logits2, bias)
        assert torch.equal(ids2, rids2) and torch.equal(w2, rw2)


@requires_gfx950
@pytest.mark.parametrize("T", [3, 200])
@pytest.mark.parametrize("expert_start", [0, 8])
def test_feeds_moe(T, expert_start):
    """Routing output drives ``tlx.ops.moe`` like the same routing computed in torch."""
    K, I, H, E_local, E_total = 256, 256, 384, 8, 32
    gen = torch.Generator(device="cuda").manual_seed(T)
    x = torch.randn(T, K, device="cuda", generator=gen).to(torch.bfloat16)
    w13 = (torch.randn(E_local, 2 * I, K, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E_local, H, I, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    logits, bias = _separated(T, E_total, seed=T)
    weights, ids = moe_routing(logits, bias, 4)
    rw, rids = reference(logits, bias, 4)
    _check((weights, ids), (rw, rids))
    got = tlx_moe(x, weights, ids, w13, w2, expert_start=expert_start)
    torch.testing.assert_close(got, tlx_moe(x, rw, rids, w13, w2, expert_start=expert_start))


@requires_gfx950
@pytest.mark.parametrize("expert_start", [0, 8])
def test_moe_skips_unfilled_slots(expert_start):
    """``tlx.ops.moe`` routes an id -1 slot to no expert, like a zero-weight slot."""
    T, K, I, H, E_local, E_total = 4, 256, 256, 384, 8, 32
    gen = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(T, K, device="cuda", generator=gen).to(torch.bfloat16)
    w13 = (torch.randn(E_local, 2 * I, K, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E_local, H, I, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    logits, bias = _separated(T, E_total, seed=6)
    logits[:2, 2:] = float("nan")
    weights, ids = moe_routing(logits, bias, 4)
    assert (ids[:2, 2:] == -1).all() and (ids[2:] >= 0).all()
    zero_weight = torch.where(ids >= 0, ids, ids[:, :1])
    torch.testing.assert_close(tlx_moe(x, weights, ids, w13, w2, expert_start=expert_start),
                               tlx_moe(x, weights, zero_weight, w13, w2, expert_start=expert_start))


@requires_gfx950
@pytest.mark.parametrize("topk", [1, 8])
@pytest.mark.parametrize("row_stride", [0, "short"])
def test_rejects_out_rows_that_overlap(topk, row_stride):
    T = 4
    logits, bias = torch.randn(T, 256, device="cuda"), torch.zeros(256, device="cuda")
    stride = 0 if row_stride == 0 else max(topk - 1, 0)
    w = torch.empty(T * topk, device="cuda").as_strided((T, topk), (stride, 1))
    ids = torch.empty(T, topk, device="cuda", dtype=torch.int32)
    for out in ((w, ids), (ids.float(), ids.view(-1).as_strided((T, topk), (stride, 1)))):
        with pytest.raises(InvalidInput):
            moe_routing(logits, bias, topk, out=out)


@requires_gfx950
def test_rejects_invalid_inputs():
    logits = torch.randn(4, 256, device="cuda")
    bias = torch.zeros(256, device="cuda")
    bad = [
        dict(router_logits=logits.half()),
        dict(router_logits=logits[0]),
        dict(correction_bias=torch.zeros(255, device="cuda")),
        dict(correction_bias=torch.zeros(256, device="cuda", dtype=torch.int32)),
        dict(topk=0),
        dict(topk=33),
        dict(topk=8.0),
        dict(router_logits=torch.randn(4, 2048, device="cuda"), correction_bias=torch.zeros(2048, device="cuda")),
        dict(router_logits=torch.randn(4, 4, device="cuda"), correction_bias=torch.zeros(4, device="cuda")),
        dict(route_scale=torch.tensor(1.0)),
        dict(num_valid_tokens=torch.tensor([1, 2], device="cuda", dtype=torch.int32)),
        dict(num_valid_tokens=torch.tensor([1.0], device="cuda")),
        dict(out=(torch.empty(4, 8, device="cuda"), )),
        dict(out=(torch.empty(4, 8, device="cuda"), torch.empty(4, 8, device="cuda"))),
        dict(out=(torch.empty(8, 4, device="cuda").t(), torch.empty(4, 8, device="cuda", dtype=torch.int32))),
        dict(out=(logits[:, :8], torch.empty(4, 8, device="cuda", dtype=torch.int32))),
        dict(correction_bias=torch.zeros(256)),
        dict(space="full"),
    ]
    for override in bad:
        args = dict(router_logits=logits, correction_bias=bias, topk=8)
        args.update(override)
        with pytest.raises(InvalidInput):
            moe_routing(args.pop("router_logits"), args.pop("correction_bias"), args.pop("topk"), **args)
    grad_weights = torch.empty(4, 8, device="cuda", requires_grad=True)
    with pytest.raises(UnsupportedBackward):
        moe_routing(logits, bias, 8, out=(grad_weights, torch.empty(4, 8, device="cuda", dtype=torch.int32)))
    with pytest.raises(UnsupportedBackward):
        moe_routing(logits.requires_grad_(), bias, 8)
