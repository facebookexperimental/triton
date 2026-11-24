"""Public ``tlx.ops.moe_finalize`` coverage for sm100, forward and backward."""

import pytest
import torch

from triton._internal_testing import is_blackwell
from triton.tlx.ops import InvalidInput, UnsupportedBackward, moe_finalize
from triton.tlx.ops._catalog import CATALOG
from triton.tlx.ops.kernels.moe_finalize import sm100
from triton.tlx.ops.kernels.moe_finalize.sm100 import finalize_backward
from triton.tlx.ops.kernels.moe_finalize._shapes import (CORRECTNESS_SHAPES, EPS, MoeFinalizeShape, inputs, label,
                                                         reference, unfused_finalize, unpermute)

requires_sm100 = pytest.mark.skipif(not is_blackwell(), reason="Requires sm100")
NORMS = [(True, True), (True, False), (False, True), (False, False)]  # (input_scale, has weight)
BITWISE_SHAPES = [
    MoeFinalizeShape(1, 4096, 8, 256, 64),
    MoeFinalizeShape(151, 4096, 8, 256, 64),
    MoeFinalizeShape(300, 2048, 4, 64, 8),
    MoeFinalizeShape(512, 1536, 8, 256, 256),
    MoeFinalizeShape(64, 8192, 8, 256, 256),
    MoeFinalizeShape(37, 1000, 2, 16, 4),
    MoeFinalizeShape(63, 1000, 3, 16, 4),
]


def _norm(case, input_scale, has_weight):
    return dict(norm_weight=case["norm_weight"] if has_weight else None, input_scale=input_scale, eps=EPS)


def _gout(T, H, seed=11):
    return torch.randn(T, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed)).bfloat16()


def _grads(fn, leaves, gout):
    for leaf in leaves:
        leaf.grad = None
    fn().backward(gout)
    return [leaf.grad for leaf in leaves]


def _leaves(case, has_weight):
    y = case["expert_out"].clone().requires_grad_()
    w = case["topk_weights"].clone().requires_grad_()
    gamma = case["norm_weight"].clone().requires_grad_() if has_weight else None
    return y, w, gamma, [t for t in (y, w, gamma) if t is not None]


def _ideal(dense, topk_weights, norm_weight, input_scale, live):
    """The op's math in fp64 with no rounding after the inputs."""
    x = dense.double()
    w = norm_weight.double() if norm_weight is not None else None
    if input_scale:
        n = x * w if w is not None else x
        y = n * torch.rsqrt(n.pow(2).mean(-1, keepdim=True) + EPS)
    else:
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
        if w is not None:
            y = y * w
    return (y * topk_weights.double()[:, :, None] * live[:, :, None]).sum(1)


def _err(got, ideal):
    d = got.double() - ideal.double()
    return (d.norm() / ideal.double().norm()).item(), (d.abs().max() / ideal.double().abs().max()).item()


def test_catalog_entry():
    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("moe_finalize", "sm100")]
    assert len(specs) == 1
    spec, = specs
    assert spec.dtypes == frozenset({"bfloat16"})
    assert spec.supports_backward


@requires_sm100
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=label)
def test_matches_reference_registered_shapes(shape):
    case = inputs(shape)
    got = moe_finalize(case["expert_out"], case["topk_weights"], row_index=case["row_index"], **_norm(case, True, True))
    ref = reference(unpermute(case["expert_out"], case["row_index"]), case["topk_weights"], case["norm_weight"])
    # The reference sums squares in another order; a different rstd rounding can move an output by an ulp.
    assert (got == ref).float().mean() > 0.99
    torch.testing.assert_close(got.float(), ref.float(), rtol=8e-3, atol=8e-3)


@requires_sm100
@pytest.mark.parametrize("shape", BITWISE_SHAPES, ids=label)
@pytest.mark.parametrize("input_scale,has_weight", NORMS)
def test_bitwise_equal_to_the_unfused_finalize(shape, input_scale, has_weight):
    case = inputs(shape, seed=1)
    norm = _norm(case, input_scale, has_weight)
    dense = unpermute(case["expert_out"], case["row_index"])
    want = unfused_finalize(dense, case["topk_weights"], norm["norm_weight"], input_scale)
    by_rows = moe_finalize(case["expert_out"], case["topk_weights"], row_index=case["row_index"], **norm)
    masked = moe_finalize(dense, case["topk_weights"], topk_ids=case["topk_ids"], expert_mask=case["expert_mask"],
                          **norm)
    assert torch.equal(by_rows, want) and torch.equal(masked, want)


@requires_sm100
@pytest.mark.parametrize("H", [4096, 1024, 512])
def test_slots_are_summed_in_order(H):
    # Routing weights 2**-5 apart per slot: the BF16 terms span more than FP32's 24 bits, so the sum depends on the
    # order of the additions. After the BF16 rounding of the output few elements still differ (19 of 4M at H = 4096),
    # so the batch is large enough for the reversed order to show. Short rows check that the tile and the slot loop
    # (``row_index``) keep the same order when the tile has fewer columns than eight warps cover.
    case = inputs(MoeFinalizeShape(1024, H, 8, 256, 256), seed=8)
    spread = case["topk_weights"] * 2.0**(-5 * torch.arange(8, device="cuda"))
    dense = unpermute(case["expert_out"], case["row_index"])
    got = moe_finalize(dense, spread, eps=EPS)
    assert torch.equal(got, unfused_finalize(dense, spread, None, True))
    assert torch.equal(got, moe_finalize(case["expert_out"], spread, row_index=case["row_index"], eps=EPS))
    terms = moe_finalize(dense.flatten(0, 1)[:, None].contiguous(), spread.reshape(-1, 1), eps=EPS).float()
    terms = terms.view(1024, 8, -1)
    assert torch.equal(got, terms.cumsum(1)[:, -1].to(torch.bfloat16))
    if H == 4096:
        assert not torch.equal(got, terms.flip(1).cumsum(1)[:, -1].to(torch.bfloat16))


@requires_sm100
@pytest.mark.parametrize("input_scale,has_weight", NORMS)
def test_layouts_are_bitwise_identical(input_scale, has_weight):
    case = inputs(MoeFinalizeShape(200, 4096, 8, 256, 64), seed=2)
    norm = _norm(case, input_scale, has_weight)
    dense = unpermute(case["expert_out"], case["row_index"])
    by_rows = moe_finalize(case["expert_out"], case["topk_weights"], row_index=case["row_index"], **norm)
    masked = moe_finalize(dense, case["topk_weights"], topk_ids=case["topk_ids"], expert_mask=case["expert_mask"],
                          **norm)
    every_slot = moe_finalize(dense, case["topk_weights"], **norm)
    assert torch.equal(by_rows, masked) and torch.equal(by_rows, every_slot)


@requires_sm100
def test_slots_that_do_not_contribute_are_not_read():
    case = inputs(MoeFinalizeShape(64, 4096, 8, 256, 64), seed=3)
    want = moe_finalize(case["expert_out"], case["topk_weights"], row_index=case["row_index"])
    dense = unpermute(case["expert_out"], case["row_index"])
    dead = case["row_index"] < 0
    dense[dead] = float("nan")
    ids = case["topk_ids"].clone()
    slot = torch.arange(8, device="cuda")
    ids[dead & (slot % 3 == 1)] = -1
    ids[dead & (slot % 3 == 2)] = 256
    got = moe_finalize(dense, case["topk_weights"], topk_ids=ids, expert_mask=case["expert_mask"].int())
    assert torch.equal(got, want)
    padded = torch.cat([case["expert_out"], torch.full_like(case["expert_out"][:7], float("nan"))])
    assert torch.equal(moe_finalize(padded, case["topk_weights"], row_index=case["row_index"]), want)


@requires_sm100
@pytest.mark.parametrize("input_scale,has_weight", NORMS)
def test_eps_zero_skips_dead_and_padding_slots(input_scale, has_weight):
    # K = 3 pads the slot tile to 4 rows. A zero row normalized with eps = 0 would give 0 * rsqrt(0) = NaN.
    case = inputs(MoeFinalizeShape(63, 1024, 3, 16, 4), seed=6)
    norm = dict(norm_weight=case["norm_weight"] if has_weight else None, input_scale=input_scale, eps=0.0)
    dense = unpermute(case["expert_out"], case["row_index"])
    by_rows = moe_finalize(case["expert_out"], case["topk_weights"], row_index=case["row_index"], **norm)
    masked = moe_finalize(dense, case["topk_weights"], topk_ids=case["topk_ids"], expert_mask=case["expert_mask"],
                          **norm)
    assert torch.isfinite(by_rows.float()).all() and torch.equal(by_rows, masked)
    ref = reference(dense, case["topk_weights"], norm["norm_weight"], input_scale, 0.0, live=case["row_index"] >= 0)
    assert (by_rows == ref).float().mean() > 0.99
    every = inputs(MoeFinalizeShape(63, 1024, 3, 16, 16), seed=6)
    by_rows = moe_finalize(every["expert_out"], every["topk_weights"], row_index=every["row_index"], **norm)
    every_slot = moe_finalize(unpermute(every["expert_out"], every["row_index"]), every["topk_weights"], **norm)
    assert torch.isfinite(by_rows.float()).all() and torch.equal(by_rows, every_slot)


@requires_sm100
def test_tokens_without_local_slots_get_zeros():
    case = inputs(MoeFinalizeShape(50, 4096, 8, 256, 64), seed=4)
    row_index = case["row_index"].clone()
    row_index[::3] = -1
    got = moe_finalize(case["expert_out"], case["topk_weights"], row_index=row_index)
    assert (got[::3] == 0).all()
    dense = unpermute(case["expert_out"], row_index)
    ids = torch.where(row_index >= 0, case["topk_ids"], -1)
    assert torch.equal(moe_finalize(dense, case["topk_weights"], topk_ids=ids, expert_mask=case["expert_mask"]), got)


@requires_sm100
@pytest.mark.parametrize("layout", ["rows", "dense"])
def test_rows_do_not_depend_on_the_batch(layout):
    case = inputs(MoeFinalizeShape(1000, 4096, 8, 256, 256), seed=5)
    dense = unpermute(case["expert_out"], case["row_index"])

    def run(lo, hi):
        w = case["topk_weights"][lo:hi].contiguous()
        if layout == "rows":
            return moe_finalize(case["expert_out"], w, row_index=case["row_index"][lo:hi].contiguous())
        return moe_finalize(dense[lo:hi].contiguous(), w)

    full = run(0, 1000)
    for lo, hi in ((0, 1), (17, 18), (300, 451), (999, 1000)):
        assert torch.equal(run(lo, hi), full[lo:hi])


@requires_sm100
def test_row_order_does_not_matter():
    case = inputs(MoeFinalizeShape(300, 4096, 8, 256, 64), seed=6)
    want = moe_finalize(case["expert_out"], case["topk_weights"], row_index=case["row_index"])
    perm = torch.randperm(case["expert_out"].shape[0], device="cuda")
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device="cuda")
    row_index = torch.where(case["row_index"] >= 0, inverse[case["row_index"].long().clamp(min=0)], -1)
    got = moe_finalize(case["expert_out"][perm].contiguous(), case["topk_weights"], row_index=row_index.to(torch.int64))
    assert torch.equal(got, want)


@requires_sm100
def test_offsets_past_int32():
    H = 4096
    case = inputs(MoeFinalizeShape(4, H, 8, 256, 64), seed=7)
    want = moe_finalize(case["expert_out"], case["topk_weights"], row_index=case["row_index"])
    # (T, K, H) slots past 2**31 elements; only the last tokens have local slots.
    T = (1 << 31) // (8 * H) + 2
    dense = torch.zeros(T, 8, H, device="cuda", dtype=torch.bfloat16)
    dense[-4:] = unpermute(case["expert_out"], case["row_index"])
    topk_weights = torch.zeros(T, 8, device="cuda")
    topk_weights[-4:] = case["topk_weights"]
    ids = torch.full((T, 8), -1, device="cuda", dtype=torch.int32)
    ids[-4:] = case["topk_ids"]
    got = moe_finalize(dense, topk_weights, topk_ids=ids, expert_mask=case["expert_mask"])
    assert torch.equal(got[-4:], want) and (got[:-4] == 0).all()
    # The backward addresses the same slots and the gradient rows past 2**31 elements.
    dense.requires_grad_()
    gout = torch.zeros(T, H, device="cuda", dtype=torch.bfloat16)
    gout[-4:] = _gout(4, H)
    moe_finalize(dense, topk_weights, topk_ids=ids, expert_mask=case["expert_mask"]).backward(gout)
    small = unpermute(case["expert_out"], case["row_index"]).requires_grad_()
    moe_finalize(small, case["topk_weights"], topk_ids=case["topk_ids"],
                 expert_mask=case["expert_mask"]).backward(gout[-4:])
    assert torch.equal(dense.grad[-4:], small.grad) and not dense.grad[:-4].any()


@requires_sm100
def test_empty_batch():
    case = inputs(MoeFinalizeShape(3, 4096, 8, 256, 64))
    y, w, gamma, leaves = _leaves(case, True)
    out = moe_finalize(y, w[:0], row_index=case["row_index"][:0], norm_weight=gamma)
    assert out.shape == (0, 4096)
    out.sum().backward()
    assert not y.grad.any() and not gamma.grad.any()


@requires_sm100
@pytest.mark.parametrize("T", [2, 3000])
def test_graph_capture_replays_with_new_inputs(T):
    shape = MoeFinalizeShape(T, 4096, 8, 256, 64)
    case = inputs(shape)
    new = inputs(shape, seed=9)
    rows = max(case["expert_out"].shape[0], new["expert_out"].shape[0])
    expert_out = torch.zeros(rows, 4096, device="cuda", dtype=torch.bfloat16)
    args = dict(row_index=case["row_index"].clone(), norm_weight=case["norm_weight"].clone(), eps=EPS)
    topk_weights = case["topk_weights"].clone()
    out = torch.empty(T, 4096, device="cuda", dtype=torch.bfloat16)
    moe_finalize(expert_out, topk_weights, out=out, **args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_finalize(expert_out, topk_weights, out=out, **args)
    expert_out[:new["expert_out"].shape[0]] = new["expert_out"]
    topk_weights.copy_(new["topk_weights"])
    args["row_index"].copy_(new["row_index"])
    args["norm_weight"].copy_(new["norm_weight"])
    graph.replay()
    want = moe_finalize(new["expert_out"], new["topk_weights"], row_index=new["row_index"],
                        norm_weight=new["norm_weight"], eps=EPS)
    assert torch.equal(out, want)


@requires_sm100
def test_torch_compile_matches_eager():
    case = inputs(MoeFinalizeShape(40, 4096, 8, 256, 256))
    gout = _gout(40, 4096)

    def finalize(y, w, gamma):
        return moe_finalize(y, w, row_index=case["row_index"], norm_weight=gamma)

    # Catalog dispatch imports the kernel module, which dynamo does not trace, so the op runs at a graph break.
    y, w, gamma, leaves = _leaves(case, True)
    eager = _grads(lambda: finalize(y, w, gamma), leaves, gout)
    compiled = _grads(lambda: torch.compile(finalize)(y, w, gamma), leaves, gout)
    assert all(torch.equal(a, b) for a, b in zip(eager, compiled))


@requires_sm100
@pytest.mark.parametrize("shape", [
    MoeFinalizeShape(37, 1000, 3, 16, 16),
    MoeFinalizeShape(300, 2048, 4, 64, 8),
    MoeFinalizeShape(512, 1536, 8, 256, 256),
    MoeFinalizeShape(64, 8192, 8, 256, 256),
], ids=label)
@pytest.mark.parametrize("input_scale,has_weight", NORMS)
def test_backward_no_less_accurate_than_the_eager_chain(shape, input_scale, has_weight):
    """Against the fp64 math, every gradient is within 1.1x of the eager chain's relative L2 error and 1.5x of its
    largest element error; the chain is ``reference`` on the gathered layout, differentiated by autograd."""
    case = inputs(shape, seed=1)
    gout = _gout(shape.tokens, shape.H)
    live = (case["row_index"] >= 0).double()
    rows = case["row_index"]
    y, w, gamma, leaves = _leaves(case, has_weight)
    got = _grads(lambda: moe_finalize(y, w, row_index=rows, norm_weight=gamma, input_scale=input_scale, eps=EPS),
                 leaves, gout)
    chain = _grads(lambda: reference(unpermute(y, rows), w, gamma, input_scale, EPS, live=live), leaves, gout)
    ideal = _grads(lambda: _ideal(unpermute(y, rows), w, gamma, input_scale, live), leaves, gout.double())
    for name, g, c, i in zip(("expert_out", "topk_weights", "norm_weight"), got, chain, ideal):
        g_l2, g_max = _err(g, i)
        c_l2, c_max = _err(c, i)
        assert g_l2 <= 1.1 * c_l2 + 1e-7, (name, g_l2, c_l2)
        assert g_max <= 1.5 * c_max + 1e-7, (name, g_max, c_max)


@requires_sm100
@pytest.mark.parametrize("input_scale,has_weight", NORMS)
def test_backward_layouts_are_bitwise_identical(input_scale, has_weight):
    case = inputs(MoeFinalizeShape(200, 2048, 8, 256, 64), seed=2)
    gout = _gout(200, 2048)
    norm = dict(input_scale=input_scale, eps=EPS)
    y, w, gamma, leaves = _leaves(case, has_weight)
    g_rows = _grads(lambda: moe_finalize(y, w, row_index=case["row_index"], norm_weight=gamma, **norm), leaves, gout)
    dense = unpermute(case["expert_out"], case["row_index"]).requires_grad_()
    w2 = case["topk_weights"].clone().requires_grad_()
    gamma2 = case["norm_weight"].clone().requires_grad_() if has_weight else None
    leaves2 = [t for t in (dense, w2, gamma2) if t is not None]
    g_masked = _grads(
        lambda: moe_finalize(dense, w2, topk_ids=case["topk_ids"], expert_mask=case["expert_mask"], norm_weight=gamma2,
                             **norm), leaves2, gout)
    live = case["row_index"] >= 0
    assert torch.equal(g_masked[0][live], g_rows[0][case["row_index"][live].long()])
    assert not g_masked[0][~live].any()
    assert torch.equal(g_masked[1], g_rows[1]) and not g_rows[1][~live].any()
    if has_weight:
        # The weight gradient sums per-program partials whose split follows the slot loop of each layout.
        torch.testing.assert_close(g_masked[2].float(), g_rows[2].float(), rtol=1e-2, atol=1e-2)


@requires_sm100
@pytest.mark.parametrize("input_scale,has_weight", NORMS)
def test_backward_eps_zero_dead_and_padding_slots_get_zero_gradient(input_scale, has_weight):
    # 63 tokens x 3 slots leave padding rows in the last slot block, which must not reach the weight gradient.
    case = inputs(MoeFinalizeShape(63, 1024, 3, 16, 4), seed=6)
    gout = _gout(63, 1024)
    norm = dict(input_scale=input_scale, eps=0.0)
    y, w, gamma, leaves = _leaves(case, has_weight)
    g_rows = _grads(lambda: moe_finalize(y, w, row_index=case["row_index"], norm_weight=gamma, **norm), leaves, gout)
    dense = unpermute(case["expert_out"], case["row_index"]).requires_grad_()
    w2 = case["topk_weights"].clone().requires_grad_()
    gamma2 = case["norm_weight"].clone().requires_grad_() if has_weight else None
    leaves2 = [t for t in (dense, w2, gamma2) if t is not None]
    g_masked = _grads(
        lambda: moe_finalize(dense, w2, topk_ids=case["topk_ids"], expert_mask=case["expert_mask"], norm_weight=gamma2,
                             **norm), leaves2, gout)
    assert all(torch.isfinite(g.float()).all() for g in g_rows + g_masked)
    live = case["row_index"] >= 0
    assert not g_masked[0][~live].any() and not g_masked[1][~live].any() and not g_rows[1][~live].any()
    assert torch.equal(g_masked[0][live], g_rows[0][case["row_index"][live].long()])
    assert torch.equal(g_masked[1], g_rows[1])
    if has_weight:
        torch.testing.assert_close(g_masked[2].float(), g_rows[2].float(), rtol=1e-2, atol=1e-2)


@requires_sm100
def test_backward_rows_no_slot_names_get_zero_gradient():
    case = inputs(MoeFinalizeShape(64, 4096, 8, 256, 64), seed=3)
    padded = torch.cat([case["expert_out"], torch.randn_like(case["expert_out"][:7])]).requires_grad_()
    moe_finalize(padded, case["topk_weights"], row_index=case["row_index"]).backward(_gout(64, 4096))
    assert not padded.grad[-7:].any() and padded.grad[:-7].any()


@requires_sm100
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_rows_past_the_end_do_not_contribute(index_dtype):
    case = inputs(MoeFinalizeShape(64, 4096, 8, 256, 64), seed=3)
    R, H = case["expert_out"].shape
    rows = case["row_index"].to(index_dtype)
    past = rows.clone()
    dead = past < 0
    past[dead & (torch.arange(8, device="cuda") % 2 == 0)] = R
    past[dead & (torch.arange(8, device="cuda") % 2 == 1)] = torch.iinfo(index_dtype).max
    big = torch.full((R + 8, H), float("nan"), device="cuda", dtype=torch.bfloat16)
    big[:R] = case["expert_out"]
    y = big[:R]
    want = moe_finalize(y, case["topk_weights"], row_index=rows, norm_weight=case["norm_weight"])
    assert torch.equal(moe_finalize(y, case["topk_weights"], row_index=past, norm_weight=case["norm_weight"]), want)

    gout = _gout(64, H)
    rest = (None, None, case["norm_weight"], True, EPS)
    grad_buf = torch.full((R + 8, H), 7.0, device="cuda", dtype=torch.bfloat16)
    gy, gs, gw = finalize_backward(gout, y, case["topk_weights"], past, *rest, grad_expert_out=grad_buf[:R])
    want_gy, want_gs, want_gw = finalize_backward(gout, y, case["topk_weights"], rows, *rest)
    assert (grad_buf[R:] == 7.0).all()
    assert torch.equal(gy, want_gy) and torch.equal(gs, want_gs) and torch.equal(gw, want_gw)


@requires_sm100
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_backward_counts_the_slots_naming_each_row(index_dtype, monkeypatch):
    # A failed device-side assertion ends the process, so the test records the counts passed to the check instead.
    counts = []
    monkeypatch.setattr(sm100, "_check_rows_named_once", lambda named: counts.append(named.clone()))
    case = inputs(MoeFinalizeShape(64, 2048, 8, 256, 64), seed=5)
    rows = case["row_index"].to(index_dtype)
    twice = rows.clone()
    live = (rows >= 0).nonzero()
    twice[live[-1][0], live[-1][1]] = rows[live[0][0], live[0][1]]
    gout = _gout(64, 2048)
    for row_index in (rows, twice):
        finalize_backward(gout, case["expert_out"], case["topk_weights"], row_index, None, None, case["norm_weight"],
                          True, EPS)
    first = rows[live[0][0], live[0][1]].item()
    assert (counts[0] == 1).all()
    assert counts[1][first] == 2 and int(counts[1].sum()) == len(live)


@requires_sm100
def test_rows_named_twice_fail_a_device_assertion():
    named = torch.ones(5000, device="cuda", dtype=torch.int32)
    kernel = sm100._rows_named_once_kernel[(2, )](named, named.numel(), BLOCK=sm100._CHECK_BLOCK)
    torch.cuda.synchronize()
    assert "__assertfail" in kernel.asm["ptx"]


@requires_sm100
def test_backward_is_deterministic():
    case = inputs(MoeFinalizeShape(16384, 1536, 8, 256, 256), seed=4)
    gout = _gout(16384, 1536)
    y, w, gamma, leaves = _leaves(case, True)
    first = _grads(lambda: moe_finalize(y, w, row_index=case["row_index"], norm_weight=gamma), leaves, gout)
    second = _grads(lambda: moe_finalize(y, w, row_index=case["row_index"], norm_weight=gamma), leaves, gout)
    assert all(torch.equal(a, b) for a, b in zip(first, second))


@requires_sm100
def test_backward_returns_only_requested_gradients():
    case = inputs(MoeFinalizeShape(30, 2048, 8, 256, 256))
    w = case["topk_weights"].clone().requires_grad_()
    moe_finalize(case["expert_out"], w, row_index=case["row_index"], norm_weight=case["norm_weight"]).sum().backward()
    assert w.grad is not None and w.grad.shape == (30, 8)
    with torch.no_grad():
        y = case["expert_out"].clone().requires_grad_()
        assert not moe_finalize(y, case["topk_weights"], row_index=case["row_index"]).requires_grad


def _centered_weight(H, seed=5):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (0.1 * torch.randn(H, device="cuda", generator=gen)).bfloat16()


@requires_sm100
@pytest.mark.parametrize("layout", ["rows", "dense"])
@pytest.mark.parametrize("input_scale", [True, False])
def test_gain_center_equals_the_shifted_weight(layout, input_scale):
    """``gain_center=c`` gives the bits of ``norm_weight=bf16(w + c)``, forward and backward. The two calls compile
    separate kernels; with the weight applied after the norm, the FP32 gradient of the row may contract differently,
    so it is held to one BF16 ulp there."""
    shape = MoeFinalizeShape(300, 1536, 8, 256, 64)
    case = inputs(shape, seed=3)
    w = _centered_weight(shape.H)
    shifted = (w.float() + 1.0).bfloat16()
    if layout == "rows":
        y, kw = case["expert_out"], dict(row_index=case["row_index"])
    else:
        y = unpermute(case["expert_out"], case["row_index"])
        kw = dict(topk_ids=case["topk_ids"], expert_mask=case["expert_mask"])
    norm = dict(input_scale=input_scale, eps=EPS, **kw)
    gout = _gout(shape.tokens, shape.H)
    grads = []
    for weight, gain in ((w, 1.0), (shifted, 0.0)):
        leaves = [
            y.clone().requires_grad_(), case["topk_weights"].clone().requires_grad_(),
            weight.clone().requires_grad_()
        ]
        out = moe_finalize(leaves[0], leaves[1], norm_weight=leaves[2], gain_center=gain, **norm)
        out.backward(gout)
        grads.append([out.detach()] + [t.grad for t in leaves])
    (out, gy, gs, gw), (out_ref, gy_ref, gs_ref, gw_ref) = grads
    assert torch.equal(out, out_ref) and torch.equal(gs, gs_ref) and torch.equal(gw, gw_ref)
    if input_scale:
        assert torch.equal(gy, gy_ref)
    else:
        diff = gy.float() != gy_ref.float()
        assert diff.float().mean() <= 1e-4
        torch.testing.assert_close(gy.float(), gy_ref.float(), rtol=2**-7, atol=0)


@requires_sm100
def test_fp32_norm_weight_is_rounded_after_the_gain():
    """An FP32 weight is shifted, then rounded to BF16; its gradient is the same FP32 sum, returned in FP32."""
    shape = MoeFinalizeShape(300, 2048, 8, 256, 64)
    case = inputs(shape, seed=4)
    w32 = _centered_weight(shape.H).float() + 1e-3
    effective = (w32 + 1.0).bfloat16()
    gout = _gout(shape.tokens, shape.H)
    grads = []
    for weight, gain in ((w32, 1.0), (effective, 0.0)):
        leaf = weight.clone().requires_grad_()
        out = moe_finalize(case["expert_out"], case["topk_weights"], row_index=case["row_index"], norm_weight=leaf,
                           gain_center=gain, eps=EPS)
        out.backward(gout)
        grads.append((out.detach(), leaf.grad))
    (out32, g32), (out16, g16) = grads
    assert torch.equal(out32, out16)
    assert g32.dtype == torch.float32 and torch.equal(g32.bfloat16(), g16)


@requires_sm100
def test_rows_tagged_with_token_ids():
    """A token dispatcher's combine input (routed rows, each with its token id and score, K rows per token) maps to
    ``row_index`` by a stable sort of the token ids; with ``gain_center`` the result is the eager post-expert norm
    chain's (scale by ``bf16(w + 1)``, RMSNorm, router score, sum of the token's rows in row order)."""
    T, K, H, eps = 512, 8, 1536, 1e-5
    gen = torch.Generator(device="cuda").manual_seed(6)
    token_ids = torch.arange(T, device="cuda").repeat_interleave(K)[torch.randperm(T * K, device="cuda", generator=gen)]
    x = (torch.randn(T * K, H, device="cuda", generator=gen) * 3).bfloat16()
    scores = torch.rand(T * K, device="cuda", generator=gen)
    weight = _centered_weight(H)
    rows = torch.argsort(token_ids, stable=True).view(T, K)
    topk_weights = scores[rows].contiguous()
    got = moe_finalize(x, topk_weights, row_index=rows.to(torch.int32), norm_weight=weight, gain_center=1.0, eps=eps)

    w = (weight.float() + 1.0).bfloat16()
    scaled = (x * w).float()
    normed = (scaled * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + eps)).bfloat16()
    terms = (normed.float() * scores[:, None]).bfloat16().float()
    acc = terms[rows[:, 0]]
    for k in range(1, K):
        acc = acc + terms[rows[:, k]]
    want = acc.bfloat16()
    assert torch.equal(got, unfused_finalize(unpermute(x, rows), topk_weights, w, eps=eps))
    assert (got == want).float().mean() > 0.999
    torch.testing.assert_close(got.float(), want.float(), rtol=8e-3, atol=8e-3)


@requires_sm100
def test_rejects_invalid_inputs():
    case = inputs(MoeFinalizeShape(4, 4096, 8, 256, 64))
    y, w, rows = case["expert_out"], case["topk_weights"], case["row_index"]
    dense = unpermute(y, rows)
    bad = [
        dict(expert_out=y.float()),
        dict(expert_out=dense),
        dict(expert_out=y.t().contiguous().t()),
        dict(topk_weights=w.double()),
        dict(topk_weights=w.t().contiguous().t()),
        dict(row_index=rows.float()),
        dict(row_index=rows[:, :4].contiguous()),
        dict(topk_ids=case["topk_ids"], expert_mask=case["expert_mask"]),
        dict(row_index=None, expert_out=dense, topk_ids=case["topk_ids"]),
        dict(row_index=None, expert_out=dense[:, :4].contiguous()),
        dict(expert_out=torch.zeros(9, 16384, device="cuda", dtype=torch.bfloat16)),
        dict(norm_weight=case["norm_weight"].half()),
        dict(gain_center=1.0),
        dict(norm_weight=case["norm_weight"], gain_center=float("nan")),
        dict(norm_weight=case["norm_weight"], gain_center=True),
        dict(eps=-1.0),
        dict(out=torch.empty(4, 4096, device="cuda")),
        dict(out=y[:4]),
        dict(space="full"),
    ]
    for override in bad:
        args = dict(expert_out=y, topk_weights=w, row_index=rows)
        args.update(override)
        with pytest.raises(InvalidInput):
            moe_finalize(args.pop("expert_out"), args.pop("topk_weights"), **args)
    out = torch.empty(4, 4096, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(UnsupportedBackward):
        moe_finalize(y.clone().requires_grad_(), w, row_index=rows, out=out)
    with pytest.raises(UnsupportedBackward):
        moe_finalize(y, w, row_index=rows, out=out.clone().requires_grad_())
