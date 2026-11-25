"""Public ``tlx.ops.moe_router_topk_align`` coverage for gfx950."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import InvalidInput, UnsupportedBackward, moe_router_topk_align
from triton.tlx.ops._catalog import CATALOG
from triton.tlx.ops.kernels.moe.gfx950 import _align
from triton.tlx.ops.kernels.fused.moe_router_topk_align._shapes import (CORRECTNESS_SHAPES, MoeRouterTopkAlignShape,
                                                                        inputs, label, reference)

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
# The router GEMV, every split-K bucket through prefill sizes, and partial row tiles.
TOKENS = (1, 2, 5, 9, 10, 16, 17, 32, 33, 63, 64, 65, 151, 600, 2049)
# (local experts, first local expert): EP8 ranks at both ends, an EP4 rank, and every expert local.
LOCAL = ((32, 0), (32, 224), (64, 64), (256, 0))


def _ep8(T, local=32, start=64, block_m=16):
    return MoeRouterTopkAlignShape(T, 256, 8192, 8, local, start, block_m)


def _assert_equal(got, ref):
    names = ("topk_weights", "topk_ids", "sorted_ids", "expert_ids", "num_tokens_post_padded")
    for name, g, r in zip(names, got, ref, strict=True):
        assert g.dtype == r.dtype and g.shape == r.shape, name
        assert torch.equal(g, r), name


def test_catalog_entry():
    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("moe_router_topk_align", "gfx950")]
    assert len(specs) == 1
    spec, = specs
    assert spec.dtypes == frozenset({"bfloat16"})
    assert not spec.supports_backward
    assert spec.accepts({"E": 256, "K": 8192, "topk": 8}) and spec.accepts({"E": 1024, "K": 1024, "topk": 6})
    assert not spec.accepts({"E": 250, "K": 8192, "topk": 8}) and not spec.accepts({"E": 256, "K": 8000, "topk": 8})


@requires_gfx950
def test_router_partials_add_up_to_router_gemm():
    """The split-K partials the top-k kernel adds are router_gemm's, and so is the order it adds them in."""
    from triton.tlx.ops import router_gemm
    from triton.tlx.ops.kernels.router_gemm import gfx950 as router

    for T in (10, 16, 33, 151):
        args = inputs(_ep8(T), seed=T)
        partials, splits = router.router_gemm_partials(args["x"], args["weight"])
        assert splits > 1
        logits = partials[0].clone()
        for s in range(1, splits):
            logits += partials[s]
        assert torch.equal(logits, router_gemm(args["x"], args["weight"]))


@requires_gfx950
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=label)
def test_registered_shapes(shape):
    args = inputs(shape)
    _assert_equal(moe_router_topk_align(**args), reference(**args))


@requires_gfx950
@pytest.mark.parametrize("T", TOKENS)
@pytest.mark.parametrize("local,start", LOCAL)
def test_matches_op_chain(T, local, start):
    args = inputs(_ep8(T, local, start), seed=T)
    _assert_equal(moe_router_topk_align(**args), reference(**args))


@requires_gfx950
@pytest.mark.parametrize("T", [1, 16, 64])
@pytest.mark.parametrize("block_m", [32, 64])
def test_block_sizes(T, block_m):
    args = inputs(_ep8(T, block_m=block_m), seed=T)
    _assert_equal(moe_router_topk_align(**args), reference(**args))


@requires_gfx950
@pytest.mark.parametrize("T", [5, 40])
@pytest.mark.parametrize("topk", [1, 6])
def test_other_topk(T, topk):
    args = dict(inputs(MoeRouterTopkAlignShape(T, 128, 4096, topk, 64, 0, 16), seed=T), topk=topk)
    _assert_equal(moe_router_topk_align(**args), reference(**args))


@requires_gfx950
@pytest.mark.parametrize("T", [8, 40])
def test_matches_fp64_routing(T):
    """The selection and weights of a sigmoid router computed from the fp64 logits, where no two biased scores tie
    within the logits' rounding."""
    args = inputs(_ep8(T, 256, 0), seed=11)
    weights, ids, *_ = moe_router_topk_align(**args)
    logits = args["x"].double() @ args["weight"].double().T
    scores = torch.sigmoid(logits * args["router_multiplier"])
    biased = scores + args["correction_bias"].double()
    order = biased.sort(dim=1, descending=True)
    # Rows whose 8th and 9th biased scores are within 1e-5 of each other could rank either way: leave them out.
    clear = (order.values[:, 7] - order.values[:, 8]) > 1e-5
    assert clear.sum() >= T // 2
    ref_ids = order.indices[:, :8]
    picked = scores.gather(1, ref_ids)
    ref_weights = picked * (args["route_scale"] / (picked.sum(dim=1, keepdim=True) + args["eps"]))
    assert torch.equal(ids[clear].long(), ref_ids[clear])
    torch.testing.assert_close(weights[clear].double(), ref_weights[clear], rtol=1e-6, atol=0)


@requires_gfx950
def test_nan_rows_and_no_local_slots():
    args = inputs(_ep8(16), seed=5)
    args["x"][3] = float("nan")
    # Biased toward experts outside the local range, so most rows route nothing locally.
    args["correction_bias"][:128] += 4.0
    got = moe_router_topk_align(**args)
    _assert_equal(got, reference(**args))
    assert (got[1][3] == -1).all() and not got[0][3].any()


@requires_gfx950
@pytest.mark.parametrize("T", [4, 16, 64])
def test_graph_capture_replays_with_new_inputs(T):
    args = inputs(_ep8(T))
    moe_router_topk_align(**args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = moe_router_topk_align(**args)
    new = inputs(_ep8(T), seed=3)
    for name in ("x", "weight", "correction_bias"):
        args[name].copy_(new[name])
    for _ in range(3):
        graph.replay()
    _assert_equal(got, reference(**args))


@requires_gfx950
@pytest.mark.parametrize("T", [3, 40, 100])
def test_row_strided_x(T):
    args = inputs(_ep8(T), seed=T)
    wide = torch.zeros(T, 8192 + 64, device="cuda", dtype=torch.bfloat16)
    wide[:, 32:32 + 8192] = args["x"]
    _assert_equal(moe_router_topk_align(**dict(args, x=wide[:, 32:32 + 8192])), reference(**args))


@requires_gfx950
def test_torch_compile():
    args = inputs(_ep8(33))
    compiled = torch.compile(lambda a: moe_router_topk_align(**a))
    _assert_equal(compiled(args), moe_router_topk_align(**args))


@requires_gfx950
def test_zero_tokens():
    args = inputs(_ep8(1))
    args["x"] = args["x"][:0]
    weights, ids, sorted_ids, expert_ids, num_post = moe_router_topk_align(**args)
    assert weights.shape == ids.shape == (0, 8) and num_post.item() == 0
    assert torch.equal(sorted_ids, _align(ids, 32, 64, 16)[0])
    assert (expert_ids == -1).all()


@requires_gfx950
def test_invalid_inputs():
    args = inputs(_ep8(4))
    bad = (
        dict(x=args["x"].float()),
        dict(weight=args["weight"].bfloat16()),
        dict(correction_bias=args["correction_bias"].bfloat16()),
        dict(x=args["x"][0]),
        dict(weight=args["weight"][:, :4096]),
        dict(weight=args["weight"][:250].contiguous(), correction_bias=args["correction_bias"][:250]),
        dict(x=args["x"].t().contiguous().t()),
        dict(topk=33),
        dict(topk=0),
        dict(num_local_experts=0),
        dict(num_local_experts=32, local_expert_start=240),
        dict(block_m=8),
        dict(block_m=24),
        dict(router_multiplier=None),
        dict(weight=args["weight"].cpu()),
    )
    for change in bad:
        with pytest.raises(InvalidInput):
            moe_router_topk_align(**dict(args, **change))
    with pytest.raises(UnsupportedBackward):
        moe_router_topk_align(**dict(args, weight=args["weight"].clone().requires_grad_()))
