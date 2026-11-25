"""``tlx.ops.moe_mxfp4_sort_quant`` (gfx950) against a PyTorch model of AITER's expert-masked ``moe_sorting``,
``dynamic_mxfp4_quant`` and ``moe_mxfp4_sort``: every output byte that the A4W4 stage kernels read must match."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx import ops
from triton.tlx.ops import InvalidInput, UnsupportedBackward
from triton.tlx.ops.kernels.fused.moe_mxfp4_sort_quant import gfx950 as impl
from triton.tlx.ops.kernels.fused.moe_mxfp4_sort_quant._shapes import (CORRECTNESS_SHAPES, MoeMxfp4SortQuantShape,
                                                                       inputs, label, quant_reference, scale_offset,
                                                                       sort_reference)

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")


def _check(got, kw):
    q, sorted_ids, sorted_eids, num_valid, scales = got
    x, ids, start, nloc = kw["x"], kw["topk_ids"], kw["local_expert_start"], kw["num_local_experts"]
    T, K = x.shape
    rq, rs = quant_reference(x)
    rows, eids = sort_reference(ids, start, nloc)
    n = len(rows)
    assert torch.equal(q.cpu(), rq)
    assert num_valid.tolist() == [n, T]
    assert sorted_ids[:n].tolist() == rows
    assert sorted_eids[:n // 32].tolist() == eids
    assert sorted_ids.numel() == impl.sorted_capacity(T, ids.shape[1], kw["num_experts"], 32)
    flat = scales.view(-1).cpu()
    groups = K // 32
    for r, v in enumerate(rows):
        t = v & 0xFFFFFF
        if t < T:
            off = torch.tensor([scale_offset(r, c, groups) for c in range(groups)])
            assert torch.equal(flat[off], rs[t]), (r, t)


@requires_gfx950
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=label)
def test_matches_reference_registered_shapes(shape):
    kw = inputs(shape)
    _check(ops.moe_mxfp4_sort_quant(**kw), kw)


@requires_gfx950
@pytest.mark.parametrize("T", [1, 2, 3, 7, 16, 17, 32, 33, 64])
@pytest.mark.parametrize("local_start", [0, 96, 224])
@pytest.mark.parametrize("K", [4096, 1024])
def test_matches_reference(T, local_start, K):
    kw = inputs(MoeMxfp4SortQuantShape(T, K, 256, 8, 32, local_start), seed=T * 13 + local_start)
    kw["x"][-1, 32:64] = 1e30  # a group whose scale is near the top of the E8M0 range
    _check(ops.moe_mxfp4_sort_quant(**kw), kw)


@requires_gfx950
def test_every_bf16_group_maximum():
    """Each positive finite BF16 value is the largest magnitude of one group; the remaining groups hold 1.75 * 2**127
    throughout. Group maxima from 1.75 * 2**127 up select scale byte 254, whose factor 2**-127 is subnormal in fp32."""
    T, K = 64, 16384
    amax = torch.arange(1, 0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    v = torch.full((T * K // 32, ), 1.75 * 2.0**127)
    v[:amax.numel()] = amax
    c = torch.rand(v.numel(), 32, generator=torch.Generator().manual_seed(0)) * 2 - 1
    c[:, 0] = 1
    c[amax.numel():] = 1
    kw = inputs(MoeMxfp4SortQuantShape(T, K, 256, 8, 32, 64))
    kw["x"] = (v[:, None] * c).to(torch.bfloat16).view(T, K).to(kw["x"].device)
    _check(ops.moe_mxfp4_sort_quant(**kw), kw)


@requires_gfx950
@pytest.mark.parametrize("T,K", [(4, 256), (4, 768), (17, 1280), (33, 1792), (64, 768)])
def test_k_not_a_multiple_of_the_default_tile(T, K):
    kw = inputs(MoeMxfp4SortQuantShape(T, K, 256, 8, 32, 96), seed=T + K)
    _check(ops.moe_mxfp4_sort_quant(**kw), kw)


@requires_gfx950
@pytest.mark.parametrize("topk", [1, 2, 3, 5, 6, 7])
@pytest.mark.parametrize("T", [5, 33])
def test_topk(T, topk):
    kw = inputs(MoeMxfp4SortQuantShape(T, 1024, 64, topk, 16, 16), seed=T * 7 + topk)
    _check(ops.moe_mxfp4_sort_quant(**kw), kw)


@requires_gfx950
@pytest.mark.parametrize("T,experts,local,start", [(64, 256, 24, 40), (33, 256, 5, 0), (7, 64, 1, 63),
                                                   (64, 128, 48, 80)])
def test_local_expert_count_not_a_power_of_two(T, experts, local, start):
    kw = inputs(MoeMxfp4SortQuantShape(T, 1024, experts, 8, local, start), seed=T + local)
    _check(ops.moe_mxfp4_sort_quant(**kw), kw)


@requires_gfx950
def test_no_local_slots():
    ids = torch.full((4, 8), 255, dtype=torch.int32)
    ids[:, 1:] = torch.arange(200, 207, dtype=torch.int32)
    x = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
    *_, num_valid, _ = ops.moe_mxfp4_sort_quant(x, ids.cuda(), num_local_experts=32, num_experts=256)
    assert num_valid.tolist() == [0, 4]


@requires_gfx950
def test_rows_do_not_depend_on_the_batch():
    kw = inputs(MoeMxfp4SortQuantShape(64, 4096, 256, 8, 32, 64), seed=5)
    full = ops.moe_mxfp4_sort_quant(**kw)[0]
    part = ops.moe_mxfp4_sort_quant(**{**kw, "x": kw["x"][:9], "topk_ids": kw["topk_ids"][:9].contiguous()})[0]
    assert torch.equal(full[:9], part)


@requires_gfx950
def test_rejects_unsupported():
    kw = inputs(MoeMxfp4SortQuantShape(4, 4096, 256, 8, 32, 64))
    bad = (dict(x=kw["x"].half()), dict(topk_ids=kw["topk_ids"].long()), dict(block_m=64),
           dict(x=torch.zeros(65, 4096, dtype=torch.bfloat16, device="cuda"),
                topk_ids=torch.zeros(65, 8, dtype=torch.int32,
                                     device="cuda")), dict(x=kw["x"][:, :1000]), dict(local_expert_start=240),
           dict(topk_ids=kw["topk_ids"].t()), dict(topk_ids=kw["topk_ids"][:, :0].contiguous()),
           dict(num_local_experts=65), dict(topk_ids=torch.cat([kw["topk_ids"], kw["topk_ids"][:, :1] + 1], dim=1)))
    for change in bad:
        with pytest.raises(InvalidInput):
            ops.moe_mxfp4_sort_quant(**{**kw, **change})
    with pytest.raises(UnsupportedBackward):
        ops.moe_mxfp4_sort_quant(**{**kw, "x": kw["x"].clone().requires_grad_()})


@requires_gfx950
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=label)
def test_torch_reference_matches(shape):
    """The benchmark's vectorized PyTorch reference gives the CPU model's bytes."""
    from triton.tlx.ops.kernels.fused.moe_mxfp4_sort_quant._shapes import reference_torch

    kw = inputs(shape)
    _check(reference_torch(**kw), kw)
