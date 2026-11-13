"""Public ``tlx.ops.norm_residual_norm`` coverage for gfx950."""

import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx import ops
from triton.tlx.ops import InvalidInput, UnsupportedBackward
from triton.tlx.ops._catalog import CATALOG
from triton.tlx.ops.kernels.fused.norm_residual_norm import gfx950 as impl
from triton.tlx.ops.kernels.fused.norm_residual_norm._shapes import (CORRECTNESS_SHAPES, NormResidualNormShape, inputs,
                                                                     label, reference)

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
# "persistent_streaming": the persistent kernel with non-temporal row loads and stores.
KERNELS = ("tlx", "early", "persistent", "persistent_streaming")


def _check(got, kw):
    out, new_stream = got
    ref_out, ref_stream = reference(**kw)
    # The branch norm rounds to BF16 before the add: a different reduction order can move it by one BF16 ulp,
    # which moves the stream by beta times that ulp.
    torch.testing.assert_close(new_stream, ref_stream, atol=4e-2, rtol=1e-5)
    assert ((new_stream - ref_stream).abs() > 1e-5 * ref_stream.abs() + 1e-6).float().mean() < 1e-3
    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=1e-2)
    assert (out != ref_out).float().mean() < 1e-3


def _force(monkeypatch, kind):
    monkeypatch.setattr(impl, "pick_kernel", lambda rows, H, num_cus: kind.removesuffix("_streaming"))
    monkeypatch.setattr(impl, "use_streaming_accesses", lambda rows, H: kind.endswith("_streaming"))


def test_catalog_entries():
    specs = {spec.op: spec for spec in CATALOG if spec.arch == "gfx950"}
    spec = specs["norm_residual_norm"]
    assert spec.dtypes == frozenset({"bfloat16"}) and not spec.supports_backward
    assert spec.accepts(dict(H=8192)) and spec.accepts(dict(H=4096)) and spec.accepts(dict(H=16384))
    assert not spec.accepts(dict(H=6144)) and not spec.accepts(dict(H=20480))
    # The kernels hold a row in one tl.arange, which needs a power of two.
    assert not spec.accepts(dict(H=12288))
    for shape in CORRECTNESS_SHAPES:
        assert spec.accepts(dict(H=shape.H)), label(shape)


def test_ops_are_exported():
    assert "norm_residual_norm" in ops.__all__


def test_kernel_choice_follows_cu_count_and_call_size():
    for cus in (80, 256, 304):
        for H in (4096, 8192, 16384):
            assert impl.pick_kernel(1, H, cus) == "tlx"
            assert impl.pick_kernel(cus // 2, H, cus) == "tlx"
            assert impl.pick_kernel(cus // 2 + 1, H, cus) == "early"
            last_early = 10 * 8192 * cus // H
            assert impl.pick_kernel(last_early, H, cus) == "early"
            assert impl.pick_kernel(last_early + 1, H, cus) == "persistent"


def test_streaming_choice_follows_the_call_size():
    for H in (4096, 8192, 16384):
        rows = (3 << 23) // H
        assert impl.use_streaming_accesses(rows, H) and not impl.use_streaming_accesses(rows - 1, H)


@requires_gfx950
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=label)
def test_matches_reference_registered_shapes(shape):
    kw = inputs(shape)
    _check(ops.norm_residual_norm(**kw), kw)


@requires_gfx950
@pytest.mark.parametrize("kind", KERNELS)
@pytest.mark.parametrize("T,H", [(1, 4096), (37, 8192), (300, 16384)])
@pytest.mark.parametrize("boundary", ["attn", "ffn", "final"])
def test_each_kernel_matches_reference(monkeypatch, kind, T, H, boundary):
    _force(monkeypatch, kind)
    kw = inputs(NormResidualNormShape(T, H, boundary))
    _check(ops.norm_residual_norm(**kw), kw)


@requires_gfx950
@pytest.mark.parametrize("kind", ["early", "persistent", "persistent_streaming"])
def test_64bit_row_offsets(monkeypatch, kind):
    _force(monkeypatch, kind)
    monkeypatch.setattr(impl, "INT32_OFFSETS", 0)
    kw = inputs(NormResidualNormShape(300, 8192, "final"), seed=9)
    _check(ops.norm_residual_norm(**kw), kw)


def _assert_bitwise(a, b):
    assert torch.equal(a[0].view(torch.int16), b[0].view(torch.int16))
    assert torch.equal(a[1].view(torch.int32), b[1].view(torch.int32))


@requires_gfx950
@pytest.mark.parametrize("boundary", ["attn", "ffn", "final"])
@pytest.mark.parametrize("T,H", [(64, 8192), (600, 8192), (300, 4096), (301, 16384)])
@pytest.mark.parametrize("variant", ["full", "scalar_gate", "no_gate_no_weights"])
def test_all_kernels_are_bitwise_equal(monkeypatch, boundary, T, H, variant):
    kw = inputs(NormResidualNormShape(T, H, boundary), seed=3)
    if variant == "scalar_gate":
        kw.update(alpha=kw["alpha"][:1].clone(), beta=kw["beta"][:1].clone())
    elif variant == "no_gate_no_weights":
        kw.update(alpha=None, beta=None, post_weight=None, pre_weight=None)
    results = []
    for kind in KERNELS:
        _force(monkeypatch, kind)
        results.append(ops.norm_residual_norm(**kw))
    for other in results[1:]:
        _assert_bitwise(results[0], other)


@requires_gfx950
@pytest.mark.parametrize("streaming", [False, True])
def test_rows_do_not_depend_on_the_batch(streaming):
    """Slices that take the TLX, early and persistent kernels match the full batch row for row; with ``streaming`` the
    full batch is large enough for non-temporal accesses."""
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    T = max(10 * cus, impl.STREAMING_ELEMENTS // 8192) + 77 if streaming else 10 * cus + 77
    assert impl.use_streaming_accesses(T, 8192) == streaming
    kw = inputs(NormResidualNormShape(T, 8192, "final"), seed=11)
    full = ops.norm_residual_norm(**kw)
    for start, stop in ((0, 1), (5, 5 + cus // 2), (100, 100 + 3 * cus), (0, T - 1), (1, T)):
        part = {k: v[start:stop] if k in ("stream", "branch") else v for k, v in kw.items()}
        got = ops.norm_residual_norm(**part)
        _assert_bitwise(got, (full[0][start:stop], full[1][start:stop]))


@requires_gfx950
@pytest.mark.parametrize("kind", KERNELS)
@pytest.mark.parametrize("variant", ["no_gate", "scalar_gate", "no_weights", "post_weight_not_input_scale"])
def test_optional_features(monkeypatch, kind, variant):
    _force(monkeypatch, kind)
    kw = inputs(NormResidualNormShape(19, 8192, "ffn"), seed=5)
    if variant == "no_gate":
        kw.update(alpha=None, beta=None)
    elif variant == "scalar_gate":
        kw.update(alpha=kw["alpha"][:1].clone(), beta=kw["beta"][:1].clone())
    elif variant == "no_weights":
        kw.update(post_weight=None, pre_weight=None)
    else:
        kw.update(post_norm_input_scale=False)
    _check(ops.norm_residual_norm(**kw), kw)


@requires_gfx950
def test_inputs_are_not_mutated_and_buffers_are_filled():
    kw = inputs(NormResidualNormShape(40, 8192, "final"), seed=7)
    before = {k: v.clone() for k, v in kw.items() if isinstance(v, torch.Tensor)}
    out = torch.full((40, 8192), float("nan"), device="cuda", dtype=torch.bfloat16)
    new_stream = torch.full((40, 8192), float("nan"), device="cuda")
    got = ops.norm_residual_norm(**kw, out=out, new_stream=new_stream)
    assert got[0] is out and got[1] is new_stream
    _check(got, kw)
    for k, v in before.items():
        assert torch.equal(kw[k], v), k


@requires_gfx950
def test_empty_batch():
    kw = inputs(NormResidualNormShape(0, 8192, "attn"))
    out, new_stream = ops.norm_residual_norm(**kw)
    assert out.shape == (0, 8192) and new_stream.shape == (0, 8192)


@requires_gfx950
@pytest.mark.parametrize("T", [9, 2000])
def test_graph_capture_replays_with_new_inputs(T):
    kw = inputs(NormResidualNormShape(T, 8192, "ffn"), seed=1)
    out = torch.empty(T, 8192, device="cuda", dtype=torch.bfloat16)
    new_stream = torch.empty(T, 8192, device="cuda")
    ops.norm_residual_norm(**kw, out=out, new_stream=new_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.norm_residual_norm(**kw, out=out, new_stream=new_stream)
    fresh = inputs(NormResidualNormShape(T, 8192, "ffn"), seed=2)
    kw["stream"].copy_(fresh["stream"])
    kw["branch"].copy_(fresh["branch"])
    graph.replay()
    _check((out, new_stream), kw)


@requires_gfx950
def test_rejects_invalid_inputs():
    kw = inputs(NormResidualNormShape(4, 8192, "ffn"))
    call = ops.norm_residual_norm
    with pytest.raises(InvalidInput):
        call(**dict(kw, branch=kw["branch"].float()))
    with pytest.raises(InvalidInput):
        call(**dict(kw, stream=kw["stream"][:, :4096]))
    with pytest.raises(InvalidInput):
        call(**dict(kw, beta=None))
    with pytest.raises(InvalidInput):
        call(**dict(kw, pre_weight=kw["pre_weight"].float()))
    with pytest.raises(InvalidInput):
        call(**kw, new_stream=kw["stream"])
    with pytest.raises(InvalidInput):
        call(**kw, space="full")
    for H in (6144, 12288):
        with pytest.raises(InvalidInput):
            call(**inputs(NormResidualNormShape(4, H, "attn")))
    for name, like in (("out", kw["branch"]), ("new_stream", kw["stream"])):
        with pytest.raises(UnsupportedBackward):
            call(**kw, **{name: torch.empty_like(like, requires_grad=True)})
    with pytest.raises(UnsupportedBackward):
        call(**dict(kw, branch=kw["branch"].requires_grad_()))
