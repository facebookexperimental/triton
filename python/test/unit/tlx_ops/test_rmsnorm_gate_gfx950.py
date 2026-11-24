"""Public ``tlx.ops.rmsnorm_gate`` coverage for gfx950."""

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip_cdna4
from triton.tlx import ops
from triton.tlx.ops import InvalidInput, UnsupportedBackward
from triton.tlx.ops._catalog import CATALOG
from triton.tlx.ops.kernels.fused.rmsnorm_gate import gfx950 as impl
from triton.tlx.ops.kernels.fused.rmsnorm_gate._shapes import (CORRECTNESS_SHAPES, RmsnormGateShape, inputs, label,
                                                               reference)

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
# (elements per lane, XCDs used): every layout the launcher can pick.
LAYOUTS = ((2, 1), (2, 2), (2, 4), (2, 8), (8, 8))


@triton.jit
def _two_element_lanes_kernel(X, G, Y, eps, N, H, stride_x, stride_g, stride_y, BLOCK_M: tl.constexpr):
    """The plain Triton formulation: BLOCK_M (token, head) rows of 64 per program; with four waves each head gets 32
    lanes of two elements."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 64)
    mask = (rows < N)[:, None]
    tok = rows // H
    head = rows % H
    x = tl.load(X + tok[:, None] * stride_x + head[:, None] * 64 + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / 64
    rstd = 1.0 / tl.sqrt(var + eps)
    x = x * rstd[:, None]
    g = tl.load(G + tok[:, None] * stride_g + head[:, None] * 64 + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    x = x * tl.sigmoid(g)
    tl.store(Y + tok[:, None] * stride_y + head[:, None] * 64 + cols[None, :], x.to(tl.bfloat16), mask=mask)


def _two_element_lanes(x, gate, eps):
    T, N = x.shape
    y = torch.empty(T, N, device=x.device, dtype=torch.bfloat16)
    rows = T * (N // 64)
    _two_element_lanes_kernel[(triton.cdiv(rows, 8), )](x, gate, y, eps, rows, N // 64, x.stride(0), gate.stride(0),
                                                        y.stride(0), BLOCK_M=8, num_warps=4)
    return y


def _bits(t):
    return t.view(torch.int16)


def _assert_bitwise(a, b):
    assert torch.equal(_bits(a), _bits(b))


def _check(got, kw):
    ref = reference(**kw)
    # The reference uses correctly rounded sqrt and exp; the kernel uses the hardware approximations, which can move
    # a BF16 rounding by one ulp.
    torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=8e-3)
    assert (_bits(got) != _bits(ref)).float().mean() < 1e-3


def _force(monkeypatch, layout):
    monkeypatch.setattr(impl, "pick_layout", lambda programs, num_cus: layout)


def test_catalog_entry():
    spec = {s.op: s for s in CATALOG if s.arch == "gfx950"}["rmsnorm_gate"]
    assert spec.dtypes == frozenset({"bfloat16"}) and not spec.supports_backward
    assert spec.accepts(dict(head_dim=64, H=16)) and spec.accepts(dict(head_dim=64, H=8))
    assert spec.accepts(dict(head_dim=64, H=40))
    assert not spec.accepts(dict(head_dim=128, H=16)) and not spec.accepts(dict(head_dim=64, H=12))
    for shape in CORRECTNESS_SHAPES:
        assert spec.accepts(dict(head_dim=64, H=shape.heads)), label(shape)


def test_op_is_exported():
    assert "rmsnorm_gate" in ops.__all__


@pytest.mark.parametrize("cus", [256, 304, 80])
def test_layout_follows_cu_count_and_call_size(cus):
    per_xcd = cus // 8
    assert impl.pick_layout(1, cus) == (2, 1)
    assert impl.pick_layout(per_xcd, cus) == (2, 1)
    assert impl.pick_layout(per_xcd + 1, cus) == (2, 2)
    assert impl.pick_layout(2 * per_xcd + 1, cus) == (2, 4)
    assert impl.pick_layout(4 * per_xcd + 1, cus) == (2, 8)
    assert impl.pick_layout(3 * cus, cus) == (2, 8)
    assert impl.pick_layout(3 * cus + 1, cus) == (8, 8)


@requires_gfx950
def test_launch_follows_the_layout(monkeypatch):
    """The launcher passes the picked layout to the kernel, with 8 * ceil(programs / xcds) programs."""
    launches = []
    real = impl._kernel

    class Spy:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                launches.append((grid, kwargs["ELEMS"], kwargs["XCDS"], kwargs["num_warps"]))
                return real[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(impl, "_kernel", Spy())
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    for T in (1, cus // 16, cus // 16 + 1, 3 * cus // 2, 3 * cus // 2 + 1):
        kw = inputs(RmsnormGateShape(T, 16))
        _check(ops.rmsnorm_gate(**kw), kw)
        elems, xcds = impl.pick_layout(2 * T, cus)
        assert launches[-1] == ((triton.cdiv(2 * T, xcds) * 8, ), elems, xcds, 4 if elems == 2 else 1)
    assert {launch[1:3] for launch in launches} == {(2, 1), (2, 2), (2, 8), (8, 8)}


@requires_gfx950
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=label)
def test_matches_reference_registered_shapes(shape):
    kw = inputs(shape)
    _check(ops.rmsnorm_gate(**kw), kw)


@requires_gfx950
@pytest.mark.parametrize("layout", LAYOUTS, ids=str)
@pytest.mark.parametrize("T,heads", [(1, 16), (5, 8), (37, 16), (300, 40), (2048, 16)])
def test_each_layout_is_bitwise_the_two_element_kernel(monkeypatch, layout, T, heads):
    _force(monkeypatch, layout)
    kw = inputs(RmsnormGateShape(T, heads), seed=3)
    _assert_bitwise(ops.rmsnorm_gate(**kw), _two_element_lanes(**kw))


@requires_gfx950
@pytest.mark.parametrize("scale", [1e-30, 1e-3, 1.0, 1e3, 1e30])
def test_bitwise_over_magnitudes_and_gate_tails(scale):
    """Tiny and huge inputs (eps dominates, squares overflow) and gates far in the sigmoid's tails."""
    kw = inputs(RmsnormGateShape(64, 16), seed=4)
    kw["x"] = (kw["x"].float() * scale).to(torch.bfloat16)
    kw["gate"] = kw["gate"] * 20
    _assert_bitwise(ops.rmsnorm_gate(**kw), _two_element_lanes(**kw))


@requires_gfx950
def test_rows_do_not_depend_on_the_batch():
    """Slices that take every layout match the full batch row for row."""
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    T = 3 * cus // 2 + 77
    kw = inputs(RmsnormGateShape(T, 16), seed=11)
    full = ops.rmsnorm_gate(**kw)
    for start, stop in ((0, 1), (3, 3 + cus // 16), (9, 9 + cus // 8), (40, 40 + cus // 4), (20, 20 + cus), (0, T - 1),
                        (1, T)):
        part = dict(kw, x=kw["x"][start:stop], gate=kw["gate"][start:stop])
        _assert_bitwise(ops.rmsnorm_gate(**part), full[start:stop])


@requires_gfx950
@pytest.mark.parametrize("layout", [(2, 1), (2, 8), (8, 8)], ids=str)
def test_64bit_token_offsets(monkeypatch, layout):
    _force(monkeypatch, layout)
    monkeypatch.setattr(impl, "INT32_OFFSETS", 0)
    kw = inputs(RmsnormGateShape(300, 16), seed=9)
    _assert_bitwise(ops.rmsnorm_gate(**kw), _two_element_lanes(**kw))


@requires_gfx950
def test_eps_is_applied():
    kw = inputs(RmsnormGateShape(16, 16), seed=6)
    kw["x"] = (kw["x"].float() * 1e-3).to(torch.bfloat16)
    small, large = ops.rmsnorm_gate(**dict(kw, eps=1e-8)), ops.rmsnorm_gate(**dict(kw, eps=1e-2))
    _check(small, dict(kw, eps=1e-8))
    _check(large, dict(kw, eps=1e-2))
    assert not torch.equal(small, large)


@requires_gfx950
def test_strided_inputs_and_out_buffer():
    """Row-strided x and out (views into wider buffers); inputs are not mutated and the buffer is returned."""
    T, N = 40, 1024
    kw = inputs(RmsnormGateShape(T, 16), seed=7)
    wide_x = torch.zeros(T, N + 64, device="cuda", dtype=torch.bfloat16)
    wide_x[:, 32:32 + N] = kw["x"]
    x = wide_x[:, 32:32 + N]
    before = (x.clone(), kw["gate"].clone())
    wide_out = torch.full((T, N + 128), float("nan"), device="cuda", dtype=torch.bfloat16)
    out = wide_out[:, 64:64 + N]
    got = ops.rmsnorm_gate(x, kw["gate"], eps=kw["eps"], out=out)
    assert got is out
    _assert_bitwise(got, _two_element_lanes(kw["x"], kw["gate"], kw["eps"]))
    assert torch.isnan(wide_out[:, :64]).all() and torch.isnan(wide_out[:, 64 + N:]).all()
    assert torch.equal(x, before[0]) and torch.equal(kw["gate"], before[1])


@requires_gfx950
def test_empty_batch():
    kw = inputs(RmsnormGateShape(0, 16))
    assert ops.rmsnorm_gate(**kw).shape == (0, 1024)


@requires_gfx950
@pytest.mark.parametrize("T", [4, 2000])
def test_graph_capture_replays_with_new_inputs(T):
    kw = inputs(RmsnormGateShape(T, 16), seed=1)
    out = torch.empty(T, 1024, device="cuda", dtype=torch.bfloat16)
    ops.rmsnorm_gate(**kw, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.rmsnorm_gate(**kw, out=out)
    fresh = inputs(RmsnormGateShape(T, 16), seed=2)
    kw["x"].copy_(fresh["x"])
    kw["gate"].copy_(fresh["gate"])
    graph.replay()
    _assert_bitwise(out, _two_element_lanes(**kw))


@requires_gfx950
def test_non_default_stream():
    kw = inputs(RmsnormGateShape(33, 16), seed=5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        got = ops.rmsnorm_gate(**kw)
    torch.cuda.current_stream().wait_stream(stream)
    _assert_bitwise(got, _two_element_lanes(**kw))


@requires_gfx950
def test_rejects_invalid_inputs():
    kw = inputs(RmsnormGateShape(4, 16))
    call = ops.rmsnorm_gate
    with pytest.raises(InvalidInput):
        call(**dict(kw, x=kw["x"].float()))
    with pytest.raises(InvalidInput):
        call(**dict(kw, gate=kw["gate"][:, :512]))
    with pytest.raises(InvalidInput):
        call(**dict(kw, x=kw["x"].t().contiguous().t()))
    with pytest.raises(InvalidInput):
        call(**kw, head_dim=128)
    with pytest.raises(InvalidInput):
        call(**dict(kw, eps=0.0))
    with pytest.raises(InvalidInput):
        call(**kw, out=kw["x"])
    with pytest.raises(InvalidInput):
        call(**kw, out=torch.empty(4, 1024, device="cuda"))
    with pytest.raises(InvalidInput):
        call(**kw, space="full")
    with pytest.raises(InvalidInput):
        call(**inputs(RmsnormGateShape(4, 12)))
    with pytest.raises(UnsupportedBackward):
        call(**kw, out=torch.empty(4, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True))
    with pytest.raises(UnsupportedBackward):
        call(**dict(kw, x=kw["x"].clone().requires_grad_()))
    with pytest.raises(UnsupportedBackward):
        call(**dict(kw, gate=kw["gate"].clone().requires_grad_()))
    with torch.no_grad():
        call(**dict(kw, x=kw["x"].clone().requires_grad_()))
