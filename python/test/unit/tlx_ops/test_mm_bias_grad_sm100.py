"""Blackwell correctness for explicit linear weight and bias gradients."""

from __future__ import annotations

import pytest
import torch
from triton._internal_testing import is_blackwell
from triton.tlx.ops import InvalidInput, mm_bias_grad, UnsupportedOp
from triton.tlx.ops.kernels.mm_bias_grad import sm100
from triton.tlx.ops.kernels.mm_bias_grad._shapes import CORRECTNESS_SHAPES

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="Requires sm100")


def _reference(dout, input_act):
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return dout.float().T @ input_act.float(), dout.float().sum(0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _assert_gradients(actual, expected):
    for result, reference in zip(actual, expected):
        assert result.shape == reference.shape
        assert torch.isfinite(result).all()
        precision = 8e-3 if result.dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(result.float(), reference, rtol=precision,
                                   atol=precision * reference.abs().max().item())


@pytest.mark.parametrize("batch, out_features, in_features", CORRECTNESS_SHAPES)
def test_gradients(batch, out_features, in_features):
    dtype = torch.bfloat16
    torch.manual_seed(0)
    dout = torch.randn((batch, out_features), device="cuda", dtype=dtype)
    input_act = torch.randn((batch, in_features), device="cuda", dtype=dtype)
    result = mm_bias_grad(dout, input_act)
    torch.cuda.synchronize()
    assert result[0].dtype == dtype and result[1].dtype == dtype
    _assert_gradients(result, _reference(dout, input_act))


@pytest.mark.parametrize("tile, split, buffers, tmem, interleave", [
    ((128, 128, 64, 1, 1), 1, 2, 1, 0),
    ((128, 128, 64, 1, 1), 4, 2, 1, 0),
    ((128, 256, 64, 1, 1), 4, 2, 1, 0),
    ((128, 128, 64, 1, 1), 4, 3, 2, 0),
    ((128, 256, 64, 1, 1), 4, 3, 1, 0),
    ((128, 128, 128, 1, 1), 4, 3, 2, 0),
    ((64, 128, 64, 1, 1), 4, 3, 2, 0),
])
@pytest.mark.parametrize("out_features, in_features", [(384, 384), (512, 512)])
def test_split_k_and_feature_padding(monkeypatch, tile, split, buffers, tmem, interleave, out_features, in_features):
    batch = 4097
    dtype = torch.bfloat16
    torch.manual_seed(42)
    dout = torch.randn((batch, out_features), device="cuda", dtype=dtype)
    input_act = torch.randn((batch, in_features), device="cuda", dtype=dtype)
    config = sm100._config(tile, buffers, tmem, split, interleave=interleave)
    monkeypatch.setattr(sm100, "_configs", lambda *args, **kwargs: [config])
    sm100._tuned.cache_clear()
    try:
        for _ in range(3):
            _assert_gradients(mm_bias_grad(dout, input_act), _reference(dout, input_act))
    finally:
        sm100._tuned.cache_clear()


def test_bias_ownership_and_cancellation():
    dout = torch.full((4097, 512), 0.03125, dtype=torch.bfloat16, device="cuda")
    dout[1::2].neg_()
    input_act = torch.ones((4097, 768), dtype=dout.dtype, device=dout.device)
    weight, bias = mm_bias_grad(dout, input_act)
    torch.testing.assert_close(bias, torch.full_like(bias, 0.03125), rtol=0, atol=0)
    torch.testing.assert_close(weight, torch.full_like(weight, 0.03125), rtol=0, atol=0)


def test_cuda_graph_replay_overwrites_outputs():
    dout = torch.ones((4096, 512), device="cuda", dtype=torch.bfloat16)
    input_act = torch.ones((4096, 384), device="cuda", dtype=torch.bfloat16)
    out = (torch.empty((512, 384), device="cuda",
                       dtype=dout.dtype), torch.empty((512, ), device="cuda", dtype=dout.dtype))
    mm_bias_grad(dout, input_act, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = mm_bias_grad(dout, input_act, out=out)
    assert result[0] is out[0] and result[1] is out[1]
    for value in (0.125, -0.25, 0.0):
        dout.fill_(value)
        out[0].fill_(float("nan"))
        out[1].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out[0], torch.full_like(out[0], value * 4096), rtol=0, atol=0)
        torch.testing.assert_close(out[1], torch.full_like(out[1], value * 4096), rtol=0, atol=0)


def test_explicit_gradients_accept_saved_activations():
    dout = torch.randn((65, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    input_act = torch.randn((65, 128), device="cuda", dtype=dout.dtype, requires_grad=True)
    result = mm_bias_grad(dout, input_act)
    assert not result[0].requires_grad and not result[1].requires_grad
    _assert_gradients(result, _reference(dout, input_act))


@pytest.mark.parametrize("shapes", [((0, 128), (0, 128)), ((16, 128), (17, 128)), ((16, 7), (16, 128))])
def test_rejects_invalid_shapes(shapes):
    tensors = [torch.empty(shape, device="cuda", dtype=torch.bfloat16) for shape in shapes]
    with pytest.raises(InvalidInput):
        mm_bias_grad(*tensors)


def test_rejects_invalid_dtype_layout_and_space():
    a = torch.ones((64, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(InvalidInput):
        mm_bias_grad(a, a.float())
    with pytest.raises(InvalidInput):
        mm_bias_grad(a.float(), a.float())
    with pytest.raises(InvalidInput):
        mm_bias_grad(a.T, a.T)
    with pytest.raises(InvalidInput):
        mm_bias_grad(a, a, space="unknown")


def test_rejects_cpu():
    a = torch.ones((64, 128), dtype=torch.bfloat16)
    with pytest.raises(UnsupportedOp):
        mm_bias_grad(a, a)


def test_rejects_aliasing_outputs():
    a = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(InvalidInput, match="overlap"):
        mm_bias_grad(a, a, out=(a, torch.empty((128, ), device=a.device, dtype=a.dtype)))
    weight = torch.empty_like(a)
    with pytest.raises(InvalidInput, match="overlap"):
        mm_bias_grad(a, a, out=(weight, weight[0]))


def test_rejects_empty_split():
    config = sm100._config((128, 128, 64, 1, 1), 2, 1, 3)
    assert not sm100._valid_config(config, 128, 128, 65)


def test_cluster_config_in_full_space():
    config = sm100._config((256, 256, 64, 2, 2), 4, 1, 32)
    candidates = sm100._configs(512, 384, 294912, 148, space="full")
    assert any(c.kwargs == config.kwargs and c.num_warps == 8 and c.ctas_per_cga == (2, 1, 1) for c in candidates)


@pytest.mark.parametrize("buffers,epilogue,interleave", [(2, 4, 0), (3, 4, 1), (4, 8, 0), (4, 8, 1)])
def test_two_cta_protocol(monkeypatch, buffers, epilogue, interleave):
    torch.manual_seed(13)
    dout = torch.randn((4097, 512), device="cuda", dtype=torch.bfloat16)
    input_act = torch.randn((4097, 384), device="cuda", dtype=dout.dtype)
    config = sm100._config((256, 256, 64, 2, 2), buffers, 1, 4, interleave=interleave, epilogue=epilogue)
    monkeypatch.setattr(sm100, "_configs", lambda *args, **kwargs: [config])
    sm100._tuned.cache_clear()
    try:
        for _ in range(3):
            _assert_gradients(mm_bias_grad(dout, input_act), _reference(dout, input_act))
    finally:
        sm100._tuned.cache_clear()


def test_prunes_partial_m_split_workspace():
    config = sm100._config((256, 256, 64, 2, 2), 4, 1, 4)
    assert not sm100._valid_config(config, 384, 384, 4096)
