"""L1 correctness for ``tlx.ops.flash_attn_mxfp8`` on Blackwell."""

from unittest import mock

import pytest
import torch
from triton._internal_testing import is_blackwell
from triton.tlx.ops.kernels.flash_attn_mxfp8._shapes import CORRECTNESS_SHAPES

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.flash_attn_mxfp8 requires sm100")

MULTI_WAVE_SHAPE = (1, 64, 1024, 128)
GRAD_SUBSETS = [
    pytest.param((True, False, False), id="q"),
    pytest.param((False, True, False), id="k"),
    pytest.param((False, False, True), id="v"),
    pytest.param((True, True, False), id="qk"),
    pytest.param((True, False, True), id="qv"),
    pytest.param((False, True, True), id="kv"),
    pytest.param((True, True, True), id="qkv"),
]


def _qkv(shape, *, requires_grad=False):
    return [(torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5).requires_grad_(requires_grad)
            for _ in range(3)]


def _sdpa(q, k, v, causal, scale):
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=causal,
        scale=scale,
    )


def _cosine(actual, expected):
    return torch.nn.functional.cosine_similarity(
        actual.float().flatten(),
        expected.float().flatten(),
        dim=0,
    ).item()


@pytest.fixture(params=[False, True], ids=["exp2", "expcast"])
def p_expcast(request):
    """Run the forward with MUFU exp2 P (default) and with the ExpCast-FP8 P encode."""
    return request.param


def test_quantize_mxfp8_32x32_operand_layouts():
    from triton.tlx.ops.kernels.flash_attn_mxfp8.sm100 import (
        _quantize_mxfp8_32x32_operand, )

    ref = torch.randn((2, 3, 256, 128), device="cuda", dtype=torch.bfloat16)
    data, normal_scale, swapped_scale = _quantize_mxfp8_32x32_operand(ref)

    assert data.shape == ref.shape
    assert data.dtype == torch.float8_e4m3fn
    assert data.is_contiguous()
    assert normal_scale.shape == (6, 2, 1, 2, 256)
    assert swapped_scale.shape == (6, 1, 2, 2, 256)
    assert normal_scale.dtype == torch.float8_e8m0fnu
    assert swapped_scale.dtype == torch.float8_e8m0fnu


def test_quantize_mxfp8_32x32_matches_torchao():
    try:
        from torchao.prototype.mx_formats.kernels import (
            triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale, )
    except ImportError:
        pytest.skip("installed torchao does not provide the 32x32 reference quantizer")

    from triton.tlx.ops.kernels.flash_attn_mxfp8.sm100 import (
        _quantize_mxfp8_32x32_operand,
        swizzled_to_tma_preshuffled,
    )

    ref = torch.randn((2, 3, 256, 128), device="cuda", dtype=torch.bfloat16)
    data, normal_scale, swapped_scale = _quantize_mxfp8_32x32_operand(ref)
    ref_data, ref_normal_scale, ref_swapped_scale = (triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale(
        ref.reshape(-1, 128)))
    ref_normal_scale = swizzled_to_tma_preshuffled(ref_normal_scale, 256, 128, 32, 6)
    ref_swapped_scale = swizzled_to_tma_preshuffled(ref_swapped_scale, 128, 256, 32, 6)

    assert torch.equal(data.view(torch.uint8), ref_data.reshape_as(ref).view(torch.uint8))
    assert torch.equal(normal_scale.view(torch.uint8), ref_normal_scale.view(torch.uint8))
    assert torch.equal(swapped_scale.view(torch.uint8), ref_swapped_scale.view(torch.uint8))


@pytest.mark.parametrize("needs_input_grad", GRAD_SUBSETS)
def test_flash_attn_mxfp8_shares_32x32_qkdo_payloads(monkeypatch, needs_input_grad):
    from triton.tlx.ops import flash_attn_mxfp8
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import sm100

    calls = []
    v_calls = []
    quantize = sm100._quantize_mxfp8_32x32_operand
    quantize_v = sm100._quantize_mxfp8_operand

    def record_quantize(ref, **kwargs):
        calls.append((ref, kwargs))
        return quantize(ref, **kwargs)

    def record_quantize_v(ref, **kwargs):
        v_calls.append((ref, kwargs))
        return quantize_v(ref, **kwargs)

    monkeypatch.setattr(sm100, "_quantize_mxfp8_32x32_operand", record_quantize)
    monkeypatch.setattr(sm100, "_quantize_mxfp8_operand", record_quantize_v)

    q, k, v = [tensor.requires_grad_(needed) for tensor, needed in zip(_qkv((1, 1, 256, 128)), needs_input_grad)]
    do = torch.randn_like(q)
    flash_attn_mxfp8(q, k, v, space="smoke").backward(do)

    assert len(calls) == 3
    assert [tensor.data_ptr() for tensor, _ in calls] == [q.data_ptr(), k.data_ptr(), do.data_ptr()]
    need_dq, need_dk, need_dv = needs_input_grad
    expected_scale_requests = [(True, need_dk), (True, need_dq), (need_dq or need_dk, need_dv)]
    for (_, kwargs), (normal, swapped) in zip(calls, expected_scale_requests):
        assert kwargs.get("need_normal_scale", True) is normal
        assert kwargs.get("need_swapped_scale", True) is swapped
    assert len(v_calls) == 1 + (need_dq or need_dk)
    assert all(tensor.data_ptr() == v.data_ptr() for tensor, _ in v_calls)
    assert v_calls[0][1].get("transpose_for_reduction", False) is True
    if need_dq or need_dk:
        assert v_calls[1][1].get("transpose_for_reduction", False) is False


@pytest.mark.parametrize("need_normal_scale,need_swapped_scale", [(True, False), (False, True), (True, True)])
def test_flash_attn_mxfp8_quantizer_allocates_requested_scales(need_normal_scale, need_swapped_scale):
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import sm100

    ref = torch.empty((1, 1, 256, 128), dtype=torch.bfloat16)
    with mock.patch.object(sm100, "_mxfp8_32x32_qdata_dual_scale_kernel") as kernel:
        with mock.patch.object(sm100.torch, "empty", wraps=torch.empty) as allocate:
            data, normal, swapped = sm100._quantize_mxfp8_32x32_operand(ref, need_normal_scale=need_normal_scale,
                                                                        need_swapped_scale=need_swapped_scale)
    assert data.shape == ref.shape and data.dtype == torch.float8_e4m3fn
    assert (normal is not None) is need_normal_scale
    assert (swapped is not None) is need_swapped_scale
    assert allocate.call_count == need_normal_scale + need_swapped_scale
    assert kernel.__getitem__.return_value.call_count == 1


@pytest.mark.parametrize("needs_input_grad", GRAD_SUBSETS)
def test_flash_attn_mxfp8_backward_allocates_requested_work(needs_input_grad):
    from contextlib import ExitStack
    from types import SimpleNamespace
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import sm100

    need_dq, need_dk, need_dv = needs_input_grad
    need_ds = need_dq or need_dk
    shape = (1, 1, 256, 128)
    payload = torch.empty(shape, device="meta", dtype=torch.float8_e4m3fn)
    scale = torch.empty((1, 1, 1, 2, 256), device="meta", dtype=torch.float8_e8m0fnu)
    master = torch.empty(shape, device="meta", dtype=torch.bfloat16)
    lse = torch.empty(shape[:-1], device="meta", dtype=torch.float32)
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(sm100, "TensorDescriptor"))
        stack.enter_context(mock.patch.object(sm100.triton, "set_allocator"))
        stack.enter_context(
            mock.patch.object(torch.cuda, "get_device_properties",
                              return_value=SimpleNamespace(multi_processor_count=1)))
        allocate = stack.enter_context(mock.patch.object(sm100.torch, "empty", wraps=torch.empty))
        allocate_delta = stack.enter_context(mock.patch.object(sm100.torch, "empty_like", wraps=torch.empty_like))
        preprocess = stack.enter_context(mock.patch.object(sm100, "_attn_bwd_preprocess")).__getitem__.return_value
        backward = stack.enter_context(mock.patch.object(sm100, "_attn_bwd_mxf8_ws")).__getitem__.return_value
        gradients = sm100.attention_bwd(payload, payload, payload, payload if need_ds else None, master, lse, scale,
                                        scale if need_dk else None, scale, scale if need_dq else None,
                                        scale if need_ds else None, scale if need_ds else None,
                                        scale if need_dv else None, 0.5, do_bf16=master,
                                        needs_input_grad=needs_input_grad)

    assert allocate.call_count == sum(needs_input_grad)
    assert allocate_delta.call_count == int(need_ds)
    assert preprocess.call_count == int(need_ds)
    assert backward.call_count == 1
    assert [(tensor is not None) for tensor in gradients] == list(needs_input_grad)
    for tensor, dtype in zip(gradients, (torch.float32, torch.bfloat16, torch.bfloat16)):
        if tensor is not None:
            assert tensor.shape == shape and tensor.dtype == dtype
    for name, needed in zip(("NEED_DQ", "NEED_DK", "NEED_DV"), needs_input_grad):
        assert backward.call_args.kwargs[name] is needed
    assert (backward.call_args.args[2] is not None) is need_ds
    assert (backward.call_args.args[9] is not None) is need_ds
    if need_ds:
        assert preprocess.call_args.args[2] is gradients[0]
        assert preprocess.call_args.kwargs["NEED_DQ"] is need_dq


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype_name", CORRECTNESS_SHAPES)
def test_flash_attn_mxfp8_fwd(Z, H, N_CTX, HEAD_DIM, causal, dtype_name, p_expcast):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv((Z, H, N_CTX, HEAD_DIM))
    scale = 0.5
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke", p_expcast=p_expcast)
    ref = _sdpa(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=0.2, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_mxfp8_fwd_multiple_cta_waves(causal, p_expcast):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv(MULTI_WAVE_SHAPE)
    scale = 0.5
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke", p_expcast=p_expcast)
    ref = _sdpa(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0)


def _rel_l2(actual, expected):
    return ((actual.float() - expected).norm() / expected.norm()).item()


# Smoke bound, not a guarantee. The scales span flat (128**-0.5), mid-entropy (0.25) and peaked (0.5) rows.
EXPCAST_RATIO_SHAPES = [MULTI_WAVE_SHAPE, (1, 8, 4096, 128)]
EXPCAST_RATIO_SCALES = [128**-0.5, 0.25, 0.5]


@pytest.mark.parametrize("scale", EXPCAST_RATIO_SCALES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", EXPCAST_RATIO_SHAPES)
def test_flash_attn_mxfp8_expcast_error_vs_exp2(shape, causal, scale):
    """ExpCast-FP8 trades P accuracy for speed; bound its error relative to the exp2 path on the same inputs."""
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv(shape)
    ref = _sdpa(q.float(), k.float(), v.float(), causal, scale)
    rel_l2 = {}
    for p_expcast in (False, True):
        out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke", p_expcast=p_expcast)
        rel_l2[p_expcast] = _rel_l2(out, ref)
    # Measured worst case is 1.09x (flat logits at the default scale).
    assert rel_l2[True] <= 1.15 * rel_l2[False], rel_l2


@pytest.mark.parametrize("scale", EXPCAST_RATIO_SCALES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", EXPCAST_RATIO_SHAPES)
def test_flash_attn_mxfp8_expcast_bwd_error_vs_exp2(shape, causal, scale):
    """The bwd recomputes P with exp2, but consumes the ExpCast forward's O and logsumexp."""
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv(shape)
    do = torch.randn_like(q)
    rq, rk, rv = (t.detach().float().requires_grad_() for t in (q, k, v))
    _sdpa(rq, rk, rv, causal, scale).backward(do.float())
    rel_l2 = {}
    for p_expcast in (False, True):
        tq, tk, tv = (t.detach().clone().requires_grad_() for t in (q, k, v))
        flash_attn_mxfp8(tq, tk, tv, causal=causal, sm_scale=scale, space="smoke", p_expcast=p_expcast).backward(do)
        rel_l2[p_expcast] = {
            label: _rel_l2(actual.grad, expected.grad)
            for label, actual, expected in (("dq", tq, rq), ("dk", tk, rk), ("dv", tv, rv))
        }
    # Measured worst case is 1.06x (dv, causal, flat logits).
    for label in ("dq", "dk", "dv"):
        assert rel_l2[True][label] <= 1.10 * rel_l2[False][label], (label, rel_l2)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_mxfp8_bwd_multiple_cta_waves(causal, p_expcast):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv(MULTI_WAVE_SHAPE, requires_grad=True)
    rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    scale = 0.5
    do = torch.randn_like(q)

    flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke", p_expcast=p_expcast).backward(do)
    _sdpa(rq, rk, rv, causal, scale).backward(do)

    for label, actual, expected in (("dq", q.grad, rq.grad), ("dk", k.grad, rk.grad), ("dv", v.grad, rv.grad)):
        cosine = _cosine(actual, expected)
        assert cosine >= 0.98, f"{label} cosine_similarity={cosine:.6f}"


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype_name", CORRECTNESS_SHAPES)
def test_flash_attn_mxfp8_bwd(Z, H, N_CTX, HEAD_DIM, causal, dtype_name, p_expcast):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv((Z, H, N_CTX, HEAD_DIM), requires_grad=True)
    rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    scale = 0.5
    do = torch.randn_like(q)

    flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke", p_expcast=p_expcast).backward(do)
    _sdpa(rq, rk, rv, causal, scale).backward(do)

    for label, actual, expected in (("dq", q.grad, rq.grad), ("dk", k.grad, rk.grad), ("dv", v.grad, rv.grad)):
        cosine = _cosine(actual, expected)
        assert cosine >= 0.98, f"{label} cosine_similarity={cosine:.6f}"


@pytest.mark.parametrize("shape", [(1, 1, 256, 128), MULTI_WAVE_SHAPE], ids=["small", "multiple_cta_waves"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("needs_input_grad", GRAD_SUBSETS)
def test_flash_attn_mxfp8_backward_requested_gradients(shape, causal, needs_input_grad, p_expcast):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    full_inputs = _qkv(shape, requires_grad=True)
    inputs = [tensor.detach().clone().requires_grad_(needed) for tensor, needed in zip(full_inputs, needs_input_grad)]
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in full_inputs]
    do = torch.randn_like(inputs[0])
    scale = 0.5

    flash_attn_mxfp8(*full_inputs, causal=causal, sm_scale=scale, space="smoke", p_expcast=p_expcast).backward(do)
    flash_attn_mxfp8(*inputs, causal=causal, sm_scale=scale, space="smoke", p_expcast=p_expcast).backward(do)
    _sdpa(*reference_inputs, causal, scale).backward(do)

    for label, tensor, full_tensor, reference, needed in zip(("dq", "dk", "dv"), inputs, full_inputs, reference_inputs,
                                                             needs_input_grad):
        if not needed:
            assert tensor.grad is None
            continue
        assert tensor.grad.dtype == torch.bfloat16 and bool(tensor.grad.isfinite().all())
        torch.testing.assert_close(tensor.grad, full_tensor.grad)
        cosine = _cosine(tensor.grad, reference.grad)
        assert cosine >= 0.98, f"{label} cosine_similarity={cosine:.6f}"


@pytest.mark.parametrize(
    "shape,dtype,match",
    [
        ((1, 1, 256, 64), torch.bfloat16, "does not support"),
        ((1, 1, 384, 128), torch.bfloat16, "does not support"),
        ((1, 1, 256, 128), torch.float16, "does not support"),
    ],
)
def test_flash_attn_mxfp8_rejects_unsupported_inputs(shape, dtype, match):
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q, k, v = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
    with pytest.raises(InvalidInput, match=match):
        flash_attn_mxfp8(q, k, v, space="smoke")


def test_flash_attn_mxfp8_rejects_mismatched_shapes():
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q = torch.randn((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    with pytest.raises(InvalidInput, match="identical shapes"):
        flash_attn_mxfp8(q, k, v, space="smoke")


def test_flash_attn_mxfp8_rejects_unknown_space():
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q, k, v = _qkv((1, 1, 256, 128))
    with pytest.raises(InvalidInput, match="does not provide space"):
        flash_attn_mxfp8(q, k, v, space="heuristic")
