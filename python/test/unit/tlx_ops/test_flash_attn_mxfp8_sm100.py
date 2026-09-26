"""L1 correctness for ``tlx.ops.flash_attn_mxfp8`` on Blackwell."""

import pytest
import torch
from triton._internal_testing import is_blackwell
from triton.tlx.ops.kernels.flash_attn_mxfp8._shapes import CORRECTNESS_SHAPES

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.flash_attn_mxfp8 requires sm100")

MULTI_WAVE_SHAPE = (1, 64, 1024, 128)


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


def test_quantize_mxfp8_32x32_operand_layouts():
    from triton.tlx.ops.kernels.flash_attn_mxfp8.sm100 import (
        _quantize_mxfp8_32x32_operand,
    )

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
            triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
        )
    except ImportError:
        pytest.skip("installed torchao does not provide the 32x32 reference quantizer")

    from triton.tlx.ops.kernels.flash_attn_mxfp8.sm100 import (
        _quantize_mxfp8_32x32_operand,
        swizzled_to_tma_preshuffled,
    )

    ref = torch.randn((2, 3, 256, 128), device="cuda", dtype=torch.bfloat16)
    data, normal_scale, swapped_scale = _quantize_mxfp8_32x32_operand(ref)
    ref_data, ref_normal_scale, ref_swapped_scale = (
        triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale(ref.reshape(-1, 128))
    )
    ref_normal_scale = swizzled_to_tma_preshuffled(ref_normal_scale, 256, 128, 32, 6)
    ref_swapped_scale = swizzled_to_tma_preshuffled(ref_swapped_scale, 128, 256, 32, 6)

    assert torch.equal(data.view(torch.uint8), ref_data.reshape_as(ref).view(torch.uint8))
    assert torch.equal(normal_scale.view(torch.uint8), ref_normal_scale.view(torch.uint8))
    assert torch.equal(swapped_scale.view(torch.uint8), ref_swapped_scale.view(torch.uint8))


def test_flash_attn_mxfp8_shares_32x32_qkdo_payloads(monkeypatch):
    from triton.tlx.ops import flash_attn_mxfp8
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import sm100

    calls = []
    quantize = sm100._quantize_mxfp8_32x32_operand

    def record_quantize(ref):
        calls.append(ref)
        return quantize(ref)

    monkeypatch.setattr(sm100, "_quantize_mxfp8_32x32_operand", record_quantize)

    q, k, v = _qkv((1, 1, 256, 128), requires_grad=True)
    do = torch.randn_like(q)
    flash_attn_mxfp8(q, k, v, arch=ARCH, space="smoke").backward(do)

    assert len(calls) == 3
    assert calls[0].data_ptr() == q.data_ptr()
    assert calls[1].data_ptr() == k.data_ptr()
    assert calls[2].data_ptr() == do.data_ptr()


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype_name", CORRECTNESS_SHAPES)
def test_flash_attn_mxfp8_fwd(Z, H, N_CTX, HEAD_DIM, causal, dtype_name):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv((Z, H, N_CTX, HEAD_DIM))
    scale = 0.5
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke")
    ref = _sdpa(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=0.2, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_mxfp8_fwd_multiple_cta_waves(causal):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv(MULTI_WAVE_SHAPE)
    scale = 0.5
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke")
    ref = _sdpa(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_mxfp8_bwd_multiple_cta_waves(causal):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv(MULTI_WAVE_SHAPE, requires_grad=True)
    rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    scale = 0.5
    do = torch.randn_like(q)

    flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke").backward(do)
    _sdpa(rq, rk, rv, causal, scale).backward(do)

    for label, actual, expected in (("dq", q.grad, rq.grad), ("dk", k.grad, rk.grad), ("dv", v.grad, rv.grad)):
        cosine = _cosine(actual, expected)
        assert cosine >= 0.98, f"{label} cosine_similarity={cosine:.6f}"


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype_name", CORRECTNESS_SHAPES)
def test_flash_attn_mxfp8_bwd(Z, H, N_CTX, HEAD_DIM, causal, dtype_name):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv((Z, H, N_CTX, HEAD_DIM), requires_grad=True)
    rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    scale = 0.5
    do = torch.randn_like(q)

    flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke").backward(do)
    _sdpa(rq, rk, rv, causal, scale).backward(do)

    for label, actual, expected in (("dq", q.grad, rq.grad), ("dk", k.grad, rk.grad), ("dv", v.grad, rv.grad)):
        cosine = _cosine(actual, expected)
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
