"""Public ``tlx.ops.flash_attn`` coverage for gfx950."""

from unittest import mock

import pytest
import torch

from triton._internal_testing import is_hip_cdna4


def test_flash_attn_gfx950_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, InvalidInput, check_inputs

    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("flash_attn", "gfx950")]
    assert len(specs) == 1
    spec = specs[0]
    assert spec.impl == "triton.tlx.ops.kernels.flash_attn.gfx950:flash_attn"
    assert spec.supports_backward

    check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=64)
    check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=128)
    with pytest.raises(InvalidInput, match="does not support float16"):
        check_inputs(spec, dtype=torch.float16, HEAD_DIM=64)
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=256)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_catalog_resolves_public_implementation():
    from triton.tlx.ops._catalog import impl_for
    from triton.tlx.ops.kernels.flash_attn.gfx950 import flash_attn

    implementation, spec = impl_for("flash_attn", arch="gfx950")
    assert implementation is flash_attn
    assert spec.supports_backward


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize(
    ("requires_grad", "grad_enabled"),
    ((False, True), (True, False)),
)
def test_flash_attn_gfx950_inference_skips_backward_state(requires_grad, grad_enabled):
    from triton.tlx.ops.kernels.flash_attn import gfx950

    q, k, v = _qkv(64, requires_grad=requires_grad, context=256)
    sentinel = object()
    with (
            mock.patch.object(gfx950, "_flash_attn_inference", return_value=sentinel) as inference,
            mock.patch.object(gfx950._attention, "apply") as differentiable,
            torch.set_grad_enabled(grad_enabled),
    ):
        assert gfx950.flash_attn(q, k, v, space="smoke") is sentinel

    inference.assert_called_once_with(q, k, v, 64**-0.5, False, "smoke")
    differentiable.assert_not_called()


def _qkv(head_dim, *, requires_grad, context=None):
    context = context or (512 if head_dim == 64 else 1024)
    return [
        torch.randn(
            (1, 4, context, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        ).requires_grad_(requires_grad) for _ in range(3)
    ]


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_gfx950_forward_returns_natural_log_lse(causal):
    from triton.tlx.ops.kernels.flash_attn.gfx950 import _flash_attn_forward

    torch.manual_seed(1)
    q, k, v = _qkv(64, requires_grad=False, context=256)
    scale = q.shape[-1]**-0.5

    _out, lse = _flash_attn_forward(q, k, v, scale, causal, "smoke")
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        causal_mask = torch.ones((q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.bool).triu(1)
        scores.masked_fill_(causal_mask, float("-inf"))
    expected = torch.logsumexp(scores, dim=-1)

    assert lse.dtype is torch.float32
    assert lse.shape == q.shape[:-1]
    torch.testing.assert_close(lse, expected, atol=2e-2, rtol=2e-3)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_gfx950_forward_and_backward(head_dim, causal):
    from triton.tlx.ops import flash_attn

    torch.manual_seed(0)
    q, k, v = _qkv(head_dim, requires_grad=True)
    rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    if head_dim == 64 and not causal:
        grad_out = torch.randn((*q.shape[:-1], 2 * head_dim), device=q.device, dtype=q.dtype)[..., ::2]
        assert not grad_out.is_contiguous()
    else:
        grad_out = torch.randn_like(q)
    scale = None if causal else 0.75 * head_dim**-0.5

    if head_dim == 64 and not causal:
        # Exercise the public default and the STORE_LSE/ALLOW_PRESCALE_Q
        # autotuner-key ABI; the remaining cases use the smaller smoke space.
        out = flash_attn(q, k, v, causal=causal, sm_scale=scale)
    else:
        out = flash_attn(q, k, v, causal=causal, sm_scale=scale, space="smoke")
    reference = torch.nn.functional.scaled_dot_product_attention(
        rq,
        rk,
        rv,
        is_causal=causal,
        scale=scale,
    )

    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)
    out.backward(grad_out)
    reference.backward(grad_out)

    for name, actual, expected in zip(
        ("dq", "dk", "dv"),
        (q.grad, k.grad, v.grad),
        (rq.grad, rk.grad, rv.grad),
            strict=True,
    ):
        assert torch.isfinite(actual).all(), name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        tolerance = 5e-3 if head_dim == 64 else 1e-2
        assert relative_l2.item() < tolerance, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_rejects_fp16():
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = [torch.randn((1, 1, 128, 64), device="cuda", dtype=torch.float16) for _ in range(3)]
    with pytest.raises(InvalidInput, match="does not support float16"):
        flash_attn(q, k, v, space="smoke")


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_rejects_unknown_space():
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = _qkv(64, requires_grad=False)
    with pytest.raises(InvalidInput, match="does not provide space"):
        flash_attn(q, k, v, space="heuristic")


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_rejects_unsupported_backward_shape():
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = _qkv(64, requires_grad=True, context=192)
    with pytest.raises(InvalidInput, match="supported MHA shapes"):
        flash_attn(q, k, v, space="smoke")


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("operand", ("q", "k", "v"))
def test_flash_attn_gfx950_rejects_noncontiguous_backward_inputs(operand):
    from triton.tlx.ops import InvalidInput, flash_attn

    tensors = _qkv(64, requires_grad=True, context=256)
    padded = torch.randn(
        (*tensors[0].shape[:-1], 128),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    tensors[("q", "k", "v").index(operand)] = padded[..., ::2]
    assert not tensors[("q", "k", "v").index(operand)].is_contiguous()

    with pytest.raises(InvalidInput, match=rf"{operand} must be contiguous"):
        flash_attn(*tensors, space="smoke")
