"""L1 correctness for ``tlx.ops.flash_attn_mxfp8`` on Blackwell."""

import pytest
import torch
from triton._internal_testing import is_blackwell

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.flash_attn_mxfp8 requires sm100")

ARCH = "sm100"
SHAPES = [
    (1, 1, 256, 128, False),
    (1, 1, 256, 128, True),
]
MULTI_WAVE_SHAPE = (1, 64, 1024, 128)


def _qkv(shape, *, requires_grad=False):
    return [
        (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5).requires_grad_(requires_grad)
        for _ in range(3)
    ]


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


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal", SHAPES)
def test_flash_attn_mxfp8_fwd(Z, H, N_CTX, HEAD_DIM, causal):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv((Z, H, N_CTX, HEAD_DIM))
    scale = 0.5
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, arch=ARCH, space="smoke")
    ref = _sdpa(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=0.12, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_mxfp8_fwd_multiple_cta_waves(causal):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv(MULTI_WAVE_SHAPE)
    scale = 0.5
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, arch=ARCH, space="smoke")
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

    flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, arch=ARCH, space="smoke").backward(do)
    _sdpa(rq, rk, rv, causal, scale).backward(do)

    for label, actual, expected in (("dq", q.grad, rq.grad), ("dk", k.grad, rk.grad), ("dv", v.grad, rv.grad)):
        cosine = _cosine(actual, expected)
        assert cosine >= 0.98, f"{label} cosine_similarity={cosine:.6f}"


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal", SHAPES)
def test_flash_attn_mxfp8_bwd(Z, H, N_CTX, HEAD_DIM, causal):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv((Z, H, N_CTX, HEAD_DIM), requires_grad=True)
    rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    scale = 0.5
    do = torch.randn_like(q)

    flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, arch=ARCH, space="smoke").backward(do)
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
        flash_attn_mxfp8(q, k, v, arch=ARCH, space="smoke")


def test_flash_attn_mxfp8_rejects_mismatched_shapes():
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q = torch.randn((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    with pytest.raises(InvalidInput, match="identical shapes"):
        flash_attn_mxfp8(q, k, v, arch=ARCH, space="smoke")


def test_flash_attn_mxfp8_rejects_unknown_space():
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q, k, v = _qkv((1, 1, 256, 128))
    with pytest.raises(InvalidInput, match="does not provide space"):
        flash_attn_mxfp8(q, k, v, arch=ARCH, space="heuristic")
