"""L1 correctness for the Hopper ``tlx.ops.flash_attn`` backend."""

import pytest
import torch
from triton._internal_testing import is_hopper

pytestmark = pytest.mark.skipif(not is_hopper(), reason="requires an sm90 GPU")

ARCH = "sm90"
SHAPES = [
    (1, 1, 1024, 128, False),
    (1, 1, 1024, 128, True),
]


def _qkv(Z, H, N_CTX, HEAD_DIM, dtype, requires_grad=False):
    return [
        torch.randn(
            (Z, H, N_CTX, HEAD_DIM),
            device="cuda",
            dtype=dtype,
        ).requires_grad_(requires_grad) for _ in range(3)
    ]


def _sdpa(q, k, v, causal, scale=None):
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=causal,
        scale=scale,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, causal", SHAPES)
def test_flash_attn_fwd(Z, H, N_CTX, HEAD_DIM, causal, dtype):
    from triton.tlx.ops import flash_attn

    torch.manual_seed(0)
    q, k, v = _qkv(Z, H, N_CTX, HEAD_DIM, dtype)
    scale = None if causal else 0.7
    out = flash_attn(
        q,
        k,
        v,
        causal=causal,
        sm_scale=scale,
        arch=ARCH,
        space="smoke",
    )
    ref = _sdpa(q, k, v, causal, scale=scale)
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, causal", SHAPES)
def test_flash_attn_bwd(Z, H, N_CTX, HEAD_DIM, causal, dtype):
    from triton.tlx.ops import flash_attn

    torch.manual_seed(0)
    q, k, v = _qkv(Z, H, N_CTX, HEAD_DIM, dtype, requires_grad=True)
    rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    do = torch.randn_like(q)

    flash_attn(q, k, v, causal=causal, arch=ARCH, space="smoke").backward(do)
    _sdpa(rq, rk, rv, causal).backward(do)

    for got, expected in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, expected, atol=0.2, rtol=0.1)


def test_flash_attn_rejects_d64():
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = _qkv(1, 1, 128, 64, torch.float16)
    with pytest.raises(InvalidInput, match="does not support"):
        flash_attn(q, k, v, arch=ARCH, space="smoke")
