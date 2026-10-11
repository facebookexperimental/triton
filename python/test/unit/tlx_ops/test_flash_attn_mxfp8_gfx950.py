"""L1 correctness for ``tlx.ops.flash_attn_mxfp8`` on gfx950. Forward only."""

import pytest
import torch
from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.flash_attn_mxfp8._shapes import CORRECTNESS_SHAPES

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="tlx.ops.flash_attn_mxfp8 on gfx950 requires CDNA4")

MULTI_WAVE_SHAPE = (1, 64, 1024, 128)


def _qkv(shape, *, requires_grad=False):
    # Clip to [-1, 1]: unbounded Gaussian outliers combine with MXFP8 score
    # noise to spike short causal rows past the threshold (rare but real, e.g.
    # max 0.30 over 5000 seeds unclipped). Clipping bounds the tail (~0.13 max
    # over 5000 seeds per shape) while keeping realistic magnitudes.
    return [((torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5).clamp(-1.0,
                                                                                   1.0)).requires_grad_(requires_grad)
            for _ in range(3)]


def _sdpa(q, k, v, causal, scale):
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=causal,
        scale=scale,
    )


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype_name", CORRECTNESS_SHAPES)
def test_flash_attn_mxfp8_fwd(Z, H, N_CTX, HEAD_DIM, causal, dtype_name):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv((Z, H, N_CTX, HEAD_DIM))
    scale = 0.5
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke")
    ref = _sdpa(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=0.2, rtol=0)


def _blackwell_mxfp8(x, rows, cols):
    # Blackwell's MXFP8 rule for each rows x cols block of x's last two dims:
    # E8M0 byte RCEIL(amax * fp32(1 / 448)) (cvt.rp.satfinite.ue8m0x2.f32),
    # E4M3 data x / 2**(byte - 127) rounded to nearest.
    z, h, n, d = x.shape
    blocks = x.float().reshape(z, h, n // rows, rows, d // cols, cols)
    amax = blocks.abs().amax(dim=(3, 5), keepdim=True)
    byte = ((amax * (1.0 / 448.0)).view(torch.int32) + 0x7FFFFF) >> 23
    data = (blocks * torch.exp2(127.0 - byte.float())).clamp(-448, 448).to(torch.float8_e4m3fn)
    return data.reshape(z, h, n, d), byte.reshape(z, h, n // rows, d // cols).to(torch.uint8)


def test_flash_attn_mxfp8_quantizers_match_blackwell():
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950 import quantize_mxfp8_head, quantize_mxfp8_v

    torch.manual_seed(20)
    shape = (2, 3, 512, 128)
    # Token magnitudes that vary across each 32-row block, and an all-zero block.
    x, v = [torch.randn(shape, device="cuda") * torch.randn(shape[:3] + (1, ), device="cuda").exp() for _ in range(2)]
    x[0, 0, :32, :32] = 0
    x, v = x.to(torch.bfloat16), v.to(torch.bfloat16)
    data, scale = quantize_mxfp8_head(x)
    ref_data, ref_scale = _blackwell_mxfp8(x, 32, 32)
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(scale, ref_scale.repeat_interleave(32, 2))
    data, scale = quantize_mxfp8_v(v)
    ref_data, ref_scale = _blackwell_mxfp8(v, 32, 1)
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(scale, ref_scale)


def test_flash_attn_mxfp8_fast_exp_bound():
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950

    torch.manual_seed(20)
    q, k, v = _qkv((1, 4, 1024, 128))
    # A flat softmax, where the bit-trick exp2 errs the most against the rest
    # of the MXFP8 error (about 9% more error here, 0-6% at sharper ones).
    sm_scale = 128**-0.5
    assert gfx950._default_config(False, q.shape[2])["pingpong"]
    q8, qs = gfx950.quantize_mxfp8_head(q)
    k8, ks = gfx950.quantize_mxfp8_head(k, pack_k=True)
    v8, vs = gfx950.quantize_mxfp8_v(v, transposed=True)
    fast = gfx950._launch_quantized(q8, k8, v8, qs, ks, vs, False, sm_scale).float()
    exact = gfx950._launch_quantized(q8, k8, v8, qs, ks, vs, False, sm_scale, fast_exp=False).float()
    ref = _sdpa(q.float(), k.float(), v.float(), False, sm_scale)

    def rms(t):
        return t.pow(2).mean().sqrt().item()

    assert rms(fast - ref) < 1.15 * rms(exact - ref)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_mxfp8_fwd_multiple_cta_waves(causal):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv(MULTI_WAVE_SHAPE)
    scale = 0.5
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke")
    ref = _sdpa(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0)


def test_flash_attn_mxfp8_rejects_backward():
    from triton.tlx.ops import UnsupportedBackward, flash_attn_mxfp8

    q, k, v = _qkv((1, 1, 256, 128), requires_grad=True)
    with pytest.raises(UnsupportedBackward, match="does not support backward"):
        flash_attn_mxfp8(q, k, v, space="smoke")


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
