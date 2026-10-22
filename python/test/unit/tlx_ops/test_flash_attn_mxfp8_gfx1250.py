"""L1 correctness for ``tlx.ops.flash_attn_mxfp8`` on gfx1250. Forward only."""

import pytest
import torch
from triton._internal_testing import is_hip_gfx1250
from triton.tlx.ops.kernels.flash_attn_mxfp8._shapes import CORRECTNESS_SHAPES

pytestmark = pytest.mark.skipif(not is_hip_gfx1250(), reason="tlx.ops.flash_attn_mxfp8 on gfx1250 requires gfx1250")

MULTI_WAVE_SHAPE = (1, 64, 1024, 128)


def _qkv(shape, *, requires_grad=False):
    # Clipped to [-1, 1] as in the gfx950 tests: unbounded Gaussian outliers
    # combine with MXFP8 score noise to spike short causal rows.
    return [((torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5).clamp(-1.0,
                                                                                   1.0)).requires_grad_(requires_grad)
            for _ in range(3)]


def _sdpa(q, k, v, causal, scale):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype_name", CORRECTNESS_SHAPES)
def test_flash_attn_mxfp8_fwd(Z, H, N_CTX, HEAD_DIM, causal, dtype_name):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv((Z, H, N_CTX, HEAD_DIM))
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=0.5, space="smoke")
    torch.testing.assert_close(out, _sdpa(q, k, v, causal, 0.5), atol=0.2, rtol=0)


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


def test_flash_attn_mxfp8_quantizer_matches_blackwell():
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx1250 import quantize_mxfp8_qkv

    torch.manual_seed(20)
    shape = (2, 3, 512, 128)
    # Token magnitudes that vary across each 32-row block, and an all-zero block.
    q, k, v = [
        torch.randn(shape, device="cuda") * torch.randn(shape[:3] + (1, ), device="cuda").exp() for _ in range(3)
    ]
    q[0, 0, :32, :32] = 0
    v[1, 2, 64:96, :] = 0
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    q8, k8, v8, qs, ks, vs = quantize_mxfp8_qkv(q, k, v)
    for x, data, scale in ((q, q8, qs), (k, k8, ks)):
        ref_data, ref_scale = _blackwell_mxfp8(x, 32, 32)
        assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
        assert torch.equal(scale, ref_scale.repeat_interleave(32, 2))
    ref_data, ref_scale = _blackwell_mxfp8(v, 32, 1)
    z, h, n, d = shape
    # V's data are transposed per 128-key tile.
    assert torch.equal(v8.view(torch.uint8), ref_data.view(torch.uint8).reshape(z, h, n // 128, 128, d).transpose(3, 4))
    # Per 128-key tile, column d holds its four 32-key blocks' scales.
    assert torch.equal(vs, ref_scale.reshape(z, h, n // 128, 4, d).transpose(3, 4))


# Causal Q tiles of 256 rows: one tile (only the masked steps), two (one
# unmasked step) and more.
@pytest.mark.parametrize("shape", [(1, 1, 256, 128), (1, 2, 512, 128), (1, 3, 768, 128), (2, 2, 1280, 128)])
def test_flash_attn_mxfp8_fwd_causal_lengths(shape):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv(shape)
    out = flash_attn_mxfp8(q, k, v, causal=True, sm_scale=0.5, space="smoke")
    torch.testing.assert_close(out, _sdpa(q, k, v, True, 0.5), atol=0.2, rtol=0)


# 1024 takes the persistent kernels.
@pytest.mark.parametrize("causal,n_ctx", [(False, 1024), (True, 1024)])
def test_flash_attn_mxfp8_fast_exp_bound(causal, n_ctx):
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx1250

    torch.manual_seed(20)
    q, k, v = _qkv((1, 4, n_ctx, 128))
    # A flat softmax, where the bit-trick exp2 errs the most against the rest
    # of the MXFP8 error.
    sm_scale = 128**-0.5
    args = gfx1250.quantize_mxfp8_qkv(q, k, v)
    fast = gfx1250._launch_quantized(*args, causal, sm_scale).float()
    exact = gfx1250._launch_quantized(*args, causal, sm_scale, fast_exp=False).float()
    ref = _sdpa(q.float(), k.float(), v.float(), causal, sm_scale)

    def rms(t):
        return t.pow(2).mean().sqrt().item()

    assert rms(fast - ref) < 1.15 * rms(exact - ref)


# A zero sm_scale weighs all visible keys equally: the output is the mean of V
# over them. With V = 1 only NaNs show; V growing with the key also shows keys
# missing from the mean. Causal calls take the non-persistent kernel here.
@pytest.mark.parametrize("fast_exp", [True, False])
@pytest.mark.parametrize("causal,n_ctx,v_kind", [(True, 512, "ones"), (False, 1024, "key"), (True, 1024, "key")])
def test_flash_attn_mxfp8_zero_scale(causal, n_ctx, v_kind, fast_exp):
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx1250

    torch.manual_seed(20)
    shape = (1, 2, n_ctx, 128)
    if v_kind == "ones":
        q = k = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
        v = torch.ones_like(q)
    else:
        q, k, _ = _qkv(shape)
        v = (torch.arange(n_ctx, device="cuda") / n_ctx)[:, None].expand(shape).to(torch.bfloat16).contiguous()
    out = gfx1250._launch_quantized(*gfx1250.quantize_mxfp8_qkv(q, k, v), causal, 0.0, fast_exp=fast_exp)
    torch.testing.assert_close(out.float(), _sdpa(q.float(), k.float(), v.float(), causal, 0.0), atol=0.02, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_mxfp8_fwd_multiple_cta_waves(causal):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv(MULTI_WAVE_SHAPE)
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=0.5, space="smoke")
    torch.testing.assert_close(out, _sdpa(q, k, v, causal, 0.5), atol=0.15, rtol=0)


def test_flash_attn_mxfp8_rejects_backward():
    from triton.tlx.ops import UnsupportedBackward, flash_attn_mxfp8

    q, k, v = _qkv((1, 1, 256, 128), requires_grad=True)
    with pytest.raises(UnsupportedBackward, match="does not support backward"):
        flash_attn_mxfp8(q, k, v, space="smoke")


@pytest.mark.parametrize(
    "shape,dtype",
    [((1, 1, 256, 64), torch.bfloat16), ((1, 1, 384, 128), torch.bfloat16), ((1, 1, 256, 128), torch.float16)],
)
def test_flash_attn_mxfp8_rejects_unsupported_inputs(shape, dtype):
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q, k, v = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
    with pytest.raises(InvalidInput, match="does not support"):
        flash_attn_mxfp8(q, k, v, space="smoke")


def test_flash_attn_mxfp8_rejects_mismatched_shapes():
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q = torch.randn((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(InvalidInput, match="identical shapes"):
        flash_attn_mxfp8(q, k, torch.randn_like(k), space="smoke")


def test_flash_attn_mxfp8_rejects_unknown_space():
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q, k, v = _qkv((1, 1, 256, 128))
    with pytest.raises(InvalidInput, match="does not provide space"):
        flash_attn_mxfp8(q, k, v, space="heuristic")
