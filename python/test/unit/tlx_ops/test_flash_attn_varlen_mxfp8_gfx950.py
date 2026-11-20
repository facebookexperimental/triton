"""L1 correctness for ``tlx.ops.flash_attn_varlen_mxfp8`` on gfx950."""

import itertools

import pytest
import torch
from triton._internal_testing import is_hip_cdna4

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="tlx.ops.flash_attn_varlen_mxfp8 requires CDNA4")


def _cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)


def _reference(q, k, v, cu_q, cu_k, scale):
    out = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    lse = torch.full((q.shape[1], q.shape[0]), float("-inf"), device=q.device)
    for b in range(cu_q.numel() - 1):
        q0, q1 = cu_q[b].item(), cu_q[b + 1].item()
        k0, k1 = cu_k[b].item(), cu_k[b + 1].item()
        if q1 == q0 or k1 == k0:
            continue
        qs = q[q0:q1].float()
        ks = k[k0:k1].float()
        vs = v[k0:k1].float()
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        lse[:, q0:q1] = torch.logsumexp(scores, dim=-1)
        out[q0:q1] = torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), vs)
    return out, lse


@pytest.mark.parametrize(
    "lens_q,lens_k,fast_exp",
    [
        ([96, 128], [160, 192], False),
        ([320, 512], [384, 640], False),
        ([320, 512], [4096, 5000], False),
        ([0, 320, 0, 128], [0, 384, 0, 0], False),
        ([320, 512], [384, 640], True),
        ([320, 512, 320], [100, 300, 4100], True),
        ([128, 0, 320], [0, 300, 100], False),
    ],
)
def test_flash_attn_varlen_mxfp8_four_wave(lens_q, lens_k, fast_exp):
    from triton.tlx.ops import flash_attn_varlen_mxfp8

    torch.manual_seed(20)
    q = torch.randn(sum(lens_q), 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lens_k), 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)
    out = flash_attn_varlen_mxfp8(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(lens_q),
        max(lens_k),
        causal=False,
        fast_exp=fast_exp,
    )
    ref, _ = _reference(q, k, v, cu_q, cu_k, 128**-0.5)
    assert out.shape == q.shape and out.dtype == q.dtype
    torch.testing.assert_close(out.float(), ref, atol=0.08, rtol=0.08)


def test_flash_attn_varlen_mxfp8_prequantized():
    from triton.tlx.ops.kernels.flash_attn_varlen_mxfp8 import gfx950

    torch.manual_seed(20)
    lens_q, lens_k = [320, 512], [100, 300]
    q = torch.randn(sum(lens_q), 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lens_k), 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)
    scale = 128**-0.5

    q8, q_scales = gfx950.prepare_q(q, cu_q)
    k8, k_scales, v8, v_scales = gfx950.prepare_kv(k, v, cu_k)
    out_pre, lse_pre = gfx950.forward_prequantized(
        q8,
        q_scales,
        k8,
        k_scales,
        v8,
        v_scales,
        cu_q,
        cu_k,
        scale,
        total_kv=k.shape[0],
    )
    out, lse, _ = gfx950.forward(q, k, v, cu_q, cu_k, scale)
    ref, ref_lse = _reference(q, k, v, cu_q, cu_k, scale)

    torch.testing.assert_close(out_pre, out, atol=0, rtol=0)
    torch.testing.assert_close(lse_pre, lse, atol=0, rtol=0)
    torch.testing.assert_close(out.float(), ref, atol=0.08, rtol=0.08)
    torch.testing.assert_close(lse, ref_lse, atol=0.25, rtol=0.05)


@pytest.mark.parametrize("lens_k,fast", [([384, 640], False), ([4096, 5000], True)])
def test_flash_attn_varlen_mxfp8_default_fast_exp(lens_k, fast):
    """Without fast_exp, sequences averaging 4096 keys or more get the approximate exponential."""
    from triton.tlx.ops import flash_attn_varlen_mxfp8
    from triton.tlx.ops.kernels.flash_attn_varlen_mxfp8 import gfx950

    torch.manual_seed(20)
    lens_q = [320, 512]
    q = torch.randn(sum(lens_q), 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lens_k), 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)
    scale = 128**-0.5

    kv_state = gfx950.prepare_kv(k, v, cu_k)
    out, lse, _ = gfx950.forward(q, k, v, cu_q, cu_k, scale, kv_state=kv_state)
    for explicit in (False, True):
        out_e, lse_e, _ = gfx950.forward(q, k, v, cu_q, cu_k, scale, fast_exp=explicit, kv_state=kv_state)
        assert torch.equal(lse, lse_e) == (explicit == fast)
        assert torch.equal(out, out_e) == (explicit == fast)
    torch.testing.assert_close(flash_attn_varlen_mxfp8(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k)), out, atol=0,
                               rtol=0)
    ref, ref_lse = _reference(q, k, v, cu_q, cu_k, scale)
    torch.testing.assert_close(out.float(), ref, atol=0.08, rtol=0.08)
    torch.testing.assert_close(lse, ref_lse, atol=0.25, rtol=0.05)


def test_flash_attn_varlen_mxfp8_rejects_invalid_inputs():
    from triton.tlx.ops import InvalidInput, flash_attn_varlen_mxfp8

    q = torch.randn(8, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(16, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(15, 4, 128, device="cuda", dtype=torch.bfloat16)
    cu_q = _cu([8])
    cu_k = _cu([16])
    with pytest.raises(InvalidInput, match="shape"):
        flash_attn_varlen_mxfp8(q, k, v, cu_q, cu_k, 8, 16)
    empty_offsets = torch.empty(0, device="cuda", dtype=torch.int32)
    with pytest.raises(InvalidInput, match="sequence offsets"):
        flash_attn_varlen_mxfp8(q, k, k, empty_offsets, empty_offsets, 8, 16)
