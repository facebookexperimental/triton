"""Test TLX Flash Attention with asymmetric head dimensions (DSv3: HD_K=192, HD_V=128).

Tests:
1. Symmetric HD=128 (baseline, should always work)
2. Asymmetric HD_K=192, HD_V=128 (DSv3 config)
3. Symmetric HD=64 (small baseline)
"""
import pytest
import torch


# Skip on non-SM100 (Blackwell) GPUs
def _is_sm100():
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10 and props.minor == 0


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="SM100 only")


def _ref_attention(q, k, v, sm_scale):
    """Reference attention implementation."""
    # q: (Z, H, N, D_K), k: (Z, H, N, D_K), v: (Z, H, N, D_V)
    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v.float())
    return out


def test_symmetric_hd128():
    """Baseline: symmetric HD=128."""
    from triton.language.extra.tlx.tutorials.blackwell_fa_ws import attention

    Z, H, N, D = 1, 1, 256, 128
    q = torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16)
    sm_scale = 1.0 / (D**0.5)

    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "NUM_BUFFERS_K": 2,
        "NUM_BUFFERS_V": 2,
        "NUM_BUFFERS_QK": 1,
        "NUM_MMA_GROUPS": 1,
    }
    out = attention(q, k, v, sm_scale, config=config)
    ref = _ref_attention(q, k, v, sm_scale)

    max_diff = (out.float() - ref).abs().max().item()
    rel_err = max_diff / ref.abs().max().item()
    print(f"\nSymmetric HD=128: max_diff={max_diff:.6f}, rel_err={rel_err:.6f}")
    assert rel_err < 0.05, f"Symmetric HD=128 failed: rel_err={rel_err}"


def test_dsv3_asymmetric_padded():
    """DSv3: HD_K=192 (QK), HD_V=128 (PV) -- with pow2 padding.

    HD_K=192 is NPOT and gets padded to 256.
    Use BLOCK_N=64 to fit in SMEM (K: 64x256x2=32KB per buf, V: 64x128x2=16KB per buf).
    SMEM: Q=128x256x2=64KB + K=32x2=64KB + V=16x2=32KB = 160KB (fits 227KB limit).
    """
    from triton.language.extra.tlx.tutorials.blackwell_fa_ws import attention

    Z, H, N = 1, 1, 256
    D_K, D_V = 192, 128
    q = torch.randn(Z, H, N, D_K, device="cuda", dtype=torch.float16)
    k = torch.randn(Z, H, N, D_K, device="cuda", dtype=torch.float16)
    v = torch.randn(Z, H, N, D_V, device="cuda", dtype=torch.float16)
    sm_scale = 1.0 / (D_K**0.5)

    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "NUM_BUFFERS_K": 2,
        "NUM_BUFFERS_V": 2,
        "NUM_BUFFERS_QK": 1,
        "NUM_MMA_GROUPS": 1,
    }
    out = attention(q, k, v, sm_scale, config=config)

    assert out.shape == (Z, H, N, D_V), f"Wrong output shape: {out.shape}"

    ref = _ref_attention(q, k, v, sm_scale)
    max_diff = (out.float() - ref).abs().max().item()
    rel_err = max_diff / ref.abs().max().item()
    print(f"\nDSv3 HD_K=192(padded->256), HD_V=128, BN=64: max_diff={max_diff:.6f}, rel_err={rel_err:.6f}")
    assert rel_err < 0.05, f"DSv3 asymmetric (padded) failed: rel_err={rel_err}"


def test_dsv3_asymmetric_native():
    """DSv3: HD_K=192 (QK), HD_V=128 (PV) -- native NPOT (no padding).

    This test verifies that native NPOT K=192 works in the warp-specialized kernel.
    Uses BLOCK_N=64 to fit SMEM budget with native 192.
    SMEM: Q=128x192x2=48KB + K=64x192x2=24KB*2=48KB + V=64x128x2=16KB*2=32KB = 128KB (fits).
    """
    import os
    # Save/restore TRITON_ALLOW_NPOT so it does not leak into later tests.
    _prev_allow_npot = os.environ.get("TRITON_ALLOW_NPOT")
    os.environ["TRITON_ALLOW_NPOT"] = "1"
    from triton.language.extra.tlx.tutorials.blackwell_fa_ws import attention

    Z, H, N = 1, 1, 256
    D_K, D_V = 192, 128
    q = torch.randn(Z, H, N, D_K, device="cuda", dtype=torch.float16)
    k = torch.randn(Z, H, N, D_K, device="cuda", dtype=torch.float16)
    v = torch.randn(Z, H, N, D_V, device="cuda", dtype=torch.float16)
    sm_scale = 1.0 / (D_K**0.5)

    # Monkey-patch to disable padding (force native NPOT)
    import triton.language.extra.tlx.tutorials.blackwell_fa_ws as fa_mod
    orig_needs_padding = fa_mod._needs_hd_padding
    fa_mod._needs_hd_padding = lambda x: False

    try:
        config = {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "NUM_BUFFERS_K": 2,
            "NUM_BUFFERS_V": 2,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 1,
        }
        out = attention(q, k, v, sm_scale, config=config)
        torch.cuda.synchronize()
    finally:
        fa_mod._needs_hd_padding = orig_needs_padding
        if _prev_allow_npot is None:
            os.environ.pop("TRITON_ALLOW_NPOT", None)
        else:
            os.environ["TRITON_ALLOW_NPOT"] = _prev_allow_npot

    assert out.shape == (Z, H, N, D_V), f"Wrong output shape: {out.shape}"

    ref = _ref_attention(q, k, v, sm_scale)
    max_diff = (out.float() - ref).abs().max().item()
    rel_err = max_diff / ref.abs().max().item()
    print(f"\nDSv3 HD_K=192(native NPOT), HD_V=128: max_diff={max_diff:.6f}, rel_err={rel_err:.6f}")
    assert rel_err < 0.05, f"DSv3 asymmetric (native NPOT) failed: rel_err={rel_err}"


def test_symmetric_hd64():
    """Baseline: symmetric HD=64."""
    from triton.language.extra.tlx.tutorials.blackwell_fa_ws import attention

    Z, H, N, D = 1, 2, 512, 64
    q = torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16)
    sm_scale = 1.0 / (D**0.5)

    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "NUM_BUFFERS_K": 2,
        "NUM_BUFFERS_V": 2,
        "NUM_BUFFERS_QK": 1,
        "NUM_MMA_GROUPS": 1,
    }
    out = attention(q, k, v, sm_scale, config=config)
    ref = _ref_attention(q, k, v, sm_scale)

    max_diff = (out.float() - ref).abs().max().item()
    rel_err = max_diff / ref.abs().max().item()
    print(f"\nSymmetric HD=64: max_diff={max_diff:.6f}, rel_err={rel_err:.6f}")
    assert rel_err < 0.05, f"Symmetric HD=64 failed: rel_err={rel_err}"
