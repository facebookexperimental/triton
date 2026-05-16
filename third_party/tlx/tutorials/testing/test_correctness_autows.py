"""
Correctness tests for autoWS (automatic warp specialization) flash attention
kernels. These kernels use tl.range(warp_specialize=True) and are compiled
with the automatic warp specialization flow, not the TLX manual WS flow.

Run:
    pytest third_party/tlx/tutorials/testing/test_correctness_autows.py
"""

import pytest

import torch

import triton

from triton.language.extra.tlx.tutorials.fused_attention_ws_device_tma import (
    attention as _autows_fa, )
from triton.language.extra.tlx.tutorials.fused_attention_ws_device_tma_dp import (
    attention as _autows_fa_dp, )

from triton._internal_testing import is_blackwell

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# =============================================================================
# Common utilities
# =============================================================================


class FlashAttention:
    """Common utilities for autoWS Flash Attention correctness tests."""

    # (Z, H, N_CTX, HEAD_DIM)
    SHAPES = [(4, 32, 8192, 128)]

    CONFIGS = {
        "autows_fa_dp": {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "SUBTILING": True,
            "SUBTILING_P": False,
            "VECT_MUL": 1,
            "FADD2_REDUCE": False,
            "GROUP_SIZE_N": 1,
        },
    }

    @staticmethod
    def create_inputs(Z, H, N_CTX, HEAD_DIM, dtype=torch.bfloat16):
        torch.manual_seed(20)
        q = torch.empty((Z, H, N_CTX, HEAD_DIM), device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5)
        k = torch.empty((Z, H, N_CTX, HEAD_DIM), device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5)
        v = torch.empty((Z, H, N_CTX, HEAD_DIM), device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5)
        return q, k, v

    @staticmethod
    def get_reference(q, k, v, sm_scale, causal):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=causal)


# =============================================================================
# AutoWS Flash Attention DP — Non-Causal
# =============================================================================


@pytest.mark.parametrize("SUBTILING", [True, False])
@pytest.mark.parametrize("SUBTILING_P", [True, False])
@pytest.mark.parametrize("VECT_MUL", [1, 3])
@pytest.mark.parametrize("FADD2_REDUCE", [False])
@pytest.mark.parametrize("BLOCK_N", [64, 128])
@pytest.mark.parametrize("GROUP_SIZE_N", [1])
@pytest.mark.parametrize("maxRegAutoWS", [152, 192])
@pytest.mark.parametrize("pingpongAutoWS", [True, False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_autows_fa_dp_non_causal(SUBTILING, SUBTILING_P, VECT_MUL, FADD2_REDUCE,
                                 BLOCK_N, GROUP_SIZE_N, maxRegAutoWS, pingpongAutoWS):
    config = FlashAttention.CONFIGS["autows_fa_dp"].copy()
    config["SUBTILING"] = SUBTILING
    config["SUBTILING_P"] = SUBTILING_P
    config["VECT_MUL"] = VECT_MUL
    config["FADD2_REDUCE"] = FADD2_REDUCE
    config["BLOCK_N"] = BLOCK_N
    config["GROUP_SIZE_N"] = GROUP_SIZE_N
    if maxRegAutoWS == 152:
        pytest.skip("maxRegAutoWS=152 fails accuracy after compiler upgrade, fix pending")
    config["maxRegAutoWS"] = maxRegAutoWS
    config["pingpongAutoWS"] = pingpongAutoWS
    sm_scale = 0.5
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal=False)
        tri_out = _autows_fa_dp(q, k, v, False, sm_scale, "ws_persistent", config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


# =============================================================================
# AutoWS Flash Attention DP — Causal
# =============================================================================


@pytest.mark.skip(reason="Causal DP kernel is broken on current commit, fix pending")
@pytest.mark.parametrize("SUBTILING", [True, False])
@pytest.mark.parametrize("SUBTILING_P", [False])
@pytest.mark.parametrize("VECT_MUL", [1, 3])
@pytest.mark.parametrize("FADD2_REDUCE", [False])
@pytest.mark.parametrize("BLOCK_N", [64, 128])
@pytest.mark.parametrize("GROUP_SIZE_N", [4])
@pytest.mark.parametrize("maxRegAutoWS", [152, 192])
@pytest.mark.parametrize("pingpongAutoWS", [False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_autows_fa_dp_causal(SUBTILING, SUBTILING_P, VECT_MUL, FADD2_REDUCE,
                             BLOCK_N, GROUP_SIZE_N, maxRegAutoWS, pingpongAutoWS):
    config = FlashAttention.CONFIGS["autows_fa_dp"].copy()
    config["SUBTILING"] = SUBTILING
    config["SUBTILING_P"] = SUBTILING_P
    config["VECT_MUL"] = VECT_MUL
    config["FADD2_REDUCE"] = FADD2_REDUCE
    config["BLOCK_N"] = BLOCK_N
    config["GROUP_SIZE_N"] = GROUP_SIZE_N
    config["maxRegAutoWS"] = maxRegAutoWS
    config["pingpongAutoWS"] = pingpongAutoWS
    sm_scale = 0.5
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal=True)
        tri_out = _autows_fa_dp(q, k, v, True, sm_scale, "ws_persistent", config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


# =============================================================================
# AutoWS Flash Attention with Compiler Data Partitioning — Non-Causal
# =============================================================================


@pytest.mark.parametrize("SUBTILING", [True, False])
@pytest.mark.parametrize("VECT_MUL", [0, 1])
@pytest.mark.parametrize("FADD2_REDUCE", [False])
@pytest.mark.parametrize("baseVariant", ["ws_persistent", "ws"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_autows_fa_non_causal(SUBTILING, VECT_MUL, FADD2_REDUCE, baseVariant):
    if SUBTILING:
        pytest.skip("SUBTILING=True leads to accuracy mismatch, fix pending")
    sm_scale = 0.5
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal=False)
        tri_out = _autows_fa(q, k, v, False, sm_scale, baseVariant,
                             SUBTILING, VECT_MUL, FADD2_REDUCE)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


# =============================================================================
# AutoWS Flash Attention with Compiler Data Partitioning — Causal
# =============================================================================


@pytest.mark.skip(reason="Causal kernel is broken on current commit, fix is yet to land")
@pytest.mark.parametrize("SUBTILING", [True, False])
@pytest.mark.parametrize("VECT_MUL", [0, 1])
@pytest.mark.parametrize("FADD2_REDUCE", [False])
@pytest.mark.parametrize("baseVariant", ["ws_persistent", "ws"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_autows_fa_causal(SUBTILING, VECT_MUL, FADD2_REDUCE, baseVariant):
    sm_scale = 0.5
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal=True)
        tri_out = _autows_fa(q, k, v, True, sm_scale, baseVariant,
                             SUBTILING, VECT_MUL, FADD2_REDUCE)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
