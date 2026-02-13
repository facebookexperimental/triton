import pytest

import torch

import triton

from triton.language.extra.tlx.tutorials.blackwell_gemm_ws import (
    matmul as _blackwell_gemm_ws, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_clc import (
    matmul as _blackwell_gemm_clc, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_pipelined import (
    matmul as _blackwell_gemm_pipelined, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_2cta import (
    matmul as _blackwell_gemm_2cta, )
from triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent import (
    attention as _blackwell_fa_ws_pipelined_persistent, )
from triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent_mxfp8 import (
    attention as _blackwell_fa_ws_pipelined_persistent_mxfp8,
    generate_attention_inputs as _generate_mxfp8_attention_inputs,
)
from triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined import (
    attention as _blackwell_fa_ws_pipelined, )
from triton.language.extra.tlx.tutorials.blackwell_fa_ws_persistent import (
    attention as _blackwell_fa_ws_persistent, )
from triton.language.extra.tlx.tutorials.blackwell_fa_ws import (
    attention as _blackwell_fa_ws, )
from triton.language.extra.tlx.tutorials.hopper_gemm_pipelined import (
    matmul as _hopper_gemm_pipelined, )
from triton.language.extra.tlx.tutorials.hopper_gemm_ws import (
    matmul as _hopper_gemm_ws, )
from triton.language.extra.tlx.tutorials.hopper_fa_ws_pipelined_pingpong_persistent import (
    attention as _hopper_fa_ws_pipelined_pingpong_persistent, )
from triton.language.extra.tlx.tutorials.hopper_fa_ws_pipelined_pingpong import (
    attention as _hopper_fa_ws_pipelined_pingpong, )
from triton.language.extra.tlx.tutorials.hopper_fa_ws_pipelined import (
    attention as _hopper_fa_ws_pipelined, )
from triton.language.extra.tlx.tutorials.hopper_fa_ws import (
    attention as _hopper_fa_ws, )

from triton._internal_testing import is_blackwell, is_hopper

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# =============================================================================
# GEMM: Common utilities and configs
# =============================================================================


class Gemm:
    """Common utilities and configs for GEMM tests."""

    SHAPES = [(4096, 4096, 4096), (8192, 8192, 8192)]

    CONFIGS = {
        "blackwell_gemm_ws": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "NUM_SMEM_BUFFERS": 2,
            "NUM_TMEM_BUFFERS": 2,
            "EPILOGUE_SUBTILE": 1,
            "PAIR_CTA": False,
        },
        "blackwell_gemm_clc": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "NUM_SMEM_BUFFERS": 2,
            "NUM_TMEM_BUFFERS": 2,
            "EPILOGUE_SUBTILE": True,
        },
        "blackwell_gemm_pipelined": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "NUM_STAGES": 4,
        },
        "blackwell_gemm_2cta": None,  # Uses fixed config internally
        "hopper_gemm_pipelined": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "NUM_STAGES": 3,
        },
        "hopper_gemm_ws": {
            "BM": 128,
            "BN": 256,
            "BK": 64,
            "GROUP_SIZE_M": 8,
            "NUM_STAGES": 3,
            "NUM_MMA_WARPS": 8,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": False,
        },
    }

    @staticmethod
    def run_test(matmul_fn, config, shapes=None):
        if shapes is None:
            shapes = Gemm.SHAPES
        for shape in shapes:
            M, N, K = shape
            torch.manual_seed(0)
            a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
            b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
            torch_output = torch.matmul(a, b)
            triton_output = matmul_fn(a, b, config=config)
            torch.testing.assert_close(triton_output, torch_output)


# =============================================================================
# Flash Attention: Common utilities and configs
# =============================================================================


class FlashAttention:
    """Common utilities and configs for Flash Attention tests."""

    # (Z, H, N_CTX, HEAD_DIM)
    SHAPES = [(4, 8, 1024, 128)]

    CONFIGS = {
        "blackwell_fa_ws": {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_KV": 3,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
        },
        "blackwell_fa_ws_persistent": {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 3,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
        },
        "blackwell_fa_ws_pipelined": {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_KV": 3,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
        },
        "blackwell_fa_ws_pipelined_persistent": {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 3,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_MMA_SLICES": 2,
            "GROUP_SIZE_N": 1,
        },
        "blackwell_fa_ws_pipelined_persistent_mxfp8": {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 3,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "GROUP_SIZE_N": 1,
        },
        "hopper_fa_ws": {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "NUM_BUFFERS": 2,
            "NUM_MMA_WARPS": 8,
            "NUM_MMA_GROUPS": 2,
        },
        "hopper_fa_ws_pipelined": {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "NUM_BUFFERS": 2,
            "NUM_MMA_WARPS": 8,
            "NUM_MMA_GROUPS": 2,
        },
        "hopper_fa_ws_pipelined_pingpong": {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "NUM_BUFFERS": 2,
            "NUM_MMA_WARPS": 8,
            "NUM_MMA_GROUPS": 2,
        },
        "hopper_fa_ws_pipelined_pingpong_persistent": {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 2,
            "NUM_MMA_WARPS": 8,
            "NUM_MMA_GROUPS": 2,
        },
    }

    @staticmethod
    def create_inputs(Z, H, N_CTX, HEAD_DIM, dtype=torch.float16):
        torch.manual_seed(20)
        q = torch.empty((Z, H, N_CTX, HEAD_DIM), device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()
        k = torch.empty((Z, H, N_CTX, HEAD_DIM), device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()
        v = torch.empty((Z, H, N_CTX, HEAD_DIM), device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()
        return q, k, v

    @staticmethod
    def get_reference(q, k, v, sm_scale, causal):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=causal)


# =============================================================================
# Blackwell GEMM Tests
# =============================================================================


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws():
    Gemm.run_test(_blackwell_gemm_ws, Gemm.CONFIGS["blackwell_gemm_ws"])


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_clc():
    Gemm.run_test(_blackwell_gemm_clc, Gemm.CONFIGS["blackwell_gemm_clc"])


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_pipelined():
    Gemm.run_test(_blackwell_gemm_pipelined, Gemm.CONFIGS["blackwell_gemm_pipelined"])


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_2cta():
    Gemm.run_test(_blackwell_gemm_2cta, Gemm.CONFIGS["blackwell_gemm_2cta"])


# =============================================================================
# Blackwell Flash Attention Tests
# =============================================================================


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws():
    config = FlashAttention.CONFIGS["blackwell_fa_ws"]
    sm_scale = 0.5
    causal = False  # ws kernel doesn't support causal attention
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _blackwell_fa_ws(q, k, v, sm_scale, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_persistent():
    config = FlashAttention.CONFIGS["blackwell_fa_ws_persistent"]
    sm_scale = 0.5
    causal = True
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _blackwell_fa_ws_persistent(q, k, v, sm_scale, causal, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined():
    config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined"]
    sm_scale = 0.5
    causal = True
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _blackwell_fa_ws_pipelined(q, k, v, sm_scale, causal, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent():
    config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent"]
    sm_scale = 0.5
    causal = True
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _blackwell_fa_ws_pipelined_persistent(q, k, v, sm_scale, causal, 64, 1, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_mxfp8(HEAD_DIM, causal):
    config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent_mxfp8"]
    if HEAD_DIM == 128:
        pytest.skip("HEAD_DIM=128 not supported yet")
    sm_scale = 0.5
    dtype = torch.float8_e4m3fn
    shapes = [(8, 16, 1024)]
    for Z, H, N_CTX in shapes:
        torch.manual_seed(20)
        shape = (Z, H, N_CTX, HEAD_DIM)
        (q, q_scale, q_ref), (k, k_scale, k_ref), (v, v_scale,
                                                   v_ref) = _generate_mxfp8_attention_inputs(shape, DEVICE, dtype)
        ref_out = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, scale=sm_scale,
                                                                   is_causal=causal)
        tri_out = _blackwell_fa_ws_pipelined_persistent_mxfp8(q, k, v, q_scale, k_scale, v_scale, sm_scale, causal,
                                                              config=config)
        tri_out = tri_out.to(ref_out.dtype)
        if causal:
            atol = 0.1
        else:
            atol = 0.03
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)


# =============================================================================
# Hopper GEMM Tests
# =============================================================================


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper GPU")
def test_hopper_gemm_pipelined():
    Gemm.run_test(_hopper_gemm_pipelined, Gemm.CONFIGS["hopper_gemm_pipelined"])


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper GPU")
def test_hopper_gemm_ws():
    Gemm.run_test(_hopper_gemm_ws, Gemm.CONFIGS["hopper_gemm_ws"])


# =============================================================================
# Hopper Flash Attention Tests
# =============================================================================


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper GPU")
def test_hopper_fa_ws():
    config = FlashAttention.CONFIGS["hopper_fa_ws"]
    sm_scale = 0.5
    causal = False
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _hopper_fa_ws(q, k, v, sm_scale, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper GPU")
def test_hopper_fa_ws_pipelined():
    config = FlashAttention.CONFIGS["hopper_fa_ws_pipelined"]
    sm_scale = 0.5
    causal = False
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _hopper_fa_ws_pipelined(q, k, v, sm_scale, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper GPU")
def test_hopper_fa_ws_pipelined_pingpong():
    config = FlashAttention.CONFIGS["hopper_fa_ws_pipelined_pingpong"]
    sm_scale = 0.5
    causal = False
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _hopper_fa_ws_pipelined_pingpong(q, k, v, sm_scale, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper GPU")
def test_hopper_fa_ws_pipelined_pingpong_persistent():
    config = FlashAttention.CONFIGS["hopper_fa_ws_pipelined_pingpong_persistent"]
    sm_scale = 0.5
    causal = False
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _hopper_fa_ws_pipelined_pingpong_persistent(q, k, v, sm_scale, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
