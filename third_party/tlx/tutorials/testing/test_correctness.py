import importlib.util
from pathlib import Path

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
try:
    from triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent_mxfp8 import (
        attention as _blackwell_fa_ws_pipelined_persistent_mxfp8,
        generate_attention_inputs as _generate_mxfp8_attention_inputs,
    )
except ModuleNotFoundError as e:
    if e.name != "torchao":
        raise
    _blackwell_fa_ws_pipelined_persistent_mxfp8 = None
    _generate_mxfp8_attention_inputs = None
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

from triton._internal_testing import is_blackwell, is_hip_gfx1250, is_hopper

DEVICE = triton.runtime.driver.active.get_active_torch_device()

_AMD_MXFP_GEMM_MODULE = None


def _load_amd_mxfp_gemm_module():
    global _AMD_MXFP_GEMM_MODULE
    if _AMD_MXFP_GEMM_MODULE is None:
        path = Path(__file__).resolve().parents[1] / "amd-mxfp-gemm-tdm-pipelined_test.py"
        spec = importlib.util.spec_from_file_location("tlx_amd_mxfp_gemm_tdm_pipelined_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _AMD_MXFP_GEMM_MODULE = module
    return _AMD_MXFP_GEMM_MODULE


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
            "NUM_MMA_GROUPS": 1,
            "EPILOGUE_SUBTILE": 1,
            "NUM_CTAS": 1,
            "SPLIT_K": 1,
            "INTERLEAVE_EPILOGUE": 0,
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
        "blackwell_gemm_ws_warp_barrier": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "NUM_SMEM_BUFFERS": 2,
            "NUM_TMEM_BUFFERS": 2,
            "NUM_MMA_GROUPS": 1,
            "EPILOGUE_SUBTILE": 1,
            "NUM_CTAS": 1,
            "SPLIT_K": 1,
            "INTERLEAVE_EPILOGUE": 0,
            "USE_WARP_BARRIER": True,
        },
        "blackwell_gemm_clc_warp_barrier": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "NUM_SMEM_BUFFERS": 2,
            "NUM_TMEM_BUFFERS": 2,
            "EPILOGUE_SUBTILE": True,
            "USE_WARP_BARRIER": True,
        },
        "hopper_gemm_ws_warp_barrier": {
            "BM": 128,
            "BN": 256,
            "BK": 64,
            "GROUP_SIZE_M": 8,
            "NUM_STAGES": 3,
            "NUM_MMA_WARPS": 8,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": False,
            "USE_WARP_BARRIER": True,
        },
    }

    @staticmethod
    def run_test(matmul_fn, config, shapes=None, dtype=torch.float16):
        if shapes is None:
            shapes = Gemm.SHAPES
        for shape in shapes:
            M, N, K = shape
            torch.manual_seed(0)
            a = torch.randn((M, K), device=DEVICE, dtype=dtype)
            b = torch.randn((K, N), device=DEVICE, dtype=dtype)
            torch_output = torch.matmul(a, b)
            triton_output = matmul_fn(a, b, config=config)
            torch.testing.assert_close(triton_output, torch_output)


class AmdMXFPGemm:
    """AMD gfx1250 MXFP GEMM e2e configs."""

    CONFIGS = [
        pytest.param(
            {
                "M": 256,
                "N": 256,
                "K": 768,
                "SCHEDULE": "sliceMNK",
                "DTYPE_A": "float8_e4m3",
                "DTYPE_B": "float4",
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 256,
                "TRANSPOSE_B": True,
                "NUM_BUFFERS": 3,
                "SCALE_PRESHUFFLE": True,
                "WITH_A_SCALE": True,
                "TDM_FUSION": "partial",
                "L2_PREFETCH_DISTANCE": -1,
                "TDM_SPLIT": True,
            },
            id="sliceMNK-split-partial-fp8e4m3-fp4-bT-3buf",
        ),
        pytest.param(
            {
                "M": 256,
                "N": 256,
                "K": 768,
                "SCHEDULE": "sliceMNK",
                "DTYPE_A": "float8_e4m3",
                "DTYPE_B": "float4",
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 256,
                "TRANSPOSE_B": False,
                "NUM_BUFFERS": 2,
                "SCALE_PRESHUFFLE": True,
                "WITH_A_SCALE": True,
                "TDM_FUSION": "partial",
                "L2_PREFETCH_DISTANCE": -1,
                "TDM_SPLIT": True,
            },
            id="sliceMNK-split-partial-fp8e4m3-fp4-b-2buf",
        ),
        pytest.param(
            {
                "M": 128,
                "N": 128,
                "K": 768,
                "SCHEDULE": "sliceMNK",
                "DTYPE_A": "float8_e4m3",
                "DTYPE_B": "float8_e4m3",
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 256,
                "TRANSPOSE_B": True,
                "NUM_BUFFERS": 3,
                "SCALE_PRESHUFFLE": True,
                "WITH_A_SCALE": True,
                "TDM_FUSION": "none",
                "L2_PREFETCH_DISTANCE": -1,
                "TDM_SPLIT": True,
            },
            id="sliceMNK-split-none-fp8e4m3-fp8e4m3-bT-128x128-3buf",
        ),
    ]

    @staticmethod
    def run_test(config):
        mxfp = _load_amd_mxfp_gemm_module()
        M = config["M"]
        N = config["N"]
        K = config["K"]
        dtype_a = config["DTYPE_A"]
        dtype_b = config["DTYPE_B"]

        torch.manual_seed(0)
        a = mxfp._init_data(dtype_a, M, K)
        b = mxfp._init_data(dtype_b, K, N)
        a_scale = mxfp.MXScaleTensor(size=(M, triton.cdiv(K, 32))).random(high=32.0).data
        b_scale = mxfp.MXScaleTensor(size=(N, triton.cdiv(K, 32))).random(high=32.0).data
        ref = mxfp.torch_gemm_mxfp(a, b, a_scale, b_scale, 32, M, N, K)

        a_scale_input = mxfp.pack_scale(a_scale) if config["SCALE_PRESHUFFLE"] else a_scale
        b_scale_input = mxfp.pack_scale(b_scale) if config["SCALE_PRESHUFFLE"] else b_scale
        if dtype_a == "float4":
            a = a.to_packed_tensor(dim=1)
        if dtype_b == "float4":
            b = b.to_packed_tensor(dim=0)

        a_d = a.data.contiguous().cuda() if dtype_a == "float4" else a.contiguous().cuda()
        if dtype_b == "float4":
            b_d = b.data.T.contiguous().cuda() if config["TRANSPOSE_B"] else b.data.contiguous().cuda()
        else:
            b_d = b.T.contiguous().cuda() if config["TRANSPOSE_B"] else b.contiguous().cuda()

        out = mxfp.mxgemm_tdm_pipelined(
            a_d,
            b_d,
            a_scale_input.cuda(),
            b_scale_input.cuda(),
            config["BLOCK_M"],
            config["BLOCK_N"],
            config["BLOCK_K"],
            config["TRANSPOSE_B"],
            config["NUM_BUFFERS"],
            mxfp.DTYPE_TO_TRITON[dtype_a],
            mxfp.DTYPE_TO_TRITON[dtype_b],
            config["SCALE_PRESHUFFLE"],
            config["WITH_A_SCALE"],
            config["SCHEDULE"],
            config["L2_PREFETCH_DISTANCE"],
            M,
            N,
            K,
            config["TDM_FUSION"],
            config["TDM_SPLIT"],
        )
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=2e-2)


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
        "blackwell_fa_ws_pipelined_persistent_warp_barrier": {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 3,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_MMA_SLICES": 2,
            "GROUP_SIZE_N": 1,
            "USE_WARP_BARRIER": True,
        },
        "blackwell_fa_ws_pipelined_persistent_mxfp8": {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 3,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_Q_SCALE_TMEM_BUFFERS": 1,
            "NUM_KV_SCALE_TMEM_BUFFERS": 2,
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws(dtype):
    Gemm.run_test(_blackwell_gemm_ws, Gemm.CONFIGS["blackwell_gemm_ws"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_clc(dtype):
    Gemm.run_test(_blackwell_gemm_clc, Gemm.CONFIGS["blackwell_gemm_clc"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_warp_barrier(dtype):
    Gemm.run_test(_blackwell_gemm_ws, Gemm.CONFIGS["blackwell_gemm_ws_warp_barrier"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_clc_warp_barrier(dtype):
    Gemm.run_test(_blackwell_gemm_clc, Gemm.CONFIGS["blackwell_gemm_clc_warp_barrier"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_pipelined(dtype):
    Gemm.run_test(_blackwell_gemm_pipelined, Gemm.CONFIGS["blackwell_gemm_pipelined"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_2cta(dtype):
    Gemm.run_test(_blackwell_gemm_2cta, Gemm.CONFIGS["blackwell_gemm_2cta"], dtype=dtype)


# =============================================================================
# AMD gfx1250 MXFP GEMM Tests
# =============================================================================


@pytest.mark.parametrize("config", AmdMXFPGemm.CONFIGS)
@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires AMD gfx1250 GPU")
def test_amd_mxfp_gemm_tdm_split(config):
    AmdMXFPGemm.run_test(config)


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


@pytest.mark.parametrize("RESCALE_OPT,USE_WHERE", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_warp_barrier(causal, RESCALE_OPT, USE_WHERE):
    config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent_warp_barrier"].copy()
    config["RESCALE_OPT"] = RESCALE_OPT
    config["USE_WHERE"] = USE_WHERE
    sm_scale = 0.5
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _blackwell_fa_ws_pipelined_persistent(q, k, v, sm_scale, causal, 64, 1, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_mxfp8(HEAD_DIM, causal):
    if _blackwell_fa_ws_pipelined_persistent_mxfp8 is None or _generate_mxfp8_attention_inputs is None:
        pytest.skip("Requires torchao")
    config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent_mxfp8"]
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
            if HEAD_DIM == 64:
                # Max atol measured was 0.09375
                atol = 0.1
            else:
                # Max atol measured was 0.10986328125
                assert HEAD_DIM == 128
                atol = 0.11
        else:
            if HEAD_DIM == 64:
                # Max atol measured was 0.029296875
                atol = 0.03
            else:
                # Max atol measured was 0.0625
                assert HEAD_DIM == 128
                atol = 0.07
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


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper GPU")
def test_hopper_gemm_ws_warp_barrier():
    Gemm.run_test(_hopper_gemm_ws, Gemm.CONFIGS["hopper_gemm_ws_warp_barrier"])


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
