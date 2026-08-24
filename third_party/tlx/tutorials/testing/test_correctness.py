import math
import random

import pytest

import torch

import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from triton.language.extra.tlx.tutorials.blackwell_gemm_ws import (
    matmul as _blackwell_gemm_ws, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_ws_mxfp8 import (
    matmul as _blackwell_gemm_ws_mxfp8, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_clc import (
    matmul as _blackwell_gemm_clc, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_pipelined import (
    matmul as _blackwell_gemm_pipelined, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_2cta import (
    matmul as _blackwell_gemm_2cta, )
from triton.language.extra.tlx.tutorials.blackwell_scaled_mm_ws import (
    blackwell_scaled_mm_ws as _blackwell_scaled_mm_ws, )
from triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent import (
    attention as _blackwell_fa_ws_pipelined_persistent,
    _attn_bwd_preprocess as _blackwell_fa_bwd_preprocess,
    _attn_bwd_dq_postprocess as _blackwell_fa_bwd_dq_postprocess,
    _attn_bwd_ws as _blackwell_fa_bwd_ws,
    _attn_fwd_ws as _blackwell_fa_fwd_ws,
    _host_descriptor_pre_hook as _blackwell_fa_fwd_pre_hook,
    configs as _configs_fwd,
    configs_bwd_1cta as _configs_bwd_1cta,
    configs_bwd_2cta as _configs_bwd_2cta,
    _bwd_selected_meta,
    prune_configs_by_hdim as _prune_fwd_configs,
    prune_bwd_configs as _prune_bwd_configs,
)
from triton.language.extra.tlx.tutorials.blackwell_fa_clc import (
    attention as _blackwell_fa_clc, )
from triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent_mxfp8 import (
    _attn_fwd_mxf8_ws,
    _mxf8_host_descriptor_pre_hook,
    attention as _blackwell_fa_ws_pipelined_persistent_mxfp8,
    attention_bwd,
    generate_attention_inputs as _generate_mxfp8_attention_inputs,
    swizzled_to_tma_preshuffled,
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
from triton.language.extra.tlx.tutorials.amd_fa_pipelined import (
    attention as _amd_fa_pipelined, )
from triton.language.extra.tlx.tutorials.amd_fa_persistent import (
    attention as _amd_fa_persistent, )
from triton.language.extra.tlx.tutorials.amd_fa_cluster import (
    attention as _amd_fa_cluster, )
from triton.language.extra.tlx.tutorials.amd_fa_cluster import (
    persistent_attention as _amd_fa_cluster_persistent, )
from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
    fa_backward as _amd_fa_backward, )
from triton.language.extra.tlx.tutorials.amd_pa_decode import (
    pa_decode_tlx as _amd_pa_decode,
    build_inputs as _amd_pa_decode_build_inputs,
    ref_decode as _amd_pa_decode_ref,
)
from triton.language.extra.tlx.tutorials.amd_tdm_gemm_pipelined import (
    matmul as _amd_tdm_gemm_pipelined, )
from triton.language.extra.tlx.tutorials.amd_gemm_warp_pipeline import (
    matmul as _amd_gemm_warp_pipeline, )
from triton.language.extra.tlx.tutorials.amd_gemm_pipelined import (
    matmul as _amd_gemm_pipelined, )
from triton.language.extra.tlx.tutorials.amd_gemm_gfx942 import (
    matmul as _amd_gemm_gfx942, )
from triton.language.extra.tlx.tutorials.amd_addmm_gfx942 import (
    addmm as _amd_addmm_gfx942, )
from triton.language.extra.tlx.tutorials.amd_bmm_gfx942 import (
    bmm as _amd_bmm_gfx942,
    make_bmm_inputs as _amd_bmm_gfx942_inputs,
)
from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a16w16.matmul_kernel_split_m import (
    matmul as _amd_gemm_pingpong, )
from triton.language.extra.tlx.tutorials.amd_bmm import (
    bmm as _amd_bmm,
    make_bmm_inputs as _amd_bmm_inputs,
)
from triton.language.extra.tlx.tutorials.amd_mxfp_gemm_tdm_pipelined import (
    matmul as _amd_mxfp_gemm_tdm_pipelined,
    pack_scale as _amd_mxfp_pack_scale,
)
from triton.language.extra.tlx.tutorials.amd_addmm_glu import (
    KERNEL_REGISTRY as _amd_addmm_glu_registry,
    pytorch_baseline as _amd_addmm_glu_baseline,
    M as _amd_addmm_glu_M,
    N as _amd_addmm_glu_N,
)
from triton.language.extra.tlx.tutorials.gfx950_gdpa import (
    gdpa as _gfx950_gdpa,
    gdpa_ref as _gfx950_gdpa_ref,
    generate_gdpa_data as _gfx950_gdpa_gen,
    gelu_approx_error as _gfx950_gdpa_approx_error,
)
from triton.language.extra.tlx.tutorials.amd_addmm_gfx950 import (
    addmm as _amd_addmm,
    available_paths as _amd_addmm_paths,
)
from triton.language.extra.tlx.tutorials import amd_hstu_attn as _hstu
from triton.tools.mxfp import MXScaleTensor

from triton.language.extra.tlx.tutorials.ikbo.ikbo_lce_triton import (
    create_inputs as _ikbo_lce_create_inputs,
    ikbo_lce as _ikbo_lce,
    lce_reference as _ikbo_lce_reference,
)
from triton.language.extra.tlx.tutorials.ikbo.ikbo_fa_triton import (
    create_inputs as _ikbo_fa_create_inputs,
    fa_reference as _ikbo_fa_reference,
    ikbo_fa as _ikbo_fa,
)

from triton.language.extra.tlx.tutorials.testing.multi_cta_layer_norm import (
    multi_cta_layernorm as _multi_cta_layernorm,
    multi_cta_layernorm_2d as _multi_cta_layernorm_2d,
)

from triton._internal_testing import (is_blackwell, is_hopper, is_hopper_or_newer, is_hip, is_hip_cdna3, is_hip_cdna4,
                                      is_hip_gfx1250)
from triton.language.extra.tlx.tutorials.testing.gemm_shapes import (
    BLACKWELL_GEMM_WS as _BLACKWELL_GEMM_WS_MORE_SHAPES, )

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
            "NUM_CTAS": 1,
        },
        "blackwell_gemm_ws_2cta_2group": {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 2,
            "NUM_SMEM_BUFFERS": 2,
            "NUM_TMEM_BUFFERS": 2,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": 2,
            "NUM_CTAS": 2,
            "SPLIT_K": 1,
            "INTERLEAVE_EPILOGUE": 1,
            "USE_WARP_BARRIER": False,
            "num_warps": 4,
            "num_stages": 1,
            "ctas_per_cga": (2, 1, 1),
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
            "NUM_CTAS": 1,
        },
        "amd_tdm_gemm_pipelined": {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
        },
        "amd_gemm_warp_pipeline": {
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "NUM_BUFFERS": 3,
            "num_warps": 8,
        },
        "amd_mxfp_gemm_tdm_pipelined": {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_SIZE_M": 8,
            "NUM_BUFFERS": 2,
            "DTYPE_A": "e5m2",
            "DTYPE_B": "e5m2",
            "SCALE_BLOCK": 32,
            "num_warps": 4,
            "waves_per_eu": 1,
        },
        "amd_gemm_pipelined": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4,
            "NUM_STAGES": 2,
            "kpack": 1,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": 0,
            "num_warps": 8,
        },
        # Register path of the gfx950 standalone addmm. A mid-size 128x128x64
        # tile is the safe pin for the whole shape list: the kernel masks its K
        # tail and store, so it is valid down to K=24 and up to M=32768.
        "amd_standalone_addmm_register": {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "NUM_XCDS": 1,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": 0,
            "kpack": 1,
            "num_warps": 8,
            "num_stages": 2,
        },
    }

    @staticmethod
    def run_test(matmul_fn, config, shapes=None, dtype=torch.float16):
        if shapes is None:
            shapes = Gemm.SHAPES
        for shape in shapes:
            M, N, K = shape
            torch.manual_seed(0)
            a = (torch.randn((M, K), device=DEVICE, dtype=dtype) + 1) / K
            b = (torch.randn((K, N), device=DEVICE, dtype=dtype) + 1) / K
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
            "USE_WARP_BARRIER": False,
        },
        "blackwell_fa_ws_pipelined_persistent_2cta": {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 5,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_MMA_SLICES": 2,
            "GROUP_SIZE_N": 1,
            "USE_WARP_BARRIER": False,
            "NUM_CTAS": 2,
        },
        "blackwell_fa_clc": {
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
            "RESCALE_OPT": True,
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
        "amd_fa_pipelined": {
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "num_warps": 4,
        },
        "amd_fa_pipelined_prefetch": {
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "num_warps": 8,
            "PREFETCH": True,
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
# Scaled-MM: Common utilities and configs
# =============================================================================


class Mxfp8Gemm:
    """Utilities for native Blackwell MXFP8 scaled-MMA tests."""

    SHAPES = [
        (128, 128, 128),
        (256, 256, 256),
        (384, 256, 512),
    ]

    CONFIG_2CTA = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 2,
        "NUM_SMEM_BUFFERS": 3,
        "NUM_TMEM_BUFFERS": 1,
        "NUM_MMA_GROUPS": 1,
        "EPILOGUE_SUBTILE": 4,
        "NUM_CTAS": 2,
        "SPLIT_K": 1,
        "ctas_per_cga": (2, 1, 1),
    }

    @staticmethod
    def run_test(shape, config=None):
        from torchao.prototype.mx_formats.mx_tensor import MXTensor, ScaleCalculationMode

        M, N, K = shape
        torch.manual_seed(0)
        a = torch.empty((M, K), device=DEVICE, dtype=torch.bfloat16).normal_(std=0.5)
        b = torch.empty((N, K), device=DEVICE, dtype=torch.bfloat16).normal_(std=0.5)
        a_mx = MXTensor.to_mx(
            a,
            torch.float8_e4m3fn,
            scaling_mode=ScaleCalculationMode.RCEIL,
            is_swizzled_scales=True,
        )
        b_mx = MXTensor.to_mx(
            b,
            torch.float8_e4m3fn,
            scaling_mode=ScaleCalculationMode.RCEIL,
            is_swizzled_scales=True,
        )

        out = _blackwell_gemm_ws_mxfp8(
            a_mx.qdata,
            b_mx.qdata,
            a_mx.scale,
            b_mx.scale,
            config=config,
        )
        ref = torch.matmul(
            a_mx.dequantize(torch.float32),
            b_mx.dequantize(torch.float32).T,
        ).to(torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=1e-1, rtol=0.01)


class ScaledMM:
    """Common utilities and configs for FP8 scaled_mm tests (blockwise / rowwise / tensorwise)."""

    # (M, N, K), N and K multiples of 128: square (small/large) plus igctr
    # production moderate / tall (large N, small K) / wide (small N, large K).
    SHAPES = [
        (1024, 1024, 1024),  # small: exercises the occupancy-aware BLOCK_M=64 tile
        (2048, 2048, 2048),
        (8192, 8192, 8192),
        (4096, 6144, 4608),
        (4096, 16896, 3840),
        (4096, 4608, 16896),
    ]

    SCALE_MODES = ["blockwise", "rowwise", "tensorwise"]

    @staticmethod
    def create_inputs(M, N, K, scale_mode):
        torch.manual_seed(0)
        a = (torch.randn(M, K, device=DEVICE) * 0.1).to(torch.float8_e4m3fn)
        b = (torch.randn(N, K, device=DEVICE) * 0.1).to(torch.float8_e4m3fn)
        if scale_mode == "blockwise":
            # DeepSeek: scale_a M-major [M, K//128], scale_b row-major [N//128, K//128].
            scale_a = torch.rand(M, K // 128, device=DEVICE, dtype=torch.float32).t().contiguous().t()
            scale_b = torch.rand(N // 128, K // 128, device=DEVICE, dtype=torch.float32)
        elif scale_mode == "rowwise":
            scale_a = torch.rand(M, device=DEVICE, dtype=torch.float32)
            scale_b = torch.rand(N, device=DEVICE, dtype=torch.float32)
        else:  # tensorwise: one scalar per operand
            scale_a = torch.rand(1, device=DEVICE, dtype=torch.float32)
            scale_b = torch.rand(1, device=DEVICE, dtype=torch.float32)
        return a, b, scale_a, scale_b

    @staticmethod
    def get_reference(a, b, scale_a, scale_b, scale_mode):
        af, bf = a.to(torch.float32), b.to(torch.float32)
        if scale_mode == "blockwise":
            # Scales are K-dependent: rescale-and-sum each 128-wide K group.
            M, K = a.shape
            N = b.shape[0]
            out = torch.zeros((M, N), dtype=torch.float32, device=a.device)
            for g in range(K // 128):
                partial = af[:, g * 128:(g + 1) * 128] @ bf[:, g * 128:(g + 1) * 128].t()
                sa = scale_a[:, g][:, None]
                sb = scale_b[:, g].repeat_interleave(128)[None, :]
                out += partial * sa * sb
            return out.to(torch.bfloat16)
        # K-independent: accumulate all K, then apply scales once.
        prod = af @ bf.t()
        if scale_mode == "rowwise":
            return (prod * scale_a[:, None] * scale_b[None, :]).to(torch.bfloat16)
        return (prod * scale_a * scale_b).to(torch.bfloat16)  # tensorwise

    @staticmethod
    def run_test(scale_mode, shapes=None):
        if shapes is None:
            shapes = ScaledMM.SHAPES
        for M, N, K in shapes:
            a, b, scale_a, scale_b = ScaledMM.create_inputs(M, N, K, scale_mode)
            ref = ScaledMM.get_reference(a, b, scale_a, scale_b, scale_mode)
            out = _blackwell_scaled_mm_ws(a, b, scale_a, scale_b, scale_mode=scale_mode)
            torch.testing.assert_close(out, ref, atol=1e-1, rtol=0.05)


# =============================================================================
# Blackwell GEMM Tests
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws(dtype):
    Gemm.run_test(_blackwell_gemm_ws, Gemm.CONFIGS["blackwell_gemm_ws"], dtype=dtype)


@pytest.mark.parametrize(
    "shape",
    Mxfp8Gemm.SHAPES,
    ids=[f"{m}x{n}x{k}" for m, n, k in Mxfp8Gemm.SHAPES],
)
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8(shape):
    Mxfp8Gemm.run_test(shape)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8_bn256_1cta():
    Mxfp8Gemm.run_test(
        (256, 384, 256),
        config={
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4,
            "NUM_SMEM_BUFFERS": 2,
            "NUM_TMEM_BUFFERS": 1,
            "EPILOGUE_SUBTILE": 1,
            "NUM_CTAS": 1,
            "SPLIT_K": 1,
        },
    )


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8_split_k():
    Mxfp8Gemm.run_test(
        (128, 128, 640),
        config={
            "SPLIT_K": 4,
            "NUM_SMEM_BUFFERS": 4,
        },
    )


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8_lean_pipeline():
    Mxfp8Gemm.run_test(
        (256, 256, 256),
        config={
            "GROUP_SIZE_M": 4,
            "NUM_SMEM_BUFFERS": 4,
            "NUM_TMEM_BUFFERS": 1,
            "EPILOGUE_SUBTILE": 1,
        },
    )


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8_deep_k_split_k():
    Mxfp8Gemm.run_test(
        (128, 128, 2048),
        config={
            "SPLIT_K": 4,
            "NUM_SMEM_BUFFERS": 4,
        },
    )


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8_2cta():
    Mxfp8Gemm.run_test((256, 256, 256), config=Mxfp8Gemm.CONFIG_2CTA.copy())


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8_2cta_64_columns_per_cta():
    config = Mxfp8Gemm.CONFIG_2CTA.copy()
    config["BLOCK_SIZE_N"] = 128
    Mxfp8Gemm.run_test((256, 128, 256), config=config)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8_2cta_64_columns_odd_m_tiles():
    config = Mxfp8Gemm.CONFIG_2CTA.copy()
    config["BLOCK_SIZE_N"] = 128
    Mxfp8Gemm.run_test((384, 128, 256), config=config)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8_2cta_tall_short_k():
    config = Mxfp8Gemm.CONFIG_2CTA.copy()
    config.update({
        "GROUP_SIZE_M": 4,
        "NUM_SMEM_BUFFERS": 4,
        "EPILOGUE_SUBTILE": 1,
    })
    Mxfp8Gemm.run_test((512, 256, 256), config=config)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8_2cta_uneven_split_k():
    config = Mxfp8Gemm.CONFIG_2CTA.copy()
    config.update({"SPLIT_K": 4, "NUM_SMEM_BUFFERS": 4})
    Mxfp8Gemm.run_test((256, 256, 640), config=config)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8_2cta_odd_m_tiles():
    Mxfp8Gemm.run_test((384, 256, 256), config=Mxfp8Gemm.CONFIG_2CTA.copy())


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_2cta_2group():
    Gemm.run_test(
        _blackwell_gemm_ws,
        Gemm.CONFIGS["blackwell_gemm_ws_2cta_2group"],
        shapes=[(1024, 12800, 1152)],
        dtype=torch.float16,
    )


@pytest.mark.parametrize(
    "shape",
    _BLACKWELL_GEMM_WS_MORE_SHAPES,
    ids=[f"{m}x{n}x{k}" for m, n, k in _BLACKWELL_GEMM_WS_MORE_SHAPES],
)
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_more_shapes(shape):
    Gemm.run_test(
        _blackwell_gemm_ws,
        Gemm.CONFIGS["blackwell_gemm_ws"],
        shapes=[shape],
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_clc(dtype):
    Gemm.run_test(_blackwell_gemm_clc, Gemm.CONFIGS["blackwell_gemm_clc"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_warp_barrier(dtype):
    Gemm.run_test(_blackwell_gemm_ws, Gemm.CONFIGS["blackwell_gemm_ws_warp_barrier"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_clc_warp_barrier(dtype):
    Gemm.run_test(
        _blackwell_gemm_clc,
        Gemm.CONFIGS["blackwell_gemm_clc_warp_barrier"],
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_pipelined(dtype):
    Gemm.run_test(_blackwell_gemm_pipelined, Gemm.CONFIGS["blackwell_gemm_pipelined"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_2cta(dtype):
    Gemm.run_test(_blackwell_gemm_2cta, Gemm.CONFIGS["blackwell_gemm_2cta"], dtype=dtype)


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


@pytest.mark.parametrize("RESCALE_OPT,USE_WHERE", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("BLOCK_M", [256, 128])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent(causal, RESCALE_OPT, USE_WHERE, BLOCK_M):
    config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent"].copy()
    config["RESCALE_OPT"] = RESCALE_OPT
    config["USE_WHERE"] = USE_WHERE
    config["BLOCK_M"] = BLOCK_M
    sm_scale = 0.5
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _blackwell_fa_ws_pipelined_persistent(q, k, v, sm_scale, causal, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_fast_f16(causal):
    # Exercise the selected production route: long FP16 D64 uses the four-slice
    # fixed-gauge path by default. Numerically sensitive callers can opt out via
    # an explicit configuration; that path is covered separately.
    Z, H, N_CTX, HEAD_DIM = 1, 1, 32768, 64
    sm_scale = 0.5
    q, k, v = FlashAttention.create_inputs(
        Z, H, N_CTX, HEAD_DIM, dtype=torch.float16
    )
    ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
    tri_out = _blackwell_fa_ws_pipelined_persistent(
        q, k, v, sm_scale, causal
    )
    assert torch.isfinite(tri_out).all()
    torch.testing.assert_close(tri_out, ref_out, atol=1.5e-2, rtol=0)


@pytest.mark.parametrize("RESCALE_OPT,USE_WHERE", [(False, False), (True, False), (True, True)])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_2cta(RESCALE_OPT, USE_WHERE):
    # 2-CTA (M-split) forward: HEAD_DIM=128, non-causal (v1 scope).
    config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent_2cta"].copy()
    config["RESCALE_OPT"] = RESCALE_OPT
    config["USE_WHERE"] = USE_WHERE
    causal = False
    sm_scale = 0.5
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        if HEAD_DIM != 128:
            continue
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _blackwell_fa_ws_pipelined_persistent(q, k, v, sm_scale, causal, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


def _run_blackwell_fa_numeric(q, k, v, sm_scale, *, fast_fixed=True, rescale_opt=False):
    Z, H, N_CTX, HEAD_DIM = q.shape
    config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent_2cta"].copy()
    config.update({
        "NUM_BUFFERS_KV": 3,
        "RESCALE_OPT": rescale_opt,
        "USE_WHERE": False,
        "USE_WARP_BARRIER": True,
        "PIPELINED": True,
        "DENSE_REGS": 176,
        "FAST_FIXED": fast_fixed,
    })

    o = torch.full_like(q, float("nan"))
    m = torch.full((Z, H, N_CTX), float("nan"), device=q.device, dtype=torch.float32)
    y_dim = Z * H * N_CTX
    dummy_block = [1, 1]
    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
    nargs = {
        **config,
        "HEAD_DIM": HEAD_DIM,
        "desc_q": desc_q,
        "desc_k": desc_k,
        "desc_v": desc_v,
        "desc_o": desc_o,
    }
    _blackwell_fa_fwd_pre_hook(nargs)

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    num_ctas = config["NUM_CTAS"]
    work_ctas = triton.cdiv(N_CTX, config["BLOCK_M"] * num_ctas) * Z * H * num_ctas
    grid_ctas = min(torch.cuda.get_device_properties(q.device).multi_processor_count, work_ctas)
    grid_ctas -= grid_ctas % num_ctas
    _blackwell_fa_fwd_ws.fn[(grid_ctas, 1, 1)](
        sm_scale,
        m,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        N_CTX=N_CTX,
        HEAD_DIM=HEAD_DIM,
        STAGE=1,
        num_stages=1,
        num_warps=4,
        ctas_per_cga=(2, 1, 1),
        **config,
    )
    return o, m


def _make_attention_numeric_inputs(shape, dtype, distribution):
    torch.manual_seed(20)
    if distribution == "uniform_random":
        return tuple(torch.empty(shape, device=DEVICE, dtype=dtype).uniform_(-0.5, 0.5) for _ in range(3))
    if distribution == "normal_random":
        return tuple(
            torch.empty(shape, device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5)
            for _ in range(3)
        )
    q = torch.ones(shape, device=DEVICE, dtype=dtype)
    q[:, :, shape[2] // 2:, :] = -1
    amplitude = 0.625 if shape[3] == 128 else 0.2
    k = torch.full(shape, amplitude, device=DEVICE, dtype=dtype)
    k[:, :, :128, :] = -amplitude
    v = torch.empty(shape, device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5)
    return q, k, v


@pytest.mark.parametrize("distribution", ["uniform_random", "normal_random", "far_apart"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_fast_fixed_bf16_numerics(distribution):
    Z, H, N_CTX, HEAD_DIM = 4, 48, 1024, 128
    shape = (Z, H, N_CTX, HEAD_DIM)
    sm_scale = 0.5
    q, k, v = _make_attention_numeric_inputs(shape, torch.bfloat16, distribution)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    ref_o = torch.matmul(torch.softmax(scores, dim=-1), v.float())
    ref_m = torch.logsumexp(scores, dim=-1) * math.log2(math.e)

    outputs = [_run_blackwell_fa_numeric(q, k, v, sm_scale) for _ in range(3)]
    tri_o, tri_m = outputs[0]
    for repeat_o, repeat_m in outputs:
        assert torch.isfinite(repeat_o).all()
        assert torch.isfinite(repeat_m).all()
        torch.testing.assert_close(repeat_o, tri_o, atol=0, rtol=0)
        torch.testing.assert_close(repeat_m, tri_m, atol=0, rtol=0)
    o_error = (tri_o.float() - ref_o).abs()
    print(
        f"{distribution}: O max/RMSE="
        f"{o_error.max().item():.8g}/{o_error.square().mean().sqrt().item():.8g}"
    )
    torch.testing.assert_close(tri_o.float(), ref_o, atol=1e-2, rtol=0)
    # The fixed-gauge BF16 exp approximation has a bounded bias in the saved
    # base-2 log-sum-exp; backward consumes the matching value from forward.
    torch.testing.assert_close(tri_m, ref_m, atol=0.125, rtol=0)


@pytest.mark.parametrize("rescale_opt", [False, True])
@pytest.mark.parametrize("distribution", ["uniform_random", "normal_random", "far_apart"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_2cta_non_fast_fixed_numerics(distribution, rescale_opt):
    Z, H, N_CTX, HEAD_DIM = 4, 48, 1024, 128
    shape = (Z, H, N_CTX, HEAD_DIM)
    sm_scale = 0.5
    q, k, v = _make_attention_numeric_inputs(shape, torch.bfloat16, distribution)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    ref_o = torch.matmul(torch.softmax(scores, dim=-1), v.float())
    ref_m = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
    outputs = [
        _run_blackwell_fa_numeric(
            q,
            k,
            v,
            sm_scale,
            fast_fixed=False,
            rescale_opt=rescale_opt,
        )
        for _ in range(3)
    ]
    tri_o, tri_m = outputs[0]
    for repeat_o, repeat_m in outputs:
        assert torch.isfinite(repeat_o).all()
        assert torch.isfinite(repeat_m).all()
        torch.testing.assert_close(repeat_o, tri_o, atol=0, rtol=0)
        torch.testing.assert_close(repeat_m, tri_m, atol=0, rtol=0)
    o_error = (tri_o.float() - ref_o).abs()
    print(
        f"{distribution}, RESCALE_OPT={rescale_opt}: "
        f"O max/RMSE={o_error.max().item():.8g}/{o_error.square().mean().sqrt().item():.8g}"
    )
    torch.testing.assert_close(tri_o.float(), ref_o, atol=1e-2, rtol=0)
    torch.testing.assert_close(tri_m, ref_m, atol=0.125, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_fp16_fixed_gauge_opt_out(causal):
    shape = (1, 1, 1024, 64)
    q, k, v = _make_attention_numeric_inputs(shape, torch.float16, "far_apart")
    config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent"].copy()
    config.update({
        "NUM_CTAS": 1,
        "NUM_MMA_SLICES": 2,
        "RESCALE_OPT": False,
        "USE_WHERE": False,
        "FAST_FIXED": False,
    })
    ref_o = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=0.5, is_causal=causal
    )
    outputs = [
        _blackwell_fa_ws_pipelined_persistent(
            q, k, v, 0.5, causal, config=config
        )
        for _ in range(3)
    ]
    for repeat_o in outputs:
        assert torch.isfinite(repeat_o).all()
        torch.testing.assert_close(repeat_o, outputs[0], atol=0, rtol=0)
    torch.testing.assert_close(outputs[0], ref_o, atol=1e-2, rtol=0)


def test_blackwell_fa_ws_pipelined_persistent_2cta_pruning():
    selected = _prune_fwd_configs(
        _configs_fwd,
        {},
        HEAD_DIM=128,
        STAGE=1,
        N_CTX=768,
    )
    assert selected
    assert all(config.kwargs.get("NUM_CTAS", 1) == 1 for config in selected)

    selected = _prune_fwd_configs(
        _configs_fwd,
        {},
        HEAD_DIM=128,
        STAGE=1,
        N_CTX=1024,
    )
    assert any(config.kwargs.get("NUM_CTAS", 1) == 2 for config in selected)


@pytest.mark.parametrize("Z,H", [(4, 8), (4, 48), (24, 8)])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_2cta_probe(Z, H):
    # Cover multiple persistent waves and equivalent batch/head decompositions.
    config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent_2cta"].copy()
    config.update({
        "NUM_BUFFERS_KV": 3,
        "RESCALE_OPT": True,
        "USE_WHERE": False,
        "USE_WARP_BARRIER": True,
        "PIPELINED": True,
        "DENSE_REGS": 176,
    })
    sm_scale = 0.5
    N_CTX, HEAD_DIM = 1024, 128
    q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
    ref_out = FlashAttention.get_reference(q, k, v, sm_scale, False)
    tri_out = _blackwell_fa_ws_pipelined_persistent(q, k, v, sm_scale, False, config=config)
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_bench_2cta_vs_1cta():
    """Benchmark: 2-CTA M-split vs 1-CTA pipelined persistent."""
    import triton.testing as tt

    config_1cta = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent"].copy()
    config_1cta["RESCALE_OPT"] = True
    config_1cta["USE_WHERE"] = False

    config_2cta = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent_2cta"].copy()
    config_2cta["RESCALE_OPT"] = True
    config_2cta["USE_WHERE"] = False

    shapes = [
        (4, 8, 1024, 128),
        (4, 48, 1024, 128),
        (4, 48, 2048, 128),
        (4, 48, 4096, 128),
        (4, 48, 8192, 128),
        (4, 48, 16384, 128),
    ]

    def calc_tflops(Z, H, N, D, ms):
        return 4.0 * Z * H * N * N * D / (ms * 1e-3) / 1e12

    print(f"\n{'Shape':>30s} | {'1CTA ms':>9s} {'TFLOPS':>8s} | {'2CTA ms':>9s} {'TFLOPS':>8s} | {'Ratio':>7s}")
    print("-" * 90)

    for Z, H, N, D in shapes:
        q, k, v = FlashAttention.create_inputs(Z, H, N, D)
        sm_scale = 0.5

        # warmup + bench 1-CTA
        _blackwell_fa_ws_pipelined_persistent(q, k, v, sm_scale, False, config=config_1cta)
        torch.cuda.synchronize()
        ms1 = tt.do_bench(lambda: _blackwell_fa_ws_pipelined_persistent(q, k, v, sm_scale, False, config=config_1cta))
        tf1 = calc_tflops(Z, H, N, D, ms1)

        # warmup + bench 2-CTA
        _blackwell_fa_ws_pipelined_persistent(q, k, v, sm_scale, False, config=config_2cta)
        torch.cuda.synchronize()
        ms2 = tt.do_bench(lambda: _blackwell_fa_ws_pipelined_persistent(q, k, v, sm_scale, False, config=config_2cta))
        tf2 = calc_tflops(Z, H, N, D, ms2)

        label = f"Z={Z} H={H} N={N} D={D}"
        ratio = tf2 / tf1
        print(f"{label:>30s} | {ms1:9.4f} {tf1:8.1f} | {ms2:9.4f} {tf2:8.1f} | {ratio:7.3f}x")

    print()


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
        tri_out = _blackwell_fa_ws_pipelined_persistent(q, k, v, sm_scale, causal, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.parametrize("RESCALE_OPT,USE_WHERE", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("N_CTX", [1024, 2048, 4096, 8192])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_clc(N_CTX, causal, RESCALE_OPT, USE_WHERE):
    config = FlashAttention.CONFIGS["blackwell_fa_clc"].copy()
    config["RESCALE_OPT"] = RESCALE_OPT
    config["USE_WHERE"] = USE_WHERE
    sm_scale = 0.5
    Z, H, HEAD_DIM = 4, 8, 128
    q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
    ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
    tri_out = _blackwell_fa_clc(q, k, v, sm_scale, causal, config=config)
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.parametrize("NUM_CTAS", [1, 2])
@pytest.mark.parametrize("USE_WARP_BARRIER", [False, True])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("RESCALE_OPT,USE_WHERE", [(False, False), (True, False), (True, True)])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_bwd(
    causal, RESCALE_OPT, USE_WHERE, HEAD_DIM, USE_WARP_BARRIER, NUM_CTAS
):
    if NUM_CTAS == 2 and USE_WARP_BARRIER:
        pytest.skip("the 2-CTA configuration uses cluster barriers")
    fwd_config: dict[str,
                     bool | int] = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent_warp_barrier"].copy()
    fwd_config["RESCALE_OPT"] = RESCALE_OPT
    fwd_config["USE_WHERE"] = USE_WHERE
    sm_scale = 0.5

    for Z, H, N_CTX, _ in FlashAttention.SHAPES:
        direct_dq_output = NUM_CTAS == 2 and HEAD_DIM == 128
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)

        # Reference backward via PyTorch autograd
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        do = torch.randn_like(ref_out)
        ref_out.backward(do)
        ref_dq, ref_dk, ref_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad, k.grad, v.grad = None, None, None

        # Forward with known-good config (no autotuning)
        stage = 3 if causal else 1
        o = torch.empty_like(q)
        M = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
        y_dim = Z * H * N_CTX
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
        desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)

        nargs = {
            **fwd_config,
            "HEAD_DIM": HEAD_DIM,
            "desc_q": desc_q,
            "desc_k": desc_k,
            "desc_v": desc_v,
            "desc_o": desc_o,
        }
        _blackwell_fa_fwd_pre_hook(nargs)

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)
        grid = (triton.cdiv(N_CTX, fwd_config["BLOCK_M"]) * Z * H, 1, 1)
        _blackwell_fa_fwd_ws.fn[grid](
            sm_scale,
            M,
            Z,
            H,
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            N_CTX=N_CTX,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
            **fwd_config,
        )
        torch.testing.assert_close(o, ref_out, atol=1e-2, rtol=0)

        # Backward: preprocess
        RCP_LN2 = 1.4426950408889634
        arg_k = k if direct_dq_output else k * (sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        pre_grid = (N_CTX // PRE_BLOCK, Z * H)
        delta = torch.empty_like(M)
        _blackwell_fa_bwd_preprocess[pre_grid](o, do, delta, N_CTX, BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM)

        # Backward: main kernel
        dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32) if direct_dq_output else torch.empty(
            q.shape, device=q.device, dtype=torch.float32
        )
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        _HALF_HD = HEAD_DIM // 2
        dq_accum = dq if direct_dq_output else torch.zeros(
            [Z, H, N_CTX, HEAD_DIM], device=q.device, dtype=torch.float32
        )

        dummy_block_4d = [1, 1, 1, 1]
        desc_shape = [Z, H, N_CTX, HEAD_DIM]
        desc_strides = [H * N_CTX * HEAD_DIM, N_CTX * HEAD_DIM, HEAD_DIM, 1]
        desc_bk = TensorDescriptor(arg_k, shape=desc_shape, strides=desc_strides, block_shape=dummy_block_4d)
        desc_bv = TensorDescriptor(v, shape=desc_shape, strides=desc_strides, block_shape=dummy_block_4d)
        desc_bq = TensorDescriptor(q, shape=desc_shape, strides=desc_strides, block_shape=dummy_block_4d)
        desc_do = TensorDescriptor(do, shape=desc_shape, strides=desc_strides, block_shape=dummy_block_4d)
        if direct_dq_output:
            _dq_desc_shape = desc_shape
            _dq_desc_strides = desc_strides
        else:
            _dq_desc_shape = [Z, H, 2 * N_CTX, _HALF_HD]
            _dq_desc_strides = [H * N_CTX * HEAD_DIM, N_CTX * HEAD_DIM, _HALF_HD, 1]
        desc_dq = TensorDescriptor(dq_accum, shape=_dq_desc_shape, strides=_dq_desc_strides, block_shape=dummy_block_4d)
        desc_dk = TensorDescriptor(dk, shape=desc_shape, strides=desc_strides, block_shape=dummy_block_4d)
        desc_dv = TensorDescriptor(dv, shape=desc_shape, strides=desc_strides, block_shape=dummy_block_4d)
        desc_m = TensorDescriptor(M, shape=[Z * H * N_CTX], strides=[1], block_shape=[1])
        desc_delta = TensorDescriptor(delta, shape=[Z * H * N_CTX], strides=[1], block_shape=[1])

        # Descriptors for 2-CTA B-operand transposed views.
        # In 1-CTA mode these are passed but unused by the kernel.
        desc_kt = TensorDescriptor(arg_k, shape=desc_shape, strides=desc_strides, block_shape=dummy_block_4d)
        desc_qt = TensorDescriptor(q, shape=desc_shape, strides=desc_strides, block_shape=dummy_block_4d)
        desc_dot = TensorDescriptor(do, shape=desc_shape, strides=desc_strides, block_shape=dummy_block_4d)

        BLK_SLICE_FACTOR = 2

        source_configs = _configs_bwd_1cta if NUM_CTAS == 1 else _configs_bwd_2cta
        bwd_configs = [
            config
            for config in source_configs
            if config.kwargs["USE_WARP_BARRIER"] == USE_WARP_BARRIER
        ]
        assert len(bwd_configs) == 1
        bwd_kernel = triton.autotune(configs=bwd_configs, key=["N_CTX", "HEAD_DIM"])(_blackwell_fa_bwd_ws.fn)

        def grid_persistent(meta):
            n_tiles = triton.cdiv(N_CTX, meta["BLOCK_N1"])
            num_ctas = meta.get("NUM_CTAS", 1)
            n_tiles = triton.cdiv(n_tiles, num_ctas) * num_ctas
            return (n_tiles, H, Z)

        bwd_kernel[grid_persistent](
            desc_bq,
            desc_bk,
            desc_bv,
            sm_scale,
            desc_do,
            desc_dq,
            desc_dk,
            desc_dv,
            desc_m,
            desc_delta,
            M,
            delta,
            H,
            Z,
            N_CTX,
            desc_kt,
            desc_qt,
            desc_dot,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
            DQ_STAGE_COUNT=2,
            SCALE_QK_IN_KERNEL=direct_dq_output,
        )

        if not direct_dq_output:
            _blk = _bwd_selected_meta["BLOCK_M1"] // _bwd_selected_meta["NUM_CTAS"]
            post_grid = (N_CTX // PRE_BLOCK, Z * H)
            _blackwell_fa_bwd_dq_postprocess[post_grid](
                dq_accum,
                dq,
                N_CTX,
                BLK=_blk,
                HALF_HD=HEAD_DIM // 2,
                BLOCK_M=PRE_BLOCK,
                HEAD_DIM=HEAD_DIM,
            )

        torch.testing.assert_close(dv, ref_dv, atol=1e-2, rtol=0)
        torch.testing.assert_close(dk, ref_dk, atol=1e-2, rtol=0)
        torch.testing.assert_close(dq.to(ref_dq.dtype), ref_dq, atol=1e-2, rtol=0)


def test_blackwell_fa_ws_pipelined_persistent_direct_dq_pruning():
    configs = _configs_bwd_1cta + _configs_bwd_2cta
    selected = _prune_bwd_configs(
        configs,
        {},
        SCALE_QK_IN_KERNEL=True,
        HEAD_DIM=128,
        N_CTX=1024,
    )
    assert selected
    assert all(config.kwargs.get("NUM_CTAS", 1) == 2 for config in selected)
    with pytest.raises(AssertionError):
        _prune_bwd_configs(
            configs,
            {},
            SCALE_QK_IN_KERNEL=True,
            HEAD_DIM=64,
            N_CTX=1024,
        )

    odd_tiles = _prune_bwd_configs(
        configs,
        {},
        SCALE_QK_IN_KERNEL=False,
        HEAD_DIM=128,
        N_CTX=384,
    )
    assert odd_tiles
    assert all(config.kwargs.get("NUM_CTAS", 1) == 1 for config in odd_tiles)


@pytest.mark.parametrize(
    "dtype,N_CTX,causal",
    [
        (torch.float16, 128, False),
        (torch.float16, 384, False),
        (torch.float16, 1024, False),
        (torch.bfloat16, 1024, False),
        (torch.bfloat16, 1024, True),
        (torch.bfloat16, 4096, False),
    ],
)
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_backward_public_paths(dtype, N_CTX, causal):
    shape = (1, 1, N_CTX, 128)
    torch.manual_seed(20)
    q0, k0, v0 = [
        torch.empty(shape, device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5)
        for _ in range(3)
    ]
    do = torch.empty(shape, device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5)

    ref_q, ref_k, ref_v = [tensor.detach().clone().requires_grad_() for tensor in (q0, k0, v0)]
    ref_o = torch.nn.functional.scaled_dot_product_attention(
        ref_q, ref_k, ref_v, scale=0.5, is_causal=causal
    )
    ref_o.backward(do)
    reference = (ref_q.grad, ref_k.grad, ref_v.grad)

    results = []
    for _ in range(3):
        q, k, v = [tensor.detach().clone().requires_grad_() for tensor in (q0, k0, v0)]
        out = _blackwell_fa_ws_pipelined_persistent(q, k, v, 0.5, causal)
        out.backward(do)
        result = (q.grad, k.grad, v.grad)
        assert all(torch.isfinite(grad).all() for grad in result)
        results.append(result)

    atol = 6.25e-2 if dtype == torch.bfloat16 else 1.5e-2
    for result in results:
        for grad, ref_grad in zip(result, reference):
            torch.testing.assert_close(grad, ref_grad, atol=atol, rtol=0)


@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_mxfp8(HEAD_DIM, causal):
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
                # Max atol measured was 0.033203125
                atol = 0.04
            else:
                # Max atol measured was 0.07421875
                assert HEAD_DIM == 128
                atol = 0.08
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)


def _quantize_mxfp8_bwd_operand(ref, dtype, transpose_for_reduction=False):
    from torchao.prototype.mx_formats.mx_tensor import MXTensor, ScaleCalculationMode

    Z, H, N_CTX, HEAD_DIM = ref.shape
    flat = ref.reshape(Z * H * N_CTX, HEAD_DIM).contiguous()
    quant_input = flat.t().contiguous() if transpose_for_reduction else flat
    mx = MXTensor.to_mx(
        quant_input,
        dtype,
        scaling_mode=ScaleCalculationMode.RCEIL,
        is_swizzled_scales=True,
    )
    if transpose_for_reduction:
        data = mx.qdata.t().reshape_as(ref).contiguous()
        scale = swizzled_to_tma_preshuffled(mx.scale, HEAD_DIM, N_CTX, 32, Z * H)
    else:
        data = mx.qdata.reshape_as(ref).contiguous()
        scale = swizzled_to_tma_preshuffled(mx.scale, N_CTX, HEAD_DIM, 32, Z * H)
    return data, scale


def _cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_flat = actual.float().reshape(-1)
    expected_flat = expected.float().reshape(-1)
    actual_norm = actual_flat.norm().item()
    expected_norm = expected_flat.norm().item()
    if actual_norm == 0.0 or expected_norm == 0.0:
        return 1.0 if actual_norm == 0.0 and expected_norm == 0.0 else 0.0
    return torch.dot(actual_flat, expected_flat).item() / (actual_norm * expected_norm)


def _assert_close_with_cosine(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    min_cosine: float,
) -> None:
    cosine = _cosine_similarity(actual, expected)
    # TODO: Enable value-based checking once MXFP8 backward tolerances settle.
    assert cosine >= min_cosine, f"{label} cosine_similarity={cosine:.6f} fell below min_cosine={min_cosine:.6f}"


@pytest.mark.parametrize(
    "Z,H,N_CTX",
    [
        (1, 1, 256),
        (1, 1, 1024),
        (2, 2, 256),
        (2, 4, 512),
        # Test the persistent case
        (8, 16, 2048),
        # Failing N_CTX/N_BLOCK odd case. Seems likely to be a quantization bug.
        # (2, 1, 1152),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_mxfp8_bwd(Z, H, N_CTX, causal):
    """MXFP8 backward correctness vs PyTorch autograd on randomized inputs."""
    sm_scale = 0.5
    dtype = torch.float8_e4m3fn
    head_dim = 128
    shape = (Z, H, N_CTX, head_dim)
    bwd_min_cosine: float = 0.98
    torch.manual_seed(20)

    (q, q_scale, q_ref), (k, k_scale, k_ref), (v, v_scale,
                                               v_ref) = _generate_mxfp8_attention_inputs(shape, DEVICE, dtype)
    q_ref = q_ref.detach().requires_grad_(True)
    k_ref = k_ref.detach().requires_grad_(True)
    v_ref = v_ref.detach().requires_grad_(True)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, scale=sm_scale, is_causal=causal)
    do_bf16 = torch.randn_like(ref_out)
    ref_out.backward(do_bf16)

    q_dk, q_scale_dk = _quantize_mxfp8_bwd_operand(q_ref.detach(), dtype, transpose_for_reduction=True)
    k_dq, k_scale_dq = _quantize_mxfp8_bwd_operand(k_ref.detach(), dtype, transpose_for_reduction=True)
    v_bwd, v_scale_bwd = _quantize_mxfp8_bwd_operand(v_ref.detach(), dtype)
    do_fp8, do_scale = _quantize_mxfp8_bwd_operand(do_bf16, dtype)
    do_fp8_dv, do_scale_dv = _quantize_mxfp8_bwd_operand(do_bf16, dtype, transpose_for_reduction=True)

    fwd_config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent_mxfp8"]
    y_dim = Z * H * N_CTX
    o = torch.empty(q.shape, device=DEVICE, dtype=torch.bfloat16)
    M = torch.empty((Z, H, N_CTX), device=DEVICE, dtype=torch.float32)
    dummy_block = [1, 1]
    dummy_5d = [1, 1, 1, 1, 1]
    desc_q = TensorDescriptor(q, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=dummy_block)
    desc_k = TensorDescriptor(k, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=dummy_block)
    desc_v = TensorDescriptor(v, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=dummy_block)
    desc_m = TensorDescriptor(M, shape=[y_dim], strides=[1], block_shape=[1])
    desc_q_scale = TensorDescriptor.from_tensor(q_scale, block_shape=dummy_5d)
    desc_k_scale = TensorDescriptor.from_tensor(k_scale, block_shape=dummy_5d)
    desc_v_scale = TensorDescriptor.from_tensor(v_scale, block_shape=dummy_5d)
    nargs = {
        **fwd_config,
        "HEAD_DIM": head_dim,
        "desc_q": desc_q,
        "desc_k": desc_k,
        "desc_v": desc_v,
        "desc_o": desc_o,
        "desc_m": desc_m,
        "desc_q_scale": desc_q_scale,
        "desc_k_scale": desc_k_scale,
        "desc_v_scale": desc_v_scale,
    }
    _mxf8_host_descriptor_pre_hook(nargs)

    def alloc_fn(size, align, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    fwd_grid = (
        triton.cdiv(N_CTX, fwd_config["BLOCK_M"]) * Z * H,
        1,
        1,
    )
    _attn_fwd_mxf8_ws.fn[fwd_grid](
        sm_scale,
        desc_m,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        desc_q_scale,
        desc_k_scale,
        desc_v_scale,
        N_CTX=N_CTX,
        HEAD_DIM=head_dim,
        STAGE=3 if causal else 1,
        num_stages=1,
        num_warps=4,
        **fwd_config,
    )

    dq, dk, dv = attention_bwd(
        do_fp8,
        do_fp8_dv,
        q,
        q_dk,
        k,
        k_dq,
        v_bwd,
        o,
        M,
        q_scale,
        q_scale_dk,
        k_scale,
        k_scale_dq,
        v_scale_bwd,
        do_scale,
        do_scale_dv,
        sm_scale,
        do_bf16=do_bf16,
        causal=causal,
    )
    ref_dq = q_ref.grad.detach()
    ref_dk = k_ref.grad.detach()
    ref_dv = v_ref.grad.detach()

    dq_bf16 = dq.to(torch.bfloat16)
    _assert_close_with_cosine(
        dq_bf16,
        ref_dq,
        label="dq",
        min_cosine=bwd_min_cosine,
    )
    _assert_close_with_cosine(
        dk,
        ref_dk,
        label="dk",
        min_cosine=bwd_min_cosine,
    )
    _assert_close_with_cosine(
        dv,
        ref_dv,
        label="dv",
        min_cosine=bwd_min_cosine,
    )


# =============================================================================
# Blackwell Scaled-MM (FP8) Tests
# =============================================================================


@pytest.mark.parametrize("scale_mode", ScaledMM.SCALE_MODES)
@pytest.mark.parametrize("shape", ScaledMM.SHAPES, ids=[f"{m}x{n}x{k}" for m, n, k in ScaledMM.SHAPES])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_scaled_mm_ws(shape, scale_mode):
    M, N, K = shape
    ScaledMM.run_test(scale_mode, shapes=[shape])


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


# =============================================================================
# AMD Flash Attention Tests
# =============================================================================


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("config_name", ["amd_fa_pipelined", "amd_fa_pipelined_prefetch"])
# Gated to gfx950 (CDNA4): the kernel passes on MI350 but fails to lower
# (MLIR -> LLVM `unrealized_conversion_cast`) on gfx942/MI300, matching the
# arch-gating of the sibling AMD GEMM tests below.
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_fa_pipelined(config_name, causal):
    config = FlashAttention.CONFIGS[config_name]
    sm_scale = 0.5
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _amd_fa_pipelined(q, k, v, sm_scale, causal, config=config)
        torch.testing.assert_close(tri_out, ref_out, atol=2e-2, rtol=0)


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "nocausal"])
@pytest.mark.parametrize("N_CTX", [128, 192, 256, 500, 512, 1024])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_fa_persistent(N_CTX, causal):
    """Persistent AMD FA fwd: async prefetch + XCD-grouped zig-zag scheduler."""
    torch.manual_seed(42)
    B, H, D = 1, 4, 128
    dtype = torch.bfloat16
    q = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=dtype)
    sm = 1.0 / math.sqrt(D)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm)
    out = _amd_fa_persistent(q, k, v, sm, causal)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "nocausal"])
@pytest.mark.parametrize(
    "q_len,kv_len",
    [(256, 1024), (1024, 256), (1, 1024), (1024, 1024)],
    ids=["cross_qlt", "cross_qgt", "decode", "square"],
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_fa_persistent_cross_attention(q_len, kv_len, causal):
    """Persistent kernel with q_len != kv_len (cross-attention / decode).

    Causal uses bottom-right alignment (key j attends iff j <= i + (kv_len -
    q_len)) — the decode/KV-cache and FlashAttention convention.
    """
    torch.manual_seed(42)
    B, H, D = 1, 8, 128
    dtype = torch.bfloat16
    q = torch.randn(B, H, q_len, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, kv_len, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, kv_len, D, device=DEVICE, dtype=dtype)
    sm = 1.0 / math.sqrt(D)
    if not causal:
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm)
    else:
        i = torch.arange(q_len, device=q.device)[:, None]
        j = torch.arange(kv_len, device=q.device)[None, :]
        bias = torch.zeros(q_len, kv_len, device=q.device,
                           dtype=q.dtype).masked_fill(~(j <= i + (kv_len - q_len)), float("-inf"))
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=sm)
    out = _amd_fa_persistent(q, k, v, sm, causal)
    valid = ~torch.isnan(ref.float())  # fully-masked rows (q_len > kv_len) are undefined
    torch.testing.assert_close(out.float()[valid], ref.float()[valid], atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal", [False, True], ids=["nocausal", "causal"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_fa_cluster(causal, dtype, HEAD_DIM):
    torch.manual_seed(42)
    B, H, N_CTX, D = 1, 4, 1024, HEAD_DIM
    q = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=dtype)
    sm = 1.0 / math.sqrt(D)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm)
    out = _amd_fa_cluster(q, k, v, sm, causal)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("persistent", [False, True], ids=["direct", "persistent"])
@pytest.mark.parametrize("causal", [False, True], ids=["nocausal", "causal"])
@pytest.mark.parametrize("use_direct_load", [None, False, True], ids=["autotune", "lds", "direct-load"])
@pytest.mark.parametrize(
    "N_CTX,BLOCK_M",
    [(128, 128), (129, 128), (257, 256)],
    ids=["short", "short-unaligned", "pipeline-unaligned"],
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_fa_cluster_block_n_boundaries(persistent, causal, use_direct_load, N_CTX, BLOCK_M):
    torch.manual_seed(42)
    B, H, D = 1, 4, 64
    dtype = torch.bfloat16
    q = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=dtype)
    sm = 1.0 / math.sqrt(D)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm)
    kernel = _amd_fa_cluster_persistent if persistent else _amd_fa_cluster
    config = {"BLOCK_M": BLOCK_M, "BLOCK_N": 64}
    if use_direct_load is not None:
        config["USE_DIRECT_LOAD"] = use_direct_load
    if persistent:
        config.update({"NUM_SMS": 16, "NUM_XCDS": 4})
    out = kernel(q, k, v, sm, causal, config=config)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal", [False, True], ids=["nocausal", "causal"])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_fa_cluster_persistent_scheduler_knobs(causal, HEAD_DIM):
    torch.manual_seed(42)
    B, H, N_CTX, D = 2, 9, 1024, HEAD_DIM
    q = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=torch.bfloat16)
    sm = 1.0 / math.sqrt(D)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm)
    out = _amd_fa_cluster_persistent(q, k, v, sm, causal, config={"NUM_SMS": 16, "NUM_XCDS": 4})
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("B", "Hq", "Hkv", "N_CTX"),
    [(1, 1, 1, 512),  # MHA
     (1, 8, 1, 512),  # GQA8
     ],
    ids=["mha", "gqa8"],
)
@pytest.mark.parametrize("causal", [False, True], ids=["nocausal", "causal"])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_fa_bwd_d64(B, Hq, Hkv, N_CTX, causal):
    torch.manual_seed(42)
    D = 64
    q = torch.randn(B, Hq, N_CTX, D, device=DEVICE, dtype=torch.bfloat16).contiguous()
    k = torch.randn(B, Hkv, N_CTX, D, device=DEVICE, dtype=torch.bfloat16).contiguous()
    v = torch.randn(B, Hkv, N_CTX, D, device=DEVICE, dtype=torch.bfloat16).contiguous()
    do = torch.randn_like(q)
    sm_scale = D**-0.5

    state = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, causal, False, scale=sm_scale)
    o, lse = state[0], state[1]
    cum_q, cum_k, max_q, max_k, rng, unused = state[2:8]
    ref_dq, ref_dk, ref_dv = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        do, q, k, v, o, lse, cum_q, cum_k, max_q, max_k, 0.0, causal, rng, unused, scale=sm_scale)

    dq, dk, dv = _amd_fa_backward(q, k, v, o.contiguous(), do, lse.contiguous(), sm_scale, causal)

    for name, actual, expected in (("dq", dq, ref_dq), ("dk", dk, ref_dk), ("dv", dv, ref_dv)):
        assert torch.isfinite(actual).all(), name
        rel_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert rel_l2.item() < 5e-3, (name, rel_l2.item())


# =============================================================================
# AMD Paged-Attention Decode Tests (gfx950)
# =============================================================================


@pytest.mark.parametrize("query_length", [1, 2, 3, 4], ids=lambda q: f"qlen{q}")
@pytest.mark.parametrize("num_splits", [1, 4], ids=["split1", "split4"])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_pa_decode(num_splits, query_length):
    """Split-K paged decode with bf16 KV cache and GQA, incl. multi-token
    prediction (query_length 1-4). Reference is dense fp32 attention gathered
    from the page table with bottom-right causal masking over the query block.
    """
    num_kv_heads, group = 2, 4
    num_q_heads = num_kv_heads * group
    head_dim, page_size = 128, 16
    ctx_lens = [40, 71]
    num_seqs = len(ctx_lens)
    sm_scale = 1.0 / math.sqrt(head_dim)

    query, key_cache, value_cache, context_lens, block_tables = _amd_pa_decode_build_inputs(
        num_seqs, ctx_lens, num_q_heads, num_kv_heads, head_dim, page_size, query_length=query_length, device=DEVICE)

    out = torch.empty_like(query)
    _amd_pa_decode(out, query, key_cache, value_cache, context_lens, block_tables, sm_scale, query_length=query_length,
                   num_splits=num_splits)

    ref = _amd_pa_decode_ref(query, key_cache, value_cache, context_lens, block_tables, sm_scale, num_q_heads,
                             num_kv_heads, query_length)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_pa_decode_page64_high_splits():
    num_kv_heads, group = 2, 4
    num_q_heads = num_kv_heads * group
    head_dim, page_size = 64, 64
    ctx_lens = [65, 257, 513]
    sm_scale = 1.0 / math.sqrt(head_dim)

    query, key_cache, value_cache, context_lens, block_tables = _amd_pa_decode_build_inputs(
        len(ctx_lens), ctx_lens, num_q_heads, num_kv_heads, head_dim, page_size, device=DEVICE)
    out = torch.empty_like(query)
    _amd_pa_decode(out, query, key_cache, value_cache, context_lens, block_tables, sm_scale, num_splits=128)

    ref = _amd_pa_decode_ref(query, key_cache, value_cache, context_lens, block_tables, sm_scale, num_q_heads,
                             num_kv_heads, 1)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_pa_decode_page16_tile_boundaries():
    num_kv_heads, group = 2, 4
    num_q_heads = num_kv_heads * group
    head_dim, page_size = 128, 16
    ctx_lens = [15, 16, 17, 63, 64, 65]
    sm_scale = 1.0 / math.sqrt(head_dim)

    query, key_cache, value_cache, context_lens, block_tables = _amd_pa_decode_build_inputs(
        len(ctx_lens), ctx_lens, num_q_heads, num_kv_heads, head_dim, page_size, device=DEVICE)
    out = torch.empty_like(query)
    _amd_pa_decode(out, query, key_cache, value_cache, context_lens, block_tables, sm_scale, num_splits=4)

    ref = _amd_pa_decode_ref(query, key_cache, value_cache, context_lens, block_tables, sm_scale, num_q_heads,
                             num_kv_heads, 1)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


# =============================================================================
# AMD HSTU Attention Tests (gfx950)
# =============================================================================


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("batch_size, max_seq_len, sparsity, heads, attn_dim, hidden_dim", _hstu.get_inputs())
def test_hstu_attention(batch_size, max_seq_len, sparsity, heads, attn_dim, hidden_dim):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    dropout_pr = 0.0
    target_size = 20
    sl_alpha = 2.0

    # In prod, BF16 is used by HSTU attention
    dtype = torch.bfloat16
    invalid_attn_mask_type = "lower_triangular"
    causal = True
    alpha = 1.0 / attn_dim * 10000

    # generate inputs
    torch.manual_seed(1001)  # for reproducibility
    lengths = _hstu.generate_sparse_seq_len(
        size=batch_size,
        max_seq_len=max_seq_len,
        sparsity=sparsity,
        device=torch.device("cuda"),
    )
    lengths = _hstu.apply_SL(lengths, sl_alpha, max_seq_len=max_seq_len)
    num_targets = torch.randint(
        1,
        target_size + 1,
        (batch_size, ),
        device=lengths.device,
        dtype=lengths.dtype,
    )
    num_targets = torch.where(num_targets > lengths, lengths, num_targets)
    seq_offsets = torch.zeros((batch_size + 1, ), dtype=torch.int64, device=torch.device("cuda"))
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    L = int(seq_offsets[-1].item())
    x = torch.empty(
        (L, heads, attn_dim * 2 + hidden_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)

    q = _hstu.switch_to_contiguous_if_needed(q)
    k = _hstu.switch_to_contiguous_if_needed(k)
    v = _hstu.switch_to_contiguous_if_needed(v)

    _hstu.sanity_check_attention(
        max_seq_len=max_seq_len,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        invalid_attn_mask_type=invalid_attn_mask_type,
        dropout_pr=dropout_pr,
        attn_bias=None,
        max_attn_len=None,
        contextual_seq_len=0,
    )

    def triton_attn():
        return _hstu.triton_hstu_attention_fwd(max_seq_len, alpha, q, k, v, seq_offsets, causal, num_targets,
                                               0,  # max_attn_len,
                                               0,  # contextual_seq_len
                                               True,  # sort_by_length,
                                               )

    def torch_attn():
        return _hstu.torch_hstu_attention(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            causal,
            dropout_pr=0.0,
            training=False,
            num_targets=num_targets,
            max_attn_len=0,
            contextual_seq_len=0,
            min_full_attn_seq_len=0,
        )

    out = triton_attn() * max_seq_len
    out_ref = torch_attn() * max_seq_len
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=0)


# =============================================================================
# AMD TDM GEMM Tests (gfx1250)
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_amd_tdm_gemm_pipelined(dtype):
    Gemm.run_test(_amd_tdm_gemm_pipelined, Gemm.CONFIGS["amd_tdm_gemm_pipelined"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_gemm_warp_pipeline(dtype):
    Gemm.run_test(_amd_gemm_warp_pipeline, Gemm.CONFIGS["amd_gemm_warp_pipeline"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_gemm_pingpong(dtype):
    # Specialized kernel: config is baked in, so it can't go through Gemm.run_test.
    for M, N, K in Gemm.SHAPES:
        torch.manual_seed(0)
        a = (torch.randn((M, K), device=DEVICE, dtype=dtype) + 1) / K
        b = (torch.randn((K, N), device=DEVICE, dtype=dtype) + 1) / K
        torch.testing.assert_close(_amd_gemm_pingpong(a, b), torch.matmul(a, b))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_hip(), reason="Requires AMD GPU")
def test_amd_gemm_pipelined(dtype):
    Gemm.run_test(_amd_gemm_pipelined, Gemm.CONFIGS["amd_gemm_pipelined"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_hip_cdna3(), reason="Requires gfx942 hardware (MI300X / CDNA3)")
def test_amd_gemm_gfx942(dtype):
    # Autotuned kernel: no fixed config (config=None).
    Gemm.run_test(_amd_gemm_gfx942, None, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("bias_shape", ["1d", "2d"])
@pytest.mark.skipif(not is_hip_cdna3(), reason="Requires gfx942 hardware (MI300X / CDNA3)")
def test_amd_addmm_gfx942(bias_shape, dtype):
    # Covers both bias layouts: 1-D (N,) broadcast down the rows (stride_biasm == 0,
    # the Linear case) and a full (M, N).
    for M, N, K in [(1024, 1024, 1024), (4096, 4096, 4096), (255, 129, 130)]:
        torch.manual_seed(0)
        a = (torch.randn((M, K), device=DEVICE, dtype=dtype) + 1) / K
        b = (torch.randn((K, N), device=DEVICE, dtype=dtype) + 1) / K
        bias = torch.randn((N, ) if bias_shape == "1d" else (M, N), device=DEVICE, dtype=dtype)
        torch.testing.assert_close(_amd_addmm_gfx942(bias, a, b), torch.addmm(bias, a, b))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_hip_cdna3(), reason="Requires gfx942 hardware (MI300X / CDNA3)")
def test_amd_bmm_gfx942(dtype):
    # Shared-A (a.stride(0) == 0) is the benchmark convention; the odd shape also
    # exercises the M/N wraparound and the partial K tile.
    for M, N, K, B in [(256, 256, 256, 8), (512, 512, 512, 16), (255, 129, 130, 4)]:
        a, b = _amd_bmm_gfx942_inputs(B, M, N, K, DEVICE, dtype=dtype)
        # make_bmm_inputs yields unscaled randn, so the accumulator grows like
        # sqrt(K) and a different summation order than torch.bmm's costs more
        # than the default 1e-5 atol allows -- measured worst case here is one
        # element in 4.2M off by 1.5e-5 (3.3e-3 relative). The gfx950 sibling
        # uses 2e-2/2e-2 for the same reason; this is an order tighter.
        torch.testing.assert_close(_amd_bmm_gfx942(a, b), torch.bmm(a, b), atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_hip_cdna3(), reason="Requires gfx942 hardware (MI300X / CDNA3)")
def test_amd_bmm_gfx942_distinct_a(dtype):
    # Distinct per-batch A, i.e. a non-zero batch stride on both operands.
    B, M, N, K = 8, 256, 256, 256
    torch.manual_seed(0)
    a = (torch.randn((B, M, K), device=DEVICE, dtype=dtype) + 1) / K
    b = (torch.randn((B, K, N), device=DEVICE, dtype=dtype) + 1) / K
    torch.testing.assert_close(_amd_bmm_gfx942(a, b), torch.bmm(a, b))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_hip_cdna3(), reason="Requires gfx942 hardware (MI300X / CDNA3)")
def test_amd_gemm_gfx942_odd_shapes(dtype):
    # M/N wraparound + masked store, and a partial K tile -- none of which the
    # block-aligned Gemm.SHAPES exercise. Small shapes also make the autotuner
    # prune down to the narrow tiles, covering that path.
    shapes = [(255, 129, 130), (1000, 1000, 200), (64, 64, 4096), (3000, 500, 700)]
    Gemm.run_test(_amd_gemm_gfx942, None, shapes=shapes, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires AMD gfx950 (CDNA4)")
def test_amd_bmm(dtype):
    # a16w16 batched GEMM (col-major B). Covers both load paths of the single kernel:
    # aligned K (K % 32 == 0) -> direct-to-LDS; odd / unaligned K -> register path.
    # K=264 is the boundary case: 8-aligned but NOT BLOCK_K-aligned, so it must take
    # the register path -- the direct path does no K-tail masking and would over-read.
    for M, N, K, B in [(256, 256, 256, 8), (395, 256, 320, 8), (262, 256, 294, 8), (176, 256, 257, 8),
                       (256, 256, 264, 8)]:
        a, b = _amd_bmm_inputs(B, M, N, K, DEVICE, dtype=dtype)
        out = _amd_bmm(a, b)
        ref = torch.bmm(a, b)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


# =============================================================================
# AMD MXFP TDM GEMM Tests (gfx1250)
# =============================================================================


def _mxfp_e8m0_to_float32(scale):
    scale = scale.view(torch.uint8).to(torch.int32)
    scale = scale << 23
    return scale.view(torch.float32)


def _torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K):
    a_scale_f32 = _mxfp_e8m0_to_float32(a_scale).repeat_interleave(scale_block, dim=1)[:M, :K]
    b_scale_f32 = _mxfp_e8m0_to_float32(b_scale).repeat_interleave(scale_block, dim=1).T.contiguous()[:K, :N]
    return torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32)


def _init_fp8_e5m2(rows, cols):
    return torch.randint(20, 40, (rows, cols), dtype=torch.uint8).view(torch.float8_e5m2)


@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_amd_mxfp_gemm_tdm_pipelined(TRANSPOSE_B):
    torch.manual_seed(0)
    M = N = 256
    K = 512
    scale_block = Gemm.CONFIGS["amd_mxfp_gemm_tdm_pipelined"]["SCALE_BLOCK"]
    a = _init_fp8_e5m2(M, K)
    b = _init_fp8_e5m2(K, N)
    a_scale = MXScaleTensor(size=(M, triton.cdiv(K, scale_block))).random(high=32.0).data
    b_scale = MXScaleTensor(size=(N, triton.cdiv(K, scale_block))).random(high=32.0).data
    ref = _torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K)

    a_scale = _amd_mxfp_pack_scale(a_scale)
    b_scale = _amd_mxfp_pack_scale(b_scale)
    a_d = a.contiguous().to(DEVICE)
    b_d = (b.T.contiguous() if TRANSPOSE_B else b.contiguous()).to(DEVICE)

    config = Gemm.CONFIGS["amd_mxfp_gemm_tdm_pipelined"].copy()
    config["TRANSPOSE_B"] = TRANSPOSE_B
    out = _amd_mxfp_gemm_tdm_pipelined(a_d, b_d, a_scale.to(DEVICE), b_scale.to(DEVICE), config=config)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=2e-2)


# =============================================================================
# AMD addmm Tests (gfx950)
# =============================================================================


# The addmm launcher's default `path=None` times its candidate paths against
# each other with `do_bench` and keeps the winner. Correctness tests pin the
# path instead, for two reasons: the timing race costs more wall clock than the
# assertion it guards, and it admits a candidate only once that candidate
# already agrees with `register` -- so a wrong `inter_wave` would be dropped
# from the race and the suite would still pass. Iterating `available_paths`
# asserts every path a shape can take against torch, independently.
def _check_addmm_all_paths(bias, a, b, dtype=torch.float16):
    ref = torch.addmm(bias, a, b)
    config = Gemm.CONFIGS["amd_standalone_addmm_register"]
    for path in _amd_addmm_paths(bias, a, b):
        out = _amd_addmm(bias, a, b, path=path, config=config)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2, msg=lambda m, path=path: f"path={path}\n{m}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("bias_2d,split_k", [(False, 1), (True, 2)], ids=["1d-direct", "2d-split-k"])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_standalone_addmm(dtype, bias_2d, split_k):
    M, N, K = 256, 256, 2048
    torch.manual_seed(0)
    a = (torch.randn(M, K, device=DEVICE, dtype=dtype) + 1) / K
    b = ((torch.randn(N, K, device=DEVICE, dtype=dtype) + 1) / K).T
    bias_shape = (1, N) if bias_2d else (N, )
    bias = torch.randn(bias_shape, device=DEVICE, dtype=dtype)
    ref = torch.addmm(bias, a, b)

    if split_k > 1:
        # SPLIT_K > 1 is inter-wave only, and the launcher routes it directly.
        out = _amd_addmm(bias, a, b, SPLIT_K=split_k)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    else:
        _check_addmm_all_paths(bias, a, b, dtype=dtype)


@pytest.mark.parametrize(
    "M,N,K",
    [
        pytest.param(1024, 896, 1840, id="1024x896x1840"),
        pytest.param(1024, 896, 24, id="1024x896x24"),
        pytest.param(1024, 896, 104, id="1024x896x104"),
        pytest.param(1024, 1536, 2048, id="1024x1536x2048"),
        pytest.param(1024, 6144, 512, id="1024x6144x512"),
        pytest.param(7000, 256, 256, id="7000x256x256"),
        pytest.param(32768, 256, 256, id="32768x256x256"),
    ],
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_standalone_addmm_stock_triton_shapes(M, N, K):
    torch.manual_seed(0)
    a = (torch.randn(M, K, device=DEVICE, dtype=torch.float16) + 1) / K
    b = ((torch.randn(N, K, device=DEVICE, dtype=torch.float16) + 1) / K).T
    bias = torch.randn(N, device=DEVICE, dtype=torch.float16)

    _check_addmm_all_paths(bias, a, b)


# =============================================================================
# AMD addmm + GLU Tests (gfx950)
# =============================================================================


@pytest.mark.parametrize("K", [256, 512, 1024])
@pytest.mark.parametrize("kernel_name", list(_amd_addmm_glu_registry))
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_addmm_glu(kernel_name, K):
    M, N = _amd_addmm_glu_M, _amd_addmm_glu_N
    torch.manual_seed(0)
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(N, device=DEVICE, dtype=torch.float16)
    y = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
    ref = _amd_addmm_glu_baseline(bias, a, b, y)
    out = _amd_addmm_glu_registry[kernel_name](a, b, bias, y)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


# =============================================================================
# gfx950 GDPA (Generalized Dot-Product Attention) Tests
# =============================================================================
#
# The kernel computes gelu via ads_mkl's AMD `fast_gelu`, which is the tanh
# approximation rewritten as x * sigmoid(k*x*(1 + c*x^2)) so it lowers to
# fast_expf/fast_dividef; gfx950 has no tanh.approx.f32. The reference uses exact
# erf gelu, so the tolerance has to absorb the approximation error and is looser
# than the other AMD attention tests. `gelu_approx_error` reports the
# approximation's own contribution -- if a failure is at or near that floor it
# is the approximation, not the kernel.

# Threshold: the measured gelu-approximation floor is rel_l2 ~2.3e-3 across all
# cases below, and bf16 output rounding adds ~2e-3. 1e-2 leaves ~3x headroom
# over that combined floor while still catching a real kernel bug. Compared via
# relative L2 rather than elementwise max-rel: the reference has near-zero
# elements (max_rel reaches 5e3 on them) which make elementwise ratios useless.

# Pinned instead of sweeping the shipped 15-config space once per distinct
# (H, MAX_M, DFF, QK_SCALE). BLOCK_M=128 is the smallest shipped m-tile, so it
# covers the short-Q cases (max_M=64, 137) without wasting ragged-tail rows.
GDPA_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 64,
    "NUM_BUFFERS": 2,
    "matrix_instr_nonkdim": 16,
    "waves_per_eu": 0,
    "num_stages": 1,
    "num_warps": 4,
}


@pytest.mark.parametrize(
    "B,max_M,H,dff,sparsity,seq_len_mode",
    [(8, 500, 4, 256, 0.68, "uniform"),  # prod geometry, small batch
     (8, 500, 4, 256, 0.68, "random"),  # genuinely ragged sequence lengths
     (8, 500, 4, 256, 1.0, "uniform"),  # dense Q
     (4, 137, 4, 256, 0.68, "random"),  # ragged max_M (not a multiple of BLOCK_M)
     (4, 500, 4, 192, 0.68, "random"),  # dff not a power of two
     (1, 64, 2, 256, 1.0, "uniform"),  # single batch, short Q
     ],
    ids=["uniform", "jagged", "dense", "ragged_m", "npot_dff", "single"],
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_gfx950_gdpa(B, max_M, H, dff, sparsity, seq_len_mode):
    """GDPA forward: jagged Q x dense KV, out = gelu(q @ k.T) @ v per sequence."""
    D = H * 64  # head_dim = 64, matching the production shape
    data = _gfx950_gdpa_gen(B, max_M, D, H, dff, sparsity=sparsity, dtype=torch.bfloat16, device=DEVICE, seed=42,
                            seq_len_mode=seq_len_mode)
    q, k, v, q_offsets = data["q"], data["k"], data["v"], data["q_offsets"]

    ref = _gfx950_gdpa_ref(q, k, v, q_offsets, dff, qk_scale=1.0)
    out = _gfx950_gdpa(q, k, v, q_offsets, dff, qk_scale=1.0, config=GDPA_CONFIG)

    assert out.shape == q.shape and out.dtype == q.dtype
    diff = (out.float() - ref.float()).abs()
    rel_l2 = (diff.norm() / ref.float().norm().clamp_min(1e-6)).item()
    if rel_l2 >= 1e-2:
        floor = _gfx950_gdpa_approx_error(q, k, v, q_offsets, dff, qk_scale=1.0)
        pytest.fail(f"GDPA rel_l2={rel_l2:.4e} exceeds 1e-2; "
                    f"gelu-approximation floor is rel_l2={floor['rel_l2']:.4e} "
                    f"(max_abs={floor['max_abs']:.4e}) -- a result near the floor "
                    f"means the approximation, not the kernel")


# =============================================================================
# Multi-CTA Layer Normalization Tests
# =============================================================================


class LayerNorm:
    """Common utilities for multi-CTA layer normalization tests."""

    # (M, N) shapes
    SHAPES = [(4, 16384), (1152, 16384), (4, 32768)]

    @staticmethod
    def run_test(layernorm_fn, shapes=None, dtype=torch.float16, num_ctas=2, **kwargs):
        if shapes is None:
            shapes = LayerNorm.SHAPES
        eps = 1e-5
        for M, N in shapes:
            torch.manual_seed(0)
            x = torch.randn(M, N, device=DEVICE, dtype=dtype)
            weight = torch.randn(N, device=DEVICE, dtype=dtype)
            bias = torch.randn(N, device=DEVICE, dtype=dtype)
            ref_out = torch.nn.functional.layer_norm(x, (N, ), weight, bias, eps)
            tri_out, _, _ = layernorm_fn(x, weight, bias, eps, NUM_CTAS=num_ctas, **kwargs)
            torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_ctas", [1, 2, 4], ids=["1cta", "2cta", "4cta"])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or Blackwell GPU")
def test_multi_cta_layer_norm(num_ctas):
    LayerNorm.run_test(_multi_cta_layernorm, num_ctas=num_ctas)


@pytest.mark.parametrize("num_ctas", [2, 4], ids=["2cta", "4cta"])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or Blackwell GPU")
def test_multi_cta_layer_norm_2d(num_ctas):
    LayerNorm.run_test(_multi_cta_layernorm_2d, num_ctas=num_ctas, BLOCK_SIZE_M=4)


# =============================================================================
# IKBO (In-Kernel Broadcast Optimization) Tests
# =============================================================================

# IKBO is the one tutorial pair that supports both backends explicitly
# (`ikbo_fa_triton` carries separate `_amd_configs` / `_nvidia_configs` and
# flips ALLOW_TF32 on `_is_hip`), so it is gated on "a GPU this kernel targets"
# rather than on gfx950 alone -- a CDNA4-only gate would drop the NVIDIA
# coverage the module is written for.
_ikbo_supported = is_hip_cdna4() or is_hopper_or_newer()


class IkboLce:
    """Common utilities for IKBO LCE tests."""

    # (B, M, N, K_USER, K_CAND, cand_to_user_ratio)
    SHAPES = [
        (512, 128, 256, 1024, 1024, 70),
        (1024, 433, 256, 1184, 872, 100),
    ]

    # Correctness pins the smallest tile rather than sweeping the 48-config
    # space (2x2x2 tiles x 3 stages x 2 warp counts, and no early_config_prune)
    # once per shape. 64x64x64 is valid for every shape: the K loop is masked.
    CONFIG = {"BM": 64, "BN": 64, "BK": 64, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 4}

    ERROR_MULTIPLIER = 1.0
    ERROR_FLOOR = 1e-4

    @staticmethod
    def check_vs_fp32(out, ref_fp16, ref_fp32):
        baseline_err = (ref_fp16.float() - ref_fp32).abs().max().item()
        kernel_err = (out.float() - ref_fp32).abs().max().item()
        threshold = max(IkboLce.ERROR_MULTIPLIER * baseline_err, IkboLce.ERROR_FLOOR)
        assert kernel_err <= threshold, (
            f"IKBO LCE error exceeds baseline: kernel={kernel_err:.4e}, baseline={baseline_err:.4e}")


class IkboFa:
    """Common utilities for IKBO Flash Attention tests."""

    # (B, n_seed, num_heads, d_head, max_seq_len, cand_to_user_ratio)
    SHAPES = [
        (512, 64, 1, 128, 512, 64),
        (1024, 64, 2, 128, 1024, 64),
    ]

    # Smallest tile on either backend; num_warps differs because the AMD and
    # NVIDIA config lists do.
    CONFIG = {
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "num_stages": 2,
        "num_warps": 2 if is_hip() else 4,
    }


@pytest.mark.parametrize(
    "B, M, N, K_USER, K_CAND, ratio",
    IkboLce.SHAPES,
    ids=[f"B{s[0]}_M{s[1]}" for s in IkboLce.SHAPES],
)
@pytest.mark.skipif(not _ikbo_supported, reason="Requires gfx950 (CDNA4) or Hopper+ GPU")
def test_ikbo_lce(B, M, N, K_USER, K_CAND, ratio):
    torch.manual_seed(0)
    cw_c, cw_u, e_c, e_u, idx = _ikbo_lce_create_inputs(
        B,
        M,
        N,
        K_USER,
        K_CAND,
        ratio,
        device=DEVICE,
    )
    ref_fp32 = _ikbo_lce_reference(
        cw_c.float(),
        cw_u.float(),
        e_c.float(),
        e_u.float(),
        idx,
    )
    ref_fp16 = _ikbo_lce_reference(cw_c, cw_u, e_c, e_u, idx)
    out = _ikbo_lce(cw_c, cw_u, e_c, e_u, idx, config=IkboLce.CONFIG)
    IkboLce.check_vs_fp32(out, ref_fp16, ref_fp32)


@pytest.mark.parametrize(
    "B, n_seed, num_heads, d_head, max_seq_len, ratio",
    IkboFa.SHAPES,
    ids=[f"B{s[0]}_h{s[2]}_d{s[3]}" for s in IkboFa.SHAPES],
)
@pytest.mark.skipif(not _ikbo_supported, reason="Requires gfx950 (CDNA4) or Hopper+ GPU")
def test_ikbo_fa(B, n_seed, num_heads, d_head, max_seq_len, ratio):
    random.seed(0)
    torch.manual_seed(0)
    query, key, value, cand_to_user_index, cand_grid = _ikbo_fa_create_inputs(
        B,
        n_seed,
        num_heads,
        d_head,
        max_seq_len,
        cand_to_user_ratio=ratio,
        device=DEVICE,
    )
    ref_out = _ikbo_fa_reference(
        query,
        key,
        value,
        cand_to_user_index,
        n_seed,
        num_heads,
        d_head,
        max_seq_len,
    )
    tri_out = _ikbo_fa(
        query,
        key,
        value,
        cand_to_user_index,
        cand_grid,
        n_seed,
        num_heads,
        d_head,
        max_seq_len,
        config=IkboFa.CONFIG,
    )
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
