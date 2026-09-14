import math
import random
from dataclasses import replace

import pytest

import torch

import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from triton._internal_testing import (is_blackwell, is_hopper, is_hopper_or_newer, is_hip, is_hip_cdna4, is_hip_gfx1250)

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

# Ungated despite being AMD: the launcher-rejection tests below assert host-side
# validation and carry no arch skipif, so these names must bind everywhere.
from triton.language.extra.tlx.tutorials.amd_addmm_gfx950 import (
    addmm as _amd_addmm,
    available_paths as _amd_addmm_paths,
)
from triton.language.extra.tlx.tutorials.amd_bmm_shared_a import (
    _MT64X256_MI32_KERNEL_SPEC,
    _MT224X160_MI16_KERNEL_SPEC,
    _RESIDENT_OPERAND_B,
    bmm as _shared_a_bmm,
    bmm_register_staged_template,
)
from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a16w16 import (
    matmul_kernel as _amd_gemm, )

# Arch-gated: importing another arch's tutorials is not free. They pull that
# arch's optional deps (the mxfp8 FA tutorial imports torchao), and at module
# scope a missing one is an ImportError that fails the whole file instead of
# skipping the tests that need it. Names left unbound here are only ever
# referenced inside tests that carry the matching skipif.

if is_blackwell():
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
        _attn_fwd_ws as _blackwell_fa_fwd_ws,
        _host_descriptor_pre_hook as _blackwell_fa_fwd_pre_hook,
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

if is_hopper_or_newer():
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

if is_hip():
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
    from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a16w16.matmul_kernel_split_m import (
        matmul as _amd_gemm_pingpong, )
    from triton.language.extra.tlx.tutorials.gfx9_gemm.a16w16.v9_beyond_hotloop.matmul_kernel import (
        matmul as _amd_gemm_v9_beyond_hotloop, )
    from triton.language.extra.tlx.tutorials.amd_mxfp_gemm_tdm_pipelined import (
        matmul as _amd_mxfp_gemm_tdm_pipelined,
        pack_scale as _amd_mxfp_pack_scale,
    )
    from triton.language.extra.tlx.tutorials.gfx950_gdpa import (
        gdpa as _gfx950_gdpa,
        gdpa_ref as _gfx950_gdpa_ref,
        generate_gdpa_data as _gfx950_gdpa_gen,
        gelu_approx_error as _gfx950_gdpa_approx_error,
    )
    from triton.language.extra.tlx.tutorials.amd_bmm import (
        bmm as _amd_bmm,
        make_bmm_inputs as _amd_bmm_inputs,
    )
    from triton.language.extra.tlx.tutorials.amd_addmm_glu import (
        KERNEL_REGISTRY as _amd_addmm_glu_registry,
        pytorch_baseline as _amd_addmm_glu_baseline,
        M as _amd_addmm_glu_M,
        N as _amd_addmm_glu_N,
    )
else:
    _amd_addmm_glu_registry = {}

_IKBO_SUPPORTED = is_hip_cdna4() or is_hopper_or_newer()

# Spelled out rather than read off the registry: the parametrize is evaluated at
# collection time, on arches where the registry was never imported. The test
# asserts this matches the real keys when it does run.
_AMD_ADDMM_GLU_KERNELS = ("tlx_baseline", "tlx_simple_async", "tlx_optimized_async", "tlx_optimized", "tlx_persistent")

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# =============================================================================
# GEMM: Common utilities and configs
# =============================================================================


class Gemm:
    """Common utilities and configs for GEMM tests."""

    SHAPES = [(4096, 4096, 4096)]

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
            "NUM_BUFFERS_KV": 3,
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
    SHAPES = [(1024, 1024, 1024)]

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

# mxfp8 keeps its full config matrix rather than one smoke case: tlx.ops has no
# mxfp8 implementation, so unlike the fp16/bf16 kernels there is nothing in
# python/test/unit/tlx_ops/ backstopping it. Prune these only once mxfp8 lands
# there.


@pytest.mark.parametrize(
    "shape",
    Mxfp8Gemm.SHAPES,
    ids=[f"{m}x{n}x{k}" for m, n, k in Mxfp8Gemm.SHAPES],
)
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8(shape):
    Mxfp8Gemm.run_test(shape)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_ws_mxfp8_2cta():
    Mxfp8Gemm.run_test((256, 256, 256), config=Mxfp8Gemm.CONFIG_2CTA.copy())


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


# CONFIG_2CTA is BLOCK_SIZE_N=256 over 2 CTAs, i.e. 128 columns each. Halving it
# to 64 per CTA is the narrow-tile split (EPILOGUE_SUBTILE=4 then cuts 32-column
# subtiles), and an odd M-tile count pads by a whole tile row.
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
def test_blackwell_gemm_ws_mxfp8_2cta_odd_m_tiles():
    Mxfp8Gemm.run_test((384, 256, 256), config=Mxfp8Gemm.CONFIG_2CTA.copy())


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
def test_blackwell_gemm_ws_mxfp8_split_k():
    # Split-K against block scales: the failure mode is silently wrong rows.
    Mxfp8Gemm.run_test(
        (128, 128, 640),
        config={
            "SPLIT_K": 4,
            "NUM_SMEM_BUFFERS": 4,
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
def test_blackwell_gemm_ws_mxfp8_2cta_uneven_split_k():
    config = Mxfp8Gemm.CONFIG_2CTA.copy()
    config.update({"SPLIT_K": 4, "NUM_SMEM_BUFFERS": 4})
    Mxfp8Gemm.run_test((256, 256, 640), config=config)


@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_clc(dtype):
    Gemm.run_test(_blackwell_gemm_clc, Gemm.CONFIGS["blackwell_gemm_clc"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_pipelined(dtype):
    Gemm.run_test(_blackwell_gemm_pipelined, Gemm.CONFIGS["blackwell_gemm_pipelined"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
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
        return tuple(torch.empty(shape, device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5) for _ in range(3))
    if distribution in ("positive_shift", "negative_shift"):
        q = torch.full(shape, 1.5, device=DEVICE, dtype=dtype)
        k = torch.full(shape, 1.5 if distribution == "positive_shift" else -1.5, device=DEVICE, dtype=dtype)
        v = torch.empty(shape, device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5)
        return q, k, v
    q = torch.ones(shape, device=DEVICE, dtype=dtype)
    q[:, :, shape[2] // 2:, :] = -1
    amplitude = 0.625 if shape[3] == 128 else 0.2
    k = torch.full(shape, amplitude, device=DEVICE, dtype=dtype)
    k[:, :, :128, :] = -amplitude
    v = torch.empty(shape, device=DEVICE, dtype=dtype).normal_(mean=0.0, std=0.5)
    return q, k, v


@pytest.mark.parametrize("RESCALE_OPT,USE_WHERE", [(False, False)])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("N_CTX", [1024])
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


@pytest.mark.parametrize("HEAD_DIM", [64])
@pytest.mark.parametrize("causal", [True])
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
    [(1, 1, 256)],
)
@pytest.mark.parametrize("causal", [True])
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
@pytest.mark.parametrize("shape", ScaledMM.SHAPES[:1], ids=([f"{m}x{n}x{k}" for m, n, k in ScaledMM.SHAPES])[:1])
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
@pytest.mark.parametrize("causal", [False, True])
def test_hopper_fa_ws_pipelined_pingpong(causal):
    config = FlashAttention.CONFIGS["hopper_fa_ws_pipelined_pingpong"]
    sm_scale = 0.5
    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
        q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM)
        ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
        tri_out = _hopper_fa_ws_pipelined_pingpong(
            q,
            k,
            v,
            sm_scale,
            causal=causal,
            config=config,
        )
        torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper GPU")
@pytest.mark.parametrize("causal", [False, True])
def test_hopper_fa_ws_pipelined_pingpong_bwd(causal):
    # The forward-only tests above do not exercise _attn_bwd_tlx, so compare
    # both dense and causal gradients against SDPA independently.
    shape = (1, 1, 1024, 128)
    torch.manual_seed(20)
    q0, k0, v0 = [torch.empty(shape, device=DEVICE, dtype=torch.bfloat16).normal_(mean=0.0, std=0.5) for _ in range(3)]
    do = torch.empty(shape, device=DEVICE, dtype=torch.bfloat16).normal_(mean=0.0, std=0.5)

    ref_q, ref_k, ref_v = [tensor.detach().clone().requires_grad_() for tensor in (q0, k0, v0)]
    ref_o = torch.nn.functional.scaled_dot_product_attention(
        ref_q,
        ref_k,
        ref_v,
        scale=0.5,
        is_causal=causal,
    )
    ref_o.backward(do)
    reference = (ref_q.grad, ref_k.grad, ref_v.grad)

    q, k, v = [tensor.detach().clone().requires_grad_() for tensor in (q0, k0, v0)]
    out = _hopper_fa_ws_pipelined_pingpong(q, k, v, 0.5, causal=causal)
    out.backward(do)
    result = (q.grad, k.grad, v.grad)
    assert all(torch.isfinite(grad).all() for grad in result)
    for grad, ref_grad in zip(result, reference):
        torch.testing.assert_close(grad, ref_grad, atol=2e-1, rtol=1e-1)


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


@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("config_name", ["amd_fa_pipelined"])
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


@pytest.mark.parametrize("causal", [True], ids=["causal"])
@pytest.mark.parametrize("N_CTX", [128])
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


@pytest.mark.parametrize("causal", [True], ids=["causal"])
@pytest.mark.parametrize(
    "q_len,kv_len",
    [(256, 1024)],
    ids=["cross_qlt"],
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


@pytest.mark.parametrize("causal", [False], ids=["nocausal"])
@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
@pytest.mark.parametrize("HEAD_DIM", [64])
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


@pytest.mark.parametrize("N_CTX", [384])
@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_fa_cluster_short_causal_classes(N_CTX, dtype):
    torch.manual_seed(42)
    B, H, D = 1, 8, 128
    q = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sm = 1.0 / math.sqrt(D)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm)
    out = _amd_fa_cluster(q, k, v, sm, True)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("N_CTX", [128])
@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_fa_cluster_persistent_short_causal_lds_normalizes_once(N_CTX, dtype):
    """The persistent BM128 LDS path does not renormalize its predicated diagonal."""
    torch.manual_seed(42)
    B, H, D = 1, 1, 128
    q = torch.randn(B, H, N_CTX, D, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sm = 1.0 / math.sqrt(D)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm)
    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "num_warps": 4,
        "num_stages": 3,
        "waves_per_eu": 0,
        "USE_DIRECT_LOAD": False,
        "NUM_SMS": 8,
        "NUM_XCDS": 8,
        "enable_sched_group_barrier_scheduler": False,
    }
    out = _amd_fa_cluster_persistent(q, k, v, sm, True, config=config)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("persistent", [False], ids=["direct"])
@pytest.mark.parametrize("causal", [False], ids=["nocausal"])
@pytest.mark.parametrize("use_direct_load", [None], ids=["autotune"])
@pytest.mark.parametrize(
    "N_CTX,BLOCK_M",
    [(128, 128)],
    ids=["short"],
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


@pytest.mark.parametrize("causal", [False], ids=["nocausal"])
@pytest.mark.parametrize("HEAD_DIM", [64])
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
@pytest.mark.parametrize("causal", [False], ids=["nocausal"])
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


@pytest.mark.parametrize("query_length", [1], ids=lambda q: f"qlen{q}")
@pytest.mark.parametrize("num_splits", [1], ids=["split1"])
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


# =============================================================================
# AMD TDM GEMM Tests (gfx1250)
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_amd_tdm_gemm_pipelined(dtype):
    Gemm.run_test(_amd_tdm_gemm_pipelined, Gemm.CONFIGS["amd_tdm_gemm_pipelined"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_gemm_warp_pipeline(dtype):
    Gemm.run_test(_amd_gemm_warp_pipeline, Gemm.CONFIGS["amd_gemm_warp_pipeline"], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_gemm_pingpong(dtype):
    # Specialized kernel: config is baked in, so it can't go through Gemm.run_test.
    for M, N, K in Gemm.SHAPES:
        torch.manual_seed(0)
        a = (torch.randn((M, K), device=DEVICE, dtype=dtype) + 1) / K
        b = (torch.randn((K, N), device=DEVICE, dtype=dtype) + 1) / K
        torch.testing.assert_close(_amd_gemm_pingpong(a, b), torch.matmul(a, b))


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_gemm_v9_beyond_hotloop_is_deterministic():
    M, N, K = 131072, 512, 256
    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((N, K), device=DEVICE, dtype=torch.float16).T
    reference = torch.matmul(a, b)

    for _ in range(5):
        actual = _amd_gemm_v9_beyond_hotloop(a, b)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
@pytest.mark.skipif(not is_hip(), reason="Requires AMD GPU")
def test_amd_gemm_pipelined(dtype):
    Gemm.run_test(_amd_gemm_pipelined, Gemm.CONFIGS["amd_gemm_pipelined"], dtype=dtype)


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


@pytest.mark.parametrize(
    "dtype,m,n,k,kernel_spec",
    [
        (torch.float16, 40, 160, 64, _MT64X256_MI32_KERNEL_SPEC),
        (
            torch.bfloat16,
            224,
            160,
            65,
            replace(
                _MT224X160_MI16_KERNEL_SPEC,
                resident_operand_policy=_RESIDENT_OPERAND_B,
            ),
        ),
    ],
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_register_staged_bmm_resident_operand_policies(
    dtype, m, n, k, kernel_spec
):
    """Exercise both dot orders, input dtypes, and partial output tiles."""
    batch = 2
    torch.manual_seed(0)
    a_storage = torch.randn((1, m, k), device=DEVICE, dtype=dtype)
    a = a_storage.expand(batch, -1, -1)
    b = torch.randn((batch, k, n), device=DEVICE, dtype=dtype)

    actual = bmm_register_staged_template(a, b, kernel_spec)
    expected = torch.bmm(a, b)
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "m,n,k,error",
    [
        (64, 256, 31, "K must be >= BLOCK_K=32"),
        (31, 256, 64, "BLOCK_M=64 requires M >= 32"),
        (64, 127, 64, "BLOCK_N=256 requires N >= 128"),
    ],
)
def test_register_staged_bmm_rejects_unsupported_wrap_contracts(m, n, k, error):
    a = torch.empty((1, m, k), dtype=torch.float16)
    b = torch.empty((1, k, n), dtype=torch.float16)

    with pytest.raises(AssertionError, match=error):
        bmm_register_staged_template(a, b, _MT64X256_MI32_KERNEL_SPEC)


@pytest.mark.parametrize(
    "batch,m,n,k,dtype",
    [
        (63, 40, 256, 1956, torch.float16),
        (255, 262, 256, 294, torch.float16),
        (256, 448, 160, 931, torch.float16),
        (64, 1195, 256, 2309, torch.bfloat16),
    ],
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_shared_a_bmm_production_dispatches_at_group_boundaries(
    batch, m, n, k, dtype
):
    """Cover every tuned dispatch at the batch size that enables grouping."""
    torch.manual_seed(0)
    a = torch.randn((1, m, k), device=DEVICE, dtype=dtype).expand(batch, -1, -1)
    b = torch.randn((batch, k, n), device=DEVICE, dtype=dtype)

    actual = _shared_a_bmm(a, b)
    expected = torch.bmm(a, b)
    atol = 5e-1 if dtype == torch.bfloat16 else 2e-2
    torch.testing.assert_close(actual, expected, atol=atol, rtol=2e-2)


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


@pytest.mark.parametrize("TRANSPOSE_B", [False])
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
def test_amd_gemm_offset_width_selection():
    i32_max_element = (1 << 30) - 1
    within_i32 = torch.empty((i32_max_element + 1, ), device="meta", dtype=torch.float16)
    beyond_i32 = torch.empty((i32_max_element + 2, ), device="meta", dtype=torch.float16)

    assert not _amd_gemm._needs_i64_offsets(within_i32)
    assert _amd_gemm._needs_i64_offsets(beyond_i32)


def test_amd_gemm_output_offset_width_selection(monkeypatch):
    launches = []

    class FakeKernel:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                launches.append((grid, kwargs["USE_I64_C_OFFSETS"]))

            return launch

    monkeypatch.setattr(_amd_gemm, "a16w16_8wave", FakeKernel())
    for M, N in [(256, 256), (925210, 4096)]:
        a = torch.empty((M, 128), device="meta", dtype=torch.float16)
        b = torch.empty((128, N), device="meta", dtype=torch.float16)
        _amd_gemm._launch(a, b, SPLIT_K=1, TILE=(256, 256))

    assert [use_i64_c_offsets for _, use_i64_c_offsets in launches] == [False, True]


def test_amd_gemm_input_offset_width_selection(monkeypatch):
    launches = []

    class FakeKernel:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                launches.append(
                    (
                        kwargs["USE_I64_A_OFFSETS"],
                        kwargs["USE_I64_B_OFFSETS"],
                        kwargs["HAS_M_TAIL"],
                        kwargs["HAS_N_TAIL"],
                    )
                )

            return launch

    monkeypatch.setattr(_amd_gemm, "a16w16_8wave", FakeKernel())
    cases = [
        ((256, 256, 4096), (False, False, False, False)),
        ((257, 256, 4096), (False, False, True, False)),
        ((256, 257, 4096), (False, False, False, True)),
        ((262400, 256, 4096), (True, False, False, False)),
        ((256, 262400, 4096), (False, True, False, False)),
    ]
    for (M, N, K), _ in cases:
        a = torch.empty((M, K), device="meta", dtype=torch.float16)
        b = torch.empty((K, N), device="meta", dtype=torch.float16)
        _amd_gemm._launch(a, b, SPLIT_K=1, TILE=(256, 256))

    assert launches == [expected for _, expected in cases]


@pytest.mark.parametrize(
    "split_k,defer_epilogue",
    [(2, False)],
    ids=["split-k"],
)
def test_amd_gemm_rejects_large_workspace(split_k, defer_epilogue):
    M, N, K = 262145, 2048, 256
    a = torch.empty((M, K), device="meta", dtype=torch.float16)
    b = torch.empty((K, N), device="meta", dtype=torch.float16)

    with pytest.raises(ValueError, match="FP32 workspace exceeds signed-i32 byte offsets"):
        _amd_gemm._launch(
            a,
            b,
            SPLIT_K=split_k,
            TILE=(256, 256),
            DEFER_EPILOGUE=defer_epilogue,
        )


def test_amd_addmm_rejects_register_config_for_inter_wave():
    a = torch.empty((256, 2048), device="meta", dtype=torch.float16)
    b = torch.empty((256, 2048), device="meta", dtype=torch.float16).T
    bias = torch.empty(256, device="meta", dtype=torch.float16)

    with pytest.raises(ValueError, match="config is only supported by the register path"):
        _amd_addmm(bias, a, b, path="inter_wave", config={})


def _check_addmm_all_paths(bias, a, b, split_k=None):
    ref = torch.addmm(bias, a, b)
    config = Gemm.CONFIGS["amd_standalone_addmm_register"]
    for path in _amd_addmm_paths(bias, a, b):
        path_config = config if path == "register" else None
        out = _amd_addmm(bias, a, b, SPLIT_K=split_k, path=path, config=path_config)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2, msg=lambda m, path=path: f"path={path}\n{m}")


def _check_addmm_default_matches_register_exact(bias, a, b, split_k):
    config = Gemm.CONFIGS["amd_standalone_addmm_register"]
    expected = _amd_addmm(bias, a, b, SPLIT_K=split_k, path="register", config=config)
    actual = _amd_addmm(bias, a, b, SPLIT_K=split_k, config=config)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "bias_2d,split_k,N",
    [(False, 1, 256), (True, 2, 256), (False, 1, 384)],
    ids=["1d-direct", "2d-split-k", "1d-direct-n-tail"],
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_standalone_addmm(dtype, bias_2d, split_k, N):
    M, K = 256, 2048
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
        _check_addmm_all_paths(bias, a, b, split_k)
        _check_addmm_default_matches_register_exact(bias, a, b, split_k)


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
@pytest.mark.parametrize("kernel_name", _AMD_ADDMM_GLU_KERNELS)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_amd_addmm_glu(kernel_name, K):
    assert tuple(_amd_addmm_glu_registry) == _AMD_ADDMM_GLU_KERNELS, "registry drifted from the parametrize list"
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
    [(8, 500, 4, 256, 0.68, "uniform")],
    ids=["uniform"],
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
    SHAPES = [(4, 16384)]

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


@pytest.mark.parametrize("num_ctas", [1], ids=["1cta"])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or Blackwell GPU")
def test_multi_cta_layer_norm(num_ctas):
    LayerNorm.run_test(_multi_cta_layernorm, num_ctas=num_ctas)


@pytest.mark.parametrize("num_ctas", [2], ids=["2cta"])
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
_ikbo_supported = _IKBO_SUPPORTED


class IkboLce:
    """Common utilities for IKBO LCE tests."""

    # (B, M, N, K_USER, K_CAND, cand_to_user_ratio)
    SHAPES = [(512, 128, 256, 1024, 1024, 70)]

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
    SHAPES = [(512, 64, 1, 128, 512, 64)]

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
    IkboLce.SHAPES[:1],
    ids=([f"B{s[0]}_M{s[1]}" for s in IkboLce.SHAPES])[:1],
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
    IkboFa.SHAPES[:1],
    ids=([f"B{s[0]}_h{s[2]}_d{s[3]}" for s in IkboFa.SHAPES])[:1],
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
