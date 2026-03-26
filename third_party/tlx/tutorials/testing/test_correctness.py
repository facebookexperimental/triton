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
    attention as _blackwell_fa_ws_pipelined_persistent,
    _attn_bwd_preprocess as _blackwell_fa_bwd_preprocess,
    _attn_bwd_ws as _blackwell_fa_bwd_ws,
    _attn_fwd_ws as _blackwell_fa_fwd_ws,
    _host_descriptor_pre_hook as _blackwell_fa_fwd_pre_hook,
)
from triton.language.extra.tlx.tutorials.blackwell_fa_clc import (
    attention as _blackwell_fa_clc, )
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

from triton.language.extra.tlx.tutorials.testing.multi_cta_layer_norm import (
    multi_cta_layernorm as _multi_cta_layernorm,
    multi_cta_layernorm_2d as _multi_cta_layernorm_2d,
)

from triton._internal_testing import is_blackwell, is_hopper, is_hopper_or_newer
from triton.language.extra.tlx.tutorials.testing.gemm_shapes import BLACKWELL_GEMM_WS as _BLACKWELL_GEMM_WS_MORE_SHAPES

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


@pytest.mark.parametrize(
    "shape",
    _BLACKWELL_GEMM_WS_MORE_SHAPES,
    ids=[f"{m}x{n}x{k}" for m, n, k in _BLACKWELL_GEMM_WS_MORE_SHAPES],
)
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_gemm_more_shapes(shape):
    Gemm.run_test(_blackwell_gemm_ws, Gemm.CONFIGS["blackwell_gemm_ws"], shapes=[shape], dtype=torch.bfloat16)


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
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent(causal, RESCALE_OPT, USE_WHERE):
    config = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent"].copy()
    config["RESCALE_OPT"] = RESCALE_OPT
    config["USE_WHERE"] = USE_WHERE
    sm_scale = 0.5
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


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("RESCALE_OPT,USE_WHERE", [(False, False), (True, False), (True, True)])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_ws_pipelined_persistent_bwd(causal, RESCALE_OPT, USE_WHERE):
    from triton.tools.tensor_descriptor import TensorDescriptor

    fwd_config: dict[str,
                     bool | int] = FlashAttention.CONFIGS["blackwell_fa_ws_pipelined_persistent_warp_barrier"].copy()
    fwd_config["RESCALE_OPT"] = RESCALE_OPT
    fwd_config["USE_WHERE"] = USE_WHERE
    sm_scale = 0.5
    BWD_BLOCK_M1 = 64
    GROUP_SIZE_M = 1

    for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
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
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = (min(NUM_SMS, triton.cdiv(N_CTX, fwd_config["BLOCK_M"]) * Z * H), 1, 1)
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
        arg_k = k * (sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        pre_grid = (N_CTX // PRE_BLOCK, Z * H)
        delta = torch.empty_like(M)
        _blackwell_fa_bwd_preprocess[pre_grid](o, do, delta, N_CTX, BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM)

        # Backward: main kernel
        dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        desc_bk = TensorDescriptor(arg_k, shape=[Z * H * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                   block_shape=dummy_block)
        desc_bv = TensorDescriptor(v, shape=[Z * H * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
        desc_bq = TensorDescriptor(q, shape=[Z * H * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
        desc_do = TensorDescriptor(do, shape=[Z * H * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
        desc_dq = TensorDescriptor(dq, shape=[Z * H * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
        desc_dk = TensorDescriptor(dk, shape=[Z * H * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
        desc_dv = TensorDescriptor(dv, shape=[Z * H * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)

        BLK_SLICE_FACTOR = 2
        EPILOGUE_SUBTILE = 4 if BWD_BLOCK_M1 == 128 and HEAD_DIM == 128 else 2

        def grid_persistent(meta):
            return (min(NUM_SMS, triton.cdiv(N_CTX, meta["BLOCK_N1"]) * Z * H), 1, 1)

        _blackwell_fa_bwd_ws[grid_persistent](
            desc_bq,
            desc_bk,
            desc_bv,
            sm_scale,
            desc_do,
            desc_dq,
            desc_dk,
            desc_dv,
            M,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            H,
            Z,
            N_CTX,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
            BLOCK_M1=BWD_BLOCK_M1,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            GROUP_SIZE_M=GROUP_SIZE_M,
        )

        tri_dq = dq.to(q.dtype)
        torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=0)
        torch.testing.assert_close(dk, ref_dk, atol=1e-2, rtol=0)
        torch.testing.assert_close(dv, ref_dv, atol=1e-2, rtol=0)


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
