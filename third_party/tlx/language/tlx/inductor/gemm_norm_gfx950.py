"""gfx950 addmm + normalization fusion for TorchInductor."""

from __future__ import annotations

import functools
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from torch._inductor import config
from torch._inductor.pattern_matcher import fwd_only, Match, register_replacement
from torch.library import wrap_triton

from ..hw.target import current_target
from .gfx950_addmm_warppipe_core import gfx950_addmm_warppipe_compute_async


_SUPPORTED_N = (2048, 2560, 4096)
_BLOCK_M = 8
_BLOCK_N = 128
_BLOCK_K = 64
_NUM_BUFFERS = 2
_GEMM_NORM_CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": _BLOCK_M,
            "BLOCK_N": _BLOCK_N,
            "BLOCK_K": _BLOCK_K,
            "NUM_BUFFERS": _NUM_BUFFERS,
            "waves_per_eu": 1,
        },
        num_warps=8,
        num_stages=1,
    )
]


@triton.jit
def _tlx_gfx950_addmm_norm_body(
    x_ptr,
    weight_ptr,
    gemm_bias_ptr,
    scale_ptr,
    norm_bias_ptr,
    output_ptr,
    M,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_gemm_bias,
    stride_scale,
    stride_norm_bias,
    stride_om,
    stride_on,
    EPS: tl.constexpr,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    K: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    rows = rows % M
    offs_k = tl.arange(0, BLOCK_K)

    smem_a = tlx.local_alloc(
        (BLOCK_M, BLOCK_K),
        tl.bfloat16,
        NUM_BUFFERS,
    )
    smem_b = tlx.local_alloc(
        (BLOCK_N, BLOCK_K),
        tl.bfloat16,
        NUM_BUFFERS,
    )

    # The addmm output is materialized to BF16 before normalization in the
    # unfused graph. Retaining that representation preserves its cast boundary.
    retained_layout: tl.constexpr = tlx.swizzled_layout(0, 0, 0, order=[1, 0])
    retained = tlx.local_alloc(
        (BLOCK_M, N_PAD),
        tl.bfloat16,
        1,
        layout=retained_layout,
    )
    retained_view = tlx.local_view(retained, 0)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    row_sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    a_base = rows[:, None] * stride_xm
    k_iters = K // BLOCK_K

    for n_start in range(0, N, BLOCK_N):
        cols = n_start + tl.arange(0, BLOCK_N)
        acc = gfx950_addmm_warppipe_compute_async(
            x_ptr,
            weight_ptr,
            smem_a,
            smem_b,
            a_base,
            cols * stride_wn,
            offs_k,
            0,
            k_iters,
            k_iters * BLOCK_K,
            K - k_iters * BLOCK_K,
            stride_xk,
            stride_wk,
            False,
            tl.float32,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            NUM_BUFFERS,
        )

        gemm_bias = tl.load(
            gemm_bias_ptr + cols * stride_gemm_bias,
            mask=cols < N,
            other=0.0,
        )
        value = (acc + gemm_bias[None, :]).to(tl.bfloat16)
        local_tile = tlx.local_slice(
            retained_view,
            [0, n_start],
            [BLOCK_M, BLOCK_N],
        )
        tlx.local_store(local_tile, value)
        value_fp32 = value.to(tl.float32)
        row_sum_sq += tl.sum(value_fp32 * value_fp32, axis=1)
        if not IS_RMS_NORM:
            row_sum += tl.sum(value_fp32, axis=1)

    tl.debug_barrier()
    if IS_RMS_NORM:
        inverse_std = tl.rsqrt(row_sum_sq / N + EPS)
        mean = tl.zeros((BLOCK_M,), dtype=tl.float32)
    else:
        mean = row_sum / N
        variance = tl.maximum(row_sum_sq / N - mean * mean, 0.0)
        inverse_std = tl.rsqrt(variance + EPS)

    for n_start in range(0, N, BLOCK_N):
        cols = n_start + tl.arange(0, BLOCK_N)
        local_tile = tlx.local_slice(
            retained_view,
            [0, n_start],
            [BLOCK_M, BLOCK_N],
        )
        value = tlx.local_load(local_tile).to(tl.float32)
        scale = tl.load(
            scale_ptr + cols * stride_scale,
            mask=cols < N,
            other=0.0,
        ).to(tl.float32)
        normalized = (value - mean[:, None]) * inverse_std[:, None]
        normalized *= scale[None, :]
        if not IS_RMS_NORM:
            norm_bias = tl.load(
                norm_bias_ptr + cols * stride_norm_bias,
                mask=cols < N,
                other=0.0,
            ).to(tl.float32)
            normalized += norm_bias[None, :]
        output_offsets = rows[:, None] * stride_om + cols[None, :] * stride_on
        tl.store(
            output_ptr + output_offsets,
            normalized.to(output_ptr.dtype.element_ty),
            mask=row_mask[:, None] & (cols[None, :] < N),
        )


@triton.autotune(configs=_GEMM_NORM_CONFIGS, key=["N", "K"])
@triton.jit
def tlx_gfx950_addmm_rmsnorm(
    x_ptr,
    weight_ptr,
    gemm_bias_ptr,
    scale_ptr,
    norm_bias_ptr,
    output_ptr,
    M,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_gemm_bias,
    stride_scale,
    stride_norm_bias,
    stride_om,
    stride_on,
    EPS: tl.constexpr,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    _tlx_gfx950_addmm_norm_body(
        x_ptr,
        weight_ptr,
        gemm_bias_ptr,
        scale_ptr,
        norm_bias_ptr,
        output_ptr,
        M,
        stride_xm,
        stride_xk,
        stride_wk,
        stride_wn,
        stride_gemm_bias,
        stride_scale,
        stride_norm_bias,
        stride_om,
        stride_on,
        EPS,
        N,
        N_PAD,
        K,
        True,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BUFFERS,
    )


@triton.autotune(configs=_GEMM_NORM_CONFIGS, key=["N", "K"])
@triton.jit
def tlx_gfx950_addmm_layernorm(
    x_ptr,
    weight_ptr,
    gemm_bias_ptr,
    scale_ptr,
    norm_bias_ptr,
    output_ptr,
    M,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_gemm_bias,
    stride_scale,
    stride_norm_bias,
    stride_om,
    stride_on,
    EPS: tl.constexpr,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    _tlx_gfx950_addmm_norm_body(
        x_ptr,
        weight_ptr,
        gemm_bias_ptr,
        scale_ptr,
        norm_bias_ptr,
        output_ptr,
        M,
        stride_xm,
        stride_xk,
        stride_wk,
        stride_wn,
        stride_gemm_bias,
        stride_scale,
        stride_norm_bias,
        stride_om,
        stride_on,
        EPS,
        N,
        N_PAD,
        K,
        False,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BUFFERS,
    )


def _fused_gfx950_addmm_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[1]
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    wrap_triton(tlx_gfx950_addmm_rmsnorm)[
        (triton.cdiv(m, _BLOCK_M),)
    ](
        x,
        weight,
        gemm_bias,
        scale,
        scale,
        output,
        m,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        gemm_bias.stride(0),
        scale.stride(0),
        scale.stride(0),
        output.stride(0),
        output.stride(1),
        EPS=eps,
        N=n,
        N_PAD=triton.next_power_of_2(n),
        K=k,
    )
    return output


def _fused_gfx950_addmm_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[1]
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    wrap_triton(tlx_gfx950_addmm_layernorm)[
        (triton.cdiv(m, _BLOCK_M),)
    ](
        x,
        weight,
        gemm_bias,
        scale,
        norm_bias,
        output,
        m,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        gemm_bias.stride(0),
        scale.stride(0),
        norm_bias.stride(0),
        output.stride(0),
        output.stride(1),
        EPS=eps,
        N=n,
        N_PAD=triton.next_power_of_2(n),
        K=k,
    )
    return output


def _aten_gfx950_addmm_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    value = torch.addmm(gemm_bias, x, weight)
    return torch.nn.functional.rms_norm(value, (value.shape[-1],), scale, eps)


def _aten_gfx950_addmm_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    value = torch.addmm(gemm_bias, x, weight)
    return torch.nn.functional.layer_norm(
        value,
        (value.shape[-1],),
        scale,
        norm_bias,
        eps,
    )


@torch.library.custom_op("torch_tlx::gfx950_addmm_rmsnorm", mutates_args=())
def gfx950_addmm_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _aten_gfx950_addmm_rmsnorm(x, weight, gemm_bias, scale, eps)


@gfx950_addmm_rmsnorm.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[1]),
        device=x.device,
        dtype=x.dtype,
    )


@torch.library.custom_op("torch_tlx::gfx950_addmm_layernorm", mutates_args=())
def gfx950_addmm_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _aten_gfx950_addmm_layernorm(
        x,
        weight,
        gemm_bias,
        scale,
        norm_bias,
        eps,
    )


@gfx950_addmm_layernorm.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[1]),
        device=x.device,
        dtype=x.dtype,
    )


def _eligible(match: Match) -> bool:
    if config.triton.tlx_mode not in ("allow", "force"):
        return False
    if not current_target().is_gfx950:
        return False
    addmm = next(
        (
            node
            for node in match.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.addmm.default
        ),
        None,
    )
    if addmm is None or len(addmm.users) != 1:
        return False
    tensor_names = ["x", "weight", "gemm_bias", "scale"]
    if "norm_bias" in match.kwargs:
        tensor_names.append("norm_bias")
    tensors = [
        match.kwargs[name].meta.get("val")
        for name in tensor_names
    ]
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return False
    x, weight, gemm_bias, scale, *optional_norm_bias = tensors
    if x.ndim != 2 or weight.ndim != 2:
        return False
    vector_inputs = [gemm_bias, scale, *optional_norm_bias]
    if any(value.ndim != 1 for value in vector_inputs):
        return False
    try:
        k = int(x.shape[1])
        weight_k = int(weight.shape[0])
        n = int(weight.shape[1])
        vector_sizes = [int(value.shape[0]) for value in vector_inputs]
    except (TypeError, ValueError):
        return False
    return bool(
        x.dtype == torch.bfloat16
        and weight.dtype == x.dtype
        and gemm_bias.dtype == x.dtype
        and scale.dtype == x.dtype
        and all(value.dtype == x.dtype for value in optional_norm_bias)
        and k == weight_k
        and n in _SUPPORTED_N
        and all(size == n for size in vector_sizes)
        and n % _BLOCK_N == 0
        and k % _BLOCK_K == 0
    )


def _register_autotuned_region(custom_op, fused_impl, aten_impl, name: str) -> None:
    from torch._inductor.kernel.custom_op import (
        CustomOpConfig,
        register_custom_op_autotuning,
    )
    from torch._inductor.lowering import user_lowerings

    register_custom_op_autotuning(
        custom_op,
        configs=[CustomOpConfig(fused_impl), CustomOpConfig(aten_impl)],
        name=f"{name}_allow",
        include_fallback=False,
    )
    op_overload = custom_op._opoverload
    allow_lowering = user_lowerings[op_overload]

    register_custom_op_autotuning(
        custom_op,
        configs=[CustomOpConfig(fused_impl)],
        name=name,
        include_fallback=False,
    )
    force_lowering = user_lowerings[op_overload]

    @functools.wraps(allow_lowering)
    def lowering(*args, **kwargs):
        if config.triton.tlx_mode == "force":
            return force_lowering(*args, **kwargs)
        return allow_lowering(*args, **kwargs)

    user_lowerings[op_overload] = lowering


@functools.cache
def register_gemm_norm_patterns() -> None:
    from torch._inductor.fx_passes.post_grad import pass_patterns

    _register_autotuned_region(
        gfx950_addmm_rmsnorm,
        _fused_gfx950_addmm_rmsnorm,
        _aten_gfx950_addmm_rmsnorm,
        "tlx_gfx950_addmm_rmsnorm",
    )
    _register_autotuned_region(
        gfx950_addmm_layernorm,
        _fused_gfx950_addmm_layernorm,
        _aten_gfx950_addmm_layernorm,
        "tlx_gfx950_addmm_layernorm",
    )

    n = _SUPPORTED_N[0]
    example_x = torch.empty((2, 64), dtype=torch.bfloat16)
    example_weight = torch.empty((64, n), dtype=torch.bfloat16)
    example_gemm_bias = torch.empty((n,), dtype=torch.bfloat16)
    example_scale = torch.empty((n,), dtype=torch.bfloat16)
    example_norm_bias = torch.empty((n,), dtype=torch.bfloat16)

    def rmsnorm_pattern(x, weight, gemm_bias, scale):
        value = torch.addmm(gemm_bias, x, weight)
        value_fp32 = torch.ops.prims.convert_element_type.default(
            value,
            torch.float32,
        )
        mean_square = torch.ops.aten.mean.dim(
            torch.ops.aten.pow.Tensor_Scalar(value_fp32, 2),
            [1],
            True,
        )
        inverse_std = torch.ops.aten.rsqrt.default(
            torch.ops.aten.add.Scalar(mean_square, 1.0e-5)
        )
        normalized = torch.ops.aten.mul.Tensor(value_fp32, inverse_std)
        scaled = torch.ops.aten.mul.Tensor(normalized, scale)
        return torch.ops.prims.convert_element_type.default(scaled, torch.bfloat16)

    def rmsnorm_replacement(x, weight, gemm_bias, scale):
        return gfx950_addmm_rmsnorm(
            x,
            weight,
            gemm_bias,
            scale,
            1.0e-5,
        )

    def layernorm_pattern(x, weight, gemm_bias, scale, norm_bias):
        value = torch.addmm(gemm_bias, x, weight)
        return torch.nn.functional.layer_norm(
            value,
            (value.shape[-1],),
            scale,
            norm_bias,
            1.0e-5,
        )

    def layernorm_replacement(x, weight, gemm_bias, scale, norm_bias):
        return gfx950_addmm_layernorm(
            x,
            weight,
            gemm_bias,
            scale,
            norm_bias,
            1.0e-5,
        )

    rmsnorm_inputs = (
        example_x,
        example_weight,
        example_gemm_bias,
        example_scale,
    )
    layernorm_inputs = (
        example_x,
        example_weight,
        example_gemm_bias,
        example_scale,
        example_norm_bias,
    )
    register_replacement(
        rmsnorm_pattern,
        rmsnorm_replacement,
        rmsnorm_inputs,
        fwd_only,
        pass_patterns[0],
        extra_check=_eligible,
        pattern_name="tlx_gfx950_addmm_rmsnorm",
    )
    register_replacement(
        layernorm_pattern,
        layernorm_replacement,
        layernorm_inputs,
        fwd_only,
        pass_patterns[0],
        extra_check=_eligible,
        pattern_name="tlx_gfx950_addmm_layernorm",
    )
