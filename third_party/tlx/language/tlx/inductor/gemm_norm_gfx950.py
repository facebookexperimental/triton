"""gfx950 addmm + normalization fusion for TorchInductor."""

from __future__ import annotations

import functools
import torch
import triton
import triton.language as tl
from torch._inductor import config
from torch._inductor.pattern_matcher import fwd_only, Match, register_replacement
from torch.library import wrap_triton
from triton.tlx.ops.kernels.mm.gfx950 import a16w16_8wave

from ..hw.target import current_target


_LAYERNORM_SHAPE = (2032, 2560, 2560)
_RMSNORM_SHAPE = (677, 8192, 4096)
_PROD_CONFIGS = {
    _LAYERNORM_SHAPE: (128, 128, 1, 4, 4),
    _RMSNORM_SHAPE: (192, 256, 4, 4, 8),
}
_BLOCK_K = 64
_GEMM_BLOCK_N = 128
_NORM_BLOCK_N = 4096
_STATS_BLOCKS = 32
_SPLIT_REDUCE_BLOCK_M = 32
_SPLIT_REDUCE_BLOCK_N = 256


@triton.jit
def tlx_gfx950_addmm_rmsnorm_stats(
    workspace_ptr,
    gemm_bias_ptr,
    raw_ptr,
    row_sum_sq_ptr,
    M,
    N,
    stride_gemm_bias,
    stride_raw_m,
    stride_raw_n,
    SPLIT_K: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offsets = rows[:, None] * N + cols[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split_k in range(SPLIT_K):
        acc += tl.load(
            workspace_ptr + split_k * M * N + offsets,
            mask=mask,
            other=0.0,
        )
    gemm_bias = tl.load(
        gemm_bias_ptr + cols * stride_gemm_bias,
        mask=cols < N,
        other=0.0,
    )
    value = (acc + gemm_bias[None, :]).to(tl.bfloat16)
    raw_offsets = rows[:, None] * stride_raw_m + cols[None, :] * stride_raw_n
    tl.store(raw_ptr + raw_offsets, value, mask=mask)

    stats_offsets = rows * N_BLOCKS + pid_n
    value_fp32 = value.to(tl.float32)
    tl.store(
        row_sum_sq_ptr + stats_offsets,
        tl.sum(value_fp32 * value_fp32, axis=1),
        mask=rows < M,
    )


@triton.jit
def tlx_gfx950_apply_norm(
    raw_ptr,
    row_sum_ptr,
    row_sum_sq_ptr,
    scale_ptr,
    norm_bias_ptr,
    output_ptr,
    M,
    stride_raw_m,
    stride_raw_n,
    stride_scale,
    stride_norm_bias,
    stride_output_m,
    stride_output_n,
    EPS: tl.constexpr,
    N: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    USE_PARTIAL_STATS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_STATS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    raw_offsets = row * stride_raw_m + cols * stride_raw_n
    value = tl.load(raw_ptr + raw_offsets, mask=col_mask, other=0.0).to(
        tl.float32
    )
    if USE_PARTIAL_STATS:
        stats_cols = tl.arange(0, BLOCK_STATS)
        stats_mask = stats_cols < N_BLOCKS
        stats_offsets = row * N_BLOCKS + stats_cols
        row_sum_sq = tl.sum(
            tl.load(row_sum_sq_ptr + stats_offsets, mask=stats_mask, other=0.0),
            axis=0,
        )
    else:
        row_sum_sq = tl.sum(value * value, axis=0)
    if IS_RMS_NORM:
        mean = 0.0
        inverse_std = tl.rsqrt(row_sum_sq / N + EPS)
    else:
        if USE_PARTIAL_STATS:
            row_sum = tl.sum(
                tl.load(
                    row_sum_ptr + stats_offsets,
                    mask=stats_mask,
                    other=0.0,
                ),
                axis=0,
            )
        else:
            row_sum = tl.sum(value, axis=0)
        mean = row_sum / N
        variance = tl.maximum(row_sum_sq / N - mean * mean, 0.0)
        inverse_std = tl.rsqrt(variance + EPS)

    scale = tl.load(
        scale_ptr + cols * stride_scale,
        mask=col_mask,
    ).to(tl.float32)
    normalized = (value - mean) * inverse_std * scale
    if not IS_RMS_NORM:
        normalized += tl.load(
            norm_bias_ptr + cols * stride_norm_bias,
            mask=col_mask,
        ).to(tl.float32)
    output_offsets = row * stride_output_m + cols * stride_output_n
    tl.store(output_ptr + output_offsets, normalized, mask=col_mask)


def _launch_gfx950_addmm_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    *,
    is_rms_norm: bool,
) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[1]
    block_m, block_n, split_k, group_size_m, num_warps = _PROD_CONFIGS[
        (m, k, n)
    ]
    grid_mn = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    raw = torch.empty((m, n), device=x.device, dtype=x.dtype)
    stats_block_n = block_n if split_k == 1 else _SPLIT_REDUCE_BLOCK_N
    n_blocks = triton.cdiv(n, stats_block_n)
    output = torch.empty_like(raw)
    row_sum_sq = (
        raw
        if split_k == 1
        else torch.empty(
            (m, n_blocks),
            device=x.device,
            dtype=torch.float32,
        )
    )
    workspace = (
        raw
        if split_k == 1
        else torch.empty(
            (split_k * m, n),
            device=x.device,
            dtype=torch.float32,
        )
    )

    wrap_triton(a16w16_8wave)[(grid_mn * split_k,)](
        x,
        weight,
        gemm_bias,
        raw,
        workspace,
        m,
        n,
        k,
        k // split_k,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        0,
        gemm_bias.stride(0),
        workspace.stride(0),
        workspace.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=_BLOCK_K,
        GROUP_SIZE_M=group_size_m,
        NUM_XCDS=8,
        GRID_MN=grid_mn,
        SPLIT_K=split_k,
        ADD_BIAS=split_k == 1,
        HAS_REGISTER_TAIL=False,
        USE_I64_A_OFFSETS=False,
        USE_I64_B_OFFSETS=False,
        USE_I64_C_OFFSETS=False,
        UNEVEN_SPLIT_K=False,
        HAS_M_TAIL=True,
        HAS_N_TAIL=False,
        PIN_OFFSET_LAYOUT=False,
        DEFER_EPILOGUE=split_k > 1,
        num_warps=num_warps,
        num_stages=1,
        matrix_instr_nonkdim=16,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
        enable_sched_group_barrier_scheduler=True,
    )

    if split_k > 1:
        wrap_triton(tlx_gfx950_addmm_rmsnorm_stats)[
            (
                triton.cdiv(m, _SPLIT_REDUCE_BLOCK_M),
                n_blocks,
            )
        ](
            workspace,
            gemm_bias,
            raw,
            row_sum_sq,
            m,
            n,
            gemm_bias.stride(0),
            raw.stride(0),
            raw.stride(1),
            SPLIT_K=split_k,
            N_BLOCKS=n_blocks,
            BLOCK_M=_SPLIT_REDUCE_BLOCK_M,
            BLOCK_N=_SPLIT_REDUCE_BLOCK_N,
            num_warps=4,
        )

    wrap_triton(tlx_gfx950_apply_norm)[(m,)](
        raw,
        raw,
        row_sum_sq,
        scale,
        norm_bias,
        output,
        m,
        raw.stride(0),
        raw.stride(1),
        scale.stride(0),
        norm_bias.stride(0),
        output.stride(0),
        output.stride(1),
        EPS=eps,
        N=n,
        N_BLOCKS=n_blocks,
        IS_RMS_NORM=is_rms_norm,
        USE_PARTIAL_STATS=split_k > 1,
        BLOCK_N=_NORM_BLOCK_N,
        BLOCK_STATS=_STATS_BLOCKS,
        num_warps=8,
    )
    return output


def _fused_gfx950_addmm_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _launch_gfx950_addmm_norm(
        x,
        weight,
        gemm_bias,
        scale,
        scale,
        eps,
        is_rms_norm=True,
    )


def _fused_gfx950_addmm_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _launch_gfx950_addmm_norm(
        x,
        weight,
        gemm_bias,
        scale,
        norm_bias,
        eps,
        is_rms_norm=False,
    )


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


def _eligible(match: Match, expected_shape: tuple[int, int, int]) -> bool:
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
        m = int(x.shape[0])
        k = int(x.shape[1])
        weight_k = int(weight.shape[0])
        n = int(weight.shape[1])
        vector_sizes = [int(value.shape[0]) for value in vector_inputs]
        x_strides = tuple(int(stride) for stride in x.stride())
        weight_strides = tuple(int(stride) for stride in weight.stride())
        vector_strides = [int(value.stride(0)) for value in vector_inputs]
    except (TypeError, ValueError):
        return False
    return bool(
        (m, k, n) == expected_shape
        and x.dtype == torch.bfloat16
        and weight.dtype == x.dtype
        and gemm_bias.dtype == x.dtype
        and scale.dtype == x.dtype
        and all(value.dtype == x.dtype for value in optional_norm_bias)
        and all(value.device == x.device for value in tensors)
        and k == weight_k
        and all(size == n for size in vector_sizes)
        and x_strides == (k, 1)
        and weight_strides == (1, k)
        and all(stride == 1 for stride in vector_strides)
        and n % _GEMM_BLOCK_N == 0
        and k % _BLOCK_K == 0
    )


def _eligible_rmsnorm(match: Match) -> bool:
    return _eligible(match, _RMSNORM_SHAPE)


def _eligible_layernorm(match: Match) -> bool:
    return _eligible(match, _LAYERNORM_SHAPE)


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

    n = _LAYERNORM_SHAPE[2]
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
        extra_check=_eligible_rmsnorm,
        pattern_name="tlx_gfx950_addmm_rmsnorm",
    )
    register_replacement(
        layernorm_pattern,
        layernorm_replacement,
        layernorm_inputs,
        fwd_only,
        pass_patterns[0],
        extra_check=_eligible_layernorm,
        pattern_name="tlx_gfx950_addmm_layernorm",
    )
