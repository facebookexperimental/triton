"""gfx950 addmm + RMSNorm fusion for TorchInductor."""

from __future__ import annotations

import functools

import torch
from torch._inductor.pattern_matcher import fwd_only, Match, register_replacement

from .gemm_norm_gfx950 import (
    _NORM_BLOCK_N,
    _SPLIT_REDUCE_BLOCK_M,
    _SPLIT_REDUCE_BLOCK_N,
    _SPLIT_STATS,
    _eligible,
    _launch_gfx950_addmm_norm,
    _make_lds_plans,
    _register_autotuned_region,
)

_SHAPE = (677, 8192, 4096)

# TODO: Full-workspace split-K does not beat the non-split register GEMM plus
# standalone RMSNorm for this shape. Revisit split-K with Stream-K or another
# design that avoids a SPLIT_K * M * N FP32 workspace.

# A plan is (kind, block_m, block_n, block_k, group_m, num_xcds, split_k,
# num_warps, num_stages, matrix_instr_nonkdim, waves_per_eu, kpack,
# disable_agpr). Keeping this immutable makes it a valid CustomOpConfig value.
_DEFAULT_PLAN = (
    "lds",
    192,
    256,
    64,
    4,
    8,
    4,
    8,
    1,
    16,
    0,
    1,
    True,
)

_LDS_PLANS = _make_lds_plans((
    (128, 128, 1),
    (128, 256, 2),
    (192, 256, 2),
    (192, 256, 4),
    (256, 128, 2),
    (256, 256, 2),
    (256, 256, 4),
    (256, 256, 5),
))

_FOCUSED_PLANS = (
    ("register", 128, 64, 64, 4, 8, 1, 4, 3, 16, 0, 1, False),
    ("register", 128, 128, 128, 16, 1, 1, 8, 2, 16, 0, 1, False),
    ("lds", 256, 256, 64, 4, 8, 4, 8, 1, 16, 0, 1, True),
    ("lds", 256, 256, 64, 4, 8, 5, 8, 1, 16, 0, 1, True),
)


def _fused_gfx950_addmm_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    *,
    gemm_plan: tuple[object, ...] = _DEFAULT_PLAN,
    norm_impl: str = _SPLIT_STATS,
    norm_num_warps: int = 8,
    apply_block_n: int = _NORM_BLOCK_N,
    stats_block_m: int = _SPLIT_REDUCE_BLOCK_M,
    stats_block_n: int = _SPLIT_REDUCE_BLOCK_N,
    stats_num_warps: int = 4,
) -> torch.Tensor:
    return _launch_gfx950_addmm_norm(
        x,
        weight,
        gemm_bias,
        scale,
        scale,
        eps,
        is_rms_norm=True,
        gemm_plan=gemm_plan,
        norm_impl=norm_impl,
        norm_num_warps=norm_num_warps,
        apply_block_n=apply_block_n,
        stats_block_m=stats_block_m,
        stats_block_n=stats_block_n,
        stats_num_warps=stats_num_warps,
    )


def _aten_gfx950_addmm_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    value = torch.addmm(gemm_bias, x, weight)
    return torch.nn.functional.rms_norm(value, (value.shape[-1], ), scale, eps)


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
def _fake_gfx950_addmm_rmsnorm(
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


def _eligible_rmsnorm(match: Match) -> bool:
    return _eligible(match, _SHAPE)


@functools.cache
def register_gemm_rmsnorm_pattern() -> None:
    from torch._inductor.fx_passes.post_grad import pass_patterns

    _register_autotuned_region(
        gfx950_addmm_rmsnorm,
        _fused_gfx950_addmm_rmsnorm,
        _aten_gfx950_addmm_rmsnorm,
        "tlx_gfx950_addmm_rmsnorm",
        lds_plans=_LDS_PLANS,
        focused_plans=_FOCUSED_PLANS,
        is_rms_norm=True,
    )

    n = _SHAPE[2]
    example_inputs = (
        torch.empty((2, 64), dtype=torch.bfloat16),
        torch.empty((64, n), dtype=torch.bfloat16),
        torch.empty((n, ), dtype=torch.bfloat16),
        torch.empty((n, ), dtype=torch.bfloat16),
    )

    def pattern(x, weight, gemm_bias, scale):
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
        inverse_std = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Scalar(mean_square, 1.0e-5))
        normalized = torch.ops.aten.mul.Tensor(value_fp32, inverse_std)
        scaled = torch.ops.aten.mul.Tensor(normalized, scale)
        return torch.ops.prims.convert_element_type.default(scaled, torch.bfloat16)

    def replacement(x, weight, gemm_bias, scale):
        return gfx950_addmm_rmsnorm(
            x,
            weight,
            gemm_bias,
            scale,
            1.0e-5,
        )

    register_replacement(
        pattern,
        replacement,
        example_inputs,
        fwd_only,
        pass_patterns[0],
        extra_check=_eligible_rmsnorm,
        pattern_name="tlx_gfx950_addmm_rmsnorm",
    )
