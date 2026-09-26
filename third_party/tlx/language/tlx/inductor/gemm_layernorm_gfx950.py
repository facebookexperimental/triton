"""gfx950 addmm + LayerNorm fusion for TorchInductor."""

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

_SHAPE = (2032, 2560, 2560)

# A plan is (kind, block_m, block_n, block_k, group_m, num_xcds, split_k,
# num_warps, num_stages, matrix_instr_nonkdim, waves_per_eu, kpack,
# disable_agpr). Keeping this immutable makes it a valid CustomOpConfig value.
_DEFAULT_PLAN = (
    "lds",
    128,
    128,
    64,
    4,
    8,
    1,
    4,
    1,
    16,
    0,
    1,
    True,
)

_LDS_PLANS = _make_lds_plans((
    (128, 128, 1),
    (128, 256, 1),
    (192, 256, 1),
    (192, 256, 2),
    (256, 128, 1),
    (256, 256, 1),
    (256, 256, 2),
))

_FOCUSED_PLANS = (
    ("register", 128, 128, 64, 8, 1, 1, 8, 2, 16, 0, 1, False),
    ("register", 128, 128, 64, 8, 8, 1, 4, 2, 16, 0, 1, False),
    ("register", 128, 128, 64, 16, 8, 1, 4, 2, 16, 0, 1, False),
)


def _fused_gfx950_addmm_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    norm_bias: torch.Tensor,
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
        norm_bias,
        eps,
        is_rms_norm=False,
        gemm_plan=gemm_plan,
        norm_impl=norm_impl,
        norm_num_warps=norm_num_warps,
        apply_block_n=apply_block_n,
        stats_block_m=stats_block_m,
        stats_block_n=stats_block_n,
        stats_num_warps=stats_num_warps,
    )


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
        (value.shape[-1], ),
        scale,
        norm_bias,
        eps,
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
def _fake_gfx950_addmm_layernorm(
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


def _eligible_layernorm(match: Match) -> bool:
    return _eligible(match, _SHAPE)


@functools.cache
def register_gemm_layernorm_pattern() -> None:
    from torch._inductor.fx_passes.post_grad import pass_patterns

    _register_autotuned_region(
        gfx950_addmm_layernorm,
        _fused_gfx950_addmm_layernorm,
        _aten_gfx950_addmm_layernorm,
        "tlx_gfx950_addmm_layernorm",
        lds_plans=_LDS_PLANS,
        focused_plans=_FOCUSED_PLANS,
        is_rms_norm=False,
    )

    n = _SHAPE[2]
    example_inputs = (
        torch.empty((2, 64), dtype=torch.bfloat16),
        torch.empty((64, n), dtype=torch.bfloat16),
        torch.empty((n, ), dtype=torch.bfloat16),
        torch.empty((n, ), dtype=torch.bfloat16),
        torch.empty((n, ), dtype=torch.bfloat16),
    )

    def pattern(x, weight, gemm_bias, scale, norm_bias):
        value = torch.addmm(gemm_bias, x, weight)
        return torch.nn.functional.layer_norm(
            value,
            (value.shape[-1], ),
            scale,
            norm_bias,
            1.0e-5,
        )

    def replacement(x, weight, gemm_bias, scale, norm_bias):
        return gfx950_addmm_layernorm(
            x,
            weight,
            gemm_bias,
            scale,
            norm_bias,
            1.0e-5,
        )

    register_replacement(
        pattern,
        replacement,
        example_inputs,
        fwd_only,
        pass_patterns[0],
        extra_check=_eligible_layernorm,
        pattern_name="tlx_gfx950_addmm_layernorm",
    )
