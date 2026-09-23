# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
import torch.nn.functional as F

NAME = "gfx950_02_gemm_layernorm"
CANDIDATE_NAME = "gemm_layernorm"
_COMMON_CONFIG: dict[str, object] = {
    "force_disable_caches": True,
    "max_autotune": True,
    "max_autotune_gemm_backends": "TRITON",
    "enable_caching_generated_triton_templates": False,
}
BASELINE_CONFIG: dict[str, object] = {
    **_COMMON_CONFIG,
    "triton.tlx_mode": None,
}
CANDIDATE_CONFIG: dict[str, object] = {
    **_COMMON_CONFIG,
    "triton.tlx_mode": "force",
}
ATOL = 3.0e-2
RTOL = 3.0e-2

M = 2032
K = 2560
N = 2560
EPS = 1.0e-5


def add_arguments(parser) -> None:
    pass


def configure(args) -> None:
    pass


def problem() -> str:
    return f"M={M} K={K} N={N} dtype=bf16"


def model(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    norm_bias: torch.Tensor,
) -> torch.Tensor:
    value = torch.addmm(gemm_bias, x, weight.t())
    return F.layer_norm(value, (N, ), scale, norm_bias, EPS)


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    return (
        torch.randn((M, K), device="cuda", dtype=torch.bfloat16),
        torch.randn((N, K), device="cuda", dtype=torch.bfloat16),
        torch.randn((N, ), device="cuda", dtype=torch.bfloat16),
        torch.randn((N, ), device="cuda", dtype=torch.bfloat16),
        torch.randn((N, ), device="cuda", dtype=torch.bfloat16),
    )
