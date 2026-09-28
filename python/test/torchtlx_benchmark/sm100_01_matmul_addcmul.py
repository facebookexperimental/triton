# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "sm100_01_matmul_addcmul"
CANDIDATE_NAME = "matmul_addcmul"
CANDIDATE_CODE_MARKERS = ("matmul_addcmul_kernel", )
CANDIDATE_FORBIDDEN_CODE_MARKERS = ("torch.ops.torch_tlx.sm100_01_fused_kernel.default", )
_COMMON_CONFIG: dict[str, object] = {
    "force_disable_caches": True,
    "max_autotune": True,
    "max_autotune_gemm_backends": "ATEN,TRITON",
    "enable_caching_generated_triton_templates": False,
}
BASELINE_CONFIG: dict[str, object] = {
    **_COMMON_CONFIG,
    "triton.tlx_mode": None,
}
CANDIDATE_CONFIG: dict[str, object] = {
    **_COMMON_CONFIG,
    "triton.tlx_mode": "allow",
}
ATOL = 5.0e-2
RTOL = 5.0e-2

# Production shape reported for the Blackwell GEO kernel.
M = 1152
K = 1024
N = 12800
DTYPE = "bf16"


def add_arguments(parser) -> None:
    pass


def configure(args) -> None:
    pass


def problem() -> str:
    return f"M={M} K={K} N={N} dtype={DTYPE}"


def model(
    s: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x0: torch.Tensor,
    layer_input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    accumulator = s.float() @ weight.float().T
    projection_fp32 = accumulator + bias.float()
    projection = projection_fp32.to(torch.bfloat16)
    output = (layer_input.float() + x0.float() * projection_fp32).to(torch.bfloat16)
    return output, projection


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    dtype = torch.float16 if DTYPE == "fp16" else torch.bfloat16
    operand_scale = K**-0.25
    s = torch.randn((M, K), device="cuda", dtype=dtype) * operand_scale
    weight = torch.randn((N, K), device="cuda", dtype=dtype) * operand_scale
    bias = torch.randn((N, ), device="cuda", dtype=dtype) * 0.1
    x0 = torch.randn((M, N), device="cuda", dtype=dtype) * 0.1
    layer_input = torch.randn((M, N), device="cuda", dtype=dtype) * 0.1
    return s, weight, bias, x0, layer_input
