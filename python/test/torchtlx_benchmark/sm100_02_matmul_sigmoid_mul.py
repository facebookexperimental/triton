# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "sm100_02_matmul_sigmoid_mul"
CANDIDATE_NAME = "matmul_sigmoid_mul"
CANDIDATE_CODE_MARKERS = ("matmul_sigmoid_mul_kernel", )
CANDIDATE_FORBIDDEN_CODE_MARKERS = ("torch.ops.torch_tlx.sm100_02_fused_kernel.default", )
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
ATOL = 1.0e-2
RTOL = 1.0e-2

# Production shape reported for the Blackwell GEO kernel.
M = 3836160
K = 256
N = 256
DTYPE = "bf16"


def add_arguments(parser) -> None:
    pass


def configure(args) -> None:
    pass


def problem() -> str:
    return f"M={M} K={K} N={N} dtype={DTYPE}"


def model(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    accumulator = x.float() @ weight.float()
    sigmoid = torch.sigmoid(accumulator)
    output = (2.0 * x.float() * sigmoid).to(torch.bfloat16)
    return output, sigmoid.to(torch.bfloat16)


def make_inputs() -> tuple[torch.Tensor, ...]:
    if K != N:
        raise ValueError("matmul_sigmoid_mul requires K == N so x can multiply the projection")
    torch.manual_seed(0)
    dtype = torch.float16 if DTYPE == "fp16" else torch.bfloat16
    x = torch.randn((M, K), device="cuda", dtype=dtype)
    weight = torch.randn((K, N), device="cuda", dtype=dtype)
    return x, weight
