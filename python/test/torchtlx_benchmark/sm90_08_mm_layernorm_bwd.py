# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Projection dgrad plus residual feeding the LayerNorm backward row reduction (dx only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "sm90_08_mm_layernorm_bwd"
CANDIDATE_NAME = "mm_layernorm_bwd"
_COMMON_CONFIG: dict[str, object] = {"force_disable_caches": True}
BASELINE_CONFIG: dict[str, object] = {**_COMMON_CONFIG, "triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {**_COMMON_CONFIG, "triton.tlx_mode": "allow"}

# A production H100 training model: per-sequence LayerNorms over [1536, L, 128]
# whose gradient carries a 128-wide projection dgrad (L = 200 and 1000).
SHAPES = (
    {"m": 307200, "k": 128, "n": 128, "dtype": "bf16"},
    {"m": 1536000, "k": 128, "n": 128, "dtype": "bf16"},
)
M = SHAPES[0]["m"]
K = 128
N = 128


def add_arguments(parser) -> None:
    for name in ("m", "k", "n"):
        parser.add_argument(f"--{name}", type=int)


def configure(args) -> None:
    global M, K, N
    M, K, N = args.m, args.k, args.n


def problem() -> str:
    return f"M={M} K={K} N={N} (LayerNorm backward dx epilogue) dtype=bf16"


def model(
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    grad_residual: torch.Tensor,
    x_hat: torch.Tensor,
    rstd: torch.Tensor,
    ln_weight: torch.Tensor,
) -> torch.Tensor:
    import torch

    g = (torch.mm(grad_out, weight) + grad_residual).float() * ln_weight.float()
    x_hat = x_hat.float()
    dx = rstd * (g - g.mean(-1, keepdim=True) - x_hat * (g * x_hat).mean(-1, keepdim=True))
    return dx.to(torch.bfloat16)


def make_inputs() -> tuple[torch.Tensor, ...]:
    import torch

    torch.manual_seed(0)
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    return (
        torch.randn((M, K), **bf16),
        torch.randn((K, N), **bf16) / K**0.5,
        torch.randn((M, N), **bf16),
        torch.randn((M, N), **bf16),
        torch.rand((M, 1), device="cuda", dtype=torch.float32) + 0.5,
        torch.randn((N, ), **bf16),
    )
