# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Projection dgrad feeding the LayerNorm backward row reduction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx950_08_mm_layernorm_bwd"
CANDIDATE_NAME = "mm_layernorm_bwd"
BASELINE_CONFIG: dict[str, object] = {"triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {"triton.tlx_mode": "allow"}
GRAD_INPUTS = (0, 1, 2, 3, 4, 5)

D = 256
SHAPES = (
    {"m": 4202342, "k": 256, "n": 256, "dtype": "bf16"},
    {"m": 1319440, "k": 256, "n": 256, "dtype": "bf16"},
)
M = SHAPES[0]["m"]


def add_arguments(parser) -> None:
    parser.add_argument("--m", type=int)


def configure(args) -> None:
    global M
    M = args.m


def problem() -> str:
    return f"M={M} K={D} N={D} (layernorm backward epilogue) dtype=bf16"


def model(
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    grad_residual: torch.Tensor,
    x_hat: torch.Tensor,
    rstd: torch.Tensor,
    ln_weight: torch.Tensor,
) -> torch.Tensor:
    g = (torch.mm(grad_out, weight) + grad_residual).float() * ln_weight.float()
    x_hat = x_hat.float()
    dx = rstd * (g - g.mean(-1, keepdim=True) - x_hat * (g * x_hat).mean(-1, keepdim=True))
    return dx.to(torch.bfloat16)


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    return (
        torch.randn((M, D), **bf16),
        torch.randn((D, D), **bf16) / 16,
        torch.randn((M, D), **bf16),
        torch.randn((M, D), **bf16),
        torch.rand((M, 1), device="cuda", dtype=torch.float32) + 0.5,
        torch.randn((D, ), **bf16),
    )
