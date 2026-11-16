# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""FFN down-projection with residual-add epilogue."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx950_07_addmm_residual_add"
CANDIDATE_NAME = "addmm_residual_add"
BASELINE_CONFIG: dict[str, object] = {"triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {"triton.tlx_mode": "allow"}
GRAD_INPUTS = (0, 1, 2, 3)

K = 512
N = 256
SHAPES = (
    {"m": 4202342, "k": 512, "n": 256, "dtype": "bf16"},
    {"m": 1319440, "k": 512, "n": 256, "dtype": "bf16"},
)
M = SHAPES[0]["m"]


def add_arguments(parser) -> None:
    parser.add_argument("--m", type=int)


def configure(args) -> None:
    global M
    M = args.m


def problem() -> str:
    return f"M={M} K={K} N={N} (residual add epilogue) dtype=bf16"


def model(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return torch.addmm(bias, x, weight.t()) + residual


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    return (
        torch.randn((M, K), **bf16),
        torch.randn((N, K), **bf16) / 16,
        torch.randn((N, ), **bf16),
        torch.randn((M, N), **bf16),
    )
