# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Weight-grad GEMM over the token dim plus the bias-grad column sum of the same gradient."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx950_09_mm_bias_grad"
CANDIDATE_NAME = "mm_bias_grad"
BASELINE_CONFIG: dict[str, object] = {"triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {"triton.tlx_mode": "allow"}
GRAD_INPUTS = (0, 1)

D = 256
SHAPES = (
    {"m": 4202342, "n": 256, "dtype": "bf16"},
    {"m": 1319440, "n": 256, "dtype": "bf16"},
)
M = SHAPES[0]["m"]


def add_arguments(parser) -> None:
    parser.add_argument("--m", type=int)


def configure(args) -> None:
    global M
    M = args.m


def problem() -> str:
    return f"M={D} N={D} K={M} (+ bias-grad sum over K) dtype=bf16"


def model(grad_out: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.mm(grad_out.t(), x), grad_out.sum(0)


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    return (
        torch.randn((M, D), **bf16) / 16,
        torch.randn((M, D), **bf16) / 16,
    )
