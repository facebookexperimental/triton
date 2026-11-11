# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""FFN down-projection dgrad with GELU-backward epilogue."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx950_10_mm_gelu_bwd"
CANDIDATE_NAME = "mm_gelu_bwd"
BASELINE_CONFIG: dict[str, object] = {"triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {"triton.tlx_mode": "allow"}
GRAD_INPUTS = (0, 1, 2)

K = 256
N = 512
SHAPES = (
    {"m": 4202342, "k": 256, "n": 512, "dtype": "bf16"},
    {"m": 1319440, "k": 256, "n": 512, "dtype": "bf16"},
)
M = SHAPES[0]["m"]


def add_arguments(parser) -> None:
    parser.add_argument("--m", type=int)


def configure(args) -> None:
    global M
    M = args.m


def problem() -> str:
    return f"M={M} K={K} N={N} (gelu backward epilogue) dtype=bf16"


def model(grad_out: torch.Tensor, weight: torch.Tensor, pre_act: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.gelu_backward(torch.mm(grad_out, weight), pre_act)


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    return (
        torch.randn((M, K), **bf16),
        torch.randn((K, N), **bf16) / 16,
        torch.randn((M, N), **bf16),
    )
