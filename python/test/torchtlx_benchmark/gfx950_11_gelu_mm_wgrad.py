# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""FFN down-projection weight grad with the GELU forward recomputed as a B-operand prologue."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx950_11_gelu_mm_wgrad"
CANDIDATE_NAME = "gelu_mm_wgrad"
BASELINE_CONFIG: dict[str, object] = {"triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {"triton.tlx_mode": "allow"}
GRAD_INPUTS = (0, 1)

D_OUT = 256
D_IN = 512
SHAPES = (
    {"m": 4202342, "n_out": 256, "n_in": 512, "dtype": "bf16"},
    {"m": 1319440, "n_out": 256, "n_in": 512, "dtype": "bf16"},
)
M = SHAPES[0]["m"]


def add_arguments(parser) -> None:
    parser.add_argument("--m", type=int)


def configure(args) -> None:
    global M
    M = args.m


def problem() -> str:
    return f"M={D_OUT} N={D_IN} K={M} (gelu prologue) dtype=bf16"


def model(grad_out: torch.Tensor, pre_act: torch.Tensor) -> torch.Tensor:
    return torch.mm(grad_out.t(), torch.nn.functional.gelu(pre_act))


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    return (
        torch.randn((M, D_OUT), **bf16) / 16,
        torch.randn((M, D_IN), **bf16),
    )
