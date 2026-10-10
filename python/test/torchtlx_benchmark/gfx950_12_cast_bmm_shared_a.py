# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Learned-query pooling: shared-A baddbmm over a per-sample sequence with an fp32->bf16 cast prologue."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx950_12_cast_bmm_shared_a"
CANDIDATE_NAME = "cast_bmm_shared_a"
BASELINE_CONFIG: dict[str, object] = {"triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {"triton.tlx_mode": "allow"}
GRAD_INPUTS = (0, 1, 2)

Q = 10
L = 1988
D = 256
SHAPES = ({"b": 1024, "q": 10, "l": 1988, "d": 256, "dtype": "bf16"}, )
M = SHAPES[0]["b"]


def add_arguments(parser) -> None:
    parser.add_argument("--b", type=int)


def configure(args) -> None:
    global M
    M = args.b


def problem() -> str:
    return f"B={M} M={Q} N={D} K={L} shared A (fp32 cast prologue) dtype=bf16"


def model(bias: torch.Tensor, query: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    return torch.baddbmm(bias.expand(M, Q, D), query.expand(M, Q, L), seq.to(torch.bfloat16))


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    return (
        torch.randn((Q, 1), device="cuda", dtype=torch.bfloat16),
        torch.randn((Q, L), device="cuda", dtype=torch.bfloat16) / 32,
        torch.randn((M, L, D), device="cuda", dtype=torch.float32),
    )
