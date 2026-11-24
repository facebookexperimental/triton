# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Learned-query pooling backward: shared-A bmm (K=10) with a bf16->fp32 cast epilogue."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx950_13_bmm_cast_bwd"
CANDIDATE_NAME = "bmm_cast_bwd"
BASELINE_CONFIG: dict[str, object] = {"triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {"triton.tlx_mode": "allow"}
GRAD_INPUTS = (0, 1)

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
    return f"B={M} M={L} N={D} K={Q} shared A (fp32 cast epilogue) dtype=bf16"


def model(query: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    return torch.bmm(query.t().expand(M, L, Q), grad_out).float()


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    return (
        torch.randn((Q, L), device="cuda", dtype=torch.bfloat16),
        torch.randn((M, Q, D), device="cuda", dtype=torch.bfloat16),
    )
