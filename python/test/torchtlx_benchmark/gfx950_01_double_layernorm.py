# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx950_01_double_layernorm"
CANDIDATE_NAME = "local_buffer_retention"
BASELINE_CONFIG: dict[str, object] = {"triton.tlx_mode": None}
# Retention is offered as a MultiKernel choice; the default value 0 would
# compile the candidate into the baseline and make the A/B meaningless.
CANDIDATE_CONFIG: dict[str, object] = {
    "triton.tlx_mode": "allow",
    "triton.multi_kernel": 1,
}
ATOL = 2.0e-2
RTOL = 2.0e-2

M = 1024
N = 6144
EPS = 1.0e-5


def add_arguments(parser) -> None:
    parser.add_argument("--n", type=int, default=N)


def configure(args) -> None:
    global N
    N = args.n


def problem() -> str:
    return f"M={M} N={N} dtype=fp16"


def model(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    z = residual + torch.nn.functional.layer_norm(x, (N, ), weight1, bias1, EPS)
    return z * torch.sigmoid(torch.nn.functional.layer_norm(z, (N, ), weight2, bias2, EPS))


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)
    residual = torch.randn((M, N), device="cuda", dtype=torch.float16)
    parameters = tuple(torch.randn((N, ), device="cuda", dtype=torch.float16) for _ in range(4))
    return x, residual, *parameters
