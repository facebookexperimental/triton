# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "sm100_01_matmul_addcmul"
CANDIDATE_NAME = "torchtlx_fused"
BASELINE_CONFIG: dict[str, object] = {"triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {"triton.tlx_mode": "force"}
ATOL = 2.0e-2
RTOL = 2.0e-2

# Production shape reported for the Blackwell GEO kernel.
M = 1152
K = 1024
N = 12800
DTYPE = "bf16"


def add_arguments(parser) -> None:
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default=DTYPE)


def configure(args) -> None:
    global M, K, N, DTYPE
    M, K, N, DTYPE = args.m, args.k, args.n, args.dtype


def problem() -> str:
    return f"M={M} K={K} N={N} dtype={DTYPE}"


def model(
    s: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x0: torch.Tensor,
    layer_input: torch.Tensor,
) -> torch.Tensor:
    projection = s @ weight.T + bias
    return torch.addcmul(layer_input, x0, projection)


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    dtype = torch.float16 if DTYPE == "fp16" else torch.bfloat16
    s = torch.randn((M, K), device="cuda", dtype=dtype)
    weight = torch.randn((N, K), device="cuda", dtype=dtype)
    bias = torch.randn((N, ), device="cuda", dtype=dtype)
    x0 = torch.randn((M, N), device="cuda", dtype=dtype)
    layer_input = torch.randn((M, N), device="cuda", dtype=dtype)
    return s, weight, bias, x0, layer_input
