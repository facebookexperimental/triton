# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "sm100_02_matmul_sigmoid_mul"
CANDIDATE_NAME = "torchtlx_fused"
BASELINE_CONFIG: dict[str, object] = {"triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {"triton.tlx_mode": "force"}
ATOL = 2.0e-2
RTOL = 2.0e-2

# Production shape reported for the Blackwell GEO kernel.
M = 3836160
K = 256
N = 256
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


def model(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sigmoid = torch.sigmoid(x @ weight)
    return 2.0 * x * sigmoid, sigmoid


def make_inputs() -> tuple[torch.Tensor, ...]:
    if K != N:
        raise ValueError("matmul_sigmoid_mul requires K == N so x can multiply the projection")
    torch.manual_seed(0)
    dtype = torch.float16 if DTYPE == "fp16" else torch.bfloat16
    x = torch.randn((M, K), device="cuda", dtype=dtype)
    weight = torch.randn((K, N), device="cuda", dtype=dtype)
    return x, weight
