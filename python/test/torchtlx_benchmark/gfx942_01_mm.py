# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx942_01_mm"
CANDIDATE_NAME = "tlx_gfx942_mm"
CANDIDATE_CODE_MARKERS = ("tlx_gfx942_mm", )
BASELINE_CONFIG: dict[str, object] = {"triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {
    "triton.tlx_mode": "force",
    "max_autotune": True,
    "max_autotune_gemm_backends": "TRITON",
}
ATOL = 2.0e-2
RTOL = 2.0e-2

M = 4096
N = 4096
K = 4096
SHAPES = ({"m": M, "n": N, "k": K, "dtype": "fp16"}, )


def add_arguments(parser) -> None:
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)


def configure(args) -> None:
    global M, N, K
    M = args.m if args.m is not None else M
    N = args.n if args.n is not None else N
    K = args.k if args.k is not None else K


def problem() -> str:
    return f"M={M} N={N} K={K} dtype=fp16"


def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    return (a, b)
