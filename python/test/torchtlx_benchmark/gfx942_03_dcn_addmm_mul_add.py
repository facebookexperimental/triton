# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""DCNv2 cross-layer up-projection with ``x0 * y + x`` epilogue."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx942_03_dcn_addmm_mul_add"
CANDIDATE_NAME = "dcn_addmm_mul_add"
_COMMON_CONFIG: dict[str, object] = {"force_disable_caches": True}
BASELINE_CONFIG: dict[str, object] = {**_COMMON_CONFIG, "triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {**_COMMON_CONFIG, "triton.tlx_mode": "allow"}

# Production ranking-model merge-net over-arch DCN up-projections at the median batch.
SHAPES = (
    {"m": 4662, "k": 512, "n": 18688, "dtype": "fp16"},
    {"m": 4662, "k": 256, "n": 18688, "dtype": "fp16"},
)
M = SHAPES[0]["m"]
K = SHAPES[0]["k"]
N = SHAPES[0]["n"]


def add_arguments(parser) -> None:
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)


def configure(args) -> None:
    global M, K, N
    M, K, N = args.m, args.k, args.n


def problem() -> str:
    return f"M={M} K={K} N={N} (x0 * addmm + x epilogue) dtype=fp16"


def model(
    h: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x0: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    return x0 * torch.addmm(bias, h, weight.t()) + x


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    fp16 = {"device": "cuda", "dtype": torch.float16}
    return (
        torch.randn((M, K), **fp16),
        torch.randn((N, K), **fp16) / K**0.5,
        torch.randn((N, ), **fp16),
        torch.randn((M, N), **fp16),
        torch.randn((M, N), **fp16),
    )
