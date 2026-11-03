# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""MLP layer: addmm, optional residual add, then zero-centered RMSNorm and swish."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx942_04_addmm_swish_norm"
CANDIDATE_NAME = "addmm_swish_norm"
_COMMON_CONFIG: dict[str, object] = {"force_disable_caches": True}
BASELINE_CONFIG: dict[str, object] = {**_COMMON_CONFIG, "triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {**_COMMON_CONFIG, "triton.tlx_mode": "allow"}

# Production ranking-model MLP layers at the median batch.
SHAPES = (
    {"m": 4662, "k": 2560, "n": 2560, "residual": True, "dtype": "fp16"},
    {"m": 4662, "k": 2560, "n": 2560, "residual": False, "dtype": "fp16"},
    {"m": 4662, "k": 18688, "n": 2560, "residual": False, "dtype": "fp16"},
)
M = SHAPES[0]["m"]
K = SHAPES[0]["k"]
N = SHAPES[0]["n"]
RESIDUAL = SHAPES[0]["residual"]
EPS = 1.0e-5


def add_arguments(parser) -> None:
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)


def configure(args) -> None:
    global M, K, N, RESIDUAL
    M, K, N, RESIDUAL = args.m, args.k, args.n, args.residual


def problem() -> str:
    return f"M={M} K={K} N={N} residual={RESIDUAL} dtype=fp16"


def model(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    value = torch.addmm(bias, x, weight.t())
    if RESIDUAL:
        value = residual + value
    dtype = value.dtype
    value = value.to(torch.float32)
    value = value - value.mean(dim=-1, keepdim=True)
    value = value / torch.sqrt(torch.square(value).mean(dim=-1, keepdim=True) + EPS)
    value = (value * scale.to(torch.float32)).to(dtype)
    return value * torch.sigmoid(value)


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    fp16 = {"device": "cuda", "dtype": torch.float16}
    return (
        torch.randn((M, K), **fp16),
        torch.randn((N, K), **fp16) / K**0.5,
        torch.randn((N, ), **fp16),
        torch.randn((M, N), **fp16),
        torch.ones((N, ), **fp16),
    )
