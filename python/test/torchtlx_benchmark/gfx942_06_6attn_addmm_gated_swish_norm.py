# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""6-attn MoE expert MLP layer: addmm, optional residual add, then gated SwishWithNorm."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx942_06_6attn_addmm_gated_swish_norm"
CANDIDATE_NAME = "addmm_gated_swish_norm"
_COMMON_CONFIG: dict[str, object] = {"force_disable_caches": True}
BASELINE_CONFIG: dict[str, object] = {**_COMMON_CONFIG, "triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {**_COMMON_CONFIG, "triton.tlx_mode": "allow"}

# Expert x_ff fc0-fc2 at the median batch (VG 28764920146482456).
SHAPES = (
    {"m": 440, "k": 6400, "n": 4096, "residual": False, "dtype": "fp16"},
    {"m": 440, "k": 4096, "n": 2048, "residual": False, "dtype": "fp16"},
    {"m": 440, "k": 2048, "n": 4096, "residual": True, "dtype": "fp16"},
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
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
) -> torch.Tensor:
    value = torch.nn.functional.linear(x, weight, bias)
    if RESIDUAL:
        value = residual + value
    dtype = value.dtype
    # AOTI re-lowering clips the centered value to the fp16 range.
    value = value.to(torch.float32)
    centered = torch.clip(value - value.mean(dim=[-1], keepdim=True), -65504.0, 65504.0)
    normed = torch.nn.functional.rms_norm(centered, [N], weight=None, eps=EPS)
    normed = normed * norm_weight.to(torch.float32) + norm_bias.to(torch.float32)
    return (value * torch.sigmoid(normed)).to(dtype)


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    fp16 = {"device": "cuda", "dtype": torch.float16}
    return (
        torch.randn((M, K), **fp16),
        torch.randn((N, K), **fp16) / K**0.5,
        torch.randn((N, ), **fp16),
        torch.randn((M, N), **fp16),
        torch.ones((N, ), **fp16),
        torch.zeros((N, ), **fp16),
    )
