# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""LayerNorm(x + pos_emb[pos]) prologue feeding three QKV projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "gfx950_05_layernorm_qkv_addmm"
CANDIDATE_NAME = "layernorm_qkv_addmm"
BASELINE_CONFIG: dict[str, object] = {"triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {"triton.tlx_mode": "allow"}
GRAD_INPUTS = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10)

D = 256
MAX_POS = 5000
EPS = 1.0e-5
SHAPES = (
    {"m": 4202342, "k": 256, "n": 256, "dtype": "bf16"},
    {"m": 1319440, "k": 256, "n": 256, "dtype": "bf16"},
)
M = SHAPES[0]["m"]


def add_arguments(parser) -> None:
    parser.add_argument("--m", type=int)


def configure(args) -> None:
    global M
    M = args.m


def problem() -> str:
    return f"M={M} K={D} N={D} x3 (layernorm prologue) dtype=bf16"


def model(
    x: torch.Tensor,
    pos: torch.Tensor,
    pos_emb: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    w_q: torch.Tensor,
    b_q: torch.Tensor,
    w_k: torch.Tensor,
    b_k: torch.Tensor,
    w_v: torch.Tensor,
    b_v: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    y = torch.nn.functional.layer_norm(x + pos_emb[pos], (D, ), ln_weight, ln_bias, EPS)
    return tuple(torch.addmm(b, y, w.t()) for w, b in ((w_q, b_q), (w_k, b_k), (w_v, b_v)))


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    return (
        torch.randn((M, D), **bf16),
        torch.randint(0, MAX_POS, (M, ), device="cuda"),
        torch.randn((MAX_POS, D), **bf16),
        torch.randn((D, ), **bf16),
        torch.randn((D, ), **bf16),
        *(t for _ in range(3) for t in (torch.randn((D, D), **bf16) / 16, torch.randn((D, ), **bf16))),
    )
