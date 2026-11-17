from __future__ import annotations

import math
from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class SwigluFfnShape(NamedTuple):
    tokens: int
    d_model: int
    dtype: str


SYNTHETIC: tuple[SwigluFfnShape, ...] = (
    SwigluFfnShape(1000, 128, "fp16"),
    SwigluFfnShape(1000, 256, "fp16"),
)

# The feed-forward of a production ranking model's attention block: the smallest, quartile and largest token counts of
# the 88 calls in its trace.
SM90_1 = FocusSuite(
    name="sm90_1",
    op="swiglu_ffn",
    shapes=tuple(SwigluFfnShape(tokens, 256, "fp16") for tokens in (20480, 32304, 35988, 40960, 55800)),
)

FOCUS_SUITES = (SM90_1, )
DEFAULT_SUITES = {"sm90": ("sm90_1", )}
FOCUS = FocusRegistry("swiglu_ffn", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def label(tokens: int, d_model: int, dtype: str) -> str:
    return f"(({tokens}, {d_model}), {{'dtype': '{dtype}'}})"


def flops(tokens: int, d_model: int) -> int:
    """The three (tokens, d_model) x (d_model, d_model) products."""
    return 6 * tokens * d_model * d_model


def inputs(tokens: int, d_model: int, dtype, device="cuda"):
    """`x`, `norm_weight`, `w1`, `w2`, `w3`. The weights are scaled so the feed-forward branch is of order 1: with
    default initializations it would be small next to the residual, and a wrong branch would still compare close."""
    import torch

    gen = torch.Generator(device=device).manual_seed(0)
    bound = 1.6 / math.sqrt(d_model)
    w1, w2, w3 = (((torch.rand(d_model, d_model, generator=gen, device=device) * 2 - 1) * bound).to(dtype)
                  for _ in range(3))
    norm_weight = (0.5 + torch.rand(d_model, generator=gen, device=device)).to(dtype)
    x = torch.randn(tokens, d_model, generator=gen, device=device).to(dtype)
    return x, norm_weight, w1, w2, w3


def reference(x, norm_weight, w1, w2, w3, eps=1e-5):
    """The eager module: RMSNorm in FP32 rounded to the input dtype, then the SwiGLU feed-forward, with residual."""
    import torch

    xf = x.float()
    n = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * norm_weight.float()).to(x.dtype)
    return x + (torch.nn.functional.silu(n @ w1.t()) * (n @ w3.t())) @ w2.t()
