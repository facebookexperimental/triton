from __future__ import annotations

import math
from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class RoutingShape(NamedTuple):
    """``tokens`` rows of fp32 router logits over ``experts`` experts, routed to ``topk`` of them."""
    tokens: int
    experts: int
    topk: int


SYNTHETIC: tuple[RoutingShape, ...] = tuple(RoutingShape(tokens, 64, 4) for tokens in (1, 300, 5000))

# Router of a 256-expert top-8 MoE whose scores scale the logits by 1/sqrt(32) (the router_multiplier below).
ROUTER_MULTIPLIER = 1.0 / math.sqrt(32.0)
ROUTE_EPS = 1e-15

# Decode, including a verify batch of 8 draft tokens + 1.
GFX950_1 = FocusSuite(name="gfx950_1", op="moe_routing",
                      shapes=tuple(RoutingShape(t, 256, 8) for t in (1, 4, 8, 9, 12, 16, 18)))
# Prefill: chunked extends through long prompts, across the one-row and four-rows-per-wave layouts.
GFX950_2 = FocusSuite(
    name="gfx950_2",
    op="moe_routing",
    shapes=tuple(RoutingShape(t, 256, 8) for t in (27, 64, 151, 512, 1024, 1536, 1920, 3072, 3840, 4096, 8192, 16384)),
)
GFX950_ALL = FocusSuite(name="gfx950_all", op="moe_routing", includes=("gfx950_1", "gfx950_2"))

FOCUS_SUITES = (GFX950_1, GFX950_2, GFX950_ALL)
DEFAULT_SUITES = {"gfx950": ("gfx950_all", )}
FOCUS = FocusRegistry("moe_routing", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def label(shape: RoutingShape) -> str:
    return f"(({shape.tokens}, {shape.experts}), {{'dtype': 'fp32', 'topk': '{shape.topk}'}})"


def inputs(shape: RoutingShape, seed: int = 0, device="cuda"):
    """``router_logits`` and ``correction_bias`` at the spread of a trained router."""
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(shape.tokens, shape.experts, device=device, generator=gen) * 4.0
    bias = torch.randn(shape.experts, device=device, generator=gen) * 0.05
    return logits, bias


def reference(logits, bias, topk, router_multiplier=1.0, route_scale=1.0, eps=0.0):
    """Eager torch routing with the op's selection order and weight rounding."""
    import torch

    scores = torch.sigmoid(logits.float() * router_multiplier)
    biased = scores + bias.float()
    # A stable descending sort picks the lower id among equal values.
    ids = torch.sort(biased, dim=1, descending=True, stable=True).indices[:, :topk]
    picked = scores.gather(1, ids)
    total = picked[:, 0].clone()
    for k in range(1, topk):
        total = total + picked[:, k]
    weights = picked * (route_scale / (total + eps))[:, None]
    return weights, ids.to(torch.int32)
