from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class MoeShape(NamedTuple):
    """One expert-parallel rank's slice of a routed MoE layer: ``tokens`` routed over ``num_experts`` experts, of
    which ``local_experts`` live on this rank starting at ``expert_start``."""
    tokens: int
    K: int
    I: int
    H: int
    local_experts: int
    num_experts: int
    topk: int
    expert_start: int
    dtype: str


SYNTHETIC: tuple[MoeShape,
                 ...] = tuple(MoeShape(tokens, 256, 256, 384, 8, 32, 4, 8, "bf16") for tokens in (1, 64, 4090))


def _muse_spark_open(tokens: int) -> MoeShape:
    # Muse-Spark-Open-816B-A42B at TP8/EP8: moe_hidden_size = moe_intermediate_size = 4096, 256 experts, top-8; rank 3.
    return MoeShape(tokens, 4096, 4096, 4096, 32, 256, 8, 96, "bf16")


# Decode, including the DFLASH verify batch of 8 draft tokens + 1.
GFX950_1 = FocusSuite(name="gfx950_1", op="moe", shapes=tuple(_muse_spark_open(t) for t in (4, 8, 9, 12, 16, 18)))
# Prefill: radix-cached extends through long prompts, spanning the register and quadrant engines.
GFX950_2 = FocusSuite(
    name="gfx950_2",
    op="moe",
    shapes=tuple(_muse_spark_open(t) for t in (27, 64, 151, 512, 1024, 1536, 1920, 3840, 4096, 8192, 16384)),
)
GFX950_ALL = FocusSuite(name="gfx950_all", op="moe", includes=("gfx950_1", "gfx950_2"))

FOCUS_SUITES = (GFX950_1, GFX950_2, GFX950_ALL)
DEFAULT_SUITES = {"gfx950": ("gfx950_all", )}
FOCUS = FocusRegistry("moe", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def label(shape: MoeShape) -> str:
    return (f"(({shape.tokens}, {shape.K}), {{'dtype': '{shape.dtype}', 'I': '{shape.I}', 'H': '{shape.H}', "
            f"'E': '{shape.local_experts}/{shape.num_experts}', 'topk': '{shape.topk}'}})")


def flops(local_rows: int, K: int, I: int, H: int) -> int:
    """Gate+up and down GEMMs over the rows routed to this rank's experts."""
    return 2 * local_rows * (2 * I * K + H * I)


def inputs(shape: MoeShape, seed: int = 0, device="cuda"):
    """``x``, ``topk_weights``, ``topk_ids``, ``w13``, ``w2``, ``post_norm_weight``; each token routes to ``topk``
    distinct experts drawn uniformly from all ``num_experts``, with normalized weights."""
    import torch

    dtype = {"bf16": torch.bfloat16}[shape.dtype]
    gen = torch.Generator(device=device).manual_seed(seed)
    E, K, I, H = shape.local_experts, shape.K, shape.I, shape.H
    w13 = (torch.randn(E, 2 * I, K, device=device, generator=gen) * 0.02).to(dtype)
    w2 = (torch.randn(E, H, I, device=device, generator=gen) * 0.02).to(dtype)
    norm = (torch.randn(H, device=device, generator=gen) * 0.1).to(dtype)
    x = torch.randn(shape.tokens, K, device=device, generator=gen).to(dtype)
    ids = torch.rand(shape.tokens, shape.num_experts, device=device, generator=gen).argsort(dim=1)[:, :shape.topk]
    weights = torch.rand(shape.tokens, shape.topk, device=device, generator=gen)
    return x, weights / weights.sum(dim=1, keepdim=True), ids.to(torch.int32), w13, w2, norm
