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
SYNTHETIC_FP8 = tuple(MoeShape(t, 256, 256, 256, 4, 4, 2, 0, "fp8") for t in (17, 128, 513))
FP8_SCALES = (0.5, 0.25, 0.0625, 0.125)


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

GFX1250_1 = FocusSuite(name="gfx1250_1", op="moe",
                       shapes=tuple(MoeShape(t, 256, 256, 256, 4, 4, 2, 0, "fp8") for t in (128, 512, 2048, 8192)))

FOCUS_SUITES = (GFX950_1, GFX950_2, GFX950_ALL, GFX1250_1)
DEFAULT_SUITES = {"gfx950": ("gfx950_all", ), "gfx1250": ("gfx1250_1", )}
FOCUS = FocusRegistry("moe", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(s for s in dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())) if s.dtype == "bf16")


def label(shape: MoeShape) -> str:
    return (f"(({shape.tokens}, {shape.K}), {{'dtype': '{shape.dtype}', 'I': '{shape.I}', 'H': '{shape.H}', "
            f"'E': '{shape.local_experts}/{shape.num_experts}', 'topk': '{shape.topk}'}})")


def flops(local_rows: int, K: int, I: int, H: int) -> int:
    """Gate+up and down GEMMs over the rows routed to this rank's experts."""
    return 2 * local_rows * (2 * I * K + H * I)


def inputs(shape: MoeShape, seed: int = 0, device="cuda"):
    """``x``, ``topk_weights``, ``topk_ids``, ``w13``, ``w2``, ``post_norm_weight``; each token routes to ``topk``
    distinct experts drawn uniformly from all ``num_experts``, with normalized weights.
    Plain FP8 suites use FP8_SCALES and return no post-normalization weight.
    """
    import torch

    dtype = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[shape.dtype]
    gen = torch.Generator(device=device).manual_seed(seed)
    E, K, I, H = shape.local_experts, shape.K, shape.I, shape.H
    std = 0.5 if shape.dtype == "fp8" else 0.02
    w13 = (torch.randn(E, 2 * I, K, device=device, generator=gen) * std).to(dtype)
    w2 = (torch.randn(E, H, I, device=device, generator=gen) * std).to(dtype)
    norm = None if shape.dtype == "fp8" else (torch.randn(H, device=device, generator=gen) * 0.1).to(dtype)
    x = (torch.randn(shape.tokens, K, device=device, generator=gen) * (0.5 if shape.dtype == "fp8" else 1.)).to(dtype)
    ids = torch.rand(shape.tokens, shape.num_experts, device=device, generator=gen).argsort(dim=1)[:, :shape.topk]
    weights = torch.rand(shape.tokens, shape.topk, device=device, generator=gen)
    return x, weights / weights.sum(dim=1, keepdim=True), ids.to(torch.int32), w13, w2, norm
