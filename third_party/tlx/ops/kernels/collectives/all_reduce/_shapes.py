from __future__ import annotations

from typing import NamedTuple

from ..._shape_suites import FocusRegistry, FocusSuite


class AllReduceShape(NamedTuple):
    """``tokens`` rows of a BF16 ``(tokens, H)`` tensor summed over the ranks of one node."""
    tokens: int
    H: int


SYNTHETIC: tuple[AllReduceShape, ...] = tuple(AllReduceShape(tokens, 4096) for tokens in (1, 3, 300))

# Tensor-parallel residual branches of hidden 8192 and 4096: decode including 8-token speculative verify batches,
# then chunked prefill extends.
_DECODE = (1, 4, 8, 9, 12, 16, 18)
_PREFILL = (27, 32, 48, 64, 151, 512, 1024, 1536, 1920, 3072, 4096, 8192, 16384)
GFX950_1 = FocusSuite(name="gfx950_1", op="all_reduce",
                      shapes=tuple(AllReduceShape(t, H) for H in (8192, 4096) for t in _DECODE))
GFX950_2 = FocusSuite(name="gfx950_2", op="all_reduce",
                      shapes=tuple(AllReduceShape(t, H) for H in (8192, 4096) for t in _PREFILL))
GFX950_ALL = FocusSuite(name="gfx950_all", op="all_reduce", includes=("gfx950_1", "gfx950_2"))

FOCUS_SUITES = (GFX950_1, GFX950_2, GFX950_ALL)
DEFAULT_SUITES = {"gfx950": ("gfx950_all", )}
FOCUS = FocusRegistry("all_reduce", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def label(shape: AllReduceShape) -> str:
    return f"(({shape.tokens}, {shape.H}), {{'dtype': 'bf16'}})"


def partial(shape: AllReduceShape, rank: int, seed: int = 0, device="cuda"):
    """Rank ``rank``'s BF16 partial: distinct per rank so a dropped or repeated peer changes the sum."""
    import torch

    gen = torch.Generator(device=device).manual_seed(seed * 1000 + rank)
    return (torch.randn(shape.tokens, shape.H, device=device, generator=gen) * 2).to(torch.bfloat16)


def reference(partials):
    """The FP32 sum of the partials in rank order, rounded to BF16."""
    import torch

    total = partials[0].float()
    for p in partials[1:]:
        total = total + p.float()
    return total.to(torch.bfloat16)
