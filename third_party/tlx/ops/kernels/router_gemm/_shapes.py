from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class RouterGemmShape(NamedTuple):
    """``tokens`` bf16 rows of width ``hidden`` against an fp32 ``(experts, hidden)`` router weight."""
    tokens: int
    experts: int
    hidden: int


SYNTHETIC: tuple[RouterGemmShape, ...] = (
    RouterGemmShape(1, 128, 1024),
    RouterGemmShape(33, 128, 4096),
    RouterGemmShape(300, 64, 5120),
)

# Router of a 256-expert MoE with an 8192 hidden size. Decode, a verify batch of 8 draft tokens + 1, and speculative
# batches up to 64 tokens:
GFX950_1 = FocusSuite(name="gfx950_1", op="router_gemm",
                      shapes=tuple(RouterGemmShape(t, 256, 8192) for t in (1, 4, 8, 9, 12, 16, 18, 32, 64)))
# Prefill: chunked extends through long prompts.
GFX950_2 = FocusSuite(
    name="gfx950_2",
    op="router_gemm",
    shapes=tuple(RouterGemmShape(t, 256, 8192) for t in (128, 151, 256, 512, 1024, 2048, 4096, 8192, 16384)),
)
GFX950_ALL = FocusSuite(name="gfx950_all", op="router_gemm", includes=("gfx950_1", "gfx950_2"))
# The same router on Blackwell.
SM100_1 = FocusSuite(name="sm100_1", op="router_gemm", shapes=GFX950_1.shapes)
SM100_2 = FocusSuite(name="sm100_2", op="router_gemm", shapes=GFX950_2.shapes)
SM100_ALL = FocusSuite(name="sm100_all", op="router_gemm", includes=("sm100_1", "sm100_2"))

FOCUS_SUITES = (GFX950_1, GFX950_2, GFX950_ALL, SM100_1, SM100_2, SM100_ALL)
DEFAULT_SUITES = {"gfx950": ("gfx950_all", ), "sm100": ("sm100_all", )}
FOCUS = FocusRegistry("router_gemm", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def label(shape: RouterGemmShape) -> str:
    return f"(({shape.tokens}, {shape.hidden}), {{'experts': '{shape.experts}'}})"


def inputs(shape: RouterGemmShape, seed: int = 0, device="cuda"):
    """Normalized bf16 hidden states and an fp32 router weight at the scale of a trained router."""
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape.tokens, shape.hidden, device=device, generator=gen).to(torch.bfloat16)
    weight = torch.randn(shape.experts, shape.hidden, device=device, generator=gen) * shape.hidden**-0.5
    return x, weight


def reference(x, weight):
    """Logits in fp64 from the same inputs."""
    return (x.double() @ weight.double().T)
