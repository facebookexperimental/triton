from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class FlashAttentionMXFP8Shape(NamedTuple):
    batch: int
    heads: int
    context: int
    head_dim: int
    causal: bool
    dtype: str


SYNTHETIC: tuple[FlashAttentionMXFP8Shape, ...] = (
    FlashAttentionMXFP8Shape(1, 1, 256, 128, False, "bf16"),
    FlashAttentionMXFP8Shape(1, 1, 256, 128, True, "bf16"),
)

SM100_1 = FocusSuite(
    name="sm100_1",
    op="flash_attn_mxfp8",
    shapes=(
        FlashAttentionMXFP8Shape(4, 32, 1024, 128, False, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 1024, 128, True, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 2048, 128, False, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 2048, 128, True, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 4096, 128, False, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 4096, 128, True, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 8192, 128, False, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 8192, 128, True, "bf16"),
    ),
)

FOCUS_SUITES = (SM100_1, )
DEFAULT_SUITES = {"sm100": ("sm100_1", )}
FOCUS = FocusRegistry("flash_attn_mxfp8", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def qkv(batch, heads, context, head_dim, dtype, requires_grad=False, device="cuda"):
    import torch

    shape = (batch, heads, context, head_dim)
    return [(torch.randn(shape, device=device, dtype=dtype) * 0.5).requires_grad_(requires_grad) for _ in range(3)]


def flops(batch, heads, context, head_dim, causal, direction="fwd") -> int:
    total = 2 * (2.0 * batch * heads * context * context * head_dim)
    if causal:
        total *= 0.5
    if direction == "bwd":
        total *= 2.5
    return int(total)


def label(batch, heads, context, head_dim, causal, dtype, direction="fwd") -> str:
    return (f"((), {{'dtype': '{dtype}', 'causal': '{causal}', 'dir': '{direction}', "
            f"'Z': '{batch}', 'H': '{heads}', 'N_CTX': '{context}', 'HEAD_DIM': '{head_dim}'}})")
