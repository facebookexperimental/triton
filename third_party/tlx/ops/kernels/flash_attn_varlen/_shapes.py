from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class FlashAttentionVarlenShape(NamedTuple):
    """`batch` sequences whose lengths are drawn uniformly from `[min_seqlen, max_seqlen]`; Q and K/V share them."""

    batch: int
    min_seqlen: int
    max_seqlen: int
    heads: int
    head_dim: int
    causal: bool
    dtype: str


SYNTHETIC: tuple[FlashAttentionVarlenShape, ...] = (
    FlashAttentionVarlenShape(3, 1, 300, 1, 128, False, "fp16"),
    FlashAttentionVarlenShape(3, 1, 300, 2, 128, True, "bf16"),
)

GFX950_1 = FocusSuite(
    name="gfx950_1",
    op="flash_attn_varlen",
    shapes=(
        FlashAttentionVarlenShape(10, 1000, 1000, 1, 128, False, "fp16"),
        FlashAttentionVarlenShape(10, 700, 1200, 1, 128, False, "fp16"),
        FlashAttentionVarlenShape(1, 1000, 1000, 1, 128, False, "fp16"),
        FlashAttentionVarlenShape(13, 1000, 1000, 1, 128, False, "fp16"),
        FlashAttentionVarlenShape(13, 700, 1200, 1, 128, False, "fp16"),
    ),
)
FOCUS_SUITES = (GFX950_1, )
DEFAULT_SUITES = {"gfx950": ("gfx950_1", )}
FOCUS = FocusRegistry("flash_attn_varlen", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def seqlens(batch, min_seqlen, max_seqlen) -> list[int]:
    """The shape's sequence lengths, seeded by the shape so every run draws the same batch."""
    import torch

    generator = torch.Generator().manual_seed(batch * 100003 + min_seqlen * 1009 + max_seqlen)
    return torch.randint(min_seqlen, max_seqlen + 1, (batch, ), generator=generator).tolist()


def inputs(lengths, heads, head_dim, dtype, device="cuda"):
    """Packed `(tokens, heads, head_dim)` Q/K/V and the int32 `cu_seqlens` for `lengths`."""
    import itertools

    import torch

    tokens = sum(lengths)
    q, k, v = (torch.randn(tokens, heads, head_dim, device=device, dtype=dtype) for _ in range(3))
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lengths)], device=device, dtype=torch.int32)
    return q, k, v, cu_seqlens


def flops(lengths, heads, head_dim, causal) -> int:
    """`flash_attn`'s convention per sequence: two GEMMs, halved when causal."""
    total = sum(2 * (2.0 * heads * n * n * head_dim) for n in lengths)
    return int(total * 0.5 if causal else total)


def label(batch, min_seqlen, max_seqlen, heads, head_dim, causal, dtype) -> str:
    return (f"((), {{'dtype': '{dtype}', 'causal': '{causal}', 'B': '{batch}', "
            f"'SEQLEN': '{min_seqlen}-{max_seqlen}', 'H': '{heads}', 'HEAD_DIM': '{head_dim}'}})")
