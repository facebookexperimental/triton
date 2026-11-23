from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class WindowAttentionVarlenShape(NamedTuple):
    """`batch` sequences whose lengths are drawn uniformly from `[min_seqlen, max_seqlen]`; row `i` attends keys
    `i - left .. i + right` of its own sequence."""

    batch: int
    min_seqlen: int
    max_seqlen: int
    heads: int
    kv_heads: int
    head_dim: int
    left: int
    right: int
    dtype: str


SYNTHETIC: tuple[WindowAttentionVarlenShape, ...] = (
    WindowAttentionVarlenShape(3, 1, 300, 1, 1, 128, 64, 64, "fp16"),
    WindowAttentionVarlenShape(3, 1, 300, 2, 1, 64, 16, 0, "bf16"),
    WindowAttentionVarlenShape(5, 1, 200, 2, 2, 128, 0, 100, "fp16"),
)

# Ranking self-attention over a user's event history: one sequence per request, 5-12 requests per call, histories of
# up to 12,460 events (mean about 5,700), one head of 128 columns, window (64, 64), fp16. The lengths per request are
# estimates that match the call's measured total token count.
SM90_1 = FocusSuite(
    name="sm90_1",
    op="window_attn_varlen",
    shapes=tuple(
        WindowAttentionVarlenShape(batch, 500, 11000, 1, 1, 128, 64, 64, "fp16")
        for batch in (5, 6, 7, 8, 9, 10, 11, 12)) +
    (WindowAttentionVarlenShape(5, 12460, 12460, 1, 1, 128, 64, 64, "fp16"), ),
)
FOCUS_SUITES = (SM90_1, )
DEFAULT_SUITES = {"sm90": ("sm90_1", )}
FOCUS = FocusRegistry("window_attn_varlen", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def seqlens(batch, min_seqlen, max_seqlen) -> list[int]:
    """The shape's sequence lengths, seeded by the shape so every run draws the same batch."""
    import torch

    generator = torch.Generator().manual_seed(batch * 100003 + min_seqlen * 1009 + max_seqlen)
    return torch.randint(min_seqlen, max_seqlen + 1, (batch, ), generator=generator).tolist()


def inputs(lengths, heads, kv_heads, head_dim, dtype, device="cuda"):
    """Packed `(tokens, heads, head_dim)` Q, `(tokens, kv_heads, head_dim)` K/V and the int32 `cu_seqlens`."""
    import itertools

    import torch

    tokens = sum(lengths)
    q = torch.randn(tokens, heads, head_dim, device=device, dtype=dtype)
    k, v = (torch.randn(tokens, kv_heads, head_dim, device=device, dtype=dtype) for _ in range(2))
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lengths)], device=device, dtype=torch.int32)
    return q, k, v, cu_seqlens


def flops(lengths, heads, head_dim, left, right) -> int:
    """Two GEMMs over the keys each row sees."""
    seen = sum(min(n - 1, i + right) - max(0, i - left) + 1 for n in lengths for i in range(n))
    return int(2 * 2.0 * heads * head_dim * seen)


def label(batch, min_seqlen, max_seqlen, heads, kv_heads, head_dim, left, right, dtype) -> str:
    return (f"((), {{'dtype': '{dtype}', 'B': '{batch}', 'SEQLEN': '{min_seqlen}-{max_seqlen}', 'H': '{heads}', "
            f"'KV_H': '{kv_heads}', 'HEAD_DIM': '{head_dim}', 'WINDOW': '{left},{right}'}})")
