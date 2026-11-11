from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class FlashAttentionVarlenShape(NamedTuple):
    """`batch` sequences whose lengths are drawn uniformly from `[min_seqlen, max_seqlen]`.

    Self-attention (`kv_max_seqlen == 0`): K/V share the query lengths. Cross-attention: key/value lengths are drawn
    from `[kv_min_seqlen, kv_max_seqlen]` instead.
    """

    batch: int
    min_seqlen: int
    max_seqlen: int
    heads: int
    head_dim: int
    causal: bool
    dtype: str
    kv_min_seqlen: int = 0
    kv_max_seqlen: int = 0


SYNTHETIC: tuple[FlashAttentionVarlenShape, ...] = (
    FlashAttentionVarlenShape(3, 1, 300, 1, 128, False, "fp16"),
    FlashAttentionVarlenShape(3, 1, 300, 2, 128, True, "bf16"),
    FlashAttentionVarlenShape(3, 1, 300, 2, 128, False, "bf16", 1, 500),
    FlashAttentionVarlenShape(3, 1, 300, 1, 64, True, "fp16", 1, 200),
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
# OneFeed ranking merge-net attention (Vanguard case 1857594728405650): 24 query rows per candidate, 2 heads,
# head_dim 128. `sm90_1` is the candidate-over-user-history attention, whose key side is the history (1K-5K events);
# `sm90_2` is the same query side over a short 256-row key side. Their lengths are estimates.
SM90_1 = FocusSuite(
    name="sm90_1",
    op="flash_attn_varlen",
    shapes=(
        FlashAttentionVarlenShape(8, 720, 3360, 2, 128, False, "bf16", 1024, 5120),
        FlashAttentionVarlenShape(8, 2160, 2160, 2, 128, False, "bf16", 3072, 3072),
        FlashAttentionVarlenShape(12, 3360, 3360, 2, 128, False, "bf16", 3072, 3072),
        FlashAttentionVarlenShape(8, 2160, 2160, 2, 128, False, "bf16", 5120, 5120),
        FlashAttentionVarlenShape(8, 2160, 2160, 2, 128, False, "bf16", 1024, 1024),
    ),
)
SM90_2 = FocusSuite(
    name="sm90_2",
    op="flash_attn_varlen",
    shapes=(
        FlashAttentionVarlenShape(8, 2160, 2160, 2, 128, False, "bf16", 256, 256),
        FlashAttentionVarlenShape(12, 720, 3360, 2, 128, False, "bf16", 256, 256),
    ),
)
# The same two attentions at the call sizes of the Zoomer trace of that case (profiling run 1086146450672601): 4-12
# requests per call, up to 144 candidates (3456 query rows) per request, fp16. The UHM history is 3K-5K events, which
# matches the trace's fitted mean; the SNN key side is 256 rows.
SM90_3 = FocusSuite(
    name="sm90_3",
    op="flash_attn_varlen",
    shapes=tuple(
        FlashAttentionVarlenShape(requests, 24 * 8, 24 * 144, 2, 128, False, "fp16", kv_min, kv_max)
        for kv_min, kv_max in ((3072, 5120), (256, 256))
        for requests in range(4, 13)),
)
SM90_ALL = FocusSuite(name="sm90_all", op="flash_attn_varlen", includes=("sm90_1", "sm90_2", "sm90_3"))
FOCUS_SUITES = (GFX950_1, SM90_1, SM90_2, SM90_3, SM90_ALL)
DEFAULT_SUITES = {"gfx950": ("gfx950_1", ), "sm90": ("sm90_all", )}
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


def label(batch, min_seqlen, max_seqlen, heads, head_dim, causal, dtype, kv_min_seqlen=0, kv_max_seqlen=0) -> str:
    kv = f"'KV_SEQLEN': '{kv_min_seqlen}-{kv_max_seqlen}', " if kv_max_seqlen else ""
    return (f"((), {{'dtype': '{dtype}', 'causal': '{causal}', 'B': '{batch}', "
            f"'SEQLEN': '{min_seqlen}-{max_seqlen}', {kv}'H': '{heads}', 'HEAD_DIM': '{head_dim}'}})")


def is_cross_attention(shape) -> bool:
    """Whether the shape gives K/V their own lengths rather than sharing the query lengths."""
    return bool(shape.kv_max_seqlen)


def lengths(shape) -> tuple[list[int], list[int]]:
    """The shape's query and key/value lengths, each drawn by `seqlens` from its own length range."""
    lens_q = seqlens(shape.batch, shape.min_seqlen, shape.max_seqlen)
    if not shape.kv_max_seqlen:
        return lens_q, lens_q
    return lens_q, seqlens(shape.batch, shape.kv_min_seqlen, shape.kv_max_seqlen)


def cross_inputs(lens_q, lens_k, heads, head_dim, dtype, device="cuda"):
    """Packed `(tokens, heads, head_dim)` Q/K/V and the int32 `cu_seqlens` for the query and key/value lengths."""
    import itertools

    import torch

    q = torch.randn(sum(lens_q), heads, head_dim, device=device, dtype=dtype)
    k, v = (torch.randn(sum(lens_k), heads, head_dim, device=device, dtype=dtype) for _ in range(2))
    cu_q = torch.tensor([0, *itertools.accumulate(lens_q)], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, *itertools.accumulate(lens_k)], device=device, dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def cross_flops(lens_q, lens_k, heads, head_dim, causal) -> int:
    """`flash_attn`'s convention per sequence: two GEMMs, halved when causal."""
    total = sum(2 * (2.0 * heads * n * m * head_dim) for n, m in zip(lens_q, lens_k))
    return int(total * 0.5 if causal else total)
