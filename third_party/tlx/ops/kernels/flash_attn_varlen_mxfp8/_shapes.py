from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class FlashAttentionVarlenMXFP8Shape(NamedTuple):
    batch: int
    total_q: int
    total_kv: int
    max_q: int
    max_kv: int
    dtype: str


SYNTHETIC: tuple[FlashAttentionVarlenMXFP8Shape, ...] = (FlashAttentionVarlenMXFP8Shape(3, 224, 352, 128, 192,
                                                                                        "bf16"), )

# Production cross-attention aggregates. The original traces retain totals and
# maxima, not every sequence offset, so inputs use a deterministic long-tail
# proxy with the same batch, totals, and maxima.
GFX950_SHORT = FocusSuite(
    name="gfx950_short",
    op="flash_attn_varlen_mxfp8",
    shapes=tuple(
        FlashAttentionVarlenMXFP8Shape(432, total_q, total_kv, max_q, max_kv, "bf16")
        for total_q, total_kv, max_q, max_kv in (
            (24688, 264119, 192, 1004),
            (27248, 264237, 160, 828),
            (29232, 264686, 240, 748),
            (31856, 263664, 224, 788),
            (32736, 263726, 240, 748),
            (26672, 259407, 224, 964),
            (27200, 264751, 224, 748),
            (30576, 264690, 240, 748),
            (30928, 264436, 224, 748),
            (32784, 264520, 240, 748),
        )),
)
GFX950_LONG = FocusSuite(
    name="gfx950_long",
    op="flash_attn_varlen_mxfp8",
    shapes=tuple(
        FlashAttentionVarlenMXFP8Shape(432, total_q, total_kv, max_q, max_kv, "bf16")
        for total_q, total_kv, max_q, max_kv in (
            (24688, 3135104, 192, 32738),
            (27248, 3142170, 160, 21144),
            (29232, 3161784, 240, 16354),
            (31856, 3146362, 224, 18851),
            (32736, 3139507, 240, 16354),
            (26672, 3117994, 224, 29926),
            (27200, 3164011, 224, 16354),
            (30576, 3164659, 240, 16354),
            (30928, 3180031, 224, 16354),
            (32784, 3156939, 240, 16354),
        )),
)
GFX950_PRODUCTION = FocusSuite(
    name="gfx950_production",
    op="flash_attn_varlen_mxfp8",
    includes=("gfx950_short", "gfx950_long"),
)
FOCUS_SUITES = (GFX950_SHORT, GFX950_LONG, GFX950_PRODUCTION)
DEFAULT_SUITES = {"gfx950": ("gfx950_production", )}
FOCUS = FocusRegistry("flash_attn_varlen_mxfp8", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def _lengths(count: int, total: int, maximum: int, seed: int) -> list[int]:
    """Deterministic long-tail lengths with the requested aggregate shape."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    weights = -torch.log(torch.rand(count, generator=generator).clamp_min(1e-6))
    values = torch.clamp((weights / weights.sum() * total).to(torch.int64), 1, maximum)
    values[0] = maximum
    result = values.tolist()
    delta = total - sum(result)
    while delta:
        if delta > 0:
            for index in range(1, count):
                added = min(maximum - result[index], delta)
                result[index] += added
                delta -= added
                if not delta:
                    break
        else:
            for index in range(1, count):
                removed = min(result[index] - 1, -delta)
                result[index] -= removed
                delta += removed
                if not delta:
                    break
    return result


def lengths(shape: FlashAttentionVarlenMXFP8Shape) -> tuple[list[int], list[int]]:
    query_lengths = [16 * value for value in _lengths(
        shape.batch,
        shape.total_q // 16,
        shape.max_q // 16,
        seed=3,
    )]
    key_lengths = _lengths(shape.batch, shape.total_kv, shape.max_kv, seed=5)
    return query_lengths, key_lengths


def inputs(query_lengths, key_lengths, dtype, device="cuda"):
    import itertools

    import torch

    q = torch.randn(sum(query_lengths), 4, 128, device=device, dtype=dtype)
    k = torch.randn(sum(key_lengths), 4, 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, *itertools.accumulate(query_lengths)], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, *itertools.accumulate(key_lengths)], device=device, dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def flops(query_lengths, key_lengths) -> int:
    return int(sum(4.0 * 4 * q * k * 128 for q, k in zip(query_lengths, key_lengths)))


def label(batch, total_q, total_kv, max_q, max_kv, dtype) -> str:
    return (f"((), {{'dtype': '{dtype}', 'B': '{batch}', 'TOTAL_Q': '{total_q}', "
            f"'TOTAL_KV': '{total_kv}', 'MAX_Q': '{max_q}', 'MAX_KV': '{max_kv}'}})")
