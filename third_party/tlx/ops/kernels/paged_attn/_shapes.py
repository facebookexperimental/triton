"""BF16 GQA8 D64, page64 decode benchmark matrices."""

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class Shape(NamedTuple):
    batch: int
    query_length: int
    context: int
    window: int
    packed: bool = True
    kv_heads: int = 2
    profile: str = "uniform"


def context_lengths(batch, query_length, context, profile="uniform"):
    """Deterministic ragged batches, including unaligned lengths and the bound."""
    if profile == "uniform":
        return [context] * batch
    if profile != "mixed":
        raise ValueError(f"unknown context profile {profile!r}")
    lengths = (context, query_length, context // 4 - 1, 63, context // 2 + 7, 129, context - 63, context // 16 + 1)
    return [max(query_length, min(context, lengths[b % len(lengths)])) for b in range(batch)]


SYNTHETIC = (Shape(1, 1, 65, 0), Shape(3, 7, 65, 0), Shape(3, 8, 2049, 2048, False, 4, "mixed"))
PRIMARY = tuple(
    Shape(b, q, k, w) for b in (1, 8, 32, 128) for k in (2048, 8192, 16384, 32768) for q in (1, 8) for w in (0, 2048))
EXTENDED = tuple(
    Shape(b, q, k, w) for b in (2, 16, 64, 256) for k in (4096, 16384, 65536) for q in (1, 8) for w in (0, 2048))
INTERMEDIATE = tuple(
    Shape(b, q, k, 0, True, 2, profile)
    for b in (12, 24)
    for q in (1, 7, 8)
    for k in (4096, 8192, 16384)
    for profile in ("uniform", "mixed"))
FULL_HEADS = tuple(s._replace(kv_heads=h) for h in (4, 16) for s in PRIMARY)
MIXED = tuple(
    Shape(b, q, k, w, True, h, "mixed")
    for h in (2, 4, 16)
    for b in (8, 32, 128)
    for k in (8192, 32768)
    for q in (1, 8)
    for w in (0, 2048))
# Seven-token drafting and eight-token verification, including 128K contexts.
SBD = tuple(
    shape._replace(kv_heads=h) for h in (2, 4, 16) for shape in (
        Shape(16, 1, 8192, 0),
        Shape(64, 1, 8192, 0),
        Shape(16, 8, 8192, 0),
        Shape(32, 8, 8192, 0),
        Shape(64, 8, 8192, 0),
        Shape(64, 7, 8192, 0),
        Shape(64, 8, 8192, 2048),
        Shape(8, 8, 131072, 0),
        Shape(8, 8, 131072, 2048),
        Shape(16, 8, 131072, 0),
    ))
FOCUS = FocusRegistry(
    "paged_attention_decode",
    (FocusSuite(name="gfx1250_primary", op="paged_attention_decode",
                shapes=PRIMARY), FocusSuite(name="gfx1250_extended", op="paged_attention_decode", shapes=EXTENDED),
     FocusSuite(name="gfx1250_heads", op="paged_attention_decode",
                shapes=FULL_HEADS), FocusSuite(name="gfx1250_mixed", op="paged_attention_decode", shapes=MIXED),
     FocusSuite(name="gfx1250_intermediate", op="paged_attention_decode",
                shapes=INTERMEDIATE), FocusSuite(name="gfx1250_sbd", op="paged_attention_decode", shapes=SBD),
     FocusSuite(name="gfx1250_4d", op="paged_attention_decode", shapes=tuple(s._replace(packed=False)
                                                                             for s in PRIMARY))),
    {"gfx1250": ("gfx1250_primary", "gfx1250_heads", "gfx1250_mixed", "gfx1250_intermediate", "gfx1250_sbd")},
)


def flops(batch, query_length, context, window, packed=True, kv_heads=2, profile="uniform"):
    """Useful QK + PV FLOPs, FMA=2; excludes masked work and softmax."""
    pairs = sum(
        min(max(0, length - query_length + i + 1), window or length)
        for length in context_lengths(batch, query_length, context, profile)
        for i in range(query_length))
    return 4 * kv_heads * 8 * 64 * pairs
