"""BF16 GQA8 D64, page64 prefill benchmark matrices."""

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class Shape(NamedTuple):
    batch: int
    query_length: int
    context: int
    window: int
    kv_heads: int = 2


SYNTHETIC = (Shape(1, 65, 129, 0), Shape(2, 769, 2817, 2048, 4))
PRIMARY = tuple(Shape(1, k, k, w) for k in (512, 2048, 8192, 16384) for w in (0, 2048))
FULL_HEADS = tuple(s._replace(kv_heads=h) for h in (4, 16) for s in PRIMARY)
BATCHED = tuple(Shape(b, k, k, w, h) for h in (2, 4, 16) for b in (2, 4) for k in (2048, 8192) for w in (0, 2048))
LONG = tuple(Shape(b, k, k, w, h) for h in (2, 4, 16) for b in (1, 2) for k in (32768, 65536) for w in (0, 2048))
PREFIX = tuple(Shape(b, 2048, k, w, h) for h in (2, 4, 16) for b in (1, 2, 4) for k in (8192, 32768) for w in (0, 2048))
FOCUS = FocusRegistry(
    "paged_prefill",
    tuple(
        FocusSuite(name=name, op="paged_prefill", shapes=shapes)
        for name, shapes in (("gfx1250_1", PRIMARY), ("gfx1250_heads", FULL_HEADS), ("gfx1250_batched", BATCHED),
                             ("gfx1250_long", LONG), ("gfx1250_prefix", PREFIX))),
    {"gfx1250": ("gfx1250_1", "gfx1250_heads", "gfx1250_batched", "gfx1250_long", "gfx1250_prefix")},
)


def flops(batch, sq, sk, window, kv_heads=2):
    """Useful QK + PV work, FMA=2; exclude masks, softmax and normalization."""
    first = sk - sq + 1
    if not window:
        pairs = sq * (first + sk) // 2
    else:
        ramp = max(0, min(sq, window - first + 1))
        pairs = ramp * (2 * first + ramp - 1) // 2 + (sq - ramp) * window
    return 4 * batch * kv_heads * 8 * 64 * pairs
