"""Fixed BF16 16Q/2KV D64, page64 prefill benchmark shapes."""

from .._shape_suites import FocusRegistry, FocusSuite

# (batch, query length, KV length, window including self)
SYNTHETIC = ((1, 65, 129, 0), (1, 769, 2817, 2048))
GFX1250 = FocusSuite(
    name="gfx1250_1",
    op="paged_prefill",
    shapes=tuple((1, length, length, window) for length in (512, 2048, 8192, 16384) for window in (0, 2048)),
)
FOCUS = FocusRegistry("paged_prefill", (GFX1250, ), {"gfx1250": ("gfx1250_1", )})


def flops(batch, sq, sk, window):
    """Useful QK + PV work, FMA=2; exclude masks, softmax and normalization."""
    first = sk - sq + 1
    if not window:
        pairs = sq * (first + sk) // 2
    else:
        ramp = max(0, min(sq, window - first + 1))
        pairs = ramp * (2 * first + ramp - 1) // 2 + (sq - ramp) * window
    return 4 * batch * 16 * 64 * pairs
