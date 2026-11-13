"""BF16 16Q/2KV D64, page64 decode benchmark matrices."""

from .._shape_suites import FocusRegistry, FocusSuite

# (batch, query length, KV length, window including self, packed cache)
SYNTHETIC = ((1, 1, 65, 0, True), (3, 8, 2049, 2048, False))
PRIMARY = tuple(
    (b, q, k, w, True) for b in (1, 8, 32, 128) for k in (2048, 8192, 32768) for q in (1, 8) for w in (0, 2048))
EXTENDED = tuple(
    (b, q, k, w, True) for b in (2, 16, 64, 256) for k in (4096, 16384, 65536) for q in (1, 8) for w in (0, 2048))
FOCUS = FocusRegistry(
    "paged_attention_decode",
    (FocusSuite(name="gfx1250_primary", op="paged_attention_decode",
                shapes=PRIMARY), FocusSuite(name="gfx1250_extended", op="paged_attention_decode", shapes=EXTENDED),
     FocusSuite(name="gfx1250_4d", op="paged_attention_decode", shapes=tuple((*s[:4], False) for s in PRIMARY))),
    {"gfx1250": ("gfx1250_primary", )},
)


def flops(batch, query_length, context, window, packed):
    """Useful QK + PV FLOPs, FMA=2; excludes masked work and softmax."""
    pairs = sum(min(max(0, context - query_length + i + 1), window or context) for i in range(query_length))
    return 4 * batch * 16 * 64 * pairs
