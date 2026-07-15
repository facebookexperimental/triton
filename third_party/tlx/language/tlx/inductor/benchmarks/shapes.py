"""Named GEMM shape sets for the torchTLX perf harness.

Shapes are (M, N, K). Keep the default set small so a batch run finishes quickly
and does not wedge the GPU; use ``--shapes`` for ad-hoc shapes or ``--shape-set``
to pick a larger curated set.
"""

# fmt: off

# A handful of representative shapes, one per heuristic regime (mirrors the
# categories exercised by test_torchtlx_templates.TEMPLATE_TEST_SHAPES, but in
# (M, N, K) order).
DEFAULT = [
    (4096, 4096, 4096),     # GPU-saturated general (Rule 7)
    (16384, 2048, 4096),    # tall-M
    (1152, 1024, 16384),    # undersaturated large-output -> split-K (Rule 5)
    (256, 256, 8192),       # undersaturated + large K -> split-K (Rule 6)
    (512, 128, 16384),      # undersaturated small-output, large K -> split-K
]

# Square sweep -- easy to reason about tflops scaling.
SQUARE = [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]

# ads-style tall-M / skinny shapes (production-representative).
ADS = [
    (542442, 512, 896),
    (294912, 512, 512),
    (147456, 448, 192),
    (73728, 256, 768),
    (32768, 1024, 1152),
]

SHAPE_SETS = {
    "default": DEFAULT,
    "square": SQUARE,
    "ads": ADS,
}

# fmt: on


def parse_shapes(spec: str):
    """Parse a ``MxNxK,MxNxK,...`` string into a list of (M, N, K) tuples."""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.lower().replace(" ", "").split("x")
        if len(parts) != 3:
            raise ValueError(f"bad shape {chunk!r}; expected MxNxK")
        out.append(tuple(int(p) for p in parts))
    return out
