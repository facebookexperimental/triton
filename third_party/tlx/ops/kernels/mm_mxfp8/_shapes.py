from __future__ import annotations

from .._shape_suites import FocusRegistry, FocusSuite

PRODUCTION_SHAPES: tuple[tuple[int, int, int], ...] = (
    (65536, 1024, 4096),
    (65536, 8192, 1024),
    (65536, 4096, 1024),
    (65536, 3072, 1024),
    (65536, 1024, 1024),
    (264192, 1024, 512),
)

# The two batch sizes for the 15 unique linear shapes that dominate the two
# model workloads. Non-aligned logical inputs are zero-padded to a split-friendly
# multiple of 256; benchmark FLOPs continue to use the logical K dimension.
MODEL_LINEAR_SHAPES: tuple[tuple[int, int, int], ...] = tuple(
    (m, n, k)
    for m in (1024, 2048)
    for k, n in (
        (159744, 6144),
        (22656, 6144),
        (6144, 32768),
        (159744, 1024),
        (1024, 159744),
        (1024, 153600),
        (67584, 6144),
        (39936, 6144),
        (15360, 3072),
        (6144, 24576),
        (24576, 1024),
        (1024, 64512),
        (1024, 36864),
        (12416, 3072),
        (3072, 8192),
    )
)

LOGICAL_K = {22656: 22592, 12416: 12337}

FOCUS_SUITES = (
    FocusSuite(name="gfx950_1", op="mm_mxfp8", shapes=PRODUCTION_SHAPES),
    FocusSuite(name="gfx950_model_linears", op="mm_mxfp8", shapes=MODEL_LINEAR_SHAPES),
)
DEFAULT_SUITES = {"gfx950": ("gfx950_1", )}
FOCUS = FocusRegistry("mm_mxfp8", FOCUS_SUITES, DEFAULT_SUITES)
