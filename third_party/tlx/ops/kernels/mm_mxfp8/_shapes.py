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

# M=85 is the in-model request-row regime. M=128 and M=2048 retain the
# production shape family's boundary and throughput regression coverage.
PRODUCTION_SHORT_M_SHAPES: tuple[tuple[int, int, int], ...] = tuple((m, n, k) for m in (85, 128, 2048) for n, k in (
    (23040, 1792),
    (22464, 1024),
    (15360, 3584),
    (14976, 1024),
    (3584, 25600),
    (3072, 18048),
))

# The two batch sizes for the 15 unique linear shapes that dominate the two
# model workloads. Non-aligned logical inputs are zero-padded to a split-friendly
# multiple of 256; benchmark FLOPs continue to use the logical K dimension.
MODEL_LINEAR_SHAPES: tuple[tuple[int, int, int], ...] = tuple((m, n, k) for m in (1024, 2048) for k, n in (
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
))

# Tile-native caller scale layouts used by the production model-linear routes.
# Repeats are the A/B MFMA row groups: 256x192 => (4, 3), 128x192 =>
# (2, 3), 128x128 => (2, 2), 128x256 => (2, 4), 256x256 => (4, 4).
MODEL_LINEAR_PRESHUFFLE_REPEATS: dict[tuple[int, int, int], tuple[int, int]] = {
    (1024, 6144, 159744): (4, 3),
    (1024, 6144, 22656): (4, 3),
    (1024, 6144, 67584): (4, 3),
    (1024, 1024, 24576): (2, 4),
    (1024, 36864, 1024): (2, 3),
    (1024, 8192, 3072): (2, 2),
    (2048, 6144, 159744): (4, 3),
    (2048, 6144, 22656): (4, 3),
    (2048, 32768, 6144): (4, 4),
    (2048, 6144, 67584): (4, 3),
    (2048, 6144, 39936): (4, 3),
    (2048, 3072, 15360): (4, 3),
    (2048, 1024, 24576): (2, 4),
    (2048, 36864, 1024): (2, 3),
    (2048, 3072, 12416): (2, 3),
}

LOGICAL_K = {22656: 22592, 12416: 12337, 25600: 25536}

SYNTHETIC_SHAPES: tuple[tuple[int, int, int], ...] = (
    (128, 128, 128),
    (256, 384, 512),
    (1024, 1024, 1024),
)

FOCUS_SUITES = (
    FocusSuite(name="gfx950_1", op="mm_mxfp8", shapes=PRODUCTION_SHAPES),
    FocusSuite(name="gfx950_short_m", op="mm_mxfp8", shapes=PRODUCTION_SHORT_M_SHAPES),
    FocusSuite(name="gfx950_model_linears", op="mm_mxfp8", shapes=MODEL_LINEAR_SHAPES),
)
DEFAULT_SUITES = {"gfx950": ("gfx950_1", "gfx950_short_m", "gfx950_model_linears")}
FOCUS = FocusRegistry("mm_mxfp8", FOCUS_SUITES, DEFAULT_SUITES)


def label(m: int, n: int, k: int, sf_layout: str) -> str:
    logical_k = LOGICAL_K.get(k, k)
    kernel_k = f", 'K_kernel': '{k}'" if logical_k != k else ""
    return (f"((), {{'dtype': 'fp8e4m3', 'sf_layout': '{sf_layout}', "
            f"'M': '{m}', 'N': '{n}', 'K': '{logical_k}'{kernel_k}}})")


def flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * LOGICAL_K.get(k, k)
