from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class BMMShape(NamedTuple):
    batch: int
    m: int
    n: int
    k: int
    a_strides: tuple[int, int, int]
    b_strides: tuple[int, int, int]
    dtype: str


SYNTHETIC: tuple[BMMShape, ...] = (
    BMMShape(8, 256, 256, 272, (256 * 272, 272, 1), (272 * 256, 256, 1), "fp16"),
    BMMShape(8, 256, 256, 272, (256 * 272, 272, 1), (272 * 256, 256, 1), "bf16"),
    # PERF: Exercises the register-load fallback.
    BMMShape(2, 128, 128, 259, (128 * 259, 259, 1), (259 * 128, 128, 1), "fp16"),
    # PERF: Exercises the shared-LHS specialization.
    BMMShape(2, 40, 256, 1956, (0, 1956, 1), (1956 * 256, 256, 1), "fp16"),
)

# PERF: The odd-K fallback stays synthetic-only until it is competitive.
GFX950_1 = FocusSuite(
    name="gfx950_1",
    op="bmm",
    shapes=(
        BMMShape(8, 256, 256, 272, (256 * 272, 272, 1), (272 * 256, 256, 1), "fp16"),
        BMMShape(8, 256, 256, 272, (256 * 272, 272, 1), (272 * 256, 256, 1), "bf16"),
        BMMShape(2, 448, 160, 931, (0, 931, 1), (931 * 160, 160, 1), "fp16"),
        BMMShape(2, 1195, 256, 2309, (0, 2309, 1), (2309 * 256, 256, 1), "fp16"),
    ),
)

GFX950_2 = FocusSuite(
    name="gfx950_2",
    op="bmm",
    shapes=(
        BMMShape(3072, 128, 128, 64, (8192, 64, 1), (8192, 128, 1), "fp16"),
        BMMShape(3072, 240, 128, 128, (30720, 128, 1), (16384, 128, 1), "fp16"),
    ),
)

# Production ranking-model bmm shapes covering 70% of their autotuned bmm time, heaviest first.
GFX950_3 = FocusSuite(
    name="gfx950_3",
    op="bmm",
    shapes=(
        BMMShape(3072, 128, 192, 1892, (242176, 1892, 1), (363264, 192, 1), "fp16"),
        BMMShape(1024, 832, 256, 944, (0, 944, 1), (241664, 256, 1), "bf16"),
        BMMShape(3072, 472, 112, 726, (0, 726, 1), (84000, 112, 1), "fp16"),
        BMMShape(2048, 304, 96, 1256, (0, 1256, 1), (120576, 96, 1), "bf16"),
        BMMShape(3072, 288, 112, 422, (0, 422, 1), (47264, 112, 1), "fp16"),
        BMMShape(3072, 200, 256, 256, (51200, 256, 1), (65536, 256, 1), "fp16"),
        BMMShape(3072, 288, 112, 448, (0, 448, 1), (52864, 112, 1), "fp16"),
        BMMShape(3072, 64, 192, 607, (38848, 607, 1), (116544, 192, 1), "fp16"),
        BMMShape(23, 1024, 2048, 1024, (1024, 1024, 1), (2097152, 2048, 1), "bf16"),
        BMMShape(3072, 240, 112, 351, (0, 351, 1), (39312, 112, 1), "fp16"),
        BMMShape(3072, 16, 192, 112, (17920, 112, 1), (0, 1, 112), "fp16"),
        BMMShape(3072, 160, 112, 422, (0, 422, 1), (47264, 112, 1), "fp16"),
        BMMShape(3072, 288, 112, 240, (0, 240, 1), (29568, 112, 1), "fp16"),
        BMMShape(3072, 40, 556, 112, (4480, 112, 1), (62272, 1, 112), "fp16"),
        BMMShape(1024, 48, 984, 256, (12288, 256, 1), (251904, 1, 256), "bf16"),
        BMMShape(3072, 2, 192, 607, (1214, 607, 1), (116544, 192, 1), "fp16"),
        BMMShape(6, 3072, 2048, 2048, (2048, 2048, 1), (4194304, 1, 2048), "fp16"),
        BMMShape(1024, 48, 256, 984, (47232, 984, 1), (251904, 256, 1), "bf16"),
        BMMShape(8, 3072, 896, 1024, (1024, 1024, 1), (917504, 896, 1), "fp16"),
        BMMShape(4, 3072, 9216, 512, (512, 512, 1), (4718592, 1, 512), "fp16"),
        BMMShape(3072, 40, 112, 556, (22240, 556, 1), (62272, 112, 1), "fp16"),
        BMMShape(2048, 16, 96, 1240, (0, 1240, 1), (119040, 96, 1), "bf16"),
        BMMShape(80, 3072, 112, 768, (768, 768, 1), (86016, 112, 1), "fp16"),
        BMMShape(68, 3072, 768, 128, (128, 128, 1), (98304, 1, 128), "fp16"),
        BMMShape(6, 3072, 1024, 1024, (1024, 1024, 1), (1048576, 1024, 1), "fp16"),
        BMMShape(8, 3072, 1024, 768, (768, 768, 1), (786432, 1024, 1), "fp16"),
        BMMShape(1024, 144, 256, 536, (0, 536, 1), (137216, 256, 1), "bf16"),
        BMMShape(6, 3072, 896, 1024, (1024, 1024, 1), (917504, 896, 1), "fp16"),
        BMMShape(3072, 128, 112, 336, (0, 336, 1), (37632, 112, 1), "fp16"),
        BMMShape(58, 3072, 768, 80, (80, 80, 1), (61440, 1, 80), "fp16"),
        BMMShape(80, 3072, 112, 512, (512, 512, 1), (57344, 112, 1), "fp16"),
        BMMShape(1024, 256, 256, 352, (0, 352, 1), (90112, 256, 1), "bf16"),
        BMMShape(3072, 104, 112, 296, (0, 296, 1), (33152, 112, 1), "fp16"),
        BMMShape(1024, 48, 256, 536, (0, 536, 1), (251904, 256, 1), "bf16"),
        BMMShape(5, 3072, 896, 1024, (1024, 1024, 1), (917504, 896, 1), "fp16"),
        BMMShape(3072, 16, 160, 112, (17920, 112, 1), (0, 1, 112), "fp16"),
        BMMShape(3072, 160, 112, 192, (0, 192, 1), (21504, 112, 1), "fp16"),
        BMMShape(3072, 40, 252, 112, (4480, 112, 1), (28224, 1, 112), "fp16"),
        BMMShape(3072, 224, 112, 112, (0, 112, 1), (12544, 112, 1), "fp16"),
        BMMShape(3072, 144, 192, 64, (9216, 64, 1), (12288, 192, 1), "fp16"),
        BMMShape(3072, 32, 112, 336, (0, 336, 1), (37632, 112, 1), "fp16"),
        BMMShape(3072, 40, 112, 252, (10080, 252, 1), (28224, 112, 1), "fp16"),
    ),
)

GFX950_ALL = FocusSuite(
    name="gfx950_all",
    op="bmm",
    includes=("gfx950_1", "gfx950_2", "gfx950_3"),
)

FOCUS_SUITES = (GFX950_1, GFX950_2, GFX950_3, GFX950_ALL)
DEFAULT_SUITES = {"gfx950": ("gfx950_all", )}
FOCUS = FocusRegistry("bmm", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def operand(batch, rows, cols, strides, dtype, device="cuda"):
    """A (batch, rows, cols) tensor whose strides are exactly `strides`.

    One buffer spans every element the layout addresses, so recorded
    broadcast, padded, transposed and overlapping layouts all build.
    """
    import torch

    size = (batch, rows, cols)
    span = 1 + sum((n - 1) * s for n, s in zip(size, strides))
    return torch.randn(span, device=device, dtype=dtype).as_strided(size, strides)


def inputs(entry, dtype, device="cuda"):
    batch, m, n, k, a_strides, b_strides, _ = entry
    a = operand(batch, m, k, a_strides, dtype, device=device)
    b = operand(batch, k, n, b_strides, dtype, device=device)
    return a, b


def flops(batch, m, n, k):
    return 2 * batch * m * n * k


def label(batch, m, n, k, a_strides, b_strides, dtype) -> str:
    strides = f"[[{', '.join(map(str, a_strides))}], [{', '.join(map(str, b_strides))}]]"
    return (f"((), {{'dtype': '{dtype}', 'strides': '{strides}', 'B': '{batch}', "
            f"'M': '{m}', 'N': '{n}', 'K': '{k}'}})")
