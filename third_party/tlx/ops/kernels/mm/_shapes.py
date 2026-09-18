from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class MMShape(NamedTuple):
    m: int
    n: int
    k: int
    a_strides: tuple[int, int]
    b_strides: tuple[int, int]
    dtype: str


SYNTHETIC: tuple[MMShape, ...] = (
    MMShape(256, 256, 256, (256, 1), (256, 1), "fp16"),
    MMShape(1024, 1024, 1024, (1024, 1), (1024, 1), "fp16"),
    MMShape(2048, 512, 1024, (1024, 1), (512, 1), "fp16"),
    MMShape(512, 4096, 1024, (1024, 1), (1, 1024), "fp16"),
    MMShape(1024, 2048, 512, (1, 1024), (2048, 1), "fp16"),
    MMShape(2048, 2048, 2048, (1, 2048), (1, 2048), "fp16"),
    MMShape(136, 256, 128, (128, 1), (256, 1), "fp16"),
    MMShape(1000, 1000, 1024, (1024, 1), (1000, 1), "fp16"),
    MMShape(1000, 1000, 200, (200, 1), (1000, 1), "fp16"),
    MMShape(256, 256, 16384, (16384, 1), (256, 1), "fp16"),
    MMShape(64, 4096, 4096, (4096, 1), (4096, 1), "fp16"),
    MMShape(256, 256, 256, (256, 1), (256, 1), "bf16"),
    MMShape(1024, 1024, 1024, (1024, 1), (1024, 1), "bf16"),
    MMShape(2048, 512, 1024, (1024, 1), (512, 1), "bf16"),
    MMShape(512, 4096, 1024, (1024, 1), (1, 1024), "bf16"),
    MMShape(1024, 2048, 512, (1, 1024), (2048, 1), "bf16"),
    MMShape(2048, 2048, 2048, (1, 2048), (1, 2048), "bf16"),
    MMShape(136, 256, 128, (128, 1), (256, 1), "bf16"),
    MMShape(1000, 1000, 1024, (1024, 1), (1000, 1), "bf16"),
    MMShape(1000, 1000, 200, (200, 1), (1000, 1), "bf16"),
    MMShape(256, 256, 16384, (16384, 1), (256, 1), "bf16"),
    MMShape(64, 4096, 4096, (4096, 1), (4096, 1), "bf16"),
    # PERF: Cover K-dominant and N-dominant split-K geometries.
    MMShape(384, 3072, 64512, (64512, 1), (3072, 1), "bf16"),
    MMShape(384, 64512, 3072, (3072, 1), (64512, 1), "bf16"),
)

FOCUS_SUITES = (
    FocusSuite(
        name="sm100_baseline",
        op="mm",
        shapes=(
            MMShape(8192, 8192, 8192, (8192, 1), (8192, 1), "fp16"),
            MMShape(8192, 8192, 8192, (8192, 1), (8192, 1), "bf16"),
            MMShape(8192, 8192, 1024, (1024, 1), (8192, 1), "fp16"),
            MMShape(8192, 8192, 1024, (1024, 1), (8192, 1), "bf16"),
            MMShape(8192, 8192, 16384, (16384, 1), (8192, 1), "fp16"),
            MMShape(8192, 8192, 16384, (16384, 1), (8192, 1), "bf16"),
            MMShape(8192, 8192, 8192, (8192, 1), (1, 8192), "fp16"),
            MMShape(8192, 8192, 8192, (8192, 1), (1, 8192), "bf16"),
        ),
    ),
    FocusSuite(
        name="gfx942_baseline",
        op="mm",
        shapes=(
            # PERF: These BF16 layouts select gfx942's direct-load paths.
            MMShape(819200, 1024, 192, (192, 1), (1, 192), "bf16"),
            MMShape(4096, 1894, 242432, (242432, 1), (1, 242432), "bf16"),
            MMShape(1024, 6144, 20480, (20480, 1), (1, 20480), "bf16"),
            MMShape(2048, 25408, 10240, (10240, 1), (1, 10240), "bf16"),
            MMShape(61440, 2048, 5120, (5120, 1), (1, 5120), "bf16"),
            MMShape(819200, 192, 1024, (1024, 1), (192, 1), "fp16"),
            MMShape(4096, 242432, 1894, (1894, 1), (242432, 1), "fp16"),
            MMShape(1024, 20480, 6144, (6144, 1), (20480, 1), "fp16"),
            MMShape(2048, 10240, 25408, (25408, 1), (10240, 1), "fp16"),
            MMShape(61440, 5120, 2048, (2048, 1), (5120, 1), "fp16"),
            MMShape(2252800, 256, 256, (256, 1), (256, 1), "fp16"),
            MMShape(61440, 3840, 4096, (4096, 1), (3840, 1), "fp16"),
            MMShape(4096, 4096, 2048, (2048, 1), (4096, 1), "fp16"),
            MMShape(61440, 5120, 7744, (7744, 1), (5120, 1), "fp16"),
            MMShape(1024, 6144, 4096, (4096, 1), (6144, 1), "fp16"),
        ),
    ),
    FocusSuite(
        name="gfx950_baseline",
        op="mm",
        shapes=(
            MMShape(7, 8192, 2048, (2048, 1), (1, 2048), "fp16"),
            MMShape(7, 2048, 4096, (4096, 1), (1, 4096), "fp16"),
        ),
    ),
)

DEFAULT_SUITES = {
    "sm100": ("sm100_baseline", ),
    "gfx942": ("gfx942_baseline", ),
    "gfx950": ("gfx950_baseline", ),
}
FOCUS = FocusRegistry("mm", FOCUS_SUITES, DEFAULT_SUITES)

SM100_FOCUS = FOCUS.shapes("sm100")
GFX942_FOCUS = FOCUS.shapes("gfx942")
GFX950_FOCUS = FOCUS.shapes("gfx950")

ALL: tuple[MMShape,
           ...] = tuple(dict.fromkeys(SYNTHETIC + tuple(shape for suite in FOCUS_SUITES for shape in suite.shapes)))


def operand(rows, cols, strides, dtype, device="cuda"):
    """A (rows, cols) tensor whose strides are exactly `strides`.

    Recorded strides carry three things a row/column-major flag cannot:
    a leading stride wider than the row (a slice of a padded buffer),
    stride 0 (a broadcast operand), and which of the two dims is contiguous.
    """
    import torch

    s0, s1 = strides
    if s0 == 0:  # broadcast down the rows
        # At rows == 1 the expand is a no-op and stride(0) stays `cols`. That is
        # not a miss: a stride on a dim of extent 1 addresses nothing, so the
        # element layout is identical either way.
        return torch.randn((1, cols), device=device, dtype=dtype).expand(rows, cols)
    if s1 == 1:  # row-major, possibly padded to s0 >= cols
        return torch.randn((rows, s0), device=device, dtype=dtype)[:, :cols]
    if s0 == 1:  # column-major, possibly padded to s1 >= rows
        return torch.randn((cols, s1), device=device, dtype=dtype)[:, :rows].T
    raise ValueError(f"unsupported strides {strides} for ({rows}, {cols})")


def label(M, N, K, a_strides, b_strides, dtype) -> str:
    strides = f"[[{a_strides[0]}, {a_strides[1]}], [{b_strides[0]}, {b_strides[1]}]]"
    return (f"((), {{'dtype': '{dtype}', 'strides': '{strides}', "
            f"'M': '{M}', 'N': '{N}', 'K': '{K}'}})")


def flops(M, N, K):
    return 2 * M * N * K
