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
GFX950_FOCUS_SUITE = FocusSuite(
    name="gfx950_baseline",
    op="bmm",
    shapes=(
        BMMShape(8, 256, 256, 272, (256 * 272, 272, 1), (272 * 256, 256, 1), "fp16"),
        BMMShape(8, 256, 256, 272, (256 * 272, 272, 1), (272 * 256, 256, 1), "bf16"),
        BMMShape(2, 448, 160, 931, (0, 931, 1), (931 * 160, 160, 1), "fp16"),
        BMMShape(2, 1195, 256, 2309, (0, 2309, 1), (2309 * 256, 256, 1), "fp16"),
    ),
)

FOCUS_SUITES = (GFX950_FOCUS_SUITE, )
DEFAULT_SUITES = {"gfx950": ("gfx950_baseline", )}
FOCUS = FocusRegistry("bmm", FOCUS_SUITES, DEFAULT_SUITES)

GFX950_FOCUS = FOCUS.shapes("gfx950")


def operand(batch, rows, cols, strides, dtype, device="cuda"):
    import torch

    batch_stride, row_stride, col_stride = strides
    if row_stride != cols or col_stride != 1:
        raise ValueError(f"unsupported bmm strides {strides}")
    if batch_stride == 0:
        return torch.randn((rows, cols), device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)
    if batch_stride == rows * cols:
        return torch.randn((batch, rows, cols), device=device, dtype=dtype)
    raise ValueError(f"unsupported bmm strides {strides}")


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
