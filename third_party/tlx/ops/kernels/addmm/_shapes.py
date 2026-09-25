from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite
from ..mm._shapes import operand


class AddMMShape(NamedTuple):
    m: int
    n: int
    k: int
    a_strides: tuple[int, int]
    b_strides: tuple[int, int]
    dtype: str


SYNTHETIC: tuple[AddMMShape, ...] = (
    AddMMShape(256, 256, 272, (272, 1), (1, 272), "fp16"),
    AddMMShape(256, 256, 272, (272, 1), (1, 272), "bf16"),
    AddMMShape(264, 256, 328, (328, 1), (1, 328), "fp16"),
    # PERF: Exercises the register-load fallback.
    AddMMShape(256, 192, 259, (259, 1), (1, 259), "fp16"),
)

# PERF: The odd-K fallback stays synthetic-only until it is competitive.
FOCUS_SUITES = (
    FocusSuite(
        name="gfx942_1",
        op="addmm",
        shapes=(
            AddMMShape(819200, 1024, 192, (192, 1), (1, 192), "bf16"),
            AddMMShape(4096, 1894, 242432, (242432, 1), (1, 242432), "bf16"),
            AddMMShape(1024, 6144, 20480, (20480, 1), (1, 20480), "bf16"),
            AddMMShape(61440, 2048, 5120, (5120, 1), (1, 5120), "bf16"),
        ),
    ),
    FocusSuite(
        name="gfx950_1",
        op="addmm",
        shapes=(
            AddMMShape(4096, 192, 2048, (2048, 1), (1, 2048), "fp16"),
            AddMMShape(4096, 192, 2048, (2048, 1), (1, 2048), "bf16"),
            AddMMShape(1024, 1024, 864, (864, 1), (1, 864), "fp16"),
            AddMMShape(64000, 256, 256, (256, 1), (1, 256), "fp16"),
        ),
    ),
)

DEFAULT_SUITES = {
    "gfx942": ("gfx942_1", ),
    "gfx950": ("gfx950_1", ),
}
FOCUS = FocusRegistry("addmm", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def inputs(entry, dtype, device="cuda"):
    m, n, k, a_strides, b_strides, _ = entry
    import torch

    bias = torch.randn(n, device=device, dtype=dtype)
    a = operand(m, k, a_strides, dtype, device=device)
    b = operand(k, n, b_strides, dtype, device=device)
    return bias, a, b


def flops(m, n, k):
    return 2 * m * n * k


def label(m, n, k, a_strides, b_strides, dtype) -> str:
    strides = f"[[{a_strides[0]}, {a_strides[1]}], [{b_strides[0]}, {b_strides[1]}]]"
    return (f"((), {{'dtype': '{dtype}', 'strides': '{strides}', "
            f"'M': '{m}', 'N': '{n}', 'K': '{k}', 'bias': '[{n}]'}})")
