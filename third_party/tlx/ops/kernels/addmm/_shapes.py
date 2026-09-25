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
    bias_strides: tuple[int, ...]
    dtype: str


SYNTHETIC: tuple[AddMMShape, ...] = (
    AddMMShape(256, 256, 272, (272, 1), (1, 272), (1, ), "fp16"),
    AddMMShape(256, 256, 272, (272, 1), (1, 272), (1, ), "bf16"),
    AddMMShape(264, 256, 328, (328, 1), (1, 328), (1, ), "fp16"),
    # PERF: Exercises the register-load fallback.
    AddMMShape(256, 192, 259, (259, 1), (1, 259), (1, ), "fp16"),
)

# PERF: The odd-K fallback stays synthetic-only until it is competitive.
FOCUS_SUITES = (
    FocusSuite(
        name="gfx942_1",
        op="addmm",
        shapes=(
            AddMMShape(819200, 1024, 192, (192, 1), (1, 192), (1, ), "bf16"),
            AddMMShape(4096, 1894, 242432, (242432, 1), (1, 242432), (1, ), "bf16"),
            AddMMShape(1024, 6144, 20480, (20480, 1), (1, 20480), (1, ), "bf16"),
            AddMMShape(61440, 2048, 5120, (5120, 1), (1, 5120), (1, ), "bf16"),
        ),
    ),
    FocusSuite(
        name="gfx950_1",
        op="addmm",
        shapes=(
            AddMMShape(4096, 192, 2048, (2048, 1), (1, 2048), (1, ), "fp16"),
            AddMMShape(4096, 192, 2048, (2048, 1), (1, 2048), (1, ), "bf16"),
            AddMMShape(1024, 1024, 864, (864, 1), (1, 864), (1, ), "fp16"),
            AddMMShape(64000, 256, 256, (256, 1), (1, 256), (1, ), "fp16"),
        ),
    ),
    FocusSuite(
        name="gfx950_2",
        op="addmm",
        shapes=(
            AddMMShape(819200, 192, 1024, (1024, 1), (1, 1024), (1, ), "fp16"),
            AddMMShape(4096, 242432, 1894, (1894, 1), (1, 1894), (1, ), "fp16"),
            AddMMShape(1024, 20480, 6144, (6144, 1), (1, 6144), (1, ), "fp16"),
            AddMMShape(61440, 5120, 2048, (2048, 1), (1, 2048), (1, ), "fp16"),
            AddMMShape(2252800, 256, 256, (256, 1), (1, 256), (1, ), "fp16"),
            AddMMShape(61440, 3840, 4096, (4096, 1), (1, 4096), (1, ), "fp16"),
            AddMMShape(4096, 4096, 2048, (2048, 1), (1, 2048), (1, ), "fp16"),
            AddMMShape(61440, 5120, 7744, (7744, 1), (1, 7744), (1, ), "fp16"),
            AddMMShape(1024, 6144, 4096, (4096, 1), (1, 4096), (1, ), "fp16"),
            AddMMShape(61440, 2560, 4096, (4096, 1), (1, 4096), (1, ), "fp16"),
            AddMMShape(819200, 1024, 192, (192, 1), (1, 192), (1, ), "fp16"),
            AddMMShape(4096, 2048, 6144, (6144, 1), (1, 6144), (1, ), "fp16"),
            AddMMShape(1024, 4096, 6144, (6144, 1), (1, 6144), (1, ), "fp16"),
            AddMMShape(1024, 6144, 22272, (22272, 1), (1, 22272), (1, ), "fp16"),
            AddMMShape(4096, 2048, 4096, (4096, 1), (1, 4096), (1, ), "fp16"),
            AddMMShape(61440, 10000, 1024, (1024, 1), (1, 1024), (1, ), "fp16"),
        ),
    ),
    FocusSuite(
        name="gfx950_3",
        op="addmm",
        shapes=(
            AddMMShape(256, 2048, 359, (359, 1), (1, 359), (1, ), "fp16"),
            AddMMShape(256, 282, 282, (282, 1), (1, 282), (1, ), "fp16"),
            AddMMShape(3072, 2048, 2048, (2048, 1), (1, 2048), (1, ), "fp16"),
            AddMMShape(3072, 10240, 128, (128, 1), (1, 128), (1, ), "fp16"),
            AddMMShape(3072, 4096, 7424, (7424, 1), (1, 7424), (1, ), "fp16"),
            AddMMShape(3072, 8192, 2048, (2048, 1), (1, 2048), (1, ), "fp16"),
            AddMMShape(256, 1024, 2048, (2048, 1), (1, 2048), (1, ), "fp16"),
            AddMMShape(3072, 1024, 2048, (2048, 1), (1, 2048), (1, ), "fp16"),
            AddMMShape(3072, 4096, 2048, (2048, 1), (1, 2048), (1, ), "fp16"),
            AddMMShape(3072, 3, 1024, (30720, 1), (1, 1024), (1, ), "fp16"),
            AddMMShape(3072, 2048, 797, (797, 1), (1, 797), (1, ), "fp16"),
            AddMMShape(3072, 2048, 2048, (30720, 1), (1, 4096), (2048, 1), "fp16"),
            AddMMShape(3072, 4096, 797, (797, 1), (1, 1156), (4096, 1), "fp16"),
        ),
    ),
    FocusSuite(
        name="gfx950_all",
        op="addmm",
        includes=("gfx950_1", "gfx950_2", "gfx950_3"),
    ),
)

DEFAULT_SUITES = {
    "gfx942": ("gfx942_1", ),
    "gfx950": ("gfx950_all", ),
}
FOCUS = FocusRegistry("addmm", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def inputs(entry, dtype, device="cuda"):
    m, n, k, a_strides, b_strides, bias_strides, _ = entry
    import torch

    if len(bias_strides) == 1:
        if bias_strides != (1, ):
            raise ValueError(f"unsupported addmm bias strides {bias_strides}")
        bias = torch.randn(n, device=device, dtype=dtype)
    elif len(bias_strides) == 2:
        bias = operand(m, n, bias_strides, dtype, device=device)
    else:
        raise ValueError(f"unsupported addmm bias rank {len(bias_strides)}")
    a = operand(m, k, a_strides, dtype, device=device)
    b = operand(k, n, b_strides, dtype, device=device)
    return bias, a, b


def flops(m, n, k):
    return 2 * m * n * k


def label(m, n, k, a_strides, b_strides, bias_strides, dtype) -> str:
    strides = f"[[{a_strides[0]}, {a_strides[1]}], [{b_strides[0]}, {b_strides[1]}]]"
    bias = f"[{n}]" if len(bias_strides) == 1 else f"[{m}, {n}]"
    return (f"((), {{'dtype': '{dtype}', 'strides': '{strides}', "
            f"'M': '{m}', 'N': '{n}', 'K': '{k}', 'bias': '{bias}'}})")
