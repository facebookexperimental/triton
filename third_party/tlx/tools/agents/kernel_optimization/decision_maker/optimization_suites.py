"""Agent-owned benchmark cohorts that are not production shape requests."""

from __future__ import annotations

from typing import NamedTuple


class OptimizationShape(NamedTuple):
    m: int
    n: int
    k: int
    a_strides: tuple[int, int]
    b_strides: tuple[int, int]
    dtype: str


GFX942_MM_OPTIMIZATION_SUITES = {
    "gfx942_correctness_narrow": (
        OptimizationShape(344064, 32, 527, (527, 1), (1, 527), "fp16"),
        OptimizationShape(262144, 32, 91, (91, 1), (1, 91), "bf16"),
    ),
    "gfx942_group_a": (
        OptimizationShape(128, 8960, 6852, (6852, 1), (1, 6852), "fp16"),
        OptimizationShape(128, 2688, 3840, (3840, 1), (1, 3840), "fp16"),
        OptimizationShape(128, 61040, 576, (576, 1), (1, 576), "fp16"),
        OptimizationShape(128, 2688, 3360, (7840, 1), (1, 3360), "fp16"),
        OptimizationShape(128, 1536, 6144, (6144, 1), (1, 6144), "fp16"),
        OptimizationShape(128, 23520, 576, (576, 1), (1, 576), "fp16"),
        OptimizationShape(128, 20580, 576, (576, 1), (1, 576), "fp16"),
        OptimizationShape(128, 16384, 256, (256, 1), (1, 256), "fp16"),
        OptimizationShape(128, 8960, 1152, (1152, 1), (1, 1152), "fp16"),
        OptimizationShape(128, 7840, 576, (576, 1), (1, 576), "fp16"),
        OptimizationShape(128, 1536, 3072, (3072, 1), (1, 3072), "fp16"),
        OptimizationShape(128, 256, 6144, (6144, 1), (1, 6144), "fp16"),
        OptimizationShape(128, 1344, 1152, (1152, 1), (1, 1152), "fp16"),
        OptimizationShape(128, 2688, 2688, (6272, 1), (1, 2688), "fp16"),
        OptimizationShape(128, 8192, 256, (256, 1), (1, 256), "fp16"),
        OptimizationShape(128, 1792, 2688, (6272, 1), (1, 2688), "fp16"),
        OptimizationShape(128, 256, 3072, (3072, 1), (1, 3072), "fp16"),
        OptimizationShape(128, 6272, 384, (384, 1), (1, 384), "fp16"),
        OptimizationShape(128, 896, 768, (768, 1), (1, 768), "fp16"),
        OptimizationShape(279, 8192, 2048, (2048, 1), (1, 2048), "fp16"),
        OptimizationShape(279, 2048, 4096, (4096, 1), (1, 4096), "fp16"),
        OptimizationShape(279, 4096, 4352, (4352, 1), (1, 4352), "fp16"),
        OptimizationShape(7, 8192, 2048, (2048, 1), (1, 2048), "fp16"),
        OptimizationShape(7, 2048, 4096, (4096, 1), (1, 4096), "fp16"),
    ),
    "gfx942_group_b": (
        OptimizationShape(294912, 128, 128, (128, 1), (1, 128), "fp16"),
        OptimizationShape(409600, 32, 32, (32, 1), (1, 32), "bf16"),
        OptimizationShape(14336, 56, 563, (563, 1), (1, 563), "fp16"),
        OptimizationShape(14336, 126, 144, (144, 1), (1, 144), "fp16"),
        OptimizationShape(14336, 70, 144, (144, 1), (1, 144), "fp16"),
        OptimizationShape(14336, 70, 126, (126, 1), (1, 126), "fp16"),
        OptimizationShape(3072, 128, 112, (112, 1), (1, 112), "fp16"),
    ),
    "gfx942_group_b_tall_128": (
        OptimizationShape(294912, 128, 128, (128, 1), (1, 128), "fp16"),
    ),
    "gfx942_group_b_narrow_deep": (
        OptimizationShape(14336, 56, 563, (563, 1), (1, 563), "fp16"),
    ),
    "gfx942_group_b_shallow": (
        OptimizationShape(409600, 32, 32, (32, 1), (1, 32), "bf16"),
        OptimizationShape(14336, 126, 144, (144, 1), (1, 144), "fp16"),
        OptimizationShape(14336, 70, 144, (144, 1), (1, 144), "fp16"),
        OptimizationShape(14336, 70, 126, (126, 1), (1, 126), "fp16"),
        OptimizationShape(3072, 128, 112, (112, 1), (1, 112), "fp16"),
    ),
    "gfx942_group_c": (
        OptimizationShape(3072, 1024, 4096, (4096, 1), (1, 4096), "fp16"),
        OptimizationShape(2032, 2560, 18688, (18688, 1), (1, 18688), "fp16"),
        OptimizationShape(2032, 512, 18688, (18688, 1), (1, 18688), "fp16"),
    ),
    "gfx942_group_d": (
        OptimizationShape(262144, 1195, 2373, (2373, 1), (1, 2373), "fp16"),
    ),
}
