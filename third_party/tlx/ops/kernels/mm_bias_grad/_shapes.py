from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class MMBiasGradShape(NamedTuple):
    batch: int
    out_features: int
    in_features: int


SYNTHETIC: tuple[MMBiasGradShape, ...] = (
    MMBiasGradShape(1, 8, 8),
    MMBiasGradShape(65, 96, 144),
    MMBiasGradShape(1025, 136, 264),
    MMBiasGradShape(4097, 512, 512),
    MMBiasGradShape(19459, 384, 384),
)

FOCUS_SUITES = (FocusSuite(
    name="sm100_1",
    op="mm_bias_grad",
    shapes=(
        MMBiasGradShape(1024, 384, 384),
        MMBiasGradShape(4096, 512, 384),
        MMBiasGradShape(8192, 256, 256),
        MMBiasGradShape(294912, 512, 384),
    ),
), )

FOCUS = FocusRegistry(
    op="mm_bias_grad",
    suites=FOCUS_SUITES,
    defaults={"sm100": ("sm100_1", )},
)

CORRECTNESS_SHAPES = tuple(dict.fromkeys((*FOCUS.all_shapes(), *SYNTHETIC)))
