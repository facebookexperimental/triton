from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class KDADecodeShape(NamedTuple):
    batch: int
    heads: int
    key_dim: int
    value_dim: int
    dtype: str


SYNTHETIC: tuple[KDADecodeShape, ...] = (KDADecodeShape(1, 4, 128, 128, "bf16"), )

GFX950_1 = FocusSuite(
    name="gfx950_1",
    op="kda_recurrent_decode",
    shapes=tuple(KDADecodeShape(batch, heads, 128, 128, "bf16") for heads in (4, 12) for batch in (1, 2, 4, 8, 16, 32)),
)

FOCUS_SUITES = (GFX950_1, )
DEFAULT_SUITES = {"gfx950": ("gfx950_1", )}
FOCUS = FocusRegistry("kda_recurrent_decode", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))
