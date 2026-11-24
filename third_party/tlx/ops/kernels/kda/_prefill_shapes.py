from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class KDAPrefillShape(NamedTuple):
    total_tokens: int
    sequences: int
    heads: int
    key_dim: int
    value_dim: int
    dtype: str


SYNTHETIC: tuple[KDAPrefillShape, ...] = (KDAPrefillShape(64, 1, 4, 128, 128, "bf16"), )

GFX950_1 = FocusSuite(
    name="gfx950_1",
    op="kda_paged_prefill",
    shapes=(
        KDAPrefillShape(4096, 1, 4, 128, 128, "bf16"),
        KDAPrefillShape(4096, 4, 4, 128, 128, "bf16"),
        KDAPrefillShape(131072, 1, 4, 128, 128, "bf16"),
        KDAPrefillShape(131072, 8, 4, 128, 128, "bf16"),
        KDAPrefillShape(4096, 1, 12, 128, 128, "bf16"),
        KDAPrefillShape(4096, 4, 12, 128, 128, "bf16"),
        KDAPrefillShape(131072, 1, 12, 128, 128, "bf16"),
        KDAPrefillShape(131072, 8, 12, 128, 128, "bf16"),
    ),
)

FOCUS_SUITES = (GFX950_1, )
DEFAULT_SUITES = {"gfx950": ("gfx950_1", )}
FOCUS = FocusRegistry("kda_paged_prefill", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))
