from __future__ import annotations

from .._shape_suites import FocusRegistry, FocusSuite
from ..moe._shapes import MoeShape, flops, label

__all__ = ["MoeShape", "flops", "label", "FOCUS", "CORRECTNESS_SHAPES"]


def _muse_spark_open(tokens: int) -> MoeShape:
    # Muse-Spark-Open-816B-A42B at TP4/EP4: moe_hidden_size = moe_intermediate_size = 4096, 256 experts, top-8; rank 3.
    return MoeShape(tokens, 4096, 4096, 4096, 64, 256, 8, 192, "bf16")


# Decode, and the short tail chunk of a chunked prefill.
GFX1250_1 = FocusSuite(name="gfx1250_1", op="moe_mxfp", shapes=tuple(_muse_spark_open(t) for t in (1, 40)))
# A full 8192-token chunked-prefill chunk.
GFX1250_2 = FocusSuite(name="gfx1250_2", op="moe_mxfp", shapes=(_muse_spark_open(8192), ))
GFX1250_ALL = FocusSuite(name="gfx1250_all", op="moe_mxfp", includes=("gfx1250_1", "gfx1250_2"))

FOCUS_SUITES = (GFX1250_1, GFX1250_2, GFX1250_ALL)
DEFAULT_SUITES = {"gfx1250": ("gfx1250_all", )}
FOCUS = FocusRegistry("moe_mxfp", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = FOCUS.all_shapes()
