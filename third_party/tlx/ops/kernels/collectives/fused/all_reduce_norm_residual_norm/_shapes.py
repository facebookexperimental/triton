from __future__ import annotations

from typing import NamedTuple

from ...._shape_suites import FocusRegistry, FocusSuite
from ...all_reduce import _shapes as all_reduce_shapes


class AllReduceNormResidualNormShape(NamedTuple):
    """``tokens`` rows of a rank's BF16 ``(tokens, H)`` branch partial, summed over the ranks of one node and fed to a
    ``norm_residual_norm`` boundary of kind ``boundary`` (``attn``, ``ffn`` or ``final``)."""
    tokens: int
    H: int
    boundary: str


SYNTHETIC: tuple[AllReduceNormResidualNormShape, ...] = tuple(
    AllReduceNormResidualNormShape(tokens, 4096, boundary)
    for tokens, boundary in ((1, "attn"), (3, "ffn"), (300, "final")))


def _ffn(suite):
    return tuple(AllReduceNormResidualNormShape(s.tokens, s.H, "ffn") for s in suite.shapes)


# The all_reduce suites at the boundary after the FFN: decode, where the fused kernel runs, then chunked prefill
# extends, where the op runs the all-reduce and the standalone boundary kernel.
GFX950_1 = FocusSuite(name="gfx950_1", op="all_reduce_norm_residual_norm", shapes=_ffn(all_reduce_shapes.GFX950_1))
GFX950_2 = FocusSuite(name="gfx950_2", op="all_reduce_norm_residual_norm", shapes=_ffn(all_reduce_shapes.GFX950_2))
# The boundary after attention at hidden 8192 at the decode batches of a TP8 Muse Spark Open server at concurrency 1-32:
# the cooperative kernel up to 8 rows, the all-reduce followed by the standalone boundary above.
GFX950_3 = FocusSuite(name="gfx950_3", op="all_reduce_norm_residual_norm",
                      shapes=tuple(AllReduceNormResidualNormShape(t, 8192, "attn") for t in (1, 2, 4, 8, 16, 24, 32)))
GFX950_ALL = FocusSuite(name="gfx950_all", op="all_reduce_norm_residual_norm",
                        includes=("gfx950_1", "gfx950_2", "gfx950_3"))

FOCUS_SUITES = (GFX950_1, GFX950_2, GFX950_3, GFX950_ALL)
DEFAULT_SUITES = {"gfx950": ("gfx950_all", )}
FOCUS = FocusRegistry("all_reduce_norm_residual_norm", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def label(shape: AllReduceNormResidualNormShape) -> str:
    return f"(({shape.tokens}, {shape.H}), {{'dtype': 'bf16', 'boundary': '{shape.boundary}'}})"
