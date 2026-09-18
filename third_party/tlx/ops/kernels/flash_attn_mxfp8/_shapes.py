from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class FlashAttentionMXFP8Shape(NamedTuple):
    batch: int
    heads: int
    context: int
    head_dim: int
    causal: bool
    dtype: str


SM100_FOCUS_SUITE = FocusSuite(
    name="sm100_baseline",
    op="flash_attn_mxfp8",
    shapes=(
        FlashAttentionMXFP8Shape(4, 32, 1024, 128, False, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 1024, 128, True, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 2048, 128, False, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 2048, 128, True, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 4096, 128, False, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 4096, 128, True, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 8192, 128, False, "bf16"),
        FlashAttentionMXFP8Shape(4, 32, 8192, 128, True, "bf16"),
    ),
)

FOCUS_SUITES = (SM100_FOCUS_SUITE, )
DEFAULT_SUITES = {"sm100": ("sm100_baseline", )}
FOCUS = FocusRegistry("flash_attn_mxfp8", FOCUS_SUITES, DEFAULT_SUITES)

SM100_FOCUS = FOCUS.shapes("sm100")
