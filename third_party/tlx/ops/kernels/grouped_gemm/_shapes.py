from __future__ import annotations

from typing import NamedTuple


class GroupedGemmShape(NamedTuple):
    label: str
    shapes: tuple[tuple[int, int, int], ...]


SYNTHETIC: tuple[GroupedGemmShape, ...] = (
    GroupedGemmShape("ragged", ((1024, 1024, 1024), (512, 512, 512), (256, 256, 256), (128, 128, 128))),
    GroupedGemmShape("moe", ((4096, 4096, 4096), (2048, 4096, 4096), (1000, 4096, 4096), (333, 4096, 4096))),
    GroupedGemmShape("unaligned", ((512, 300, 4000), (333, 1000, 1500), (128, 128, 100), (256, 704, 320))),
    GroupedGemmShape("tiny", ((1, 64, 64), (33, 128, 50))),
)

CORRECTNESS_SHAPES = SYNTHETIC
