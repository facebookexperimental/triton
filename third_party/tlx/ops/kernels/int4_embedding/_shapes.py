from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class Int4EmbeddingShape(NamedTuple):
    tables: int
    batch: int
    rows_per_bag: int
    dim: int
    rows_per_table: int


SYNTHETIC: tuple[Int4EmbeddingShape, ...] = (
    Int4EmbeddingShape(1, 1, 1, 64, 100),
    Int4EmbeddingShape(9, 3, 37, 64, 1000),
    Int4EmbeddingShape(33, 2, 129, 64, 10_000),
    Int4EmbeddingShape(128, 1, 50, 64, 1000),
)

FOCUS_SUITES = (
    FocusSuite(
        name="sm90_1",
        op="int4_embedding",
        shapes=(
            # OneFeed merge-net sequence lookup: 24 tables, 6 long bags each.
            Int4EmbeddingShape(24, 6, 1697, 64, 3_300_000),
            Int4EmbeddingShape(24, 24, 424, 64, 3_300_000),
            Int4EmbeddingShape(24, 384, 26, 64, 3_300_000),
        ),
    ), )

FOCUS = FocusRegistry(
    op="int4_embedding",
    suites=FOCUS_SUITES,
    defaults={"sm90": ("sm90_1", )},
)

CORRECTNESS_SHAPES = tuple(dict.fromkeys((*FOCUS.all_shapes(), *SYNTHETIC)))
