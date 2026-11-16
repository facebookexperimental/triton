from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class FlexAttentionShape(NamedTuple):
    batch: int
    heads: int
    q_len: int
    kv_len: int
    head_dim: int
    shared_kv: bool
    mod: str = "noop"


SM90_ALL = FocusSuite(
    name="sm90_all",
    op="flex_attention",
    shapes=(
        FlexAttentionShape(16, 2, 32, 1920, 80, True),
        FlexAttentionShape(16, 2, 32, 300, 96, True),
        FlexAttentionShape(16, 2, 16, 100, 80, True),
        FlexAttentionShape(16, 2, 32, 1536, 80, False),
        FlexAttentionShape(16, 2, 64, 200, 128, False),
        FlexAttentionShape(16, 2, 1, 77, 64, True),
        FlexAttentionShape(4, 2, 32, 200, 96, True, "score_mask"),
    ),
)
FOCUS = FocusRegistry("flex_attention", (SM90_ALL, ), {"sm90": ("sm90_all", )})
