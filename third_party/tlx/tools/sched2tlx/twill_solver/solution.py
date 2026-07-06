"""The solver's output: a per-loop schedule + warp assignment + buffering that
``graph_writer`` stamps back into ``schedule_graph.json``.

A ``LoopSolution`` is exactly the set of fields the emitter consumes; everything
else in the schedule graph passes through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoopSolution:
    loop_id: int
    II: int
    node_cycle: dict[int, int]      # node id -> cycle
    node_stage: dict[int, int]      # node id -> stage
    node_cluster: dict[int, int]    # node id -> cluster (intra (wg,stage) order)
    node_wg: dict[int, int]         # node id -> warp_group
    warp_groups: list[dict[str, Any]]      # [{id, pipelines, num_warps}]
    buffer_count: dict[int, int]           # buffer id -> depth
    cross_wg_barriers: list[dict[str, Any]]  # verbatim barrier dicts

    def cost(self) -> tuple[int, int, int]:
        """Predicted cost, lower is better (lexicographic):
        (II, total prologue stage depth, -total buffering).
        Used as the no-regression pre-filter vs the baseline.
        """
        sum_stage = sum(self.node_stage.values())
        sum_count = sum(self.buffer_count.values())
        return (self.II, sum_stage, -sum_count)


@dataclass
class Solution:
    loops: dict[int, LoopSolution] = field(default_factory=dict)
    # Graph-level: emit eager MMA SMEM-operand release. Set by the solver only
    # when its memory-aware analysis proves every MMA operand ring is
    # double-buffered (count>=2), which hides the MMA read latency.
    eager_smem_release: bool = False

    def cost(self) -> tuple:
        # Concatenate per-loop costs in loop-id order for a total order.
        parts: list[int] = []
        for lid in sorted(self.loops):
            parts.extend(self.loops[lid].cost())
        return tuple(parts)

    def is_better_than(self, other: "Solution") -> bool:
        return self.cost() < other.cost()
