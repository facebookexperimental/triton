"""Machine model + tunables for the Twill-inspired solver.

Defaults mirror the values the C++ modulo scheduler dumps in ``ddg.json``'s
``config`` block (``smem_budget_bytes``, ``tmem_budget_bytes``). When a case's
``ddg.json`` is present those per-case values override these.
"""

from __future__ import annotations

from dataclasses import dataclass

# Hardware / resource budgets (Blackwell B200 defaults; overridden from ddg.json).
DEFAULT_SMEM_BUDGET_BYTES = 233472  # 228 KiB usable SMEM
DEFAULT_TMEM_BUDGET_BYTES = 262144  # 128 rows x 512 cols x 4 B  ==  512 TMEM columns
TMEM_BYTES_PER_COL = 512  # 128 rows x 4 B/f32 ; cols = tmem_bytes // this

# Registers/thread available per warp-count for a warp group, from the C++
# partitioner's regsForWarpCount table (ModuloSchedulePass.cpp). 2 interpolated.
REGS_FOR_WARP_COUNT = {1: 24, 2: 88, 4: 152, 8: 232}
HW_REG_LIMIT_PER_THREAD = 255  # ptxas hard cap

# Cross-warp spill latency: transferring a register value producer->consumer on
# different warps goes through SMEM + an mbarrier round-trip. Tie to the
# local_alloc / mbarrier wait cost (~30 cyc). Tunable.
DELTA_SPILL = 30

# Buffer multi-buffering depth cap (Twill exposes streaming pipeline depth as a
# tunable; we bound the search).
MAX_BUFFER_COUNT = 6
MIN_BUFFER_COUNT = 1

# Max warp groups the emitter/HW can express on Hopper/Blackwell.
MAX_WARP_GROUPS = 6
WARP_COUNT_CHOICES = (1, 2, 4, 8)

# Ascending-II search: how far above MinII to try before giving up (fall back to
# baseline). Every example here is Rec/Res-MII-bound so MinII normally succeeds.
II_SEARCH_CAP = 8

# Default CP-SAT wall-clock budget per solve (seconds).
DEFAULT_TIME_LIMIT_S = 10.0


@dataclass
class MachineModel:
    smem_budget_bytes: int = DEFAULT_SMEM_BUDGET_BYTES
    tmem_budget_bytes: int = DEFAULT_TMEM_BUDGET_BYTES
    delta_spill: int = DELTA_SPILL
    max_buffer_count: int = MAX_BUFFER_COUNT

    def reg_budget(self, num_warps: int) -> int:
        """Registers/thread available to a warp group of ``num_warps`` warps."""
        return REGS_FOR_WARP_COUNT.get(num_warps, HW_REG_LIMIT_PER_THREAD)
