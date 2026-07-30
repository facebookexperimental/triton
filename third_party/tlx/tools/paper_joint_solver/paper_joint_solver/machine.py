"""Machine model (paper sec 5: target-architecture description).

Capacities and per-op resource usage for Blackwell (B200).  Functional-unit
kinds mirror the DDG's pipeline classification; each has capacity 1 at the
whole-SM tile-op granularity the paper schedules at.  Costs (latency /
occupancy) come from the DDG dump, which is microbenchmark-calibrated — the
paper allows "documentation or direct measurement".

Memory capacities note: the resource model accounts for storage objects named
by the loop DDG, including MMA accumulators. Function-scope staging that is not
represented in that graph must be supplied through the per-kernel fixed
overhead knobs.
"""

from dataclasses import dataclass, field

# Functional units of the machine description D.  TMEM load/store ports are
# distinct hardware from the general-purpose ALUs on Blackwell, so they get
# their own unit (the dump folds them into CUDA, which would fully saturate
# the ALU column and make MinII packing artificially infeasible).
PIPELINES = ("TC", "TMA", "TMEM", "SFU", "CUDA")  # NONE ops use no resource

SMEM_BYTES = 232448  # B200 max dynamic SMEM per CTA
TMEM_COLS = 512
REGS_PER_SM = 65536  # 32-bit registers per SM (per CTA at 1 CTA/SM)
WARP_SIZE = 32
REGS_PER_THREAD = 255
MAX_PHYSICAL_WARPS = 32
GROUP_WIDTHS = (1, 2, 4, 8)
TMEM_COLUMN_BYTES = 128 * 4
MBARRIER_PAIR_BYTES = 16  # one 8-byte full + one 8-byte empty barrier


def required_group_width(min_warps: int) -> int:
    for width in GROUP_WIDTHS:
        if min_warps <= width:
            return width
    raise ValueError(f"operation requires unsupported warp width {min_warps}")


def physical_group_width(
    active_warps: int, supported_widths: tuple[int, ...]
) -> int:
    if active_warps < 1:
        raise ValueError("a used physical group must contain at least one warp")
    for width in sorted(supported_widths):
        if active_warps <= width:
            return width
    raise ValueError(
        f"{active_warps} active warps exceed supported group widths "
        f"{sorted(supported_widths)}"
    )


def register_allocation_words(value_words: int, width: int) -> int:
    return -(-value_words // width) * width


@dataclass
class MachineModel:
    capacities: dict[str, int] = field(
        default_factory=lambda: {p: 1 for p in PIPELINES})
    smem_bytes: int = SMEM_BYTES
    tmem_cols: int = TMEM_COLS
    # Fixed SMEM the emitter allocates outside the schedule-graph buffer list
    # (prologue staging, epilogue staging, reduce staging).  Per-kernel.
    smem_fixed_overhead: int = 0
    # Fixed function-scope TMEM columns not represented in the loop DDG.
    tmem_fixed_cols: int = 0
    # Group labels are an emitter-facing representation of sets of physical
    # warps. The default label pool cannot restrict the paper's physical set.
    num_warpgroups: int = MAX_PHYSICAL_WARPS
    num_warps: int = MAX_PHYSICAL_WARPS
    # Physical warps used by function-scope/default work absent from the DDG.
    warp_fixed_overhead: int = 0
    group_widths: tuple[int, ...] = GROUP_WIDTHS
    regs_per_sm: int = REGS_PER_SM
    regs_fixed_overhead: int = 0
    regs_per_warp: int = WARP_SIZE * REGS_PER_THREAD
    tmem_column_bytes: int = TMEM_COLUMN_BYTES
    barrier_pair_bytes: int = MBARRIER_PAIR_BYTES
    # Cross-warp spill cost (paper: annotation; SMEM round-trip latency).
    spill_cost: int = 30

    def cap(self, pipeline: str) -> int:
        return self.capacities.get(pipeline, 0)

    def scheduled_warp_budget(self, total_warps: int | None = None) -> int:
        total = self.num_warps if total_warps is None else total_warps
        if total < 1 or total > self.num_warps:
            raise ValueError(
                f"physical warp budget {total} is outside "
                f"[1, {self.num_warps}]"
            )
        if not 0 <= self.warp_fixed_overhead < total:
            raise ValueError(
                "fixed warp overhead must leave at least one physical warp "
                "for the scheduled DDG"
            )
        return total - self.warp_fixed_overhead

    def registers_fit(
        self,
        group_values: dict[int, list[tuple[int, frozenset[int]]]],
        group_widths: dict[int, int],
    ) -> bool:
        total = 0
        for group, values in group_values.items():
            group_width = group_widths[group]
            if any(
                not lanes or max(lanes) >= group_width for _, lanes in values
            ):
                return False
            for lane in range(group_width):
                per_warp = sum(
                    -(-value // len(lanes))
                    for value, lanes in values
                    if lane in lanes
                )
                if per_warp > self.regs_per_warp:
                    return False
            total += sum(
                register_allocation_words(value, len(lanes))
                for value, lanes in values
            )
        return total + self.regs_fixed_overhead <= self.regs_per_sm
