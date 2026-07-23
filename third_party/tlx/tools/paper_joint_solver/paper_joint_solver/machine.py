"""Machine model (paper sec 5: target-architecture description).

Capacities and per-op resource usage for Blackwell (B200).  Functional-unit
kinds mirror the DDG's pipeline classification; each has capacity 1 at the
whole-SM tile-op granularity the paper schedules at.  Costs (latency /
occupancy) come from the DDG dump, which is microbenchmark-calibrated — the
paper allows "documentation or direct measurement".

Memory capacities note: schedule_graph buffer accounting excludes the
emitter's fixed SMEM staging (K/V prologue tiles, epilogue c_smem, dq_smem —
measured ~115 KB on the FA backward) and the MMA-accumulator TMEM tiles.
MEMORYCAPACITY must be enforced against the *emitted* totals, so the model
exposes both the hardware capacity and a per-kernel fixed-overhead knob.
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
MAX_WARPGROUPS = 8  # scheduling slots for opw assignment


@dataclass
class MachineModel:
    capacities: dict[str, int] = field(
        default_factory=lambda: {p: 1 for p in PIPELINES})
    smem_bytes: int = SMEM_BYTES
    tmem_cols: int = TMEM_COLS
    # Fixed SMEM the emitter allocates outside the schedule-graph buffer list
    # (prologue staging, epilogue staging, reduce staging).  Per-kernel.
    smem_fixed_overhead: int = 0
    # Fixed TMEM columns for MMA accumulators (also outside the buffer list).
    tmem_fixed_cols: int = 0
    num_warpgroups: int = MAX_WARPGROUPS
    # Per-warp(group) register budget for REGISTERLIMIT.  The paper's
    # reduced-register run ("-LR") shrinks this.
    regs_per_warpgroup: int = REGS_PER_SM // 4
    # Cross-warp spill cost (paper: annotation; SMEM round-trip latency).
    spill_cost: int = 30

    def cap(self, pipeline: str) -> int:
        return self.capacities.get(pipeline, 0)
