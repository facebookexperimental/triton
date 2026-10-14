# Ping-pong (`pingpongAutoWS`, autoWS GEMM, sm100)

Scope: when `pingpongAutoWS` can change a GEMM kernel on B200, which configs
it silently leaves unchanged, and what it trades. How regions are formed and
synchronized is in [PingPongScheduling.md](../PingPongScheduling.md). Ping-pong only matters for fusions with transcendental (SFU) ops, such as softmax or attention epilogues; it has no effect on a raw GEMM and is out of scope for v0.

Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` (`WS/`) unless they start at the repository root.

## Knobs

| Knob | How set | What it does | Existing doc |
|---|---|---|---|
| `pingpongAutoWS` | compile option (`triton.Config(pingpongAutoWS=True)`, default False) | Makes two consumer partitions take turns on an "expensive" op | [PingPongScheduling.md, "Pipeline Integration"](../PingPongScheduling.md#pipeline-integration) |

## Never try (any problem)

Every condition below leaves the kernel unchanged with no diagnostic (debug log only).

| Condition | Status | Why | Code |
|---|---|---|---|
| no SFU op with a rank > 1 tensor result in two partitions (e.g. a plain GEMM or addmm) | silently ignored | On sm_100 the only expensive ops are SFU math (`math.exp`, `exp2`, `sin`, `cos`, `tanh`, `sqrt`, `rsqrt`, or pure SFU `.approx` inline asm) on 2-D+ tensors; MMAs are never ping-ponged | `WS/PingPong.cpp`, `CriticalRegionManager::isExpensiveOp`, `isSFUOp` |
| compute capability other than 90 or 100 | silently ignored | `isExpensiveOp` only has cases for 90 (wgmma) and 100 (SFU) | `WS/PingPong.cpp`, `CriticalRegionManager::isExpensiveOp` |
| the expensive-op group spans 1 or more than 2 partitions | silently ignored | "Only handle pingpong for the case of 2 different partitions" | `WS/PingPong.cpp`, `doPingPongPrep` |
| the expensive ops have no enclosing `scf.for` (e.g. an epilogue directly in a dynamic or CLC outer `while`) | silently ignored | "No parent ForOp found, skipping this critical region" | `WS/PingPong.cpp`, `doPingPongPrep` |
| the ops differ in op name, sit in different blocks or across control flow, or have a real memory effect between them in one partition | silently ignored | The ops are not grouped (group formation rules) | `WS/PingPong.cpp`, `doPingPongPrep`; [PingPongScheduling.md, "Step 1: Group Expensive Ops"](../PingPongScheduling.md#step-1-group-expensive-ops) |
| `pingpongAutoWS=True` on a GEMM, even with every condition below met (static persistent loop, dp=2 at `BLOCK_M=256`, a pure SFU op with a 2-D result, any `merge_epilogue` × `separate_epilogue_store` combination) | duplicate config (reproduced) | Both data-partition halves' SFU ops land in one partition, so no region has two partitions and the TTGIR is identical with pingpong on or off | `WS/PingPong.cpp`, `doPingPongPrep`; `WS/PartitionSchedulingMeta.cpp` |
| the two partitions' ops do not alternate cleanly in the loop schedule | silently ignored | "Skipping group due to invalid pingpong schedule pattern" | `WS/PingPong.cpp`, `arrivesFirst`, `doPingPongPrep` |

## Mechanism

**What it changes.**
- Each accepted region gets a two-element mbarrier group in SMEM; the two partitions wait/arrive around their critical regions so only one executes the expensive op at a time, with wait phases derived from the enclosing loops ([PingPongScheduling.md, "Barrier Allocation"](../PingPongScheduling.md#barrier-allocation); `WS/PingPong.cpp`, `doPingPongSync`).
- For SFU ops the region extends forward to the next op with memory effects ([PingPongScheduling.md, "Expensive Op Identification"](../PingPongScheduling.md#expensive-op-identification)).

**What it relieves.** Contention between two consumer partitions issuing SFU work at the same time.

**What it costs.** Serialized critical regions; a barrier wait/arrive per region crossing per iteration; two barriers of SMEM per region.

**Where a GEMM can trigger it.**
- A GEMM's second consumer partition comes from `data_partition_factor=2` (two computation groups; see [data_partitioning.md](data_partitioning.md)).
- Documented conditions: dp=2, SFU math on 2-D tiles in both groups, inside an `scf.for` (static persistent). Not enough on this build: both data-partition halves sit in the default region (see [warps_and_registers.md](warps_and_registers.md#mechanism)), so their SFU ops share one partition and no region forms.

## Interactions

See [interactions.md](interactions.md). Pairs these rules depend on:
- `pingpongAutoWS` × `data_partition_factor` (see [data_partitioning.md](data_partitioning.md))
- `pingpongAutoWS` × tile scheduler (see [clc_persistence.md](clc_persistence.md))
- `pingpongAutoWS` × epilogue fusion (SFU math)
- `pingpongAutoWS` × `TRITON_PROMOTE_MBAR_TO_NAMED_BARRIER`

## Hazards

None known; see [hazards.md](hazards.md).

## Not yet established

See [TRACKER.md](TRACKER.md):
- Whether ping-pong ever pays on a GEMM epilogue with SFU math and dp=2. No measurement exists.
