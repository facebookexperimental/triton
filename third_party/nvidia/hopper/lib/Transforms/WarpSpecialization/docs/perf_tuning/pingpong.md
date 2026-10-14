# Ping-pong (`pingpongAutoWS`, autoWS GEMM, sm100)

Scope: when `pingpongAutoWS` can change a GEMM kernel on B200, which configs
it silently leaves unchanged, and what it trades. How regions are formed and
synchronized is in [PingPongScheduling.md](../PingPongScheduling.md).

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
| the two partitions' ops do not alternate cleanly in the loop schedule | silently ignored | "Skipping group due to invalid pingpong schedule pattern" | `WS/PingPong.cpp`, `arrivesFirst`, `doPingPongPrep` |

## Mechanism

**What it changes.**
- Each accepted region gets a two-element mbarrier group in SMEM. The two partitions wait/arrive around their critical regions so only one executes the expensive op at a time, with wait phases derived from the enclosing loops ([PingPongScheduling.md, "Barrier Allocation"](../PingPongScheduling.md#barrier-allocation); `WS/PingPong.cpp`, `doPingPongSync`).
- For SFU ops the region extends forward to the next op with memory effects ([PingPongScheduling.md, "Expensive Op Identification"](../PingPongScheduling.md#expensive-op-identification)).

**What it relieves.** Contention between two consumer partitions issuing SFU work at the same time.

**What it costs.**
- The two partitions' critical regions are serialized.
- Each region crossing adds a barrier wait/arrive per iteration.
- Two barriers of SMEM per region.

**Where a GEMM can trigger it.**
- In a GEMM, a second consumer partition comes from `data_partition_factor=2`, which gives two computation groups (see [data_partitioning.md](data_partitioning.md)).
- So a GEMM can only form a region when all of these hold:
  - dp=2;
  - an epilogue (or fused prologue) applies SFU math to 2-D tiles in both groups;
  - that code sits inside an `scf.for`, i.e. a static persistent schedule.

## When to try it (depends on the problem)

- **IF** the kernel is a GEMM/addmm with no SFU math in the epilogue, **THEN** skip `pingpongAutoWS=True`, **BECAUSE** no region forms and it is a duplicate config.
- **IF** `data_partition_factor` is 1, **THEN** skip `pingpongAutoWS=True`, **BECAUSE** a GEMM has no second consumer partition to alternate with.
- **IF** the tile schedule is CLC or dynamic (outer `scf.while`) and the SFU math is in the epilogue, **THEN** skip `pingpongAutoWS=True` (this removes it), **BECAUSE** the epilogue has no enclosing `scf.for`. SFU math inside the K loop can still form a region.
- **IF** dp=2 and the epilogue applies SFU math (`exp`, `tanh`, `rsqrt`, ...) to 2-D tiles under a static persistent schedule, **THEN** try `pingpongAutoWS` on and off (this keeps both), **BECAUSE** it only trades SFU contention against serialization.

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
