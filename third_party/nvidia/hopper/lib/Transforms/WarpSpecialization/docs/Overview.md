# AutoWS Overview

Automatic Warp Specialization (AutoWS) is a compiler optimization that
partitions a kernel's operations into specialized warp groups — typically a
**producer** group that handles memory loads and a **consumer** group that
handles computation (MMA/tensor core ops). By assigning different hardware
resources to each group, warp specialization enables overlap of memory
transfers, CUDA core work, and tensor core work, improving SM utilization.

## Two Pipelines

There are two implementations of AutoWS in the codebase:

### New Pipeline (`TritonGPUAutomaticWarpSpecialization`)

The upstream-style pipeline, defined in
`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp`.
It composes an ordered sub-pipeline of modular passes:

```
PartitionScheduling
  → InsertAref
  → LoadMMASpecialization    (currently active; InsertTmemAref is #if 0'd)
  → RewritePartitionDependencies
  → SCCP + CSE
  → LowerAref
  → PartitionLoops
  → LowerWarpGroup
  → [canonicalization cleanup]
  → OptimizePartitionWarps
  → ScheduleLoops
```

Bails out if the module contains manual TLX warp specialization ops
(`tlx.async_tasks`).

### Legacy Pipeline (`NVGPUWarpSpecialization`)

The Meta-internal pipeline, defined in
`third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp`. It
orchestrates sub-passes as function calls within a single monolithic pass:

```
doTaskPartition          (Hopper only; skipped on Blackwell)
  → doTaskIdPropagate
  → doDataPartition      (Hopper only; skipped on Blackwell)
  → doPingPongPrep       (optional, if pingpongAutoWS is set)
  → doBufferAllocation
  → doMemoryPlanner
  → doCodePartitionPost
  → doPingPongSync       (optional)
  → doTokenLowering
  → doLoopSchedulePreprocessing + scheduleLoops
```

On Blackwell, only `doTaskIdPropagate` runs for annotation (task partition and
data partition are skipped). The task assignments are expected to come from
an earlier partition scheduling pass.

### When Each Pipeline Is Used

Selection happens in `third_party/nvidia/backend/compiler.py`:

| GPU | Condition | Pipeline |
|-----|-----------|----------|
| Hopper (SM80/SM90) | Always | Legacy (`add_hopper_warpspec`) |
| Blackwell (SM100+) | `use_meta_ws = False` | New (`add_warp_specialize`) |
| Blackwell (SM100+) | `use_meta_ws = True` | Legacy (`add_hopper_warpspec`), preceded by `add_partition_scheduling` or `add_partition_scheduling_meta` |
| Any | Module has TLX `async_tasks` | Neither (manual WS path via TLX) |

The `use_meta_ws` knob is in `python/triton/knobs.py`. When `use_meta_partition`
is also set, the Meta variant of partition scheduling is used.

## File Map

### Core Infrastructure (`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/`)

| File | Pass / Component | Description |
|------|-----------------|-------------|
| `AutomaticWarpSpecialization.cpp` | `TritonGPUAutomaticWarpSpecialization` | Top-level new pipeline orchestration |
| `PartitionScheduling.cpp` | `tritongpu-partition-scheduling` | Assigns ops to partitions (load/MMA/default/epilogue) |
| `LoadMMASpecialization.cpp` | `tritongpu-load-mma-specialization` | Creates multi-buffered TMEM for MMA, handles TMA load lowering with barrier fusion |
| `RewritePartitionDependencies.cpp` | `tritongpu-rewrite-partition-dependencies` | Rewrites cross-partition SSA deps through shared memory |
| `PartitionLoops.cpp` | `tritongpu-partition-loops` | Splits scheduled loops into `ttg.warp_specialize` regions |
| `OptimizePartitionWarps.cpp` | `tritongpu-optimize-partition-warps` | Adjusts warp counts per partition based on register pressure |
| `Partition.cpp` | — | `Partition`/`PartitionSet` data structures and iteration utilities |
| `PartitionBuilder.cpp` | — | Builder helpers for creating ops with partition metadata |

### NVIDIA Hopper Backend (`third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/`)

| File | Function / Pass | Description |
|------|----------------|-------------|
| `WarpSpecialization.cpp` | `NVGPUWarpSpecialization` | Top-level legacy pipeline orchestration |
| `WSTaskPartition.cpp` | `doTaskPartition` | Assigns `async_task_id` to anchor ops (loads, dots, stores) |
| `TaskIdPropagation.cpp` | — | `TaskIdBackwardPropagation` sparse dataflow analysis |
| `WSTaskIdPropagate.cpp` | `doTaskIdPropagate` | Runs analysis and materializes task IDs |
| `WSDataPartition.cpp` | `doDataPartition` | Splits ops along M/N dimensions across warp groups |
| `WSBuffer.cpp` | `doBufferAllocation` | Creates SMEM/TMEM buffers for register channels |
| `WSMemoryPlanner.cpp` | `doMemoryPlanner` | Plans SMEM and TMEM allocation (multi-buffering, liveness) |
| `WSCodePartition.cpp` | `doCodePartitionPost` | Generates warp-specialized code with channels and barriers |
| `WSSpecialize.cpp` | `specializeRegion` | Duplicates ops into `ttg.WarpSpecializeOp` regions |
| `WSLowerToken.cpp` | `doTokenLowering` | Lowers `ProducerAcquireOp`/`ConsumerWaitOp` to hardware barriers |
| `WSLowerMem.cpp` | — | Memory lowering: async copies between global/shared memory |
| `PingPong.cpp` | `doPingPongPrep` / `doPingPongSync` | Named barrier insertion for ping-pong scheduling |
| `TMEMAlloc1D.cpp` | `TMEM1DAllocator` | 1D tensor memory allocation for cross-partition values |
| `WSTMAStoreLowering.cpp` | `doTMAStoreLowering` | Pre-pass lowering of `tt.descriptor_store` for WS visibility |
| `PartitionSchedulingMeta.cpp` | `nvgpu-partition-scheduling-meta` | Meta's fork of partition scheduling |
| `CodePartitionUtility.cpp` | — | Channel data structures, operand D handling, buffer management |
| `Utility.cpp` | — | `AsyncTaskId` helpers, `OpBuilderWithAsyncTaskIds` |

### NVWS Dialect (`third_party/nvidia/lib/Dialect/NVWS/Transforms/`)

| File | Pass | Description |
|------|------|-------------|
| `InsertAref.cpp` | `nvws-insert-aref` | Creates async reference (aref) ops for cross-partition data flow |
| `InsertTmemAref.cpp` | `nvws-insert-tmem-aref` | Creates arefs for TMEM ownership transfer (currently disabled) |
| `LowerAref.cpp` | `nvws-lower-aref` | Lowers arefs to concrete mbarrier allocations; contains `combineArefs()` fusion |
| `AssignStagePhase.cpp` | — | Assigns stage/phase indices to aref operations |
| `LowerWarpGroup.cpp` | `nvws-lower-warp-group` | Lowers warp group abstractions |

### Headers

| File | Description |
|------|-------------|
| `include/triton/Dialect/TritonGPU/Transforms/Partition.h` | `Partition`, `PartitionSet`, partition attribute helpers |
| `include/triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h` | `PartitionBuilder`, `StageCluster` |
| `include/triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h` | `rewritePartitionDependencies()`, `partitionLoop()` |
| `include/triton/Dialect/TritonGPU/Transforms/Passes.td` | TableGen for core AutoWS passes |
| `third_party/nvidia/hopper/include/Transforms/Passes.td` | TableGen for NVIDIA-specific passes |
| `third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td` | TableGen for NVWS dialect passes |
| Internal headers in `WarpSpecialization/` | `Utility.h`, `TaskIdPropagation.h`, `CodePartitionUtility.h`, `TMEMUtils.h` |

## Glossary

| Term | Definition |
|------|-----------|
| **Partition** | A group of operations assigned to run on the same warp group. Identified by a partition ID (integer). |
| **Async Task** | Synonym for partition in the legacy pipeline. Identified by `async_task_id` attribute on ops. |
| **Channel** | A producer-consumer data dependency between partitions. Can be SMEM-backed (`ChannelPost`) or TMEM-backed (`TmemDataChannelPost`). |
| **Aref (Async Reference)** | An abstraction in the new pipeline for cross-partition data flow. An aref enter/exit pair brackets a data transfer, with associated barriers. Lowered to concrete mbarrier ops by `LowerAref`. |
| **Reuse Group** | A set of channels sharing a single physical buffer (`buffer.id`). See [ReuseGroups.md](ReuseGroups.md). |
| **Multi-buffering** | Allocating N copies of a buffer so the producer can fill copy N+1 while the consumer reads copy N. Controlled by `buffer.copy`. |
| **Operand D** | The MMA accumulator — the TMEM allocation that both receives MMA output and carries accumulated results across loop iterations. |
| **Ping-pong** | Named-barrier-based mutual exclusion between two consumer partitions executing expensive ops. |
| **Stage / Phase** | Pipeline stage index (which buffer slot) and phase (parity bit for mbarrier wait/arrive). |

## Further Reading

- [Task Partitioning & ID Propagation](TaskPartitionAndPropagation.md) — how ops are assigned to partitions
- [Operand D Handling](OperandDHandling.md) — MMA accumulator lifecycle through WS
- [TMEM Allocation Heuristics](TMEMAllocationHeuristics.md) — TMEM memory planning algorithms
- [SMEM Allocation Design](SmemAllocationDesign.md) — SMEM budget-aware allocation
- [Barrier Fusion](BarrierFusion.md) — TMA fusion, tcgen05_commit, aref combining
- [Reuse Groups](ReuseGroups.md) — buffer sharing mechanics
- [Ping-Pong Scheduling](PingPongScheduling.md) — named barrier insertion for expensive ops
- [Memory Planner Visualization](MemoryPlannerVisualization.md) — debug DOT graph tools
