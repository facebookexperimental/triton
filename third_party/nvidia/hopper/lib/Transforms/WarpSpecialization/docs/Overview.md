# AutoWS Overview

Automatic Warp Specialization (AutoWS) is a compiler optimization that
partitions a kernel's operations into specialized warp groups — typically a
**producer** group that handles memory loads and a **consumer** group that
handles computation (MMA/tensor core ops). By assigning different hardware
resources to each group, warp specialization enables overlap of memory
transfers, CUDA core work, and tensor core work, improving SM utilization.

## Pipeline

The AutoWS pipeline is defined in the adjacent `WarpSpecialization.cpp`. It
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
  → doLoopSchedulePreprocessing + scheduleLoops  (external, not in this directory)
```

On Blackwell, only `doTaskIdPropagate` runs for annotation (task partition and
data partition are skipped). The task assignments are expected to come from
an earlier partition scheduling pass (`PartitionSchedulingMeta`).

## File Map

| File | Function / Pass | Description |
|------|----------------|-------------|
| `WarpSpecialization.cpp` | `NVGPUWarpSpecialization` | Top-level pipeline orchestration |
| `PartitionSchedulingMeta.cpp` | `nvgpu-partition-scheduling-meta` | Partition scheduling for Blackwell (assigns `ttg.partition` attributes) |
| `WSTaskPartition.cpp` | `doTaskPartition` | Assigns `async_task_id` to anchor ops (loads, dots, stores) — Hopper only |
| `TaskIdPropagation.cpp` | — | `TaskIdBackwardPropagation` sparse dataflow analysis |
| `WSTaskIdPropagate.cpp` | `doTaskIdPropagate` | Runs analysis and materializes task IDs |
| `WSDataPartition.cpp` | `doDataPartition` | Splits ops along M/N dimensions across warp groups — Hopper only |
| `PingPong.cpp` | `doPingPongPrep` / `doPingPongSync` | Named barrier insertion for ping-pong scheduling |
| `WSCodePartition.cpp` | `doBufferAllocation` | Channel discovery and SMEM/TMEM allocation hoisting (pre-pass) |
| `WSBuffer.cpp` | `appendAccumCntsForOps` | Accumulation counter infrastructure for multi-buffer indexing |
| `WSMemoryPlanner.cpp` | `doMemoryPlanner` | Plans SMEM and TMEM allocation (multi-buffering, liveness) |
| `WSCodePartition.cpp` | `doCodePartitionPost` | Creates channels, inserts async copies and barriers |
| `WSLowerMem.cpp` | — | Memory lowering: async copies between global/shared/tensor memory |
| `WSSpecialize.cpp` | `specializeRegion` | Clones ops into `ttg.WarpSpecializeOp` regions |
| `WSLowerToken.cpp` | `doTokenLowering` | Lowers `ProducerAcquireOp`/`ConsumerWaitOp` to hardware barriers |
| `WSTMAStoreLowering.cpp` | `doTMAStoreLowering` | Pre-pass lowering of `tt.descriptor_store` for WS visibility |
| `WSTMAStoreLowering.cpp` | `doAnnotateTMAStoreWaits` | Annotate TMA store waits with multi-buffer rotation count |
| `WSTMAStoreLowering.cpp` | `doValidateTMAStoreAnnotations` | Safety check: strip invalid annotations |
| `WSTMAStoreLowering.cpp` | `doTMAStoreWaitReorder` | Reschedule TMA store waits using SWP CoarseSchedule |
| `TMEMAlloc1D.cpp` | `TMEM1DAllocator` | 1D tensor memory allocation for cross-partition values |
| `CodePartitionUtility.cpp` | — | Channel data structures, operand D handling, barrier fusion, buffer management |
| `Utility.cpp` | — | `AsyncTaskId` helpers, `OpBuilderWithAsyncTaskIds` |

### Headers

| File | Description |
|------|-------------|
| `Utility.h` | `AsyncTaskId` typedef, `OpBuilderWithAsyncTaskIds`, `LoopScheduleInfo`, task ID helpers |
| `TaskIdPropagation.h` | `TaskId` lattice, `TaskIdLattice`, `TaskIdBackwardPropagation` analysis |
| `CodePartitionUtility.h` | `Channel`, `ChannelPost`, `TmemDataChannel`, `TmemDataChannelPost`, `ReuseGroup`, `ReuseConfig`, `CommChannel` |
| `TMEMUtils.h` | `TMEM1DAllocator`, `sliceAndReinterpretMDTMEM`, `createTMEMDesc` |
| `WSBarrierAnalysis.h` | `WSBarrierAttr`, `buildChannelGraph`, `buildWSBarrierOrderedRegionRanges`, `injectChannelGraph` — channel graph and ordered-region construction for barrier constraints |
| `nvidia/hopper/include/Transforms/WSBarrierReorder.h` | `canAdvanceWSBarrier`, `canAdvanceWSBarrierArrivePastWait`, `sinkWSArrives`, `raiseWSWaits`, `buildBarrierToMemoryOpMap`, `optimizeWSBarrierLocations` — barrier reordering utilities consumed by `InterleaveTMem` |

## Glossary

| Term | Definition |
|------|-----------|
| **Partition** | A group of operations assigned to run on the same warp group. Identified by a partition ID (integer). |
| **Async Task** | Synonym for partition. Identified by `async_task_id` attribute on ops. |
| **Channel** | A producer-consumer data dependency between partitions. Can be SMEM-backed (`ChannelPost`) or TMEM-backed (`TmemDataChannelPost`). |
| **Reuse Group** | A set of channels sharing a single physical buffer (`buffer.id`). See [ReuseGroups.md](ReuseGroups.md). |
| **Multi-buffering** | Allocating N copies of a buffer so the producer can fill copy N+1 while the consumer reads copy N. Controlled by `buffer.copy`. |
| **Operand D** | The MMA accumulator — the TMEM allocation that both receives MMA output and carries accumulated results across loop iterations. |
| **Ping-pong** | Named-barrier-based mutual exclusion between two consumer partitions executing expensive ops. |
| **Stage / Phase** | Pipeline stage index (which buffer slot) and phase (parity bit for mbarrier wait/arrive). |
| **Token** | Abstract synchronization primitive (`CreateTokenOp`) that is lowered to hardware mbarrier pairs. |
| **AccumCnt** | Accumulation counter — a loop-carried value that tracks the current buffer slot for multi-buffered channels. |

## Further Reading

- [Task Partitioning & ID Propagation](TaskPartitionAndPropagation.md) — how ops are assigned to partitions
- [Data Partitioning](DataPartition.md) — splitting tensor dimensions across consumer warp groups
- [Code Partitioning](CodePartition.md) — channel discovery, buffer creation, sync insertion
- [Code Specialization](CodeSpecialization.md) — how ops are cloned into WarpSpecializeOp regions
- [Memory Lowering](MemoryLowering.md) — async copy creation and TMA store lowering
- [Token & Barrier Lowering](TokenBarrierLowering.md) — lowering abstract tokens to hardware mbarriers
- [Buffer Allocation](BufferAllocation.md) — channel discovery and SMEM/TMEM allocation hoisting
- [Accumulation Counters](AccumulationCounters.md) — accumulation counter infrastructure for multi-buffering
- [Operand D Handling](OperandDHandling.md) — MMA accumulator lifecycle through WS
- [TMEM Allocation Heuristics](TMEMAllocationHeuristics.md) — TMEM memory planning algorithms
- [SMEM Allocation Design](SmemAllocationDesign.md) — SMEM budget-aware allocation
- [Barrier Fusion](BarrierFusion.md) — TMA fusion, tcgen05_commit combining
- [Barrier Constraints](BarrierConstraints.md) — generic barrier constraints and WSBarrier reordering metadata
- [WS Barrier Ordered Region Tracking](WSBarrierOrderedRegionTracking.md) — V2 ordered-region metadata for overlapping channel graphs
- [Reuse Groups](ReuseGroups.md) — buffer sharing mechanics
- [Ping-Pong Scheduling](PingPongScheduling.md) — named barrier insertion for expensive ops
- [Utilities](Utilities.md) — `OpBuilderWithAsyncTaskIds`, task ID helpers, location utilities
- [Memory Planner Visualization](MemoryPlannerVisualization.md) — debug DOT graph tools
- [TMA Store Wait Pipeline](TMAStoreWaitPipeline.md) — annotation, reordering, and lowering of TMA store waits
