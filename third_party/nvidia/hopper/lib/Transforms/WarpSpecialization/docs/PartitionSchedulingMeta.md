# Partition Scheduling Meta

This document covers the `PartitionSchedulingMeta` pass, which assigns partition
IDs to operations for warp specialization. This is the first pass in the AutoWS
pipeline — it determines which warp group each operation will execute on.

**File**: `PartitionSchedulingMeta.cpp`

## Overview

The pass walks all `scf.for` loops with the `tt.warp_specialize` attribute and
assigns each operation inside the loop (and post-loop consumers) to a
**partition**. Each partition maps to a warp group at runtime.

```
Phase 1: Categorize operations         (OpCategorizer + collectMMABackwardSlices)
Phase 2: Create partition layout       (createPartitionLayout with tuning knobs)
Phase 3: Schedule anchor ops           (loads, epilogue stores, MMAs)
Phase 4: Propagate users               (load users, correction, reductions)
Phase 5: Create computation partitions (per-MMA user scheduling)
Post:    propagatePartitions + schedulePostLoopOps + optimizeSchedule
         + splitDataPartitionedIfOps
```

## Tuning Knobs

Partition layout is controlled by `SchedulingOptions`, exposed as pass options
in `Passes.td`:

| Knob | Pass Option | Default | Effect |
|------|-------------|---------|--------|
| `mergeCorrection` | `--merge-correction` | false | Correction ops → computation[dpId] |
| `mergeEpilogue` | `--merge-epilogue-into-computation` | false | Non-store epilogue ops → correction/reduction/computation |
| `mergeReduction` | `--merge-reduction` | false | Reduction ops → computation[dpId] |
| `separateEpilogueStore` | `--separate-epilogue-store` | false | DescriptorStore → own 1-warp partition |

**`mergeEpilogue` routing**: When true, non-store epilogue ops go to the
correction partition (if it exists), else the reduction partition, else
computation[dpId].

### Target Partition Layouts

| Case | Knobs | Partitions |
|------|-------|------------|
| Blackwell FA fwd | default | correction, gemm, load, epilogue, comp×2 |
| Blackwell flex fwd | default (no epilogue) | correction, gemm, load, comp×2 |
| Blackwell FA bwd | default | reduction, gemm, load, epilogue, comp |
| Hopper FA fwd | mergeCorrection + mergeEpilogue | load, comp×2 |
| Simple GEMM | default | default, gemm, load, epilogue |

## Phase 1: Operation Categorization (`OpCategorizer`)

### Categories

| Category | Ops | Purpose |
|----------|-----|---------|
| `Load` | `DescriptorLoadOp`, `DescriptorGatherOp` | TMA loads |
| `MMA` | `MMAv5OpInterface`, `WarpGroupDotOp` | Tensor core operations |
| `MemDescView` | ops with `MemDescViewTrait` | Memory descriptor views feeding MMA |
| `EpilogueStore` | `DescriptorStoreOp`, `AsyncTMACopyLocalToGlobalOp` | Output stores |
| `TMAReduction` | `DescriptorReduceOp`, `AsyncTMAReduceOp` | Atomic reductions |
| `Correction` | Cross-iteration MMA users | Online softmax rescaling |
| `DataPartition` | Exclusive ops in one MMA's backward slice | Per-MMA-group computation |

### MMA Type Support

The pass supports both Blackwell and Hopper MMA types via the `isMMAOp()`
helper:
- **MMAv5** (`tc_gen5_mma`): Blackwell tensor cores. Gets its own `gemm`
  partition for TMEM-based accumulation.
- **WarpGroupDot** (`warp_group_dot`): Hopper tensor cores. No separate `gemm`
  partition — MMA ops go directly into computation partitions.

### Categorization Order

```
categorizeLoads()
categorizeMMAs()
categorizeEpilogueStores()
categorizeTMAReductions()
categorizeCorrectionOps()       ← runs before DataPartition
categorizeDataPartitionOps()    ← skips already-categorized ops
```

Correction runs before DataPartition so that correction ops (accumulator
rescaling) are not stolen by the data partition categorizer.

### Central dpId Assignment (`collectMMABackwardSlices`)

`collectMMABackwardSlices` is the single source of truth for data partition ID
(dpId) assignment. It:

1. **Collects backward slices** for each MMA, **entering `scf.if` regions**
   selectively — only following yield operands that correspond to results
   consumed by the current slice. This captures ops like `tmem_load QK` and
   `mulf(QK*scale)` in flex attention without pulling in ops from the other
   data partition.
2. **Groups dependent MMAs** via union-find. MMA B depends on MMA A if A's
   forward user set overlaps B's backward slice (e.g., QK MMA feeds PV MMA).
3. **Builds `opToDpId` map** for ALL reachable ops:
   - **Inner-loop ops**: From backward slices, using normalized group IDs.
     Ops appearing in multiple groups get `SHARED_DPID` sentinel.
   - **Pre-loop ops**: Following MMA operands backward across the loop
     boundary (Q loads, allocs).
   - **Post-loop ops**: Following loop results forward to post-loop consumers
     (descriptor stores, normalization).

All `categorize*` functions look up dpId from `opToDpId` via `addCategorizedOp`,
which auto-resolves the dpId when not explicitly provided.

### Data Partition Factor Detection

1. **Collect backward slices** for each MMA.
2. **Identify shared ops** — ops appearing in multiple slices.
3. **Union-find grouping** — MMAs whose forward user sets overlap another MMA's
   backward slice are grouped together.
4. **Count groups with exclusive ops** — only groups with at least one
   non-shared, non-constant op count. This becomes `dataPartitionFactor`.

For FA forward with `data_partition_factor=2`, this yields `dpFactor=2`.

## Phase 2: Partition Layout (`createPartitionLayout`)

Creates partitions based on the categorizer results and `SchedulingOptions`.
Replaces the old template system (`UnifiedFATemplate`, `GEMMTemplate`,
`selectTemplate`).

Partition creation order determines the partition index. The first partition
created gets index 0, which becomes the "default" warp group in
`tt.warp_specialize` (receives 4 warps):

1. **Default** — created first when no correction/reduction exists (GEMM case).
   Holds uncategorized ops (post-loop tmem_load, truncf, etc.).
2. **Correction** — when `!mergeCorrection && hasCorrection`. Serves as default
   for FA/flex (shared ops, load users go here). Created first → index 0.
3. **Reduction** — when `!mergeReduction && hasReduction`. Serves as default for
   bwd. Created first → index 0.
4. **Gemm** — only when MMAv5 ops exist (Blackwell). Hopper `warp_group_dot`
   is not MMAv5, so no gemm partition is created for Hopper.
5. **Load** — always.
6. **Epilogue** — when `!mergeEpilogue && hasEpilogue`.
7. **Epilogue store** — when `separateEpilogueStore && hasEpilogue`. Gets 1 warp.
8. **Computation** — created dynamically in Phase 5 per data partition.

When correction or reduction exists, it serves as the default partition (shared
ops, load users route there). When merged (`mergeCorrection=true`), no
correction partition is created and those ops go to computation[dpId].

## Phase 3–5: Partition Assignment

### Phase 3: Anchor Ops

1. **Loads** → `load` partition. Includes `LocalAllocOp` users with matching
   shared encoding and `TMEMAllocOp` users.
2. **Epilogue stores** → `epilogue` partition (when it exists).
3. **MMAs** → `gemm` partition (MMAv5 only). Non-MMAv5 MMAs (WarpGroupDot) are
   left for Phase 5 where they go to computation partitions.
4. **MemDesc views** → `gemm` partition (MMAv5 only). Skipped when no gemm
   partition exists.

### Phase 4: Propagate Users

1. **Load users** → default/correction partition.
2. **Correction ops** → correction partition (+ `scheduleUsers` for transitive users).
3. **TMA reduction ops** → reduction partition (+ backward slice producers).

### Phase 5: Computation Partitions

Pre-assigns `DataPartition`-categorized ops to per-dpId computation partitions,
then iterates over MMAs:

- **Pre-assigned MMAs** (PV MMAs): Use the pre-assigned computation partition.
- **Non-pre-assigned MMAs** (QK MMAs): Look up dpId from `opToDpId` to find the
  correct existing computation partition. This prevents creating extra partitions.
- **Non-MMAv5** (Hopper): MMA ops themselves are scheduled into the computation
  partition (not gemm, since no gemm partition exists).

## Post-Processing

### `propagatePartitions`

Handles unscheduled ops by forming clusters and assigning them based on their
def-use relationships to already-scheduled partitions. When
`createComputePartitions=false` (Hopper with all merges), unscheduled clusters
merge into existing computation partitions instead of creating new ones.

### `schedulePostLoopOps`

Schedules post-loop operations. Epilogue store ops go to the epilogue partition.
Non-store post-loop ops go to the default partition. When no default or epilogue
exists (Hopper with all merges), falls back to the first computation partition.

### `optimizeSchedule`

Clones `BroadcastOp` and `ExpandDimsOp` into each partition that has users.

### `splitDataPartitionedIfOps`

Splits `scf.if` ops whose results feed different computation partitions into
separate per-partition `scf.if` ops. Required for flex attention masking where
a single `scf.if` yields values for both data partitions.

## Partition Type Summary

For FA forward with `dpFactor=2` (Blackwell):
```
partition 0: correction  — correction ops, load users (shared computation)
partition 1: gemm        — MMA operations + mem desc views
partition 2: load        — TMA loads + associated allocs
partition 3: epilogue    — descriptor stores (+ post-loop ops)
partition 4: computation — MMA user group 0
partition 5: computation — MMA user group 1
```

For flex attention forward with `dpFactor=2` (Blackwell):
```
partition 0: correction  — correction ops, load users, sparse indexing
partition 1: gemm        — MMA operations + mem desc views
partition 2: load        — TMA loads + associated allocs
partition 3: computation — MMA user group 0 (includes QK tmem_load + scale)
partition 4: computation — MMA user group 1 (includes QK tmem_load + scale)
```

For FA forward with `dpFactor=2` (Hopper, mergeCorrection + mergeEpilogue):
```
partition 0: load        — TMA loads + associated allocs
partition 1: computation — MMA group 0 (QK + PV + softmax + correction + epilogue)
partition 2: computation — MMA group 1 (QK + PV + softmax + correction + epilogue)
```

For GEMM (no correction/reduction):
```
partition 0: default     — uncategorized ops (post-loop tmem_load, truncf)
partition 1: gemm        — MMA operations + mem desc views
partition 2: load        — TMA loads + associated allocs
partition 3: epilogue    — descriptor stores
```

## Debug

- `TRITON_LLVM_DEBUG_ONLY="tritongpu-partition-scheduling"` enables debug logging.
- The categorizer prints all ops grouped by category with dpId.
- `createPartitionLayout` logs which partitions are created.
- Phase 5 logs MMA processing with dpId and pre-assignment status.
