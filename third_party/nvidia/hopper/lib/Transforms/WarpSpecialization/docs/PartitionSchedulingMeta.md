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
Phase 5: Create computation partitions (per-MMA user scheduling + dpId assignment)
Post:    propagatePartitions + schedulePostLoopOps + optimizeSchedule
         + splitDataPartitionedIfOps
```

## Tuning Knobs

Partition layout is controlled by `SchedulingOptions`, exposed as pass options
in `Passes.td`:

| Knob | Pass Option | Default | Effect |
|------|-------------|---------|--------|
| `mergeCorrection` | `--merge-correction` | false | Correction ops → computation[dpId] |
| `mergeEpilogue` | `--merge-epilogue` | false | Epilogue ops → correction/reduction/computation |
| `mergeEpilogueToComputation` | `--merge-epilogue-to-computation` | false | Epilogue ops → computation[dpId] directly |
| `mergeReduction` | `--merge-reduction` | false | Reduction ops → computation[dpId] |
| `separateEpilogueStore` | `--separate-epilogue-store` | false | Epilogue store ops → own 1-warp partition |

Per-loop `tt.merge_epilogue` attribute overrides the `mergeEpilogue` pass option.

### Epilogue Terminology

Post-loop operations are split into two categories:

- **Epilogue ops**: Non-store post-loop operations (tmem_load acc, normalize,
  truncf, convert_layout). These are computation that must happen after the
  main loop before the final store.
- **Epilogue store ops**: Post-loop TMA store operations (DescriptorStoreOp,
  AsyncTMACopyLocalToGlobalOp). These write the final results to global memory.

The epilogue tuning knobs control where these go:

**`mergeEpilogue` routing**: When true, epilogue ops go to the correction
partition (if it exists), else the reduction partition, else computation[dpId].
This preserves the priority: correction > reduction > computation. Used by
FA forward where epilogue ops (normalize acc) belong in the correction
partition.

**`mergeEpilogueToComputation` routing**: When true, epilogue ops go directly
to computation[dpId], even if a correction or reduction partition exists. This
is used by FA backward where post-loop ops (tmem_load dK/dV, reshape, split,
truncf) are data-partitioned and should stay with their corresponding
computation partition rather than being merged into the reduction partition.

`mergeEpilogueToComputation` takes priority over `mergeEpilogue` when both are
set.

Epilogue store ops are independent of these knobs — they always go to
`epilogue_store` (when `separateEpilogueStore`) or `epilogue` partition.

### Target Partition Layouts

| Case | Knobs | Partitions |
|------|-------|------------|
| Blackwell FA fwd | mergeEpilogue + separateEpilogueStore | correction, gemm, load, epilogue_store, comp×2 |
| Blackwell FA bwd | mergeEpilogueToComputation (merge_epilogue=true) | reduction, gemm, load, computation |
| Blackwell flex fwd | mergeEpilogue | correction, gemm, load, comp×2 |
| Hopper FA fwd | mergeCorrection + mergeEpilogue | load, comp×2 |
| Simple GEMM | separateEpilogueStore | gemm, load, epilogue, epilogue_store |

## Phase 1: Operation Categorization (`OpCategorizer`)

### Categories

| Category | Ops | Purpose |
|----------|-----|---------|
| `Load` | `DescriptorLoadOp`, `DescriptorGatherOp` | TMA loads |
| `MMA` | `MMAv5OpInterface`, `WarpGroupDotOp` | Tensor core operations |
| `MemDescView` | ops with `MemDescViewTrait` | Memory descriptor views feeding MMA |
| `EpilogueStore` | `DescriptorStoreOp`, `AsyncTMACopyLocalToGlobalOp` | Epilogue store ops (TMA output stores) |
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
For FA backward, MMAs are data-dependent (QK feeds PV via the same accumulator),
so all MMAs group together → `dpFactor=1`.

## Phase 2: Partition Layout (`createPartitionLayout`)

Creates partitions based on the categorizer results and `SchedulingOptions`.
Replaces the old template system (`UnifiedFATemplate`, `GEMMTemplate`,
`selectTemplate`).

Partition creation order determines the partition index. The first partition
created gets index 0, which becomes the "default" warp group in
`tt.warp_specialize` (receives 4 warps):

1. **Correction** — when `!mergeCorrection && hasCorrection`. Serves as default
   for FA/flex (shared ops, load users go here). Created first → index 0.
2. **Reduction** — when `!mergeReduction && hasReduction`. Serves as default for
   bwd. Created first → index 0.
3. **Gemm** — only when MMAv5 ops exist (Blackwell). Hopper `warp_group_dot`
   is not MMAv5, so no gemm partition is created for Hopper.
4. **Load** — always.
5. **Epilogue** — when `!mergeEpilogue && !mergeEpilogueToComputation &&
   hasEpilogue`. Holds epilogue ops (non-store post-loop computation).
6. **Epilogue store** — when `separateEpilogueStore && hasEpilogue`. Gets 1
   warp. Holds epilogue store ops (TMA stores). When no separate epilogue store
   partition exists, epilogue store ops go to the epilogue partition instead.
7. **Computation** — pre-created in Phase 5 per data partition (reverse dpId
   order for consistent partition index assignment).

There is no dedicated "default" partition. Uncategorized ops (e.g., pre-loop
acc inits, shared ops, load users) that are not assigned by any phase are
routed to existing partitions with the fallback priority:
correction → reduction → epilogue → computation.

When merged (`mergeCorrection=true`), no correction partition is created and
those ops go to the next available partition in the fallback chain.

## Phase 3–5: Partition Assignment

### Phase 3: Anchor Ops

1. **Loads** → `load` partition. Includes `LocalAllocOp` users with matching
   shared encoding and `TMEMAllocOp` users.
2. **Epilogue store ops** → `epilogue_store` partition (when it exists), else
   follow the same routing as regular epilogue ops.
3. **MMAs** → `gemm` partition (MMAv5 only). Non-MMAv5 MMAs (WarpGroupDot) are
   left for Phase 5 where they go to computation partitions.
4. **MemDesc views** → `gemm` partition (MMAv5 only). Skipped when no gemm
   partition exists.

### Phase 4: Propagate Users

1. **Load users** → routed with the uncategorized op fallback priority:
   correction → reduction → epilogue → computation.
2. **Correction ops** → correction partition (+ `scheduleUsers` for transitive
   users).
3. **TMA reduction ops** → reduction partition (+ backward slice producers).

### Phase 5: Computation Partitions

Pre-creates computation partitions for each dpId that has `DataPartition`-
categorized ops (in reverse dpId order to match legacy partition index ordering).
Then iterates over MMAs:

- **Pre-assigned MMAs** (PV MMAs): Use the pre-assigned computation partition.
- **Non-pre-assigned MMAs** (QK MMAs): First check user partitions, then look up
  dpId from `opToDpId` to find the correct existing computation partition. This
  prevents creating extra partitions.
- **Non-MMAv5** (Hopper): MMA ops themselves are scheduled into the computation
  partition (not gemm, since no gemm partition exists).

### dpId-Based Inner-Loop Assignment

After Phase 5, some inner-loop ops may remain unscheduled (e.g., `l_ij` reduce,
`tmem_alloc` p, `l_i*alpha`, `l_i+l_ij`). These ops have dpIds but aren't
reached by `scheduleUsers` because they're downstream of correction ops
(already scheduled in Phase 4) whose use chains `scheduleUsers` skips.

For each unscheduled inner-loop op with a tensor result:
1. Look up dpId from `opToDpId`.
2. If no entry, **trace through operands** to find the dpId from an operand
   that IS in `opToDpId` or already assigned to a computation partition.
3. Assign to the corresponding `dpIdToPartition` computation partition.

Scalar integer ops (loop counters) and `scf.yield` are excluded from this
assignment since they are loop-control ops, not data-partition computation ops.

## Post-Processing

### `propagatePartitions`

Handles unscheduled ops by forming **clusters** — groups of adjacent
unscheduled ops connected via the SSA def-use graph. Each cluster tracks:

- **defPartitions**: Partitions of already-scheduled ops that feed into the
  cluster (upstream).
- **sinkPartitions**: Partitions of already-scheduled ops that consume the
  cluster's outputs (downstream).

**Nested loop visibility**: `iterateUsers` follows use chains into nested
inner loops to find partitioned consumers. When a captured value (e.g.,
`tt.splat` producing `tensor<!tt.ptr>`) is used inside a nested `scf.for`,
`iterateUsers` walks the use chain inside the nested loop until it finds an
op with a partition annotation. This ensures the cluster gets the correct
sink partition (e.g., computation) rather than falling back to the def
partition (e.g., reduction). Without this, `propagatePartitions` would
assign pointer tensor ops to reduction, creating cross-partition channels
for pointer types that crash `WSCodePartition`.

**Scalar op exclusion**: During cluster assignment, ops that produce only
scalar results (non-tensor, non-memdesc) are skipped. These ops can be
rematerialized in any partition and should not force partition assignment.
Clusters with empty `defPartitions` (containing only scalar ops) are also
skipped.

Cluster assignment rules:

1. **Multiple def or sink partitions**: The cluster sits between multiple
   partitions. For BWD-like kernels (has reduction, no epilogue, has
   computation), assign to the existing computation partition. Otherwise
   create a new computation partition (unless `createComputePartitions=false`,
   in which case merge into existing computation).
2. **No sink partition** (no downstream consumers with partitions): Assign
   the entire cluster to its def partition.
3. **Single def and single sink**: Assign to the sink partition (downstream
   consumer), or to the def partition if they're the same.

### `schedulePostLoopOps`

Schedules post-loop operations:

- **Epilogue store ops** → `epilogue_store` partition (when it exists), else
  follow the same routing as regular epilogue ops.
- **Epilogue ops** (non-store) → routing depends on tuning knobs:
  - `mergeEpilogueToComputation`: → computation[dpId] directly
  - `mergeEpilogue`: → correction (if exists) → reduction → computation[dpId]
  - Neither: → `epiloguePartition` (if exists) → correction/reduction →
    computation

The `postLoopPartition` fallback order (for epilogue ops when no merge knob
is active) is:
1. `epiloguePartition` (when it exists)
2. Correction/reduction partition (whichever serves as default)
3. First `dpIdToPartition` entry (Hopper with all merges, last resort)

### `optimizeSchedule`

Clones `BroadcastOp` and `ExpandDimsOp` into each partition that has users.

### `splitDataPartitionedIfOps`

Splits `scf.if` ops whose results feed different computation partitions into
separate per-partition `scf.if` ops. Required for flex attention masking where
a single `scf.if` yields values for both data partitions.

## Partition Type Summary

For FA forward with `dpFactor=2`, `mergeEpilogue` + `separateEpilogueStore`
(Blackwell):
```
partition 0: correction      — correction ops, load users, epilogue ops (normalize acc)
partition 1: gemm            — MMA operations + mem desc views
partition 2: load            — TMA loads + associated allocs
partition 3: epilogue_store  — descriptor stores
partition 4: computation     — MMA user group 1 (PV_1 chain)
partition 5: computation     — MMA user group 0 (PV_0 chain)
```

For FA backward with `dpFactor=1`, `mergeEpilogueToComputation` (Blackwell):
```
partition 0: reduction   — TMA reduction ops, pre-loop tmem_stores
partition 1: gemm        — MMA operations + mem desc views
partition 2: load        — TMA loads + associated allocs
partition 3: computation — all MMA users + epilogue ops (tmem_load dK/dV,
                           reshape, split, truncf, descriptor_store)
```

For flex attention forward with `dpFactor=2`, `mergeEpilogue` (Blackwell):
```
partition 0: correction  — correction ops, load users, sparse indexing,
                           epilogue ops (normalize acc)
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

For GEMM with `separateEpilogueStore` (no correction/reduction):
```
partition 0: gemm           — MMA operations + mem desc views
partition 1: load           — TMA loads + associated allocs
partition 2: epilogue       — epilogue ops (post-loop tmem_load, truncf)
partition 3: epilogue_store — TMA stores (descriptor_store, async_tma_copy)
```

## Debug

- `TRITON_LLVM_DEBUG_ONLY="tritongpu-partition-scheduling"` enables debug logging.
- The categorizer prints all ops grouped by category with dpId.
- `createPartitionLayout` logs which partitions are created.
- Phase 5 logs MMA processing with dpId and pre-assignment status.
