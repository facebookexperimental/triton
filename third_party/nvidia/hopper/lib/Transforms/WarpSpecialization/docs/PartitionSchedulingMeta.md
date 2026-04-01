# Partition Scheduling Meta

This document covers the `PartitionSchedulingMeta` pass, which assigns partition
IDs to operations for warp specialization on Blackwell. This is the first pass
in the AutoWS pipeline — it determines which warp group each operation will
execute on.

**File**: `PartitionSchedulingMeta.cpp`

## Overview

The pass walks all `scf.for` loops with the `tt.warp_specialize` attribute and
assigns each operation inside the loop (and post-loop consumers) to a
**partition**. Each partition maps to a warp group at runtime. The pass has five
phases:

```
Phase 1: Categorize operations     (OpCategorizer)
Phase 2: Select scheduling template (selectTemplate)
Phase 3: Schedule anchor ops        (loads, epilogue stores, MMAs)
Phase 4: Propagate users            (load users, correction, reductions)
Phase 5: Create computation partitions (per-MMA user scheduling)
Post:    propagatePartitions + schedulePostLoopOps + optimizeSchedule
         + splitDataPartitionedIfOps + mergeExtraComputationPartitions
```

## Phase 1: Operation Categorization (`OpCategorizer`)

The `OpCategorizer` classifies every operation in the loop into categories:

| Category | Ops | Purpose |
|----------|-----|---------|
| `Load` | `DescriptorLoadOp`, `DescriptorGatherOp` | TMA loads |
| `MMA` | `MMAv5OpInterface` | Tensor core operations |
| `MemDescView` | ops with `MemDescViewTrait` | Memory descriptor views feeding MMA |
| `EpilogueStore` | `DescriptorStoreOp`, `AsyncTMACopyLocalToGlobalOp` | Output stores |
| `TMAReduction` | `DescriptorReduceOp`, `AsyncTMAReduceOp` | Atomic reductions |
| `Correction` | Cross-iteration MMA users | Online softmax rescaling |
| `DataPartition` | Exclusive ops in one MMA's backward slice | Per-MMA-group computation |

### Data Partition Factor Detection (`collectMMABackwardSlices`)

Determines how many independent MMA groups exist in the innermost loop:

1. **Collect backward slices** for each MMA — the set of ops feeding into it.
2. **Identify shared ops** — ops appearing in multiple slices.
3. **Union-find grouping** — MMAs whose forward user sets overlap another MMA's
   backward slice are grouped together (they are data-dependent).
4. **Count groups with exclusive ops** — only groups with at least one non-shared,
   non-constant op count. This becomes `dataPartitionFactor`.

For Flash Attention forward with `data_partition_factor=2` (two independent
`tl.dot` calls), this yields `dpFactor=2`.

## Phase 2: Template Selection (`selectTemplate`)

Based on the categorized ops, one of two templates is selected:

### `UnifiedFATemplate`

Selected when: `hasCorrection || (mmas.size() > 1 && mmas.size() != dpFactor)`

Creates partitions dynamically based on what's needed:

| Partition | Type | Stage | Created When |
|-----------|------|-------|-------------|
| `default` | Correction + fallback | 0 | fwd (no reduction) |
| `reduction` | TMA reductions | 0 | bwd (replaces default at index 0) |
| `gemm` | MMA operations | 1 | Always |
| `load` | TMA loads | 0 | Always |
| `epilogue` | Descriptor stores | 0 | When epilogue stores exist |
| `computation` | Per-MMA-group ops | 0 | Created dynamically in Phase 5 |

### `GEMMTemplate`

Selected when: single MMA or `mmas.size() == dpFactor` (1:1 mapping).

Creates exactly 4 partitions: `default`, `gemm`, `load`, `epilogue`.

## Phase 3: Schedule Anchor Ops

Anchor operations are assigned directly to their partitions:

1. **Loads** → `load` partition. Includes `LocalAllocOp` users with matching
   shared encoding and `TMEMAllocOp` users.
2. **Epilogue stores** → `epilogue` partition. Includes both in-loop and
   post-loop `DescriptorStoreOp`/`AsyncTMACopyLocalToGlobalOp`. For post-loop
   stores, the **backward slice** is also scheduled into the epilogue (e.g.,
   `tmem_load`, `truncf` that feed the store).
3. **MMAs** → `gemm` partition. Includes `TMEMStoreOp` for the accumulator
   if it's unrelated to read-modify-write patterns.
4. **MemDesc views** feeding MMAs → `gemm` partition. Views are duplicated if
   they have users in other partitions.

## Phase 4: Propagate Users

After anchor ops are assigned, their transitive users are scheduled:

1. **Load users** → `default` partition (shared computation like `qk_scale`).
   Skipped when `default` is absent (bwd).
2. **Correction ops** (cross-iteration MMA users) → `correction` partition
   (aliased to `default` for fwd). These are the online softmax rescaling
   operations (`alpha * acc`, `l_i * alpha + l_ij`).
3. **TMA reduction ops** → `reduction` partition, including their backward
   slice (producers of the reduction value).

## Phase 5: Computation Partitions

MMA users that aren't already scheduled create computation partitions:

- **`dpFactor > 1` (fwd)**: Each independent MMA group gets its own dynamic
  partition via `scheduleUsers(nullptr)`. This creates separate computation
  partitions per data partition.
- **`dpFactor == 1` (bwd)**: All MMA users share a single computation
  partition to avoid creating too many partitions.

### DataPartition Pre-Assignment

When `dpFactor > 1`, the pass pre-assigns `DataPartition`-categorized ops to
their respective computation partitions BEFORE `scheduleUsers` runs. This
prevents Phase 5's greedy `scheduleUsers` from absorbing all computation ops
into the first MMA's partition.

The pre-assignment creates one computation partition per `dpId` and schedules
each exclusive op into its partition. Shared ops (those appearing in multiple
MMA backward slices, e.g., `scf.if` for masking) are pre-assigned to the
`defaultPartition` when it exists, preventing `propagatePartitions` from
forming cross-partition clusters that would collapse the split.

**Backward slice limitation with `scf.if`**: `getBackwardSlice` stops at
`scf.if` ops without entering their regions. For flex attention, QK
`tmem_load` and `mulf(QK*scale)` feed into `scf.if` yield operands but are
NOT in PV MMA's backward slice. These ops are not categorized as
`DataPartition` and are handled later by the merge step in post-processing
(see [Merge Extra Computation Partitions](#merge-extra-computation-partitions)).
This does not affect FA which has no `scf.if`.

#### Why `defaultPartition` is absent for flex attention

The `defaultPartition` is created when
`hasCorrection || hasEpilogue || numDataPartitions > 1`. For flex attention:

1. **`hasCorrection = false`**: The correction ops (accumulator rescaling)
   are categorized as `DataPartition` by `categorizeDataPartitionOps` (which
   runs first) before `categorizeCorrectionOps` can claim them. The secondary
   detection in `selectTemplate` also misses them because the PV MMA token
   is only yielded (no non-yield users in the current iteration).

2. **`hasEpilogue = false`**: Flex attention uses pointer-based `tt.store`
   ops, not `DescriptorStoreOp`/`AsyncTMACopyLocalToGlobalOp`.

3. **`numDataPartitions > 1`**: This condition was added to ensure a
   `defaultPartition` is created for multi-data-partition kernels even
   without epilogue stores. This enables Phase 4 load user propagation,
   which pre-claims shared ops and prevents Phase 5 collapse.

#### Impact on TMEM allocation

Without the data partition split, both alpha stores and both bf16 accumulator
stores land on the same `async_task_id` (e.g., task 4). The downstream
memory planner's `alongDependencyChain` check requires the alpha producer's
task to match the QK consumer's task for TMEM column reuse. When both alphas
are on the same task, they both funnel into one QK's TMEM slot (e.g., `qk_1`
at task 4), overfilling it (128/128 columns), leaving no room.

With the split, alpha_0 lands on one computation partition (matching `qk_0`'s
consumer task) and alpha_1 on another (matching `qk_1`'s consumer task). Each
QK slot gets one alpha + one bf16 buffer, keeping `maxColOffset` manageable.

## Post-Processing

### `propagatePartitions`

Handles unscheduled ops by forming **clusters** — groups of adjacent unscheduled
ops connected by def-use chains. Each cluster is assigned based on its
relationship to already-scheduled partitions:

- **Single def + single sink**: ops on the critical path go to the def
  partition; critical path ops may be rematerialized into the sink partition.
- **Multiple defs or sinks**: the cluster gets its own new `computation`
  partition (with special cases for bwd and GEMM to avoid partition explosion).

### `schedulePostLoopOps`

Schedules operations **after** the WS loop into the epilogue partition. Collects
all uses of the loop's results and recursively assigns them to the epilogue.

**⚠️ Known issue**: This function schedules ALL post-loop consumers of loop
results into the epilogue, including `TMEMLoadOp` (accumulator reads) and
arithmetic ops (e.g., `acc / l_i`). When the WS loop is an inner loop (non-
persistent FA forward), this puts TMEM ops in the epilogue, forcing the
epilogue to use 4 warps (TMEM lane coverage constraint) instead of 1. In the
persistent case, the WS loop is the outer loop and post-loop code is minimal,
so the epilogue only contains a TMA store and can use 1 warp.

### `optimizeSchedule`

Clones `BroadcastOp` and `ExpandDimsOp` into each partition that has users of
them, reducing cross-partition data transfer.

### `splitDataPartitionedIfOps`

When `dpFactor > 1`, the `scf.if` used for masking in flex attention yields
results for both data partitions. After partition assignment, the two results
feed different computation partitions. This step splits such multi-result
`scf.if` ops into separate per-partition `scf.if` ops, each yielding only the
results consumed by one computation partition. This ensures each `scf.if` has a
single partition assignment, which downstream passes require.

The split creates new `scf.if` ops grouped by consumer partition:
```
// Before: one scf.if with results feeding partitions 3 and 4
%r:2 = scf.if %cond -> (T, T) {
  yield %a, %b
} else {
  yield %c, %d
}
use(%r#0) {partition = [3]}
use(%r#1) {partition = [4]}

// After: two scf.if ops, one per partition
%r0 = scf.if %cond -> (T) { yield %a } else { yield %c } {partition = [3]}
%r1 = scf.if %cond -> (T) { yield %b } else { yield %d } {partition = [4]}
use(%r0) {partition = [3]}
use(%r1) {partition = [4]}
```

### Merge Extra Computation Partitions

After `propagatePartitions` and `splitDataPartitionedIfOps`, flex attention may
have more computation partitions than expected (e.g., 4 instead of 2). This
happens because the backward slice analysis cannot cross `scf.if` region
boundaries:

1. `getBackwardSlice` stops at `scf.if` ops without entering their regions.
2. QK `tmem_load` and `mulf(QK*scale)` feed into `scf.if` yield operands, so
   they are NOT in PV MMA's backward slice.
3. These ops are not categorized as `DataPartition` and are not pre-assigned.
4. `propagatePartitions` creates new computation partitions for them.

The merge step fixes this by finding the two largest computation partitions
(the "main" ones) and merging smaller ones into them. To find the correct
merge target, it checks each small partition op's users. If a user is inside
an `scf.if` region (e.g., a `scf.yield` op with no partition attribute), the
merge looks at the parent `scf::IfOp`'s partition instead. This traces through
the region boundary created by `splitDataPartitionedIfOps`.

After merging, a compaction step removes trailing empty partition entries from
the serialized `partition.types` and `partition.stages` arrays by finding the
highest partition index still referenced by any op.

This issue does not affect FA because FA has no `scf.if` — the QK `tmem_load`
and `mulf(QK*scale)` are directly in PV MMA's backward slice and are correctly
categorized as `DataPartition` ops during Phase 1.

## Partition Type Summary

For FA forward with `dpFactor=2`:
```
partition 0: default     — correction ops, load users (shared computation)
partition 1: gemm        — MMA operations + mem desc views
partition 2: load        — TMA loads + associated allocs
partition 3: epilogue    — descriptor stores (+ post-loop ops)
partition 4: computation — MMA user group 0
partition 5: computation — MMA user group 1
```

For flex attention forward with `dpFactor=2` (no epilogue stores, `scf.if`
masking):
```
partition 0: default     — correction ops, load users, sparse indexing
partition 1: gemm        — MMA operations + mem desc views
partition 2: load        — TMA loads + associated allocs
partition 3: computation — MMA user group 0 (includes QK tmem_load + scale)
partition 4: computation — MMA user group 1 (includes QK tmem_load + scale)
```

Key difference: flex uses pointer-based `tt.store` (not `DescriptorStoreOp`),
so no epilogue partition is created. The global stores fall into the default
partition via `schedulePostLoopOps`.

## Debug

- `TRITON_LLVM_DEBUG_ONLY="tritongpu-partition-scheduling"` enables debug logging.
- The categorizer prints all ops grouped by category.
- Template selection is logged.
