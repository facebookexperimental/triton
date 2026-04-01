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

## Partition Type Summary

For FA forward with `dpFactor=2`:
```
partition 0: default    — correction ops, load users (shared computation)
partition 1: gemm       — MMA operations + mem desc views
partition 2: load       — TMA loads + associated allocs
partition 3: epilogue   — descriptor stores (+ post-loop ops)
partition 4: computation — MMA user group 0
partition 5: computation — MMA user group 1
```

## Debug

- `TRITON_LLVM_DEBUG_ONLY="tritongpu-partition-scheduling"` enables debug logging.
- The categorizer prints all ops grouped by category.
- Template selection is logged.
