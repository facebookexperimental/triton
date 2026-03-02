# Partition Scheduling Redesign: Template-Based Warp Specialization

## Overview

This stack redesigns `PartitionScheduling.cpp` from a monolithic, ad-hoc
partition assignment into a structured **categorize → template → schedule**
pipeline. The new code replaces the old approach of directly hardcoding
partition creation with an abstract template system that can handle FA forward,
FA backward, and GEMM kernels through a single unified code path.

**Files changed:**
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp` — all scheduling logic (692 → 1707 lines)
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/Partition.cpp` — DOT visualization infrastructure only

**Performance:** FA forward achieves **1025 TFLOPS** (matching the original implementation).

---

## Architecture: Three-Phase Pipeline

The old code directly created partitions and scheduled ops in a single interleaved
pass inside `getInitialSchedule()`. The new code separates this into distinct phases:

### Phase 1: Op Categorization (`OpCategorizer`)

A new `OpCategorizer` class (lines 465–807) analyzes all ops in the loop and assigns
each one a semantic category:

| Category | Ops | Purpose |
|---|---|---|
| `Load` | `DescriptorLoadOp`, `DescriptorGatherOp` | TMA loads |
| `LocalAlloc` | `LocalAllocOp`, `TMEMAllocOp` | Buffer allocations co-scheduled with loads |
| `MMA` | `TCGen5MMAOp` | Matrix multiply-accumulate |
| `MemDescView` | Ops with `MemDescViewTrait` | Memory views feeding into MMAs |
| `EpilogueStore` | `DescriptorStoreOp` | Epilogue stores |
| `TMAReduction` | `DescriptorReduceOp`, `AsyncTMAReduceOp` | TMA atomic reductions (new) |
| `Correction` | Cross-iteration MMA users | Online softmax correction |
| `DataPartition` | Ops exclusive to one MMA's backward slice | Per-MMA computation chains |
| `Shared` | Ops in multiple MMA backward slices | Shared computation (e.g., Q/K loads) |

The categorizer also detects the **data partition factor** — the number of independent
MMA computation groups that can run in parallel.

### Phase 2: Template Selection (`selectTemplate()`)

Based on the categorization results, a scheduling template is selected:

- **`UnifiedFATemplate`** — Selected when `mmas.size() > 1 || hasCorrection || dpFactor > 1`.
  Handles both FA forward (dpFactor=2, 4 MMAs in 2 independent groups) and FA backward
  (dpFactor=1, 5 MMAs all transitively dependent).

- **`GEMMTemplate`** — Fallback for simple single-MMA kernels. Creates 4 fixed partitions
  (default, gemm, load, epilogue).

### Phase 3: Template-Based Scheduling

The template creates **abstract partitions** (`Gemm`, `Load`, `Correction`, `Computation[N]`,
`Epilogue`, `Reduction`, `Default`) that map to physical `WarpSchedule` partitions. Scheduling
then proceeds in sub-phases:

1. **Schedule core ops** — Loads → load partition, MMAs → gemm partition, stores → epilogue partition
2. **Propagate users** — Load users → default, MMA cross-iteration users → correction partition
3. **Schedule reductions** — TMA reductions → reduction partition
4. **Create per-MMA partitions** — MMA users → computation partitions (data partitioning)

---

## Key New Components

### Abstract Partition Types and Templates

```
enum class AbstractPartition {
  Gemm, Correction, Epilogue, Load, Reduction, Computation, Default
};
```

The `SchedulingTemplate` base class defines the interface:
- `createPartitions(WarpSchedule&)` — Create physical partitions
- `getPartition(AbstractPartition, dpId)` — Map abstract → physical partition
- `getEpiloguePartition()` — Get the epilogue partition (must be last)

### `TemplateOptions` — Configurable Partition Merging

```cpp
struct TemplateOptions {
  bool mergeEpilogueIntoComputation = false;  // Merge epilogue into last computation
  bool mergeReductionIntoComputation = false;  // Merge reduction into last computation
  bool hasCorrection = false;                  // Whether correction ops exist
  bool hasReduction = false;                   // Whether TMA reductions exist
  unsigned numDataPartitions = 1;              // Number of independent MMA groups
};
```

When `hasCorrection` or `hasReduction` is false, the corresponding partition is **aliased
to the default partition** rather than created, avoiding unnecessary warp groups and
register pressure.

### `UnifiedFATemplate` — Partition Creation Order

The partition creation order in `UnifiedFATemplate::createPartitions()` matches the
original `FAForwardTemplate` ordering, which is critical for correctness:

```
P0: default (stage 0)
P1: gemm    (stage 1)    ← MMA partition must be stage 1
P2: load    (stage 0)
P3..P(3+N): computation[0..N-1] (stage 0)
P(next): correction      (only if hasCorrection)
P(next): reduction       (only if hasReduction, or merged into computation)
P(last): epilogue        (must be last — schedulePostLoopOps relies on this)
```

### Dependency-Based Data Partition Detection

The old code treated each MMA as an independent data partition (dpFactor = number of MMAs).
The new code uses **forward reachability analysis with union-find grouping** to detect
actual inter-MMA dependencies:

1. For each MMA, collect its **forward reachable set** — all ops transitively reachable from
   the MMA's results, including cross-iteration paths through yield → iter arg.
2. For each other MMA, check if its **backward slice** overlaps with the forward set.
3. If overlap exists, the two MMAs are **dependent** and grouped together via union-find.

Results:
- **FA forward**: 4 MMAs → 2 independent groups (dpFactor=2) ✓
- **FA backward**: 5 MMAs → 1 group (dpFactor=1, all transitively dependent) ✓

### `DescriptorReduceOp` Support

The new code recognizes `tt::DescriptorReduceOp` (Triton IR level) as a TMA reduction,
alongside the existing `ttng::AsyncTMAReduceOp` (TritonNvidiaGPU IR level). This is
important for FA backward which uses `descriptor_reduce` for atomic gradient accumulation.

### DOT Visualization

Two DOT dumpers are now available (enabled via `TRITON_DUMP_PARTITION_DOT=1`):

1. **`/tmp/partition_scheduling.dot`** — Dumps from `PartitionScheduling.cpp` after
   `getInitialSchedule()`. Shows template-based partitions with `OpCategorizer` annotations
   (category labels like `[MMA]`, `[TMAReduction]`, `[EpilogueStore]`).

2. **`/tmp/partitions.dot`** — Dumps from `Partition.cpp` after `propagatePartitions()`.
   Shows the final partition state with semantic categories and data partition IDs.

---

## What Stays the Same

The downstream scheduling logic is **mostly unchanged**, with targeted fixes for
persistent kernel support:

- `propagatePartitions()` — Cluster-based unassigned op propagation.
  **Fix:** For BWD kernels (has reduction partition, no epilogue partition), reuses
  the existing computation partition instead of creating new ones when a cluster has
  multiple def/sink partitions. This prevents an extra "computation" partition that
  would split pointer-typed ops across partitions and crash `createLocalAlloc`.
  **Fix:** Skips ops outside the loop (post-loop ops) so they are left for
  `schedulePostLoopOps` to assign correctly.
- `optimizeSchedule()` — Broadcast/ExpandDims/ConvertLayout rematerialization
- `schedulePostLoopOps()` — Post-loop epilogue scheduling.
  **Fix:** Only assigns `DescriptorStoreOp` to the epilogue partition. All other
  post-loop ops (tmem_load, log2, addf, divf, truncf, tt.store, etc.) go to the
  root/default partition. This matches persistent FWD behavior where the heavy
  softmax rescaling computation is inside the outer loop body, keeping the epilogue
  partition lightweight (1 warp) and avoiding thread count overflow.
- `PartitionScheduling::runOnOperation()` — Top-level pass structure

The `getInitialSchedule()` function still produces the same `WarpSchedule` output, just
through the new template-based pipeline instead of direct hardcoding.

### Lit Tests

Three partition scheduling lit tests verify the partition assignments:

| Test | File | Checks |
|---|---|---|
| FWD persistent | `ws_partition_scheduling_fwd_persist.mlir` | gemm=1, load=2, epilogue(descriptor_store)=3, computation(2D mulf/exp2)=5 |
| FWD non-persistent | `ws_partition_scheduling_fwd.mlir` | Same as persistent; epilogue has NO heavy ops (exp2, divf, tmem_load) |
| BWD | `ws_partition_scheduling_bwd.mlir` | gemm=1, load=2, computation/epilogue(exp2,mulf,descriptor_store)=3, reduction(128x32 mulf)=0 |

---

## Commit Stack (bottom to top)

| # | Commit | Description |
|---|--------|-------------|
| 1 | `detectDataPartitionFactor` | Add backward-slice-based MMA analysis for data partition detection |
| 2 | `New phases` | Introduce OpCategorizer and the categorize → template → schedule pipeline |
| 3 | `FA bwd` | Add backward pass support and DOT visualization to Partition.cpp |
| 4 | `update to partition plot` | Enhance Partition.cpp DOT dumper with semantic categories |
| 5 | `use data partition in scheduling` | Wire data partition factor into template-based scheduling |
| 6 | `clean up dot` | Separate DOT output files to avoid collision between the two dumpers |
| 7 | `redesign: categorizer → abstract partitions → partitions` | Major restructure: introduce AbstractPartition, SchedulingTemplate, UnifiedFATemplate, GEMMTemplate |
| 8 | `fix fwd` | Fix FA forward: match partition ordering, skip unused partitions, restore correction scheduling |
| 9 | `more fix` | Additional fixes for template selection and merge options |
| 10 | `fix dp factor for bwd to 1` | Fix dependency analysis: use forward reachability instead of counting MMAs |

Plus uncommitted changes adding `DescriptorReduceOp` support and refining template selection
(`mmas.size() > 1` condition).
