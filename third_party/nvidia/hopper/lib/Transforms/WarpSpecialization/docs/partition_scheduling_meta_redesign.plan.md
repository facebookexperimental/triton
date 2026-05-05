## Context

The current `PartitionSchedulingMeta` pass has accumulated several design issues:

1. **Hacky secondary correction detection**: `selectTemplate()` has ~35 lines re-detecting correction ops that the categorizer missed because `categorizeDataPartitionOps()` runs first and claims them.
2. **dpId only on DataPartition**: Only `DataPartition`-categorized ops carry a `dataPartitionId`. Other categories (Load, MMA, Correction, EpilogueStore) don't, making it impossible to merge them into the correct per-dpId computation partition.
3. **Template system is over-engineered**: `UnifiedFATemplate` vs `GEMMTemplate` selection adds indirection. The partition layout should be driven by tuning knobs, not by detecting which "pattern" the kernel matches.
4. **Default partition semantics are inconsistent**: The "default" partition is sometimes created, sometimes not, and serves multiple unrelated roles (correction, load users, post-loop ops, uncategorized ops).
5. **`getBackwardSlice` stops at `scf.if` boundaries**: MLIR's `getBackwardSlice` adds an `scf.if` op to the slice and follows its condition, but does NOT enter the then/else regions to follow yield operands. This causes QK `tmem_load` and `mulf(QK*scale)` ops in flex attention to be missed, requiring the post-hoc merge workaround.
6. **New Hopper case impossible**: FA on Hopper wants 3 partitions (load + computation×2), requiring `mergeCorrection` and `mergeEpilogue` — none of which exist today.
7. **No control over epilogue store placement**: On Blackwell, `DescriptorStoreOp` benefits from a dedicated 1-warp partition.

### Target partition layouts

| Case | Knobs | Partitions |
|------|-------|------------|
| Blackwell FA fwd (current) | default | correction, gemm, load, epilogue, comp×2 |
| Blackwell FA fwd (optimized) | separateEpilogueStore | correction, gemm, load, epilogue_store (1-warp), comp×2 |
| Blackwell FA fwd (merged epi) | mergeEpilogue | correction (+ epilogue ops), gemm, load, comp×2 |
| Blackwell FA bwd | default | reduction, gemm, load, epilogue, comp |
| Blackwell flex fwd | default (no epilogue) | correction, gemm, load, comp×2 |
| Hopper FA fwd | mergeCorrection+mergeEpilogue | load, comp×2 |
| Simple GEMM (dpFactor=1) | default | default, gemm, load, epilogue |
| Data-partitioned GEMM (dpFactor=2) | default | default, gemm, load, epilogue |

Note: Both GEMM cases produce identical partition layouts. With dpFactor=2, each MMA's exclusive backward slice only contains loads/memdesc_views (already categorized as Load), so no DataPartition or computation entries are created. Post-loop ops (tmem_load, truncf for output conversion) go to the uncategorized partition, labeled "default".

---

## Phase 1: Enhance `collectMMABackwardSlices` as central dpId assignment

**File**: `PartitionSchedulingMeta.cpp`

The core change: `collectMMABackwardSlices` becomes the single source of truth for dpId assignment. It already computes backward slices and union-find groups. Enhance it to (a) enter `scf.if` regions, (b) build an `opToDpId` map for ALL reachable ops, and (c) extend beyond the innermost loop boundary.

### 1a. Enter `scf.if` regions in backward slice analysis

Enhance `collectMMABackwardSlice` so that when an `scf.if` op is added to the slice, its yield operands in the then/else blocks are also followed backward. This captures ops like `tmem_load QK` and `mulf(QK*scale)` that feed into `scf.if` yield operands in flex attention.

Implementation: after the initial `getBackwardSlice` call, iterate over any `scf::IfOp` in the slice and recursively call `getBackwardSlice` on their yield operands:

```
collectMMABackwardSlice(loop, mmaOp):
  slice = getBackwardSlice(mmaOp operands, options)
  // Enter scf.if regions: follow yield operands backward
  repeat until no new ops:
    for each scf.IfOp in slice:
      for each region (then, else):
        for each yield operand:
          getBackwardSlice(operand, &slice, options)
  return slice
```

This eliminates the root cause of the flex attention issue. The post-hoc merge-extra-computation-partitions logic and compaction step can be removed.

### 1b. Assign dpId to all ops (inside and outside innermost loop)

After union-find grouping, build `opToDpId` for every reachable op:

**Inside innermost loop** — iterate over all MMAs and their (now-complete) backward slices:
```
For each MMA group g:
  For each MMA m in group g:
    opToDpId[m] = g
    For each op in backwardSlice[m]:
      if op not in opToDpId:
        opToDpId[op] = g
      else if opToDpId[op] != g:
        opToDpId[op] = SHARED_DPID
```

**Pre-loop ops** (Q loads, allocs): Follow MMA operands backward across the loop boundary. Assign dpId based on which MMA group they feed exclusively into, or `SHARED_DPID` if shared.

**Post-loop ops** (descriptor_stores, normalization): Follow loop results forward. Each result traces back to a specific MMA group's yield value. The post-loop consumer chain gets that group's dpId.

### 1c. Expose dpId map from OpCategorizer

Add `opToDpId` as a member of `OpCategorizer`. All `categorize*` functions look up dpId from this map when creating `CategorizedOp` entries, instead of computing dpId independently. `CategorizedOp.dataPartitionId` is populated for ALL categories.

### 1d. Fix categorization order

Move `categorizeCorrectionOps()` BEFORE `categorizeDataPartitionOps()`:
```
categorizeLoads();            // dpId from opToDpId
categorizeMMAs();             // dpId from opToDpId
categorizeEpilogueStores();   // dpId from opToDpId
categorizeTMAReductions();    // dpId from opToDpId
categorizeCorrectionOps();    // dpId from opToDpId ← moved up
categorizeDataPartitionOps(); // dpId from opToDpId, skips already-categorized
```

This eliminates the root cause of the secondary correction detection hack.

---

## Phase 2: Replace template system with tuning knobs

**File**: `PartitionSchedulingMeta.cpp`

### 2a. Tuning knobs

```cpp
struct SchedulingOptions {
  bool mergeCorrection = false;        // correction → computation[dpId]
  bool mergeEpilogue = false;          // non-store epilogue ops → see routing below
  bool mergeReduction = false;         // reduction → computation[dpId]
  bool separateEpilogueStore = false;  // descriptor_store → own 1-warp partition
  unsigned numDataPartitions = 1;
};
```

No `mergeGemm` — MMAv5 always gets its own gemm partition.

**`mergeEpilogue` routing logic** (for non-store epilogue ops):
1. If a **correction** partition exists (`!mergeCorrection && hasCorrection`): merge into correction partition.
2. Else if a **reduction** partition exists (`!mergeReduction && hasReduction`): merge into reduction partition.
3. Else: merge into `computation[dpId]`.

Rationale: correction ops (acc rescaling) and epilogue ops (acc normalization, output writes) are part of the same accumulator pipeline. When correction has its own partition, epilogue naturally belongs there. Same logic applies for reduction in bwd.

**`separateEpilogueStore`**: When true, `DescriptorStoreOp`/`AsyncTMACopyLocalToGlobalOp` always get their own 1-warp partition, regardless of `mergeEpilogue`.

**Full interaction matrix** (non-store epilogue ops):

| `mergeCorrection` | `mergeEpilogue` | correction exists? | non-store epilogue → |
|---|---|---|---|
| false | false | yes | epilogue partition |
| false | true | yes | **correction partition** |
| true | false | no | epilogue partition |
| true | true | no | computation[dpId] |

**Full interaction matrix** (descriptor_store ops):

| `mergeEpilogue` | `separateEpilogueStore` | descriptor_store → |
|---|---|---|
| false | false | epilogue partition |
| false | true | **epilogue_store (1-warp)** |
| true | false | follows non-store epilogue routing above |
| true | true | **epilogue_store (1-warp)** |

Expose as pass options and/or `scf.for` attributes.

### 2b. Simplify partition creation

Remove `UnifiedFATemplate`, `GEMMTemplate`, and `selectTemplate()`. Replace with direct partition creation:

1. **Always** create `computation[0..dpFactor-1]` partitions (when dpFactor > 1).
2. Create `gemm` only if there are MMA-categorized ops (MMAv5). When present, MMAv5 always gets its own partition.
3. **Always** create `load` partition.
4. Create `correction` only if `!mergeCorrection && hasCorrection`.
5. Create `reduction` only if `!mergeReduction && hasReduction`.
6. Create `epilogue` only if `!mergeEpilogue && hasEpilogue && !separateEpilogueStore`. (Also create when `!mergeEpilogue` and there are non-store epilogue ops even when `separateEpilogueStore` is true.)
7. Create `epilogue_store` only if `separateEpilogueStore && hasEpilogueStores`. This partition gets 1 warp.
8. Create `uncategorized` partition for leftovers → label as `"default"` at the end if it has ops, or remove it.

### 2c. Remove secondary correction detection

Delete the ~35 lines in `selectTemplate()` that re-detect correction by walking MMA forward users.

---

## Phase 3: Refactor partition assignment

**File**: `PartitionSchedulingMeta.cpp`

### 3a. Category-to-partition routing with dpId

Replace current Phase 3-5 logic with category-based assignment using dpId:

```
For each categorized op:
  switch (category):
    Load          → loadPartition (shared; dpId is informational)
    MMA           → gemmPartition (always separate for MMAv5)
    MemDescView   → gemmPartition (same as MMA)
    Correction    → correctionPartition (or computation[dpId] if mergeCorrection)
    EpilogueStore → if separateEpilogueStore: epilogueStorePartition (1-warp)
                    else: follow Epilogue routing below
    Epilogue      → if !mergeEpilogue: epiloguePartition
                    else if correctionPartition exists: correctionPartition
                    else if reductionPartition exists: reductionPartition
                    else: computation[dpId]
    Reduction     → reductionPartition (or computation[dpId] if mergeReduction)
    DataPartition → computation[dpId]
    Default       → uncategorizedPartition
```

For ops with `dpId = SHARED_DPID`, route to the uncategorized/default partition.

### 3c. Partition reordering — select the default partition

After all ops are assigned, reorder partitions so that the **default partition** (partition index 0 in `tt.warp_specialize`) is one that requires 4 warps. The `tt.warp_specialize` lowering assigns 4 warps to the first partition and distributes remaining warps to others.

Selection priority:
1. If a **reduction** partition exists → make it partition 0 (bwd: reduction needs 4 warps for TMEM coverage).
2. Else if a **correction** partition exists → make it partition 0 (fwd: correction/rescaling needs 4 warps for TMEM ops).
3. Else → make `computation[0]` partition 0 (fallback: e.g., Hopper with all categories merged).

Implementation: after partition assignment is complete, swap the chosen partition to index 0 and update all ops' `ttg.partition` attributes to reflect the new numbering.

With the `scf.if` region fix (Phase 1a) and dpId-aware routing:
- Merge-extra-computation-partitions step is **removed** (no extra partitions created).
- Compaction step is **removed** (no empty partitions to compact).
- `splitDataPartitionedIfOps` remains for flex attention.
- `propagatePartitions` and `schedulePostLoopOps` still needed for uncategorized ops.

---

## Phase 4: Add Hopper FA lit test

**File**: `test/Hopper/WarpSpecialization/partition-scheduling-meta-hopper-fa.mlir`

Create from `hopper.part.prior`:
- 3 partitions: `load`, `computation`, `computation`
- Pass options: `--nvgpu-partition-scheduling-meta="merge-correction merge-epilogue"`
- Hopper uses `warp_group_dot` (not MMAv5), so no MMA-categorized ops → no gemm partition created
- Correction ops + epilogue ops → computation[dpId] (both merged, no correction/reduction partition exists)
- Loads → shared load partition
- Result: load + comp×2 = 3 partitions

---

## Phase 5: Verify all existing lit tests

Run all existing `partition-scheduling-meta-*.mlir` tests with default knobs (no merging) to verify backward compatibility.

---

## Verification

1. `ninja -j$(nproc) triton-opt` to rebuild
2. Run all partition-scheduling-meta lit tests with FileCheck
3. Run `triton-opt` on `fa.part.prior`, `flex.part.prior`, `hopper.part.prior` and verify partition types
4. Run FA fwd tutorial: `TRITON_USE_META_WS=1 python python/tutorials/fused-attention-ws-device-tma.py`

---

## Critical files

- `PartitionSchedulingMeta.cpp` — main pass implementation (all phases)
- `docs/PartitionSchedulingMeta.md` — documentation updates
- `test/Hopper/WarpSpecialization/partition-scheduling-meta-*.mlir` — lit tests
- `include/nvidia/hopper/include/Transforms/Passes.td` — pass option definitions for merge/separation knobs
