# Subtile Operator — Design & Implementation Overview

## Motivation

In warp-specialized GEMM epilogues with `EPILOGUE_SUBTILE > 1`, the
accumulator is split into N subtiles (e.g., 128×256 → 2×128×128). Each
subtile flows through the same computation (truncf, convert, store) but with
different data and offsets. The **subtile operator** (`ttng.subtiled_region`)
captures this structure so that per-tile barrier placement, memory planning,
and code generation can reason about the repetition rather than seeing N
copies of inlined code.

## Architecture

### Op Definition

`SubtiledRegionOp` (`ttng.subtiled_region`) has three regions:

- **setup**: Computes shared values (tmem_load → reshape → trans → split).
  Terminated by `subtiled_region_yield` whose values are indexed by tile
  mappings.
- **tile**: Per-tile body, replicated during lowering. Block arguments are
  substituted from setup outputs via `tileMappings`. An optional trailing
  i32 argument receives the tile index (0, 1, …).
- **teardown**: Runs once after all tiles. Its yield values become the op's
  results.

Key attributes:
- `tileMappings: ArrayAttr` — one `DenseI32ArrayAttr` per tile mapping tile
  block args to setup yield indices
- `barrierAnnotations: ArrayAttr` — where to insert wait/arrive barrier ops
  during lowering (uses `subtile_op_id` for stable targeting)
- `tokenAnnotations: ArrayAttr` — NVWS token-layer annotations, converted to
  barrier annotations during token lowering

Defined in `include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td`.

### Passes

#### 1. GenerateSubtiledRegion
**File:** `lib/Dialect/TritonNvidiaGPU/Transforms/GenerateSubtiledRegion.cpp`
**Pass:** `triton-nvidia-gpu-test-generate-subtiled-region`

Finds `tmem_load → reshape → trans{[0,2,1]} → split` patterns and wraps the
per-tile chains into `SubtiledRegionOp`s.

Key capabilities:
- **2-tile and N-tile** (4, 8, …) via nested split tree walking
  (`collectSplitTreeLeaves`). Both paths share the same N-tile build
  functions (`buildSingleSubtiledRegionN`, `buildMultiTaskSubtiledRegionsN`)
  — the 2-tile path is a thin wrapper that converts inputs.
- **Auxiliary collection** (`collectPerTileChain`): always recursively
  captures ops needed by the chain but not depending on the split result
  (e.g., `descriptor_load`, `arith.addi` for address computation). For
  N-tile nested splits, inner setup ops are excluded via `excludeOps`.
- **Identity insertion** for asymmetric chains (e.g., one tile has an extra
  `arith.addi` for column offset). The "canonical identity set" approach in
  `checkStructuralEquivalenceN` compares the template against the shortest
  chain first to discover all identity ops, then re-compares other chains
  with `forcedIdentityOps` for consistent identity counts.
- **Multi-task segmentation** for chains crossing async task boundaries.
  Each segment becomes a separate `SubtiledRegionOp` with SMEM transitions
  (implicit buffer via `local_store`/`local_load`). Allocs are assumed to
  be pre-hoisted by the memory planner.
- **Multi-chain support** (addmm): when task IDs are non-contiguous
  (e.g., task 2 → 3 → 2 → 1), segments are merged by task ID and
  topologically sorted by data dependency, producing contiguous regions
  (e.g., task 3 → 2 → 1).

Structural equivalence (`checkStructuralEquivalence`) compares per-tile
chains pairwise, recording differing operands and identity-compatible ops.
`checkStructuralEquivalenceN` wraps this for N chains with consistent
identity handling.

#### 2. OptimizeTMemLayouts (+ PushSharedSetupToTile)
**File:** `lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp`
**Pass:** `triton-nvidia-optimize-tmem-layouts`

This pass serves dual purposes:

1. **TMem layout optimization** (pattern-based): Converts
   `tmem_load → reshape → trans → split` chains into
   `tmem_subslice → tmem_load` pairs, eliminating reshape/trans overhead.
   Also handles `tmem_store + join` patterns and layout selection for
   vectorization.

2. **SubtiledRegionOp setup push** (imperative, after patterns fire): Walks
   all `SubtiledRegionOp`s and calls `pushSubtiledRegionSetupToTile()`, which
   runs three transformations:
   - `addSubsliceRangeToSetup` — extracts per-tile N offsets from
     `tmem_subslice` ops as i32 tile args
   - `pushTmemLoadsToTile` — moves per-tile `tmem_load` chains from setup
     into tile body, interleaving loads with compute
   - `pushSharedSetupToTile` — sinks "shared" tile arguments (uniform across
     tiles) into the tile body

The push logic lives in `PushSharedSetupToTile.cpp` and is exposed via the
`pushSubtiledRegionSetupToTile()` entry point declared in `Dialect.h`.

#### 3. LowerSubtiledRegion
**File:** `lib/Dialect/TritonNvidiaGPU/Transforms/LowerSubtiledRegion.cpp`
**Pass:** `triton-nvidia-gpu-lower-subtiled-region`

Expands each `SubtiledRegionOp` into flat IR:
1. Inlines setup ops
2. Replicates tile body N times with value substitution from tile mappings
3. Inserts `WaitBarrierOp`/`ArriveBarrierOp` at positions specified by
   barrier annotations (using `subtile_op_id` for stable op targeting and
   `tileMask` for selective per-tile firing)
4. Inlines teardown ops

Also exported as a public function `lowerSubtiledRegion(SubtiledRegionOp)`
for use by other passes (e.g., WSCodePartition for multi-task fallback).

### Pipeline Integration

The subtile pipeline spans two compilation phases: the WS mega-pass generates
and annotates the SubtiledRegionOps, then the main TTGIR pipeline optimizes
and lowers them.

**Inside `NVGPUWarpSpecialization` pass** (`WarpSpecialization.cpp`):

```
doTaskIdPropagate
doBufferAllocation
doHoistLoopInvariantTMEMStore
doMemoryPlanner
doGenerateSubtiledRegion          ← only runs GenerateSubtiledRegion pass
doAnnotateTMAStoreWaits
doValidateTMAStoreAnnotations
doCodePartitionPost               ← adds token annotations on SubtiledRegionOps;
                                    multi-task SubtiledRegionOps lowered here
doTokenLowering                   ← converts tokens → barrier annotations
scheduleLoops                       (SubtiledRegionOps survive with annotations)
```

**In the main TTGIR pipeline** (`compiler.py`), after the WS pass:

```
...
add_optimize_tmem_layouts         ← pattern rewrites (split → tmem_subslice)
                                    + pushSubtiledRegionSetupToTile()
add_lower_subtiled_region         ← expands tile bodies with per-tile barriers
add_tma_lowering
...
```

This separation is critical: `doGenerateSubtiledRegion` only creates the
SubtiledRegionOps (no tmem optimization, no setup push). The SubtiledRegionOps
survive through the WS pass where they receive barrier annotations via token
lowering. Only after the WS pass completes does `add_optimize_tmem_layouts`
transform the setup chains (both inside SubtiledRegionOps and bare splits
elsewhere), and `add_lower_subtiled_region` expands the tile bodies.

This avoids the earlier problem where `OptimizeTMemLayouts` ran inside
`doGenerateSubtiledRegion` and transformed bare (non-SubtiledRegionOp) splits
into `tmem_subslice` ops lacking `async_task_id`, crashing `createChannelPost`.

Multi-task SubtiledRegionOps (tile body spanning multiple tasks) are still
lowered as a fallback inside `doCodePartitionPost` before `specializeRegion`.

### Compiler Option

- Kernel kwarg: `generate_subtiled_region=True`
- Knob: `triton.knobs.nvidia.generate_subtiled_region = True`
- Env var: `TRITON_GENERATE_SUBTILED_REGION=1`
- Autotuning config option: `generate_subtiled_region`

Default: `False`.

### Barrier & Token Annotations

`BarrierAnnotationAttr` specifies per-tile barrier placement:
- `barrierIdx` — index into the op's barriers/accumCnts
- `placement` — BEFORE or AFTER target op
- `targetOpIdx` — matched via `subtile_op_id` attribute on tile body ops
- `barrierOpKind` — `"wait_barrier"` or `"arrive_barrier"`
- `tileMask` — per-tile enable mask (empty = all tiles)
- `region` — TILE, SETUP, or TEARDOWN
- `numBuffers` — for multi-buffer phase/index computation

`TokenAnnotationAttr` is the NVWS token-layer equivalent, resolved to
`BarrierAnnotationAttr` during `doTokenLowering`.

### Test Coverage

| Test file | Coverage |
|-----------|----------|
| `test/TritonNvidiaGPU/lower_subtiled_region.mlir` | 13 LIT tests for lowering |
| `test/TritonNvidiaGPU/generate_subtiled_region_multi_task.mlir` | Multi-task, identity, addmm patterns |
| `test/TritonNvidiaGPU/generate_subtiled_region_ntile.mlir` | 4-tile, 8-tile nested splits |
| `test/TritonNvidiaGPU/generate_subtiled_region_tmem_split.mlir` | tmem_subslice + push-to-tile optimization |
| `test/TritonNvidiaGPU/push_shared_setup_to_tile.mlir` | Setup-to-tile push transformations |
| `test/TritonNvidiaGPU/invalid.mlir` | Verifier error cases |
| `python/test/unit/language/test_tutorial09_warp_specialization.py` | Blackwell GEMM e2e (parametrized with `generate_subtiled_region`) |
| `python/test/unit/language/test_autows_addmm.py` | Addmm e2e (parametrized with `generate_subtiled_region`) |
| `test_subtile_gemm.py` | Standalone addmm + subtile e2e |

## Known TODOs

1. **N-tile multi-task with identity ops in segments.** When per-segment
   chains have identity ops (from the canonical-identity forced identity),
   `checkStructuralEquivalenceN` on the segment bails. This prevents
   N-tile multi-task subtiling when tiles have asymmetric address offsets
   (e.g., 4-tile with `arith.addi` in 3 of 4 tiles). The fix: propagate
   `forcedIdentityOps` into per-segment equivalence checks.

2. **Merge `tryGenerateForSplit` 2-tile/N-tile branches.** The two branches
   (~120 lines each) share ~60% logic. Now that `buildSingleSubtiledRegion`
   and `buildMultiTaskSubtiledRegions` both delegate to N-tile versions,
   the branch condition (`numTiles > 2`) could be eliminated by always using
   the N-tile data structures. The 2-tile path's non-contiguous task
   merging + topological sort needs to be generalized to N tiles.

3. **Non-tensor cross-segment values in N-tile multi-task** (e.g., scalar
   offsets) bail out. These need to be passed through as differing operands
   without SMEM buffering.

4. **The `isFirstSegment` assumption in `buildMultiTaskSubtiledRegionsN`.**
   After merge-and-reorder, the first segment may not use the split result
   (e.g., task 3 bias load segment). The unused split result tile arg is
   wasted.

5. **Full e2e test coverage.** The `generate_subtiled_region=True` parameter
   is added to `test_tutorial09_warp_specialization.py` and
   `test_autows_addmm.py` but full test suite results are pending validation.
