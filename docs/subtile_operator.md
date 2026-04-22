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
  (`collectSplitTreeLeaves`)
- **Identity insertion** for asymmetric chains (e.g., one tile has an extra
  `arith.addi` for column offset)
- **Multi-task segmentation** for chains crossing async task boundaries.
  Each segment becomes a separate `SubtiledRegionOp` with SMEM transitions
  (Option 1: explicit `local_alloc`; Option 2: implicit buffer via
  `local_store`/`local_load`)
- **Multi-chain support** (addmm): recursive auxiliary collection captures
  independent data flows (e.g., bias `descriptor_load` chain) in the per-tile
  chain. When task IDs are non-contiguous (e.g., task 2 → 3 → 2 → 1),
  segments are merged by task ID and topologically sorted by data dependency,
  producing contiguous regions (e.g., task 3 → 2 → 1)

Structural equivalence (`checkStructuralEquivalence`) compares per-tile
chains, recording differing operands and identity-compatible ops.

#### 2. OptimizeTMemLayouts
**Pass:** `triton-nvidia-optimize-tmem-layouts`

Converts `tmem_load → reshape → trans → split` inside SubtiledRegionOp setup
regions into `tmem_subslice → tmem_load` pairs, eliminating the reshape/trans
overhead.

#### 3. PushSharedSetupToTile
**File:** `lib/Dialect/TritonNvidiaGPU/Transforms/PushSharedSetupToTile.cpp`
**Pass:** `triton-nvidia-gpu-push-shared-setup-to-tile`

Three transformations on each `SubtiledRegionOp`:
1. `addSubsliceRangeToSetup` — extracts per-tile N offsets from
   `tmem_subslice` ops as i32 tile args
2. `pushTmemLoadsToTile` — moves per-tile `tmem_load` chains from setup into
   tile body, interleaving loads with compute
3. `pushSharedSetupToTile` — sinks "shared" tile arguments (uniform across
   tiles) into the tile body

#### 4. LowerSubtiledRegion
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

Inside `NVGPUWarpSpecialization` pass (`WarpSpecialization.cpp`):

```
doTaskIdPropagate
doBufferAllocation
doHoistLoopInvariantTMEMStore
doMemoryPlanner
doGenerateSubtiledRegion          ← sub-pipeline: Generate + OptimizeTMem + PushShared
doAnnotateTMAStoreWaits
doValidateTMAStoreAnnotations
doCodePartitionPost               ← adds token annotations on SubtiledRegionOps
doTokenLowering                   ← converts tokens → barrier annotations
lowerSubtiledRegion               ← expands tile bodies with per-tile barriers
scheduleLoops
```

Multi-task SubtiledRegionOps (tile body spanning multiple tasks) are lowered
as a fallback inside `doCodePartitionPost` before `specializeRegion`.

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
| `test/TritonNvidiaGPU/generate_subtiled_region_tmem_split.mlir` | tmem_subslice optimization |
| `test/TritonNvidiaGPU/push_shared_setup_to_tile.mlir` | Setup-to-tile push transformations |
| `test/TritonNvidiaGPU/invalid.mlir` | Verifier error cases |
| `python/test/unit/language/test_tutorial09_warp_specialization.py` | Blackwell GEMM e2e (parametrized) |
| `python/test/unit/language/test_autows_addmm.py` | Addmm e2e (parametrized) |
| `test_subtile_gemm.py` | Standalone addmm + subtile e2e |

## Known TODOs

1. **E2e pipeline crash with `generate_subtiled_region=True`.**
   `OptimizeTMemLayouts` runs unconditionally inside `doGenerateSubtiledRegion`
   and replaces `tmem_load → reshape → trans → split` with `tmem_subslice →
   tmem_load` even when the generation pass doesn't wrap the split in a
   SubtiledRegionOp. The resulting bare `tmem_subslice` ops have no
   `async_task_id`, causing an assertion failure in `createChannelPost`
   (`CodePartitionUtility.cpp:2666`). Fix: scope `OptimizeTMemLayouts` to
   only operate inside SubtiledRegionOp setup regions, or propagate task IDs
   to the new ops.

2. **Cross-SubtiledRegionOp barrier insertion for multi-chain (addmm).**
   The 3-region model (task 3 bias load → task 2 compute → task 1 store)
   produces 3 single-task SubtiledRegionOps with SMEM transitions. The code
   partition pass needs to detect `local_store`/`local_load` crossing task
   boundaries between SubtiledRegionOps and insert barrier annotations. This
   path is blocked by TODO 1.

3. **N-tile multi-task Option 1** (explicit `local_alloc` at segment
   boundaries) is not yet supported for N > 2. The code bails out.

4. **Non-tensor cross-segment values in N-tile multi-task** (e.g., scalar
   offsets) bail out. These need to be passed through as differing operands
   without SMEM buffering.

5. **`PushSharedSetupToTile` for multi-segment SubtiledRegionOps.** Non-first
   segments don't clone setup ops. The push pass may not handle SMEM buffer
   tile args correctly.

6. **The `isFirstSegment` assumption in `buildMultiTaskSubtiledRegions`.**
   After merge-and-reorder, the first segment may not use the split result
   (e.g., task 3 bias load segment). The unused split result tile arg is
   wasted. The setup region also clones the entire tmem_load → split chain
   unnecessarily.
