# Subtiled Region Operator — Overview

This document summarizes the `SubtiledRegionOp` infrastructure for the explicit
subtiling optimization in the Triton compiler. See the
[design doc](https://docs.google.com/document/d/1lBqS0lqDN7VI20_dyoJ2TpwC5VIQYG-N1FhKAgPoips/edit)
for motivation and full design.

## What is SubtiledRegionOp?

`SubtiledRegionOp` encapsulates per-tile operations in the epilogue (e.g.,
after a GEMM accumulator is produced). It has three regions:

- **Setup region**: Ops that run once before any tile (e.g., `tmem_subslice`).
- **Tile region**: Ops that run once per tile (e.g., `tmem_load`, type
  conversions, stores).
- **Teardown region**: Ops that run once after all tiles.

The op also carries `BarrierAnnotation` metadata describing where barriers
should be inserted (BEFORE the first tile, AFTER the last tile, etc.).

## Implemented Passes

### Pass 1: AddSubtileRegions (Generation)

**File**: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/AddSubtileRegions.cpp`

Pattern-matches `split` trees in the epilogue IR, identifies which ops are
"setup" (shared across tiles) vs. "per-tile", and wraps them in a
`SubtiledRegionOp`. Handles both same-task and cross-task TMA store patterns.

- Activated by the `useSubtiledRegionOperator` option on `NVGPUWarpSpecialization`.
- Controlled at runtime by `TRITON_USE_SUBTILED_REGION_OPERATOR=1`.
- Validates SWP consistency (async_task_id, loop.stage, loop.cluster).
- Called during `add_hopper_warpspec`, after `doTaskIdPropagate`.

### Pass 1.75: FuseSubtileRegions (Fusion)

**File**: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/AddSubtileRegions.cpp`
(function `doFuseSubtileRegions`)

Runs after `doCodePartitionPost` and before `doTokenLowering` within the WS
pipeline (gated by `TRITON_FUSE_SUBTILED_REGIONS=1`). When
`doCodePartitionPost` produces two adjacent `SubtiledRegionOp`s that are
independent, this pass fuses them into a single op that interleaves their
tile arguments. This changes sequential execution (all tiles of A, then all
tiles of B) into interleaved execution (tile 0 of A, tile 0 of B, tile 1 of
A, tile 1 of B), improving locality and enabling better barrier placement
around the combined operation.

Two ops are fusible when they:
- Are in the same block with no side-effectful ops between them
- Have the same number of tiles
- Both have empty barriers (fusion runs before `doAnnotateSubtileBarriers`)
- Both have empty teardown regions
- Have compatible attributes (`async_task_id`, `loop.stage`, `loop.cluster`)

The pass supports chain-fusion: three or more adjacent fusible ops are
iteratively fused into a single op.

### Pass 1.8: AnnotateSubtileBarriers (Barrier Annotation)

**File**: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/AddSubtileRegions.cpp`
(function `doAnnotateSubtileBarriers`)

Runs after `doTokenLowering` within the WS pipeline. Scans for
`WaitBarrierOp`/`ArriveBarrierOp` ops adjacent to `SubtiledRegionOp` and
absorbs them into the op's barrier operands and `BarrierAnnotation` metadata.

- Wait barriers before the op → `BEFORE` annotation (targetOpIdx=0).
- Arrive barriers after the op → `AFTER` annotation (targetOpIdx=last).
- The absorbed barriers are erased from the flat IR.

### Pass 2: SubtiledRegionSetupPush (Register Pressure Reduction)

**File**: `lib/Dialect/TritonNvidiaGPU/Transforms/SubtiledRegionSetupPush.cpp`

After `OptimizeTMemLayouts` has split TMEM loads into per-tile
`tmem_subslice + tmem_load` pairs, this pass pushes the `tmem_load` ops from
the setup region into the tile body. This is the key optimization for register
pressure reduction: only one tile's loaded tensor is live at a time.

**Before setup push** (setup region loads all tiles' data):
```
setup {
  %sub0 = tmem_subslice %base
  %load0 = tmem_load %sub0
  %sub1 = tmem_subslice %base
  %load1 = tmem_load %sub1
  yield %load0, %load1
}
tile(%val) { compute(%val); store(%val) }
```

**After setup push** (each tile loads its own data):
```
setup {
  %sub0 = tmem_subslice %base
  %sub1 = tmem_subslice %base
  yield %sub0, %sub1
}
tile(%memdesc) { %val = tmem_load %memdesc; compute(%val); store(%val) }
```

### Pass 3: LowerSubtiledRegion (Lowering)

**File**: `lib/Dialect/TritonNvidiaGPU/Transforms/LowerSubtiledRegion.cpp`

Lowers `SubtiledRegionOp` back to flat IR by:
1. Inlining the setup region.
2. Replicating the tile region for each tile in the tile mapping.
3. Inserting barrier ops (`wait_barrier`/`arrive_barrier`) at annotated
   positions (before first tile / after last tile).
4. Inlining the teardown region.

## Current Pipeline Placement (Blackwell path)

```
add_hopper_warpspec(pm, ..., use_subtiled_region_operator=True)
  ├── doTaskIdPropagate
  ├── doAddSubtileRegions          ← generates SubtiledRegionOp
  ├── doBufferAllocation
  ├── doMemoryPlanner
  ├── doCodePartitionPost
  ├── doFuseSubtileRegions          ← fuses adjacent ops (gated by TRITON_FUSE_SUBTILED_REGIONS)
  ├── doTokenLowering
  └── doAnnotateSubtileBarriers    ← absorbs adjacent barriers
add_pipeline(pm, ...)
...
optimize_tmem_layouts(pm)          ← TMemSplitLoadPattern fires in setup region
subtiled_region_setup_push(pm)     ← pushes tmem_load into tile body
tma_lowering(pm)
remove_layout_conversions(pm)
interleave_tmem(pm)
lower_subtiled_region(pm)          ← lowers back to flat IR
...
```

The `SubtiledRegionOp` survives through the middle of the pipeline, allowing
`optimize_tmem_layouts` to decompose the setup region's TMEM loads and
`subtiled_region_setup_push` to move per-tile loads into the tile body for
register pressure reduction. The op is lowered after `interleave_tmem` and
before the remaining passes that require flat IR.

## Knob

```bash
TRITON_USE_SUBTILED_REGION_OPERATOR=1
```

When unset (default), no `SubtiledRegionOp` is created and the subtiling
passes are not added to the pipeline. The compilation path is unchanged.

```bash
TRITON_FUSE_SUBTILED_REGIONS=1
```

When set (and `TRITON_USE_SUBTILED_REGION_OPERATOR=1` is also set), adjacent
`SubtiledRegionOp`s produced by `doCodePartitionPost` are fused into a single
op with interleaved tile arguments.

## Testing

- **LIT tests** (pass-level):
  - `test/TritonNvidiaGPU/lower_subtiled_region.mlir`
  - `test/TritonNvidiaGPU/subtiled_region_invalid.mlir`
  - `test/TritonNvidiaGPU/subtiled_region_ops.mlir`
  - `test/TritonNvidiaGPU/add_subtile_regions.mlir`
  - `test/TritonNvidiaGPU/add_subtile_regions_invalid.mlir`
  - `test/TritonNvidiaGPU/fuse_subtile_regions.mlir`
- **End-to-end**:
  ```bash
  TRITON_USE_SUBTILED_REGION_OPERATOR=1 \
    pytest python/test/unit/language/test_tutorial09_warp_specialization.py -k "0-1-True-False-False-4-2-64-128-128-128-128-128"
  ```

## What's Not Yet Implemented

1. **Existing pass extensions for deeper survival**: Some passes
   (`Allocation.cpp`, `Membar.cpp`) use region-aware analysis that would need
   `RegionBranchOpInterface` on `SubtiledRegionOp` to handle it natively.
   The current approach lowers before those passes run.
2. **Fine-grain barriers** and **TMA store pipelining** (future optimizations,
   explicitly out of scope).
