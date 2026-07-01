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

`SubtiledRegionOp` (`ttng.subtiled_region`) is `IsolatedFromAbove`: all
values used in the tile body must be passed as `perTileArgs` or `sharedArgs`.

It has one region:

- **tile**: Per-tile body, replicated `numTiles` times during lowering.
  Tile block arguments are ordered:
  `[perTile0, ..., perTileK-1, shared0, ..., sharedM-1, tileIdx?]`.
  Terminated by `subtiled_region_yield` which optionally yields per-tile
  results.

Key operands:
- `perTileArgs: Variadic<AnyType>` — `numTiles * K` operands grouped by
  position. For K per-tile arg positions, operands `[j*N..(j+1)*N)` are
  the values for position j across all tiles.
- `sharedArgs: Variadic<AnyType>` — M operands broadcast to all tiles.

Key attributes:
- `numTiles: I32Attr` — number of tile replications.

Key methods:
- `addSharedArg(Value)` — appends a shared arg and adds a tile block
  argument. Used by `insertAsyncComm` to make NVWS token / accumCnt / base-alloc
  values accessible inside the tile body.
- `removePerTilePosition(unsigned)` — erases a per-tile position's `numTiles`
  operands and its tile block argument (segment-aware, via
  `getPerTileArgsMutable().erase`). Used to drop buffer positions left dead after
  the in-body view rewire.
- `getNumPerTilePositions()` — returns `perTileArgs.size() / numTiles`.
- `hasTileIndex()` / `getTileIndexArg()` — query / fetch the trailing optional
  i32 tile-index block argument.

If the tile body yields M values, the op produces `numTiles * M` results,
grouped by yield position (matching `tt.join` argument order).

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

#### 2. Lowering (`lowerSubtiledRegion`)
**File:** `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`

`lowerSubtiledRegion(SubtiledRegionOp)` expands a SubtiledRegionOp into flat
IR: replicates the tile body N times, substituting per-tile args for each
tile and broadcasting shared args. Called from:
- `WarpSpecialization.cpp` — inlines SubtiledRegionOps with NVWS ops before
  doTokenLowering, and lowers all remaining before doTMAStoreWaitReorder
- `WSCodePartition.cpp` — inlines multi-task SubtiledRegionOps before
  specializeRegion

### Pipeline Integration

**Inside `NVGPUWarpSpecialization` pass** (`WarpSpecialization.cpp`):

```
doTaskIdPropagate
doBufferAllocation
doHoistLoopInvariantTMEMStore
doMemoryPlanner
doGenerateSubtiledRegion          ← creates SubtiledRegionOps
doAnnotateTMAStoreWaits
doValidateTMAStoreAnnotations
doCodePartitionPost               ← creates inline NVWS ops in SubtiledRegionOps;
                                    multi-task SubtiledRegionOps lowered here
doLowerSubtiledRegionsWithNVWSOps ← inlines SubtiledRegionOps with NVWS ops
doTokenLowering                   ← resolves NVWS ops → hardware barrier ops
scheduleLoops
doLowerRemainingSubtiledRegions   ← inlines all surviving SubtiledRegionOps
doTMAStoreWaitReorder
```

All SubtiledRegionOps are lowered inside the WS pass.

### Compiler Option

- Kernel kwarg: `generate_subtiled_region=True`
- Knob: `triton.knobs.nvidia.generate_subtiled_region = True`
- Env var: `TRITON_GENERATE_SUBTILED_REGION=1`
- Autotuning config option: `generate_subtiled_region`

Default: `False`.

### NVWS Sync Ops in Tile Bodies

When `insertAsyncComm` (WSCodePartition) discovers a sync point inside a
SubtiledRegionOp's tile body, it creates the NVWS op (ProducerAcquireOp,
ConsumerWaitOp, etc.) directly inside the tile body. The token is threaded as a
`addSharedArg` (one logical channel for all tiles).

#### Per-tile SMEM rotation (in-body, off the builtin `tileIdx`)

The barrier slot/phase AND the staging-buffer slot must **not** be shared across
tiles. A subtile group is a reuse group of `numTiles` distinct, concurrent
buffers that all share **one** barrier pair and **one** physical multibuffer
alloc, so every tile must occupy a distinct *generation*.

Every SMEM-rotation value is therefore computed **inside the tile body** from the
op's **builtin `tileIdx`** block arg, with `accumCnt` and the representative
multibuffer alloc threaded in as two `SubtiledRegionOp::addSharedArg`s. For each
multi-buffered subtiled reuse member, `insertAsyncComm` emits, once per region at
the tile-body entry (`getOrComputeSubtiledSlot` in `WSCodePartition.cpp`):

```
flattened = accumCnt + tileIdx                   // tileIdx = builtin block arg
bufferIdx = flattened % numBuffers               // getBufferIdxAndPhase
phase     = (flattened / numBuffers) & 1
view      = memdesc_index[baseArg, bufferIdx]    // built from the in-body base
```

The `numTiles` factor lives on the **loop-carried counter**, not in this index
math: the subtiled reuse group's `accumCnt` advances by `numTiles` per outer
iteration (`getReuseGroupStride` / `getAccumForReuseGroup` in `WSBuffer.cpp`),
so the flattened stream is still `iter*numTiles + tileIdx`. Keeping the stride
on the counter — rather than an in-body `accumCnt * numTiles` — enforces a single
**per-channel** stride rule and stops the `numTiles` factor from leaking onto
co-resident non-subtile counters (e.g. the depth-2 TMEM accumulator, whose
slot/phase otherwise collapse → deadlock). See `docs/AccumulationCounters.md`.

`lowerSubtiledRegion` replaces `tileIdx` with `arith.constant t` per tile, so
the shared barrier behaves as one monotonic stream advanced `numTiles` times per
outer iteration (the advance now comes from `accumCnt += numTiles`). The producer
SMEM-store dest and consumer `async_tma_copy` source are rewired to `view`, and
the now-dead per-tile buffer positions are removed (`removePerTilePosition`) —
`columnOffset` and the data leaf stay per-tile operands.

- The offset is the **builtin tile index** (producer/consumer replication
  order), NOT the reuse-group position. Producer-tile-`t` and consumer-tile-`t`
  are the same logical subtile by construction (both regions replicate from the
  same ordered split-tree leaves), so deriving the slot from `tileIdx` makes
  producer and consumer agree on `slot = (accumCnt + t) % numBuffers` (with
  `accumCnt` advancing by `numTiles`/iter) with **no operand matching**. An
  earlier approach that keyed the count off the
  *threaded* per-tile buffer operands (matched via `traceToBufferBase`) permuted
  the tile→count mapping differently between producer and consumer (the consumer
  carries the SMEM buffer at two per-tile positions) and corrupted data.
- **The data buffer slot uses this SAME in-body count**, so it equals the barrier
  generation `% numBuffers`. A data slot may be reused only `>= numBuffers`
  flattened generations apart; sharing the barrier's count guarantees the shared
  barrier serializes every slot reuse. The generic reuse-group
  `accumCnt + reuseGroupPosition` stagger is wrong here: it collapses distinct
  subtiles onto one slot (the EPILOGUE_SUBTILE>2 staging-buffer race;
  `numTiles=2` was correct only by coincidence).
- This is correct even when `numTiles > numBuffers`: two same-iteration tiles may
  land on one slot, but because flattened order == tile-walk order, the later
  tile's acquire simply *waits* for the earlier tile's slot to be released (a
  serialization, not a race).

Non-reuse / single-copy (`numBuffers == 1`) / non-subtiled cases fall back to a
shared `addSharedArg` barrier index/phase.

Before `doTokenLowering` runs, all SubtiledRegionOps containing NVWS ops
are inlined via `lowerSubtiledRegion`. This puts the NVWS ops in flat IR
where `doTokenLowering` processes them normally — replacing them with
hardware `WaitBarrierOp`/`ArriveBarrierOp`.

### Test Coverage

| Test file | Coverage |
|-----------|----------|
| `test/TritonNvidiaGPU/generate_subtiled_region_dp1.mlir` | DP=1 epilogue subtiling |
| `test/TritonNvidiaGPU/generate_subtiled_region_multi_task.mlir` | Multi-task, identity, addmm patterns |
| `test/TritonNvidiaGPU/generate_subtiled_region_ntile.mlir` | 4-tile, 8-tile nested splits |
| `test/TritonNvidiaGPU/generate_subtiled_region_tmem_split.mlir` | tmem_subslice optimization |
| `test/TritonNvidiaGPU/ops.mlir` | Round-trip parse/print, per-tile results |
| `test/TritonNvidiaGPU/invalid.mlir` | Verifier error cases |
| `test/Hopper/WarpSpecialization/ws_token_lowering_subtiled_region.mlir` | Token lowering with SubtiledRegionOps inside warp_specialize |
| `test/Hopper/WarpSpecialization/ws_code_partition_subtiled_region.mlir` | Code partition with SMEM channels between SubtiledRegionOps |
| `python/test/unit/language/test_tutorial09_warp_specialization.py` | Blackwell GEMM e2e (parametrized with `generate_subtiled_region`) |
| `python/test/unit/language/test_autows_addmm.py` | Addmm e2e (parametrized with `generate_subtiled_region`) |
| `test_subtile_gemm.py` | Standalone addmm + subtile e2e |
