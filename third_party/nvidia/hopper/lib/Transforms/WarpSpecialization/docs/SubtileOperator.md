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
- `addPerTilePosition(ValueRange)` — appends a NEW per-tile position (inverse of
  `removePerTilePosition`): takes exactly `numTiles` values (tile 0, 1, …) and
  adds one tile block argument that lowering substitutes per tile. Used by
  `insertAsyncComm` to thread a **per-tile producer token** for inside→outside
  channels whose siblings are distinct buffers (see below).
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
doCodePartition               ← creates inline NVWS ops in SubtiledRegionOps;
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

#### Channel topologies: inside→outside vs both-endpoints-subtiled

An epilogue SMEM channel whose producer is inside a `ttng.subtiled_region` has
two consumer shapes:

- **Inside→outside** (`separate_epilogue_store=True`, the producer subtiled but
  each subtile's consumer a *flat* op outside): represented as `numTiles` sibling
  `AllocChannel`s sharing one in-body template producer, each with its own flat
  consumer. There are two sub-shapes depending on staging `buffer.copy`:
  - **Multi-buffered** (`buffer.copy > 1`): the siblings are one reuse group over
    one physical multibuffer; the in-body slot rotation (`getOrComputeSubtiledSlot`)
    fires and the producer acquire/commit use one shared reuse-group token with a
    per-tile *slot*. Emitted by exactly one sibling (`emittedSubtiledProducerTokens`).
  - **Single-buffered** (`buffer.copy == 1`): the siblings are **distinct buffers**
    (different `buffer.id`, NOT a reuse group), so `getOrComputeSubtiledSlot`
    returns invalid and there is no shared slot. Each tile writes its own buffer
    (a per-tile buffer position) and must acquire/commit ONLY its own buffer's
    barrier. Because the tile body is replicated per tile, the `numTiles` sibling
    tokens are threaded as ONE **per-tile** arg (`addPerTilePosition`, ordered by
    the template store's per-tile buffer operands so token[t] matches the buffer
    tile t writes) and a single producer acquire/commit references it — so tile t
    handshakes exactly sibling t's barrier. Emitted once per region
    (`emittedSubtiledPerTileProducer`). Threading the tokens *shared* instead made
    every replicated tile arrive on *every* sibling's barrier (over-commit → the
    producer/consumer handshake is imbalanced → **runtime deadlock**).

  Two additional straddle hazards on this path (fixed): (a) the outer
  `bufferIdx`/`phase` for a flat consumer scheduled *before* the producer region
  must be anchored at the earliest endpoint, else it fails SSA dominance
  (verifier: `arith.trunci ... destroyed but still has uses`); (b) the flat
  consumer's `consumer_release` must be routed off its own per-token consumer, not
  the group-wide `tailConsumer` (which the straddle resolves to the region itself,
  misrouting the release into the producer partition and dropping it). See bug #10
  in the partition-scheduler rules and
  `test/Hopper/WarpSpecialization/ws_subtiled_inside_outside_channel.mlir`.
- **Same-task interleaved** (`separate_epilogue_store=False`): the producer
  `local_store` and the consumer `async_tma_copy_local_to_global` are both in the
  epilogue task, so there is no cross-task channel and *no WS reuse barrier* — the
  only drain sync is the per-tile `async_tma_store_token_wait`. These two endpoints
  must therefore live in **one** `SubtiledRegionOp` whose tile body is
  `store_t → copy_t → token_wait` (per tile), so the TMA wait drains a staging slot
  before a later subtile reuses it. `collectPerTileChain`
  (`GenerateSubtiledRegion.cpp`) achieves this by following a `local_store`'s SMEM
  buffer to a **same-task** TMA copy and pulling it into the same per-tile chain;
  `buildSingleSubtiledRegionN` then emits one region (the store and copy share one
  per-tile buffer position). Emitting two *sequential* same-task regions instead
  (all stores, then all copies) races the staging slot whenever
  `numTiles > buffer.copy` — the slot is overwritten before its copy drains —
  because, unlike the cross-task paths, there is no concurrency and no barrier to
  serialize the reuse. A debug assert in the separate-region branch guards this
  invariant. Because a same-task tile body now has both a write and a read of the
  one slot, `getReuseGroupStride` (`WSBuffer.cpp`) counts slot *lifecycles*
  (writes, falling back to reads), not raw buffer-touching ops, so the counter
  stride stays `numTiles`.
- **Both-endpoints-subtiled** (producer subtiled AND consumer subtiled, in
  *different* async tasks — the `DATA_PARTITION_FACTOR=2` epilogue): the
  `numTiles` per-tile allocs of one (producer region, consumer region) pair are
  **collapsed** into a single `AllocChannel` in `collectAllocChannels`
  (`getSubtiledChannelEndpoints` resolves the two regions; the dedup key is the
  region pair, gated on the regions being in different tasks). The collapsed
  channel is the sole member of a degenerate size-1 subtiled reuse group (see
  [Reuse Groups](ReuseGroups.md)), so the in-body slot math above is unchanged.
  This collapse fires for **any `buffer.copy`, including `buffer.copy == 1`** (the
  DP=1 epilogue): the `numBuffers > 1` reuse-group guards are relaxed for channels
  flagged `AllocChannel::isCollapsedBothSubtiled` (queried via
  `channelIsCollapsedBothSubtiled` — the *narrow* predicate, NOT the broad
  `channelIsSubtiled`, which would also match a consumer-only-subtiled bias load),
  the in-body math collapses all subtiles onto one physical slot (`bufferIdx == 0`,
  alternating phase), and the skipped sibling per-tile allocs
  (`AllocChannel::collapsedSiblingAllocs`) are erased after the rewire. Omitting
  any of this leaves the sibling staging alloc live → SMEM OOM (bug #13).
  Per-data-partition separation of the shared physical staging buffer is done in
  the memory planner (cross-partition staging split). See bug #11.

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
| `test/Hopper/WarpSpecialization/ws_code_partition_subtiled_region_inbody.mlir` | Both-endpoints-subtiled in-body SMEM rotation (DP=1, buffer.copy=3) |
| `test/Hopper/WarpSpecialization/ws_code_partition_subtiled_region_inbody_copy1.mlir` | Both-endpoints-subtiled in-body SMEM rotation (DP=1, buffer.copy=1: single-slot collapse + sibling erase, bug #13) |
| `test/Hopper/WarpSpecialization/ws_subtiled_region_inside_outside.mlir` | Asymmetric inside→outside (flat consumer) channel |
| `test/Hopper/WarpSpecialization/ws_subtiled_inside_outside_channel.mlir` | Inside→outside STRADDLE, single-copy distinct buffers: dominance anchor + per-tile producer token + per-token consumer release (addmm EPI=2, `separate_epilogue_store`, non-early-TMA) |
| `test/Hopper/WarpSpecialization/ws_subtiled_region_dp2_both_subtiled.mlir` | Both-endpoints-subtiled DP=2: cross-partition staging split + channel collapse |
| `python/test/unit/language/test_tutorial09_warp_specialization.py` | Blackwell GEMM e2e (parametrized with `generate_subtiled_region`) |
| `python/test/unit/language/test_autows_addmm.py` | Addmm e2e (parametrized with `generate_subtiled_region`) |
| `test_subtile_gemm.py` | Standalone addmm + subtile e2e |
