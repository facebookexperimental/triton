# Code Partitioning

Code partitioning is the central step of the AutoWS pipeline — it discovers
cross-partition data dependencies, creates channels and buffers, inserts
synchronization primitives (tokens, barriers), and materializes async copies.
This is the largest and most complex file in the WS pipeline.

**File**: `WSCodePartition.cpp`

## Two Phases (one production flow)

Code partitioning is the second half of a **two-phase flow** that runs the same
way on Hopper and Blackwell (partitions are assigned earlier, by
`PartitionSchedulingMeta` on Blackwell or `doTaskPartition` on Hopper):

1. **Buffer-allocation phase** — `doBufferAllocation` discovers channels from raw
   producers and hoists single-copy allocations (see the pre-pass below and
   [Buffer Allocation](BufferAllocation.md)).
2. **Memory-planner phase** — `doMemoryPlanner` decides multi-buffering
   (`buffer.copy`) and reuse (`buffer.id`).
3. **Code-partition phase** — `doCodePartition` rediscovers channels from the
   now-existing allocations, builds multi-buffer arrays, and inserts the
   synchronization protocol.

The two phases use two channel representations. The buffer-allocation phase uses
the raw base `Channel` (anchored on a consumer op + operand index, discovered by
`collectAsyncChannels`/`createChannel`, kinds `SMEM`/`TMEM`/`REG`). The
code-partition phase uses `AllocChannel` / `TmemAllocChannel` (anchored on an
existing `local_alloc` / `tmem_alloc`, kinds `SMEMAlloc`/`TMEMAlloc`). This is a
**phase** distinction, not a target distinction — production runs
`doCodePartition` on every architecture.

### `doCodePartition` — code-partition phase

```
Step 1: collectAllocChannels        — discover channels from existing allocs
Step 2: collectRegionsWithChannels — find control flow with channels
Step 3: detect reuse groups        — group channels by buffer.id
Step 4: appendAccumCntsForOps      — add accumulation counter loop args
Step 5: createBufferForAllocs           — create multi-buffer arrays for existing allocs
Step 6: createToken                — create tokens and barriers
Step 7: insertAsyncComm            — insert sync ops; lower TMA loads (optimizeTMALoads)
Step 8: fuseTcgen05CommitBarriers  — fuse redundant tcgen05_commit ops
Step 9: cleanupTmemTokens          — replace TMEM op tokens with poison
Step 10: replaceBufferReuse        — rewrite non-representative allocs
Step 11: specializeRegion          — clone ops into WarpSpecializeOp regions
```

## `doBufferAllocation` — Pre-pass

**Function**: `doBufferAllocation(funcOp)`

A separate entry point for pre-processing before the main pipeline.
See [Buffer Allocation](BufferAllocation.md) for details.

```
Step 0:   swapTransposedLocalAllocs   — normalize transposed alloc layouts
Step 0.5: mergeDuplicateLocalAllocs   — deduplicate allocs with same source
Step 1:   collectAsyncChannels        — discover channels
Step 2:   reorderEpilogOps            — interleave epilogue stores
Step 3:   createBuffer                — allocate buffers (single copy)
Step 4:   separateLocalAllocWithSrc   — split local_alloc(src) → alloc + store
```

## Channel Discovery

### `collectAsyncChannels`

Walks the function to find all cross-partition data dependencies:

1. For each operation with `async_task_id`, check if it is a **channel anchor
   op** (`isChannelAnchorOp`).
2. If so, call `createChannel` to identify consumers in different partitions.

### `isChannelAnchorOp`

An operation can be a channel endpoint if it is:
- A load (`LoadOp`, `DescriptorLoadOp`)
- An MMA/dot op (`DotOpInterface`)
- A `TMEMStoreOp`
- A `LocalAllocOp` with a source operand
- Any op producing a `RankedTensorType` result

### `createChannel`

The core channel creation logic:

1. For each result of the producer op, collect all **transitive users**
   (`getTransitiveUsers`) — tracking through loop-carried values to reach real
   users across iterations. For `scf.for`, this follows `scf.yield` to loop
   results. For `scf.while`, after-region `scf.yield` values feed the next
   iteration's before-region arguments, while `scf.condition` forwarded
   operands feed while results and after-region arguments.
2. Filter by **dominance**: only consider users properly dominated by the
   producer.
3. For each user in a **different partition** (different `async_task_id`),
   create a `Channel` with the appropriate kind (`SMEM`, `TMEM`, or `REG`).

### `collectAllocChannels`

For the post-allocated path, channels are discovered from existing
`LocalAllocOp` and `TMEMAllocOp` operations rather than from raw producers.
Creates `AllocChannel` (SMEM) or `TmemAllocChannel` (TMEM) objects. Also
calls `handleOperandD` to create operand D channels for MMA accumulators.

## Epilogue Reordering

### `reorderEpilogOps`

Groups epilogue stores by type (`DescriptorStoreOp` vs `StoreOp`), then
interleaves them so earlier-completed producers are consumed first. Uses
forward/backward slicing to pack dependent ops close together.

For TMA stores, `DescriptorStoreOp` order is also the order used by the
epilogue-store partition after buffering. `createBuffer` inserts
`local_store` ops in producer order, so `reorderEpilogOps` restores producer
order for channels feeding the same descriptor to match the descriptor-store
order. This keeps the producer-side `local_store` sequence aligned with the
consumer-side TMA store sequence.

## Buffer Creation

### `createBuffer` / `createBufferForAllocs`

Creates SMEM or TMEM allocations for each channel:

- **`hoistLocalAlloc`**: Moves allocations to function entry, converting
  `local_alloc(src)` into `local_alloc() + local_store(src)`.
- **`createLocalAlloc`**: Creates new allocations, choosing between SMEM and
  TMEM based on tensor dimensionality. Selects shared memory encoding
  (`NVMMAShared` for MMA consumers, unswizzled for others, TMA encoding for
  TMA stores).
- **`createBufferForAllocs`**: For the post-allocated path, groups channels
  sharing the same `allocOp` and creates multi-buffer arrays.

## Token and Barrier Creation

### `createToken`

Creates synchronization tokens for each channel group:

- For each consumer group, creates a `CreateTokenOp` with `numBuffers` slots.
- **TMA barrier pre-allocation**: When any channel in a group has a TMA
  producer, an mbarrier array is pre-allocated via `BarrierAllocOp`.
- **Gen5 inline barriers**: For MMAv5 consumers (`TCGen5MMAOp` and
  `TCGen5MMAScaledOp`), decides whether to use the MMA op's built-in
  completion barrier instead of a separate token (checked via
  `ProducerIsGen5`).
- Results are stored in a `CommChannel` struct per channel, containing
  `tokens` (per consumer task ID), optional `producerBarrier` (for TMA/gen5),
  and optional `consumerBarriers` (for gen5 inline barriers).

## Synchronization Insertion

### `insertAsyncComm`

The largest function (~950 lines) — inserts the full synchronization protocol
for each channel group. See [Barrier Insertion](BarrierInsertion.md) for the
detailed decision tree, code paths, and a worked FA BWD example.

1. **Compute head/tail**: Find the first and last producer/consumer ops.
2. **Scope lifting**: When producer and consumer are at different nesting
   levels, uses `isAinNestedRegion` and `getSameLevelOp` to lift operations
   to the correct scope.
3. **Insert sync ops**: For each channel:
   - `ProducerAcquireOp` before the producer (wait for buffer to be free)
   - `ProducerCommitOp` after the producer (signal data is ready)
   - `ConsumerWaitOp` before the consumer (wait for data)
   - `ConsumerReleaseOp` after the consumer (signal buffer is free)
4. **`desyncMMAv5Op`**: Makes MMAv5 ops fully asynchronous by attaching a
   completion barrier and creating a `WaitBarrierOp`.
5. **Consumer release placement**: `consumerReleaseHeuristic` uses
   post-dominance analysis to find optimal placement.
6. **Data-partitioned commit replacement**: In data-partitioned loops
   (`tt.data_partition_factor > 1`) with multiple MMAs, the D-channel
   creation sites generate `wait_barrier` + `arrive_barrier` pairs directly
   instead of `tcgen05_commit` ops. Each MMA gets a per-MMA wait on the
   MMA's existing inline A/B barrier (from the final loop iteration)
   followed by an arrive on the D barrier, enabling per-MMA completion
   tracking. This avoids the problem with `tcgen05_commit`, which is a
   global fence that commits ALL pending async operations — the first
   commit would wait for every MMA to finish, serializing them. When there
   is only a single MMA in the loop, the standard `tcgen05_commit` is used
   since there is no serialization concern. The replacement is handled by
   `replaceCommitWithBarrierSync`, called at the two commit creation sites
   in `insertAsyncComm` (the `producerBarrier` and `consumerBarrier` paths).
   **Invariant**: each call to `replaceCommitWithBarrierSync` must represent
   the work of exactly one MMA — the commit being replaced must correspond
   to a single MMA's D-channel, not aggregate work from multiple MMAs. This
   is structurally guaranteed because the call sites iterate per-channel
   (each D-channel maps to one MMA), and the `mmaCount > 1` guard at each
   call site ensures the replacement is only attempted when data partitioning
   has produced multiple distinct per-MMA channels.

### Channel Loop Detection

- **`isForwardOfChannelLoop`** / **`isBackwardOfChannelLoop`**: Detect
  operand D TMEM channel cycles where the same TMEM allocation is both
  produced and consumed in the same loop iteration (wrap-around channels).
- **Guard channel handling**: `isSameIterGuard` channels protect
  `tmem_load` → `tmem_store` resource hazards within the same iteration.
  Uses token-based synchronization instead of gen5 inline barriers.

## IR Cleanup Passes

### `foldLocalLoads`

Eliminates redundant `local_load` + `local_alloc` patterns when the load
result has a single use that is an alloc.

### `cleanupTmemTokens`

Replaces TMEM operation tokens with poison values since synchronization is
now handled by the WS infrastructure.

### `separateLocalAllocWithSrc`

Splits `local_alloc(src)` into `local_alloc() + local_store(src)` so
downstream channel detection can identify cross-task SMEM channels.

### `swapTransposedLocalAllocs`

When a transposed `local_alloc` feeds into `memdesc_trans` which feeds MMA
operand A, swaps the layouts so the alloc uses non-transposed layout. This
enables buffer sharing with other allocs of the same source.

### `mergeDuplicateLocalAllocs`

Merges `LocalAllocOp`s that have the same source value and layout into a
single allocation.
