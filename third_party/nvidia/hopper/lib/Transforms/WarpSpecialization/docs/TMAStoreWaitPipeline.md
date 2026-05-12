# TMA Store Wait Pipeline

**File**: `WSTMAStoreLowering.cpp`, `WSMemoryPlanner.cpp`

After `doTMAStoreLowering` converts `tt::DescriptorStoreOp` into
`LocalAllocOp` + `AsyncTMACopyLocalToGlobalOp` + `TMAStoreTokenWaitOp`
(see [Memory Lowering](MemoryLowering.md#tma-store-lowering)), the
memory planner and a sequence of sub-passes handle these staging buffers.

Before this lowering, `doBufferAllocation` uses each `DescriptorStoreOp` as
the ordering anchor for channels feeding TMA stores. The producer-side
`local_store` order must match the descriptor-store order for the same TMA
descriptor; later wait annotation and rotation reason about the sequence of
stores to a shared staging buffer.

## Memory Planner: `isTMAStoreStaging` Handling

**File**: `WSMemoryPlanner.cpp` (within `allocateSmemBuffers`)

When `early_tma_store_lowering` is enabled, the `local_alloc` ops created
for TMA store staging are visible to the memory planner. These allocs feed
`AsyncTMACopyLocalToGlobalOp` and are detected by checking users:

```cpp
for (auto user : alloc->getUsers()) {
    if (isa<ttng::AsyncTMACopyLocalToGlobalOp>(user))
        buf.isTMAStoreStaging = true;
}
```

The `isTMAStoreStaging` flag triggers a special path through four phases:

### Phase 3.5: TMA Store Staging Fusion

All `isTMAStoreStaging` WSBuffers are merged into a single `bufferId`
(via `fuseEpilogueWSBuffers`). This groups the dk/dv epilogue store
staging buffers together. The merge uses the first buffer's ID for all.

Note: the shared `bufferId` affects `computeTotalSmem`'s cost model
(`max(size) × copies` per ID) but does **not** cause physical alloc
merging downstream — each alloc remains separate through
`AllocateSharedMemoryNv`.

### Phase 4.5: Epilogue Group Copy Increase

The merged TMA store group is treated as a P2_Other epilogue group.
`increaseFusedEpilogueCopies` iteratively increases copies (up to
`numBuffers`) while checking `computeTotalSmem ≤ smemBudget`.

Since `computeTotalSmem` excludes `isTMAStoreStaging` buffers from its
total, the budget check is effectively a no-op — copies always increase
to `numBuffers`. This is by design: TMA store staging buffers live
outside the pipelined inner loop and don't compete with channel buffers
for pipeline depth.

### Phase 4.6: Combined SMEM Budget Validation

After Phase 4.5, the combined SMEM cost is checked:

```
channelSmem = computeTotalSmem(wsBuffers)           // excludes TMA staging
tmaStoreSmem = computeTMAStoreStagingSmem(wsBuffers) // per-entry counting
if (channelSmem + tmaStoreSmem > smemBudget):
    cap all isTMAStoreStaging copies to 1
```

`computeTMAStoreStagingSmem` counts `numEntries × size × copies` (not
`max(size) × copies`) because the allocs are NOT merged into one physical
alloc downstream.

This prevents SMEM overflow for tight-budget configs where Phase 4.5
would otherwise increase TMA staging copies unchecked. For example:
BWD config 1 (BLOCK_M1=64, EPILOGUE_SUBTILE=2) has 4 TMA store staging
allocs of 16KB each — at 2 copies this is 128KB, exceeding the budget.
Phase 4.6 caps copies to 1 (64KB), fitting within hardware limits.

### Phase 6: Hoist Before Outermost Loop

All `isTMAStoreStaging` allocs are moved before the outermost enclosing
`scf.for` loop. This is required for the rotation mechanism
(`doAnnotateTMAStoreWaits`) which reads `buffer.copy` and only annotates
allocs that are outside all loops.

## Wait Annotation and Reordering Pipeline

Within the AutoWS monolithic pass (`WarpSpecialization.cpp`), three
functions handle the wait ops after the memory planner:

```
doMemoryPlanner
  → doAnnotateTMAStoreWaits      ← annotate waits with buffer count
  → doValidateTMAStoreAnnotations ← safety check
  → doCodePartitionPost
  → ...
  → scheduleLoops                 ← SWP assigns pipeline stages
  → doTMAStoreWaitReorder         ← move waits using the SWP schedule
```

Each function is also available as a standalone MLIR pass for use outside
the monolithic pipeline.

## Step 1: `doAnnotateTMAStoreWaits`

**Test pass**: `nvgpu-test-annotate-tma-store-waits` (`NVGPUTestAnnotateTMAStoreWaitsPass`)

This pass walks `scf.for` loops and inspects every `TMAStoreTokenWaitOp`.
For each wait, it traces the token back to the defining
`AsyncTMACopyLocalToGlobalOp`, then looks at the SMEM buffer used by that
store:

1. Get the `LocalAllocOp` that produces the buffer.
2. Read the `buffer.copy` attribute (set earlier by the memory planner),
   which records how many physical copies of this buffer exist.
3. If `buffer.copy = K`, set `can_rotate_by_buffer_count = K`
   on the wait op.

The attribute means: "K buffer copies exist, so this wait can be delayed
until the K-th subsequent TMA store to the same buffer — at that point
the buffer slot is about to be overwritten and the earlier store must
have finished reading."

### Token Tracing

`getDefiningTMAStore` handles two cases:

| Case | Pattern |
|------|---------|
| **Direct** | Token is the direct SSA result of `AsyncTMACopyLocalToGlobalOp` |
| **Loop-carried** | Token is a block argument of the `scf.for` body; the function follows the corresponding yield operand back to its `AsyncTMACopyLocalToGlobalOp` |

## Step 2: `doValidateTMAStoreAnnotations`

This is a safety pass that runs immediately after annotation. It
re-checks every annotated wait and strips the `can_rotate_by_buffer_count`
attribute if the defining TMA store or its `LocalAllocOp` can no longer
be resolved. This guards against IR transformations between annotation
and reordering that might invalidate assumptions.

## Step 3: `doTMAStoreWaitReorder`

**Test pass**: `nvgpu-test-tma-store-token-wait-reorder` (`NVGPUTestTMAStoreTokenWaitReorderPass`)

This pass runs **after** `scheduleLoops` has assigned pipeline stages and
clusters to every op. It uses the SWP `CoarseSchedule` to move waits
forward in the linearized pipeline order.

### Algorithm

For each annotated `TMAStoreTokenWaitOp` with `can_rotate_by_buffer_count = K`:

1. **Deserialize the schedule** from the `scf.for` loop. If no schedule
   exists, create a trivial single-stage schedule so the logic can still
   proceed.

2. **Linearize from the defining TMA store**: use
   `schedule.linearized(forOp, tmaStore)` to get an iterator that walks
   ops in pipeline-unrolled order (wrapping across stages up to
   `numStages + K`). Note: That we may only increase by 1 stage (we move
   by K TMA stores, not necessarily K pipeline stages).

3. **Count K copies**: walk the linearized schedule, counting
   `AsyncTMACopyLocalToGlobalOp` ops. Stop at the K-th copy — this is the
   point where the buffer slot would be reused.

4. **Adjust for barriers**: scan backwards from the insertion target to
   find a preceding `WaitBarrierOp`. If one exists, insert before it
   instead — this avoids placing the TMA store wait between a barrier
   wait and the ops it guards.

5. **Update the schedule**: split the cluster at the insertion target and
   create a new cluster for the wait op, assigned to the target's pipeline
   stage. Serialize the modified schedule back to the loop.

6. **Remove the annotation**: strip `can_rotate_by_buffer_count` from the
   wait op.

### Example

With `buffer.copy = 2` (double-buffered) and a 3-stage pipeline:

```
Stage 0: AsyncTMACopyLocalToGlobal (store to buffer[0])
         TMAStoreTokenWait          ← originally placed here
Stage 1: ...compute...
Stage 2: AsyncTMACopyLocalToGlobal (store to buffer[1])
```

After reordering with K=2, the wait moves forward to just before the 2nd
copy (which would overwrite buffer[0]):

```
Stage 0: AsyncTMACopyLocalToGlobal (store to buffer[0])
Stage 1: ...compute...
Stage 2: TMAStoreTokenWait          ← moved here
         AsyncTMACopyLocalToGlobal (store to buffer[1])
```

This allows the compute in stage 1 to overlap with the asynchronous TMA
store instead of stalling.

## Final Lowering: `NVGPUTMAStoreTokenWaitLoweringPass`

**Pass**: `nvgpu-tma-store-token-wait-lowering`

After reordering, a separate pass lowers each `TMAStoreTokenWaitOp` into
concrete hardware operations:

1. **Compute pendings**: count `AsyncTMACopyLocalToGlobalOp` ops between
   the defining store and the wait (in program order). For loop-carried
   tokens, this wraps around the loop body boundary.
2. **Emit `TMAStoreWaitOp`**: waits until at most `pendings` TMA stores
   remain in flight.
3. **Emit `ArriveBarrierOp`**: for each barrier attached to the wait,
   signals that the SMEM buffer is now free for reuse.
4. **Erase** the original `TMAStoreTokenWaitOp`.

See also [Memory Lowering](MemoryLowering.md) for the broader context of
how TMA stores fit into the WS memory lowering pipeline.
