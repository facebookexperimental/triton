# Buffer Allocation

Buffer allocation is a pre-pass that discovers cross-partition channels,
creates or hoists SMEM and TMEM allocations to function scope, and
normalizes `local_alloc` ops for downstream code partitioning passes.

**File**: `WSCodePartition.cpp`
**Function**: `doBufferAllocation(funcOp)`
**Pass**: `NVGPUTestWSBufferAllocation`

## Pipeline Context

```
doTaskIdPropagate       ← assigns async_task_id to all ops
  → doBufferAllocation  ← THIS STEP: channels + alloc hoisting
  → doMemoryPlanner     ← decides multi-buffering (buffer.copy)
  → doCodePartitionPost ← inserts accumCnts, async copies, sync ops
```

`doBufferAllocation` creates single-copy buffers. Multi-buffering is
decided later by the memory planner. Code partitioning then uses
[accumulation counters](AccumulationCounters.md) to index into
multi-buffered allocations.

## Algorithm

### Step 0: `swapTransposedLocalAllocs`

When a `local_alloc` uses a transposed `#shared2` (NVMMAShared with
`transposed=true`) layout and its only use is a `memdesc_trans` back to
non-transposed `#shared` feeding MMA operand A, swap the layouts:

```
Before:  local_alloc → #shared_transposed  →  memdesc_trans → #shared
After:   local_alloc → #shared             →  memdesc_trans → #shared_transposed
```

This enables the alloc to share a buffer with other allocs of the same
source that already use `#shared` layout.

### Step 0.5: `mergeDuplicateLocalAllocs`

After layout normalization, merge `LocalAllocOp`s that have the same
source value and the same `MemDescType` — replace duplicates with the
first alloc.

### Step 1: `collectAsyncChannels`

Walk the function to find cross-partition data dependencies. For each
operation with a single `async_task_id` that is a **channel anchor op**
(loads, dots, allocs with source, etc.), call `createChannel` to identify
consumers in different partitions. All channels are created with
`numBuffers=1` (single-buffered).

### Step 2: `reorderEpilogOps`

Reorder epilogue operations (stores after the main loop) to align with
the expected producer completion order. Groups stores by type
(`DescriptorStoreOp` vs `StoreOp`) and interleaves them so
earlier-completed producers are consumed first.

### Step 3: `createBuffer`

The core step. For each channel (grouped by producer), create or hoist
the backing allocation to function entry:

- **TMEM channels** (existing `TMEMAllocOp` or `TCGen5MMAOp` source):
  Hoist the existing alloc to function entry via `hoistLocalAlloc`.

- **SMEM channels** (existing `LocalAllocOp` source):
  Hoist the existing alloc to function entry via `hoistLocalAlloc`.

- **Tensor-typed channels** (no existing alloc):
  Call `createLocalAlloc` which creates a new `LocalAllocOp` (SMEM)
  or `TMEMAllocOp` (for 1D tensors on Blackwell ≥ cc100). For
  post-channels (`isPost=true`), also inserts `LocalStoreOp` after
  the producer and `LocalLoadOp` before the consumer.

Channels sharing the same producer value share the same buffer.

### Step 4: `separateLocalAllocWithSrc`

Split any remaining `local_alloc %val` (alloc-with-source) into
`local_alloc` + `local_store %val`. This normalization exposes
cross-partition SMEM dependencies as separate store ops, enabling
downstream `doCodePartition`/`doCodePartitionPost` to detect them
as channels.

## Key Distinction

`doBufferAllocation` does **not** insert:
- Accumulation counters (see [Accumulation Counters](AccumulationCounters.md))
- Async copies or TMA lowering
- Tokens or synchronization ops (barriers, acquire/release)

Those are handled by `doCodePartition` / `doCodePartitionPost`.

## Key Functions

| Function | File | Description |
|----------|------|-------------|
| `doBufferAllocation` | `WSCodePartition.cpp` | Entry point |
| `swapTransposedLocalAllocs` | `WSCodePartition.cpp` | Layout normalization for buffer sharing |
| `mergeDuplicateLocalAllocs` | `WSCodePartition.cpp` | Dedup same-source allocs |
| `collectAsyncChannels` | `WSCodePartition.cpp` | Channel discovery |
| `reorderEpilogOps` | `WSCodePartition.cpp` | Epilogue store reordering |
| `createBuffer` | `WSCodePartition.cpp` | Buffer creation / hoisting |
| `createLocalAlloc` | `WSCodePartition.cpp` | New SMEM/TMEM alloc for tensor channels |
| `hoistLocalAlloc` | `WSCodePartition.cpp` | Move existing alloc to function entry |
| `separateLocalAllocWithSrc` | `WSCodePartition.cpp` | Split alloc+src into alloc + store |
