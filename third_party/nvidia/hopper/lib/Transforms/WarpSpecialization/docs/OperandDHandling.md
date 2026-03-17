# Operand D Handling in AutoWS

Operand D is the MMA accumulator — the result of a matrix multiply-accumulate
operation. On Blackwell, it resides in TMEM (`TMEMAllocOp`) and is written by
`TCGen5MMAOp`. On Hopper, it is the result of `WarpGroupDotOp`. Operand D
requires careful handling throughout the WS pipeline because it often crosses
partition boundaries (the MMA runs on the consumer, but the result may be read
by other partitions) and it carries state across loop iterations (accumulation).

## Overview of the Challenges

1. **Cross-partition communication**: The MMA (consumer partition) produces
   operand D, but downstream ops (e.g., epilogue stores, softmax rescaling)
   may run on different partitions. The accumulator value must be communicated
   via TMEM with proper barrier synchronization.

2. **Loop-carried accumulation**: In many kernels (e.g., Flash Attention), the
   accumulator persists across loop iterations — iteration N+1 reads the result
   of iteration N. This creates a loop-carried dependency that interacts with
   multi-buffering.

3. **Read-modify-write patterns**: When the accumulator is loaded, modified
   (e.g., rescaled), and stored back, multi-buffering of the accumulator is
   not possible because the value must be in-place.

## New Pipeline: `LoadMMASpecialization`

**File**: `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/LoadMMASpecialization.cpp`

The core handler is `pipelineMMA()` (lines 474-851), which processes each MMA
operation in the loop.

### Accumulator Multi-Buffering Decision

Before creating multi-buffered TMEM, the pass checks whether multi-buffering
is safe:

| Function | File | What It Checks |
|----------|------|---------------|
| `hasAccReadModifyWrite()` | `MMAv5PipelineUtility.cpp` | Traces `tmem_load` results through the loop body. Returns true if they flow back to a `tmem_store` to the same allocation (a read-modify-write cycle). |
| `isAccMultibufferingPossible()` | `MMAv5PipelineUtility.cpp` | Returns true if `use_acc` is false OR the accumulator is overwritten each iteration. |
| `disallow_acc_multi_buffer` | Loop attribute | An explicit opt-out on the `scf.for`. |

Multi-buffering is enabled when:
- `isAccMultibufferingPossible()` returns true, AND
- `disallow_acc_multi_buffer` is not set, AND
- The accumulator is actually read within the loop (`requiresAccMultiBuffering`).

When enabled, `createTMemAlloc()` prepends an extra leading dimension to the
TMEM allocation shape, giving `numMmaStages` copies of the accumulator buffer.

### The 3-Node Chain

`pipelineMMA()` models the accumulator lifecycle as three operations:

```cpp
struct Node {
  Operation *op;
  Value barPrev;  // barrier from previous node
  Value barNext;  // barrier to next node
  Value index;    // buffer index (for multi-buffering)
  Value phase;    // phase (parity bit for mbarrier wait)
};
SmallVector<Node, 3> nodes{Node{overwriteOp}, Node{mmaOp}, Node{readOp}};
```

| Node | Operation | Description |
|------|-----------|-------------|
| `overwriteOp` | `TMEMStoreOp` | Writes/initializes the accumulator (producer) |
| `mmaOp` | `TCGen5MMAOp` | The MMA operation that reads and writes the accumulator |
| `readOp` | `TMEMLoadOp` | Reads the MMA result (consumer) |

### Barrier Insertion at Partition Boundaries

Barriers are only inserted between adjacent nodes that are in **different
partitions** (different `async_task_id`). For example:

- If `overwriteOp` and `mmaOp` are in the same partition but `readOp` is in
  a different partition, a barrier is inserted between `mmaOp` and `readOp`.
- If all three are in the same partition, no barriers are needed.

Each barrier pair consists of:
- A **ready barrier** (signaling data is available to read)
- An **empty barrier** (signaling the buffer slot is available to write)

### Loop-Carried Dependencies

When the accumulator carries state across iterations (accumulation mode), the
loop-carried dependency is handled through TMEM buffer views:

- The `index` and `phase` values are threaded as loop-carried arguments.
- On each iteration, the index advances to the next buffer slot (modular
  arithmetic for circular buffering).
- The `overwriteOp` uses the next buffer slot while the `readOp` uses the
  current slot, allowing overlap.

### TMEM Allocation Shape Change

When multi-buffering is enabled, the TMEM allocation changes from:

```
tmem_alloc [M, N] → tmem_alloc [numStages, M, N]
```

Buffer views (`MemDescSubsliceOp`) index into the leading dimension to select
the current buffer slot.

## Legacy Pipeline: `CodePartitionUtility`

**File**: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/CodePartitionUtility.cpp`

The core handler is `handleOperandD()` (lines 2173-2479).

### Channel Creation

Operand D channels are `TmemDataChannelPost` objects with special flags:

| Flag | Meaning |
|------|---------|
| `isOperandD` | True when this channel represents the MMA accumulator |
| `isOperandDNoAcc` | True when `use_accumulator` is false (MMA overwrites rather than accumulates) |
| `isSameIterGuard` | True for same-iteration resource-hazard guards |

Detection in `createChannelPost()`:
```cpp
if (auto mmaOp = dyn_cast<TCGen5MMAOp>(user)) {
  if (mmaOp.getD() == allocOp->getResult(0)) {
    if (!isConstFalse(mmaOp.useAccumulator())) {
      isOperandD = true;
    }
  }
}
```

### Three Producer Patterns

`handleOperandD()` recognizes three patterns for how the accumulator is
initialized or updated:

1. **`TMEMStoreOp` outside the loop**: The accumulator is initialized before
   the loop begins (e.g., zeroed out). A channel from the store to the MMA
   is created.

2. **MMA with `use_accumulator = false`**: On the first iteration (or every
   iteration in non-accumulating kernels), the MMA overwrites the accumulator
   entirely. The channel gets `isOperandDNoAcc = true`.

3. **`TMEMStoreOp` inside the loop**: The accumulator is re-initialized
   mid-loop (e.g., after an epilogue store flushes results). This creates a
   wrap-around dependency.

### Wrap-Around Channels

For loop-carried accumulation, `handleOperandD()` creates **wrap-around
channels**: the MMA output at the end of iteration N feeds into the
`TMEMLoadOp` at the start of iteration N+1. These channels have special
ordering requirements in the code partitioning pass to maintain correctness:

```
tmem_load(dstOp of channel B) ...
tmem_store(srcOp of channel F) ...
gen5(srcOp of channel B, dstOp of channel F)
```

### Same-Iteration Guard Channels

When a `TMEMStoreOp` overwrites the accumulator in the same iteration that a
`TMEMLoadOp` reads it, a **guard channel** (`isSameIterGuard = true`) is
created. This prevents the store from executing before the load has finished
reading, which would corrupt the data. The guard channel adds a barrier
between the load and the store within the same iteration.

## Memory Planner: Operand D Priority

**File**: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlanner.cpp`

Operand D receives special treatment in the TMEM memory planner:

### Allocation Priority

TMEM allocations are sorted before allocation with operand D getting the
**highest priority** (line 1235):

```cpp
if (aCh->isOperandD && !bCh->isOperandD)
    return true;  // operandD always comes first
```

This ensures accumulators — which tend to have the longest liveness and the
largest TMEM footprint — are allocated first, getting the best row positions.

### Liveness Computation

For operand D channels, **all users** of the `TMEMAllocOp` result are
collected for liveness analysis, not just the channel's source and destination
ops (line 1022-1025 in `getAllTmemUsers`). This is because the accumulator is
both written by MMA and read by `tmem_load`, potentially across different
partitions, and all these uses must be accounted for to compute correct
liveness intervals.

### Region Collection

In `collectRegionsWithChannelsPost()` (line 577), for operand D, the function
iterates over **all users** of the alloc op to find enclosing regions. This
ensures correct accumulation counter tracking when the accumulator is used in
multiple nested regions.

## Task Partition: Operand D Assignment

In `WSTaskPartition.cpp`, the dot/MMA op is always assigned to the **consumer
partition**. Only operands A and B are backward-sliced to find producer ops:

```cpp
SetVector<Operation *> backwardSlice;
(void)getBackwardSlice(dotOp.getA(), &backwardSlice, opt);
(void)getBackwardSlice(dotOp.getB(), &backwardSlice, opt);
```

Operand D (the accumulator) stays with the MMA in the consumer partition.
Communication of the result to other partitions is handled by the channel
mechanism described above.

## Code Partitioning: Operand D Synchronization

**File**: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSCodePartition.cpp`

### `ProducerIsGen5()`

Checks if the producer of a TMEM channel is a `TCGen5MMAOp` by comparing
`mmaOp.getD()` with the alloc result. This determines whether the channel
represents an operand D flow.

### `desyncTCGen5MMAOp()`

Makes the MMA asynchronous with barriers for operand D communication between
partitions. When the MMA's result needs to cross a partition boundary, this
function:
1. Adds completion barriers to the MMA op
2. Sets the MMA as asynchronous (`setIsAsync(true)`)
3. The barriers are signaled via `tcgen05_commit` when the MMA finishes,
   allowing the consumer partition to safely read the result

See also [Barrier Fusion](BarrierFusion.md) for how `tcgen05_commit` is used
for operand D synchronization.
