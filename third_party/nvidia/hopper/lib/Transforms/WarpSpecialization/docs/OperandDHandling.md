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

## Channel Creation: `CodePartitionUtility`

**File**: `CodePartitionUtility.cpp`

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

**File**: `WSMemoryPlanner.cpp`

Operand D receives special treatment in the TMEM memory planner:

### Allocation Priority

TMEM allocations are sorted before allocation with operand D getting the
**highest priority**:

```cpp
if (aCh->isOperandD && !bCh->isOperandD)
    return true;  // operandD always comes first
```

This ensures accumulators — which tend to have the longest liveness and the
largest TMEM footprint — are allocated first, getting the best row positions.

### Liveness Computation

For operand D channels, **all users** of the `TMEMAllocOp` result are
collected for liveness analysis, not just the channel's source and destination
ops (in `getAllTmemUsers`). This is because the accumulator is both written by
MMA and read by `tmem_load`, potentially across different partitions, and all
these uses must be accounted for to compute correct liveness intervals.

### Region Collection

In `collectRegionsWithChannelsPost()`, for operand D, the function iterates
over **all users** of the alloc op to find enclosing regions. This ensures
correct accumulation counter tracking when the accumulator is used in multiple
nested regions.

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

**File**: `WSCodePartition.cpp`

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

## Partition Scheduling: Operand D Markers

**File**: `PartitionSchedulingMeta.cpp`

The partition scheduling pass inserts `tmem.start` and `tmem.end` marker
attributes on operations to delineate the MMA accumulator's lifecycle. These
markers are used later by `TmemDataChannelPost` to identify the source
(`tmem.start`) and destination (`tmem.end`) operations of operand D channels.
