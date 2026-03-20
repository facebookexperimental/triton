# Data Partitioning

Data partitioning physically splits tensor dimensions across multiple consumer
warp groups. After task assignment (which determines *which* ops run on
producers vs consumers), data partitioning determines *how* each consumer warp
group gets its slice of the data. For example, an M=256 accumulator is split
into two M=128 pieces for two consumer groups.

**File**: `WSDataPartition.cpp`
**Function**: `doDataPartition(funcOp, numConsumerGroups)`

## Pipeline Context

```
doTaskPartition          ← assigns ops to partitions
  → doTaskIdPropagate   ← propagates task IDs to all ops
  → doDataPartition     ← THIS STEP: splits tensor dimensions (Hopper only)
  → doPingPongPrep
```

Data partitioning runs only on Hopper. On Blackwell, the partition scheduling
pass (`PartitionSchedulingMeta`) handles spatial splitting differently.

## `DataPartitionScheme`

The central data structure tracking what to partition and how:

```cpp
struct DataPartitionScheme {
    unsigned numPartitions;                          // number of consumer groups
    SetVector<Operation *> ops;                      // ops to partition
    DenseMap<Operation *, unsigned> opPartitionDims;  // op → which dim to split
    DenseMap<Operation *, unsigned> dotPartitionOperand; // dot → which operand
    DenseMap<Operation *, SetVector<unsigned>> rematerializedOps; // ops to clone
    DenseSet<Operation *> opsToSkip;                 // ops exempt from partitioning
    DenseMap<unsigned, unsigned> funcArgPartitionDims; // func arg → partition dim
};
```

- `noOpPartitionDim`: Special sentinel value — ops with this dim are
  duplicated (cloned for each partition) rather than sliced.

## Algorithm

### Step 1: Task ID Fixup (`fixTaskId`)

Before partitioning, ensures all ops in def-use chains carry correct
`async_task_id` attributes via bidirectional propagation:

- **Backward**: If an op uses a value defined by an `arith` op that lacks the
  consumer's task ID, propagate backward.
- **Forward**: If a `YieldOp` or `IfOp` has a single-use operand whose
  defining op has extra task IDs, propagate forward.

Runs to a fixed point.

### Step 2: Compute Partition Scheme (`computePartitionScheme`)

Drives partitioning from dot/MMA ops:

1. Collect all `WarpGroupDotOp` and `TCGen5MMAOp` operations.
2. For each dot with multiple `async_task_id` values, determine the partition
   dimension from the accumulator shape:
   - **M dimension** (dim 0): if `shapePerCTA[0] / numPartitions >= 64`
   - **N dimension** (dim 1): if `shapePerCTA[1] / numPartitions >= 128`
   - M is preferred; N is fallback.
3. Call `getSliceToPartition` to trace the partition dimension through the
   dataflow graph.

### Step 3: Slice Propagation (`getSliceToPartition`)

Traces the partition dimension backward and forward from the accumulator:

- **`getBackwardSliceToPartition`**: From the accumulator, walks backward
  through operand definitions. Tracks how the partition dimension transforms
  through transposes (`TransOp`), expands (`ExpandDimsOp`), reshapes, and
  other shape-changing ops. Stops at loads, block arguments, and ops that
  produce scalar types.

- **`getForwardSliceToPartition`**: From the accumulator, walks forward
  through result users. Handles `YieldOp` (follow to loop result users),
  `IfOp` (follow to if result), and tracks dimension remapping through
  layout-changing ops.

### Step 4: Rematerialization (`rewriteRematerializedOps`)

When an op is reached with **conflicting partition dimensions** (e.g., used by
two dots partitioning along different dims), it is marked for rematerialization.
Only `LocalAllocOp` and `arith::ConstantOp` are eligible. The op is cloned —
one copy per partition dimension — and users are updated to reference the
appropriate clone.

### Step 5: Rewrite (`sliceOp`)

For each partition offset (0 to `numPartitions - 1`):

1. Clone each partitioned op with types adjusted — divide
   `shape[partitionDim]` by `numPartitions`.
2. An op with `async_task_id = [1, 2]` gets split into two copies: one with
   `[1]` and one with `[2]`.
3. Function arguments with `TensorDescType` have their block type sliced to
   match the partition factor.

### Step 6: Cleanup (`doDeepCleanup`)

After rewriting, runs dead code elimination and removes orphaned operations
that are no longer referenced after partitioning.

## Key Design Points

### Partition Dimension Tracking

The partition dimension is tracked through shape-changing operations:
- `TransOp`: remaps dimension via permutation order
- `ExpandDimsOp`: shifts dimension index if expansion is before the partition
  dim
- `SplatOp`, `BroadcastOp`: partition dim propagates unchanged
- `MakeRangeOp`, `LoadOp`: stop — these produce fresh data

### Function Argument Slicing

When a `TensorDescType` function argument feeds a partitioned op, its block
type is sliced. The `funcArgPartitionDims` map tracks which arguments need
slicing and along which dimension.

### Interaction with Task IDs

Data partitioning operates **after** task ID assignment. The offset parameter
selects which task ID from the original array. This is how N consumer warp
groups each get their slice of the data.
