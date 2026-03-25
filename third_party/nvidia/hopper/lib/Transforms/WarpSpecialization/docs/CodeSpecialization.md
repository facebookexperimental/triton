# Code Specialization

Code specialization is the step that physically separates operations into
distinct `WarpSpecializeOp` regions — one region per partition. Before this
step, operations coexist in a single function body with `async_task_id`
annotations. After specialization, each partition has its own isolated region
that will execute on a dedicated warp group.

**File**: `WSSpecialize.cpp`
**Function**: `specializeRegion(funcOp, requestedRegisters)`

## Pipeline Context

```
doCodePartitionPost     ← channels and barriers created
  → specializeRegion    ← THIS STEP: ops cloned into regions
  → doPingPongSync      ← named barriers inserted within regions
  → doTokenLowering     ← abstract tokens lowered to hardware barriers
```

## Algorithm

### Step 1: Create `WarpSpecializeOp`

A `ttg.WarpSpecializeOp` is created with:
- A **default region** for the producer (task 0)
- **N partition regions** for consumers (tasks 1 through N)
- Per-partition warp counts

### Step 2: Collect and Sort Operations

All operations with `async_task_id` attributes are collected and
topologically sorted. Each operation is then assigned to the appropriate
region based on its task ID.

### Step 3: Clone Operations

For each partition (starting with the default region, then each consumer
region), `SpecializeOp` recursively clones operations into the target region
using `IRMapping`.

#### `SpecializeForOp`

`scf::ForOp` requires special handling because different partitions may use
different subsets of the loop's block arguments and yield values:

1. Collect only the block arguments used by the specific task.
2. Create a **trimmed loop** with only the needed arguments.
3. Recursively clone body ops that belong to this partition.
4. Build a yield that only produces values used by this partition.

This means the same source loop may become different loops in different
partition regions, each with a reduced set of loop-carried values.

#### `SpecializeIfOp`

Similarly, `scf::IfOp` regions are cloned with reduced result sets — only
results used by the partition are kept.

### Step 4: Handle Captures

Values defined outside the `WarpSpecializeOp` but used inside it become
**captures**:

- **Constants** (`arith::ConstantOp`): rematerialized inside each region
  that uses them. This avoids unnecessary captures for trivially recomputable
  values.
- **Other values**: threaded as operands to the `WarpSpecializeOp` and
  mapped to corresponding block arguments in each region.

### Step 5: Cleanup

After all operations are cloned into their respective regions:
- Dead code elimination (DCE) removes unused operations within each region.
- Original operations in the function body are erased.

## Key Design Decisions

### Trimmed Loops

Instead of cloning the full loop into every partition, each partition gets a
loop with only the block arguments and yield values it actually uses. This
reduces register pressure and eliminates unnecessary loop-carried values.

### Constant Rematerialization

Constants are cheap to recompute, so they are cloned into each region rather
than captured. This avoids register file pressure from captures that would
otherwise hold constant values across the `WarpSpecializeOp` boundary.

### Topological Ordering

Operations are processed in topological order to ensure that when an
operation is cloned, all of its operand definitions (within the same
partition) have already been cloned and are available in the `IRMapping`.
