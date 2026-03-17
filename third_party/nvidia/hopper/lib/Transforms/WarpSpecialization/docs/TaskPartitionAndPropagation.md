# Task Partitioning & ID Propagation

This document explains how operations in a kernel are assigned to warp groups
(partitions) for warp specialization. Task partitioning is the first step in
the AutoWS pipeline — it decides which ops run on producer warp groups versus
consumer warp groups.

## Concepts

- **Partition / Async Task**: A group of operations that will execute on the
  same warp group. Identified by an integer ID.
- **Anchor op**: An operation whose partition assignment is determined directly
  (loads, MMAs, stores). Non-anchor ops are assigned by propagation.
- **Producer**: The warp group responsible for memory loads (typically task 0).
- **Consumer**: The warp group responsible for computation — MMA / tensor core
  ops (task 1+).
- **Data partitioning**: After task assignment, consumer ops can be further
  split along spatial dimensions (M/N) across multiple consumer warp groups.

## Partition Scheduling: `PartitionSchedulingMeta`

**File**: `PartitionSchedulingMeta.cpp`

An extended partition scheduling pass with template-based scheduling for Flash
Attention and GEMM patterns. This pass runs before the main WS pipeline on
Blackwell, assigning `ttg.partition` attributes that are later converted to
`async_task_id` by `WSTaskIdPropagate`.

### Op Categorizer

Ops are classified into rich categories:

| Category | Description |
|----------|-------------|
| `TMALoad` | `DescriptorLoadOp`, `AsyncTMACopyGlobalToLocalOp` |
| `MMA` | `TCGen5MMAOp`, `WarpGroupDotOp` |
| `EpilogueStore` | `DescriptorStoreOp`, stores at loop end |
| `TMEMStore` | `TMEMStoreOp` |
| `TMEMLoad` | `TMEMLoadOp` |
| `BlockPointerAdvance` | `AdvanceOp` for TMA descriptors |
| `DataPartition` | Ops exclusive to one MMA's backward slice (detected via union-find grouping of dependent MMAs) |
| `Correction` | Cross-iteration MMA users (e.g., softmax rescaling) |
| `TMAReduction` | `DescriptorReduceOp`, `AsyncTMAReduceOp` |

### Scheduling Templates

- **`UnifiedFATemplate`**: For Flash Attention patterns (correction ops, multiple
  MMAs, or data partition factor > 1). Creates reduction partition (BWD) or
  correction partition (FWD) in addition to load/MMA/epilogue.
- **`GEMMTemplate`**: Simple default/gemm/load/epilogue.

Template selection: use `UnifiedFATemplate` if correction ops exist, multiple
MMAs exist, or `dpFactor > 1`. Otherwise `GEMMTemplate`.

### Partition Assignment

| Op Type | Partition |
|---------|-----------|
| TMA loads, block pointer advances | Partition 0 (producer) |
| MMA ops | Partition 1+ (consumer) |
| Epilogue stores | Epilogue partition |
| Correction ops | Correction/reduction partition |

### Key Differences From Upstream

**Propagation**: For BWD-like kernels (has reduction, no epilogue), ambiguous
clusters reuse the existing computation partition rather than creating new ones.

**Operand D handling**: Inserts `tmem.start`/`tmem.end` marker attributes and
creates operand-D channels for MMA accumulator lifecycle management.

**Partition type annotation**: Tags loops with `tt.partition_types` (producer,
compute, epilogue).

### Output

Ops are tagged with `ttg.partition` attributes. The pass skips if manual TLX
`async_tasks` are present.

## Task Partition: `WSTaskPartition`

**File**: `WSTaskPartition.cpp`

A simpler approach using backward slicing from dot/MMA ops. Used on Hopper.

### Algorithm

1. Collect all `scf::ForOp` loops, `WarpGroupDotOp`, load ops, and store ops.
2. For each dot, compute the backward slice of operands A and B.
3. Any `DescriptorLoadOp` (or expensive `LoadOp`) in the backward slice is a
   **producer** (task ID 0).
4. All dots are **consumers** (task IDs 1 through `numWarpGroups - 1`).
5. All stores get consumer task IDs.

**Key point**: only operands A and B are backward-sliced. The dot itself (and
its accumulator / operand D) always stays in the consumer partition.

## Task ID Propagation

**Files**:
- `TaskIdPropagation.cpp` (analysis)
- `WSTaskIdPropagate.cpp` (materialization)

After anchors are assigned task IDs, many intermediate ops remain unannotated.
Task ID propagation fills these gaps.

### Dataflow Analysis

`TaskIdBackwardPropagation` is a sparse backward dataflow analysis using MLIR's
analysis framework.

**Lattice**: `TaskId` has three states:
- **Uninitialized**: not yet visited
- **Known**: a set of task IDs (e.g., `{0, 1}`)
- **Unknown**: conflicting information

**Meet operation**: union of task ID sets. An op used by tasks `{0, 1}` and
`{1, 2}` gets `{0, 1, 2}`.

**Transfer function** (`visitOperation`):
- **Anchor ops** (non-scalar ops with `async_task_id`): define partitioning
  boundaries. Task IDs flow backward to operands but are not overridden.
- **Non-anchor ops** (including scalar arith/math): standard backward
  propagation — task IDs flow from results to operands.
- Scalar arith/math ops are always non-anchors, allowing task IDs to flow
  through shared address computations.

### Materialization (`doTaskIdPropagate`)

1. Convert `ttg.partition` → `async_task_id` (normalize indices by subtracting
   the minimum partition ID).
2. Handle operand D initialization: find `TMEMStoreOp` before the loop that
   writes to the MMA's accumulator, assign it the appropriate task ID.
3. Mark all `scf::ForOp` loops with the union of all task IDs.
4. Run the backward dataflow solver.
5. Materialize: update `async_task_id` on all ops from the solver's lattice.
6. `labelParentOps`: ensure parent ops have the union of their children's
   task IDs.

## Data Partitioning

**File**: `WSDataPartition.cpp`

After task assignment, data partitioning physically splits tensor dimensions
across multiple consumer warp groups. For example, an M=256 accumulator is split
into two M=128 pieces for two consumer groups.

### Algorithm

1. **Compute partition scheme**: For each dot/MMA, determine which dimension
   to split (M if `shapePerCTA[0] / numPartitions >= 64`, else N if
   `shapePerCTA[1] / numPartitions >= 128`).

2. **Backward + forward slicing**: From the accumulator, trace backward through
   operand definitions and forward through result users, adjusting the partition
   dimension through transposes, expands, and other shape-changing ops.

3. **Rematerialization**: If an op is reached with conflicting partition
   dimensions, clone it (only `LocalAllocOp` and `arith::ConstantOp`).

4. **Rewrite**: For each partition offset, clone ops with types adjusted
   (divide `shape[dim]` by `numPartitions`). An op with
   `async_task_id = [1, 2]` gets split into two copies: one with `[1]` and
   one with `[2]`.

### Relationship to Task IDs

Data partitioning operates **after** task ID assignment. The offset parameter
selects which task ID from the original array. This is how N consumer warp
groups each get their slice of the data.
