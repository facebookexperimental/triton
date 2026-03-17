# Task Partitioning & ID Propagation

This document explains how operations in a kernel are assigned to warp groups
(partitions) for warp specialization. Task partitioning is the first step in
both AutoWS pipelines — it decides which ops run on producer warp groups versus
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

## New Pipeline: `PartitionScheduling`

**File**: `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp`

This pass runs on `scf.for` loops annotated with `tt.warp_specialize`. It
creates a `PartitionSet` and assigns each op to one or more partitions.

### Op Classification

| Category | Ops | Default Partition |
|----------|-----|------------------|
| Load | `DescriptorLoadOp`, `DescriptorGatherOp` | `loadPartition` (stage 0) |
| MMA | `MMAv5OpInterface` | `mmaPartition` (stage 1) |
| Store | `DescriptorStoreOp` | `epiloguePartition` (stage 0) |
| View | Ops with `MemDescViewTrait` | Cloned into `mmaPartition` if shared |
| Other | Everything else | Determined by propagation |

### Algorithm

1. **Create partitions**: Four initial partitions — `defaultPartition` (stage 0),
   `mmaPartition` (stage 1), `loadPartition` (stage 0), `epiloguePartition`
   (stage 0).

2. **Schedule anchors**:
   - TMA loads → `loadPartition`. Their `LocalAllocOp` / `TMEMAllocOp` users
     with matching shared encoding also go to `loadPartition`.
   - MMA ops → `mmaPartition`. Accumulator-initializing `TMEMStoreOp` (value
     defined outside loop, no read-modify-write) also goes to `mmaPartition`.
   - Stores → `epiloguePartition`.

3. **Schedule load users**: Transitively place users of load results into
   `defaultPartition`.

4. **Schedule cross-iteration MMA users** ("correction" pattern, e.g., online
   softmax rescaling): ops that consume the MMA result yielded from the previous
   iteration go to `defaultPartition`.

5. **Per-MMA computation partitions**: For each MMA, `scheduleUsers(nullptr)`
   creates a new dynamic partition for ops that exclusively use that MMA's
   results. This enables distinct computation partitions for different MMA
   operations (useful in multi-MMA kernels like Flash Attention).

6. **Propagate to remaining ops** (`propagatePartitions`):
   - Builds `OpCluster` objects — sets of adjacent unscheduled ops in the SSA
     graph, tracking which partitions provide definitions (`defPartitions`) and
     which consume results (`sinkPartitions`).
   - For clusters with multiple def/sink partitions: create a new partition.
   - For clusters with a single def and single sink: analyze the critical path.
     If all ops are on the critical path, assign to the def partition. Otherwise,
     rematerialize critical-path ops into the sink partition and assign the rest
     to the def partition.

7. **Optimize schedule**: Clone `BroadcastOp`, `ExpandDimsOp`, `ConvertLayoutOp`
   into each consuming partition to transfer smaller pre-broadcast values through
   shared memory.

8. **Assign root partition**: Any still-unscheduled ops are assigned to all
   partitions (they are needed everywhere).

### Output

Ops are tagged with `ttg.partition` attributes (integer indices into the
`PartitionSet`). These are later converted to `async_task_id` by
`WSTaskIdPropagate`.

## Meta Variant: `PartitionSchedulingMeta`

**File**: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/PartitionSchedulingMeta.cpp`

An extended version with template-based scheduling for Flash Attention and GEMM.

### Additions Over Upstream

**Op categorizer** with richer categories:
- `DataPartition`: ops exclusive to one MMA's backward slice (detected via
  union-find grouping of dependent MMAs)
- `Correction`: cross-iteration MMA users
- `TMAReduction`: `DescriptorReduceOp`, `AsyncTMAReduceOp`

**Scheduling templates**:
- **`UnifiedFATemplate`**: For Flash Attention patterns (correction ops, multiple
  MMAs, or data partition factor > 1). Creates reduction partition (BWD) or
  correction partition (FWD) in addition to load/MMA/epilogue.
- **`GEMMTemplate`**: Simple default/gemm/load/epilogue.

Template selection: use `UnifiedFATemplate` if correction ops exist, multiple
MMAs exist, or `dpFactor > 1`. Otherwise `GEMMTemplate`.

**Key difference in propagation**: For BWD-like kernels (has reduction, no
epilogue), ambiguous clusters reuse the existing computation partition rather
than creating new ones.

## Legacy Pipeline: `WSTaskPartition`

**File**: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSTaskPartition.cpp`

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
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/TaskIdPropagation.cpp`
  (analysis)
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSTaskIdPropagate.cpp`
  (materialization)

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

**File**: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSDataPartition.cpp`

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
