# TMEM Allocation Heuristics

This document covers the TMEM (Tensor Memory) allocation algorithms in the
AutoWS memory planner. For SMEM allocation, see
[SmemAllocationDesign.md](SmemAllocationDesign.md). For reuse group mechanics
shared between SMEM and TMEM, see [ReuseGroups.md](ReuseGroups.md). For debug
visualization, see [MemoryPlannerVisualization.md](MemoryPlannerVisualization.md).

**File**: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlanner.cpp`

## TMEM vs SMEM Classification

The decision of what goes in TMEM vs SMEM is **not made by the memory planner**.
It is determined earlier in the pipeline during channel collection
(`collectPostChannels`). Channels are tagged at creation time based on the
operations involved:

| Channel Kind | Created For |
|-------------|------------|
| `TMEMPost` | `TMEMAllocOp` used by `TCGen5MMAOp`, MMA operand A/B via `TMEMStoreOp`, operand D (accumulator) |
| `SMEMPost` | `LocalAllocOp`, TMA loads (`AsyncTMACopyGlobalToLocalOp`, `DescriptorLoadOp`), `LocalStoreOp` |

The memory planner handles each kind independently: SMEM through
`MemoryPlanner` and TMEM through `MemoryPlannerTmem`.

## Entry Point: `doMemoryPlanner`

The top-level function (line 2289) orchestrates five steps:

```
Step 1: collectPostChannels      — gather all SMEM and TMEM channels
Step 2: SMEM planning            — MemoryPlanner::run() or allocateSmemBuffers()
Step 3: Visualization dump       — combined DOT graph
Step 4: TMEM planning            — MemoryPlannerTmem::run()
Step 5: Decision serialization   — optional JSON read/write for reproducibility
```

SMEM runs first and returns `lastBufferId`. TMEM starts numbering from there,
ensuring globally unique `buffer.id` values.

## TMEM Allocation Overview

TMEM on Blackwell has **512 rows** and a configurable number of columns. Each
`TMEMAllocOp` requires a contiguous block of rows and columns. The planner's
job is to assign `(rowOffset, colOffset)` to each allocation, minimizing total
row usage while respecting liveness constraints.

Key output attributes set on each `TMEMAllocOp`:
- `buffer.id` — groups allocations that share physical space
- `buffer.copy` — always 1 for TMEM (no multi-buffering at the TMEM level)
- `buffer.offset` — column offset within the owner's space (for reusing
  allocations)

## Sorting Priority

Before allocation, all `TMEMAllocOp`s are sorted (line 1217) with this
priority:

1. **Operand D first**: Accumulators (`isOperandD`) get highest priority.
   They tend to have the longest liveness and largest footprint, so allocating
   them first gives them the best row positions.

2. **Larger buffers first**: By total size (`numRows * numCols`), then by
   `numCols` alone, then `numRows` alone.

3. **Earlier liveness first**: For same-sized buffers, earlier
   `liveInterval.start()` wins.

4. **Buffers without channels last**: Allocations not associated with any
   channel are placed at the end.

## Liveness Computation

TMEM liveness is computed by `livenessForTmemChannel` (line 1040) and
`getLiveIntervals` (line 1140).

### User Collection

For each TMEM allocation, liveness is determined by collecting all operations
that use the allocation:

- **Operand D**: `getAllTmemUsers` collects **all direct users** of the
  `TMEMAllocOp` result, not just the channel endpoints. This is because the
  accumulator is both written by MMA and read by `tmem_load`, potentially
  across different partitions.

- **Non-operand-D**: Uses `getAllActualUsersForChannel` which traces the
  source op and actual consumers through the channel.

### Scope Normalization

`updateLiveOpsAcrossScopes` normalizes users to the same scope level and
collects all operations between first and last user. It also follows
`MemDescIndexOp` and `MemDescReinterpretOp` chains to capture subslice users.

The liveness interval is then `[firstUser, lastUser)` in the operation ID
space (from `buildOperationIdMap`).

## Algorithm 1: Greedy (`allocateTMemAllocs`)

The greedy algorithm processes sorted allocations sequentially.

### Core Logic

For each candidate allocation:

1. **`allInterfere` check**: If the candidate's liveness overlaps with ALL
   previously allocated buffers, it must get new row space (no reuse is
   possible since everything is live simultaneously).

2. **`findReuseChannel`**: Try to reuse an existing buffer's columns. The
   reuse criteria depend on the relationship between the candidate and the
   potential reuse owner:

   - **Different loops** (`!sameLoop`): Reuse if they have the same
     partitions (`samePartition`). The `partitionCondition` parameter controls
     strictness:
     - 0: always allow
     - 1: compare dst partition of owner with src partition of candidate
     - 2: compare combined task sets of all users

   - **Same loop** (`sameLoop`): Reuse if there is a data dependency chain
     (`alongDependencyChain`). Checks whether the consumer of the owner feeds
     into the producer of the candidate.

   After finding a potential owner, two additional checks run:
   - `findReuseSpace`: finds the first available column offset within the
     owner's space
   - `checkOtherReuses`: verifies no liveness overlap with other buffers
     already reusing the same owner at the computed column offset

3. **`allocateNewSpace`** (fallback): If no reuse is possible, allocate new
   row space at the maximum row offset so far. Enforces the **512-row limit**
   (line 1966).

### Column Reuse (Subslicing)

When one buffer has fewer columns than the owner, it gets a column offset
within the owner's row space. For example:

- A 128x128 f32 accumulator occupies 128 rows and 128 columns
- A 128x64 bf16 operand can reuse the same 128 rows at column offset 0,
  because it only needs 64 columns

This is implemented through `buffer.offset` and later materialized by
`sliceAndReinterpretMDTMEM` in code partitioning.

### All TMEM buffers get `buffer.copy = 1`

Unlike SMEM, TMEM does not support multi-buffering at the memory planner
level. Each TMEM allocation has exactly one copy.

## Algorithm 2: Backtracking (`allocateTMemAllocs2`)

A more sophisticated algorithm using recursive backtracking search.

### Data Structures

```cpp
struct AllocationState {
  DenseMap<BufferT *, std::pair<BufferT *, size_t>> assignment;  // buf → (owner, colOffset)
  DenseSet<BufferT *> owners;                                    // set of space owners
  size_t usedRows = 0;                                           // total rows consumed
};
```

### `hasPotentialReuse`

Returns a priority score for reusing an owner's space:
- **0**: cannot reuse (column too wide, liveness overlap, or no data
  dependency)
- **1**: can reuse (columns fit, no liveness overlap, has bidirectional data
  dependency)
- **2**: exact column size match (preferred)

The data dependency check uses bidirectional SSA def-use chain walking:
```cpp
isDataDependent(srcCh->getDstOp(), dstCh->getSrcOp()) ||
isDataDependent(dstCh->getDstOp(), srcCh->getSrcOp())
```
This verifies that there is a producer-consumer relationship between the two
channels in either direction.

### `tryAllocate` (Recursive Backtracking)

```
tryAllocate(allocs, idx, state, maxRows, ctrlOp):
  if idx == allocs.size(): return true  // base case: all allocated

  buf = allocs[idx]

  // Collect reuse candidates sorted by priority (2 = exact, 1 = can reuse)
  candidates = [(owner, priority) for owner in state.owners
                if hasPotentialReuse(owner, buf) > 0]
  sort(candidates, by priority descending)

  // Try each candidate
  for (owner, priority) in candidates:
    colOffset = computeColOffset(buf, owner, state)
    if colOffset is valid:
      assign buf → (owner, colOffset) in state
      if tryAllocate(allocs, idx+1, state, maxRows):
        return true
      // backtrack
      remove buf from state

  // Fallback: allocate new row space
  if state.usedRows + buf.rowSize <= maxRows:
    make buf an owner in state
    if tryAllocate(allocs, idx+1, state, maxRows):
      return true
    // backtrack
    remove buf from owners

  return false  // allocation failed
```

### `computeColOffset`

Determines where a candidate fits within an owner's column space:

1. For each existing reuser of the same owner, check if it can share columns
   with the candidate (via `hasPotentialReuse` in both directions).
2. If they **can** share columns: overlapping is OK (they are never live at
   the same time).
3. If they **cannot** share: place the candidate after the reuser's column
   range.
4. Return the maximum column offset, or `INVALID` if the candidate doesn't
   fit within the owner's total column width.

## Algorithm Selection

The algorithm is selected per-loop via the `tt.tmem_alloc_algo` attribute on
the `scf.for` operation:

| Value | Algorithm | When to Use |
|-------|-----------|-------------|
| 1 (default) | Greedy | Fast, works well for most kernels |
| 2 | Backtracking | Better packing for complex kernels with many TMEM buffers |

## Debug Tools

- **DOT graph visualization**: Set `TRITON_DUMP_WS_GRAPHS=/path/to/dir` to
  dump TMEM liveness graphs. See
  [MemoryPlannerVisualization.md](MemoryPlannerVisualization.md).

- **JSON serialization**: The `writeDecisionFile` / `readDecisionFile`
  parameters allow saving and replaying allocation decisions for
  reproducibility and debugging.

- **Debug logging**: `TRITON_LLVM_DEBUG_ONLY="nvgpu-ws-memory-planner"` enables
  detailed allocation step logging.
