# Barrier Analysis Implementation

`BarrierAnalysis.cpp` implements a comprehensive analysis pass for understanding barrier synchronization patterns in Triton's TTGIR. Here's how it works:

## Core Architecture

The implementation centers around the `BarrierExecutionOrderAnalysis` class, which runs a 5-stage pipeline:

```cpp
void BarrierExecutionOrderAnalysis::run() {
    collectBarrierOps();    // Stage 1: Gather all barrier operations
    assignBarrierIds();     // Stage 2: Assign unique IDs to each barrier
    groupByWarpGroup();     // Stage 3: Organize ops by async task
    analyzeDependencies();  // Stage 4: Build dependency graph
    detectIssues();         // Stage 5: Find potential problems
}
```

## Stage 1: Collecting Barrier Operations (`collectBarrierOps`)

Walks the function in pre-order, using LLVM's `TypeSwitch` to identify barrier operations from the `nvidia_gpu` dialect:

- **Memory barriers (mbarrier)**: `InitBarrierOp`, `InvalBarrierOp`, `BarrierExpectOp`, `WaitBarrierOp`, `ArriveBarrierOp`
- **Named barriers**: `NamedBarrierArriveOp`, `NamedBarrierWaitOp`

For each operation, it extracts:
- Operation kind and source location
- Static values (phase, expected bytes, thread count) when available from `arith::ConstantIntOp`
- Order index for topological ordering
- Async task ID by walking up the operation hierarchy looking for `async_task_id` attributes

## Stage 2: Assigning Barrier IDs (`assignBarrierIds`)

Uses a two-pass approach:
1. **First pass**: Assigns unique IDs to `InitBarrierOp` allocations, storing them in `allocToBarrierId` map
2. **Second pass**: Traces other barrier ops back to their allocation via `getBarrierAllocId()`, which follows the SSA def-use chain through view/subview operations

## Stage 3: Grouping by Warp Group (`groupByWarpGroup`)

Creates `WarpGroupInfo` structures for each async task ID:
- Groups barrier operations by their `asyncTaskId`
- Labels each warp group as **Producer** (init/arrive/expect/TMA ops) or **Consumer** (wait ops)
- Sorts operations within each group by execution order

## Stage 4: Analyzing Dependencies (`analyzeDependencies`)

Builds three types of dependency relationships by matching barrier IDs:

| Dependency Type | Producer → Consumer | Meaning |
|----------------|---------------------|---------|
| `InitThenUse` | init → arrive/wait | Barrier must be initialized before use |
| `ArriveThenWait` | arrive → wait | Arrive signals completion to wait |
| `ExpectThenWait` | expect → wait | Expected bytes + TMA satisfies wait |

## Stage 5: Detecting Issues (`detectIssues`)

Identifies potential synchronization problems:
- **`MISSING_ARRIVE`**: Wait without corresponding arrive (likely bug)
- **`MISSING_WAIT`**: Arrive without corresponding wait (may be intentional)
- Placeholder for deadlock detection via wait-for graph cycle detection

## Visualization Methods

### 1. `print()`
Structured text output showing ops grouped by warp group, dependencies, and cross-warp-group sync points.

### 2. `printExecutionTrace()`
ASCII timeline visualization:
```
Order:  0     2     4     6     8
        │───────────────────────────
main  │      I  E  W  A  X
```
Uses single-character symbols (I=init, E=expect, W=wait, A=arrive, X=inval).

### 3. `printDependencyGraph()`
Outputs DOT format for GraphViz visualization with nodes for each op and edges showing dependencies (cross-warp-group edges highlighted in red).

## Helper Functions

- `barrierOpKindToString()`: Converts enum to readable string
- `dependencyKindToString()`: Describes dependency relationships
- `issueKindToString()`: Labels detected problems
- `getAsyncTaskId()`: Walks parent ops to find async task context
- `getBarrierAllocId()`: Traces SSA chain to find barrier allocation

## Usage

```bash
triton-opt your_module.mlir -triton-print-barrier-analysis
```

## Example Output

```
========================================
 Barrier Execution Order Analysis
========================================

Barrier Operations by Warp Group:
─────────────────────────────────

[main] (Producer) (Consumer)
  [5] init_barrier (bar=0) @ loc("test.mlir":23:3)
  [6] barrier_expect (bar=0, bytes=256) @ loc("test.mlir":26:3)
  [7] wait_barrier (bar=0, phase=0) @ loc("test.mlir":29:3)
  [8] arrive_barrier (bar=0) @ loc("test.mlir":32:3)

Dependencies:
─────────────
  init_barrier[5] --[init->use]--> arrive_barrier[8]
  init_barrier[5] --[init->use]--> wait_barrier[7]
  arrive_barrier[8] --[arrive->wait]--> wait_barrier[7]
```
