# Utilities

This document covers the foundational utility infrastructure used throughout
the AutoWS pipeline.

## Files

| File | Description |
|------|-------------|
| `Utility.h` | `AsyncTaskId` typedef, `OpBuilderWithAsyncTaskIds`, `LoopScheduleInfo`, task ID helpers, location utilities |
| `Utility.cpp` | Implementation of task ID manipulation functions |

## Async Task ID Management

### Type

```cpp
typedef int AsyncTaskId;
```

Task IDs are stored as `DenseI32ArrayAttr` under the `"async_task_id"` key on
each operation. They can also be read from `ttg.partition` attributes (used by
`PartitionSchedulingMeta` before conversion to `async_task_id`).

### Functions

| Function | Description |
|----------|-------------|
| `getAsyncTaskIds(op)` | Returns sorted task IDs from `async_task_id` or `ttg.partition` attribute |
| `hasAsyncTaskId(op, id)` | Checks if an op has a specific task ID |
| `setAsyncTaskIds(op, ids)` | Sets the `async_task_id` attribute (sorted) |
| `addAsyncTaskIds(op, ids)` | Adds task IDs without duplicates |
| `removeAsyncTaskId(op, id)` | Removes a single task ID |
| `removeAsyncTaskIds(op)` | Removes the entire `async_task_id` attribute |
| `getNestedAsyncTaskIds(op)` | Collects task IDs from op and all nested ops |
| `labelParentOps(op)` | Propagates an op's task IDs upward to all parent ops |

### `labelParentOps`

After task IDs are assigned to leaf ops, parent ops (loops, if-ops) need the
union of their children's task IDs. `labelParentOps` walks the parent chain
up to the enclosing `FuncOp`, calling `addAsyncTaskIds` at each level.

## `OpBuilderWithAsyncTaskIds`

A custom `OpBuilder` subclass that **automatically sets `async_task_id` and
loop scheduling attributes** on every operation it creates. This is the
builder used throughout the entire WS pipeline.

### Key Methods

| Method | Description |
|--------|-------------|
| `createWithAsyncTaskIds<OpTy>(args...)` | Creates an op with the builder's current task IDs and loop schedule info |
| `create<OpTy>(args...)` | Alias for `createWithAsyncTaskIds` |
| `setAsyncTaskIdsFromOp(op)` | Copy task IDs from an existing op |
| `setAsynTaskIdsFromArray(ids)` | Set task IDs from an explicit array |
| `setAsyncTaskIdsFromValueUsers(value)` | Set task IDs from the union of all users of a value |
| `setLoopScheduleInfoFromOp(op)` | Copy `loop.stage` and `loop.cluster` from an op |
| `clearLoopScheduleInfo()` | Stop setting loop schedule attributes |

### Usage Pattern

```cpp
OpBuilderWithAsyncTaskIds builder(someOp);  // inherits task IDs + schedule
builder.setInsertionPointAfter(someOp);
auto newOp = builder.createWithAsyncTaskIds<SomeOp>(loc, args...);
// newOp automatically has async_task_id and loop.stage/loop.cluster set
```

## Loop Schedule Info

```cpp
struct LoopScheduleInfo {
    IntegerAttr stage;    // loop.stage attribute
    IntegerAttr cluster;  // loop.cluster attribute
};
```

These attributes are used by downstream loop scheduling passes to control
software pipelining. `OpBuilderWithAsyncTaskIds` preserves these attributes
through WS transformations so that pipeline stage assignments survive code
partitioning and specialization.

### `copyLoopScheduleInfo(newOp, oldOp)`

Copies `loop.stage` and `loop.cluster` attributes from `oldOp` to `newOp`.
Used when creating replacement operations where the dependency exists without
a direct SSA use (e.g., barrier operations that replace abstract tokens).

## Location Utilities

Helper functions for manipulating MLIR `Location` objects, used to give
meaningful debug names to channels and allocations:

| Function | Description |
|----------|-------------|
| `appendToNameLoc(loc, suffix, ctx)` | Appends a suffix to the innermost `NameLoc` in a location hierarchy |
| `getOutermostNameFromLoc(loc)` | Extracts the outermost `NameLoc` name, unwrapping `CallSiteLoc` |
| `replaceOutermostNameLoc(loc, name)` | Replaces the outermost name while preserving the `CallSiteLoc` wrapper and innermost child location |

These are used throughout channel creation to capture source-level names
(e.g., variable names from the Python DSL) for debug output and DOT graph
visualization.
