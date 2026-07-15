# Utilities

This document covers the foundational utility infrastructure used throughout
the AutoWS pipeline.

## Files

| File | Description |
|------|-------------|
| `Utility.h` | `WSPartitionId` typedef, `OpBuilderWithPartitionIds`, `LoopScheduleInfo`, partition ID helpers, location utilities |
| `Utility.cpp` | Implementation of partition ID manipulation functions |

## Async Task ID Management

### Type

```cpp
typedef int WSPartitionId;
```

Partition IDs are stored as `DenseI32ArrayAttr` under the `ttg.partition` key
on each operation. The helper names still use `WSPartitionId` terminology, but
they operate on `ttg.partition`.

### Functions

| Function | Description |
|----------|-------------|
| `getWSPartitionIds(op)` | Returns sorted partition IDs from the `ttg.partition` attribute |
| `hasWSPartitionId(op, id)` | Checks if an op has a specific partition ID |
| `setWSPartitionIds(op, ids)` | Sets the `ttg.partition` attribute (sorted) |
| `addWSPartitionIds(op, ids)` | Adds partition IDs without duplicates |
| `removeWSPartitionId(op, id)` | Removes a single partition ID |
| `removeWSPartitionIds(op)` | Removes the entire `ttg.partition` attribute |
| `getNestedWSPartitionIds(op)` | Collects partition IDs from op and all nested ops |
| `labelParentOps(op)` | Propagates an op's partition IDs upward to all parent ops |

### `labelParentOps`

After partition IDs are assigned to leaf ops, parent ops (loops, if-ops, while-ops)
need the union of their children's partition IDs. `labelParentOps` walks the parent
chain up to the enclosing `FuncOp`, calling `addWSPartitionIds` at each level.

## `OpBuilderWithPartitionIds`

A custom `OpBuilder` subclass that **automatically sets `ttg.partition` and
loop scheduling attributes** on every operation it creates. This is the
builder used throughout the entire WS pipeline.

### Key Methods

| Method | Description |
|--------|-------------|
| `createWithPartitionIds<OpTy>(args...)` | Creates an op with the builder's current partition IDs and loop schedule info |
| `create<OpTy>(args...)` | Alias for `createWithPartitionIds` |
| `setPartitionIdsFromOp(op)` | Copy partition IDs from an existing op |
| `setPartitionIdsFromArray(ids)` | Set partition IDs from an explicit array |
| `setPartitionIdsFromValueUsers(value)` | Set partition IDs from the union of all users of a value |
| `setLoopScheduleInfoFromOp(op)` | Copy `loop.stage` and `loop.cluster` from an op |
| `clearLoopScheduleInfo()` | Stop setting loop schedule attributes |

### Usage Pattern

```cpp
OpBuilderWithPartitionIds builder(someOp);  // inherits partition IDs + schedule
builder.setInsertionPointAfter(someOp);
auto newOp = builder.createWithPartitionIds<SomeOp>(loc, args...);
// newOp automatically has ttg.partition and loop.stage/loop.cluster set
```

## Loop Schedule Info

```cpp
struct LoopScheduleInfo {
    IntegerAttr stage;    // loop.stage attribute
    IntegerAttr cluster;  // loop.cluster attribute
};
```

These attributes are used by downstream loop scheduling passes to control
software pipelining. `OpBuilderWithPartitionIds` preserves these attributes
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
