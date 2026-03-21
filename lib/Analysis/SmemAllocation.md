# SMEM Allocation Analysis

This document describes Triton's core shared memory (SMEM) allocation analysis,
implemented in `Allocation.cpp`. This analysis assigns non-overlapping SMEM
offsets to all buffers that are live at the same time, minimizing total SMEM
usage.

> **Scope.** This covers the _core Triton_ allocator (`lib/Analysis/`), which
> runs as part of the standard TTGIR pipeline for all backends. The AutoWS
> memory planner (`WSMemoryPlanner`) is a separate, more specialized allocator
> documented in its own `docs/` directory under the warp specialization passes.

## Overview

The allocator has three phases:

1. **Buffer discovery** — find every SMEM buffer and compute its size
2. **Liveness analysis** — determine when each buffer is live
3. **Offset assignment** — assign SMEM offsets so that simultaneously-live
   buffers don't overlap

The algorithm is based on the paper
[_Algorithms for Compile-Time Memory Optimization_](https://dl.acm.org/doi/pdf/10.5555/314500.315082).

## Buffer Kinds

Every SMEM buffer has one of three kinds:

| Kind | Source | Example |
|------|--------|---------|
| **Explicit** | `ttg.local_alloc` | User-requested SMEM allocation |
| **Scratch** | Ops that need temp space | `ttg.convert_layout`, `tt.reduce`, `tt.scan`, `tt.atomic_rmw`, `ttng.tensormap_create`, `ttg.warp_specialize` (for captures) |
| **Virtual** | `triton.call` | Cross-function scratch forwarded to callees |

Buffer sizes are computed in `getExplicitValueSize` (for Explicit) and
`getScratchValueSize` (for Scratch/Virtual). Backends can provide a custom
`AllocationAnalysisScratchSizeFn` to override scratch sizes for
target-specific ops.

## Liveness Analysis

### Operation IDs

Every operation under the root is assigned a numeric ID via a **post-order
walk**. Post-order ensures that a parent operation's ID is greater than all its
children's IDs. This is critical for values defined in a parent region but used
inside a child region (e.g., a value defined before an `scf.for` but used inside
the loop body) — the parent's higher ID extends the value's liveness range to
cover the child.

### SSA Liveness

For **Explicit** buffers (from `ttg.local_alloc`), liveness is computed using
MLIR's built-in `Liveness` analysis (`liveness.resolveLiveness(value)`), which
returns all operations where the SSA value is live. The liveness interval is
`[min operation ID, max operation ID + 1)`.

For **Scratch** buffers, liveness is the single operation that owns them (a
point interval), except for function-level scratch which spans the entire
function.

For **Alias** buffers (values that alias an explicit buffer through block
arguments or subviews), liveness is the union of the alias's own range and the
underlying buffer's range.

### Liveness Extensions for Async Operations

SSA liveness tracks _when a value is referenced in the IR_, but some operations
launch asynchronous hardware work that continues reading or writing SMEM after
the SSA use completes. Without extensions, the allocator would consider the
buffer dead too early and allow another buffer to alias the same SMEM, causing
data races.

The allocator handles three such cases:

#### 1. Remote SMEM Stores (`RemoteShmemStoreOp`, `AsyncRemoteShmemStoreOp`)

Remote stores write to another CTA's shared memory in a cluster. The receiving
CTA has no SSA dependency on the write, so the buffer must remain live for the
entire function to avoid races with local reuse. Without this, an expensive
cluster barrier would be needed before and after every remote store.

**Extension:** Liveness → entire function (`[0, operationId.size())`).

#### 2. Warp Specialization Barriers (`InitBarrierOp`)

Barriers for warp specialization are allocated once at the start of the function
but may be used across multiple sequential warp-specialized loops. If two
barriers in different loops got the same offset, they would corrupt each other
when both are initialized.

**Extension:** Liveness → entire function (`[0, operationId.size())`).

#### 3. Async TMA Store Buffers (`AsyncTMACopyLocalToGlobalOp`)

Early TMA store lowering creates this pattern:

```
%buf = local_alloc %tensor        // write tensor data into SMEM
%tok = async_tma_copy_local_to_global %buf  // TMA starts async read from SMEM
tma_store_token_wait %tok         // wait for TMA to finish reading
```

SSA liveness ends the buffer at `async_tma_copy_local_to_global` (the last
direct use of `%buf`). But the TMA hardware continues reading from SMEM
asynchronously until the token wait completes. If another buffer is allocated at
the same SMEM offset and written between the copy and the wait, the TMA reads
corrupted data.

This is a real bug that manifests with data partitioning (DP=2): two epilogue
accumulators each get their own `local_alloc → tma_copy → token_wait` sequence.
`TritonGPUReorderInstructions` can move the second `local_alloc` before the
first `token_wait` (since there's no SSA dependency), and if both buffers share
offset 0, the second write corrupts the first TMA read.

**Extension:** Liveness is extended to cover the `TMAStoreTokenWaitOp` that
consumes the token. The forward SSA slice from the `local_alloc`'s defining op
is walked to find the token wait, and `maxId` is set to that op's ID + 1. This
is more precise than extending to the full function — it only extends as far as
the async operation actually needs.

### How Extensions Are Implemented

All extensions use `hasOpOfAnyTypeInForwardSlice<OpType>(defOp)`, which walks the
transitive SSA forward slice of the buffer's defining operation and checks for
specific op types. When a match is found, the buffer's liveness interval is
widened accordingly.

The general pattern for adding a new extension:

```cpp
// In getValueLivenessRange lambda, after computing base [minId, maxId]:
if (hasOpOfAnyTypeInForwardSlice<SomeAsyncOp>(defOp)) {
  // Option A: extend to full function
  minId = 0;
  maxId = operationId.size();

  // Option B: extend to a specific downstream op
  llvm::SetVector<Operation *> forwardSlice;
  getForwardSlice(defOp, &forwardSlice);
  for (Operation *op : forwardSlice) {
    if (isa<SomeWaitOp>(op)) {
      maxId = std::max(maxId, operationId[op] + 1);
    }
  }
}
```

## Offset Assignment

### Initial Placement (Triple Algorithm)

The `calculateStarts` method assigns initial SMEM offsets using the triple-based
algorithm from the paper. It maintains a set of _(offset, available range)_
triples representing free SMEM slots. Buffers are processed in descending size
order to reduce fragmentation — large buffers are placed first.

For each buffer, the algorithm finds a triple whose available time range
intersects the buffer's liveness range, places the buffer at that offset, and
splits the triple into up to three new triples representing the remaining free
space.

### Interference Graph

After initial placement, `buildInterferenceGraph` identifies buffer pairs that
**both** overlap in SMEM offset space **and** are live at the same time. Two
buffers interfere if:

- Their `[offset, offset + size)` intervals intersect **and** their liveness
  intervals intersect, **or**
- They are in different regions of the same `AsyncRegions` parent (e.g.,
  different partitions of a `warp_specialize` op) and their offset intervals
  intersect — regardless of liveness, since async regions execute concurrently.

### Graph Coloring

The `allocate` method resolves interferences using first-fit graph coloring.
Each buffer gets a color; buffers with the same color don't interfere. Buffers
with non-zero colors are bumped to offsets past the highest-offset interfering
neighbor.

Since bumping can create new interferences, the interference graph is rebuilt
and coloring re-run in a loop until no interferences remain (fixed point).

### Total SMEM Size

The final `sharedMemorySize` is the maximum `offset + size` across all buffers.

## Module-Level Allocation

`ModuleAllocation` extends the analysis to an entire module by walking the call
graph in post-order. Each function is analyzed independently, and `triton.call`
ops are treated as Virtual scratch buffers sized to the callee's total SMEM
usage. The module's total SMEM size is the maximum across all root functions.

## Debugging

Enable debug output with:

```bash
LLVM_DEBUG_TYPE=allocation-shared-memory
```

This prints buffer ranges, interference graphs, and final allocation sizes.
The `dumpBuffers`, `dumpInterferenceGraph`, and `dumpAllocationSize` methods
provide structured output for each phase.
