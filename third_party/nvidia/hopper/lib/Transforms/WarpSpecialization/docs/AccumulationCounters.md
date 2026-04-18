# Accumulation Counters

Accumulation counter insertion threads `accumCnt` loop-carried values into
the IR — `i64` values that track which buffer slot to use in multi-buffered
pipelines. This runs as part of code partitioning (`doCodePartition` step 6,
`doCodePartitionPost` step 4), after channels and buffers have been created.

**File**: `WSBuffer.cpp`
**Function**: `appendAccumCntsForOps(taskTopOps, channels, regionsWithChannels, config)`

## Pipeline Context

```
doCodePartition / doCodePartitionPost
  Step 1-3: channel discovery, grouping, buffer creation
  ...
  → appendAccumCntsForOps  ← THIS: inserts accumCnt loop arguments
  ...
  → insertAsyncCopy / insertAsyncComm  ← uses accumCnt to index buffers
```

## What Is an Accumulation Counter?

An **accumulation counter** (`accumCnt`) is an `i64` loop-carried value that
starts at 0 and increments by 1 each time a buffer slot is consumed. It is
used to compute:

```
bufferIdx = accumCnt % numBuffers    // which buffer slot
phase     = (accumCnt / numBuffers) & 1  // mbarrier phase bit
```

Each channel (or reuse group of channels) that is multi-buffered needs its
own `accumCnt` argument threaded through the enclosing control flow.

## Algorithm

### Step 1: Identify Channels Needing AccumCnt

A channel needs an accumulation counter when it has `numBuffers > 1` (is
multi-buffered). Channels in a reuse group share a single `accumCnt`.

### Step 2: Extend Loop Arguments (`createNewLoop`)

For each `scf::ForOp` that contains multi-buffered channels:

1. Create a new loop with additional `i64` block arguments — one per
   accumulation counter.
2. All arguments start at 0 (`arith::ConstantOp(0)`).
3. The original loop body is moved into the new loop.

`createNewLoopWrapper` handles the case where the loop is wrapped in an
outer structure.

### Step 3: Extend If-Op Results (`rewriteIfOp`)

When `scf::IfOp` appears inside a loop with accumulation counters, its
results must be extended to carry the `accumCnt` values through both the
then and else branches:

- `generateYieldCntsForThenBlock`: generates yield values for the then branch
- `generateYieldCntsForIfOp`: generates yield values for both branches

### Step 4: Update Counter Values (`updateAccumLoopCount`)

Recursively processes nested `ForOp`/`IfOp` to thread `accumCnt` values
correctly through all control flow. The counter is incremented at each
point where a buffer slot is consumed (i.e., at the channel's destination
operation).

### Step 5: Generate Yield Values

- `generateYieldCntsForForOp`: at each loop yield, the `accumCnt` is
  incremented by the number of times it was consumed in the loop body.
- For reuse groups, the counter is shared — each channel in the group
  offsets its buffer index by its position within the group.

## Interaction with Reuse Groups

When channels share a reuse group (same `buffer.id`), they share a single
`accumCnt`:

- `getAccumForReuseGroup`: computes the `accumCnt` SSA value at a given
  operation by walking back through the channel list.
- `getBufferIdxAndPhase`: for the first channel in the group, uses
  `accumCnt` directly. Each subsequent channel at position N adds N to
  stagger its slot within the shared circular buffer.

See [Reuse Groups](ReuseGroups.md) for more details.

## Key Functions

| Function | Description |
|----------|-------------|
| `appendAccumCntsForOps` | Entry point: identifies channels needing counters |
| `createNewLoop` / `createNewLoopWrapper` | Extends `scf::ForOp` with extra block arguments |
| `rewriteIfOp` | Extends `scf::IfOp` results with accumCnt outputs |
| `updateAccumLoopCount` | Recursively threads counters through nested control flow |
| `generateYieldCntsForForOp` | Generates loop yield values for counters |
| `generateYieldCntsForIfOp` | Generates if-op yield values for counters |
| `getAccumCount` | Retrieves the accumCnt value for an op from its enclosing loop |
| `getAccumCnts` | Returns the number of accumCnt arguments for a control flow op |
| `getAccumArgIdx` | Returns the starting index of accumCnt arguments in a block argument list |
