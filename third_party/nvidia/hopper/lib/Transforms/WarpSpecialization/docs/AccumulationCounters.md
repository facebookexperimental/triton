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

## TMA Staging Buffers: same-partition vs cross-partition

`getStaggeredAccumCnt` (`CodePartitionUtility.cpp`) has a special path for
early-TMA staging buffers (`buffer.tmaStaging > 0`, e.g. the FA-bwd `dk/dv`
store-staging, the `dq` reduce-staging, and the FA-fwd `desc_o` output
staging). These hold S subtiles (the `EPILOGUE_SUBTILE` / `DQ_SUBTILE` stores,
or the `BLOCK_M`/128 output halves) that rotate through K = `buffer.copy` slots
of one circular buffer. **The slot/phase rule depends on how the buffer is
drained, which is determined by whether the channel's producer and consumer are
in the same warp task:**

| Staging kind | Producer vs consumer task | Drain | Slot/phase from `getStaggeredAccumCnt` |
|---|---|---|---|
| **Same-partition** (bwd `dk`/`dv`/`dq`) | same task (e.g. `local_store` and `async_tma_copy`/`async_tma_reduce` both in the store task) | fixed in-flight-count `cp.async.bulk.wait_group(K-1)`, no mbarrier | bare subtile index `theIdx` → `slot = theIdx % K`, repeating every tile |
| **Cross-partition** (fwd `desc_o`) | different tasks (produced in compute task, TMA-stored in epilogue-store task) | producer/consumer **mbarrier** (full/empty), plus the wait_group for the TMA itself | continuous `accumCnt` (+`theIdx`) → `slot`/`phase` rotate every persistent tile |

Why they differ: the cross-partition mbarrier's empty/full **phase** is derived
from the same returned count and must keep flipping across persistent tiles, so
the count must stay continuous; collapsing it to the per-tile `theIdx` pins the
phase and deadlocks the handshake after the first tile. The same-partition ring
has no such phase (the wait_group alone gates slot reuse) and instead needs the
fixed per-tile `theIdx` so same-slot stores stay exactly K apart.

The discriminator is `getAsyncTaskIds(ch->getSrcOp()) == getAsyncTaskIds(ch->getDstOp())`.
The same-partition `theIdx % K` rotation is only correct when **K | S**; that
invariant is enforced defensively by `increaseFusedEpilogueCopies`
(see [TMA Store Wait Pipeline](TMAStoreWaitPipeline.md)). The cross-partition
mbarrier rotation tolerates any K.

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
