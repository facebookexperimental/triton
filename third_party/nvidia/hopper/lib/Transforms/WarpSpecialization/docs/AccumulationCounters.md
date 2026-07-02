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

The per-iteration increment (stride) is **per channel**, not per loop — it is the
number of buffer slots that channel consumes in one iteration:

- `generateYieldCntsForForOp`: a channel that lives directly in the loop body
  (TMEM accumulator, A/B loads) consumes one slot per iteration, so its
  `accumCnt` increments by **1**.
- For reuse groups, the counter is shared — each channel in the group offsets
  its buffer index by its position within the group. A **subtiled** reuse group
  shares one multibuffer across `numTiles` tiles consumed per iteration, so its
  counter advances by **numTiles** (`getReuseGroupStride` →
  `getAccumForReuseGroup`); the subtile index math is then just
  `accumCnt + tileIdx` (see [Subtile Operator](SubtileOperator.md)).
  `getReuseGroupStride` derives the per-tile factor by counting staging-slot
  *lifecycles* in the tile body (writes, falling back to reads), **not** raw
  buffer-touching ops: a same-task interleaved tile body
  (`separate_epilogue_store=False`) holds both a `local_store` and an
  `async_tma_copy` of the **same** slot, so counting both would wrongly double the
  stride.

> A loop-scoped increment (returning numTiles whenever the loop contains *any*
> `SubtiledRegionOp`) is wrong: it stamps the subtile stride onto co-resident
> non-subtile counters and collapses their slot/phase — the AutoWS subtile GEMM
> deadlock.

## Persistent `scf.while` Loops

A static persistent kernel's outer loop may be an `scf.while` instead of an
`scf.for` (e.g. `while tile_id < num_tiles:` with `tile_id += NUM_SMS`). The
accumulation counters must still be carried across persistent iterations so
buffer indices and mbarrier phases stay in phase, exactly as for the `scf.for`
persistent case. Without this, every persistent iteration would restart the
counter at 0 while the hardware mbarrier phases keep advancing, producing a
runtime deadlock, and the buffer-index lookup would mis-resolve to a non-integer
loop-carried value (e.g. a tensor accumulator), crashing in `cast<IntegerType>`.

The `scf.while` is threaded analogously to `scf.for` (`createNewWhileWrapper` /
`createNewWhileLoop`):

1. Each accumCnt is added as an `i64` to the **init operands**, the **before**
   block args, and the **after** block args.
2. The `scf.condition` op **forwards** the before-region accumCnt args into the
   after region (and the while results).
3. The after-region `scf.yield` yields the next accumCnt value back, so it flows
   to the next iteration's before args.

Two kinds of counters can live on a persistent while:

- **Nested-loop counter** — for the inner warp-specialized `scf.for` (e.g. the
  A/B TMA load channel). Seeded from the while's after-region arg; the inner
  loop yields its final value back.
- **Direct counter** — for a channel whose producer/consumer lives directly in
  the while's after region (e.g. the accumulator/epilogue channel). Like a
  channel directly in an `scf.for`, it advances by one per persistent iteration.

For ops directly in the after region, `getAccumCount` resolves the counter from
the while's after-region arguments (no enclosing `scf.for`). In
`createBufferPost`, operandD/TMEM users (and other channel users) that live
directly in the after region must use `getBufferIdxAndPhase(builder, user, ...)`
so their buffer slot rotates with the while-carried counter — matching the
writer's slot — rather than the truly-outside-loop fallback (which would pin the
slot to 0 and race across persistent iterations).

`scf.while` regions are registered in `regionsWithChannels`
(`collectRegionsWithChannels` / `collectRegionsWithChannelsPost`), counted by
`getAccumCnts` / `getAccumCntsPreOrder`, and indexed by `getAccumArgIdx` the same
way `scf.for`/`scf.if` are.

### Reuse groups directly in a persistent `scf.while`

A subtiled epilogue (`EPILOGUE_SUBTILE > 1`) emits several TMA stores that share
one multi-buffered staging buffer — a **reuse group** whose channels live
directly in the while's after region (a *second* channel directly in the while,
in addition to the operand-D accumulator). Reuse groups get their own appended
accumCnt, after the per-region counters. The while path mirrors the `scf.for`
reuse handling end-to-end:

- `createNewWhileWrapper` appends a reuse-group counter to `initialAccums`
  (seeded via `getAccumForReuseGroup`, which is a constant 0 for the outermost
  while) and wires its yield with the staggered next value.
- `getReuseChannels`, `needAccumCntForReuse`, and `getAccumForReuseGroup` accept
  `scf::WhileOp` (iterating / indexing the after region).
- `getStaggeredAccumCnt` finds the enclosing loop as `scf::ForOp` **or**
  `scf::WhileOp` — using `getParentOfType<scf::ForOp>()` alone is null for an op
  in the while's after region and would crash `getReuseChannels`.

Missing any of these makes the per-channel accumCnt index run past the counters
actually threaded into the while (an out-of-bounds after-region argument /
non-integer accumCnt), the same failure class the `getAccumCount` `isIntOrIndex`
assertion guards against.

### Redundant accumulator zero-store removal

`removeRedundantTmemZeroStores` (`WSCodePartition.cpp`) drops the explicit
`tmem_store 0` that initializes the accumulator when the MMA already zeroes it
via `useAccumulator=false` on the first iteration. Without this, the zero-store
becomes a *cross-partition* channel (epilogue zeroes, gemm reads) that, in a
persistent loop, races/hangs across tiles. It must recognize the persistent
outer loop as an `scf.while` (not only `scf.for`) — the zero-store lives directly
in the while's after region, so its enclosing loop is found via
`getParentOfType<scf::WhileOp>()`. The erase step forwards the store's input dep
token (`getDep()`) to its result token (`getToken()`) so the accumulator
dependency chain skips the removed store; forwarding the wrong operand leaves the
store un-erased and reintroduces the cross-tile race.

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
| `createNewWhileLoop` / `createNewWhileWrapper` | Threads accumCnt through a persistent `scf::WhileOp` (before/after regions, condition, yield) |
| `rewriteIfOp` | Extends `scf::IfOp` results with accumCnt outputs |
| `updateAccumLoopCount` | Recursively threads counters through nested control flow |
| `generateYieldCntsForForOp` | Generates loop yield values for counters |
| `generateYieldCntsForIfOp` | Generates if-op yield values for counters |
| `getAccumCount` | Retrieves the accumCnt value for an op from its enclosing loop |
| `getAccumCnts` | Returns the number of accumCnt arguments for a control flow op |
| `getAccumArgIdx` | Returns the starting index of accumCnt arguments in a block argument list |
