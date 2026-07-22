# Reuse Groups

Reuse groups are the autoWS memory planner's mechanism for letting multiple
channels with non-overlapping lifetimes share a single physical buffer
allocation. When two channels never hold live data at the same time, the planner
assigns them the same `buffer.id` so that downstream code partitioning replaces
all but one allocation with views into a single representative buffer. This
reduces SMEM and TMEM pressure without changing program semantics.

## Reuse Categories (A1–A6)

Code partitioning classifies each reuse group (channels sharing a `buffer.id`)
by how it must be synchronized. These labels are used throughout this doc and in
the `WSCodePartition.cpp` / `CodePartitionUtility.cpp` comments:

| | Category | Mem | Shape | Predicate / handler |
|---|---|---|---|---|
| **A1** | SMEM circular reuse | SMEM | multi-buffered (`numCopies > 1`), all producers/consumers in one block; cycles channels through *time slots* of one circular buffer via `accumCnt` staggering | `verifyReuseGroup1` |
| **A2** | 2-buffer real reuse | TMEM/SMEM | exactly 2 single-copy channels with **overlapping** reuse AND a consumer→producer dependency chain — TMEM (overlapping columns **and** chain) or SMEM (chain) | `verifyReuseGroup2` |
| **A3** | N-buffer SMEM epilogue | SMEM | ≥2 single-copy channels sharing one circular buffer, stored/loaded sequentially, all producers in one block (epilogue subtiling) | `verifyReuseGroupN` + inline chain |
| **A4** | TMEM column packing | TMEM | same `buffer.id`, **disjoint** columns (`buffer.offset`) — spatial packing, no cross-channel sync; materialized by `replaceBufferReuse`'s column slice | none (`tmemReuseGroupOverlaps` false) |
| **A5** | Cross-partition reuse | TMEM | single-copy ≥3-channel group whose producers span >1 partition AND admit a unique **dependency-chain** order; synced like A2 on the chain **endpoints** | `verifyReuseGroupCrossPartition` |
| **A6** | Whole-allocation-overwrite hub | TMEM | a `useC=false` owner overwrites the **whole** allocation, clobbering spatially-packed (distinct-`buffer.offset`) siblings; emits back-edges to those siblings | `isWholeAllocationOverwriteReuseOwner` |

A1 uses temporal slot staggering (multi-buffered). A2/A3/A5/A6 are single-copy
(`buffer.copy = 1`) and get explicit reuse barriers. A4 needs no synchronization
(the columns are disjoint). Each category is detailed in its own section below.

## Requirements for Reuse

Two channels can share a buffer when:

1. They have the **same `buffer.id`** assigned by the memory planner.
2. They reference **different `allocOp`s**. If all channels with the same
   `buffer.id` point to the same `allocOp`, they are lifecycle phases of one
   buffer (e.g., multi-buffered pipeline stages), not reuse candidates.

Beyond these common requirements, SMEM and TMEM have additional constraints:

### SMEM Circular Reuse

Handled in `WSMemoryPlanner.cpp` Phase 4 (`allocateSmemBuffers`). Requires:

- Exactly **2 innermost-loop candidates** in the same priority group
- **Compatible element types** (both allocs must have the same `elemType`)
- Multi-dimensional allocs (`numD >= 2`) whose users live in the innermost loop

When these conditions hold, buffer B is given buffer A's `bufferId` and both
receive the same `numCopies`. The number of copies is then maximized by the
SMEM memory planner's incremental allocation algorithm described in
[SMEM Allocation Design](SmemAllocationDesign.md).

### TMEM Packing

Handled in `WSMemoryPlanner.cpp` (`applyAllocationState`). Requires:

- **Non-overlapping liveness intervals** in the column dimension, checked by
  `hasPotentialReuse` during allocation planning
- A valid column offset found by the backtracking allocator `tryAllocate`

Owner buffers get a fresh `buffer.id`; non-owner (reusing) buffers receive the
same `buffer.id` as their owner plus a `buffer.offset` encoding the column
offset within the owner's TMEM row.

## Data Structures

Defined in `CodePartitionUtility.h`:

```cpp
struct ReuseGroup {
  std::vector<unsigned> channelIDs;
  std::vector<Channel *> channels;
};

struct ReuseConfig {
  std::vector<ReuseGroup> groups;
  unsigned getGroupSize() { return groups.size(); }
  ReuseGroup *getGroup(unsigned idx);
};
```

`ReuseGroup` holds a set of channels that all share the same physical buffer.
The first channel (`channels[0]`) is always the **representative** — the owner
of the physical memory. `ReuseConfig` is the collection of all reuse groups for
a given kernel.

## Formation Algorithm

Reuse groups are formed in `doCodePartition` (`WSCodePartition.cpp`):

1. **Group by `buffer.id`**: Iterate over all ordered channels. For each
   channel, look up the `buffer.id` attribute on its `allocOp` and insert the
   channel into a `bufferIdToChannels` map.

2. **Filter same-allocOp sets**: For each `buffer.id` with more than one
   channel, check whether all channels reference the same `allocOp`. If so,
   they are lifecycle phases of one buffer — skip them.

3. **Order channels**: Stable-partition the channels so that the one
   **without** a `buffer.offset` attribute comes first. This channel becomes
   the representative (`channels[0]`), the owner of the physical allocation.

4. **Create `ReuseGroup`**: Push the ordered channel list into a new
   `ReuseGroup` and append it to `config.groups`.

### Degenerate size-1 subtiled reuse group

Normally a reuse group needs ≥ 2 channels (two different `allocOp`s that share a
`buffer.id`). The **both-endpoints-subtiled** epilogue channel is the exception:
`collectAllocChannels` collapses the `numTiles` per-tile staging allocs of one
(producer `ttng.subtiled_region`, consumer `ttng.subtiled_region`) pair into a
**single** `AllocChannel` (the per-tile buffers become in-body instances indexed
by the builtin `tileIdx`). That lone channel still needs the reuse-group
machinery — the in-body per-tile slot rotation (`getOrComputeSubtiledSlot` fires
only for `reuseGrp >= 0`) and the `numTiles` loop-counter stride
(`getReuseGroupStride`). So `doCodePartition` forms a **size-1** group for it
whenever `channelIsCollapsedBothSubtiled(ch)` — **independent of `buffer.copy`**.
The `numBuffers > 1` / `getNumBuffers() <= 1` guards that normally gate the reuse
machinery are relaxed for these channels at **six** sites that must agree:
`needAccumCntForReuse`, `getReuseChannels`, `channelInReuseGroup`
(`CodePartitionUtility.cpp`), `createNewLoopWrapper` (`WSBuffer.cpp`),
`getOrComputeSubtiledSlot` and the `size1Subtiled` group formation
(`WSCodePartition.cpp`). The accumCnt-counting subset (`needAccumCntForReuse` →
`getAccumCnts`, `getReuseChannels`, `createNewLoopWrapper`) must stay in lockstep
or the loop-arg count and accumCnt indices disagree → out-of-bounds
`getArgument` in `getAccumCount`.

The discriminator is the **narrow** `channelIsCollapsedBothSubtiled`
(`AllocChannel::isCollapsedBothSubtiled`, set in `collectAllocChannels` only when
producer AND consumer are in different-task subtiled regions), **not** the broad
`channelIsSubtiled`. The broad form is also true for a *consumer-only-subtiled*
channel — e.g. an epilogue **bias** load whose `local_store` sits outside the
region but whose `local_load` is inside the producer subtiled region. Such a
channel has no per-tile staging buffer to rotate; routing it through the in-body
path (at `buffer.copy == 1`) would push its region into `subtiledRegionsTouched`
with nothing to rewire → empty `deadPositions` assert in `insertAsyncComm`. The
pre-existing `channels.size() <= 1` subtiled exceptions still use the broad
`channelIsSubtiled` (a size-1 group is only ever the collapsed channel, so the
two agree there).

At `buffer.copy == 1` (the DP=1 both-subtiled epilogue) the in-body math reduces
to `bufferIdx = (accumCnt + tileIdx) % 1 == 0` with `phase = (accumCnt + tileIdx) & 1`:
all `numTiles` subtiles share **one** physical staging slot and serialize via the
alternating barrier phase (later tiles *wait* for the earlier tile's slot
release — a serialization, not a race). Collapsing to one physical slot is what
avoids the SMEM `OutOfResources`. The skipped sibling per-tile allocs are
recorded on the representative `AllocChannel` (`collapsedSiblingAllocs`) in
`collectAllocChannels` and **erased** in `insertAsyncComm` once the in-body view
rewire makes them `use_empty` (asserting if a recorded sibling still has a use —
a missed collapse). Without that erase the dead sibling alloc (mutable SMEM →
carries an `Allocate` effect) is double-counted at allocation.

The collapse only fires for a **cross-task** pair (producer-region task ≠
consumer-region task); when `separate_epilogue_store=False` puts both regions in
the epilogue task there is no cross-task channel, so the per-tile allocs are
handled by the ordinary same-task reuse machinery and must NOT be collapsed
(collapsing drops a sibling alloc that is never folded → SMEM OOM). See
[Subtile Operator](SubtileOperator.md).

### Cross-partition staging split (memory planner)

For a data-partitioned epilogue (`tt.data_partition_factor > 1`) with
`early_tma_store_lowering`, every partition's TMA-store staging buffer targets the
**same** descriptor. `WSMemoryPlanner.cpp` `fuseEpilogueWSBuffers` therefore keys
the TMA-staging fusion on `(descriptor, originalLoad)` — `originalLoad` traces the
`local_store` source back to the originating `ttng.tmem_load` / accumulator — so
the two partitions get **distinct** `buffer.id`s instead of sharing one physical
buffer + barrier (which aliases concurrent partitions → corruption + deadlock).
This mirrors the non-staging `loadGroups` discriminator.

## What Reuse Groups Affect

### 1. Accumulation Counters

> **SMEM-only.** Cross-group count accumulation applies **only to
> multi-buffered SMEM circular reuse**. TMEM reuse groups do **not** accumulate
> across the group — see [SMEM vs TMEM](#smem-vs-tmem-accumulation) below.

When channels in a reuse group share a multi-buffered circular buffer, a shared
**accumulation counter** (`accumCnt`) tracks which buffer slot to use. The
counter is carried as a loop argument and incremented as channels are consumed.

Key functions:
- `needAccumCntForReuse` — returns true when **the group is multi-buffered**
  (`channels[0]->getNumBuffers() > 1`) **and** a loop/if region contains at
  least one src or dst op of the reuse group. It short-circuits to `false` for
  single-buffered groups (`CodePartitionUtility.cpp:346-348`), so no shared
  `accumCnt` loop argument is created for them.
- `getAccumForReuseGroup` — computes the `accumCnt` SSA value at a given
  operation by walking back through the channel list to find the nearest
  preceding region op, then arithmetically adding the remaining offset
- `getBufferIdxAndPhase` / `getStaggeredAccumCnt` — for the first channel in the
  ordered list, uses `accumCnt` directly; each subsequent channel at position N
  adds N to stagger its slot within the shared circular buffer. **Subtiled-region
  reuse groups do NOT flow through here for their staging slot:** the N subtiles
  share one barrier pair and one physical alloc, so both the data slot and the
  barrier `bufferIdx`/`phase` are computed *inside the tile body* from the builtin
  `tileIdx` (`flattened = accumCnt + tileIdx`, `% numBuffers`; `accumCnt`
  advances by `numTiles` per iteration), keeping data slot == barrier generation.
  The generic `+ position` stagger is wrong for subtiles (it aliases distinct
  subtiles within a `numBuffers` window and races — the EPILOGUE_SUBTILE>2
  staging-buffer bug). See
  [SubtileOperator](SubtileOperator.md).
- `getReuseAccumArgIdx` — returns the position of a group's `accumCnt`
  argument within the region's full argument list

#### Condition check (A1)

Because the memory planner has already committed to aliasing a multi-buffered
SMEM circular reuse group, code partitioning **validates** the A1 preconditions
before emitting any `accumCnt` for it. After reuse-group formation (and before
`appendAccumCntsForOps`), `doCodePartition` calls `verifyReuseGroup1` on
every multi-buffered group and `report_fatal_error`s if it fails:

- **Multi-buffered** — `channels[0]->getNumBuffers() > 1` (this is also what
  classifies the group as A1 rather than A2/A3).
- **Common basic block** — every channel's producer (`getSrcOp()`) and consumer
  (`getDstOp()`) live in one common block, so the shared `accumCnt` staggering
  is well-defined.

A violation is a hard error (not a silent fallback) because the buffers are
already physically aliased; proceeding would emit incorrect synchronization.

#### SMEM vs TMEM accumulation

The `accumCnt` stagger is a **temporal** mechanism: it cycles the channels of a
group through the *time slots* of one shared multi-buffered circular buffer.
This is exactly how **SMEM circular reuse** works (the group is multi-buffered).

**TMEM reuse does not use it.** TMEM column-packed reuse groups are
**single-buffered** (`numBuffers == 1`); each accumulator is placed at its own
**spatial column** via `buffer.offset` in `replaceBufferReuse`, not in a shared
time slot. Consequently:

1. `needAccumCntForReuse` returns `false` (single-buffered) → no shared
   `accumCnt` loop argument is added for the group.
2. The accumCnt/buffer-index path calls `channelInReuseGroup(channel, config)`
   with the default `reuseBarrier=true`, which skips groups whose representative
   has `numBuffers <= 1` → returns `-1`.
3. With `reuseGroupIdx < 0`, `getBufferIdxAndPhase` takes the plain per-channel
   branch — **no `+N` stagger across the group.**

So `accumCnt` accumulation is exclusive to the SMEM circular-reuse path. TMEM
reuse participates in reuse groups only for **spatial packing** (`buffer.offset`)
and **barrier sharing** (looked up with `reuseBarrier=false`), never for
cross-group count accumulation.

### 2. Token/Barrier Sharing

In `createToken`, the representative channel (first in the group) creates
barriers; non-representative channels reuse them. `channelInReuseGroup` looks
up which group a channel belongs to (returning -1 if none). The `reuseBarrier`
flag skips groups whose representative has `numBuffers <= 1` (single-buffered
channels share no circular barrier).

#### Dedicated reuse-WAR tokens (synthetic)

Separately from the shared per-channel tokens above, two reuse hazards are
guarded by a **freshly minted** token (`ttnvws::CreateTokenOp` at function
entry) plus a `producer_acquire`/`consumer_release` pair, each tagged
`WSBarrierAttr::forDstTask(...)` at the *other* task. These are **not** A1–A6
categories — they are cross-iteration write-after-read edges layered on top of
buffer reuse:

| Site (`WSCodePartition.cpp`) | Mem | Reuse kind | Hazard guarded |
|---|---|---|---|
| operand-D **same-iteration guard** (`guardToken`, ~L3760) | TMEM | same `buffer.id`, same partition, `isSameIterGuard` | the guard channel's `tmem_load` must read before the next iteration's `tmem_store` (MMA operand D) overwrites the accumulator |
| **staging↔operand** reuse (`reuseToken`, Step 7.5, ~L5530) | SMEM | cross-`buffer.id` via `allocation.reuseTarget`, cross partition | the next persistent tile's operand load must not overwrite the SMEM until the previous tile's staging TMA store has drained (bug #9 / D109859261) |

Both mirror the manual TLX empties-token idiom. They differ only in the
`bufferIdx`/`phase` source: the guard token uses
`getBufferIdxAndPhase(numBuffers)`; the Step 7.5 token uses `numBuffers = 1` with
a loop-carried phase derived from the outer induction variable.

**Redundancy (candidate for a helper).** The two sites currently duplicate (a)
the func-entry token-mint idiom (`OpBuilder(funcOp)` →
`setInsertionPointToStart(&funcOp.getBody().front())` → `CreateTokenOp::create`)
and (b) the acquire-then-release `forDstTask` WAR-pair emission. These could be
factored into shared helpers (e.g. `createReuseWarToken` +
`emitTokenWarPair`); not yet done.

### 3. Buffer Replacement

`replaceBufferReuse` rewrites all IR uses of non-representative alloc ops to
point at the representative's alloc. It iterates `config->groups` **directly**
(the source of truth for what must be collapsed) and folds every
non-representative channel of each group into `channels[0]`. It deliberately
does **not** iterate the post-merge channel lists: the reuse-group
consumer-merge in `doCodePartition` removes a non-representative channel
from `orderedChannels` whenever it shares a consumer op with the representative
(e.g. epilogue subtiles that all feed one `ttng.subtiled_region`). Iterating
`orderedChannels` would therefore skip those merged-out channels and leave their
duplicate physical buffers alive — doubling SMEM and causing `OutOfResources`.
Iterating groups visits every non-representative channel regardless of merge
state.

- **SMEM channels (same buffer.id, same type)**: When the alloc types match,
  uses direct `replaceUsesOfWith` to swap the alloc result, then erases the
  old alloc. This generic rewrite also updates `ttng.subtiled_region` operands,
  which are ordinary users of the alloc result.

- **SMEM channels (same buffer.id, different type)**: Type mismatch within
  the *same* `bufferId` group is skipped here — `AllocateSharedMemoryNv`
  resolves these via liveness-based overlap when the two channels are
  not co-live.

- **SMEM channels with `allocation.reuseTarget = N` (cross-buffer reuse)**:
  This is a **distinct mechanism from the A1–A6 categories** above — those group
  channels that share **one `buffer.id`**, whereas here the staging and host keep
  **different `buffer.id`s** linked only by `reuseTarget`, so they never form a
  `ReuseGroup`. The distinct `buffer.id`s are **required, not stylistic**: Step
  7.5 identifies the host operand by resolving `bufferIdToChannel[reuseTarget]`.
  If the staging shared the host's `buffer.id`, that map (last-wins per id)
  collides — the target resolves to the staging itself, so `loadTask ==
  stagingTask` and the cross-tile WAR barrier is **silently dropped** (empirically:
  the Step 7.5 token + acquire/release vanish, no compensating barrier, and the
  bug-#9 race returns with no error).
  Driven by the planner's Phase 3.6 hint (see
  [TMAStoreWaitPipeline.md §Phase 3.6](TMAStoreWaitPipeline.md#phase-36-inter-buffer-smem-reuse-via-allocationreuetarget)).
  The staging alloc carries `allocation.reuseTarget = <host bufferId>`;
  the follow-up staging-reuse merge rewrites the staging alloc into a
  `memdesc_reinterpret` view of the host alloc (the one with
  `buffer.id = N`) and erases the staging alloc. The reinterpret is sound
  because the host alloc's storage covers ≥ `staging.size × staging.numCopies`
  bytes (guaranteed by `findReuseCandidate`'s size check) **and** the two
  encodings match (`areReuseEncodingsCompatible`; an encoding-incompatible reuse
  is never marked, or realization would silently drop it and under-count SMEM —
  bug #10).

  Because the staging *aliases* the host operand SMEM, the two must never be
  live at once. On the persistent (outer-tile) path this is a **cross-tile**
  write-after-read: the next tile's operand load (load task) must not overwrite
  that SMEM until the previous tile's staging TMA store (staging task) has
  drained. That edge is enforced by the Step 7.5 barrier in `doCodePartition` —
  a dedicated single-buffered **cross-partition** reuse token (not the host
  operand's own barrier, whose consumer warp count differs and would trip
  `WSLowerToken`'s `consumerWarps == nWarps`): the load task `producer_acquire`s
  it at the top of the outer loop with a loop-carried phase (from the induction
  variable), and the staging task `consumer_release`s it at the bottom after the
  staging stores drain. (An earlier degenerate version emitted a constant
  `bufferIdx=0`/`phase=0` acquire on the host token that `WSLowerToken` elided to
  a no-op — harmless on the single-tile path, a cross-tile SMEM race on the
  persistent path. Bug #9 / D109859261; see
  [TMAStoreWaitPipeline.md](TMAStoreWaitPipeline.md) and
  `.llms/rules/partition-scheduler-bugs.md` #9, lit test
  `ws_code_partition_bwd_persist_staging_war.mlir`.)

- **TMEM channels**: Inserts a `sliceAndReinterpretMDTMEM` op at the
  `buffer.offset` column within the representative's TMEM allocation. If the
  primary representative's type cannot accommodate the slice, other group
  representatives are tried before emitting an error.

Without this reinterpret step for SMEM cross-buffer reuse, `AllocateSharedMemoryNv`
would place the staging alloc at a fresh offset (it has no awareness of
`allocation.reuseTarget`), and the planner's Phase 3.6 "free" accounting would
silently overrun the SMEM budget at codegen time.

### 4. `allocation.shareGroup` Attribute

Buffers in a reuse group are tagged with an `allocation.shareGroup` attribute
for consumption by downstream passes.

## 2-Buffer Reuse Group Synchronization

When two channels share the same physical buffer (a **reuse group** with
2 buffers and `buffer.copy=1`), we must ensure that one channel's consumer
has fully released the buffer before the other channel's producer acquires it.
The code shares tokens between reuse group channels but must also reason
about the ordering of `producer_acquire` across the two channels.

### Background: Current `producer_acquire` Insertion

`producer_acquire` is inserted at one of these points in `insertAsyncComm`:

| Mechanism | Condition | Insertion Point |
|-----------|-----------|-----------------|
| `ProducerAcquireOp` (token-based) | `consumerBarriers` empty | Before `headProducer` (or `producerAcquireForChannelLoop`) |
| `WaitBarrierOp` (gen5 inline) | `consumerBarriers` populated | Before the producer, via `desyncMMAv5Op(..., asProducerAcquire=true)` |

The variable `producerAcquireForChannelLoop` already handles the case of
**forward/backward channel loops** (same alloc, same block, cycle through
gen5 operand D). The 2-buffer reuse group design extends that concept.

### Requirements

For a reuse group with 2 buffers A and B (`buffer.copy=1`):

1. **Verification**: Each buffer must have exactly one channel, and there must
   be a dependency chain from one buffer's consumer to the other's producer.
2. **Ordering**: Determine which buffer is "early" (A) and which is "late" (B).
   If `A.producer → A.consumer → B.producer`, then A is early.
3. **Case analysis**: Check whether there is an ordering from B's consumer back
   to A's producer:
   - **Implicit ordering** (e.g. `qk/pp`): B's consumer and A's producer are
     both in the same partition (e.g. gemm). The partition-internal ordering
     already guarantees B's consumer_release happens after A's producer_acquire.
     No additional synchronization needed.
   - **Explicit wait needed** (e.g. `dp/dq`): B's consumer and A's producer
     are in different partitions (or same partition but wrong order). We must
     move B's `producer_acquire` to be before A's producer, so A's producer
     waits for B's consumer_release before writing.

### Helper Functions

#### `verifyReuseGroup2`

```cpp
// Verify a 2-buffer reuse group:
// - Exactly 2 channels.
// - Each channel has 1 copy (getNumBuffers() == 1).
// - A dependency chain exists between one channel's consumer and the other's producer.
// Returns true if valid.
bool verifyReuseGroup2(ReuseGroup *group);
```

`verifyReuseGroup2` recognizes only **real** 2-buffer reuse — two channels that
share the same physical space and reuse it across time. The signal differs by
memory kind:

- **TMEM**: real reuse iff the two channels' **column ranges overlap**
  (`[buffer.offset, buffer.offset + numCols)`) **and** there is a
  consumer→producer **dependency chain** in either direction. Disjoint columns
  are *spatial packing* (e.g. two independent accumulators side-by-side),
  materialized by `replaceBufferReuse`'s column slice — they need no
  cross-channel sync and are **not** treated as a reuse group here. Requiring
  the chain in addition to overlap means the two accesses are provably
  temporally ordered (a real reuse, not concurrently-live aliasing) and gives
  `orderReuseGroup2` a reliable early/late ordering — the same standard SMEM
  already meets.
- **SMEM**: real reuse iff there is a consumer→producer **dependency chain** in
  either direction (e.g. `qk/pp`).

The SMEM **epilogue-subtile** case (producers in the same block, no dependency
chain) is deliberately **not** handled here — it is the N-buffer path
(`verifyReuseGroupN`). This split keeps `verifyReuseGroup2` strictly about
overlapping-space reuse.

Implementation:
```
verifyReuseGroup2(group):
  assert group.channels.size() == 2
  A = group.channels[0], B = group.channels[1]

  // Only single-copy buffers are handled.
  if A.getNumBuffers() != 1 || B.getNumBuffers() != 1:
    return false

  // TMEM: real reuse iff the column ranges overlap (same space) AND a
  // consumer->producer dependency chain orders the two (real temporal reuse,
  // not concurrent aliasing).
  if A and B are both TMEM:
    if not tmemReuseGroupOverlaps(group):
      return false                                    // disjoint = spatial packing
    return hasDependencyChain(A, B) || hasDependencyChain(B, A)

  // SMEM: real reuse iff a consumer→producer dependency chain exists.
  return hasDependencyChain(A, B) || hasDependencyChain(B, A)
```

`hasDependencyChain(A, B)` walks the transitive users of `A.dstOp` looking for
`B.srcOp`, and also treats same-block program order (`A.dstOp` before
`B.srcOp`) as an implicit dependency. `tmemReuseGroupOverlaps` computes each
channel's column interval from `getTmemAllocSizes(...).numCols` and
`buffer.offset` and returns true if any two intervals intersect.

#### `orderReuseGroup2`

```cpp
// For a verified 2-buffer reuse group, determine which channel is early (A)
// and which is late (B).
// Returns {earlyChannel, lateChannel}.
std::pair<Channel *, Channel *> orderReuseGroup2(ReuseGroup *group);
```

Implementation:
```
orderReuseGroup2(group):
  A = group.channels[0], B = group.channels[1]
  if hasDependencyChain(A, B):    // A.consumer -> B.producer
    return {A, B}
  if hasDependencyChain(B, A):
    return {B, A}
  // Unreachable for a verified group: verifyReuseGroup2 now requires a chain in
  // one direction (both SMEM and overlapping TMEM). A group with no chain is an
  // overlapping-but-unordered (concurrently aliased) pair — a memory-planner
  // contract violation — so fail loudly instead of guessing a barrier direction
  // from program order.
  report_fatal_error("orderReuseGroup2: reuse group has no dependency chain")
```

#### `needExplicitReuseWait`

```cpp
// Given ordered channels {A (early), B (late)}, determine whether we need to
// explicitly wait for B's consumer_release before A's producer_acquire.
// Returns false when B's consumer and A's producer are in the same partition
// and program order guarantees correctness.
bool needExplicitReuseWait(Channel *earlyChannel, Channel *lateChannel);
```

Implementation:
```
needExplicitReuseWait(earlyChannel, lateChannel):
  aProducerOp = earlyChannel.srcOp
  // Resolve through memdesc_trans etc. — there may be more than one.
  for bConsumerOp in getActualConsumers(lateChannel.dstOp):
    if getWSPartitionIds(bConsumerOp) shares a taskId with getWSPartitionIds(aProducerOp):
      // Same partition: program order already serializes the release before
      // the next acquire.
      if aProducerOp, bConsumerOp in same block and appearsBefore(aProducerOp, bConsumerOp):
        return false  // No explicit wait needed (qk/pp case)

  return true  // Need explicit wait (dp/dq case)
```

### Integration into `insertAsyncComm`

In the main channel processing loop, after computing
`producerAcquireForChannelLoop`, the reuse group logic is added:

```cpp
Operation *producerAcquireForChannelLoop = nullptr;
if (headProducer->getBlock() == headConsumer->getBlock()) {
  auto *bwdCh = isForwardOfChannelLoop(masterChannel);
  if (bwdCh)
    producerAcquireForChannelLoop = bwdCh->getDstOp();
}

// --- 2-buffer reuse group handling ---
Operation *producerAcquireForReuse = nullptr;
int reuseGrp = channelInReuseGroup(masterChannel, config);
if (reuseGrp >= 0) {
  auto *group = config->getGroup(reuseGrp);
  if (group->channels.size() == 2) {
    verifyReuseGroup2(group);
    auto [earlyChannel, lateChannel] = orderReuseGroup2(group);

    if (masterChannel == earlyChannel) {
      // Early buffer (A): check if we need explicit wait for late buffer's
      // consumer_release. No change needed here — the key change is for
      // the LATE buffer (below).
      if (needExplicitReuseWait(earlyChannel, lateChannel)) {
        // implicit: early buffer uses default producer_acquire placement
      }
    } else {
      // Late buffer (B): if explicit wait is needed, move this buffer's
      // producer_acquire to before the early buffer's producer.
      assert(masterChannel == lateChannel);
      if (needExplicitReuseWait(earlyChannel, lateChannel)) {
        producerAcquireForReuse = earlyChannel->getSrcOp();
      }
    }
  }
}

// Combine with existing producerAcquireForChannelLoop
if (producerAcquireForReuse && !producerAcquireForChannelLoop) {
  producerAcquireForChannelLoop = producerAcquireForReuse;
}
```

This reuses the existing `producerAcquireForChannelLoop` mechanism which
flows through to both `ProducerAcquireOp` insertion and gen5 inline barrier
`desyncMMAv5Op` insertion.

### Processing Order

The early channel should be processed before the late channel so that when
the late channel is processed, it can reference the early channel's producer
as an insertion point. In `orderedChannelsGroupedByConsumers` construction,
ensure that within a reuse group, the early channel appears first:

```cpp
for (unsigned idx = 0; idx < config.getGroupSize(); idx++) {
  auto *group = config.getGroup(idx);
  if (group->channels.size() == 2) {
    auto [early, late] = orderReuseGroup2(group);
    // Ensure early appears before late in orderedChannelsGroupedByConsumers
  }
}
```

### Examples

#### `dp/dq` (explicit wait needed)

```
dp: producer = tc_gen5_mma (task 1, gemm)    → consumer = tmem_load (task 3, computation)
dq: producer = tc_gen5_mma (task 1, gemm)    → consumer = tmem_load (task 0, computation)
```

- Ordering: `dp` is early (dp.producer → dp.consumer → dq.producer).
- `dq.consumer` (task 0) and `dp.producer` (task 1) are in **different
  partitions** → `needExplicitReuseWait` returns `true`.
- Action: Move `dq`'s `producer_acquire` to before `dp`'s producer. This
  ensures `dp`'s producer waits (via the shared token) until `dq`'s consumer
  releases the buffer.

#### `qk/pp` (implicit ordering)

```
qk: producer = TMA load (task 2, load)       → consumer = tc_gen5_mma (task 1, gemm)
pp: producer = local_store (task 3, comp)     → consumer = tc_gen5_mma (task 1, gemm)
```

- Ordering: `pp` is early (pp.producer → pp.consumer → qk.producer).
- `pp.consumer` (task 1, gemm) and `qk.producer` (task 1, gemm) are in the
  **same partition** and `qk.producer` appears before `pp.consumer` →
  `needExplicitReuseWait` returns `false`.
- Action: No change. Partition-internal ordering guarantees correctness.

## N-Buffer Reuse Group Synchronization

For reuse groups with more than two channels (`group->channels.size() > 2`),
`insertAsyncComm` handles two distinct shapes. Both run only for single-copy
groups (`getNumBuffers() == 1`, always true for TMEM).

### Same-block linear chain (SMEM epilogue subtiling)

When every channel's producer is in the **same block**, the channels are ordered
by producer program order and a dependency chain is built: channel `i`'s producer
back-waits on channel `i-1`'s consumer, and the first channel wraps around to wait
on the last channel's consumer from the previous iteration. This is the original
N>2 path and covers epilogue subtiling, where N subtiles share one SMEM buffer and
are stored/loaded sequentially.

### Whole-allocation overwrite owner ("hub" case)

Column-packed **TMEM** reuse groups (TMEM Packing, above) break the same-block
assumption: the representative (the QK accumulator) is produced by a
`tc_gen5_mma` in the **inner** loop, while the packed scalar siblings
(`alpha`/`m_ij`/`l_i0` at `buffer.offset` 64-66) are produced by `tmem_store` ops
in the **outer** loop body. The producers are in different blocks, so the linear
chain does not apply.

The hazard: a `tc_gen5_mma` with `useC=false` **zeros the entire physical
allocation** before writing — clobbering every packed sibling's columns, not just
the owner's. The owner channel's barrier only frees the buffer with respect to the
QK *result* consumer (released inside the inner loop); nothing makes the
next-iteration MMA wait for the default partition to finish reading the packed
scalars (read after the inner loop). This is the FA-fwd-persistent TMEM aliasing
race — latent on the non-early path, exposed by `early_tma_store`.

The fix: when the representative satisfies
`isWholeAllocationOverwriteReuseOwner`, the owner back-waits on the packed
siblings - but **cadence matters**, and getting it wrong
deadlocks:

- **Inner-cadence siblings** (produced *inside* the owner's inner loop, e.g.
  `alpha`): produced and consumed within the same inner iteration, already ordered
  by their own per-iteration channel barriers. **No extra edge is added.**
- **Outer-cadence siblings** (produced *outside* the inner loop, e.g.
  `m_ij`/`l_i0`, read once per tile in the epilogue): these are the ones the next
  tile's first inner MMA clobbers. For each such sibling whose consumer is in a
  **different partition** than the owner producer, emit a `producer_acquire` on the
  sibling's token **before the owner's inner loop** (once per outer tile), using
  the **sibling's own outer-loop phase** (`getBufferIdxAndPhase(..., op = the inner
  `scf.for`, reuseGroupIdx = -1, ch = sibling)`), so the next tile's first MMA
  waits for the previous tile's epilogue read.

Two details are essential and were the source of an initial deadlock:

1. **Placement**: the wait goes *before the inner loop*, not at the MMA. The MMA
   runs at inner cadence; an outer-cadence barrier waited on every inner iteration
   never matches its phase and hangs.
2. **Phase basis**: use the *sibling's* phase, not the owner's. The siblings are
   single-buffered, so they sync via the per-channel path (`reuseGroupIdx = -1`);
   `getReuseChannels` skips the multi-buffer reuse-group accumCnt path for
   `numBuffers <= 1`. The inner-loop op's parent is the outer loop, so
   `getAccumCount` resolves the same outer-loop accumCnt the sibling's real
   producer_acquire/consumer_release use.

Siblings consumed within the owner producer's own partition (e.g. the `P` matrix at
`buffer.offset = 0`, consumed by the PV MMA in the same gemm partition) are skipped
— program order already orders those.

```
Without the fix (race):              With the fix (per outer tile):
  [outer tile body]                    [outer tile body]
    inner loop {                         wait m_ij backward (task 0, outer phase)
      tc_gen5_mma useC=false <-- 1st     wait l_i0 backward (task 0, outer phase)
      ... }                              inner loop {
    epilogue: task0 reads m_ij/l_i0        tc_gen5_mma useC=false  (now gated)
                                           ... }
                                         epilogue: task0 reads m_ij/l_i0
```

`isWholeAllocationOverwriteReuseOwner(ownerCh)` returns true when `ownerCh` is the
representative (its alloc has no `buffer.offset`) and is a `TmemAllocChannel`
with `isOperandDNoAcc == true` (set in `createAllocChannel` when the producer MMA's
`useAccumulator()` is constant-false). The outer-cadence siblings are collected in
`fullOverwriteOuterSiblings`; a debug-only assert in the emission step guards
against silently dropping a required sibling back-edge (the bug class reappearing).
Regression test:
`test/Hopper/WarpSpecialization/ws_code_partition_tmem_packed_reuse_backward.mlir`;
runtime validation is the FA-fwd-persistent dp kernel determinism check.

### A6 applies only to spatial packing (distinct `buffer.offset`)

The N>2 dispatch checks the A6 hub case **first**, but only when the group is
**spatial packing** — its channels occupy **≥2 distinct `buffer.offset`s** within
the owner's columns (e.g. alpha/m_ij/l_i0 at offsets 64/65/66 inside the QK
accumulator). A group whose channels **all share the same offset** is
**full-overlap temporal reuse**, not packing, and must NOT enter A6 even though
its `useC=false` owner satisfies `isWholeAllocationOverwriteReuseOwner`: the hub
back-edges assume packed siblings live in *different* columns, so applying them to
a full-overlap group emits **incorrect** synchronization.

Realized case: FA-bwd `{dpT, dq, dsT}` (`buffer.id=5`, all at offset 0). Routed
to A6 it mis-synchronizes the `dq` overwrite against `dk`'s read of `dsT` →
wrong `dq` (this is exactly the [BwdTmemReuseSlotHazard](BwdTmemReuseSlotHazard.md)
class, re-exposed when Nick's A6 landed in `origin/main`). Excluded from A6, the
group relies on its standard per-channel barriers plus the kernel's dk-before-dq
ordering, which is correct.

The gate (`WSCodePartition.cpp`, N>2 dispatch): A6 fires only when an
`isWholeAllocationOverwriteReuseOwner` is present **and** the group spans ≥2
distinct `buffer.offset`s (`isSpatialPacking`). Note the discriminators that do
**not** work here: column-range overlap (`tmemReuseGroupOverlaps`) is true for
both shapes (the owner spans the packed siblings' columns either way), and
block-based `verifyReuseGroupCrossPartition` is unusable at this stage because
partitions are still `ttg.partition` tags within a single block.

## A5: Cross-Partition Reuse (N ≥ 3, realized as TMEM)

A fifth shape that does **not** fit A1–A4: a single-copy reuse group of **≥ 3
channels** sharing one **overlapping** buffer, whose producers span **more than
one partition/block**. It is distinct from every other category:

| | A5 group `{dpT, dq, dsT}` |
|---|---|
| A1 SMEM circular | ✗ single-copy, not multi-buffered |
| A2 2-buffer (`verifyReuseGroup2`) | ✗ 3 channels, not 2 |
| A3 SMEM epilogue (`verifyReuseGroupN`) | ✗ TMEM, and producers not all same block |
| A4 TMEM column packing | ✗ columns **overlap** (temporal reuse, not disjoint) |

**Realized case** — FA-bwd config `_BWD_DOT_ATTRS_TMEM`
(`fused_attention_ws_device_tma.py`): three dots share one TMEM `buffer.id`
(`dpT.opndD`, `dq.opndD`, `dk.opndA = dsT`). The distinguishing annotation vs
the default is `dk: "opndA,tmem,1,5"` — `dk` reads `dsT` from the same TMEM
buffer that `dpT`/`dq` write. The group is formed **by annotation**
(`FrozenDotAttrs` channel specs pre-assign the shared `buffer.id`), not by the
overlap/liveness heuristics.

### Predicate

`verifyReuseGroupCrossPartition` (`CodePartitionUtility.cpp`) accepts the group
iff:

1. **≥ 3 channels**, all **single-copy** (`getNumBuffers() == 1`).
2. Producers span **> 1 partition** (`relation.first` — the producer
   `ttg.partition`). This is detected by partition id, **not** block: at
   `doCodePartition` the partitions are still `ttg.partition` tags within one
   block, so a block-based test would always see "one block".
3. The channels admit a **unique total dependency-chain order** —
   `orderReuseGroupChain(group)` returns a non-empty ordering (channel `i`'s
   consumer reaches channel `i+1`'s producer via SSA use-def or same-block
   program order). For `{dpT, dsT, dq}` this is `dpT → dsT → dq`.

If no unique chain order exists the predicate returns false and the group falls
back to the per-channel barriers.

### Synchronization

Handled inline in `insertAsyncComm` (its own `verifyReuseGroupCrossPartition`
branch, **decoupled** from the A3 same-block chain — it pushes nothing onto
`wrapAroundChannelsForReuseSync`). The chain `dpT → dsT → dq` is ordered by
**inherent edges**, so the shared slot is used strictly in order within a tile,
and only the cross-iteration WAR needs an explicit barrier:

- **dpT → dsT** is a data dependency (`dsT` is computed from `dpT` in the
  computation partition), so `dsT`'s write follows `dpT`'s read for free.
- **dsT → dq** is gemm-partition program order within the same SWP stage (the
  `dk` MMA reads `dsT` before the `dq` MMA overwrites the slot; consecutive
  tcgen05 MMAs execute in issue order), so `dq`'s write follows `dsT`'s read for
  free. (This is why `dk` must be emitted before `dq` — see the accuracy fix.)
- **cross-iteration WAR** (the only explicit edge): the next tile's first write
  (`dpT`) must wait for the previous tile's last read (`dq`). Emitted exactly
  like the 2-buffer **A2** case applied to the chain **endpoints** — early =
  first (`dpT`), late = last (`dq`): the late channel's `producer_acquire` is
  relocated ahead of the early channel's producer so the shared slot's empty
  barrier (flipped by `dq`'s consumer release) gates the `dpT` overwrite, plus an
  intra-iteration wait of the late writer on the early reader (subsumed by
  program order; kept for parity with A2).

Because each channel still gets **its own per-channel barrier** and `dsT`'s
consumer (`dk` MMA) lives in a *different task* than the representative's
consumer, `createToken` allocates a **dedicated gen5 consumer barrier** for
that consumer task — otherwise `dsT`'s consumer-release silently drops and `dk`
deadlocks (the `BwdTmemDotAttrsDeadlock` fix).

### Status / caveats

- A5 enters via the `group->channels.size() > 2` clause of the dispatch gate;
  `verifyReuseGroupCrossPartition` validates treatability and is checked **before**
  the A3 `allSameBlock` chain so the cross-partition group does not fall into A3.
- For the **contract** (bail on unhandled reuse), a size-> 2 group is "handled"
  iff `verifyReuseGroupN(group) || verifyReuseGroupCrossPartition(group)`; any
  other size-> 2 group should bail.

## Key Attributes

| Attribute | Description | Set by | Read by |
|-----------|-------------|--------|---------|
| `buffer.id` | Groups channels that share physical memory | `WSMemoryPlanner` (SMEM + TMEM) | `doCodePartition` (group formation) |
| `buffer.copy` | Number of pipeline copies (multi-buffering depth) | `WSMemoryPlanner` | Buffer allocation, `needAccumCntForReuse` |
| `buffer.offset` | Column offset within the owner's TMEM allocation | `WSMemoryPlanner` (`applyAllocationState`) | `replaceBufferReuse` (TMEM slice offset) |
| `allocation.shareGroup` | Tags buffers for downstream passes | `doCodePartition` | Downstream passes |

## Key Functions Reference

| Function | File | Purpose |
|----------|------|---------|
| `ReuseGroup`, `ReuseConfig` | `CodePartitionUtility.h` | Data structures |
| `channelInReuseGroup` | `CodePartitionUtility.cpp` | Look up reuse group index for a channel |
| `needAccumCntForReuse` | `CodePartitionUtility.cpp` | Check if a region needs an `accumCnt` argument |
| `getReuseChannels` | `CodePartitionUtility.cpp` | Build ordered list of dst ops in a region |
| `getReuseAccumArgIdx` | `CodePartitionUtility.cpp` | Position of group's `accumCnt` in argument list |
| `getBufferIdxAndPhase` | `CodePartitionUtility.cpp` | Compute buffer index with per-channel stagger |
| `getAccumForReuseGroup` | `WSBuffer.cpp` | Compute `accumCnt` SSA value at a given op |
| `replaceBufferReuse` | `WSCodePartition.cpp` | Rewrite alloc uses to point at representative |
| Reuse group formation | `WSCodePartition.cpp` (`doCodePartition`) | Group channels by `buffer.id`, form `ReuseConfig` |
| SMEM `buffer.id` assignment | `WSMemoryPlanner.cpp` | Assign `buffer.id` to SMEM allocs |
| SMEM circular reuse (Phase 4) | `WSMemoryPlanner.cpp` | Form SMEM reuse pairs, maximize copies |
| TMEM `applyAllocationState` | `WSMemoryPlanner.cpp` | Assign `buffer.id` + `buffer.offset` to TMEM allocs |
| `verifyReuseGroup1` | `CodePartitionUtility.cpp` | Verify A1 (SMEM circular reuse): multi-buffered + all producers/consumers in one block |
| `verifyReuseGroup2` | `CodePartitionUtility.cpp` | Real 2-buffer reuse: TMEM (column overlap AND dependency chain), or SMEM (dependency chain) |
| `tmemReuseGroupOverlaps` | `CodePartitionUtility.cpp` | True if a TMEM group's channel column ranges overlap (real reuse vs spatial packing) |
| `orderReuseGroup2` | `CodePartitionUtility.cpp` | Determine early/late channel ordering |
| `needExplicitReuseWait` | `CodePartitionUtility.cpp` | Check if explicit cross-channel wait is needed |
| `isWholeAllocationOverwriteReuseOwner` | `CodePartitionUtility.cpp` | Detect a representative whose producer overwrites the whole allocation (needs back-edges to live packed siblings) |
| `hasDependencyChain` | `CodePartitionUtility.cpp` | True if A's consumer reaches B's producer (transitive or program order) |
| `verifyReuseGroupN` | `CodePartitionUtility.cpp` | **Live** gate for the SMEM epilogue path: SMEM, single-copy, producers same block, N ≥ 2 |
| N-buffer sync (epilogue) | `WSCodePartition.cpp` (`insertAsyncComm`) | Inline chain + wrap-around `ProducerAcquireOp` insertion |
| `orderReuseGroupN` | `CodePartitionUtility.cpp` | Producer-order sort helper — **still unused** (inline does its own sort) |
| `verifyReuseGroupCrossPartition` | `CodePartitionUtility.cpp` | A5 predicate: cross-partition N≥3 reuse (producers span >1 task) with a unique dependency-chain order (FA-bwd `{dpT,dsT,dq}`) |
| `orderReuseGroupChain` | `CodePartitionUtility.cpp` | Topologically orders an N-channel single-copy reuse group into one dependency chain (`dpT→dsT→dq`); empty if no unique order |
