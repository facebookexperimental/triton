# Reuse Groups

Reuse groups are the autoWS memory planner's mechanism for letting multiple
channels with non-overlapping lifetimes share a single physical buffer
allocation. When two channels never hold live data at the same time, the planner
assigns them the same `buffer.id` so that downstream code partitioning replaces
all but one allocation with views into a single representative buffer. This
reduces SMEM and TMEM pressure without changing program semantics.

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

Reuse groups are formed in `doCodePartitionPost` (`WSCodePartition.cpp`):

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
`collectPostChannels` collapses the `numTiles` per-tile staging allocs of one
(producer `ttng.subtiled_region`, consumer `ttng.subtiled_region`) pair into a
**single** `ChannelPost` (the per-tile buffers become in-body instances indexed
by the builtin `tileIdx`). That lone channel still needs the reuse-group
machinery — the in-body per-tile slot rotation (`getOrComputeSubtiledSlot` fires
only for `reuseGrp >= 0`) and the `numTiles` loop-counter stride
(`getReuseGroupStride`). So `doCodePartitionPost` forms a **size-1** group for it
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
(`ChannelPost::isCollapsedBothSubtiled`, set in `collectPostChannels` only when
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
recorded on the representative `ChannelPost` (`collapsedSiblingAllocs`) in
`collectPostChannels` and **erased** in `insertAsyncComm` once the in-body view
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

When channels in a reuse group share a multi-buffered circular buffer, a shared
**accumulation counter** (`accumCnt`) tracks which buffer slot to use. The
counter is carried as a loop argument and incremented as channels are consumed.

Key functions:
- `needAccumCntForReuse` — returns true when a loop/if region contains at
  least one src or dst op of the reuse group and the group is multi-buffered
- `getAccumForReuseGroup` — computes the `accumCnt` SSA value at a given
  operation by walking back through the channel list to find the nearest
  preceding region op, then arithmetically adding the remaining offset
- `getBufferIdxAndPhase` / `getStaggeredAccumCnt` — for the first channel in the
  ordered list, uses `accumCnt` directly; each subsequent channel at position N
  adds N to stagger its slot within the shared circular buffer. **Subtiled-region
  reuse groups do NOT flow through here for their staging slot:** the N subtiles
  share one barrier pair and one physical alloc, so both the data slot and the
  barrier `bufferIdx`/`phase` are computed *inside the tile body* from the builtin
  `tileIdx` (`flattened = accumCnt * numTiles + tileIdx`, `% numBuffers`), keeping
  data slot == barrier generation. The generic `+ position` stagger is wrong for
  subtiles (it aliases distinct subtiles within a `numBuffers` window and races —
  the EPILOGUE_SUBTILE>2 staging-buffer bug). See
  [SubtileOperator](SubtileOperator.md).
- `getReuseAccumArgIdx` — returns the position of a group's `accumCnt`
  argument within the region's full argument list

### 2. Token/Barrier Sharing

In `createTokenPost`, the representative channel (first in the group) creates
barriers; non-representative channels reuse them. `channelInReuseGroup` looks
up which group a channel belongs to (returning -1 if none). The `reuseBarrier`
flag skips groups whose representative has `numBuffers <= 1` (single-buffered
channels share no circular barrier).

### 3. Buffer Replacement

`replaceBufferReuse` rewrites all IR uses of non-representative alloc ops to
point at the representative's alloc. It iterates `config->groups` **directly**
(the source of truth for what must be collapsed) and folds every
non-representative channel of each group into `channels[0]`. It deliberately
does **not** iterate the post-merge channel lists: the reuse-group
consumer-merge in `doCodePartitionPost` removes a non-representative channel
from `orderedChannels` whenever it shares a consumer op with the representative
(e.g. epilogue subtiles that all feed one `ttng.subtiled_region`). Iterating
`orderedChannels` would therefore skip those merged-out channels and leave their
duplicate physical buffers alive — doubling SMEM and causing `OutOfResources`.
Iterating groups visits every non-representative channel regardless of merge
state.

- **SMEM channels**: When the alloc types match, uses direct
  `replaceUsesOfWith` to swap the alloc result, then erases the old alloc. This
  generic rewrite also updates `ttng.subtiled_region` operands, which are
  ordinary users of the alloc result. Type mismatches are skipped (SMEM cannot
  be reinterpreted like TMEM).

- **TMEM channels**: Inserts a `sliceAndReinterpretMDTMEM` op at the
  `buffer.offset` column within the representative's TMEM allocation. If the
  primary representative's type cannot accommodate the slice, other group
  representatives are tried before emitting an error.

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
| `WaitBarrierOp` (gen5 inline) | `consumerBarriers` populated | Before the producer, via `desyncTCGen5MMAOp(..., asProducerAcquire=true)` |

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

Implementation:
```
verifyReuseGroup2(group):
  assert group.channels.size() == 2
  A = group.channels[0], B = group.channels[1]
  assert A.getNumBuffers() == 1 && B.getNumBuffers() == 1

  // Check dependency chain: A.consumer → B.producer or B.consumer → A.producer
  hasAtoB = isDependencyChain(A.dstOp, B.srcOp)
  hasBtoA = isDependencyChain(B.dstOp, A.srcOp)
  assert (hasAtoB || hasBtoA) // At least one direction
  return true
```

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
  if isDependencyChain(A.dstOp, B.srcOp):
    return {A, B}
  return {B, A}
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
  bConsumerOp = getUniqueActualConsumer(lateChannel.dstOp, consumerTaskId)
  aProducerOp = earlyChannel.srcOp

  bConsumerTasks = getAsyncTaskIds(bConsumerOp)
  aProducerTasks = getAsyncTaskIds(aProducerOp)

  if bConsumerTasks and aProducerTasks share a common taskId:
    if appearsBefore(aProducerOp, bConsumerOp):
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
`desyncTCGen5MMAOp` insertion.

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
representative (its alloc has no `buffer.offset`) and is a `TmemDataChannelPost`
with `isOperandDNoAcc == true` (set in `createChannelPost` when the producer MMA's
`useAccumulator()` is constant-false). The outer-cadence siblings are collected in
`fullOverwriteOuterSiblings`; a debug-only assert in the emission step guards
against silently dropping a required sibling back-edge (the bug class reappearing).
Regression test:
`test/Hopper/WarpSpecialization/ws_code_partition_tmem_packed_reuse_backward.mlir`;
runtime validation is the FA-fwd-persistent dp kernel determinism check.

## Key Attributes

| Attribute | Description | Set by | Read by |
|-----------|-------------|--------|---------|
| `buffer.id` | Groups channels that share physical memory | `WSMemoryPlanner` (SMEM + TMEM) | `doCodePartitionPost` (group formation) |
| `buffer.copy` | Number of pipeline copies (multi-buffering depth) | `WSMemoryPlanner` | Buffer allocation, `needAccumCntForReuse` |
| `buffer.offset` | Column offset within the owner's TMEM allocation | `WSMemoryPlanner` (`applyAllocationState`) | `replaceBufferReuse` (TMEM slice offset) |
| `allocation.shareGroup` | Tags buffers for downstream passes | `doCodePartitionPost` | Downstream passes |

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
| Reuse group formation | `WSCodePartition.cpp` (`doCodePartitionPost`) | Group channels by `buffer.id`, form `ReuseConfig` |
| SMEM `buffer.id` assignment | `WSMemoryPlanner.cpp` | Assign `buffer.id` to SMEM allocs |
| SMEM circular reuse (Phase 4) | `WSMemoryPlanner.cpp` | Form SMEM reuse pairs, maximize copies |
| TMEM `applyAllocationState` | `WSMemoryPlanner.cpp` | Assign `buffer.id` + `buffer.offset` to TMEM allocs |
| `verifyReuseGroup2` | `CodePartitionUtility.cpp` | Verify 2-buffer reuse group constraints |
| `orderReuseGroup2` | `CodePartitionUtility.cpp` | Determine early/late channel ordering |
| `needExplicitReuseWait` | `CodePartitionUtility.cpp` | Check if explicit cross-channel wait is needed |
| `isWholeAllocationOverwriteReuseOwner` | `CodePartitionUtility.cpp` | Detect a representative whose producer overwrites the whole allocation (needs back-edges to live packed siblings) |
