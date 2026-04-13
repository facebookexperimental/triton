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
- `getBufferIdxAndPhase` — for the first channel in the ordered list, uses
  `accumCnt` directly; each subsequent channel at position N adds N to stagger
  its slot within the shared circular buffer
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
point at the representative's alloc:

- **SMEM channels**: When the alloc types match, uses direct
  `replaceUsesOfWith` to swap the alloc result, then erases the old alloc.
  Type mismatches are skipped (SMEM cannot be reinterpreted like TMEM).

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
