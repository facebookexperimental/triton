# Design: 2-Buffer Reuse Group Synchronization

## Problem

When two channels share the same physical buffer (a **reuse group** with 2 buffers and `buffer.copy=1`), we must ensure that one channel's consumer has fully released the buffer before the other channel's producer acquires it. Today the code shares tokens between reuse group channels but does not reason about the ordering of `producer_acquire` across the two channels.

## Background

### Reuse Groups

Channels sharing the same `buffer.id` are placed into a `ReuseGroup` (populated in `WSCodePartition.cpp` lines 3468–3507). All channels in a reuse group share a single `CommChannel` (tokens, barriers) via `tokenMap`.

### Current `producer_acquire` Insertion

`producer_acquire` is inserted at one of these points (lines 2873–2898):

| Mechanism | Condition | Insertion Point |
|-----------|-----------|-----------------|
| `ProducerAcquireOp` (token-based) | `consumerBarriers` empty | Before `headProducer` (or `producerAcquireForChannelLoop`) |
| `WaitBarrierOp` (gen5 inline) | `consumerBarriers` populated | Before the producer, via `desyncTCGen5MMAOp(..., asProducerAcquire=true)` |

The variable `producerAcquireForChannelLoop` (line 2698) already handles the case of **forward/backward channel loops** (same alloc, same block, cycle through gen5 operand D). This design extends that concept to **reuse groups**.

## Requirements (from `reuse.cp`)

For a reuse group with 2 buffers A and B (`buffer.copy=1`):

1. **Verification**: Each buffer must have exactly one channel, and there must be a dependency chain from one buffer's consumer to the other's producer.
2. **Ordering**: Determine which buffer is "early" (A) and which is "late" (B). If `A.producer → A.consumer → B.producer`, then A is early.
3. **Case analysis**: Check whether there is an ordering from B's consumer back to A's producer:
   - **Implicit ordering** (e.g. `qk/pp`): B's consumer and A's producer are both in the same partition (e.g. gemm). The partition-internal ordering already guarantees B's consumer_release happens after A's producer_acquire. No additional synchronization needed.
   - **Explicit wait needed** (e.g. `dp/dq`): B's consumer and A's producer are in different partitions (or same partition but wrong order). We must move B's `producer_acquire` to be before A's producer, so A's producer waits for B's consumer_release before writing.

## Design

### 1. Helper Functions

Add to `CodePartitionUtility.h` / `CodePartitionUtility.cpp`:

#### `verifyReuseGroup2`

```cpp
// Verify a 2-buffer reuse group:
// - Exactly 2 channels.
// - Each channel has 1 copy (getNumBuffers() == 1).
// - A dependency chain exists between one channel's consumer and the other's producer.
// Returns true if valid.
bool verifyReuseGroup2(ReuseGroup *group);
```

#### `orderReuseGroup2`

```cpp
// For a verified 2-buffer reuse group, determine which channel is early (A)
// and which is late (B).
// Channel A is early if: A.producer appears before A.consumer appears before
// B.producer in program order.
// Returns {earlyChannel, lateChannel}.
std::pair<Channel *, Channel *> orderReuseGroup2(ReuseGroup *group);
```

#### `needExplicitReuseWait`

```cpp
// Given ordered channels {A (early), B (late)} in a reuse group,
// determine whether we need to explicitly wait for B's consumer_release
// before A's producer_acquire.
//
// Returns false (no explicit wait needed) when:
//   B's consumer (or actual consumer) and A's producer are in the same
//   partition AND B's consumer appears after A's producer in program order
//   (partition-internal ordering guarantees correctness).
//
// Returns true when B's consumer and A's producer are in different
// partitions or when there is no ordering guarantee.
bool needExplicitReuseWait(Channel *earlyChannel, Channel *lateChannel);
```

### 2. Implementation Details

#### `verifyReuseGroup2`

```
verifyReuseGroup2(group):
  assert group.channels.size() == 2
  A = group.channels[0], B = group.channels[1]
  assert A.getNumBuffers() == 1 && B.getNumBuffers() == 1

  // Check dependency chain: A.consumer → B.producer or B.consumer → A.producer
  // Use getActualConsumers() to find transitive consumers of A.dstOp,
  // then check if B.srcOp is reachable.
  hasAtoB = isDependencyChain(A.dstOp, B.srcOp)
  hasBtoA = isDependencyChain(B.dstOp, A.srcOp)
  assert (hasAtoB || hasBtoA) // At least one direction
  return true
```

#### `orderReuseGroup2`

```
orderReuseGroup2(group):
  A = group.channels[0], B = group.channels[1]

  // If A.consumer → B.producer dependency exists, A is early
  if isDependencyChain(A.dstOp, B.srcOp):
    return {A, B}

  // Otherwise B.consumer → A.producer, so B is early
  return {B, A}
```

The dependency chain check can leverage `appearsBefore` on the ops within the same block, or walk the def-use chain from the consumer op to the producer op.

#### `needExplicitReuseWait`

```
needExplicitReuseWait(earlyChannel, lateChannel):
  // earlyChannel = A, lateChannel = B
  // We need to know: does B's consumer release happen before A's producer acquire
  // without explicit synchronization?

  bConsumerOp = getUniqueActualConsumer(lateChannel.dstOp, consumerTaskId)
  aProducerOp = earlyChannel.srcOp  // or headProducer for A

  bConsumerTasks = getAsyncTaskIds(bConsumerOp)
  aProducerTasks = getAsyncTaskIds(aProducerOp)

  // If both are in the same partition, check program order
  if bConsumerTasks and aProducerTasks share a common taskId:
    // Same partition: check if B's consumer appears AFTER A's producer
    // in the original program order. If so, partition-internal ordering
    // guarantees B's consumer_release happens before A's next producer_acquire
    // (since consumer_release is after B's consumer, and A's producer_acquire
    // is before A's producer, and they're ordered within the partition).
    if appearsBefore(aProducerOp, bConsumerOp):
      return false  // No explicit wait needed (qk/pp case)

  return true  // Need explicit wait (dp/dq case)
```

### 3. Integration into `insertAsyncComm`

In the main channel processing loop (around line 2698), after computing `producerAcquireForChannelLoop`, add the reuse group logic:

```cpp
Operation *producerAcquireForChannelLoop = nullptr;
if (headProducer->getBlock() == headConsumer->getBlock()) {
  auto *bwdCh = isForwardOfChannelLoop(masterChannel);
  if (bwdCh)
    producerAcquireForChannelLoop = bwdCh->getDstOp();
}

// --- NEW: 2-buffer reuse group handling ---
Operation *producerAcquireForReuse = nullptr;
int reuseGrp = channelInReuseGroup(masterChannel, config);
if (reuseGrp >= 0) {
  auto *group = config->getGroup(reuseGrp);
  if (group->channels.size() == 2) {
    verifyReuseGroup2(group);
    auto [earlyChannel, lateChannel] = orderReuseGroup2(group);

    if (masterChannel == earlyChannel) {
      // Early buffer (A): check if we need explicit wait for late buffer's
      // consumer_release.
      if (needExplicitReuseWait(earlyChannel, lateChannel)) {
        // Move late buffer's producer_acquire to before early buffer's
        // producer, so early buffer's producer_acquire waits for late
        // buffer's consumer_release.
        // The insertion point for early buffer's acquire is the same as
        // its headProducer (default behavior) — no change needed here.
        // The key change is for the LATE buffer (see below).
      }
      // else: implicit ordering (qk/pp case), no change needed.
    } else {
      // Late buffer (B): if explicit wait is needed, move this buffer's
      // producer_acquire to before the early buffer's producer.
      assert(masterChannel == lateChannel);
      if (needExplicitReuseWait(earlyChannel, lateChannel)) {
        producerAcquireForReuse = earlyChannel->getSrcOp(); // A's producer
        LDBG("move producer_acquire for reuse group late buffer "
             << masterChannel->uniqID);
      }
    }
  }
}

// Combine with existing producerAcquireForChannelLoop
if (producerAcquireForReuse && !producerAcquireForChannelLoop) {
  producerAcquireForChannelLoop = producerAcquireForReuse;
}
// --- END NEW ---
```

This reuses the existing `producerAcquireForChannelLoop` mechanism which already flows through to the `ProducerAcquireOp` insertion (line 2884) and the gen5 inline barrier `desyncTCGen5MMAOp` insertion (line 2847).

### 4. Processing Order

The reuse group ordering must be respected during channel iteration. The early channel should be processed before the late channel so that when the late channel is processed, it can reference the early channel's producer as an insertion point.

In `orderedChannelsGroupedByConsumers` construction (lines 2407–2430), ensure that within a reuse group, the early channel appears first. This may require a sorting step:

```cpp
// After building orderedChannelsGroupedByConsumers, reorder within reuse groups
for (unsigned idx = 0; idx < config.getGroupSize(); idx++) {
  auto *group = config.getGroup(idx);
  if (group->channels.size() == 2) {
    auto [early, late] = orderReuseGroup2(group);
    // Ensure early appears before late in orderedChannelsGroupedByConsumers
    // (swap if necessary)
  }
}
```

## Examples

### `dp/dq` (explicit wait needed)

```
dp: producer = tc_gen5_mma (task 1, gemm)    → consumer = tmem_load (task 3, computation)
dq: producer = tc_gen5_mma (task 1, gemm)    → consumer = tmem_load (task 0, computation)
```

- Ordering: `dp` is early (dp.producer → dp.consumer → dq.producer in program order).
- `dq.consumer` (task 0) and `dp.producer` (task 1) are in **different partitions** → `needExplicitReuseWait` returns `true`.
- Action: Move `dq`'s `producer_acquire` to before `dp`'s producer. This ensures `dp`'s producer waits (via the shared token) until `dq`'s consumer releases the buffer.

### `qk/pp` (implicit ordering)

```
qk: producer = TMA load (task 2, load)       → consumer = tc_gen5_mma (task 1, gemm)
pp: producer = local_store (task 3, comp)     → consumer = tc_gen5_mma (task 1, gemm)
```

- Ordering: `pp` is early (pp.producer → pp.consumer → qk.producer).
- `pp.consumer` (task 1, gemm) and `qk.producer` (task 1, gemm) are in the **same partition** and `qk.producer` appears before `pp.consumer` → `needExplicitReuseWait` returns `false`.
- Action: No change. Partition-internal ordering guarantees correctness.

## Files to Modify

| File | Changes |
|------|---------|
| `CodePartitionUtility.h` | Add `verifyReuseGroup2`, `orderReuseGroup2`, `needExplicitReuseWait` declarations |
| `CodePartitionUtility.cpp` | Implement the three helper functions |
| `WSCodePartition.cpp` | Integrate reuse group logic into `insertAsyncComm` (~line 2698), add processing order enforcement |

## Testing

- Use the existing lit test command: `triton-opt --mlir-print-debuginfo --mlir-use-nameloc-as-prefix --nvgpu-test-ws-code-partition="num-buffers=1 post-channel-creation=1" t.dump.cp`
- Verify `dp/dq` case: `dq`'s `producer_acquire` appears before `dp`'s producer in the output.
- Verify `qk/pp` case: no change in `producer_acquire` placement.
