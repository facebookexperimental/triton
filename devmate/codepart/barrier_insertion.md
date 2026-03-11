# Barrier Insertion in Warp Specialization Code Partition

This document describes how `producer_acquire`, `consumer_release`, and related synchronization primitives are inserted during the warp specialization code partition pass (`WSCodePartition.cpp` → `insertAsyncComm()`).

## Overview

When data flows between two partitions (tasks), the pass creates a **communication channel** with synchronization primitives. The choice of primitives depends on whether the producer or consumer is a `TCGen5MMAOp` (gen5 MMA).

There are two synchronization mechanisms:
1. **Token-based**: Explicit `ProducerAcquireOp` / `ProducerCommitOp` / `ConsumerWaitOp` / `ConsumerReleaseOp`.
2. **Gen5 inline barrier**: `WaitBarrierOp` + the MMA's built-in completion barrier. No explicit acquire/release ops.

## Key Decision: `useGen5Barrier`

```cpp
// Line 872
bool useGen5Barrier = isa<ttng::TCGen5MMAOp>(consumerOp) &&
                      producerOp->getBlock() == consumerOp->getBlock();
```

This is `true` when:
1. The **consumer** op is a `TCGen5MMAOp`, **AND**
2. Producer and consumer are in the **same basic block**.

When true → `consumerBarriers` is populated (an inline barrier alloc is created).
When false → only a **token** (`nvws.create_token`) is created.

Separately, a **`producerBarrier`** is allocated when the producer is a TMA load (`DescriptorLoadOp`) or gen5 MMA (`ProducerIsGen5`).

## Path 1: Token-based (consumer is NOT gen5)

Applies when `commChannel.consumerBarriers` is empty.

### `ProducerAcquireOp` (lines 2876–2898)

```cpp
if (commChannel.consumerBarriers.empty()) {
    auto producerAcquirePoint =
        getSameLevelOp(headConsumer, tmaHeadProducer);
    if (producerAcquireForChannelLoop) {
        builder.setInsertionPoint(producerAcquireForChannelLoop);
    } else {
        builder.setInsertionPoint(producerAcquirePoint);
    }
    builder.createWithAsyncTaskIds<ttnvws::ProducerAcquireOp>(
        headProducer->getLoc(), token, bufferIdx, phase);
}
```

- Inserted **before** the head producer.
- For loop-carried channels, moved to before the backward channel's `dstOp`.
- Uses the **producer's** async task IDs.

### `ConsumerReleaseOp` (lines 2978–2992)

```cpp
if (commChannel.consumerBarriers.empty()) {
    auto consumerReleasePoint =
        consumerReleaseHeuristic(tailProducer, tailConsumer, consumerTaskId);
    builder.setInsertionPointAfter(consumerReleasePoint);
    builder.createWithAsyncTaskIds<ttnvws::ConsumerReleaseOp>(
        consumerReleasePoint->getLoc(), token, bufferIdx);
}
```

- Inserted **after** `consumerReleasePoint`.
- `consumerReleaseHeuristic` (lines 2359–2399) finds the latest point where the consumer data is still needed by tracing `getActualConsumers()` and computing the common post-dominator.

### `ProducerCommitOp` (lines 2900–2963)

Only when there is **no `producerBarrier`** (producer is neither TMA nor gen5):

- Inserted **after** `tailProducer`.
- Special case for TMEM channels where producer is `TMEMStoreOp` feeding gen5 operand A: commit is delayed to after both tmem_stores (data + acc D).

### `ConsumerWaitOp` (lines 2968–2976)

Only when there is **no `producerBarrier`**:

- Inserted **before** `headConsumer`.

## Path 2: Gen5 inline barrier (consumer IS gen5)

Applies when `commChannel.consumerBarriers` is populated.

### Producer acquire → `WaitBarrierOp` with inverted phase

`desyncTCGen5MMAOp()` (line 2865) is called with `asProducerAcquire=true`. It inserts a `WaitBarrierOp` **before the producer** using **inverted phase** (`xor true` at line 2081). This waits for the buffer-empty barrier — semantically equivalent to a producer_acquire.

```cpp
if (asProducerAcquire) {
    Value _1_1b = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
    phase = builder.createWithAsyncTaskIds<mlir::arith::XOrIOp>(loc, inPhase, _1_1b);
}
phase = builder.createWithAsyncTaskIds<arith::ExtUIOp>(loc, i32Type, phase);
auto waitOp = builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
    loc, producerBarrier, phase);
```

### Consumer release → implicit via gen5 inline barrier

The gen5 MMA's inline barrier is attached as a **completion barrier operand** (line 2062):

```cpp
mmaOp.addCompletionBarrier(consumerBarrier, pred);
mmaOp.setIsAsync(true);
```

When the MMA completes, it signals this barrier. No explicit `ConsumerReleaseOp` is emitted — the MMA lowering handles it.

## Path for gen5 as producer (`producerBarrier` set)

When the **producer** is gen5 (lines 2767–2797), `desyncTCGen5MMAOp()` is called with `asProducerAcquire=false`:

- The MMA's inline barrier is attached as a **completion barrier** (producer_commit).
- A `WaitBarrierOp` is inserted **before the consumer** as a consumer_wait.

## Summary Table

| Scenario | `consumerBarriers` | Producer Acquire | Producer Commit | Consumer Wait | Consumer Release |
|---|---|---|---|---|---|
| Consumer is gen5 (same block) | populated | `WaitBarrierOp` (inverted phase) before producer | Implicit via gen5 inline barrier | Implicit via gen5 inline barrier | Implicit via gen5 inline barrier |
| Consumer is NOT gen5, producer is NOT gen5/TMA | empty | `ProducerAcquireOp` before head producer | `ProducerCommitOp` after tail producer | `ConsumerWaitOp` before head consumer | `ConsumerReleaseOp` after last actual consumer |
| Consumer is NOT gen5, producer IS gen5 | empty | `ProducerAcquireOp` before head producer | Implicit via gen5 inline barrier + `WaitBarrierOp` before consumer | `WaitBarrierOp` before head consumer | `ConsumerReleaseOp` after last actual consumer |
| Consumer is NOT gen5, producer IS TMA | empty | `ProducerAcquireOp` before head producer | TMA barrier expect (via `optimizeTMALoads`) | `WaitBarrierOp` on TMA barrier before consumer | `ConsumerReleaseOp` after last actual consumer |

## Example: `dq` and `dsT` in FA BWD

### Channel `dq` (TMEM, gen5 → tmem_load)

- **Producer**: `tc_gen5_mma` (task 1, gemm) computes `dq = dsT^T @ k` into tmem.
- **Consumer**: `tmem_load` (task 0, computation) reads the result.
- **`producerBarrier`** is set (producer is gen5).
- **`useGen5Barrier = false`** (consumer `tmem_load` is not gen5) → `consumerBarriers` empty.
- Result:
  - `ProducerAcquireOp` before the MMA (token-based).
  - Gen5 inline barrier signals MMA completion (producer_commit).
  - `WaitBarrierOp` before `tmem_load` (consumer_wait on the producer barrier).
  - `ConsumerReleaseOp` after `tmem_load` (token-based).

### Channel `dsT` (SMEM, local_store → gen5)

- **Producer**: `local_store` (task 3, computation) writes `dsT` to SMEM.
- **Consumer**: `tc_gen5_mma` for dk and dq (task 1, gemm) reads `dsT` as an operand.
- **`producerBarrier`** is not set (producer is `local_store`, not TMA/gen5).
- **`useGen5Barrier = true`** (consumer is gen5, same block) → `consumerBarriers` populated.
- Result:
  - `WaitBarrierOp` with inverted phase before `local_store` (acts as producer_acquire via gen5 inline barrier).
  - `ProducerCommitOp` after `local_store`.
  - `ConsumerWaitOp` before gen5 MMA.
  - Gen5 inline barrier signals buffer-empty on MMA completion (acts as consumer_release).
  - **No** explicit `ProducerAcquireOp` or `ConsumerReleaseOp`.
