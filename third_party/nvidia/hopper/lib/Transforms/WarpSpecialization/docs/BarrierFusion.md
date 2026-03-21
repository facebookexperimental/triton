# Barrier Fusion

This document describes how barriers are created, fused, and lowered for
different async operation types in the AutoWS pipeline. Barrier fusion reduces
the number of mbarrier allocations and arrive/wait operations, improving
performance by amortizing synchronization overhead.

## Background: mbarrier Semantics

An **mbarrier** (memory barrier) is an SMEM-allocated synchronization primitive.
Key properties:

- **Arrive count**: initialized via `InitBarrierOp`. The barrier completes when
  this many arrivals are registered.
- **Wait**: blocks until the arrive count is reached for the current phase.
- **Phase**: a parity bit (0 or 1) that alternates between uses, allowing
  reuse of the same mbarrier across iterations.
- **Expect**: `BarrierExpectOp` sets the number of bytes the barrier should
  expect from TMA operations before it completes.

**Named barriers** (indices 0-15) are hardware-allocated and do not require
SMEM. They are used for ping-pong scheduling (see
[PingPongScheduling.md](PingPongScheduling.md)), not for the data-flow barriers
described here.

## Producer-Consumer Protocol

The full synchronization protocol for a multi-buffered channel:

```
Producer (load partition):              Consumer (MMA/compute partition):
───────────────────────────             ──────────────────────────────────
wait(emptyBarrier[i], phase)            wait(readyBarrier[i], phase)
  ↓ buffer slot i is free to write        ↓ data is available to read
BarrierExpectOp(readyBarrier[i], bytes) use the data (LocalLoad, MMA, ...)
TMA copies → readyBarrier[i]              ↓ done reading
  ↓ TMA hardware auto-arrives            arrive(emptyBarrier[i])
                                          ↓ signal buffer slot is free
advance i, flip phase                   advance i, flip phase
```

The **ready barriers** ("full barriers") signal that data is available. The
**empty barriers** signal that a buffer slot is free for the producer to reuse.

## TMA Barrier Fusion

**File**: `WSLowerMem.cpp` (`optimizeTMALoads`)

TMA (Tensor Memory Accelerator) barrier fusion is the most common form of
barrier fusion. When multiple TMA loads share the same dominant consumer
operation (e.g., they all feed into the same MMA), they are fused onto a
**single mbarrier** with a **single `BarrierExpectOp`** whose byte count is
the sum of all loads' sizes.

### Why This Works

TMA load operations take an mbarrier operand. When the hardware completes
the copy, it automatically decrements the barrier's pending count by the
number of bytes transferred. No software arrive is needed. By pointing
multiple TMA loads at the same barrier and setting the expected byte count
to their sum, a single barrier wait covers all loads.

### Algorithm (`optimizeTMALoads`)

1. **Group channels by consumer**: Channels with the same consumer operation
   are grouped together. Each group gets a single barrier pair (ready + empty).

2. **Compute combined byte count**: `BarrierExpectOp` is emitted once with
   the total `txCount` summed across all TMA loads in the group.

3. **Issue TMA copies**: All `AsyncTMACopyGlobalToLocalOp` operations in the
   group reference the same ready barrier. The hardware auto-arrives on this
   barrier when each copy completes.

4. **Single wait**: The consumer issues a single `WaitBarrierOp` on the ready
   barrier, which completes when all TMA copies have arrived.

### Where It's Called

`optimizeTMALoads` is called from `insertAsyncCopy` in `WSCodePartition.cpp`
during the `doCodePartitionPost` pass. It processes groups of channels whose
producers are TMA descriptor loads.

## tcgen05_commit Barrier Fusion

**File**: `CodePartitionUtility.cpp` (`fuseTcgen05CommitBarriers`)

`TCGen5CommitOp` is the instruction that makes an mbarrier track the
completion of all prior asynchronous tcgen05 operations (MMA and TMEM copy).
Instead of a software `ArriveBarrierOp`, the system emits a `TCGen5CommitOp`
that atomically tracks completion of all preceding async operations.

### How It Works

The `TCGen5CommitOp` uses **commit groups** — sequential groups of async
operations. When `TCGen5CommitOp` is issued with barrier A, that barrier's
arrive count is decremented when all preceding async tcgen05 operations
complete. A subsequent `TCGen5CommitOp` with barrier B is guaranteed to
arrive after barrier A, preserving ordering.

### Fusion Algorithm (`fuseTcgen05CommitBarriers`)

When multiple `TCGen5CommitOp`s in the same block share the same barrier,
they can be fused into a single commit:

1. **Collect commit groups** (`collectCommitGroup`): Walk the block and group
   `TCGen5CommitOp`s that reference the same barrier value. Operations between
   commits are checked for interference — if an intervening op uses a different
   barrier, the group is split.

2. **Match phases** (`hasMatchingPhase`): Verify that the commit ops being
   fused operate on the same phase of the barrier. Phases are tracked through
   `MemDescIndexOp` chains to ensure correctness.

3. **Merge subgroups** (`mergeSubgroups`): For commit ops that can be safely
   combined, keep only the last one in program order and erase the others.
   The last commit subsumes all preceding ones because tcgen05_commit is
   cumulative — it covers all async ops issued since the previous commit.

### Where It's Used

`fuseTcgen05CommitBarriers` is called from `doCodePartitionPost` in
`WSCodePartition.cpp` after channels and barriers have been created. It is
also used for operand D synchronization, where `desyncTCGen5MMAOp` (in
`WSCodePartition.cpp`) adds completion barriers to MMA ops, and the resulting
`tcgen05_commit` operations are then fused by this pass.

## Token Lowering: Barrier Materialization

**File**: `WSLowerToken.cpp`

Barrier fusion interacts with token lowering. `CreateTokenOp` produces
abstract synchronization tokens that are lowered to concrete mbarrier
allocations by `doTokenLowering`. Each token becomes two barrier arrays
(ready and empty), each with `numBuffers` entries. When channels share
tokens (from the grouping in `doCodePartitionPost`), they share the
materialized barriers, which is another form of barrier reduction.

See [Token & Barrier Lowering](TokenBarrierLowering.md) for the full
lowering algorithm.

## Data-Partitioned Commit Replacement

**File**: `WSCodePartition.cpp` (`replaceDataPartitionedCommits`)

In data-partitioned loops (`tt.data_partition_factor > 1`), the D-channel
creation emits multiple standalone `tcgen05_commit` ops after the inner for
loop — one per data-partitioned MMA. Because `tcgen05_commit` is a global
fence that commits ALL pending async tcgen05 operations, using it for
per-MMA D-channel signaling is unnecessarily coarse: the first commit must
wait for every outstanding MMA, serializing completion.

`replaceDataPartitionedCommits` replaces each commit with a targeted
barrier-based synchronization:

1. **Group detection**: Walk the function for groups of consecutive
   `TCGen5CommitOp`s (skipping interleaved `MemDescIndexOp`s that compute
   barrier indices). Groups with 2+ commits indicate data-partitioned
   D-channels.

2. **MMA matching**: For each commit in the group, find the corresponding
   `TCGen5MMAOp` inside the preceding for loop by matching `async_task_id`.

3. **Replacement**: For each commit except the last:
   - Compute the final-iteration buffer index and phase for the MMA's
     inline A/B barrier (via `getOutOfScopeBufferIdxAndPhase`).
   - Emit `WaitBarrierOp` on the A/B barrier — waits for that specific MMA
     to finish its final iteration.
   - Emit `ArriveBarrierOp` on the D barrier — signals the D-channel
     consumer that the MMA result is available.
   - Erase the original `TCGen5CommitOp`.

The last commit in the group is kept because `tcgen05_commit` is
cumulative — it covers all async ops since the previous commit, so the
final one ensures all remaining MMA operations complete. This enables
per-MMA completion tracking for earlier commits while preserving
correctness for the last one.

## Summary: Forms of Barrier Fusion

| Fusion Type | What Gets Fused | Result | Where |
|------------|----------------|--------|-------|
| **TMA fusion** | Multiple TMA loads to same consumer | Single mbarrier, single `BarrierExpectOp` with summed bytes | `WSLowerMem.cpp::optimizeTMALoads` |
| **tcgen05_commit** | Multiple commits to same barrier | Single `TCGen5CommitOp` (last one kept) | `CodePartitionUtility.cpp::fuseTcgen05CommitBarriers` |
| **DP commit replacement** | Consecutive commits from data-partitioned D-channels | Per-MMA `WaitBarrierOp` + `ArriveBarrierOp` | `WSCodePartition.cpp::replaceDataPartitionedCommits` |
| **Token sharing** | Channels grouped by consumer | Shared `CreateTokenOp` → shared barrier pair | `WSCodePartition.cpp::doCodePartitionPost` |
