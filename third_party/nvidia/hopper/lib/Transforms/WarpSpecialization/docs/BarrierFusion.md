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
barrier fusion. At this point TMA producers are buffered
`nvws.descriptor_load` operations. When multiple loads share the same dominant consumer
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
   the `txCount` attributes summed across all NVWS loads in the group. Each
   count is computed from the descriptor's CTA-local shape.

3. **Issue TMA copies**: All `AsyncTMACopyGlobalToLocalOp` operations in the
   group reference the same ready barrier. The hardware auto-arrives on this
   barrier when each copy completes.

4. **Single wait**: The consumer issues a single `WaitBarrierOp` on the ready
   barrier, which completes when all TMA copies have arrived.

Fused loads must also agree on their issue protocol: a barrier group is either
entirely ordinary per-CTA loads or entirely cooperative 2-CTA loads. Mixing the
two would route ordinary completion to the pair leader while computing its
expected byte count with per-CTA semantics. `optimizeTMALoads` checks this
invariant before materializing the shared barrier: a mixed group emits the
error `TMA barrier fusion cannot mix cooperative and per-CTA loads` and the
function returns `nullptr`.

### 2-CTA Cooperative Loads and `TRITON_DISABLE_2CTA_COOPERATIVE_LOADS`

**Files**: `Transform2CTALoads.cpp` (marking), `WSLowerMem.cpp`
(`optimizeTMALoads`, lowering), `LoadStoreOpToLLVM.cpp` (PTX)

In a 2-CTA kernel (`ctas_per_cga=(2,1,1)`, MMAs with `two_ctas`), each CTA
issues its own TMA copies into its own SMEM: its A tile, and its half of B
after `Transform2CTALoads` halves the B descriptor. What differs between the
two protocols below is only how copy completion is tracked. Data destinations
and coordinates are the same in both.

**Marking.** At the end of `Transform2CTALoads::runOnOperation`, the pass
adds the `two_cta_load` unit attribute (`ttng::AttrTwoCTALoadName`) to every
`tt.descriptor_load` whose result is rank 2. This happens only when all of the
following hold:

- The module is a physical cluster.
- The module is not a TLX kernel.
- The module contains at least one MMAv5 op with `two_ctas`.

The marking is module-wide. It covers every rank-2 tile load, including A
operands and rank-2 epilogue loads, not only the B operand. Rank-1 loads stay
unmarked.

**Lowering.** `optimizeTMALoads` reads the attribute per barrier group:

| | Per-CTA (unmarked) | Cooperative (`two_cta_load`) |
|---|---|---|
| Completion barrier | The CTA's own ready barrier | The pair leader's ready barrier (`mapBarrierTo2CTALeader`; leader = even cluster rank) |
| `BarrierExpectOp` | Every CTA expects its own bytes | Only the leader expects, and it expects `2 ×` the summed `txCount` |
| TMA copy | `AsyncTMACopyGlobalToLocalOp` | Same op with `twoCta` set: `cp.async.bulk.tensor ... .shared::cluster ... .cta_group::2` |
| Consumer wait | One `WaitBarrierOp` per consumer task | Leader waits. It then relays to the follower with `FenceAsyncSharedOp` plus an `ArriveBarrierOp` on the follower's barrier (`mapBarrierTo2CTAPeer`), and the follower waits on its own barrier. |

The relay uses a remote arrive rather than a cluster barrier because only the
consuming partition takes part in the handoff.

**The knob.** If `TRITON_DISABLE_2CTA_COOPERATIVE_LOADS` is set,
`Transform2CTALoads` skips the marking step entirely, so every load in the
module takes the per-CTA column. The B-halving transform itself still runs.

| Property | Behavior |
|---|---|
| How it is set | Process environment variable, read with `triton::tools::getBoolEnv`. |
| Legal values | `1`, `true`, or `on` (case-insensitive) disable cooperative loads. Any other value, or unset, leaves them enabled. |
| Default | Unset: cooperative loads are on for every 2-CTA module. |
| Granularity | Whole module. There is no per-load or per-config control. |
| Cache key | Listed in `CACHE_INVALIDATING_ENV_VARS` (`include/triton/Tools/Sys/GetEnv.h`). |
| Resources | No SMEM, TMEM, warp, or barrier-count change. The cooperative path adds the leader-to-follower relay instructions (fence, remote arrive, extra wait) in each consumer partition. |

The code comment in `Transform2CTALoads` describes cooperative marking as a
performance choice, not a correctness requirement: both protocols wait for all
bytes the consumer reads. The cross-CTA MMA rendezvous that `Insert2CTASync`
inserts does not read the attribute, so it is the same in both modes.

Fallbacks and failure modes:

- Only the Meta WS lowering (`optimizeTMALoads`) reads `two_cta_load`. If a
  load does not go through that path, the attribute has no effect, and setting
  or clearing the variable changes nothing for that load.
- A barrier group that mixes marked and unmarked loads is a compile error
  (`TMA barrier fusion cannot mix cooperative and per-CTA loads`). It is never
  silently miscounted. A group can mix only when a rank-1 load is fused with
  rank-2 loads. With the variable set nothing is marked, so this error cannot
  occur.

### Where It's Called

`optimizeTMALoads` is called from `insertAsyncComm` in `WSCodePartition.cpp`
during the `doCodePartition` pass. It processes groups of channels whose
producers are `nvws.descriptor_load` operations. Each operation already owns
its destination memdesc, so planner buffer replacement cannot be bypassed by a
separate lowering-side buffer lookup. Lowering derives the allocation from
that destination and rebuilds its stage view with the fused barrier's buffer
index, keeping the copy slot and barrier phase aligned.

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

`fuseTcgen05CommitBarriers` is called from `doCodePartition` in
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
tokens (from the grouping in `doCodePartition`), they share the
materialized barriers, which is another form of barrier reduction.

See [Token & Barrier Lowering](TokenBarrierLowering.md) for the full
lowering algorithm.

## Data-Partitioned Commit Replacement

**File**: `WSCodePartition.cpp` (`replaceCommitWithBarrierSync`)

In data-partitioned loops (`tt.data_partition_factor > 1`) with multiple MMAs,
the D-channel creation sites generate `wait_barrier` + `arrive_barrier` pairs
directly instead of `tcgen05_commit` ops. Because `tcgen05_commit` is a global
fence that commits ALL pending async tcgen05 operations, using it for per-MMA
D-channel signaling is unnecessarily coarse: the first commit must wait for
every outstanding MMA, serializing completion.

The replacement is performed inline at the two commit creation sites in
`insertAsyncComm` (the `producerBarrier` and `consumerBarrier` paths), rather
than as a separate post-pass. This has two advantages: (1) the MMA's inline
A/B barrier is already available at channel creation time (A/B channels are
processed before D-channels in program order), and (2) there is a direct 1:1
mapping between each D-channel and its MMA, avoiding the need for heuristic
commit-to-MMA matching.

### How It Works

At each D-channel commit creation site, when `mmaCount > 1` in the nested loop:

1. **A/B barrier lookup**: Retrieve the MMA's inline completion barrier (set
   by the A/B consumer_release channel processed earlier). Trace through the
   `MemDescIndexOp` to get the underlying barrier allocation.

2. **Final-iteration index**: Compute the buffer index and phase for the A/B
   barrier's final loop iteration via `getOutOfScopeBufferIdxAndPhase`.

3. **Wait on A/B barrier**: Emit `WaitBarrierOp` on the A/B barrier — waits
   for that specific MMA to finish its final iteration.

4. **D barrier index**: Compute the buffer index for the D barrier (which may
   have a different number of buffers than the A/B barrier — e.g., 1 buffer
   vs 3).

5. **Arrive on D barrier**: Emit `ArriveBarrierOp` on the D barrier — signals
   the D-channel consumer that the MMA result is available.

**Invariant**: each call to `replaceCommitWithBarrierSync` must represent the
work of exactly one MMA — the commit being replaced must correspond to a single
MMA's D-channel, not aggregate work from multiple MMAs. This is structurally
guaranteed because the call sites iterate per-channel (each D-channel maps to
one MMA), and the `mmaCount > 1` guard ensures the replacement is only
attempted when data partitioning has produced multiple distinct per-MMA
channels.

When there is only a single MMA in the loop, or when the MMA lacks an inline
A/B barrier, the standard `tcgen05_commit` is emitted as a fallback.

## Summary: Forms of Barrier Fusion

| Fusion Type | What Gets Fused | Result | Where |
|------------|----------------|--------|-------|
| **TMA fusion** | Multiple TMA loads to same consumer | Single mbarrier, single `BarrierExpectOp` with summed bytes | `WSLowerMem.cpp::optimizeTMALoads` |
| **tcgen05_commit** | Multiple commits to same barrier | Single `TCGen5CommitOp` (last one kept) | `CodePartitionUtility.cpp::fuseTcgen05CommitBarriers` |
| **DP commit replacement** | Per-MMA D-channel commits (when multiple MMAs) | Per-MMA `WaitBarrierOp` + `ArriveBarrierOp` | `WSCodePartition.cpp::replaceCommitWithBarrierSync` |
| **Token sharing** | Channels grouped by consumer | Shared `CreateTokenOp` → shared barrier pair | `WSCodePartition.cpp::doCodePartition` |
