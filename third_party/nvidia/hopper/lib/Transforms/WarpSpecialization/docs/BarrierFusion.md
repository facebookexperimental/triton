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

### New Pipeline: `LoadMMASpecialization`

**File**: `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/LoadMMASpecialization.cpp`

In `PipelinedLoadGroup::lowerLoads()` (lines 369-468):

1. **Group loads**: Loads are grouped by common dominant consumer ops
   (lines 866-875 in `lowerLoops`).
2. **Share barriers**: A single pair of `emptyBars` / `readyBars` is allocated
   for all loads in the group (lines 337-338).
3. **Accumulate byte counts**: `loadSizeInBytes` sums across all loads
   (lines 386-388).
4. **Single expect**: One `BarrierExpectOp` is emitted with the combined size
   (lines 389-391).
5. **TMA copies point to shared barrier**: All copies via `lowerTMACopy()`
   reference the same `curLoadBar` (lines 349-367, 439).

### Legacy Pipeline: `LowerAref`

**File**: `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp`

In `lowerTMALoad()` (lines 287-324):

1. When an `ArefPutEnterOp` has TMA loads as users, accumulate `txCount`
   across all loads (line 299).
2. Create a single `BarrierExpectOp` with the combined byte count on the
   **full barrier** (lines 307-310).
3. Issue all TMA copies pointing to the same barrier.
4. In `insertArriveBarrier()` (line 454): for `AsyncOp::TMALoad`, **nothing
   is emitted** — the arrive is done by hardware.

## tcgen05_commit Barrier Fusion

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

### New Pipeline: Completion Barriers on MMA

In `LoadMMASpecialization.cpp` (lines 770-786):

MMA operations directly carry completion barriers:
```cpp
mmaOp.addCompletionBarrier(bar, userPred);
mmaOp.setIsAsync(true);
```

The MMA op definition (`TritonNvidiaGPUOps.td`, line 649) accepts variadic
barrier operands (`Variadic<TTG_MemDescType>:$barriers`). At LLVM lowering,
this generates `tcgen05.commit` followed by arrive on those barriers.

This is used to signal the empty barrier — when the MMA finishes reading from
a buffer, it signals the producer that the buffer slot is free.

### Legacy Pipeline: Arrive Dispatch

In `LowerAref.cpp`, `insertArriveBarrier()` (lines 438-463) dispatches based
on the `AsyncOp` type:

| `AsyncOp` | Barrier Mechanism | Software/Hardware |
|-----------|------------------|-------------------|
| `NONE` | `ArriveBarrierOp` (count=1) | Software arrive |
| `WGMMA` | `ArriveBarrierOp` (count=1) | Software arrive |
| `TMALoad` | Nothing emitted | Hardware auto-arrive |
| `TC5MMA` | `TCGen5CommitOp` | Hardware-tracked commit |
| `TMEMCopy` | `TCGen5CommitOp` | Hardware-tracked commit |

## Aref Combining

Aref (Async Reference) combining is a barrier fusion optimization at the
abstraction level.

**File**: `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp`,
`combineArefs()` (lines 744-824)

### Algorithm

1. **Group by dominant consumer**: Collect all `ArefGetEnterOp` operations
   and group them by their dominant consumer operation (lines 750-756).

2. **Check combinability**: Multiple arefs feeding the same consumer can be
   combined if their producers are in the same partition.

3. **Merge enter/exit pairs**: `createCombinedArefOps()` (lines 648-715)
   merges multiple enter/exit pairs into single combined enter/exit
   operations.

4. **Union async ops**: The combined exit's `async_ops` attribute is the
   union of all individual arefs' async ops (lines 668-671). For example,
   if one aref tracks `TMALoad` and another tracks `TC5MMA`, the combined
   aref tracks both.

### Impact

This directly reduces:
- The number of mbarrier allocations (fewer arefs = fewer barriers)
- The number of arrive/wait operations
- The number of `BarrierExpectOp` / `TCGen5CommitOp` instructions

## Arrive Count Computation

**File**: `LowerAref.cpp`, `getArrivalCount()` (lines 142-195)

The arrive count determines how many arrivals are needed before a barrier
completes. It is computed per-partition:

### Empty Barriers (producer-side)

`producerPendingCount` = number of distinct consumer partition groups. Each
consumer partition arrives once when it finishes reading from the buffer slot.

Different `AsyncOp` types contribute to the producer pending count:
- `TC5MMA`, `WGMMA`, `NONE`: each consumer partition group adds 1
- `TMALoad`: does not contribute (TMA is a producer, not a consumer)

### Ready Barriers (consumer-side)

`consumerPendingCount` = number of distinct producer partition groups. Each
producer partition arrives once when it finishes filling the buffer slot.

Different `AsyncOp` types contribute to the consumer pending count:
- `TC5MMA`, `TMALoad`, `NONE`: each producer partition group adds 1
- `WGMMA`: does not contribute to consumer count

## Phase Tracking

**File**: `third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignStagePhase.cpp`

Each aref enter/exit operation carries a **stage** (buffer index) and **phase**
(parity bit). `AssignStagePhase` assigns these through control flow:

- **Stage**: which buffer slot to use (0 to numStages-1, wrapping)
- **Phase**: parity bit that flips each time a stage wraps around

Producers and consumers use **opposite initial phases** (producer=0,
consumer=1) to ensure proper synchronization — the consumer must not read a
slot until the producer has filled it.

For `scf::ForOp`, stage and phase are threaded as loop-carried values,
advancing on each iteration. For `scf::IfOp`, both branches receive the same
stage/phase.

## Summary: Forms of Barrier Fusion

| Fusion Type | What Gets Fused | Result | Where |
|------------|----------------|--------|-------|
| **TMA fusion** | Multiple TMA loads to same consumer | Single mbarrier, single `BarrierExpectOp` with summed bytes | `LoadMMASpecialization`, `LowerAref::lowerTMALoad` |
| **tcgen05_commit** | MMA/TMEM completion signaling | `TCGen5CommitOp` replaces software `ArriveBarrierOp` | `LoadMMASpecialization` (completion barriers), `LowerAref::insertArriveBarrier` |
| **Aref combining** | Multiple arefs to same consumer | Single combined aref with union of async ops | `LowerAref::combineArefs` |
