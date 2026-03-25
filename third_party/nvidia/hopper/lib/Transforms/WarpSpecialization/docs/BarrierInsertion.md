# Barrier Insertion

This document describes how `producer_acquire`, `consumer_release`, and
related synchronization primitives are inserted during the warp specialization
code partition pass. This is the implementation-level complement to the
high-level overview in [Code Partitioning](CodePartition.md) and the
optimization-focused [Barrier Fusion](BarrierFusion.md).

**File**: `WSCodePartition.cpp` → `insertAsyncComm()`

## Overview

When data flows between two partitions (tasks), the pass creates a
**communication channel** with synchronization primitives. The choice of
primitives depends on whether the producer or consumer is a `TCGen5MMAOp`
(gen5 MMA).

There are two synchronization mechanisms:
1. **Token-based**: Explicit `ProducerAcquireOp` / `ProducerCommitOp` /
   `ConsumerWaitOp` / `ConsumerReleaseOp`.
2. **Gen5 inline barrier**: `WaitBarrierOp` + the MMA's built-in completion
   barrier. No explicit acquire/release ops.

## Key Decision: `useGen5Barrier`

```cpp
bool useGen5Barrier = isa<ttng::TCGen5MMAOp>(consumerOp) &&
                      producerOp->getBlock() == consumerOp->getBlock();
```

This is `true` when:
1. The **consumer** op is a `TCGen5MMAOp`, **AND**
2. Producer and consumer are in the **same basic block**.

When true → `consumerBarriers` is populated (an inline barrier alloc is
created).
When false → only a **token** (`nvws.create_token`) is created.

Separately, a **`producerBarrier`** is allocated when the producer is a TMA
load (`DescriptorLoadOp`) or gen5 MMA (`ProducerIsGen5`).

## Path 1: Token-Based (Consumer is NOT gen5)

Applies when `commChannel.consumerBarriers` is empty.

### `ProducerAcquireOp`

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

### `ConsumerReleaseOp`

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
- `consumerReleaseHeuristic` finds the latest point where the consumer data is
  still needed by tracing `getActualConsumers()` and computing the common
  post-dominator.

### `ProducerCommitOp`

Only when there is **no `producerBarrier`** (producer is neither TMA nor gen5):

- Inserted **after** `tailProducer`.
- Special case for TMEM channels where producer is `TMEMStoreOp` feeding gen5
  operand A: commit is delayed to after both tmem_stores (data + acc D).

### `ConsumerWaitOp`

Only when there is **no `producerBarrier`**:

- Inserted **before** `headConsumer`.

## Path 2: Gen5 Inline Barrier (Consumer IS gen5)

Applies when `commChannel.consumerBarriers` is populated.

### Producer Acquire → `WaitBarrierOp` with Inverted Phase

`desyncTCGen5MMAOp()` is called with `asProducerAcquire=true`. It inserts
a `WaitBarrierOp` **before the producer** using **inverted phase**
(`xor true`). This waits for the buffer-empty barrier — semantically
equivalent to a producer_acquire.

```cpp
if (asProducerAcquire) {
    Value _1_1b = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
        loc, 1, 1);
    phase = builder.createWithAsyncTaskIds<mlir::arith::XOrIOp>(
        loc, inPhase, _1_1b);
}
phase = builder.createWithAsyncTaskIds<arith::ExtUIOp>(loc, i32Type, phase);
auto waitOp = builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
    loc, producerBarrier, phase);
```

### Consumer Release → Implicit via gen5 Inline Barrier

The gen5 MMA's inline barrier is attached as a **completion barrier
operand**:

```cpp
mmaOp.addCompletionBarrier(consumerBarrier, pred);
mmaOp.setIsAsync(true);
```

When the MMA completes, it signals this barrier. No explicit
`ConsumerReleaseOp` is emitted — the MMA lowering handles it.

## Path for gen5 as Producer (`producerBarrier` set)

When the **producer** is gen5, `desyncTCGen5MMAOp()` is called with
`asProducerAcquire=false`:

- The MMA's inline barrier is attached as a **completion barrier**
  (producer_commit).
- A `WaitBarrierOp` is inserted **before the consumer** as a consumer_wait.

## Summary Table

| Scenario | `consumerBarriers` | Producer Acquire | Producer Commit | Consumer Wait | Consumer Release |
|---|---|---|---|---|---|
| Consumer is gen5 (same block) | populated | `WaitBarrierOp` (inverted phase) before producer | Implicit via gen5 inline barrier | Implicit via gen5 inline barrier | Implicit via gen5 inline barrier |
| Consumer is NOT gen5, producer is NOT gen5/TMA | empty | `ProducerAcquireOp` before head producer | `ProducerCommitOp` after tail producer | `ConsumerWaitOp` before head consumer | `ConsumerReleaseOp` after last actual consumer |
| Consumer is NOT gen5, producer IS gen5 | empty | `ProducerAcquireOp` before head producer | Implicit via gen5 inline barrier + `WaitBarrierOp` before consumer | `WaitBarrierOp` before head consumer | `ConsumerReleaseOp` after last actual consumer |
| Consumer is NOT gen5, producer IS TMA | empty | `ProducerAcquireOp` before head producer | TMA barrier expect (via `optimizeTMALoads`) | `WaitBarrierOp` on TMA barrier before consumer | `ConsumerReleaseOp` after last actual consumer |

## Examples: FA BWD Channels

### Channel `dq` (TMEM, gen5 → tmem_load)

- **Producer**: `tc_gen5_mma` (task 1, gemm) computes `dq = dsT^T @ k`.
- **Consumer**: `tmem_load` (task 0, computation) reads the result.
- **`producerBarrier`** is set (producer is gen5).
- **`useGen5Barrier = false`** (consumer `tmem_load` is not gen5) →
  `consumerBarriers` empty.
- Result:
  - `ProducerAcquireOp` before the MMA (token-based).
  - Gen5 inline barrier signals MMA completion (producer_commit).
  - `WaitBarrierOp` before `tmem_load` (consumer_wait on the producer
    barrier).
  - `ConsumerReleaseOp` after `tmem_load` (token-based).

### Channel `dsT` (SMEM, local_store → gen5)

- **Producer**: `local_store` (task 3, computation) writes `dsT` to SMEM.
- **Consumer**: `tc_gen5_mma` for dk and dq (task 1, gemm) reads `dsT` as
  an operand.
- **`producerBarrier`** is not set (producer is `local_store`, not TMA/gen5).
- **`useGen5Barrier = true`** (consumer is gen5, same block) →
  `consumerBarriers` populated.
- Result:
  - `WaitBarrierOp` with inverted phase before `local_store` (acts as
    producer_acquire via gen5 inline barrier).
  - `ProducerCommitOp` after `local_store`.
  - `ConsumerWaitOp` before gen5 MMA.
  - Gen5 inline barrier signals buffer-empty on MMA completion (acts as
    consumer_release).
  - **No** explicit `ProducerAcquireOp` or `ConsumerReleaseOp`.

---

## FA BWD HD64 Barrier Map

This section provides a complete barrier map for the Flash Attention BWD
persistent kernel with `HEAD_DIM=64`, serving as a concrete reference for
how all the pieces fit together.

### Partitions

| Partition | Type | async_task_id | Warps | Role |
|-----------|------|---------------|-------|------|
| default / partition0 | reduction | 0 | 1 | dQ epilogue: tmem_load dQ → scale → TMA atomic_add to global |
| partition1 | gemm | 1 | 1 | All MMA operations: qkT, dpT, dV, dK, dQ |
| partition2 | load | 2 | 8 | TMA loads: k, v, q, do |
| partition3 | computation | 3 | 8 | Softmax, ppT, dsT computation; tmem_load qkT/dpT; tmem_store ppT |

### TMEM Allocations

| Name | Shape | shareGroup | buffer.id | Encoding |
|------|-------|-----------|-----------|----------|
| dpT  | 1×128×128×f32 | 2 | 8 | blockM=128, blockN=128 |
| qkT  | 1×128×128×f32 | 0 | 7 | blockM=128, blockN=128 |
| dv   | 1×128×64×f32  | 1 | 6 | blockM=128, blockN=64  |
| dk   | 1×128×64×f32  | 3 | 5 | blockM=128, blockN=64  |

### SMEM Allocations

| Name | Shape | buffer.id | Notes |
|------|-------|-----------|-------|
| dsT  | 2×128×128×f16 | 0 | double-buffered |
| do   | 2×128×64×f16  | 1 | double-buffered |
| q    | 2×128×64×f16  | 2 | double-buffered |
| v    | 1×128×64×f16  | 3 | single-buffered |
| k    | 1×128×64×f16  | 4 | single-buffered |

### MMA Operations (all in Task 1 / partition1)

| MMA | Operand D (TMEM) | useAcc | Commit barriers |
|-----|-----------------|--------|-----------------|
| qkT MMA | qkT (memdesc_index) | `false` | 1×1 HW commit |
| dpT MMA | dpT (memdesc_index) | `false` | 2×1 (do consumed) + 1×1 (HW commit) |
| dV MMA  | dv (memdesc_index)  | loop-carried | 1×1 HW commit |
| dK MMA  | dk (memdesc_index)  | loop-carried | 2×1 (q consumed) |
| dQ MMA  | dq (tmem_subslice of dpT, cols 0-63) | `false` | 2×1 (dsT consumed) + 1×1 (dQ commit for Task 0) |

### dQ Operand D Chain

The dQ MMA's operand D is NOT a separate TMEM allocation. It is derived from
the dpT allocation via:

```
%dpT_86 = tmem_subslice %dpT_9 {N = 0}        → cols 0-63 of dpT (128×128)
%dpT_87 = memdesc_reinterpret %dpT_86          → 1×128×64
%dq_88  = memdesc_index %dpT_87[0]             → 128×64
dQ MMA writes to %dq_88
```

This is safe because of the **transitive dependency chain** — by the time dQ
MMA executes, dpT has been consumed by Task 3 (see dpT flow below).

### Complete Barrier Map

| warp_spec arg | Partition arg | Size | Purpose |
|---|---|---|---|
| `%23` | `%arg22` | 2×1 | q TMA load complete |
| `%26` | `%arg25` | 1×1 | qkT MMA HW commit |
| `%31` | `%arg28` | 2×1 | do TMA load complete |
| `%34` | `%arg29` | 1×1 | dV MMA HW commit |
| `%28` | `%arg32` | 2×1 | dpT MMA commit (do consumed) |
| `%36` | `%arg33` | 1×1 | dpT MMA HW commit |
| `%20` | `%arg36` | 2×1 | dK MMA commit (q consumed) |
| `%38` | `%arg37` | 2×1 | dQ MMA commit #1 (dsT consumed) |
| `%41` | `%arg38` | 1×1 | dQ MMA commit #2 (for Task 0 dQ consumer) |
| `%14` | `%arg39` | 1×1 | dK epilog commit |
| `%16` | `%arg40` | 1×1 | dK epilog commit #2 |
| `%18` | `%arg41` | 1×1 | dV epilog commit |
| `%8`  | `%arg42` | 1×1 | k TMA load gate (outer tile) |
| `%44` | `%arg57` | 1×1 | dQ consumed (by Task 0 → Task 1) |
| `%47` | `%arg58` | 2×1 | dsT ready (Task 3 → Task 1) |
| `%54` | `%arg59` | 1×1 | dpT consumed (Task 3 → Task 1) |
| `%57` | `%arg60` | 1×1 | ppT stored / dV consumed (Task 3 → Task 1) |
| `%62` | `%arg61` | 1×1 | qkT consumed (Task 3 → Task 1) |

### Producer-Consumer Barrier Flows

#### Flow 1: qkT (shareGroup 0)

```
Task 1: wait %arg61 (qkT consumed) → qkT MMA → commit %arg25 (HW)
Task 3: wait %arg25 (qkT committed) → tmem_load qkT → arrive %arg61 (qkT consumed)
```

#### Flow 2: dpT (shareGroup 2) — most complex

```
Task 1: wait %arg57 (dQ consumed) + wait %arg59 (dpT consumed) → dpT MMA →
        commit %arg32 (do consumed) + %arg33 (HW)
Task 3: wait %arg33 (dpT committed) → tmem_load dpT → arrive %arg59 (dpT consumed)
Task 2: wait %arg32 (do consumed) → TMA load do
```

#### Flow 3: dV (shareGroup 1)

```
Task 0: tmem_store zeros → dV (init)
Task 3: wait %arg29 (dV committed) → tmem_store ppT → arrive %arg60 (ppT ready)
Task 1: wait %arg60 (ppT ready) → dV MMA (useAcc=true) → commit %arg29 (HW)
Task 3 (epilog): wait %arg41 → tmem_load dV → TMA store to global
```

#### Flow 4: dK (shareGroup 3)

```
Task 0: tmem_store zeros → dK (init)
Task 1: wait %arg58 (dsT ready) → dK MMA (useAcc=true) → commit %arg36 (q consumed)
Task 2: wait %arg36 (q consumed) → TMA load q
Task 3 (epilog): wait %arg39 → tmem_load dK → TMA store to global
```

#### Flow 5: dQ (subslice of dpT, shareGroup 2)

```
Task 1: dQ MMA (after dK MMA) → commit %arg37 (dsT consumed) +
        %arg38 (dQ ready for Task 0)
Task 0: wait %arg38 (dQ committed) → tmem_load dQ (4 × 128×16 chunks) →
        cp.reduce → arrive %arg57 (dQ consumed)
Task 1: wait %arg57 (dQ consumed) → dpT MMA (next iteration)
Task 3: wait %arg37 (dsT consumed) → store next dsT to SMEM
```

#### Flow 6: dsT (SMEM, double-buffered)

```
Task 3: wait %arg37 (dsT consumed) → local_store dsT → arrive %arg58 (dsT ready)
Task 1: wait %arg58 (dsT ready) → dK MMA (reads dsT) → dQ MMA (reads dsT)
Task 1: dQ MMA commit → arrive %arg37 (dsT consumed)
```

### Key Insight: dpT/dQ TMEM Sharing Is Safe

The dQ MMA writes to columns 0-63 of the dpT TMEM buffer. This does NOT race
with Task 3's `tmem_load dpT` because of the **transitive dependency chain**:

```
dpT MMA (Task 1)
  → commit %arg33 (dpT HW commit)
    → Task 3 waits %arg33
      → tmem_load dpT (Task 3 CONSUMES dpT)
        → compute dsT = pT * (dpT - Di)
          → local_store dsT to SMEM
            → arrive %arg58 (dsT READY)
              → Task 1 waits %arg58
                → dK MMA (reads dsT from SMEM)
                  → dQ MMA (writes to dpT subslice) ← dpT already consumed!
```

### Barrier Initialization

All barriers are initialized with `init_barrier ..., 1` (arrival count = 1).
Barriers are separated by `gpu.barrier` calls to ensure visibility across
warp groups before the `warp_specialize` region begins.

Single-buffered barriers (`1×1`): phase alternates `curr_m & 1`.
Double-buffered barriers (`2×1`): indexed by `tile_idx % 2`.

---

## Known Issues: BWD Persistent Kernel Bugs

This section documents known bugs found during BWD persistent kernel
bring-up. Some are fixed; others remain open.

### Bug 1 — 2-Buffer Reuse Group Fires Incorrectly (NaN results)

**Status:** Fixed (commit `92a456c0`)

The 2-buffer reuse group logic moved `producer_acquire` for a late channel
before an early channel's producer **even when the late channel's consumer was
in a different control block**. In the BWD kernel this corrupted the MMA
pipeline ordering, leading to reads of uninitialized TMEM.

**Fix:** Added a guard condition requiring the late consumer to be in the
**same block** as the early producer. See [Reuse Groups](ReuseGroups.md) for
the full 2-buffer reuse group design.

### Bug 2 — TMA Store Column Offset

**Status:** Fixed (commit `b56dee56`)

With `EPILOGUE_SUBTILE = 4`, all four TMA store chunks used hardcoded column
offset `0`, causing every chunk to overwrite the first 32 columns. This was
a kernel authoring bug, not a compiler bug.

### Bug 3 — dK Race Condition (Reduction Zeros TMEM Before Computation Reads)

**Status:** Fixed

The gemm partition's `tc_gen5_commit` signaled both bar_A (for the reduction's
tmem_store) and bar_B (for the computation's tmem_load) simultaneously. The
tmem_store zeroed dk TMEM while tmem_load was still reading it.

See [Operand D Handling](OperandDHandling.md#the-operand-d-race--and-the-fix)
for the full race analysis, the token-based fix, and the same-task guard for
FA FWD.

### Bug 4 — dV Accuracy at BM64 (Open)

**Status:** Open — root cause confirmed via runtime diagnostics

**Error:** `max|err| = 0.98` (non-deterministic). Affected gradient: dV only.
First tile per CTA always passes; subsequent tiles fail ~18% of the time.

**Root cause:** Same race pattern as Bug 3 — the reduction partition zeroes dV
TMEM for the next outer iteration while the computation partition is still
reading dV. The TTGIR-level guard channel barrier wiring is correct for both
dk and dv. The error is **downstream of TTGIR** — in token/barrier lowering
or TMEM physical allocation.

**Analysis:** The autoWS compiler generates redundant cross-partition TMEM
zeroing (`tmem_store dense<0.0>`) that creates an unresolvable race condition.
TLX relies entirely on the MMA's `useC=false` flag on the first inner loop
iteration to zero the accumulator, avoiding the race entirely.

Confirmed via `TRITON_KERNEL_OVERRIDE`: removing the two `tmem_store` zeroing
instructions from the reduction partition while keeping all barrier
waits/arrives intact produces **ALL PASS** with 0.0 error.

**Remaining hypotheses:**
1. **Token/barrier lowering bug** (`WSLowerToken.cpp`): The guard token's
   lowering may produce incorrect barrier semantics for dv.
2. **TMEM allocation collision**: Physical TMEM column assignments may overlap
   under high SM occupancy (>1 tile per CTA).
3. **Async MMA pipeline ordering**: The dV MMA's completion may be reordered
   relative to the guard channel arrive.

## Code Locations

| Function | File | Purpose |
|----------|------|---------|
| `insertAsyncComm` | `WSCodePartition.cpp` | Main sync insertion (~950 lines) |
| `desyncTCGen5MMAOp` | `WSCodePartition.cpp` | Make MMA async with barriers |
| `createTokenPost` | `WSCodePartition.cpp` | Allocate tokens and barriers |
| `consumerReleaseHeuristic` | `WSCodePartition.cpp` | Find optimal consumer release point |
| `ProducerIsGen5` | `WSCodePartition.cpp` | Check if producer traces to gen5 MMA |
| `fuseTcgen05CommitBarriers` | `CodePartitionUtility.cpp` | Fuse redundant commits (see [Barrier Fusion](BarrierFusion.md)) |
| `optimizeTMALoads` | `WSLowerMem.cpp` | TMA barrier fusion (see [Barrier Fusion](BarrierFusion.md)) |
