# OperandD Channel and Barrier Creation

This document describes how data channels and synchronization barriers are
created for the **D operand** (accumulator) of `TCGen5MMAOp` during warp
specialization code partition.  It covers the full pipeline: channel discovery
in `handleOperandD`, token/barrier allocation in `createTokenPost`, and
synchronization insertion in `insertAsyncComm`.

---

## 1. Data Structures

### Channel types

| Type | Header | Used for |
|------|--------|----------|
| `TmemDataChannelPost` | `CodePartitionUtility.h:157` | Operand-D TMEM channels (post-scheduling) |
| `TmemDataChannel` | `CodePartitionUtility.h:133` | Non-operand-D TMEM channels (pre-scheduling) |

`TmemDataChannelPost` carries:
- `isOperandD = true` — flags this as an accumulator channel
- `allocOp` — the `ttng.tmem_alloc` that backs the TMEM buffer
- Inherits `channelKind = DataChannelKind::TMEMPost`

### CommChannel

```cpp
struct CommChannel {                           // CodePartitionUtility.h:120
    DenseMap<int, Value> tokens;               // task-id → token (nvws.create_token)
    std::optional<Value>  producerBarrier;     // barrier for TMA / gen5 producer
    DenseMap<int, Value>  consumerBarriers;    // task-id → barrier for gen5 consumer
};
```

A single `CommChannel` is shared by all channels in the same
`channelsGroupedByConsumers` group, and optionally by all channels in the
same reuse group.

---

## 2. Channel Creation — `handleOperandD`

**File:** `CodePartitionUtility.cpp:2225`
**Entry:** called from `createChannelPost` when a `tmem_alloc` is identified
as the D operand of a `TCGen5MMAOp` (i.e. `mmaOp.getD() == tmemAllocOp`).

### Algorithm

`handleOperandD` walks the `scf.for` loop body in **program order**, tracking
a sliding window of producers (`currentProds`).  Each TMEM user is classified:

| Op type | Action |
|---------|--------|
| `TMEMStoreOp` | Clears `currentProds`, becomes new sole producer |
| `TCGen5MMAOp` (same as `mmaOp`) | Both consumer (of `currentProds`) **and** producer.  Creates channel `currentProds → mmaOp`, then sets `currentProds = [mmaOp]` |
| `TCGen5MMAOp` (different MMA) | Consumer only (reads the TMEM as an operand other than D).  Creates channel `currentProds → this MMA` |
| `TMEMLoadOp` | Consumer only.  Creates channel `currentProds → tmem_load` |

A channel is created only when `needsChannel(producerTaskId, consumerIds)`
returns true — i.e. the producer and consumer are in **different partitions**.

### Pre-loop producers

Before iterating the loop body, `handleOperandD` scans all users of the
`tmem_alloc` for a `TMEMStoreOp` outside the `scf.for`.  If found (e.g. an
initialisation store before the loop), it seeds `currentProds` with that store.

### Deferred (back-edge) channels

When a `TMEMLoadOp` appears **before** any producer inside the loop body
(i.e. `currentProds` is empty), it is recorded in `channelsToBeUpdate`.
After the loop-body scan completes, these deferred channels are patched:
their producer is set to the last entry in `currentProds` (the last
producer in program order), creating a **back-edge** channel.

### Post-loop consumers

After the loop body, any remaining users of the `tmem_alloc` outside the
`scf.for` (e.g. a `TMEMLoadOp` after the loop) are paired with the final
`currentProds` to create forward channels.

### Concrete example — FA BWD dk

```
Loop body (merge_epilogue):
  tmem_store 0 → dk   (task 0, reduction)     ← zeros accumulator
  tc_gen5_mma → dk     (task 1, gemm)          ← inner loop, accumulates dk
  tmem_load dk         (task 3, computation)    ← reads result

Channels created:
  Channel A (id=N):   tmem_store(task 0) → gen5_mma(task 1)   "zero → accumulate"
  Channel B (id=N+1): gen5_mma(task 1)   → tmem_load(task 3)  "accumulate → read"
```

Both are `TmemDataChannelPost` with `isOperandD = true` and share the same
`allocOp` (the `tmem_alloc` for dk).

**Important:** No back-edge channel is created from `tmem_load → tmem_store`.
The loop-carried dependency "tmem_load must finish before tmem_store zeros in
the next iteration" is handled separately during barrier insertion (see
§4.4).

---

## 3. Token / Barrier Allocation — `createTokenPost`

**File:** `WSCodePartition.cpp:1143`

For each channel (or channel group), `createTokenPost` allocates the
`CommChannel` contents: tokens, `producerBarrier`, and `consumerBarriers`.

### Decision tree per channel

```
producerOp = channel->getSrcOp()
consumerOp = actual consumer (resolved via getActualConsumers)

1. producerBarrier
   ├─ Producer is gen5 MMA?  → producerBarrier = createBarrierAlloc(numBuffers)
   └─ Producer is TMA load?  → producerBarrier = createBarrierAlloc(numBuffers)
   (Otherwise producerBarrier stays empty.)

2. For each consumer task ID:
   a. Resolve the actual consumer op (via getActualConsumers).
   b. useGen5Barrier = ALL actual consumers are TCGen5MMAOp?
   c. Token:
      ├─ hasProdBar AND useGen5Barrier → no token needed (fully inline)
      └─ otherwise → tokens[taskId] = CreateTokenOp(numBuffers, tokenLoadType)
   d. consumerBarriers:
      ├─ useGen5Barrier → consumerBarriers[taskId] = createBarrierAlloc(numBuffers)
      └─ otherwise → (empty)
```

### Applied to FA BWD dk

**Channel A** (tmem_store → gen5 MMA):
```
producerOp = tmem_store          → NOT gen5, NOT TMA
                                 → producerBarrier = ∅
                                   Wait — producerBarrier IS set because
                                   ProducerIsGen5() traces the tmem_store's
                                   dst to the tmem_alloc, finds the gen5 MMA
                                   with matching D, and returns truthy.
                                 → producerBarrier = createBarrierAlloc(...)  ✓

consumerOp = gen5 MMA (task 1)   → useGen5Barrier = true
                                 → consumerBarriers[task1] = createBarrierAlloc(...)
                                 → tokens[task1] = CreateTokenOp(...)
                                   (created because !hasProdBar || !useGen5Barrier
                                    evaluates: hasProdBar depends on whether
                                    gen5 is the producer — here it is not the
                                    direct producer, so the token IS created)
```

Result: `{producerBarrier=bar_p, consumerBarriers={task1: bar_A}, tokens={task1: tok_A}}`

**Channel B** (gen5 MMA → tmem_load):
```
producerOp = gen5 MMA            → IS gen5
                                 → producerBarrier = createBarrierAlloc(...)  ✓

consumerOp = tmem_load (task 3)  → NOT gen5 → useGen5Barrier = false
                                 → consumerBarriers = ∅
                                 → tokens[task3] = CreateTokenOp(...)
```

Result: `{producerBarrier=bar_B, consumerBarriers={}, tokens={task3: tok_B}}`

---

## 4. Barrier / Sync Insertion — `insertAsyncComm`

**File:** `WSCodePartition.cpp:2297`

`insertAsyncComm` iterates over all channels in dependency order and inserts
the synchronization primitives.  TMEM channels (`TMEMPost`) are processed
**after** SMEM channels.

### 4.1 Channel B (gen5 MMA → tmem_load): gen5-as-producer path

Enters the block at line 2807: `if (commChannel.producerBarrier)`.

```
headProducer = gen5 MMA → dyn_cast<TCGen5MMAOp> succeeds → mmaOp is valid

desyncTCGen5MMAOp(mmaOp, bar_B, ..., headConsumer=tmem_load,
                  asProducerAcquire=false, addCompletionBarrier=true)
  → mmaOp.addCompletionBarrier(bar_B)     // tc_gen5_commit signals bar_B
  → WaitBarrierOp(bar_B, phase)           // before tmem_load (consumer_wait)
```

Then at line 3010–3130, token-based synchronisation:

```
consumerBarriers.empty() → true

ProducerAcquireOp(tok_B, bufferIdx, phase)   // before gen5 MMA
                                              // (producer must wait for buffer)
ConsumerReleaseOp(tok_B, bufferIdx)          // after tmem_load
                                              // (signals buffer free)
```

**Full Channel B sync chain:**
```
ProducerAcquire(tok_B)  →  gen5 MMA  →  tc_gen5_commit(bar_B)
                                              │
                                    WaitBarrier(bar_B)
                                              │
                                         tmem_load
                                              │
                                    ConsumerRelease(tok_B)  ←─── loops back
```

### 4.2 Channel A (tmem_store → gen5 MMA): gen5-as-consumer path

Enters the `for (auto &consumerTaskId : ...)` loop at line 2840.

```
consumerBarriers.count(task1) → true → enters gen5-consumer block

mmaOp = gen5 MMA (the consumer)
consumerBarrier = bar_A
producerAcquirePoint = headProducer = tmem_store
addCompletionBarrier = true

desyncTCGen5MMAOp(mmaOp, bar_A, ..., producerAcquirePoint=tmem_store,
                  asProducerAcquire=true, addCompletionBarrier=true)
  → mmaOp.addCompletionBarrier(bar_A)      // tc_gen5_commit signals bar_A
  → WaitBarrierOp(bar_A, phase XOR 1)      // before tmem_store
                                            // (inverted phase = producer_acquire)
```

**Channel A sync chain (before fix):**
```
WaitBarrier(bar_A, inverted)  →  tmem_store zeros dk  →  gen5 MMA accumulates dk
                                                              │
                                                    tc_gen5_commit(bar_A)
                                                              │
                                                    signals bar_A  ←─── loops back
```

Token-based ProducerAcquire/ConsumerRelease at lines 3010–3130 is
**skipped** because `consumerBarriers` is not empty.

### 4.3 Combined picture — the MMA's completion barriers

After processing both channels, the gen5 MMA has **two** completion
barriers: `bar_A` (from Channel A) and `bar_B` (from Channel B).

```
tc_gen5_commit
  ├─→ bar_A signaled → WaitBarrier(bar_A) before tmem_store satisfied
  └─→ bar_B signaled → WaitBarrier(bar_B) before tmem_load  satisfied
```

Both the tmem_store and tmem_load are unblocked **simultaneously** when the
MMA commits.  There is no ordering between them.

### 4.4 The Race — and the Fix

Because both fire at the same time, the tmem_store (which zeros dk for the
next iteration) can race with the tmem_load (which reads dk for the current
iteration's epilogue).  This is the root cause of Bug 3 in
`ws_code_partition_bwd_findings.md`.

**Fix** (implemented at line ~2920 of `WSCodePartition.cpp`):

When processing Channel A where the producer is a `TMEMStoreOp` for
operand D, the code detects the pattern and finds the **sibling Channel B**
(same `allocOp`, gen5 MMA → tmem_load).  Instead of creating a
`WaitBarrierOp(bar_A)` before the tmem_store, it:

1. **Still adds** `bar_A` as a completion barrier on the MMA
   (so `tc_gen5_commit` still signals bar_A — needed for phase tracking).
2. **Creates a new token** (`tok_consumed`) for the tmem_load → tmem_store
   dependency.
3. **Inserts `ProducerAcquireOp(tok_consumed)`** before the tmem_store —
   this blocks until `ConsumerRelease(tok_consumed)` fires.
4. **Inserts `ConsumerReleaseOp(tok_consumed)`** after Channel B's
   tmem_load consumer — signals that dk has been read and the TMEM is
   free to be zeroed.

**Fixed sync chains:**

```
Channel B (unchanged):
  ProducerAcquire(tok_B) → gen5 MMA → tc_gen5_commit(bar_B) →
  WaitBarrier(bar_B) → tmem_load → ConsumerRelease(tok_B)

Channel A (fixed):
  ProducerAcquire(tok_consumed) → tmem_store zeros dk → gen5 MMA →
  tc_gen5_commit(bar_A)

Cross-channel dependency (NEW):
  tmem_load → ConsumerRelease(tok_consumed) ──→ ProducerAcquire(tok_consumed)
                                                       │
                                                 tmem_store zeros dk  (safe!)
```

The tmem_store now waits for the tmem_load to finish reading before it
zeros the TMEM buffer.

### 4.5 FA FWD Accumulators — Same-Task Guard

FA fwd has a structurally similar operand-D lifecycle for the output
accumulator (`%acc`), but crucially the `tmem_store` and `tmem_load` are
in the **same partition** (computation), so there is no cross-partition
race.

**FA fwd acc lifecycle (inside the loop):**

```
Loop body (non-persistent):
  tmem_load %acc[token]      (task 3/5, computation)  ← read previous acc
  ... rescale acc (mulf, subf, exp2, broadcast, inline_asm) ...
  tmem_store rescaled, %acc  (task 3/5, computation)  ← write rescaled acc back
  tc_gen5_mma P, V, %acc     (task 1, gemm)           ← accumulate P*V into acc
```

**Channels created by `handleOperandD`:**

```
Channel A: tmem_store(task 3, computation) → gen5_mma(task 1, gemm)
Channel B: gen5_mma(task 1, gemm) → tmem_load(task 3, computation)  [back-edge]
```

Both channels are `TmemDataChannelPost` with `isOperandD = true`.
Channel B is a **deferred (back-edge) channel** — the `tmem_load`
appears before the `tmem_store` in program order, so it has no in-loop
producer when first encountered.

**Why the token fix must NOT fire:**

Channel A's producer is a `TMEMStoreOp` on an operand-D channel, and
the sibling Channel B has `TCGen5MMAOp` → `TMEMLoadOp` on the same
`allocOp`.  This matches all the structural conditions of the operand-D
race fix.  However:

- The `tmem_store` (computation, task 3) and `tmem_load` (computation,
  task 3) are in the **same task/partition**.
- Program order within the warp group already guarantees that the
  `tmem_load` completes before the `tmem_store` writes (they execute
  sequentially in the same warp group).
- The original `desyncTCGen5MMAOp` path creates a `WaitBarrier(bar_A)`
  before the `tmem_store` that waits for `tc_gen5_commit` — this is
  correct and sufficient.
- Applying the token-based fix creates a circular dependency:
  `ProducerAcquire(tok_consumed)` before `tmem_store` waits for
  `ConsumerRelease(tok_consumed)` after `tmem_load`, but both are in
  the same warp group and the `tmem_load` is gated on the MMA's
  `WaitBarrier(bar_B)` which in turn depends on the `tmem_store` →
  MMA → commit chain.  This causes a **deadlock**.

**Same-task guard (line ~2940 of `WSCodePartition.cpp`):**

```cpp
int storeTaskId = masterChannel->relation.first;
auto &loadTaskIds = sibCh->relation.second;
if (llvm::is_contained(loadTaskIds, storeTaskId))
  continue;
```

If the `tmem_store`'s producer task ID appears in the sibling
`tmem_load`'s consumer task IDs, the fix is skipped.  This ensures:

- **FA BWD (fires):** `storeTaskId = 0` (reduction), `loadTaskIds = {3}`
  (computation).  `0 ∉ {3}` → different tasks → token fix applied.
- **FA FWD (skipped):** `storeTaskId = 3` (computation),
  `loadTaskIds = {3}` (computation).  `3 ∈ {3}` → same task →
  `continue`, falls through to `desyncTCGen5MMAOp`.

**FA fwd summary table (per accumulator):**

| | Channel A | Channel B |
|---|---|---|
| **Producer** | tmem_store (computation, task 3) | gen5 MMA (gemm, task 1) |
| **Consumer** | gen5 MMA (gemm, task 1) | tmem_load (computation, task 3) |
| **Token fix?** | **No** — same-task guard | N/A |
| **Sync mechanism** | `WaitBarrier(bar_A)` before tmem_store (original `desyncTCGen5MMAOp`) | `WaitBarrier(bar_B)` before tmem_load + `ConsumerRelease(tok_B)` after tmem_load |

---

## 5. Summary Table — OperandD Channels (FA BWD)

For a single TMEM accumulator (e.g. dk) with the cross-partition pattern
`tmem_store(reduction) → gen5_mma(gemm) → tmem_load(computation)`:

| | Channel A | Channel B |
|---|---|---|
| **Kind** | `TMEMPost` (operand D) | `TMEMPost` (operand D) |
| **Producer** | tmem_store (reduction, task 0) | gen5 MMA (gemm, task 1) |
| **Consumer** | gen5 MMA (gemm, task 1) | tmem_load (computation, task 3) |
| **producerBarrier** | set (via `ProducerIsGen5` trace) | set (producer IS gen5) |
| **consumerBarriers** | `{task1: bar_A}` (consumer is gen5) | ∅ (consumer is tmem_load) |
| **tokens** | `{task1: tok_A}` (unused for sync) | `{task3: tok_B}` |
| **MMA completion barrier** | bar_A (via addCompletionBarrier) | bar_B (via addCompletionBarrier) |
| **Producer acquire** | `ProducerAcquire(tok_consumed)` before tmem_store *(fixed)* | `ProducerAcquire(tok_B)` before gen5 MMA |
| **Consumer release** | Implicit via gen5 inline barrier (bar_A) | `ConsumerRelease(tok_B)` after tmem_load |
| **Cross-channel** | `ConsumerRelease(tok_consumed)` after tmem_load *(new)* | — |

---

## 6. Code Locations

| Step | File | Function | Line |
|------|------|----------|------|
| Channel discovery | `CodePartitionUtility.cpp` | `handleOperandD` | 2225 |
| Channel creation helper | `CodePartitionUtility.cpp` | `createChannelsForProducers` | 813 |
| Entry point | `CodePartitionUtility.cpp` | `createChannelPost` | 2443 |
| Token/barrier alloc | `WSCodePartition.cpp` | `createTokenPost` | 1143 |
| Sync insertion | `WSCodePartition.cpp` | `insertAsyncComm` | 2297 |
| Gen5 desync helper | `WSCodePartition.cpp` | `desyncTCGen5MMAOp` | 2044 |
| Operand-D race fix | `WSCodePartition.cpp` | `insertAsyncComm` (inline) | ~2920 |
| Same-task guard | `WSCodePartition.cpp` | `insertAsyncComm` (inline) | ~2940 |
