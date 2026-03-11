# WSCodePartition BWD Persistent Kernel — Findings

## Overview

Investigation of accuracy issues in the backward persistent fused-attention
kernel (`_attn_bwd_persist`) when compiled with warp specialization flags:

```
TRITON_USE_META_PARTITION=1 TRITON_ALWAYS_COMPILE=1 TRITON_USE_META_WS=1
```

Three bugs were found.  Two are fixed; one remains open.

---

## Bug 1 — 2-Buffer Reuse Group Fires Incorrectly (NaN results)

**Status:** Fixed  
**Commit:** `20f7b3e26f` (first bad), fix in `92a456c0`  
**File:** `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSCodePartition.cpp`  
**Lit test:** `test/Hopper/WarpSpecialization/reuse_group_2buffer_fwd.mlir`

### Symptom
Running `test_tlx_bwd_from_fused_attention.py` with WS flags produced **NaN** for
all gradients (dQ, dK, dV).

### Root Cause
The 2-buffer reuse group logic (lines 2706-2740 of `WSCodePartition.cpp`)
moved `producer_acquire` for a late channel before an early channel's
producer **even when the late channel's consumer was in a different control
block** than the early channel's producer.  In the BWD kernel this corrupted
the MMA pipeline ordering, leading to reads of uninitialised TMEM.

### Fix
Added a guard condition at line 2724-2726:

```cpp
if (earlyProducer->getBlock() == headProducer->getBlock() &&
    lateConsumer->getBlock() == earlyProducer->getBlock() &&  // NEW
    appearsBefore(earlyProducer, headProducer)) {
```

This ensures the 2-buffer reuse group optimisation only fires when the late
consumer lives in the **same block** as the early producer, which is the only
configuration where the hoisted `producer_acquire` is safe.

### Bisection
First bad commit: `20f7b3e26f` ("Lit test + integration to handle barrier
for buffer reuse").  The reuse-group-2-buffer feature was introduced here.

---

## Bug 2 — TMA Store Column Offset (dQ, dV, dK all wrong)

**Status:** Fixed  
**Commit:** `b56dee56b4d9` ("fix accuracy issue due to kernel authoring")  
**File:** `python/tutorials/fused-attention-ws-device-tma.py`

### Symptom
After fixing Bug 1, dQ and dV produced large errors; dK was worse.

### Root Cause
With `EPILOGUE_SUBTILE = 4`, the 128-wide output tensors (dV, dK, dQ) are
split into four 32-wide chunks by `_split_n()` and stored via four separate
TMA store operations.  **All four stores used hardcoded column offset `0`**,
so every chunk overwrote the first 32 columns.

### Fix (3 locations)

**`_attn_bwd_core` — dV store (line ~912):**
```python
slice_size = HEAD_DIM // EPILOGUE_SUBTILE
dvs = _split_n(dv, EPILOGUE_SUBTILE)
for slice_id in tl.static_range(0, EPILOGUE_SUBTILE):
    dvN = dvs[slice_id]
    desc_dv.store(
        [(off_bh + start_n).to(tl.int32), slice_id * slice_size],  # was 0
        dvN.to(dtype),
    )
```

**`_attn_bwd_core` — dK store (line ~920):**
```python
dks = _split_n(dk, EPILOGUE_SUBTILE)
for slice_id in tl.static_range(0, EPILOGUE_SUBTILE):
    dkN = dks[slice_id] * sm_scale
    desc_dk.store(
        [(off_bh + start_n).to(tl.int32), slice_id * slice_size],  # was 0
        dkN.to(dtype),
    )
```

**`_attn_bwd_dkdv_inner` — dQ atomic_add (line ~675):**
```python
slice_size: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
for slice_id in tl.static_range(0, EPILOGUE_SUBTILE):
    dqN = dqs[slice_id] * LN2
    desc_dq.atomic_add(
        [(off_bh + curr_m).to(tl.int32), slice_id * slice_size],  # was 0
        dqN,
    )
```

After this fix **dQ and dV pass** (error < 0.01).  **dK still fails.**

---

## Bug 3 — dK Accuracy (race condition — reduction zeroes dk TMEM before computation reads it)

**Status:** Open — root cause fully traced in TTGIR, compiler fix needed  
**Error:** `max|err| = 2.544922e+00` (Original vs Reference)  
**TLX kernel:** Passes (`2.929688e-03`)

### Key Observations

| Property | Value |
|---|---|
| Error magnitude | 2.544922 (deterministic max, but pattern varies per run) |
| Affected gradient | dK only (dQ and dV pass) |
| TLX (blackwell pipelined) | Passes — confirms reference is correct |
| EPILOGUE_SUBTILE=1 | Still fails → not a split issue |
| Non-META_PARTITION scheduler | Still fails → not META_PARTITION-specific |
| fuseTcgen05CommitBarriers disabled | Still fails → barrier fusion is NOT the cause |

### Debug Analysis Results

Python-level element-wise analysis revealed the true nature of the error:

**Non-zero elements are PERFECT** (max|err| = 2.929e-3, ratio mean = 0.99998):
```
  Non-zero element accuracy (5530567 / 16777216 elements):
    max|err| among non-zero orig = 2.929688e-03
    ratio orig/ref — mean=0.99998349  median=1.00000000  std=0.02263965
```

**~67% of elements are ZERO** (dk not written, not a scaling error):
```
  Per N-tile zero fraction: 0.65 – 0.74 per tile (uniform across tiles)
  Per batch-head: 23 all-zero, 86 partial, 19 fully-correct (out of 128 heads)
```

**Zero pattern is RANDOM and varies between runs** (non-deterministic):
```
  Per-head N-tile zero pattern (Z=zeros, .=ok):
    head [ 0, 0]: ZZZZZZZZ    ← entire head zero
    head [ 1, 3]: pZ.Zpppp    ← partial
    head [ 1, 4]: p.p.p..p    ← mostly ok
    head [ 0, 9]: Zp.ZZZZZ    ← partial
```

### Root Cause (CONFIRMED via TTGIR barrier trace)

The bug is a **race condition between the reduction partition (task 0) and the
computation partition (task 3)**:

1. The gemm partition (task 1) accumulates dk in TMEM via async MMA ops
2. After the inner loop, `tc_gen5_commit` signals dk ready via **two barriers**:
   `%8` (= `%arg39`) for the computation partition, and `%10` (= `%arg40`)
   for the reduction partition
3. **BOTH** the computation and reduction see dk-committed simultaneously:
   - **Computation** waits on `%arg39` (= `%8`) → starts reading dk via `tmem_load`
   - **Reduction** waits on `%arg40` (= `%10`) → **immediately zeroes dk TMEM**
     via `tmem_store %cst_0`
4. The reduction's dk zeroing **races** with the computation's dk reading

The effect: dk values ARE computed correctly by the gemm, but are
**overwritten with zeros by the reduction partition** before the computation
partition finishes reading them.  Whether a particular tile's dk is overwritten
depends on timing, producing the random zero pattern observed.

### Why dV Works But dK Doesn't

Both dv and dk have the **same race pattern** (reduction waits on committed,
not consumed). But the race outcome differs due to **execution order asymmetry**:

| | Reduction order | Computation order |
|---|---|---|
| dk | Zeroed **FIRST** (line 248) | Read **SECOND** (line 847+) |
| dv | Zeroed **SECOND** (line 254) | Read **FIRST** (line 812+) |

The computation reads dv FIRST (before dk), and the reduction zeroes dk FIRST
(before dv). This timing asymmetry means:
- **dv:** Computation reads dv before reduction gets to zeroing dv → dv is safe
- **dk:** Reduction zeroes dk before computation gets to reading dk → dk is corrupted

### Full TTGIR Barrier Trace

Generated from fresh TTGIR at:
`/tmp/triton_ttgir_trace/.../attn_bwd_persist.ttgir` (1073 lines)

#### Warp Specialise Operand → Barrier Mapping

All partitions share the same block arguments (`%argN`) from the
`ttg.warp_specialize` operands. Key barrier allocations:

| Operand idx | WS operand | %argN | Role | Producer→Consumer |
|-------------|-----------|-------|------|-------------------|
| 23 | `%8` | `%arg39` | dk COMMITTED #1 | gemm→computation |
| 24 | `%10` | `%arg40` | dk COMMITTED #2 | gemm→**reduction** |
| 25 | `%12` | `%arg41` | dv COMMITTED #1 | gemm→computation |
| 26 | `%14` | `%arg42` | dv COMMITTED #2 | gemm→**reduction** |
| 46 | `%56` | `%arg62` | dv ZEROED | reduction→gemm |
| 48 | `%64` | `%arg64` | dk ZEROED | reduction→gemm |
| 47 | `%61` | `%arg63` | dv CONSUMED | computation→gemm |
| 49 | `%69` | `%arg65` | dk CONSUMED | computation→gemm |

#### dk Lifecycle (BROKEN)

```
Reduction (task 0) — OUTER LOOP, runs FIRST in epilogue:
  L247: wait_barrier  memdesc_index(%10)[0], phase   ← dk COMMITTED #2
  L248: tmem_store 0.0 → dk TMEM                     ← ZEROES dk! ⚠️
  L250: arrive_barrier memdesc_index(%64)[0]          ← dk ZEROED

Gemm (task 1) — OUTER LOOP PROLOGUE:
  L352: wait_barrier memdesc_index(%arg64)[0], phase  ← dk ZEROED
  L354: wait_barrier memdesc_index(%arg65)[0], phase  ← dk CONSUMED (from prev)
                     ↓
  Inner loop:
  L413: wait_barrier memdesc_index(%arg58)[0], phase  ← dsT READY (not dk!)
  L415: tc_gen5_mma dsT, q → dk TMEM                 ← accumulate dk
                     ↓
  L481: tc_gen5_commit %arg39                         ← dk COMMITTED #1 (→compute)
  L482: tc_gen5_commit %arg40                         ← dk COMMITTED #2 (→reduce)

Computation (task 3) — OUTER LOOP EPILOGUE:
  L847: wait_barrier memdesc_index(%arg39)[0], phase  ← dk COMMITTED #1
  L855-858: tmem_load dk (4 × 128×32 chunks)          ← read dk from TMEM
  L859: arrive_barrier memdesc_index(%arg65)[0]        ← dk CONSUMED
  L860+: multiply by sm_scale, truncf, TMA store      ← dk epilogue
```

**The race:** Steps L247-248 (reduction zeroes dk) and L847-858 (computation
reads dk) both fire after the gemm commits dk. There is **NO ordering**
between them — the reduction should wait for dk CONSUMED (`%69`/`%arg65`)
before zeroing dk, but instead waits on dk COMMITTED (`%10`/`%arg40`).

#### dv Lifecycle (same pattern, but survives due to timing)

```
Reduction (task 0):
  L253: wait_barrier  memdesc_index(%14)[0], phase   ← dv COMMITTED #2
  L254: tmem_store 0.0 → dv TMEM                     ← ZEROES dv
  L256: arrive_barrier memdesc_index(%56)[0]          ← dv ZEROED

Computation (task 3):
  L812: wait_barrier memdesc_index(%arg41)[0], phase  ← dv COMMITTED #1
  L820-823: tmem_load dv (4 × 128×32 chunks)          ← read dv
  L824: arrive_barrier memdesc_index(%arg63)[0]        ← dv CONSUMED
```

dv has the SAME incorrect barrier dependency (reduction waits on committed,
not consumed), but survives because computation reads dv BEFORE dk, and
reduction zeroes dk BEFORE dv. With more TMEM channels or different scheduling,
dv could also be corrupted.

### Code-Level Root Cause — Barrier Insertion Trace

The wrong barrier is passed to `desyncTCGen5MMAOp` at line 2905 of
`WSCodePartition.cpp`. Here is the precise trace showing how it happens.

#### Step 1: `handleOperandD` creates two channels per TMEM accumulator

For dk TMEM (`tmemAllocOp`), `handleOperandD` (CodePartitionUtility.cpp:2224)
walks the loop body in program order and creates channels between each
producer→consumer pair:

```
Loop body order (outer loop, with merge_epilogue):
  tmem_store 0 → dk   (task 0, reduction)    ← zeros accumulator
  tc_gen5_mma → dk     (task 1, gemm)         ← inner loop, accumulates dk
  tmem_load dk         (task 3, computation)   ← reads result

Channels created:
  Channel A: tmem_store(task 0) → gen5_mma(task 1)   "zero → accumulate"
  Channel B: gen5_mma(task 1) → tmem_load(task 3)    "accumulate → read"
```

**No back-edge channel is created from tmem_load → tmem_store.**
The dependency "tmem_load must finish before tmem_store can zero" is missing.

#### Step 2: `createToken` sets up CommChannel for each channel

**Channel A** (tmem_store → gen5 MMA):
```
Line 861: ProducerIsGen5(tmem_store) → follows dst → allocOp → finds
          gen5 MMA with matching D → returns MMA → truthy
          → commChannel.producerBarrier = createBarrierAlloc(...)     ← SET

Line 872: useGen5Barrier = isa<TCGen5MMAOp>(consumerOp=gen5 MMA) → true
Line 921: → commChannel.consumerBarriers[task1] = createBarrierAlloc(...)
Line 917: → commChannel.tokens[task1] = CreateTokenOp(...)
```
Result: `{producerBarrier=✓, consumerBarriers={task1: bar_A}, tokens={task1: tok_A}}`

**Channel B** (gen5 MMA → tmem_load):
```
Line 861: ProducerIsGen5(gen5 MMA) → true
          → commChannel.producerBarrier = createBarrierAlloc(...)     ← SET

Line 872: useGen5Barrier = isa<TCGen5MMAOp>(consumerOp=tmem_load) → false
Line 917: → commChannel.tokens[task3] = CreateTokenOp(...)
```
Result: `{producerBarrier=✓, consumerBarriers={}, tokens={task3: tok_B}}`

#### Step 3: `insertAsyncComm` processes Channel A — THE BUG

For Channel A (tmem_store → gen5 MMA):

```
Line 2807: commChannel.producerBarrier IS set
Line 2810: mmaOp = dyn_cast<TCGen5MMAOp>(headProducer=tmem_store) → NULL
           → gen5-as-producer block (2811-2837) SKIPPED
             (headProducer is tmem_store, not gen5 MMA)

Lines 2840-2910: consumer is gen5 MMA, consumerBarriers has task1
  Line 2880: consumerBarrier = commChannel.consumerBarriers[task1]  ← bar_A
  Line 2884: producerAcquirePoint = headProducer = tmem_store

  Line 2905:                                                     ⚠️ BUG HERE
    desyncTCGen5MMAOp(builder, mmaOp,
                      consumerBarrier,        ← bar_A (gen5 inline barrier)
                      bufferIdx, phase, ...,
                      producerAcquirePoint,   ← tmem_store
                      ...,
                      asProducerAcquire=true, ← creates wait BEFORE tmem_store
                      ...,
                      addCompletionBarrier);  ← adds bar_A to MMA's tc_gen5_commit
```

Inside `desyncTCGen5MMAOp` (line 2044):
```
Line 2062: mmaOp.addCompletionBarrier(bar_A, pred)
           → tc_gen5_commit will signal bar_A

Line 2072: producerBarrier = getBarrierForPipelineStage(bar_A, bufferIdx)
Line 2081: phase = inPhase XOR 1     (asProducerAcquire=true → invert phase)
Line 2089: WaitBarrierOp(producerBarrier, phase)  ← inserted BEFORE tmem_store
```

**Result:** A `wait_barrier` is inserted before `tmem_store` that waits on
`bar_A` — Channel A's consumerBarrier. This barrier is signaled by the MMA's
`tc_gen5_commit`. As soon as the MMA commits, the tmem_store is unblocked
and zeroes dk TMEM.

#### Step 4: `insertAsyncComm` processes Channel B — creates consumer release

For Channel B (gen5 MMA → tmem_load):
```
Line 2807: commChannel.producerBarrier IS set
Line 2810: mmaOp = dyn_cast<TCGen5MMAOp>(headProducer=gen5 MMA) → valid
Line 2832: desyncTCGen5MMAOp(mmaOp, *producerBarrier, ...,
                             headConsumer=tmem_load,
                             asProducerAcquire=false, ...)
           → adds producerBarrier (bar_B) to MMA's tc_gen5_commit
           → creates wait_barrier before tmem_load on bar_B

Line 2915: consumerBarriers.empty() → true
           → ProducerAcquireOp before gen5 MMA using tok_B

Line 3020: consumerBarriers.empty() → true
           → ConsumerReleaseOp after tmem_load using tok_B
```

**Result:** Channel B creates a proper token cycle:
  ProducerAcquire(tok_B) → MMA → tc_gen5_commit(bar_B) →
  wait_barrier(bar_B) → tmem_load → ConsumerRelease(tok_B)

The ConsumerRelease(tok_B) after tmem_load signals "dk consumed", but
**nothing connects this to Channel A's tmem_store wait**.

#### Step 5: The Race in Summary

After tc_gen5_commit fires, it signals BOTH bar_A and bar_B simultaneously:
```
tc_gen5_commit
  ├─→ bar_A signaled → tmem_store wait satisfied → ZEROS dk TMEM  ⚠️
  └─→ bar_B signaled → tmem_load wait satisfied  → READS dk TMEM  ⚠️
                                                                   RACE!
```

Channel B's ConsumerRelease (tok_B, after tmem_load) feeds back to
Channel B's ProducerAcquire (tok_B, before gen5 MMA). It does NOT
feed back to Channel A's tmem_store. So the tmem_store zeroes dk
while tmem_load is still reading it.

### Required Fix

The tmem_store's producer_acquire must wait for Channel B's consumer release
(tmem_load done), not Channel A's consumer release (MMA done):

```
CURRENT (BROKEN):
  tc_gen5_commit ──→ bar_A ──→ tmem_store zeros dk  (races with tmem_load)

CORRECT:
  tc_gen5_commit ──→ bar_B ──→ tmem_load reads dk
                                    │
                          ConsumerRelease(tok_B)
                                    │
                                    ▼
                          tmem_store zeros dk   (safe, tmem_load done)
```

The fix site is at **line 2905 in WSCodePartition.cpp** where Channel A's
`consumerBarrier` (bar_A) is passed to `desyncTCGen5MMAOp`. Instead, the
tmem_store's wait should use Channel B's consumer release token (tok_B).

This could be achieved by:
1. In `handleOperandD`: create a back-edge channel from tmem_load(task 3) →
   tmem_store(task 0) so the dependency is explicitly modeled
2. OR in `insertAsyncComm`: when processing a Channel A whose producer is
   tmem_store for operand D, find the sibling Channel B (same TMEM, gen5 MMA →
   tmem_load) and use Channel B's consumer release instead of Channel A's
   consumerBarrier

### Why `fuseTcgen05CommitBarriers` Is Not The Root Cause

Even though `fuseTcgen05CommitBarriers` deletes dk's `wait_barrier` (merging
it with dv's because they share `async_task_id = 3`), **disabling it entirely
produces the exact same error**.  This is because:

- `tc_gen5_commit` flushes ALL preceding async MMAs.  So dv's `wait_barrier`
  on the single fused barrier is sufficient to guarantee both dv AND dk TMEM
  writes are complete for the **current** iteration.
- The problem is not "is dk ready?" but "is dk still there?" — the **reduction
  partition zeroes dk TMEM** before the computation partition reads it.

---

## Compiled TTGIR Structure (for reference)

The BWD persistent kernel is partitioned into 4 tasks:

| Task | Partition | Stage | Role |
|------|-----------|-------|------|
| 0 | reduction | 0 | TMEM zeroing, dQ reduce-store |
| 1 | gemm | 1 | Async MMA for dv, dk, dq, qkT, dpT |
| 2 | load | 0 | TMA loads (q, k, v, do) |
| 3 | computation | 0 | Softmax, dk/dv epilogue stores |

### TMEM Buffer Assignments

| Buffer | buffer.id | shareGroup | Content |
|--------|-----------|------------|---------|
| dpT | 8 | 2 | Dot product for backward |
| qkT | 7 | 0 | Query-key scores |
| dv | 6 | 1 | Gradient for V |
| dk | 5 | 3 | Gradient for K |

### Partition Regions in TTGIR

| Region | Lines | Task ID | Type |
|--------|-------|---------|------|
| default | 208–309 | 0 | reduction |
| partition0 | 310–490 | 1 | gemm |
| (unnamed) | 491–604 | 2 | load |
| (unnamed) | 605–888 | 3 | computation |

---

## Next Steps

- [ ] Fix `insertAsyncComm` in `WSCodePartition.cpp` to make reduction's
      TMEM zeroing wait on the CONSUMED barrier (from computation) instead of
      the COMMITTED barrier (from gemm) for TMEM channels in `merge_epilogue`
- [ ] Verify dv also gets the fix (currently survives by timing, not correctness)
- [ ] Add a lit test for this barrier ordering invariant

---

## Files Modified

| File | Change | Commit |
|------|--------|--------|
| `WSCodePartition.cpp:2724` | Added `lateConsumer->getBlock() == earlyProducer->getBlock()` guard | `92a456c0` |
| `fused-attention-ws-device-tma.py` (3 locations) | Fixed TMA store column offsets from `0` to `slice_id * slice_size` | `b56dee56` |
| `reuse_group_2buffer_fwd.mlir` | New lit test for FWD 2-buffer reuse group regression | `92a456c0` |
| `test_tlx_bwd_from_fused_attention.py` | Added `_debug_dk()` diagnostic function | (uncommitted) |

## Files Examined (read-only)

- `CodePartitionUtility.cpp` — `fuseTcgen05CommitBarriers`, `updateSubgroup`, `mergeSubgroups`, `collectCommitGroup`
- `CodePartitionUtility.h` — `CommChannel` struct, `DataChannelKind` enum
- `test_tlx_bwd_from_fused_attention.py` — test harness comparing Original vs TLX vs PyTorch reference
- `_attn_bwd_persist.ttgir` (compiled, 1073 lines) — full barrier and tmem_load trace across all 4 partitions
