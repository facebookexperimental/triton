# Operand D Channel-Loop Fix — Per-Phase Desired Outputs

## Status

**Bug**: D91359924 (commit `d35e994174`) added a pre-loop scan in
`handleOperandD` that unconditionally seeds `currentProds` with any
out-of-loop `TMEMStoreOp` user of the alloc. When the loop body has the
**channel-loop pattern** (in-loop `tmem_load` appears in program order
*before* the in-loop `tmem_store` / `gen5_mma`), the pre-loop scan
short-circuits the deferred-channel mechanism that 96870473d's barrier-
insertion logic in `insertAsyncComm` depends on. As a result, the
**backward channel** (`gen5 → tmem_load`) is silently dropped.

This doc describes the **desired output** at each phase of
`doCodePartition` for the canonical channel-loop pattern, against
which the implementation can be checked.

## Canonical Input — Channel-Loop Pattern

Operand D accumulator with all three triggering conditions:
1. Out-of-loop `tmem_store` initializing the accumulator (single task,
   e.g. correction partition T_corr).
2. In-loop body order: `tmem_load (T_corr)` → `tmem_store (T_corr)` →
   `gen5_mma (T_mma)`.
3. `tmem_load` and `tmem_store` are in the SAME partition (T_corr) —
   different from the gen5 partition (T_mma).

```
%acc = ttng.tmem_alloc                                 // op0
%init = ttng.tmem_store %zeros, %acc          T_corr   // op1 (outside loop)
scf.for ... iter_args(%tok = %init) {
  %v, %tok2 = ttng.tmem_load %acc[%tok]       T_corr   // op2 (load before store!)
  ... rescale v ...
  %tok3 = ttng.tmem_store %rescaled, %acc     T_corr   // op3
  %tok4 = ttng.tc_gen5_mma A, B, %acc[%tok3]  T_mma    // op4
  scf.yield %tok4
}
%post, _ = ttng.tmem_load %acc[%tok_final]    T_corr   // op5 (post-loop, optional)
```

**Naming for the rest of the doc:**
- `init` = out-of-loop tmem_store (op1)
- `load_in` = in-loop tmem_load (op2)
- `store_in` = in-loop tmem_store (op3)
- `mma` = in-loop tc_gen5_mma (op4)
- `load_post` = post-loop tmem_load (op5, may be absent)
- T_corr = correction partition (e.g. 0 or 3)
- T_mma = MMA partition (e.g. 1)

---

## Phase 1 — Channel Discovery (`handleOperandD`)

### File

`third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/CodePartitionUtility.cpp`
(function `handleOperandD`).

### Desired Output (channels created)

For the canonical pattern, `handleOperandD` must produce these channels
on the operand-D `tmem_alloc`:

| ID | Producer | Producer Task | Consumer | Consumer Task | Role | Notes |
|----|----------|---------------|----------|---------------|------|-------|
| F | `store_in` | T_corr | `mma` | T_mma | **Forward** (in-loop) | tmem.start on store_in, tmem.end on mma |
| B | `mma` | T_mma | `load_in` | T_corr | **Backward** (back-edge) | tmem.start on mma, tmem.end on load_in. Created via deferred (`channelsToBeUpdate`) path |
| W (optional) | `init` | T_corr | `mma` | T_mma | First-iter init | Only when `init` is in the same block as `mma`'s parent. Carries the very-first-iteration semantic. May be merged with F via `currentProds` (multi-producer) |
| P (optional) | `mma` | T_mma | `load_post` | T_corr | Post-loop forward | Created by the "consumers outside ForOp" loop |

### Algorithm Outline

```
forOp = mmaOp.getParentOfType<scf::ForOp>()

// Step A: classify the body to detect channel-loop pattern.
hasBodyChannelLoop = bodyHasInLoopLoadBeforeStoreAndMma(forOp, tmemAllocOp)

// Step B: pre-loop init store.
initStore = findOutOfLoopTmemStoreUser(tmemAllocOp, forOp)

// Step C: seed currentProds.
if hasBodyChannelLoop:
  // Do NOT seed currentProds with initStore — leaving it empty allows
  // the body's first tmem_load to take the deferred (back-edge) path.
  currentProds = []
  // initStore will be wired separately via channel W (see Step F).
else:
  // Original d35e9941 behavior is correct here (no in-body load before
  // in-body store/gen5).
  if initStore:
    currentProds = [initStore]
    handledUsers.insert(initStore)

// Step D: walk body in program order (same as today).
//  - tmem_load with currentProds empty   -> defer (channelsToBeUpdate),
//                                            channel B in flight.
//  - tmem_store                          -> currentProds = [store]
//  - mmaOp                               -> create channel F
//                                            (currentProds -> mma),
//                                            currentProds = [mma]

// Step E: post body, patch deferred channels.
//  - For each idx in channelsToBeUpdate:
//      lastProd = currentProds.back()         // = mma
//      channels[idx].producer = task(lastProd) // T_mma
//      Mark tmem.start on lastProd
//      // *** Important: also bump numChannelsCreated and update
//      //     firstProducer/lastConsumer so the wrap-around block
//      //     at the end of handleOperandD treats the back-edge as
//      //     a real channel.
//      numChannelsCreated++
//      if !firstProducer: firstProducer = lastProd
//      lastConsumer = channels[idx].dstOp

// Step F: wire init store as channel W.
if hasBodyChannelLoop and initStore:
  // initStore -> first-iter mma. Create a dedicated channel so the
  // first iteration has a producer for operand D. The token must be
  // distinct from F (different src op).
  push channel W: producer = init (T_corr),
                  consumer = mma (T_mma),
                  isOperandD = true,
                  isFirstIterInit = true   // new flag (optional; helps
                                            // downstream emit acquire
                                            // outside the loop)

// Step G: post-loop consumers (unchanged).
//  - For each unhandled tmem_load user outside the loop:
//      channel P: producer = mma, consumer = load_post.
```

### Diagnostic to Add

```
LDBG("[handleOperandD] body has channel-loop pattern: "
     << hasBodyChannelLoop
     << " ; init store: " << (initStore ? "yes" : "no"));
```

This makes regressions visible in `t.dump`.

### Verification

`dumpChannelsForOperandD` after `handleOperandD` should show 3 channels
(F, B, W) when init store is present, or 2 channels (F, B) when not. The
existing `=== Channels for OperandD ===` debug already prints producer /
consumer / src / dst — verify that channel B has
`SrcOp = mma, DstOp = load_in`.

---

## Phase 2 — Region & Reuse-Group Setup

### Files

- `WSCodePartition.cpp::doCodePartition` (steps "find top-level
  ops", `collectRegionsWithChannels`, `bufferIdToChannels`,
  `mergedChannels`).

### Desired Output

- `regionsWithChannels` includes the inner `scf.for` (because all of F,
  B, and W reference ops inside it or its parent).
- `bufferIdToChannels[buffer.id(of tmem_alloc)]` contains all 3 channels
  (F, B, W). The check `allSameAlloc` skips creating a `ReuseGroup`
  (these are lifecycle phases of one buffer, not a reuse group).
- `channelsGroupedByConsumers` retains all 3 channels separately —
  `haveMatchingConsumers` returns true for `TMEMAlloc`, but the dst-op
  equality check fails (F's dst = mma, B's dst = load_in, W's dst =
  mma), so F and W can be considered for merging only if their consumer
  ops match exactly. Recommended: do NOT merge F and W — they have
  different src ops and merging them prevents the back-edge sync
  insertion.

### Verification

```
LDBG("bufferIdToChannels[" << buf_id << "] size = "
     << bufferIdToChannels[buf_id].size());
// Expect 3
```

---

## Phase 3 — Buffer Creation (`createBufferForAllocs`)

### File

`WSCodePartition.cpp::createBufferForAllocs`.

### Desired Output

- All 3 channels (F, B, W) share the SAME `tmem_alloc` buffer (operand
  D is in-place; no multi-buffering across the channel-loop pattern).
- `bufferMap[F] == bufferMap[B] == bufferMap[W]` — they all map to the
  single `ttng.tmem_alloc` op (no per-channel buffer).
- `numBuffers == 1` for these channels (operand D in the channel-loop
  pattern cannot be multi-buffered: load and store must hit the same
  TMEM cell within an iteration).

### Verification

After `createBufferForAllocs`:
```
LDBG("bufferMap[F] = " << bufferMap[F]);
LDBG("bufferMap[B] = " << bufferMap[B]);
LDBG("bufferMap[W] = " << bufferMap[W]);
// Expect all three equal.
```

---

## Phase 4 — Token / Barrier Allocation (`createToken`)

### File

`WSCodePartition.cpp::createToken`.

### Desired Output (per channel)

| Channel | producerBarrier | consumerBarriers | tokens |
|---------|-----------------|------------------|--------|
| F (`store_in → mma`) | set (via `ProducerIsGen5` trace through alloc → mmaOp) | `{T_mma: bar_F}` (consumer is gen5) | `{T_mma: tok_F}` |
| B (`mma → load_in`) | set (producer IS gen5) | `{}` (consumer is tmem_load) | `{T_corr: tok_B}` |
| W (`init → mma`) | set (via `ProducerIsGen5` trace) | `{T_mma: bar_W}` (consumer is gen5) | `{T_mma: tok_W}` |

Note: F and W produce two completion barriers on the gen5 MMA
(`bar_F` and `bar_W`), which fuse into the same `tcgen05_commit`. This
is fine — `fuseTcgen05CommitBarriers` (later in the pipeline) will
deduplicate.

### Verification

```
LDBG("F: producerBarrier=" << !!commF.producerBarrier
     << " consumerBarriers={" << print(commF.consumerBarriers) << "}"
     << " tokens={" << print(commF.tokens) << "}");
// Same for B, W.
```

Expected:
- F: `producerBarrier=1, consumerBarriers={T_mma: bar_F}, tokens={T_mma: tok_F}`
- B: `producerBarrier=1, consumerBarriers={}, tokens={T_corr: tok_B}`
- W: `producerBarrier=1, consumerBarriers={T_mma: bar_W}, tokens={T_mma: tok_W}`

---

## Phase 5 — Sync Insertion (`insertAsyncComm`)

### File

`WSCodePartition.cpp::insertAsyncComm` (with the 96870473d helpers
`isForwardOfChannelLoop` / `isBackwardOfChannelLoop` /
`producerAcquireForChannelLoop`).

### Desired Output (IR after sync insertion)

#### In the loop body (T_corr partition)

```
// === Channel B (back-edge gen5 -> load_in) ===
%bidx_B, %ph_B = getBufferIdxAndPhase(B)
nvws.consumer_wait %tok_B, %bidx_B, %ph_B            // before load_in
%v, _ = ttng.tmem_load %acc[%tok]                     // load_in (T_corr)
nvws.consumer_release %tok_B, %bidx_B                 // after load_in

// === Channel F (forward store_in -> mma) ===
// 96870473d optimization: hoist producer_acquire of F to BEFORE load_in
// (i.e. to producerAcquireForChannelLoop = bwdCh->getDstOp() = load_in)
// In code-gen this becomes:
//   nvws.producer_acquire %tok_F, %bidx_F, %ph_F  (PLACED BEFORE load_in,
//                                                  not before store_in)
... rescale v ...
%tok3 = ttng.tmem_store %rescaled, %acc               // store_in (T_corr)
nvws.producer_commit %tok_F, %bidx_F                  // after store_in
```

#### In the loop body (T_mma partition)

```
// === Channel F consumer side ===
nvws.consumer_wait %tok_F, %bidx_F, %ph_F             // before mma
%tok4 = ttng.tc_gen5_mma A, B, %acc[%tok3]            // mma (T_mma)
   { tc_gen5_commit -> bar_F, bar_B, (bar_W if first iter) }
nvws.consumer_release %tok_F, %bidx_F                 // after mma
// (Note: bar_F is signaled by tc_gen5_commit on the MMA;
//  the WaitBarrier(bar_F) is inserted INSIDE the partition — this is
//  the standard desyncTCGen5MMAOp behavior.)
```

#### Outside the loop (T_corr partition)

```
// === Channel W producer_acquire (first-iter init) ===
nvws.producer_acquire %tok_W, %bidx_W, %ph_W          // before init
%init = ttng.tmem_store %zeros, %acc                  // init (T_corr)
nvws.producer_commit %tok_W, %bidx_W                  // after init
scf.for ... { ... }
```

#### Loop body in T_mma partition for channel W

```
nvws.consumer_wait %tok_W, %bidx_W, %ph_W             // before mma (first iter only)
%tok4 = ttng.tc_gen5_mma ...
nvws.consumer_release %tok_W, %bidx_W                 // after mma
```

(The W consumer-wait/release will need to be guarded to fire only
on the first iteration; one option is to peel iteration 0 in
`insertAsyncComm`, another is to use the existing
`accum-counter` mechanism.)

### Required Path in `insertAsyncComm`

The 96870473d helpers must trigger:

1. While processing channel F (`store_in → mma`):
   - `headProducer.getBlock() == headConsumer.getBlock()` — true.
   - `appearsBefore(headProducer, headConsumer)` — true (store_in
     before mma).
   - Call `isForwardOfChannelLoop(F)`. It scans `orderedChannels` for
     a channel chB such that:
     - `chB.allocOp == F.allocOp` — true (B has same alloc).
     - `chB.srcOp == F.dstOp` — true (B's src is mma, F's dst is mma).
     - `chB.dstOp == F.srcOp` — should be true if `chB.dstOp` (load_in)
       and `F.srcOp` (store_in) are in the same block.
     - **Wait** — the check is `withSameTask(chB.dstOp, F.srcOp)`,
       which compares partition IDs of load_in and store_in. Both T_corr —
       OK.
     - Plus all in same block.
     - And `appearsBefore(chB.dstOp, F.srcOp)` — load_in before
       store_in → true.
   - Returns `bwdCh = B` → `producerAcquireForChannelLoop = B.dstOp =
     load_in`.
   - The `ProducerAcquireOp` for F is then inserted at `load_in`'s
     position (instead of before `store_in`), which is the desired
     behavior from 96870473d.

2. While processing channel B (`mma → load_in`):
   - `headProducer.getBlock() == headConsumer.getBlock()` — true.
   - `appearsBefore(headProducer, headConsumer)` — false (mma is
     after load_in in the body), so we hit the loop-carried branch.
   - The new branch (96870473d-style) skips B as the master via
     `isBackwardOfChannelLoop(B)` — returns true → `continue` (the
     barrier was already placed when F was processed).

### Verification

After `insertAsyncComm`, dump the IR and check for these substrings:

```
// CHECK: nvws.producer_acquire %tok_F                 // should be BEFORE load_in
// CHECK-NEXT: ttng.tmem_load                          // load_in
// CHECK: ttng.tmem_store                              // store_in
// CHECK-NEXT: nvws.producer_commit %tok_F
//
// On the gen5 partition:
// CHECK: nvws.consumer_wait %tok_F
// CHECK-NEXT: ttng.tc_gen5_mma {{.*}} commit_with %bar_F, %bar_B
// CHECK-NEXT: nvws.consumer_release %tok_F
```

---

## Phase 6 — Specialization (`specializeRegion`)

### File

`WSCodePartition.cpp::specializeRegion`.

### Desired Output

Three warp-specialized partitions emit the right operations:

- **T_corr partition** (default): receives `producer_acquire`,
  `tmem_load`, `... rescale ...`, `tmem_store`, `producer_commit`,
  `consumer_wait` (for B), `consumer_release` (for B).
- **T_mma partition**: receives `consumer_wait` (for F + W),
  `tc_gen5_mma` (with commit fused for bar_F + bar_B + bar_W),
  `consumer_release` (for F + W).
- **Outside-loop init partition** (T_corr): runs `producer_acquire(W)`,
  `init = tmem_store`, `producer_commit(W)`.

No partition should reference an `nvws.consumer_wait` whose token has
no matching `nvws.consumer_release` — that indicates a dropped channel.

### Verification

```
ttg.warp_specialize {
  default { /* T_corr ops as above */ }
  partition0 { /* T_mma ops as above */ }
}
```

---

## Phase 7 — Barrier Fusion (`fuseTcgen05CommitBarriers`)

### File

`WSCodePartition.cpp::fuseTcgen05CommitBarriers`.

### Desired Output

The MMA's two/three completion barriers (bar_F, bar_B, optionally
bar_W) are emitted as a single `tcgen05.commit` instruction (or fused
where the hardware allows). No barrier is dropped.

---

## Cross-Phase Invariants (assertions to add)

1. After Phase 1, for any operand D `tmem_alloc` whose body has the
   channel-loop pattern, there MUST exist a back-edge channel B with:
   - `B.allocOp == tmem_alloc`
   - `isa<TCGen5MMAOp>(B.srcOp) && B.srcOp.getD() == tmem_alloc`
   - `isa<TMEMLoadOp>(B.dstOp)`
   - `B.srcOp` is in the loop body, `B.dstOp` is in the loop body
   - `appearsBefore(B.dstOp, B.srcOp)` — load_in is before mma in body
2. After Phase 4, B's `tokens` map is non-empty (a back-edge with no
   token cannot be enforced).
3. After Phase 5, every `nvws.consumer_wait` has a matching
   `nvws.consumer_release` in the SAME partition (no dangling waits).

These three invariants form a regression net. Each should be a
LLVM_DEBUG-guarded assertion or a verifier check.

---

## Test Plan

### Test Coverage Gap (existing lit tests)

A scan of `test/Hopper/WarpSpecialization/*.mlir` confirms that **no
existing lit test directly exercises this bug**. The triggering
combination requires:
1. outside-loop `tmem_store` initializing the operand-D alloc,
2. in-body order `tmem_load → tmem_store → tc_gen5_mma` on the same
   alloc,
3. the init store survives the `use_acc=false` optimization, and
4. the init store has a single partition id (passes the strict
   `producerTaskIds.size() != 1` check).

Today's coverage:

| Test | Pattern | Why bug doesn't fire |
|---|---|---|
| `ws_code_partition_wrap_around_tmem_channel.mlir` | init + gen5 + post-loop load | No in-body channel loop (condition 2 fails) |
| `blackwell_fa_code_partition.mlir`, `blackwell_fa_fwd_persist_code_partition.mlir`, `reuse_group_2buffer_fwd.mlir`, `ws_memory_planner_fwd.mlir` | FA fwd variants with channel-loop in body | Init store eliminated by `use_acc=false` opt (conditions 1+3 fail) |
| `fa_code_partition.mlir`, `1D_tmem.mlir`, `blackwell_ws_data_partition.mlir` | channel-loop in body, no outside init | Condition 1 fails |
| `ws_memory_planner.mlir` (XFAIL) | FA bwd, multi-task init | Strict partition-id check fires before load handling (condition 4 fails) |

### New lit test: `ws_code_partition_operand_d_channel_loop.mlir`

Synthetic MLIR exercising the canonical pattern above with:
- Single-task init store outside the loop.
- In-body order: `tmem_load (T0)` → `tmem_store (T0)` → `gen5 (T1)`.
- Post-loop tmem_load (T0).

CHECK lines verify:
- `=== Channels for OperandD ===` dump shows 3 channels.
- Channel B has `SrcOp: %{{.*}} = ttng.tc_gen5_mma` and
  `DstOp: %{{.*}} = ttng.tmem_load`.
- Final IR has `producer_acquire` on tok_F **before** `tmem_load`
  (96870473d optimization), not before `tmem_store`.

### Coverage tests (existing)

- `ws_code_partition_wrap_around_tmem_channel.mlir` — must still pass
  (4 channels: F, gen5→post_load, wrap-around, guard).
- `blackwell_fa_code_partition.mlir` — must still pass (init store
  eliminated by `use_acc=false` opt; no channel-loop in body).
- `ws_memory_planner.mlir` (XFAIL) — unchanged.
- `blackwell_fa_fwd_persist_code_partition.mlir` — was crashing due to
  stale-annotation collision; fixed by the idempotent-cleanup commit.
- `reuse_group_2buffer_fwd.mlir` — requires CHECK-line update because
  the back-edge channel is now legitimately emitted (see below).

### Existing test CHECK-line updates required

`reuse_group_2buffer_fwd.mlir` is the FA-fwd persistent test that
verifies the operand-D race fix's same-task guard. With the back-edge
channel emitted, the accumulator MMA now carries 2 distinct channel
ids in `tmem.start` (back-edge B id + post-loop forward P id) instead
of the baseline's single id duplicated by buffer-reuse processing.

The structural intent of the test is preserved:
- No `producer_acquire` between the qk MMA and the accumulator
  `consumer_wait`.
- The accumulator MMA's `tmem.start` array contains two channel ids
  (now distinct rather than a single id repeated 2x).

**Before (baseline, under-synchronized — back-edge silently dropped):**
```mlir
// CHECK: ttng.tc_gen5_mma {{.*}}loop.cluster = 4{{.*}}loop.stage = 1{{.*}}tmem.start = array<i32: 17, 17>
// CHECK: ttng.tc_gen5_mma {{.*}}loop.cluster = 1{{.*}}loop.stage = 2{{.*}}tmem.start = array<i32: 15, 15>
```

**After (with back-edge channel correctly emitted):**
```mlir
// CHECK: ttng.tc_gen5_mma {{.*}}loop.cluster = 4{{.*}}loop.stage = 1{{.*}}tmem.end = array<i32: {{.+}}>, tmem.start = array<i32: {{.+}}, {{.+}}>
// CHECK: ttng.tc_gen5_mma {{.*}}loop.cluster = 1{{.*}}loop.stage = 2{{.*}}tmem.end = array<i32: {{.+}}>, tmem.start = array<i32: {{.+}}, {{.+}}>
```

Concrete observed values (for reference; may shift with channel-ID
reassignment):
- `cluster=4, stage=1` gen5: `tmem.end = array<i32: 18>, tmem.start = array<i32: 17, 19>` (was `array<i32: 17, 17>`)
- `cluster=1, stage=2` gen5: `tmem.end = array<i32: 15>, tmem.start = array<i32: 14, 16>` (was `array<i32: 15, 15>`)

The two `tmem.start` ids correspond to the new back-edge channel B
(`gen5 → in-body tmem_load`) and the post-loop forward channel P
(`gen5 → post-loop tmem_load`). The single `tmem.end` id is the
in-body forward channel F (`in-body tmem_store → gen5`).

The original `array<i32: 17, 17>` was a buffer-reuse duplication of a
single channel id; the new `array<i32: 17, 19>` is two distinct
channels, which is the correct cross-partition synchronization shape.

---

## Code Locations Summary

| Phase | File | Function | Change |
|-------|------|----------|--------|
| 1 | `CodePartitionUtility.cpp` | `handleOperandD` | Add channel-loop detection; conditionally skip pre-loop seeding; update `numChannelsCreated`/`firstProducer`/`lastConsumer` in deferred path; emit channel W |
| 2 | `WSCodePartition.cpp` | `doCodePartition` (region/reuse setup) | None expected; verify F/W don't accidentally merge |
| 3 | `WSCodePartition.cpp` | `createBufferForAllocs` | None expected |
| 4 | `WSCodePartition.cpp` | `createToken` | None expected; verify W gets its own token |
| 5 | `WSCodePartition.cpp` | `insertAsyncComm` | None expected (96870473d code already correct once B exists); add assert that `isBackwardOfChannelLoop(B)` succeeds |
| 6 | `WSCodePartition.cpp` | `specializeRegion` | None expected |
| 7 | `WSCodePartition.cpp` | `fuseTcgen05CommitBarriers` | None expected |
