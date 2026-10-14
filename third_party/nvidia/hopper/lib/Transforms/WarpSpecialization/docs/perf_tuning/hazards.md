# AutoWS correctness hazards (open)

These are the conditions under which a Meta WS (`TRITON_USE_META_WS=1`) kernel can still give a wrong answer, hang, or fault at runtime. Each entry gives:
- the trigger condition;
- the symptom;
- the code location, as a file and symbol, of the missing or faulty piece.

An entry is deleted as soon as the fix lands; this file only lists what is open. When a tuning file says "see hazards", treat that region of the search space as off-limits until the entry is gone.

Two general rules apply to every entry:
- These hazards are races or depend on layout, so a single passing run does not rule them out.
- compute-sanitizer's racecheck and synccheck do not model mbarrier, TMA or tcgen05 ordering. They do not catch these hazards. See [DebuggingAccuracyAndDeadlocks.md](../DebuggingAccuracyAndDeadlocks.md), section "Process".

### TMA tile read into registers, released without a proxy fence

- **Status:** from code inspection; not reproduced in testing. The IR does show the release arrive with no fence.
- **Trigger:** a TMA-loaded SMEM buffer is read into registers in one partition (`local_load`) and then handed back to the TMA producer. The hand-back is either a token release or the release attached to a TMA-store wait. Typical sources:
  - a register-consumed descriptor load: a cast prologue, a bias or other epilogue input loaded with TMA, or a reduction side input;
  - an MMA operand that is also read in registers.
- **Symptom (expected from the memory model):** the next TMA write into the slot can overtake the in-flight generic-proxy `ld.shared` reads, which would give a nondeterministic wrong answer.
- **Code:**
  - `WSLowerToken.cpp`, `processConsumerReleaseOp`: lowers the release to a plain mbarrier arrive. No `fence.proxy.async` is emitted, whatever the token's load type.
  - `WSLowerToken.cpp`, `lowerTokenOperations`, the `TMAStoreTokenWaitOp` case: same gap.
  - For comparison, `LowerAref.cpp`, `rewriteGetExitOp` emits the fence when a generic-proxy consumer releases a TMA-produced buffer.
  - `ProxyFenceInsertion.cpp` analyses one region at a time, so it never sees the reader and the writer, which sit in different partitions.

### Single K tile with an additive epilogue loaded by TMA

- **Status:** reproduced. The same kernel is exact with a pointer-loaded epilogue input, with a K loop, or with warp specialization off.
- **Trigger:** all of the following hold, under Meta WS:
  - a persistent kernel whose IR has no K loop: a single `tl.dot` with K a compile-time constant (`tl.constexpr`) <= BLOCK_K. A runtime K launched with K <= BLOCK_K still has a K loop and is not affected;
  - the GEMM result is combined with an `[M, N]` epilogue input, e.g. a bias or residual added to it;
  - that input is loaded with a TMA descriptor.
- **Symptom:** wrong GEMM output on every run.
- **Code:**
  - `Combine.cpp`, `CombineDotAddFPattern` (and `CombineDotAddIPattern` for integers) folds `dot(a, b, 0) + d` into `dot(a, b, d)` when the dot feeds the add directly. The addend therefore becomes an accumulator-init `tmem_store` in a different task from the MMA.
  - `WSCodePartition.cpp`, `insertAsyncComm`, operand-D guard-channel path: the builder's task IDs are left on the MMA's task. The plain `ProducerCommitOp` that follows is created without resetting them to the producer (`relation.first`), so the commit lands in the MMA partition.

### TMA landing buffer smaller than 128 bytes, multi-buffered

- **Trigger:** a descriptor load whose per-copy SMEM size is not a multiple of 128 bytes gets more than one copy from the planner. Example: a `[1, BLOCK_N / EPILOGUE_SUBTILE]` bf16 bias slice narrower than 64 elements, at `EPILOGUE_SUBTILE >= 2`.
- **Symptom:** a misaligned-address fault at runtime.
- **Code:**
  - `WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` (Phase 3.7) and the copy assignment in `allocateSmemBuffers` never check the 128-byte TMA alignment.
  - The non-WS pipeliner does check it: `PipeliningUtility.cpp`, `canPipelineTMALoad` (`TMA_ALIGN`).

### Atomic inside the warp-specialized tile loop with no partition

- **Status:** reproduced.
- **Trigger:** a `tl.atomic_*` (an `AtomicRMWOp`) inside the WS loop whose operands do not depend on the accumulator, so no partition reaches it, e.g. a counter `atomic_add(cnt, 1)` or adding a constant to a column. Atomics of accumulator-derived values (a column sum, a scalar sum) are kept and are exact.
- **Symptom:** the atomic is removed from the TTGIR, so its result stays 0 on every run, while warp specialization is still applied.
  - Per the code, an atomic replicated to more than one partition instead makes the atomic broadcast reject and strips warp specialization from the kernel; that variant was not reproduced in testing. Under `two_ctas`, stripping WS hangs (see [2cta.md](2cta.md#never-try-any-problem)).
- **Code:**
  - `PartitionSchedulingMeta.cpp`, `runOnOperation`: the fix-up that schedules unreachable stores handles `tt.store` only and explicitly leaves atomics alone.
  - `WSAtomicBroadcast.cpp`, `classifyAtomic`: atomics with at most one task ID pass through unchanged.
  - `WSSpecialize.cpp`, `SpecializeOp`: an op that has no task ID and that no partition needs is not cloned into any partition, and the original loop is erased.

### 2-CTA with an unsplit B

- **Status:** reproduced. The same kernels as 1-CTA, and `BLOCK_N=32` as 2-CTA, are exact. A register-modified A is exact under 2-CTA, because A is promoted to TMEM (`PromoteLHSToTMem.cpp`, `LHSToTMem`).
- **Trigger:** `tl.dot(..., two_ctas=True)` where the B split is skipped:
  - `BLOCK_N/2` < 16 (e.g. `BLOCK_N=16`); or
  - B is not a direct descriptor load feeding the MMA, e.g. TMA-loaded and then scaled or cast in registers (a B-side prologue).
- **Symptom:** a silent wrong answer on every run, with no error or warning.
- **Code:** `third_party/nvidia/hopper/lib/Transforms/Transform2CTALoads.cpp`: `runOnOperation` skips an MMA whose B is not from a descriptor load ("B not from descriptor load"), and `transformBLoad` / `transformSplitBLoad` skip when `halfN < 16` ("too small, skipping"). Both are debug logs only. The MMA lowering still reads a half-width B per CTA (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`, `convertDotImpl`).

### 2-CTA with the atomic dynamic tile scheduler

- **Status:** reproduced. Static persistent + 2-CTA and dynamic + 1-CTA are exact.
- **Trigger:** `DynamicPersistent1DScheduler` with `tl.dot(..., two_ctas=True)` under Meta WS.
- **Symptom:** compiles silently, then a wrong answer or a hang at runtime.
- **Code:** `python/triton/language/schedule.py`, `DynamicPersistent1DScheduler`: the tile claim is a per-CTA global atomic with no cluster-level claim, so the two CTAs of a pair can take different tiles; nothing rejects the combination at compile time. Design doc: [DynamicPersistentAutoWSGaps.md § 2-CTA Status](../DynamicPersistentAutoWSGaps.md#2-cta-status).

### 2-CTA without warp specialization

- **Status:** reproduced.
- **Trigger and symptom:**
  - Meta WS off (`TRITON_USE_META_WS` unset): a persistent 2-CTA GEMM gives a wrong answer. With a pointer-loaded B it is wrong on every run; with a descriptor-loaded B it is wrong on most runs (a race).
  - Meta WS on but the loop not warp-specialized (`warp_specialize=False`): the first iteration is correct, then the kernel hangs.
- **Code:** without warp specialization, two 2-CTA synchronization schemes apply at once: the cross-CTA rendezvous inserted by `third_party/nvidia/hopper/lib/Transforms/Insert2CTASync.cpp` (`insertSyncBeforeMMA`; gated only on Meta WS, so it also runs after WS is stripped) and the cluster arrive the MMA lowering emits when the MMA is not inside a `ttg.warp_specialize` (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`, `convertDotImpl`). Without Meta WS, `make_ttgir` never warp-specializes a cluster kernel (`third_party/nvidia/backend/compiler.py`).
- **Also applies to:** any condition that silently strips warp specialization from a 2-CTA kernel (`num_warps` < 4, warp estimate > 16, a rejected replicated atomic).

### Same-partition TMA staging floor that does not divide the subtile count

- **Status:** from code inspection; no reproducer yet.
- **Trigger:** TMA-store staging is drained in the same partition with K copies, and S subtiles rotate through those copies. The planner's cross-stage correctness floor already forces K > 1, and S is not a multiple of K.
- **Symptom:** a store can overwrite a staging slot before its TMA store has drained, giving a wrong answer.
- **Code:** `WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies`. It logs a debug-only warning ("wait_group rotation may be unsafe") and continues. The copy count can only be raised to values that divide S (see [TMAStoreWaitPipeline.md](../TMAStoreWaitPipeline.md), section "Phase 3.7: Epilogue Group Copy Increase"); a floor that already violates this is left alone.

### auto_tma promotion drops the load mask

- **Status:** reproduced for all three cases below: each gives wrong elements with `auto_tma=True`, while `auto_tma=False` is exact.
- **Trigger:** `auto_tma=True` (or `TRITON_AUTO_TMA=1`) promotes a pointer `tl.load` to a TMA descriptor load. Three cases are unsafe:
  1. **Extra predicate in the mask.** The load uses `other=0` and its mask has a predicate beyond the per-dimension bounds, e.g. causal `offs_m >= offs_n` or a lower bound `offs >= lo`. The promoted load returns real data where the original read 0.
  2. **Wraparound read.** The offset is `(arange + splat(off)) % arg` and the wraparound is real, e.g. an Inductor dynamic broadcast `x0 = xindex % ks0`. The modulus is taken as the dimension's extent, so positions past `arg` read 0 instead of wrapping.
  3. **Wrong extent.** The load's own mask bound is not a bare kernel argument, e.g. `offs < M - 1`. The extent then becomes the first `offs < arg` comparison on that offset found anywhere in the kernel.

  GEMM operands with the usual `% M` wrap or K-tail masks are not affected.
- **Symptom:** a wrong answer, silently.
- **Code:** `PromoteLoadToTMA.cpp` (mechanics in [AutoTMAPromotion.md](../AutoTMAPromotion.md)).
  - The load promotion path (`isTMAEligible` and the load walk) checks only that `other` is zero and never checks that the load mask is rectangular. Stores and atomics do check it, with `storeMaskIsRectangular`.
  - `match1DOffset` peels a `remsi` by a kernel argument as the out-of-bounds idiom.
  - `findShapeArgForDim` falls back to a kernel-wide search for the first matching bound.

### Plan search pins TMEM accumulator copies to 1

- **Status:** reproduced. The in-tree test `test_autows_addmm_pointer_bias_single_k_tile` fails under plan search and passes without it; with K = 2 x BLOCK_K it passes either way.
- **Trigger:** `TRITON_WS_SMEM_PLAN_SEARCH=1` on a persistent kernel with no K loop (K == BLOCK_K as a compile-time constant) and an epilogue input, with more tiles than SMs. Kernels with scaled MMA or a `SubtiledRegionOp` fall back to the heuristic planner and are not affected.
- **Symptom:** a silent wrong answer on every run.
- **Code:** `WSMemoryPlanner.cpp`: `allocateTmemBuffersViaSearch` stamps `buffer.copy=1` on every TMEM alloc. The heuristic allocator's TMEM multi-buffering post-processing gives a per-iteration accumulator one slot per in-flight iteration, because the loop scheduler can place its initialization and its epilogue load in different stages; plan search skips that step.

### Relaxed operand-copy floor with a TMA-loaded epilogue input, several tiles per CTA

- **Status:** reproduced. One tile per CTA is exact, and relaxed plans without the TMA-loaded epilogue input are exact. Root cause not yet located.
- **Trigger:** all of the following, under Meta WS:
  - `data_partition_factor` > 1, where the static copy-safety floor applies, and the planner relaxes that floor below `num_stages` because it does not fit (e.g. `EPILOGUE_SUBTILE=2`, `num_stages=3`, ending at A=2/2 and B=1 copies after the auxiliary reservation), with or without `disallow_acc_multi_buffer`;
  - an epilogue input (e.g. a bias) loaded with a TMA descriptor;
  - a CTA processes several tiles (more tiles than CTAs).
- **Symptom:** the kernel hangs.
- **Code:** `WSMemoryPlanner.cpp`, `getStaticSmemCopySafetyFloor` and `relaxCopySafetyFloorToBudget`.

Reported wrong answers and hangs whose root cause is not yet located in code are tracked in [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause), not here.
