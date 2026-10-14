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

- **Trigger:** a TMA-loaded SMEM buffer is read into registers in one partition (`local_load`) and then handed back to the TMA producer. The hand-back is either a token release or the release attached to a TMA-store wait. Typical sources:
  - a register-consumed descriptor load: a cast prologue, a bias or other epilogue input loaded with TMA, or a reduction side input;
  - an MMA operand that is also read in registers.
- **Symptom:** the next TMA write into the slot can overtake the in-flight generic-proxy `ld.shared` reads. The result is a nondeterministic wrong answer.
- **Code:**
  - `WSLowerToken.cpp`, `processConsumerReleaseOp`: lowers the release to a plain mbarrier arrive. No `fence.proxy.async` is emitted, whatever the token's load type.
  - `WSLowerToken.cpp`, `lowerTokenOperations`, the `TMAStoreTokenWaitOp` case: same gap.
  - For comparison, `LowerAref.cpp`, `rewriteGetExitOp` emits the fence when a generic-proxy consumer releases a TMA-produced buffer.
  - `ProxyFenceInsertion.cpp` analyses one region at a time, so it never sees the reader and the writer, which sit in different partitions.

### Register-written MMA operand read by a 2-CTA MMA

- **Trigger:** `tl.dot(..., two_ctas=True)` with an operand that a partition writes from registers into SMEM. Examples: a multi-consumer operand that goes through a register copy, or an operand cast in-kernel.
- **Symptom:** intermittent garbage output.
- **Code:**
  - `WSLowerToken.cpp`, `processProducerCommitOp`: the producer commits with a plain arrive and emits no fence on the writer side.
  - `FenceInsertion.cpp`, `FenceInsertionPass`: the only fence for a register-written dot operand is placed before the MMA, and it is CTA-scoped (`bCluster=false`). A 2-CTA MMA also reads the peer CTA's SMEM.
  - For comparison, `LowerAref.cpp`, `rewritePutExitOp` fences on the producer side.

### Single K tile with an additive epilogue loaded by TMA

- **Status:** reproduced. The same kernel is exact with a pointer-loaded epilogue input, with a K loop, or with warp specialization off.
- **Trigger:** all of the following hold, under Meta WS:
  - a persistent kernel with no K loop: a single `tl.dot`, K <= BLOCK_K;
  - the GEMM result is combined with an `[M, N]` epilogue input, e.g. a bias or residual added to it;
  - that input is loaded with a TMA descriptor.
- **Symptom:** wrong GEMM output on every run.
- **Code:**
  - `Combine.cpp`, `CombineDotAddFPattern` folds `dot(a, b, 0) + d` into `dot(a, b, d)`. The addend therefore becomes an accumulator-init `tmem_store` in a different task from the MMA.
  - `WSCodePartition.cpp`, `insertAsyncComm`, operand-D guard-channel path: the builder's task IDs are left on the MMA's task. The plain `ProducerCommitOp` that follows is created without resetting them to the producer (`relation.first`), so the commit lands in the MMA partition.

### TMA landing buffer smaller than 128 bytes, multi-buffered

- **Trigger:** a descriptor load whose per-copy SMEM size is not a multiple of 128 bytes gets more than one copy from the planner. Example: a `[1, BLOCK_N / EPILOGUE_SUBTILE]` bf16 bias slice narrower than 64 elements, at `EPILOGUE_SUBTILE >= 2`.
- **Symptom:** a misaligned-address fault at runtime.
- **Code:**
  - `WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` (Phase 3.7) and the copy assignment in `allocateSmemBuffers` never check the 128-byte TMA alignment.
  - The non-WS pipeliner does check it: `PipeliningUtility.cpp`, `canPipelineTMALoad` (`TMA_ALIGN`).

### Atomic inside the warp-specialized tile loop with no partition

- **Trigger:** a `tl.atomic_*` inside the WS loop that no partition reaches. This is typical of an atomic whose result is unused, e.g. a cross-tile reduction done by `atomic_add`.
- **Symptom:** the atomic is silently dropped, so the reduction output is wrong.
  - An atomic replicated to more than one partition behaves differently: warp specialization is stripped from the whole kernel, which is a silent perf loss.
- **Code:**
  - `PartitionSchedulingMeta.cpp`, `runOnOperation`: the fix-up that schedules unreachable stores handles `tt.store` only and explicitly leaves atomics alone.
  - `WSAtomicBroadcast.cpp`, `classifyAtomic`: atomics with at most one task ID pass through unchanged.
  - `WSSpecialize.cpp`, `SpecializeOp`: an op that has no task ID and that no partition needs is not cloned into any partition, and the original loop is erased.

### 2-CTA with `BLOCK_N/2` below the B-split minimum

- **Status:** reproduced. The same `BLOCK_N` as 1-CTA, and the next legal `BLOCK_N` as 2-CTA, are exact.
- **Trigger:** `tl.dot(..., two_ctas=True)` with `BLOCK_N/2` < 16 (e.g. `BLOCK_N=16`).
- **Symptom:** a silent wrong answer on every run, with no error or warning.
- **Code:** `third_party/nvidia/hopper/lib/Transforms/Transform2CTALoads.cpp`, `transformBLoad` and `transformSplitBLoad`: when `halfN < 16` the B split is skipped with only a debug log ("too small, skipping"), while the 2-CTA MMA lowering still reads a half-width B per CTA.

### 2-CTA without warp specialization

- **Status:** reproduced.
- **Trigger and symptom:**
  - Meta WS off (`TRITON_USE_META_WS` unset): a persistent 2-CTA GEMM gives a wrong answer. With a pointer-loaded B it is wrong on every run; with a descriptor-loaded B it is wrong on most runs (a race).
  - Meta WS on but the loop not warp-specialized (`warp_specialize=False`): the first iteration is correct, then the kernel hangs.
- **Code (root cause not yet located):** two 2-CTA synchronization schemes can apply at once without warp specialization: the cross-CTA rendezvous inserted by `third_party/nvidia/hopper/lib/Transforms/Insert2CTASync.cpp` (`insertSyncBeforeMMA`) and the inline cluster arrive in the MMA lowering (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`, `convertDotImpl`).

### Same-partition TMA staging floor that does not divide the subtile count (from code inspection; no reproducer yet)

- **Trigger:** TMA-store staging is drained in the same partition with K copies, and S subtiles rotate through those copies. The planner's cross-stage correctness floor already forces K > 1, and S is not a multiple of K.
- **Symptom:** a store can overwrite a staging slot before its TMA store has drained, giving a wrong answer.
- **Code:** `WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies`. It logs a debug-only warning ("wait_group rotation may be unsafe") and continues. The copy count can only be raised to values that divide S (see [TMAStoreWaitPipeline.md](../TMAStoreWaitPipeline.md), section "Phase 3.7: Epilogue Group Copy Increase"); a floor that already violates this is left alone.

### auto_tma promotion drops the load mask (from code inspection; no reproducer yet)

- **Trigger:** `auto_tma=True` (or `TRITON_AUTO_TMA=1`) promotes a pointer `tl.load` to a TMA descriptor load. Three cases are unsafe:
  1. **Extra predicate in the mask.** The load uses `other=0` and its mask has a predicate beyond the per-dimension bounds, e.g. causal `offs_m >= offs_n` or a lower bound `offs >= lo`. The promoted load returns real data where the original read 0.
  2. **Wraparound read.** The offset is `(arange + splat(off)) % arg` and the wraparound is real, e.g. an Inductor dynamic broadcast `x0 = xindex % ks0`. The modulus is taken as the dimension's extent, so positions past `arg` read 0 instead of wrapping.
  3. **Wrong extent.** The load's own mask bound is not a bare kernel argument, e.g. `offs < M - 1`. The extent then becomes the first `offs < arg` comparison on that offset found anywhere in the kernel.

  GEMM operands with the usual `% M` wrap or K-tail masks are not affected.
- **Symptom:** a wrong answer, silently.
- **Code:** `PromoteLoadToTMA.cpp`.
  - The load promotion path (`isTMAEligible` and the load walk) checks only that `other` is zero and never checks that the load mask is rectangular. Stores and atomics do check it, with `storeMaskIsRectangular`.
  - `match1DOffset` peels a `remsi` by a kernel argument as the out-of-bounds idiom.
  - `findShapeArgForDim` falls back to a kernel-wide search for the first matching bound.

Reported wrong answers and hangs whose root cause is not yet located in code are tracked in [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause), not here.
