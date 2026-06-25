# Partition Scheduler Known Issues & Patterns

> **For full architectural context**, load the `partition-scheduler` skill which points to the design docs (PartitionSchedulingMeta.md, BufferAllocation.md, etc).

> Update this file when an issue is triaged/fixed and PartitionSchedulingMeta.md if necessary

## Code Location
- Partition assignment: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/PartitionSchedulingMeta.cpp`
- Buffer allocation: `WSCodePartition.cpp` → `doBufferAllocation()` → `createLocalAlloc()`
- Code partition: `WSCodePartition.cpp` → `doCodePartition()`

## Debugging Regression between directory A and B
- If IR dumps are provided after each pass:
  - Find the IR right before partition scheduler for the right kernel, and save as file
- Do not guess, run triton-opt for the partition scheduler pass with debugging enabled or add debugging when needed, to check what happened at each phase (phases are defined in the PartitionSchedulingMeta.md)
- Run directory A's triton-opt on A's IR dump, and run directory B's triton-opt on B's IR dump, and compare
- Show the differences and figure out which phase caused the issue
- **Important**: Check BOTH directories for the same kernel. MetaMain at `~/local/MetaMain/triton/t.dump` may have both fwd and bwd kernels.

## Known Bugs & Fixes

### 1. getIntOrFloatBitWidth crash on pointer-typed 1D tensors (2026-04-14)
- **Symptom**: `Assertion 'isIntOrFloat()' failed` in `doBufferAllocation`
- **Manifestation**: We hit this when trying to create a 1D channel for pointer tensor. In general, partition scheduler should not put produer and consumer associated with pointer tensor in different partitions. So we will not have a need for a channel that is a pointer tensor. The root cause is in PSM.

### 2. Shared memory overflow from alpha cross-partition channel (2026-04-14, fixed)
- **Symptom**: `OutOfResources: shared memory, Required: 232712, Hardware limit: 232448` in FA forward persistent with dp=2
- **Manifestation**: After rebasing to upstream Triton, `TritonGPURemoveLayoutConversions` chose `#linear` layout instead of `#blocked` for the accumulator. This inserted a `ConvertLayoutOp` between `ExpandDimsOp` and `BroadcastOp` in the alpha correction chain.
- **Fix applied**: Added `cloneOperandChain` in `optimizeSchedule` that walks backward from a cloned `BroadcastOp`/`ExpandDimsOp` and also clones any `ConvertLayoutOp`/`BroadcastOp`/`ExpandDimsOp` feeding it from a different partition.
- **Commit**: `67af25ea`

### 3. optimizeSchedule too broad / too narrow for Blackwell vs Hopper (2026-04-17, fixed)
- **Symptom (Blackwell)**: `channels sharing the same producer must be in the same task` assertion in `WSCodePartition.cpp:createBuffer` when using the broad `isPure(op)` filter.
- **Symptom (Hopper)**: `producerTaskIds.size() == 1` assertion in `CodePartitionUtility.cpp:createChannelPost` when using a restrictive filter that excludes `MemDescTransOp`.
- **Root cause**: The `optimizeSchedule` op filter must be selective:
  - Too broad (any pure single-result op): cascading cloning of expensive ops (`tt.reduce`, `arith.mulf`, etc.) into computation partitions on Blackwell, violating channel invariants.
  - Too narrow (only `ConvertLayoutOp/BroadcastOp/ExpandDimsOp`): `memdesc_trans` shared by two `warp_group_dot` ops in different partitions on Hopper doesn't get cloned, creating a cross-partition memdesc dependency WS can't handle.
- **Fix**: Added `MemDescTransOp` to the allowed op list: `isa<MemDescTransOp, ConvertLayoutOp, BroadcastOp, ExpandDimsOp>(op)`. `MemDescTransOp` is metadata-only (reinterprets shared memory layout) so it's safe and cheap to clone.
- **Lit test**: `partition-scheduling-meta-hopper-fa.mlir` checks for two `memdesc_trans` copies with different partitions.

### 4. Non-deterministic epilogue partition assignment from DenseMap iteration (2026-04-17, fixed)
- **Symptom**: `producerTaskIds.size() == 1` assertion — `math.log2` for dp1's result gets partition 2 (dp0's) instead of partition 1, creating a cross-partition dependency with its downstream `arith.addf` in partition 1.
- **Root cause**: Two issues:
  1. Yield operands for `l_i` (softmax sum) and similar non-MMA-feeding ops are NOT in `opToDpId` (they're not in any MMA's backward slice). The post-loop dpId assignment at lines 576-578 skips these results.
  2. The fallback `dpIdToPartition.begin()->second` in `getEpilogueTarget` uses `DenseMap` iteration, which is non-deterministic across builds. Different binaries pick different partitions.
- **Fix**:
  1. Added `findDpIdBackward` helper that walks backward from a yield def through its operand chain to find an ancestor in `opToDpId` (e.g., finds `alpha_exp` which has the correct dpId).
  2. Replaced `dpIdToPartition.begin()->second` with `std::min_element` on the key for deterministic fallback.
- **Lit test**: `partition-scheduling-meta-hopper-fa.mlir` checks that `tt.expand_dims` on `#1` (dp0) gets partition 2 and `#4` (dp1) gets partition 1.

### 5. BWD softmax chain assigned to reduction instead of computation (2026-04-18, fixed)
- **Symptom**: In BWD FA with TMA descriptor_load for m/Di values, the pT chain (`convert_layout → expand_dims → broadcast → arith.subf → math.exp2 → arith.truncf → tmem_alloc`) gets partition 0 (reduction) instead of partition 3 (computation).
- **Root cause**: The load-user scheduling (Phase 4) walks forward from every categorized `descriptor_load` and assigns all transitive users to `defaultPartition`. For BWD, `defaultPartition` falls back to `reductionPartition` (partition 0) via `getDefaultPartition()` since no correction/epilogue/computation partition exists yet. When m/Di values come through `descriptor_load` (TMA), this walk transitively pulls the entire softmax chain into the reduction partition. The lit test used `tt.load` (pointer-based) for m/Di which is NOT categorized as a Load, so the issue was hidden.
- **Fix**: Added guard `defaultPartition != reductionPartition` to the load-user scheduling condition. When `defaultPartition` is just a fallback to reduction (BWD case), the load-user walk is skipped. Phase 5's MMA forward walk correctly assigns the softmax ops to computation instead.
- **Key insight**: The `loops` array in `getInitialSchedule` is ordered `[inner, outer]` (not `[outer, inner]`). Phase 5's `loops[0]` check matches inner-loop MMAs, so `scheduleUsers` DOES run on them. The issue was purely in Phase 4's load-user scheduling being too aggressive.

### 6. Forward set dead-ends at scf.yield inside causal-mask scf.if (2026-05-28, fixed)
- **Symptom**: TMEM exhaustion (`640 vs 512` limit) in causal FA w/ manual dp. The pass produces 8 partitions / 4 computation instead of the expected 6 / 2.
- **Root cause**: In `collectMMABackwardSlices`, the QK MMA's forward user set walk dead-ends at `scf.yield` (a terminator with no SSA results) when it enters the causal-mask `scf.if` region. The forward set is trapped inside the region and never reaches the softmax chain that the PV MMA's backward slice contains. As a result, union-find fails to merge the QK and PV MMAs, so they land in separate computation partitions.
- **Fix**: Make the forward walk follow `scf.yield` → the parent `scf.if`'s corresponding result, using the specific yield operand index so multiple data partitions (e.g., flex attention) are not merged. For multi-result `scf.if`, the walk follows one value at a time from the worklist. With the enlarged forward set, the QK MMA's forward set overlaps the PV MMA's backward slice and they share a compute partition.
- **Lit test**: `partition-scheduling-meta-causal-attention.mlir` (new) checks exactly 6 partitions / 2 computation (fails with 8 / 4 without the fix). Also removed a stale `CHECK-NOT` on `arith.mulf` scores ops in `partition-scheduling-meta-flex-attention.mlir`, which now correctly get computation partitions through the enlarged forward set.

### 7. Empty-producer assertion from partition-less shared `local_alloc` (2026-06-10, fixed)
- **Symptom**: `producerTaskIds.size() == 1` assertion in `CodePartitionUtility.cpp` `createChannelPost` (NDEBUG: `handleOperandD: expected exactly one producer task ID, got 0` + segfault).
- **Manifestation**: FA3 backward (`_attn_bwd_persist`, `cuda:100`, `data_partition_factor=2`). `separateLocalAllocWithSrc` splits a shared `local_alloc` that carries no `async_task_id` (hoisted above all partitions) into `local_alloc + local_store`; the store inherits the empty task-id set, so `producerTaskIds` is empty in `createChannelPost`.
- **Root cause**: `createChannelPost`'s `else` branch assumes exactly one producer task. An *empty* set (size 0) — not a multi-producer — is what trips the assert; the `size() > 1` branch already handles multiple producers.
- **Fix**: Guard `if (producerTaskIds.empty()) return;` — an alloc that belongs to no partition needs no cross-partition channel. Companion: `separateLocalAllocWithSrc` now tags the split `local_store` with the source op's single task ID (e.g. the TMA load) so a single-source shared buffer forms a clean 1-producer→N-consumer channel instead of one buffer per consumer.
- **Lit test**: `ws_code_partition_empty_producer_guard.mlir` runs `_attn_bwd_persist` through `--nvgpu-warp-specialization`; aborts without the guard, passes with it.

### 8. Cross-stage SMEM buffer silently downgraded to depth 1 (2026-06-22, fixed)
- **Symptom**: Runtime mbarrier **deadlock** in `_attn_bwd_persist` (FA backward, `ws_persistent`, `cuda:100`) with `early_tma_store_lowering=True` + `num_stages=2` + the heuristic `_BWD_DOT_ATTRS_SCHED` schedule (`bwd_config_idx=1`). Every warp-group spins on `q = desc_q.load` / dependent MMAs.
- **Root cause** (`WSMemoryPlanner.cpp`, `allocateSmemBuffers`): `q` (128×128×f16) is cross-stage — consumed by qkT MMA at `loop.stage=0` and dk MMA at `loop.stage=1` — so it needs `buffer.copy=2`. Phase 2 set `q`'s 2nd copy then budget-checked via `computeTotalSmem`, which counted the 3 early-TMA staging allocs (dq reduce + dk/dv store, ~32 KB) at full size, so `q`'s 2nd copy exceeded `smem_budget` (200000) and Phase 2 **silently reverted `q` to copy 1**. Phase 3.6 reuse — which would alias the staging SMEM and free the space — only runs `if (baseTotal > smemBudget)`, but the revert put the total back under budget, so reuse never fired. Chicken-and-egg: reverting the floor kept us under budget, so the space the floor needs was never reclaimed.
- **Fix**: Make the cross-stage minimum a strict, first-applied floor. (1) Phase 2 sets `minCopies = max(maxConsumerStage - minConsumerStage + 1, 1)` (capped at `num_buffers`) via new `getSmemCrossStageDepth`, applied unconditionally with **no budget revert**. (2) Phase 3.6 now reclaims **only discretionary** space — TMA-staging buffers — and excludes co-live operand buffers (new `isSmemLiveAcrossInnerLoop` guard) so `q`/`k`/`v` are never aliased. (3) If the floor still doesn't fit after staging reuse, ship it anyway (HW SMEM limit / `OutOfResources` at codegen is the backstop) rather than reverting. Phase 4/4.5 reverts/splits clamp to `minCopies`. This realigns the code with the documented design in `SmemAllocationDesign.md` ("no budget check in Phase 2").
- **Lit tests**: `ws_memory_planner_bwd_persist_early_tma.mlir` (new — real 128-wide failing IR: `q` now `buffer.copy=2`, store-staging reused). Updated `ws_memory_planner_bwd_persist.mlir` and `ws_memory_planner_bwd.mlir` (`q` 1→2, both cross-stage buffers get their floor; `v`/`k` not aliased). E2E: `fused_attention_ws_device_tma.py::test_op[1-False-0-False-triton-fp16-ws_persistent-bwd-False-128-1024-16-8]` now passes (was a hang); `0-…` config unregressed.

### 9. Subtile accumCnt stride computed per-loop instead of per-channel (2026-06-25, fixed)
- **Symptom**: Runtime mbarrier **deadlock** in `matmul_kernel_tma_persistent_ws` with `generate_subtiled_region=True` + `EPILOGUE_SUBTILE`/`numTiles ≥ 2` (e.g. the `…-4-3-64-128-256-8192-8192-1024` tutorial09 config). Hangs on the 2nd output tile.
- **Root cause** (`WSBuffer.cpp`): `getAccumCntIncrement` was **loop-scoped** — it returned `numTiles` whenever the loop contained *any* `SubtiledRegionOp`, and `generateYieldCntsForForOp` stamped that stride onto the *first* accumCnt, which is the **TMEM-accumulator** channel (a non-subtile, loop-body-scope channel). The TMEM accumulator is depth-2; with `slot = count % 2`, `phase = (count/2)&1`, a `+numTiles` stride pins `slot ≡ 0`/`phase ≡ const`, so the depth-2 slot/phase never alternate and the gemm↔epilogue handshake deadlocks. Meanwhile the subtile C-store reuse-group counter carried `+1` and leaned on an in-body `accumCnt * numTiles` in `getOrComputeSubtiledSlot` (`WSCodePartition.cpp`) — the stride was on the wrong counter.
- **Fix**: Make the stride **per-channel**. (1) `generateYieldCntsForForOp` always yields `+1` (deleted `getAccumCntIncrement`). (2) New `getReuseGroupStride` detects a subtiled reuse group (its channel src/dst is in a `SubtiledRegionOp`) and returns `numTiles × perTileConsumptions` (`perTile==1` asserted); `getAccumForReuseGroup` multiplies its per-position `lit` by this stride, so only subtile-body channels step by `numTiles`. (3) `getOrComputeSubtiledSlot` drops the in-body `* numTiles` → `flattened = accumCnt + tileIdx`. Equivalent flattened stream (`iter*numTiles + tileIdx`); the TMEM counter reverts to `+1` and alternates again.
- **Lit tests**: updated `ws_code_partition_subtiled_region_inbody.mlir`, `ws_subtiled_region_per_tile_barrier.mlir`, `ws_subtiled_region_per_tile_buffer_index.mlir` (in-body `mul`→`add`; assert the subtile counter steps `+numTiles` while the non-subtile counter steps `+1`). E2E: the `…-4-3-64-128-256-8192-8192-1024` tutorial09 config now passes (was a hang); base-vs-fix sweep confirmed no regressions (every still-failing config — `separate_epilogue_store=False` / `early_tma=False` / `DP=2` subtile paths — already failed/hung on base; those are separate pre-existing issues).

## Debugging Workflow
- `t.dump` captures IR after each WarpSpec pass (doTaskIdPropagate → doBufferAllocation → doMemoryPlanner → doCodePartition → ...)
- IR after PartitionSchedulingMeta uses `ttg.partition = array<i32: N>` attributes (not `async_task_id`)
- IR after doTaskIdPropagate converts `ttg.partition` to `async_task_id` annotations
- To check partition assignments: look at IR between `NVGPUPartitionSchedulingMeta` and `NVGPUWarpSpecialization` dump sections
- Build: see xxx/build-triton.txt
- To run a single pass: `triton-opt --nvgpu-partition-scheduling-meta="merge-epilogue-to-computation=true" input.mlir`
- To enable debug: add `-debug-only=tritongpu-partition-scheduling`
- To add stack traces on specific ops: instrument `setPartition()` in `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/Partition.cpp`

## Key Concepts
- `PartitionSchedulingMeta` assigns `ttg.partition` attributes → `doTaskIdPropagate` converts to `async_task_id`
- Pointer-typed tensors (`!tt.ptr<T>`) should not be cross-partition
