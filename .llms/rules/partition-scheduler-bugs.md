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

## Debugging Workflow
- `t.dump` captures IR after each WarpSpec pass (doTaskIdPropagate → doBufferAllocation → doMemoryPlanner → doCodePartition → ...)
- IR after PartitionSchedulingMeta uses `ttg.partition = array<i32: N>` attributes (not `async_task_id`)
- IR after doTaskIdPropagate converts `ttg.partition` to `async_task_id` annotations
- To check partition assignments: look at IR between `NVGPUPartitionSchedulingMeta` and `NVGPUWarpSpecialization` dump sections
- Build: see `devmate/step1.txt`
- To run a single pass: `triton-opt --nvgpu-partition-scheduling-meta="merge-epilogue-to-computation=true" input.mlir`
- To enable debug: add `-debug-only=tritongpu-partition-scheduling`
- To add stack traces on specific ops: instrument `setPartition()` in `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/Partition.cpp`

## Key Concepts
- `PartitionSchedulingMeta` assigns `ttg.partition` attributes → `doTaskIdPropagate` converts to `async_task_id`
- Pointer-typed tensors (`!tt.ptr<T>`) should not be cross-partition
