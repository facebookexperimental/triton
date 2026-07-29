# Dynamic Persistent AutoWS Support Gaps

**Files**: `PartitionSchedulingMeta.cpp`, `WSAtomicBroadcast.cpp`,
`WarpSpecialization.cpp`, `WSDataPartition.cpp`, `WSSpecialize.cpp`,
`WSBuffer.cpp`, `WSCodePartition.cpp`, and TritonGPU's
`WarpSpecialization/Partition.{h,cpp}`.

## Scope

This document describes the remaining work for making a non-countable dynamic
persistent `scf.while` the loop selected for automatic warp specialization.
The target frontend shape is:

```python
sched = tl.DynamicPersistent1DScheduler.initialize(...)
while tl.condition(
    sched.is_valid(),
    warp_specialize=True,
    data_partition_factor=...,
):
    ... one output tile ...
    sched = sched.advance()  # scalar atomic tile claim
```

This is distinct from the dynamic persistent path that works today. The current
GEMM test marks the nested K `scf.for` with `warp_specialize=True`; the outer
dynamic `scf.while` is cloned as surrounding control flow after partitions have
already been assigned from that inner loop.

## What Works Today

The existing inner-loop path supports dynamic persistent GEMM on Hopper and
Blackwell:

1. `PartitionSchedulingMeta` assigns partitions from the annotated K
   `scf.for`.
2. Task-id propagation extends those assignments over the enclosing
   `scf.while`.
3. `doDynamicTileBroadcast` retags the loop-carried scalar atomic to one owner
   partition and broadcasts its result through an SMEM channel to every
   partition.
4. Code specialization clones the persistent while for each partition.
5. While-aware accumulation counters rotate channel buffers and mbarrier phases
   across persistent iterations.

The following pieces already understand `scf.while` and are not the primary
blocker:

- task-id propagation;
- data partition slicing and while-carried result updates;
- physical warp-group cloning (`SpecializeWhileOp`);
- channel discovery and code partitioning;
- accumulation-counter threading through the before/after regions;
- direct-while channel buffer index and phase calculation;
- run-once scalar atomic and CLC result broadcast.

Existing coverage includes:

- `test_tutorial09_matmul_tma_dynamic_persistent_while_loop_warp_specialize`
  with `EPILOGUE_SUBTILE` 1, 2, and 4;
- `ws_atomic_broadcast_transform.mlir`, including broadcast depth 2;
- `ws_atomic_broadcast_reject.mlir` for graceful rejection of an unsupported
  scatter atomic;
- simple while task-id, data-partition, and code-partition cases in
  `ws_while_loop_autows.mlir`.

## Partition Assignment Foundation

`PartitionSchedulingMeta` now discovers annotated `scf::ForOp` and supported
ordered-subset-carry `scf::WhileOp` loops. Its categorization, cross-iteration
tracing, partition propagation, schedule optimization, and serialization APIs
use `LoopLikeOpInterface`, while nested software-pipelined K-loops remain
`scf::ForOp`.

The pass and its helpers need a loop-body abstraction over
`LoopLikeOpInterface`:

| Concept | `scf.for` | `scf.while` |
|---|---|---|
| scheduled body | body region | after region |
| carried body arguments | body args after the IV | all after-region args |
| next-iteration values | body `scf.yield` | after-region `scf.yield` |
| condition forwarding | implicit | before-region `scf.condition` args |
| induction variable | explicit body arg 0 | none |

The partition scheduler accepts direct, unique, non-empty, order-preserving subsets of
before-region arguments. It maps after-region arguments back through
`scf.condition` to the matching yield slots and stops scheduled-body traversal
for condition-only slots. Empty, reordered, duplicate, and computed forwarding retain
a documented safe no-op behavior.

### Implemented PSM changes

- Collect annotated `scf::ForOp` and `scf::WhileOp` loops.
- Generalize `getInitialSchedule`, `OpCategorizer`, MMA backward-slice
  collection, partition propagation, schedule optimization, and serialization.
- Discover nested K `scf.for` loops from the while after-region.
- Trace carried definitions and users through `yield -> before arg ->
  condition arg -> after arg` without the for-loop induction-variable offset.
- Assign partitions to before-region condition computation and terminators so
  every specialized partition evaluates the same broadcast tile ID.
- Generalize `Partition`, `PartitionSet::fromLoop`, `serialize`,
  `swapPartitions`, and the `iterate*` helpers.
- Keep the existing `scf.for` behavior unchanged.

Focused lit coverage validates direct outer-while scheduling, schedule
round-tripping, CLC-shaped ordered-subset carry, condition/task replication,
warp-budget cleanup, and graceful rejection of reordered or computed
condition forwarding.

The atomic-broadcast half of the outer-while path is now also validated in
isolation (`ws_atomic_broadcast_from_psm.mlir`): starting from an
**unpartitioned** dynamic-persistent GEMM `scf.while` with a scalar
`tt.atomic_rmw` tile claim, the chain `nvgpu-partition-scheduling-meta ->
nvgpu-test-taskid-propagate -> nvgpu-test-ws-atomic-broadcast` transforms the
claim into run-once-in-owner + broadcast-to-all. **Key finding:** no PSM change
was needed to feed the broadcast — task-id propagation already flows the full
partition union backward from the loop-carried value (used by every partition)
onto the atomic, so `classifyAtomic` sees `taskIds.size() == allParts.size()`
and classifies it `Transform`. The owner resolves to the TMA-load partition via
`getOwnerPartition`, exactly as intended. (`nvgpu-test-ws-atomic-broadcast` is a
new thin test pass wrapping `doDynamicTileBroadcast`, mirroring
`nvgpu-test-taskid-propagate`; the step has no standalone pass otherwise.)

The same fixture now continues through the production warp-specialization path,
validating channel/barrier synthesis, while accumulation counters, physical
specialization, and depth-2 broadcast slot/phase rotation for a full GEMM while
body. The unified `DynamicPersistent1DScheduler` outer-while path is also
correct on Blackwell. Hopper runtime and the CLC sibling remain to be validated.

## Cleanup and Bailout Foundation

PSM's local `dropWarpSpec` helper strips loop metadata from both `scf.for` and
`scf.while`, including:

- `tt.warp_specialize`;
- `ttg.partition.stages`;
- `ttg.partition.types`;
- `ttg.warp_specialize.tag`.

It also removes op-level partition metadata, so warp-budget rejection and
invalid condition forwarding cannot leave a half-specialized function.

The post-scheduling dead-op cleanup treats `scf.while` as structural control
flow and never erases it as a use-empty, single-result operation.

The physical `NVGPUWarpSpecialization` fallback scan recognizes annotated
`scf.for` and `scf.while` loops when no partition/task attributes exist.

## Software-Pipeline Boundary

The non-countable outer while should not be software-pipelined: it has no static
trip count, and the atomic broadcast channel provides cross-tile producer
run-ahead. The nested K `scf.for` remains the software-pipelined loop.

Consequently, full while support is not required in `AssignLatencies`,
`ScheduleLoops`, or the software-pipeline expander. The required audit is
narrower:

- preserve the inner K-loop schedule while the outer while is partitioned and
  cloned;
- post-WS loop-schedule preprocessing recognizes an annotated outer
  `scf.while` and marks its already-staged innermost `scf.for` loops for
  schedule completion;
- do not interpret `tt.num_stages` on the outer dynamic while as a request to
  software-pipeline that while.

## Atomic Broadcast Restrictions

`doDynamicTileBroadcast` is ready for the standard scheduler shape, but it is
intentionally narrow. A replicated atomic is transformable only when it is:

- scalar (not a tensor/scatter atomic);
- directly yielded as a carried value of an enclosing `scf.while`;
- mapped to every partition, not a strict subset;
- suitable for ownership by one partition and broadcast to the others.

Any other replicated atomic causes a transactional, graceful rejection of
AutoWS for the function. This means outer-while support will not automatically
cover arbitrary work queues, transformed/cast atomic results, or bodies with
unrelated replicated atomics.

The owner-selection heuristic prefers the partition containing a TMA load and
otherwise uses the lowest partition ID. That is correct for the current GEMM
shape but needs validation for dynamic persistent kernels without TMA loads.

Broadcast depth defaults to one, which keeps partitions in lockstep at the tile
claim. Depth greater than one is covered by lit but lacks runtime correctness
and performance coverage.

## 2-CTA Gap

Dynamic persistence with a 2-CTA MMA requires one logical tile claim per CTA
cluster. The current scheduler seeds work from physical program IDs, and the
atomic broadcast synchronizes warp partitions within a CTA. It does not define:

- physical-CTA to logical-cluster tile mapping;
- which CTA owns the cluster's atomic claim;
- how the claimed tile ID is distributed to the peer CTA;
- counter initialization and termination accounting per cluster.

Therefore dynamic 2-CTA should be treated as a separate feature after basic
outer-while AutoWS works. Reusing the intra-CTA atomic broadcast without a
cluster-level ownership protocol would allow both CTAs to advance the global
counter independently.

## Frontend and Configuration Gaps

Because the current workaround annotates the inner K loop, outer-while options
do not have a complete end-to-end contract for dynamic persistence. This
includes:

- `data_partition_factor`;
- `merge_epilogue`, `merge_epilogue_to_computation`, and `merge_correction`;
- `separate_epilogue_store`;
- SMEM/TMEM allocation options;
- generated subtiled regions;
- register-budget and ping-pong combinations.

`num_stages` on the dynamic outer while should remain inert or produce a clear
frontend diagnostic; stages belong on the nested K loop.

## Test Gaps

The unified `DynamicPersistent1DScheduler` and `ClcTileScheduler` now validate
an annotated outer while with an unannotated K loop on Blackwell, including
numerical correctness and the absence of incomplete pipeline-stage diagnostics.
Cross-architecture and feature-combination coverage remains open.

Required coverage:

1. **PSM lit** [done]: annotated full-carry and CLC-shaped ordered-subset
   `scf.while` loops with a nested MMA or serialized partition schedule;
   check partitions on condition, tile mapping, loads, MMA, epilogue, atomic,
   and while terminators. (`ws_while_loop_autows.mlir`.)
2. **Negative PSM lit** [done]: reordered/computed condition forwarding is skipped
   without partial metadata. (`ws_while_loop_autows.mlir`.)
3. **Atomic integration lit** [done]: PSM + task propagation + atomic broadcast
   from an initially unpartitioned outer while, extended through code
   partitioning, physical specialization, depth-2 slot/phase rotation, and
   nested K-loop rescheduling. (`ws_atomic_broadcast_from_psm.mlir`.)
4. **Unified E2E** [Blackwell done]: `DynamicPersistent1DScheduler` and
   `ClcTileScheduler` with outer `tl.condition(..., warp_specialize=True)` and
   an unannotated K loop. The dynamic scheduler remains pending on Hopper.
5. **Feature E2E**: epilogue subtiles, DP=2, separate epilogue store, generated
   subtiled regions, and broadcast depths greater than one.
6. **Bailout E2E/lit**: scatter, strict-subset, non-carried, and unrelated
   replicated atomics leave a compilable non-WS kernel.
7. **Hopper and Blackwell**: CLC sibling correctness is covered on Blackwell;
   unified dynamic atomic correctness remains to be run on Hopper.
8. **Performance**: compare static persistent, dynamic inner-loop AutoWS,
   dynamic outer-loop AutoWS at depth 1, and tuned broadcast depth.

## Recommended Implementation Order

1. Complete unified dynamic E2E coverage on Hopper.
2. Add feature combinations.
3. Address dynamic 2-CTA as a separate cluster-level design.

## Definition of Done

Basic dynamic outer-loop AutoWS is complete when a unified scheduler kernel can
place `warp_specialize=True` only on its dynamic `tl.condition`, leave the K
loop unannotated, and satisfy all of the following:

- PSM serializes a valid partition schedule on the `scf.while`;
- exactly one atomic tile claim executes per logical CTA iteration;
- all partitions evaluate the same current and terminating tile IDs;
- physical `ttg.warp_specialize` regions are emitted;
- buffer slots and mbarrier phases rotate correctly across persistent tiles;
- correctness passes for epilogue subtiles on Hopper and Blackwell;
- unsupported while/atomic shapes cleanly fall back without residual metadata.
