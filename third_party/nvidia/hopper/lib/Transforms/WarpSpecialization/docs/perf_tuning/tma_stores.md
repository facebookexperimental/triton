# TMA output stores

Scope: the GEMM output epilogue under Meta WS (`TRITON_USE_META_WS=1`) on Blackwell. Covered here:
- which partition issues the store;
- the SMEM staging ring that feeds the TMA store;
- how store waits are drained.

Subtiling choice (`EPILOGUE_SUBTILE`) is covered in [epilogue.md](epilogue.md). Extra epilogue inputs are covered in `epilogue_prologue_tma_loads.md`.

## Knobs

| Knob | How set | One line | Existing doc |
|---|---|---|---|
| TMA store | `desc.store(...)` on a tensor descriptor (or a pointer store promoted by `auto_tma` in a WS kernel) | The only store form the partition scheduler treats as an epilogue store. | [MemoryLowering.md](../MemoryLowering.md), section "TMA Store Lowering" |
| `separate_epilogue_store` | `tl.range(..., separate_epilogue_store=True)` (loop attribute `tt.separate_epilogue_store`; also a pass option) | Gives TMA stores their own 1-warp `epilogue_store` partition. | [PartitionSchedulingMeta.md](../PartitionSchedulingMeta.md), sections "Epilogue Terminology" and "Target Partition Layouts" |
| `merge_epilogue` / `merge_epilogue_to_computation` | `tl.range(...)` loop attributes `tt.merge_epilogue` / `tt.merge_epilogue_to_computation` | Puts the post-loop epilogue in an existing partition instead of a separate `epilogue` partition. | [PartitionSchedulingMeta.md](../PartitionSchedulingMeta.md), section "Target Partition Layouts" |
| `TRITON_WS_STAGING_COPIES` | env var | Caps the depth of the TMA-store staging ring that the planner may grow to. It also caps other Phase 3.7 epilogue groups and, unless `TRITON_WS_TMA_REDUCE_STAGING_COPIES` is set, TMA-reduce staging. | [TMAStoreWaitPipeline.md](../TMAStoreWaitPipeline.md#triton_ws_staging_copies-capping-the-phase-37-target), section "`TRITON_WS_STAGING_COPIES`: capping the Phase 3.7 target" |
| `TRITON_WS_TMA_REDUCE_STAGING_COPIES` | env var | Copy target for TMA-reduce staging. It is independent of `num_stages`. | [TMAStoreWaitPipeline.md](../TMAStoreWaitPipeline.md), section "Phase 3.7: Epilogue Group Copy Increase" |
| `tma_store_pipelining` | compile option, on by default | Annotates and reorders TMA store waits so that staging slots can be reused. | [TMAStoreWaitPipeline.md](../TMAStoreWaitPipeline.md), section "Wait Annotation and Reordering Pipeline" |

## Never try (any problem)

| Condition | Status | Why | Code ref |
|---|---|---|---|
| `separate_epilogue_store=True` with a pointer `tl.store` epilogue (no TMA store) | duplicate config | The epilogue-store category contains only `DescriptorStoreOp`, `AsyncTMACopyLocalToGlobalOp` and `TMAStoreTokenWaitOp`. A pointer store goes to the partition of the loop's other stores, or to the default partition. | `PartitionSchedulingMeta.cpp`, `isEpilogueStoreOp`; the unreachable-store fix-up in `runOnOperation` |
| `TRITON_WS_STAGING_COPIES` at `EPILOGUE_SUBTILE=1` | duplicate config | Phase 3.7 only deepens staging groups that hold 2 or more subtile buffers. | `WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |
| `TRITON_WS_STAGING_COPIES` with `two_ctas=True` | silently ignored | Ordinary 2-CTA TMA output staging is kept at one copy. | `WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` (2-CTA branch) |
| 2-CTA TMA output staging whose correctness floor needs more than one copy | compile error | The planner emits "2CTA ordinary TMA output staging requires one copy" and marks the plan invalid. | `WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |
| A staging copy count, for staging drained in the same partition, that does not divide the subtile count | duplicate config | The planner skips that depth: store waits drain a fixed in-flight count, so same-slot stores must be exactly K apart in issue order. | `WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies`; [TMAStoreWaitPipeline.md](../TMAStoreWaitPipeline.md), section "Phase 3.7: Epilogue Group Copy Increase" |

## Mechanism

**Store partition placement.**
- **What changes:** with `separate_epilogue_store`, partition layout creates an `epilogue_store` partition and schedules TMA stores, plus any op whose input comes from that partition, into it. Without it, stores follow the epilogue target: the `epilogue` partition, or the merged partition when a `merge_epilogue*` attribute is set. Per-loop `tt.separate_epilogue_store` / `tt.merge_epilogue` attributes override the pass options.
- **What it relieves:** the store wait and staging handshake move off the partition that drains TMEM and does epilogue math.
- **What it costs:** one more partition (1 warp at minimum producer register budget), plus a cross-partition staging channel with full/empty barriers.
- Code: `PartitionSchedulingMeta.cpp`, `createPartitionLayout` and `schedulePostLoopOps`.

**Every annotated loop is partitioned on its own.**
- **What changes:** partition scheduling separately schedules every `scf.for` or `scf.while` carrying `warp_specialize`, inner loop first, including one nested inside another annotated loop.
- **What it costs:** if a persistent GEMM's inner K loop also carries `warp_specialize=True`, it is partitioned first, and the outer tile loop's epilogue placement works around an already-partitioned loop.
- Code: `PartitionSchedulingMeta.cpp`, `runOnOperation`.

**Staging ring and store waits** (shared mechanics in [epilogue.md](epilogue.md#mechanism)).
- **What changes:** each TMA store is fed from a staging ring of BLOCK_M x BLOCK_N / EPILOGUE_SUBTILE x output element size per copy (one store's subtile buffers fused). Phase 3.7 grows its copy count K up to `min(num_stages, TRITON_WS_STAGING_COPIES)` within the SMEM budget. Same-partition drain: each wait is `cp.async.bulk.wait_group(K-1)` and K must divide the subtile count S. Different-partition drain: mbarrier rotation, any K.
- **What it relieves:** with K > 1, the next subtile's `local_store` proceeds while earlier TMA stores drain.
- **What it costs:** K x the subtile staging size of SMEM, from the same budget as the operand ring.
- Code: `WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` and `getStagingCopiesCap`; `WSTMAStoreLowering.cpp` (planned pending count = K-1). Background: [TMAStoreWaitPipeline.md](../TMAStoreWaitPipeline.md), sections "Phase 3.5: TMA Staging Fusion" and "Phase 3.7: Epilogue Group Copy Increase".

## Interactions

See [interactions.md](interactions.md). Pairs this file expects to hold:
- **TMA store x 2-CTA:** 2-CTA requires a TMA store, and its staging is held at one copy.
- **TMA store x epilogue subtiling:** staging size per copy is BLOCK_M x BLOCK_N / ES, and K must divide ES when the store is drained in the same partition.
- **TMA store x CLC / dynamic persistence:** CLC composes with `separate_epilogue_store`.
- **TMA store x data partitioning:** each data partition stores its own tile through its own staging slot.

## Hazards

- [Same-partition TMA staging floor that does not divide the subtile count](hazards.md#same-partition-tma-staging-floor-that-does-not-divide-the-subtile-count)

## Not yet established

Tracked in [TRACKER.md](TRACKER.md).

- The size of the gain from a separate store partition, and whether 2-CTA kernels see it as 1-CTA kernels do.
- Whether `separate_epilogue_store` matters for fused 1-CTA kernels, given that it ties on plain 2-CTA addmm.
- Folding the store into the epilogue warps with a double-buffered C tile and deferring the final store wait past the persistent loop, as hand-written kernels do.
- Whether a TMA store of an extra epilogue output pays when SMEM is tight.
- Whether merging the store partition into the default partition matters, beyond fewer barrier stalls, once the dominant stall is fixed.
