# TMA output stores

Scope: the GEMM output epilogue under Meta WS (`TRITON_USE_META_WS=1`) on Blackwell. Covered here:
- which partition issues the store;
- the SMEM staging ring that feeds the TMA store;
- how store waits are drained.

Subtiling choice (`EPILOGUE_SUBTILE`) is covered in [epilogue.md](epilogue.md). Extra epilogue inputs are covered in `epilogue_prologue_tma_loads.md`.

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| `early_tma_store_lowering` | compile option | None (= True) / False | True | P0 | Kernels with TMA stores. | — |
| `tma_store_pipelining` | compile option | None (= True) / False | True | P0 | Kernels with TMA stores. | — |

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| `separate_epilogue_store=True` with a pointer `tl.store` epilogue (no TMA store) | duplicate config | The epilogue-store category contains only `DescriptorStoreOp`, `AsyncTMACopyLocalToGlobalOp` and `TMAStoreTokenWaitOp`. |
| `TRITON_WS_STAGING_COPIES` at `EPILOGUE_SUBTILE=1` | duplicate config | Phase 3.7 only deepens staging groups that hold 2 or more subtile buffers. |
| `TRITON_WS_STAGING_COPIES` with `two_ctas=True` | silently ignored | Ordinary 2-CTA TMA output staging is kept at one copy. |
| 2-CTA TMA output staging whose correctness floor needs more than one copy | compile error | The planner emits "2CTA ordinary TMA output staging requires one copy" and marks the plan invalid. |
| A staging copy count, for staging drained in the same partition, that does not divide the subtile count | duplicate config | The planner skips that depth: store waits drain a fixed in-flight count, so same-slot stores must be exactly K apart in issue order. |

## Mechanism

- **Store partition placement:** changes, with `separate_epilogue_store`, the partition layout to create an `epilogue_store` partition and schedule TMA stores, plus any op whose input comes from that partition, into it; without it, stores follow the epilogue target (the `epilogue` partition, or the merged partition when a `merge_epilogue*` attribute is set), and per-loop `tt.separate_epilogue_store` / `tt.merge_epilogue` attributes override the pass options (`PartitionSchedulingMeta.cpp`, `createPartitionLayout` and `schedulePostLoopOps`) -> relieves by moving the store wait and staging handshake off the partition that drains TMEM and does epilogue math -> costs one more partition (1 warp at minimum producer register budget) plus a cross-partition staging channel with full/empty barriers.
- **Every annotated loop is partitioned on its own:** partition scheduling separately schedules every `scf.for` or `scf.while` carrying `warp_specialize`, inner loop first, including one nested inside another annotated loop (`PartitionSchedulingMeta.cpp`, `runOnOperation`) -> costs: if a persistent GEMM's inner K loop also carries `warp_specialize=True`, it is partitioned first, and the outer tile loop's epilogue placement works around an already-partitioned loop.
- **Staging ring and store waits (shared mechanics in [epilogue.md](epilogue.md#mechanism)):** changes each TMA store to be fed from a staging ring of BLOCK_M x BLOCK_N / EPILOGUE_SUBTILE x output element size per copy (one store's subtile buffers fused), whose copy count K Phase 3.7 grows up to `min(num_stages, TRITON_WS_STAGING_COPIES)` within the SMEM budget; same-partition drain waits with `cp.async.bulk.wait_group(K-1)` and K must divide the subtile count S, different-partition drain uses mbarrier rotation with any K (`WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` and `getStagingCopiesCap`; `WSTMAStoreLowering.cpp`, planned pending count = K-1; background: [TMAStoreWaitPipeline.md](../TMAStoreWaitPipeline.md), sections "Phase 3.5: TMA Staging Fusion" and "Phase 3.7: Epilogue Group Copy Increase") -> relieves, with K > 1, by letting the next subtile's `local_store` proceed while earlier TMA stores drain -> costs K x the subtile staging size of SMEM, from the same budget as the operand ring.
- **`early_tma_store_lowering` and `tma_store_pipelining`:** the first (default on) sets `ttg.early_tma_store_lowering`, so TMA stores are lowered to staging allocations before the WS memory planner runs and the planner can size them (`third_party/nvidia/backend/compiler.py`, `make_ttgir`; `WSTMAStoreLowering.cpp`); the second (default on) is passed to the WS pass and annotates, validates and reorders TMA store waits so staging slots can be reused (`doAnnotateTMAStoreWaits`, `doTMAStoreWaitReorder`) -> relieves staging reuse -> costs nothing when off except the reuse ([TMAStoreWaitPipeline.md, "Wait Annotation and Reordering Pipeline"](../TMAStoreWaitPipeline.md#wait-annotation-and-reordering-pipeline)).

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
