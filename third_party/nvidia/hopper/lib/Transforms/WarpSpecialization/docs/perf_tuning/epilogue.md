# Epilogue: subtiling, separate store, store placement (autoWS GEMM, sm100)

Scope:
- epilogue subtiling: the kernel splits the accumulator into `ES` column slices before storing, e.g. the `EPILOGUE_SUBTILE` constexpr in `python/test/unit/language/test_autows_addmm.py`;
- the output-staging ring the planner builds for TMA stores;
- where the store runs (its own partition or the epilogue partition).

Budget numbers are in [resources.md](resources.md). Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` (`WS/`) unless they start at the repository root.

## Knobs

| Knob | How set | What it does | Existing doc |
|---|---|---|---|
| Epilogue subtile count `ES` | kernel source (accumulator split into `ES` slices of `BLOCK_M × BLOCK_N/ES`) | Number of staging slices per tile and the size of each `tmem_load` | [SubtileOperator.md, "Motivation"](../SubtileOperator.md#motivation) |
| `generate_subtiled_region` / `TRITON_GENERATE_SUBTILED_REGION` | compile option / env | Represents the subtiled epilogue as a `SubtiledRegionOp` | [SubtileOperator.md, "Compiler Option"](../SubtileOperator.md#compiler-option) |
| `separate_epilogue_store` | `tl.range(...)` kwarg | Moves TMA epilogue stores into their own one-warp partition | [PartitionSchedulingMeta.md, "Target Partition Layouts"](../PartitionSchedulingMeta.md#target-partition-layouts) |
| `merge_epilogue`, `merge_epilogue_to_computation` | `tl.range(...)` kwargs | Route non-store epilogue ops into an existing partition instead of a separate epilogue partition | [PartitionSchedulingMeta.md, "Tuning Knobs"](../PartitionSchedulingMeta.md#tuning-knobs) |
| `TRITON_WS_STAGING_COPIES` | env | Caps output-staging copies at min(`num_stages`, value). The same cap applies to other Phase 3.7 epilogue groups and, unless `TRITON_WS_TMA_REDUCE_STAGING_COPIES` is set, to TMA-reduce staging | [TMAStoreWaitPipeline.md, `TRITON_WS_STAGING_COPIES`](../TMAStoreWaitPipeline.md#triton_ws_staging_copies-capping-the-phase-37-target) |
| `TRITON_WS_TMA_REDUCE_STAGING_COPIES` | env | Copy target for TMA-reduce staging; may exceed `num_stages` | same |
| `early_tma_store_lowering` | compile option (on unless set to `False`) | Lowers TMA stores to staging allocations before the WS planner runs | [TMAStoreWaitPipeline.md, "Memory Planner: `buffer.tmaStaging` Handling"](../TMAStoreWaitPipeline.md#memory-planner-buffertmastaging-handling) |
| `tma_store_pipelining` | compile option (default on) | Annotates and reorders TMA store waits | [TMAStoreWaitPipeline.md, "Wait Annotation and Reordering Pipeline"](../TMAStoreWaitPipeline.md#wait-annotation-and-reordering-pipeline) |

## Never try (any problem)

| Condition | Status | Why | Code |
|---|---|---|---|
| `separate_epilogue_store=True` with pointer (`tl.store`) epilogue stores | duplicate config | Only `descriptor_store`, `async_tma_copy_local_to_global` and TMA store waits count as epilogue stores; with none, no store partition is created | `WS/PartitionSchedulingMeta.cpp`, `isEpilogueStoreOp`, `createPartitionLayout` |
| `TRITON_WS_STAGING_COPIES` with `ES=1` | duplicate config | A staging group with fewer than 2 slices is never bumped, and Phase 4 never raises staging | `WS/WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` ("only ... buffer(s) in group, skipping"), `allocateSmemBuffers` (Phase 4 never raises staging) |
| Staging copies under `two_ctas` | silently ignored | Staging is held at one copy: 2-CTA TMA stores have one completion phase per CTA-pair tile | `WS/WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |
| A correctness floor needing > 1 staging copy under `two_ctas` | compile error | "2CTA ordinary TMA output staging requires one copy; the correctness floor requires N" | `WS/WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |
| Same-partition store (`separate_epilogue_store=False`) with a staging cap that does not divide `ES` | duplicate config | Rounded down to the largest divisor of `ES` that fits: the wait_group-drained ring is correct only when copies divide the slice count | `WS/WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` (`K | S` check) |
| Staging cap above `num_stages` (store staging) | duplicate config | Capped at `num_stages`: the bump target is min(`num_stages`, cap); only TMA-reduce staging may exceed it | `WS/WSMemoryPlanner.cpp`, `getStagingCopiesCap`, `increaseFusedEpilogueCopies` |
| `TRITON_WS_STAGING_COPIES` or `TRITON_WS_TMA_REDUCE_STAGING_COPIES` with `smem_alloc_algo=0` | silently ignored | Both are read only by the default planner's Phase 3.7 | `WS/WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |

## Mechanism

**Staging footprint:**
- The `ES` staging slices of one store (same descriptor, source accumulator and block) share one buffer id. The footprint is one slice × copies, not `ES` slices × copies: `fuseEpilogueWSBuffers` and `computeTotalSmem` in `WS/WSMemoryPlanner.cpp`.
- Staging SMEM = `BLOCK_M × (BLOCK_N/ES) × output element size × copies`.

Mechanics: [TMAStoreWaitPipeline.md, "Phase 3.5: TMA Staging Fusion"](../TMAStoreWaitPipeline.md#phase-35-tma-staging-fusion).

**Staging depth.** Staging copies are raised before operand copies (`increaseFusedEpilogueCopies` runs before Phase 4), up to min(`num_stages`, `TRITON_WS_STAGING_COPIES`) and the budget. The legal copy counts depend on where the store runs:
- **Store in its own partition** (`separate_epilogue_store=True`): any count, because the slot and phase come from a continuous producer/consumer mbarrier rotation.
- **Store in the epilogue partition**: only divisors of `ES`, because the ring is drained by `cp.async.bulk.wait_group(K-1)`.

Mechanics: [TMAStoreWaitPipeline.md, "`K | S` cap for same-partition (wait_group-drained) staging"](../TMAStoreWaitPipeline.md#k--s-cap-for-same-partition-wait_group-drained-staging) and [AccumulationCounters.md, "What Is an Accumulation Counter?"](../AccumulationCounters.md#what-is-an-accumulation-counter).

**Registers.**
- One slice's `tmem_load` holds `BLOCK_M × (BLOCK_N/ES) / (32 × warps)` 32-bit values per thread for an fp32 accumulator, where warps is the count of the partition doing the load.
- Example: at `BLOCK_M=128, BLOCK_N=256` with 4 warps, an `ES=2` slice is 128 registers per thread and an `ES=4` slice is 64.
- In the GEMM layouts the epilogue runs in the default region, which partition-warp halving never shrinks (see [warps_and_registers.md](warps_and_registers.md#mechanism)), so `ES` does not change the epilogue's warp count.

**Store placement** (`WS/PartitionSchedulingMeta.cpp`, `createPartitionLayout`):
- `separate_epilogue_store` adds an `epilogue_store` partition (one warp after `optimizePartitionNumWarps`) when TMA epilogue stores exist.
- `merge_epilogue` / `merge_epilogue_to_computation` suppress the separate `epilogue` partition.

**Plan search.** A `SubtiledRegionOp` makes the TMEM plan search fall back to the heuristic planner (`WS/WSMemoryPlanner.cpp`, `allocateTmemBuffersViaSearch`). Accumulator multi-buffering therefore stays on for subtiled-region kernels even under `TRITON_WS_SMEM_PLAN_SEARCH=1`.

**Change → relief → cost:**
- **Raising `ES`:**
  - relieves: per-copy staging SMEM and per-slice epilogue registers;
  - costs: more TMA stores and barrier rounds per tile. With a same-partition store, the legal copy counts change to the divisors of the new `ES`.
- **Raising staging copies:**
  - relieves: lets stores of consecutive slices or tiles overlap;
  - costs: operand ring depth, since staging is reserved first.
- **`separate_epilogue_store=True`:**
  - relieves: frees staging depth from the divisor rule and takes TMA issue out of the epilogue partition;
  - costs: one more worker warp, which can cost a whole warp group of registers (see [warps_and_registers.md](warps_and_registers.md)).

## When to try it (depends on the problem)

- **IF** the epilogue uses pointer stores, **THEN** skip `separate_epilogue_store=True` variants (this removes them), **BECAUSE** they are duplicate configs of `False`.
- **IF** `ES=1` or `two_ctas=True`, **THEN** skip staging-copy variants, **BECAUSE** staging stays at one copy.
- **IF** `separate_epilogue_store=False`, **THEN** only try staging caps that divide `ES`, **BECAUSE** other caps round down to a divisor.
- **IF** two configs differ only in `ES` and the larger `ES` is what makes the config fit, **THEN** prefer the larger `ES` (this removes the smaller-`ES` variant that does not fit), **BECAUSE** total staging SMEM falls as `BLOCK_N/ES` shrinks at equal copies.
- **IF** total staging SMEM is compared across `ES` values, **THEN** compute it as slice × copies with copies bounded as above, **BECAUSE** it does not shrink monotonically with `ES`: copies can rise toward min(`num_stages`, `ES`) for a same-partition store, or `num_stages` for a separate store.

## Interactions

See [interactions.md](interactions.md). Pairs these rules depend on:
- `ES` × `separate_epilogue_store`
- `ES` × staging copies
- staging copies × `num_stages`
- staging copies × `two_ctas`
- `separate_epilogue_store` × pointer vs TMA store
- `ES` × `generate_subtiled_region`
- `generate_subtiled_region` × `TRITON_WS_SMEM_PLAN_SEARCH`
- `merge_epilogue` × `separate_epilogue_store`

## Hazards

None in [hazards.md](hazards.md). Reported but not yet located in code (rows in [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause)):
- rank-1 TMA bias loads in a subtiled epilogue;
- `data_partition_factor=2` with a subtiled region and per-slice scaling.

## Not yet established

Tracked in [TRACKER.md](TRACKER.md).

- Whether runtime is flat across `ES` once a config fits. This has been observed for plain GEMM but not shown in general.
- The best `ES` per tile and 2-CTA setting. Observed results are not shape-stable.
- Whether splitting the epilogue into slices cuts registers only when each slice's `tmem_load` is issued and consumed before the next one; observed, not traced in code.
- Whether applying row/column scales per slice instead of to the full tile is bitwise identical in general.
- Whether `early_tma_store_lowering=False` takes output staging out of the WS planner's accounting entirely. The TMA store lowering that creates staging allocations runs only when the attribute is set (`WS/WSTMAStoreLowering.cpp`, `NVGPUWSTMAStoreLoweringPass`); what happens downstream has not been checked.
- How many layout conversions each slice adds for non-TMA epilogue loads.
