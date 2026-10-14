# Epilogue: subtiling, separate store, store placement (autoWS GEMM, sm100)

Scope:
- epilogue subtiling: the kernel splits the accumulator into `ES` column slices before storing, e.g. the `EPILOGUE_SUBTILE` constexpr in `python/test/unit/language/test_autows_addmm.py`;
- the output-staging ring the planner builds for TMA stores;
- where the store runs (its own partition or the epilogue partition).

Budget numbers are in [resources.md](resources.md). Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` (`WS/`) unless they start at the repository root.

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| `separate_epilogue_store` | `tl.range` kwarg | bool | False | P0 | Kernels with a TMA output store. With a pointer `tl.store` it is a duplicate config. | A TMA output store. |
| Epilogue subtiling (ES) | kernel pattern (accumulator split into ES column slices) | 1 / 2 / 4 / 8 | 1 | P0 | Every kernel with a subtiled epilogue store. | A side-input slice under 128 B at ES >= 2 needs #3939 (OPEN). |
| `TRITON_WS_STAGING_COPIES` | env var | int K >= 1 | unset (no cap) | P0 | 1-CTA, ES >= 2, TMA store. At ES=1 it is a duplicate config; under `two_ctas` it is silently ignored. | Legal ring depths for same-partition staging: #3571 (OPEN). |
| `TRITON_WS_TMA_REDUCE_STAGING_COPIES` | env var | int >= 1 | unset (min(`num_stages`, staging cap)) | P0 | Only kernels with a TMA-reduce output (`desc.atomic_add` / `store_reduce`). | A TMA-reduce output. |
| `merge_epilogue` | `tl.range` kwarg | bool | False | P1 | Kernels with a correction or reduction partition; otherwise epilogue ops already go to computation. | — |
| `generate_subtiled_region` | compile option, or env `TRITON_GENERATE_SUBTILED_REGION`‡ | bool | False | P1 | Only an ES >= 2 split; with no split there is nothing to wrap. | Not with dp=2 + a per-subtile scale (TRACKER: hang). |
| `merge_epilogue_to_computation`, `merge_correction` | `tl.range` kwargs | bool | False | P2 | Route epilogue or correction ops into computation; a GEMM without a correction partition has nothing for `merge_correction` to move. | — |

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| `separate_epilogue_store=True` with pointer (`tl.store`) epilogue stores | duplicate config | Only `descriptor_store`, `async_tma_copy_local_to_global` and TMA store waits count as epilogue stores; with none, no store partition is created. |
| `TRITON_WS_STAGING_COPIES` with `ES=1` | duplicate config | A staging group with fewer than 2 slices is never bumped, and Phase 4 never raises staging. |
| Staging copies under `two_ctas` | silently ignored | Staging is held at one copy: 2-CTA TMA stores have one completion phase per CTA-pair tile. |
| A correctness floor needing > 1 staging copy under `two_ctas` | compile error | "2CTA ordinary TMA output staging requires one copy; the correctness floor requires N". |
| Same-partition store (`separate_epilogue_store=False`) with a staging cap that does not divide `ES` | duplicate config | Rounded down to the largest divisor of `ES` that fits: the wait_group-drained ring is correct only when copies divide the slice count. |
| Staging cap above `num_stages` (store staging) | duplicate config | Capped at `num_stages`: the bump target is min(`num_stages`, cap); only TMA-reduce staging may exceed it. |

## Mechanism

- **`EPILOGUE_SUBTILE` (`ES`):** changes the staging footprint: the `ES` staging slices of one store (same descriptor, source accumulator and block) share one buffer id, so staging SMEM = `BLOCK_M × (BLOCK_N/ES) × output element size × copies`, one slice × copies rather than `ES` slices × copies (`fuseEpilogueWSBuffers`, `computeTotalSmem` in `WS/WSMemoryPlanner.cpp`; [TMAStoreWaitPipeline.md, "Phase 3.5: TMA Staging Fusion"](../TMAStoreWaitPipeline.md#phase-35-tma-staging-fusion)) -> relieves per-copy staging SMEM and per-slice epilogue registers -> costs more TMA stores and barrier rounds per tile, and with a same-partition store the legal copy counts become the divisors of the new `ES`.
- **Epilogue registers:** one slice's `tmem_load` holds `BLOCK_M × (BLOCK_N/ES) / (32 × warps)` 32-bit values per thread for an fp32 accumulator (warps = the loading partition's count); at `BLOCK_M=128, BLOCK_N=256` with 4 warps, an `ES=2` slice is 128 registers per thread and `ES=4` is 64; in the GEMM layouts the epilogue runs in the default region, which partition-warp halving never shrinks (see [warps_and_registers.md](warps_and_registers.md#mechanism)), so `ES` does not change the epilogue's warp count.
- **Staging copies:** changes staging depth, raised before operand copies (`increaseFusedEpilogueCopies` runs before Phase 4) up to min(`num_stages`, `TRITON_WS_STAGING_COPIES`) and the budget; with the store in its own partition any count is legal (slot and phase come from a continuous producer/consumer mbarrier rotation), in the epilogue partition only divisors of `ES` (the ring is drained by `cp.async.bulk.wait_group(K-1)`) ([TMAStoreWaitPipeline.md, "`K | S` cap for same-partition (wait_group-drained) staging"](../TMAStoreWaitPipeline.md#k--s-cap-for-same-partition-wait_group-drained-staging), [AccumulationCounters.md, "What Is an Accumulation Counter?"](../AccumulationCounters.md#what-is-an-accumulation-counter)) -> relieves: stores of consecutive slices or tiles can overlap -> costs operand ring depth, since staging is reserved first.
- **Store placement:** in `WS/PartitionSchedulingMeta.cpp` (`createPartitionLayout`), `merge_epilogue` / `merge_epilogue_to_computation` suppress the separate `epilogue` partition.
- **`separate_epilogue_store=True`:** changes placement by adding an `epilogue_store` partition (one warp after `optimizePartitionNumWarps`) when TMA epilogue stores exist -> relieves: frees staging depth from the divisor rule and takes TMA issue out of the epilogue partition -> costs one more worker warp, which can cost a whole warp group of registers (see [warps_and_registers.md](warps_and_registers.md)), and with `ES` >= 2 and `TRITON_WS_STAGING_COPIES` unset, staging may take up to `num_stages` copies before the operands, which starves the B ring (see the rules below).
- **Plan search:** `generate_subtiled_region` wraps the `ES` copies in a `SubtiledRegionOp` ([SubtileOperator.md, "Compiler Option"](../SubtileOperator.md#compiler-option)); a `SubtiledRegionOp` makes the TMEM plan search fall back to the heuristic planner (`WS/WSMemoryPlanner.cpp`, `allocateTmemBuffersViaSearch`), so accumulator multi-buffering stays on for subtiled-region kernels even under `TRITON_WS_SMEM_PLAN_SEARCH=1`.

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
