# Tiles and stages (autoWS GEMM, sm100)

Scope:
- the tile shape (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`);
- `num_stages`;
- how operand ring depth comes out of the memory planner;
- which tile/stage configs are illegal, silently changed, or duplicates.

Budget numbers are in [resources.md](resources.md). Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` (`WS/`) unless they start at the repository root.

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| `BLOCK_M` | kernel constexpr (M extent of the `tl.dot` tile) | power of 2, >= 64 | — | P0 | Every tcgen05 GEMM. A power of 2 below 64 gets no tcgen05 MMA and no gemm partition. | — |
| `BLOCK_N` | kernel constexpr (N extent of the `tl.dot` tile) | power of 2, multiple of 8 | — | P0 | Every tcgen05 GEMM. Otherwise there is no tcgen05 MMA. | Under `two_ctas`, `BLOCK_N/2` >= 16. |
| `BLOCK_K` | kernel constexpr (K step of the inner loop) | power of 2, >= 256 / operand bit width | — | P0 | Every tcgen05 GEMM. Below 256 / bit width (16 bf16, 32 fp8) there is no tcgen05 MMA. | — |
| `num_stages` | compile option | int >= 1 | 3 | P0 | Every WS loop. Below 1 is a compile error; above the depth that fits the SMEM budget no operand copies are added. | — |
| `flatten` | `tl.range` kwarg | False | False | P0 | `flatten=False` only: `flatten=True` drops warp specialization. | — |
| `num_stages` (per loop) | `tl.range` kwarg | int >= 1 | kernel `num_stages` | P1 | `scf.for` loops only; silently ignored on a dynamic or CLC `while`. | — |
| `loop_unroll_factor`, `disable_licm` | `tl.range` kwargs | int / bool | None / False | P2 | `loop_unroll_factor` on a persistent WS loop is a compile error. `disable_licm` is not WS-specific. | — |

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| `BLOCK_M` a power of 2 below 64 (e.g. 32) | silently ignored | MMAv5 needs a power-of-2 M that is a multiple of 64, so the dot lowers to mma.sync and autoWS has no gemm partition. |
| `BLOCK_N` not a power of 2 | silently ignored | Same check: N must be a power of 2 and a multiple of 8, so there is no tcgen05. |
| `BLOCK_K` < 256 / operand bit width (16 for bf16, 32 for fp8) | silently ignored | No tcgen05: "If k size is smaller than the native mma size, we cannot use MMA". |
| fp32 × fp32 dot with `input_precision="ieee"` | silently ignored | No tensor-core MMA: `supportMMA` rejects fp32 operands. |
| `BLOCK_M=128` with `data_partition_factor=2` | duplicate config | Runs as dp=1. |
| `BLOCK_M=64` with `data_partition_factor=2` and `BLOCK_N` < 256 | compile error | Neither an M slice >= 64 nor an N slice >= 128 exists, so `computePartitionScheme` fails and the pass signals failure. |
| `tl.dot(two_ctas=True)` with `BLOCK_M` < 64 | silently ignored | Covered by the first row: no tcgen05 at all. |
| `TRITON_WS_SMEM_PLAN_SEARCH=1` with no K loop (K == BLOCK_K as a compile-time constant) and more tiles than SMs | wrong answer | Plan search keeps TMEM accumulator copies at 1, bypassing the per-iteration accumulator multi-buffering. |
| `loop_unroll_factor` >= 2 on a persistent warp-specialized loop | compile error | "SMEM circular reuse group is ill-formed": both the unrolled loop and the remainder loop keep `tt.warp_specialize`. |
| `flatten=True` | silently ignored | Flattening drops warp specialization from the loop. |
| `num_stages` < 1 | compile error | "nvgpu-warp-specialization requires num-stages >= 1". |
| Annotated `numCopies` below the buffer's cross-stage depth | hangs | The planner keeps the annotation with only a warning, and a depth below the cross-stage floor deadlocks. |

## Mechanism

- **Tile shape (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`):** changes the A slot (`BLOCK_M × BLOCK_K`) and B slot (`BLOCK_N × BLOCK_K`) in the operand element type, the TMEM columns per accumulator copy (`BLOCK_N × ceil(BLOCK_M/128)` for fp32, hence the accumulator copy count; [resources.md](resources.md)), and the epilogue tile, with `EPILOGUE_SUBTILE` ([epilogue.md](epilogue.md)); SMEM is sized from these compile-time shapes alone (`WS/WSMemoryPlanner.cpp`, `getSmemAllocSizeBytes`), so a smaller problem never makes a config fit.
- **MMA instruction shape:** M = 128 when `BLOCK_M` >= 128, else 64; N = `min(BLOCK_N, 256)`; K = 256 / operand bit width; larger tiles issue several MMAs (`lib/Dialect/TritonGPU/Transforms/Utility.cpp`, `mmaVersionToInstrShape`).
- **`num_stages`:** changes the planner's `numBuffers` cap: output staging is raised first, up to that cap (`increaseFusedEpilogueCopies`), then Phase 4 adds one copy at a time to the innermost operand buffers (TMA-fed first, then non-TMA, in first-use order) until the next copy would exceed the budget or `num_stages` ([SmemAllocationDesign.md, "Phase 4: Iterative Copy Increase"](../SmemAllocationDesign.md#phase-4-iterative-copy-increase), [TMAStoreWaitPipeline.md, "Phase 3.7: Epilogue Group Copy Increase"](../TMAStoreWaitPipeline.md#phase-37-epilogue-group-copy-increase)) -> relieves: deeper operand rings let the load partition run further ahead of the MMA -> costs: operand copies and staging compete for the same SMEM, and deeper rings add barrier slots to the auxiliary reservation (`estimateAuxiliarySmemBytes` scales barrier bytes by depth).
- **Resulting operand depth:** depth = the largest d <= `num_stages` such that floors + staging + d·(A slot + B slot) fits the budget, plus possibly one more copy of the operand used first; with equal A/B slots the two depths differ by at most 1, with unequal slots the smaller operand keeps rising after the larger stops, so the gap can be larger.
- **Correctness floors:** can force copies above `num_stages`; the cross-stage depth is never given back; the copy-safety floor (in particular, under `data_partition_factor` > 1 with `disallow_acc_multi_buffer`, every TMA operand of the MMA is floored at `num_stages`; `getStaticSmemCopySafetyFloor`) is given back when it does not fit (`relaxCopySafetyFloorToBudget`).
- **Annotated channel copies:** the unannotated planner raises A and B in lockstep, so A/B depths more than ±1 apart at equal slot sizes need annotated channel copies (`tl.dot(attrs={"channels": ...})`), never below the cross-stage depth.

## Interactions

See [interactions.md](interactions.md). Pairs these rules depend on:
- `BLOCK_M` × `data_partition_factor`
- `BLOCK_M` × `two_ctas`
- `num_stages` × `TRITON_WS_STAGING_COPIES`
- `num_stages` × `EPILOGUE_SUBTILE` (same-partition store)
- `num_stages` × `disallow_acc_multi_buffer` (dp > 1)
- `BLOCK_M`/`BLOCK_N` × accumulator copies (TMEM)
- `num_stages` × CLC/dynamic `while` (loop `num_stages` inert)

## Hazards

None in [hazards.md](hazards.md). Reported but not yet located in code (row in [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause)): single-copy register copy of a multi-consumer operand, with few K tiles.

## Not yet established

Tracked in [TRACKER.md](TRACKER.md).

- Exactly how `BLOCK_M=32` fails under autoWS. The fallback to mma.sync is in code; the crash in code partitioning has been observed but not traced.
- When extra stages help at small `K` (one or two K tiles). The persistent producer runs ahead across tiles, so this cannot be read from code.
- Which stage counts suit 2-CTA `BLOCK_K=128, BLOCK_N=256`.
- When asymmetric A/B depth pays; observed to depend on whether the deeper operand can actually run ahead.
- Whether larger `BLOCK_N` or `BLOCK_K=128` beats deeper rings at smaller `BLOCK_K` for a given shape.
- Whether deeper rings hurt saturated large-M shapes.
