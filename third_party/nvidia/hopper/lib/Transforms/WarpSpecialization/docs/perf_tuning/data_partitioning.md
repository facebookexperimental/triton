# Data partitioning (`data_partition_factor`, autoWS GEMM, sm100)

Scope: when to set `data_partition_factor` (dp) on the persistent tile loop of
a GEMM. This covers which dp configs are duplicates of dp=1 or fail to compile,
what dp trades, and the pruning that follows. How the pass slices the IR is in
[DataPartition.md](../DataPartition.md); budget numbers are in
[resources.md](resources.md).

Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` (`WS/`) unless they start at the repository root.

## Knobs

| Knob | How set | What it does | Existing doc |
|---|---|---|---|
| `data_partition_factor` | `tl.range(..., data_partition_factor=)` | Splits the accumulator and its consumers into dp consumer groups | [DataPartition.md, "Step 2: Compute Partition Scheme (`computePartitionScheme`)"](../DataPartition.md#step-2-compute-partition-scheme-computepartitionscheme) |
| `disallow_acc_multi_buffer` | `tl.range(..., disallow_acc_multi_buffer=True)` | Under Meta WS, floors every TMA-fed SMEM operand of the MMA at `num_stages` when dp > 1; it does not cap TMEM accumulator copies | [AutoWSAnnotations.md, "`disallow_acc_multi_buffer`"](../AutoWSAnnotations.md#disallow_acc_multi_buffer) |

## Never try (any problem)

| Condition | Status | Why | Code |
|---|---|---|---|
| `BLOCK_M=128`, dp=2 (any `BLOCK_N`) | duplicate config | Runs as dp=1, warning only. The M split is tried first. A 64-row slice would change the TMEM encoding's `blockM`, so the pass warns "skipping M-dimension data partitioning" and resets to no partitioning. It returns without trying the N split, even when `BLOCK_N/2` >= 128 | `WS/WSDataPartition.cpp`, `DataPartitionScheme::partitionIsCompatible`, `computePartitionScheme` |
| `BLOCK_M=64`, dp=2, `BLOCK_N` < 256 | compile error | No M slice >= 64 and no N slice >= 128 exists, so `computePartitionScheme` fails and the pass signals failure | `WS/WSDataPartition.cpp`, `computePartitionScheme`, `doDataPartition` |
| different `data_partition_factor` on two annotated loops of one function | silently ignored | Release builds keep the last loop's value. The check is an assert ("data partition factor mismatch") | `WS/WSDataPartition.cpp`, `NVGPUWSDataPartitionPass::runOnFuncOp` |
| a host `TensorDescType` kernel argument passed as a loop init arg | compile error | The descriptor slicing does not update loop iter-arg types; `doDataPartition` bails ("used as loop init arg; not supported yet") | `WS/WSDataPartition.cpp`, `doDataPartition` |
| MMAs that accumulate into one TMEM tile land in dp groups with no common partition | compile error | "warp specialization assigned accumulator-chained MMAs with no common partition"; concurrent accumulating MMAs would race | `WS/PartitionSchedulingMeta.cpp` (accumulator-chain backstop) |
| the warp estimate (default partition `num_warps` plus each other partition's minimum) exceeds 16 | silently ignored | Warp specialization is dropped for the whole function. "Warp budget exceeded. Skipping warp specialization." (debug log only). Under `two_ctas` this hangs instead (see [2cta.md](2cta.md#never-try-any-problem)). In the GEMM layouts both data-partition halves sit in the default region and the workers are 1 warp each, so dp=2 does not push a GEMM over the cap (`BLOCK_M=256`, dp=2, `num_warps=8` keeps WS with 12 warps) | `WS/PartitionSchedulingMeta.cpp` (`kMaxWarps`); `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`, `getMinWarpsForOp` |
| total TMEM above 512 columns | requires | Total TMEM must fit 512 columns; otherwise `OutOfResources("tensor memory")` is raised when the kernel is loaded, not at compile time | `python/triton/compiler/compiler.py`, `CompiledKernel._init_handles` (`max_tmem_size`) |
| `disallow_acc_multi_buffer` at dp=1 | duplicate config (reproduced) | Under Meta WS the SMEM planner reads the attribute only when `data_partition_factor` > 1; the TTGIR is identical apart from the attribute | `WS/WSMemoryPlanner.cpp`, `getStaticSmemCopySafetyFloor` |

## Mechanism

**What it changes.**
- **The accumulator is split into dp slices,** along M when `BLOCK_M/dp` >= 64 and the slice keeps the TMEM `blockM`. If an M split is tried and skipped (it would change `blockM`), partitioning is dropped without trying N; an N split (`BLOCK_N/dp` >= 128) is tried only when no M split was attempted. Each slice gets its own MMA and its own consumer (epilogue/computation) group, and the operand loads that feed the split dimension are sliced too ([DataPartition.md, "Algorithm"](../DataPartition.md#algorithm)).
- In practice the useful case is `BLOCK_M=256`, dp=2: two 128-row groups.
- **DP0 runs before DP1.** After slicing, the pass rebuilds straight-line blocks so the whole DP0 chain precedes DP1 (`WS/WSDataPartition.cpp`, `serializeDataPartitionedOps`).

**What it relieves.**
- Each consumer group handles `BLOCK_M/dp` rows, so the per-thread epilogue live set (TMEM → registers → SMEM) shrinks by dp at a fixed `num_warps`.
- A 256-row tile can be expressed without 2-CTA and without one 256-row epilogue.

**What it costs.**
- **TMEM is the same as the undivided tile:** `dp · (BLOCK_M/dp) · BLOCK_N` fp32, i.e. `2·BLOCK_N` columns at `BLOCK_M=256`. The planner raises accumulator copies round-robin only while the total stays within 512 columns (`WS/WSMemoryPlanner.cpp`, the TMEM `buffer.copy` post-processing with `tmemColLimit`). At `BLOCK_N=256` that leaves one accumulator copy, so the epilogue of tile i cannot overlap the MMAs of tile i+1.
- **Extra consumer groups:** one per slice. In the GEMM layouts both sit in the default region.
- **With `disallow_acc_multi_buffer`,** every TMA-fed SMEM operand of the MMA is floored at the full `num_stages` (`WS/WSMemoryPlanner.cpp`, `getStaticSmemCopySafetyFloor`). When that floor does not fit, the planner gives it back one copy at a time (`relaxCopySafetyFloorToBudget`). The floor matters: a relaxed floor with a TMA-loaded epilogue input hangs once a CTA runs several tiles (reproduced; [hazards.md](hazards.md#relaxed-operand-copy-floor-with-a-tma-loaded-epilogue-input-several-tiles-per-cta)).

## When to try it (depends on the problem)

- **IF** `BLOCK_M=128`, **THEN** skip dp=2, **BECAUSE** it compiles to the dp=1 kernel.
- **IF** `BLOCK_M=64`, **THEN** only try dp=2 with `BLOCK_N=256` (N split), **BECAUSE** every other `BLOCK_N` fails to compile.
- **IF** `BLOCK_M=256`, **THEN** try dp=1 and dp=2, **BECAUSE** dp=2 is the only way to split a 256-row tile into two 128-row consumer groups without 2-CTA.
- **IF** dp=2 and `BLOCK_N=256`, **THEN** prefer configs whose epilogue does not depend on cross-tile overlap, or keep a dp=1 / smaller-`BLOCK_N` candidate, **BECAUSE** TMEM holds only one accumulator copy.
- **IF** dp > 1 and the epilogue has a TMA-loaded input, **THEN** only try `num_stages` whose full operand ring fits the SMEM budget (this removes the rest), **BECAUSE** a relaxed copy-safety floor hangs once a CTA runs several tiles (reproduced).
- **IF** dp=1, **THEN** skip `disallow_acc_multi_buffer=True` (this removes it), **BECAUSE** it is a duplicate config of `False`.
- **IF** a kernel has more than one annotated loop, **THEN** use one `data_partition_factor` for all of them, **BECAUSE** mismatches are not diagnosed in release builds.

## Interactions

See [interactions.md](interactions.md). Pairs these rules depend on:
- `data_partition_factor` × `BLOCK_M` / `BLOCK_N`
- `data_partition_factor` × `num_warps` (both halves in the default region)
- `data_partition_factor` × `disallow_acc_multi_buffer`
- `data_partition_factor` × accumulator copies (TMEM)
- `data_partition_factor` × `EPILOGUE_SUBTILE` / `generate_subtiled_region`
- `data_partition_factor` × `two_ctas` (see [2cta.md](2cta.md))
- `data_partition_factor` × CLC / dynamic `while` (see [clc_persistence.md](clc_persistence.md))
- `data_partition_factor` × `pingpongAutoWS` (see `pingpong.md`)
- `data_partition_factor` × host-side TMA descriptors

## Hazards

None in [hazards.md](hazards.md). Reported but not yet located in code (row in [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause)): dp=2 with `generate_subtiled_region` and a per-row/column scale applied per subtile hangs.

## Not yet established

See [TRACKER.md](TRACKER.md):
- When dp=2 at `BLOCK_M=256` beats dp=1 at the same tile. Best-vs-best comparisons have confounded dp with the tile size.
- Correctness and value of `BLOCK_M=64, BLOCK_N=256`, dp=2 (N split).
- Whether splitting only the TMEM drain/epilogue along N (not the MMA) would recover cross-tile overlap at `BLOCK_N=256`.
- How dp=2 configs that win unfused rank once an epilogue is fused.
