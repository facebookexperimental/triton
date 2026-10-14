# Data partitioning (`data_partition_factor`, autoWS GEMM, sm100)

Scope: when to set `data_partition_factor` (dp) on the persistent tile loop of
a GEMM. This covers which dp configs are duplicates of dp=1 or fail to compile,
what dp trades, and the pruning that follows. How the pass slices the IR is in
[DataPartition.md](../DataPartition.md); budget numbers are in
[resources.md](resources.md).

Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` (`WS/`) unless they start at the repository root.

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| `data_partition_factor` | `tl.range` kwarg | int; < 2 is a no-op | None | P0 | M split at `BLOCK_M=256`. At `BLOCK_M=128`, dp=2 is a duplicate config of dp=1; at `BLOCK_M=64` with `BLOCK_N` < 256 it is a compile error. | — |
| `disallow_acc_multi_buffer` | `tl.range` kwarg | bool | False | P1 | Only `data_partition_factor` > 1, where it floors operand copies at `num_stages`. At dp=1 it is a duplicate config. | Not with dp=2, a TMA-loaded epilogue input, a relaxed floor and several tiles per CTA (hangs). |

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| `BLOCK_M=128`, dp=2 (any `BLOCK_N`) | duplicate config | Runs as dp=1, warning only. |
| `BLOCK_M=64`, dp=2, `BLOCK_N` < 256 | compile error | No M slice >= 64 and no N slice >= 128 exists, so `computePartitionScheme` fails and the pass signals failure. |
| different `data_partition_factor` on two annotated loops of one function | silently ignored | Release builds keep the last loop's value. |
| a host `TensorDescType` kernel argument passed as a loop init arg | compile error | The descriptor slicing does not update loop iter-arg types; `doDataPartition` bails ("used as loop init arg; not supported yet"). |
| dp > 1 with a TMA-loaded epilogue input, a `num_stages` whose full operand ring does not fit (the copy-safety floor is relaxed), and several tiles per CTA | hangs | Root cause not yet located. |
| MMAs that accumulate into one TMEM tile land in dp groups with no common partition | compile error | "warp specialization assigned accumulator-chained MMAs with no common partition"; concurrent accumulating MMAs would race. |
| the warp estimate (default partition `num_warps` plus each other partition's minimum) exceeds 16 | silently ignored | Warp specialization is dropped for the whole function. |
| total TMEM above 512 columns | requires | Total TMEM must fit 512 columns; otherwise `OutOfResources("tensor memory")` is raised when the kernel is loaded, not at compile time. |
| `disallow_acc_multi_buffer` at dp=1 | duplicate config | Under Meta WS the SMEM planner reads the attribute only when `data_partition_factor` > 1; the TTGIR is identical apart from the attribute. |

## Mechanism

- **Split rule:** the accumulator is split into dp slices along M when `BLOCK_M/dp` >= 64 and the slice keeps the TMEM `blockM`, along N (`BLOCK_N/dp` >= 128) only when no M split was attempted (an M split skipped because it would change `blockM` drops partitioning without trying N); each slice gets its own MMA and consumer (epilogue/computation) group, and operand loads feeding the split dimension are sliced too ([DataPartition.md, "Algorithm"](../DataPartition.md#algorithm)); at `BLOCK_M=256`, dp=2 gives two 128-row groups.
- **Per-slice consumer groups:** changes each consumer group to `BLOCK_M/dp` rows -> relieves: the per-thread epilogue live set (TMEM -> registers -> SMEM) shrinks by dp at fixed `num_warps`, and a 256-row tile runs without 2-CTA and without one 256-row epilogue -> costs one extra consumer group per slice; in the GEMM layouts both sit in the default region.
- **DP0 runs before DP1:** the pass rebuilds straight-line blocks so the whole DP0 chain precedes DP1 (`WS/WSDataPartition.cpp`, `serializeDataPartitionedOps`).
- **TMEM same as the undivided tile:** `dp · (BLOCK_M/dp) · BLOCK_N` fp32, i.e. `2·BLOCK_N` columns at `BLOCK_M=256`; accumulator copies are raised round-robin only within 512 columns (`WS/WSMemoryPlanner.cpp`, the TMEM `buffer.copy` post-processing with `tmemColLimit`), so at `BLOCK_N=256` that leaves one copy and tile i's epilogue cannot overlap tile i+1's MMAs.
- **`disallow_acc_multi_buffer`:** every TMA-fed SMEM operand of the MMA is floored at the full `num_stages` (`WS/WSMemoryPlanner.cpp`, `getStaticSmemCopySafetyFloor`), given back one copy at a time when it does not fit (`relaxCopySafetyFloorToBudget`); a relaxed floor with a TMA-loaded epilogue input hangs once a CTA runs several tiles (reproduced; [hazards.md](hazards.md#relaxed-operand-copy-floor-with-a-tma-loaded-epilogue-input-several-tiles-per-cta)).

## Interactions

See [interactions.md](interactions.md). Pairs these rules depend on:
- `data_partition_factor` × `BLOCK_M` / `BLOCK_N`
- `data_partition_factor` × `num_warps` (both halves in the default region)
- `data_partition_factor` × `disallow_acc_multi_buffer`
- `data_partition_factor` × accumulator copies (TMEM)
- `data_partition_factor` × `EPILOGUE_SUBTILE` / `generate_subtiled_region`
- `data_partition_factor` × `two_ctas` (see [2cta.md](2cta.md))
- `data_partition_factor` × CLC / dynamic `while` (see `clc_persistence.md`)
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
