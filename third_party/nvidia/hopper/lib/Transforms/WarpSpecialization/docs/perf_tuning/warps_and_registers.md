# Warps and registers (autoWS GEMM, sm100)

Scope:
- how `num_warps` and the register knobs become per-partition warp counts and `setmaxnreg` budgets;
- which settings are illegal or silently drop warp specialization.

Hardware limits are in [resources.md](resources.md). Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` (`WS/`) unless they start at the repository root.

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| `warp_specialize` | `tl.range` kwarg | True | False | P0 | Every autoWS kernel. Silently dropped when `num_warps` < 4, when no partition forms, or when the WS warp estimate exceeds 16. | `TRITON_USE_META_WS=1`; `num_warps` in {4, 8}. |
| `num_warps` | compile option | 4 or 8 | 4 | P0 | Every kernel. Other values get no tcgen05 MMA; below 4, warp specialization is dropped. | — |
| `minRegAutoWS` | compile option | multiple of 8 | 24 | P1 | Every WS kernel. Requests that do not fit are silently not applied. | — |
| `maxRegAutoWS` | compile option | multiple of 8 | None | P1 | Only non-default tensor partitions; in these GEMM layouts the epilogue is the default region. | — |
| `maxnreg` | compile option | int | None | P2 | Every kernel. Requests that do not fit are silently dropped. | — |

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| `num_warps` < 4 | silently ignored | The kernel runs without WS: "Warp specialization requires at least 4 warps". |
| `num_warps` not 4 or 8 | silently ignored | `supportMMA` for MMAv5 accepts only 4 or 8 warps (the TMEM load/store layouts), so the dot falls back to mma.sync and there is no gemm partition. |
| WS warp estimate > 16 | silently ignored | All WS metadata is removed. |
| `minRegAutoWS` or `maxRegAutoWS` not a multiple of 8 | compile error | Requests are emitted per warp group in steps of 8. |
| Register requests that do not fit `maxnreg` × total warps, or a shared share below 24 | silently ignored | `AllocateWarpGroups` returns without setting actual registers, so there is no per-partition register split. |

## Mechanism

- **Partition warp counts:** the default region gets `num_warps` (`WS/PartitionSchedulingMeta.cpp`, warp budget estimate) and every other partition is created with `num_warps` too (`WS/WSSpecialize.cpp`, construction of `ttg.warp_specialize`); `optimizePartitionNumWarps` (`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp`) then shrinks them: no tensors -> 1 warp, TMEM ops -> stays >= 4 (`getMinWarpsForOp`), other tensor partitions halve while 2 × (registers of the largest single tensor) fits the per-thread share 65,536 / 32 / (total warps after halving), not counting tensors live at the same time; then, with both a "reduction" and a "computation" partition and no cluster (no 2-CTA attribute), the computation partition is set to 8 warps.
- **GEMM layouts:** in the partition scheduler's GEMM layouts the epilogue and, under `data_partition_factor=2`, both data-partition halves sit in the default region, which is never shrunk, and the workers (load, MMA issue, store) are 1 warp each, so the 16-warp cap and halving do not come into play for a plain or epilogue-subtiled GEMM: `BLOCK_M=256`, dp=2, `num_warps=8` keeps WS with 12 total warps, and `EPILOGUE_SUBTILE` does not change worker warps (which logical partition is the default region: [PartitionSchedulingMeta.md, "Phase 2: Partition Layout (`createPartitionLayout`)"](../PartitionSchedulingMeta.md#phase-2-partition-layout-createpartitionlayout), [default_partition_selection.md, "Hardware mechanism: the default partition is the register sink"](../../../../../../../../docs/design/modulo_schedule/default_partition_selection.md#hardware-mechanism-the-default-partition-is-the-register-sink)).
- **Warp groups and register ceiling:** in `lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp` (persistent kernels only, one CTA per SM, so occupancy does not enter) registers are assigned per 4-warp group with per-thread floor `kMinLegalRegisters` = 24; `padToMaxWarpGroups` pads worker warps to whole warp groups with partitions that only return, each padding warp requesting 24 and doing no work, so total launched warps = `num_warps` + 4 × ceil(worker warps / 4), recorded as `ttg.total-num-warps` (which is what `metadata["num_warps"]` of a compiled WS kernel reports, `third_party/nvidia/backend/compiler.py`; relaunching from TTGIR reads `ttg.num-warps`, the default-region count, `python/triton/compiler/compiler.py`, `parse_options`); the per-thread ceiling is `maxnreg` if set, else 65,536 / total warps / 32, rounded down to a multiple of 8.
- **Per-group register requests:** each warp group requests the maximum over its partitions with a positive estimate, rounded up to 8; without `maxRegAutoWS`, non-tensor partitions request `minRegAutoWS` and tensor partitions and the default region take an even share of the rest (tensor partitions carry a share sentinel that is skipped, so a group of one-warp partitions requests `minRegAutoWS`, default 24); with `maxRegAutoWS`, non-default tensor partitions request it and the default region takes the leftover share ([RegisterBudgets.md](../RegisterBudgets.md)).
- **Raising `num_warps`:** changes the thread count of the default region and every tensor partition that does not shrink -> relieves: more threads and a smaller per-thread share of each tensor -> costs: a lower ceiling for all threads, and the warp estimate counts the default region at full size.
- **Adding a partition:** changes the work split -> relieves: separated work can overlap -> costs: a whole warp group once worker warps cross a multiple of 4, lowering the ceiling for every thread.

## Interactions

See [interactions.md](interactions.md). Pairs these rules depend on:
- `num_warps` × `data_partition_factor` (both halves in the default region)
- `num_warps` × `separate_epilogue_store`
- `num_warps` × `merge_epilogue`
- cluster (`ctas_per_cga`) × reduction/computation warp override
- `maxnreg` × `minRegAutoWS`/`maxRegAutoWS`

## Hazards

None in [hazards.md](hazards.md).

## Not yet established

Tracked in [TRACKER.md](TRACKER.md).

- Whether `num_warps=4` or `8` is better for plain GEMM versus epilogue-heavy fusions. This has been observed to depend on the kernel.
- The register budgets partitions actually get in a GEMM (the computation-partition range).
- The cost of single-thread mbarrier arrives in 8-warp partitions.
- Whether PyTorch Inductor's register-spill filter, which assumes the accumulator is in registers, prunes valid autoWS configs in practice.
