# Warps and registers (autoWS GEMM, sm100)

Scope:
- how `num_warps` and the register knobs become per-partition warp counts and `setmaxnreg` budgets;
- which settings are illegal or silently drop warp specialization.

Hardware limits are in [resources.md](resources.md). Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` (`WS/`) unless they start at the repository root.

## Knobs

| Knob | How set | What it does | Existing doc |
|---|---|---|---|
| `num_warps` | kernel compile option | Warps of the default region; also the starting warp count of every other partition | [default_partition_selection.md, "Per-op minimum warps"](../../../../../../../../docs/design/modulo_schedule/default_partition_selection.md#per-op-minimum-warps) |
| `minRegAutoWS` | kernel compile option | Per-thread registers requested by partitions that hold no tensors (default 24) | [RegisterBudgets.md](../RegisterBudgets.md); [Overview.md, "Register Budgets"](../Overview.md#register-budgets) |
| `maxRegAutoWS` | kernel compile option | When set, per-thread registers requested by non-default tensor partitions; the default region gets the leftover share | [RegisterBudgets.md](../RegisterBudgets.md); [Overview.md, "Register Budgets"](../Overview.md#register-budgets) |
| `maxnreg` | kernel compile option | Replaces the computed per-thread register ceiling (`ttg.maxnreg`) | [RegisterBudgets.md](../RegisterBudgets.md) |
| `merge_epilogue`, `merge_epilogue_to_computation`, `separate_epilogue_store` | `tl.range(...)` kwargs | Add or remove partitions, which changes the worker warp total | [PartitionSchedulingMeta.md, "Target Partition Layouts"](../PartitionSchedulingMeta.md#target-partition-layouts) |

## Never try (any problem)

| Condition | Status | Why | Code |
|---|---|---|---|
| `num_warps` < 4 | silently ignored | The kernel runs without WS: "Warp specialization requires at least 4 warps". Under `two_ctas` this hangs instead (see [2cta.md](2cta.md#never-try-any-problem)) | `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp`, `runOnFuncOp` |
| `num_warps` not 4 or 8 | silently ignored | `supportMMA` for MMAv5 accepts only 4 or 8 warps (the TMEM load/store layouts), so the dot falls back to mma.sync and there is no gemm partition | `lib/Analysis/Utility.cpp`, `supportMMA`; `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`, `getMMAVersionSafe` |
| WS warp estimate > 16 | silently ignored | All WS metadata is removed. Estimate = `num_warps` + Σ over other partitions of their minimum warps (4 for partitions holding TMEM load/store/alloc, 2 for gather/scatter, else 1). Under `two_ctas` this hangs instead (see [2cta.md](2cta.md#never-try-any-problem)) | `WS/PartitionSchedulingMeta.cpp`, "Warp budget exceeded. Skipping warp specialization."; `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`, `getMinWarpsForOp` |
| `minRegAutoWS` or `maxRegAutoWS` not a multiple of 8 | compile error | Requests are emitted per warp group in steps of 8 | `third_party/nvidia/backend/compiler.py`, `_check_reg_auto_ws_alignment` |
| Register requests that do not fit `maxnreg` × total warps, or a shared share below 24 | silently ignored | `AllocateWarpGroups` returns without setting actual registers, so there is no per-partition register split. Without a user `maxnreg` every warp keeps the uniform ceiling; with one, the lowering still emits a register reallocation to `maxnreg` (`ConvertWarpSpecializeToLLVM.cpp`) | `lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp`, register allocation in `runOnOperation` |

## Mechanism

**Warp counts.** The steps, in order:
1. The default region gets `num_warps` (`WS/PartitionSchedulingMeta.cpp`, warp budget estimate).
2. Every other partition is also created with `num_warps` (`WS/WSSpecialize.cpp`, construction of `ttg.warp_specialize`).
3. `optimizePartitionNumWarps` (`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp`) then shrinks them:
   - partitions without tensors go to 1 warp;
   - partitions holding TMEM ops stay at >= 4 (`getMinWarpsForOp`);
   - any other tensor partition halves while 2 × (registers of its largest single tensor) fits the per-thread share 65,536 / 32 / (total warps after halving).
4. The estimate sees only the largest tensor, not tensors that are live at the same time.
5. When the schedule has both a "reduction" and a "computation" partition and the module has no cluster (no 2-CTA attribute), the computation partition is set to 8 warps.
6. In the GEMM layouts the partition scheduler builds, the epilogue and, under `data_partition_factor=2`, both data-partition halves sit in the default region, which is never shrunk; the worker partitions (load, MMA issue, store) are 1 warp each. So the 16-warp cap and partition-warp halving do not come into play for a plain or epilogue-subtiled GEMM: `BLOCK_M=256`, dp=2, `num_warps=8` keeps WS with 12 total warps, and `EPILOGUE_SUBTILE` does not change worker warps.

Which logical partition is the default region: [PartitionSchedulingMeta.md, "Phase 2: Partition Layout (`createPartitionLayout`)"](../PartitionSchedulingMeta.md#phase-2-partition-layout-createpartitionlayout) and [default_partition_selection.md, "Hardware mechanism: the default partition is the register sink"](../../../../../../../../docs/design/modulo_schedule/default_partition_selection.md#hardware-mechanism-the-default-partition-is-the-register-sink).

**Warp groups and registers** (`lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp`):
- `padToMaxWarpGroups` pads worker warps to whole warp groups. Total launched warps = `num_warps` + 4 × ceil(worker warps / 4), recorded as `ttg.total-num-warps`.
- The per-thread ceiling is `maxnreg` if set. Otherwise it is 65,536 / total warps / 32, rounded down to a multiple of 8.
- Each warp group requests the maximum of its partitions' requests, rounded up to 8:
  - without `maxRegAutoWS`, non-tensor partitions request `minRegAutoWS` and tensor partitions take an even share of what is left, together with the default region;
  - with `maxRegAutoWS`, non-default tensor partitions request it and the default region takes the leftover share.
- A warp group's request is the maximum over its partitions that have a positive estimate; without `maxRegAutoWS`, tensor partitions carry a share sentinel that is skipped, so a group of one-warp partitions requests `minRegAutoWS` (default 24). Full mechanics: [RegisterBudgets.md](../RegisterBudgets.md).

**Reading metadata.** `metadata["num_warps"]` of a compiled WS kernel is `ttg.total-num-warps` (workers plus padding; `third_party/nvidia/backend/compiler.py`). Relaunching from TTGIR reads `ttg.num-warps`, the default-region count (`python/triton/compiler/compiler.py`, `parse_options`).

**Change → relief → cost:**
- **Raising `num_warps`:**
  - relieves: more threads and a smaller per-thread share of each tensor in the default region and in every tensor partition that does not shrink;
  - costs: a lower ceiling for all threads, and the warp estimate counts the default region at full size.
- **Adding a partition:**
  - relieves: separates work so it can overlap;
  - costs: a whole warp group once worker warps cross a multiple of 4, which lowers the ceiling for every thread.

## When to try it (depends on the problem)

- **IF** `num_warps` is not 4 or 8, **THEN** skip, **BECAUSE** no tcgen05 autoWS GEMM is generated.
- **IF** `num_warps` + 4 × (number of non-default partitions holding TMEM ops) + (other non-default partitions) > 16, **THEN** skip, **BECAUSE** WS is stripped and the config duplicates a non-WS kernel (or hangs under `two_ctas`). In the GEMM layouts the epilogue and data-partition halves are in the default region, so this needs extra non-default TMEM partitions to trigger.
- **IF** you set `minRegAutoWS`/`maxRegAutoWS`, **THEN** use multiples of 8 and check that requests × warps fit the ceiling, **BECAUSE** otherwise compilation fails or the split is silently dropped.
- **IF** a config adds a one-warp partition and worker warps are already at a multiple of 4, **THEN** count it as a full warp group of register cost, **BECAUSE** padding adds 4 warps.
- **IF** you compare register use across configs, **THEN** read `setmaxnreg` per partition in the PTX, **BECAUSE** the uniform ceiling and the per-partition budgets differ.

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
