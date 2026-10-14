# B200 resource limits (autoWS GEMM)

Scope: the shared-memory (SMEM), tensor-memory (TMEM), register and warp limits that decide whether an autoWS GEMM config on sm100 compiles, does what was asked, or is silently changed. The rules below cover only what follows from those limits. For how the planners work, follow the links.

Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` (`WS/`) unless they start at the repository root.

## Limits

| Resource | Limit the compiler uses | Where it comes from |
|---|---|---|
| SMEM per CTA | 232,448 B on sm100 | `third_party/nvidia/backend/compiler.py`, `_max_shared_mem_for_capability` (table entry for capability 100), passed as `smem_budget` to `add_hopper_warpspec` |
| SMEM the WS planner may fill | 232,448 B (or the loop's `smem_budget`) minus a per-kernel auxiliary estimate | `WS/WSMemoryPlanner.cpp`, `doMemoryPlanner` and `estimateAuxiliarySmemBytes` |
| TMEM | 128 lanes × 512 32-bit columns | `WS/WSMemoryPlanner.cpp`, `tmemColLimit` and `kMaxTMemCols` |
| Registers | 65,536 per SM; `setmaxnreg` values in [24, 256] in steps of 8 | `lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp`, `kMinLegalRegisters` and the `maxnreg` computation; `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp`, `nTotalRegs` |
| Warps | WS needs `num_warps` >= 4; the WS warp estimate must be <= 16; tcgen05 needs `num_warps` of 4 or 8 | `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp` ("Warp specialization requires at least 4 warps"); `WS/PartitionSchedulingMeta.cpp`, `kMaxWarps`; `lib/Analysis/Utility.cpp`, `supportMMA` |

## SMEM

The planner's budget is the target's opt-in maximum, or `tt.smem_budget` if the loop sets one, minus `estimateAuxiliarySmemBytes`. That estimate counts:
- barrier slots, scaled by buffer depth and CTA count;
- barrier-array metadata;
- captured kernel arguments and allocations;
- tensor maps;
- the largest single op scratch before the last WS loop, such as a layout conversion (`estimateOpScratchBytes`).

The estimate depends on the kernel, so no single "planner budget" number holds for every kernel. Buffers are multiples of 16 KiB for bf16 GEMM tiles, so bf16 plans often total exactly 229,376 B (14 × 16 KiB). That total is a result of the buffer sizes, not a constant in the code.

Mechanics (link, don't repeat):
- Phases, priorities and the copy-increase loop: [SmemAllocationDesign.md, "Algorithm"](../SmemAllocationDesign.md#algorithm).
- Staging fusion, the staging copy increase and the `K | S` cap: [TMAStoreWaitPipeline.md, "Memory Planner: `buffer.tmaStaging` Handling"](../TMAStoreWaitPipeline.md#memory-planner-buffertmastaging-handling).
- Plan-space search: [MemoryPlannerSearch.md, "9. The plan-space beam search"](../MemoryPlannerSearch.md#9-the-plan-space-beam-search-wsmemoryplansearchhcpp).

Facts these rules depend on, all in `WS/WSMemoryPlanner.cpp`, `allocateSmemBuffers`:
1. **Staging comes first.** TMA output staging is reserved before the operand rings: `increaseFusedEpilogueCopies` runs before the Phase 4 operand copy increase.
2. **The operand depth cap is `num_stages`.** Phase 4 raises the innermost operand buffers, in the order they are first used, one level at a time. It stops when nothing fits or the level would pass `num_stages`. It never uses free SMEM beyond `num_stages`.
3. **The cross-stage floor is never dropped.** A plan that still exceeds the budget after the discretionary reuse step only produces the warning "exceeding configured smem-budget". The hardware check at launch is the backstop. The copy-safety floor (dp > 1) is different: it is given back one copy at a time when it does not fit (`relaxCopySafetyFloorToBudget`). A relaxed floor combined with a TMA-loaded epilogue input hangs when a CTA processes several tiles (reproduced; [hazards.md](hazards.md#relaxed-operand-copy-floor-with-a-tma-loaded-epilogue-input-several-tiles-per-cta)).
4. **Under budget pressure, staging can alias operand rings.** Innermost operand buffers are never aliased (`findReuseCandidate`).

## TMEM

Mechanics: [TMEMAllocationHeuristics.md, "TMEM Allocation Overview"](../TMEMAllocationHeuristics.md#tmem-allocation-overview) and [MemoryPlannerSearch.md, "2. Mental model"](../MemoryPlannerSearch.md#2-mental-model).

Accumulator copies (`WS/WSMemoryPlanner.cpp`, the "TMEM multi-buffering post-processing" step at the end of the heuristic TMEM allocator's `run`):
- After placement, the heuristic TMEM planner raises `buffer.copy` round-robin on every loop-carried MMA accumulator. A per-iteration accumulator overwritten by a `tmem_store` at the top of the body also counts as loop-carried.
- It stops when the next copy would pass 512 columns minus the columns reserved for scaled-MMA scales.
- No user knob bounds this; `num_stages` does not either.
- A kernel whose total TMEM exceeds 512 columns is not rejected at compile time: it raises `OutOfResources("tensor memory")` when it is loaded (`python/triton/compiler/compiler.py`, `CompiledKernel._init_handles`).

Columns per accumulator copy (fp32) = `BLOCK_N × ceil(BLOCK_M / 128)`:
- Instruction M is 128 whenever `BLOCK_M` >= 128 (and 64 below that), instruction N is `min(BLOCK_N, 256)` and instruction K is 256 / operand bit width (`lib/Dialect/TritonGPU/Transforms/Utility.cpp`, `mmaVersionToInstrShape`). TMEM has 128 lanes.
- So `BLOCK_M=128` gives 4 copies at `BLOCK_N=128` and 2 copies at `BLOCK_N=256`.
- `BLOCK_M=256, BLOCK_N=256` gives 1 copy, so tile i+1's mainloop cannot overlap tile i's epilogue.

## Registers and warps

Mechanics: [default_partition_selection.md, "Hardware mechanism: the default partition is the register sink"](../../../../../../../../docs/design/modulo_schedule/default_partition_selection.md#hardware-mechanism-the-default-partition-is-the-register-sink) and [Overview.md, "Register Budgets"](../Overview.md#register-budgets).

Facts:
1. **Starting warp counts.** The default region gets `num_warps`. Every other partition is created with `num_warps` too (`WS/WSSpecialize.cpp`, construction of the `ttg.warp_specialize` op).
2. **Shrinking (`optimizePartitionNumWarps`).** Partitions without tensors drop to 1 warp. Partitions holding TMEM loads, stores or allocs stay at 4 or more (`getMinWarpsForOp`). Other tensor partitions halve while twice their largest single tensor still fits the even per-thread register share.
3. **Padding (`padToMaxWarpGroups`, `AllocateWarpGroups.cpp`).** Worker warps are padded to whole warp groups:
   - total launched warps = `num_warps` + 4 × ceil(worker warps / 4) (`ttg.total-num-warps`);
   - base `maxnreg` = 65,536 / total warps / 32, rounded down to a multiple of 8;
   - non-tensor partitions request `minRegAutoWS` (default 24), and tensor partitions share the rest;
   - if those requests do not fit, the register split is silently skipped.

## Never try (any problem)

| Condition | Status | Why | Code |
|---|---|---|---|
| Auxiliary SMEM estimate >= budget | compile error | "estimated auxiliary shared-memory allocation ... exhausts the shared-memory budget" | `WS/WSMemoryPlanner.cpp`, `doMemoryPlanner` |
| Plan larger than the hardware limit after floors and reuse | requires | The plan must fit the hardware limit; otherwise `OutOfResources` is raised at launch, not at compile time. The cross-stage floor is never dropped for budget, so the planner only warns | `WS/WSMemoryPlanner.cpp`, "exceeding configured smem-budget" warning |
| `num_warps` < 4 | silently ignored | WS bails out before partitioning, so the kernel runs without WS. Under `two_ctas` this hangs instead (see [2cta.md](2cta.md#never-try-any-problem)) | `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp`, `runOnFuncOp` |
| `num_warps` not 4 or 8 | silently ignored | `supportMMA` for MMAv5 accepts only 4 or 8 warps, so there is no tcgen05 and no gemm partition | `lib/Analysis/Utility.cpp`, `supportMMA`; `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`, `getMMAVersionSafe` |
| WS warp estimate > 16 | silently ignored | All WS metadata is removed. Estimate = `num_warps` + Σ per-partition minimum warps (TMEM ops count as 4, gather/scatter as 2; `getMinWarpsForOp`). Under `two_ctas` this hangs instead (see [2cta.md](2cta.md#never-try-any-problem)) | `WS/PartitionSchedulingMeta.cpp`, "Warp budget exceeded. Skipping warp specialization." |
| `minRegAutoWS` / `maxRegAutoWS` not a multiple of 8 | compile error | Register requests are emitted per warp group in steps of 8 | `third_party/nvidia/backend/compiler.py`, `_check_reg_auto_ws_alignment` |
| `TRITON_WS_SMEM_PLAN_SEARCH=1` with no K loop (K == BLOCK_K as a compile-time constant) and more tiles than SMs | wrong answer (reproduced) | Plan search pins every TMEM accumulator to one copy, bypassing the per-iteration accumulator multi-buffering. See [hazards.md](hazards.md#plan-search-pins-tmem-accumulator-copies-to-1) | `WS/WSMemoryPlanner.cpp`, `allocateTmemBuffersViaSearch` |
| `num_stages` < 1 | compile error | "nvgpu-warp-specialization requires num-stages >= 1" | `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp`, `runOnOperation` |

## When to try it (depends on the problem)

- **IF** `num_warps` is not 4 or 8, **THEN** skip the config, **BECAUSE** it never produces a tcgen05 autoWS GEMM.
- **IF** two configs differ only in `num_stages`, both are at or above the operand depth that fits, and output staging is single-copy (`EPILOGUE_SUBTILE=1` or `two_ctas`), **THEN** keep one (this removes the other), **BECAUSE** the operand plan is the same; only barrier bytes and the loop schedule differ. With `EPILOGUE_SUBTILE` >= 2 staging deepens with `num_stages`, so the plans differ (see the worked example).
- **IF** a config uses `TRITON_WS_SMEM_PLAN_SEARCH=1`, **THEN** do not compare it against heuristic-planner configs for epilogue overlap, and skip it for kernels with no K loop and more tiles than SMs, **BECAUSE** the plan search pins every TMEM `buffer.copy` to 1, which also bypasses the per-iteration accumulator multi-buffering that guards a cross-tile race (wrong answer, reproduced: [hazards.md](hazards.md#plan-search-pins-tmem-accumulator-copies-to-1)). The exceptions are kernels with scaled MMA or a `SubtiledRegionOp`, which fall back to the heuristic planner (`WS/WSMemoryPlanner.cpp`, `allocateTmemBuffersViaSearch`).
- **IF** `BLOCK_M × BLOCK_N` fp32 already takes all 512 TMEM columns, **THEN** expect no accumulator overlap across tiles, and add no TMEM-resident loop state, **BECAUSE** the accumulator has exactly one copy.
- **IF** a worker partition would push worker warps past a multiple of 4, **THEN** count it as costing registers for every thread, **BECAUSE** padding adds a whole warp group, which lowers the base `maxnreg`.

## Worked example (bf16 in and out, fp32 accumulator, 1-CTA)

Config: `BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_stages=4, EPILOGUE_SUBTILE=2, separate_epilogue_store=True, num_warps=4`.

**SMEM slot sizes** (elements × 2 B):

| Buffer | Shape | Bytes |
|---|---|---|
| A slot | 128 × 64 | 16,384 |
| B slot | 256 × 64 | 32,768 |
| Staging slot (one subtile) | 128 × 128 | 32,768 |

Both staging subtiles share one buffer id, so staging costs one slot × copies.

**Staging is reserved first.**
- The store runs in its own partition, so any copy count up to `num_stages` is legal.
- 4 copies take 131,072 B, which fits. Staging gets 4 copies.

**Then the operands**, one level at a time, A first:

| Level | A copies | B copies | Data total | Fits? |
|---|---|---|---|---|
| 1 | 1 | 1 | 180,224 | yes |
| 2 | 2 | 1 | 196,608 | yes |
| 2 | 2 | 2 | 229,376 | only if the auxiliary estimate is <= 3,072 B |
| 3 | 3 | 2 | 245,760 | no (> 232,448) |
| 3 (if B=2 did not fit) | 3 | 1 | 212,992 | only if the auxiliary estimate is <= 19,456 B |

Phase 4 keeps raising any buffer that still fits, so the result is A=2 / B=2 when the auxiliary estimate is <= 3,072 B, and A=3 / B=1 when it is between 3,072 and 19,456 B.

**With `TRITON_WS_STAGING_COPIES=2`**, staging takes 65,536 B and the operands reach A=4 / B=3. That totals 229,376 B under the same condition. The choice is between staging depth and operand depth, under the same total.

**`num_stages` is not a duplicate here.** Staging took 4 copies because `num_stages=4`; at `num_stages=3` staging takes 3 copies (98,304 B) and the operands get the freed 32 KiB.

**TMEM:** 256 columns per accumulator copy, so 2 copies (512 columns).

**Registers:** one epilogue subtile is a 128 × 128 fp32 `tmem_load`. Across 4 warps that is 128 registers per thread.

**Warps (illustrative):** worker partitions totalling 3 warps are padded to 4. That gives 8 warps in all and a base `maxnreg` of 256.

## Interactions

See [interactions.md](interactions.md). Pairs where these limits decide the outcome:
- staging copies × `num_stages`
- `EPILOGUE_SUBTILE` × staging copies
- `BLOCK_M`/`BLOCK_N` × accumulator copies
- `data_partition_factor` × TMEM columns
- `num_warps` × extra partitions
- 2-CTA × staging copies

## Hazards

None in [hazards.md](hazards.md).

## Not yet established

- Which auxiliary-estimate size flips the worked example between B=2 and B=1. That needs the planner debug output for the real kernel.
- Whether an over-budget plan can still pass the planner and then fail at launch, given that `estimateOpScratchBytes` is sized on the IR the planner sees rather than the IR the allocator sees.
