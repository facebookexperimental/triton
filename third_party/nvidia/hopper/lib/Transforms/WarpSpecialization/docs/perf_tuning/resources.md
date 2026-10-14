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

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| `tmem_alloc_algo` | `tl.range` kwarg | 1 (greedy) or other (backtracking) | 1 | P1 | Every WS kernel with TMEM allocations. | — |
| `smem_alloc_algo` | `tl.range` kwarg | 1 (WSBuffer) or 0 (legacy) | 1 | P2 | `=0` only; under 0, `tt.autows` channel pins are ignored. | — |
| `smem_budget` | `tl.range` kwarg | bytes | device opt-in maximum | P2 | Every WS kernel. | — |
| `smem_circular_reuse` | `tl.range` kwarg | bool / None | None | P2 | Only with exactly 2 candidate buffers. | — |
| `tt.autows` channels | `tl.dot(attrs={"channels": [...]})` | `"operand,smem\|tmem,copies,id[,offset]"` | unset | P2 | Only with `smem_alloc_algo=1`. | Not below a buffer's cross-stage depth (hangs). |
| `mem_plan_pick` + `TRITON_WS_MEM_PLAN_TOPK` / `_PICK` | `tl.range` kwarg + env vars | rank int; TOPK int | 0; TOPK 1 | P2 | Plans are enumerated when TOPK > 1 or the pick is > 0. | — |
| `TRITON_WS_SMEM_PLAN_SEARCH` | env var | 0 / 1 | 0 | P2 | Every WS kernel. With no K loop, an epilogue input and more tiles than SMs it gives a wrong answer. | — |

## SMEM

Budget = the target's opt-in maximum (or the loop's `tt.smem_budget`) minus `estimateAuxiliarySmemBytes`, which counts:
- barrier slots, scaled by buffer depth and CTA count;
- barrier-array metadata;
- captured kernel arguments and allocations;
- tensor maps;
- the largest single op scratch before the last WS loop, such as a layout conversion (`estimateOpScratchBytes`).

The estimate depends on the kernel, so no single "planner budget" number holds. bf16 GEMM tile buffers are multiples of 16 KiB, so bf16 plans often total exactly 229,376 B (14 × 16 KiB): a result of buffer sizes, not a constant in the code.

Mechanics (link, don't repeat):
- Phases, priorities and the copy-increase loop: [SmemAllocationDesign.md, "Algorithm"](../SmemAllocationDesign.md#algorithm).
- Staging fusion, the staging copy increase and the `K | S` cap: [TMAStoreWaitPipeline.md, "Memory Planner: `buffer.tmaStaging` Handling"](../TMAStoreWaitPipeline.md#memory-planner-buffertmastaging-handling).
- Plan-space search: [MemoryPlannerSearch.md, "9. The plan-space beam search"](../MemoryPlannerSearch.md#9-the-plan-space-beam-search-wsmemoryplansearchhcpp).

Facts these rules depend on, all in `WS/WSMemoryPlanner.cpp`, `allocateSmemBuffers`:
1. **Staging comes first:** `increaseFusedEpilogueCopies` reserves TMA output staging before the Phase 4 operand copy increase.
2. **The operand depth cap is `num_stages`:** Phase 4 raises the innermost operand buffers one level at a time, in first-use order, until nothing fits or the level would pass `num_stages`; it never uses free SMEM beyond `num_stages`.
3. **The cross-stage floor is never dropped:** an over-budget plan after the discretionary reuse step only warns "exceeding configured smem-budget"; the hardware check at launch is the backstop. The copy-safety floor (dp > 1) is given back one copy at a time when it does not fit (`relaxCopySafetyFloorToBudget`); a relaxed floor with a TMA-loaded epilogue input hangs when a CTA processes several tiles (reproduced; `hazards.md`).
4. **Under budget pressure, staging can alias operand rings;** innermost operand buffers are never aliased (`findReuseCandidate`).

**Closed-form operand depth.**
`d_fit = min(num_stages, floor((232448 - aux - S_stg) / slot))`
- `slot = (BM + BN') * BK * in_bytes`, `BN' = BN/2` under 2-CTA; add `BN * BK * in_bytes` for a multi-consumer B (`tma_loads.md`) and one landing buffer per TMA side input (`epilogue_prologue_tma_loads.md`).
- `S_stg = BM * (BN/ES) * out_bytes * staging copies` (one copy when `ES=1` or 2-CTA); since staging is raised first, an uncapped staging ring reduces operand depth.
- `aux` = `estimateAuxiliarySmemBytes` (barriers, captures, tensor maps, op scratch); kernel-dependent.
- `d_fit` is the balanced A/B depth; Phase 4 may give the first-used operand one more copy (worked example).

## TMEM

Mechanics: [TMEMAllocationHeuristics.md, "TMEM Allocation Overview"](../TMEMAllocationHeuristics.md#tmem-allocation-overview) and [MemoryPlannerSearch.md, "2. Mental model"](../MemoryPlannerSearch.md#2-mental-model).

Accumulator copies (`WS/WSMemoryPlanner.cpp`, the "TMEM multi-buffering post-processing" step at the end of the heuristic TMEM allocator's `run`):
- After placement, `buffer.copy` is raised round-robin on every loop-carried MMA accumulator (including a per-iteration accumulator overwritten by a `tmem_store` at the top of the body), until the next copy would pass 512 columns minus the columns reserved for scaled-MMA scales.
- No user knob bounds this, `num_stages` included.
- Packing algorithm: `tmem_alloc_algo` 1 (default) = greedy (`allocateTMemAllocs`), any other value = backtracking (`allocateTMemAllocs2`); see [TMEMAllocationHeuristics.md, "Algorithm Selection"](../TMEMAllocationHeuristics.md#algorithm-selection).
- Total TMEM above 512 columns is not rejected at compile time: it raises `OutOfResources("tensor memory")` at load (`python/triton/compiler/compiler.py`, `CompiledKernel._init_handles`).

Columns per accumulator copy (fp32) = `BLOCK_N × ceil(BLOCK_M / 128)`:
- TMEM has 128 lanes; instruction M is 128 whenever `BLOCK_M` >= 128 (64 below), N is `min(BLOCK_N, 256)`, K is 256 / operand bit width (`lib/Dialect/TritonGPU/Transforms/Utility.cpp`, `mmaVersionToInstrShape`).
- `BLOCK_M=128`: 4 copies at `BLOCK_N=128`, 2 at `BLOCK_N=256`.
- `BLOCK_M=256, BLOCK_N=256`: 1 copy, so tile i+1's mainloop cannot overlap tile i's epilogue.

## Registers and warps

Mechanics: [default_partition_selection.md, "Hardware mechanism: the default partition is the register sink"](../../../../../../../../docs/design/modulo_schedule/default_partition_selection.md#hardware-mechanism-the-default-partition-is-the-register-sink) and [Overview.md, "Register Budgets"](../Overview.md#register-budgets).

Facts:
1. **Starting warp counts:** the default region and every other partition start at `num_warps` (`WS/WSSpecialize.cpp`, construction of the `ttg.warp_specialize` op).
2. **Shrinking (`optimizePartitionNumWarps`):** no tensors → 1 warp; TMEM loads, stores or allocs → stays >= 4 (`getMinWarpsForOp`); other tensor partitions halve while twice their largest single tensor fits the even per-thread register share.
3. **Padding (`padToMaxWarpGroups`, `AllocateWarpGroups.cpp`):** worker warps are padded to whole warp groups:
   - total launched warps = `num_warps` + 4 × ceil(worker warps / 4) (`ttg.total-num-warps`);
   - base `maxnreg` = 65,536 / total warps / 32, rounded down to a multiple of 8;
   - non-tensor partitions request `minRegAutoWS` (default 24); tensor partitions share the rest;
   - if those requests do not fit, the register split is silently skipped.

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| Auxiliary SMEM estimate >= budget | compile error | "estimated auxiliary shared-memory allocation ... |
| Plan larger than the hardware limit after floors and reuse | requires | The plan must fit the hardware limit; otherwise `OutOfResources` is raised at launch, not at compile time. |
| `num_warps` < 4 | silently ignored | WS bails out before partitioning, so the kernel runs without WS. |
| `num_warps` not 4 or 8 | silently ignored | `supportMMA` for MMAv5 accepts only 4 or 8 warps, so there is no tcgen05 and no gemm partition. |
| WS warp estimate > 16 | silently ignored | All WS metadata is removed. |
| `minRegAutoWS` / `maxRegAutoWS` not a multiple of 8 | compile error | Register requests are emitted per warp group in steps of 8. |
| `TRITON_WS_SMEM_PLAN_SEARCH=1` with no K loop (K == BLOCK_K as a compile-time constant) and more tiles than SMs | wrong answer | Plan search pins every TMEM accumulator to one copy, bypassing the per-iteration accumulator multi-buffering. |
| `num_stages` < 1 | compile error | "nvgpu-warp-specialization requires num-stages >= 1". |

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


**Applying `d_fit`** (with `aux` = 4 KiB).

- 128x256x64, `ES=1`, 1-CTA: `S_stg` 65,536, slot 49,152 -> `d_fit = floor(162,816 / 49,152) = 3`. `num_stages` above the fit adds no operand copies.
- 128x256x64, `ES=2`, separate store, `num_stages=4` (the config above): staging 4 x 32,768 -> `floor(97,280 / 49,152) = 1` balanced level; Phase 4 then adds A copies (A=3 / B=1 at a 4 KiB auxiliary bound, as in the table).
- 128x256x128, `ES=2`: 1-CTA slot 98,304 -> `d_fit` 1; 2-CTA slot 65,536 -> `d_fit` 2. Only the 2-CTA variant double-buffers the operand ring here (`2cta.md`).

## Interactions

See [interactions.md](interactions.md). Pairs where these limits decide the outcome:
- staging copies × `num_stages`
- `EPILOGUE_SUBTILE` × staging copies
- `BLOCK_M`/`BLOCK_N` × accumulator copies
- `data_partition_factor` × TMEM columns
- `num_warps` × extra partitions
- 2-CTA × staging copies

## Hazards

None in `hazards.md`.

## Not yet established

- Which auxiliary-estimate size flips the worked example between B=2 and B=1. That needs the planner debug output for the real kernel.
- Whether an over-budget plan can still pass the planner and then fail at launch, given that `estimateOpScratchBytes` is sized on the IR the planner sees rather than the IR the allocator sees.
