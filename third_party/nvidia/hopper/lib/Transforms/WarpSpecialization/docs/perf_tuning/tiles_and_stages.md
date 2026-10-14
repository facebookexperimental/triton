# Tiles and stages (autoWS GEMM, sm100)

Scope:
- the tile shape (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`);
- `num_stages`;
- how operand ring depth comes out of the memory planner;
- which tile/stage configs are illegal, silently changed, or duplicates.

Budget numbers are in [resources.md](resources.md). Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` (`WS/`) unless they start at the repository root.

## Knobs

| Knob | How set | What it does | Existing doc |
|---|---|---|---|
| `BLOCK_M`, `BLOCK_N`, `BLOCK_K` | kernel constexprs (tile shape of `tl.dot`) | Size every operand slot, the accumulator, and the epilogue tile | — |
| `num_stages` | kernel compile option (`triton.Config(num_stages=...)`) | Upper bound on the copies of every innermost operand buffer and of output staging | [SmemAllocationDesign.md, "Phase 4: Iterative Copy Increase"](../SmemAllocationDesign.md#phase-4-iterative-copy-increase) |
| loop `num_stages` | `tl.range(..., num_stages=)` | Software-pipeline depth of that loop; inert on a dynamic or CLC `while` | [AutoWSAnnotations.md, "`num_stages` / pipelining on a `while`: not applicable"](../AutoWSAnnotations.md#num_stages--pipelining-on-a-while-not-applicable) |
| per-operand copies | `tl.dot(..., attrs={"channels": ["opndA,smem,<copies>,<id>", ...]})` | Pins one operand's buffer depth and id | [AnnotationBasedBufferPreAssignment.md, "User-Facing API"](../AnnotationBasedBufferPreAssignment.md#user-facing-api) |
| `smem_budget` | `tl.range(..., smem_budget=)` | Replaces the target SMEM limit as the planner's starting budget | [AutoWSAnnotations.md, "Annotation → IR attribute → consumer"](../AutoWSAnnotations.md#annotation--ir-attribute--consumer) |

## Never try (any problem)

| Condition | Status | Why | Code |
|---|---|---|---|
| `BLOCK_M` a power of 2 below 64 (e.g. 32) | silently ignored | MMAv5 needs a power-of-2 M that is a multiple of 64, so the dot lowers to mma.sync and autoWS has no gemm partition | `lib/Analysis/Utility.cpp`, `isMmaOutputShapeSupported` / `supportMMA`; `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`, `getMMAVersionSafe` |
| `BLOCK_N` not a power of 2 | silently ignored | Same check: N must be a power of 2 and a multiple of 8, so there is no tcgen05 | `lib/Analysis/Utility.cpp`, `isMmaOutputShapeSupported` |
| `BLOCK_K` < 256 / operand bit width (16 for bf16, 32 for fp8) | silently ignored | No tcgen05: "If k size is smaller than the native mma size, we cannot use MMA" | `lib/Analysis/Utility.cpp`, `supportMMA` |
| fp32 × fp32 dot with `input_precision="ieee"` | silently ignored | No tensor-core MMA: `supportMMA` rejects fp32 operands. `"tf32"`, `"tf32x3"`, `"bf16x3"` and `"bf16x6"` all reach tcgen05, because `F32DotTC` rewrites the multi-pass forms into tf32 or bf16 dots before `AccelerateMatmul` | `lib/Analysis/Utility.cpp`, `supportMMA`; `lib/Dialect/TritonGPU/Transforms/F32DotTC.cpp`; `third_party/nvidia/backend/compiler.py`, `make_ttgir` (`add_f32_dot_tc` before `add_accelerate_matmul`) |
| `BLOCK_M=128` with `data_partition_factor=2` | duplicate config | Runs as dp=1. A 64-row M slice would change TMEM `blockM`, so the pass warns "skipping M-dimension data partitioning" and drops partitioning. It does not fall back to an N split, even when `BLOCK_N/2` >= 128 | `WS/WSDataPartition.cpp`, `DataPartitionScheme::partitionIsCompatible`, `computePartitionScheme` |
| `BLOCK_M=64` with `data_partition_factor=2` and `BLOCK_N` < 256 | compile error | Neither an M slice >= 64 nor an N slice >= 128 exists, so `computePartitionScheme` fails and the pass signals failure | `WS/WSDataPartition.cpp`, `computePartitionScheme`, `doDataPartition` |
| `tl.dot(two_ctas=True)` with `BLOCK_M` < 64 | silently ignored | Covered by the first row: no tcgen05 at all. `BLOCK_M=64` with `two_ctas` compiles as real 2-CTA | `lib/Analysis/Utility.cpp`, `isMmaOutputShapeSupported` |
| `TRITON_WS_SMEM_PLAN_SEARCH=1` with no K loop (K == BLOCK_K as a compile-time constant) and more tiles than SMs | wrong answer (reproduced) | Plan search keeps TMEM accumulator copies at 1, bypassing the per-iteration accumulator multi-buffering. See [hazards.md](hazards.md#plan-search-pins-tmem-accumulator-copies-to-1) | `WS/WSMemoryPlanner.cpp`, `allocateTmemBuffersViaSearch` |
| `loop_unroll_factor` >= 2 on a persistent warp-specialized loop | compile error | "SMEM circular reuse group is ill-formed": both the unrolled loop and the remainder loop keep `tt.warp_specialize`. See [AutoWSAnnotations.md § `loop_unroll_factor`](../AutoWSAnnotations.md#loop_unroll_factor) | `WS/WSCodePartition.cpp`; `lib/Dialect/Triton/Transforms/LoopUnroll.cpp` |
| `num_stages` < 1 | compile error | "nvgpu-warp-specialization requires num-stages >= 1" | `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp`, `runOnOperation` |
| Annotated `numCopies` below the buffer's cross-stage depth | hangs | Inferred from the planner, not reproduced: it keeps the annotation and only warns, and its own rationale for never lowering a cross-stage floor is that a lower depth "deadlocks at runtime"; annotations bypass that floor | `WS/WSMemoryPlanner.cpp`, `allocateSmemBuffers` Phase 2 ("annotated cross-stage SMEM buffer has buffer.copy=...") |

## Mechanism

**What tile shape changes.** The three tile dimensions set:
- the A slot (`BLOCK_M × BLOCK_K`) and the B slot (`BLOCK_N × BLOCK_K`), in the operand element type;
- TMEM columns per accumulator copy (`BLOCK_N × ceil(BLOCK_M/128)` for fp32), and from that the number of accumulator copies (see [resources.md](resources.md));
- the epilogue tile, in combination with `EPILOGUE_SUBTILE` (see [epilogue.md](epilogue.md)).

The MMA instruction shape follows from the tile: instruction M is 128 when `BLOCK_M` >= 128 and 64 otherwise, instruction N is `min(BLOCK_N, 256)`, and instruction K is 256 / operand bit width; larger tiles are issued as several MMAs (`lib/Dialect/TritonGPU/Transforms/Utility.cpp`, `mmaVersionToInstrShape`).

SMEM is sized from these compile-time shapes alone (`WS/WSMemoryPlanner.cpp`, `getSmemAllocSizeBytes`). A smaller problem never makes a config fit.

**What `num_stages` changes.** It is the planner's `numBuffers`:
- Phase 4 raises the innermost operand buffers, TMA-fed first, then non-TMA, in the order they are first used, one copy at a time;
- it stops when the next copy would exceed the budget or `num_stages`;
- output staging was already raised, up to the same cap, before Phase 4 ran (`increaseFusedEpilogueCopies`).

Mechanics: [SmemAllocationDesign.md, "Phase 4: Iterative Copy Increase"](../SmemAllocationDesign.md#phase-4-iterative-copy-increase) and [TMAStoreWaitPipeline.md, "Phase 3.7: Epilogue Group Copy Increase"](../TMAStoreWaitPipeline.md#phase-37-epilogue-group-copy-increase).

**Resulting operand depth:**
> depth = the largest d <= `num_stages` such that floors + staging + d·(A slot + B slot) fits the budget, plus possibly one more copy of the operand used first.
- With equal-size A and B slots, the two depths differ by at most 1.
- With unequal slots, the smaller operand keeps rising after the larger one stops, so the gap can be larger.

**What it relieves:** deeper operand rings let the load partition run further ahead of the MMA.

**What it costs:** every operand copy is SMEM that staging could have used, and the reverse. Deeper rings also add barrier slots to the auxiliary reservation (`estimateAuxiliarySmemBytes` scales barrier bytes by depth).

**Floors:**
- Correctness floors (cross-stage depth, copy-safety floor) can force copies above `num_stages`.
- In particular, under `data_partition_factor` > 1 with `disallow_acc_multi_buffer`, every TMA operand of the MMA is floored at `num_stages` (`getStaticSmemCopySafetyFloor`).
- That copy-safety floor is given back when it does not fit (`relaxCopySafetyFloorToBudget`). The cross-stage floor is never given back.

## When to try it (depends on the problem)

- **IF** `BLOCK_M` < 64, or `BLOCK_N` is not a power of 2, or `BLOCK_K` < 256/bits, **THEN** skip, **BECAUSE** no tcgen05 MMA is generated.
- **IF** `BLOCK_M=128` and `data_partition_factor=2`, **THEN** skip, **BECAUSE** it is a duplicate config of dp=1.
- **IF** `BLOCK_M=64` and `data_partition_factor=2`, **THEN** only try `BLOCK_N=256`, **BECAUSE** every other `BLOCK_N` fails to compile.
- **IF** two configs differ only in `num_stages`, both are at or above the operand depth that fits, and output staging is single-copy (`EPILOGUE_SUBTILE=1` or `two_ctas`), **THEN** keep the smaller (this removes the larger), **BECAUSE** the operand plan is the same; only the barrier reservation and the loop schedule differ. With `EPILOGUE_SUBTILE` >= 2, a larger `num_stages` deepens staging (Phase 3.7, `increaseFusedEpilogueCopies`) and makes the operand ring shallower, so it is a different config.
- **IF** the goal is a deeper operand ring at a fixed tile, **THEN** lower staging copies first (`TRITON_WS_STAGING_COPIES`, or `EPILOGUE_SUBTILE` with a same-partition store), **BECAUSE** staging is reserved before operands under the same cap.
- **IF** you want asymmetric A/B depths beyond ±1 at equal slot sizes, **THEN** annotate the channel copies, and never below the cross-stage depth, **BECAUSE** the unannotated planner raises A and B in lockstep.

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
