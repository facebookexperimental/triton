# Feature interactions (autoWS GEMM, sm100)

Scope: what happens when two autoWS GEMM features or knobs are combined. Each row is one pair. For what a single feature does, follow the links in the feature files ([resources.md](resources.md), [tiles_and_stages.md](tiles_and_stages.md), [warps_and_registers.md](warps_and_registers.md), `epilogue.md`).

Status is one of:
- **works**
- **compile error**
- **silently ignored**: the knob or feature has no effect, possibly with only a compiler warning or debug log
- **duplicate config**: compiles to the same kernel as another config
- **hangs**
- **wrong answer**
- **requires**: the first feature needs the second

Open wrong-answer and hang conditions are listed in [hazards.md](hazards.md) and only linked here.

Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` (`WS/`) unless they start at the repository root. Design docs are under `docs/design/` at the repository root.

## 2-CTA (`tl.dot(two_ctas=True)`)

| Pair | Status | Mechanism | Code ref |
|---|---|---|---|
| `two_ctas` × `ctas_per_cga=(2,1,1)` | requires | Without cluster-dim-x >= 2 the dot falls back to a 1-CTA MMA with a warning, so the config duplicates the 1-CTA one | `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`, `BlockedToMMAv5` ("two_ctas=True requires ctas_per_cga=(2,1,1) or larger") |
| `two_ctas` × cluster shape other than (2,1,1) | requires | Only (2,1,1) is supported; (2,2,1) and (4,1,1) are not; the only code check is cluster-dim-x >= 2 | design doc: [2cta-autoWS-sync.md § Supported Configurations](../../../../../../../../docs/design/2cta-autoWS-sync.md#supported-configurations) |
| `two_ctas` × `BLOCK_M` < 64 | silently ignored | Falls back to 1-CTA MMA with a warning. `BLOCK_M=64` is accepted and uses the `TwoCTA_RHS` TMEM mode | `AccelerateMatmul.cpp`, `BlockedToMMAv5` ("two_ctas=True with BLOCK_M < 64 is not supported") |
| `two_ctas` × a different `two_ctas` on another dot | compile error | All tcgen05 ops in a kernel must use one `cta_group` | `lib/Dialect/TritonNvidiaGPU/Transforms/CheckMatmulTwoCTAs.cpp` ("inconsistent two_ctas setting across matmuls") |
| `two_ctas` × a 2-CTA dot consuming another 2-CTA dot's result | compile error | Rejected unless the pass option `allow-dependent-chains` (the dependent-exchange path) is set | `CheckMatmulTwoCTAs.cpp`, `allowDependentChains` |
| `two_ctas` × rhs-only scaled `tl.dot_scaled` | silently ignored | The dot is transposed, so 2-CTA is dropped with a warning | `AccelerateMatmul.cpp`, `ScaledBlockedToMMAv5` ("two_ctas=True is not supported when only the rhs ...") |
| `two_ctas` × B from a descriptor (TMA) load | requires | Only a descriptor-loaded B is split into per-CTA halves (`BLOCK_N/2` each); other B operands skip the split with a debug log only. See [hazards.md](hazards.md) for the unsplit-B case | `third_party/nvidia/hopper/lib/Transforms/Transform2CTALoads.cpp` |
| `two_ctas` × TMA (descriptor) output store, under Meta WS | requires | Pointer-store epilogues do not form the Meta WS epilogue partition; only descriptor stores and TMA copies count as epilogue stores | design doc: [2cta-autoWS-sync.md § Current Limitations](../../../../../../../../docs/design/2cta-autoWS-sync.md#current-limitations); `WS/PartitionSchedulingMeta.cpp`, `isEpilogueStoreOp` |
| `two_ctas` × `BLOCK_N/2` < 16 | wrong answer (reproduced) | The B split is skipped with only a debug log while the MMA still reads half-width B; silent. See [hazards.md](hazards.md#2-cta-with-block_n2-below-the-b-split-minimum) | `third_party/nvidia/hopper/lib/Transforms/Transform2CTALoads.cpp`, `transformBLoad`, `transformSplitBLoad` |
| `two_ctas` × pointer-loaded B, under Meta WS | compile error | "pipeliner doesn't know how to predicate this op" | `lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp`, `predicateOp` |
| `two_ctas` × Meta WS off | wrong answer (reproduced) | A persistent non-WS 2-CTA GEMM is wrong on every run with a pointer-loaded B and on most runs with a descriptor-loaded B. See [hazards.md](hazards.md#2-cta-without-warp-specialization) | `third_party/nvidia/hopper/lib/Transforms/Insert2CTASync.cpp`; `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`, `convertDotImpl` |
| `two_ctas` × `warp_specialize=False`, under Meta WS | hangs (reproduced) | The first iteration completes, then the kernel hangs. See [hazards.md](hazards.md#2-cta-without-warp-specialization) | same |
| `two_ctas` × output staging copies (`TRITON_WS_STAGING_COPIES`, or depth from `num_stages`) | duplicate config | Ordinary TMA output staging is held at one copy under 2-CTA | `WS/WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |
| `two_ctas` × a staging correctness floor > 1 | compile error | "2CTA ordinary TMA output staging requires one copy; the correctness floor requires N" | `WS/WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |
| `two_ctas` × `num_stages` | works | The cross-CTA rendezvous barrier is multi-buffered to the MMA operand depth and indexed by the same slot and phase | `third_party/nvidia/hopper/lib/Transforms/Insert2CTASync.cpp`, `getOperandPipelineDepth`; design doc: [2cta-autoWS-sync.md § Current Limitations](../../../../../../../../docs/design/2cta-autoWS-sync.md#current-limitations) |
| `two_ctas` × tile schedule | requires | Pair-aligned tile schedule: the two CTAs of a cluster must cover adjacent `pid_m` on the same `pid_n`; odd `grid_m` must be padded to even. The compiler cannot verify this, and a misaligned schedule gives a wrong answer | design doc: [2cta-autoWS-sync.md § Current Limitations](../../../../../../../../docs/design/2cta-autoWS-sync.md#current-limitations) |
| `two_ctas` × `data_partition_factor` | works | Data partitioning remaps pinned buffer ids per partition and serializes the DP0 chain before DP1 | design doc: [2cta-autoWS-sync.md § Current Limitations](../../../../../../../../docs/design/2cta-autoWS-sync.md#current-limitations) |
| `two_ctas` × CLC tile scheduler | works | Hardware CLC gives cluster-level tile ownership; clustered CLC state is hoisted and captured by every partition | design doc: [DynamicPersistentAutoWSGaps.md § 2-CTA Status](../DynamicPersistentAutoWSGaps.md#2-cta-status) |
| `two_ctas` × CLC grid padding | requires | Every grid dim must be divisible by the cluster dim, padded CTAs must still run (no early exit), and every member must reach `advance()` | design doc: [triton-clc-tile-scheduler.md § PTX cluster grouping contract](../../../../../../../../docs/design/triton-clc-tile-scheduler.md#ptx-cluster-grouping-contract); `python/triton/language/schedule.py` |
| `two_ctas` × computation-partition 8-warp override | silently ignored | The "computation partition = 8 warps when a reduction partition exists" override is skipped on 2-CTA kernels | `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp`, `optimizePartitionNumWarps` |
| `two_ctas` × post-WS loop rescheduling | works | 2-CTA re-runs loop scheduling after WS; 1-CTA does not, because re-running it miscompiles kernels with a separate epilogue-store partition | `third_party/nvidia/backend/compiler.py`, Meta WS branch of `make_ttgir` |

## Data partitioning (`data_partition_factor`)

| Pair | Status | Mechanism | Code ref |
|---|---|---|---|
| dp=2 × `BLOCK_M=128` | duplicate config | Runs as dp=1. The 64-row M slice would change TMEM `blockM`, so partitioning is skipped with a warning; the N split is never tried | `WS/WSDataPartition.cpp`, `DataPartitionScheme::partitionIsCompatible`, `computePartitionScheme` |
| dp=2 × `BLOCK_M=64`, `BLOCK_N` < 256 | compile error | No M slice >= 64 and no N slice >= 128 exists; `doDataPartition` fails | `WS/WSDataPartition.cpp`, `computePartitionScheme`, `doDataPartition` |
| dp=2 × `BLOCK_M=256` | works | M is split into two 128-row halves | `WS/WSDataPartition.cpp`, `computePartitionScheme` |
| dp=2 × `BLOCK_N` (TMEM) | works | Two `128 × BLOCK_N` fp32 accumulators take `2 × BLOCK_N` columns, so at `BLOCK_N=256` there is one accumulator copy (no cross-tile overlap). A footprint over 512 columns is rejected as OutOfResources | `WS/WSMemoryPlanner.cpp`, TMEM multi-buffering post-processing; `python/triton/compiler/compiler.py`, `CompiledKernel._init_handles` (`tmem_size` check) |
| dp > 1 × accumulator-chained MMAs in partitions with no common partition | compile error | "warp specialization assigned accumulator-chained MMAs with no common partition" | `WS/PartitionSchedulingMeta.cpp`, accumulator-chain check |
| dp > 1 × host `TensorDescType` kernel argument used as a loop init arg | compile error | `doDataPartition` returns false ("used as loop init arg; not supported yet"), so the pass fails | `WS/WSDataPartition.cpp`, `doDataPartition` |
| dp on one loop × a different dp on another loop in the same function | silently ignored | The consistency check is an `assert` only; release builds keep the last loop's value | `WS/WSDataPartition.cpp`, `runOnFuncOp` ("data partition factor mismatch") |
| dp > 1 × `disallow_acc_multi_buffer` | works | Operand rings are pinned to `num_stages`: every TMA operand of the MMA is floored at the configured depth; the floor is given back only if it does not fit | `WS/WSMemoryPlanner.cpp`, `getStaticSmemCopySafetyFloor`, `relaxCopySafetyFloorToBudget` |
| dp=2 × `num_warps=8` | silently ignored | WS is stripped when the warp estimate exceeds 16; each extra TMEM-holding partition adds 4 (see [warps_and_registers.md](warps_and_registers.md)). Whether the GEMM templates cross 16 is not yet established | `WS/PartitionSchedulingMeta.cpp`, warp budget estimate |

## Pingpong (`pingpongAutoWS`)

| Pair | Status | Mechanism | Code ref |
|---|---|---|---|
| pingpong × plain GEMM / addmm epilogue | silently ignored | On sm_100 only SFU math (`exp`, `exp2`, `sin`, `cos`, `tanh`, `sqrt`, `rsqrt`, or SFU `.approx` asm) with a rank > 1 result is "expensive"; MMAs are never pingponged | `WS/PingPong.cpp`, expensive-op classification (`case 100`) |
| pingpong × capability other than 90 or 100 | silently ignored | Only those two capabilities have a classification | `WS/PingPong.cpp` |
| pingpong × `data_partition_factor` | requires | For a GEMM epilogue, pingpong needs dp=2: a region needs exactly 2 partitions holding the same SFU op | `WS/PingPong.cpp`, `doPingPongPrep`; [PingPongScheduling.md § Step 2: Validate and Assign `pingpong_id`](../PingPongScheduling.md#step-2-validate-and-assign-pingpong_id) |
| pingpong × CLC / dynamic `scf.while` tile loop | silently ignored | A region needs an enclosing `scf.for`; epilogue ops under a `while` tile loop have none | `WS/PingPong.cpp`, `doPingPongPrep` |

## TMA multicast

| Pair | Status | Mechanism | Code ref |
|---|---|---|---|
| `desc.load(multicast=True)` or `multicast` compile option × any config | silently ignored | The planning pass only attaches `tt.multicast_axes`; no TMA lowering reads it, so no `.multicast::cluster` copy is emitted | `lib/Dialect/TritonNvidiaGPU/Transforms/TMAMulticast.cpp`; `WS/WSLowerMem.cpp`, descriptor-load lowering |

## Persistent schedules (static, dynamic, CLC)

| Pair | Status | Mechanism | Code ref |
|---|---|---|---|
| loop `num_stages` × CLC or dynamic outer `while` | silently ignored | Pipelining does not apply to a raw `scf.while`; stages belong on the inner K loop | [AutoWSAnnotations.md § `num_stages` / pipelining on a `while`: not applicable](../AutoWSAnnotations.md#num_stages--pipelining-on-a-while-not-applicable) |
| CLC / dynamic `while` × `data_partition_factor`, `separate_epilogue_store`, generated subtiled regions | works | Covered end to end on Blackwell | design doc: [DynamicPersistentAutoWSGaps.md § Frontend and Configuration Gaps](../DynamicPersistentAutoWSGaps.md#frontend-and-configuration-gaps) |
| upstream `clc=True` option × Meta WS | compile error | "clc=True does not support Meta AutoWS; use tl.clc_tile_scheduler()" | `third_party/nvidia/backend/compiler.py`, `CUDAOptions` validation |
| upstream `clc=True` × physical cluster (`ctas_per_cga`, `cluster_dims`) | compile error | `clc=True` requires `num_ctas=1` and no physical cluster | `third_party/nvidia/backend/compiler.py`, `CUDAOptions.__post_init__` |

## Stages and the memory planner

| Pair | Status | Mechanism | Code ref |
|---|---|---|---|
| `num_stages` × SMEM budget | duplicate config | Above the depth that fits: operand copies stop at the budget; staging is reserved first under the same cap | `WS/WSMemoryPlanner.cpp`, `allocateSmemBuffers`, `increaseFusedEpilogueCopies` |
| `num_stages` < 1 × WS | compile error | "nvgpu-warp-specialization requires num-stages >= 1" | `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp` |
| annotated channel copies × cross-stage depth | hangs | When the annotation is below the cross-stage depth: the planner keeps the annotation with only a warning, though its own rationale is that a depth below the cross-stage floor deadlocks | `WS/WSMemoryPlanner.cpp`, `allocateSmemBuffers` Phase 2 ("annotated cross-stage SMEM buffer has buffer.copy=...") |
| `TRITON_WS_SMEM_PLAN_SEARCH=1` × accumulator multi-buffering | silently ignored | Accumulator copies are pinned to 1: the TMEM plan search stamps `buffer.copy=1` on every TMEM alloc; kernels with scaled MMA or a `SubtiledRegionOp` fall back to the heuristic planner and keep multi-buffering | `WS/WSMemoryPlanner.cpp`, `allocateTmemBuffersViaSearch` |

## Warps and registers

| Pair | Status | Mechanism | Code ref |
|---|---|---|---|
| `num_warps` < 4 × WS | silently ignored | No WS: "Warp specialization requires at least 4 warps" | `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp`, `runOnFuncOp` |
| `num_warps` not 4 or 8 × tcgen05 | silently ignored | Falls back to mma.sync, so there is no gemm partition; MMAv5 accepts only 4 or 8 warps | `lib/Analysis/Utility.cpp`, `supportMMA` |
| `num_warps` × number of partitions | silently ignored | WS is stripped when the estimate exceeds 16. Estimate = `num_warps` + Σ per-partition minimum warps (4 with TMEM ops) | `WS/PartitionSchedulingMeta.cpp` ("Warp budget exceeded"); `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`, `getMinWarpsForOp` |
| `minRegAutoWS` / `maxRegAutoWS` × values not a multiple of 8 | compile error | ValueError from option validation | `third_party/nvidia/backend/compiler.py`, `_check_reg_auto_ws_alignment` |
| `maxnreg` × register requests that do not fit | silently ignored | No per-partition `setmaxnreg`: register allocation returns without setting actual registers | `lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp` |

## Tiles, loads and stores

| Pair | Status | Mechanism | Code ref |
|---|---|---|---|
| `BLOCK_M` a power of 2 below 64, `BLOCK_N` not a power of 2, or `BLOCK_K` < 256/bits × tcgen05 | silently ignored | No tcgen05 (MMAv5 shape rules) | `lib/Analysis/Utility.cpp`, `isMmaOutputShapeSupported`, `supportMMA` |
| fp32 × fp32 dot × `input_precision` other than `"tf32"` | silently ignored | No tensor-core MMA: `supportMMA` rejects fp32 operands without TF32 | `lib/Analysis/Utility.cpp`, `supportMMA` |
| pointer `tl.load` operand × Meta WS | works | Not producer-pipelined. Only descriptor loads/gathers are routed to the load partition; a pointer-loaded operand reaches the MMA through a register-to-SMEM copy in a consumer partition | `WS/PartitionSchedulingMeta.cpp`, `categorizeLoads`; `WS/WSCodePartition.cpp`, `isChannelAnchorOp` |
| descriptor load × a second (register) consumer besides the MMA | works | Costs two buffers. A single-use load is written by TMA straight into the MMA buffer; with another consumer it gets its own landing buffer plus a register copy into the MMA buffer | `WS/WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS` |
| `separate_epilogue_store` × pointer (`tl.store`) epilogue | duplicate config | Only descriptor stores, TMA copies and TMA store waits count as epilogue stores; with none, no store partition is created | `WS/PartitionSchedulingMeta.cpp`, `isEpilogueStoreOp`, `createPartitionLayout` |
| `merge_epilogue` × separate epilogue partition | works | Suppresses the separate epilogue partition. Non-store epilogue ops are routed into an existing partition | `WS/PartitionSchedulingMeta.cpp`, `createPartitionLayout` |
| `tl.atomic_*` × WS tile loop | wrong answer | A replicated tensor atomic makes the atomic broadcast reject, which strips WS for the kernel. An unpartitioned atomic is dropped. See [hazards.md](hazards.md) | `WS/WSAtomicBroadcast.cpp`; `WS/PartitionSchedulingMeta.cpp` (late store fixup skips atomics) |

## Epilogue subtiling and staging

| Pair | Status | Mechanism | Code ref |
|---|---|---|---|
| `TRITON_WS_STAGING_COPIES` × `ES=1` | duplicate config | A staging group with fewer than 2 slices is never bumped | `WS/WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |
| staging copies × same-partition store (`separate_epilogue_store=False`) | duplicate config | For caps that do not divide `ES`: the wait_group-drained ring accepts only copy counts that divide the slice count; other caps round down | `WS/WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies`; [TMAStoreWaitPipeline.md § `K \| S` cap for same-partition (wait_group-drained) staging](../TMAStoreWaitPipeline.md#k--s-cap-for-same-partition-wait_group-drained-staging) |
| staging copies × separate store partition | works | Any count up to the cap. Cross-partition staging rotates on a continuous mbarrier counter | same |
| `TRITON_WS_STAGING_COPIES` × `num_stages` | duplicate config | Above `num_stages`: the bump target is min(`num_stages`, cap); only `TRITON_WS_TMA_REDUCE_STAGING_COPIES` may exceed `num_stages` | `WS/WSMemoryPlanner.cpp`, `getStagingCopiesCap`, `getTmaReduceStagingCopies` |
| `ES` × partition warps | works | An 8-warp epilogue partition can drop to 4: warp halving sizes from the largest single tensor, which shrinks as `ES` grows | `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp`, `optimizePartitionNumWarps` |
| `ES` >= 2 × epilogue that reads one accumulator slice twice (fused reduction) | works | `InterleaveTMem` keeps both reads of a subtile inside the accumulator wait/arrive: barrier moves are bounded by the pre-reorder block order | `lib/Dialect/TritonNvidiaGPU/Transforms/InterleaveTMem.cpp`, `buildOriginalBlockOrder`, `processBlock` |

## Not yet established

Tracked in [TRACKER.md](TRACKER.md).

- `two_ctas` × `BLOCK_M=64` (`TwoCTA_RHS`): correctness not run.
- `two_ctas` × `BLOCK_N=64`: an LLVM abort was observed; it may no longer reproduce now that the halved tile's swizzle is re-derived.
- `two_ctas` × atomic dynamic scheduler: documented as unsupported (no cluster tile-claim protocol); not rejected in code; failure mode unknown.
- `two_ctas` × `smem_alloc_algo=0`: the one-copy staging pin is only on the default allocator path.
- dp=2 × `BLOCK_M=64`, `BLOCK_N=256` (N split): correctness not run.
- dp=2 × generated subtiled region × per-slice scaling: hang observed.
- `num_warps=8` × dp=2 against the 16-warp estimate, for the GEMM templates.
- pingpong × GEMM epilogue with SFU math at dp=2: no data.
- CLC / dynamic `while` × `merge_*`, SMEM/TMEM allocation options, register budgets, pingpong: no end-to-end contract.
- `early_tma_store_lowering=False` × staging knobs: whether staging leaves the planner's accounting.
- `ES` × runtime once a config fits; best `ES` per tile and 2-CTA setting.
