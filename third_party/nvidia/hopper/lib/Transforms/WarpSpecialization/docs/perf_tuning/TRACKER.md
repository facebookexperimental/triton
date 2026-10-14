# Perf Tuning Coverage Tracker

Coverage checklist for the files described in [README.md](README.md).
Reviewers: add rows for missing context. State is `todo`, `in progress` or
`done`.

## perf_tuning files

| File | State | Note |
|------|-------|------|
| [`knobs.md`](knobs.md) | done | Index of the GEMM knobs: one line per knob plus a link to the doc that explains it.  |
| [`resources.md`](resources.md) | done | SMEM, TMEM and register budgets as rules; links to the existing SMEM/TMEM/register docs for the mechanics. Open: its "Not yet established" items. |
| [`interactions.md`](interactions.md) | done | Feature-pair table grouped by feature: pair \| status ("works", "compile error", "silently ignored", "duplicate config", "hangs", "wrong answer", "requires") \| mechanism + code location. Open: its "Not yet established" items. |
| [`hazards.md`](hazards.md) | done | Open wrong-answer and hang conditions, each with its trigger and code location. Entries are deleted when fixed.  |
| [`shape_regimes.md`](shape_regimes.md) | done | Routing index from derived shape quantities to the rules that key on them; no rules of its own.  |

## Feature files

| File | State | Note |
|------|-------|------|
| [`tiles_and_stages.md`](tiles_and_stages.md) | done | `BLOCK_M`/`BLOCK_N`/`BLOCK_K` and `num_stages`. Open: its "Not yet established" items. |
| [`warps_and_registers.md`](warps_and_registers.md) | done | `num_warps` and the AutoWS register budgets. Open: its "Not yet established" items. |
| [`epilogue.md`](epilogue.md) | done | Epilogue subtiling, separate epilogue store, store placement. Open: its "Not yet established" items. |
| [`2cta.md`](2cta.md) | done | 1-CTA vs 2-CTA (`two_ctas`). Open: its "Not yet established" items. |
| [`data_partitioning.md`](data_partitioning.md) | done | `data_partition_factor`. Open: its "Not yet established" items. |
| [`clc_persistence.md`](clc_persistence.md) | done | Static persistent vs dynamic and CLC persistent. Open: its "Not yet established" items. |
| [`tma_loads.md`](tma_loads.md) | done | TMA vs pointer operand loads. Open: its "Not yet established" items. |
| [`tma_stores.md`](tma_stores.md) | done | TMA vs pointer output stores. Open: its "Not yet established" items. |
| [`split_k.md`](split_k.md) | done | Splitting one output tile's K reduction (split-K, decomposeK, stream-K) and the single-K-tile path. Open: its "Not yet established" items. |
| [`pingpong.md`](pingpong.md) | done | Ping-pong scheduling. Open: its "Not yet established" items. |
| [`multicast.md`](multicast.md) | done | TMA multicast. Open: its "Not yet established" items. |
| [`epilogue_prologue_tma_loads.md`](epilogue_prologue_tma_loads.md) | done | TMA loads of extra epilogue/prologue inputs. Open: its "Not yet established" items. |
| [`fused_reductions.md`](fused_reductions.md) | done | Reductions fused into the epilogue. Open: its "Not yet established" items. |

## Docs for undocumented knobs

These go in the existing docs folder (`../`), not in `perf_tuning/`, so that
`perf_tuning/` only links to them.

| Knob | State | Note |
|------|-------|------|
| `auto_tma` | done | `TRITON_AUTO_TMA`, `python/triton/knobs.py`. Documented in [AutoTMAPromotion.md](../AutoTMAPromotion.md). |
| `TRITON_WS_STAGING_COPIES` | done | Read in `WSMemoryPlanner.cpp`. Documented in [TMAStoreWaitPipeline.md § `TRITON_WS_STAGING_COPIES`](../TMAStoreWaitPipeline.md#triton_ws_staging_copies-capping-the-phase-37-target). |
| `TRITON_DISABLE_2CTA_COOPERATIVE_LOADS` | done | Read in `Transform2CTALoads.cpp`. Documented in [BarrierFusion.md § 2-CTA Cooperative Loads](../BarrierFusion.md#2-cta-cooperative-loads-and-triton_disable_2cta_cooperative_loads). |
| `maxnreg` under WS | done | `CUDAOptions.maxnreg`, `third_party/nvidia/backend/compiler.py`. Documented in [RegisterBudgets.md](../RegisterBudgets.md). |
| `disallow_acc_multi_buffer` | done | `AutoWSLoopOptions` in `python/triton/language/core.py`. Documented in [AutoWSAnnotations.md § `disallow_acc_multi_buffer`](../AutoWSAnnotations.md#disallow_acc_multi_buffer). |
| `loop_unroll_factor`, `disable_licm` | done | `AutoWSLoopOptions` in `python/triton/language/core.py`; generic loop transforms, not WS-specific. Documented in [AutoWSAnnotations.md § `loop_unroll_factor` and `disable_licm`](../AutoWSAnnotations.md#loop_unroll_factor-and-disable_licm). |
| Descriptor loads/stores (kernel pattern) | done | `tl.make_tensor_descriptor` or host `TensorDescriptor` with `desc.load` / `desc.store`. Documented in [TMADescriptors.md](../TMADescriptors.md). |

## Reported wrong answers / hangs needing repro + code root cause

Reported but not located in code, so not in [hazards.md](hazards.md). Each needs a reproducer and a root cause in code before it moves there.

| Report | State | Note |
|--------|-------|------|
| Pointer-loaded MMA operand | todo | **Trigger:** a pointer `tl.load` result feeds `tl.dot` inside a WS loop. **Symptom:** an assertion in `NVGPUWarpSpecialization` (`Interval` `Start <= End` in `Allocation.h`) on builds with assertions enabled. Release builds have not been characterized, and a hang has also been seen. |
| Single-copy register copy of a multi-consumer operand, with few K tiles | todo | **Trigger:** an operand loaded by TMA is consumed by both the MMA and register code; its register-to-SMEM copy for the MMA gets one copy; `k_tiles` is small relative to `num_stages`: 1 at 3 stages, or up to 2 at 4 stages; there are more tiles than SMs. **Symptom:** NaN or garbage on later persistent tiles. Some larger-K cases at 4 stages hang. |
| Rank-1 TMA bias with subtiled epilogue | todo | **Trigger:** a rank-1 TMA descriptor for a bias, at BLOCK_M=128, BLOCK_N=256, `EPILOGUE_SUBTILE=2`, 2 stages, 8 warps. **Symptom:** subtile 0 gets the wrong bias in a fraction of tiles. **Workaround:** a 2-D `[1, N]` descriptor or a pointer load. |
| Store of a loop-carried value after the WS loop | todo | **Trigger:** a per-CTA partial is carried across the persistent loop and stored after the loop ends. **Symptom:** a GPU hang. **Workaround:** store it inside the loop, predicated on the last tile. |
| Two K steps per loop body over a shared operand | todo | **Trigger:** two TMA loads of the same operand and two MMAs per loop body, with a deferred reduction. **Symptom:** sparse, large corruption at larger shapes. |
| Data partitioning with generated subtiled regions and a subtiled scale | todo | **Trigger:** `data_partition_factor=2`, `generate_subtiled_region` on, and a per-subtile scale in the epilogue. **Symptom:** a hang. |
| 2-CTA with `smem_alloc_algo=0` | todo | Suspected from code: the one-copy staging pin (`increaseFusedEpilogueCopies`) lives only in the default planner. Not reproduced in testing: under algo 0, output staging stayed a single buffer and results were exact. |
| Register-written SMEM operand read by a 2-CTA MMA | todo | Code facts: `WSLowerToken.cpp`, `processProducerCommitOp` commits with a plain arrive, and `FenceInsertion.cpp`, `FenceInsertionPass` places a CTA-scoped fence (`bCluster=false`) before the MMA, while a 2-CTA MMA also reads the peer CTA's SMEM. Not reproduced: a register-modified A is promoted to TMEM (`PromoteLHSToTMem.cpp`), and a register-modified B is not split (see [hazards.md](hazards.md#2-cta-with-an-unsplit-b)), so no tested config reaches this path. |

## Fixes to existing docs

| Doc | State | Note |
|-----|-------|------|
| [TMEMAllocationHeuristics.md](../TMEMAllocationHeuristics.md) "All TMEM buffers get `buffer.copy = 1`" | todo | Out of date: the heuristic planner's TMEM multi-buffering post-processing (`WSMemoryPlanner.cpp`) raises accumulator copies. |
| [2cta-autoWS-sync.md](../../../../../../../../docs/design/2cta-autoWS-sync.md) "Current Limitations" | done | `BLOCK_M` fallback threshold is 64, not 128. |
| [Overview.md](../Overview.md) | done | Ping-pong uses an mbarrier pair, not named barriers. |
| [triton_tma_multicast.md](../../../../../../../../docs/design/triton_tma_multicast.md) | done | Multicast lowering is planned, not implemented. |
| [BarrierFusion.md](../BarrierFusion.md) | done | `optimizeTMALoads` mixed-group behavior. |

## Knob coverage

One row per entry in [knobs.md](knobs.md): where its when-to-use rules live. `covered` = named in a "Never try" or "When to try it" section of the linked files; `mentioned` = named elsewhere in a rules file only; `todo` = no rules yet.

| Knob | Rules in | State |
|------|----------|-------|
| `BLOCK_M` | [resources.md](resources.md#when-to-try-it-depends-on-the-problem); [tiles_and_stages.md](tiles_and_stages.md#never-try-any-problem); [2cta.md](2cta.md#never-try-any-problem); [data_partitioning.md](data_partitioning.md#never-try-any-problem) | covered |
| `BLOCK_N` | [resources.md](resources.md#when-to-try-it-depends-on-the-problem); [tiles_and_stages.md](tiles_and_stages.md#never-try-any-problem); [epilogue.md](epilogue.md#when-to-try-it-depends-on-the-problem); [2cta.md](2cta.md#never-try-any-problem); [data_partitioning.md](data_partitioning.md#never-try-any-problem) | covered |
| `BLOCK_K` | [resources.md](resources.md#never-try-any-problem); [tiles_and_stages.md](tiles_and_stages.md#never-try-any-problem); [split_k.md](split_k.md#when-to-try-it-depends-on-the-problem); [epilogue_prologue_tma_loads.md](epilogue_prologue_tma_loads.md#never-try-any-problem) | covered |
| `num_stages` | [resources.md](resources.md#never-try-any-problem); [tiles_and_stages.md](tiles_and_stages.md#never-try-any-problem); [epilogue.md](epilogue.md#never-try-any-problem); [2cta.md](2cta.md#when-to-try-it-depends-on-the-problem); [data_partitioning.md](data_partitioning.md#when-to-try-it-depends-on-the-problem); [clc_persistence.md](clc_persistence.md#never-try-any-problem); [tma_stores.md](tma_stores.md#when-to-try-it-depends-on-the-problem) | covered |
| `num_stages` (per loop) | [clc_persistence.md](clc_persistence.md#never-try-any-problem) | covered |
| `flatten` | — | todo |
| `disallow_acc_multi_buffer` | [data_partitioning.md](data_partitioning.md#never-try-any-problem) | covered |
| `loop_unroll_factor`, `disable_licm` | [tiles_and_stages.md](tiles_and_stages.md#never-try-any-problem) | covered |
| `warp_specialize` | [tiles_and_stages.md](tiles_and_stages.md#never-try-any-problem); [2cta.md](2cta.md#never-try-any-problem); [tma_stores.md](tma_stores.md#when-to-try-it-depends-on-the-problem) | covered |
| `data_partition_factor` | [tiles_and_stages.md](tiles_and_stages.md#never-try-any-problem); [data_partitioning.md](data_partitioning.md#never-try-any-problem); [pingpong.md](pingpong.md#when-to-try-it-depends-on-the-problem) | covered |
| `separate_epilogue_store` | [epilogue.md](epilogue.md#never-try-any-problem); [tma_stores.md](tma_stores.md#never-try-any-problem) | covered |
| `merge_epilogue` | — | mentioned |
| `merge_epilogue_to_computation`, `merge_correction` | — | mentioned |
| `pingpongAutoWS` | [clc_persistence.md](clc_persistence.md#never-try-any-problem); [pingpong.md](pingpong.md#when-to-try-it-depends-on-the-problem) | covered |
| `multi_cta` | [split_k.md](split_k.md#never-try-any-problem); [fused_reductions.md](fused_reductions.md#never-try-any-problem) | covered |
| `two_ctas` | [resources.md](resources.md#never-try-any-problem); [tiles_and_stages.md](tiles_and_stages.md#never-try-any-problem); [warps_and_registers.md](warps_and_registers.md#never-try-any-problem); [epilogue.md](epilogue.md#never-try-any-problem); [2cta.md](2cta.md#never-try-any-problem); [data_partitioning.md](data_partitioning.md#never-try-any-problem); [clc_persistence.md](clc_persistence.md#never-try-any-problem); [tma_stores.md](tma_stores.md#never-try-any-problem); [epilogue_prologue_tma_loads.md](epilogue_prologue_tma_loads.md#never-try-any-problem); [fused_reductions.md](fused_reductions.md#never-try-any-problem) | covered |
| `allowDependentTwoCTA` | [2cta.md](2cta.md#never-try-any-problem) | covered |
| `ctas_per_cga` | [2cta.md](2cta.md#never-try-any-problem); [clc_persistence.md](clc_persistence.md#never-try-any-problem); [split_k.md](split_k.md#never-try-any-problem) | covered |
| Tile scheduler | [2cta.md](2cta.md#never-try-any-problem); [clc_persistence.md](clc_persistence.md#never-try-any-problem) | covered |
| `clc` | [clc_persistence.md](clc_persistence.md#never-try-any-problem) | covered |
| `tt.autows` stage/order | — | todo |
| Descriptor loads/stores | — | mentioned |
| `multicast` (per load) | [tma_loads.md](tma_loads.md#never-try-any-problem); [multicast.md](multicast.md#never-try-any-problem) | covered |
| `multicast` | [tma_loads.md](tma_loads.md#never-try-any-problem); [multicast.md](multicast.md#never-try-any-problem) | covered |
| `auto_tma` / `auto_tma_device` | [tma_loads.md](tma_loads.md#never-try-any-problem) | covered |
| `TRITON_DISABLE_2CTA_COOPERATIVE_LOADS` | — | mentioned |
| Epilogue subtiling (ES) | [resources.md](resources.md#when-to-try-it-depends-on-the-problem); [tiles_and_stages.md](tiles_and_stages.md#when-to-try-it-depends-on-the-problem); [epilogue.md](epilogue.md#never-try-any-problem); [2cta.md](2cta.md#never-try-any-problem); [tma_stores.md](tma_stores.md#never-try-any-problem); [epilogue_prologue_tma_loads.md](epilogue_prologue_tma_loads.md#never-try-any-problem) | covered |
| `generate_subtiled_region` | — | mentioned |
| `early_tma_store_lowering` | — | mentioned |
| `tma_store_pipelining` | — | mentioned |
| `TRITON_WS_STAGING_COPIES` | [tiles_and_stages.md](tiles_and_stages.md#when-to-try-it-depends-on-the-problem); [epilogue.md](epilogue.md#never-try-any-problem); [2cta.md](2cta.md#never-try-any-problem); [tma_stores.md](tma_stores.md#never-try-any-problem) | covered |
| `TRITON_WS_TMA_REDUCE_STAGING_COPIES` | [epilogue.md](epilogue.md#never-try-any-problem) | covered |
| `num_warps` | [resources.md](resources.md#never-try-any-problem); [warps_and_registers.md](warps_and_registers.md#never-try-any-problem); [2cta.md](2cta.md#never-try-any-problem); [data_partitioning.md](data_partitioning.md#never-try-any-problem) | covered |
| `minRegAutoWS` | [resources.md](resources.md#never-try-any-problem); [warps_and_registers.md](warps_and_registers.md#never-try-any-problem) | covered |
| `maxRegAutoWS` | [resources.md](resources.md#never-try-any-problem); [warps_and_registers.md](warps_and_registers.md#never-try-any-problem) | covered |
| `maxnreg` | [resources.md](resources.md#when-to-try-it-depends-on-the-problem); [warps_and_registers.md](warps_and_registers.md#never-try-any-problem) | covered |
| `sched4reg`, `enable_tree_reduction` | — | todo |
| `tmem_alloc_algo` | — | todo |
| `smem_alloc_algo` | [epilogue.md](epilogue.md#never-try-any-problem) | covered |
| `smem_budget` | — | mentioned |
| `smem_circular_reuse` | — | todo |
| `tt.autows` channels | — | todo |
| `mem_plan_pick` + `TRITON_WS_MEM_PLAN_TOPK` / `_PICK` | — | todo |
| `TRITON_WS_SMEM_PLAN_SEARCH` | [resources.md](resources.md#never-try-any-problem); [tiles_and_stages.md](tiles_and_stages.md#never-try-any-problem) | covered |
| `TRITON_ENABLE_INTERLEAVE_TMEM`‡, `TRITON_ENABLE_UNIFY_WS_BARRIER_LOCATIONS`‡, `TRITON_DISABLE_WSBARRIER_REORDER`, `TRITON_PROMOTE_MBAR_TO_NAMED_BARRIER` | — | mentioned |
| `TRITON_WS_TILE_PREFETCH_DEPTH`‡ | — | mentioned |
| Alternative schedulers: `TRITON_USE_MODULO_SCHEDULE` (+ `TRITON_MODULO_*`), `TRITON_USE_JOINT_SCHEDULE`‡, `TRITON_USE_LLM_SCHEDULE`‡, `TRITON_USE_LIST_SCHEDULE` | — | todo |
| `list_schedule_pick` | — | todo |

## Other

| Item | State | Note |
|------|-------|------|
| Inductor reference definition | todo | Definition pending. Until then `knobs.md` has no column for what Inductor sets. |
| Leaves outside AutoWS | todo | cuBLAS or non-WS leaves; later. |
