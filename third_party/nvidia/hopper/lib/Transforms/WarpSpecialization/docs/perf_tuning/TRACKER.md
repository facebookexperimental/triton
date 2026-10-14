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
| `shape_regimes.md` | todo | Routing index from derived shape quantities to the rules that key on them; no rules of its own. |

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
| Register-written SMEM operand read by a 2-CTA MMA | todo | Code facts: `WSLowerToken.cpp`, `processProducerCommitOp` commits with a plain arrive, and `FenceInsertion.cpp`, `FenceInsertionPass` places a CTA-scoped fence (`bCluster=false`) before the MMA, while a 2-CTA MMA also reads the peer CTA's SMEM. Not reproduced: a register-modified A is promoted to TMEM (`PromoteLHSToTMem.cpp`), and a register-modified B is not split (see [hazards.md](hazards.md#2-cta-with-an-unsplit-b)), so no tested config reaches this path. |

## Fixes to existing docs

| Doc | State | Note |
|-----|-------|------|
| [TMEMAllocationHeuristics.md](../TMEMAllocationHeuristics.md) "All TMEM buffers get `buffer.copy = 1`" | todo | Out of date: the heuristic planner's TMEM multi-buffering post-processing (`WSMemoryPlanner.cpp`) raises accumulator copies. |

## Knob coverage

One row per entry in [knobs.md](knobs.md), with its priority (P0 highest, P2 lowest). Covered in: the files whose "Pruning ruleset" or "Mechanism" section names the knob. `covered` = named there; `mentioned` = named elsewhere in a feature file only; `todo` = not covered yet.

| Knob | Priority | Covered in | State |
|------|----------|------------|-------|
| `BLOCK_M` | P0 | [resources.md](resources.md#tmem); [tiles_and_stages.md](tiles_and_stages.md#pruning-ruleset); [warps_and_registers.md](warps_and_registers.md#mechanism); [epilogue.md](epilogue.md#mechanism); [2cta.md](2cta.md#pruning-ruleset); [data_partitioning.md](data_partitioning.md#pruning-ruleset); [tma_loads.md](tma_loads.md#mechanism); [tma_stores.md](tma_stores.md#mechanism); [split_k.md](split_k.md#mechanism); [pingpong.md](pingpong.md#pruning-ruleset) | covered |
| `BLOCK_N` | P0 | [resources.md](resources.md#tmem); [tiles_and_stages.md](tiles_and_stages.md#pruning-ruleset); [epilogue.md](epilogue.md#mechanism); [2cta.md](2cta.md#pruning-ruleset); [data_partitioning.md](data_partitioning.md#pruning-ruleset); [tma_loads.md](tma_loads.md#mechanism); [tma_stores.md](tma_stores.md#mechanism); [split_k.md](split_k.md#mechanism); [epilogue_prologue_tma_loads.md](epilogue_prologue_tma_loads.md#mechanism) | covered |
| `BLOCK_K` | P0 | [resources.md](resources.md#pruning-ruleset); [tiles_and_stages.md](tiles_and_stages.md#pruning-ruleset); [2cta.md](2cta.md#mechanism); [tma_loads.md](tma_loads.md#mechanism); [split_k.md](split_k.md#mechanism); [epilogue_prologue_tma_loads.md](epilogue_prologue_tma_loads.md#pruning-ruleset) | covered |
| `num_stages` | P0 | [resources.md](resources.md#pruning-ruleset); [tiles_and_stages.md](tiles_and_stages.md#pruning-ruleset); [epilogue.md](epilogue.md#pruning-ruleset); [2cta.md](2cta.md#mechanism); [data_partitioning.md](data_partitioning.md#pruning-ruleset); [clc_persistence.md](clc_persistence.md#pruning-ruleset); [tma_stores.md](tma_stores.md#mechanism); [fused_reductions.md](fused_reductions.md#mechanism) | covered |
| `flatten` | P0 | [tiles_and_stages.md](tiles_and_stages.md#pruning-ruleset) | covered |
| `warp_specialize` | P0 | [resources.md](resources.md#registers-and-warps); [tiles_and_stages.md](tiles_and_stages.md#pruning-ruleset); [warps_and_registers.md](warps_and_registers.md#mechanism); [2cta.md](2cta.md#pruning-ruleset); [tma_stores.md](tma_stores.md#mechanism) | covered |
| `data_partition_factor` | P0 | [tiles_and_stages.md](tiles_and_stages.md#pruning-ruleset); [warps_and_registers.md](warps_and_registers.md#mechanism); [data_partitioning.md](data_partitioning.md#pruning-ruleset); [clc_persistence.md](clc_persistence.md#mechanism); [pingpong.md](pingpong.md#mechanism) | covered |
| `separate_epilogue_store` | P0 | [epilogue.md](epilogue.md#pruning-ruleset); [clc_persistence.md](clc_persistence.md#mechanism); [tma_stores.md](tma_stores.md#pruning-ruleset); [pingpong.md](pingpong.md#pruning-ruleset) | covered |
| `two_ctas` | P0 | [tiles_and_stages.md](tiles_and_stages.md#pruning-ruleset); [epilogue.md](epilogue.md#pruning-ruleset); [2cta.md](2cta.md#pruning-ruleset); [clc_persistence.md](clc_persistence.md#pruning-ruleset); [tma_stores.md](tma_stores.md#pruning-ruleset); [epilogue_prologue_tma_loads.md](epilogue_prologue_tma_loads.md#pruning-ruleset) | covered |
| `ctas_per_cga` | P0 | [2cta.md](2cta.md#pruning-ruleset); [clc_persistence.md](clc_persistence.md#pruning-ruleset); [split_k.md](split_k.md#pruning-ruleset) | covered |
| Tile scheduler | P0 (static persistent); P1 (CLC, dynamic) | [2cta.md](2cta.md#pruning-ruleset); [clc_persistence.md](clc_persistence.md#pruning-ruleset) | covered |
| Descriptor loads/stores | P0 | [tma_loads.md](tma_loads.md#mechanism); [split_k.md](split_k.md#mechanism); [epilogue_prologue_tma_loads.md](epilogue_prologue_tma_loads.md#mechanism) | covered |
| Epilogue subtiling (ES) | P0 | [resources.md](resources.md#smem); [tiles_and_stages.md](tiles_and_stages.md#mechanism); [warps_and_registers.md](warps_and_registers.md#mechanism); [epilogue.md](epilogue.md#pruning-ruleset); [2cta.md](2cta.md#pruning-ruleset); [tma_stores.md](tma_stores.md#pruning-ruleset); [epilogue_prologue_tma_loads.md](epilogue_prologue_tma_loads.md#pruning-ruleset); [fused_reductions.md](fused_reductions.md#mechanism) | covered |
| `early_tma_store_lowering` | P0 | [tma_stores.md](tma_stores.md#mechanism) | covered |
| `tma_store_pipelining` | P0 | [tma_stores.md](tma_stores.md#mechanism) | covered |
| `TRITON_WS_STAGING_COPIES` | P0 | [epilogue.md](epilogue.md#pruning-ruleset); [2cta.md](2cta.md#pruning-ruleset); [tma_stores.md](tma_stores.md#pruning-ruleset); [fused_reductions.md](fused_reductions.md#mechanism) | covered |
| `TRITON_WS_TMA_REDUCE_STAGING_COPIES` | P0 | [fused_reductions.md](fused_reductions.md#mechanism) | covered |
| `num_warps` | P0 | [resources.md](resources.md#pruning-ruleset); [warps_and_registers.md](warps_and_registers.md#pruning-ruleset); [2cta.md](2cta.md#pruning-ruleset); [data_partitioning.md](data_partitioning.md#pruning-ruleset) | covered |
| `num_stages` (per loop) | P1 | [clc_persistence.md](clc_persistence.md#pruning-ruleset) | covered |
| `disallow_acc_multi_buffer` | P1 | [tiles_and_stages.md](tiles_and_stages.md#mechanism); [data_partitioning.md](data_partitioning.md#pruning-ruleset) | covered |
| `merge_epilogue` | P1 | [epilogue.md](epilogue.md#mechanism); [tma_stores.md](tma_stores.md#mechanism); [pingpong.md](pingpong.md#pruning-ruleset) | covered |
| `multicast` (per load) | P1 | [tma_loads.md](tma_loads.md#pruning-ruleset); [multicast.md](multicast.md#pruning-ruleset) | covered |
| `multicast` | P1 | [tma_loads.md](tma_loads.md#pruning-ruleset); [multicast.md](multicast.md#pruning-ruleset) | covered |
| `generate_subtiled_region` | P1 | [epilogue.md](epilogue.md#mechanism) | covered |
| `minRegAutoWS` | P1 | [resources.md](resources.md#pruning-ruleset); [warps_and_registers.md](warps_and_registers.md#pruning-ruleset) | covered |
| `maxRegAutoWS` | P1 | [resources.md](resources.md#pruning-ruleset); [warps_and_registers.md](warps_and_registers.md#pruning-ruleset) | covered |
| `tmem_alloc_algo` | P1 | [resources.md](resources.md#tmem) | covered |
| `loop_unroll_factor`, `disable_licm` | P2 | [tiles_and_stages.md](tiles_and_stages.md#pruning-ruleset) | covered |
| `merge_epilogue_to_computation`, `merge_correction` | P2 | [epilogue.md](epilogue.md#mechanism) | covered |
| `pingpongAutoWS` | P2 | [clc_persistence.md](clc_persistence.md#pruning-ruleset); [pingpong.md](pingpong.md#pruning-ruleset) | covered |
| `multi_cta` | P2 | [split_k.md](split_k.md#pruning-ruleset); [fused_reductions.md](fused_reductions.md#pruning-ruleset) | covered |
| `allowDependentTwoCTA` | P2 | [2cta.md](2cta.md#pruning-ruleset) | covered |
| `clc` | P2 | [clc_persistence.md](clc_persistence.md#pruning-ruleset) | covered |
| `tt.autows` stage/order | P2 | — | todo |
| `auto_tma` / `auto_tma_device` | P2 | [tma_loads.md](tma_loads.md#pruning-ruleset) | covered |
| `TRITON_DISABLE_2CTA_COOPERATIVE_LOADS` | P2 | [2cta.md](2cta.md#mechanism) | covered |
| `maxnreg` | P2 | [resources.md](resources.md#registers-and-warps); [warps_and_registers.md](warps_and_registers.md#pruning-ruleset) | covered |
| `sched4reg`, `enable_tree_reduction` | P2 | — | todo |
| `smem_alloc_algo` | P2 | — | todo |
| `smem_budget` | P2 | [resources.md](resources.md#smem) | covered |
| `smem_circular_reuse` | P2 | — | todo |
| `tt.autows` channels | P2 | — | todo |
| `mem_plan_pick` + `TRITON_WS_MEM_PLAN_TOPK` / `_PICK` | P2 | — | todo |
| `TRITON_WS_SMEM_PLAN_SEARCH` | P2 | [resources.md](resources.md#pruning-ruleset); [tiles_and_stages.md](tiles_and_stages.md#pruning-ruleset); [epilogue.md](epilogue.md#mechanism) | covered |
| `TRITON_ENABLE_INTERLEAVE_TMEM`‡, `TRITON_ENABLE_UNIFY_WS_BARRIER_LOCATIONS`‡, `TRITON_DISABLE_WSBARRIER_REORDER`, `TRITON_PROMOTE_MBAR_TO_NAMED_BARRIER` | P2 | — | mentioned |
| `TRITON_WS_TILE_PREFETCH_DEPTH`‡ | P2 | — | mentioned |
| Alternative schedulers: `TRITON_USE_MODULO_SCHEDULE` (+ `TRITON_MODULO_*`), `TRITON_USE_JOINT_SCHEDULE`‡, `TRITON_USE_LLM_SCHEDULE`‡, `TRITON_USE_LIST_SCHEDULE` | P2 | — | todo |
| `list_schedule_pick` | P2 | — | todo |

## Other

| Item | State | Note |
|------|-------|------|
| Inductor reference definition | todo | Definition pending. Until then `knobs.md` has no column for what Inductor sets. |
| Leaves outside AutoWS | todo | cuBLAS or non-WS leaves; later. |
| Multicast once lowering lands | todo | Multicast is inert in this tree; revisit `multicast.md` when the lowering is implemented. |

## When-to-use guidance

Deferred; not in this version.
