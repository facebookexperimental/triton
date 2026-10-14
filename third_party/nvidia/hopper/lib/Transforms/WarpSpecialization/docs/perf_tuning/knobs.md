# AutoWS GEMM knobs on B200: index

This page lists every knob that changes generated code for an autoWS GEMM on B200. That covers mm, addmm, bmm and grouped GEMM, compiled with `TRITON_USE_META_WS=1`. For each knob it gives how to set it, its legal values, one line on what it does, and where an existing doc explains it.

**Scope:**
- Knobs that do not change generated code are out of scope. That includes IR/graph dumps, plan-dump paths and verifier printouts.
- Env-var knobs that change generated code but are **not part of the compile cache key** are marked ‡. Changing one of these does not invalidate kernels that are already cached. The key covers only the variables in `CACHE_INVALIDATING_ENV_VARS` (`include/triton/Tools/Sys/GetEnv.h`), via `get_cache_key` (`python/triton/runtime/cache.py`).

**How each knob is set:**
- `tl.range` = keyword argument on `tl.range(...)` / `tl.condition(...)`. Defined in `AutoWSLoopOptions` (`python/triton/language/core.py`).
- `op` = argument on a single op.
- `option` = compile option, set via `triton.Config(...)` or a launch kwarg. Defined in `CUDAOptions` (`third_party/nvidia/backend/compiler.py`).
- `env` = environment variable, or the matching `triton.knobs.nvidia.*` field in `nvidia_knobs` (`python/triton/knobs.py`).
- `pattern` = a way of writing the kernel, not a single setting.

**Doc links:** relative to this folder.

Prerequisite for everything below: `TRITON_USE_META_WS=1` (env; in the cache key). It switches warp specialization from the upstream pass to Meta's partition scheduler plus the WS pipeline.

## Tile and pipeline

| Knob | Set via | Values | What it does | Doc |
|---|---|---|---|---|
| `num_stages` | option | int ≥ 1, default 3 | Depth of the SMEM operand ring for the WS loop. The memory planner may give a buffer fewer copies to fit the SMEM budget. | [AutoWSAnnotations: Annotation → IR attribute → consumer](../AutoWSAnnotations.md#annotation--ir-attribute--consumer) |
| `num_stages` (per loop) | tl.range | int | Overrides the kernel-level `num_stages` for one loop. Has no effect on a dynamic or CLC `while`. | [AutoWSAnnotations: `num_stages` / pipelining on a `while`](../AutoWSAnnotations.md#num_stages--pipelining-on-a-while-not-applicable) |
| `flatten` | tl.range | bool, default False | Fuses the tile loop and the K loop into one loop. The WS mark is carried onto the fused loop (`FuseNestedLoops.cpp`). | [AutoWSAnnotations: Annotation → IR attribute → consumer](../AutoWSAnnotations.md#annotation--ir-attribute--consumer) (table row only) |
| `disallow_acc_multi_buffer` | tl.range | bool, default False | Under Meta WS it does not cap TMEM accumulator copies. It changes the MMA latency the software pipeliner assigns (`AssignLatencies.cpp`), and when dp > 1 it floors every TMA-fed SMEM operand of the MMA at `num_stages` (`getStaticSmemCopySafetyFloor`). | [AutoWSAnnotations: `disallow_acc_multi_buffer`](../AutoWSAnnotations.md#disallow_acc_multi_buffer) |
| `loop_unroll_factor`, `disable_licm` | tl.range | int / bool | Generic loop transforms: IR-level unrolling, and turning off loop-invariant hoisting. Not specific to WS. | [AutoWSAnnotations: `loop_unroll_factor` and `disable_licm`](../AutoWSAnnotations.md#loop_unroll_factor-and-disable_licm) |

## Partitioning and scheduling

| Knob | Set via | Values | What it does | Doc |
|---|---|---|---|---|
| `warp_specialize` | tl.range | bool, default False | Marks the loop for autoWS: partitions for load, MMA (gemm), epilogue and, optionally, epilogue store. Silently dropped if `num_warps` < 4, if no partition can be formed, or if the warp estimate exceeds `kMaxWarps` = 16 (`PartitionSchedulingMeta.cpp`). | [Overview: Pipeline](../Overview.md#pipeline); [AutoWSAnnotations](../AutoWSAnnotations.md#annotation--ir-attribute--consumer) |
| `data_partition_factor` | tl.range | int; values < 2 are a no-op | Splits the accumulator and its consumers into N slices. Split rule: M if `BM/N ≥ 64`, otherwise N if `BN/N ≥ 128`. The split is skipped, with a warning, if it would change the TMEM blockM (`NVGPUWSDataPartitionPass`). | [DataPartition: Step 2](../DataPartition.md#step-2-compute-partition-scheme-computepartitionscheme) |
| `separate_epilogue_store` | tl.range | bool, default False | Moves the TMA store into its own 1-warp partition. | [PartitionSchedulingMeta: Tuning Knobs](../PartitionSchedulingMeta.md#tuning-knobs), [Target Partition Layouts](../PartitionSchedulingMeta.md#target-partition-layouts) |
| `merge_epilogue` | tl.range | bool, default False | Folds epilogue ops into the correction or reduction partition, or into computation if neither exists. | [PartitionSchedulingMeta: Tuning Knobs](../PartitionSchedulingMeta.md#tuning-knobs) |
| `pingpongAutoWS` | option | bool, default False | Makes two consumer partitions take turns at "expensive" ops. On Blackwell those are SFU math ops only, not MMAs. Needs exactly 2 partitions (`doPingPongPrep`). | [PingPongScheduling: Expensive Op Identification](../PingPongScheduling.md#expensive-op-identification) |
| `multi_cta` | tl.range | bool | Splits loop iterations across the CTAs of a cluster and reduces across them through DSMEM. Needs `ctas_per_cga`. | [multi_cta_autows_requirements: Summary](../../../../../../../../docs/design/multi_cta_autows_requirements.md#summary) |
| `two_ctas` | op (`tl.dot`, `tl.dot_scaled`) | bool | Two-CTA collective MMA. The compiler halves the B load per CTA (`Transform2CTALoads`). Falls back to 1-CTA, with only a warning, when there is no cluster or BM < 64 (`AccelerateMatmul.cpp`). All dots in the kernel must use the same setting. | [2cta-autoWS-sync: User API](../../../../../../../../docs/design/2cta-autoWS-sync.md#user-api), [Known Limitations](../../../../../../../../docs/design/2cta-autoWS-sync.md#known-limitations-and-future-work) |
| `ctas_per_cga` | option | 3-tuple, e.g. (2,1,1) | Physical cluster shape. Forces `num_ctas = 1`. Cannot be combined with `num_ctas > 1`. Needed by 2-CTA, multicast and `multi_cta`. | [2cta-autoWS-sync: Two Approaches to 2-CTA](../../../../../../../../docs/design/2cta-autoWS-sync.md#two-approaches-to-2-cta) |
| Tile scheduler | pattern (`triton.language.schedule`, `tl.clc_tile_scheduler()`) | non-persistent / static / dynamic / CLC | Shape of the outer loop. Static persistent becomes an `scf.for`, so every knob applies. Dynamic and CLC stay `scf.while`, and only some knobs apply there. | [AutoWSAnnotations: Status on `scf.while`](../AutoWSAnnotations.md#status-on-scfwhile-current-gaps); [triton-clc-tile-scheduler: API](../../../../../../../../docs/design/triton-clc-tile-scheduler.md#api) |
| `clc` | option | bool, default False | Upstream single-CTA CLC. Rejected when Meta WS is on. | [triton-clc-tile-scheduler](../../../../../../../../docs/design/triton-clc-tile-scheduler.md) |
| `tt.autows` stage/order | op (`tl.dot(attrs=...)`, `desc.load(attrs=...)`) | JSON string fields `stage`, `order` | Pins MMA order and the load prefetch rank inside the loop. | [AutoWSAnnotations: Operation scheduling annotations](../AutoWSAnnotations.md#operation-scheduling-annotations) |

## TMA

| Knob | Set via | Values | What it does | Doc |
|---|---|---|---|---|
| Descriptor loads/stores | pattern (`tl.make_tensor_descriptor` or host `TensorDescriptor`; `desc.load` / `desc.store`) | — | TMA producer and store partitions are the shape autoWS expects. Pointer stores do not form an epilogue-store partition. | [TMADescriptors](../TMADescriptors.md) |
| `multicast` (per load) | op (`desc.load(multicast=)`) | None / True / False | Gives permission for compiler-selected TMA multicast within the cluster. Silently falls back to per-CTA loads. | [triton_tma_multicast: Current implementation](../../../../../../../../docs/design/triton_tma_multicast.md#current-implementation) |
| `multicast` | option | bool, default False | Default multicast policy for descriptor loads in the kernel. | [triton_tma_multicast: Decision summary](../../../../../../../../docs/design/triton_tma_multicast.md#decision-summary) |
| `auto_tma` / `auto_tma_device` | option, or env `TRITON_AUTO_TMA`‡ / `TRITON_AUTO_TMA_DEVICE`‡ (OR'd together) | bool | Rewrites pointer loads as TMA (`add_promote_load_to_tma`, `PromoteLoadToTMA.cpp`), using host or device descriptors. | [AutoTMAPromotion](../AutoTMAPromotion.md) |
| `TRITON_DISABLE_2CTA_COOPERATIVE_LOADS` | env | 0/1 | 2-CTA uses per-CTA TMA loads instead of cooperative pair loads (`Transform2CTALoads.cpp`). | [BarrierFusion: 2-CTA Cooperative Loads](../BarrierFusion.md#2-cta-cooperative-loads-and-triton_disable_2cta_cooperative_loads) |

## Epilogue

| Knob | Set via | Values | What it does | Doc |
|---|---|---|---|---|
| Epilogue subtiling (ES) | pattern (split the accumulator into ES column slices) | 1 / 2 / 4 / 8 | Stores the BM×BN tile as ES pieces of BM×(BN/ES). | [SubtileOperator: Motivation](../SubtileOperator.md#motivation) |
| `generate_subtiled_region` | option, or env `TRITON_GENERATE_SUBTILED_REGION`‡ (OR'd together) | bool, default False | Wraps the ES copies in `ttng.subtiled_region` so barriers and memory are planned per subtile. | [SubtileOperator: Compiler Option](../SubtileOperator.md#compiler-option) |
| `early_tma_store_lowering` | option | None (= True) / False | Lowers TMA stores before WS, so the memory planner can see and size the staging buffers. | [TMAStoreWaitPipeline: Memory Planner `buffer.tmaStaging` Handling](../TMAStoreWaitPipeline.md#memory-planner-buffertmastaging-handling) |
| `tma_store_pipelining` | option | None (= True) / False | Annotates, validates and reorders TMA-store waits. | [TMAStoreWaitPipeline: Wait Annotation and Reordering Pipeline](../TMAStoreWaitPipeline.md#wait-annotation-and-reordering-pipeline) |
| `TRITON_WS_STAGING_COPIES` | env | int K; unset = no cap | Caps the Phase 3.7 copy target of every eligible group: TMA-store staging, other epilogue groups and, unless `TRITON_WS_TMA_REDUCE_STAGING_COPIES` is set, TMA-reduce staging (`getStagingCopiesCap`, `WSMemoryPlanner.cpp`). | [TMAStoreWaitPipeline: `TRITON_WS_STAGING_COPIES`](../TMAStoreWaitPipeline.md#triton_ws_staging_copies-capping-the-phase-37-target) |
| `TRITON_WS_TMA_REDUCE_STAGING_COPIES` | env | int; unset = default | Copy target for TMA-reduce output staging (`getTmaReduceStagingCopies`). | [TMAStoreWaitPipeline: Phase 3.7](../TMAStoreWaitPipeline.md#phase-37-epilogue-group-copy-increase) |

## Registers and warps

| Knob | Set via | Values | What it does | Doc |
|---|---|---|---|---|
| `num_warps` | option | power of 2, default 4; ≥ 4 for WS | Warp count of the default (compute/epilogue) partition only. Producer partitions are added on top and sized by `optimizePartitionNumWarps` (`OptimizePartitionWarps.cpp`). | [default_partition_selection: The general rule](../../../../../../../../docs/design/modulo_schedule/default_partition_selection.md#the-general-rule) (written for the modulo path) |
| `minRegAutoWS` | option | multiple of 8, default 24 | Register budget for partitions that hold no tensors (load, MMA issue, store) (`_check_reg_auto_ws_alignment`, `optimizePartitionNumWarps`). | [RegisterBudgets](../RegisterBudgets.md); [Overview: Register Budgets](../Overview.md#register-budgets) |
| `maxRegAutoWS` | option | multiple of 8, default None | If set, non-default tensor partitions request this many registers and the default region takes the leftover share. If unset, tensor partitions split what is left evenly. | [RegisterBudgets](../RegisterBudgets.md); [Overview: Register Budgets](../Overview.md#register-budgets); [default_partition_selection: Hardware mechanism](../../../../../../../../docs/design/modulo_schedule/default_partition_selection.md#hardware-mechanism-the-default-partition-is-the-register-sink) |
| `maxnreg` | option | int, default None | Register cap for the whole module. It also turns on dynamic register reallocation between partitions (`ConvertWarpSpecializeToLLVM.cpp`, `AllocateWarpGroups.cpp`). | [RegisterBudgets](../RegisterBudgets.md) |

## Memory planner

| Knob | Set via | Values | What it does | Doc |
|---|---|---|---|---|
| `tmem_alloc_algo` | tl.range | 1 = greedy (default); any other value = backtracking | How TMEM buffers are packed into the 512 columns. | [TMEMAllocationHeuristics: Algorithm Selection](../TMEMAllocationHeuristics.md#algorithm-selection) |
| `smem_alloc_algo` | tl.range | 1 = WSBuffer (default); 0 = legacy | Which SMEM planner runs. `tt.autows` channel pins are ignored under 0. | [SmemAllocationDesign](../SmemAllocationDesign.md#goal); [AnnotationBasedBufferPreAssignment: Implementation Status](../AnnotationBasedBufferPreAssignment.md#implementation-status) |
| `smem_budget` | tl.range | bytes; default = the device opt-in maximum (`_max_shared_mem_for_capability`) | SMEM cap for the planner. Auxiliary SMEM is reserved first. | [AutoWSAnnotations: Annotation → IR attribute → consumer](../AutoWSAnnotations.md#annotation--ir-attribute--consumer) (table row only) |
| `smem_circular_reuse` | tl.range | bool / None | Lets two buffers whose lifetimes don't overlap share storage. | [SmemAllocationDesign: Terminology](../SmemAllocationDesign.md#terminology) |
| `tt.autows` channels | op (`tl.dot(attrs={"channels": [...]})`) | `"operand,smem\|tmem,copies,id[,offset]"` | Pins copies, buffer id and TMEM offset for each MMA operand. | [AnnotationBasedBufferPreAssignment: Channel Format](../AnnotationBasedBufferPreAssignment.md#channel-format) |
| `mem_plan_pick` + `TRITON_WS_MEM_PLAN_TOPK` / `_PICK` | tl.range + env | rank int (0 = cost-best); TOPK int, default 1 | Applies the N-th ranked TMEM packing. Has no effect unless TOPK > 1. | [MemoryPlannerSearch: 6. Knobs](../MemoryPlannerSearch.md#6-knobs) |
| `TRITON_WS_SMEM_PLAN_SEARCH` | env | 0/1 | Allocates SMEM (and TMEM) with a beam search over the plan space instead of the greedy planner (`smemPlanSearch`). | [MemoryPlannerSearch: 6. Knobs](../MemoryPlannerSearch.md#6-knobs) |

## Pass toggles and experimental env vars

| Knob | Set via | Values | What it does | Doc |
|---|---|---|---|---|
| `TRITON_ENABLE_INTERLEAVE_TMEM`‡, `TRITON_ENABLE_UNIFY_WS_BARRIER_LOCATIONS`‡, `TRITON_DISABLE_WSBARRIER_REORDER`, `TRITON_PROMOTE_MBAR_TO_NAMED_BARRIER` | env | 0/1 (the first two default to 1) | Switch individual post-WS barrier and TMEM passes on or off. | [TMEMInterleave](../TMEMInterleave.md); [WSBarrierLocationUnification: Pipeline and Controls](../WSBarrierLocationUnification.md#pipeline-and-controls); [named_barrier_api_changes](../../../../../../../../docs/design/named_barrier_api_changes.md) |
| `TRITON_WS_TILE_PREFETCH_DEPTH`‡ | env | int, default 1 | Buffer depth of the tile-id broadcast in dynamic-persistent and CLC loops. | [CrossPartitionAtomicSupport: Tunable: broadcast depth](../CrossPartitionAtomicSupport.md#tunable-broadcast-depth) |
| Alternative schedulers: `TRITON_USE_MODULO_SCHEDULE` (+ `TRITON_MODULO_*`), `TRITON_USE_JOINT_SCHEDULE`‡, `TRITON_USE_LLM_SCHEDULE`‡, `TRITON_USE_LIST_SCHEDULE` | env | unset / 1 / algo name | Replace the default loop schedule. **`TRITON_USE_LIST_SCHEDULE` turns off WS entirely** (`make_ttgir`). | design docs under `docs/design/modulo_schedule/` |

‡ Changes generated code but is not in `CACHE_INVALIDATING_ENV_VARS`, so it is not part of the compile cache key.
