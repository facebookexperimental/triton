# Perf Tuning Coverage Tracker

Coverage checklist for the files described in [README.md](README.md).
Reviewers: add rows for missing context. State is `todo`, `in progress` or
`done`.

## perf_tuning files

| File | State | Note |
|------|-------|------|
| `knobs.md` | todo | Index of the GEMM knobs: one line per knob plus a link to the doc that explains it. |
| `resources.md` | todo | SMEM, TMEM and register budgets as rules; links to the existing SMEM/TMEM/register docs for the mechanics. |
| `interactions.md` | todo | Feature-pair table grouped by feature: pair \| status ("works", "compile error", "silently ignored", "duplicate config", "hangs", "wrong answer", "requires") \| mechanism + code location. |
| `hazards.md` | todo | Open wrong-answer and hang conditions, each with its trigger and code location. Entries are deleted when fixed. |
| `shape_regimes.md` | todo | Routing index from derived shape quantities to the rules that key on them; no rules of its own. |

## Feature files

| File | State | Note |
|------|-------|------|
| `tiles_and_stages.md` | todo | `BLOCK_M`/`BLOCK_N`/`BLOCK_K` and `num_stages`. |
| `warps_and_registers.md` | todo | `num_warps` and the AutoWS register budgets. |
| `epilogue.md` | todo | Epilogue subtiling, separate epilogue store, store placement. |
| `2cta.md` | todo | 1-CTA vs 2-CTA (`two_ctas`). |
| `data_partitioning.md` | todo | `data_partition_factor`. |
| `clc_persistence.md` | todo | Static persistent vs dynamic and CLC persistent. |
| `tma_loads.md` | todo | TMA vs pointer operand loads. |
| `tma_stores.md` | todo | TMA vs pointer output stores. |
| `split_k.md` | todo | Splitting one output tile's K reduction (split-K, decomposeK, stream-K) and the single-K-tile path. |
| `pingpong.md` | todo | Ping-pong scheduling. |
| `multicast.md` | todo | TMA multicast. |
| `epilogue_prologue_tma_loads.md` | todo | TMA loads of extra epilogue/prologue inputs. |
| `fused_reductions.md` | todo | Reductions fused into the epilogue. |

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
| Descriptor loads/stores (kernel pattern) | todo | `tl.make_tensor_descriptor` or host `TensorDescriptor` with `desc.load` / `desc.store`. |

## Fixes to existing docs

| Doc | State | Note |
|-----|-------|------|
| [BarrierFusion.md](../BarrierFusion.md) | done | `optimizeTMALoads` mixed-group behavior. |

## Other

| Item | State | Note |
|------|-------|------|
| Inductor reference definition | todo | Definition pending. Until then `knobs.md` has no column for what Inductor sets. |
| Leaves outside AutoWS | todo | cuBLAS or non-WS leaves; later. |
