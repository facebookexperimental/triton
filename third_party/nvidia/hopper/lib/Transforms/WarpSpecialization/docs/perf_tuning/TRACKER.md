# Perf Tuning Coverage Tracker

Coverage checklist for the files described in [README.md](README.md).
Reviewers: add rows for missing context. State is `todo`, `in progress` or
`done`.

## perf_tuning files

| File | State | Note |
|------|-------|------|
| [`knobs.md`](knobs.md) | done | Index of the GEMM knobs: one line per knob plus a link to the doc that explains it.  |
| [`resources.md`](resources.md) | done | SMEM, TMEM and register budgets as pruning rules; links to the existing SMEM/TMEM/register docs for the mechanics. Open: its "Not yet established" items. |
| [`interactions.md`](interactions.md) | done | Feature-pair table grouped by feature: pair \| status ("works", "compile error", "silently ignored", "duplicate config", "hangs", "wrong answer", "requires") \| mechanism + code location. Open: its "Not yet established" items. |
| [`hazards.md`](hazards.md) | done | Open wrong-answer and hang conditions, each with its trigger and code location. Entries are deleted when fixed.  |

## Feature files

| File | State | Note |
|------|-------|------|
| [`tiles_and_stages.md`](tiles_and_stages.md) | done | `BLOCK_M`/`BLOCK_N`/`BLOCK_K` and `num_stages`. Open: its "Not yet established" items. |
| `warps_and_registers.md` | todo | `num_warps` and the AutoWS register budgets. |
| `epilogue.md` | todo | Epilogue subtiling, separate epilogue store, store placement. |
| `2cta.md` | todo | 1-CTA vs 2-CTA (`two_ctas`). |
| `data_partitioning.md` | todo | `data_partition_factor`. |
| `clc_persistence.md` | todo | Static persistent vs dynamic and CLC persistent. |
| `tma_loads.md` | todo | TMA vs pointer operand loads. |
| `tma_stores.md` | todo | TMA vs pointer output stores. |
| `large_k.md` | todo | K iterations far above `num_stages`, including split-K style K splitting. |
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
| `loop_unroll_factor`, `disable_licm` | todo | `AutoWSLoopOptions` in `python/triton/language/core.py`; generic loop transforms, not WS-specific. |
| Descriptor loads/stores (kernel pattern) | todo | `tl.make_tensor_descriptor` or host `TensorDescriptor` with `desc.load` / `desc.store`: why autoWS expects TMA producer and store partitions. |

## Reported wrong answers / hangs needing repro + code root cause

Reported but not located in code, so not in [hazards.md](hazards.md). Each needs a reproducer and a root cause in code before it moves there.

| Report | State | Note |
|--------|-------|------|
| Pointer-loaded MMA operand | todo | **Trigger:** a pointer `tl.load` result feeds `tl.dot` inside a WS loop. **Symptom:** an assertion in `NVGPUWarpSpecialization` (`Interval` `Start <= End` in `Allocation.h`) on builds with assertions enabled. Release builds have not been characterized, and a hang has also been seen. |
| Single-copy register copy of a multi-consumer operand, with few K tiles | todo | **Trigger:** an operand loaded by TMA is consumed by both the MMA and register code; its register-to-SMEM copy for the MMA gets one copy; `k_tiles` is small relative to `num_stages`: 1 at 3 stages, or up to 2 at 4 stages; there are more tiles than SMs. **Symptom:** NaN or garbage on later persistent tiles. Some larger-K cases at 4 stages hang. |
| Rank-1 TMA bias with subtiled epilogue | todo | **Trigger:** a rank-1 TMA descriptor for a bias, at BLOCK_M=128, BLOCK_N=256, `EPILOGUE_SUBTILE=2`, 2 stages, 8 warps. **Symptom:** subtile 0 gets the wrong bias in a fraction of tiles. **Workaround:** a 2-D `[1, N]` descriptor or a pointer load. |
| Store of a loop-carried value after the WS loop | todo | **Trigger:** a per-CTA partial is carried across the persistent loop and stored after the loop ends. **Symptom:** a GPU hang. **Workaround:** store it inside the loop, predicated on the last tile. |
| Two K steps per loop body over a shared operand | todo | **Trigger:** two TMA loads of the same operand and two MMAs per loop body, with a deferred reduction. **Symptom:** sparse, large corruption at larger shapes. |
| 2-CTA without warp specialization | todo | **Trigger:** `tl.dot(..., two_ctas=True)` launched with WS off. **Symptom:** a GPU hang. |
| Data partitioning with generated subtiled regions and a subtiled scale | todo | **Trigger:** `data_partition_factor=2`, `generate_subtiled_region` on, and a per-subtile scale in the epilogue. **Symptom:** a hang. |
| Warp-specialized inner loop with zero iterations | todo | **Trigger:** a persistent WS kernel whose inner K loop can run zero iterations for some tile, e.g. source-level split-K where a split gets no K tiles. **Symptom:** a hang; producer and consumer partitions wait on barriers that are never armed. Documented as a precondition in `test_tutorial09_warp_specialization.py`, `matmul_kernel_tma_persistent_ws_splitk`. |

## Fixes to existing docs

| Doc | State | Note |
|-----|-------|------|
| [TMEMAllocationHeuristics.md](../TMEMAllocationHeuristics.md) "All TMEM buffers get `buffer.copy = 1`" | todo | Out of date: the heuristic planner's TMEM multi-buffering post-processing (`WSMemoryPlanner.cpp`) raises accumulator copies. |

## Other

| Item | State | Note |
|------|-------|------|
| Inductor reference definition | todo | Definition pending. Until then `knobs.md` has no column for what Inductor sets. |
| Leaves outside AutoWS | todo | cuBLAS or non-WS leaves; later. |
