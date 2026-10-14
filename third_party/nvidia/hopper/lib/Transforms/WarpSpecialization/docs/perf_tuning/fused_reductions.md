# Fused epilogue reductions

Scope: GEMMs under Meta WS (`TRITON_USE_META_WS=1`) on Blackwell whose epilogue also reduces the accumulator. Covered here:
- row reductions (over N, e.g. RMSNorm statistics);
- column reductions (over M);
- cross-tile reductions that combine partial results from many output tiles.

Epilogue side inputs are covered in [epilogue_prologue_tma_loads.md](epilogue_prologue_tma_loads.md). Subtile choice is covered in [epilogue.md](epilogue.md).

## Knobs

| Knob | How set | One line | Existing doc |
|---|---|---|---|
| in-tile reduction | `tl.sum` / `tl.max` / `tl.reduce` on the accumulator, or on a subtile, in the epilogue | Reduces within the tile in registers, after the TMEM load. | none |
| TMA reduce | `desc.atomic_add(...)` or `desc.store(..., store_reduce="add")` | Accumulates a tile into global memory with a TMA reduction. It is scheduled into the reduction partition. | [TMAStoreWaitPipeline.md](../TMAStoreWaitPipeline.md), section "Memory Planner: `buffer.tmaStaging` Handling" |
| `TRITON_WS_TMA_REDUCE_STAGING_COPIES` | env var | Copy target for the TMA-reduce staging ring, independent of `num_stages`. | [TMAStoreWaitPipeline.md](../TMAStoreWaitPipeline.md), section "Phase 3.7: Epilogue Group Copy Increase" |
| pointer atomics | `tl.atomic_add(ptr, ...)` | See the hard prunes below; do not use inside the WS tile loop. | [CrossPartitionAtomicSupport.md](../CrossPartitionAtomicSupport.md), section "Pass: `doDynamicTileBroadcast`" |
| `multi_cta` | `tl.range(..., multi_cta=True)` plus `ctas_per_cga` | Splits a purely additive loop across cluster CTAs and combines downstream `tl.sum`/`tl.reduce` results through distributed shared memory. | [large_k.md](large_k.md#knobs) |
| `EPILOGUE_SUBTILE` | kernel constexpr | Reducing per subtile means the same accumulator subtile is read once for the store and once for the reduction. | [epilogue.md](epilogue.md) |

## Hard prunes

| Condition | Status | Why | Code ref |
|---|---|---|---|
| `tl.atomic_*` on a tensor, or a scalar atomic that is not a loop-carried tile counter, replicated to more than one partition inside the WS loop | silently ignored | Warp specialization is stripped from the whole kernel. The atomic is classified as unsupported, so `doDynamicTileBroadcast` fails and the WS metadata is removed. The kernel compiles without warp specialization. | `WSAtomicBroadcast.cpp`, `classifyAtomic` and `doDynamicTileBroadcast`; `WarpSpecialization.cpp`, `runOnFuncOp` (`bailOut`) |
| `tl.atomic_*` inside the WS loop that no partition reaches (e.g. its result is unused) | wrong answer | The atomic is dropped. See [hazards.md](hazards.md#atomic-inside-the-warp-specialized-tile-loop-with-no-partition). | `PartitionSchedulingMeta.cpp`, `runOnOperation`; `WSSpecialize.cpp`, `SpecializeOp` |
| `multi_cta` on a loop whose downstream `tl.reduce` has a defined reduction ordering, or whose combiner is not an add | compile error | Partial results are combined across CTAs by addition, which is only valid for an unordered add. | `MultiCTAReduction.cpp`, `transformMultiCTALoop` and `verifyReduceCombinerIsAdd` |

## Mechanism

**Cross-lane reductions: fp32 max/min are one instruction, fp32 sum is a shuffle tree.**
- **What changes:** on sm_100-family targets, a reduction across all 32 lanes of a warp with an fp32 max or min combiner lowers to a single `redux.sync`. fp32 add has no `redux` form (`redux` add is integer-only, up to 32 bits), so it lowers to a log2(32) = 5-step shuffle tree. Partial-lane reductions always use shuffles.
- **What it costs:** an fp32 sum across lanes costs about 5x the cross-lane instructions of an fp32 max.
- Accumulator rows map to threads in the TMEM register layout. So a reduction over N (a row statistic) is mostly within a thread, while a reduction over M (a column statistic) crosses lanes and warps.
- Code: `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.cpp`, `matchReduxKind` and `TargetInfo::warpReduce`.

**Cross-tile reductions must use an op the partition scheduler places.**
- **What changes:**
  - A TMA reduce (`DescriptorReduceOp` / `AsyncTMAReduceOp`) is categorized as `TMAReduction`. It is scheduled into a dedicated `reduction` partition, and its completion wait stays in the same partition as the reduce.
  - A pointer `tl.atomic_*` gets no such category. Either it is replicated, and WS is stripped, or it is unreached, and dropped.
- **What it relieves:** the cross-tile accumulation is issued asynchronously from the reduction partition.
- **What it costs:**
  - one more partition;
  - a TMA-reduce staging ring, sized by `TRITON_WS_TMA_REDUCE_STAGING_COPIES` (default: the loop depth).
- Code: `PartitionSchedulingMeta.cpp`, `categorizeTMAReductions`, `isReduceTokenWait` and `createPartitionLayout`; `WSMemoryPlanner.cpp`, `getTmaReduceStagingCopies`.

**Per-subtile reductions read each subtile twice.**
- **What changes:** with `EPILOGUE_SUBTILE >= 2`, a reduction placed next to the stored output reads each accumulator subtile from TMEM once for the store and once for the reduction. `InterleaveTMem` sinks each read to its first use, and keeps the accumulator wait and arrive around both reads by bounding barrier moves with the pre-reorder block order (`buildOriginalBlockOrder`).
- **What it costs:** two TMEM reads per subtile.
- Code: `InterleaveTMem.cpp`, `processBlock`, `buildOriginalBlockOrder`; [TMEMInterleave.md](../TMEMInterleave.md), section "Load Sinking"; [BarrierConstraints.md](../BarrierConstraints.md), section "Barrier Reordering".

## Pruning rules

- IF the fusion needs a sum across output tiles, THEN only try a TMA reduce (`desc.atomic_add` / `store_reduce="add"`), or a per-CTA partial stored inside the loop plus a finishing kernel, BECAUSE a pointer atomic in the WS loop either strips warp specialization or is dropped.
- IF a fused reduction is over M (column), THEN prefer max/min-type statistics where the math allows, and expect sum over M to cost more cross-lane work, BECAUSE only fp32 max/min have a single-instruction warp reduction.
- IF a reduced statistic is needed after the K loop over the full K sum, e.g. a row norm of the GEMM output, THEN skip split-K for that kernel ([large_k.md](large_k.md#pruning-rules)), BECAUSE the statistic is nonlinear in the partials.

## Interactions

See [interactions.md](interactions.md). Pairs this file expects to hold:
- **Fused reduction x epilogue subtiling:** duplicate subtile reads, both kept inside the accumulator wait/arrive.
- **Fused reduction x TMA store:** TMA-reduce waits stay with the reduce, and store waits stay in the epilogue-store partition.
- **Fused reduction x data partitioning:** each data partition reduces its own accumulator half.
- **Cross-tile reduction x CLC / dynamic persistence:** a tile-claim atomic is the one supported replicated atomic. Any other atomic strips WS.

## Hazards

- [Atomic inside the warp-specialized tile loop with no partition](hazards.md#atomic-inside-the-warp-specialized-tile-loop-with-no-partition)

Reported but not yet located in code (rows in [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause)): store of a loop-carried value after the WS loop; two K steps per loop body over a shared operand.

## Not yet established

Tracked in [TRACKER.md](TRACKER.md).


- Carrying a `(BLOCK_M, 8)` partial across the K loop with one cross-lane reduce per tile, and pairwise reduction trees: the gains and their numerics.
- How much of a fused kernel's time the reduction itself takes (ablation bound), compared with the mainloop gap to cuBLAS.
- The number of extra epilogue loads beyond which epilogue fusion stops paying.
- Choosing among near-tied unfused configs by fused time, since the unfused winner is often not the best fused config.
- The cost of fp32 intermediates through the epilogue, compared with spilling.
- The conditions under which an rrms-to-epilogue rewrite (`rrms * (x @ gamma*B)`) pays.
