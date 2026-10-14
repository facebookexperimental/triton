# Fused epilogue reductions

Scope: GEMMs under Meta WS (`TRITON_USE_META_WS=1`) on Blackwell whose epilogue also reduces the accumulator. Covered here:
- row reductions (over N, e.g. RMSNorm statistics);
- column reductions (over M);
- cross-tile reductions that combine partial results from many output tiles.

Epilogue side inputs are covered in [epilogue_prologue_tma_loads.md](epilogue_prologue_tma_loads.md). Subtile choice is covered in [epilogue.md](epilogue.md).

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| Fused reductions | kernel pattern (row, column or scalar reduction of the accumulator) | — | — | P1 | A GEMM followed by a reduction. An atomic in the WS loop that does not depend on the accumulator is removed. | — |

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| `tl.atomic_*` (`AtomicRMWOp`) on a tensor, or a scalar atomic that is not a loop-carried tile counter, replicated to more than one partition inside the WS loop | silently ignored | Per the code, warp specialization is stripped from the whole kernel: the atomic is classified as unsupported, so `doDynamicTileBroadcast` fails and the WS metadata is removed. |
| `tl.atomic_*` inside the WS loop whose operands do not depend on the accumulator (e.g. a counter `atomic_add(cnt, 1)`, or a constant added to a column) | wrong answer | No partition reaches it, so the atomic is removed from the TTGIR and its result stays 0, while WS is still applied. |
| Cross-tile sum inside the WS loop | requires | A TMA reduce (`desc.atomic_add` / `store_reduce="add"`), or a per-CTA partial stored inside the loop plus a finishing kernel. |
| `multi_cta` on a loop whose downstream `tl.reduce` has a defined reduction ordering, or whose combiner is not an add | compile error | Partial results are combined across CTAs by addition, which is only valid for an unordered add. |

## Mechanism

- **Cross-lane fp32 max/min vs sum:** on sm_100-family targets a full 32-lane warp reduction with an fp32 max or min combiner lowers to one `redux.sync`, while fp32 add has no `redux` form (`redux` add is integer-only, up to 32 bits) and lowers to a log2(32) = 5-step shuffle tree; partial-lane reductions always use shuffles (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.cpp`, `matchReduxKind` and `TargetInfo::warpReduce`) -> an fp32 cross-lane sum costs about 5x the cross-lane instructions of an fp32 max.
- **Reduction axis vs TMEM layout:** in the TMEM register layout accumulator rows map to threads, so a reduction over N (row statistic) is mostly within a thread, while one over M (column statistic) crosses lanes and warps.
- **Cross-tile reduction via TMA reduce:** a TMA reduce (`DescriptorReduceOp` / `AsyncTMAReduceOp`) is categorized as `TMAReduction` and scheduled into a dedicated `reduction` partition with its completion wait in the same partition, while a pointer `tl.atomic_*` gets no category (`PartitionSchedulingMeta.cpp`, `categorizeTMAReductions`, `isReduceTokenWait` and `createPartitionLayout`) -> the cross-tile accumulation is issued asynchronously from the reduction partition -> costs one more partition and a TMA-reduce staging ring sized by `TRITON_WS_TMA_REDUCE_STAGING_COPIES` (default: min(`num_stages`, `TRITON_WS_STAGING_COPIES`); `WSMemoryPlanner.cpp`, `getTmaReduceStagingCopies`).
- **Per-subtile reductions:** with `EPILOGUE_SUBTILE >= 2`, a reduction next to the stored output reads each accumulator subtile from TMEM once for the store and once for the reduction; `InterleaveTMem` sinks each read to its first use and keeps the accumulator wait and arrive around both reads by bounding barrier moves with the pre-reorder block order (`InterleaveTMem.cpp`, `processBlock`, `buildOriginalBlockOrder`; [TMEMInterleave.md](../TMEMInterleave.md), section "Load Sinking"; [BarrierConstraints.md](../BarrierConstraints.md), section "Barrier Reordering") -> costs two TMEM reads per subtile.

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
