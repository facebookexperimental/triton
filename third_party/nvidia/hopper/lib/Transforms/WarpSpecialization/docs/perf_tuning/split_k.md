# Split-K, decomposeK and stream-K

Scope: techniques that split one output tile's K reduction across more than one CTA or kernel, applied to Meta WS (`TRITON_USE_META_WS=1`) GEMMs on Blackwell. Covered here:
- what the compiler does and does not do to parallelize K;
- what source-level split-K looks like under warp specialization;
- the K-dimension facts that decide whether splitting is legal or useful, including the separate compile path taken when K fits in one tile.

The regime that routes here is long K with few output tiles: many `K / BLOCK_K` iterations and fewer output tiles than SMs. See `shape_regimes.md`. Inductor's decompose-K (a cuBLAS bmm plus a reduction, chosen above a K threshold) lives outside this directory and is not described here. Stream-K has no compiler support (see Mechanism).

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| Source-level split-K | kernel pattern (tile id over `SPLIT_K x tiles`, partial to a workspace, reduce kernel) | SPLIT_K >= 1 | 1 | P1 | Kernels whose epilogue is linear in the accumulator. | — |
| `multi_cta` | `tl.range` kwarg | bool | False | P2 | Purely additive loops feeding `tl.sum` / `tl.reduce`. On a GEMM K loop with a cluster it is a compile error; without a cluster it is silently ignored. | `ctas_per_cga` > 1. |

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| `multi_cta=True` on a GEMM K loop (accumulator carried by `tl.dot`) with a cluster of more than 1 CTA | compile error | The loop must yield `arith.addf`/`arith.addi` of its own iter_arg; a dot-carried accumulator is rejected ("only additive accumulation ... is supported"). |
| `multi_cta=True` without a cluster (`ctas_per_cga` unset or 1) | silently ignored | The attribute is removed and the loop runs on one CTA. |

## Mechanism

- **Within autoWS, K is reduced serially per output tile:** the warp-specialization passes have no split-K or stream-K; one CTA accumulates each output tile's whole K range, one MMA per K step, into one TMEM accumulator (absence in `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` and `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/`) -> costs latency per tile that grows linearly with `cdiv(K, BLOCK_K)`, idle extra SMs when tiles are fewer than SMs, and accumulation error that grows with the number of serial K steps.
- **Source-level split-K under warp specialization:** changes the persistent tile loop to run over `SPLIT_K x (M/BLOCK_M) x (N/BLOCK_N)` tiles, each accumulating a K slice and TMA-storing its partial into a `(SPLIT_K * M, N)` workspace, with a second, non-WS kernel summing the slabs; legal only when the epilogue is linear in the accumulator (a per-row normalization or nonlinearity over the full K sum cannot be split) -> relieves K-bound skinny problems with SPLIT_K times more tiles to fill the SMs and a shorter serial K chain per tile -> costs a workspace of `SPLIT_K x M x N` elements, a second kernel launch and its memory traffic, and the full epilogue in every split.
- **Operand SMEM is sized by BLOCK_K, not by K:** each A/B ring stage is a descriptor-block-sized buffer (BLOCK_M x BLOCK_K, BLOCK_N x BLOCK_K) (`WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS`, buffer type from the descriptor block); when K < BLOCK_K, TMA zero-fills the extra K columns (`python/triton/tools/tensor_descriptor.py`, `TensorDescriptor`, `padding="zero"`) -> the MMA is correct -> costs the SMEM, TMA bytes and MMA work still paid at every stage.
- **No K loop takes a separate compile path:** needs a compile-time K (`tl.constexpr`) <= BLOCK_K so the IR has no K loop (a runtime K launched with K <= BLOCK_K keeps its loop); changes the IR so `CombineDotAddFPattern` folds an additive epilogue (e.g. a bias) into the dot's accumulator initialization, and the per-tile TMEM accumulator, written by a `tmem_store` at the top of every tile iteration, is multi-buffered across tiles by the memory planner (`Combine.cpp`, `CombineDotAddFPattern`; `WSMemoryPlanner.cpp`, TMEM copy assignment for accumulators whose first access in the body is a `tmem_store`) -> costs its own hazards, listed below; correctness sweeps must include `k_tiles = 1` as well as `k_tiles > 1`.

## Interactions

See [interactions.md](interactions.md). Pairs this file expects to hold:
- **Split-K x TMA store:** the partial tile goes through the normal TMA store epilogue into the workspace descriptor.
- **Single K tile x TMA epilogue load:** see the hazard below.
- **Single K tile x multi-consumer operand:** see the hazard below.
- **Large K x 2-CTA:** 2-CTA halves B SMEM per CTA but does not shorten the serial K chain.

## Hazards

- [Single K tile with an additive epilogue loaded by TMA](hazards.md#single-k-tile-with-an-additive-epilogue-loaded-by-tma)

Reported but not yet located in code (row in [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause)): single-copy register copy of a multi-consumer operand, with few K tiles.

## Not yet established

Tracked in [TRACKER.md](TRACKER.md).

- How the autoWS gap to the best non-WS kernel changes with K for fused epilogues. The observation so far is that per-output-tile epilogue wins shrink as K grows.
- Whether source-level split-K with a WS mainloop beats Inductor's decompose-K for skinny-N, huge-K shapes.
- The behavior of the WS template at K in the millions: indexing width, and accumulation error against a bf16 tolerance.
- Whether smaller BLOCK_M or fewer stages beats a smaller BLOCK_K when BLOCK_K exceeds K.
