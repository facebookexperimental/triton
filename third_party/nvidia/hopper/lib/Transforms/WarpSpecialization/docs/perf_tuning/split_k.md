# Split-K, decomposeK and stream-K

Scope: techniques that split one output tile's K reduction across more than one CTA or kernel, applied to Meta WS (`TRITON_USE_META_WS=1`) GEMMs on Blackwell. Covered here:
- what the compiler does and does not do to parallelize K;
- what source-level split-K looks like under warp specialization;
- the K-dimension facts that decide whether splitting is legal or useful, including the separate compile path taken when K fits in one tile.

The regime that routes here is long K with few output tiles: many `K / BLOCK_K` iterations and fewer output tiles than SMs. See [shape_regimes.md](shape_regimes.md). Inductor's decompose-K (a cuBLAS bmm plus a reduction, chosen above a K threshold) lives outside this directory and is not described here. Stream-K has no compiler support (see Mechanism).

## Knobs

| Knob | How set | One line | Existing doc |
|---|---|---|---|
| `BLOCK_K` | kernel constexpr; descriptor block shape | Sets operand SMEM per stage and the number of K iterations, `cdiv(K, BLOCK_K)`. | none |
| `num_stages` | `triton.Config` / launch option | Requested operand ring depth. The planner caps it at what fits. | [tiles_and_stages.md](tiles_and_stages.md) |
| source-level split-K | kernel structure: tile id over `SPLIT_K x tiles`, partial tile to a workspace, separate reduce kernel | The only way to spread one output tile's K range over several CTAs under warp specialization. | none (example: `python/test/unit/language/test_tutorial09_warp_specialization.py`, `matmul_kernel_tma_persistent_ws_splitk`) |
| `multi_cta` | `tl.range(..., multi_cta=True)` plus `ctas_per_cga` at launch | Partitions the iterations of a purely additive loop across the CTAs of a cluster, and combines downstream `tl.sum`/`tl.reduce` results through distributed shared memory. | `docs/design/multi_cta_grid_requirements.md`; `python/triton/language/core.py`, `range` docstring |

## Never try (any problem)

| Condition | Status | Why | Code ref |
|---|---|---|---|
| `multi_cta=True` on a GEMM K loop (accumulator carried by `tl.dot`) with a cluster of more than 1 CTA | compile error | The loop must yield `arith.addf`/`arith.addi` of its own iter_arg; a dot-carried accumulator is rejected ("only additive accumulation ... is supported"). | `MultiCTAReduction.cpp`, `verifyAdditiveAccumulation` |
| `multi_cta=True` without a cluster (`ctas_per_cga` unset or 1) | silently ignored | The attribute is removed and the loop runs on one CTA. | `MultiCTAReduction.cpp`, `transformMultiCTALoop` |

## Mechanism

**Within autoWS, K is reduced serially per output tile (no compiler split-K or stream-K).**
- **What changes:** nothing in the WS pipeline splits K. No split-K or stream-K exists in the warp-specialization passes. Each output tile's whole K range is accumulated by one CTA, one MMA per K step, into one TMEM accumulator.
- **What it costs:**
  - latency per tile grows linearly with `cdiv(K, BLOCK_K)`;
  - when tiles are fewer than SMs, the extra SMs stay idle;
  - accumulation error grows with the number of serial K steps.
- Code: absence in `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` and `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/`.

**Source-level split-K under warp specialization.**
- **What changes:** the persistent tile loop runs over `SPLIT_K x (M/BLOCK_M) x (N/BLOCK_N)` tiles. Each tile accumulates a K slice and TMA-stores its partial into a `(SPLIT_K * M, N)` workspace. A second, non-WS kernel sums the slabs.
- **What it relieves:** K-bound skinny problems get SPLIT_K times more tiles to fill the SMs, and each tile's serial K chain is shorter.
- **What it costs:**
  - a workspace of `SPLIT_K x M x N` elements;
  - a second kernel launch and its memory traffic;
  - every split pays the full epilogue.
- It is legal only when the epilogue is linear in the accumulator. A per-row normalization or nonlinearity applied over the full K sum cannot be split.

**Operand SMEM is sized by BLOCK_K, not by K.**
- **What changes:** each stage of the A/B ring is a descriptor-block-sized buffer (BLOCK_M x BLOCK_K and BLOCK_N x BLOCK_K). When K < BLOCK_K, TMA zero-fills the extra K columns. The MMA over the padded K is correct, but the SMEM, the TMA bytes and the MMA work are still paid at every stage.
- Code: `WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS` (buffer type from the descriptor block); `python/triton/tools/tensor_descriptor.py`, `TensorDescriptor` (`padding="zero"`).

**No K loop takes a separate compile path.** This needs K to be a compile-time constant (`tl.constexpr`) <= BLOCK_K, so that the IR has no K loop; a runtime K launched with K <= BLOCK_K keeps its loop.
- **What changes:** there is no K loop, so:
  - `CombineDotAddFPattern` folds an additive epilogue such as a bias into the dot's accumulator initialization;
  - the per-tile TMEM accumulator, written by a `tmem_store` at the top of every tile iteration, is multi-buffered across tiles by the memory planner.
- **What it costs:** this path has its own hazards, listed below. Correctness sweeps must include `k_tiles = 1` in addition to `k_tiles > 1`.
- Code: `Combine.cpp`, `CombineDotAddFPattern`; `WSMemoryPlanner.cpp`, TMEM copy assignment for accumulators whose first access in the body is a `tmem_store`.

## When to try it (depends on the problem)

- IF K < BLOCK_K, THEN skip every BLOCK_K above the smallest legal BLOCK_K that covers K, BECAUSE a larger BLOCK_K pays SMEM and MMA work for zero-filled columns at every stage.
- IF tuning a GEMM K loop, THEN skip `multi_cta`, BECAUSE it rejects dot-carried accumulators (with a cluster) or is removed (without one).
- IF the epilogue applies a per-row or nonlinear function of the full K sum, THEN skip split-K for that kernel, BECAUSE partial sums cannot be combined after the epilogue.
- IF the problem has fewer output tiles than SMs and many K steps, THEN include a split-K formulation in the search, not only deeper stages, BECAUSE autoWS reduces K serially per tile and deeper stages do not add parallelism.

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
