# Shape regimes: where each derived quantity enters

This file adds no rules of its own. Each section names one quantity derived from the problem shape (defined in [README.md](README.md#decision-tree-skeleton), step 2) and links to the constraints and mechanisms it enters. The fact and its code reference live at the link.

## Output tile count against SM count (`T` vs `P`)

- [clc_persistence.md, Mechanism](clc_persistence.md#mechanism): static persistence fixes each program's tiles, so its tail cost is set by `ceil(T/P)`; dynamic and CLC schedules rebalance at a per-tile claim cost.
- [split_k.md, Mechanism](split_k.md#mechanism): autoWS reduces K serially per tile, so with `T < P` the idle SMs stay idle unless K is split in source.
- [2cta.md, Mechanism](2cta.md#mechanism): one 2-CTA MMA covers a `2*BLOCK_M x BLOCK_N` tile per CTA pair; `T` counts each CTA's `BLOCK_M x BLOCK_N` tile.
- [2cta.md, Never try (any problem)](2cta.md#never-try-any-problem): the pair-aligned schedule, including padding an odd M-tile count.
- Tiny, launch-bound problems (a handful of tiles and K steps) route to the "Leaves outside AutoWS" row in [TRACKER.md](TRACKER.md#other).

## K iterations against `num_stages` (`k_iters`)

- [tiles_and_stages.md, Mechanism](tiles_and_stages.md#mechanism): `num_stages` sets the operand ring depth; the producer can run ahead at most `k_iters` steps within a tile, and across tiles under persistence.
- [split_k.md, Mechanism](split_k.md#mechanism): no K loop (K a compile-time constant <= BLOCK_K) takes a separate compile path, so correctness sweeps need `k_tiles = 1`; operand SMEM is sized by `BLOCK_K`, not by K.
- [epilogue_prologue_tma_loads.md, Never try (any problem)](epilogue_prologue_tma_loads.md#never-try-any-problem): no K loop with a TMA-loaded addend gives a wrong answer (reproduced).
- [hazards.md](hazards.md#single-k-tile-with-an-additive-epilogue-loaded-by-tma): the single-K-tile hazard entry.
- [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause): the few-K-tiles failure of the single-copy register copy of a multi-consumer operand (reported, not located in code).

## Arithmetic intensity (`AI_min` vs `ridge`)

- [tiles_and_stages.md, Mechanism](tiles_and_stages.md#mechanism): operand SMEM fill per FLOP is `1/I_t`, which matters when the GEMM is compute bound.
- [split_k.md, Mechanism](split_k.md#mechanism): a memory-bound GEMM needs enough SMs issuing loads, which ties `AI_min` back to `T`.

## M/N tail waste against tile size (`w`)

- [tma_loads.md, Mechanism](tma_loads.md#mechanism): out-of-bounds TMA reads return zeros, so M/N/K tails need no masks, and the TMA store bounds discard the output tail.
- [epilogue_prologue_tma_loads.md, Mechanism](epilogue_prologue_tma_loads.md#mechanism): a zero pad of M or K folds into the descriptor extent exactly.
- [tiles_and_stages.md, Never try (any problem)](tiles_and_stages.md#never-try-any-problem): `BLOCK_M` below 64 gets no tcgen05 MMA, which sets the smallest M tile.
- [clc_persistence.md, Never try (any problem)](clc_persistence.md#never-try-any-problem): clustered CLC on a padded tile grid must let padded tiles run.

## Batch count (bmm)

- Batch enters only through the output tile count above.
