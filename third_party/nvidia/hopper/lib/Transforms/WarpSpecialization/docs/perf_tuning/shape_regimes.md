# Shape regimes: routing index

This file adds no rules of its own. Each section below names one quantity derived from the problem shape and links to the existing rules that key on it, each with a one-line pointer. The rule, its mechanism and its code reference live at the link. The quantities are defined in [README.md](README.md#decision-tree-skeleton), step 2.

## Output tile count against SM count

Tile count is `ceil(M/BLOCK_M) * ceil(N/BLOCK_N) * batch`, compared with 148 SMs on B200.

- [clc_persistence.md, When to try it (depends on the problem)](clc_persistence.md#when-to-try-it-depends-on-the-problem): with fewer tiles than SMs, prefer non-persistent or static persistent over CLC/dynamic.
- [clc_persistence.md, When to try it (depends on the problem)](clc_persistence.md#when-to-try-it-depends-on-the-problem): when tiles mod SMs is small under static persistence, change the tile shape to move the tile count.
- [clc_persistence.md, When to try it (depends on the problem)](clc_persistence.md#when-to-try-it-depends-on-the-problem): uniform tiles prefer static persistence, and uneven (grouped or jagged) tiles try CLC.
- [clc_persistence.md, Mechanism](clc_persistence.md#mechanism): the static persistent tail cost is set by `ceil(tiles/P)`.
- [split_k.md, When to try it (depends on the problem)](split_k.md#when-to-try-it-depends-on-the-problem): with fewer tiles than SMs and many K steps, include split-K in the search.
- [split_k.md, Mechanism](split_k.md#mechanism): autoWS reduces K serially per tile, so idle SMs stay idle.
- [2cta.md, Mechanism](2cta.md#mechanism): one 2-CTA MMA covers a `2*BLOCK_M x BLOCK_N` tile per CTA pair.
- Tiny, launch-bound problems (a handful of tiles and K steps) route to the "Leaves outside AutoWS" row in [TRACKER.md](TRACKER.md#other).

## K iterations against `num_stages`

K iterations are `ceil(K/BLOCK_K)`.

- [split_k.md, When to try it (depends on the problem)](split_k.md#when-to-try-it-depends-on-the-problem): when K < BLOCK_K, skip BLOCK_K values above the smallest one that covers K.
- [split_k.md, Mechanism](split_k.md#mechanism): no K loop (K a compile-time constant <= BLOCK_K) takes a separate compile path, so correctness sweeps need `k_tiles = 1`.
- [epilogue_prologue_tma_loads.md, Never try (any problem)](epilogue_prologue_tma_loads.md#never-try-any-problem): no K loop with a TMA-loaded addend gives a wrong answer (reproduced).
- [epilogue_prologue_tma_loads.md, When to try it (depends on the problem)](epilogue_prologue_tma_loads.md#when-to-try-it-depends-on-the-problem): skip TMA for the addend when K <= BLOCK_K.
- [hazards.md](hazards.md#single-k-tile-with-an-additive-epilogue-loaded-by-tma): the single-K-tile hazard entry.
- [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause): the few-K-tiles failure of the single-copy register copy of a multi-consumer operand (reported, not located in code).
- [tiles_and_stages.md, Not yet established](tiles_and_stages.md#not-yet-established): whether extra stages help at one or two K tiles.

## Arithmetic intensity

Arithmetic intensity is `2*M*N*K / bytes(A, B, C)`.

- No rule in the feature files keys on it yet. The memory-bound end overlaps the tiny-problem leaf above ([TRACKER.md](TRACKER.md#other), "Leaves outside AutoWS").

## M/N tail waste against tile size

- [tma_loads.md, Mechanism](tma_loads.md#mechanism): out-of-bounds TMA reads return zeros, so M/N/K tails need no masks, and the TMA store bounds discard the output tail.
- [epilogue_prologue_tma_loads.md, Mechanism](epilogue_prologue_tma_loads.md#mechanism): a zero pad of M or K folds into the descriptor extent exactly.
- [tiles_and_stages.md, Never try (any problem)](tiles_and_stages.md#never-try-any-problem): `BLOCK_M` below 64 gets no tcgen05 MMA, which sets the smallest M tile.
- [2cta.md, Never try (any problem)](2cta.md#never-try-any-problem): the 2-CTA tile schedule must pair adjacent `pid_m` on the same `pid_n`, and odd `grid_m` padding falls under it.
- [clc_persistence.md, When to try it (depends on the problem)](clc_persistence.md#when-to-try-it-depends-on-the-problem): with clustered CLC on a padded tile grid, let padded tiles run instead of exiting early.

## Batch count (bmm)

- Batch enters only through the output tile count above. No rule in the feature files keys on batch count separately yet.
