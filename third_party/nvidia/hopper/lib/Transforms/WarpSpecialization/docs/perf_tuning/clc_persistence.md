# Persistence and CLC (tile schedulers, autoWS GEMM, sm100)

Scope: which tile schedule a GEMM's outer tile loop uses: non-persistent,
static persistent, dynamic (atomic) persistent, or CLC (Cluster Launch Control).
This covers which schedule/knob combinations are rejected, ignored or
unsupported, what each schedule trades, and the pruning that follows. The
scheduler API is in `python/triton/language/schedule.py`. CLC lowering is in
[triton-clc-tile-scheduler.md](../../../../../../../../docs/design/triton-clc-tile-scheduler.md),
and autoWS on a `while` loop is in
[DynamicPersistentAutoWSGaps.md](../DynamicPersistentAutoWSGaps.md).

Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` (`WS/`) unless they start at the repository root.

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| Tile scheduler | kernel pattern (`triton.language.schedule`, `tl.clc_tile_scheduler()`) | non-persistent / static / dynamic / CLC | static persistent | P0 (static persistent); P1 (CLC, dynamic) | Static persistent becomes an `scf.for`, so every knob applies. On a dynamic or CLC `while`, only dp, `separate_epilogue_store`, subtiled regions and prefetch depth are covered end to end. | CLC: grid divisible by the cluster and every member calls `advance()`. Dynamic: not with `two_ctas`. |
| `clc` | compile option | bool | False | P2 | A compile error with Meta WS. | — |
| `TRITON_WS_TILE_PREFETCH_DEPTH`‡ | env var | int >= 1 | 1 | P2 | Only dynamic or CLC persistent loops. | — |

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| `clc=True` with `TRITON_USE_META_WS=1` | compile error | "clc=True does not support Meta AutoWS; use tl.clc_tile_scheduler()". |
| `clc=True` with a physical cluster (`ctas_per_cga`, `cluster_dims`) | compile error | "clc=True requires num_ctas=1 and no physical cluster". |
| `ClcTileScheduler` with `ctas_per_cga` where a grid dimension is not divisible by the cluster dimension, or a cluster member skips `advance()` (e.g. an early-exit bounds guard on a padded tile) | requires | A divisible grid and uniform `advance()`: the CLC reservation is collective for the physical cluster; "Inactive padded CTAs are unsupported". |
| `DynamicPersistent1DScheduler` with `two_ctas` | wrong answer; hangs | Compiles silently, then gives a wrong answer or hangs: the atomic scheduler has no cluster-level tile claim, so both CTAs advance the counter independently. |
| loop `num_stages` on a dynamic or CLC outer `while` | silently ignored | Software pipelining is `scf.for`-only; the stages belong on the K loop. |
| a replicated atomic in the WS loop that is not a scalar, while-carried value mapped to every partition | silently ignored | The atomic broadcast rejects it, so warp specialization is dropped for the function. |
| `pingpongAutoWS` with a dynamic or CLC outer `while`, for SFU math in the epilogue | silently ignored | Ping-pong regions need an enclosing `scf.for`; epilogue ops directly under the `while` have none (SFU math inside the K loop still can) (see [pingpong.md](pingpong.md)). |

## Mechanism

- **Static persistent:** changes the grid to min(programs, tiles), with program p walking tiles p, p+P, p+2P, ... (`StaticPersistent1DScheduler`), and the outer `while` uplifted to an `scf.for` so every loop annotation takes effect ([AutoWSAnnotations.md, "Status on `scf.while` (current gaps)"](../AutoWSAnnotations.md#status-on-scfwhile-current-gaps)) -> relieves launch cost per tile and lets the accumulator double buffer and the store partition overlap tile i's epilogue with tile i+1's mainloop -> costs a fixed tail: the slowest program runs ceil(tiles/P) tiles, so a small tiles mod P leaves SMs idle in the last wave, and uneven per-tile work is not rebalanced.
- **Dynamic (atomic) persistent:** changes tile assignment so each program claims its next tile from a global atomic counter (`DynamicPersistent1DScheduler`); under autoWS one partition (the one with the TMA load, if any) owns the claim and broadcasts the tile id to the others (`WS/WSAtomicBroadcast.cpp`, `doDynamicTileBroadcast`) -> relieves imbalance from uneven tiles -> costs one atomic per tile, partitions held in lockstep at the claim (broadcast depth 1), an outer `scf.while` that is not software-pipelined, and a host-allocated counter.
- **CLC:** changes to an oversubscribed grid, one CTA per logical tile, where a CTA that finishes cancels a pending cluster and takes over its tile (`ClcTileScheduler`; [triton-clc-tile-scheduler.md, "PTX cluster grouping contract"](../../../../../../../../docs/design/triton-clc-tile-scheduler.md#ptx-cluster-grouping-contract)); the cancel is issued from one producer partition, hoisted for overlap, and its result broadcast to the other partitions -> relieves imbalance without a software counter, and its cluster-level ownership makes it the dynamic schedule that works with `two_ctas` -> costs one CLC request per tile, the `while`-loop knob restrictions (next bullet), and the clustered contract (divisible grid, no inactive CTAs).
- **CLC knob coverage:** the outer loop stays an `scf.while`; end-to-end coverage there is `data_partition_factor`, `separate_epilogue_store`, generated subtiled regions and broadcast depth, while the `merge_*` knobs, the SMEM/TMEM allocation options, register budgets and ping-pong have no contract ([DynamicPersistentAutoWSGaps.md, "Frontend and Configuration Gaps"](../DynamicPersistentAutoWSGaps.md#frontend-and-configuration-gaps)).
- **Tile order:** does not change the work done; it changes which tiles are co-resident across the P programs, and so the L2 reuse of A and B tiles.

## Interactions

See [interactions.md](interactions.md). Pairs these rules depend on:
- tile scheduler × `two_ctas` (see [2cta.md](2cta.md))
- tile scheduler × `data_partition_factor` (see [data_partitioning.md](data_partitioning.md))
- tile scheduler × `separate_epilogue_store` / `EPILOGUE_SUBTILE`
- tile scheduler × loop `num_stages`
- tile scheduler × `pingpongAutoWS` (see [pingpong.md](pingpong.md))
- tile scheduler × TMA multicast (see [multicast.md](multicast.md))
- tile scheduler × `TRITON_WS_TILE_PREFETCH_DEPTH`
- tile scheduler × `merge_*` / SMEM-TMEM allocation knobs
- `clc` × `TRITON_USE_META_WS`

## Hazards

- [2-CTA with the atomic dynamic tile scheduler](hazards.md#2-cta-with-the-atomic-dynamic-tile-scheduler)

## Not yet established

See [TRACKER.md](TRACKER.md):
- When CLC beats static persistent on a GEMM. No GEMM measurement exists.
- Whether `GROUP_M` / group-by-N tile order matters for autoWS GEMMs. It has been observed to be flat.
- Runtime correctness and value of `TRITON_WS_TILE_PREFETCH_DEPTH` > 1. The design doc is inconsistent about whether it has runtime coverage.
- The tile-count threshold below which a non-persistent schedule beats a persistent one.
