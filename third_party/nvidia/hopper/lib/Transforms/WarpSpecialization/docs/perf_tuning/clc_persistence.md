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

| Knob | How set | What it does | Existing doc |
|---|---|---|---|
| tile scheduler | `tl.constexpr` scheduler class (`NonPersistentScheduler`, `StaticPersistent1DScheduler`, `DynamicPersistent1DScheduler`, `ClcTileScheduler`), or a hand-written `tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True)` loop | Chooses how a program gets its next tile | [AutoWSAnnotations.md, "Status on `scf.while` (current gaps)"](../AutoWSAnnotations.md#status-on-scfwhile-current-gaps) |
| `tl.clc_tile_scheduler()` | kernel code | Factory for `ClcTileScheduler` | [triton-clc-tile-scheduler.md, "API"](../../../../../../../../docs/design/triton-clc-tile-scheduler.md#api) |
| `TRITON_WS_TILE_PREFETCH_DEPTH` | env var (default 1) | How many tile claims the owner partition runs ahead of the others | [DynamicPersistentAutoWSGaps.md, "Atomic Broadcast Restrictions"](../DynamicPersistentAutoWSGaps.md#atomic-broadcast-restrictions) |
| `clc` | compile option (`triton.Config(clc=True)`) | Upstream single-CTA CLC path; not the autoWS path | — |
| tile order (`GROUP_M`, group-by-N) | kernel code mapping the linear tile id to `(pid_m, pid_n)` | Which tiles are co-resident across programs | — |

## Never try (any problem)

| Condition | Status | Why | Code |
|---|---|---|---|
| `clc=True` with `TRITON_USE_META_WS=1` | compile error | "clc=True does not support Meta AutoWS; use tl.clc_tile_scheduler()" | `third_party/nvidia/backend/compiler.py`, `CUDABackend.parse_options`, `CUDABackend.make_ttgir` |
| `clc=True` with a physical cluster (`ctas_per_cga`, `cluster_dims`) | compile error | "clc=True requires num_ctas=1 and no physical cluster" | `third_party/nvidia/backend/compiler.py`, `CUDAOptions.__post_init__` |
| `ClcTileScheduler` with `ctas_per_cga` where a grid dimension is not divisible by the cluster dimension, or a cluster member skips `advance()` (e.g. an early-exit bounds guard on a padded tile) | requires | A divisible grid and uniform `advance()`: the CLC reservation is collective for the physical cluster; "Inactive padded CTAs are unsupported" | `python/triton/language/schedule.py`, `ClcTileScheduler`; [triton-clc-tile-scheduler.md, "PTX cluster grouping contract"](../../../../../../../../docs/design/triton-clc-tile-scheduler.md#ptx-cluster-grouping-contract) |
| `DynamicPersistent1DScheduler` with `two_ctas` | wrong answer (reproduced); hangs (reproduced) | Compiles silently, then gives a wrong answer or hangs: the atomic scheduler has no cluster-level tile claim, so both CTAs advance the counter independently. Use `ClcTileScheduler` for a dynamic 2-CTA schedule. See [hazards.md](hazards.md#2-cta-with-the-atomic-dynamic-tile-scheduler) | [DynamicPersistentAutoWSGaps.md, "2-CTA Status"](../DynamicPersistentAutoWSGaps.md#2-cta-status) |
| loop `num_stages` on a dynamic or CLC outer `while` | silently ignored | Software pipelining is `scf.for`-only; the stages belong on the K loop | [AutoWSAnnotations.md, "`num_stages` / pipelining on a `while`: not applicable"](../AutoWSAnnotations.md#num_stages--pipelining-on-a-while-not-applicable) |
| a replicated atomic in the WS loop that is not a scalar, while-carried value mapped to every partition | silently ignored | Per the code, warp specialization is dropped for the function (not reproduced in testing). Only that shape can be owned by one partition and broadcast. Under `two_ctas`, stripping WS hangs (see [2cta.md](2cta.md#never-try-any-problem)) | `WS/WSAtomicBroadcast.cpp` (`AtomicWSCase::Reject`); [DynamicPersistentAutoWSGaps.md, "Atomic Broadcast Restrictions"](../DynamicPersistentAutoWSGaps.md#atomic-broadcast-restrictions) |
| `pingpongAutoWS` with a dynamic or CLC outer `while`, for SFU math in the epilogue | silently ignored | Ping-pong regions need an enclosing `scf.for`; epilogue ops directly under the `while` have none (SFU math inside the K loop still can) (see `pingpong.md`) | `WS/PingPong.cpp`, `doPingPongPrep` |

## Mechanism

**Static persistent.**
- Grid = min(programs, tiles); program p walks tiles p, p+P, p+2P, ... (`StaticPersistent1DScheduler`). The outer `while` is uplifted to an `scf.for`, so every loop annotation takes effect ([AutoWSAnnotations.md, "Status on `scf.while` (current gaps)"](../AutoWSAnnotations.md#status-on-scfwhile-current-gaps)).
- **Relieves:** launch cost per tile; lets the accumulator double buffer and the store partition overlap tile i's epilogue with tile i+1's mainloop.
- **Costs:** a fixed tail: the slowest program runs ceil(tiles/P) tiles, so a small tiles mod P leaves SMs idle in the last wave. Uneven per-tile work is not rebalanced.

**Dynamic (atomic) persistent.**
- Each program claims its next tile from a global atomic counter (`DynamicPersistent1DScheduler`). Under autoWS one partition (the one with the TMA load, if any) owns the claim and broadcasts the tile id to the others (`WS/WSAtomicBroadcast.cpp`, `doDynamicTileBroadcast`).
- **Relieves:** imbalance from uneven tiles.
- **Costs:** one atomic per tile; partitions held in lockstep at the claim (broadcast depth 1); an outer `scf.while` that is not software-pipelined; a host-allocated counter.

**CLC.**
- Oversubscribed grid, one CTA per logical tile; a CTA that finishes cancels a pending cluster and takes over its tile (`ClcTileScheduler`; [triton-clc-tile-scheduler.md, "PTX cluster grouping contract"](../../../../../../../../docs/design/triton-clc-tile-scheduler.md#ptx-cluster-grouping-contract)). The cancel is issued from one producer partition, hoisted for overlap, and its result broadcast to the other partitions.
- The outer loop stays an `scf.while`. End-to-end coverage there: `data_partition_factor`, `separate_epilogue_store`, generated subtiled regions, broadcast depth. No contract: the `merge_*` knobs, the SMEM/TMEM allocation options, register budgets, ping-pong ([DynamicPersistentAutoWSGaps.md, "Frontend and Configuration Gaps"](../DynamicPersistentAutoWSGaps.md#frontend-and-configuration-gaps)).
- **Relieves:** imbalance without a software counter; its cluster-level ownership makes it the dynamic schedule that works with `two_ctas`.
- **Costs:** one CLC request per tile; the `while`-loop knob restrictions above; the clustered contract (divisible grid, no inactive CTAs).

**Tile order** does not change the work done; it changes which tiles are co-resident across the P programs, and so the L2 reuse of A and B tiles.

## Interactions

See [interactions.md](interactions.md). Pairs these rules depend on:
- tile scheduler × `two_ctas` (see [2cta.md](2cta.md))
- tile scheduler × `data_partition_factor` (see [data_partitioning.md](data_partitioning.md))
- tile scheduler × `separate_epilogue_store` / `EPILOGUE_SUBTILE`
- tile scheduler × loop `num_stages`
- tile scheduler × `pingpongAutoWS` (see `pingpong.md`)
- tile scheduler × TMA multicast (see `multicast.md`)
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
