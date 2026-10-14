# 2-CTA (`two_ctas`, autoWS GEMM, sm100)

Scope: when to compile a GEMM as a 2-CTA cluster (`tl.dot(..., two_ctas=True)`
with `ctas_per_cga=(2,1,1)`) instead of 1-CTA. This covers which 2-CTA configs
give a compile error, silently run as 1-CTA, hang or give a wrong answer, and what 2-CTA trades. How 2-CTA is
lowered is in
[2cta-autoWS-sync.md](../../../../../../../../docs/design/2cta-autoWS-sync.md);
budget numbers are in [resources.md](resources.md).

Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/` (`hopper/`)
or to the repository root.

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| `two_ctas` | `tl.dot` / `tl.dot_scaled` arg | bool | False | P0 | tcgen05 GEMMs with `BLOCK_M` >= 64 and `ttg.cluster-dim-x` >= 2; otherwise silently ignored. Mixed settings across dots are a compile error. | `ctas_per_cga=(2,1,1)`; B a direct descriptor load with `BLOCK_N/2` >= 16 (#3428, OPEN, covers only an fp32-to-bf16-truncated B); a TMA output store; a pair-aligned tile schedule; warp specialization applied; not with `DynamicPersistent1DScheduler`. |
| `ctas_per_cga` | compile option | 3-tuple, e.g. (2,1,1) | None | P0 | Required by 2-CTA, multicast and `multi_cta`. With `num_ctas` > 1 it is a compile error. | Grid divisible by the cluster shape. |
| `allowDependentTwoCTA` | compile option | bool | False | P2 | Only kernels where one 2-CTA dot consumes another's result; a single-dot GEMM has none. | `two_ctas`. |
| `TRITON_DISABLE_2CTA_COOPERATIVE_LOADS` | env var | 0 / 1 | 0 | P2 | Only `two_ctas` kernels. | `two_ctas`. |

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| `two_ctas=True` with module attribute `ttg.cluster-dim-x` < 2 (e.g. `ctas_per_cga` unset); check it in the TTGIR module header | silently ignored | Falls back to 1-CTA MMA with only a warning: "two_ctas=True requires ctas_per_cga=(2,1,1) or larger". |
| `two_ctas=True` with `BLOCK_M` < 64 | silently ignored | A power-of-2 `BLOCK_M` below 64 gets no tcgen05 MMA at all, 1-CTA or 2-CTA (see [tiles_and_stages.md](tiles_and_stages.md#pruning-ruleset)). |
| `two_ctas=True` on a `tl.dot_scaled` with only the rhs scaled | silently ignored | Falls back to 1-CTA MMA with only a warning: "two_ctas=True is not supported when only the rhs ..." (the dot is transposed). |
| `two_ctas` differs between dots in one kernel | compile error | "inconsistent two_ctas setting across matmuls"; PTX requires one `cta_group` for every tcgen05 op in a kernel. |
| a 2-CTA dot consumes a value derived from another 2-CTA dot | compile error | "two_ctas=True does not currently support dependent matmul chains", unless the compile option `allowDependentTwoCTA` is set (pass option `allow-dependent-chains`). |
| cluster shape other than `(2,1,1)` | requires | `ctas_per_cga=(2,1,1)`: Y/Z must be 1; larger even X is listed as future work. |
| pointer (non-descriptor) output store | requires | A TMA output store: pointer stores do not create the Meta WS epilogue partition. |
| tile schedule where the two CTAs of a cluster (`program_id(0)` = 2k, 2k+1) get different `pid_n`, or non-adjacent `pid_m` | wrong answer | The pair shares one B tile. |
| `TRITON_WS_STAGING_COPIES` > 1 | silently ignored | TMA output staging stays at one copy: "Ordinary 2CTA TMA stores have one completion phase per paired-CTA tile". |
| the output staging correctness floor needs more than one copy | compile error | "2CTA ordinary TMA output staging requires one copy". |
| B not split: `BLOCK_N/2` < 16 (e.g. `BLOCK_N=16`), or B is not a direct descriptor load feeding the MMA (e.g. TMA-loaded, then scaled or cast in registers: a B-side prologue) | wrong answer | The B split is skipped with only a debug log, but the MMA still reads a half-width B per CTA; no error or warning. |
| pointer-loaded (non-descriptor) B under Meta WS | compile error | "pipeliner doesn't know how to predicate this op". |
| Meta WS off (`TRITON_USE_META_WS` unset), persistent 2-CTA GEMM | wrong answer | Without warp specialization, the `Insert2CTASync` rendezvous and the MMA lowering's own cluster arrive both apply. |
| Meta WS on, loop not warp-specialized (`warp_specialize=False`) | hangs | The first iteration completes, then the kernel hangs. |
| any condition that silently strips warp specialization: `num_warps` < 4, warp estimate > 16, a rejected replicated atomic | hangs | Stripping WS leaves the loop as if `warp_specialize=False`, which hangs: `Insert2CTASync` still runs (it is gated only on Meta WS) and the MMA lowering adds its own cluster arrive. |
| `DynamicPersistent1DScheduler` (atomic dynamic tile scheduler) | wrong answer; hangs | Compiles silently; the atomic scheduler has no cluster-level tile claim, so the two CTAs advance the counter independently. |

## Mechanism

- **B halved per CTA:** changes each CTA's B descriptor load (and the host-side descriptor argument types) to `BLOCK_K x BLOCK_N/2`, with the swizzle re-derived for the half tile (`hopper/Transform2CTALoads.cpp`, `transformBLoad`, `transformSplitBLoad`, `shrinkSwizzleForShape`) -> relieves SMEM per operand stage, `(BLOCK_M + BLOCK_N)·BLOCK_K·elt` -> `(BLOCK_M + BLOCK_N/2)·BLOCK_K·elt`, and halves B global/L2 traffic per CTA, so a wide logical tile (e.g. `BLOCK_N=256`), or a deeper operand ring at a fixed tile, fits where 1-CTA does not -> costs a cluster of 2 that the tile schedule must pair-align.
- **One MMA per pair:** it computes a `2*BLOCK_M x BLOCK_N` tile; the lowering predicates it on the CTA rank so only the even-rank (leader) CTA issues it (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`, `convertDotImpl`).
- **Rendezvous before every MMA:** both CTAs arrive on the leader's barrier and the leader waits; the barrier is multi-buffered to the MMA's operand pipeline depth, so any `num_stages` works (`hopper/Insert2CTASync.cpp`, `getOperandPipelineDepth`, `insertSyncBeforeMMA`; [2cta-autoWS-sync.md, "Insert2CTASync: "Arrive Remote, Wait Local" Pattern"](../../../../../../../../docs/design/2cta-autoWS-sync.md#insert2ctasync-arrive-remote-wait-local-pattern)) -> costs one cross-CTA rendezvous per MMA.
- **Ordinary TMA output staging is one copy** (see "Pruning ruleset") -> costs: the epilogue cannot rotate staging slots across tiles, so store overlap comes only from subtiling inside a tile.
- **Cooperative TMA by default:** rank-2 descriptor loads are marked as cooperative pair transactions unless `TRITON_DISABLE_2CTA_COOPERATIVE_LOADS` is set (`Transform2CTALoads.cpp`, `AttrTwoCTALoadName`).
- **Cluster-keyed partition and schedule differences:** keyed on the cluster (`ctas_per_cga`), not on `two_ctas`, so they also apply to `two_ctas=False` cluster kernels: the rule giving the computation partition 8 warps when a reduction partition exists is skipped when the module carries the 2-CTA attribute, which `CheckMatmulTwoCTAs.cpp` sets whenever a cluster is present (`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp`, `optimizePartitionNumWarps`, `AttrTwoCTAsName`); loop scheduling re-runs after warp specialization when `max(cluster_dims) >= 2` (`third_party/nvidia/backend/compiler.py`, `make_ttgir`).

## Interactions

See [interactions.md](interactions.md). Pairs these rules depend on:
- `two_ctas` × `ctas_per_cga`
- `two_ctas` × `BLOCK_M` (64 uses the RHS-split TMEM mode; below 64 falls back)
- `two_ctas` × `BLOCK_N` (half-B width, swizzle)
- `two_ctas` × TMA output store
- `two_ctas` × `TRITON_WS_STAGING_COPIES`
- `two_ctas` × `EPILOGUE_SUBTILE`
- `two_ctas` × `data_partition_factor`
- `two_ctas` × `num_stages`
- `two_ctas` × `num_warps` with a reduction partition
- `two_ctas` × `tl.dot_scaled`
- `two_ctas` × CLC tile scheduler (see [clc_persistence.md](clc_persistence.md))
- `two_ctas` × atomic dynamic tile scheduler
- `two_ctas` × TMA multicast (see `multicast.md`)

## Hazards

- [2-CTA with an unsplit B](hazards.md#2-cta-with-an-unsplit-b)
- [2-CTA without warp specialization](hazards.md#2-cta-without-warp-specialization)
- [2-CTA with the atomic dynamic tile scheduler](hazards.md#2-cta-with-the-atomic-dynamic-tile-scheduler)

Suspected from code, not reproduced in testing (rows in [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause)): a register-written SMEM operand read by a 2-CTA MMA without a cluster-scope fence.

## Not yet established

See [TRACKER.md](TRACKER.md):
- Value of `BLOCK_M=64` 2-CTA (it compiles as real 2-CTA and is correct).
- Root cause of the hang with warp specialization off.
- When 2-CTA beats the best 1-CTA config. It has been observed to depend on kernel class: plain GEMM vs epilogues that add an SMEM consumer.
- The throughput cost of one output staging copy.
- Which `num_stages` suit 2-CTA at `BLOCK_K=128, BLOCK_N=256`.
- Cooperative vs per-CTA operand loads (`TRITON_DISABLE_2CTA_COOPERATIVE_LOADS`).
