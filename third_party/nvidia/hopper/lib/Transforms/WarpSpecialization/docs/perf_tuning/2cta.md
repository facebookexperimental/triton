# 2-CTA (`two_ctas`, autoWS GEMM, sm100)

Scope: when to compile a GEMM as a 2-CTA cluster (`tl.dot(..., two_ctas=True)`
with `ctas_per_cga=(2,1,1)`) instead of 1-CTA. This covers which 2-CTA configs
are illegal, silently 1-CTA, or wrong, and what 2-CTA trades. How 2-CTA is
lowered is in
[2cta-autoWS-sync.md](../../../../../../../../docs/design/2cta-autoWS-sync.md);
budget numbers are in [resources.md](resources.md).

Code paths are relative to `third_party/nvidia/hopper/lib/Transforms/` (`hopper/`)
or to the repository root.

## Knobs

| Knob | How set | What it does | Existing doc |
|---|---|---|---|
| `two_ctas` | `tl.dot(..., two_ctas=True)` / `tl.dot_scaled(..., two_ctas=True)` | Requests one `tcgen05.mma.cta_group::2` across a CTA pair | [2cta-autoWS-sync.md, "User API"](../../../../../../../../docs/design/2cta-autoWS-sync.md#user-api) |
| `ctas_per_cga` | compile option (`triton.Config(ctas_per_cga=(2,1,1))`) | Physical cluster shape; without it `two_ctas` is dropped | [2cta-autoWS-sync.md, "Supported Configurations"](../../../../../../../../docs/design/2cta-autoWS-sync.md#supported-configurations) |
| `TRITON_DISABLE_2CTA_COOPERATIVE_LOADS` | env var | Each CTA issues its own TMA instead of one cooperative pair transaction | [2cta-autoWS-sync.md, "Transform2CTALoads Pass"](../../../../../../../../docs/design/2cta-autoWS-sync.md#transform2ctaloads-pass) |

## Hard prunes

| Condition | Status | Why | Code |
|---|---|---|---|
| `two_ctas=True` without a cluster of x-dimension >= 2 (`ctas_per_cga` unset) | silently ignored | Falls back to 1-CTA MMA with only a warning: "two_ctas=True requires ctas_per_cga=(2,1,1) or larger ... Falling back to 1-CTA MMA." | `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`, `BlockedToMMAv5`, `ScaledBlockedToMMAv5` |
| `two_ctas=True` with `BLOCK_M` < 64 | silently ignored | Falls back to 1-CTA MMA with only a warning: "two_ctas=True with BLOCK_M < 64 is not supported" | `AccelerateMatmul.cpp`, `BlockedToMMAv5` |
| `two_ctas=True` on a `tl.dot_scaled` with only the rhs scaled | silently ignored | Falls back to 1-CTA MMA with only a warning: "the dot is transposed here" | `AccelerateMatmul.cpp`, `ScaledBlockedToMMAv5` |
| `two_ctas` differs between dots in one kernel | compile error | "inconsistent two_ctas setting across matmuls"; PTX requires one `cta_group` for every tcgen05 op in a kernel | `lib/Dialect/TritonNvidiaGPU/Transforms/CheckMatmulTwoCTAs.cpp` |
| a 2-CTA dot consumes a value derived from another 2-CTA dot | compile error | "two_ctas=True does not currently support dependent matmul chains", unless the pass option `allow-dependent-chains` is set | `CheckMatmulTwoCTAs.cpp`, `allowDependentChains` |
| cluster shape other than `(2,1,1)` | requires | `ctas_per_cga=(2,1,1)`: Y/Z must be 1; larger even X is listed as future work | [2cta-autoWS-sync.md, "Supported Configurations"](../../../../../../../../docs/design/2cta-autoWS-sync.md#supported-configurations) |
| pointer (non-descriptor) output store | requires | A TMA output store: pointer stores do not create the Meta WS epilogue partition | [2cta-autoWS-sync.md, "Current Limitations"](../../../../../../../../docs/design/2cta-autoWS-sync.md#current-limitations) |
| tile schedule where the two CTAs of a cluster (`program_id(0)` = 2k, 2k+1) get different `pid_n`, or non-adjacent `pid_m` | wrong answer | The pair shares one B tile. The compiler cannot prove a schedule pair-aligned and does not check it | [2cta-autoWS-sync.md, "Current Limitations"](../../../../../../../../docs/design/2cta-autoWS-sync.md#current-limitations) |
| `TRITON_WS_STAGING_COPIES` > 1 | silently ignored | TMA output staging stays at one copy: "Ordinary 2CTA TMA stores have one completion phase per paired-CTA tile" | `hopper/WarpSpecialization/WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |
| the output staging correctness floor needs more than one copy | compile error | "2CTA ordinary TMA output staging requires one copy" | `WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |

## Mechanism

**What it changes.**
- **B is halved per CTA.** Each CTA's B descriptor load is rewritten to half width (`BLOCK_K x BLOCK_N/2`); host-side descriptor argument types are rewritten too. The swizzle is re-derived for the half tile (`hopper/Transform2CTALoads.cpp`, `transformBLoad`, `transformSplitBLoad`, `shrinkSwizzleForShape`).
- **One MMA covers the pair.** It computes a `2*BLOCK_M x BLOCK_N` tile. Only the even-rank (leader) CTA issues it; the lowering predicates the MMA on the CTA rank (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`, `convertDotImpl`).
- **A rendezvous gates every MMA.** Both CTAs arrive on the leader's barrier, and the leader waits before the MMA. The barrier is multi-buffered to the MMA's operand pipeline depth, so any `num_stages` works (`hopper/Insert2CTASync.cpp`, `getOperandPipelineDepth`, `insertSyncBeforeMMA`; [2cta-autoWS-sync.md, "Insert2CTASync: "Arrive Remote, Wait Local" Pattern"](../../../../../../../../docs/design/2cta-autoWS-sync.md#insert2ctasync-arrive-remote-wait-local-pattern)).
- **Ordinary TMA output staging is one copy** (see Hard prunes).
- **Cooperative TMA is the default.** Rank-2 descriptor loads are marked as cooperative pair transactions unless `TRITON_DISABLE_2CTA_COOPERATIVE_LOADS` is set (`Transform2CTALoads.cpp`, `AttrTwoCTALoadName`).

**What it relieves.**
- SMEM per operand stage drops from `(BLOCK_M + BLOCK_N)·BLOCK_K·elt` to `(BLOCK_M + BLOCK_N/2)·BLOCK_K·elt`.
- B global/L2 traffic per CTA halves.

So 2-CTA lets a wide logical tile (e.g. `BLOCK_N=256`), or a deeper operand ring at a fixed tile, fit where 1-CTA does not.

**What it costs.**
- One cross-CTA rendezvous per MMA.
- One output staging copy. The epilogue cannot rotate staging slots across tiles, so store overlap comes only from subtiling inside a tile.
- A cluster of 2 that the tile schedule must pair-align.

**Partition and schedule differences from 1-CTA at an equal config.**
- The rule that gives the computation partition 8 warps when a reduction partition exists is skipped for 2-CTA (`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp`, `optimizePartitionNumWarps`).
- 2-CTA re-runs loop scheduling after warp specialization; 1-CTA does not (`third_party/nvidia/backend/compiler.py`, `make_ttgir`).

## Pruning rules

- **IF** the launch has no `ctas_per_cga=(2,1,1)`, **THEN** skip every `two_ctas=True` config, **BECAUSE** each compiles as the 1-CTA kernel (duplicate config).
- **IF** `BLOCK_M` < 64, **THEN** skip `two_ctas=True`, **BECAUSE** it falls back to 1-CTA.
- **IF** any dot in the kernel cannot be 2-CTA (e.g. an rhs-only scaled `dot_scaled`), **THEN** skip 2-CTA for the whole kernel, **BECAUSE** every tcgen05 op must use one `cta_group`.
- **IF** the epilogue uses pointer stores, **THEN** skip 2-CTA, **BECAUSE** 2-CTA requires a TMA output store.
- **IF** the tile schedule is changed (grouped-M, group-by-N, custom order), **THEN** only try orders where tile ids 2k and 2k+1 land on the same `pid_n` and adjacent `pid_m`, **BECAUSE** anything else is a wrong answer. Arithmetic for a static persistent loop `range(program_id(0), num_tiles, num_programs)` with grouped-M ordering: the tile-M count, the group size (`GROUP_M`) and the number of programs must all be even, so pad the M-tile count up to even and let TMA clip the padded tile.
- **IF** the target tile does not fit 1-CTA SMEM at two or more operand stages, but does at `BLOCK_N/2` per CTA, **THEN** only try that tile as 2-CTA, **BECAUSE** halving B is the only change that keeps the tile.
- **IF** two 2-CTA configs differ only in `TRITON_WS_STAGING_COPIES`, **THEN** keep one, **BECAUSE** staging is pinned to one copy.
- **IF** the epilogue depends on overlapping the store of tile i with the epilogue of tile i+1 through multiple staging copies, **THEN** keep a 1-CTA candidate alongside the 2-CTA one, **BECAUSE** 2-CTA removes that overlap.
- **IF** a 2-CTA sweep is limited to `num_stages=1`, **THEN** widen it, **BECAUSE** the rendezvous barrier is multi-buffered to the operand depth.

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
- `two_ctas` × CLC tile scheduler (see `clc_persistence.md`)
- `two_ctas` × atomic dynamic tile scheduler
- `two_ctas` × `smem_alloc_algo`
- `two_ctas` × TMA multicast (see `multicast.md`)

## Hazards

None in [hazards.md](hazards.md) yet. Suspected from code but not reproduced, so tracked as rows in [TRACKER.md](TRACKER.md):
- `two_ctas` when warp specialization is off or silently stripped (e.g. the warp estimate exceeds 16). Two sync protocols apply at once: the rendezvous from `Insert2CTASync` and the inline cluster arrive from `convertDotImpl`.
- `two_ctas` whose B is not a descriptor load, or `BLOCK_N` < 32: the B split is skipped with only a debug log, while the MMA lowering still reads a half-width B.
- `two_ctas` with the atomic dynamic tile scheduler: there is no cluster-level tile claim, and no compile-time rejection.
- `two_ctas` with `smem_alloc_algo=0`: the one-copy staging pin lives only in the default planner.

## Not yet established

See [TRACKER.md](TRACKER.md):
- Correctness and value of `BLOCK_M=64` 2-CTA (RHS-split TMEM mode).
- Whether `BLOCK_N=64` 2-CTA still aborts compilation (`getTMABlockShape failed`) now that the half-tile swizzle is re-derived.
- Root cause of the hang with warp specialization off.
- When 2-CTA beats the best 1-CTA config. It has been observed to depend on kernel class: plain GEMM vs epilogues that add an SMEM consumer.
- The throughput cost of one output staging copy.
- Which `num_stages` suit 2-CTA at `BLOCK_K=128, BLOCK_N=256`.
- Cooperative vs per-CTA operand loads (`TRITON_DISABLE_2CTA_COOPERATIVE_LOADS`).
