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

| Knob | How set | What it does | Existing doc |
|---|---|---|---|
| `two_ctas` | `tl.dot(..., two_ctas=True)` / `tl.dot_scaled(..., two_ctas=True)` | Requests one `tcgen05.mma.cta_group::2` across a CTA pair | [2cta-autoWS-sync.md, "User API"](../../../../../../../../docs/design/2cta-autoWS-sync.md#user-api) |
| `ctas_per_cga` | compile option (`triton.Config(ctas_per_cga=(2,1,1))`) | Physical cluster shape; without it `two_ctas` is dropped | [2cta-autoWS-sync.md, "Supported Configurations"](../../../../../../../../docs/design/2cta-autoWS-sync.md#supported-configurations) |
| `TRITON_DISABLE_2CTA_COOPERATIVE_LOADS` | env var | Each CTA issues its own TMA instead of one cooperative pair transaction | [2cta-autoWS-sync.md, "Transform2CTALoads Pass"](../../../../../../../../docs/design/2cta-autoWS-sync.md#transform2ctaloads-pass) |

## Never try (any problem)

| Condition | Status | Why | Code |
|---|---|---|---|
| `two_ctas=True` without a cluster of x-dimension >= 2 (`ctas_per_cga` unset) | silently ignored | Falls back to 1-CTA MMA with only a warning: "two_ctas=True requires ctas_per_cga=(2,1,1) or larger ... Falling back to 1-CTA MMA." | `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`, `BlockedToMMAv5`, `ScaledBlockedToMMAv5` |
| `two_ctas=True` with `BLOCK_M` < 64 | silently ignored | A power-of-2 `BLOCK_M` below 64 gets no tcgen05 MMA at all, 1-CTA or 2-CTA (see [tiles_and_stages.md](tiles_and_stages.md#never-try-any-problem)). The "two_ctas=True with BLOCK_M < 64 is not supported" warning in `BlockedToMMAv5` is reached only by non-power-of-2 M. `BLOCK_M=64` compiles as real 2-CTA (`TwoCTA_RHS`) | `lib/Analysis/Utility.cpp`, `isMmaOutputShapeSupported`; `AccelerateMatmul.cpp`, `BlockedToMMAv5` |
| `two_ctas=True` on a `tl.dot_scaled` with only the rhs scaled | silently ignored | Falls back to 1-CTA MMA with only a warning: "two_ctas=True is not supported when only the rhs ..." (the dot is transposed) | `AccelerateMatmul.cpp`, `transposeDotOp` (called from `transposeDots`) |
| `two_ctas` differs between dots in one kernel | compile error | "inconsistent two_ctas setting across matmuls"; PTX requires one `cta_group` for every tcgen05 op in a kernel | `lib/Dialect/TritonNvidiaGPU/Transforms/CheckMatmulTwoCTAs.cpp` |
| a 2-CTA dot consumes a value derived from another 2-CTA dot | compile error | "two_ctas=True does not currently support dependent matmul chains", unless the compile option `allowDependentTwoCTA` is set (pass option `allow-dependent-chains`) | `CheckMatmulTwoCTAs.cpp`, `allowDependentChains`; `third_party/nvidia/backend/compiler.py`, `CUDAOptions.allowDependentTwoCTA` |
| cluster shape other than `(2,1,1)` | requires | `ctas_per_cga=(2,1,1)`: Y/Z must be 1; larger even X is listed as future work | [2cta-autoWS-sync.md, "Supported Configurations"](../../../../../../../../docs/design/2cta-autoWS-sync.md#supported-configurations) |
| pointer (non-descriptor) output store | requires | A TMA output store: pointer stores do not create the Meta WS epilogue partition | [2cta-autoWS-sync.md, "Current Limitations"](../../../../../../../../docs/design/2cta-autoWS-sync.md#current-limitations) |
| tile schedule where the two CTAs of a cluster (`program_id(0)` = 2k, 2k+1) get different `pid_n`, or non-adjacent `pid_m` | wrong answer | The pair shares one B tile. The compiler cannot prove a schedule pair-aligned and does not check it | [2cta-autoWS-sync.md, "Current Limitations"](../../../../../../../../docs/design/2cta-autoWS-sync.md#current-limitations) |
| `TRITON_WS_STAGING_COPIES` > 1 | silently ignored | TMA output staging stays at one copy: "Ordinary 2CTA TMA stores have one completion phase per paired-CTA tile" | `hopper/WarpSpecialization/WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |
| the output staging correctness floor needs more than one copy | compile error | "2CTA ordinary TMA output staging requires one copy" | `WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |
| B not split: `BLOCK_N/2` < 16 (e.g. `BLOCK_N=16`), or B is not a direct descriptor load feeding the MMA (e.g. TMA-loaded, then scaled or cast in registers: a B-side prologue) | wrong answer (reproduced) | The B split is skipped with only a debug log, but the MMA still reads a half-width B per CTA; no error or warning. A register-modified A is fine: it is promoted to TMEM. See [hazards.md](hazards.md#2-cta-with-an-unsplit-b) | `hopper/Transform2CTALoads.cpp`, `runOnOperation` ("B not from descriptor load"), `transformBLoad`, `transformSplitBLoad` (`halfN < 16`); `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`, `convertDotImpl` |
| pointer-loaded (non-descriptor) B under Meta WS | compile error | "pipeliner doesn't know how to predicate this op" | `lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp`, `predicateOp` |
| Meta WS off (`TRITON_USE_META_WS` unset), persistent 2-CTA GEMM | wrong answer (reproduced) | Wrong on every run with a pointer-loaded B, on most runs with a descriptor-loaded B. See [hazards.md](hazards.md#2-cta-without-warp-specialization) | `hopper/Insert2CTASync.cpp`, `insertSyncBeforeMMA`; `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`, `convertDotImpl` |
| Meta WS on, loop not warp-specialized (`warp_specialize=False`) | hangs (reproduced) | The first iteration completes, then the kernel hangs. See [hazards.md](hazards.md#2-cta-without-warp-specialization) | same |
| any condition that silently strips warp specialization: `num_warps` < 4, warp estimate > 16, a rejected replicated atomic | hangs | Stripping WS leaves the loop as if `warp_specialize=False`, which hangs (reproduced): `Insert2CTASync` still runs (it is gated only on Meta WS) and the MMA lowering adds its own cluster arrive | `hopper/Insert2CTASync.cpp`; `MMAv5.cpp`, `convertDotImpl` |
| `DynamicPersistent1DScheduler` (atomic dynamic tile scheduler) | wrong answer (reproduced); hangs (reproduced) | Compiles silently; the atomic scheduler has no cluster-level tile claim, so the two CTAs advance the counter independently. Static + 2-CTA and dynamic + 1-CTA are exact. See [hazards.md](hazards.md#2-cta-with-the-atomic-dynamic-tile-scheduler) | `python/triton/language/schedule.py`, `DynamicPersistent1DScheduler`; design doc: [DynamicPersistentAutoWSGaps.md § 2-CTA Status](../DynamicPersistentAutoWSGaps.md#2-cta-status) |

## Mechanism

**What it changes.**
- **B is halved per CTA.** Each CTA's B descriptor load is rewritten to half width (`BLOCK_K x BLOCK_N/2`); host-side descriptor argument types are rewritten too. The swizzle is re-derived for the half tile (`hopper/Transform2CTALoads.cpp`, `transformBLoad`, `transformSplitBLoad`, `shrinkSwizzleForShape`).
- **One MMA covers the pair.** It computes a `2*BLOCK_M x BLOCK_N` tile. Only the even-rank (leader) CTA issues it; the lowering predicates the MMA on the CTA rank (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`, `convertDotImpl`).
- **A rendezvous gates every MMA.** Both CTAs arrive on the leader's barrier, and the leader waits before the MMA. The barrier is multi-buffered to the MMA's operand pipeline depth, so any `num_stages` works (`hopper/Insert2CTASync.cpp`, `getOperandPipelineDepth`, `insertSyncBeforeMMA`; [2cta-autoWS-sync.md, "Insert2CTASync: "Arrive Remote, Wait Local" Pattern"](../../../../../../../../docs/design/2cta-autoWS-sync.md#insert2ctasync-arrive-remote-wait-local-pattern)).
- **Ordinary TMA output staging is one copy** (see "Never try (any problem)").
- **Cooperative TMA is the default.** Rank-2 descriptor loads are marked as cooperative pair transactions unless `TRITON_DISABLE_2CTA_COOPERATIVE_LOADS` is set (`Transform2CTALoads.cpp`, `AttrTwoCTALoadName`).

**What it relieves.**
- SMEM per operand stage drops from `(BLOCK_M + BLOCK_N)·BLOCK_K·elt` to `(BLOCK_M + BLOCK_N/2)·BLOCK_K·elt`.
- B global/L2 traffic per CTA halves.

So 2-CTA lets a wide logical tile (e.g. `BLOCK_N=256`), or a deeper operand ring at a fixed tile, fit where 1-CTA does not.

**What it costs.**
- One cross-CTA rendezvous per MMA.
- One output staging copy. The epilogue cannot rotate staging slots across tiles, so store overlap comes only from subtiling inside a tile.
- A cluster of 2 that the tile schedule must pair-align.

**Partition and schedule differences from a kernel without a cluster.** Both are keyed on the cluster (`ctas_per_cga`), not on the `two_ctas` value, so they also apply to `two_ctas=False` cluster kernels:
- The rule that gives the computation partition 8 warps when a reduction partition exists is skipped when the module carries the 2-CTA attribute, which `CheckMatmulTwoCTAs.cpp` sets whenever a cluster is present (`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp`, `optimizePartitionNumWarps`, `AttrTwoCTAsName`).
- Loop scheduling is re-run after warp specialization when `max(cluster_dims) >= 2` (`third_party/nvidia/backend/compiler.py`, `make_ttgir`).

## When to try it (depends on the problem)

- **IF** warp specialization is off or stripped for the loop (Meta WS off, `warp_specialize=False`, or a strip condition), **THEN** skip `two_ctas=True`, **BECAUSE** it gives a wrong answer or hangs.
- **IF** the launch has no `ctas_per_cga=(2,1,1)`, **THEN** skip every `two_ctas=True` config, **BECAUSE** `two_ctas` is silently ignored and each compiles as the 1-CTA kernel.
- **IF** `BLOCK_M` < 64, **THEN** skip `two_ctas=True`, **BECAUSE** there is no tcgen05 MMA at that `BLOCK_M`. `BLOCK_M=64` is a valid 2-CTA config.
- **IF** any dot in the kernel cannot be 2-CTA (e.g. an rhs-only scaled `dot_scaled`), **THEN** skip 2-CTA for the whole kernel, **BECAUSE** every tcgen05 op must use one `cta_group`.
- **IF** the epilogue uses pointer stores, **THEN** skip 2-CTA, **BECAUSE** 2-CTA requires a TMA output store.
- **IF** the tile schedule is changed (grouped-M, group-by-N, custom order), **THEN** only try orders where tile ids 2k and 2k+1 land on the same `pid_n` and adjacent `pid_m`, **BECAUSE** anything else is a wrong answer. Arithmetic for a static persistent loop `range(program_id(0), num_tiles, num_programs)` with grouped-M ordering: the tile-M count, the group size (`GROUP_M`) and the number of programs must all be even, so pad the M-tile count up to even and let TMA clip the padded tile.
- **IF** the target tile does not fit 1-CTA SMEM at two or more operand stages, but does at `BLOCK_N/2` per CTA, **THEN** only try that tile as 2-CTA (this removes the 1-CTA variant of that tile), **BECAUSE** halving B keeps the tile and its ring depth.
- **IF** two 2-CTA configs differ only in `TRITON_WS_STAGING_COPIES`, **THEN** keep one, **BECAUSE** staging is pinned to one copy.
- **IF** the epilogue depends on overlapping the store of tile i with the epilogue of tile i+1 through multiple staging copies, **THEN** keep a 1-CTA candidate alongside the 2-CTA one (this keeps configs; it removes none), **BECAUSE** 2-CTA removes that overlap.
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
- `two_ctas` × CLC tile scheduler (see [clc_persistence.md](clc_persistence.md))
- `two_ctas` × atomic dynamic tile scheduler
- `two_ctas` × `smem_alloc_algo`
- `two_ctas` × TMA multicast (see [multicast.md](multicast.md))

## Hazards

- [2-CTA with an unsplit B](hazards.md#2-cta-with-an-unsplit-b)
- [2-CTA without warp specialization](hazards.md#2-cta-without-warp-specialization)
- [2-CTA with the atomic dynamic tile scheduler](hazards.md#2-cta-with-the-atomic-dynamic-tile-scheduler)

Suspected from code, not reproduced in testing (rows in [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause)): `two_ctas` with `smem_alloc_algo=0`; a register-written SMEM operand read by a 2-CTA MMA without a cluster-scope fence.

## Not yet established

See [TRACKER.md](TRACKER.md):
- Value of `BLOCK_M=64` 2-CTA (it compiles as real 2-CTA and is correct).
- Root cause of the hang with warp specialization off.
- When 2-CTA beats the best 1-CTA config. It has been observed to depend on kernel class: plain GEMM vs epilogues that add an SMEM consumer.
- The throughput cost of one output staging copy.
- Which `num_stages` suit 2-CTA at `BLOCK_K=128, BLOCK_N=256`.
- Cooperative vs per-CTA operand loads (`TRITON_DISABLE_2CTA_COOPERATIVE_LOADS`).
