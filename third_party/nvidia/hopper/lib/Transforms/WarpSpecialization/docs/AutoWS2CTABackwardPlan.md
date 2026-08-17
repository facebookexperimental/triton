# AutoWS 2-CTA Flash-Attention Backward Design

This document describes the current compiler design for Blackwell AutoWS
two-CTA Flash-Attention backward. Historical ablations, generated IR dumps,
and performance measurements belong in task T282685712 rather than this design
reference.

The implementation is exercised by:

```text
third_party/tlx/tutorials/fused_attention_ws_device_tma.py
```

The handwritten matching target is:

```text
third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py
```

## Status and scope

BM64 and direct-grid BM128 two-CTA backward are correct for dQ, dK, and dV.
BM128 has no known deadlock. Its final TTGIR matches TLX in the primary
structural counts: loops, MMAs, TMEM allocations, TMEM loads, output TMA
stores, and dQ TMA reductions.

The implementation supports:

- all five backward MMAs (`qK`, `dK`, `dP`, `dQ`, and `dV`) as two-CTA
  collectives;
- dependent two-CTA MMA chains;
- cooperative descriptor loads for rank-two MMA operands;
- the dS peer exchange required by dQ;
- the physical `TwoCTA_RHS` dQ accumulator and packed epilogue; and
- direct-grid ownership in which adjacent N tiles form a two-CTA cluster.

An automatic cost model for choosing one CTA versus two CTAs and arbitrary
dependent-MMA graphs are out of scope.

## Module-wide two-CTA uniformity

`cta_group::2` is not a per-operation choice. Once any `tcgen05` MMA in a
kernel issues as a two-CTA collective, every `tcgen05` MMA in that kernel must
issue the same way, and both CTAs of the cluster must reach each collective
issue point. The design therefore treats two-CTA as a module property, not an
attribute a single dot can opt into.

Enforcement lives in `CheckMatmulTwoCTAs`, which runs before any two-CTA
lowering and is not FA-specific:

- it walks every `tt.dot`, `ttng.tc_gen5_mma`, and `ttng.tc_gen5_mma_scaled`
  in the module and requires an identical `two_ctas` setting; a mismatch is a
  hard compile error naming both matmuls, never a silent downgrade to one CTA;
- it stamps the agreed value on the module as `ttg.two_ctas` so later passes
  read one authoritative mode instead of re-deriving it per operation; and
- it rejects the dependency shapes that are not yet realizable. The backward
  work relaxes that second check to permit the dependent two-CTA chains listed
  above, but it does not relax the uniformity check.

Two-CTA remains opt-in per matmul in the kernel source: the user decorates the
GEMMs and the pass proves the decoration is uniform. Inferring two-CTA
automatically would require the cost model that is out of scope here.

The remaining unproven condition is control flow. Uniformity of the *operations*
is checked; uniformity of the *paths that reach them* is not. Both CTAs
executing identical collective-MMA and peer-exchange control flow is currently
an assumption the kernel must satisfy (see "Execution model"), not an analysis
result. A divergent branch, an early loop exit on one CTA, or a rank-dependent
trip count breaks the collective without a diagnostic. That analysis is shared
with TMA multicast rather than specific to this design, and is tracked as the
last stage in "Delivery sequence".

## Execution model

Two physical CTAs form one cluster and execute each collective MMA together.
The compiler emits one logical operation:

```mlir
ttng.tc_gen5_mma ... {two_ctas}
```

Both CTAs must execute identical collective-MMA and peer-exchange control flow.
Neither CTA may leave a loop while its peer can still issue a collective MMA,
a remote DSMEM transaction, or an associated barrier operation.

The kernel supplies separate descriptor views when one tensor is needed with
incompatible collective block shapes:

```text
Q:  desc_q and desc_qt
dO: desc_do and desc_dot
K:  desc_k and desc_kt
```

The intended steady-state MMA order is:

```text
qK -> dK -> dP -> dQ -> dV
```

dP and dQ are ordered deliberately because dP/dS and dQ reuse tensor memory.
Changes to this order must be validated against the reuse plan and barrier
graph, not treated as a local scheduling change.

## MMA shapes and dQ representation

BM128 uses the TLX physical dQ formulation:

```text
dS gather: [256,64]
dQ A:      [64,256]
dQ B:      [256,64]
dQ D:      [64,128] with TwoCTA_RHS
```

Each CTA owns half of the logical M rows:

```text
CTA 0: dQ[M 0:64,   H 0:128]
CTA 1: dQ[M 64:128, H 0:128]
```

`TwoCTA_RHS` stores the logical `[64,128]` result through a physical
`[128,64]` TMEM view. The epilogue must unload that physical view and address
the packed dQ descriptor consistently. Slicing it as an ordinary logical
`[64,128]` tensor duplicates the first H half.

BM64 uses the same collective principle with a `[64,128]` dQ contraction and
the direct peer-completion protocol. BM128 uses a larger peer payload and a
dedicated relay partition.

## Dependency classification and peer exchange

`Analyze2CTADependencies` classifies producer-to-consumer MMA edges as:

- `replicated`;
- `partition_preserving`;
- `collective_contraction`;
- `requires_peer_gather`; or
- `unsupported`.

dV and dK are collective contractions. dQ operand A requires a peer gather
because each CTA needs the peer-owned half of transposed dS.

`Plan2CTAExchange` inserts an abstract `ttng.two_cta_peer_gather` before
partition scheduling and memory planning. BM128 also receives a
`ttng.two_cta_peer_relay` marker. Inserting these operations early makes the
dependency and destination lifetime visible to AutoWS.

After software pipelining, `Materialize2CTAExchange` lowers the plan to packed
local stores, asynchronous remote DSMEM copies, fences, and barrier
publication. The BM128 one-warp relay waits for peer-copy completion and then
publishes the existing dQ consumer channel. It is not a CTA-wide cluster
rendezvous; placing a cluster barrier inside a warp-specialized partition can
deadlock because not every CTA warp participates.

## Cooperative TMA loads

`Transform2CTALoads` marks and reshapes descriptor-backed rank-two MMA operand
loads for the hardware cooperative CTA-group TMA protocol. Both CTAs issue the
load, only the pair leader executes `barrier_expect`, and completion is relayed
to the follower. Rank-one metadata loads remain ordinary per-CTA loads.

A fused TMA barrier group must be homogeneous: every member is either a
cooperative two-CTA load or an ordinary per-CTA load. Mixing the protocols
would give the barrier inconsistent completion routing and expected-byte
semantics. `WSLowerMem::optimizeTMALoads` enforces this invariant.

## Synchronization lifetime

Collective-MMA issue synchronization and DSMEM peer-copy completion are
separate protocols.

Software pipelining creates prologue, steady-state, and epilogue copies of an
MMA. Some copies reside directly in a warp-specialize partition rather than an
enclosing `scf.for`. `Insert2CTASync` therefore allocates and initializes their
two-CTA issue barriers at kernel-entry lifetime and captures them into the
partition. Initializing an issue barrier inside the partition permits a remote
arrival to race initialization and produces a compute-sanitizer `Missing init`
error.

Any change to issue handshakes, relay completion, or completion-commit
coalescing requires barrier visualization and compute-sanitizer synccheck
before performance testing.

## Partitioning and memory planning

The BM128 schedule contains these logical roles:

```text
computation, reduction, gemm, load, relay
```

The relay is a typed one-warp partition. The computation partition retains the
softmax work; the GEMM partition owns the software-pipelined collective MMAs;
the load partition owns descriptor loads; and the reduction partition owns
dQ's TMA reduction path.

Important storage relationships are:

- qK and P share storage where their lifetimes permit;
- dP and dS time-multiplex one TMEM region;
- dQ reuses the dP/dS TMEM group;
- dK and dV use independent accumulator storage; and
- remote DSMEM views alias local shared allocations rather than allocating
  additional storage.

Reuse candidates must have compatible tensor-memory encodings, two-CTA modes,
element types, and physical extents. Required placement should be expressed as
general lifetime or compatibility constraints, not FA-specific memory-planner
rules.

## Pass pipeline and ownership

```text
CheckMatmulTwoCTAs
  -> validate the module CTA mode and supported dependent chains

AccelerateMatmul
  -> create two-CTA TCGen5 MMA operations
  -> select the default or TwoCTA_RHS accumulator encoding

Analyze2CTADependencies
  -> classify collective contractions and dQ peer gather

Transform2CTALoads
  -> form cooperative descriptor-backed operand loads

Plan2CTAExchange
  -> insert the abstract dS peer gather and BM128 relay marker

PartitionSchedulingMeta
  -> assign computation, reduction, gemm, load, and relay partitions

AutoWS code partition and memory planning
  -> allocate channels, barriers, SMEM, and TMEM reuse groups

OptimizePartitionWarps
  -> assign the requested warps, including one relay warp

Materialize2CTAExchange
  -> lower the planned peer exchange after software pipelining

Insert2CTASync
  -> materialize the hardware two-CTA MMA issue protocol

TMA/TMEM and LLVM lowering
```

## Delivery sequence

The work is split so that each stage is independently reviewable and the
general two-CTA machinery lands before the FA-specific parts:

1. **General two-CTA validation.** Tighten and generalize the existing
   handling so a two-CTA module is provably uniform (see "Module-wide two-CTA
   uniformity"), keeping user decoration of every GEMM as the conservative
   interim rule. Nothing FA-specific and no new lowering.
2. **Dependency classification.** `Analyze2CTADependencies` and the FA-backward
   edge classes, including the dQ peer gather. Analysis metadata only.
3. **Basic implementation, no relay partition.** Cooperative loads, the
   collective MMAs, and the dS exchange using the direct peer-completion
   protocol, in which the remote transaction completes the barrier the consumer
   MMA already waits on.
4. **Relay partition.** The typed one-warp relay is a follow-up on top of a
   working stage 3, not a prerequisite for it. BM64 needs no relay. For BM128
   the relay must be justified before it lands: either the larger peer payload
   makes the direct protocol insufficient, in which case it is a correctness
   requirement and the reason belongs in this document, or it only shortens the
   dQ critical path, in which case it is a latency optimization and is measured
   as one.
5. **Safety analysis for divergent cluster control flow.** Prove — or reject —
   that both CTAs reach every collective issue point. This is the most likely
   source of silent breakage and is shared with other multi-CTA operations such
   as TMA multicast, so it should be solved once, generally, rather than for
   this kernel.

## Validation

Compiler changes require full production-derived TTGIR fixtures captured
immediately before the changed pass. Fixtures must preserve target metadata,
encodings, cluster dimensions, loop structure, descriptors, and side effects.
Keep only live debug-location aliases and use repository-relative source paths.

Current BM128 coverage includes:

```text
test/Hopper/TwoCTA/analyze_2cta_dependencies_bwd_bm128_persistent.mlir
test/Hopper/TwoCTA/plan_2cta_exchange_bwd_bm128_persistent.mlir
test/Hopper/TwoCTA/plan_2cta_exchange_bwd_bm128_nw8.mlir
test/Hopper/TwoCTA/materialize_2cta_exchange_bwd_bm128_persistent.mlir
test/Hopper/TwoCTA/insert_2cta_sync_bwd_bm128_inner.mlir
test/Hopper/TwoCTA/partition_scheduling_bwd_bm128_computation_default.mlir
test/Hopper/TwoCTA/pipeline_bwd_bm128_gemm_only.mlir
test/Conversion/allocate_warp_groups_autows_bwd_bm128.mlir
test/Hopper/WarpSpecialization/optimize_partition_warps_bwd_bm128_nonpersistent.mlir
test/Hopper/WarpSpecialization/ws_remove_redundant_tmem_zero_bwd_bm128_inner.mlir
```

Runtime validation requires:

1. forced-recompile dQ, dK, and dV correctness against Torch;
2. repeated-launch correctness for the BM128 two-CTA nonpersistent kernel;
3. synccheck for issue-barrier or relay changes;
4. a final-TTGIR structural comparison against TLX; and
5. a matched single-kernel benchmark before reporting performance.

### Completion criteria

Backward is not the whole surface. The work is complete when:

1. dQ, dK, and dV are correct and deadlock-free for BM64 and BM128, on both the
   direct and persistent grids;
2. Flash-Attention **forward** still compiles, is numerically correct, and does
   not regress in performance. Forward does not use two CTAs today, so it is a
   one-CTA module and the generalized validation must leave it untouched;
   because uniformity is enforced module-wide, forward and backward must remain
   separate kernels for as long as backward is a two-CTA module;
3. enabling two-CTA forward requires no FA-backward-specific compiler code. It
   is not a goal of this work, but if it cannot be attempted without touching
   the backward paths, the generalization in stages 1 and 2 of the delivery
   sequence is incomplete; and
4. the two-CTA lit fixtures above pass alongside the existing one-CTA
   WarpSpecialization suite.

## Remaining matching work

The remaining gap is schedule realization rather than kernel structure:

1. align `descriptor_store_wait` placement with TLX and keep epilogue
   `local_load` values live only until their corresponding store;
2. align TMA-load software-pipeline ordering and descriptor/phase lifetimes;
3. preserve the cooperative two-CTA load protocol while reducing redundant
   software synchronization;
4. reduce excess GEMM issue waits/arrivals; and
5. revisit completion-commit coalescing only after issue synchronization is
   matched.

Do not infer progress from generated source shape alone. Re-run correctness,
synccheck where applicable, the final-TTGIR comparison, and the matched kernel
benchmark after each synchronization or scheduling checkpoint.
