# BM128 2-CTA Relay Design

This document records the current design and validation requirements for the
BM128 Flash-Attention backward peer exchange. Historical experiments and
performance logs belong in task T282685712 rather than this design reference.

## Scope

The BM128 2-CTA backward schedule uses five two-CTA MMAs. Four operand flows are
collective contractions. The dQ operand-A flow is different: each CTA needs a
peer fragment of the transposed dS tile, so it requires an explicit peer gather.

The implementation is split across these passes:

- `Analyze2CTADependencies` classifies collective contractions and peer gathers.
- `Plan2CTAExchange` inserts an abstract gather and relay dependency before
  AutoWS scheduling and memory planning.
- `Materialize2CTAExchange` lowers the planned gather after pipelining.
- `Insert2CTASync` provides the hardware two-CTA MMA issue protocol.

## Relay protocol

The producer writes its local dS half into a packed shared-memory destination.
A dedicated one-warp relay partition copies the fragment to the peer CTA with
an asynchronous DSMEM copy. Completion is published through the existing dQ
consumer channel; the GEMM partition waits before issuing dQ.

The relay is intentionally not a CTA-wide cluster rendezvous. Warp-specialized
partitions do not include every warp in the CTA, while cluster barriers require
full CTA participation and deadlock when placed inside only one partition.

## Barrier lifetime

Two-CTA issue barriers must dominate every peer arrival. Software pipelining
creates prologue, steady-state, and epilogue MMA copies, some directly inside a
warp-specialize partition. `Insert2CTASync` hoists their barrier allocation and
initialization to kernel-entry lifetime and captures the barriers into the
partition. Initializing those barriers inside the partition permits a remote
arrival to race initialization and produces a compute-sanitizer `Missing init`
error.

## TMA loads

Rank-two operand descriptor loads use the hardware cooperative CTA-group TMA
protocol. Both CTAs issue the load, only the pair leader executes
`barrier_expect`, and completion is relayed to the follower. Rank-one metadata
loads remain ordinary per-CTA transactions.

Loads fused onto one TMA barrier must use the same issue protocol. A group may
contain only cooperative loads or only ordinary per-CTA loads; mixing them
would give the shared barrier inconsistent routing and byte-count semantics.

## Required coverage

Production-derived fixtures must cover the actual BM128 IR immediately before
each changed pass:

- `analyze_2cta_dependencies_bwd_bm128_persistent.mlir`
- `plan_2cta_exchange_bwd_bm128_persistent.mlir`
- `plan_2cta_exchange_bwd_bm128_nw8.mlir`
- `materialize_2cta_exchange_bwd_bm128_persistent.mlir`
- `insert_2cta_sync_bwd_bm128_inner.mlir`

The fixtures should retain only live debug-location aliases and use
repository-relative source paths.

Runtime validation requires:

1. focused dQ/dK/dV correctness for BM128 2-CTA nonpersistent backward;
2. repeated comparison against Torch and the handwritten TLX kernel;
3. compute-sanitizer synccheck for any issue-barrier or relay change;
4. a fresh final-TTGIR comparison before performance conclusions.

## Current status

The direct-grid BM128 kernel is correct and has no known deadlock. Its loop and
MMA structure, TMEM allocations, TMEM loads, and output-store counts match TLX.
The remaining performance work is schedule realization: descriptor-store wait
placement, TMA-load software pipelining, excess GEMM issue synchronization, and
completion-commit coalescing.
