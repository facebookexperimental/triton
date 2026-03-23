# TTGIR Control Flow Ops

Warp specialization structure, pipeline control, and cluster launch control.

## Warp Specialization

**`ttg.warp_specialize`**: Top-level op for running different code on different
warp groups simultaneously. Contains a "default" region (implicit capture) and
N "partition" regions (isolated from above, explicit captures as block args).
All regions start simultaneously and join at the end.

Key attributes: `partitionNumWarps`, `warpGroupStartIds`,
`requestedRegisters` / `actualRegisters`.

Related ops:
- `ttg.warp_specialize.partitions`: Container for partition regions
  (the `IsolatedFromAbove` boundary)
- `ttg.warp_yield`: Terminates the default region; operands become the
  `warp_specialize` results
- `ttg.warp_return`: Terminates partition regions; no operands (partitions
  communicate via SMEM/barriers)

## Pipeline Control

- `ttg.predicate_stage`: Generates a predicate for a software pipeline stage
  given `(iv, ub, step, maxStage, stage)`.
- `ttg.mask` / `ttg.mask.return`: Guarded execution region — operations inside
  only execute when the predicate is true.

## Cluster Launch Control (Blackwell only)

CLC enables dynamic persistent kernels with work stealing on SM100+.

- `ttng.async_clc_try_cancel`: Request atomic cancellation of a not-yet-launched
  cluster. Writes opaque 16-byte response to SMEM. Tracked by mbarrier.
  PTX: `clusterlaunchcontrol.try_cancel.async.shared::cta`.
- `ttng.clc_query_cancel`: Extract CTA ID from cancel response. Returns -1 if
  cancellation failed (cluster already launched).
