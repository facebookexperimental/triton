# Contracted-Graph Modulo Scheduler

## Goal

Explore a small, useful family of two-stage software-pipelined schedules while
making GEMM placement the primary decision. Global memory operations and
SMEM/TMEM buffering must not dominate standalone exploration. Long elementwise
chains should influence legality, but should be contracted for candidate
ranking unless they directly face a GEMM.

The motivating FA backward schedule has:

```text
stage 0: qkT, dpT, dv
stage 1: dk, dq

cluster(qkT) < cluster(dk) <= cluster(dq) < cluster(dpT) <= cluster(dv)
```

Cluster order is modulo order. A stage-1 GEMM may therefore have a lower
cluster than a stage-0 producer from the preceding logical iteration.

## Scope

The implementation is selected with:

```text
TRITON_USE_MODULO_SCHEDULE=contracted
```

It initially remains a standalone exploration mode. The existing exhaustive,
random, SMS, and IMS schedulers retain their behavior.

## Graph Model

Each original DDG node is classified as one of:

- **GEMM anchor**: a TC-pipeline node.
- **GEMM-facing**: a direct distance-0 producer of a GEMM.
- **Computation**: a pure CUDA, SFU, or NONE node that is not GEMM-facing.
- **Memory**: a TMA/global load or store.
- **Boundary**: a loop-carried, side-effecting, or multiply-owned node that
  cannot safely be folded into one computation group.

For every computation node, a forward distance-0 walk finds its reachable GEMM
anchors. Maximal connected regions with the same reachable-GEMM set are one
contracted computation group. Direct GEMM producers remain visible. Parallel
group edges retain the strongest dependence. Computation with no downstream
GEMM is contracted as a sink-side group and capped relative to the smallest
GEMM latency.

The original DDG remains the source of truth for dependency topology. Admission
uses structural issue quanta: one quantum per resource-using operation, one per
emitted member of a data-partition bundle, zero for pipeline-free IR nodes, and
the already-established inner II for a nested scheduled loop. Modeled result
latency is not used for dependence constraints, resource reservations, or the
II lower bound.

## Timing

The scheduler uses two timing views:

- **Structural timing** supplies candidate legality. Pipeline kind and issue
  count provide a hard resource lower bound; dependence topology provides
  ordering. No result-latency or RecMII estimate participates.
- **Estimated timing** is the critical path through a computation group,
  capped relative to the nearest GEMM latency. It is only a tie-breaker among
  candidates already admitted by structural timing.

The initial cap is 25% of the nearest GEMM latency and can be changed for
experiments with `TRITON_CONTRACTED_COMPUTE_RATIO`.

Structural dependence delay uses the producer's issue quantum. In particular,
it does not use TC result latency or occupancy. Using full modeled result
latency would forbid a stage-0 GEMM near the end of the modulo interval from
feeding an earlier modulo cluster in stage 1, which is the schedule family this
mode is intended to expose.

## Search

For `G` GEMM anchors, enumerate the nontrivial `G`-bit stage assignments. GEMMs
are pinned to stages 0 or 1 and both stages must be used. Non-GEMM epilogue
nodes may occupy stage 2 when a loop-carried token requires it. Reject assignments
that violate distance-0 GEMM reachability. For each assignment:

1. Compute a structural resource lower bound, excluding latency-derived RecMII.
2. Schedule the original DDG in topological order.
3. Place pinned GEMMs in their requested stages.
4. Keep the leading stage-0 GEMM early and pack later stage-0 GEMMs at the end
   of the modulo interval. This leaves wrapped low clusters for stage-1 GEMMs.
5. Place every operation with its structural issue duration and the reservation
   table.
6. Validate every original-DDG dependence before ranking. Intra-iteration
   edges require producer issue before consumer issue. Loop-carried edges use
   the standard SWP structural ordering constraint, shifting the consumer by
   `distance * II` without imposing the result-latency model.

Candidate identity is `(II, ordered (GEMM node, stage, modulo cluster) tuple)`,
not the full per-node stage vector. This removes duplicates caused only by
loads or elementwise placement while preserving schedules whose different II
can change resource slack and downstream memory planning.

Unlike the production schedulers, contracted mode emits clusters as one global
dense rank of `cycle % II`, rather than restarting the rank in every stage.
Cross-stage cluster inequalities therefore describe the explored modulo order.

Candidates are ranked lexicographically by II, exact two-stage shape,
contracted critical-path cost, TC utilization, computation-cluster count, and
a stable GEMM signature. The final bounded frontier reserves roughly half its
slots for evenly spaced feasible II values, including both endpoints when
possible, then fills the rest from that ranking. The frontier is finally
presented in ranking order, so rank 0 remains the lowest-II default while
larger-II candidates cannot all be displaced by alternatives at the lower
bound. Buffer depth and SMEM/TMEM headroom are excluded.

## Diagnostics

`-debug-only=modulo-scheduling-contracted` reports graph sizes, classified and
contracted nodes, stage assignments considered/rejected, and the GEMM-only
signature for each retained top-K schedule.

## Testing

`test/TritonGPU/modulo-exhaustive-fa-bwd-bm64-tmem.mlir` is the primary lit
test. It checks GEMM stages and clusters only; descriptor loads/stores are not
part of the scheduling objective. The test must contain a top-K candidate with
the target FA backward GEMM schedule above. Pipeline-free and epilogue work may
extend the loop's `tt.scheduled_max_stage` beyond the two GEMM stages. A second
pick checks that the same top-K contains a larger-II candidate with enough
structural slack to keep the epilogue within stage 1.

## Implementation Status

- [x] Design and compatibility boundary documented.
- [x] Node classification and downstream GEMM ownership.
- [x] Contracted computation groups and ranking-only latency.
- [x] Structural resource lower bound without result latency or RecMII.
- [x] Structural dependence and reservation-table admission.
- [x] II-aware candidate identity and bounded II-diverse frontier.
- [x] Exact two-stage GEMM assignment enumeration.
- [x] Original-DDG placement and validation.
- [x] GEMM-signature top-K deduplication and diagnostics.
- [x] FA backward lit coverage for the target schedule.

Keep this section synchronized with implementation changes.
