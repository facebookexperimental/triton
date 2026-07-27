# Lowering Plan

**Files**: `ModuloScheduleGraph.h`, `ModuloSchedulePass.cpp`,
`python/triton/tools/modulo_joint_solver.py`, and
`third_party/tlx/tools/sched2tlx/sched2tlx/schedule_graph.py`

The lowering plan is the versioned contract for making synchronization and
data-movement work visible to the joint scheduler. It is currently a
**fixed-II shadow model**: it participates in the joint cycle/warp-group solve
and is dumped for comparison with the emitter, but `sched2tlx` still emits from
`cross_wg_barriers` and its own semaphore derivation.

This distinction is intentional. A shadow plan can expose model/emitter drift
without making an incomplete event model part of the correctness path.

## Data Model

`LoweringTemplate` describes conditional work before warp groups are chosen.
Its relation is `always`, `same_wg`, or `different_wg`. Each event records:

- kind (`wait`, `arrive`, `expect`, `local_store`, `local_load`, `fence`, or
  `tc_commit`), owner, anchor node, and before/after placement;
- issue pipeline, duration, completion latency, blocking/async properties,
  frequency, and loop distance;
- optional buffer, fusion, and dedup identities plus bytes, depth, and
  semaphore role.

`LoweringPlan` instantiates active templates after solving. Each concrete event
has an absolute cycle, owner warp group, and per-warp-group stream order.
Schema `lowering-plan-0.1` has four states:

| Status | Meaning |
| --- | --- |
| `absent` | No joint lowering model was produced. |
| `shadow_unmodeled` | A v1 partition solve, including a v2-to-v1 arbitration result, materialized coordinates without event-resource constraints. |
| `shadow_verified` | A v2 fixed-II solve modeled event issue slots and the plan still matches final node ownership after compiler post-processing. |
| `shadow_stale` | Later ownership changes or a structural mismatch invalidated the solver result. The emitter must not consume it. |

The schedule-graph dump contains both `lowering_templates` and
`lowering_plan`. A plan without its templates is not self-describing and is
rejected by the Python parser.

`issue_duration` is per dynamic occurrence; its modeled stream occupancy is
`issue_duration * frequency`. Synchronization instructions use pipeline
`NONE`: they occupy the owner's warp-group issue stream, not the chip-wide
pipeline of the anchor operation.

## Current Coverage

The first shadow model covers two protocols derived from DDG edges:

1. Tensor-core completion consumed by CUDA/SFU work: `tc_commit` plus a
   blocking `wait`.
2. Loop-carried CUDA/SFU-to-tensor-core signals crossing warp groups: `arrive`
   plus `wait`.

Event issue intervals occupy the owner warp group's issue stream. A
lexicographic second solve minimizes independent CUDA/SFU work left behind
blocking waits while preserving a proven-optimal primary objective. Every
schedule, partition, arbitration, and secondary CP-SAT solve uses one worker
and a fixed seed plus a deterministic-time budget so the final response is
independent of host load. The joint solve also fixes every node to its
committed stage because stage moves are not representable by the current
emitter.

This is not yet lowering-aware II search. The request fixes `ii` to the
existing schedule's II, and the independent `JointSolverScheduler` II sweep
does not yet receive these templates. TMA handoffs, local store/load bridges,
full/empty recycle, fan-in/out, fences, and final synthesized buffer identities
remain emitter-derived.

## Validation Boundary

The solver response is validated into temporaries before any schedule state is
committed. For v2, the compiler first recomputes buffer counts and synthesized
channel depth from the solved cycles, re-merges buffers, deduplicates channels
by producer result, preserves explicit ring-depth floors, and checks that
loop's SMEM/TMEM hard caps. After scalar demotion, infrastructure propagation,
cross-loop warp group reconciliation, barrier synthesis, and buffer merging,
it performs a fail-closed whole-kernel SMEM and per-loop TMEM audit. The SMEM
audit mirrors sched2tlx's signal-only elimination and unsafe-alias fallback to
private allocations. Until SemIR itself is shared, the audit conservatively
reserves a full/empty barrier pair for every non-signal SMEM/TMEM data buffer,
with existing and cross-WG barrier objects deducted. It then checks template
presence and event ownership again. Only a matching v2 plan is marked
`shadow_verified`.

The sched2tlx parser accepts old graphs with no plan, parses all template/event
semantics, and rejects an unknown version, unknown state, duplicate IDs,
invalid resources, non-contiguous stream order, or ownership mismatch in a
verified plan. Parsing is observability only; code emission is unchanged.

## Promotion To Authoritative Lowering

The plan can replace emitter re-derivation only after all of the following hold:

1. A shared planner represents every emitted protocol event and final channel
   identity, including fusion and deduplication.
2. Fixed-assignment evaluation matches emitter traces for the case corpus.
3. The plan generates `cross_wg_barriers` and synthesized buffers instead of
   being computed before them.
4. sched2tlx consumes supported plan versions fail-closed and cannot add,
   remove, reorder, or deepen protocol work.
5. The same event model participates in the proven-feasible fixed-II sweep;
   an `UNKNOWN` lower II may not be skipped.

The generic II sweep already stops fail-closed at an `UNKNOWN` candidate. Its
remaining gap is that it does not yet receive the lowering event model.

Until these gates pass, `shadow_verified` means structural agreement with the
modeled subset, not exact effective-II ownership.
