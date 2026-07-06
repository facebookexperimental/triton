# Sub-tiling in front of the modulo solver — design note

Status: design, 2026-07-06 (SolverMigrationNotes step-4 item 4). Written
against branch hwu27/modulo-cpsat-joint-v2; all file:line references as of
that branch.

## Why

Ping-pong/FA4-class schedules — two M-half-tiles whose softmax and MMA
phases interleave in anti-phase — are ABSENT from the solver's solution
space today, not merely unreached: the DDG has one node per tile-level op,
so "sub-tile 0's softmax overlaps sub-tile 1's MMA" is not expressible.
Twill reports its joint formulation is UNSAT without sub-tiling; our
measured ~665 TFLOPS plateau on case3 (reproduced by three independent
solver configs) is the single-instance softmax-chain bound of the current
design point. Sub-tiling is the only lever that changes that bound rather
than approaching it.

## The target, concretely (from the hand-written kernels)

- **Hopper ping-pong** (`hopper_fa_ws_pipelined_pingpong.py`): BLOCK_M
  128 → 2×64, the consumer body replicated (`replicate=2`), and the ONLY
  addition over the non-pingpong variant is a named-barrier 9/10 handshake
  around the QK `async_dot` issue. In modulo-DDG terms that handshake is
  exactly one distance-0 edge (QK0@i → QK1@i) plus one distance-1 back
  edge (QK1@i → QK0@i+1): a recurrence pinning RecMII ≥ 2× TC occupancy
  and forcing the half-period offset. The schedule-graph IR already has
  the lowering vehicle (`CrossGroupBarrier.kind = "named"`).
- **Blackwell FA4-class** (`blackwell_fa_ws_pipelined_persistent.py`):
  same M-split, no named barriers — the interleave is program order
  inside the single 1-warp MMA task (`q0@k → p1@v_prev → q1@k → p0@v`),
  coupled to two replicated softmax tasks via per-cid mbarrier arrays and
  per-cid TMEM tiles.
- Shared vs duplicated: K/V TMA loads, their SMEM rings, and the scalar
  index math stay SHARED (empty barriers get `arrive_count=2` — the
  semaphore layer already derives this from `partition_count`,
  semaphore_ir.py:271-311). Everything from Q/QK down duplicates: QK MMA,
  the softmax chain, m_i/l_i recurrences, PV MMA, accumulator, epilogue
  stores at row offsets +0/+64.

## What it does to case3's numbers (M-split ×2, from the DDG)

- 31 nodes → ~55 (24 split, 7 shared); 40 edges → ~80; loop-carried
  iter_args 5 → 9 (m_i, l_i, qk token, acc token duplicate; the acc-init
  flag stays shared).
- The split axis MUST be M: `tt.reduce` is axis=1 (the kv axis), so an
  M-split needs no cross-sub-tile combiner; an N-split would need
  max/sum joins and serialize the PV MMAs (WSDataPartition.cpp:1978-1980
  asserts reduce axis ≠ partition dim).
- MMA latency does NOT halve (NVLatencyModel keys it on K only); MMA
  occupancy halves. Net: ResMII ~unchanged, RecMII 1325 → ~942 (the acc
  circuit's tmem_load/mulf/tmem_store halve, the 559 MMA back edge does
  not), MinII 1459 → ~1198 (CUDA-bound). **The payoff is schedulability,
  not MinII**: two independent ~1450-cycle half-chains can overlap each
  other's TC windows, which one indivisible ~2900-cycle chain cannot.
- The latency model needs ZERO changes for an IR-level split — it is
  fully shape-driven.

## Preconditions (status)

1. Guard 1 must be off for the sub-tiled graph — with 4 chained TC nodes
   its occupancy=latency inflation would price the split at TC
   ResMII=2918 (strictly worse than unsplit). DONE: the deletion is
   validated for the joint path (TRITON_MODULO_DISABLE_MMA_GUARD;
   665.7 TFLOPS at the unguarded MinII, SolverMigrationNotes step-4
   item 2).
2. A partitioner that discovers cohesion/splits from constraints rather
   than tuned scores. DONE (joint v1/v2).
3. Emission-order legality for re-solved cycles (iter-arg WAR edges).
   DONE.
4. Optional refinement: `getMinWarps` returns 4 for 64-row ops; a
   min_warps=2 refinement would let one 4-warp WG time-share both
   half-chains. Not blocking.

## Existing machinery inventory (none sufficient alone)

- **Pass A.5 DataPartitionPlan** (ModuloSchedulePass.cpp:873-967): M-split
  of the MMA accumulator, but the MMA stays ONE DDG "bundle" node with
  occupancy ×N — the sub-MMAs cannot be placed at different cycles, and
  nothing downstream (softmax) duplicates. Emit-time fan-out only.
- **Pass A.7 epilogue subtile** (TRITON_MODULO_EPILOGUE_SUBTILE): N-split
  of the outer-loop store chain only; node-level subtile fields exist in
  the JSON schema and Python parser but the C++ dumper never emits them
  (ModuloSchedulePass.cpp: "A.7 subtile fields remain a separate
  follow-up").
- **AutoWS SubtiledRegionOp** (GenerateSubtiledRegion.cpp): a pattern
  recognizer + region wrapper for split trees that ALREADY exist in the
  IR downstream of a tmem_load — it does not create splits, cannot split
  MMAs or reduces, and runs far too late (inside NVGPUWarpSpecialization).
  Not reusable as the transform; its structural-equivalence checker is a
  useful reference.
- **WSDataPartition.cpp** (~2600 lines): the real M-split slicing core —
  clones ops per partition with shape[dim]/N types, rewrites TMEM
  encodings (blockM divisibility gates at :154-171, encoding rebuild at
  :1294-1301), threads sliced loop-carried iter_args (:712-717). Its
  driver is welded to AutoWS async-task-IDs and produces per-task copies;
  reuse needs a new driver that inlines both slices in one block.

## Plan: Route A first, graduate to Route B

**Route A — env-gated IR pre-split before the modulo pass** (the cheap
experiment). A new small pass (or pre-step inside ModuloSchedulePass)
reusing WSDataPartition's slicing core with a new single-block driver:
M-split the FA-shaped inner loop ×2 (Q slice, QK MMA, softmax chain,
PV MMA, accumulators, iter_args; K/V loads shared). Downstream blast
radius is ~zero — DDG, latency model, CP-SAT, JSON, emitter are all
shape-driven and just see 55 smaller nodes. Gate:
`TRITON_MODULO_SUBTILE_M=2`. Risks: TTGIR/layout validity (blockM=64
TMEM encodings, warp-locked tcgen05.ld rows) surfaces at compile time;
split-before-solve pre-commitment is mitigated by solving both variants
and letting solver_regression compare (the A.5/A.7 env-flag precedent).

**Route B — DDG-level virtual split** (where this wants to end up: the
solver prices split vs unsplit inside ONE search, no IR risk until
emission). Requires breaking two load-bearing invariants: one node per
Operation* in the DDG builder (opToIdx; edge phases 2/3) and
one-variable-per-op_id memoization in the emitter's inner-loop renderer,
plus a shape-scaled latency query and multi-producer buffer lifetimes.
Materially larger diff concentrated in the two most delicate components.
Do it after Route A has proven the win.

## Acceptance

- Sub-tiled case3 passes correctness under solver_regression, and the
  full solver stack on the sub-tiled graph BEATS the 665-TFLOPS plateau
  at (1,32,8192) — that is the entire point; if it cannot, the design
  point hypothesis is wrong and this note gets a measured post-mortem.
- Unsplit cases (GEMM family, layernorm) are untouched (the transform
  only fires on the FA shape or behind the env flag).
- The solver reproduces the anti-phase structure (QK0/QK1 offset ≈ half
  period) without any ping-pong-specific code — the named-barrier
  recurrence should EMERGE from TC NoOverlap + the flight-window
  constraints, the same way softmax cohesion emerged in step 4 item 2.
