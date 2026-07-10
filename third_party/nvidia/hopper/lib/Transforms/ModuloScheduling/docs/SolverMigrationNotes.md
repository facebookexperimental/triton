# Heuristic guards vs. a future global solver

Status: design note, written 2026-07 alongside the throughput-based latency
model change (PR #1912). Audience: whoever replaces the current heuristic
modulo scheduler (Rau IMS / SMS / exhaustive, dispatched in
`ModuloReservationTable.cpp:runModuloScheduling`) with a global optimal
solver (ILP / CP-SAT, "Twill-style"). Extended 2026-07-05 with a gap
analysis against the Twill paper itself (Soi et al., "Optimal Software
Pipelining and Warp Specialization for Tensor Core GPUs",
arXiv:2512.18134) — see "Gap analysis" below.

## Why this note exists

The 2026-07 latency-model change made TC/TMA `occupancy` throughput-based
instead of latency-based. That is the honest hardware model, and it roughly
halved GEMM II (559 → 256) and deepened the TMA prefetch rings (2 → 3),
worth +11–40% on the GEMM-family cases. But exposing the scheduler to
tighter IIs also exposed how fragile the heuristic search + partitioner +
emitter pipeline is at those IIs. Three *guards* were added to keep the
pipeline robust. Each guard is a workaround for a limitation of the CURRENT
search, not a statement about hardware. A global solver changes the
trade-offs: some guards must become explicit constraints, some must move
into the objective, and some must be deleted or they will actively block
better solutions.

The overarching principle: **an optimal solver exploits model error
maximally**. A heuristic explores few points, so coarse costs are survivable;
a global solver drives the schedule exactly onto the model's boundary, so
every cost the model omits (cross-WG communication, barrier round-trips,
SMEM/register pressure) becomes a direction in which the solver produces
"model-optimal, hardware-catastrophic" schedules. The FA case below is a
measured example.

## Guard 1: serially-dependent MMA occupancy correction ("Phase 2.75")

Where: `DataDependenceGraph.cpp`, Phase 2.75 (between edge construction and
loop-carried edges).

What: if a loop contains ≥2 distinct TC nodes connected by a distance-0
dependence path (FA fwd: QK-MMA → softmax → PV-MMA), restore
`occupancy = latency` for ALL TC nodes in that loop.

Why it exists: with honest throughput occupancy, FA's MinII drops
1459 → 1325 (RecMII-bound). At II=1325 the Rau placement pushes the rowsum
(`tt.reduce`, input = the exp2 P-tile) past the stage boundary
(earliest = exp2_cycle + 570 > stage-1 end), the cluster partitioner then
cuts the exp2→reduce edge across warp groups, and the emitted kernel ferries
a 32KB fp32 P-tile through SMEM every iteration: **measured 5x slower**
(651 → ~100 TFLOPS on (1,32,8192)). FA's real steady-state is
softmax-chain-bound (~1745 cyc/iter), so the lower II had no upside to
begin with. The guard restores ResMII(TC) = 1459 and byte-identical
known-good schedules.

Under a global solver: **delete this guard.** It deliberately inflates
ResMII, which would fence off genuinely better schedules from a solver that
can handle them. The two truths it papers over must be expressed natively:

1. One iteration's dependent MMAs cannot overlap — already implied by
   precedence constraints; nothing extra needed.
2. Cutting a register compute chain across WGs costs a real round-trip
   (~500 cyc handshake + SMEM store/load scaled by bytes, see
   `kCrossWGRoundTripLatency` / `smemMoveCost` in `ModuloSchedulePass.cpp`)
   and SMEM capacity. This must be a term in the solver's OBJECTIVE (or the
   partition must be a joint decision variable), not an afterthought in a
   separate partitioner cost. If the solver schedules and partitions jointly
   with that term, it will discover on its own that keeping exp2+rowsum
   co-resident beats the tighter II.

## Guard 2: II search window `minII + max(10, minII/8)`

Where: `ModuloReservationTable.cpp:runModuloScheduling`.

What: the classic "+10 slack" II window from CPU modulo-scheduling folklore
was too narrow for GPU-scale IIs (hundreds of cycles, multi-hundred-cycle op
durations). Layernorm became CUDA-pipe-bound with ZERO slack at MinII=517
after the TMA-store occupancy fix and failed to schedule anywhere in
[517, 527]; it packs fine at ~534.

Under a global solver: **this code disappears entirely.** An ILP either
solves with II as a variable or sweeps II exactly with feasibility as a
constraint; reservation-table fragmentation is not a failure mode of a
complete search. No replacement needed.

**[DONE for the CP-SAT backend, 2026-07-05, branch
hwu27/modulo-cpsat-backend]** `TRITON_USE_MODULO_SCHEDULE=joint_solver` sweeps II
from minII to the true feasibility bound (critical path + serial work)
with no window; the window now applies to the heuristic paths only. The
guard-2 canary resolved exactly as predicted: CP-SAT schedules layernorm
AT MinII=517 — the perfect CUDA-row packing Rau's incomplete search could
not find anywhere in [517, 527] — with maxStage 3 vs Rau's 4 at II=534.

## Guard 3: outer WS loops forced single-WG

Where: `ModuloSchedulePass.cpp:applyGlobalWarpPartition` (the `sl.isOuter`
early-out).

What: outer persistent loops skip the cost-model partitioner and put every
op in one WG.

Why it exists: this is an EMITTER-capability constraint, not a hardware or
search limitation. sched2tlx only lowers multi-WG bodies for INNER loops;
when the partitioner split case5's outer-loop epilogue
(convert_layout → descriptor_store) into its own TMA WG, the emitter
silently dropped the store ops and the kernel produced NaN. (Silent, because
the barrier decls WERE emitted — only the WG body ops vanished.)

Under a global solver: **keep the restriction but express it as a legality
constraint** ("ops in an outer loop must share one WG" — or, better, fix the
emitter and drop it). Do NOT let it silently vanish during the rewrite: the
solver will otherwise rediscover the split (it looks profitable to any cost
model that undercounts it) and regress correctness, which is worse than
regressing performance. A solver formulation should carry an explicit list
of emitter-lowerable partition shapes as hard constraints, versioned with
the emitter.

## Related but not a guard: lifetime-based channel depth, store-consumer only

`insertCrossGroupBarriers` sizes synthesized cross-WG channels by
`lifetime/II + 1` ONLY when the consumer is an async TMA store; register-
consumer channels stay depth-1. The scoping is empirical: deepening FA's
P-tile bridge (register consumer) measured **6x slower**, while the same rule
on layernorm's compute→store channel is what removes the store-drain
serialization. A solver that owns buffer depths as decision variables (under
the SMEM budget, as `ExhaustiveScheduler`'s joint memory feasibility already
sketches) should replace both the rule and its scoping — but it must price
the register-consumer case's REAL cost (extra SMEM + barrier traffic inside
a latency-critical chain) or it will re-deepen the P-tile bridge and
re-learn the 6x the hard way. There is unexplained microarchitecture here;
treat the FA depth-2 slowdown as a benchmark to reproduce before trusting
any model of it.

## What carries over unchanged

The per-op calibration layer (`NVLatencyModel.cpp`: TMA load/store tables,
MMA latency + MACs/cycle throughput, CUDA/SFU/reduce costs — all
B200-measured) is solver-independent and transfers as-is. It is also mostly
piecewise-linear in op attributes, so it embeds directly into an ILP. What
a global solver ADDS to the model's burden is the second-order terms:
synchronization latency, channel SMEM cost, register-pressure-driven WG
sizing (`minWarps`, `wgFootprint`), and wave quantization — today these live
in the partitioner's `scoreCandidate`; in a joint formulation they belong in
the objective.

## Validation harness

Any solver rewrite should reproduce, on B200:
- case1/case5/case7: II 256/256/256, SMEM rings depth 3, gen/hw ≥ the
  2026-07 numbers (0.89–1.14x / 1.13–1.40x / 0.65–0.91x).
- case3 FA: ≥ 651 TFLOPS on (1,32,8192) — this is the guard-1 regression
  canary; anything that cuts the softmax chain across WGs fails it.
- case6 layernorm: schedules at all (guard-2 canary), store channel depth 2.
- case5: non-NaN output (guard-3 canary).
`examples/testing/run_regression.py` automates the correctness+perf sweep.

## Gap analysis: current pipeline vs. Twill (arXiv:2512.18134)

Twill (Soi, Yadav et al.) formulates SWP and WS as ONE optimization problem:
an ILP finds an optimal modulo schedule (M, I), the schedule is unrolled into
a straight-line program Q (prologue + one steady-state copy + epilogue), and
an SMT solver then jointly re-derives the schedule `op[v,i,t]` AND the warp
assignment `opw[v,w]` under memory, register, cross-warp-spill and
blocking-sync constraints. On FA fwd it rediscovers FA3 (Hopper ping-pong)
and FA4 (Blackwell 3-WG rescale split) from first principles; solve times are
19–269 s.

Our pipeline is the opposite shape — "schedule first, partition second, patch
third": DDG → MinII → Rau IMS heuristic (`ModuloReservationTable.cpp`) →
cluster partitioner scoring a frozen schedule
(`ModuloSchedulePass.cpp:scoreCandidate`) → `insertCrossGroupBarriers` →
sched2tlx emitter. The FA softmax-cut 5x regression (guard 1) is a measured
instance of exactly the failure mode Twill's joint formulation exists to
prevent: the scheduler picked a tighter II that was only realizable by
cutting a register chain, and the cost of the cut was invisible to it. The
gaps, roughly ordered by structural depth:

### 1. Schedule and WS partition are solved separately (the big one)

Twill's CROSS-WARPSPILLS constraint puts the reg→SMEM→reg round-trip
directly into the dependence delay of the joint problem; its CONCURRENCY
constraint models blocking waits interrupting same-warp issue. Our
equivalents (`kCrossWGRoundTripLatency`, `smemMoveCost`,
`computeWGBarrierCost`) live in the partitioner's after-the-fact score,
where they can only de-rank bad partitions of an already-fixed schedule.
Guard 1's migration plan ("delete the guard, price the cut in the
objective") is the first step of this; the end state is warp assignment as
a decision variable co-solved with placement.

### 2. Heuristic search → complete solver

Rau IMS + ejection + a bounded II window is an incomplete search; guard 2
(the window) exists only because of that incompleteness and disappears under
any complete method. `ExhaustiveScheduler.cpp` (branch-and-bound + joint
SMEM/TMEM feasibility, practical to ~20 ops) is the in-tree seed: its buffer
extraction and feasibility check are exactly the constraints a CP-SAT/ILP
backend needs. Twill uses CBC for the ILP and Yices2 (QF_LIA) for the joint
problem; OR-Tools CP-SAT is likely the better modern default.

### 3. Cost normalization is a tractability PREREQUISITE

Optimal modulo scheduling is exponential in the sum of edge delays, not just
|G|. Feeding raw calibrated costs (hundreds–thousands of cycles) to a solver
is intractable. Twill §5.2 solves a small side-ZLP: find integer costs C'
minimizing ratio distortion vs. C, with sum(C') ≤ U (they use U=300; SCIP,
<500 ms). Note the per-op calibration in `NVLatencyModel.cpp` transfers
unchanged — normalization is a layer on top, applied at solver-input time.
Side benefit: our per-cycle reservation table currently pretends a
resolution the model doesn't actually have; normalized costs are more
honest about that.

### 4. Scalar occupancy → resource reservation tables (RRTs)

`ModuloReservationTable` gives each pipeline one row with exclusive interval
occupancy. It cannot express pipelined units (issue throughput ≠ completion
latency), multi-instance resources (cap(f) > 1), or ops touching different
units in different cycles of their execution. The 2026-07
`occupancy`/`selfLatency` split is a two-field approximation of a pipelined
unit; an RRT (per-cycle usage vector per op + per-unit capacity, Twill §3.1)
expresses it natively and would subsume the split.

### 5. Memory and register pressure as scheduling constraints

Twill encodes liveness (`live[v,i,t]`, backward-dataflow rules as SMT
implications) and enforces MEMORYCAPACITY per memory and REGISTERLIMIT per
warp inside the joint problem. Our main path (Rau IMS) is memory-blind;
only the exhaustive branch checks budgets, and only post-placement.
Two concrete consequences for the rewrite:
- Buffer/channel depths become decision variables under the SMEM budget
  (replacing the `lifetime/II+1` store-consumer-only rule — see the channel
  depth section above, including the unexplained 6x register-consumer
  hazard that must be priced first).
- Register budget must be a conservative, tunable parameter, not the
  architectural max: Twill's Blackwell-bwd experiment found a schedule
  their own model said fits, yet ptxas spilled badly; re-solving with a
  reduced register budget produced the (faster) FA4-style 3-WG strategy.
  Expect the same ptxas gap here.

### 6. Variable-latency ops: classify, don't average

Twill §5.3 splits variable-latency ops in two: *streaming* ops (no incoming
deps, e.g. TMA input loads) get latency 0 in the cost model, live on a
dedicated warp, and their pipeline depth is exposed as an autotuning
parameter outside the solver; dependent variable-latency ops are forced
onto the designated VL warp (VARIABLELATENCY constraint). We currently give
TMA loads one calibrated fixed latency and derive depth from lifetime/II.
Adopting the streaming classification shrinks the solve space AND removes
the worst source of static-latency error from the schedule.

### 7. Blocking-sync issue interference, modeled not penalized

The `TRITON_MODULO_STAGE_SEPARATION` min(syncWork, asyncWork) penalty is a
self-declared workaround for the Fig.-2 problem in the paper (a blocking
wait on one op stalls issue of independent ops on the same warp). Twill's
CONCURRENCY constraint is the principled version: designate blocking edges;
forbid co-scheduling other ops on the consumer's warp across the wait. In a
joint formulation the penalty is deleted and the constraint does its job.

### 8. Scheduling granularity: sub-tiling

Easy to miss but load-bearing: Twill reports the joint problem is UNSAT on
the un-sub-tiled tutorial FA — FA3 ping-pong and FA4's two softmax groups
require splitting one tile-level op into two sub-tile instances that
interleave. Our DDG nodes are whole TTGIR ops, so ping-pong-class schedules
are not merely unreached by the heuristic, they are ABSENT from the solution
space. A pre-scheduling sub-tiling transform (or reuse of AutoWS subtile
machinery) is required before a solver can find them.

### 9. Emitter legality as explicit constraints

Covered by guard 3 above; restated here because it generalizes: the solver
formulation should carry a versioned whitelist of emitter-lowerable
partition/loop shapes (outer-loop single-WG, blockM≤128 TMEM, ...) as hard
constraints, each removed as the emitter gains the capability — never
encoded implicitly in costs, or the solver will route around it and regress
correctness.

### Suggested sequencing

1. **Now, heuristic unchanged**: keep calibrating the second-order terms in
   `scoreCandidate` (barrier round-trips, channel SMEM, wave quantization) —
   these become objective terms later, no work wasted.
   **[DONE 2026-07-05, branch hwu27/modulo-second-order-calibration]**
   Measured with a new in-tree microbenchmark
   (`third_party/tlx/tools/microbench/cross_wg_handoff.py`, tlx.clock64
   ping-pong between two 4-warp WGs on B200): mbarrier one-way handshake
   92 cyc, named-barrier one-way 30 cyc (the old kBarrierOverhead=30 guess
   was exact for named, ~1.5x low for mbarrier), reg→SMEM→reg hand-off
   61 + 16.1 cyc/KB. The old decomposition was wrong in both directions —
   kCrossWGRoundTripLatency=500 (now 150) had absorbed a 5-10x underpriced
   move slope (105/16384-elems, now 16 cyc/KB byte-based); the totals
   happened to agree near the one 32KB FA point the 500 was fitted on.
   Also added: scoring-time channel-SMEM capacity prediction
   (`predictChannelSmemBytes` replays `insertCrossGroupBarriers`; over-
   budget → kInfeasiblePenalty since channels were never budget-checked
   and failed only at kernel load), and the CTA co-residency bound
   (computed + logged; penalty defaults to 0 via TRITON_MODULO_CORES_PENALTY
   because persistent grids and maxnreg auto-fill pin co-residency at 1 on
   the current suite — see the constant's comment). All 5 regenerable cases
   emit byte-identical generated.py; canaries hold (case3 650 TFLOPS ≈ 651
   within do_bench noise, kernel unchanged).
2. **Mid**: grow `ExhaustiveScheduler` into a CP-SAT backend for schedule +
   buffer depths only (partitioner stays), with cost normalization. This
   alone deletes guard 2.
   **[DONE 2026-07-05, branch hwu27/modulo-cpsat-backend]**
   `TRITON_USE_MODULO_SCHEDULE=joint_solver` (opt-in; default path unchanged):
   `JointSolverScheduler.cpp` serializes the DDG + buffers and drives an OR-Tools
   CP-SAT solver in a Python subprocess
   (`python/triton/tools/modulo_joint_solver.py`, same subprocess pattern as
   LLMSchedulePass), then RE-VERIFIES the returned schedule in C++ — the
   solver is advisory, never trusted. Model: modular NoOverlap per pipeline
   (selfLatency reservations, wrap-around split), dependence + def-before-
   use, SMEM depth budget decided JOINTLY with the schedule, TMEM as exact
   cumulative (vs the exhaustive's conservative coloring), objective =
   ExhaustiveScheduler's score. Rau runs first as a warm-start incumbent
   hint. Cost normalization (Twill §5.2, side CP-SAT ILP, U=300 via
   TRITON_MODULO_JOINT_SOLVER_NORMALIZE) seeds the search — with interval
   encoding it is an accelerator, not a tractability requirement (that
   requirement is specific to time-indexed formulations).
   Validation: case1/5/7 reproduce II=256 (+8192 outer) with identical
   structure; case3 II=1459; case6 II=517 < Rau's 534 (guard-2 canary,
   see above); all correctness PASS.
   **Measured first-principle instance (KEEP THIS IN MIND FOR STEP 3):**
   at equal II=1459, CP-SAT's model-optimal placement reorders case3 FA's
   softmax rowsum after the P-tile trunc/store (the regPressure objective
   strictly prefers it; a linear earliest-placement term cannot fix this —
   it systematically prefers short-op-first on a serialized pipeline row,
   639-cycle reduce vs 105-cycle trunc) and measures 598 vs 651 TFLOPS at
   (1,32,8192) — an 8% loss the model cannot see. The recurrence-chain
   criticality of the rowsum is exactly the kind of term the joint
   formulation's objective must price before the solver is allowed to own
   these decisions. Until then: joint_solver is opt-in and the case3 canary gates
   any default flip.
3. **Long**: joint schedule+partition formulation (deletes guard 1, converts
   guard 3 to constraints), sub-tiling in front, streaming depths handed to
   the autotuner. Position it as an offline/autotune tool, not the default
   JIT path — Twill's own solve times (tens of seconds to minutes) imply
   the same.
   **[STARTED 2026-07-05, branch hwu27/modulo-cpsat-joint — two pieces
   landed]**
   (a) *Recurrence-chain criticality priced in the schedule objective*:
   for each loop-carried back edge, the forward-chain span
   `cycle[src] − cycle[dst]` is minimized at 8x the regPressure weight —
   compressing recurrence circuits so model-underestimated chains (FA's
   softmax: ~1745 real vs 1459 modeled cyc/iter) degrade the realized
   iteration time as little as possible. On case3 this recovered the step-2
   gap from 598 to 639.5 TFLOPS (canary 651). The remaining ~1.8% is a
   SCHEDULE-side blind spot no schedule-only term can fix: the alpha
   hand-off's urgency (its consumer sits in another WG) is invisible
   before a partition exists — the precise motivation for (b) and for
   joint cycles+wg (v2).
   (b) *Joint partition v1* (`TRITON_MODULO_CPSAT_JOINT=1`, opt-in;
   `partitionJointSolver` → `solve_partition` in modulo_joint_solver.py): warp
   assignment solved by CP-SAT against the committed schedule (cycles
   fixed — Twill's re-solve shape), replacing scoreCandidate's
   enumerate-and-score with constraints over the SAME calibrated costs:
   async-flight-window overlap as the split cost (Twill's CONCURRENCY
   insight — merging blocks sync work scheduled during a TMA/TC flight;
   this term is what makes or breaks the model, issue-only windows
   over-merge everything), slack-aware reg→SMEM→reg round-trips (refines
   accumulateCrossWGRoundTrip, which ignores slack), measured barrier
   issue costs, register footprint table + default-WG slack, channel SMEM
   as a hard capacity constraint. Would-be-demoted scalar arith is
   excluded (its cross-cluster edges are fictional — demotion replicates
   it per WG; serializing it let the solver glue loader WGs together
   through a shared index op).
   **Result: reproduces the committed partitions on ALL five cases —
   byte-identical generated.py — including case3's 6-WG FA structure with
   the softmax chain intact.** The partition that guard 1 and the fitted
   round-trip constant were added to protect now emerges from first
   principles; this is the concrete path to deleting guard 1 once the
   joint mode owns partitioning. Full stack (joint_solver schedule + joint
   partition, 2026-07-06 measurements): correctness passes on all cases;
   on case3 it produces a 5-WG variant (PV-MMA merged into the rescale
   WG) that measures **664.1 TFLOPS at (1,32,8192) — 2.1% ABOVE the 651
   canary** (vs 650 for the hand-protected 6-WG + Rau default, 639 for
   the joint_solver schedule under the heuristic partition). The B < A < C
   ordering is the joint-formulation thesis in one row of data: the
   solver's schedule loses to Rau under the old partition but wins when
   paired with its own matching partition. Small shapes are within noise
   (±5%); case6's II-517 schedule is perf-flat vs the II-534 default
   (memory-bound — the bottleneck is the emitter's 3-WG structure).
   Still open for v2: joint cycles+wg in one solve (fixes the alpha-order
   blind spot), guard-3 as an in-model legality constraint, sub-tiling.

### Step 4 (next, planned 2026-07-06): from v1 to owning the decisions

Agreed priority: **items 1+2 first — they are one unit of work** with
directly measurable expectations (the two open numbers from step 3: does
the joint-cycle solve lift the B config past 639, and is 664 the ceiling
of the C config or just the first point above it). Item 3 rides on
emitter-side staffing; item 4 is the most research-heavy and has the
highest ceiling. Item 5 gates any default flip and can proceed in
parallel.

1. **Step-3 v2: cycles + wg in ONE solve.** Today the joint mode holds
   the schedule fixed (Twill's two-step shape); v2 makes the flight-window
   overlap, slack-aware round-trips, and channel constraints functions of
   cycle VARIABLES instead of precomputed constants, so op order inside a
   WG can react to partition pressure. This is the only fix for the
   alpha-order blind spot (the residual 651-vs-639 gap of the B config):
   the alpha hand-off's urgency exists only relative to a partition.
   Measured targets: case3 B-config ≥ 651 without joint partitioning
   enabled being the excuse; explore whether C's 664 improves further.
   **[2026-07-06 status: infrastructure LANDED
   (TRITON_MODULO_CPSAT_JOINT=2, solve_joint in modulo_joint_solver.py: full
   dependence + per-pipeline and conditional per-WG modular NoOverlap +
   circular-difference flight exclusion + depth-from-stage SMEM + cycle
   write-back with buffer re-derivation and a C++ dependence
   re-verification), but case3 cycle re-solve is BLOCKED on a data-fidelity
   gap: ScheduleLoop edges carry issue-order latencies (20/40 edges are
   0/1, not the DDG's data latencies) and NO anti-dependence (iter-arg
   WAR) edges — the solver can legally reorder version-sensitive chains,
   and the emitter then inlines/duplicates operands with iter-arg version
   skew (measured wrong results; a strict +1 edge separation removed ties
   but not unconnected reordering, confirming the missing-WAR-edge root
   cause). A register-loop-carry ⇒ same-WG LEGALITY constraint was added
   to both joint modes along the way (iter-args have no cross-WG channel
   semantics).]**
   **[UNBLOCKED 2026-07-06 (same day): iter-arg WAR edges are now derived
   from the scf.for at serialization time — every direct reader of a
   block argument gets a strict ordering edge to the yield-operand
   producer (the emitter renders the update as a reassignment at the
   producer's cycle position). With them, ALL v2 configs pass
   correctness: case1/5/7 hint-parity, case6 re-solved cycles PASS,
   case3 PASS on both the Rau-hint path (returns to 6-WG committed
   parity — the 5-WG merge was only objective-profitable together with
   the now-forbidden version-skewed reorder) and the joint_solver-schedule path
   (5 WGs, re-solved cycles, at both guarded II=1459 and unguarded
   II=1325). Measured: 664.9 TFLOPS at (1,32,8192) for both v2
   full-stack configs — the ~665 plateau is now reproduced by THREE
   independent configurations (v1-joint 664.1, v1-joint+no-guard 665.7,
   v2 664.9), strong evidence it is the real softmax-chain bound of this
   5-WG design point rather than a solver artifact. The remaining
   fidelity item (ScheduleLoop edges carry issue-order latencies, not
   DDG data latencies) is now a PERF-modeling refinement, not a
   correctness issue.]**
2. **Delete guard 1 for the joint path.** Once v2 owns partitioning
   jointly with cycles, the Phase 2.75 occupancy inflation
   (DataDependenceGraph.cpp) is a pure solution-space fence for that
   path. Acceptance: remove it under TRITON_MODULO_CPSAT_JOINT, verify
   FA's softmax cohesion still emerges from the round-trip/flight-window
   constraints at the LOWER unguarded MinII (1325), and the case3 canary
   holds. Keep the guard for the heuristic paths (they still need it).
   **[DONE 2026-07-06 — VALIDATED, did not need v2:**
   TRITON_MODULO_DISABLE_MMA_GUARD=1 bypasses Phase 2.75; with the joint_solver
   schedule + the v1 joint partitioner, case3 solves at the unguarded
   MinII=1325, the partition keeps the softmax chain co-resident (5 WGs,
   exp2+rowsum together — the cut guard 1 existed to prevent does NOT
   happen), correctness passes, and it measures **665.7 TFLOPS at
   (1,32,8192) — the best configuration yet** (vs 664.1 guarded full
   stack, 650 default, 651 canary). The guard is now provably redundant
   for the joint path; actual code deletion should land together with
   making joint partitioning that path's default. The env bypass stays
   for A/B until then.]**
3. **Guard 3 → versioned emitter-capability constraints.** Express
   "outer loops single-WG" and "blockM ≤ 128 TMEM" as explicit legality
   constraints in the partition model, versioned with the emitter; fix
   the emitter's blockM>128 TMEM gap to unblock case2 end-to-end (the
   solver side is already verified healthy on its 256-blockM IR: II=1024,
   sane 3-WG).
   **[PARTIAL 2026-07-06: the constraint side is DONE — `EmitterCaps`
   (versioned, kVersion=1) in ModuloSchedulePass.cpp carries
   kOuterLoopSingleWG (referenced by the isOuter early-out and plumbed as
   `max_wgs` into both joint solver modes) and kMaxTMEMBlockM=128; the
   emitter now raises a clear NotImplementedError naming the capability
   for blockM>128 TMEM accumulators on all three of its TMEM render paths
   (case2's trap is a named error instead of a silently-trapping kernel;
   committed cases emit byte-identically). REMAINING: the actual
   MMA-splitting emitter feature — staffing-dependent, unchanged.]**
4. **Sub-tiling in front of the solver** (gap 8 above): without
   splitting tile-level ops into sub-tile instances, ping-pong/FA4-class
   schedules are absent from the solution space entirely. Likely reuses
   AutoWS subtile machinery; the largest expressiveness jump and the
   least scoped item — treat as its own design note when started.
   **[DESIGN NOTE DONE 2026-07-06: docs/SubTilingDesign.md.** Key
   corrections from the scout: the AutoWS SubtiledRegionOp is a pattern
   recognizer for already-split epilogue chains, NOT a splitter — the
   reusable slicing core is WSDataPartition.cpp (M-split of MMA/TMEM/
   reduce/elementwise/iter_args, needs a new non-task-ID driver). Plan:
   Route A (env-gated IR pre-split, TRITON_MODULO_SUBTILE_M=2, ~zero
   downstream blast radius since everything is shape-driven) then Route B
   (DDG-level virtual split, prices split-vs-unsplit in one solve, but
   breaks the one-node-per-Operation* and one-var-per-op_id invariants).
   All preconditions are met by items 1+2 (guard-1 off is REQUIRED —
   with 4 chained TC nodes the guard would price the split at ResMII
   2918, strictly worse). M-split only (reduce is axis-1); MinII
   1459→~1198 but the real payoff is schedulability. Acceptance: beat
   the 665 plateau on case3, anti-phase structure EMERGES from the joint
   constraints without ping-pong-specific code.]**
   **[ROUTE A EXECUTED 2026-07-06 — measured post-mortem in
   SubTilingDesign.md "Experiment log"; assets in
   `sched2tlx/examples/testing/subtiling/`. End-to-end sub-tiled solver
   kernel runs CORRECTLY after hand-patching three emitter defect
   classes (singleton cross-region channels, per-WG-deduped arrive
   counts vs per-MMA hardware arrives, dropped M_lse epilogue stores —
   the hand-patch diff is the emitter spec). Partition side of the
   emergence criterion met — CORRECTED: the 6-WG ping-pong-shaped
   partition is the HEURISTIC partitioner's own output (verified
   byte-identical; the winning kernel is Rau + heuristic end to end);
   joint-v1 diverges on this graph to an unmeasured 5-WG
   single-MMA-WG variant. As emitted: 206.7 TFLOPS.]**
   **[ROUTE A ACCEPTANCE MET same day — 703–720 TFLOPS at (1,32,8192),
   ABOVE the 665–666 single-tile plateau. The clock64 replay
   (SubTilingDesign.md "Where the 11.3K cycles actually went") showed
   the dominant cost was REGISTER SPILLS, not TMEM contention: the
   schedule hoisted each softmax WG's 128-reg acc tmem_load ~5000 cyc
   ahead of its use; sinking it to the use site is worth ×2.7 alone
   (206.7→557), P-bridge depth 2 and de-instrumentation the rest.
   Supporting calibration: tmem_port_contention.py microbench (TMEM
   round-trip 41 cyc + 4.2 cyc/KB; dual-WG interference ×1.33–1.48 —
   real but second-order) and the 1197-TFLOPS tutorial dissection
   (dedicated correction WG, ballot-gated rescale skip, in-order
   single-MMA-WG anti-phase, TMEM aliasing). Work items re-ranked in
   SubTilingDesign.md: (1) load-at-use emitter rule / register-liveness
   in model, (2) emitter multi-instance generalization, (3) shared
   engines, (4) tutorial-structure items toward 1197.]**
5. **Default-flip gate (parallelizable):** a kernel corpus beyond
   case1-7, solve-time budget policy (keep the offline/autotune
   positioning), solver configs wired into run_regression.py, and the
   case3 canary as the hard gate.
   **[FIRST CUT DONE 2026-07-06:
   `examples/testing/solver_regression.py` — for each solver config
   (joint_solver / joint1 / joint2 / full / full-noguard) × regenerable case:
   regenerate → emit → byte-parity fast path (identical codegen skips the
   GPU) → correctness + per-shape perf vs the committed kernel
   (--perf-tol) → the case3 canary as a hard absolute gate
   (--canary-tflops, default 651). Validated end-to-end (joint1 all
   parity; full-noguard case3 canary 664/651 OK). Full per-config,
   per-shape speedup tables: docs/SolverConfigMeasurements.md
   (2026-07-06). REMAINING: broader
   kernel corpus, CI wiring.]**

The validation harness above is the gate for every step, and the
"Why this note exists" principle is the order of operations: a complete
solver exploits model error maximally, so model calibration must land
BEFORE each increment of solver power, not after. (`LLMSchedulePass.cpp`,
the experimental Claude-API scheduler, shares the same DDG serialization
and should reuse the same canaries.)

## 2026-07-09: the joint solver becomes its own pass

The joint stack was promoted from env-var hooks inside the modulo pass to
a sibling pass, `JointSolverSchedulePass.cpp` (`nvgpu-joint-solver-schedule`,
`TRITON_USE_JOINT_SCHEDULE=1` from Python). Both passes are thin shells
over one shared driver (`ModuloScheduleDriver.h` / `runScheduleDriver` in
`ModuloSchedulePass.cpp`), so the annotation contract downstream consumes
is identical by construction — switching scheduler is purely a
pass-selection decision in compiler.py, and the two coexist for A/B at any
time.

Mapping from the retired knobs:

- `TRITON_MODULO_CPSAT_JOINT=2` → joint pass default (`joint-mode=0`,
  v2 → v1 fallback chain);
- `TRITON_MODULO_CPSAT_JOINT=1` → `joint-mode=1` (v1 only);
- `joint-mode=2` (strict v2, no fallback) is new;
- the joint pass always forces the joint_solver schedule backend, so the old
  rau-schedule + joint-partition combinations no longer exist;
- `TRITON_MODULO_DISABLE_MMA_GUARD` was retired with guard 1's deletion.

The modulo pass (`TRITON_USE_MODULO_SCHEDULE`) is back to pure heuristic
scheduling + heuristic partition; its `=joint_solver` backend value was left as
a schedule-only middle ground (joint_solver schedule + heuristic partition) —
UNTIL the 2026-07-10 promotion entry below: the round-1 measurements
showed that combination is a trap, and the driver now pairs every joint_solver
schedule with the CP-SAT v1 partition. `solver_regression.py` configs
updated to the pass-based selection (joint_solver / joint1 / joint2 / full).

## 2026-07-09 (second entry): Twill-gap implementation round 1

Landed (validated by `solver_regression.py`, now 6 cases × pass-based
configs including `full-stream` / `full-regcap`):

1. **Emitter tensor-capture gap closed (gap 9, first production instance).**
   case4 FA-bwd's v2 partition emitted a function-scope `tl.arange` into a
   non-default task (`WarpSpecializeOp should not capture RankedTensorType`).
   Root cause chain: `_localize_captured_reg_tensors`' graph pre-walk
   stopped at IV-dependent ops, but prologue/skew render paths inline past
   channel-bound nodes into IV-dependent expressions whose IV-INVARIANT
   sub-tensors still get referenced. Fix in two layers (emitter.py): the
   pre-walk now stops at schedule-BOUND nodes instead of IV-dependent ops
   (matches what rendering actually inlines), plus
   `_localize_rendered_captures` — a render-level safety net that scans each
   emitted non-default task body for function-scope register-tensor names
   (exactly the property the TTIR verifier enforces) and re-materializes any
   found. Committed-kernel parity stays byte-exact on all six cases.
2. **case4 FA-bwd wired into the regression suite** (correctness-only).
   It immediately exposed the NEXT v2 blocker: with the capture fixed, the
   v2 kernel compiles and runs but computes wrong dQ/dK (dV correct — the
   corruption is exactly the D-load chain, whose address add the partition
   moved cross-WG). This is another data-fidelity instance of the step-4
   "operand inlining and version skew" class: prologue inlines version 0
   while the channel delivers later versions; some version constraint is
   still missing from the joint problem. case4 stays a hard FAIL for the
   `full` config until fixed — that is the canary working as designed.
3. **Streaming variable-latency classification (gap 6, Twill §5.3),
   opt-in `TRITON_MODULO_STREAMING_VL=1`**: TMA-pipeline DDG nodes with no
   incoming distance-0 edge solve with latency-0 outgoing edges (ring
   absorbs the latency); ring depth remains a solver decision (the
   objective already rewards depth). C++ verifySolution applies the same
   effective-latency rule so streaming schedules survive re-verification.
4. **Hard register budget (gap 5, Twill REGISTERLIMIT-lite), opt-in
   `TRITON_MODULO_REG_BUDGET=<regs>`**: hard `total_regs ≤ budget` in both
   v1 and v2 partition solves (the soft-deficit model remains the default).
   Motivated by case4 v2 oversubscribing 91136 > 65536 and getting
   emitter-downscaled 152→96 regs — Twill's Blackwell finding
   (model-fits-but-ptxas-spills → re-solve at reduced budget) reproduced.

### Landing design: RRTs (gap 4, not yet implemented)

Replace `ModuloReservationTable`'s one-row-per-pipeline exclusivity with
per-op reservation vectors: `RRT[v] : cycle → (unit, instances)` rows,
plus a machine description `cap(unit)`. Concretely: (a) extend
`LatencyModel` with `getRRT(op)` defaulting to today's
`(pipeline, selfLatency)` single-row shape so all call sites migrate
incrementally; (b) `ModuloReservationTable::reserve/isFree` become
per-unit counting (capacity, not exclusivity); (c) the CP-SAT model swaps
per-pipeline NoOverlap for per-unit `AddCumulative` over modular
intervals — the wrap-around split machinery is reusable unchanged;
(d) calibration: start with TC issue-vs-completion split (the
occupancy/selfLatency pair becomes a 2-row RRT), then TMA queue depth.
Gate: byte-parity on all six cases with the default single-row RRTs, then
canary-measured II changes as real RRTs land per unit.

### Landing design: sub-tiling in the solution space (gap 8, not yet implemented)

The Route A experiment (SubTilingDesign.md, 703-720 TFLOPS hand-patched)
fixes the target; the missing piece is a pre-scheduling DDG transform, not
emitter work: (a) a `SubTileTransform` that splits an eligible tile-level
node (MMA or softmax chain member) into K sub-instances with derived
latencies/buffers and inter-instance edges, applied BEFORE DDG→solver
serialization behind `TRITON_MODULO_SUBTILE=<K>`; (b) solver sees the
sub-instances as ordinary nodes — no model change needed (this is exactly
how Twill reaches FA3/FA4: their joint problem is UNSAT on un-sub-tiled
FA); (c) the emitter already carries the A.7 subtile fields on
ScheduleNode (`subtile_count`), so emission reuses the Route A machinery;
(d) canary: case3 FA fwd at (1,32,8192) must beat the 665 plateau to
justify default-on, with the 1197-TFLOPS tutorial dissection as the
ceiling reference.

### 2026-07-09 round-1 measurements (B200, 4 configs × 6 cases)

- `full` (joint default): unchanged and healthy — case3 canary 662.4/651
  OK, case6 1.00x, case1/7 byte-parity, case4 the known v2 data-fidelity
  FAIL (the canary this round wired in).
- `full-stream` (streaming VL): IIs unchanged (256/1325/517 — this corpus
  is recurrence/resource-bound, not TMA-latency-bound), case1/7 parity
  kept, case3 662.4 OK, case6 1.00x. Safe; its value case is a kernel
  whose II is actually inflated by TMA latency — none in the suite yet.
- `full-regcap` (hard 65536 cap): case3 REGRESSES to 348.3 (1.88x) — the
  winning 662-TF partition exceeds the COUNT-BASED footprint table's
  estimate, so the hard cap forbids it, while ptxas-reality ran it fine.
  Lesson recorded: hard caps are only as good as the footprint model —
  liveness-grade REGISTERLIMIT (real per-value register footprints over
  cycle variables) is a PREREQUISITE for hard-cap default, exactly the
  RRT-adjacent calibration this doc already sequences. On case4 the capped
  partition produced a HUNG kernel (killed at 10 min) — yet another
  emitter/solver contract instance for the case4 canary pile.
- **NEW COVERAGE FINDING — `joint_solver` (schedule-only middle ground) is now a
  trap on FA**: case3 at the unguarded MinII=1325 with the HEURISTIC
  partitioner drops to 292.9 TF (2.23x). Guard 1's deletion was validated
  for the joint path ("deletion should land together with making joint
  partitioning that path's default" — #1917) but landed for every path;
  schedule-only joint_solver + heuristic partition is exactly the uncovered
  combination, reproducing the softmax-cut failure class the guard used to
  prevent. Options: route the joint_solver backend's partition through the joint
  solver too, restore a guard-1-shaped fence for the non-joint path only,
  or retire the `=joint_solver` middle ground once the joint pass is default.
  **RESOLVED 2026-07-10**: option 1 landed — see the schedule/partition
  coupling entry below (case3 back to 663.6, canary OK).

## 2026-07-10: case4 v2 data-fidelity root cause + stage-invariance gate

Root cause of the case4 FA-bwd wrong-dQ/dK result (v2 accepted, all
latency checks green, GPU silently wrong): the emitter's version
discipline — iter-arg threading, ring phases, and the inline
re-materialization of IV-dependent address chains — is derived from the
INCUMBENT schedule's stage assignment at ScheduleGraph-build time. The v2
writeback re-derives buffer liveness from the new cycles but NOT that
version structure, so any node that changes STAGE reads/writes values one
iteration off in the emitted pipeline. Channel bisection confirmed the
breadth: bypassing the suspect sem6_b10 channel with a local reload still
failed (dQ=2.17), i.e. not a single-channel bug but a whole-schedule
version-alignment bug.

Fix (compile-time rejection, "direction 3"): a stage-invariance gate in
partitionJointSolver's v2 acceptance path, after the dependence-latency
safety net and before the cycle writeback — any solution moving a node to
a different stage than the incumbent is rejected and the loop falls back
to v1 (cycles fixed, version-safe by construction). Same-stage cycle
refinements still land.

Verified on B200 (2026-07-10): `full` and `full-stream` × case1-7 ALL
GREEN — case4 correctness now PASSES (gate fires: "v2 rejected: N4 stage
change 0 -> 1", v1 partition applied); case3 canary 662.0-662.4/651 OK
with NO gate rejection (its v2 solution is same-stage — no over-reject);
case6 1.00x; case1/7 byte-parity.

Follow-up (planned, not landed): relax whole-schedule stage invariance to
per-channelized-edge stage/distance consistency once the emitter can
re-derive version structure for stage-moved nodes — that is the real
unlock for stage-changing v2 solutions, and it subsumes the gate.

## 2026-07-10 (second entry): schedule/partition coupling — joint_solver promotes to V1Only

Resolution of the round-1 coverage finding above ("`joint_solver` is now a trap
on FA"), option 1: `runScheduleDriver` promotes `JointSolverMode::Off` to
`V1Only` whenever the active schedule backend is `joint_solver`. A CP-SAT
schedule presses II to the proven minimum; the heuristic partitioners
were shaped by Rau-conservative schedules and mis-partition those (case3:
292.9 TF, 2.23x). CP-SAT schedules therefore always get the CP-SAT v1
partition — `TRITON_USE_MODULO_SCHEDULE=joint_solver` is now "joint pass minus
v2" rather than a schedule-only middle ground, and `Off` keeps meaning
"heuristic partition for heuristic schedules". Verified on B200: config
`joint_solver` case3 canary 663.6/651 OK (was 292.9 MISS), case4/5/6 clean,
case1/7 parity.

Companion fix in `applyGlobalWarpPartition`: the
`TRITON_MODULO_EXHAUSTIVE_PARTITION=0` greedy escape used to bypass the
joint chain entirely — silently disabling the CP-SAT partition for the
joint pass AND for this promotion (re-creating the very trap). The env
var now only selects which heuristic (greedy vs exhaustive scorer) serves
as the joint chain's fallback; it disables the joint solve for no one.

Doc debt paid with it: the 2026-07-09 pass-split entry's "=joint_solver remains
schedule-only" sentence, `solver_regression.py`'s `joint_solver` config comment,
and `SolverConfigMeasurements.md`'s joint_solver table (its 638.0 alpha-order
numbers were measured with the heuristic partition; superseded by the
v1-partition pairing).

### 2026-07-10 addendum: case4 intermittent correctness FAIL — investigation log

One `full`-config case4 correctness FAIL observed (1 in 24+ draws), not
yet reproduced. What the investigation ESTABLISHED:

- The CP-SAT schedule solve is nondeterministic run-to-run (wall-clock
  time limit + portfolio search): 8 consecutive draws produced 8 distinct
  cycle layouts at the same II=1033 — mostly rigid translations of each
  other (+345 etc.) plus small local perturbations. The v1 WG partition
  TOPOLOGY was identical across all draws; v2 was gate-rejected every
  time. So the varying dimension reaching the emitter is the cycle/stage
  structure of the schedule itself.
- **Emitter is translation-robust (tested)**: `TRITON_MODULO_SCHED_SHIFT=k`
  (new debug knob in JointSolverScheduler.cpp) rigidly translates the solution
  before the native re-verification/liveness pipeline; sweeping k over
  {0,86,...,1032} — including shifts that change max_stage 3→4 — passed
  correctness 13/13. Global stage-split choice is NOT the failing
  dimension.
- **Exhaustive-fallback partition on a joint_solver schedule passed (tested)**:
  forcing the v1 partition subprocess to fail (wrapper rejecting
  "mode"-carrying problems) exercised partitionExhaustive on a
  MinII-aggressive joint_solver schedule — case4 numerics correct. The
  load-induced-fallback hypothesis did not reproduce either.
- Remaining suspects: a rare LOCAL schedule structure (non-translation)
  that the emitter mis-lowers, or a transient environment fault. The
  regression harness now FREEZES failing repros (schedule graph +
  generated.py + output in a persistent solver-regression-fail-* dir) —
  the frozen graph replays deterministically through sched2tlx, so the
  next natural occurrence is caught with evidence instead of vanishing.

## 2026-07-10 (third entry): case4 flake ROOT-CAUSED — Step 4.5 merge aliasing had no hardware ordering

The intermittent case4 FAIL was caught (24-run hunt, run 7: dQ/dK/dV all
~1e3 wrong, deterministic replay from the frozen graph) and bisected by
graph hybridization: importing ONLY the buffer merge_group_id fields from
the failing draw into the passing draw reproduces the failure bit-exactly
(and the inverse repairs it). The schedule was irrelevant — node-schedule
hybrids all passed.

Mechanism: Step 4.5 merges same-shape SMEM channel buffers with
cycle-disjoint lifetimes into one physical allocation (`reuse=` in the
emitted TLX). But cycle-disjointness is a MODEL property — on hardware
only barriers order anything. In the passing draws the merged pair
happened to be chained by a same-WG data dependence (offset-channel
consumer feeds D-load producer), so program order saved it; the failing
draw merged the two D-LOAD channels (producers in wg2 and wg3, no
dependence path), so wg3's local_store raced wg5's read of the aliased
bytes. Whether solver nondeterminism produced a safe or unsafe merge was
pure lifetime-layout luck — this was never a schedule-legality bug.

Emitter fix (sched2tlx/emitter.py):
1. `_alias_group_safety`: an SMEM merge group is guardable iff every
   member is a forward cross-WG channel with a SW producer. Unguardable
   multi-member groups (HW/TMA producer, signal-only member, non-channel
   member) have their `reuse=` DROPPED at alloc — private bytes are always
   correct, merely less thrifty.
2. `_alias_wait_stmts`: guardable groups get real ordering. The group is
   one physical slot with k producer/consumer pairs per iteration in
   lifetime order (p1→c1→…→pk→ck→p1@next). A later member's producer
   waits each earlier member's empty at the CONSUMER phase (this
   iteration's reader drained); the earliest member's producer waits every
   other member's empty at the PRODUCER phase (previous iteration's
   readers; passes immediately at iter 0). Acyclic within an iteration,
   ring-closed across iterations — deadlock-free by construction.
3. The legacy `_alias_predecessors` Layer 2 (pre-SemIR path) turned out to
   be DEAD CODE on the SemIR default path — the SW producer emission never
   called it, which is why no alias wait ever appeared in any kernel.

Verified: the frozen failing graph and the MG1 hybrid (its distilled
minimal repro) both PASS with the fixed emitter, alias waits landing
exactly at the two D-load channel producers.

Note the relationship to the stage-invariance gate (first 2026-07-10
entry): stage moves change buffer liveness, which changes MERGE decisions
— some of the v2 rejections may have been this same alias bug wearing a
different hat. Worth re-testing v2 with the gate relaxed now that aliasing
is ordered; the gate stays until that experiment is run (the original v2
failure signature — dQ=2.17 with dV clean — differs from the alias
signature, so the version-structure concern may still be real).

## 2026-07-10 (fourth entry): v2 "version structure" bug ROOT-CAUSED — shared ring counter vs heterogeneous depths

The second case4 failure mode (the ORIGINAL v2 signature: dQ≈2.17,
dK≈1.67, dV clean) was caught as a v2-ACCEPTED draw (all stages match the
incumbent — the stage-invariance gate passed it; 8-WG joint partition) and
root-caused via a 3-lens analysis workflow + a two-line kernel-patch
discrimination test.

Mechanism — an EMITTER indexing bug, not a scheduling-semantics one: the
inner-loop body emits ONE shared ring counter per WG (`buf = _it %
rep_depth`, rep_depth = max count over the WG's ring buffers) and the MMA
operand renderer subscripts EVERY count>1 buffer with that literal `buf`.
When a WG touches rings of DIFFERING depths, the shallower ring is read at
the wrong slot (and out of bounds): in the failing draw wg4 held L0_smem_0
(count=3) and the dS channel L0_smem_3 (count=2) — consumers read
dS[_it%3] while the producer wrote dS[_it%2] and the barriers tracked %2.
dS = P·(dP−D) feeds dK and dQ; dV never touches it — exactly the
signature. Patching only the two subscripts on the frozen kernel collapsed
dQ/dK to dV's 1e-3 level.

Why this correlated with v2 (and looked like a "version structure"
problem): v1 keeps the incumbent cycles, where the dS ring's lifetime
spans 3 stages → count=3 = rep_depth — the shared counter is
COINCIDENTALLY correct. v2's same-stage cycle compression shortens the dS
lifetime → count=2 ≠ rep_depth=3 → heterogeneous depths in one WG. Any
schedule with mixed ring depths in one WG would trigger it; v2 was just
the only producer of such schedules in this corpus.

Fix (sched2tlx/emitter.py): `_ring_exprs(count, rctx)` — subscript each
ring buffer with its OWN modulus (`_it % count`, phase `(_it // count) &
1`) whenever its count differs from the WG's rep_depth; buffers matching
rep_depth keep the shared `buf`/`phase` (emitted-text parity for all
existing kernels). All six literal-`buf` sites (MMA operands, descriptor
loads SemIR + legacy, TMA store/reduce staging, HW-producer channel path)
route through it.

Implication for the stage-invariance gate (first entry): the gate's
factual basis is now partly explained by THIS bug. With ring indexing
fixed, stage-changing v2 solutions deserve a re-test — the gate may be
relaxable to per-channelized-edge checks or removable outright. Kept until
that experiment runs clean.

### 2026-07-10 addendum 2: ring-fix verified on real v2 draws; exhaustive-fallback HANG remains open

joint-mode=2 hunt (12 draws, ring-index fix in): every v2-ACCEPTED draw
(the 8-WG shape that was previously always wrong) now passes correctness —
3 independent accepted draws confirmed. One draw HUNG (300s kill): its v2
was gate-rejected and joint-mode=2 has no v1 fallback, so it took the
EXHAUSTIVE heuristic partition on the MinII joint_solver schedule — a 4-WG layout
folding TMA loads into compute WGs. Zero alias waits present (not the new
mechanism); same class as the earlier full-regcap hang. Reachable from the
default V2ThenV1 chain only when BOTH v2 and v1 fail (e.g. subprocess
death under load) — low exposure, but nonzero. Frozen repro:
/projects/kzhou6/hwu27/case4-hang-repro-20260710/ (schedule_graph.json
replays deterministically through sched2tlx; see its README).
OPEN — next root-cause target.
Options while open: make the terminal fallback Rau+heuristic (retreat to
the fully-heuristic pairing instead of mixing exhaustive-partition with a
MinII joint_solver schedule), or gate the exhaustive fallback behind the same
schedule/partition coupling rule as the Off->V1Only promotion.

## 2026-07-10 (fifth entry): gate-removal experiment + v1-vs-v2 measurement — gate RESTORED

Gate-removal experiment (case4, joint-mode=2, 12 draws, both emitter fixes
in): v2 acceptance went to 100% (every prior rejection was the gate; the
dependence safety-net rejects nothing) — and stage-changing v2 produced
4/12 HANGS + 1/12 wrong numerics (dQ≈0.8/dK≈0.4, dV clean — a NEW
D-misalignment signature, distinct from both fixed bugs). The gate is
therefore RESTORED verbatim: beyond merge-aliasing and ring-indexing, the
emitter's stage-move contract breaks in at least two further ways. All
failing draws frozen (hang samples in case4-hang-repro-20260710/; the
dQ≈0.8 draw in the session hunt archives) — next root-cause targets.

v1 (7-WG) vs same-run v2 (8-WG, PASSING draw) performance, B200,
perf_generated.py, gen kernel TFLOPS:

| shape (BH,N)   | v1 7-WG | v2 8-WG | v2/v1 |
|----------------|---------|---------|-------|
| 8, 8192        | 116.8   |  95.0   | 0.81x |
| 8, 16384       | 132.1   | 107.0   | 0.81x |
| 2, 32768       | 118.9   |  96.4   | 0.81x |

The model-optimal joint solution is ~19% SLOWER than v1 on hardware: the
8th WG (a CUDA cluster split out on its own) adds cross-WG hand-offs whose
real cost the objective under-prices — the design note's founding
principle ("an optimal solver exploits model error maximally") measured
end-to-end. Conclusions: (a) same-stage v2 is SAFE but not yet PROFITABLE
on case4 — the win condition for v2 is a better-calibrated objective
(RRT / sync-latency terms), not more solver freedom; (b) both gen kernels
sit at 0.46-0.68x of the handwritten no-WS baseline on case4 — the
emitter/annotation gap dominates everything the solver can influence.
