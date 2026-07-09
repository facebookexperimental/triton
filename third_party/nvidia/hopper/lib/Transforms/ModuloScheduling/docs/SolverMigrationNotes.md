# The heuristic modulo pipeline vs. a future global solver

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
worth +11–40% on the GEMM-family cases. Exposing the scheduler to tighter
IIs also exposed every place where the heuristic search + partitioner +
emitter pipeline leaned on slack: those gaps are closed by priced
constraints (the partitioner's RecMII' floor, calibrated cross-WG hand-off
costs, register/SMEM launchability) and by emitter capabilities (intra-WG
software pipelining, multi-WG outer bodies, a task-coverage hard error) —
not by workload-shaped special cases. One search-budget guard remains
(below); everything else in the pipeline is either a calibrated cost, a
structural constraint, or a named generation-time error.

The overarching principle: **an optimal solver exploits model error
maximally**. A heuristic explores few points, so coarse costs are survivable;
a global solver drives the schedule exactly onto the model's boundary, so
every cost the model omits (cross-WG communication, barrier round-trips,
SMEM/register pressure) becomes a direction in which the solver produces
"model-optimal, hardware-catastrophic" schedules. The FA softmax-cut case
below is a measured example.

## The remaining guard: II search window `minII + max(10, minII/8)`

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
- case3 FA: ≥ 651 TFLOPS on (1,32,8192) — the softmax-cut canary; anything
  that cuts the softmax chain across WGs fails it.
- case4 FA bwd: ≥ 1.2x of the no-WS baseline per shape (the recurrence-floor
  + emitter-capability canary).
- case6 layernorm: schedules at all (II-window canary), store channel depth 2.
- case5: non-NaN output (task-coverage canary — a dropped epilogue op now
  fails at generation, not at numerics).
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
sched2tlx emitter. The FA softmax-cut 5x regression is a measured
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
The partitioner's RecMII' floor (the partition-aware cycle inequality with
same-stream issue chains and stream wrap-around) is the first step of this —
it prices, after the fact, what a joint formulation would enforce natively;
the end state is warp assignment as a decision variable co-solved with
placement.

### 2. Heuristic search → complete solver

Rau IMS + ejection + a bounded II window is an incomplete search; the II
window exists only because of that incompleteness and disappears under any
complete method. `ExhaustiveScheduler.cpp` (branch-and-bound + joint
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

The solver formulation should carry a versioned whitelist of
emitter-lowerable partition/loop shapes (blockM≤128 TMEM, no outer channels
in a task that also owns an inner loop, ...) as hard constraints, each
removed as the emitter gains the capability — never encoded implicitly in
costs, or the solver will route around it and regress correctness. The
emitter's task-coverage check is the runtime backstop: any shape it cannot
lower fails generation with a named error instead of dropping ops.

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
   alone deletes the II window.
3. **Long**: joint schedule+partition formulation (subsumes the RecMII'
   floor by construction), sub-tiling in front, streaming depths handed to
   the autotuner. Position it as an offline/autotune tool, not the default
   JIT path — Twill's own solve times (tens of seconds to minutes) imply
   the same.

The validation harness above is the gate for every step, and the
"Why this note exists" principle is the order of operations: a complete
solver exploits model error maximally, so model calibration must land
BEFORE each increment of solver power, not after. (`LLMSchedulePass.cpp`,
the experimental Claude-API scheduler, shares the same DDG serialization
and should reuse the same canaries.)
