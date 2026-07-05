# Heuristic guards vs. a future global solver

Status: design note, written 2026-07 alongside the throughput-based latency
model change (PR #1912). Audience: whoever replaces the current heuristic
modulo scheduler (Rau IMS / SMS / exhaustive, dispatched in
`ModuloReservationTable.cpp:runModuloScheduling`) with a global optimal
solver (ILP / CP-SAT, "Twill-style").

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
