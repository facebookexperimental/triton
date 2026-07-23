# Retiring guard 1 on the heuristic path: emitter intra-WG MMA pipelining + partition-induced RecMII gate

Plan only — nothing in this document is implemented yet. Goal: delete the
Phase 2.75 serial-MMA guard from `DataDependenceGraph.cpp` while keeping
the heuristic (Rau + scoreCandidate) path correct and at-least-as-fast,
by fixing the two root causes the guard papers over. Grounded in the
2026-07-08 experiment series (baseline + three ablations + autopsy; all
artifacts under the session scratchpad `ablation/guardfix_*`,
`guardfix/before.*`, reproducers `bench_reorder.py`, `hang_probe*.py`).

## Evidence this plan rests on

What the guard costs (measured):
- case3 FA fwd: pinned at II=1459 while II=1325 is achievable and better
  (joint partitioner, 665.7 vs 651 TFLOPS).
- case4 FA bwd: modeled II inflated 3x (3136 guarded vs 1033 unguarded).
- Sub-tiled FA: guard prices the split graph at ResMII 2918 — needs a
  bypass env on the #1928 line.

What guard removal costs today (measured):
- The partitioner cuts the softmax chain at II=1325: 102 TFLOPS on
  #1912's synthesis (the original "5x"), 493 on #1913-calibrated.

Why the three model-side quick fixes failed (2026-07-08):
- v1 linear recurrence penalty: fires correctly (+453, probe-verified
  RecMII'=1778 > II=1325) but loses to the makespan objective — max over
  per-WG makespans structurally rewards splitting a serial chain.
- v2 synthetic depth-1 backpressure edges: added nothing to the winner.
- v3 feasibility hard gate: every judgment correct (good partition
  1445 <= 1459 feasible; cut 1445 > 1325 infeasible), but rejecting the
  cut re-routed the search to [1,1,4,4] — TC nodes folded into
  multi-warp consumer groups, a shape the emitter has never lowered.

The [1,1,4,4] autopsy (the two pathologies this plan fixes):
1. **Issue-then-inline-wait serialization**: for a TC node co-resident
   with its consumers, the emitter issues `async_dot` then immediately
   `barrier_wait`s its completion mbarrier in the same instruction
   stream ("intra-WG async result" sem pattern). Tensor core never
   overlaps the WG's CUDA work — measured 2x (0.045 vs 0.0225 ms at
   (1,4,512)).
2. **Phase-race device livelock**: the ad-hoc fan-out double-wait scheme
   (two `barrier_wait(semN_full[0], _it & 1)` sites per iteration)
   livelocks the device (100% SM utilization, warps spinning in barrier
   polls, host asleep) at >=64-CTA shapes under do_bench's exact stream
   pattern. Plain repetition, memset interleave, and unsynced deep
   queues all pass; the reproducer is `bench_reorder.py` at (1,8,1024)
   — hangs on the first do_bench, deterministically.

## Workstream 1 — emitter: intra-WG MMA software pipelining

The emitter today only software-pipelines ACROSS warp groups (cross-WG
channels + semaphores); within one WG it renders each logical iteration
linearly and stalls on its own MMAs. This workstream teaches it the
classic modulo expansion for the intra-WG case.

**Scope-narrowing precondition (what makes this tractable):** skewed
(cross-stage) edges must be memory-resident — MMA outputs are TMEM by
construction, bridges are TMEM, K/V rings are SMEM. Register values must
NOT cross a stage boundary inside a WG. This avoids modulo variable
expansion (rotating register renaming) entirely. Enforced by a static
check; violation raises a named EmitterCaps-style error, never a silent
mis-lowering.

Steps:

1.1 **Trigger + legality check.** A WG enters the pipelined-emission
    path when its nodes span more than one schedule_stage AND an
    intra-WG async producer (MMA, TMA) is consumed at a later stage in
    the same WG. Check the register-residence precondition; classify
    memory-residence per op (MMA→TMEM, descriptor_load→SMEM,
    local/tmem_store targets).

1.2 **Protocol: reuse SemIR, delete the ad-hoc scheme.** Treat intra-WG
    cross-stage async edges as same-WG channels on the standard
    full/empty semaphore machinery (ring indexing, phase expressions,
    mBarriers attachment — the battle-tested cross-WG path). This
    replaces the inline-wait + fan-out double-wait protocol wholesale
    and thereby fixes pathology 2 (the race lives in the replaced code).

1.3 **Body transformation (the main work).** Group the WG's nodes by
    stage; emit prologue (ramp-up copies), steady-state body, epilogue
    (drain). In the steady body, a node at stage s operates on logical
    iteration (_it − s): buffer indices and phase expressions change
    from `_it` to `(_it − s)`. Use explicit peeling for
    prologue/epilogue (matches hand-written kernels and the existing
    `use_acc = (t > 0)` machinery) rather than in-body predication.

1.4 **Pass-side depth fixup (a real pass↔emitter contract change).**
    Buffer counts (lifetime/II + 1) are computed before partitioning,
    but intra-WG skew extends a consumed value's lifetime by II →
    depth must grow by 1 (e.g. qk TMEM 1→2). Add a post-partition
    fixup step in `applyGlobalWarpPartition`: recompute counts for
    intra-WG async edges under skew, then re-check TMEM/SMEM budgets.
    If the budget does not fit, fall back to non-skewed emission for
    that WG and reject the partition (feeds Workstream 2's legality
    fallback).

1.5 **Acceptance ladder.**
    - The livelock reproducer (`bench_reorder.py` @ (1,8,1024)) passes.
    - The v3 [1,1,4,4] case3 kernel: correctness on all shapes AND the
      2x is recovered (do_bench completes; this partition saves two WGs
      and may become genuinely competitive once the makespan model's
      in-WG overlap assumption is finally true).
    - All committed cases regenerate byte-identically (their WGs are
      single-stage — the new path must not trigger).
    - Full perf comparison against the saved baseline.

Effort: days (1.3 dominates). Risks: peeling × iter_args interaction
(m_i init/final placement in prologue/epilogue), budget edge cases in
1.4, completeness of the memory-residence classification.

## Workstream 2 — partitioner: partition-induced RecMII feasibility gate

Definition (re-implementation of the probe-verified v3 experiment):

```
RecMII'(candidate) = max over back-edges b of
    ceil(( longestAugPath(b.dst → b.src) + lat(b) + cross(b) ) / dist(b))

where forward-path edge weights = latency + cross(e), and
cross(e) = 0                                if same WG
         = rt + smemMove(bytes)             if producer register-resident (d=0)
         = 2 × barrier handshake            if producer SMEM/TMEM-resident
                                            OR the edge is loop-carried (d≥1):
                                            the value was staged in an earlier
                                            iteration — only the release
                                            handshake remains to pay. (Found
                                            during landing: charging a fresh
                                            round-trip on back-edges priced the
                                            committed FA-fwd partition's qk
                                            anti-edge at +552 → RecMII' 1778
                                            where the achieved steady state is
                                            ~1459; with the release price the
                                            floor is 1445 — matching both the
                                            v3 probe and the measured kernel.)

Feasibility: RecMII'(candidate) > loop.II  ⇒  candidate infeasible
             (large additive penalty, both scoreCandidate and
             evalGreedyCost — identical term in both paths)
```

This is the modulo-scheduling cycle inequality (sum of cycle latencies
<= II × sum of distances) made partition-aware. It is workload-agnostic:
inputs are the dependence graph, the WG assignment, and globally
calibrated hand-off constants. Verified classifications: good case3
partition 1445 <= 1459 feasible; softmax cut 1445 > 1325 infeasible;
case1/2/5/6/7 byte-unchanged (their splits are not on cycles).

Steps:

2.1 Re-land `computePartitionRecMII` (topological longest-path per
    back-edge over the distance-0 DAG, augmented weights; O(back-edges ×
    (V+E)) per candidate — measured acceptable inside the exhaustive
    enumeration).
2.2 Hard-gate hookup in `scoreCandidate` and `evalGreedyCost` with a
    shared constant (infeasible >> any makespan scale).
2.3 **Explicit fallback**: if no candidate is both feasible and legal
    (Workstream 1's budget fallback can shrink the space), pick the
    legal candidate with minimum RecMII' excess and log loudly — never
    fail silently, never pick a pathological winner silently.
2.4 Register a `TRITON_MODULO_DEBUG_RECMII` env (GetEnv.hpp!) that
    prints the final partition's RecMII' vs II — the probe that made
    v1's diagnosis possible, kept as a permanent debugging aid.
2.5 Acceptance: case1/2/5/6/7 byte-parity; the case3 cut is rejected;
    with Workstream 1 landed, the winner at II=1325 is the cohesive
    corridor partition (TC in dedicated 1-warp WGs, softmax chain intact
    in one 4-warp WG).

Note on constants: the hand-off prices follow whatever branch this lands
on (#1912: 500 / elems-based; once stacked under #1913: 150 / 16 cyc/KB
+ resolveCrossWGChannel). The gate's structure is unchanged either way.

## Workstream 3 — guard deletion + validation + review reply

Order matters: **1 → 2 → delete guard**. With the emitter fixed first,
no interim "TC must not cohabit" legality constraint is needed — the
pathology region becomes genuinely lowerable, and v4 reduces to the gate
alone.

3.1 Delete Phase 2.75 from `DataDependenceGraph.cpp` (the block carries
    its own "a future global solver should delete it" note; this plan is
    the heuristic-path equivalent).
3.2 Regenerate all seven cases. Expected: case1/2/5/6/7 byte-identical
    to baseline; case3 re-schedules at II=1325, case4 at II=1033 with
    cohesive partitions — new fixtures, GPU-validated.
3.3 Perf gates against the saved baseline (`guardfix/before.*`):
    - case3: >= 653.8 TFLOPS at (1,32,8192) (do_bench). Honest risk:
      Rau's 1325 schedule + cohesive partition on #1912's synthesis is
      unmeasured (joint's 665.7 used its own schedule + calibrated
      synthesis). If it lands below baseline, guard retirement waits for
      the #1913 stacking and re-runs there — that outcome is a clean
      datum, not a failure of the plan.
    - case4: >= 18.4/81.3/178.8/192.8/226.9/202.3 TFLOPS per shape —
      this is where the guard's 3x II inflation should turn into a real
      win.
    - case2 unchanged (its trap is an unrelated emitter gap).
    - Sub-tiled path re-checked after the #1928 restack (the bypass env
      becomes dead code — remove it there).
3.4 Reply to wlei on #1912 with the mechanism story: dependence edges
    were never the issue (full result latency always honored); the guard
    encoded pipeline-trust, which this series replaces with (a) a real
    intra-WG pipelining capability and (b) the cycle-feasibility
    inequality in the decision layer — no workload patterns anywhere.

## Sequencing summary

| # | Item | Size | Gate to proceed |
|---|------|------|-----------------|
| 1.1–1.2 | trigger/legality + SemIR protocol swap | ~1 day | livelock reproducer passes |
| 1.3 | modulo expansion of task body | ~2 days | [1,1,4,4] correct + 2x recovered |
| 1.4 | post-partition depth fixup + budget | ~0.5 day | committed cases byte-parity |
| 2.1–2.4 | RecMII gate + fallback + debug env | ~1 day | case3 cut rejected, 5 cases parity |
| 3.1–3.3 | guard deletion + full validation | ~0.5 day | perf gates vs baseline |
| 3.4 | review reply | — | — |

Baseline artifacts, reproducers, and the v1–v3 experiment record are
preserved in the session scratchpad; the baseline numbers quoted above
are do_bench medians measured 2026-07-08 on B200.

## Execution log (2026-07-09)

**Workstream 1 — LANDED** (sched2tlx `emitter.py` + `semaphore_ir.py`):
`_compute_skew_plan` (union-find over non-skewable d=0 edges; skewable =
async MMA producer → later-stage same-WG consumer), guarded per-group body
blocks re-binding the induction var to the logical iteration, full/empty
ring on the producer's destination (depth = gap+1), same-stream dedup of
signal-only waits. Acceptance: livelock reproducer fixed at BOTH do_bench
(1,8,1024) and plain-launch (1,32,8192) — the latter hung the old serial
kernel too, so the protocol swap fixed a plain-usage correctness bug, not
just a bench artifact; correctness 5/5 shapes; committed cases regenerate
byte-identically 7/7 (no WG in them triggers the skew path). DEVIATION:
the "2× recovered" line was NOT met — the skewed [1,1,4,4] measures ≈
serial (−12 % at (2,16,2048), plain-launch A/B). Root cause is physics,
not the emitter: with softmax and the acc recurrence in one instruction
stream the partition floors at ~1900+ cycles — above the 5-WG partition's
1459. The partition must be rejected by pricing, not rescued by lowering.

**Workstream 2 — LANDED, evolved from hard gate to effective-II floor**
(`computePartitionRecMII` + `bottleneck = max(bottleneck, RecMII')` in both
scorer paths + `TRITON_MODULO_DEBUG_RECMII`): two corrections found during
landing, both probe-verified: (1) back-edges charge the release handshake
(2×30), not a fresh round-trip — otherwise the committed FA-fwd partition's
qk anti-edge reads +552 and RecMII' 1778 where reality sustains ~1459;
(2) **same-stream issue chains**: consecutive same-WG nodes (emission
order) get synthetic edges weighted by the earlier node's selfLatency — a
recurrence entering a WG at A and leaving at B pays the issue time of
everything between them. Without (2), the guard-off search re-picked the
original "5×" shape (softmax+correction merged, row-sum reduce cut out
through a 32 KB/iter channel; measured 102 TF) and the top-8 candidates
were COST-IDENTICAL to the tie-break digit — the makespan objective is
flat exactly where it matters. With (2), the merged shape floors at 2612
vs the cohesive shape's 1445, and the guard-off winner becomes the
committed partition shape. Guard-in: all 7 cases byte-identical.

**Workstream 3 — guard deletion attempted, REVERTED; deferred to the
calibrated (#1913) re-validation.** With guard off + full floor: case1/2/
5/6/7 byte-identical; case3 re-schedules at II=1325 with the committed
partition shape, correctness 5/5, do_bench A/B vs baseline: 1.000 / 1.003
/ 1.024 / 1.015 / 0.998 / **0.913** — the (1,32,8192) gate (≥ 653.8, got
598.4) fails on a statement-order artifact (the 1325 schedule emits the
845-cycle row-sum AFTER the p-bridge wait; the 1459 schedule emits it
BEFORE, hiding it under the PV-MMA drain). case4's guard-off winner
(8 WGs, II 1033 vs 3136) fails ptxas: C7602 insufficient registers — the
Layer-C register model admits a partition that does not compile. Both
blockers are model-fidelity items on the #1912 constants; neither touches
the WS1/WS2 mechanisms.

**Net state on this branch**: guard stays (with a RETIREMENT STATUS note
pointing here); WS1 + WS2 are in as byte-parity-verified improvements.
Remaining for retirement: re-run the guard-off validation on the #1913
calibrated table; fix the register-budget check that admitted case4's
non-compiling partition; consider a "hoist independent compute above
blocking channel waits" emission rule to close the case3 statement-order
gap (would change committed fixtures — needs its own byte-parity re-land).

## Calibrated re-validation (2026-07-09, #1913 cherry-picked as 06ca9d6c7)

#1913's net content (calibrated constants: kCrossWGRoundTripLatency
500→150, byte-based smemMoveCost, resolveCrossWGChannel, channel-SMEM
launchability gate) was cherry-picked onto this branch; the RecMII' floor
merged on top of it (crossCost now uses registerTensorBytes + the
calibrated slope). Guard-in: 7/7 byte-parity holds; case3's committed
floor reads 1445 ≤ 1459 on the calibrated table too.

Guard-off retry:
- case1/2/5/6/7 byte-identical; case3 and case4 pick the same partitions
  as on the uncalibrated table (cohesive at II=1325; 8-WG at II=1033) —
  calibration changed neither blocker.
- case3 A/B (do_bench): 1.000 / 1.059 / 1.024 / 1.018 / 0.996 / **0.913**
  — flagship still −8.7%, same statement-order artifact.
- case4: the register over-request is now fixed at the EMITTER — a new
  `_rescale_task_regs` step fits explicit num_regs requests to the 64K
  register file when they over-ask (152→96 for the four 4-warp tasks;
  committed kernels are under budget → byte-identical). The kernel then
  COMPILES — and **deadlocks on its first launch** at the smallest shape
  (2,1024): the 8-WG 5-MMA partition is emitter territory never exercised
  by any committed case. Root-causing that lowering gap is the remaining
  case4 blocker; the scoring-side underestimate (wgRequiredWarps counts
  clustered nodes only, so post-propagation 4-warp WGs are priced at 1
  warp — residual reads 0 where the real request is 110K/64K) is the
  model-fidelity companion to fix with it.

Verdict unchanged: guard stays. The two named blockers are now sharper —
(1) statement-order criticality in the scheduler objective (shared with
the CP-SAT known gap), (2) the 8-WG 5-MMA emitter deadlock + scoring-time
warp-count drift.

## RETIREMENT (2026-07-09, second session — guard deleted for good)

Both blockers fixed at their sources; Phase 2.75 is gone and every gate
passes on the calibrated (#1913) table.

**case3 statement-order gap → emitter triple-sinking.** A/B experiments
first: hoisting the l_i chain above the channel waits recovers the
flagship fully (656.5 vs 657.5); deepening the p-bridge to 2 slots does
nothing (596.9) — the stall was the stream idling at a channel EMPTY wait
with useful work trapped below it, not buffer capacity. Implemented as:
producer-side channel triples (wait-empty → store → arrive-full) are
DEFERRED and flushed at the end of the body or just before the next op
whose emission can wait (the flush guard matters — sinking a triple below
another wait closed a cross-WG cycle on FA-bwd and deadlocked). The five
parity cases are unaffected (case6's only triple already sat at its body
end). Guard-off case3, final: 23.8 / 135.3 / 404.5 / 475.7 / 543.6 /
656.1 TF — parity with the guard-era baseline on every shape.

**case4 → three partitioner terms + three emitter gaps.** The guard-off
search kept picking split partitions that modeled ~1.5K cyc/iter and
sustained ~5K. Missing physics, added in order of discovery:
(a) accumulatePropagatedMinWarps — scoring saw two 4-warp WGs as 1-warp
    (the 4-warp addptr joins them only at propagation), hiding a 110K/64K
    register request behind residual 0 (the ptxas C7602 winner);
(b) stream wrap-around order edges in computePartitionRecMII — barrier
    waits execute in stream order ACROSS iterations (last waiter of j
    precedes first waiter of j+1, and waits block all warps, so warp
    interleaving cannot hide them). This closes distance-1 cycles through
    whatever cross-WG dependence paths feed the waits — pricing why a
    depth-1 channel chain runs at the SUM of its hops;
(c) the effective-assignment preview — scoring runs before infra-op
    propagation, so paths through NONE ops (addptr → load, truncf →
    local_alloc → MMA) read as cross-free and severed the very cycles (b)
    should catch.
Emitter gaps exposed along the way (each was a first-launch deadlock or
garbage): tt.load was not a named in-loop op (cross-WG consumers of m/D
starved — same-WG consumers had always inlined it); register→local_alloc
cross-WG channels stored into the synthesized routing buffer while the
consuming MMAs read the alloc's own ring (unified onto the data buffer,
mirroring the TMA-channel rule); and the triple-sinking flush guard above.
Also: emitter register-request rescaling (fit num_regs to the 64K file)
turns over-budget partitions from ptxas aborts into measurable kernels.

Final guard-off case4 winner (6 WGs {2,2,19,6,4,3}, II=1033, committed
RecMII' 2211): correct, and 22.0 / 103.1 / 232.1 / 249.5 / 292.6 / 261.6
TF vs the guard-era baseline's 18.4 / 81.3 / 178.8 / 192.8 / 226.9 /
202.3 — **+20-30% on every shape**, above the hand-written WS kernel. The
guard's 3x II inflation, removed, became exactly the win the plan
predicted.

**Landed state**: Phase 2.75 deleted (comment records the replacement
mechanisms); case3/case4 fixtures regenerated (schedule_graph.json +
generated.py, self-consistency verified); case1/2/5/6/7 byte-identical;
sched2tlx unit tests pass. The [1,1,4,4] livelock fix, the skew rings,
and the RecMII' floor from the first session are all part of the final
state — the fifth case4 winner iteration exercised the skew path in a
real pipeline (wg3 MMA→tmem_load cross-stage) and lowered correctly.
