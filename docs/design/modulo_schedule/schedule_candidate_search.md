# Schedule candidate search — design plan (deferred)

Status: **PLAN ONLY.** We keep the current snapshot/re-finalize approach for now
and build this when the optimization set stabilizes (see "When to build"). This
doc captures the target design so the interim hack is intentional, not accidental.

## Problem

The modulo schedule is produced by a sequence of mutating passes, each of which
makes a *choice* (a knob):

```
Rau schedule → buffer depths → SMEM budget → buffer merge
  → data-partition (A.5, N)  → epilogue-subtile (A.7, S)
  → warp-group partition (P) → cross-WG barrier synth → re-merge
```

Today each stage commits a single cost-model winner. The only place we explore
alternatives is the **last** stage (warp-group partition), via
`TRITON_MODULO_DUMP_TOPN`: snapshot the pre-partition graph
(`ScheduledLoop::prePartitionGraph`), then `finalizeLoopPartitionForDump` re-runs
the partition tail for each candidate.

Two problems, both structural:
1. **Duplicated finalization.** The finalize sequence exists twice — the winner
   path (`applyGlobalWarpPartition`, three global passes) and the dump path
   (`finalizeLoopPartitionForDump`, per-loop). Adding a finalization step means
   editing both or the dumped variants silently diverge.
2. **Doesn't generalize.** To also tune DP and subtile we'd need a snapshot
   *before each* tunable stage and a bespoke re-run of *each* tail — combinatorial
   bespoke code, multiplying problem #1.

## Target design: replayable decision log + beam search

Make the schedule a **derived artifact of (base DDG + an ordered decision list)**,
never a hand-synced object.

```
Decision        = DataPartition{n} | EpilogueSubtile{s} | WGPartition{nodeWG[]} | ...
ScheduleRecipe  = base loop id + [Decision]            # small, serializable
materialize(recipe) -> ScheduleGraph                   # deterministic REPLAY
```

- Each pass becomes a transform `apply(graph, decision) -> graph`. `materialize`
  folds the passes over the base DDG in canonical order, applying decisions.
  Whole-module stages (barrier synth needs global WG consistency) stay
  module-granular inside `materialize`.
- **One finalize path.** The committed winner is `materialize(winnerRecipe)`;
  every dumped candidate is `materialize(candidateRecipe)`. Duplication (problem
  #1) is gone — there is only one replay.
- **Candidate list = a beam frontier of recipes.** Per tunable stage:
  `expand` each recipe by its choices (`DP∈{1,2}`, `subtile∈{1,2,4}`,
  `partition∈topN`), `score` via the cost model (replay + makespan), `prune` to
  top-K (K≈10–20). Dump the top-K recipes (problem #2 solved without blowup —
  the beam prunes).
- The JSON dump becomes "serialize K materialized graphs"; the sched2tlx emitter
  is unchanged. `prePartitionGraph` is subsumed by prefix-memoization of recipes.

This mirrors TVM/Ansor, Halide schedules, and Triton's autotuner: a small
schedule-decision IR that is cheap to store, search, and **replay**.

## Why it's deferred (complexity / risk)

- New types + beam driver are bounded/additive (~few hundred LOC).
- The cost is refactoring the **core, working** pass pipeline into clean
  replayable transforms while preserving global-ordering constraints. That
  touches the production main-compile path → real regression risk.
- The optimization set is still in flux (DP, subtile, partition, + future). Build
  the abstraction now and we likely build it around the wrong stages and churn.

Net: high risk, uncertain shape → **plan now, build later.**

## Interim (keep the hack, reduce its sharp edge)

- Keep `prePartitionGraph` snapshot + `finalizeLoopPartitionForDump` (only branches
  the WG-partition stage — the one stage we tune today).
- Small cleanup: extract `remergeBuffers(loop)` (reset mergeGroupId + clear
  physicalBuffers + `mergeNonOverlappingBuffers`) used by BOTH the winner path's
  re-merge pass and the dump path, so the most-likely-to-change block has one
  definition. Cross-reference the two paths in comments.

## When to build

Trigger any of:
- A second tunable stage (DP or epilogue-subtile) needs to be searched jointly
  with partition (not just env-override one-offs).
- We move from "verify the cost-model rank" to real schedule autotuning in CI.
- The cost model is judged un-perfectable and empirical search becomes the
  primary selection mechanism.

## Incremental migration (when triggered)
1. Extract `remergeBuffers()` (interim, do now).
2. Introduce `ScheduleRecipe` + `materialize`, WG-partition decision only —
   behavior-identical to today; winner = `materialize(winnerRecipe)`; delete
   `finalizeLoopPartitionForDump`.
3. Fold A.5 (DP) and A.7 (subtile) env overrides into first-class `Decision`s.
4. Replace per-stage greedy winners with a beam over recipes (prefix-memoized).

Validation gate at every step: the committed-winner schedule must stay
byte-identical (recipe replay reproduces today's `schedule_graph.json`).
