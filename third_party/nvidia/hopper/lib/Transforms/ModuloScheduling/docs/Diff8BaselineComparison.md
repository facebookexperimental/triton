# Diff 8: Baseline Comparison for the M2 Acceptance Criteria

## Why the original criterion was replaced

The M2 acceptance criterion read:

> Solver-produced II is valid and satisfies all resource/dependency
> constraints; **improvement over naive encoding (resource-exclusivity-only)
> demonstrated.**

The second clause cannot be satisfied. A resource-exclusivity-only model is the
full model with constraints *removed*, so its feasible set is a superset of the
full model's, and its optimal II can never exceed the full model's:

```
II*_resource-only  <=  II*_full        (always, by construction)
```

Demonstrating that the full encoding achieves a *better* II than the
resource-only encoding is therefore impossible for every input. A test written
to that wording could only pass through a bug.

The criterion is split into two well-posed ones, implemented here.

## Metric 1 — `relaxed_lower_bound` (soundness, not improvement)

`runJointSolverRelaxedLowerBound` (`JointSolverScheduler.h`) solves the same
joint model with only the `resource:` constraint groups asserted. The assertion
is the direction that is actually true:

```
II_full  >=  relaxed_lower_bound
```

A violation is **not** a performance regression. It means the full model found
a schedule the relaxed model calls impossible, which can only happen if a
constraint is mis-encoded or a "relaxation" is not actually a relaxation. That
makes it a genuinely valuable regression test, and it should be described as a
soundness check rather than dressed up as a performance claim.

The gap `II_full - relaxed_lower_bound` is reported as information only. It
says how much II the non-resource constraints cost, which is real engineering
signal, but it is not an acceptance threshold — a large gap can equally mean
those constraints are necessary.

### How the relaxation is obtained

`AssumptionTracker` already gates each constraint group behind an assumption
literal, so every group's constraints only bind when its literal is asserted
true:

```cpp
void AssumptionTracker::assertFormula(llvm::StringRef groupId, Z3_ast formula) {
  encoder.assertFormula(encoder.implies(get(groupId), formula));
}
```

Passing a *subset* of the literals to `checkWithAssumptions` therefore yields
the corresponding relaxed model with no changes to the encoder. The problem
JSON carries an optional `relax_keep_kinds` array; `solveCandidate` filters the
assumption vector through `AssumptionTracker::literalsForKinds`. This reuses
infrastructure that already has to be correct for UNSAT-core extraction to
work.

A relaxed solve stops at the first feasible II rather than running the
composite-objective binary search: only the bound is wanted, and the relaxed
objective is not comparable to the full model's anyway (the dropped groups also
drop their objective contributions).

### What the bound is NOT

The metric is named `relaxed_lower_bound`, not `resource_lower_bound`, because
not every constraint is assumption-gated. These stay hard assertions:

- cycle domain bounds (`0 <= cycle < horizon`)
- canonical-root pinning
- `smemTotal <= smem_budget`
- per-checkpoint TMEM columns `<= tmem_col_limit`

The last two survive but go **vacuous** in a resource-only relaxation: dropping
the `smem:`/`tmem:` literals zeroes their charges. Only the domain bounds and
the root pin actually bind.

The bound is consequently *tighter* than a textbook resource-only relaxation.
That is accepted deliberately: the soundness invariant holds for any
relaxation, so a tighter bound is a **stronger** test, and the alternative — a
switch that also drops the untracked assertions — would add a code path
existing solely for one test, which could itself be wrong. A reader comparing
this number against a published resource-only bound needs to know it is not
one, so the caveat is printed with the number and repeated in
`kUntrackedHardAssertions`.

The relaxed run is deliberately **not** routed through
`runJointSolverBackend`: that wrapper validates every response against the full
model, which a relaxed schedule violates by construction. It is not trusted
either — the kept kind (resource exclusivity) is re-verified in-tree before the
bound is used, because a bound that is too *high* would turn the soundness
invariant into a false alarm. The backend must also echo a `relaxation` object,
so a build whose solver ignored `relax_keep_kinds` cannot silently return the
full model's II and make the invariant compare `II_full` against itself.

## Metric 2 — improvement over Rau

```
II_full  <=  II_rau     on every fixture      (hard failure if violated)
II_full  <   II_rau     on at least one       (the acceptance criterion)
```

This is well-posed: a heuristic scheduler can and routinely does land above the
optimum.

**Rau's Iterative Modulo Scheduling is the baseline**, for four reasons:

1. It already exists and is tested — the default backend
   (`TRITON_USE_MODULO_SCHEDULE` = `1` or unset). No new scheduler to write, no
   new tie-breaking to specify and defend.
2. It is the actual production alternative: the joint solver's own fallback
   path drops through to precisely this scheduler, so beating it is the claim
   that matters operationally.
3. It unifies M2 with M3. The M3 criterion reads *"performance improvement over
   Triton's current default scheduling"*, and Triton's current default **is**
   Rau. M2 and M3 become schedule-level and wall-clock-level measurements of
   one claim rather than two unrelated exercises.
4. A purpose-built "naive greedy" would be a strawman — any scheduler written
   solely to be beaten invites "did you tune it to lose?". Rau is a published
   algorithm the team already relies on.

`SwingScheduler` (`sms`) and `ExhaustiveScheduler` (`exhaustive`) are reported
alongside at no extra cost. **Rau is the acceptance gate**; the others are free
context. `exhaustive` is especially useful where it terminates, since it
brackets the optimum from the other side — it bails out above its node/MMA
ceiling, so an absent row is expected on anything but a small loop.

A purpose-built resource-exclusivity-only greedy is deliberately **not**
written. If the literal original wording is ever required, it would be added
*in addition to* Rau, never instead of it: it is the weaker claim and should
not be the headline number.

### Every baseline is validated before its II is used

An unvalidated II is meaningless — a scheduler that produces an illegal
schedule can report any II it likes. `isLegalModuloSchedule` checks dependence
legality *and* exclusive modular reservation; a backend whose schedule fails is
reported as absent with a note rather than contributing a number.

This uses the in-tree checker rather than `Z3JointSolutionValidator`. The two
enforce the same two properties for the v1 model, and the in-tree one keeps
every baseline row available in builds without Z3, where only the joint rows
drop out.

## Running it

```bash
TRITON_MODULO_BASELINE_REPORT=1 triton-opt input.mlir -nvgpu-modulo-schedule
```

The comparison runs every backend on the DDG, so it is opt-in and never fires
on a compile path. Output goes to stderr so it does not interleave with
`triton-opt`'s IR on stdout:

Measured output on the checked-in fixture:

```
modulo-baseline-report: gemm_inner_loop
  fixture                      MinII     II_full  relaxed_LB      II_rau      II_sms II_exhaustive
  gemm_inner_loop               1091        1091        1091        1091        1091        1091
  soundness (II_full >= relaxed_LB): PASS (gap 0)
  no-regression (II_full <= II_rau): PASS
  strict improvement (II_full < II_rau): NO (every backend is at MinII 1091 — this
    fixture is MinII-bound and cannot separate schedulers; it is not evidence about
    the solver)
  relaxed_LB keeps resource exclusivity only; still-hard: cycle domain bounds,
    canonical-root pinning, SMEM/TMEM ceilings (vacuous once their gated
    contributors are dropped)
```

The two halves of criterion 2 are reported separately on purpose. `II_full ==
II_rau` satisfies the no-regression half, and printing one combined `PASS`
would let a tie read as the acceptance criterion being met. It is not.

## Fixture

The lit fixture `test/TritonGPU/modulo-schedule.mlir` (`@gemm_inner_loop`) —
real GEMM, and already the DDG the JOINT goldens pin, so the baseline numbers
are directly comparable to existing golden output.

The 2–3 node oracle fixtures are deliberately not used: they are below the size
where scheduling quality is meaningful, and Rau trivially matches the optimum
on them, producing a vacuous "no improvement" result that is evidence of
nothing.

Note this is currently the only GEMM shape checked in **as a DDG**. The
two-architecture `gemm-hopper` / `gemm-blackwell` models in
`unittest/Hopper/modulo_schedule_test.cpp` are joint-solver-0.2 **JSON**
problems, which the in-tree schedulers cannot consume — `DataDependenceGraph`
has no public constructor and is only produced by
`DataDependenceGraph::build(scf::ForOp, LatencyModel)`. Adding a second shape
means adding a second MLIR loop to the lit fixture, not a second JSON model.

## Current result: the acceptance criterion is NOT met

**As of this change, `strict improvement` is NO on the only fixture available.**
Every backend — joint solver, Rau, SMS and exhaustive — lands on II = 1091,
which is exactly `MinII`.

Root cause, and why it is not a solver result: **the fixture is MinII-bound.**
MinII = max(ResMII, RecMII) is a floor no correct scheduler can go below, so
when every backend attains it there is nothing left to optimise and no
scheduler can be separated from any other. The relaxed lower bound confirms
this from the other side: `relaxed_LB == II_full`, a gap of **0**, meaning the
resource-exclusivity constraints alone already force II = 1091. Dependences and
buffer depths cost this loop no II at all. A single-MMA GEMM inner loop
saturates one pipeline, and the schedule is decided by that pipeline's
occupancy.

So this fixture cannot produce evidence for or against the M2 improvement
claim in either direction. It is a valid soundness fixture and a valid
no-regression fixture; it is not a discriminating improvement fixture.

Per the plan, the fixture is **not** being tuned until the solver wins. Two
honest ways forward, both of which are follow-up work:

1. Add a fixture that is *not* MinII-bound — a loop with a recurrence or
   buffer pressure that forces a scheduler to leave the floor, where a
   heuristic can measurably land above the optimum. This means adding a second
   MLIR loop, since the in-tree schedulers consume a DDG, not the v0.2 JSON
   models.
2. Accept that the joint solver's value on GEMM is not II, and move the M2
   improvement claim to the dimension where it does show up — warp-group
   assignment, buffer depth, or M3's wall-clock — rather than schedule-level
   II.

This is exactly the outcome the plan flagged as plausible ("Rau is a good
heuristic and GEMM inner loops are small"). It should go to the IM alongside
the Decision 1 sign-off request, because it bears on whether the reformulated
criterion is satisfiable on the fixtures that exist today.

## Sequencing

Rau's II is machine-independent, but the joint solver's may not be until the
Diff 7 determinism work lands (`docs/Diff7DeterminismPlan.md`). Baseline IIs
must be stable for this to be a regression test rather than a flake. The lit
test therefore matches IIs as numbers rather than pinning them, and asserts
only that neither criterion reports `FAIL`.
