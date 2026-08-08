# Reproducible Solver Runs

Why the native joint solver returns the same answer on every machine, and how
the numbers that make that true were measured.

## The problem

The search used to be bounded only by a wall-clock deadline: `remainingTimeoutMs`
computed the time left against a `steady_clock` deadline and handed it to Z3 as
`timeout` before every check. When that deadline ran out the result flipped from
`ok` / `infeasible` to `inconclusive`.

That transition is a function of machine speed and load, not of the input. A
slower host, a busy CI worker, or an unlucky scheduling quantum could change the
answer. The cache added in D115014899 gives *repeat-consistency* — a
byte-identical request replays a byte-identical stored response within one
process — but not *reproducibility*: the same input solved fresh could
legitimately land somewhere else.

## The design

Two budgets, two jobs:

| Budget | Purpose | Binds in normal operation? |
| :--- | :--- | :--- |
| `rlimit` | Deterministic ceiling on search effort. Counts Z3 work units, not seconds, so the same input stops at the same point on every machine. | **Yes** |
| wall-clock `timeout` | Safety net so a solve can never hang the compiler. | No — only on pathological inputs |

Both are installed on every check. The wall clock is set well above the time the
`rlimit` budget normally takes to run out, so in practice `rlimit` is the one
that fires and reproducibility holds for every input that behaves normally.

Defaults: `kDefaultRLimit = 10000000`, `time_limit_s = 180.0`. Both are
overridable per request (`rlimit`, `random_seed`, `time_limit_s`) and, for the
two production request builders, by `TRITON_MODULO_JOINT_SOLVER_RLIMIT`,
`TRITON_MODULO_JOINT_SOLVER_SEED` and `TRITON_MODULO_JOINT_SOLVER_TIMEOUT_S`.
Because the knobs travel in the request, they land in the cache key
automatically — a cached response can never have been found under a different
budget.

## Seed pinning is solver-only, and that is a Z3 limitation

`random_seed` is set on the `Z3_solver` used by `joint-solver-0.1`. It is **not**
set on the `Z3_optimize` used by `joint-solver-0.2`, because Z3 4.15.2 has no
seed parameter at optimize scope: `opt_params.pyg` defines only
`timeout`/`rlimit` (plus engine selection), and `opt::context::collect_param_descrs`
adds nothing else. An unknown name is not ignored — `params::validate` throws,
which surfaces as `Z3_EXCEPTION` and fails the whole solve.

There is also no bare `random_seed` module parameter at all; it exists only as
`smt.random_seed`, `sat.random_seed` and `sls.random_seed`. The bare name
validates on a solver because `smt_params_helper` registers it with `smt` module
metadata and `param_descrs::get_kind_in_module` accepts either spelling.

Pinning does not change the search — Z3 already defaults those seeds to 0. It
defends the pin against an ambient override, since `gparams` and the `Z3`
environment variable are process-wide and are read when a context is built. It
is applied per solver object rather than through `Z3_global_param_set`, which
would be a data race if the compiler ever solves on more than one thread (the
cache is already mutex-protected, which anticipates that).

Residual gap: an ambient `Z3=sat.random_seed=...` could still perturb the
`joint-solver-0.2` search. Closing that would require a process-global
parameter, which is a worse trade.

## Telling the two budgets apart

`budget_exhausted` on an `inconclusive` response is `"rlimit"`, `"walltime"` or
`"none"`. `rlimit` exhaustion is deterministic and reproducible — a property of
the problem, answered by raising the budget or simplifying the model. `walltime`
exhaustion is an environment signal and, with `rlimit` binding, should
essentially never appear. Neither is ever reported as `infeasible`;
`proven_unsat` stays false.

Z3 cannot be asked which one fired. `smt::context::get_cancel_flag` records the
same `CANCELED` failure for a wall-clock cancel and for resource exhaustion, so
`reason_unknown` reads `"canceled"` either way. `RESLIMIT_EH_CALLER` — the enum
that would have distinguished them — is never raised anywhere in 4.15.2, and
`smt::failure::RESOURCE_LIMIT` is never assigned, so `"max. resource limit
exceeded"` and `"(resource limits reached)"` are both unreachable through
`reason_unknown`.

So the caller attributes the stop from its own deadline. This is sound because
`remainingTimeoutMs` rounds the remaining time **up**: Z3's timer can then only
fire at or after the deadline, so a stop observed before the deadline cannot be
the wall clock's doing.

`stats.rlimit_used` reports what the search actually cost, read from the
context's `reslimit` counter (`"rlimit count"` in solver statistics). The
counter lives on the context and is shared with any `Z3_optimize` built on it,
which is how the 0.2 path reports it despite `Z3_optimize_get_statistics`
omitting the key. Reporting it on every outcome also sharpens the determinism
tests: two runs that reach the same schedule by different search paths differ in
`rlimit_used` and are caught.

## Calibration

Measured with `--gtest_filter=*DISABLED_ReportRLimitCalibration` on a
devserver, against the DDGs committed under
`third_party/tlx/tools/sched2tlx/examples/*/ddg.json` (`loops[].ddg`), which are
the real inputs the compiler produces. Buffers are omitted from those fixtures —
the committed DDG does not carry them — so the figures slightly under-count.

| Real loop | Nodes / edges | Status | `rlimit_used` |
| :--- | :--- | :--- | ---: |
| case1_simple_gemm, inner | 5 / 5 | ok | 119,277 |
| case2_persistent_gemm, outer | 16 / 15 | inconclusive | 10,532,353 |
| case3_FA, inner | 31 / 40 | inconclusive | 16,120,168 |

The hand-written fixtures in the test file are one to two orders of magnitude
smaller and are not a useful calibration anchor on their own:

| Fixture | `rlimit_used` |
| :--- | ---: |
| v01/feasible | 8,379 |
| v01/infeasible | 1,133 |
| v01/composite | 4,793 |
| v01/depth | 26,386 |
| v02/partition | 1,297 |
| v02/blocking | 3,913 |
| v02/joint | 3,366 |
| v02/cross_issue | 2,699 |

`kDefaultRLimit = 10000000` is ~84x the largest solve that currently succeeds
(119,277). On the measurement host Z3 sustained roughly 130k–250k units/s, so
10M units is about 40–80 s of search; the 180 s wall clock therefore only takes
over on a host more than ~2.3x slower than that one.

`FixtureCorpusFitsFarInsideTheDefaultBudget` asserts 8x headroom against the
budget the response reports rather than against a copy of the constant, so the
check cannot drift out of sync with the value it guards.

## Known limitation: the solver does not scale to the larger real loops

Only the 5-node `case1_simple_gemm` inner loop solves. The 16-node persistent
GEMM outer loop and the 31-node Flash Attention inner loop both exhaust 10M
units without an answer. This is not a regression introduced here — they did not
solve under the old wall-clock budget either — but it is now **observable and
deterministic** rather than machine-dependent: both stop on `rlimit` and report
`budget_exhausted: "rlimit"` at exactly the same point every run, which
`UnsolvableRealLoopStopsAtTheSamePointEveryRun` pins.

The cost driver is the pairwise modular-exclusion constraint between every two
nodes sharing a hardware pipeline, which is quadratic in the per-pipeline node
count and uses `mod`, i.e. nonlinear integer arithmetic. Flash Attention's 17
CUDA nodes alone produce 136 such constraints.

Raising `kDefaultRLimit` would not fix this — the growth is far steeper than any
budget worth spending in a compiler. It needs a cheaper encoding.

## Re-calibrating after a Z3 upgrade

Z3's work-unit accounting is internal and is not stable across releases, so
`kDefaultRLimit` is calibrated for one vendored version
(`third-party/z3/4_15_2`). After an upgrade:

```
buck2 build @mode/opt -m ovr_config//triton:beta -c fbcode.nvcc_arch=h100a \
  fbsource//third-party/triton/beta/triton:cpp_unittest/Hopper/modulo_schedule_test.cpp \
  --show-full-output
<binary> --gtest_also_run_disabled_tests \
  --gtest_filter=*DISABLED_ReportRLimitCalibration
```

Update the table above and `kDefaultRLimit`, keeping a comfortable multiple of
the largest solve that still succeeds, and re-check that the wall clock stays
several times longer than the budget's typical duration.

## Follow-up

`budget_exhausted == "rlimit"` currently falls back silently, which is right for
a feature-flagged optional pass and wrong for a golden suite — a silent fallback
is exactly how an `rlimit` set too low becomes an invisible performance
regression. Routing it to a hard error belongs with the `strict-error` mode
contemplated for Diff 11 (`DiffSplitPlan.md`), which does not exist yet. The
attribution field is what makes that wiring a one-line change, and it makes the
condition observable in the meantime. This also promotes `strict-error` from
optional to required in the Diff 11 scope.
