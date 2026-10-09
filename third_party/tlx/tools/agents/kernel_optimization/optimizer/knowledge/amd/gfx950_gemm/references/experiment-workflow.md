# gfx950 GEMM Experiment Workflow

Use this workflow to turn a performance request into attributable evidence.
Every stage produces a recorded artifact or a decision. Do not advance through
a failed gate by changing the success criterion after seeing the result.

## 1. Freeze The Request

Create one case record for every target workload. Include the complete workload
identity from the measurement reference, plus:

```text
why the workload matters
protected incumbent and fallback
correctness oracle and tolerance
primary timing regime and noise margin
minimum per-case ratio and aggregate rule
authorized source, compiler, and dispatch scope
compile-time, workspace, and maintainability limits
```

State whether the request is an isolated kernel experiment, a production
dispatch promotion, a refactor expected to preserve performance, or a
cross-validation of the optimization workflow. They have different success
conditions.

For a production corpus, separate target cases, one-variable neighbors, old
regression cases, and blind holdouts. Keep holdout results hidden while choosing
the rule. Include a blind positive control with independently established
headroom when evaluating whether the workflow can discover optimizations.

## 2. Capture Ground Truth

Before editing, record for each protected case:

- public callable and selected provider;
- dispatch family, complete plan, compiler options, and launch grid;
- output/workspace ownership and auxiliary launches;
- correctness and raw timing samples;
- final MFMA/load/LDS forms and resource metadata; and
- external comparator identity when one is used.

Preserve an explicit callable or revision for the incumbent. After a dispatch
edit, resolve both paths again. A benchmark label can silently route both names
to the candidate and produce a false ratio of one.

For a rebase or consolidation, capture a machine-readable dispatch manifest
before and after. Route equality is necessary but not sufficient: compare final
artifacts and paired timing after the manifest passes.

## 3. Write A Headroom Verdict

Do not start from a favorite kernel mechanism. Write:

```text
incumbent latency distribution
strongest same-contract comparator or analytical bound
measurement noise floor
absolute and relative gap
one or more mechanisms capable of closing a gap of that size
```

If no gap larger than noise is known, use the case as a correctness and
regression guardrail. It cannot prove that a new optimization mechanism works.
If the external comparator has a different layout, allocation, preprocessing,
or numerical contract, it can inspire a hypothesis but cannot define the gap.

## 4. Build A Mechanism Card

Calculate inexpensive facts before compiling candidates:

```text
useful and padded MFMA work
macro tiles, workgroups, waves, CU rounds, and final-wave occupancy
K-loop trips, stage fill/drain fraction, and barrier density
compulsory bytes and expected reuse scope for each operand
accumulator, operand, future-bank, LDS, and workspace storage
launch, reduction, and epilogue cost relative to device time
```

Use these facts and profiles to classify the first-order limitation. Require
converging evidence when selecting an expensive direction:

| Suspected limit | Supporting evidence | Discriminating experiment |
|---|---|---|
| Sparse device work | Tile grid, CU rounds, idle final wave | Change ownership, tile, or K partition |
| Boundary waste | Useful/padded MFMA ratio | Try a legal irregular tile |
| Register pressure | VGPR/AGPR, scratch, lost residency | Shorten live state or change wave grid |
| Global memory/locality | Traffic bound, cache sensitivity, VMEM waits | Change mapping, reuse, or one operand policy |
| LDS/synchronization | LDS traffic, bank evidence, barriers, waits | Change staging or one pipeline boundary |
| Dependency latency | Load-to-use distance and per-wave trace | Expose independent work or deepen buffering |
| Machine ordering | Competitive structure but clumped final issue | Add a legal source window or tune scheduler |
| Fixed overhead | Device time near launch/epilogue floor | Simplify launch, prologue, dispatch, or stores |

Arithmetic intensity alone does not choose wave count, staging family,
persistence, or cache policy.

## 5. Record A Falsifiable Experiment

Use one structured record per candidate:

```text
Observation:
Hypothesis:
One mechanism changed:
Predicted artifact delta:
Predicted timing delta and affected regime:
Correctness risk:
Falsifier:
Source/compiler revision:
Resolved plan and artifact summary:
Correctness result:
Raw timing result:
Decision and residual bottleneck:
```

Change one mechanism at a time unless the components are inseparable for
correctness. A source cleanup, tile change, cache modifier, and scheduler change
in one patch do not produce reusable evidence.

Store exact shapes, dates, rejected configurations, and raw measurements in run
artifacts or case studies. Promote only the reusable boundary and decision rule
into this knowledge bundle.

## 6. Generate A Bounded Candidate Ladder

Select a small portfolio from the diagnosed mechanism rather than expanding
only the current favorite. Useful roles include:

- a low-overhead register path for short K;
- the strongest mature four-wave and higher-wave mappings;
- a geometry that reduces padded work or improves CU rounds;
- a resource variant that changes live state or occupancy;
- a locality or work-distribution variant;
- a legal source scheduling-window variant; and
- a configuration-scoped post-lowering schedule.

Order structural experiments by dependency:

1. output/K work distribution and device utilization;
2. orientation, macro tile, and boundaries;
3. MFMA ownership and wave grid;
4. register/LDS/direct-to-LDS movement, BlockK, and buffering;
5. persistence, Stream-K, LocalSplitU, mapping, and cache;
6. source-approved memory/compute scheduling windows; and
7. final machine scheduling.

This is not a requirement to try every item. Start at the first layer implicated
by evidence and stop when its hypothesis is falsified or the residual bottleneck
moves elsewhere.

## 7. Compile And Apply The Artifact Gate

Compile one discriminating case whose geometry truly exercises the proposal.
Before benchmarking, verify:

```text
MFMA opcode, logical K width, count, and ownership
global load form, width, count, and cache policy per operand
LDSDMA/direct-to-LDS, ds_read, and ds_write forms
wait, barrier, nop, and scheduling-group structure
VGPR, AGPR, LDS, scratch, and inferred occupancy
grid, waves/workgroup, persistence, and output-store width
resolved compiler options and compilation-cache identity
```

Reject immediately if the final binary spills, loses the intended resource
tier without a compensating mechanism, uses an unintended MFMA or load path,
fails to instantiate the claimed irregular fragment, or violates a pipeline
invariant. Source parameters do not prove the emitted mechanism.

## 8. Apply The Correctness Gate

Run a focused case that makes every changed fragment and path contribute. Then
expand across:

- multiple random seeds;
- partial M and N macro tiles;
- every advertised K tail and short trip count;
- alignment, nonzero offset, and stride boundaries;
- repeated launches for race-sensitive pipelines;
- cancellation and dynamic-range cases when reduction order changes; and
- fallback behavior outside the admitted contract.

A compiled candidate with one passing divisible shape is not a capability.
When direct-to-LDS or static subslices change, validate every source and
destination fragment rather than only the overall output shape.

## 9. Measure One Discriminating Case

Measure the candidate beside the saved incumbent in the same process and timing
regime. Warm all contestants, rotate order, retain raw samples, and confirm that
the callables remain distinct. Reject a clear loss without running the full
corpus.

For a close result, increase the odd sample count, repeat in independent
processes, and check absolute-latency sentinels. A result that changes sign is
unresolved. Keep the simpler incumbent.

## 10. Expand Evidence In Stages

For each surviving mechanism:

1. run several correctness seeds on the target;
2. screen one-variable neighbors with paired timing;
3. remeasure apparent wins and boundary cases in fresh processes;
4. run the full target and prior regression corpora;
5. reveal and run blind holdouts last; and
6. measure compile time, dispatch cost, workspace, and source complexity.

Stop at the first failed gate, record the falsifier, and revert an experimental
patch that is not part of the accepted result. Do not sweep a large corpus for a
candidate that already loses its discriminating case.

## 11. Derive A Promotion Rule

Explain the win through observable features such as useful/padded work, CU
rounds, aspect ratio, K depth, occupancy, reuse distance, or a verified
instruction schedule. Convert that explanation into a bounded predicate.

Test both endpoints, an interior point, and a negative neighbor across every
admitted boundary. Keep the previous dispatcher as fallback wherever evidence
is missing or unresolved. An exact-shape table can temporarily cache measured
plans, but it does not define the supported capability or the final generalized
rule.

Report:

```text
target and minimum per-case ratio
declared aggregate ratio and weights
win / parity / regression counts
raw-sample stability and noise margin
correctness coverage
compile-time, dispatch, launch, and workspace deltas
final artifact and resource deltas
```

Do not trade a known protected-case regression for a better mean unless the
acceptance policy was explicitly changed before evaluating the result.

## 12. Learn And Stop Honestly

After promotion, the winner becomes the new incumbent. Continue from its final
artifact and residual bottleneck rather than replaying a fixed checklist.

Valid outcomes include:

- a promoted performance improvement;
- a correctness-preserving refactor with measured parity;
- a dispatch fallback that removes a regression;
- a localized kernel/compiler capability gap with a reproducer;
- a rejected mechanism with a useful boundary; or
- no code change because every bounded hypothesis was falsified.

For workflow cross-validation, report separately whether the run protected
correctness, generated useful falsifiable hypotheses, and produced a repeatable
accepted speedup. If the final item is false, say that optimization generation
did not succeed; do not redefine a clean rejection as a performance win.
