# InterleaveTMem and AutoWS Scheduling Design

## Context

This note summarizes the discussion around four related changes:

- D117364688, **Remove InterleaveTMem rollback, gate at one load**
- D117430071, **Unify AutoWS barrier wait locations**
- `1a84da61e`, **Prioritize TMEM operands in AutoWS compute**
- `9d0cef208`, **Elide redundant FA P publication waits**

The original `InterleaveTMem` implementation came from OSS Triton. Meta later
extended it for AutoWS, WS barriers, and subtiled epilogues. D117364688 moves
the implementation back toward the simpler OSS policy, while the other three
changes address AutoWS-specific scheduling and barrier behavior.

The goal is to preserve all four optimizations while giving each pass one
well-defined responsibility.

## The Four Scheduling Questions

The discussion identified four distinct questions:

1. How should independent work be placed between an asynchronous MMA and the
   `tmem_load` that waits for its result?
2. How should split TMEM loads and their consumer chains be ordered to avoid
   overlapping register live ranges?
3. When should related barrier waits be co-located so PTXAS sees a better
   scheduling region?
4. What should determine tile and subtile order?

These are related, but they are not the same optimization:

- Question 1 is generic TMEM latency scheduling.
- Question 2 is AutoWS consumer and subtile scheduling.
- Question 3 is barrier-placement profitability.
- Question 4 belongs primarily to the kernel or data-partitioning layer.

## D117364688: Generic TMEM Latency Scheduling

### Purpose

`ttng.tc_gen5_mma` executes asynchronously. A later `ttng.tmem_load` cannot
complete until that MMA has completed. Moving independent work between the MMA
and the TMEM load can hide this latency.

Motivating HSTU-style example:

```text
Before:

QK = MMA(...)
qk = tmem_load QK
P = softmax(qk)
compute V address
PV = MMA(P, V)

After:

QK = MMA(...)
compute V address
qk = tmem_load QK
P = softmax(qk)
PV = MMA(P, V)
```

D117364688 makes `InterleaveTMem` sink every TMEM load greedily until a real
legality boundary. It also processes blocks with a single movable TMEM
operation and removes the old overlap-profile rollback.

### FA backward regression

T286648295 shows that the regression is not caused by rollback removal. The
regressing change is replacing the bounded split-load policy with greedy
sinking.

Measured behavior on the production FA backward 2-CTA shape:

| Configuration | Latency |
|---|---:|
| Pre-change InterleaveTMem | about 4.11 ms |
| D117364688 + D117430071 | about 4.73 ms |
| InterleaveTMem disabled on either tree | about 4.74 ms |

Thus the new pass is not worse than disabling InterleaveTMem; it removes the
approximately 13% benefit delivered by the old subtile schedule.

The final TTGIR has the same operations, buffers, task partitions, and register
budget. Only operation placement changes. The slower placement introduces 32
local loads, 40 local stores, and a 160-byte stack frame, whereas the faster
placement has no spills.

The old output keeps complete subtile work close together:

```text
tmem_load[0] -> convert[0] -> co-operand[0] -> consume[0]
tmem_load[1] -> convert[1] -> co-operand[1] -> consume[1]
...
```

The greedy output can prepare several co-operands before the TMEM loads,
keeping those values live simultaneously and causing spills.

### Intended scope

`InterleaveTMem` should have one primary objective:

> Maximize useful producer-to-load distance while respecting aliasing, memory
> effects, dominance, and synchronization constraints.

It should not also own AutoWS subtile grouping, operand ordering, barrier
elimination, and PTXAS-oriented wait placement.

## `1a84da61e`: AutoWS Consumer Operand Ordering

### Purpose

Code partitioning can place an SMEM consumer channel before a TMEM operand even
when both feed the same operation:

```text
SMEM wait/load/broadcast
TMEM wait/load
add(TMEM, broadcast)
```

The expanded SMEM value then remains live across the TMEM wait and load. The
commit moves the complete SMEM channel after the TMEM channel:

```text
TMEM wait/load
SMEM wait/load/broadcast
add(TMEM, broadcast)
```

This controls which operand is live while the other is materialized and can
eliminate substantial register spilling.

### Interaction with D117364688

The current implementation runs `prioritizeTMemOperand` before the general
InterleaveTMem sinking phase. With D117364688, the later greedy sink can move
the TMEM load through the reordered SMEM chain and undo this optimization.

This logic should therefore be extracted into an AutoWS-specific consumer
ordering pass and run after generic InterleaveTMem.

The pass should eventually operate on complete consumer bundles, including
per-subtile co-operands, rather than recognizing only a narrow
`local_load`-based pattern.

### Required safety work

The current implementation should not land unchanged:

- The acquire and release must be proven to belong to the `local_load` channel.
  Selecting the first constrained arrive after the load can accidentally select
  a foreign channel and move the real release before the read.
- Every moved pure operation must form a closed use chain. Moving a definition
  while leaving an earlier user behind can violate dominance.
- Profitability should eventually compare register footprints and chain costs,
  rather than relying only on structural shape.

## D117430071: WS Barrier Location Unification

### Purpose

Sometimes there is too little useful work between two waits to justify keeping
them separate. Co-locating them can reduce register pressure and give PTXAS a
larger scheduling region.

Motivating addmm example:

```text
Before:

wait bias-ready
load/convert/broadcast bias
wait accumulator-ready
tmem_load accumulator
add

After:

wait bias-ready
wait accumulator-ready
load/convert/broadcast bias
tmem_load accumulator
add
```

The transformation gives up overlap between bias preparation and MMA
completion, but avoids keeping a large broadcast value live across the
accumulator wait. The measured addmm case removes all local-memory spills.

The pass uses a minimum broadcast footprint to restrict the transformation to
cases where the register benefit is likely to dominate the lost overlap.

### Intended scope

This pass should move waits only. It should not decide TMEM load placement or
move entire consumer channels.

It should run after all TMEM and AutoWS consumer-order transformations so that
it sees their final placement. It should also have an independent control knob;
an old InterleaveTMem barrier-reordering knob should not silently disable it.

## `9d0cef208`: Redundant FA P Publication Wait

### The synchronization graph

FA forward aliases P onto the QK TMEM allocation:

```text
GEMM task:     QK_MMA(i) ----------------> PV_MMA(i)
                   |                          |
                   | QK_FULL                  | P consumed
                   v                          v
Softmax task:  QK_load(i) ---------------> P_store(i)
```

Across iterations, the existing edges imply:

```text
PV_MMA(i)
  -> QK_MMA(i+1)       same GEMM-task program order
  -> QK_load(i+1)      QK_FULL channel
  -> P_store(i+1)      same softmax-task program order
```

Therefore the explicit P-empty producer-acquire before `P_store(i+1)` is
redundant for hardware correctness.

### Why the current change detects early and removes late

Code partitioning is the natural place to prove redundancy because it still
has:

- logical channel endpoints;
- reuse-group membership and physical TMEM overlap;
- producer and consumer task identities;
- loop and cadence information;
- same-task program order.

However, code partitioning is followed by token lowering, loop scheduling,
software-pipeline lowering and expansion, and late TMEM reordering. Those
passes do not reconstruct the transitive cross-iteration QK/P proof. They see
the concrete wait as a side-effecting ordering boundary.

Simply omitting the wait in code partitioning therefore changes the legal
schedule selected downstream. Empirically, early removal did not reproduce the
fast final IR, while preserving the wait through scheduling and deleting it
after instruction placement did.

The current implementation consequently treats the wait as a temporary
schedule dependency:

```text
Code partitioning:
  prove the hardware wait is redundant
  emit the wait and mark it

Late InterleaveTMem:
  after operation order is fixed, erase the marked wait
```

### Can the wait instead be removed in code partitioning?

Yes. The late annotation is not fundamental. Since the wait is semantically
redundant, failing to reconstruct the best schedule should affect performance,
not correctness.

The first step should be a pass-by-pass ablation:

1. Omit only the P producer-acquire wait in code partitioning. Initially retain
   the existing PV completion barrier so that only one variable changes.
2. Compare IR after code partitioning, loop scheduling, pipeline expansion,
   and InterleaveTMem against the mark-then-erase reference.
3. Repair the first pass at which the desired order diverges.

Possible outcomes:

- If loop scheduling diverges first, model the required P publication schedule
  directly through stage/cluster placement or an explicit scheduling edge.
- If pipeline expansion diverges first, repair the pre-expansion schedule
  rather than pattern-matching expanded copies.
- If only InterleaveTMem diverges, recover the placement in the new AutoWS
  consumer-order pass.
- If only conversion placement differs, handle it at the kernel or
  data-partitioning level.

The preferred incremental solution is early wait removal plus a sufficiently
strong post-Interleave consumer-order pass. This eliminates the annotation
entirely.

### If an ordering edge must survive scheduling

If downstream scheduling genuinely requires the edge, the clean abstraction
is a first-class schedule-only dependency, for example:

```text
requiresSchedulingOrder = true
requiresHardwareWait = false
```

or a typed `nvws.schedule_dependency` operation.

Such an edge would be consumed by scheduling and lower to no hardware
instruction. Carrying an ordering fact across passes is normal compiler design;
encoding it as a real hardware wait with an ad hoc deletion attribute is the
fragile part.

A more invasive alternative is delaying token/barrier lowering until after
scheduling, but that would require substantial changes to specialization,
barrier capture handling, phase calculation, and software pipelining.

### Separate kernel-level change

The `p0.to(dtype)` and `p1.to(dtype)` source movement bundled into `9d0cef208`
is tile/consumer ordering, not barrier elimination. It should be split into a
kernel or data-partitioning change and measured separately.

## Proposed Pass Pipeline

```text
Code partitioning
  - construct channels and reuse groups
  - prove any redundant synchronization relationships
  - preferably omit redundant hardware waits
  - otherwise emit a typed schedule-only dependency

Token lowering and loop scheduling
Software-pipeline lowering and expansion

InterleaveTMem
  - generic asynchronous MMA -> TMEM-load latency scheduling
  - D117364688

OptimizeWSConsumerOrder
  - order complete AutoWS operand channels
  - serialize profitable per-subtile consumer bundles
  - generalized and safety-fixed form of 1a84da61e

FinalizeWSScheduling
  - erase schedule-only dependencies, if any remain
  - perform no profitability-driven instruction movement

UnifyWSBarrierLocations
  - co-locate remaining real waits when profitable
  - D117430071

ReduceDataDuplication
ReorderInstructions
```

If early removal plus `OptimizeWSConsumerOrder` reproduces the desired IR,
`FinalizeWSScheduling` is unnecessary for the P-publication case.

## Responsibility Summary

| Component | Responsibility | Must not own |
|---|---|---|
| `InterleaveTMem` | Hide asynchronous MMA latency by sinking TMEM reads | AutoWS subtile policy, wait deletion, PTXAS wait grouping |
| `OptimizeWSConsumerOrder` | Control complete consumer-bundle order and register liveness | Hardware synchronization elimination |
| `FinalizeWSScheduling` | Remove typed schedule-only edges | Profitability heuristics |
| `UnifyWSBarrierLocations` | Trade small amounts of overlap for register/PTXAS benefits | Moving loads or complete channels |
| Kernel/data partitioning | Choose semantic tile order and expose intended conversion grouping | Hardware barrier placement |

## Suggested Landing Strategy

1. Update D117364688 so its generic latency optimization does not erase the
   known-good FA subtile behavior. A low-risk bridge is to retain the bounded
   policy for grouped split loads while using greedy sinking for single loads.
2. Extract and repair `1a84da61e` as `OptimizeWSConsumerOrder`, running after
   generic InterleaveTMem.
3. Test early P-wait removal in code partitioning against the current
   mark-then-erase reference at every scheduling boundary.
4. If the new consumer-order pass recovers the final IR and performance, drop
   the annotation from `9d0cef208`. Otherwise replace it with a typed
   schedule-only dependency and a dedicated finalization pass.
5. Run D117430071 after all load/channel scheduling and after any schedule-only
   dependency has been eliminated.
6. Split the FA source-level conversion placement from the synchronization
   change so each performance effect can be measured independently.

## Validation Matrix

The combined design should preserve four independent wins:

| Workload | Property to preserve |
|---|---|
| Ragged HSTU | Independent address work remains between MMA and `tmem_load` |
| FA backward 2-CTA | Subtile consumer chains remain localized; zero spills |
| Addmm | Wait unification removes the large-broadcast spill |
| FA forward 2-CTA | Redundant P publication wait is absent in final IR and the approximately 14% win remains |

For FA forward wait removal, compare these variants:

1. Current marked wait plus late erasure.
2. Early removal with no repair.
3. Early removal plus `OptimizeWSConsumerOrder`.
4. Variant 3 plus D117430071.

For each variant, inspect pass-level IR, final TTGIR operation order, allocated
registers, stack size, local load/store counts, correctness, and latency.

## Open Questions

- Can final-IR structure recover the desired P schedule reliably enough to
  eliminate the temporary dependency entirely?
- What is the smallest stable definition of a subtile consumer bundle?
- Should consumer-order profitability use elements per thread, bit width,
  estimated live-range overlap, or a combination?
- Does removing the unused PV completion arrival and barrier allocation provide
  an additional win? The current proven change removes only the wait, so this
  should be a separate ablation.
- Should `UnifyWSBarrierLocations` run before or after generic
  `ReorderInstructions`? Its final position should be selected by checking
  whether generic reordering invalidates its profitability assumptions.

## Focused Triage and Design Plan for `9d0cef208`

### Goal

Determine whether the redundant P-publication wait can be removed directly in
code partitioning and whether the fast schedule can then be recovered without
carrying `ttng.redundant_publication_wait` through later passes.

### Phase 1: Separate the two changes in the commit

`9d0cef208` bundles two optimizations:

1. Compiler-side P-publication wait elimination.
2. Kernel-side early `p0/p1.to(dtype)` placement.

Benchmark four variants:

| Variant | Wait change | Conversion placement |
|---|---|---|
| A | None | Original |
| B | None | Early conversion |
| C | Late wait removal | Original |
| D | Late wait removal | Early conversion |

This establishes how much of the reported approximately 14% improvement comes
from each change.

Use the pinned FA-forward 2-CTA production configuration: FP16, `B=4`, `H=48`,
`N=4096`, `D=128`, and `BLOCK_M=256`.

### Phase 2: Compare early and late wait removal

Create four compiler variants:

1. **Reference:** emit and tag the wait in code partitioning, then erase it
   after InterleaveTMem.
2. **Early removal:** omit only the producer-acquire wait in code partitioning.
3. **Full empty-edge removal:** omit the wait and the corresponding unused PV
   completion arrival.
4. **No removal:** retain the wait through final lowering.

Initially, variant 2 should retain the completion arrival. This isolates the
scheduling effect of the wait from changes to the barrier lifecycle and MMA
completion behavior.

For every variant, dump IR after:

```text
doCodePartition
doTokenLowering
doLoopSchedule
cleanupWarpSpecializedLoops
software-pipeline expansion
InterleaveTMem
final TTGIR
```

The first question is where early removal first changes operation order or
stage/cluster assignment.

### Phase 3: Classify and repair the first divergence

#### Divergence in `doLoopSchedule`

The wait is acting as a scheduler dependency. Check whether affected operations
receive different:

- `loop.stage`;
- `loop.cluster`;
- ordering within the softmax partition;
- prologue, steady-state, or epilogue placement.

Try expressing the desired publication schedule directly:

```text
exp2
-> f16 conversion
-> P publication
-> independent row reduction
```

Possible mechanisms include an explicit scheduler edge, a priority rule for a
value published to a downstream MMA, or stage/cluster assignments that
`scheduleLoops` preserves.

#### Divergence during software-pipeline expansion

Repair the pre-expansion schedule. Avoid matching or rearranging individual
expanded copies, because their prologue/steady-state/epilogue structure is a
derived representation.

#### Divergence only in `InterleaveTMem`

Remove the wait early and recover the placement in the proposed
`OptimizeWSConsumerOrder` pass.

The pass should recognize a general publication pattern:

```text
TMEM load
-> compute value
-> conversion
-> TMEM store consumed by a downstream MMA
```

It should prioritize publication over an independent reduction when doing so
starts downstream tensor-core work sooner or reduces register pressure.

#### Divergence only in final generic reordering

Add the rule to the narrowest final scheduling pass. Do not preserve a fake
hardware wait through the entire compiler for a change introduced only by a
late generic transform.

### Phase 4: Final-TTGIR ablations

Use `ir_override` to separate the effects of individual operation movements.
All variants use the `preserve` oracle because numerical semantics should not
change.

| Variant | Hypothesis |
|---|---|
| Early-removal final TTGIR | Confirms whether the regression follows emitted IR |
| Reference order applied to early-removal IR | Tests whether ordering alone recovers performance |
| Restore only conversion placement | Measures the source/kernel component |
| Restore only P-store placement | Tests publication latency |
| Restore only reduction placement | Tests publication-versus-reduction ordering |
| Remove the paired completion arrival | Measures the cost of the remaining unused barrier endpoint |

For every variant, record:

- correctness;
- latency;
- allocated registers per thread;
- stack bytes;
- local load/store counts;
- relevant SASS instruction order.

Keep native and self-override controls beside every cross-override result so
the override mechanism itself remains validated.

### Phase 5: Select the design

#### Outcome A: A post-scheduling heuristic recovers the result

Use:

```text
CodePartition
  prove redundancy and omit the hardware wait

InterleaveTMem
  generic MMA-latency scheduling

OptimizeWSConsumerOrder
  recover the profitable P-publication order
```

No cross-pass annotation is needed.

#### Outcome B: `scheduleLoops` requires an explicit dependency

Introduce a first-class schedule-only relationship:

```text
requiresSchedulingOrder = true
requiresHardwareWait = false
```

Prefer a typed NVWS operation, channel property, or scheduler edge over a
discardable attribute on a real `wait_barrier`. Scheduling consumes the edge;
hardware lowering emits nothing.

#### Outcome C: Several downstream passes require the relationship

Retain delayed elimination, but formalize it:

- represent the relationship as a typed schedule-only dependency;
- remove it in a dedicated `FinalizeWSScheduling` pass;
- do not make InterleaveTMem responsible for deleting it.

This preserves the required cross-pass contract without representing a
non-hardware dependency as hardware synchronization.

### Integration with the other changes

Hold D117364688 and D117430071 constant during the initial `9d0cef208`
investigation. After choosing the mechanism, validate this combined order:

```text
InterleaveTMem                  D117364688
OptimizeWSConsumerOrder        repaired 1a84da61e
FinalizeWSScheduling           only if schedule-only edges remain
UnifyWSBarrierLocations        D117430071
generic instruction rewriting
```

### Exit criteria

The no-annotation design is successful if it:

- matches the reference P-publication/reduction order in final TTGIR;
- preserves the FA-forward performance improvement within measurement noise;
- passes repeated 1-CTA/2-CTA and persistent/non-persistent correctness runs;
- preserves required accumulator and reuse waits;
- introduces no FA-backward, HSTU, or addmm regression;
- requires no kernel-name or source-location matching.

If it fails, use a typed schedule-only dependency rather than the current ad
hoc wait attribute.

## Executed `9d0cef208` Triage (2026-08-31)

### Result

On the current combined pipeline, the redundant P empty edge can be removed
directly in code partitioning.  The historical need to carry a real
`wait_barrier` as a scheduling proxy could not be reproduced.  No downstream
repair heuristic or cross-pass annotation is needed.

The selected implementation keeps the existing structural proof in
`WSCodePartition`, but when the proof succeeds it emits neither endpoint of the
redundant empty edge:

```text
whole-overwrite reuse proof succeeds
  -> no P producer-acquire wait
  -> no matching PV completion arrival
  -> keep the PV MMA asynchronous for its other channels
```

`InterleaveTMem` therefore no longer removes
`ttng.redundant_publication_wait`, and that attribute is no longer produced.

### Controlled performance results

The benchmark was the pinned 2-CTA persistent FA-forward production shape on a
GB200: FP16/BF16 compute path, `B=4`, `H=48`, `N=4096`, `D=128`,
`BLOCK_M=256`, `AUTOWS_FWD_CLC=1`, and `AUTOWS_FWD_SMEM_BUDGET=132000`.

| Conversion placement | Empty-edge handling | Latency | Throughput |
|---|---|---:|---:|
| Early | Late wait erase (commit reference) | 1.269216 ms | 1299.44 TFLOP/s |
| Early | Early wait erase, retain completion arrival | 1.270176 ms | 1298.46 TFLOP/s |
| Early | Keep wait | 1.418720 ms | 1162.50 TFLOP/s |
| Original | Late wait erase | 1.274368 ms | 1294.18 TFLOP/s |
| Original | Early wait erase, retain completion arrival | 1.275328 ms | 1293.21 TFLOP/s |
| Original | Keep wait | 1.424768 ms | 1157.57 TFLOP/s |

Interpretation:

- Removing the wait accounts for about a 10.5% latency reduction, or an 11.8%
  throughput increase, in this screen.
- Early versus late wait removal differs by only 0.08% and emits byte-identical
  PTX, SASS, and cubin.
- Moving `p0/p1.to(dtype)` earlier is about 0.4% in both the kept-wait and
  removed-wait comparisons, within this run's noise.  It is independent of the
  main wait-elision benefit.

The full-edge-removal variant averaged 1.294464 ms over five alternating runs,
versus 1.302432 ms for late erasure (0.61% faster).  Individual benchmark runs
reported 7-11% dispersion, so this is evidence of no regression rather than a
claim of a separate speedup.  A final forced recompile of the selected,
annotation-free implementation measured 1.273280 ms / 1295.29 TFLOP/s.

### IR evidence

The reduced `reuse_group_2buffer_fwd.mlir` fixture shows the first and only
intentional divergence at code partitioning:

- old/reference path: two marked P producer-acquire waits;
- direct-removal path: no P producer-acquire waits;
- after `InterleaveTMem`: both paths have the same relevant P operation order.

For the generated production kernel:

| Variant | Final `wait_barrier` count | Final barrier allocations | SASS local loads/stores |
|---|---:|---:|---:|
| Late erase | 77 | 70 | 18 / 9 |
| Early wait erase | 77 | 70 | 18 / 9 |
| Full edge removal | 77 | 66 | 18 / 9 |
| Keep wait | 81 | 70 | 22 / 13 |

The late-erasure and early-wait-erasure binaries are byte-identical.  Full edge
removal preserves the same final P sequence
`exp2 -> reduction -> truncation -> split -> P stores`, while deleting four
unused barrier allocations produced by the two logical P channels after loop
expansion.  All variants retain `.maxnreg 128`; the kept-wait variant alone
adds four local loads and four local stores.

### Design conclusion

The right ownership boundary is code partitioning because it still has the
reuse group, precise TMEM ranges, task identity, program order, and loop
identity needed for the redundancy proof.  Once that proof succeeds, emitting
a hardware synchronization edge and asking a later unrelated pass to erase it
is unnecessary on the current pipeline.

No post-hoc heuristic was added.  A heuristic would have to rediscover an
already-proven semantic fact from degraded, expanded IR and would still risk
moving unrelated reductions or stores.  If a future scheduler genuinely needs
this ordering, the fallback design should be a typed schedule-only edge
consumed by scheduling—not an annotation on a real hardware wait.

### Validation

- Required native rebuild: passed.
- `reuse_group_2buffer_fwd.mlir`: passed, including a new check immediately
  after code partitioning and the existing post-`InterleaveTMem` check.
- Full `test/Hopper/WarpSpecialization` lit directory: 136 passed, 9 expected
  failures, 0 unexpected failures.
- FA forward matrix: 11 passed across 1-CTA/2-CTA,
  persistent/non-persistent, and the 2-CTA persistent multi-iteration case.
- Cross-kernel smoke tests: FA backward three-buffer TMEM reuse, HSTU
  self-attention forward, and the DP2 persistent addmm regression shape all
  passed.
- `clang-format --dry-run --Werror` and `git diff --check`: passed.
- `pre-commit run --all` could not initialize its GitHub-hosted hook because
  the environment's CONNECT proxy returned HTTP 403; no formatter finding was
  reported before that infrastructure failure.
