# Contracted Schedule and Memory-Plan Search Plan

**Status:** implementation in progress. Milestone A's structural schedule
frontier and Milestone B's bounded SMEM depth frontier are implemented.
Conservative fixed-group search now covers staging, subtile, and cross-id reuse
plans. Composing schedule and memory candidates across compilation runs remains
open.

**Implemented first slice:** Contracted search now derives its lower II from
structural issue counts, uses those same quanta for dependence and reservation
checks, and reserves modeled latency only for ranking. The FA-backward lit
oracle pins the resulting target GEMM stage and modulo-order candidate.

**Implemented second slice:** II is part of candidate identity. A bounded
frontier reserves roughly half of top-K for evenly spaced feasible II values,
including the minimum and maximum when possible, and uses the other slots for
the globally best structural alternatives. This keeps rank 0 default-neutral
while ensuring larger-II schedules remain measurable.

**Implemented third slice:** Descriptor/TMA loads with a distance-zero path to
a GEMM are discovered structurally and appended to the schedule signature.
The search generates bounded topological variants that pull each load and its
address-dependency slice forward without violating DDG edges, and reserves one
top-K slot for a non-default load order. The FA-backward oracle now checks such
a load-order candidate directly. A one-MMA two-descriptor fixture additionally
checks both A-early/B-late and A-late/B-early orders, matching the scheduling
shape needed by D120426461.

**Implemented fourth slice:** `CopySolver` now returns bounded copy vectors.
The existing greedy result remains candidate zero, while complete memory-beam
leaves also enumerate the hard-floor vector and progressively deeper legal
neighbors. The plan beam validates, scores, and deduplicates those concrete
plans before global top-K selection. A two-operand oracle covers A2/B2, A3/B2,
and A2/B3 under a budget that admits exactly one additional copy.

**Implemented fifth slice:** unsupported SMEM grouping features no longer force
an immediate all-or-nothing fallback. The heuristic first establishes its
proven grouping and staging depths; fixed-group search preserves that plan as
rank zero, pins constrained groups, and enumerates only mutable singleton
operand depths. Cross-id `allocation.reuseTarget` sources remain distinct
logical blocks but are excluded from the budget because their backing is owned
by a pinned target block.

**Implemented sixth slice:** Contracted scheduling and memory planning append
independent JSON-lines records to `TRITON_WS_SEARCH_MANIFEST`. Schedule records
contain rank, selected state, II, load-order variant, and signature; memory
records contain the active schedule pick, memory rank and selected state, pool,
score, and block layout. This is the pass-boundary contract for an external
Cartesian-product harness; no in-process scheduler object is passed to the
memory planner. `python/triton/tools/autows_search.py` consumes this contract,
runs one compile/validation command per bounded schedule-memory pair, verifies
the selected ranks, and records exit status, elapsed time, and an optional
command-reported metric as JSON lines.

**Production-shaped D120 oracle:** An annotation-free bf16 fused RMSNorm + GEMM
with 128x128x128 tiles and eight-way output subtiling has been captured at the
pre-modulo and post-buffer-allocation boundaries. The pass-local scheduler test
shows that Contracted search retains the default A0/B0/MMA1 schedule plus the
three-stage A0/B1/MMA2 and A1/B0/MMA2 alternatives required to reproduce the
D120 experiment. The pass-local memory test confirms that the existing
heuristic produces A3/B2 on the same graph. Search-mode coverage on this fixture
now recognizes its eight separately materialized output-staging allocations as
one fixed subtile group, preserves A3/B2 at rank zero, and retains A2/B2 and
A2/B3 as the first two alternatives.

**Primary implementation areas:**

- `third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlan*`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlanner.cpp`

**Reference targets:**

- [D120426461](https://www.internalfb.com/diff/D120426461), fused RMSNorm +
  GEMM with asymmetric A/B buffering.
- Flash Attention backward configurations that currently use `tt.autows`
  schedule and memory-channel annotations.
- [Schedule and Memory-Plan Search — Status and Roadmap](https://docs.google.com/document/d/1o1EDMGq6tubP7ztYB2DiaMRPghFururlx_7beJu5DRc/edit?tab=t.0#heading=h.9r1bcip08bhi).

## 1. Objective

Build a general, search-based pipeline that can recover the performance of
the hand-annotated FA-backward schedules and D120426461 without exposing
operand-specific controls such as `lhs_buffer_depth` or `rhs_buffer_depth`.

The intended pipeline is:

```text
Contracted-graph structural SWP top-K
  -> selected loop.stage / loop.cluster / nominal II
  -> search-based SMEM/TMEM planning for that schedule
  -> selected buffer.copy / buffer.id / buffer.offset / memory space
  -> code partitioning and software-pipeline expansion
```

The compiler generates legal and structurally diverse candidates. Runtime
measurement, not an inaccurate compiler latency model, determines which
candidate is fastest.

### 1.1 Success criteria

1. No operand-specific depth options are added to `tl.range`.
2. D120426461's A2/B2, A3/B2, and A2/B3 alternatives are discoverable without
   source annotations; measured search selects A3/B2 on the reported shapes.
3. FA backward reproduces the best annotated schedule and memory plan without
   `tt.autows` stage/order/copy/id/offset annotations.
4. Search-off output remains byte-identical to the current default pipeline.
5. Every emitted candidate passes schedule, buffer-safety, SMEM, and TMEM
   validation before code generation.
6. Candidate identities do not depend on `lhs`/`rhs` names, a single-MMA loop,
   or a particular kernel.

### 1.2 Non-goals

- Do not make Rau, SMS, exhaustive, random, or Z3 scheduling part of this
  production search path.
- Do not use predicted latency as a correctness condition.
- Do not require one in-process solver to choose the globally fastest
  schedule-memory pair.
- Do not initially search CTA topology, warp-group count, or epilogue subtile
  factor.

## 2. Why the current paths are insufficient

### 2.1 Contracted SWP originally depended on modeled timing

Before this implementation started, `runContractedSearch`:

1. starts from `DataDependenceGraph::computeMinII()`;
2. uses contracted/result latency in earliest-cycle placement;
3. validates distance-zero edges using contracted latency;
4. ranks lower II and contracted critical-path cost first.

The first implementation slice now uses structural issue quanta for the II
lower bound, dependence admission, and resource reservations. Contracted
latency remains only as a ranking tie-breaker. The search also preserves II
diversity and branches over GEMM-reaching load order, completing Milestone A's
structural frontier.

### 2.2 Contracted candidate identity originally hid memory-relevant schedules

The original signature contained only `(GEMM stage, GEMM cluster)`. It now also
contains structurally discovered GEMM-reaching TMA-load stage/cluster pairs,
so load-order variants remain distinct. The reduced one-GEMM oracle proves that
A-early and B-early both reach top-K; the full fused RMSNorm fixture remains.

### 2.3 The memory beam now preserves SMEM depth diversity

Partial grouping states use `CopySolver::solve()` (the legacy greedy candidate)
for inexpensive ranking. At complete leaves, `CopySolver::enumerate()` returns
a bounded set of copy vectors. For SMEM, `SmemPacker::legalJoin()` remains false,
so blocks are singletons, but top-K can now contain distinct depth assignments
such as A2/B2, A3/B2, and A2/B3.

### 2.4 Schedule-derived buffer analysis does not reach WSMemoryPlanner

`ModuloSchedulePass.cpp` already contains cycle-level lifetime analysis:

```text
producerStart = earliest cycle at which storage is occupied
lastConsumerEnd = latest real consumer completion
lifetime = lastConsumerEnd - producerStart
numBuffers = floor(lifetime / II) + 1
```

It walks through transparent views and accounts for loop-carried dependence
distance. The result is emitted as `tt.num_buffers`, but WSMemoryPlanner does
not consume that attribute and independently overwrites physical allocation
decisions.

The calculated count is a stall-free preference, not universally a correctness
floor. Fewer copies may remain correct because empty/full barriers serialize
the producer. Hard copy-safety floors remain the memory planner's concern.

### 2.5 Removing every FA-backward annotation includes memory-space choice

The current memory planner runs after `doBufferAllocation`, when each channel
is already SMEM- or TMEM-backed. Removing memtype-only annotations therefore
requires either an earlier logical memory-plan phase or a general default rule
for fixed-versus-searchable memory spaces.

## 3. Design principles

### 3.1 Schedule first, memory second

Scheduling owns logical execution order. Memory planning consumes a selected
schedule and owns physical storage. The memory planner must not mutate the
selected schedule to manufacture a desired prefetch depth.

### 3.2 II is searched, not predicted

II remains a necessary coordinate for modulo scheduling, but the compiler does
not claim that the smallest modeled II is fastest. The structural contracted
search enumerates a bounded II range and retains candidates across that range.

### 3.3 Separate hard facts from estimates

Hard candidate-admission facts:

- dependence topology and loop-carried distance;
- operation pipeline kind;
- architectural issue/resource conflicts;
- stage-count bounds;
- memory-capacity and synchronization safety.

Optional ranking or diagnostic estimates:

- result latency;
- predicted critical path;
- predicted latency hiding;
- trip frequency.

An estimate may order otherwise-retained candidates but must not remove an
entire structural alternative from the measurable frontier.

### 3.4 IR is the pass boundary

No scheduler data structure is passed directly to WSMemoryPlanner. The selected
schedule is represented by existing IR metadata:

- `loop.stage`;
- `loop.cluster`;
- `tt.modulo_ii`;
- optionally a scheduler-derived buffer-demand hint for diagnostics/ranking.

Final `buffer.copy`, `buffer.id`, `buffer.offset`, and reuse attributes remain
owned by WSMemoryPlanner.

### 3.5 Default neutrality

- Search disabled: preserve current behavior.
- Search enabled with rank 0: preserve the existing default schedule and
  memory plan.
- Experimental candidates begin at rank 1.

## 4. Work plan

### Phase 0: Establish two annotation-free oracles

Create reduced fixtures and capture IR at these boundaries:

1. after initial scheduling;
2. after partition assignment;
3. after memory planning;
4. after code partitioning;
5. after software-pipeline expansion.

#### D120426461 example

Retain three configurations as the minimal search oracle:

| Candidate | Expected role |
|---|---|
| A2/B2 | default baseline |
| A3/B2 | positive target |
| A2/B3 | negative control |

Check descriptor-load and MMA stage/cluster assignments, final SMEM rings, and
barrier counts.

The scheduler half now has a reduced one-MMA/two-descriptor oracle in
`test/TritonGPU/modulo-schedule.mlir`; the A/B copy-depth alternatives remain a
Milestone B deliverable.

#### FA-backward example

Select at least one currently annotated configuration whose desired contraction
order is:

```text
stage 0: qkT, dpT, dv
stage 1: dk, dq

cluster(qkT) < cluster(dk) <= cluster(dq)
               < cluster(dpT) <= cluster(dv)
```

Record the current annotation-selected SMEM/TMEM plan, including any
`{dpT, dsT, dQ}` reuse relationship.

**Deliverable:** tests that initially prove the annotated oracle and later pass
with the annotations removed.

### Phase 1: Introduce a structural timing view for Contracted SWP

Split DDG timing information into:

```cpp
struct StructuralTiming {
  HWPipeline pipeline;
  int issueDuration;
};

struct EstimatedTiming {
  int resultLatency;
  int criticalPathCost;
};
```

The exact API may differ, but `runContractedSearch` must be able to use only
`StructuralTiming` for admission.

Compute a hard resource lower bound:

```text
resourceLowerBound =
  max over pipelines(sum(issueDuration) / numberOfUnits)
```

Do not use latency-derived RecMII as the lower bound in structural-search mode.
Retain recurrence topology as an ordering constraint. Nested scheduled loops
may retain their already-established inner II as a structural resource fact.

#### D120426461 example

The MMA and TMA issue resources constrain candidate placement, but a modeled
TMA completion latency cannot remove the A-early or B-early alternative.

#### FA-backward example

The number and issue cost of qkT/dk/dq/dpT/dv constrain TC occupancy. Their
predicted result latencies do not decide which two-stage assignment survives.

**Likely files:** `DataDependenceGraph.{h,cpp}`, `LatencyModel.*`,
`ExhaustiveScheduler.cpp`.

### Phase 2: Search a bounded II range

For structural Contracted SWP:

```text
lowerII = hard resource lower bound
upperII = explicit internal search cap
for II in [lowerII, upperII]:
    enumerate structural schedule candidates
```

The bound is a compiler/search configuration, not a kernel-facing operand
annotation. Cap total explored states independently from the II range.

When the feasible II count exceeds top-K, retaining every II is impossible.
Reserve a bounded quota for evenly spaced II representatives, always including
the minimum and maximum when the quota permits, then fill remaining top-K slots
by deterministic structural ranking. Do not let the lowest II evict every
larger-II candidate solely because it is numerically smaller.

#### D120426461 example

At each retained II, search both A-early/B-late and A-late/B-early load
placements around the fixed MMA schedule.

#### FA-backward example

Retain the known two-stage GEMM arrangement even if it appears at a larger II
than the model-preferred candidate.

**Likely file:** `ExhaustiveScheduler.cpp::runContractedSearch`.

### Phase 3: Add load-placement branching to Contracted SWP

Identify eligible load channels structurally:

```text
descriptor/TMA producer
  -> zero-cost transparent view chain
  -> one or more real consumers
```

After choosing GEMM stages, branch over legal producer stage/modulo-position
choices rather than greedily committing to one placement. Keep address and
descriptor dependency chains with their load.

Extend candidate identity from:

```text
(GEMM stage, GEMM cluster)
```

to:

```text
(II,
 GEMM stage/cluster signature,
 descriptor-load stage/cluster signature)
```

Use stable operation order or DDG node IDs, never operand names.

#### D120426461 example

The following must be distinct schedule candidates:

```text
S_A: A issued earlier than B
S_B: B issued earlier than A
```

#### FA-backward example

Schedules with the same qkT/dk/dq/dpT/dv placement but different q/k/v
prefetch placement must remain distinct.

**Likely files:** `ExhaustiveScheduler.cpp`, `ModuloSchedulePass.cpp`,
`ContractedGraphScheduler.md`.

### Phase 4: Replace score-only pruning with a structural frontier

At each beam boundary, retain candidates in this order:

1. the existing/default schedule;
2. at least one candidate per distinct GEMM-stage signature;
3. every legal single-load movement from those signatures, within a cap;
4. one or more representatives per II;
5. remaining slots by deterministic structural tie-breaks.

Optional model scores may order candidates within a bucket, but cannot delete a
whole signature bucket.

#### D120426461 example

Both A-early and B-early survive even if the model assigns them identical cost.

#### FA-backward example

The target contraction ordering survives alongside balanced and baseline
alternatives instead of depending on one latency-ranked beam path.

### Phase 5: Define the schedule-to-memory IR contract

For the model-independent path, the scheduler emits the selected
`loop.stage`, `loop.cluster`, and `tt.modulo_ii`. It does not emit authoritative
physical allocation decisions.

The existing cycle-level `computeBufferCount()` result may be emitted as a
non-binding hint, but it must not become a hard floor or remove memory-plan
candidates. If retained, distinguish it from the selected count, for example:

```text
tt.scheduled_buffer_demand = N   // optional, non-binding
buffer.copy = N                  // selected later by WSMemoryPlanner
```

The current `tt.num_buffers` meaning must be audited before reuse because the
ModuloScheduling path may budget-reduce `ScheduleBuffer::count` before
emission.

#### D120426461 example

The A-early schedule may carry a diagnostic preference for A3, but the memory
search still retains legal A2 and A3 allocations.

#### FA-backward example

Cycle-lifetime analysis can report pressure for q/k/v and intermediates, but it
does not pin physical IDs or reuse groups.

### Phase 6: Make CopySolver return multiple copy assignments

The single-result interface has been replaced:

```cpp
CopyMap solve(...);
```

with a bounded enumeration interface, conceptually:

```cpp
CopyMaps enumerate(..., unsigned limit);
```

For each block, enumerate from its hard correctness floor through the configured
maximum. Hard floors include cross-stage/release safety and rotating-entry
requirements. Predicted latency hiding is not a legality condition.

Implemented candidate generation order:

1. the legacy greedy vector (so `TOPK=1` remains neutral);
2. the hard-floor vector;
3. legal one-buffer `+1` neighbors from that floor within the bound;
4. larger edit-distance combinations;
5. deterministic tie-breaks.

The per-buffer ceiling is part of `BufferModel`: configured `num-buffers` for
SMEM and one for TMEM. A correctness floor may exceed this ceiling and remains
authoritative. Feasibility is monotone in depth, so an over-budget vector prunes
all descendants. The enumerator is bounded by the caller's requested `K`.

Reject candidates only for hard safety violations, impossible representation,
or memory-capacity overflow.

#### D120426461 example

For two independent descriptor-backed blocks, retain:

```text
A2/B2
A3/B2
A2/B3
A3/B3, if it fits and the candidate cap permits
```

#### FA-backward example

Enumerate q/k/v operand depths and legal staging depths without relying on the
stubbed `freq` or latency-benefit score.

**Implemented in:** `WSMemoryPlanSearch.h`, `WSMemoryPlanCopies.cpp`.

### Phase 7: Compose grouping and copy search in beamSearch

Complete grouping candidates now branch over the CopySolver results:

```text
grouping candidate
  x copy-count candidates
  -> StaticCopySafetyValidator
  -> pool budget check
  -> canonical deduplication
  -> top-K structural frontier
```

SMEM initially has singleton groups because `SmemPacker::legalJoin()` is false,
but it now gains real depth diversity. TMEM continues to search grouping and
placement with copies fixed to one until TMEM multi-copy becomes legal.

Partial grouping states deliberately continue to use only the legacy greedy
copy assignment for beam ranking. Expanding the copy Cartesian product only at
complete leaves keeps the grouping beam width bounded while still exposing the
depth alternatives to `TRITON_WS_MEM_PLAN_PICK`.

#### D120426461 example

The memory beam returns distinct A3/B2 and A2/B3 plans rather than one greedy
copy solution.

#### FA-backward example

TMEM search retains alternative legal placements and reuse groups, including
the known `{dpT, dsT, dQ}` case, without hand-written IDs or offsets.

**Implemented in:** `WSMemoryPlanSearch.cpp`, `WSMemoryPlanner.cpp`.

### Phase 8: Preserve search through unsupported grouping features

Subtiled regions, multi-store staging, and some annotations require grouping
rules outside the generic packer. Preserve them through fixed-grouping mode:

1. Run the existing heuristic planner to establish a legal grouping/reuse
   structure.
2. Import that structure as a fixed `wsplan::Plan`.
3. Enumerate legal copy vectors within it.
4. Preserve staging, `K | S`, encoding, and reuse synchronization invariants.

The conservative implementation performs steps 1–4 and preserves same-id
grouping, staging, subtile, atomic-broadcast, annotation, and cross-id reuse
invariants by pinning their heuristic depths. It searches only unconstrained
singleton operands and keeps the imported heuristic allocation at rank zero.
Cross-id reuse sources are marked non-owning for budget accounting, while both
ends of the alias remain fixed so search cannot invalidate the synchronization
proof.

#### D120426461 example

The eight-way output subtile keeps its existing staging plan while A/B depth
alternatives remain searchable.

#### FA-backward example

Specialized early-TMA staging and persistent reuse keep their proven grouping
and synchronization while unpinned operand depths are searched.

### Phase 9: Compose schedule and memory searches externally

One compilation selects one schedule rank, then builds memory candidates for
that schedule. A harness sweeps the bounded Cartesian product:

```text
for schedulePick in scheduleRanks:
    compile/dump memory candidates for schedulePick
    for memoryPick in memoryRanks(schedulePick):
        compile, validate, and measure
```

Emit a manifest containing:

- schedule rank and signature;
- II;
- memory rank and signature;
- SMEM bytes and TMEM columns;
- fallback/fixed-grouping reason;
- deduplication key.

Use generic schedule and memory picks; do not expose per-operand controls.

The compiler-side manifest is implemented via `TRITON_WS_SEARCH_MANIFEST`. The
external `python/triton/tools/autows_search.py` driver discovers ranks from that
manifest and sweeps the bounded Cartesian product. Its child command must
compile exactly one searched loop, return nonzero on validation failure, and
may print a numeric value selected by `--metric-regex` for performance ranking.
For example:

```shell
python python/triton/tools/autows_search.py \
  --schedule-topk=4 --memory-topk=3 \
  --metric-regex='latency_ms=([0-9.]+)' --results=/tmp/search.jsonl -- \
  python path/to/kernel_correctness_and_benchmark.py
```

#### D120426461 example

The measured set includes at least:

```text
(A-early/B-late schedule, A3/B2 plan)
(A-early/B-late schedule, A2/B2 plan)
(A-late/B-early schedule, A2/B3 plan)
```

#### FA-backward example

Each retained qkT/dk/dq/dpT/dv schedule is evaluated with its legal SMEM/TMEM
plans. Runtime measurement selects the pair rather than a compiler cost model.

### Phase 10: Remove manual annotations incrementally

1. Remove `stage` and `order` fields after Contracted SWP retains equivalent
   schedules.
2. Remove `copies`, `bufferId`, and `colOffset` after memory-plan search retains
   equivalent physical plans.
3. Remove memtype-only annotations after memory-space choice becomes a logical
   planning axis before physical buffer creation.

The third step likely requires splitting `doBufferAllocation` into logical
channel discovery and physical memory-space materialization.

#### D120426461 example

No `lhs_buffer_depth` or `rhs_buffer_depth` frontend options are needed.

#### FA-backward example

The final kernel retains only semantic loop configuration such as
`warp_specialize` and data-partition policy; it has no manually authored
schedule or channel-allocation string.

## 5. Validation plan

### 5.1 Unit and lit coverage

- Structural resource lower-bound computation without result latency.
- II frontier retention across multiple feasible IIs.
- Descriptor-load placement candidate identity and deduplication.
- CopySolver multi-result enumeration and deterministic ordering.
- Static copy-safety and budget rejection.
- Fixed-grouping search for subtiled and staging cases.
- Default-off and rank-0 neutrality.
- End-to-end candidate presence for D120426461 and FA backward.

### 5.2 Correctness

- Run the full WarpSpecialization lit suite.
- Run focused D120 fused RMSNorm + GEMM correctness for A2/B2, A3/B2, and
  A2/B3 candidates.
- Run the FA-backward annotated baseline and annotation-free selected candidate
  over the supported correctness matrix.
- Run racecheck/synccheck only when a candidate changes buffer reuse or barrier
  topology.

### 5.3 Performance

Performance testing is a final explicit validation step, not a compiler-model
gate.

- D120426461: recover the A3/B2 improvement across its reported shape set and
  preserve A2/B3 as the negative control.
- FA backward: match or beat the best annotated configuration over the selected
  representative shapes before deleting annotations broadly.
- Record compile-time growth and candidate counts as well as kernel latency.

## 6. Rollout and milestones

### Milestone A: structural schedule frontier

- Split hard issue facts from estimated latency.
- Search a bounded II range.
- Retain GEMM- and load-placement diversity.
- Demonstrate both D120 load-order alternatives and the FA-backward target
  contraction schedule under lit.

### Milestone B: real SMEM copy search

- Make CopySolver multi-result.
- Integrate copy candidates into beamSearch.
- Demonstrate A2/B2, A3/B2, and A2/B3 without operand annotations.

### Milestone C: fallback-compatible search

- Add fixed-grouping mode.
- Exercise D120's subtiled epilogue and FA-backward staging cases.

### Milestone D: annotation-free FA backward

- Remove schedule and physical-allocation pins.
- Add memory-space selection or a general memory-space heuristic.
- Validate correctness and performance before removing the annotated fallback.

### Milestone E: evaluation and default decision

- Measure the bounded schedule-memory Cartesian product.
- Choose search limits and fallback policy based on compile-time and runtime
  data.
- Keep the feature opt-in until representative workloads show stable wins.

Estimated engineering effort, assuming a working native build and B200 access:

| Milestone | Estimate |
|---|---:|
| A | 4–6 days |
| B | 3–5 days |
| C | 2–4 days |
| D | 5–8 days |
| E | 2–4 days |

The stages overlap, but the complete annotation-removal goal is approximately
three to five engineering weeks. Milestones A–C should be sufficient to test
the D120 hypothesis before committing to memory-space search.

## 7. Risks and decision points

### Candidate explosion

Schedule ranks multiplied by memory ranks can grow quickly. Use explicit global
caps, canonical signatures, and diversity buckets. Do not silently replace
diversity preservation with latency-score pruning.

### Structural schedules that stall heavily

Removing latency from admission intentionally permits schedules that hardware
scoreboarding or barriers serialize. Runtime measurement rejects them. The
search must nevertheless enforce semantic dependence order and physical
resource legality.

### Incomplete memory-space search

FA backward cannot become entirely annotation-free until SMEM/TMEM placement
is decided before physical allocation. Treat memtype removal as its own
milestone rather than hiding a heuristic choice inside `doBufferAllocation`.

### Ambiguous ownership of `tt.num_buffers`

The scheduler currently emits it, while WSMemoryPlanner owns final copy counts.
Resolve this explicitly: either rename it as a non-binding schedule-demand hint
or stop emitting it on the structural path. Never interpret a modeled demand
as a hard correctness floor.

## 8. Completion checklist

- [x] Structural Contracted SWP does not use result latency for admission.
- [x] Multiple II values survive top-K selection.
- [x] Descriptor-load placement participates in candidate identity.
- [x] CopySolver emits multiple legal copy vectors.
- [x] Memory beam returns distinct SMEM depth plans.
- [x] Fixed-grouping search preserves multi-store staging while searching
      singleton operand depths.
- [x] Fixed-grouping search is exercised on a production-shaped subtiled
      kernel.
- [x] Cross-id `allocation.reuseTarget` footprint is modeled by fixed-group
      search without changing alias depth or topology.
- [x] Compiler emits schedule and memory JSON-lines records for external
      Cartesian-product search.
- [x] Production-shaped D120 TTGIR is captured at the pre-modulo and
      post-buffer-allocation pass boundaries.
- [x] Contracted search retains D120's A0/B1/MMA2 and A1/B0/MMA2 iteration-lead
      alternatives on the production-shaped fixture.
- [x] Memory search recognizes the production-shaped eight-subtile output
      staging group and retains the heuristic A3/B2 plan as rank zero.
- [x] External harness compiles, validates, and measures the bounded product.
- [x] D120 candidates exist without lhs/rhs depth annotations.
- [ ] D120 measured winner is A3/B2 on the target shapes.
- [ ] FA-backward target schedule exists without stage/order annotations.
- [ ] FA-backward target memory plan exists without copy/id/offset pins.
- [ ] FA-backward memory-space annotations are removed or replaced by a general
      planning rule.
- [ ] Default-off compilation remains unchanged.
- [ ] Correctness, sanitizer, compile-time, and performance gates pass.
