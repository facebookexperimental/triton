# Contracted Schedule and Memory-Plan Search Plan

For the stable architecture, algorithms, data structures, and worked examples,
start with
[ContractedScheduleMemorySearchDesign.md](ContractedScheduleMemorySearchDesign.md).
This file tracks implementation history, remaining work, and validation status.

**Status:** Milestones A-C are complete. Milestone D has reached a focused
annotation-free FA-backward correctness result, but broad correctness and
performance validation are still required before removing the annotated
fallback. Milestone E has not started. For D120426461, the production-shaped
A3/B2 candidate passes correctness; the B-early/A2-B3 candidate still hangs and
must be fixed or rejected before evaluating the full frontier.

The remaining work is:

1. Connect the implemented weighted-cycle solver to a post-memory,
   schedule-aware protocol builder and use it to reject the unsafe D120
   B-early candidates before GPU execution.
2. Measure D120 candidates on the target shapes and confirm whether A3/B2 wins.
3. Validate annotation-free FA backward over its supported correctness matrix,
   then compare it with the annotated baseline under sanitizers and performance
   measurement.
4. Remove the remaining source memtype annotations only if the annotation-free
   candidate passes those gates.
5. Verify search-off neutrality, measure compile-time/candidate-count growth,
   and choose production search caps and fallback policy.

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
runs one compile/validation command per bounded candidate quadruple, verifies
the selected ranks, and records exit status, elapsed time, and an optional
command-reported metric as JSON lines.

**Implemented seventh slice:** `PromoteLHSToTMem` exposes a bounded logical
memory-space rank before physical buffer planning. Rank zero preserves the
existing heuristic. Higher ranks enumerate subsets of unannotated direct MMA
LHS operands that the heuristic leaves in SMEM only because a transposed LHS
sibling consumes the same source. This is the FA-backward dS choice: dQ keeps
the transposed SMEM view while dK may consume a TMEM copy. Explicit memtype
annotations retain precedence. The outer driver now sweeps the full schedule ×
memory-space × SMEM-plan × TMEM-plan product, and the pass emits a
`memory-space` manifest record so every selected rank is verified.

**Implemented eighth slice:** Standalone Contracted search no longer emits
latency-derived `tt.num_buffers`, `buffer.id`, `buffer.merge_group_id`, or
`tt.num_stages`. It preserves only logical stage/cluster scheduling metadata;
the frontend pipeline depth and WSMemoryPlanner remain authoritative for
physical buffering. This prevents structural schedules from turning a large
modeled lifetime/II ratio (182 stages in the D120 oracle) into an unsafe
software-pipeline depth.

**Implemented ninth slice:** `WSChannelCycleAnalysis` now provides the common,
synchronization-independent protocol graph and deterministic weighted-cycle
solver. It decomposes the graph into strongly connected components, applies
the exact integer transform `d * (N + 1) - 1`, and reconstructs a stable edge
witness for every non-positive-distance recurrence. Direct lit coverage pins a
zero-credit rejection, its positive-credit neighbor, an acyclic negative edge,
and malformed-graph `unsupported` handling. The post-memory IR graph builder
and candidate rejection were left for subsequent units; the ordinary-loop
post-memory builder is now connected by the eleventh slice below.

**Implemented tenth slice:** `buildChannelProtocolPlan` now derives grouped
producer bounds, TMA producer placement, per-consumer-task wait/release
anchors, copy depth, and cadence without mutating IR. `insertAsyncComm`
consumes these shared decisions, including the effective acquire relocation
selected by its existing operand-D and reuse rules. This establishes the
single endpoint contract needed by the planned-protocol graph builder while
preserving the emitted synchronization protocol.

**Implemented eleventh slice:** `doCodePartition` now lowers ordinary
loop-cadence SMEM endpoint plans into the normalized protocol graph immediately
after reuse-group validation and before accumulation-counter rewriting. Each
channel contributes zero-distance Acquire/Ready/Wait/Release edges plus a
copy-depth slot-reuse edge; task serialization contributes stage-delta edges
in cluster/source order. The common weighted-cycle solver runs in production
as an audit without changing candidate selection, while the test pass exposes
status, coverage counts, graph size, and the witness. The production-shaped
D120 B-early fixture reaches this path and reports its zero-credit cycle.
Reuse-group, TMEM, subtiled, straight-line, and while protocols remain
explicitly unsupported in this slice.

The implementation is split by responsibility rather than extending the
already-large code-partition utility: `WSChannelProtocol` owns endpoint plans,
`WSChannelCycleValidator` owns IR-to-graph lowering and audit diagnostics, and
`WSChannelCycleAnalysis` owns only graph types and the weighted-cycle solver.
`WSCodePartition` retains narrow call sites because it owns the live channel
and reuse topology at both synchronization insertion and the validation point.

**Current Milestone D status:** The annotation-free BM128 FA-backward candidate
compiles through software-pipeline expansion. Pre-lowered
`ttng.async_tma_store_wait` operations selected into a peeled pipeline stage
are predicated with the same `scf.if` mechanism as the other unmaskable TMA
side effects. Separating the SMEM and TMEM rank coordinates and retaining a
conservative low-aliasing TMEM frontier candidate exposes the annotated memory
topology at schedule rank 1, memory-space rank 1, SMEM rank 0, and TMEM rank 1.
A focused backward-only correctness test passes for this annotation-free
candidate. Broader correctness and performance validation remain open.

**Production-shaped D120 oracle:** An annotation-free bf16 fused RMSNorm + GEMM
with 128x128x128 tiles and eight-way output subtiling has been captured at the
pre-modulo and post-buffer-allocation boundaries. The pass-local scheduler test
shows that Contracted search retains the default A0/B0/MMA1 schedule plus the
three-stage A0/B1/MMA2 and A1/B0/MMA2 alternatives required to reproduce the
D120 experiment. The pass-local memory test confirms that the existing
heuristic produces A3/B2 on the same graph. Search-mode coverage on this fixture
now recognizes its eight separately materialized output-staging allocations as
one fixed subtile group, preserves A3/B2 at rank zero, and retains A2/B2 and
A2/B3 as the first two alternatives. The actual 1024x12800x1024 bf16 kernel now
passes numerical correctness on B200 for schedule rank 1 plus the A3/B2 SMEM
plan. Direct-grid output staging rotates the eight straight-line stores through
the selected three-copy ring; its data slots and barrier phases are checked at
the code-partition boundary. The adjacent A-early/A2-B2 candidate also passes
correctness. The currently paired B-early/A2-B3 cross-product candidate still
hangs and remains an explicit candidate-safety gap; it must be fixed or rejected
before claiming the entire D120 frontier is executable.

**Production-shaped FA-backward oracle:** The existing BM64 pre-modulo fixture
already proves that Contracted top-K retains the target five-GEMM schedule with
no `tt.autows` stage/order metadata. At the post-buffer-allocation boundary,
the early-TMA FA-backward fixture now strips all remaining `tt.autows` channel
metadata. When the selected schedule orders `dk` before `dq`, the memory
planner uses same-partition stage/cluster order to reconstruct the annotated
three-way `{dpT, dsT, dq}` TMEM reuse group without copy/id/offset pins. It is
currently an equal-score nonzero TMEM rank, intentionally left to the outer
measurement driver instead of being forced to rank zero by an unvalidated cost
model.

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
  -> pre-allocation logical memory-space top-K
  -> search-based SMEM/TMEM planning for that schedule
  -> selected SMEM copy rank + TMEM reuse rank
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
  schedule/memory-space/memory-plan combination.
- Do not initially search CTA topology, warp-group count, or epilogue subtile
  factor.

### 1.3 Current implemented search axes

This section is the current-state reference for the implemented search. Later
phases retain the design rationale and historical implementation order, but do
not supersede the algorithms summarized here. The axes are deliberately
separate: one compiler invocation selects a schedule and logical memory-space
candidate, then constructs independent physical SMEM and TMEM frontiers for
that selected IR.

#### Contracted schedule

**Implementation:** `ExhaustiveScheduler.cpp::runContractedSearch`.

**Candidate generation:** The scheduler computes a structural lower bound for
II from issue-resource demand, then searches a bounded II range. For each II it
enumerates tensor-core stage/cluster assignments and structurally discovered
descriptor/TMA loads that have a distance-zero path to a GEMM. Bounded variants
change both load stage and the topological order of a load plus its address
dependency slice. Candidate identity contains II and the stable
stage/cluster signatures of both GEMMs and GEMM-reaching loads.

**Admission and ranking:** Dependence topology, loop-carried distance, issue
duration, resource conflicts, and stage bounds are hard constraints. Estimated
result latency is only a ranking tie-breaker. The frontier pins the existing
default at rank zero, preserves distinct GEMM/load signatures, reserves a
non-default load-order candidate, and reserves representatives from the
feasible II range before filling remaining slots by deterministic rank.

**Output:** The selected candidate writes `loop.stage`, `loop.cluster`, and
`tt.modulo_ii`. Standalone Contracted search does not write authoritative
pipeline depth, physical copy counts, buffer IDs, or reuse groups.

**Status and limits:** Implemented and covered by reduced plus
production-shaped D120 and FA-backward fixtures. The frontier intentionally
retains the D120 B-early schedule because schedule generation alone cannot know
whether the later memory plan gives every relay enough capacity. Its selected
schedule-memory combination currently deadlocks. The authoritative fix is the
post-memory channel-cycle validator in Phase 10; an earlier scheduler rule may
eventually prune candidates only when it can prove the same result.

#### Logical memory space

**Implementation:** `PromoteLHSToTMem`, before physical buffer allocation.

**Candidate generation:** Rank zero preserves the established promotion
heuristic. Higher ranks enumerate bounded subsets of unannotated direct MMA LHS
operands that remain in SMEM only because the same source also feeds a
transposed LHS. This captures the FA-backward choice in which dQ keeps its
transposed SMEM view while dK may consume a TMEM copy, without naming either
operand in the search interface.

**Admission and ranking:** Only structurally recognized, representable
promotions are enumerated. Explicit `opndA,smem` or `opndA,tmem` annotations are
authoritative and remove that choice from the searchable set. Candidates use a
stable logical signature; this axis does not rank with a latency model.

**Output:** The selected rank changes the logical SMEM/TMEM backing seen by
`doBufferAllocation`. It does not assign `buffer.id`, `buffer.offset`, or
`buffer.copy`.

**Status and limits:** The direct/transposed-LHS ambiguity needed by the
annotation-free FA-backward oracle is implemented. Other ambiguous
memory-space patterns are not yet candidate generators and continue to use the
heuristic or explicit annotation.

#### SMEM copy depth

**Implementation:** `WSMemoryPlanSearch.{h,cpp}`,
`WSMemoryPlanCopies.cpp`, and `allocateSmemBuffersViaSearch` in
`WSMemoryPlanner.cpp`.

**Candidate generation:** The plan-space beam represents each ordinary SMEM
allocation as a singleton block; `SmemPacker::legalJoin()` intentionally does
not invent SMEM alias groups. At complete beam leaves, `CopySolver` returns a
bounded set of copy vectors: the legacy greedy vector first, then the hard-floor
vector, feasible one-copy increments, and larger edit-distance combinations.
For subtiled regions, multi-store staging, explicit pins, atomic broadcast, or
cross-ID reuse, fixed-group mode first imports the heuristic grouping and pins
its proven invariants while enumerating only mutable singleton depths.

**Admission and ranking:** `StaticCopySafetyValidator` enforces schedule-derived
stage/release floors and rotating-entry requirements. `SmemPacker` rechecks the
copy-expanded byte budget; an over-budget vector prunes its deeper descendants.
Estimated latency may rank candidates but cannot make an unsafe depth legal or
remove the correctness floor. The exact heuristic plan remains rank zero.

**Output:** The selected SMEM rank writes physical `buffer.copy` and preserves
or imports the required `buffer.id`/reuse structure. Cross-ID alias sources are
represented as non-owning blocks for budget accounting without changing their
synchronization topology.

**Status and limits:** Implemented. D120 retains A3/B2 at rank zero and exposes
A2/B2 and A2/B3 alternatives while preserving the eight-subtile output-staging
group. General SMEM reuse-group discovery is not searched; unsupported grouping
features remain fixed to the heuristic topology.

#### TMEM grouping and placement

**Implementation:** `WSMemoryPlanSearch.{h,cpp}` and
`allocateTmemBuffersViaSearch` in `WSMemoryPlanner.cpp`, with the existing
`MemoryPlannerTmem` allocator as the fallback for unmodeled features.

**Candidate generation:** The beam considers joining a TMEM buffer to every
legal existing block or opening a new block. A joined block time-multiplexes
members at offset zero; legality requires disjoint liveness plus a real
dependency or a same-partition order proved by the selected
`loop.stage`/`loop.cluster` schedule. Complete plans are canonically
deduplicated. TMEM copy depth is fixed at one; the searched variables are reuse
grouping and placement.

**Admission and ranking:** Plans must satisfy the TMEM footprint limit and the
same strict reuse-chain ordering that code partitioning will use. The heuristic
plan remains rank zero. When grouping alternatives exist, the next rank is
reserved for a feasible plan with the most physical blocks, providing a
low-aliasing candidate before other score-ranked packings.

**Output:** The selected TMEM rank writes `buffer.id`, `buffer.offset`, and
`buffer.copy = 1`; search-selected groups carry the marker that makes code
partitioning validate them with schedule-aware ordering.

**Status and limits:** Implemented for representable ordinary allocations and
used to recover the FA-backward `{dpT, dsT, dQ}` topology at a nonzero rank.
Scaled-MMA scale-column reservation and subtiled TMEM regions still fall back
to the existing allocator. TMEM multi-copy search is not implemented.

#### Cross-axis composition and selection

**Implementation:** compiler JSON-lines records through
`TRITON_WS_SEARCH_MANIFEST` and the external
`python/triton/tools/autows_search.py` driver.

The searches are layered, not one joint in-process beam:

```text
schedule rank
  -> logical memory-space rank
    -> SMEM plan rank x TMEM plan rank
      -> compile, validate, and optionally measure
```

Each child compilation selects exactly one rank on each upstream axis and
re-derives downstream facts from the resulting IR. The manifest records each
frontier and verifies that the requested ranks were applied. The driver sweeps
the bounded Cartesian product, rejects compile or correctness failures, and can
select by a command-reported runtime metric. There is currently no global
compiler cost model and no scheduler data structure passed directly to the
memory planner.

**Status and limits:** The four-axis product and independent SMEM/TMEM rank
coordinates are implemented. Production search caps, broad correctness gates,
runtime measurements, and the fallback/default policy remain Milestone E work.

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
so load-order variants remain distinct. Both the reduced one-GEMM oracle and
the production-shaped fused RMSNorm fixture prove that A-early and B-early
reach top-K.

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

The reduced one-MMA/two-descriptor oracle and the production-shaped fixture now
check the schedule alternatives. The memory-planner fixture retains A2/B2,
A3/B2, and A2/B3, while the actual kernel has runtime correctness coverage for
A3/B2 and A2/B2. A2/B3 currently hangs and is the remaining safety gap.

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

**Status:** The pass-local fixtures prove the target schedule and memory
topology without stage/order/copy/id/offset pins. A focused annotation-free
BM128 backward correctness test passes. Broad correctness and performance
comparison with the annotated baseline remain.

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
    for memorySpacePick in memorySpaceRanks:
        compile/dump memory candidates for (schedulePick, memorySpacePick)
        for smemPick in smemRanks(schedulePick, memorySpacePick):
            for tmemPick in tmemRanks(schedulePick, memorySpacePick):
                compile, validate, and measure
```

Emit a manifest containing:

- schedule rank and signature;
- II;
- memory-space rank and selected LHS promotions;
- independent SMEM and TMEM ranks and signatures;
- SMEM bytes and TMEM columns;
- fallback/fixed-grouping reason;
- deduplication key.

Use generic schedule and memory picks; do not expose per-operand controls.

The compiler-side manifest is implemented via `TRITON_WS_SEARCH_MANIFEST`. The
external `python/triton/tools/autows_search.py` driver discovers ranks from that
manifest and sweeps the bounded four-dimensional product. Its child command
must compile exactly one searched loop, return nonzero on validation failure,
and may print a numeric value selected by `--metric-regex` for performance
ranking.
For example:

```shell
python python/triton/tools/autows_search.py \
  --schedule-topk=4 --memory-space-topk=2 \
  --smem-topk=3 --tmem-topk=4 \
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

Each retained qkT/dk/dq/dpT/dv schedule is evaluated with its logical
memory-space and physical SMEM/TMEM plans. Runtime measurement selects the
triple rather than a compiler cost model.

### Phase 10: Reject schedule-aware channel cycles after memory planning

The scheduler must not conservatively reject every load with a non-GEMM side
use: a later memory plan may make the schedule safe through additional relay
copies or a different reuse topology. Instead, use one normalized protocol
graph with two frontends. The authoritative candidate gate runs after
`doMemoryPlanner` has assigned physical depths and `doCodePartition` has
reconstructed its `Channel` and `ReuseConfig` topology, but before accumulation
counters, tokens, barriers, or buffer rewrites mutate the IR. A second,
post-insertion frontend builds the same graph from emitted synchronization and
audits that materialization matches the accepted plan.

The precise early insertion point is inside `doCodePartition`, after channel
collection, reuse-group construction, consumer-group merging, and existing
reuse-group validation, but before `appendAccumCntsForOps`. Running directly
between the top-level `doMemoryPlanner` and `doCodePartition` calls would be too
early: generated subtiled regions and code partition's effective channel
grouping would not yet be represented.

#### 10.1 Define one normalized protocol graph

Represent only events relevant to blocking progress:

- producer acquire / empty wait;
- producer-ready commit or TMA/MMAv5 completion;
- consumer-ready wait;
- consumer release / empty arrival.

Each normalized event records a stable channel ID, role, task, enclosing
scheduled loop, stage, cluster, stable block order, physical buffer ID, copy
count, transaction stride, and logical transaction offset. The common graph
and cycle solver must not refer to concrete token or barrier op classes.

The intended internal representation is:

```cpp
using EventId = unsigned;

enum class ProtocolEventKind { Acquire, Ready, Wait, Release };
enum class ProtocolEdgeKind { TaskOrder, DataReady, SlotReuse, ControlFlow };

struct SchedulePoint {
  Operation *scope; // The scheduled loop or straight-line parent.
  int stage;
  int cluster;
  unsigned ordinal; // Stable order inside the cluster/task.
};

struct BufferKey {
  DataChannelKind space;
  unsigned bufferId;
  int64_t offset;
  int64_t extent;
};

struct ProtocolEvent {
  EventId id;
  SmallVector<unsigned> channelIds; // More than one after fusion.
  ProtocolEventKind kind;
  AsyncTaskId task;
  Operation *anchor; // Diagnostic only; never used by the solver.
  SchedulePoint schedule;
  BufferKey buffer;
  unsigned copies;
  unsigned transactionStride;
  unsigned transactionOffset;
};

struct ProtocolEdge {
  EventId from;
  EventId to;
  int64_t iterationDistance;
  ProtocolEdgeKind kind;
  unsigned channelId; // Diagnostic provenance.
};

struct ProtocolGraph {
  SmallVector<ProtocolEvent> events;
  SmallVector<ProtocolEdge> edges;
};

enum class ProtocolStatus { Safe, Unsafe, Unsupported };

struct ProtocolValidation {
  ProtocolStatus status;
  SmallVector<ProtocolEdge> cycleWitness;
  std::string reason;
};
```

`Operation *` fields exist only in a builder-side diagnostic table. The solver
itself consumes dense event IDs and integer edge weights, which makes it
unit-testable without constructing MLIR. Subtiled or multi-rate channels are
expanded to one event template per static transaction position before solving;
different loop nests are separate cadence domains until a supported affine
mapping between them is available.

#### 10.2 Provide pre- and post-insertion graph builders

The **planned-protocol builder** is the candidate-safety authority. It consumes
the post-memory `Channel` objects, selected allocation attributes, merged
consumer groups, and `ReuseConfig`. It models the synchronization contract of a
correct channel directly:

```text
producer acquire(i) waits for consumer release(i - copies)
consumer wait(i) waits for producer ready(i)
```

Factor the endpoint-placement decisions currently embedded in
`insertAsyncComm` into pure helpers: head/tail producer, head/last actual
consumer, inline-MMAv5 versus token protocol, reuse-group channel order, and
transaction cadence. Both the validator and insertion must call those helpers.
Reuse-group staggering, subtile stride, direct-grid ordinal, persistent-loop
cadence, and synthetic staging-reuse WAR edges must use the same routines as
`getBufferIdxAndPhase`; do not duplicate slot/phase arithmetic.

The **materialized-protocol builder** is a conformance audit. It runs after
`insertAsyncComm`, channel-graph injection, and barrier fusion, but before
buffer replacement or `specializeRegion`. It pairs NVWS token endpoints by SSA
value and direct TMA/MMAv5 endpoints by barrier SSA value, then lowers them to
the same normalized event representation. When fusion combines channels,
internal channel-ID metadata must be unioned rather than discarded. This audit
answers a different question from the early gate: not “would correct channel
sync make this plan live?” but “did code partitioning emit the protocol that
was validated?”

Do not pass a scheduler-owned graph across passes. Both builders re-derive
schedule order from `loop.stage`/`loop.cluster` and the selected memory plan
from IR. The in-memory normalized graph lives only within `doCodePartition`.

#### 10.3 Add precedence edges with iteration distance

Build a directed graph whose edge label is the logical-iteration distance from
the source event to the destination event:

1. **Task order:** consecutive blocking/signaling events in one task, ordered
   by the selected stage/cluster schedule and then stable program order.
2. **Data-ready:** producer completion for transaction `i` precedes the
   matching consumer wait for transaction `i`.
3. **Slot reuse:** consumer release for transaction `i` precedes producer
   acquire for transaction `i + copies`.
4. **Control flow:** loop backedges, guarded transactions, sibling loops, and
   persistent `scf.while` counters contribute their actual transaction stride.

The copy-count edge represents initial buffer credit. A normal circular
pipeline therefore has positive iteration distance around its recurrence. A
directed cycle whose total iteration advance is non-positive has no initial
credit that can sustain it and is a deadlock. Do not use operation latency.

For each cadence domain, solve this exactly as an integer weighted-cycle
problem:

1. Drop acyclic vertices using strongly connected components. A deadlock cycle
   can exist only inside an SCC.
2. For an SCC with `N` events, transform every edge distance `d` to
   `w = d * (N + 1) - 1`.
3. Add a zero-cost synthetic source to every event and run Bellman-Ford. A
   relaxation on iteration `N` proves a negative transformed cycle.
4. Follow predecessor edges `N` times to enter that cycle, reconstruct it, and
   report the original iteration distances and event metadata.

The transform handles the strictness of event order without floating point.
For any simple cycle of length `L <= N`:

```text
sum(w) = (N + 1) * sum(iterationDistance) - L
```

Therefore `sum(w) < 0` exactly when the original cycle advances by at most zero
iterations; a cycle advancing by at least one iteration has
`sum(w) >= N + 1 - L > 0`. Complexity is `O(VE)` per SCC, and these graphs are
small relative to the compiler DDG.

For scheduled events in one task, derive task-order distance from the
deserialized `CoarseSchedule`, not lexical order alone. In one steady-state
modulo period, moving from event `u` to the next event `v` contributes the
stage difference plus a one-iteration carry when the ordered cluster sequence
wraps. Real SSA/control dependencies retain their explicit dependence
distance. If stage, cadence, or control-flow mapping is ambiguous, return
`Unsupported` instead of guessing an edge weight.

#### 10.4 Reject the selected combination, not the schedule globally

Return one of `safe`, `unsafe`, or `unsupported`:

- `safe`: continue code partitioning;
- `unsafe`: emit a deterministic diagnostic and fail this selected compiler
  invocation;
- `unsupported`: do not claim validation coverage. During rollout, preserve
  the existing rank-zero path, but do not send an unvalidated nonzero search
  candidate to runtime measurement.

Change `doCodePartition` to return `LogicalResult` so the production and
test-only passes propagate an early-gate rejection normally. The external
product driver records the failed schedule/memory ranks and continues with the
next tuple. A mismatch or new cycle in the post-insertion audit is an internal
code-partitioning error, not a candidate-quality result. Add a manifest
validation record containing the selected ranks, validation boundary, status,
reason, and cycle witness. Do not silently substitute another rank inside the
compiler.

The diagnostic cycle witness should list, in order:

```text
task, event role, source location, buffer.id, copies,
loop.stage, loop.cluster, iteration delta
```

This makes rejection explainable and lets a future planner determine whether
increasing one relay depth or changing task order would break the cycle.

#### 10.5 Land coverage in increasing scope

1. Unit-test the common weighted graph solver directly with a zero-credit cycle
   and a neighboring positive-credit recurrence.
2. Add a synthetic code-partition fixture whose planned-protocol builder finds
   the same two-task cycle, plus a safe copy-depth or schedule variant.
3. Capture the actual D120 B-early post-memory TTGIR and check that both A2/B2
   and A2/B3 are rejected by the early gate with the same relay/B wait-for
   cycle. Check that A-early with A2/B3 passes; this proves the memory plan
   itself is not being banned.
4. On safe fixtures, compare canonical planned and materialized graphs after
   insertion. Add a negative test that deliberately omits or misplaces one
   endpoint so the conformance audit fails.
5. Run the existing FA-backward memory-topology and code-partition fixtures as
   positive tests, especially the `{dpT, dsT, dQ}` TMEM reuse group and
   persistent staging-reuse channels.
6. Run the external D120 product search and verify that rejected tuples finish
   at compile time rather than launching a hanging kernel, while A3/B2,
   A2/B2, and safe A2/B3 tuples still run correctness.
7. Extend the materialized graph to overlapping physical TMEM ranges as a
   follow-up. That subsumes the late barrier-aware checker proposed in
   `BwdTmemReuseSlotHazard.md`, but is not required to reject the D120 channel
   deadlock.

#### 10.6 First implementation slice: planned-protocol validator only

Land the post-memory validator before adding the materialized-protocol audit.
The implementation sequence is:

1. **Implemented:** add `WSChannelCycleAnalysis.{h,cpp}` and register the source
   in the Hopper transforms CMake target. The normalized graph and
   weighted-cycle solver are independent of MLIR synchronization op classes.
2. **Implemented:** move the schedule-order predicate formerly local to
   `insertAsyncComm` into a shared utility. Its contract is
   `before(i) -> after(i + distance)` under the serialized
   `loop.stage`/`loop.cluster` schedule; missing metadata returns unknown rather
   than lexical-order fallback across tasks.
3. **Implemented:** factor the non-mutating endpoint decisions needed by both
   insertion and validation into `ChannelProtocolPlan`: effective
   producer-acquire anchor, producer-ready anchor, per-task consumer-wait
   anchor, consumer-release anchor, buffer depth, and cadence. Existing
   operand-D and reuse rules update the plan's acquire anchor before
   `insertAsyncComm` consumes it.
4. **Audit implemented; rejection pending:** `doCodePartition` calls the
   dedicated validator helper after `ReuseConfig` construction,
   consumer-group merging, and reuse-group shape checks, but before
   `appendAccumCntsForOps`. The rejection unit will change `doCodePartition`
   to return `LogicalResult` and propagate failure through production and
   test-only passes.
5. Initially support ordinary channels whose relevant endpoints share one
   scheduled `scf.for` cadence and have affine one-transaction-per-iteration
   behavior. Analyze supported SCCs even when unrelated channels are outside
   that scope. A proven cycle is `Unsafe`; an unsupported component is reported
   as unvalidated and does not cause a false rejection during bring-up.
6. Add debug output and a manifest validation record. The error must include
   the cycle's channel IDs, task IDs, source locations, buffer IDs/copies,
   stage/cluster coordinates, and edge distances.
7. Capture the actual B-early D120 IR at the post-buffer-allocation boundary so
   the lit pipeline still runs memory planning before validation. Extend the
   existing D120 memory-planner test rather than creating a disconnected toy
   test file: B-early with A2/B2 and A2/B3 must fail with the same cycle;
   A-early/A2-B3 must pass.
8. Run the existing annotation-free FA-backward memory and code-partition lit
   tests as positive coverage. Only after these stay clean should the external
   search driver treat a proven `Unsafe` result as a rejected tuple and avoid
   launching it.

The audit slice does not inspect emitted tokens/barriers, validate TMEM alias
reads, or claim that every unsupported control-flow shape is safe. Those are
follow-up extensions to the second graph builder. The following rejection
slice will eliminate the D120 runtime hang by rejecting the proven cycle
before code-partition mutation.

Each implementation commit must rebuild Triton in the `metamain` environment
with `/home/mren/OpenSource2/llvm-build`, run the focused D120 and FA-backward
lit tests from the configured build directory, run the full relevant
WarpSpecialization lit suite, and pass pre-commit. GPU correctness is run only
after the lit gate; no performance benchmark is part of this slice.

#### D120426461 example

For the B-early schedule, the validator should report the zero-advance cycle:

```text
A relay release(i)
  -> A relay acquire(i+1)
  -> issue B(i+1)
  -> wait for B(i+1)
  -> A relay release(i)
```

The single-copy A relay supplies enough initial credit for the first
transaction, but the selected schedule consumes that credit without creating
positive iteration advance around the steady-state cycle. Changing only B from
two to three copies cannot break this cycle. A-early/A2-B3 remains accepted.

#### FA-backward example

The selected annotation-free schedule and `{dpT, dsT, dQ}` TMEM plan must
validate as a positive control. Its cross-partition reuse dependencies advance
through real data-ready and release edges rather than forming a non-positive
cycle. Persistent `scf.while`, same-task staging, MMAv5 inline completion, and
multi-member reuse remain explicit coverage requirements before enabling the
validator for all nonzero search candidates.

**Implementation areas:** `WSChannelProtocol.{h,cpp}` for shared endpoint
planning, `WSChannelCycleAnalysis.{h,cpp}` for the generic solver,
`WSChannelCycleValidator.{h,cpp}` for IR lowering and diagnostics, narrow hooks
in `WSCodePartition.cpp` / `WarpSpecializationPipeline.h`, and
`autows_search.py` for future manifest rejection handling.

### Phase 11: Remove manual annotations incrementally

1. Remove `stage` and `order` fields after Contracted SWP retains equivalent
   schedules.
2. Remove `copies`, `bufferId`, and `colOffset` after memory-plan search retains
   equivalent physical plans.
3. Remove memtype-only annotations after memory-space choice becomes a logical
   planning axis before physical buffer creation.

The first memory-space dimension is implemented in `PromoteLHSToTMem`, which
already runs before physical buffer creation. It enumerates the general
structural ambiguity "direct LHS shares a source with a transposed LHS" rather
than naming dK/dQ or exposing an operand-specific frontend knob. This avoids a
larger `doBufferAllocation` split for the current FA case; other ambiguous
memory-space decisions can be added to the same ranked manifest contract.

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
- Post-memory schedule-aware channel-cycle rejection, including a diagnostic
  cycle witness.
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

**Status: complete.**

- Split hard issue facts from estimated latency.
- Search a bounded II range.
- Retain GEMM- and load-placement diversity.
- Demonstrate both D120 load-order alternatives and the FA-backward target
  contraction schedule under lit.

### Milestone B: real SMEM copy search

**Status: complete.**

- Make CopySolver multi-result.
- Integrate copy candidates into beamSearch.
- Demonstrate A2/B2, A3/B2, and A2/B3 without operand annotations.

### Milestone C: fallback-compatible search

**Status: complete.**

- Add fixed-grouping mode.
- Exercise D120's subtiled epilogue and FA-backward staging cases.

### Milestone D: annotation-free FA backward

**Status: in progress.** The target schedule, memory-space choice, and physical
memory topology are searchable, and focused BM128 correctness passes. The
broader correctness/performance gates and source annotation removal remain.

- Remove schedule and physical-allocation pins.
- Add memory-space selection or a general memory-space heuristic. The first
  bounded rank is implemented for direct/transposed LHS siblings.
- Validate correctness and performance before removing the annotated fallback.

### Milestone E: evaluation and default decision

**Status: not started.**

- Measure the bounded schedule × memory-space × memory-plan product.
- Choose search limits and fallback policy based on compile-time and runtime
  data.
- Keep the feature opt-in until representative workloads show stable wins.

Original engineering estimate, retained for historical context:

| Milestone | Estimate |
|---|---:|
| A | 4–6 days |
| B | 3–5 days |
| C | 2–4 days |
| D | 5–8 days |
| E | 2–4 days |

Milestones A-C and the first memory-space search axis are now implemented. The
remaining duration depends primarily on debugging the unsafe D120 candidate and
collecting representative correctness, sanitizer, compile-time, and performance
results rather than on implementing another core search layer.

## 7. Risks and decision points

### Candidate explosion

Schedule, memory-space, SMEM, and TMEM ranks multiply quickly. Use explicit
global caps, canonical signatures, and diversity buckets. Do not silently
replace diversity preservation with latency-score pruning.

### Structural schedules that stall heavily

Removing latency from admission intentionally permits schedules that hardware
scoreboarding or barriers serialize. Runtime measurement rejects them. The
search must nevertheless enforce semantic dependence order and physical
resource legality.

### Incomplete memory-space coverage

The FA-backward dS SMEM/TMEM choice is now searchable before physical
allocation, but this first candidate generator is intentionally narrow. New
memory-space ambiguities should extend the same generic rank/manifest axis and
retain explicit annotations as an override until runtime validation is done.

### Ownership of `tt.num_buffers`

The joint modulo path continues to emit authoritative `tt.num_buffers`, buffer
IDs/grouping, and `tt.num_stages`. Standalone structural Contracted search no
longer emits them: WSMemoryPlanner owns physical copy counts and grouping, and
the frontend pipeline depth remains intact. A modeled demand must never become
a hard correctness floor on the structural path.

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
- [x] Standalone Contracted search leaves pipeline depth and physical buffer
      decisions to the frontend and WSMemoryPlanner.
- [x] Production-shaped D120 TTGIR is captured at the pre-modulo and
      post-buffer-allocation pass boundaries.
- [x] Contracted search retains D120's A0/B1/MMA2 and A1/B0/MMA2 iteration-lead
      alternatives on the production-shaped fixture.
- [x] Memory search recognizes the production-shaped eight-subtile output
      staging group and retains the heuristic A3/B2 plan as rank zero.
- [x] External harness compiles, validates, and measures the bounded product.
- [x] D120 candidates exist without lhs/rhs depth annotations.
- [x] D120 schedule rank 1 with the A3/B2 memory plan passes the actual
      1024x12800x1024 bf16 correctness run on B200.
- [x] The audit-only post-memory builder detects D120 B-early through real
      planned channels and returns a zero-distance witness.
- [ ] Post-memory channel-cycle validation rejects D120 B-early before launch
      while retaining A-early with the A2/B3 memory plan.
- [ ] D120 measured winner is A3/B2 on the target shapes.
- [x] FA-backward target schedule exists without stage/order annotations.
- [x] FA-backward target memory plan exists without copy/id/offset pins.
- [x] FA-backward's remaining dS memory-space choice is represented by a
      general pre-allocation search rank and covered with captured production
      TTGIR.
- [x] Annotation-free BM128 FA backward compiles through software-pipeline
      expansion with scheduled TMA waits predicated safely.
- [x] An annotation-free BM128 FA-backward candidate reproduces the annotated
      memory topology and passes focused backward-only numerical correctness.
- [ ] Remove the source memtype annotations after correctness and performance
      select the annotation-free candidate.
- [ ] Default-off compilation remains unchanged.
- [ ] Correctness, sanitizer, compile-time, and performance gates pass.
