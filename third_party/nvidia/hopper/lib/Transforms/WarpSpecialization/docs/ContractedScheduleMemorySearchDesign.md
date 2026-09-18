# Contracted Schedule and Memory-Plan Search Design

This document describes the architecture, search spaces, core data structures,
and correctness contracts of the AutoWS schedule-and-memory search. It is the
starting point for engineers who need to understand or extend the system.

For implementation status, experiments, and remaining work, see
[ContractedScheduleMemorySearchPlan.md](ContractedScheduleMemorySearchPlan.md).
For the physical allocator in more detail, see
[MemoryPlannerSearch.md](MemoryPlannerSearch.md).

**Primary implementation areas:**

- `third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlan*`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlanner.cpp`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSChannelProtocol.{h,cpp}`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSChannelCycleAnalysis.{h,cpp}`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSChannelCycleValidator.{h,cpp}`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSCodePartition.cpp`
- `python/triton/tools/autows_search.py`

## 1. Problem and design thesis

AutoWS needs to choose two related but distinct things:

1. **When work executes:** the modulo schedule of loads, computations, and
   tensor-core operations.
2. **Where in-flight values live:** SMEM or TMEM, their copy depths, and their
   physical reuse groups.

These choices interact. Moving a load earlier may require more copies; sharing
a buffer may add a producer-acquire dependency; and a schedule that is valid
for one memory plan can deadlock with another. However, one monolithic solver
would couple several evolving legality models and require an accurate runtime
cost model that the compiler does not currently have.

The design therefore uses four bounded search axes with explicit IR boundaries:

```text
TTGIR dependence graph
  -> Contracted schedule frontier
       selected II + loop.stage + loop.cluster
  -> logical memory-space frontier
       selected SMEM/TMEM representation
  -> physical memory-plan frontiers
       SMEM copy depth x TMEM grouping/placement
  -> schedule-aware channel validation
  -> code partitioning and software-pipeline expansion
  -> external correctness and runtime measurement
```

The compiler generates structurally diverse, legal candidates. Runtime
measurement chooses among them. Estimated latency may rank candidates but is
never a correctness condition.

## 2. Ownership and pass contract

| Component | Consumes | Owns | Emits |
|---|---|---|---|
| Contracted scheduler | Loop DDG and structural resources | II and logical execution order | `tt.modulo_ii`, `loop.stage`, `loop.cluster` |
| Logical memory-space search | Scheduled MMA operand graph | SMEM-versus-TMEM representation | Rewritten logical allocation choice |
| Buffer allocation | Task assignments and SSA dataflow | Cross-partition channel discovery | Canonical single-copy allocs and stores |
| SMEM planner | Channels, schedule, SMEM budget | Copy depths and imported fixed groups | `buffer.copy`, `buffer.id`, reuse metadata |
| TMEM planner | Channels, schedule, TMEM capacity | Reuse grouping and column placement | `buffer.id`, `buffer.offset`, `buffer.copy` |
| Channel-cycle validator | Selected schedule and physical plan | Progress legality | Accept, reject, or unsupported with witness |
| Code partitioner | Validated channels and allocation | Synchronization materialization | Tokens, barriers, physical WS regions |
| External driver | Candidate manifests | Empirical selection | Result for each rank tuple and measured winner |

No scheduler-owned C++ object crosses a pass boundary. Downstream passes
reconstruct what they need from IR. In particular:

- scheduling owns `loop.stage`, `loop.cluster`, and `tt.modulo_ii`;
- memory planning owns `buffer.copy`, `buffer.id`, and `buffer.offset`;
- code partitioning owns the concrete synchronization protocol.

Standalone Contracted search does not emit `tt.num_stages`,
`tt.num_buffers`, buffer IDs, or merge groups. The frontend pipeline depth and
WSMemoryPlanner remain authoritative for physical buffering.

## 3. Search axes at a glance

| Axis | Candidate variable | Hard admission | Ranking/diversity | Current state |
|---|---|---|---|---|
| Contracted schedule | II and GEMM/load stage-cluster order | DDG dependence distance, issue resources, stage bounds | Default rank zero; preserve signatures, load order, and II representatives | Implemented |
| Logical memory space | Which ambiguous values use SMEM or TMEM | Representability and explicit annotation overrides | Heuristic rank zero; deterministic subset order | Implemented for direct/transposed LHS siblings |
| SMEM plan | Copy count per logical block | Safety floors and byte budget | Heuristic plan rank zero; structural copy neighbors | Implemented, including fixed-group mode |
| TMEM plan | Reuse groups and column placement | Dependency order and 512-column capacity | Heuristic rank zero; low-aliasing representative; remaining score order | Implemented for ordinary allocations |
| Channel progress | Selected schedule-memory combination | Absence of a non-progressing channel cycle | Not ranked; reject with witness | Post-memory builder covers ordinary SMEM, staging WAR, A1/A2/A3 SMEM reuse, and A5 TMEM reuse |

Ranks are local enumeration positions, not stable semantic identities. Tests,
manifests, and result databases should retain canonical signatures as well as
ranks.

## 4. Contracted schedule search

### 4.1 Structural timing

The search deliberately separates facts required for legality from estimates
used only for ordering candidates.

Hard facts include:

- dependence edges and loop-carried distance;
- hardware pipeline kind;
- issue duration and resource multiplicity;
- stage-count and representability bounds.

Estimated facts include result latency and predicted critical-path cost. The
structural resource lower bound is conceptually:

```text
lowerII = max_pipeline ceil(sum(issueDuration) / numberOfUnits)
```

The search enumerates a bounded II interval starting at this lower bound. It
does not use latency-derived recurrence II to exclude candidates.

### 4.2 Candidate generation

The Contracted graph identifies:

- tensor-core/GEMM nodes;
- descriptor or TMA loads with a distance-zero path to those GEMMs;
- address-calculation dependencies that must move with a load.

For every retained II, it enumerates bounded combinations of:

1. GEMM stage and cluster placement;
2. eligible load stage placement;
3. topological load-order variants.

A conceptual candidate is:

```cpp
struct ScheduleCandidate {
  int ii;
  SmallVector<StageCluster> gemms;
  SmallVector<StageCluster> loads;
  unsigned loadOrderVariant;
  ScheduleSignature signature;
  StructuralScore score;
};
```

The exact implementation may store these fields differently. The important
identity is `(II, GEMM signature, load signature)`, using stable graph order
rather than operand names.

### 4.3 Frontier policy

The bounded frontier retains candidates in this priority order:

1. the existing/default schedule at rank zero;
2. distinct GEMM stage signatures;
3. legal single-load movements and at least one non-default load order;
4. evenly spaced feasible II representatives, including endpoints when the
   quota permits;
5. remaining candidates by deterministic structural rank.

This prevents the lowest II or one latency-favored schedule from removing all
structural alternatives.

## 5. Logical memory-space search

Logical memory-space selection runs before physical buffer allocation. Rank
zero preserves the existing promotion heuristic. Higher ranks enumerate
bounded subsets of structurally ambiguous values.

The first implemented ambiguity is:

```text
one source
  -> direct MMA LHS
  -> transposed MMA LHS
```

The direct LHS may be represented in TMEM while the transposed sibling remains
in SMEM. This recovers the FA-backward dK/dQ choice without encoding operand
names or exposing a frontend tuning knob.

Explicit memory-space annotations remain authoritative and remove that value
from the search. This axis chooses only the logical representation; it does not
choose physical IDs, offsets, or copy counts.

## 6. Physical memory-plan search

### 6.1 Core representation

`WSMemoryPlanSearch` normalizes allocator-specific IR into a small interface:

```text
BufferModel
  size, liveness, stageSpan, entries, encoding, kind,
  reuseScope, latency, frequency, dependency/reuse predicates

Block
  one or more logical buffers sharing a physical allocation
  copy count
  footprint

Plan
  collection of blocks
  placement
  total footprint
  score and canonical signature
```

`stageSpan` and rotating entries contribute to correctness floors. Latency and
frequency contribute only to score; frequency is currently a placeholder for
most paths.

### 6.2 Beam search versus CopySolver

These are nested algorithms, not competing search implementations:

```text
beamSearch chooses grouping
  -> complete grouping leaf
     -> CopySolver enumerates copy vectors for that grouping
        -> safety validation
        -> capacity validation
        -> scoring and canonical deduplication
        -> global top-K
```

`beamSearch` processes buffers in deterministic liveness order. For each
buffer, it branches over joining every legal existing block or opening a new
block. Partial states use one greedy copy assignment for inexpensive ranking
and are truncated to beam width `max(16, topK)`.

Only complete grouping leaves expand copy depth. `CopySolver` returns:

1. the legacy greedy vector;
2. the correctness-floor vector;
3. every feasible one-block `+1` neighbor;
4. larger edit-distance combinations until its bound is reached.

An over-budget vector prunes its deeper descendants because footprint is
monotone in copy depth. Duplicate concrete plans are removed by canonical
signature.

### 6.3 SMEM semantics

For ordinary SMEM buffers, `SmemPacker::legalJoin()` is intentionally false.
The generic search therefore changes copy depths but does not invent alias
groups. A block costs:

```text
max(memberBytes) * copies
```

Specialized grouping remains available through fixed-group mode:

1. run the heuristic allocator;
2. import its grouping, staging, and reuse structure;
3. pin grouped, staged, subtiled, annotated, and cross-ID reuse invariants;
4. enumerate only mutable singleton depths;
5. retain the exact heuristic plan at rank zero.

Cross-ID reuse sources remain separate logical blocks but do not count toward
the budget because their physical backing belongs to the pinned target.

### 6.4 TMEM semantics

For ordinary TMEM allocations, the beam searches grouping and placement.
Buffers may join one physical block only when liveness is disjoint and their
order is proved by a real dependency or by selected stage/cluster order within
one task. Joined members time-multiplex the same columns at offset zero.

Plans must fit the Blackwell 512-column capacity and satisfy the same strict
reuse-chain ordering used by code partitioning. TMEM copies are currently fixed
to one. Rank zero preserves the existing plan; rank one is reserved for a
feasible low-aliasing plan when grouping alternatives exist.

Scaled-MMA scale columns and subtiled TMEM regions are not represented by the
modular beam and fall back to `MemoryPlannerTmem`. This legacy allocator is a
separate implementation path, not another stage of the beam.

### 6.5 Safety and scoring

Every concrete plan passes:

1. `StaticCopySafetyValidator` for schedule-derived copy floors;
2. pool capacity validation;
3. canonical deduplication.

The current score rewards estimated latency hiding and may use occupancy as a
penalty. Scores order already-legal candidates; they do not establish
correctness. Rank zero is explicitly preserved even when another candidate has
an equal or better experimental score.

## 7. Cross-axis composition

The axes are composed externally rather than through one global beam:

```text
for schedule in scheduleFrontier:
  for memorySpace in memorySpaceFrontier(schedule):
    build physical frontiers
    for smem in smemFrontier(schedule, memorySpace):
      for tmem in tmemFrontier(schedule, memorySpace):
        compile
        validate correctness
        optionally measure runtime
```

Each child compilation selects exactly one tuple and re-derives downstream
facts from IR. `TRITON_WS_SEARCH_MANIFEST` records:

- selected and available ranks;
- canonical schedule and memory signatures;
- II;
- logical memory-space promotions;
- SMEM bytes and TMEM columns;
- fallback or fixed-group reason;
- validation and process status.

`python/triton/tools/autows_search.py` discovers the frontiers, verifies that
requested ranks were applied, and records results. An expected validator
diagnostic can be classified as `rejected`, separately from a compilation or
correctness `failed` result. The driver may execute independent tuples in
parallel, but the compiler pipeline within each tuple is ordered schedule-first,
memory-second.

## 8. Schedule-aware channel validation

Memory planning can introduce a bounded channel whose backpressure conflicts
with the selected schedule. Per-channel synchronization can be correct while
the composition deadlocks. The planned validator detects this after physical
memory selection but before synchronization mutates the IR.

### 8.1 Validation boundary

Inside `doCodePartition`, first reconstruct:

- `Channel` / `AllocChannel` / `TmemAllocChannel` objects;
- merged producer and consumer groups;
- `ReuseConfig` from selected `buffer.id` values;
- effective `buffer.copy` depths.

Run the candidate gate after those structures and existing reuse-group checks
are complete, but before `appendAccumCntsForOps`. A later optional audit builds
the same graph from emitted tokens and barriers to verify code-partitioning
conformance.

### 8.2 Normalized protocol graph

The graph solver is independent of MLIR synchronization op classes:

```cpp
enum class ProtocolEventKind { Acquire, Ready, Wait, Release };
enum class ProtocolEdgeKind { TaskOrder, DataReady, SlotReuse, ControlFlow };

struct ProtocolEvent {
  EventId id;
  SmallVector<unsigned> channelIds;
  ProtocolEventKind kind;
  AsyncTaskId task;
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
  unsigned channelId;
};

struct ProtocolGraph {
  SmallVector<ProtocolEvent> events;
  SmallVector<ProtocolEdge> edges;
};
```

An edge `(u, v, d)` means event `u(i)` must complete before event
`v(i + d)` may proceed.

The graph contains:

- `Ready(i) -> Wait(i)` with distance zero;
- `Release(i) -> Acquire(i + copies)` for physical slot reuse;
- for an A1 SMEM group, cross-channel slot edges from the transaction that
  last owned the same physical slot, replacing each member's self edge;
- schedule-aware program-order edges between consecutive events in each task;
- explicit control-flow and transaction-stride edges.

For an A1 group with `N` logical transactions per loop iteration and `K`
physical slots, target member `j` has predecessor `p = (j - K) mod N` and edge
`release(p, i) -> acquire(j, i + (p + K - j) / N)`. A direct-grid group uses
the finite form `release(j-K) -> acquire(j)` only for `j >= K`; it has neither
an end-to-start task-order wrap nor a fabricated transaction after the final
subtile.

Endpoint selection and cadence calculations must be factored from
`insertAsyncComm` so insertion and validation cannot disagree.

### 8.3 Weighted-cycle algorithm

A valid circular pipeline must advance at least one logical iteration around
every directed cycle. To detect a non-positive-distance cycle:

1. Compute strongly connected components.
2. For an SCC with `N` nodes, replace each edge distance `d` by
   `w = d * (N + 1) - 1`.
3. Add a zero-cost synthetic source and use Bellman-Ford to find a negative
   transformed cycle.
4. Follow predecessor edges to reconstruct a deterministic diagnostic witness.

For a simple cycle of length `L <= N`:

```text
sum(w) = (N + 1) * sum(d) - L
```

Therefore the transformed cycle is negative exactly when its original
iteration advance is at most zero. The algorithm is structural and uses no
operation latency.

The validator returns `Safe`, `Unsafe`, or `Unsupported`. During incremental
rollout, only a proven `Unsafe` result rejects a candidate; unsupported cadence
or control flow is recorded rather than guessed.

## 9. Worked examples

### 9.1 D120426461 fused RMSNorm + GEMM

Use distinct notation for schedule stage and memory copy depth:

- schedule `A0/B1/MMA2` means A at stage 0, B at stage 1, MMA at stage 2;
- memory plan `A3/B2` means three physical A copies and two B copies.

The relevant frontier is:

| Schedule signature | SMEM plan | Expected validation | Role |
|---|---|---|---|
| A0/B1/MMA2 | A3/B2 | Accept | Current target |
| A0/B1/MMA2 | A2/B2 | Accept | Adjacent baseline |
| A0/B1/MMA2 | A2/B3 | Accept | Valid memory alternative |
| A1/B0/MMA2 | A2/B2 | Reject | B-early channel cycle |
| A1/B0/MMA2 | A2/B3 | Reject | Same cycle; B depth is irrelevant |

A also feeds RMSNorm reduction. Buffer allocation introduces a single-copy A
relay from the load task to the GEMM task. Under the B-early schedule:

```text
A release(i)
  -> A acquire(i+1)
  -> B ready(i+1)
  -> B wait(i+1)
  -> A release(i)
```

The edge distances sum to zero, so the first transaction can consume the
initial A slot but steady-state execution cannot replenish it. Increasing B's
descriptor-buffer depth does not change the cycle.

### 9.2 Flash Attention backward

The target contraction order is:

```text
stage 0: qkT, dpT, dV
stage 1: dK, dQ

cluster(qkT) < cluster(dK) <= cluster(dQ)
               < cluster(dpT) <= cluster(dV)
```

The logical memory-space axis permits dK to consume a TMEM copy of dS while dQ
retains the transposed SMEM view. The physical planner can then recover the
three-member `{dpT, dS, dQ}` TMEM reuse topology using same-task schedule order
instead of explicit copy, ID, or offset pins.

The schedule, memory-space choice, SMEM plan, and TMEM plan remain separate
candidate coordinates. A focused annotation-free BM128 configuration currently
passes correctness; broader correctness, sanitizer, and performance validation
remain roadmap items.

## 10. Invariants and failure policy

1. Search-off behavior and rank zero preserve the existing pipeline.
2. Candidate identity never depends on source operand names.
3. Latency estimates never admit or reject a candidate.
4. Physical copy counts and reuse groups are owned by WSMemoryPlanner.
5. A correctness floor may exceed a configured preference or budget; the
   hardware resource check remains the final capacity backstop.
6. Same-task stage/cluster order may prove a reuse edge; textual order across
   different tasks may not.
7. A selected invalid tuple fails with a diagnostic and manifest record. The
   compiler does not silently substitute another rank.
8. The external driver continues to the next tuple after compile or validation
   failure.
9. Unsupported analysis is distinct from proven safety or unsafety.

## 11. Current status and roadmap

Implemented:

- Structural Contracted scheduling with bounded II and load-order diversity.
- Independent logical memory-space selection for direct/transposed LHS
  siblings.
- Multi-result `CopySolver` and SMEM depth diversity.
- Fixed-group SMEM search for staging, subtile, annotation, and cross-ID reuse
  constraints.
- TMEM grouping/placement search with a conservative low-aliasing frontier
  candidate.
- Independent schedule, memory-space, SMEM, and TMEM manifests plus the
  external Cartesian-product driver.
- Production-shaped D120 and FA-backward pass-local fixtures.
- Focused correctness for D120 A-early/A3-B2 and annotation-free FA backward.
- A synchronization-independent protocol graph and deterministic weighted-cycle
  solver, including direct lit coverage for zero-credit, positive-credit,
  acyclic, and unsupported graphs.
- Shared `ChannelProtocolPlan` construction for grouped producer endpoints,
  per-task consumer wait/release anchors, buffer depth, and cadence. Existing
  synchronization insertion consumes this plan; reuse-specific acquire
  relocation updates the same record.
- A post-memory graph builder and candidate gate for ordinary, single-CTA SMEM
  channels in a common scheduled `scf.for`, including nested inner loops.
  It lowers planned endpoints to Acquire/Ready/Wait/Release events, adds copy
  depth and stage-delta edges, distinguishes asynchronous TMA/MMAv5 completion
  from task-serializing endpoints, and runs the common weighted-cycle solver.
  Nested scheduled loops are validated as finite transaction domains when
  their trip count is constant: `(event, inner iteration)` expansion removes
  edges outside the prologue/drain boundary and retains only realizable
  cycles. For dynamic inner loops, an asynchronous producer-ready and consumer
  wait in the same early stage provides explicit prologue credit to a negative
  task-order edge ending at that wait. A negative edge ending at producer
  acquire receives no credit. This accepts FA backward's seeded `m/Di/dS`
  recurrence while preserving rejection of D120's unseeded B-early A-relay
  cycle. An outer-produced channel consumed by one directly nested loop is
  represented as one outer transaction: wait at inner entry, release at inner
  drain. The production FA-backward fixture now includes its ordinary K
  outer-to-inner channel in the supported graph. Dynamic negative-total
  recurrences remain `Unsupported`. Only `Unsafe` is rejected;
  `Unsupported` continues. Endpoint planning lives in
  `WSChannelProtocol.{h,cpp}`, graph lowering and audit diagnostics in
  `WSChannelCycleValidator.{h,cpp}`, and the generic solver remains isolated in
  `WSChannelCycleAnalysis.{h,cpp}`. TMA staging allocations linked to operand
  storage by `allocation.reuseTarget` contribute the coalesced cross-tile WAR
  edge that code partitioning will materialize: staging drain in transaction
  `i` precedes operand overwrite in transaction `i + 1`. The shared protocol
  planner fails closed when the aliases do not share one persistent loop and
  task pair. Multi-buffered A1 SMEM reuse groups are also lowered using their
  actual shared-slot predecessor relation. Both cyclic `scf.for` groups and
  finite direct-grid groups are represented; the D120 eight-subtile,
  three-copy staging ring contributes 32 events and five physical slot-reuse
  edges. Single-copy A2 dependency pairs and A3 same-block chains retain their
  ordinary per-channel token edges and add the ordered physical-slot chain and
  its cross-iteration wrap. Focused lit functions cover two- and three-member
  cyclic SMEM groups. A5 full-overlap TMEM groups reuse code partitioning's
  cross-partition predicate and unique dependency-chain order. Their ordinary
  channel/task edges represent the inherent middle transitions, while the
  normalized graph adds the same first/last intra-iteration and wraparound
  endpoint constraints that synchronization insertion emits. The production
  FA-backward `{dpT, dS, dQ}` fixture checks this coverage. Exact finite-loop
  expansion preserves negative task edges and lets the prologue boundary omit
  nonexistent predecessors instead of collapsing them into false zero-distance
  cycles.

Next:

1. Complete post-memory coverage for remaining TMEM and subtiled protocol
   shapes.
2. Add the post-insertion conformance builder over the same protocol graph.
3. Validate the complete supported correctness matrix and sanitizer cases.
4. Measure D120 and FA-backward candidate frontiers on target hardware.
5. Select production search caps and fallback policy.
6. Remove remaining source annotations only after correctness and performance
   gates pass.

Current limitations:

- The logical memory-space generator covers one structural ambiguity class.
- General SMEM alias-group discovery is not searched.
- TMEM multi-copy search is not implemented.
- Scaled-MMA and subtiled TMEM cases may use the legacy allocator.
- Candidate ranks are not stable across compiler changes; signatures must be
  used for durable comparisons.
- The post-memory builder rejects proven zero-distance cycles and covers A1
  multi-buffered plus A2/A3 single-copy SMEM reuse and A5 cross-partition TMEM
  reuse. Dynamic negative-total cycles, ordinary/A2/A4/A6 TMEM protocols,
  multi-CTA, other specialized protocols, and post-insertion coverage remain
  open.

## 12. Controls and diagnostics

The search is compiler-internal and experimental. Important controls include:

| Control | Purpose |
|---|---|
| `TRITON_MODULO_TOPK`, `TRITON_MODULO_PICK` | Contracted schedule frontier and selected rank |
| `TRITON_WS_MEMORY_SPACE_TOPK`, `TRITON_WS_MEMORY_SPACE_PICK` | Logical memory-space frontier and selected rank |
| `TRITON_WS_SMEM_PLAN_TOPK`, `TRITON_WS_SMEM_PLAN_PICK` | SMEM frontier and selected rank |
| `TRITON_WS_TMEM_PLAN_TOPK`, `TRITON_WS_TMEM_PLAN_PICK` | TMEM frontier and selected rank |
| `TRITON_WS_SMEM_PLAN_SEARCH` | Enable modular plan-space search |
| `TRITON_WS_SEARCH_MANIFEST` | Append cross-axis candidate and selection records |

These are compiler/search controls, not kernel-facing operand-specific tuning
knobs. Production interfaces should not expose `lhsDepth`, `rhsDepth`, or
similar names.

## 13. Further reading

- [Overview.md](Overview.md) — complete AutoWS pass order and file map.
- [ContractedScheduleMemorySearchPlan.md](ContractedScheduleMemorySearchPlan.md)
  — implementation history, detailed phases, and completion checklist.
- [MemoryPlannerSearch.md](MemoryPlannerSearch.md) — physical plan-space search
  and legacy TMEM allocator details.
- [BufferAllocation.md](BufferAllocation.md) — channel discovery and allocation
  normalization.
- [CodePartition.md](CodePartition.md) — synchronization materialization.
- [AccumulationCounters.md](AccumulationCounters.md) — slot and phase cadence.
- [ReuseGroups.md](ReuseGroups.md) — physical buffer-sharing protocols.
- [DebuggingAccuracyAndDeadlocks.md](DebuggingAccuracyAndDeadlocks.md) — runtime
  failure investigation workflow.
