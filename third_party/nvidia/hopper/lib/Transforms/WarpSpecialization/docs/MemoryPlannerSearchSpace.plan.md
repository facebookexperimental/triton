# Memory Planner Search Space — Design & Implementation Plan

**Status**: proposal / not yet implemented
**Covers (future)**: `WSMemoryPlanner.cpp`, new `WSMemoryPlanSearch.{h,cpp}`
**Related docs**: [SmemAllocationDesign.md](SmemAllocationDesign.md),
[TMEMAllocationHeuristics.md](TMEMAllocationHeuristics.md),
[ReuseGroups.md](ReuseGroups.md), [AccumulationCounters.md](AccumulationCounters.md),
[BwdTmemReuseSlotHazard.md](BwdTmemReuseSlotHazard.md)

---

## 1. Goal

Replace the memory planner's fixed heuristic (priority buckets P0/P1/P2 +
iterative copy increase + TMEM round-robin) with a **search over the space of
legal, feasible allocation plans**, scored by a latency-aware cost model, keeping
the **top-K** plans.

Key reframing: we search the planner's **output** (the allocation plan), not its
**inputs** (the knobs `smem-circular-reuse`, `num-buffers`, `smem-alloc-algo`,
`tmem-alloc-algo`, `partition-condition`). Every knob combination only ever
produced *some* plan in this space, and never all the good ones. Searching plans
directly subsumes the knobs; the heuristic planner becomes, at most, a fallback /
bound seed.

### Locked design decisions

| Decision | Choice |
|---|---|
| Objective | **Perf**, with a hard feasibility gate (never emit a deadlocking/OOR plan) |
| Feasibility | **Static** legality+feasibility check (GPU-free), reject before scoring |
| SMEM vs TMEM | **Independent** packing (separate pools, separate searches, combine scores) |
| Search strategy | **Beam** (width `W`) to start; branch-and-bound upgrade later |
| Buffer ordering | **Swappable policy**; liveness-start first, topological later |
| Priority signal | **Latency-driven** marginal benefit from schedule annotations |
| Scope | SMEM + TMEM, all decision dimensions (grouping, copies, placement) |

---

## 2. Problem formulation

### 2.1 A "plan"

Given the buffer set `B` (every SMEM `local_alloc` and TMEM alloc), a **plan** `P`
assigns to each buffer:

- **grouping** — a partition of `B` into physical **blocks**; buffers in one
  block share physical space (reuse). → `buffer.id`
- **copies** — a multi-buffer depth per block. → `buffer.copy`
- **placement** — for TMEM, `(rowOffset, colOffset)` within the block; SMEM
  blocks are sized by `max(size)·copies`. → `buffer.offset`

This is exactly the tuple the downstream lowering already consumes, so a plan is a
complete, replayable decision — serializable via the existing
`BufferDecisionList`.

### 2.2 Legal ∧ feasible = the pruning predicate

**Legal (correctness):**
1. reuse encoding compat (same elem type/encoding, or `neutral_reuse`)
2. cross-stage floor: `copies ≥ stageSpan` for every buffer in the block
3. slot-collision floor: `copies ≥ #entries` in the block after data-partition
   expansion (the `(accumCnt + theIdx) % numBuffers` math)
4. TMEM rotation semantics valid: accumulator (per-outer-tile) vs operand
   (per-inner-iter) blocks not mixed incompatibly
5. TMEM column-share only among liveness-disjoint (or subslice-fitting) buffers

**Feasible (resources):**
6. `Σ_block max(size)·copies ≤ smem_budget` (SMEM pool)
7. TMEM blocks fit `512 rows × cols` (minus reserved scale cols)

These are the same five hazard checks discussed as the "static hazard checker,"
here repurposed as the branch predicate that carves the legal∧feasible region out
of the raw partition space.

### 2.3 The copy factorization (why the search is tractable)

The raw space is Bell-number huge (partitions) × copy counts × placements. But
**copies factor out**: given a fixed grouping+placement, each block's
footprint-per-copy and its floor are fixed, and the latency benefit of the c-th
copy is

```
marginal(block, c) = freq · II   while c·II < L_block,   else 0     (concave)
```

Separable concave gains under a shared budget → **greedy-by-marginal-benefit-per-byte
is the exact optimum** (concave resource allocation). Therefore:

- **Branch only over grouping + placement** (the genuinely combinatorial part).
- **Solve copies in closed form** per grouping via the concave knapsack — exact,
  fast, no branching. This also yields the plan's score.

---

## 3. Architecture (modular)

### 3.1 The invariant that makes ordering swappable

> **Legality and feasibility are checked explicitly against the partial plan —
> never inferred from a buffer's position in the order.**

Every check is an order-agnostic predicate (interval disjointness, entry count,
type check). Ordering then affects only which partials the beam keeps and how hard
the bound prunes — never whether a plan is admitted. This is what lets
`TopologicalOrder` drop in later with zero changes to legality/feasibility/cost.

Honest consequence: under a *beam* (approximate), changing the order changes the
*result set*. Under full branch-and-bound it changes only speed. Ordering is a
quality knob, not a correctness knob.

### 3.2 Module seams

```
BufferModel     — normalized per-buffer facts (built once from IR + schedule annotations)
OrderingPolicy  — buffer → sequence                         [SWAPPABLE]
Packer          — what a block is; join/place/footprint     [per-kind: SMEM | TMEM]
CostModel       — leaf score + admissible/optimistic bound  [SWAPPABLE]
CopySolver      — copies given a grouping (concave knapsack) [SWAPPABLE]
BeamSearch      — generic driver over all of the above       (kind-agnostic)
```

Interfaces (land in `WSMemoryPlanSearch.h`):

```cpp
struct BufferModel {
  ArrayRef<BufferId> buffers();
  Footprint      size(BufferId);        // bytes (SMEM) or rows×cols (TMEM)
  Interval<size_t> liveness(BufferId);
  unsigned       stageSpan(BufferId);   // cross-stage floor
  unsigned       entries(BufferId);     // data-partition expansion → slot floor
  EncodingKey    encoding(BufferId);    // reuse-compat key
  BufferKind     kind(BufferId);        // TMA / operand / accumulator / staging
  double         latency(BufferId);     // L_b  (from tt.self_latency)
  double         freq(BufferId);        // trip count
  bool           dependsOn(BufferId, BufferId);   // def-use — topo order & TMEM reuse
};

struct OrderingPolicy {                 // the swap seam
  virtual SmallVector<BufferId> order(const BufferModel&) const = 0;
};

struct Packer {                         // SmemPacker | TmemPacker
  virtual bool      legalJoin(const PartialPlan&, BufferId, BlockId) const = 0;
  virtual bool      feasible (const PartialPlan&, Budget) const = 0;
  virtual Footprint footprint(const Block&) const = 0;
  virtual Placement place    (const Block&, BufferId) const = 0; // TMEM offsets; SMEM no-op
};

struct CostModel {
  virtual double score(const Plan&) const = 0;                    // Σ hidden_latency − λ·occ
  virtual double bound(const PartialPlan&, ArrayRef<BufferId> rest) const = 0;
};

struct CopySolver {
  virtual CopyMap solve(const Grouping&, Budget, const CostModel&) const = 0;
};

TopKPlans beamSearch(const BufferModel&, const OrderingPolicy&, const Packer&,
                     const CostModel&, const CopySolver&, unsigned W, unsigned K);
```

### 3.3 Engine loop (beam form)

```
seq  = ordering.order(model)            // the ONLY place ordering enters
beam = { emptyPartialPlan }
for b in seq:
    next = []
    for P in beam:
        for G in P.blocks if packer.legalJoin(P, b, G):
            P' = P.join(b, G)
            if packer.feasible(P', budget): next.push(P', costModel.bound(P', rest))
        P'' = P.openBlock(b)
        if packer.feasible(P'', budget): next.push(P'', costModel.bound(P'', rest))
    beam = top-W of next by bound        // log if it drops feasible partials (no silent cap)
for P in beam:
    P.copies = copySolver.solve(P.grouping, budget, costModel)
    topK.offer(P, costModel.score(P))
return topK
```

### 3.4 SMEM/TMEM independence

Two instantiations of the same engine, per the locked decision:

```cpp
auto smem = beamSearch(smemModel, *ordering, SmemPacker{}, *cost, *copies, W, K);
auto tmem = beamSearch(tmemModel, *ordering, TmemPacker{}, *cost, *copies, W, K);
```

They differ only in the `Packer` (SMEM = circular multi-buffer groups; TMEM =
row-owner + column subslice + two rotation semantics) and the budget (SMEM bytes /
512 rows). `OrderingPolicy`, `CostModel`, `CopySolver` are shared verbatim. Final
plans are the cross-product of the two top-K lists, scored by combined occupancy
(or kept separate — they are physically independent pools).

---

## 4. Cost model

The same function orders candidates offline and (via the knapsack) drives copy
allocation.

```
Score(P) = Σ_block hidden_latency(block, copies) − λ · occupancy_penalty(P)
hidden_latency(block, c) = freq · min(c · II, L_block)
```

- `L_block` = producer `tt.self_latency`, `freq` = trip count, `II` =
  `tt.modulo_ii`.
- `bound(partial, rest)` = optimistic completion: best-case latency hidden for the
  unplaced suffix assuming free reuse + full copies. Admissible ⇒ branch-and-bound
  is exhaustive w.r.t. top-K; under beam it is just the ranking key.
- Start with `λ = 0` (pure latency hiding); add the occupancy term only if
  benchmarks show high-copy plans losing to occupancy. Keeping `λ=0` first also
  makes the proxy easy to validate against runtime.

**Schedule dependence**: the cost model needs `II` + latencies, so a plan search
is always *relative to a fixed schedule's annotations*. When later folded into the
modulo-schedule top-K search, the plan enumeration re-runs per schedule (memory
pick is conditional on schedule pick).

---

## 5. Helper inventory (reuse / refactor / new)

All line numbers refer to `WSMemoryPlanner.cpp` unless noted. **reuse** = call
as-is; **refactor** = extract/generalize existing logic behind the new interface;
**new** = must be written.

### 5.1 `BufferModel` — normalized per-buffer facts

| Fact | Existing helper | Action |
|---|---|---|
| op-ID space for liveness | `MemoryPlannerBase::buildOperationIdMap` (:80) | reuse |
| SMEM size | `getSmemAllocSizeBytes` (:1306) | reuse |
| TMEM size | `ttng::getTmemAllocSizes` / `TMemAllocation` | reuse |
| liveness interval | `resolveLiveness` (:539), `resolveExplicitBufferLiveness` (:510), `livenessForTmemChannel` (:2422), `getAllAcutalUsersForChannel` (:266), `getAllTmemUsers` (:2391), `updateLiveOpsAcrossScopes` (:399), `getUserScopes` (:327), `getLiftedScope` (:304) | reuse |
| stage span (SMEM floor) | `getSmemCrossStageDepth` (:1248), `isSmemCrossStage` (:1233), `getLoopStage` (:1217), `getLoopCluster` (:1225) | reuse |
| stage span (**TMEM** floor) | — none — | **new** `getTmemCrossStageDepth` |
| entries → slot floor | inline in `enforceMinBufferCopy` (:689, counting per `buffer.id` at :664–704) | refactor → `slotFloor(block)` |
| encoding compat | `areReuseEncodingsCompatible` (:1702), `neutralReuseEnabled` (:1698) | reuse |
| kind (TMA/innermost/staging/accum) | `isSmemTMAChannel` (:1197), `isInnermostLoop` (:137), `usersInInnermostLoop` (:464), `isInnermostSmemChannel` (:1145), `WSBuffer.tmaStaging`, channel `isOperandD`, `hasLoopCarriedAccToken` (`CodePartitionUtility.cpp:64`) | reuse |
| **latency `L_b` / `II` / freq** | — none — | **new** (read `tt.self_latency`, `tt.modulo_ii`; freq from trip counts) |
| `dependsOn(a,b)` | `isDataDependent` (:2487), `alongDependencyChain` | reuse |
| the struct itself | `WSBuffer` (:853) ≈ SMEM model; TMEM uses `BufferT` | refactor → one generic `BufferModel` |

### 5.2 `OrderingPolicy`

| Impl | Existing | Action |
|---|---|---|
| `LivenessStartOrder` | `sortChannelsByProgramOrder` (:3862), `getWSBufferUsageOrder` (:873), `getLogicalProducerOp` (:233), `getLastConsumerOrder` (:1678) / `Detailed` (:1627), `ConsumerOrder` (:1621) | refactor → wrap behind interface |
| `TopologicalOrder` | edge oracle `isDataDependent` (:2487) | **new** driver (topo sort over existing edges) |

### 5.3 `SmemPacker`

| Method | Existing | Action |
|---|---|---|
| `footprint` (Σ max·copies) | `computeTotalSmem` (:1324) | reuse |
| `feasible` (budget) | budget check in `allocateSmemBuffers` (:1845); `smemBudget` auto-detect | refactor → `SmemPacker::feasible` |
| `legalJoin` (reuse pairing) | pairing + `findReuseCandidate` (:2073) in `allocateSmemBuffers`; `areReuseEncodingsCompatible`; slot/cross-stage floors | refactor → `SmemPacker::legalJoin` |
| epilogue disjoint-liveness fusion | `fuseEpilogueWSBuffers` (:1347), `fuseEpilogueBuffers` (:712), `allAllocsCompatible` (:191), `increaseFusedEpilogueCopies` (:1427) | reuse as a join capability |
| `place` | (SMEM sized by max — no offsets) | trivial no-op |

### 5.4 `TmemPacker`

| Method | Existing | Action |
|---|---|---|
| `legalJoin` | `hasPotentialReuse` (:3171), `findReuseChannel` (:3654), `samePartition` (:3518), `alongDependencyChain`, `checkOtherReuses`, `isDataDependent` (:2487) | reuse |
| `place` / `footprint` | `AllocationState` (:3064), `OwnerPlacement` (:3058), `addOwnerToState` (:3078), `computeColOffset` (:3204), `findReuseSpace`, `allocateNewSpace`, `applyAllocationState` (:3315), `tryAllocate` (:3235), `kNumRowGroups` | reuse / refactor (wrap behind `Packer`) |
| `feasible` (512) | 512-row/col limit checks; `getMinFutureScaleTmemCols` (:2587) | reuse |
| cross-stage floor | — none — | **new** `getTmemCrossStageDepth` |
| copy>1 legality guard | `hasLoopCarriedAccToken` (`CodePartitionUtility.cpp:64`) | **new** guard (reject non-accumulator `copies>1`; see §8) |

### 5.5 `CostModel` + `CopySolver`

| Piece | Existing | Action |
|---|---|---|
| `score` | crude precursor: `WSBufferPriority` buckets | **new** (latency-hiding; delete buckets) |
| `bound` (admissible) | — none — | **new** |
| `CopySolver` (concave knapsack) | Phase-4 iterative increase in `allocateSmemBuffers`; TMEM round-robin (:3010–3022); `enforceMinBufferCopy` (:689) floor | refactor → knapsack; keep floors as constraints |

### 5.6 `BeamSearch` engine

| Piece | Existing | Action |
|---|---|---|
| DFS/backtrack skeleton | `tryAllocate` (:3235) — first-fit DFS | refactor → generalize to beam + cost-bound + top-K |
| beam width / top-K queue | — none (returns first solution) | **new** (bounded priority queue, width `W`) |

### 5.7 Output / plumbing — reuse wholesale

The plan carrier already exists; each top-K plan → one `BufferDecisionList`.

| Piece | Existing | Action |
|---|---|---|
| plan ↔ decision | `BufferDecision` (:3825), `BufferDecisionList` (:3844), `extractBufferDecision` (:3873), `applyBufferDecision` (:3897), `serializeBufferDecisions` (:3916), `serializeBufferDecisionsToString` (:3954), `writeDecisionsToFile` (:4044), `readDecisionsFromFile` (:4062) | reuse |
| debug viz | `dumpBuffers` (:772), `dumpSmem/TmemBufferLiveness`, `getLocName` (:801) | reuse |

### 5.8 Net-new code (the real work)

1. **Unified `BufferModel`** — generalize `WSBuffer` + TMEM `BufferT` into one interface.
2. **Latency/freq readers** — `tt.self_latency`, `tt.modulo_ii`, trip-count freq (needs Step 0 annotations).
3. **`getTmemCrossStageDepth`** — TMEM cross-stage floor (no helper today).
4. **`CostModel::score` + `bound`** — latency-hiding objective and its optimistic bound.
5. **`CopySolver`** — concave-benefit knapsack (replaces two ad-hoc copy loops).
6. **Generic `BeamSearch`** — beam width + top-K queue + cost-bound pruning.
7. **`OrderingPolicy` interface + `TopologicalOrder`**; lift existing sorts behind `LivenessStartOrder`.
8. **`Packer` interface** — thin; wraps existing SMEM pairing and TMEM backtracking.
9. **TMEM copy>1 legality guard** — until the general per-inner-iter rotation exists (§8).

Everything in reuse/refactor is ~70% of the surface (liveness, sizes, TMEM
placement, reuse legality, and the entire serialization/output path). The
genuinely new code is the **cost model, the knapsack, the beam/top-K driver, and
the `BufferModel`/`Packer`/`OrderingPolicy` interfaces** that tie them together.

---

## 6. Implementation steps

Ordered by dependency. Each step is independently testable against a synthetic
`BufferModel` (no IR, no GPU) unless noted.

### Step 0 — Scheduler annotations (prerequisite)
- Emit `tt.self_latency` and `tt.issue_cycle` per op in the modulo/list
  scheduler (`ModuloSchedulePass.cpp` / `LLMSchedulePass.cpp`), sourced from
  `NVLatencyModel`. `tt.modulo_ii` and `loop.stage` already exist.
- Verify the attrs survive intervening passes down to `doMemoryPlanner`
  (`loop.stage` already makes this trip — precedent exists).
- **Test**: lit test asserting the attrs appear on ops after scheduling.

### Step 1 — `BufferModel` interface + SMEM builder
- New `WSMemoryPlanSearch.h` with the interfaces from §3.2.
- SMEM builder: generalize `WSBuffer` (:853). Wire each field to its existing
  helper: `getSmemAllocSizeBytes` (:1306), `resolveLiveness` (:539),
  `getSmemCrossStageDepth` (:1248), `areReuseEncodingsCompatible` (:1702),
  `isSmemTMAChannel`/`isInnermostSmemChannel` (:1197/:1145), `WSBuffer.tmaStaging`.
- `entries()`: **refactor** the per-`buffer.id` counting from
  `enforceMinBufferCopy` (:689, :664–704) into a standalone `slotFloor` helper.
- `latency()`/`freq()`: **new** readers for the Step-0 attrs.
- `dependsOn()`: reuse `isDataDependent` (:2487).
- **Test**: synthetic + one real kernel; assert fields match the current planner's
  derived values.

### Step 2 — `OrderingPolicy` + `LivenessStartOrder`
- Lift the existing sorts (`sortChannelsByProgramOrder` :3862,
  `getWSBufferUsageOrder` :873, `getLogicalProducerOp` :233,
  `getLastConsumerOrder` :1678) behind `LivenessStartOrder`.
- Register via a factory keyed by `--mem-plan-order=liveness|topo`.
- **Test**: deterministic order on a synthetic model.

### Step 3 — `SmemPacker`
- `legalJoin`: **refactor** the reuse-pairing legality from `allocateSmemBuffers`
  (:1845) + `findReuseCandidate` (:2073), plus checks 1–3. Include the
  disjoint-liveness epilogue fusion (`fuseEpilogueWSBuffers` :1347,
  `allAllocsCompatible` :191) as a join capability.
- `footprint`: reuse `computeTotalSmem` (:1324).
- `feasible`: **refactor** the budget check from `allocateSmemBuffers`.
- `place`: no-op.
- **Guard**: keep every check an explicit predicate — no "processed in order so…"
  shortcuts (protects the ordering seam).
- **Test**: legalJoin accept/reject table; feasibility at budget boundary.

### Step 4 — `CopySolver` (concave knapsack)
- **Refactor** the Phase-4 iterative increase (`allocateSmemBuffers`) and the TMEM
  round-robin (:3010–3022) into one knapsack. Keep the floors (`minCopies`,
  cross-stage, slot-collision) as hard constraints seeded from `BufferModel`.
- **Test**: knapsack optimality vs brute force on small instances; floor
  enforcement.

### Step 5 — `CostModel`
- Implement `score` (§4) and an admissible `bound`. Delete the `WSBufferPriority`
  buckets once parity is shown.
- **Test**: monotonicity (more hidden latency ⇒ higher score); bound ≥ any
  completion's score on small instances.

### Step 6 — `BeamSearch` engine
- **Refactor** the DFS skeleton of `tryAllocate` (:3235) into the generic driver:
  add beam width `W`, a bounded top-K priority queue, and cost-bound pruning.
- **Test**: on a synthetic model, beam with `W=∞` reproduces brute-force top-K.

### Step 7 — `TmemPacker`
- Wrap the existing TMEM backtracking placer behind `Packer`: `hasPotentialReuse`
  (:3171), `findReuseChannel`, `computeColOffset` (:3204), `AllocationState`
  (:3064), `addOwnerToState` (:3078), `applyAllocationState` (:3315),
  `allocateNewSpace`, 512 limit, `getMinFutureScaleTmemCols` (:2587).
- **New** `getTmemCrossStageDepth` (TMEM cross-stage floor — no helper today).
- **Legality guard**: until the general per-inner-iteration TMEM copy path exists
  in `createBufferPost` (see [AccumulationCounters.md] / §8), `legalJoin` must
  reject `copies>1` for non-accumulator TMEM blocks. Accumulator multi-copy
  (per-outer-tile) is already supported via `hasLoopCarriedAccToken`.
- **Test**: reuse legality table; 512-row overflow rejection.

### Step 8 — TMEM `BufferModel` builder
- TMEM facts via `livenessForTmemChannel` (:2422), `getAllTmemUsers` (:2391),
  `ttng::getTmemAllocSizes`, channel `isOperandD`, `hasLoopCarriedAccToken`.
- **Test**: parity with current TMEM liveness/sizes.

### Step 9 — Wire into `doMemoryPlanner`
- Add `smem-plan-search` / `tmem-plan-search` pass options (default off).
- When on: build models → run `beamSearch` twice → emit the top-1 plan via
  `applyBufferDecision` (:3897). Emit top-K as ranked `BufferDecisionList`s
  (reuse `writeDecisionsToFile` :4044) for the downstream autotuner pick.
- Keep the heuristic path as the default fallback.
- **Test**: autoWS correctness suite (see `autows-testing` skill) with search on,
  top-1 plan, on the standard GEMM/FA kernels.

### Step 10 — `TopologicalOrder` (deferred, validates the seam)
- New `OrderingPolicy` impl using `isDataDependent` edges → topo sort.
- No changes to any other module. If this requires touching legality/feasibility,
  the §3.1 invariant was violated — fix that instead.
- **Test**: same top-K quality or better on kernels where reuse is
  dependency-dominated (TMEM).

---

## 7. Testing strategy

- **Unit** (per module, synthetic `BufferModel`, no GPU): ordering, legalJoin,
  feasible, knapsack, cost/bound, beam.
- **Parity**: search top-1 vs current heuristic on standard kernels — must be
  feasible and no worse in SMEM/TMEM footprint.
- **Correctness**: autoWS suite with search enabled (`autows-testing` skill).
- **Hazard**: replay the FA-bwd TMEM-reuse and cross-stage deadlock cases
  ([BwdTmemReuseSlotHazard.md], partition-scheduler-bugs #8) — the static gate
  must reject the deadlocking plans.

---

## 8. Open items / deferred

1. **Branch-and-bound upgrade**: replace beam with admissible-bound B&B for
   exhaustive top-K once the bound is tight enough to prune.
2. **General TMEM copy rotation**: the per-inner-iteration semantics in
   `createBufferPost` (blocks non-accumulator TMEM `copies>1`). Until then the
   Step-7 legality guard is in force.
3. **Joint schedule search**: fold `mem_decision_pick` into the modulo top-K
   config vector (conditional on schedule pick) — cross-turn follow-up.
4. **`λ` occupancy term**: enable only if benchmarks show occupancy losses.
5. **SMEM/TMEM coupling**: current plan keeps them independent; revisit if a
   kernel has buffers that can live in either pool.
