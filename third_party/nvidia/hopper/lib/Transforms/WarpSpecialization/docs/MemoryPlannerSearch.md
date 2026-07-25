# Memory Planner Search — a high-level guide

Audience: engineers touching the AutoWS memory planner. This is the mental
model + the key helpers + the knobs. For the historical design rationale and the
N-way reuse work see `MemoryPlannerSearchSpace.plan.md`; for the SMEM side see
`SmemAllocationDesign.md`; for TMEM heuristics detail see
`TMEMAllocationHeuristics.md`.

All line numbers are in
`third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/` and drift with
edits — treat them as "start here", not exact.

---

## 1. What it does & where it runs

`doMemoryPlanner` (`WSMemoryPlanner.cpp:5220`) assigns every warp-specialized
buffer a physical location so producers/consumers in different partitions can
share on-chip memory safely. It runs after channel creation and before code
partitioning emits the reuse barriers. It plans two pools independently:

- **SMEM** — bytes; multi-buffering + circular reuse. Optional plan-space beam
  search (`smemPlanSearch` / `TRITON_WS_SMEM_PLAN_SEARCH`). See
  `SmemAllocationDesign.md`.
- **TMEM** — Blackwell tensor memory, a hard **512-column** budget. This doc is
  about the TMEM search.

Output: each `ttng.tmem_alloc` gets `buffer.id` (which slot), `buffer.offset`
(column within the slot), and `buffer.copy` (multi-buffer depth). Allocs sharing
a `buffer.id` form a **reuse group**.

## 2. Mental model

- **Buffer** (`BufferT`): one TMEM alloc, with a liveness interval
  (`bufferRange`) and a column width (`colSize`, in *physical* TMEM columns —
  f16 packs 2 elems/column, so a 128×128 f16 is 64 cols, f32 is 128).
- **Reuse group** = one **owner** (holds the slot) + zero or more **reusers**
  that share the owner's columns. A reuser is placed at a `colOffset`:
  - **offset 0 = time-multiplex**: same columns, disjoint-in-time (safe only if
    the members have a happens-before order).
  - **offset > 0 = spatial packing**: side-by-side columns (always safe; costs
    more columns).
- **Owner identity is emergent**: a buffer becomes an owner only when it can't
  reuse an existing one and opens *new space*. Nobody "declares" owners.
- **Peak columns** = the max simultaneous column extent; must stay ≤ 512.

## 3. The three layers

The planner is best understood as **heuristics → search → post-pass**. The
search is entirely driven by the heuristics; the post-pass repairs what the
incremental search can't express.

```
                 ┌─────────────── heuristics (define space + ranking) ───────────────┐
                 │ hasPotentialReuse · computeColOffset · findPlacements ·            │
                 │ addOwnerToState · tmemStatePeakCols · canJoinReuseGroupChain       │
                 └───────────────────────────────────────────────────────────────────┘
                                     ▲                       ▲
        default ────────────────────┘                       └──────────── opt-in
   tryAllocate (first-fit)                          enumerateTMemAllocations (top-K)
   returns first feasible packing                   all distinct packings, ranked,
                                                     mem_plan_pick selects
                                     │                       │
                                     └──────────┬────────────┘
                                                ▼
                                repairUnsafeReuseGroups (post-pass)
                                  order-independent safety repair
                                                ▼
                                     applyAllocationState (stamp IR)
```

TMEM has **two allocators**, chosen per-loop by the `tt.tmem_alloc_algo` IR
attr (default 1):
- **algo-1** `allocateTMemAllocs` (`:4378`) — single-pass greedy
  (`findReuseChannel`). Simple, no backtracking.
- **algo-2** `allocateTMemAllocs2` (`:4230`) — the search below. Opt-in via
  `tt.tmem_alloc_algo=2` (e.g. FA-bwd). Everything in this doc is algo-2.

### 3.1 The algorithm as implemented (algo-2)

All of a loop's TMEM buffers are allocated together. Every layer operates on an
`AllocationState` — the set of **owners** (each holding a physical column range)
plus a map from each **reuser** to its `(owner, colOffset)`. Annotation-pinned
allocs seed the initial state; the search fills in the rest.

**Layer 1 — heuristics** define the *space*, never a plan. For a candidate
buffer they answer three questions, each an order-independent predicate (so the
visitation order can change only what is *found first*, never what is *legal*):
- *Can `cand` reuse `owner`?* — `hasPotentialReuse` returns >0 iff their liveness
  intervals are disjoint **and** they are bidirectionally data-dependent.
  Liveness-disjoint-but-independent buffers are refused on purpose: across warp
  partitions they can be concurrent at run time despite disjoint op-id intervals.
- *Where would it land?* — `computeColOffset` returns `0` when `cand` can
  time-multiplex the owner's columns with *every* current reuser, else the next
  free spatial offset, else MAX ("doesn't fit").
- *If it reuses nothing, what new slot does it open?* — `findPlacements` /
  `addOwnerToState`.

**Layer 2 — search** turns those predicates into a concrete packing.
- *Default — `tryAllocate` (first-fit):* a recursive DFS over the buffers. At
  each buffer it collects every owner with `hasPotentialReuse > 0` that also
  fits (`computeColOffset != MAX`), **orders them deterministically** — priority
  desc, then lowest `colOffset`, then owner liveness start — tries each in turn,
  and if none completes, opens new space. It returns the **first** state in which
  all buffers fit the 512-column budget. The tie-break by lowest `colOffset` is a
  correctness property, not just churn: it prefers the safe offset-0
  time-multiplex over a co-live spatial slot, whereas the old pointer-order tie
  corrupted an accumulator (`dq`) run-to-run.
- *Opt-in — `enumerateTMemAllocations` (top-K):* the same candidate logic, but it
  never returns early — it walks every branch and collects all *distinct*
  feasible packings (deduped by a canonical per-alloc column signature), bounded
  by a recursion budget and a solution cap. Packings are ranked by peak columns
  (`tmemStatePeakCols`, tighter first); first-fit is pinned to **rank 0** so
  `mem_plan_pick=0` is byte-identical to the default, and `mem_plan_pick=i`
  applies rank `i`.

**Layer 3 — post-pass** (`repairUnsafeReuseGroups`, runs on *both* paths).
First-fit builds groups incrementally in buffer order, so it cannot assemble a
common-ancestor sibling group whose "bridge" member sorts last (FA-bwd
`{dpT,dsT,dq}`: `dq`↔`dsT` have no pairwise dep). The repair scans the finalized
groups; for any group that is **not** a unique dependency chain (`≥3` members,
unorderable under the strict gate) it relocates one reuser into another group
where it *does* form a chain-orderable slot — but only when removing that reuser
also leaves the source group orderable, so every move strictly reduces the count
of unsafe groups and the loop converges. `canJoinReuseGroupChain` gates the move
and requires the target be orderable under **both** the strict formation gate and
the loose code-partition gate, so a repaired group can never trip code
partitioning's downstream `report_fatal_error`. The pass is **inert** unless
first-fit actually produced an unorderable group — that is why default packings
never change.

Finally `applyAllocationState` stamps `buffer.id` / `buffer.offset` /
`buffer.copy` onto the IR (TMEM `buffer.copy` is pinned to 1 — see §3.2).

### 3.2 What can be improved later

- **Ranking is occupancy-only.** The top-K search ranks packings by peak columns,
  not by a perf/latency cost model. The plan-space `CostModel` (latency-hiding,
  in `WSMemoryPlan*`) drives the SMEM/TMEM *beam* but is **not** wired into
  algo-2's enumeration, so `mem_plan_pick` currently sweeps occupancy-ranked
  plans. Wiring the cost model (and its still-stubbed `entries`/`freq` inputs)
  would make rank 0 perf-best rather than merely tightest.
- **N-way grouping is reactive.** The sibling group is assembled by the post-pass
  *after* first-fit mis-groups it, not chosen during the search. Teaching
  `tryAllocate` / enumeration to form `canJoinReuseGroupChain` groups directly
  would remove the repair round-trip and cover cases the one-move-per-fixpoint
  repair can miss.
- **TMEM multi-copy is pinned to 1.** Only loop-carried accumulators can be
  multi-buffered (per outer tile); non-accumulator `copies>1` is not yet legal,
  so the copy dimension is not searched for TMEM.
- **Coverage gaps fall back** to the heuristic / algo-1: scaled-MMA scale-column
  reservation and subtiled regions are unmodeled by the search.
- **Reuse fit-check is columns-only** — `hasPotentialReuse` (shared by first-fit
  *and* the repair's `canJoinReuseGroupChain`) checks `colSize` + liveness +
  data-dep but **not** `rowSize`/row-group. Harmless today because every TMEM
  alloc is 128 rows, but a mixed-row kernel could place a taller reuser into a
  shorter owner's slot: at apply time the reuser inherits the owner's `rowOffset`
  yet keeps its own row count, so it spills past the owner's reserved row group →
  out-of-bounds / aliasing TMEM at materialization (MLIR verifier reject at best,
  wrong results at worst — *not* silently safe). The fix is a `rowSize`/row-group
  fit-check in `hasPotentialReuse` (covers both paths), which rejects the reuse
  early so the buffer opens its own space. Tracked in T280266869.
- **Repair is greedy** — one relocation per call to a fixpoint; a global
  re-assignment could find better partitions in pathological cases.

## 4. Key helpers (algo-2)

| Helper | Where | Role |
|---|---|---|
| `hasPotentialReuse(owner, cand)` | `:3742` | **Pairwise legality.** Returns >0 if `cand` may reuse `owner`: liveness-disjoint **and** a bidirectional data dependency. The binding constraint on what groups can form. |
| `computeColOffset(cand, owner)` | `:3775` | **Placement.** 0 (time-multiplex) if `cand` can share with all current reusers, else the next free spatial offset, else MAX (doesn't fit). |
| `findPlacements` / `addOwnerToState` | `:3649` | **New-space owner.** Column range for a buffer that opens a fresh slot. |
| `tryAllocate(...)` | `:3907` | **First-fit search** (default). Recursive DFS: try each reuse candidate, then new-space; return the first packing that fits 512. |
| `enumerateTMemAllocations(...)` | `:4128` | **Top-K search** (opt-in). Same legality as `tryAllocate`, but collects *all distinct* feasible packings for ranking. |
| `tmemStatePeakCols(state)` | `:4084` | **Ranking key** — peak column occupancy (lower = tighter). |
| `canJoinReuseGroupChain(owner, cand)` | `:3821` | **N-way sibling formation.** Lets `cand` join a group with **no pairwise dep** when the whole group (≥3) is a unique dependency chain under *both* the sound (formation) and loose (code-partition) edge policies. Called only by the post-pass (`repairUnsafeReuseGroups`). |
| `repairUnsafeReuseGroups(...)` | `:3856` | **Post-pass.** Over the finalized packing, relocate a reuser out of any non-chain-orderable group into one where it forms a chain-orderable slot. Order-independent; inert unless first-fit produced an unorderable group. |
| `applyAllocationState(...)` | `:4008` | Stamp `buffer.id`/`buffer.offset`/`buffer.copy` onto the IR from the chosen state. |

**The sound predicate** (shared with code partitioning, in
`CodePartitionUtility.cpp`):

| Helper | Where | Role |
|---|---|---|
| `orderReuseGroupChain(group, crossPartitionProgOrder=true)` | `:1070` | Kahn topo-sort of a group's channels into a **unique** dependency chain; empty if none exists. The N-way legality oracle. |
| `hasDependencyChain(A, B, crossPartitionProgOrder=true)` | `:884` | Edge test: data dep (`dependsThroughMemory`), or program order. With `crossPartitionProgOrder=false` (the group-formation gate) it drops cross-partition textual order — which is **not** a happens-before — so independent cross-partition siblings are refused. |
| `verifyReuseGroup2` / `verifyReuseGroupCrossPartition` | `:994` / `:1264` | 2-way and N-way group verdicts used by code partitioning. |

> The planner's group-formation gate calls `orderReuseGroupChain(..., false)`
> (strict). Code partitioning keeps the default (`true`). Since strict ⊂ lax,
> anything the planner forms is always orderable/materializable downstream.

## 5. Control flow (`allocateTMemAllocs2`, `:4230`)

1. **Dispatch** (`:4301`): `if (topK > 1 || pick > 0)` run the top-K enumeration
   and apply the picked rank; **else** run `tryAllocate` (first-fit). Default
   (`topK=1, pick=0`) is byte-identical to plain first-fit; rank 0 is pinned to
   first-fit even under search.
2. **Post-pass** (`:4293` region): `while (repairUnsafeReuseGroups(...))` — fix
   any unorderable group left by first-fit. Runs on both paths.
3. **Stamp**: `applyAllocationState`.

Why the post-pass exists: first-fit builds groups **incrementally in buffer
order** and can't assemble a common-ancestor sibling group whose "bridge" member
sorts last (FA-bwd `{dpT,dsT,dq}`: dq↔dsT have no pairwise dep). The post-pass
repairs this order-independently. See `MemoryPlannerSearchSpace.plan.md` for the
full story.

## 6. Knobs

| Knob | Effect |
|---|---|
| `tt.tmem_alloc_algo` (IR attr on the `scf.for`) | 1 = greedy (default), 2 = backtracking search. |
| `TRITON_WS_MEM_PLAN_TOPK=K` | Enumerate K ranked TMEM packings (opt-in search). |
| `tt.mem_plan_pick` (IR attr, from `tl.range(mem_plan_pick=...)`) / `TRITON_WS_MEM_PLAN_PICK` | Apply ranked plan `pick` (0 = cost-best; autotune-native via the constexpr). |
| `TRITON_WS_MEM_PLAN_TOPK_DUMP=<path>` | Dump the ranked plans (JSON) for an external sweep harness. |
| `TRITON_WS_SMEM_PLAN_SEARCH` | Enable the SMEM plan-space beam search. |

## 7. Debugging

- **Verify harness** (`TRITON_WS_MEM_PLAN_VERIFY_GROUPS=1`): the planner
  enumerates every candidate pair/triple of a loop's TMEM allocs and prints a
  `[ws-mem-plan-verify] group {..} => ACCEPT/REJECT` verdict per group. Changes
  no planning decision. Lit test:
  `test/Hopper/WarpSpecialization/ws_mem_plan_verify_reuse_groups.mlir`.
- **LDBG** (`TRITON_LLVM_DEBUG_ONLY=nvgpu-ws-memory-planner`): buffer allocation
  order, the `hasPotentialReuse` matrix, `tryAllocate` decisions, and the Final
  Allocation. `nvgpu-ws-utility` prints `verifyReuseGroup2`/`orderReuseGroupChain`
  verdicts (also used by `verify_reuse_group_decisions.mlir`).

## 8. Invariants

- **512-column budget** for TMEM (`kMaxTMemCols`).
- **No-regression guard**: the default path (`topK=1, pick=0`) is byte-identical
  to first-fit, and the post-pass is inert unless first-fit produced an
  unorderable (≥3, single-copy) group — so packings first-fit already gets right
  never change.
- **Planner ⊆ code-partition orderability**: the strict group-formation gate
  guarantees code partitioning can always emit the reuse barrier.

## 9. The plan-space beam search (`WSMemoryPlanSearch.{h,cpp}`)

A **second, modular allocator**, separate from the algo-2
`tryAllocate`/enumerate/repair path in §3–§5. Opt-in via `smemPlanSearch` /
`TRITON_WS_SMEM_PLAN_SEARCH` (one flag, both pools). It drives:
- **SMEM** — `allocateSmemBuffersViaSearch` (whenever the flag is on).
- **TMEM** — `allocateTmemBuffersViaSearch` (flag on *and* no unmodeled feature;
  otherwise the §3–§5 `MemoryPlannerTmem` path runs).

It treats the planner's **output** (grouping + copies + placement) as the search
variable, built from six swappable seams. Each is a pure predicate/policy, so the
ordering can change *which* plans are found without changing *what is legal* (the
design invariant).

### 9.1 `BufferModel` — normalized per-buffer facts
Abstract read-only view built once from IR: `size` (`Footprint` — bytes for SMEM,
rows×cols for TMEM), `liveness` (op-order `Interval`), `stageSpan` (cross-stage
copy floor), `entries` (slot-collision floor — **SHIPPED: stubbed to 1**),
`encoding` (reuse-compat key), `kind` (TMALoad/Operand/Accumulator/Staging/
Other), `reuseScope` (SMEM same-block group gate), `latency` (producer latency
from `ttng::NVLatencyModel`, on demand), `freq` (trip count — **SHIPPED: stubbed
to 1.0**), `dependsOn(a,b)` (forward slice via the shared `dependsThroughMemory`).
Two impls:
- **`SmemBufferModel`** — one record per `local_alloc`; liveness from first/last
  user in op order; `kind` from users (TMA store → Staging, innermost TMA →
  TMALoad, else Operand); `reuseScope` = the alloc's basic block (users spanning
  blocks → a unique, ungroupable scope).
- **`TmemBufferModel`** — one record per `tmem_alloc`; footprint rows×cols;
  `kind` = Accumulator iff the channel is operand-D else Operand; `reuseScope` =
  unique per buffer (TMEM reuse is column-subslicing, never SMEM-style grouping).

### 9.2 `OrderingPolicy` — the swap seam
Produces the sequence buffers are offered to the beam; must not affect legality.
`createOrderingPolicy(name)`:
- **`LivenessStartOrder`** (default `"liveness"`) — by liveness start, then
  `BufferId` for determinism.
- **`TopologicalOrder`** (`"topo"`) — deterministic Kahn sort over `dependsOn`
  edges (liveness-start/BufferId tiebreak; a defensive fallback breaks cycles).

Callers pass `"liveness"`.

### 9.3 `Packer` — per-pool block semantics (`legalJoin` / `feasible` / `footprint` / `place`)
- **`SmemPacker`** — a block is a circular multi-buffer group sized
  `max(member bytes)·copies`. **`legalJoin` returns `false` unconditionally**: the
  search does no SMEM reuse grouping (encoding + basic block is not a safe reuse
  condition — see §3.2), so every SMEM buffer gets its own block and only its copy
  count is tuned. `feasible` = Σ block bytes ≤ budget; `place` = no-op.
- **`TmemPacker`** — a block is a row-owner whose members time-multiplex the same
  rows×cols at offset 0. **`legalJoin` = matching encoding AND disjoint liveness
  AND bidirectional `dependsOn`** (the dependency is load-bearing — independent
  liveness-disjoint buffers can be concurrent across partitions). `footprint` =
  max(member rows)×max(member cols); `feasible` = per-block cols ≤ `tmemCols` and
  Σ rows ≤ `tmemRows`; `place` = no-op (offset 0). Copies pinned to 1 by the
  caller.

### 9.4 `CostModel` — objective + bound
`LatencyCostModel(model, II, lambda=0)`:
`Score(P) = Σ_block Σ_member freq·min(copies·II, latency) − λ·occupancy`.
`score` is the exact value of a *complete* plan; `bound` is an admissible,
optimistic upper bound (assume full latency hidden, drop the ≥0 penalty) for a
future branch-and-bound. `II` = `tt.modulo_ii` (`getModuloII`, default 1); `λ`
defaults to 0 (pure latency hiding — the occupancy term is a deferred open item).

### 9.5 `CopySolver` — copies per block (concave knapsack)
`GreedyCopySolver`, given a fixed grouping: (1) set each block's **correctness
floor** = `max(max stageSpan, Σ entries, 1)`, exempt from budget; (2) greedily
add discretionary copies by **benefit density** (`Δscore / Δfootprint`) until no
block has positive benefit that still fits budget. Per-copy latency benefit is
concave, so greedy is the exact optimum for this separable knapsack. The
`CostModel` is the sole source of the objective (marginal benefit = score delta).

### 9.6 `beamSearch` — the driver
Places buffers one at a time in the ordering. Each partial branches into (a) join
`b` into every existing block the `Packer` deems legal + feasible and (b) open a
new block for `b`. Surviving partials are ranked by their **copy-solved score**
(`scoreWithCopies` runs the `CopySolver` + `CostModel`) — *not* `bound()`, since
all partials at a level share the same placed prefix — and truncated to beam
width `W`. Leaves are finalized (copies + score) and the top `K` returned. The
copy dimension is solved in closed form per grouping, never branched. Callers use
`W = max(16, topK)`, `K = topK`.

### 9.7 Wiring & safety net
`allocateSmemBuffersViaSearch` / `allocateTmemBuffersViaSearch` build the model,
run `beamSearch`, then stamp `buffer.id` / `buffer.copy` (/ `buffer.offset`) from
the picked plan (rank via `mem_plan_pick`). Both **fall back** to the heuristic
path for unmodeled features — SMEM: annotation / atomic-broadcast pins, subtiled
regions, multi-store TMA staging; TMEM: scaled MMA (scale-column reservation) and
subtiled regions. Correctness floors (cross-stage depth, per-id entry count) are
re-applied after the search as a backstop, so a plan can never drop below the
proven floor.

## 10. See also
- `MemoryPlannerSearchSpace.plan.md` — N-way reuse design, dead ends, rationale.
- `TMEMAllocationHeuristics.md` — TMEM sizing/packing detail.
- `SmemAllocationDesign.md` — the SMEM pool.
- `ReuseGroups.md`, `BufferAllocation.md` — reuse-group materialization in code
  partitioning.
- `MemoryPlannerVisualization.md` — inspecting plans.
