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

## 9. See also
- `MemoryPlannerSearchSpace.plan.md` — N-way reuse design, dead ends, rationale.
- `TMEMAllocationHeuristics.md` — TMEM sizing/packing detail.
- `SmemAllocationDesign.md` — the SMEM pool.
- `ReuseGroups.md`, `BufferAllocation.md` — reuse-group materialization in code
  partitioning.
- `MemoryPlannerVisualization.md` — inspecting plans.
