# SMEM Allocation Redesign in Memory Planner

## Goal

Redesign the SMEM allocation in `MemoryPlanner::run()` so that:

1. Each `local_alloc` is modeled as a **WSBuffer**.
2. Every WSBuffer starts with a single copy (`buffer.copy = 1`).
3. WSBuffers that span multiple `loop.stage` must have at least 2 copies.
4. `num_buffers` (the `--num-buffers` pass parameter) determines the maximum copies.
5. Copies are incrementally increased for high-priority WSBuffers while
   fitting within the SMEM budget.
6. A pass option `--smem-circular-reuse` (default: off) gates all
   reuse-group pairing logic.
7. At each iteration we choose either a **single WSBuffer** or a **pair of
   WSBuffers** and increase the copies by 1:
   - A pair is chosen only when `--smem-circular-reuse` is on and there
     are **exactly two** WSBuffers at the current highest priority.
   - A chosen pair becomes a **reuse group** (sharing a `buffer.id`).
   - If the final copy count is even, the group is split back
     (each buffer gets `numCopies/2` with its own `buffer.id`).
   - A chosen single WSBuffer has **no reuse** (its own `buffer.id`).
8. After all WSBuffers at the highest priority are handled,
   proceed to the next level.

---

## Terminology

| Term | Meaning |
|------|---------|
| **WSBuffer** | A wrapper around one `ttg.local_alloc` op, tracking its size, liveness interval, channel properties, and allocation decisions (`buffer.id`, `buffer.copy`). |
| **num_buffers** | The `--num-buffers` pass parameter. Determines the maximum `buffer.copy` value for any WSBuffer. |
| **Reuse group** | A pair of WSBuffers that share a single `buffer.id`. The physical allocation is `max(size_A, size_B) * buffer.copy`. Only formed when `--smem-circular-reuse` is on. |
| **smem-circular-reuse** | Pass option (default: off). When on, enables reuse-group pairing in Phase 4. When off, every WSBuffer keeps its own `buffer.id`. |
| **Cross-stage** | A WSBuffer whose channel has producer and consumer(s) in different `loop.stage` values. |

---

## Algorithm

### Phase 1: Initialize — One WSBuffer Per `local_alloc`, All `copy = 1`

Walk the function in **deterministic order** (sorted by operation ID). For
each `ttg.local_alloc` that is a shared memory alloc, create a **WSBuffer**:

```cpp
struct WSBuffer {
    Operation *allocOp;        // the local_alloc
    unsigned   sizeBytes;      // numElems * elemBitWidth / 8
    Interval<size_t> liveness; // [firstUser, lastUser)
    bool       isInnermost;    // users all in innermost loop, 2D+ shape
    bool       isTMA;          // channel source is TMA/descriptor_load
    bool       isCrossStage;   // src and dst in different loop.stage
    unsigned   bufferId;       // assigned buffer.id
    unsigned   numCopies;      // assigned buffer.copy (starts at 1)
};
```

All WSBuffers start with:
- A unique `bufferId` (0, 1, 2, …)
- `numCopies = 1`

### Phase 2: Enforce Cross-Stage Minimum

Any WSBuffer with `isCrossStage == true` must have at least 2 copies
(as long as `num_buffers >= 2`). For each such WSBuffer, set `numCopies = 2`.

Note: no budget check is performed here. The total SMEM may temporarily
exceed the budget after this phase. Phase 4 will resolve this — either by
grouping cross-stage buffers into reuse groups (which reduces physical SMEM)
or by confirming the allocation fits. If Phase 4 cannot bring the total
within budget, it reports the failure.

### Phase 3: Classify and Prioritize

Sort WSBuffers into priority levels. Only **innermost-loop** WSBuffers are
candidates for further copy increases. The `isCrossStage` property does
**not** affect priority — it only enforces a minimum copy count in Phase 2.

| Priority | Criteria | Description |
|----------|----------|-------------|
| **P0** (highest) | `isInnermost && isTMA` | TMA loads in innermost loop. Most critical for multi-buffering. |
| **P1** | `isInnermost && !isTMA` | Non-TMA innermost buffers. Lower priority. |
| **P2** (lowest) | `!isInnermost` | Outside-loop or non-innermost buffers. Stay at current copies. |

### Phase 4: Iterative Copy Increase

Process each priority level from P0 to P1 (P2 is never increased).

A pass option `--smem-circular-reuse` (default: off) controls whether
reuse-group pairing is attempted. When off, every WSBuffer keeps its own
`buffer.id` and only individual copy increases are tried.

#### Algorithm

For a given priority level with a set of candidate WSBuffers:

```
candidates = WSBuffers at this priority

# ── Step 0: Decide grouping upfront ──────────────────────────────
#
# When smem-circular-reuse is on and there are exactly 2 candidates,
# tentatively group them into a reuse group. The incremental loop
# operates on the group as a unit. After the loop, if the final
# copy count is even, the group is split back (Step 2) since each
# buffer gets exactly half — no circular reuse benefit.
#
# The group's starting copies must satisfy the cross-stage constraint:
# if any member has isCrossStage (needing N=2 individual copies),
# the group needs at least 2*N - 1 = 3 copies so that each member
# retains at least N effective pipeline slots.

reuseGroup = null

if smem_circular_reuse AND |candidates| == 2:
    reuseGroup = form reuse group (A, B)
    B.bufferId = A.bufferId            # B shares A's buffer.id
    maxCrossStageMin = max(A.crossStageMin, B.crossStageMin)  # 2 or 1
    if maxCrossStageMin >= 2:
        reuseGroup.numCopies = maxCrossStageMin * 2 - 1       # e.g., 3
    else:
        reuseGroup.numCopies = 1

# ── Step 1: Incremental loop ─────────────────────────────────────

if reuseGroup:
    currentGroupCopies = reuseGroup.numCopies
else:
    currentGroupCopies = 1

foundValidSolution = false

while currentGroupCopies <= num_buffers:

    if reuseGroup:
        # ── Reuse group path (handled separately) ────────────
        tentatively set group copies = currentGroupCopies
        if totalSmem(tentative) <= smemBudget:
            commit: reuseGroup.numCopies = currentGroupCopies
            currentGroupCopies += 1
            foundValidSolution = true
        else:
            break  # budget exhausted

    else:
        # ── Individual WSBuffers path ────────────────────────
        pending = [c for c in candidates if c.numCopies < currentGroupCopies]

        if not pending:
            currentGroupCopies += 1
            continue

        advanced_any = false
        for each wsBuffer in pending:
            tentatively set wsBuffer.copies = currentGroupCopies
            if totalSmem(tentative) <= smemBudget:
                commit: wsBuffer.numCopies = currentGroupCopies
                advanced_any = true
                foundValidSolution = true
            else:
                continue  # try next candidate at this level

        if not advanced_any:
            break  # budget exhausted, done with this priority

        currentGroupCopies += 1

# ── Step 2: Finalize reuse decision ──────────────────────────────
#
# If the reuse group's final numCopies is even, there is no benefit
# from circular reuse — each buffer would get exactly numCopies/2
# effective copies. Split the group back into separate buffers.

if reuseGroup AND reuseGroup.numCopies is EVEN:
    half = reuseGroup.numCopies / 2
    A.numCopies = half
    B.numCopies = half
    B.bufferId = nextBufferId++    # restore B's own buffer.id
    reuseGroup = null

# ── Step 3: Validate ─────────────────────────────────────────────
#
# After the loop, check if we found any allocation that fits.
# This catches cases where even the minimum required copies (e.g.,
# cross-stage group at 3 copies) exceeds the budget.

if not foundValidSolution:
    report error: cannot fit SMEM allocation within budget
```

#### Initial value of `currentGroupCopies`

| Scenario | Initial value | Why |
|----------|:---:|-----|
| Reuse group, one member cross-stage (N=2) | **3** (`2*2-1`) | Ensures the cross-stage member retains ≥2 effective pipeline slots |
| Reuse group, no cross-stage members | **1** | No constraint; start from bottom |
| No reuse group | **1** | Each WSBuffer increments individually |

#### Advancement of `currentGroupCopies`

`currentGroupCopies` advances by 1 after each level is processed:
- **Reuse group path:** try to bring the group to `currentGroupCopies`,
  then advance. No iteration over pending — the group is a single unit.
- **Individual path:** iterate over all pending WSBuffers at this level,
  then advance.

The loop runs while `currentGroupCopies <= num_buffers`.

**Key rules:**
- `--smem-circular-reuse` gates all pairing/reuse logic. When off,
  only single-WSBuffer increases are tried.
- When `smem-circular-reuse` is on and there are **exactly 2** candidates
  at a priority level, they are tentatively grouped into a reuse group
  before the loop begins.
- A pair is chosen (i.e., remains as a reuse group) only when there are
  **exactly 2** candidates **and** the final copy count is **odd**.
  If the final copy count is even, the group is split back in Step 2
  (each buffer gets `numCopies/2` with its own `buffer.id`).
- Once grouped, the loop increments the group's copies as a single unit
  (no iteration over pending).
- The loop terminates when budget is exhausted or
  `currentGroupCopies > num_buffers`.

### Phase 4: Total SMEM Computation

```
totalSmem = 0
for each unique buffer.id:
    groupSize = max(sizeBytes of WSBuffers sharing this buffer.id)
    copies    = buffer.copy for this group
    totalSmem += groupSize * copies
```

### Phase 5: Emit Attributes

Write `buffer.id` and `buffer.copy` attributes onto each `local_alloc` op.
For WSBuffers in a reuse group, both ops get the same `buffer.id`.

---

## BWD Test Case Walkthrough

### Setup

```
num_buffers = 2   (from --num-buffers=2 on the RUN line)
smemBudget  = 232448 bytes  (227 KB, Blackwell sm_100)
```

### SMEM WSBuffers

| # | Name   | Size   | Innermost | TMA | Cross-Stage | Why cross-stage? |
|---|--------|--------|-----------|-----|-------------|------------------|
| 0 | `dsT`  | 32 KB  | Yes | No  | No  | Producer (stage 1) → consumers (stage 1) |
| 1 | `do`   | 32 KB  | Yes | Yes | Yes | Producer (stage 0) → consumers at stage 0 and stage 1 |
| 2 | `q`    | 32 KB  | Yes | Yes | Yes | Producer (stage 0) → consumers at stage 0 and stage 1 |
| 3 | `k_42` | 32 KB  | No  | —   | —   | Outside loop |
| 4 | `v_43` | 32 KB  | No  | —   | —   | Outside loop |

### Phase 1 — Initialize

All WSBuffers get unique IDs, all `numCopies = 1`.

```
Total SMEM = 5 × 32 KB = 160 KB
```

### Phase 2 — Cross-Stage Minimum

`do` and `q` are cross-stage → set `numCopies = 2`.

```
Total SMEM = 32(dsT) + 64(do) + 64(q) + 32(k) + 32(v) = 224 KB ≤ 227 KB ✓
```

### Phase 3 — Classification

| Priority | WSBuffers |
|----------|-----------|
| P0 (innermost + TMA) | `do`, `q` |
| P1 (innermost, non-TMA) | `dsT` |
| P2 (not innermost) | `k_42`, `v_43` |

### Phase 4 — Iterative Increase

**P0: `do`, `q`**  (`smem-circular-reuse = false`)

No grouping. Each WSBuffer is independent.
Both at `numCopies = 2` from Phase 2. `currentGroupCopies = 1`.

- Level 2: pending = none (both already at 2). Advance.
- Level 3: 3 > 2 → exit (num_buffers = 2). **Done.**

**P0: `do`, `q`**  (`smem-circular-reuse = true`)

|candidates|=2 → group `do`+`q` upfront. Both are cross-stage (need 2
individual copies), so group minimum = `2*2-1 = 3`. But `num_buffers = 2`,
so `3 > num_buffers` — the group's starting copies is clamped to
`num_buffers = 2`. `currentGroupCopies = 2`.

- Level 2: group not yet at 2.
  - Group tries `numCopies = 2`: cost = max(32,32) × 2 = 64 KB.
    total = 32(dsT) + **64**(do+q) + 32(k) + 32(v) = 160 KB ≤ 227 KB ✓.
  - Commit. Advance.
- Level 3: 3 > 2 → exit (num_buffers = 2). **Done.**

**P1: `dsT`**

1 WSBuffer at P1. `numCopies = 1`, `currentGroupCopies = 1`.

With `smem-circular-reuse = false` (do=2, q=2, separate):
- Level 2: total = 64(dsT) + 64(do) + 64(q) + 32(k) + 32(v) = 256 KB > 227 KB ✗.
  Cannot increase.

With `smem-circular-reuse = true` (do+q group at 2):
- Level 2: total = 64(dsT) + 64(do+q) + 32(k) + 32(v) = 192 KB ≤ 227 KB ✓.
  Commit.

**P2: `k_42`, `v_43`**

Not innermost. **Do not increase.**

### Final Result (`smem-circular-reuse = false`)

| WSBuffer | `buffer.id` | `buffer.copy` | Reuse Group |
|----------|-------------|---------------|-------------|
| `dsT`    | 0           | 1             | — |
| `do`     | 1           | 2             | — |
| `q`      | 2           | 2             | — |
| `k_42`   | 3           | 1             | — |
| `v_43`   | 4           | 1             | — |

```
Total SMEM = 32 + 64 + 64 + 32 + 32 = 224 KB
```

### Final Result (`smem-circular-reuse = true`)

| WSBuffer | `buffer.id` | `buffer.copy` | Reuse Group |
|----------|-------------|---------------|-------------|
| `dsT`    | 0           | 2             | — |
| `do`     | 1           | 2             | `do` + `q` |
| `q`      | 1           | 2             | `do` + `q` |
| `k_42`   | 2           | 1             | — |
| `v_43`   | 3           | 1             | — |

```
Total SMEM = 64 + 64 + 32 + 32 = 192 KB
```

Grouping `do`+`q` saves 64 KB (from 224 KB to 160 KB for those two),
freeing budget for `dsT` to increase to 2 copies.

---

## Pairing Logic — Detailed Examples

### Example 1: 2 candidates, both at copies=1, `smem-circular-reuse=true`

```
P0 candidates: [A(copies=1), B(copies=1)]
  → |candidates| = 2, smem-circular-reuse → group upfront
  → group.numCopies = 1
  → Loop: level 2 → group tries 2, budget check ✓ → copies = 2
  → Loop: level 3 → group tries 3, budget check ✓ → copies = 3
  → Physical = max(sizeA, sizeB) × 3
```

### Example 2: 2 candidates, `smem-circular-reuse=false`

```
P0 candidates: [A(copies=1), B(copies=1)]
  → No grouping. Each keeps its own buffer.id.
  → Loop: level 2 → A tries 2, budget ✓ → A.copies = 2
  →                  B tries 2, budget ✓ → B.copies = 2
  → Loop: level 3 → A tries 3, budget ✓ → A.copies = 3
  →                  B tries 3, budget ✗ → B stays at 2
  → Physical = sizeA × 3 + sizeB × 2
```

### Example 3: 3 candidates, `smem-circular-reuse=true`

```
P0 candidates: [A(copies=1), B(copies=1), C(copies=1)]
  → |candidates| = 3, not exactly 2 → no grouping
  → Each keeps its own buffer.id.
  → Loop processes each individually at each level.
```

### Example 4: Different starting copies (FWD case), `smem-circular-reuse=true`

```
v(copies=2 from cross-stage), k(copies=1)
  → |candidates| = 2, smem-circular-reuse → group upfront
  → v is cross-stage (needs 2), so group starts at 2*2-1 = 3
  → Loop: level 3 → group tries 3 → 96 KB, budget ✓ → copies = 3
  → Result: both v and k share 3 pipeline slots
  → v retains ≥2 effective slots, k gets ≥1
```

### Example 5: Different starting copies, `smem-circular-reuse=false`

```
v(copies=2 from cross-stage), k(copies=1)
  → No grouping.
  → Loop: level 2 → k tries 2 → 64 KB extra, budget ✗ → k stays at 1
  → v stays at 2, k stays at 1
  → Grouping would have unlocked copies=3 for both within budget
```

---

## FWD Test Case Walkthrough

### Setup

```
num_buffers = 2   (hypothetical; the existing test uses num-buffers=3)
smemBudget  = 232448 bytes  (227 KB, Blackwell sm_100)
```

### SMEM WSBuffers

The Flash Attention forward pass (`_attn_fwd_persist`) has 6 SMEM allocations.
There is an **outer** `scf.for` (persistent tile loop, line 162) and an
**inner** `scf.for` (KV loop, line 184, `tt.scheduled_max_stage = 1`).

| # | Name    | Size  | In inner loop? | TMA? | Cross-Stage? | Notes |
|---|---------|-------|----------------|------|-------------|-------|
| 0 | `%0`    | 32 KB | No | — | — | Alloc outside all loops |
| 1 | `%1`    | 32 KB | No | — | — | Alloc outside all loops |
| 2 | `v`     | 32 KB | Yes (innermost) | Yes | **Yes** | Producer stage 0; consumers at stage 0 (MMA line 286) and stage 1 (MMA line 287) |
| 3 | `k`     | 32 KB | Yes (innermost) | Yes | **No** | Producer stage 0; all consumers at stage 0 (lines 187, 190–191) |
| 4 | `q0`    | 32 KB | No | — | — | Alloc in outer loop, used in inner loop but produced before inner loop |
| 5 | `q0_18` | 32 KB | No | — | — | Same as `q0` |

### Phase 1 — Initialize

All 6 WSBuffers get unique IDs 0–5, all `numCopies = 1`.

```
Total SMEM = 6 × 32 KB = 192 KB
```

### Phase 2 — Cross-Stage Minimum

Only `v` is cross-stage → set `v.numCopies = 2`.

```
Total SMEM = 32×1(%0) + 32×1(%1) + 32×2(v) + 32×1(k) + 32×1(q0) + 32×1(q0_18)
           = 32 + 32 + 64 + 32 + 32 + 32 = 224 KB ≤ 227 KB ✓
```

### Phase 3 — Classification

| Priority | WSBuffers |
|----------|-----------|
| P0 (innermost + TMA) | `v`, `k` |
| P1 (innermost, non-TMA) | — |
| P2 (not innermost) | `%0`, `%1`, `q0`, `q0_18` |

### Phase 4 — Iterative Increase

**P0: `v`, `k`**  (`smem-circular-reuse = false`)

No grouping. Each WSBuffer is independent.

`v` is at `numCopies = 2` (cross-stage minimum), `k` at `numCopies = 1`.
`currentGroupCopies = 1`.

- Level 2: pending = [`k`] (only `k` is below 2, `v` already at 2).
  - Single: `k` tries `numCopies = 2`:
    total = 32 + 32 + 64 + **64** + 32 + 32 = 256 KB > 227 KB ✗.
  - Cannot increase. Budget exhausted. **Done.**

**P0: `v`, `k`**  (`smem-circular-reuse = true`)

|candidates|=2 → group `v`+`k` upfront. `v` is cross-stage (needs 2
individual copies), so group starts at `2*2-1 = 3` copies.
`currentGroupCopies = 3`.

- Level 3: group not yet at 3.
  - Group tries `numCopies = 3`: cost = max(32,32) × 3 = 96 KB.
    total = 32 + 32 + **96** + 32 + 32 = 224 KB ≤ 227 KB ✓. Commit.
  - Advance.
- Level 4: 4 > 3 → exit (num_buffers = 3). **Done.**

**P1: (empty)** Skip.

**P2: `%0`, `%1`, `q0`, `q0_18`**

Not innermost. **Do not increase.**

### Final Result (`smem-circular-reuse = false`)

| WSBuffer | `buffer.id` | `buffer.copy` | Reuse Group |
|----------|-------------|---------------|-------------|
| `%0`     | 0           | 1             | — |
| `%1`     | 1           | 1             | — |
| `v`      | 2           | 2             | — |
| `k`      | 3           | 1             | — |
| `q0`     | 4           | 1             | — |
| `q0_18`  | 5           | 1             | — |

```
Total SMEM = 32 + 32 + 64 + 32 + 32 + 32 = 224 KB
```

### Final Result (`smem-circular-reuse = true`)

| WSBuffer | `buffer.id` | `buffer.copy` | Reuse Group |
|----------|-------------|---------------|-------------|
| `%0`     | 0           | 1             | — |
| `%1`     | 1           | 1             | — |
| `v`      | 2           | 3             | `v` + `k` |
| `k`      | 2           | 3             | `v` + `k` |
| `q0`     | 3           | 1             | — |
| `q0_18`  | 4           | 1             | — |

```
Total SMEM = 32 + 32 + 96 + 32 + 32 = 224 KB
```

> **Note:** The current algorithm assigns `copy = 3` to both `v` and `k`
> without reuse (total = 320 KB — exceeding budget). The new algorithm with
> `smem-circular-reuse = true` achieves the same `copy = 3` for both within
> budget via a reuse group. With reuse off, `v` stays at 2 and `k` at 1.

---

## Key Design Decisions

### 1. SMEM Budget Parameter

The hardware SMEM capacity must be known. Options:

- Derive from `ttg.target` attribute (e.g., `"cuda:100"` → 227 KB).
- Add a pass option `--smem-budget=<bytes>` for testing.
- Use a conservative default to leave room for barriers/scratch.

### 2. `num_buffers` Source

Passed as the `--num-buffers` parameter to the pass (same as today).
This is the maximum number of copies any WSBuffer can have.

### 3. Deterministic Iteration Order

Sort WSBuffers by their operation ID (from `buildOperationIdMap`) before
processing, ensuring reproducible results.

### 4. Reuse Group Constraints

Two WSBuffers can form a reuse group only if:
1. `--smem-circular-reuse` is on.
2. They are at the **same priority level**.
3. They have the same element type.

Liveness overlap and dependency ordering do not need to be checked —
the reuse group shares a circular buffer, and the circular indexing
handles producer-consumer separation.

The reuse decision is recorded by assigning the same `buffer.id` to
both WSBuffers. No additional pointer or data structure is needed —
downstream passes already group allocs by `buffer.id`.

### 5. Interaction with TMEM Planner

The SMEM planner runs first (Step 2 of `doMemoryPlanner`) and returns
`lastBufferId`. The TMEM planner (Step 4) starts numbering from there.
This interface is unchanged.

---

## Summary of Changes

| Component | Current | Proposed |
|-----------|---------|----------|
| Abstraction | Raw `BufferT` + `DenseMap` | `WSBuffer` struct per `local_alloc` |
| Initial state | Single pass, group by type | Phase 1: unique IDs, all `copy = 1` |
| Cross-stage | Not considered | Phase 2: force `copy ≥ 2` |
| Multi-buffering | Unconditional for TMA+innermost | Phase 4: iterative, budget-aware |
| Reuse | Not done | Pair of 2 same-priority WSBuffers; grouping-first when copies ≥ 2 |
| Max copies | `numBuffers` param (all-or-nothing) | `num_buffers` param (incremental cap) |
| Budget | Not checked | Enforced at every iteration |
| Iteration order | Non-deterministic | Sorted by operation ID |

---

## Pipeline Context

```
doMemoryPlanner(funcOp, numBuffers)
  ├── Step 0: reorderOpsBySchedule (disabled)
  ├── Step 1: collectPostChannels
  ├── Step 1.5: identify cross-stage channels
  ├── Step 2: MemoryPlanner::run(numBuffers)       ← THIS CHANGES
  │     ├── Phase 1: create WSBuffers, unique IDs, all copy=1
  │     ├── Phase 2: enforce cross-stage minimum (copy ≥ 2)
  │     ├── Phase 3: classify P0–P2
  │     ├── Phase 4: iterative copy increase within SMEM budget
  │     │     ├── per priority level, pair or single selection
  │     │     └── reuse group creation when paired
  │     └── Phase 5: emit buffer.id / buffer.copy attributes
  ├── Step 3: MemoryPlannerTmem::collectTMemAllocsAndLiveness
  └── Step 4: MemoryPlannerTmem::allocateBuffers(lastBufferId)
```

## Files to Modify

| File | Change |
|------|--------|
| `WSMemoryPlanner.cpp` — `MemoryPlanner` class | Add `WSBuffer` struct, rewrite `run()` with 5-phase algorithm |
| `WSMemoryPlanner.cpp` — `doMemoryPlanner` | Pass cross-stage info from Step 1.5 into the planner |
| `Passes.td` | Add `--smem-budget` and `--smem-circular-reuse` options |
| `ws_memory_planner_bwd.mlir` | Update CHECK lines for new assignments |
| `ws_memory_planner_fwd.mlir` | Update CHECK lines similarly |
