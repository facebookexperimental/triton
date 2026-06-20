# Proposal: Physical-Aliasing Barrier-Coverage Verification for AutoWS

**Status:** Proposal / RFC (not yet implemented)
**Scope:** AutoWS code partitioning (`WSCodePartition.cpp`), WS barrier analysis
(`WSBarrierAnalysis.h`), and the `barrier-visualization` agent skill.
**Author:** drafted with Claude, following the FA-fwd-persistent TMEM aliasing
race investigation.

## TL;DR

A latent synchronization bug in `_attn_fwd_persist` (Flash-Attention forward,
persistent, dp=2) shipped undetected by every existing barrier check, static and
agent-driven, because they all reason **per-token / per-`buffer.id`** and never
model the fact that one physical allocation can be **column-packed** with several
independently-synchronized buffers. This proposes a small, well-defined
**physical-aliasing coverage invariant**, an **executable verifier** that checks
it (the source of truth), and a thin **skill driver** that runs the verifier and
reports violations. The goal: turn "describe the barriers that exist" into
"prove every overwrite is ordered after every aliased read."

## Background: the bug this exposes

The memory planner packs the softmax scalars `m_ij` / `l_i0` / `alpha` into the
spare columns (offsets 64/65/66) of the 128×128 QK accumulator. They share one
physical `buffer.id` via distinct `buffer.offset` column packing, but each is its
own single-buffered channel with its **own** token.

The QK MMA is `tc_gen5_mma` with `useC=false`, which **zeros the entire
allocation** (all 128 columns) before writing the QK result. So the next tile's
first QK MMA clobbers columns 64-66 — while the default partition is still
reading `m_ij` / `l_i0` from them in the previous tile's epilogue. The required
backward edge (default's read → next MMA's write) did not exist. Result:
non-deterministic dp0 corruption — latent without `early_tma_store`, exposed with
it.

The fix (committed separately) adds, for a full-overwrite reuse-group owner, a
backward `producer_acquire` on each **outer-cadence** packed sibling, placed
**before the inner loop** using the **sibling's own outer-loop phase**. See
[ReuseGroups.md](ReuseGroups.md) → "Full-overwrite owner (hub case)".

## Why nothing caught it

Both the compiler's static analysis and the agent skill share the same blind
spot. The defect is a **missing edge across a physical-aliasing relationship that
the data model does not represent.**

1. **Local balance, not global coverage.** Every existing check is a per-barrier
   invariant (matched arrive/wait counts, phase tracking, merged byte counts).
   Each of the four tokens here is individually balanced. The defect is the
   *absence* of a cross-barrier edge, which no local check covers.

2. **`buffer.id` is the unit, and it conflates "merged" with "packed."**
   - *Merged*: one barrier (one `barrier_expect`) covers several buffers.
   - *Packed*: same `buffer.id`, distinct `buffer.offset`, **N independent
     barriers** over disjoint column ranges.
   Analyses treat same-`buffer.id` as "merged/safe" and never read
   `buffer.offset`, so packed groups look covered when they are not.

3. **No write-extent model.** The race hinges on `useC=false` ⇒ full-allocation
   write. Nothing models *which columns an op writes*, so the QK MMA is seen as
   "producer of the QK channel" only — its overwrite of the scalar columns is
   invisible.

4. **The backward-barrier heuristic is a fixed catalog.** It matches specific
   shapes (operand-D `tmem_load → tmem_store` chains, multi-buffer phase). The
   missing edge here (MMA ← scalar-consumer) matches none, so it is not derived
   from the sharing structure.

5. **No cadence/loop-level reasoning.** The MMA is inner-cadence; the scalars are
   outer-cadence. That determines *where* the edge must live (before the inner
   loop) and is why a naive fix deadlocks. No analysis relates barriers across
   nesting levels.

6. **Descriptive, not verificational.** Reports document barriers that *exist*;
   nothing enumerates barriers that *should* exist, so absence is never surfaced.

The compiler's own `WSBarrierAnalysis::buildChannelGraph` has the same gap — it
is partition-level task reachability with no buffer/offset information, used only
for reorder legality. `WSBarrierOrderedRegionTracking.md` explicitly disclaims
it: *"aliasing information is not modeled yet."*

## The invariant

> For each physical allocation `R` and each op `W` that writes a column range of
> `R`, `W` must be ordered — via a barrier reachable in `W`'s partition before
> `W`, at the correct loop cadence — after the previous-iteration read of **every
> logical buffer** whose `[buffer.offset, buffer.offset + width)` range
> intersects `W`'s write range (when that buffer's consumer is in a different
> partition than `W`).

## Proposal

### 1. Executable verifier (source of truth) — compiler, not a skill

Implement the check in the compiler, where the needed data already exists:

- **Physical layout**: group allocs by `buffer.id`; record each member's
  `[offset, width)` column interval.
- **Write extent per writer**: `tc_gen5_mma useC=false` and `tmem_store dense<0>`
  ⇒ full allocation; `useC=true` ⇒ matmul N-range; subsliced store/MMA ⇒ the
  subslice.
- **Cadence**: enclosing loop of each writer and each aliased reader.
- **Coverage**: for each (writer, aliased-buffer) intersecting pair with a
  cross-partition consumer, require a backward edge reachable before the writer
  at the reader's cadence; otherwise emit a violation.

Delivery path (incremental):

1. **(done)** narrow debug assert in `doCodePartitionPost` for the
   full-overwrite-owner case.
2. generalize that assert into the full coverage invariant (any write extent, all
   `buffer.id` groups, cadence-aware), behind `assert` / `LLVM_DEBUG`.
3. optionally expose a standalone `--nvgpu-verify-ws-aliasing` pass for CI, and
   lift `buffer.id` / `buffer.offset` into `WSBarrierAnalysis` so the channel
   graph can answer aliasing queries.

A skill cannot *be* this checker — it is agent instructions, not a deterministic
pass. Re-deriving layout/extent/cadence by parsing IR text in a script is fragile
and duplicative; the compiler already has the structures.

### 2. Coverage table (enumerate absences) — report format

Whether produced by the verifier or by the skill, the report must enumerate
absences, not just presences:

```
Physical buffer.id = 8 (owner: QK accumulator, 128 cols)
  Writer: tc_gen5_mma useC=false  (task 1, inner loop)  write-extent: cols 0-127
  Aliased buffers overwritten:
    cols 0-63  QK result   consumer task 5 (gemm-internal)   ✓ ordered (QK backward)
    col  64    alpha       consumer task 0 (inner cadence)   ✓ own per-iter barrier
    col  65    m_ij        consumer task 0 (outer cadence)   ✗ MISSING backward edge
    col  66    l_i0        consumer task 0 (outer cadence)   ✗ MISSING backward edge
```

Filling the table *forces* the missing cell to surface.

### 3. Skill driver (reuse, do not add) — `barrier-visualization`

Add a step to the existing skill: classify same-`buffer.id` sharing
(merged / packed / time-shared), run the verifier (or build the coverage table
manually when no verifier is available), and map any violation back to source.
Do **not** create a new top-level skill — it would overlap `barrier-visualization`
and add discovery/maintenance cost. Add a known-failure self-test: run the skill
against the pre-fix `ws_code_partition_tmem_packed_reuse_backward.mlir` and
confirm it flags the race; run against post-fix and confirm clean.

## Alternatives considered

- **Planner-side avoidance**: refuse to column-pack a buffer into a full-overwrite
  owner whose liveness crosses the overwrite cadence. Simpler and robust, but
  sacrifices the TMEM-packing optimization (TMEM is scarce). Rejected as the
  primary fix; viable as a fallback for cases the barrier fix cannot cover.
- **Per-shape heuristics in the skill only**: cheap but pattern-specific; the next
  novel aliasing shape escapes again. Insufficient on its own.

## Risks / open questions

- Computing write-extent for every writer requires op-specific knowledge
  (`useAccumulator`, subslice chains); start with the high-value ops (gen5 MMA,
  zeroing `tmem_store`) and widen.
- Cadence detection must handle nested loops and `scf.if` (the ordered-region
  tracking already punts on `scf.if`; the verifier should be conservative there).
- The verifier should run after `insertAsyncComm` and before specialization, where
  channels and barriers are explicit but the loop structure is still intact.

## References

- [ReuseGroups.md](ReuseGroups.md) — reuse-group sync, full-overwrite-owner case
- [BarrierConstraints.md](BarrierConstraints.md),
  [WSBarrierOrderedRegionTracking.md](WSBarrierOrderedRegionTracking.md) —
  current (partition-level, aliasing-unaware) barrier analysis
- `test/Hopper/WarpSpecialization/ws_code_partition_tmem_packed_reuse_backward.mlir`
  — regression IR (fails pre-fix)
- `.claude/skills/barrier-visualization/SKILL.md` — the agent skill driver
