# Warp Specializing `scf.while` Loops

**Files**: `PartitionSchedulingMeta.cpp` (+ a small loop-body abstraction in
`Utility.h`/`Utility.cpp`)

## Motivation

The unified tile scheduler (`triton.language.schedule`) drives a persistent
kernel with an `scf.while` outer loop instead of a hand-written `scf.for`:

```python
sched = SCHEDULE.initialize(...)
while sched.is_valid():          # <- scf.while
    ... one output tile ...
    sched = sched.advance()
```

We want the **outermost (persistent) loop** to be warp-specialized, exactly like
the hand-written `for tile_id in tl.range(..., warp_specialize=True)` persistent
GEMM (`matmul_kernel_tma_persistent_ws`). The frontend now attaches the AutoWS
annotations to the while via `tl.condition(..., warp_specialize=True)` (see
`AutoWSLoopOptions` in `language/core.py`), and those annotations survive into
TTGIR.

Two of the four schedules are already covered without touching AutoWS:

- **Static persistent** — the while is *countable*, so `triton-uplift-while-to-for`
  rewrites it to an `scf.for` (carrying the annotations) before AutoWS runs. It is
  then warp-specialized by the existing `scf.for` path.
- **Non-persistent** — the while is kept through data partitioning and initial
  loop scheduling, then `triton-simplify-single-trip-while` forwards AutoWS to
  the scheduled inner loop and eliminates the single-trip outer loop before
  partition scheduling.

That leaves the **genuinely non-countable** schedules — **dynamic** (atomic
work-stealing) and **CLC** (hardware `_valid`) — whose outer loop stays an
`scf.while`. Today they are *not* warp-specialized. This doc describes making
`PartitionSchedulingMeta` (the first AutoWS pass) schedule an `scf.while`.

## Implemented boundary (and why SWP is *not* the blocker)

`PartitionSchedulingMeta` derives the partition schedule **structurally**
(op categorization + MMA backward slices — see `PartitionSchedulingMeta.md`); it
does **not** consume the software-pipeliner's schedule. `getInitialSchedule`
only reads a pre-serialized schedule (`ttg.partition.stages` /
`ttg.warp_specialize.tag`) for *modulo*-scheduled loops; otherwise it computes
its own. So warp-specializing the `while` does **not** require software
pipelining it.

The original blocker was that the pass was typed on `scf::ForOp` and assumed
the single-region for-loop shape:

- Entry walk: `getOperation().walk([&](scf::ForOp loop){ if hasAttr(tt.warp_specialize) })`
  (`runOnOperation`, ~L2623) never sees an `scf.while`.
- The whole scheduler (`getInitialSchedule`, `OpCategorizer`,
  `collectMMABackwardSlice`, `iterateDefs`/`iterateUsers`, `scheduleUsers`,
  `findDefOpInLoop`, `hasDefPartition`) takes `scf::ForOp` and reaches the loop
  body via `loop.getBody()`, `loop.getRegionIterArg(n)`, `loop.getInductionVar()`,
  `loop.getYieldedValues()`, `loop.getOps<scf::ForOp>()`.

The partition and PSM APIs now use `LoopLikeOpInterface` for the supported
`scf.for` and ordered-subset-carry `scf.while` forms. Everything downstream of the
partition schedule already handles `scf.while`:
task-id propagation, data partition, and code partition each have `scf.while`
lit tests (`ws_while_loop_autows.mlir`). So the change is concentrated in this
one pass.

## The for/while structural difference

| Concept | `scf.for` | `scf.while` |
|---|---|---|
| body block | `getBody()` | `getAfterBody()` (after-region entry block) |
| body region | `getBodyRegion()` | `getAfter()` |
| body terminator | `scf.yield` in body | `scf.yield` in **after** region |
| loop-carried body args | body args `1..N` (arg 0 = induction var) | **all** after-region args `0..N-1` |
| induction var | `getInductionVar()` | none |
| yielded values | `getYieldedValues()` | after-region `scf.yield` operands |
| iter-arg for yield operand `i` | `getRegionIterArg(i)` | after-arg `i` (see caveat) |
| results | loop results | loop results (before-region `scf.condition` forwarded args) |

**Caveat — condition forwarding.** In an `scf.while`, loop-carried values flow
`inits → before-args → scf.condition (forwards a subset, possibly reordered) →
after-args → body → scf.yield → before-args`. The CLC scheduler carries
`(valid, x)` but forwards only `x`, so after-argument index 0 maps back to yield
slot 1. The scheduler resolves both directions through the condition operands
rather than assuming matching indices. It accepts a direct, unique, non-empty,
order-preserving subset of before arguments; condition-only slots simply have
no scheduled-body argument. Reordered, duplicate, or computed forwarding still
falls back to an unspecialized loop.

## Design: a loop-body abstraction

Introduce small free helpers (in `Utility.{h,cpp}`) that dispatch on the loop
type, and change the pass to pass loops as `LoopLikeOpInterface` (implemented by
both `scf::ForOp` and `scf::WhileOp`) instead of `scf::ForOp`:

```cpp
// Utility.h
Block   *getLoopBodyBlock(LoopLikeOpInterface loop);   // for: getBody(); while: getAfterBody()
Region  &getLoopBodyRegion(LoopLikeOpInterface loop);
Operation *getLoopBodyTerminator(LoopLikeOpInterface loop);
ValueRange getLoopYieldedValues(LoopLikeOpInterface loop);
Value    getLoopInductionVar(LoopLikeOpInterface loop); // for: iv; while: {} (null)
// Maps a body-terminator operand index to the body block arg that carries it
// into the next iteration, or null when the slot is condition-only.
BlockArgument getLoopCarriedBodyArg(LoopLikeOpInterface loop, unsigned yieldOperandIdx);
// Maps a scheduled-body argument back to its corresponding yielded value.
Value getLoopCarriedYieldedValue(LoopLikeOpInterface loop, BlockArgument bodyArg);
// True when the while's scf.condition forwards a supported ordered subset
// (always true for scf.for). Loops failing this are skipped.
bool hasSupportedLoopCarry(LoopLikeOpInterface loop);
```

The one index subtlety, already present in the code: `findDefOpInLoop`
(`~L940`) does `getYieldedValues()[arg.getArgNumber() - 1]` — the `-1` is the
for-loop induction-var offset. The paired carry helpers hide this: for
`scf.for`, body-arg `i+1` maps to yield operand `i`; for `scf.while`, an after
argument maps through its `scf.condition` operand to the corresponding before
argument and yield slot.

Nested-loop collection `mainLoop.getOps<scf::ForOp>()` (`~L334`, `~L1282`)
becomes "collect `scf::ForOp` in the body block" via
`getLoopBodyBlock(loop)->getOps<scf::ForOp>()`.

Everything else in the categorizer/scheduler operates on ops *inside* the body
(via `loop.walk`, def-use edges, partition attrs) and is loop-type-agnostic once
the body block and iter-arg mapping are abstracted.

### Entry generalization

`runOnOperation` collects **both** loop types carrying `tt.warp_specialize`:

```cpp
getOperation().walk([&](LoopLikeOpInterface loop) {
  if (!loop->hasAttr(kWarpSpecializeAttrName)) return;
  if (!hasSupportedLoopCarry(loop)) return; // safe skip
  loops.push_back(loop);
});
```

For the persistent tile-scheduler kernels, `mainLoop` is the outer loop and the
MMAs live in the nested inner `scf.for` (the K-loop), which the categorizer
already discovers through nested-for collection — so the inner loop is found the
same way for a `while` outer loop as for a `for` outer loop.

## Scope of change

- **loop-body helpers** (~60 lines): the dispatch helpers above.
- **`PartitionSchedulingMeta.cpp`**: change ~38 `scf::ForOp` occurrences to
  `LoopLikeOpInterface` and route the ~20 body/iter-arg accesses through the
  helpers. No change to the categorization/partition-assignment *logic*.
- **External `scf::ForOp`-typed helpers** the pass calls, which must be
  generalized (all small — they use the loop mostly as an `Operation*`):
  - `PartitionSet::fromLoop` / `PartitionSet::serialize` (`Partition.h/.cpp`) —
    only `getAttr`/`setAttr`/`walk`/`getLoc` on the loop → widen to
    `LoopLikeOpInterface` (or `Operation*`). Callers pass `scf::ForOp`, which
    converts implicitly, so no caller churn.
  - `getDefinitionAndDistance` (`PipeliningUtility.cpp`) uses `getBody()` +
    `getYieldedValues()[argNumber-1]` + induction-var check. Rather than change
    the shared signature (it has other callers), provide a **local**
    while-aware variant in `PartitionSchedulingMeta.cpp` that uses the loop-body
    helpers (`arg 0` is not special for a `while`, so no `-1`).
- **No pipeline change strictly required**: because the scheduler derives its own
  partitions, dynamic/CLC only need `partition_scheduling_meta` +
  `hopper_warpspec` (both already in the `use_meta_ws` path). We do **not** run
  the software pipeliner on the `while` (nor need to).

### `PartitionSet::iterate*` are meta-pass-internal, and their `distance` is SWP-only (and unused)

`PartitionSet`'s `scf::ForOp` dataflow methods `iterateInputs`/`iterateOutputs`/
`iterateDefs` (`Partition.h:49-65`) are **only** called from
`PartitionSchedulingMeta.cpp` (`iterateDefs` at `:2058`, `:2251`) — there are no
downstream callers (verified repo-wide). So they are part of *this pass's*
generalization, not a separate surface.

Crucially, the one genuinely software-pipelining concept they carry — the
cross-iteration `distance` — is **ignored** by both call sites (`:2051`, `:2246`
use only `result.getDefiningOp()`). The scheduler only needs *which op* defines a
loop-carried input, to route cluster def/sink partitions in
`propagatePartitions`. So the `scf.while` variant just follows loop-carried
inputs through the before/after regions to the defining op; no staging/distance
semantics are required. Generalizing these is mechanical (body block +
loop-carried arg mapping via the helpers). `fromLoop`/`serialize` are
attribute-only and widen trivially. `swapPartitions` is used by the SWP-oriented
upstream scheduler, not the meta path.

## Downstream (post-partition) validation surface

`PartitionSchedulingMeta` only *assigns* partitions (`ttg.partition` +
`ttg.partition.stages`). Emitting real `ttg.warp_specialize` for a `while` GEMM
then exercises `hopper_warpspec` (region split), task-id propagation,
`WSDataPartition`, `WSCodePartition` (channels/barriers), and `WSMemoryPlanner`.
None of these use `PartitionSet::iterate*`; they have their own loop handling
with `scf.while` lit coverage for *simple* bodies (`ws_while_loop_autows.mlir`).
A full GEMM tile body (TMA loads + MMA + epilogue store) is now validated
through code partitioning and physical specialization in
`ws_atomic_broadcast_from_psm.mlir`, and through the unified dynamic scheduler
on Blackwell. That validation exposed one downstream `scf::ForOp` assumption:
post-WS loop-schedule preprocessing only recognized an outer `scf.for`, so the
nested K loop inside a specialized `scf.while` kept a partial schedule and the
software pipeliner bailed. Preprocessing now recognizes both outer loop forms
while continuing to pipeline only the nested K `scf.for`.

## Risks / invariants

- **For-loop path must be byte-for-byte unchanged.** The helpers return exactly
  the current values for `scf::ForOp`; the for-loop regression suite
  (autows-testing) is the guard.
- **Condition-forwarding guard** accepts only direct ordered subsets and
  prevents mis-scheduling empty, reordered, duplicate, or computed forwarding.
- Downstream buffer sizing uses `ttg.partition.stages`; the persistent `while`
  is depth-1 at the outer level (no outer pipelining), with the inner K-loop
  providing the producer/consumer overlap — same as the for-loop persistent
  kernel.

## Test plan

1. **LIT** (`triton-opt --nvgpu-partition-scheduling-meta`): a minimal
   warp-specialized `scf.while` GEMM skeleton → check `ttg.partition` /
   `ttg.partition.stages` are assigned on the while and its body ops.
2. **For-loop regression**: full autows-testing suite stays green (no diff on the
   `scf.for` path).
3. **End-to-end**: tutorial09 `dynamic`/`clc` persistent-while kernels compile to
   `ttg.warp_specialize` and match the reference matmul (GB200).

## Status

- [x] Frontend annotation on `while` (`AutoWSLoopOptions` + `tl.condition`) — done.
- [x] `uplift-while-to-for` attribute transfer (static path) — done.
- [x] Loop-body abstraction + `PartitionSchedulingMeta` generalization — done.
- [x] Atomic broadcast validated in isolation from a PSM-assigned outer while
  (`ws_atomic_broadcast_from_psm.mlir`, via the new `nvgpu-test-ws-atomic-broadcast`
  test pass). Task-id propagation already supplies the full-union `async_task_id`
  the broadcast needs — no PSM change was required. See
  `docs/DynamicPersistentAutoWSGaps.md`.
- [x] Code partition, physical specialization, accumulation-counter rotation,
  and nested K-loop rescheduling validated from a PSM-assigned outer while.
- [x] Unified dynamic atomic scheduler correctness validated on Blackwell.
- [x] Unified CLC scheduler correctness validated on Blackwell with an
  unannotated inner K loop.
- [ ] Hopper runtime validation of the unified dynamic atomic scheduler.
