# AutoWS Loop Annotations

**Frontend**: `python/triton/language/core.py` (`AutoWSLoopOptions`, `range`,
`condition`), `python/triton/compiler/code_generator.py` (`_apply_loop_options`)

**Consumers**: the pipeliner (`AssignLatencies`, `ScheduleLoops`,
`SoftwarePipeliner`) and AutoWS (`PartitionSchedulingMeta`, `WSDataPartition`,
`LoopUnroll`, `FuseNestedLoops`).

This doc describes the loop annotations that drive automatic warp specialization
and pipelining: where they come from, what IR attribute each becomes, which pass
consumes it, and — importantly — which of them currently work on an `scf.while`
loop vs. only on `scf.for`.

## Where the annotations come from

The annotations are keyword arguments on the two loop front-ends:

```python
for i in tl.range(0, N, warp_specialize=True, num_stages=3, data_partition_factor=2):
    ...
while tl.condition(sched.is_valid(), warp_specialize=True, data_partition_factor=2):
    ...
```

Both `tl.range` (for loops) and `tl.condition` (while loops) inherit a single
dataclass, **`AutoWSLoopOptions`** (`core.py`), which owns the frontend option
set, defaults, and IR-attribute mapping. `visit_For` and `visit_While` both emit
them via the shared `_apply_loop_options` helper, so a `for` and a `while`
receive byte-identical attributes.

The C++ inventory and scheduler-loop propagation policy are centralized in
`include/triton/Dialect/Triton/IR/DiscardableAttributes.h` and
`lib/Dialect/Triton/IR/DiscardableAttributes.cpp`. The registry contains every
attribute emitted by `AutoWSLoopOptions`, including attributes consumed before
the single-trip rewrite and those that must be forwarded to the inner compute
loop. A new frontend option must add one registry entry and choose its
propagation policy; transforms such as `SimplifySingleTripWhile` filter this
registry instead of maintaining private allowlists. Internal scheduling
metadata such as `tt.scheduled_max_stage` remains separate from the frontend
AutoWS inventory.

## Annotation → IR attribute → consumer

| kwarg | IR attribute | Consumed by | Notes |
|---|---|---|---|
| `num_stages` | `tt.num_stages` | `AssignLatencies`, `SoftwarePipeliner` | software-pipeline depth |
| `warp_specialize` | `tt.warp_specialize` | `ScheduleLoops` → `PartitionSchedulingMeta` | marks the loop for WS |
| `data_partition_factor` | `tt.data_partition_factor` | `PartitionSchedulingMeta`, `WSDataPartition` | # consumer slices along M/N |
| `loop_unroll_factor` | `tt.loop_unroll_factor` | `LoopUnroll` (TTIR) | IR-level unroll; see [below](#loop_unroll_factor-and-disable_licm) |
| `flatten` | `tt.flatten` | `FuseNestedLoops` | flatten a loop nest |
| `disallow_acc_multi_buffer` | `tt.disallow_acc_multi_buffer` | `AssignLatencies`, `WSMemoryPlanner` (SMEM floor), upstream WS | keep acc single-buffered; see [below](#disallow_acc_multi_buffer) |
| `multi_cta` | `tt.multi_cta` | multi-CTA reduction | cluster-level reduction |
| `merge_epilogue` / `merge_epilogue_to_computation` / `merge_correction` | `tt.merge_*` | `PartitionSchedulingMeta` (`SchedulingOptions`) | epilogue/correction routing |
| `separate_epilogue_store` | `tt.separate_epilogue_store` | `PartitionSchedulingMeta` | epilogue store gets own partition |
| `tmem_alloc_algo` / `smem_alloc_algo` / `smem_budget` / `smem_circular_reuse` | `tt.{tmem,smem}_alloc_algo`, `tt.smem_budget`, `tt.smem_circular_reuse` | `WSMemoryPlanner` | allocation heuristics |
| `list_schedule_pick` / `mem_plan_pick` | `tt.list_schedule_pick` / `tt.mem_plan_pick` | list scheduler / `WSMemoryPlanner` | ranked schedule and memory-plan selection |
| `disable_licm` | `llvm.loop_annotation` | LLVM LICM (not `triton-licm`) | suppress hoisting; see [below](#loop_unroll_factor-and-disable_licm) |

See `PartitionSchedulingMeta.md` for how the `merge_*` / `separate_epilogue_store`
knobs shape the partition layout.

## `disallow_acc_multi_buffer`

`tl.range(..., disallow_acc_multi_buffer=True)` (or the same kwarg on
`tl.condition`) adds the unit attribute `tt.disallow_acc_multi_buffer` to the
loop. The kwarg defaults to `False`, and `False` emits nothing: every consumer
tests only whether the attribute is present (`getDisallowAccMultiBuffer` in
`PipeliningUtility.cpp`). The attribute is also moved by two loop rewrites:

- `FuseNestedLoops` puts it on the fused loop if any loop in the nest had it.
- `NormalizeWSWhileLoops` moves it from an inner warp-specialized `scf.for` to
  the enclosing CLC `scf.while`.

Which loop carries the attribute matters, because consumers look in different
places:

| Path | Consumer (file, symbol) | Where it looks | Effect when present |
|---|---|---|---|
| Software pipeliner, any WS mode | `AssignLatencies.cpp`, `AssignMMALatencies::run` | The `scf.for` whose body directly contains the MMA | Matters only when the MMA's accumulator is read inside that same loop (`requiresAccMultiBuffering`). The MMA then does not get the extra latency stage. In a non-WS loop its users stay in the MMA's stage, so `lowerMMA` does not multi-buffer the TMEM accumulator across stages. In a WS loop the MMA's latency is 1 instead of 2. |
| Meta WS memory planner | `WSMemoryPlanner.cpp`, `getStaticSmemCopySafetyFloor` | Any ancestor of an MMAv5 consumer that carries both this attribute and `tt.data_partition_factor > 1` | Every TMA-fed SMEM operand of that MMA gets a copy floor equal to the configured `num_stages`. With `data_partition_factor` 1 or unset, the SMEM planner ignores the attribute (see [TMAStoreWaitPipeline.md, Phase 3.7](TMAStoreWaitPipeline.md#phase-37-epilogue-group-copy-increase) and [MemoryPlannerSearch.md](MemoryPlannerSearch.md)). |
| Upstream WS (`TRITON_USE_META_WS` unset) | `LoadMMASpecialization.cpp`; `InsertTmemAref.cpp` (NVWS) | The WS loop (`getOuterWSLoop` for the aref path) | The accumulator's TMEM allocation or aref gets 1 stage instead of 2. |

What it does **not** control under Meta WS: TMEM accumulator copies. The
heuristic TMEM planner (`MemoryPlannerTmem::run`, post-processing) raises
`buffer.copy` round-robin on every loop-carried accumulator until the
512-column budget, minus reserved scale columns, is used. That code never
reads `tt.disallow_acc_multi_buffer`. On the plan-search path
(`allocateTmemBuffersViaSearch`) TMEM copies are always 1. With
`TRITON_USE_META_WS=1`, setting the attribute therefore does not reduce TMEM
columns. Its Meta WS effects are the two rows above: MMA latency and the SMEM
operand floor.

Resources affected:

- **TMEM.** On the upstream-WS and non-WS software-pipeliner paths, it removes
  the second accumulator buffer: one accumulator tile's columns, e.g. BN
  columns for a BM=128 fp32 tile.
- **SMEM.** On Meta WS with `data_partition_factor > 1`, the operand rings are
  raised to `num_stages` copies before Phase 3.7 output staging and Phase 4
  discretionary copies, so less budget remains for those.
- **Registers and warps.** No effect.

Fallbacks:

- **Wrong loop.** In a non-flattened persistent GEMM the annotated
  `tl.range` is usually the outer tile loop, while the MMA sits in the inner K
  loop. `AssignLatencies` checks only the MMA's own loop, so it does not see an
  outer-loop attribute. The memory-planner floor walks ancestors and does see
  it.
- **No default scheduler.** When `use_modulo_schedule`, `use_llm_schedule`,
  `use_joint_schedule`, or `use_list_schedule` is set, `make_ttgir` skips
  `AssignLatencies`, and the latency effect disappears.
- **SMEM floor relaxed.** If the pinned operand floor makes the plan exceed
  `smemBudget`, `relaxCopySafetyFloorToBudget` gives the copies back one at a
  time, largest buffer first. It never goes below the pre-floor count. The
  only record is an `LDBG` line under `-debug-only=nvgpu-ws-memory-planner`.

## `loop_unroll_factor` and `disable_licm`

Neither knob is specific to warp specialization. Both are generic loop
transforms that act on the loop before or after AutoWS sees it, so this
section only covers how they interact with AutoWS.

### `loop_unroll_factor`

| Property | Behavior |
|---|---|
| How it is set | `tl.range(..., loop_unroll_factor=N)` or the same kwarg on `tl.condition`. |
| IR | `tt.loop_unroll_factor` (int32). Only emitted when the value is not `None`. |
| Legal values | Any int. Values `<= 1` mean "do not unroll". |
| Default | `None` (no unroll). |
| Consumer | `LoopUnrollPass` (`lib/Dialect/Triton/Transforms/LoopUnroll.cpp`), added by `make_ttir` as the last TTIR pass (`add_loop_unroll`). |

The pass handles `scf.for` only. On a `tl.condition` loop it takes effect only
if `triton-uplift-while-to-for`, which runs earlier in `make_ttir` and copies
the loop's discardable attributes onto the new `scf.for`, could prove the loop
countable. Otherwise the attribute is ignored.

For each annotated loop, the pass removes `tt.loop_unroll_factor` and calls
MLIR's `loopUnrollByFactor`. If that produces a remainder loop, the pass sets
`tt.num_stages = 1` on it.

Because unrolling happens at TTIR time, AutoWS never sees the original loop.
It sees the unrolled one, with these consequences:

- **Larger loop body.** The unrolled body holds `N` copies of each load, MMA
  and store. Partition scheduling, channel creation, and the memory planner
  treat each copy as a separate op. Each copied descriptor load is its own
  channel with its own SMEM buffer, multi-buffered against `smemBudget`.
- **Remainder loop.** The pass removes only `tt.loop_unroll_factor`. Both the
  unrolled loop and the remainder loop keep every other attribute, including
  `tt.warp_specialize`. The pass pins the remainder loop to one pipeline stage.
- **Persistent WS loops: compile error (reproduced).** `loop_unroll_factor=2`
  on a persistent warp-specialized loop fails in code partitioning with "SMEM
  circular reuse group is ill-formed" (`WSCodePartition.cpp`).
- **Not forwarded.** The attribute is registered as `NotForwarded` in
  `DiscardableAttributes.cpp` (`kAutoWSLoopAttrs`), so the non-persistent
  single-trip-while rewrite does not copy it onto inner loops. By then it has
  already been consumed anyway.

Resources: SMEM, through the extra per-copy channels and buffers. Registers,
through the larger body in each partition. No direct TMEM or warp-count
effect.

### `disable_licm`

| Property | Behavior |
|---|---|
| How it is set | `tl.range(..., disable_licm=True)` or the same kwarg on `tl.condition`. |
| IR | `llvm.loop_annotation` holding an `LLVM::LoopAnnotationAttr` with LICM disabled (`get_disable_loop_licm_attr` in `python/src/ir.cc`). Only emitted when `True`. |
| Default | `False` |
| Consumer | LLVM: `fixUpLoopAnnotation` (`lib/Conversion/TritonGPUToLLVM/Utility.cpp`) re-attaches the annotation to the lowered branch, which becomes `llvm.licm.disable` loop metadata in the LLVM IR. |

The Triton TTIR LICM pass (`triton-licm`,
`lib/Dialect/Triton/Transforms/LoopInvariantCodeMotion.cpp`) does not read the
annotation. Loop-invariant code is still hoisted at TTIR/TTGIR time, including
by the `add_triton_licm` calls that run before AutoWS. The knob only stops
LLVM's LICM on the final loop.

Interaction with AutoWS:

- **Copied to every partition.** When `specializeRegion` clones a loop into
  each partition (`WSSpecialize.cpp`, the `scf.for` and `scf.while` cloning
  paths), it copies every attribute except `async_task_id`. Each partition's
  copy of the loop therefore keeps the annotation.
- **Not forwarded to inner loops.** It is `NotForwarded` in
  `kAutoWSLoopAttrs`, so the non-persistent single-trip-while rewrite does not
  move it to the inner `scf.for`.
- **Already on for the WS dispatch loop.** Independent of this knob,
  `lowerWarpSpecializeCommon` (`WarpSpecializeUtility.cpp`, `disableLICM`)
  always puts LICM-disable metadata on the switch loop that dispatches warps
  to partitions.

Resources: registers only. Code that LLVM would have hoisted is no longer kept
live across the loop.

## Operation scheduling annotations

AutoWS kernels may also attach `tt.autows` to individual dot operations and
latency-bearing loads. A dot annotation has `stage` and `order` fields and
defines the computation order within an annotated loop. `ScheduleLoops` keeps
that dot order unchanged while placing loads from each dot's backward slice in
dedicated prefetch clusters. The default load rank is:

```text
(dot stage + dot order, dot stage, dot order)
```

This rank expresses the prefetch wavefront independently of the computation
wavefront. For example, a stage-zero input for the next contraction can be
issued before a stage-one input for the current contraction.

`tensor_descriptor.load(..., attrs={"stage": "0", "order": "2"})` supplies an
explicit load rank when dependency inference is insufficient, such as scalar
metadata shared by multiple contractions. Explicit load annotations override
the inferred consumer rank. Loads outside every annotated dot's backward slice
retain dependency-based scheduling.

Both fields are JSON **strings** holding a decimal integer, matching the dot
annotation encoding; the frontend rejects non-string `attrs` values so an
integer cannot be silently dropped by the compiler-side parse. An annotation
that is not a JSON object, is missing a field, or holds a non-integer string
is ignored with a `-debug-only=triton-loop-pipeline` message and the op falls
back to the inferred rank.

## Status on `scf.while` (current gaps)

The unified tile scheduler (`triton.language.schedule`) drives a persistent
kernel with an `scf.while` outer loop. The annotations are *attached* to the
`while` identically to a `for` (via `tl.condition`) and survive into TTGIR, but
whether they take *effect* depends on the schedule:

| Schedule | outer-loop handling | annotations effective? |
|---|---|---|
| static persistent | uplifted to `scf.for` (`triton-uplift-while-to-for`) | **yes** — treated as a normal `for` |
| dynamic persistent | stays `scf.while` (atomic advance) | **partial** — see below |
| CLC | stays `scf.while` (hardware `_valid`) | **partial** — see below |
| non-persistent | kept through data partitioning, then eliminated before the default loop schedule | **yes** — AutoWS is forwarded to the first-level inner loop |

For non-persistent schedules on Hopper and Blackwell,
`triton-simplify-single-trip-while` deliberately runs in TTGIR after data
partitioning and before default latency assignment and loop scheduling. Before
inlining the single-trip while, it forwards the still-live AutoWS attributes to
`scf.for` loops exactly one loop-nesting level inside the scheduler while. This
is based on the nearest enclosing loop, so intervening non-loop control flow
such as `scf.if` is transparent, while a nested second-level loop is excluded.
This lets downstream scheduling and warp-specialization passes operate on the
inner loop without hiding the outer loop from data partitioning. An annotated
while with no first-level inner target is preserved rather than silently losing
warp specialization. Attributes whose
consumers already ran, such as `tt.list_schedule_pick`, are not forwarded;
later consumers such as the memory planner and multi-CTA reduction still
receive `tt.mem_plan_pick` and `tt.multi_cta`, respectively. Existing
attributes on the selected inner loop take precedence over outer defaults.

### `warp_specialize` on a non-countable `while`

`ttg.warp_specialize` is emitted by the pipeline `ScheduleLoops` →
`PartitionSchedulingMeta` → `hopper_warpspec`. `PartitionSchedulingMeta` accepts
`scf.while` loops whose condition forwards a direct non-empty ordered subset of carried
values, and derives their partitions structurally from the after region.
`ScheduleLoops` remains `scf.for`-only: the non-countable outer
while is not software-pipelined, while its nested K-loop retains the existing
for-loop software pipeline. Downstream dynamic/CLC validation is tracked in
**[WarpSpecializeWhileLoops.md](WarpSpecializeWhileLoops.md)**.

### `num_stages` / pipelining on a `while`: not applicable

Software pipelining is `scf.for`-only, and a non-countable `while` has no trip
count to stage. We also do **not** want to pipeline the persistent outer loop.
`num_stages` on a dynamic/CLC `while` is therefore inert (worth a frontend
warning); it applies once the static `while` is uplifted to a `for`.

### `data_partition_factor` on a `while`: the *pass* works; the *pipeline* is gated

The data-partition pass (`WSDataPartition`) **does** handle `scf.while` — it
splits a while-carried accumulator + dot along M/N and follows the before/after
regions correctly. This is verified by the lit tests `@while_data_partition`,
`@while_descriptor_data_partition`, and `@while_clc_data_partition` (the last
mirrors the CLC scheduler shape with `ttng.clc_advance`) in
`test/Hopper/WarpSpecialization/ws_while_loop_autows.mlir`.

`WSDataPartition` runs before `PartitionSchedulingMeta`, while the annotated
outer loop still exists. For a non-persistent schedule the subsequent
single-trip rewrite forwards AutoWS to the inner loop, so partition assignment
can continue there. Dynamic/CLC loops are not eliminated and remain gated on
the `PartitionSchedulingMeta` generalization described above. In other words:
the data-partition mechanism is ready for `while`; persistent while
partition-*assignment* is not.

> Known cosmetic artifact: data-partitioning a `while` leaves the original
> full-width dot as a *dead* loop result (not DCE'd, because removing a dead
> `scf.while` iter-arg needs while canonicalization). This is common to all
> `while` loops (the existing `@while_data_partition` test shows it too) and is
> harmless — the split slices are correct.

## Summary

- **Annotations are unified** across `for`/`while` via `AutoWSLoopOptions`; the
  frontend and IR emission are in lockstep, while the shared C++ registry owns
  their scheduler-loop propagation policy.
- **On `scf.for`** (including a static `while` uplifted to `for`): all
  annotations work.
- **On a non-persistent `scf.while`**: data partitioning sees the while before
  it is eliminated, then AutoWS is forwarded to the scheduled inner loop.
- **On a raw `scf.while`** (dynamic/CLC): the annotations attach and survive,
  `WSDataPartition` already understands `while`, but `warp_specialize` /
  `data_partition_factor` do not yet take effect end-to-end because
  `PartitionSchedulingMeta`/`ScheduleLoops` are `scf::ForOp`-typed. Tracked in
  [WarpSpecializeWhileLoops.md](WarpSpecializeWhileLoops.md).
