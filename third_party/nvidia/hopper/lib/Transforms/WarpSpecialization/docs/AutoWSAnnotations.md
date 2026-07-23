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
| `loop_unroll_factor` | `tt.loop_unroll_factor` | `LoopUnroll` (TTIR) | IR-level unroll |
| `flatten` | `tt.flatten` | `FuseNestedLoops` | flatten a loop nest |
| `disallow_acc_multi_buffer` | `tt.disallow_acc_multi_buffer` | pipeliner | keep acc single-buffered |
| `multi_cta` | `tt.multi_cta` | multi-CTA reduction | cluster-level reduction |
| `merge_epilogue` / `merge_epilogue_to_computation` / `merge_correction` | `tt.merge_*` | `PartitionSchedulingMeta` (`SchedulingOptions`) | epilogue/correction routing |
| `separate_epilogue_store` | `tt.separate_epilogue_store` | `PartitionSchedulingMeta` | epilogue store gets own partition |
| `tmem_alloc_algo` / `smem_alloc_algo` / `smem_budget` / `smem_circular_reuse` | `tt.{tmem,smem}_alloc_algo`, `tt.smem_budget`, `tt.smem_circular_reuse` | `WSMemoryPlanner` | allocation heuristics |
| `list_schedule_pick` / `mem_plan_pick` | `tt.list_schedule_pick` / `tt.mem_plan_pick` | list scheduler / `WSMemoryPlanner` | ranked schedule and memory-plan selection |
| `disable_licm` | `llvm.loop_annotation` | LICM | suppress hoisting |

See `PartitionSchedulingMeta.md` for how the `merge_*` / `separate_epilogue_store`
knobs shape the partition layout.

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
identity-carry `scf.while` and derives its partitions structurally from the
after region. `ScheduleLoops` remains `scf.for`-only: the non-countable outer
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
