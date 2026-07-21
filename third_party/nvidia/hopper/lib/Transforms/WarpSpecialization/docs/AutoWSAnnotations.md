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
dataclass, **`AutoWSLoopOptions`** (`core.py`), which is the *one* place the
option set, defaults, and IR-attribute mapping are defined. `visit_For` and
`visit_While` both emit them via the shared `_apply_loop_options` helper, so a
`for` and a `while` receive byte-identical attributes. Adding a new knob is a
single new field on `AutoWSLoopOptions`.

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
| `disable_licm` | `llvm.loop_annotation` | LICM | suppress hoisting |

See `PartitionSchedulingMeta.md` for how the `merge_*` / `separate_epilogue_store`
knobs shape the partition layout.

## Status on `scf.while` (current gaps)

The unified tile scheduler (`triton.language.schedule`) drives a persistent
kernel with an `scf.while` outer loop. The annotations are *attached* to the
`while` identically to a `for` (via `tl.condition`) and survive into TTGIR, but
whether they take *effect* depends on the schedule:

| Schedule | outer loop after `make_ttir` | annotations effective? |
|---|---|---|
| static persistent | uplifted to `scf.for` (`triton-uplift-while-to-for`) | **yes** — treated as a normal `for` |
| dynamic persistent | stays `scf.while` (atomic advance) | **partial** — see below |
| CLC | stays `scf.while` (hardware `_valid`) | **partial** — see below |
| non-persistent | single-trip `while` eliminated (`triton-simplify-single-trip-while`) | n/a — no loop |

### `warp_specialize` on a non-countable `while`: **no-op today**

`ttg.warp_specialize` is emitted by the pipeline `ScheduleLoops` →
`PartitionSchedulingMeta` → `hopper_warpspec`. `ScheduleLoops` and
`PartitionSchedulingMeta` walk **`scf::ForOp` only**, so a `tt.warp_specialize`
on a raw `scf.while` (dynamic/CLC) is never consumed — no partitions are
assigned and no `ttg.warp_specialize` is produced. Only the *static* schedule,
which is uplifted to an `scf.for` before these passes run, is warp-specialized.

This is a code-scope limitation, not a fundamental one, and **not** a software-
pipelining requirement (the partition scheduler derives partitions structurally
and ignores the SWP `distance`). Closing it means generalizing
`PartitionSchedulingMeta` (and `ScheduleLoops`) to `scf.while`. The full plan is
in **[WarpSpecializeWhileLoops.md](WarpSpecializeWhileLoops.md)**.

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

However, `WSDataPartition` runs *after* `PartitionSchedulingMeta` has assigned
partitions and `tt.warp_specialize` has been consumed. Because that does not
happen for a non-uplifted `while` (previous section), end-to-end data
partitioning of a dynamic/CLC `while` is gated on the same
`PartitionSchedulingMeta` generalization. In other words: the data-partition
mechanism is ready for `while`; the partition-*assignment* front-end is not.

> Known cosmetic artifact: data-partitioning a `while` leaves the original
> full-width dot as a *dead* loop result (not DCE'd, because removing a dead
> `scf.while` iter-arg needs while canonicalization). This is common to all
> `while` loops (the existing `@while_data_partition` test shows it too) and is
> harmless — the split slices are correct.

## Summary

- **Annotations are unified** across `for`/`while` via `AutoWSLoopOptions`; the
  frontend and IR emission are in lockstep.
- **On `scf.for`** (including a static `while` uplifted to `for`): all
  annotations work.
- **On a raw `scf.while`** (dynamic/CLC): the annotations attach and survive,
  `WSDataPartition` already understands `while`, but `warp_specialize` /
  `data_partition_factor` do not yet take effect end-to-end because
  `PartitionSchedulingMeta`/`ScheduleLoops` are `scf::ForOp`-typed. Tracked in
  [WarpSpecializeWhileLoops.md](WarpSpecializeWhileLoops.md).
