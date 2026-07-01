# Triton Unified Tile Scheduler

Status: **frontend landed** — `triton.language.schedule`
(`python/triton/language/schedule.py`), exposed as `tl.*`. Four schedules ship:
`NonPersistentScheduler`, `StaticPersistent1DScheduler`,
`DynamicPersistent1DScheduler`, `ClcTileScheduler`.

## Motivation

A persistent kernel iterates over output *tiles*. How a program maps to tiles,
how it advances, and how it detects "no more work" all depend on the *schedule*:

- **non-persistent** — grid == num_tiles; each program does exactly one tile.
- **static persistent** — grid == `min(num_programs, num_tiles)`; each program
  walks `pid, pid+num_programs, pid+2*num_programs, …`.
- **dynamic persistent** — each program pulls the next tile from a shared global
  atomic counter (work stealing).
- **CLC** — Blackwell hardware work stealing (see
  `docs/design/triton-clc-tile-scheduler.md`).

Historically each was hand-written inline in every kernel with a subtly
different loop shape. This module unifies them behind one opaque loop API so the
loop body is identical regardless of schedule, which lets the schedule be an
**autotunable `constexpr`**.

## API

```python
sched = SCHEDULE.initialize(lowering_args, num_tiles_fn, tile_counter)
while sched.is_valid():                 # more work?
    tile_id = sched.tile_id[0]          # current linear tile id ([1]/[2] = y/z)
    ... compute for this tile ...
    sched = sched.advance()             # claim the next tile
```

- **`SCHEDULE`** — one of the scheduler classes, passed to the kernel as a
  `tl.constexpr`. Autotuning simply varies which class is passed.
- **`initialize(lowering_args, num_tiles_fn, tile_counter)`** — seed the scheduler
  with this program's first tile. The signature is uniform across all schedules;
  each ignores the arguments it does not need (`tile_counter` for everything but
  dynamic; `num_tiles_fn`/`lowering_args` for non-persistent and CLC).
- **`is_valid()`** — the `while` condition.
- **`tile_id`** — a 3-tuple `(x, y, z)`; read `tile_id[0]` for the linear id.
- **`advance()`** — return the updated scheduler for the next tile.

### `lowering_args` + `num_tiles_fn`

The count-limited schedules (static/dynamic) need to know the total tile count.
Rather than being handed a precomputed number (which would not re-specialize per
autotune config), they carry:

- `lowering_args` — the values the tile count depends on, e.g.
  `(M, N, BLOCK_M, BLOCK_N)` (a tuple or `NamedTuple`), and
- `num_tiles_fn` — a user `@triton.jit` function that computes the count from
  `lowering_args`.

`is_valid` computes `num_tiles = num_tiles_fn(lowering_args)` **in-kernel**, over
the *tuned* block-size constexprs. Recomputing every iteration is free: it is pure
loop-invariant integer arithmetic, so CSE + LICM hoist it out of the `scf.while`.

## Design

### Class structure

- `TileScheduler` — the base contract. Its `is_valid` is a `static_assert`
  (compile-time error) so any schedule that forgets to implement it fails loudly.
- `_CountingTileScheduler(TileScheduler)` — shared count-based `is_valid`
  (`self._x < num_tiles_fn(self._lowering_args)`), inherited by the two 1D
  persistent schedules.
- The four concrete schedules are `@tl._aggregate` value types (so instances flow
  through `@triton.jit` kernels as `scf.while` loop-carried state). Each declares
  its own fields, its `initialize`/`advance`, and gets the `tile_id` property
  attached after the class.

| Schedule | `is_valid` | `advance` | extra state |
|---|---|---|---|
| `NonPersistentScheduler` | stored `_valid` (True→False) | flip `_valid` False | — |
| `StaticPersistent1DScheduler` | `_x < num_tiles` | `_x += num_programs` | `_stride` |
| `DynamicPersistent1DScheduler` | `_x < num_tiles` | `_x = atomic_add(counter,1)` | `_counter` |
| `ClcTileScheduler` | hardware `_valid` | `ttng.clc_advance` (decoded tile) | — |

### Selection is compile-time

Which schedule runs and the `num_tiles_fn` both change the generated code, so both
must be known at compile time — they are `constexpr`. `SCHEDULE` and `num_tiles_fn`
become part of the JIT cache key (a `@triton.jit` function is hashable), so there
is exactly one compile per `(SCHEDULE, num_tiles_fn, block sizes, …)` combination.

### Host responsibilities

Grid shape and the dynamic counter are **caller-provided** (they are launch-time
concerns, not kernel code):

- grid: `num_tiles` for non-persistent / CLC (CLC over-subscribes), else
  `min(NUM_SMS, num_tiles)`.
- `tile_counter`: a 1-element `int32` device tensor for the dynamic schedule,
  initialized to the number of launched programs (so static seeds `0..NUM_SMS-1`
  by program id and the counter hands out `NUM_SMS, NUM_SMS+1, …`).

## Limitations

These are consequences of how Triton's frontend / aggregates work, established
while designing this module.

1. **Schedule + `num_tiles` must be compile-time.** Anything that changes codegen
   (which schedule, which tile-count function) has to be a `constexpr`; it cannot
   be a runtime kernel argument. Runtime args are device tensors/scalars and can
   carry neither a Python object nor a `@triton.jit` function nor a control-flow
   choice.

2. **`num_tiles` is a passed-in function, not an overridable method.** Making
   `num_tiles` a method that users override by subclassing does *not* compose:
   `@tl._aggregate` values are immutable and `advance` rebuilds the concrete type
   *by name* (e.g. `StaticPersistent1DScheduler(...)`), so a subclass's override is
   dropped after the first `advance` and the `scf.while` loop-carried type no
   longer matches. The behavior that must survive `advance` therefore has to be
   carried as state — hence `num_tiles_fn` is a `constexpr` field supplied at
   `initialize`. (A fully user-authored schedule aggregate — its own fields +
   `initialize` + `advance` + `num_tiles` — does work, at the cost of
   reimplementing the stride/atomic mechanics.)

3. **A "proper schedule type" with bound methods can't be host-constructed.**
   `@jit` methods only bind `self` when the receiver is a Triton value
   (`isinstance(o, base_value)`, i.e. an `@tl._aggregate`), which exists only
   inside a kernel. A host-constructed object passed as a `constexpr` is plain
   Python: you can read its *attributes*, but calling its `@jit` methods does not
   bind `self`. So the schedule instance is created in-kernel via `initialize`; the
   host chooses the *class* and passes it as a `constexpr`.

4. **The dynamic counter must be host-allocated and host-initialized.** It is a
   single grid-shared global location. Triton's only in-kernel global allocator,
   `ttg.global_scratch_alloc`, is *private to the current program*, so it cannot be
   the shared counter; and there is no grid-wide barrier, so no CTA can safely set
   the initial value from inside the kernel without racing other CTAs. In-kernel
   scratch is also not zero-initialized. The counter therefore stays a kernel
   argument prepared on the host. (The *initial value* logic could move into
   `initialize` by claiming the first tile via `atomic_add` from a zeroed counter —
   fully dynamic — but the zeroed buffer still comes from the host.)

5. **Aggregate fields are not inherited.** `@tl._aggregate` reads only a class's
   *own* `__annotations__`, so each concrete schedule re-declares its fields;
   `tile_id` (a property, which `@_aggregate` does not copy) is attached per class.
   Only *methods* are inherited (e.g. the shared `is_valid`).

6. **`tile_id` is a 3-tuple.** Linear schedules set `y = z = 0`; kernels read
   `tile_id[0]`. This keeps CLC (which genuinely produces `x/y/z`) swappable with
   the linear schedules with no loop-body change.

7. **CLC is single-CTA and Blackwell-only.** Inherited from the CLC lowering; see
   `docs/design/triton-clc-tile-scheduler.md`.

## Files & tests

- `python/triton/language/schedule.py` — the schedulers.
- `python/triton/language/clc.py` — `ClcTileScheduler` re-export + the
  `clc_tile_scheduler()` convenience factory (backward compatible).
- `python/test/unit/language/test_tile_scheduler.py` — GPU-free IR-shape tests
  (one `scf.while`; static uses `get_num_programs`; dynamic uses an atomic; CLC
  emits `clc_advance`; the base `is_valid` requirement), plus GPU execution tests
  (each tile visited once; matmul correctness). CLC gated to Blackwell.
- `python/test/unit/language/test_tutorial09_warp_specialization.py` —
  `matmul_kernel_tma_unified_persistent_ws_while`: one warp-specialized persistent
  kernel reused across all four schedules, parametrized over `EPILOGUE_SUBTILE` ×
  schedule type.

## Future work

- A host helper that packages grid + `tile_counter` so callers don't re-derive
  them.
- Fully dynamic counter (claim the first tile via `atomic_add` from a zeroed
  counter) to drop the `NUM_SMS` seed value.
- Lift CLC single-CTA restriction (tracks the CLC lowering).
