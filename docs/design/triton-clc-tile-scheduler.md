# Triton CLC Tile Scheduler

Triton Cluster Launch Control (CLC) tile scheduler.

Status: **frontend + high-level IR op landed; lowering pass is future work.**

## Motivation

Cluster Launch Control (CLC) is a Blackwell (SM100+) hardware feature for
dynamic persistent kernels. The grid is launched with one cluster per tile
(over-subscribed relative to the SM count). A CTA that finishes early issues
`clusterlaunchcontrol.try_cancel`, which atomically cancels a *pending*
(not-yet-launched) cluster and hands the caller that cluster's program id — i.e.
it steals the pending tile's work. This load-balances better than a static
persistent schedule (the SM that finishes early does more tiles), and it
absorbs over-provisioned / "jagged" tiles that a non-persistent kernel would
otherwise launch and early-`return` on.

Today CLC is only reachable via low-level ops (Gluon `clc.try_cancel` /
`load_result`, or the TLX `clc_producer` / `clc_consumer` split), which force the
kernel author to hand-manage a response buffer, an mbarrier, and a phase. This
design adds a **core Triton (`tl.`) scheduler** that hides all of that.

## Goals / principles

- **Barrier-free pre-warp-specialization IR.** The high-level op references no
  mbarrier, no response buffer, and no phase. All of that is materialized by a
  later compiler lowering pass. This keeps the IR clean for the passes that run
  before warp specialization (WS).
- **No op is partition-aware.** The scheduler composes only single-program ops.
  Splitting work across producer/consumer warp groups is AutoWS's job, not the
  scheduler's.
- **Runs without WS.** The scheduler is usable in a plain persistent kernel
  (`warp_specialize=False`); WS is an orthogonal, later concern.
- **Minimal, opaque API.** The kernel author never sees a buffer, barrier,
  phase, or the `try_cancel`/`wait` split.

## API

```python
sched = tl.clc_tile_scheduler()      # initialize
while sched.is_valid():              # more work to steal?
    pid = sched.tile_id[0]           # current tile coord x; [1]/[2] for y/z
    ... compute for this tile ...
    sched = sched.advance()          # fetch the next tile
```

- `tl.clc_tile_scheduler()` — construct the scheduler, seeded with this CTA's
  own statically-launched tile.
- `sched.is_valid() -> bool` — is the current tile real work? (Loop condition.)
  True for the seed tile; after `advance`, reflects whether a cluster was
  successfully cancelled.
- `sched.tile_id[dim] -> i32` — the current tile's program-id coordinate along
  `dim` (0=x, 1=y, 2=z).
- `sched.advance() -> ClcTileScheduler` — fetch the next tile and return the
  updated scheduler.

There is intentionally **no `try_cancel` in the API.** Fetching the next tile
*is* `advance`; the async issue/wait split (issue early, wait late, to overlap
the CLC latency with compute) is a compiler optimization done during lowering,
not something the user expresses. Problem-level "invalid/jagged tile" handling is
just ordinary user control flow (a bounds `if` around the compute) — it is not a
scheduler concept.

## Frontend representation

The scheduler is an opaque aggregate carrying the **decoded** current tile:

```
ClcTileScheduler { _valid: i1, _x: i32, _y: i32, _z: i32 }
```

- The constructor seeds it with `is_valid = true` and `_x/_y/_z = program_id(0/1/2)`.
  The first tile needs no CLC op — it is just this CTA's static launch id.
- `is_valid()` returns `_valid`; `tile_id` returns the `(x, y, z)` tuple.
- `advance()` emits one op, `ttng.clc_advance`, which returns the decoded next
  tile `{isValid, x, y, z}`.

The persistent `scf.while` therefore carries four plain values `(i1, i32, i32,
i32)` — no memory descriptors.

### The op

```
def TTNG_CLCAdvanceOp : TTNG_Op<"clc_advance", []> {
  let results = (outs I1:$isValid, I32:$x, I32:$y, I32:$z);
}
```

Effectful (it claims a pending cluster) so it is neither DCE'd nor hoisted out of
the loop; one per persistent-loop iteration. It references no mbarrier or buffer.

### Initial TTIR

```mlir
%v0 = arith.constant true
%x0 = tt.get_program_id x
%y0 = tt.get_program_id y
%z0 = tt.get_program_id z
scf.while (%v=%v0, %x=%x0, %y=%y0, %z=%z0) : (i1,i32,i32,i32) -> (i1,i32,i32,i32) {
  scf.condition(%v) %v, %x, %y, %z
} do {
^bb0(%v: i1, %x: i32, %y: i32, %z: i32):
  ... compute using %x (guarded by problem bounds) ...
  %v1, %x1, %y1, %z1 = ttng.clc_advance : i1, i32, i32, i32
  scf.yield %v1, %x1, %y1, %z1
}
```

No `init_barrier` / `wait_barrier` / `barrier_expect` / `clc_try_cancel` /
`clc_load_result` — those belong to the lowering.

## Lowering (future work)

A TTIR→TTGIR pass expands each persistent loop containing `ttng.clc_advance`:

1. Allocate the response buffer (`memdesc<2xi64>`) and the "full" mbarrier at
   function scope; init / inval them.
2. Expand `clc_advance` into the low-level sequence:
   - **issue:** `ttng.clc_try_cancel(resp, bar)` + `barrier_expect(bar, 16)`
   - **wait+decode:** `wait_barrier(bar, phase)` + `ttng.clc_load_result(resp)` +
     `ttng.clc_is_canceled` (→ `isValid`) + `ttng.clc_get_program_id` (→ x/y/z).
3. **Pull the issue up** to the top of the loop body so the CLC latency overlaps
   the tile's compute (issue early, wait late). This is the request/acquire split
   — an optimization owned by the compiler, not the user.
4. **Infer the mbarrier phase** from the loop induction (parity toggles once per
   iteration).
5. **Drop unused coordinates.** If the kernel only reads `tile_id[0]`, the `y`/`z`
   results (and their carried loop values) are removed via dead loop-carried
   value elimination, so no work is emitted for them.

Prefetch depth (`TRITON_WS_TILE_PREFETCH_DEPTH`) is also a lowering-side concern:
depth 1 is single-stage; depth > 1 (multi-stage run-ahead) additionally requires
drain-on-exit because the CLC claim is destructive.

## Design decisions & rationale

- **`advance`-only (no user `try_cancel`).** The optimal issue point for the
  async request is invariant — the top of the loop body — so a user-placed
  `try_cancel` could only reproduce what the compiler already picks, while adding
  a footgun (the issue must fire exactly once per iteration and pair with exactly
  one wait). Collapsing to one op makes mis-pairing unrepresentable. If manual
  placement is ever needed, `try_cancel` can be re-added as an optional override
  — a strict superset, no rework.
- **Return the decoded tile (`isValid`, x, y, z), not an opaque i128.** This
  removes the opaque result type, the separate `is_canceled` / `get_program_id`
  decode ops, and the `tile_id` proxy from the frontend; the scheduler carries
  plain `tl` values and `is_valid` / `tile_id` are trivial reads. Tradeoff: all
  three coordinates are produced/carried, so dropping unused dims moves from
  "free" (never emitted) to a lowering/canonicalization step. The coordinates are
  cheap `i32`s and the pass eliminates the dead ones, so this is a good trade for
  a much simpler frontend.
- **Seed is `program_id`, not an op.** The first tile of every persistent CTA is
  its static launch id; only *subsequent* tiles come from CLC. Expressing the
  seed with `program_id` + `true` needs no new op and keeps the loop body
  uniform.
- **Phase and the issue/wait overlap are inferred in the pass.** Since the pass
  already analyzes the persistent loop, it owns phase parity and the request
  pull-up; the frontend carries no counter.

## Status & testing

- Implemented: the `ttng.clc_advance` op, the `create_clc_advance` builder
  binding, and the `tl.clc_tile_scheduler` frontend.
- Not implemented: the lowering pass. Until it lands the feature is **not
  executable**; tests in `python/test/unit/language/test_triton_clc.py` are
  IR-only (assert the high-level op appears, the loop carries `(i1,i32,i32,i32)`,
  and no barrier/buffer ops are present).
- A runnable end-to-end CLC GEMM was validated on GB200 in an earlier prototype
  that emitted the raw barrier sequence directly (git history); those correctness
  tests will be re-enabled once the lowering pass exists.

## Future work

- The TTIR→TTGIR lowering pass described above.
- AutoWS integration: because the scheduler is partition-agnostic and carries
  only decoded scalars, a WS pass can broadcast just the used coordinates across
  partitions.
- Multi-stage prefetch (`depth > 1`) with drain-on-exit.
