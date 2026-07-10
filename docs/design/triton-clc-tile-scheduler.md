# Triton CLC Tile Scheduler

Triton Cluster Launch Control (CLC) tile scheduler.

Status: **frontend + full lowering landed (Stages 1–4), single-CTA. Runs both
without WS and under AutoWS (Stage 3 fused into the run-once/broadcast pass).**

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
- **Single CTA only (for now).** Multi-cluster / multi-CTA is not yet supported;
  the materialization pass rejects `num-ctas > 1`. Multi-CTA is future work.

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

## Lowering

`clc_advance` is remapped to **two ops linked by a token**, and only later
materialized into mbarriers by a **dedicated TTGIR pass that runs before AutoWS**.
Crucially, mbarriers are not introduced by the split — the token stands in for the
async dependency until materialization.

### Token form (reuses `!ttg.async_token`)

Two ops model the intermediate form; together they are exactly `clc_advance`
(the split is result-preserving — we remap one op to two):

- `ttng.clc_try_cancel_async : !ttg.async_token` — issue the async CLC request.
  Returns **only a token**, nothing else. Effectful (claims a pending cluster).
- `ttng.clc_read(!ttg.async_token) -> (i1 isValid, i32 x, i32 y, i32 z)` — await
  the token and return the decoded tile. Same results as `clc_advance` (it must
  return `is_valid` too).

The token is `!ttg.async_token`, the same completion-handle type used by
`cp.async` / TMA / `tcgen05` — so the existing async-dependency machinery applies.

### Stage 1 — split (minimal, trivially correct)

For each `%v,%x,%y,%z = ttng.clc_advance`, emit the issue immediately followed by
the read, linked by a token — no code motion yet:
```mlir
%tok = ttng.clc_try_cancel_async : !ttg.async_token
%v,%x,%y,%z = ttng.clc_read %tok : i1, i32, i32, i32
```
This is a pure structural remap (`clc_advance` ≡ `clc_read(clc_try_cancel_async())`);
placing them adjacent is always correct. Overlap is introduced by the next pass.

### Stage 2 — `clc_try_cancel_async` hoisting / unification (dedicated TTGIR pass, before AutoWS)

Promotes each `clc_try_cancel_async` to the **earliest safe program point that
dominates its `clc_read`** consumer(s). The goal is to maximize the distance
between the async issue and the decode so the CLC round-trip is hidden behind
compute — i.e. `clc_read`'s (eventual) wait rarely stalls.

- **Same block (common case):** move the issue above the tile's compute (e.g. to
  the top of the persistent loop body).
- **Across basic blocks:** the issue can be hoisted out of conditional regions.
  E.g. if `advance` appears in *both* arms of an `if/else` (a masked path),
  **unify** the two issues into a single `clc_try_cancel_async` at the common
  dominator (before the branch); each arm's `clc_read` consumes the shared token.
  This both starts the request earlier and removes the redundant issue.

**Correctness invariant — issue/read count parity.** Every dynamic path must
execute exactly one `clc_try_cancel_async` per `clc_read`: an issue claims one
pending cluster and a read consumes one response. The pass must never introduce
an issue on a path that has no read (over-claim → lost/duplicated tiles, wrong
termination) nor drop an issue on a path that reads. Hoisting past a branch is
therefore legal only when *every* path from the hoisted location issues-and-reads
exactly once (e.g. both `if/else` arms advance); otherwise the issue stays inside
its conditional region. The token SSA edge makes this dominance-based code motion;
the effect (claims a cluster) is what bounds it.

Within a persistent loop, hoisting to the loop-body top gives one-iteration
overlap; hoisting *across* iterations (loop rotation / run-ahead) is the depth>1
prefetch case (token becomes loop-carried; needs drain-on-exit) — future.

IR after Stage 2 (hoisted issue; still **no** mbarrier / buffer / phase — the
token form flows unchanged into AutoWS):
```mlir
scf.while (%v=%v0,%x=%x0,%y=%y0,%z=%z0) : (i1,i32,i32,i32) -> (i1,i32,i32,i32) {
  scf.condition(%v) %v, %x, %y, %z
} do {
^bb0(%v: i1, %x: i32, %y: i32, %z: i32):
  %tok = ttng.clc_try_cancel_async : !ttg.async_token        // hoisted issue
  ... compute using %x (guarded by problem bounds) ...
  %v1,%x1,%y1,%z1 = ttng.clc_read %tok : i1, i32, i32, i32   // decode / await
  scf.yield %v1, %x1, %y1, %z1
}
```

### Stage 3 — AutoWS handling (fused with the atomic broadcast)

The token form flows **into** AutoWS unchanged. The CLC fetch is the same shape
as the atomic tile-counter that `WSAtomicBroadcast.cpp` already handles — a
run-once "claim the next tile" whose loop-carried result feeds *every* partition
— so the two are **handled by one merged entry point**,
`doDynamicTileBroadcast` (it processes both `tt.atomic_rmw` and `ttng.clc_read`;
called from `WarpSpecialization.cpp` right after `doTaskIdPropagate`). Without
this, each partition would run its own `clc_try_cancel_async`, cancel a different
pending cluster, and diverge → deadlock.

*Enabling WS:* like the dynamic-persistent atomic kernel, WS is requested by
`tl.range(..., warp_specialize=True)` on the inner K-loop (with `use_meta_ws`);
the outer persistent `scf.while` is auto-cloned per partition and the CLC fetch
is broadcast here. The grid launches one cluster per tile so there are pending
clusters to steal.

Primary scheme:

1. **Recognize** the CLC fetch (`clc_try_cancel_async` + `clc_read`) as a
   run-once loop-carried tile producer, alongside `atomic_rmw`.
2. **Map all CLC ops to a single partition — the producer.** Both the issue and
   the `clc_read` (decode) live only in the producer partition; no other partition
   runs any CLC op. (The token stays intra-producer, so its completion is a local
   wait, materialized in Stage 4.)
3. **Broadcast `clc_read`'s 4-tuple `(is_valid, x, y, z)` through SMEM** to all
   partitions, reusing the atomic-broadcast data path: the producer
   `splat`→`local_store`s each scalar into a small SMEM slot; every partition
   `local_load`→`unsplat`s it; the `scf.while` loop-carried tile is rewired to the
   broadcast values. The `local_store`→`local_load` edges become the usual
   `doCodePartitionPost` SMEM channels. `is_valid` is part of the broadcast tuple
   (it drives every partition's loop).
4. **Reject/bail** exactly like atomic broadcast's case-3: if the fetch's results
   are not cleanly loop-carried-to-all, fall back to `removeWarpSpecializeAttr`
   (unspecialized but compilable — never a crash).

#### Followup optimizations (to explore, not in the first cut)

- **Broadcast the response and decode per partition.** Instead of broadcasting the
  decoded 4-tuple, broadcast the raw CLC response (or keep the token) and let each
  partition run its own `clc_read`/decode. Only `clc_try_cancel_async` must remain
  single (one claim) and delayed (overlap); the decode is pure and replicable.
  This shrinks/changes the broadcast payload and composes with per-partition
  dead-coordinate dropping (each partition decodes only the coords it uses). To be
  explored as a channel optimization.
- **Dedicated tile-computation partition/buffer.** Put the generic "tile
  computation" (the `program_id` → tile-coordinate / offset math, e.g. the grouped
  swizzle in `_compute_pid`) in its own partition with its own channel/buffer,
  rather than replicating it in every partition. More flexible partitioning.

Both are followup optimizations to the channel/broadcast design; the first cut
uses the simple "producer decodes, broadcast the 4-tuple" scheme above.

### Stage 4 — materialize token → mbarrier (after AutoWS)

Runs *after* AutoWS, so it operates on the token pair as it now sits in the
producer partition (Stage 3 put all CLC ops there). It converts the async token
into the real barrier ops. This is a **separate barrier from the broadcast**:
Stage 3 / AutoWS already created the SMEM channels (with their own full/empty
mbarriers) that distribute the decoded 4-tuple to the other partitions; Stage 4
only handles the CLC-response completion barrier local to the producer.

Materialization (producer partition, depth-1):

1. Allocate the response buffer (`memdesc<2xi64>`) and one **completion mbarrier**
   at function scope; init / inval.
2. `clc_try_cancel_async` → `ttng.clc_try_cancel(resp, bar)` + `barrier_expect(bar, 16)`.
3. `clc_read %tok` → `wait_barrier(bar, phase)` + `ttng.clc_load_result(resp)` +
   `ttng.clc_is_canceled` (→ `isValid`) + `ttng.clc_get_program_id` (→ x/y/z).

#### Phase handling (one-sided channel)

The CLC response is written by hardware (`try_cancel`) and read by `clc_read` —
there is only the **"full" (data-ready) side** of a channel; there is no "empty"
(buffer-free) side, because the single response buffer is reused each iteration
and reuse is naturally program-ordered (the next iteration's issue at the loop-body
top follows this iteration's read at the bottom, in the same partition). So a
single completion mbarrier suffices, and only its phase must be tracked.

**Require the phase to be a loop-carried variable, toggled before each advance.**
Materialization adds an `i32`/`i1` phase iter_arg to the persistent `scf.while`,
initialized to 0; `clc_read` waits with the carried phase, and the phase is XOR'd
with 1 in the loop yield (once per iteration). This is:

- **the same shape the software pipeliner already uses for TMA / async waits** —
  `Pipeliner/LowerLoops.cpp` carries `phase` as an iter_arg and toggles it with
  `arith.xori` at the yield (`phase = forOp.getRegionIterArg(...)`;
  `newPhase = xori(phase, 1)`); and
- **the one-sided (`numBuffers = 1`) case of the WS channel logic** — the channel
  path computes `phase = (accumCnt / numBuffers) & 1` from a loop-carried
  `accumCnt` in `CodePartitionUtility.cpp::getBufferIdxAndPhase`, which for a
  single buffer reduces to `accumCnt & 1`, i.e. a per-iteration toggle.

**TMA lowering reference (`lib/Dialect/TritonGPU/Transforms/Pipeliner/LowerLoops.cpp`).**
The software pipeliner lowers `tt.descriptor_load` into exactly this shape; reuse it:

- **Barrier ring:** `triton::createBarrierAlloc(loop, numBuffers)` (allocates +
  inits `numBuffers` mbarriers); slot via
  `triton::createSingleBufferView(builder, alloc, idx)`.
- **Issue:** `ttng.BarrierExpectOp(bar, sizeInBytes, pred)` +
  `ttng.AsyncTMACopyGlobalToLocalOp(...)` (`createTMABarrierAndWait` ~L380-403,
  `createTMAAsyncLoad` ~L240-252). CLC substitutes `clc_try_cancel(resp, bar)`.
- **Wait:** `ttng.WaitBarrierOp(barView, phase)` (~L402). CLC follows it with
  `clc_load_result` + `clc_is_canceled` / `clc_get_program_id`.
- **Phase + indices as loop-carried iter_args** (~L535-588): `insert/extract`
  init `-1`, `phase` init `0`, added via `addIterArgsToLoop`; advanced per
  iteration with `createIncrementModulo(idx, numBuffers, …)` and
  `phase = select(cndExt, xori(phase, 1), phase)`; yield patched (~L612-618).

For CLC depth-1 (`numBuffers = 1`) the ring wraps every iteration, so `cndExt` is
always true and the phase reduces to a plain per-iteration `xori(phase, 1)` — the
"loop-carried, toggled-before-each-advance" phase, and no separate insert/extract
index is needed. Two differences to handle: CLC's loop is an `scf.while`
(persistent), not `scf.for`, so the phase iter-arg follows the while before/after
regions (cf. `getAccumCount`'s scf.while handling in `CodePartitionUtility.cpp`)
rather than the scf.for-shaped `addIterArgsToLoop`; and the "copy" is
`clc_try_cancel` (+ decode on read) instead of `async_tma_copy` (+ local_load).

Depth > 1 (future) generalizes this to a buffer ring: the carried counter becomes
`accumCnt`, `bufferIdx = accumCnt % depth`, `phase = (accumCnt / depth) & 1` (the
full channel form), and the destructive claim then also needs drain-on-exit.

### Pipeline

```
frontend ttng.clc_advance                          (TTIR)
   -> [Stage 1: split]         clc_try_cancel_async (token) + clc_read  (adjacent)
   -> [Stage 2: hoist/unify]   clc_try_cancel_async promoted to earliest safe point
   -> [Stage 3: AutoWS]        producer runs fetch + broadcast decoded 4-tuple  (fused into WSAtomicBroadcast)
   -> [Stage 4: materialize]   token -> buffer + completion mbarrier + loop-carried phase (after AutoWS)
```

Prefetch depth (`TRITON_WS_TILE_PREFETCH_DEPTH`) is a materialization-stage
concern: depth 1 is single-stage; depth > 1 (multi-stage run-ahead) additionally
requires drain-on-exit because the CLC claim is destructive.

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
- **Token-based split before mbarriers, reusing `!ttg.async_token`.** Splitting
  `clc_advance` into `clc_try_cancel_async` (→ token) + `clc_read` (token →
  decoded tile) expresses the request/acquire structure and enables the overlap
  hoist *without* introducing mbarriers, using the same async-token dependency
  model as `cp.async` / TMA. mbarriers are materialized only in a later pass
  (after AutoWS), keeping the two concerns — async structure vs. barrier
  realization — cleanly separated and independently testable.

## Status & testing

- **Implemented:** the `tl.clc_tile_scheduler` frontend and `ttng.clc_advance`
  op; the token ops `ttng.clc_try_cancel_async` / `ttng.clc_read`; Stages 1
  (`triton-nvidia-gpu-clc-split`), 2 (`triton-nvidia-gpu-clc-hoist`) and 4
  (`triton-nvidia-gpu-clc-materialize`) as TTGIR passes
  (`lib/Dialect/TritonNvidiaGPU/Transforms/CLCLowering.cpp`); and Stage 3 (AutoWS)
  merged into `doDynamicTileBroadcast` in
  `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSAtomicBroadcast.cpp`.
  Wired into the NVIDIA Blackwell `make_ttgir` — split + hoist before WS,
  materialize after; the AutoWS broadcast runs inside WS.
- **Restriction:** single CTA only — `clc-materialize` errors on `num-ctas > 1`.
- **Tests:** `python/test/unit/language/test_triton_clc.py` (frontend IR checks +
  non-WS Blackwell execution + materialized-TTGIR) and, in
  `test_tutorial09_warp_specialization.py`,
  `test_tutorial09_matmul_tma_clc_persistent_while_loop_warp_specialize` (Blackwell
  warp-specialized CLC GEMM: asserts `ttg.warp_specialize` + `ttng.clc_try_cancel`
  and correctness).

## Future work

- Multi-cluster / multi-CTA support (lift the single-CTA restriction).
- Multi-stage prefetch (`depth > 1`) with drain-on-exit (loop-carried token).
- Cross-basic-block hoist / unification in Stage 2.
- Channel optimizations: broadcast-and-decode-per-partition; a dedicated
  tile-computation partition/buffer.
