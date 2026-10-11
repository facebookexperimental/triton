# Hopper Warp-Spec Consumer Patterns

Use this guidance only for NVIDIA Hopper (H100, sm_90) kernels that already have the 2-task producer/consumer scaffold (see `hopper-warp-spec-producer-consumer.md`). It selects and builds the consumer fan-out: pingpong, cooperative, replicated rings, or a third sink-writer task.

## Select The Pattern

- **Pingpong**: two or more consumer tasks contend for the SAME dedicated unit (MMA, SFU, ALU). Goal is serialization with overlap of non-contended work.
- **Cooperative**: replicas partition the SAME output with no cross-partition dependency and share at least one input. Goal is parallel throughput scaling.
- **Replicated rings**: the consumer declares `replicate=N>=2` with per-replica sub-tiles; pick the ring layout by operand reload cadence (below).
- **Sink-writer**: the epilogue latency is exposed AND all three G-gates hold AND no B-gate fires (below).

Pingpong and cooperative are alternatives: serialize turns on one contended resource versus parallel teamwork on one work item.

## Build The Pingpong Handoff

Two tasks take turns owning the contended unit behind a two-slot handoff: each task waits on its own slot and arrives on the other's (mbarrier with `num_barriers=2`, or two distinct named-barrier IDs). Bootstrap with an initial arrive from the second task so the first task's opening wait passes. Wait BEFORE the contended region, arrive AFTER it, run non-contended work (epilogue compute, stores) after the arrive so it overlaps the partner's exclusive region, and toggle phase each iteration. Pick one formulation (mbarrier or named) and stay consistent.

Failure modes: missing bootstrap deadlocks at start; a single shared slot deadlocks (each task can pass its own next wait); wait/arrive on the wrong side leaves both tasks in the exclusive region; mixed formulations leave dead state; an untoggled phase goes stale.

## Build The Cooperative Partition

Split the output along an axis with no cross-replica dependency. Shared inputs get one `NUM_STAGES` ring whose empties barrier uses `arrive_count=N` so it flips only after ALL replicas release; partitioned inputs get a `NUM_STAGES * N` ring indexed `buf + NUM_STAGES * cid`. The producer issues one shared load plus N partitioned loads at disjoint offsets per iteration; each replica waits on its private slot and the shared slot, computes its partition, releases both, and writes its output at a `cid`-dependent offset.

Failure modes: shared empties at `arrive_count=1` lets the producer overwrite a buffer mid-read; indexing the private ring with bare `buf` reads the wrong replica's slot; a `cid`-independent output offset overlaps partitions; a single private load starves N-1 replicas.

## Lay Out Replicated Rings By Reload Cadence

For `replicate=N>=2` with `BLOCK_M_SPLIT = BLOCK_M // N` at or above the WGMMA m64 floor, pick per operand by reload cadence:

- **Tile-persistent** (loaded once per output tile): N independent slots of `(BLOCK_M_SPLIT, INNER_DIM)` with per-replica `x_full[cid]` barriers, prefetched in an unrolled prologue. The pre-hook MUST set the descriptor `block_shape` to the split shape; a full-`BLOCK_M` descriptor into a split buffer silently corrupts.
- **Inner-iteration** (reloaded every inner iter): one flat strided ring of depth `NUM_STAGES * N` with matching barriers, indexed `buf + cid * NUM_STAGES` — stride is `NUM_STAGES`, not `cid` — under ONE shared phase counter. Per-replica empties use `arrive_count=1`; only the shared ring uses `arrive_count=N`.
- **Asymmetric** (one residence-invariant outer operand plus inner-cadence operands in a persistent loop): singleton slot plus barrier pair driven at outer-tile granularity for the outer operand, multibuf rings at inner granularity for the rest.

Pair with the producer schedule: `loop_unroll_factor=N` on short bounded prologues only (N ≤ 4, TMA queue depth; never the main or tile loop), and sandwich shared loads between per-replica loads (`X[0] → Y → X[1]`) by source order alone — no new barriers, which would serialize the overlap away.

Refuse the dual-consumer split (two independent half-tile tasks) when the kernel has zero `async_task` blocks (magnitude rewrite), already declares `replicate>=2` (duplicate state), or lacks `NUM_REPLICAS >= 2`. Other failures: per-replica phase counters drift and hang on the second inner-iter; reversed stride aliases replicas; recurrent register state routed through SMEM rings instead of staying in registers.

## Gate The Third Sink-Writer Task

Splitting the epilogue into a third task (producer → dot-consumer → sink-writer over an fp32 SMEM acc ring with an `acc_full`/`acc_empty` pair) pays only when ALL hold:

- **G1 — Writer-output independence**: tile t's writer output is not a data dependency of tile t+1's consumer (atomic-RMW into shared accumulation violates this via L2 back-pressure).
- **G2 — Writer asynchrony or amortization**: the writer is a non-blocking issue (TMA store, async bulk) or its ALU/SMEM payload is comparable to the WGMMA cost — a short synchronous writer cannot cover the hand-off cost.
- **G3 — Sufficient concurrency**: the persistent grid yields steady-state iterations per CTA after the prologue (≥ 2 waves at the chosen tile), or the kernel is non-persistent with multiple resident CTAs per SM.

Refuse when ANY holds: synchronous atomic-RMW to one hot destination (redirect: two-pass reduction — per-CTA private partial stores plus a separate reduce; intrinsically disjoint destinations are already fine); 1-wave collapse at the WGMMA tile (smaller tile, pingpong, or drop persistent); lightweight synchronous writer or same-kernel feedback into the next dot (inline the epilogue, use pingpong); deep-wave baseline (mandatory — the wave scheduler already amortizes); no steady-state iteration after the prologue; compute-bound baseline with low barrier stall (no headroom); pingpong-replicate geometry already saturating registers (choose pingpong OR sink-writer, not both); already-persistent baseline (the dominant lever — persistent conversion, not the scaffold — is claimed; refuse without further gate checks, while the admit path still requires every gate above).

Geometry: producer 4 warps, dot-consumer 4 warps at `registers=232`, writer 4 warps at `registers=32–40` (asymmetric split is load-bearing; never steal from 232). Prefer `ACC_STAGES=2` when SMEM allows. Fence `async_shared` between the consumer's `local_store(acc)` and its arrive; the writer recomputes tile coordinates identically; all three tasks iterate tiles in identical order; `BLOCK_N >= 64` with `tlx.async_dot`.

Failure modes: `Stall_Barrier` up 10×+ with tensor pipe halved means misapplied gates — revert; stale acc reads mean a missing fence; absent WGMMA means `BLOCK_N < 64` or `tl.dot`; SMEM OOM at 128² with `ACC_STAGES=2` means drop to 1 first.

## Validate The Pattern

Verify correctness across shapes, boundary tiles, replica counts, and repeated launches, at the kernel's standard dtype tolerance. Re-run NCU: tensor pipe must not regress, barrier stall must not balloon (a small hand-off increase is expected), IPC must not regress. Promotion requires a stable end-to-end win.
