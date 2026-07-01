# Cross-Partition "Run-Once" Op Support (Dynamic Persistent / atomic_add) — Proposal

> Status: **DRAFT / iterating**. This is a design doc, not yet implemented.
> Scope: AutoWS (`third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/`).

## 1. Problem

A dynamic (work-stealing) persistent kernel claims its next tile from a global
counter via `tl.atomic_add` instead of a static `tile_id += NUM_SMS`:

```python
tile_id = tl.program_id(0)
while tile_id < num_tiles:
    ... compute tile ...
    tile_id = tl.atomic_add(tile_counter, 1)   # claim next tile
```

Under AutoWS the persistent `scf.while` is **cloned once per partition**
(epilogue / gemm / load). For the static kernel that is correct because
`tile_id += NUM_SMS` is **pure** — every clone computes the same value. The
`atomic_add` is **side-effecting and non-deterministic**, so cloning is wrong:

- The generated TTGIR has `tt.atomic_rmw add` in **all three** partitions
  (`async_task_id = 0`, `1`, `2`).
- Each warp group bumps the counter independently and gets a **different**
  `tile_id`, so the partitions diverge onto different tiles; their producer /
  consumer barriers never match → **runtime deadlock**.

Repro: `test_tutorial09_matmul_tma_dynamic_persistent_while_loop_warp_specialize`
(currently `xfail(run=False)`).

## 2. Requirement

The tile-claiming op must:

1. **Execute once per (CTA, iteration)** — not once per partition.
2. Have its single result **broadcast** to every partition so all clones'
   `scf.condition` agree on the same `tile_id`.

(It is "once per CTA per iteration", not literally once per program — the global
counter already serializes claims across CTAs.)

## 3. Existing CLC broadcast precedent

CLC (`third_party/tlx/language/tlx/dynamic_launch.py`,
`third_party/tlx/tutorials/blackwell_gemm_clc.py`) implements the same
run-once/broadcast shape:

- `clc_create_context(num_consumers=N)` allocates a 1-producer / N-consumer
  broadcast mailbox in SMEM:
  - `clc_responses`  — `num_stages × i128` SMEM slot(s) holding the fetched id.
  - `clc_mbars_full` — "data ready" mbarrier (producer arrives, all consumers wait).
  - `clc_mbars_empty` — "slot free" mbarrier with `arrive_count = num_consumers`
    (every consumer arrives; producer waits before refilling = back-pressure).
- **One** partition (the `"default"`/epilogue) is the producer: `wait(empty)` →
  `expect_bytes(full,16)` → async `try_cancel` (HW arrives `full`).
- **All N** partitions consume: `wait(full)` → read SMEM slot → `fence(async_shared)`
  → `arrive(empty)`.
- Multi-stage (`num_stages>1`) = a circular buffer for run-ahead.

The whole protocol is the standard mbarrier **full/empty + phase-parity**
handshake — the same primitive AutoWS already uses everywhere (SMEM channels,
accumulator handshake, accumCnt phase, TMA `expect_bytes`, `tc_gen5_commit`).

**Takeaway:** the broadcast mechanism is proven; what is missing is making AutoWS
*synthesize* it for a side-effecting control op instead of cloning the op.

## 4. Supported atomic cases (everything else: gracefully reject WS)

To keep v1 simple and correct, AutoWS supports exactly **two** atomic mappings,
classified by how task assignment maps a `tt.atomic_rmw` across partitions, and
**gracefully rejects** everything else:

1. **Single-partition atomic.** The atomic is mapped to exactly one partition
   (its effect and result live in one warp group). No cross-partition concern →
   **pass through unchanged** (identity).

2. **All-partition, loop-carried scalar atomic.** A *scalar* atomic mapped to
   *every* partition whose result is the persistent `scf.while` **loop-carried
   value** — i.e. it must execute exactly **once per while iteration**. This is
   the dynamic-persistent work counter (`tile_id = atomic_add(counter, 1)`).
   Transform: execute once in one owner partition and **broadcast** the result to
   all partitions (run-once + SMEM/REG broadcast, §D2–D5).

**Everything else → reject AutoWS gracefully (do not crash).** Any other atomic
shape — replicated to a *strict subset* of partitions, replicated but *not*
loop-carried, or *non-scalar* / tensor / scatter (data-dependent) — is out of
scope. The compiler must **bail out of warp specialization** for that kernel,
exactly like AutoWS's existing unsupported-pattern skips: e.g. the `scf.if`
else-block bail in `WarpSpecialization.cpp` (`runOnFuncOp`), which on an
unsupported pattern logs `LDBG("Warp specialization does not support else blocks.
Skipping.")`, calls `removeWarpSpecializeAttr(funcOp)`, and `return`s — leaving
the kernel **unspecialized but compilable** (the `numWarps < 4` check does the
same). This is **safe** (no silent corruption, no `llvm_unreachable`/assert).

Why these two and not the general "K>1 → dedup": restricting case 2 to
**every partition + loop-carried + scalar** makes it unambiguously the uniform
run-once-broadcast pattern. Partial replication (`1 < K < numPartitions`) and
non-scalar / data-dependent atomics (e.g. a scatter `atomic_add(base+idx, val)`,
where each partition would update a *different* region and de-duplicating would
drop updates) are precisely the ambiguous/dangerous shapes — so they reject
rather than guess.

This is **correctness-preserving**: the user wrote *one* `atomic_add`; case 2
preserves single-execution by deduping+broadcasting, case 1 is already singular,
and the reject path refuses to silently change semantics for anything else.

## 5. Design decisions

Each is an independent axis. Recommended v1 choice in **bold**.

### D1 — Detection: classify each atomic into {pass-through, transform, reject}

For each `tt.atomic_rmw`, after task assignment, classify by the §4 cases:

1. mapped to **exactly one** partition → **pass through** (identity).
2. mapped to **every** partition, **scalar**, and its result is the persistent
   `scf.while` **loop-carried** value → **transform** (run-once + broadcast).
3. anything else → **reject**: emit a diagnostic and bail out of WS (no crash).

Checks for case 2 (all must hold, else → reject):
- op is `tt.atomic_rmw` with **scalar** address & value (no tensor operand);
- after task propagation it is tagged for **all** partitions (not a strict subset);
- its result is (transitively) the `scf.while` loop-carried value / drives the
  loop condition in every partition.

The trigger is **automatic** from the case-2 predicate above. There is no
explicit opt-in marker or annotation path: once the pass proves the atomic is the
all-partition, scalar, loop-carried persistent counter, the transformation is
correctness-preserving and can be always-on.

**Reject must be graceful**, mirroring existing AutoWS bail-outs (diagnostic +
leave the kernel unspecialized), never `assert`/`llvm_unreachable`. The current
`xfail` dynamic test would, pre-feature, hit this reject rather than the runtime
deadlock.

→ **v1: automatic case-2 predicate; reject (graceful) for everything outside
cases 1–2.** Broadening to tensor / data-dependent atomics is future work.

### D2 — Owner placement

**v1: always place the atomic in the producer partition.** No heuristic, no
runtime election — the owner is unconditionally the TMA-load/producer partition
already identified by `PartitionSchedulingMeta.cpp`. The partition scheduler
creates the load partition as the producer/default warp group; this proposal
consumes that existing assignment rather than adding a second producer-detection
mechanism. This is simplest and near-optimal: the producer has the least per-tile
compute and runs ahead, and it is the **earliest consumer** of the next `tile_id`
(first to start the new tile — it loads), so the claim sits on the partition that
needs the value soonest → ~zero broadcast latency on the critical path (MMA and
epilogue receive it with slack). Works for FA too.

Owner placement is orthogonal to transport (D3) and depth (D4); it composes, it is
not an `or` against them.

Other options (not in v1):
1. **A dedicated "next-tile" warp.** A separate warp/partition whose only job is to
   claim and broadcast the next tile (e.g. an otherwise-idle warp group, if the
   layout exposes one). Keeps the producer off double duty and can run ahead.
   Layout-dependent.
2. **Earliest-tile-finisher (dynamic).** Let whichever partition finishes its tile
   first own the claim. In practice the earliest finisher is almost always the
   producer, so this would rarely differ from the v1 default — and a true dynamic
   form needs a runtime election. Low value given (default).

→ **v1: owner = producer partition, always.**

### D3 — Transport: how is the scalar broadcast?

**v1: (a) SMEM slot + full/empty mbarrier pipeline** — reusing existing AutoWS
infra (mbarriers, phase model, `arrive_count = num_partitions`). The owner stores
the claimed `tile_id` into a 1-element SMEM slot and arrives `full`; every
partition waits `full`, reads the slot, and arrives `empty`. This mirrors the
broadcast protocol (§3), differing only in that a thread arrives `full`
after the atomic instead of a hardware transfer-complete.

Other options (not v1):
- (b) Existing `DataChannelKind::REG` channel — the result is a 1-producer /
  N-consumer `i32`, the register-channel shape; could be lower-new-code if REG
  channels already broadcast a side-effecting scalar 1→N, but that is unconfirmed
  and (a) is the known-good path.
- (c) Named barrier + SMEM — simpler sync but named barriers are scarce (0–15),
  no multi-buffering.
- (d) Warp shuffle — **ruled out**: partitions are different warp groups; `shfl`
  cannot cross them. Must go through SMEM.

→ **v1: (a) SMEM slot + full/empty mbarrier pipeline.**

### D4 — Pipelining depth (tile prefetch) — orthogonal to D2 owner

Depth is independent of *who* owns (D2): any owner can claim 1-ahead or N-ahead.

**v1: default depth = 1 tile** (single-stage: claim → broadcast → all consume,
serialized per iteration). Depth is made tunable via a **new env var
`TRITON_WS_TILE_PREFETCH_DEPTH`** (default `1`):

- Register it in `include/triton/Tools/Sys/GetEnv.hpp` (alongside
  `TRITON_USE_META_WS`, `TRITON_MODULO_SMEM_BUDGET_KB`).
- Read it in the WS pass and **thread it to the memory planner**
  (`doMemoryPlanner` in `WSMemoryPlanner.cpp`), which sizes the tile-id broadcast
  channel — it allocates `tile_prefetch_depth` SMEM slots + full/empty mbarrier
  pairs (D3) and bounds how far the owner may run ahead.
- `depth = 1` is the v1 default (no prefetch); `depth > 1` lets the owner claim
  tile N+1 … N+depth-1 while consumers process N (a multi-stage run-ahead over a
  pipelined response buffer).

**Atomic-specific caveat, required when depth > 1.** Unlike a query/cancel-style
claim (which can be undone), a plain `atomic_add` is a **destructive claim** — the
instant it executes, that tile is owned. So prefetch must guarantee **every
claimed tile is processed**: if the owner buffers tile T but exits before draining
it (e.g. it also reads the terminating sentinel), T is silently dropped — stolen
from other CTAs and never computed → wrong result. `depth = 1` sidesteps this;
`depth > 1` must drain all claimed tiles at loop exit.

→ **v1: default depth = 1; `TRITON_WS_TILE_PREFETCH_DEPTH` threads depth to the
memory planner; depth > 1 requires drain-on-exit.**

### D5 — Placement / timing
The result feeds the **`scf.condition`** (before-region), so there is an
unavoidable **cross-iteration handoff**: owner produces tile N+1 at the end of
iteration N's after-region; every partition must read it before evaluating
iteration N+1's condition. Same forward/backward shape as the accumulator
handshake (`full` arrive at iteration end, `wait` before next condition).

## 6. Correctness obligations (hold for any choice above)

1. **Exactly-once**: the atomic is cloned into one task only.
2. **Same value everywhere**: all N consumers read the one SMEM slot guarded by `full`.
3. **Back-pressure**: all N consumer partitions must release the slot before the
   producer reuses it (so it isn't overwritten early). Realizable as N arrives to
   one empty barrier (`arrive_count = N`) **or** N separate barriers the producer
   waits on — equivalent, given the producer blocks before revisiting the slot
   (true at depth = 1). ⚠️ **Neither is supported by the current auto-WS token
   infra** — see §7 "Token-lowering limitation."
4. **Memory ordering**: `fence(async_shared)` between the SMEM read and `arrive(empty)`.
5. **Consistent loop exit**: all partitions read the same broadcast `tile_id`, so
   they hit the terminating condition together (a terminating-sentinel read must
   skip the empty-arrive so the producer can exit).
6. **Graceful rejection**: an atomic outside cases 1–2 (§4) must cause AutoWS to
   emit a diagnostic and bail (leave the kernel unspecialized), **never** crash /
   `assert` / `llvm_unreachable`. No silent semantic change for unsupported shapes.

## 7. Implementation touch-points

All in machinery already exercised by the while-loop accumCnt work:

- **Task assignment / propagation** (`WSTaskIdPropagate.cpp` / `TaskIdPropagation.cpp`):
  do not assign the side-effecting control op to all tasks; assign it to the
  producer partition supplied by `PartitionSchedulingMeta.cpp`.
- **Channel creation** (`WSCodePartition.cpp` / `CodePartitionUtility.cpp`):
  synthesize the SMEM-slot + full/empty mbarrier 1-producer / N-consumer broadcast
  channel for its result (D3a).
- **Specialization** (`WSSpecialize.cpp`): clone the atomic only into the owner;
  emit producer-fill (`atomic_rmw → store slot → arrive(full)`) and consumer-read
  (`wait(full) → load slot → arrive(empty)`). The owner thread arrives `full`
  after the atomic (no hardware transfer-complete involved).
- **Prefetch depth plumbing** (D4): register `TRITON_WS_TILE_PREFETCH_DEPTH` in
  `include/triton/Tools/Sys/GetEnv.hpp`, read it in `WarpSpecialization.cpp`, and
  pass it to `doMemoryPlanner` (`WSMemoryPlanner.cpp`) so it sizes the broadcast
  channel's SMEM slots / mbarrier pairs and the owner's run-ahead bound.

### Token-lowering limitation (the N-consumer back-pressure gap)

The current token lowering (`WSLowerToken.cpp`, `lowerToken`) does **not** support
the obligation-3 `empty.arrive_count = N` (one empty barrier that N *distinct*
consumer partitions each arrive on):

- It `init_barrier(..., 1)`s both full and empty barriers (the computed
  `bufferFullCount`/`bufferEmptyCount` are commented out); the per-thread count is
  carried on the *arrive*, not the init.
- It models a **1-producer ↔ 1-consumer-partition** channel — it `assert`s a
  single/homogeneous consumer warp count
  (`assert(consumerWarps == 0 || consumerWarps == nWarps)`) and
  `TmemDataChannelPost` asserts a single consumer partition. There is no
  1→N-partition broadcast channel today.

**The N-consumer back-pressure has two equivalent realizations** — and either is
correct **provided the slot is not revisited before all consumers have released**
(guaranteed at depth = 1; at depth > 1 the per-slot block / drain-on-exit, D4,
provides it):
- **(a) Several arrives to one empty barrier** — `arrive_count = N`, each
  consumer partition arrives once; the phase flips after all N.
- **(b) Multiple barriers** — one empty barrier per consumer; the producer waits
  on all N before reusing the slot.

Both are sound under the block-before-revisit condition; the difference is just
barrier count vs. arrive count. Neither is expressible through the current token
lowering (which is 1-producer↔1-consumer-partition, init count 1). This 1→N
back-pressure (multi-arrive *or* multi-barrier) is **general AutoWS
infrastructure** — it should be added to `WSLowerToken` later as a reusable
capability rather than special-cased for this feature. For v1, synthesize the
barriers directly, bypassing the standard token framework: the
`alloc_barriers` / `init_barrier` primitive already supports
`arrive_count = N`, as the broadcast protocol in §3 does. A future optimization
can then fold the direct form back into token lowering as **multi-channel →
single-barrier coalescing**: N consumer channels share *one* barrier via N
arrives (`arrive_count = N`) instead of allocating one barrier per channel. That
future work would reduce barrier count / SMEM use and is the natural
generalization of the existing `BarrierFusion` (see `BarrierFusion.md`).

## 8. Recommended v1 (concrete)

Scope (§4): support only case 1 (single-partition → pass-through) and case 2
(all-partition + scalar + loop-carried → transform); **gracefully reject** all
other atomics. Within case 2, compose these independent axes:
- **D1**: automatic case-2 predicate (scalar `atomic_rmw`, all partitions,
  loop-carried result); reject path is a clean diagnostic, never a crash.
- **D2**: owner = **producer partition**, always (next-tile warp / earliest-finisher
  are non-v1 options).
- **D3**: SMEM slot + full/empty mbarrier pipeline.
- **D4**: depth = 1 tile by default; `TRITON_WS_TILE_PREFETCH_DEPTH` (new env var)
  threads the depth to the memory planner (depth > 1 needs drain-on-exit).
- **D5 + obligations 1–6**: cross-iteration `full`/`empty` handoff sized
  `arrive_count = num_partitions`, plus graceful reject for unsupported shapes.

This is the auto-WS structural analogue of the SMEM full/empty broadcast protocol
(§3), with the owner fixed to the producer.

## 9. Resolved decisions

1. **D1 trigger**: automatic inference only. There is no explicit opt-in marker.
2. **D2 owner**: use the producer/load partition already assigned by
   `PartitionSchedulingMeta.cpp`; no new owner-detection heuristic is needed.
3. **D3 lowering**: synthesize the SMEM slot plus full/empty mbarrier broadcast
   directly in AutoWS for v1. A reusable 1→N token-lowering abstraction can be
   added later, but it is not required to resolve the dynamic persistent
   correctness bug.

## 10. Test plan

- Flip `test_tutorial09_matmul_tma_dynamic_persistent_while_loop_warp_specialize`
  from `xfail(run=False)` to a real run; assert correctness for
  `EPILOGUE_SUBTILE ∈ {1,2,4}` and many-tiles-per-CTA (small NUM_SMS) to exercise
  repeated broadcast.
- Verify the generated TTGIR has the atomic in **one** partition only and an
  SMEM full/empty broadcast channel feeding all `scf.while` conditions.
- Regression: static persistent while + the full tutorial09 / FA / addmm / LIT WS
  suites unchanged.
- **Graceful reject (case 3)**: kernels with an unsupported atomic shape —
  non-scalar/scatter atomic, an atomic replicated to a strict subset of
  partitions, or a replicated-but-not-loop-carried atomic — must produce a clean
  diagnostic and compile **unspecialized** (no crash/assert). Add a test that
  compiles such a kernel and checks it does not abort and is not warp-specialized.
