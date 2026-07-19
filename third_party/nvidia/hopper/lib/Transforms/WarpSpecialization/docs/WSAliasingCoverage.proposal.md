# WS Barrier Verifier — Static Detection of mbarrier Deadlocks and Cross-Partition Races in AutoWS

Status: proposal / design (analysis only; no implementation yet).
Authored with Claude.

This is the executable coverage verifier that the `barrier-visualization` skill
(`.claude/skills/barrier-visualization/SKILL.md`) refers to as "a future
`triton-opt` pass / `doCodePartition` invariant that models physical layout,
per-op write extent, and loop cadence." It builds on the WSBuffer / WSBarrier /
reuse-group model from the "Buffer Reuse, IR Design, and Verifier" design doc
(Google Doc `1ySM17-WBtJthF3nZULfb4lBajNtYrFvwleRZvEj4qGg`, tabs t.0 and
t.elg8342709es).

---

## 0. Goal 1 scope (what we build first)

- **Run the verifier after the pipeline expander** (`add_pipeline` ==
  `--tritongpu-pipeline`, whose internal expander step is
  `SoftwarePipeliner ... ExpandLoops`). For Blackwell (`cuda:100`) the backend
  order (`third_party/nvidia/backend/compiler.py`) is:
  `add_partition_scheduling_meta` -> `add_hopper_warpspec`
  (`doCodePartitionPost` + `doTokenLowering`, barriers materialized) ->
  **`add_pipeline`** (the expander). The verifier runs immediately after
  `add_pipeline`.
- **AutoWS path only** (not TLX) for goal 1.
- **Assume barrier -> reuse group is known:** every barrier endpoint is taken to
  carry the `buffer.id` (reuse group) it guards. The mechanism for recovering
  that mapping (§6) is deferred; goal 1 treats it as given.

### Reference IR (the HSTU DP=2 fwd deadlock)

- **Prior-code-partitioning IR** (post-memory-planner: `async_task_id` +
  `buffer.id`, no barriers — the input to `doCodePartition`):
  **[P2426552429](https://www.internalfb.com/intern/paste/P2426552429/)**
- **Post-pipeline-expander IR** (`SoftwarePipeliner ... ExpandLoops`:
  `ttg.warp_specialize` + materialized barriers; the deadlocking DP=2 module):
  **[P2426552821](https://www.internalfb.com/intern/paste/P2426552821/)**

Both were captured from an end-to-end compile of the HSTU self-attn forward at
DP=2 (`HSTU_SELF_AUTOWS=1 HSTU_SELF_DP=2 HSTU_SELF_AUTOWS_WARPS=4
HSTU_SELF_PIN=1 TRITON_USE_META_WS=1`, config `BLOCK_M=256/BLOCK_N=128/
num_stages=1/num_warps=4`) with `MLIR_ENABLE_DUMP=_hstu_attn_fwd`, extracting the
respective pass-dump sections. The launch deadlocks at runtime, confirming the
post-expander module is the buggy one.

> Note on branch drift: the frozen lit
> `test/Hopper/WarpSpecialization/ws_code_partition_dp_idle_acc_init_deadlock.mlir`
> pins an *older* HSTU DP=2 deadlock (the idle-epilogue acc-init cycle). On the
> current branch that specific channel is already removed
> (`removeRedundantTmemZeroStores` + the DP fix `e394874fd` + PSM fix
> `fdd5447c6`), so P2426552821 above has **0 backward waits in task 0** and
> deadlocks for a *different* reason (backward reuse waits now live in the gemm
> and load partitions). Both are valid verifier targets: the frozen lit is the
> clean regression, and P2426552821 is the live case. §5 walks the frozen-lit
> cycle in full because its structure is documented and minimal.

---

## 1. Motivation

AutoWS lowering (`WSCodePartition.cpp`) turns cross-partition SSA data
dependencies into physical mbarrier handshakes: `ttng.init_barrier` /
`wait_barrier` / `arrive_barrier` / `tc_gen5_commit`, the TMA-implicit arrive on
`async_tma_copy_global_to_local`, and (pre-lowering) the NVWS token ops
`nvws.producer_acquire` / `producer_commit` / `consumer_wait` /
`consumer_release`. Today the correctness of these handshakes is validated only
by (a) running on hardware and observing a hang or wrong numerics, and (b) the
manual `barrier-visualization` skill.

The known-bug catalog `.llms/rules/partition-scheduler-bugs.md` records 13
shipped WS defects; at least six are exactly the classes this pass must catch
statically at the code-partition IR level:

- #8 cross-stage SMEM downgraded to depth 1 -> runtime **mbarrier deadlock**.
- #9 persistent bwd staging cross-tile **WAR race** (degenerate self-edge guard).
- #11 fwd-persistent **deadlock** from a pinned phase (slot/phase collapse).
- FA-fwd-persistent **column-packed TMEM aliasing race**
  (`isWholeAllocationOverwriteReuseOwner`,
  `ws_code_partition_tmem_packed_reuse_backward.mlir`).
- static-persistent-while GEMM **redundant cross-partition acc init hang**
  (`removeRedundantTmemZeroStores`).
- HSTU DP=2 idle-epilogue **acc-init deadlock**
  (`ws_code_partition_dp_idle_acc_init_deadlock.mlir`, currently xfail).

Each ships as a hand-written FileCheck test pinning *one* IR signature after the
fix; none is a general invariant, so a new kernel shape that recreates the class
silently regresses (#9 and #11 explicitly escaped). The verifier turns those
per-fix signatures into one executable check that runs in CI over the
post-expander IR of every kernel and lit test.

**Central constraint (do not skip):** mbarriers are phase-based, not counting
semaphores. A raw arrive/wait count mismatch is a *candidate*, never a
conclusion. The verifier must use the phase model of §3 or it produces false
positives on the very common benign "redundant acquire polls the prior
iteration's flip" pattern.

### Framing (from the Buffer-Reuse design doc)

The verifier's core object is the **WSBuffer reuse group** (one `buffer.id`), and
its core property is **no live-range overlap on shared partitions**. Every
catalog bug is an instance of that. Vocabulary:

- **WSBuffer** — an alloc (lifted to outermost scope) that is a memory-space
  placeholder for cross-partition communication. Single producer for SMEM;
  single producer for TMEM **except** the gen5 accumulator (operand D), which is
  the one sanctioned multi-producer/multi-version WSBuffer (versions tracked by
  tokens / memory SSA).
- **Channel** — a per-`(producer, consumer(s))` record (not IR). Single WSBuffer
  per channel; multiple channels may map to one WSBuffer.
- **WSBarrier** — a barrier associated with a reuse group (`buffer.id`),
  optionally naming its member WSBuffers. Target IR form (§6); the existing
  `constraints = {WSBarrier = {...}}` dict is the bootstrap form.

Verdicts follow the doc's 3-state model: **verifiably safe / verifiably unsafe /
indeterminate** (the last for anything the model cannot discharge, e.g.
`scf.if`-nested channels or runtime trip counts).

---

## 2. IR model

The verifier consumes the post-`add_pipeline` module (see P2426552821), in which:

- partitions are `ttg.warp_specialize` regions (`ttg.partition.types =
  ["epilogue","gemm","load","computation","computation"]`); ops still carry
  `async_task_id = array<i32: N>`;
- SMEM/TMEM buffers are `ttg.local_alloc` / `ttng.tmem_alloc` with `buffer.id`,
  `buffer.copy`, `buffer.offset` (packed siblings), `buffer.tmaStaging`;
- mbarrier allocs are `ttg.local_alloc` stamped `ttg.ws_generated_barrier`
  (`kWarpSpecializeGeneratedBarrierAttrName`), sized `distance == buffer.copy`;
- barrier endpoints carry `constraints = {WSBarrier = {dstTask, channelGraph,
  direction, parentId, minRegionId, maxRegionId}}` (`WSBarrierAnalysis.h`,
  `WSBarrierAttr`). These annotations survive token lowering + the expander
  intact (verified on P2426552821: 28 barriers carry the constraint, 6 are
  `direction="backward"`).

### 2.1 Cross-partition Barrier Dependency Graph (BDG)

Nodes: one per barrier *site* = `(partition, program-position, role)`, role in
{wait, arrive, commit}. Covers `wait_barrier`, `arrive_barrier`,
`tc_gen5_commit`, the implicit TMA arrive, and (pre-token-lowering) the four
`nvws.*` ops. `partition` = the single element of `getAsyncTaskIds(op)`.
`program-position` = `(parentId, ordered-region index, index-in-block)`, reusing
the ordered-region machinery already in `WSBarrierAnalysis.h`
(`buildWSBarrierOpRegionInfo`, `getNearestWSBarrierParent`,
`getOrderedRegionBlockOffsets`).

Each node is annotated with:

- **barrier alloc identity** — the `ws_generated_barrier` `local_alloc`, traced
  through `memdesc_index` / block args / loop iter_args (§6 alias trace).
- **slot + phase parity** — from the op operands: `wait_barrier(%barView,
  %phase)`, `%barView = memdesc_index %barAlloc[%bufIdx]`, with the *symbolic*
  form `bufIdx = accumCnt % numBuffers`, `phaseParity = (accumCnt / numBuffers)
  & 1` (from `getBufferIdxAndPhase`, `CodePartitionUtility.h`). Symbolic, not
  constant-folded, so cadence holds for all trip counts.
- **direction** — `forward`/`backward` from `isWSBarrierBackwardEndpoint`
  (token type for NVWS; `constraints.WSBarrier.direction` for TTNG endpoints).
- **dstTask + channelGraph** — from `WSBarrierAttr::parse`.
- **guarded buffer** — `buffer.id` / `buffer.offset` (given in goal 1; §6).

Edges:

- **satisfy edges** (cross-partition): wait W -> every arrive/commit A on the
  same barrier alloc + same slot producing W's parity (a *may-satisfy*
  relation; §3 turns it into must-satisfy-first-execution).
- **program-order edges** (intra-partition): W -> next barrier site in the same
  partition in ordered-region order.

### 2.2 Physical Buffer Model (PBM)

Per `buffer.id`:

- **owner** = alloc with no `buffer.offset`; **packed siblings** =
  `buffer.offset > 0` sharing that `buffer.id`.
- **copies** = `buffer.copy`; **memory space** = SMEM/TMEM; **cross-stage** flag
  = consumers span >1 `loop.stage` (the #8 signal).
- **writers**: op, partition, **write extent** (a `tc_gen5_mma` with
  `useAcc=false`/`useC=false` overwrites the *entire allocation* — all columns;
  the full-overwrite owner), `loop.stage`/`loop.cluster`/enclosing loop.
- **readers**: op, partition, `loop.stage`, and **iteration offset from its
  writer** (the reuse distance; see reconstruction below).
- **reuse-in-place / versions**: single-buffered (`buffer.copy==1`) slot written
  by X then re-written by Y for a different logical value (QK result reused as P
  operand); operand-D versions tracked by tokens. From `ReuseGroup` /
  `ReuseConfig` / `TmemDataChannelPost`.
- **requiredDepth** (the under-buffering key): the live-range span in iterations
  over the buffer's producers/consumers, `= maxStage - minStage + 1` (bug #8's
  `getSmemCrossStageDepth` floor). A reuse group is **under-buffered** when
  `requiredDepth > buffer.copy` and no coarse backward WAR edge covers the whole
  multi-iteration span (see §4.4).

Populated by walking allocs and their
`memdesc_index`/`memdesc_reinterpret`/`memdesc_subview`/`memdesc_trans` chains,
grouped by `buffer.id`.

**Reconstructing the iteration difference after the expander.** `loop.stage` does
**not** survive `ExpandLoops` (verified: 0 occurrences post-expander; only the
partition-level `ttg.partition.stages` on the `warp_specialize` op survives). So
`requiredDepth` must be recovered from the physical structure the expander
produced — which is the *right* signal, because it reflects what the expander
actually did (extra issues it introduced), not the pre-expander intent:

1. **Slot-index / phase stagger** — producer writes `memdesc_index %buf[idx_p]`,
   consumer reads `[idx_c]`, both `= f(accumCnt)`; the barrier's producing arrive
   and consuming wait use `phase = (accumCnt / numBuffers) & 1`. The accumCnt
   offset between producer and consumer = the iteration difference. When
   single-buffered (`numBuffers==1`, `idx == %c0_i32`), the difference lives
   entirely in the phase expression + the loop-carried token chain.
2. **Prologue peel depth** — the expander peels `span` iterations into a
   prologue; the count of peeled producer copies before the steady-state loop =
   `span`.
3. **Loop-carried version count** — a value produced in iter i and consumed in
   iter i+k appears as `k` loop-carried iter_args / memory tokens.

If these are unambiguous the difference is deducible post-expander; if the
expander constant-folded them ambiguously the verdict is **indeterminate** and
the check falls back to the after-token-lowering run (where `loop.stage` is
explicit).

---

## 3. Deadlock analysis

### 3.1 Phase model (the guardrail)

- `wait_barrier(bar, phase)` spins until parity == phase, then returns; it
  **consumes nothing**. Any number of waits are satisfied by one flip.
- On a freshly `init_barrier`'d empty/reuse barrier, the producer's **first**
  `wait_barrier` (a `producer_acquire`, phase pre-inverted) passes with **no
  arrive** — the buffer starts free. So a backward/EMPTY first-acquire is
  **non-blocking on first execution** and must not create a wait-for edge for
  the entry-deadlock check.
- A FULL wait (parity produced only by a forward arrive/commit) blocks until a
  real arrive flips the parity.

Satisfy edges are filtered accordingly: FULL waits contribute a real wait-for
edge on first execution; backward/EMPTY first-acquires are free on first
execution (they only wait on iteration N-1's release from N>=2). **Pre-arm
detection:** a backward acquire is free only if its EMPTY barrier is pre-armed
(init-time arrive / phase pre-inverted so its phase-1 wait passes) or re-armed
loop-locally before it runs; otherwise it genuinely blocks. This is the filter
that avoids the false positive of treating phase=1 empty waits as blocking.

### 3.2 The three genuine deadlock conditions

For each FULL `wait_barrier(bar, phase)` W:

1. **No producer** — no arrive/commit anywhere drives `bar`'s slot to W's parity
   -> deadlock. (Covers a dropped producer-side arrive, e.g. #9's degenerate
   same-task guard elided by `WSLowerToken`.)
2. **Parity-cadence mismatch** — count phase flips per iteration vs the parities
   the waits require. Two waits requiring the **same** parity with one flip per
   iteration starve; two waits requiring **opposite** parities (`x` vs `NOT x`)
   with one flip per iteration are fine (the benign redundant-acquire signature
   -> classify *redundant, harmless*). This is #11: collapsing the staggered
   `accumCnt` to a constant `theIdx` pins the phase (`wait_barrier ..., %c1_i32`)
   so it never flips across persistent tiles.
3. **Cross-partition cycle** — in the combined graph (satisfy + program-order),
   a cycle where the arrive that must satisfy W is transitively ordered *after*
   W across partitions. This is the HSTU idle-acc-init deadlock (§5).

Run Tarjan SCC over the **first-execution subgraph** (backward first-acquires
excluded per §3.1); a nontrivial SCC containing >=1 cross-partition satisfy edge
and >=1 FULL wait is a deadlock cycle. Excluding the free first-acquires is what
prevents persistent-loop back-edges from spuriously closing a cycle.

---

## 4. Race analysis

**The BDG alone is not sufficient for races.** The BDG models *synchronization*
(what ordering exists); a race is about *data* (a write and a later
cross-partition read of the same memory with no covering order). Race detection
is therefore a **join of two structures, keyed by `buffer.id`**:

- the **BDG** (§2.1) — the *supply* of ordering (existing forward/backward
  barrier edges, each tagged with the `buffer.id` it guards);
- the **Access-Order Graph (AOG)** — one per reuse group, the *demand* for
  ordering, derived on demand from the PBM facts for one `buffer.id`.

The race check overlays the two: **every required AOG edge must be covered by a
synchronization path in the BDG.** An uncovered / mis-cadenced / self-edge-only
required edge is a race; a cycle in (required-order ∪ available-sync) within a
group is a deadlock. So there are two graphs sharing the `buffer.id` key — the
global **BDG** (SCC → deadlock, §3) and the small per-group **AOG** (edge-coverage
→ race, this section). The PBM (§2.2) stays the raw fact table; the AOG is
materialized from it per group when the race check runs (cheap, scoped, dumpable
— §7). §4.1–4.3 are the specific required-edge patterns; §4.4 is the
reuse-distance (under-buffering) rule that reads the iteration difference off the
WAR edge span.

**AOG data structure (iteration is first-class):**

```cpp
struct AOGNode {                 // one memory-access instance on the buffer
  Operation *op;
  AccessKind kind;               // Write | Read
  int        partition;
  int        bufferId, bufferOffset;   // which sibling/slot in the group
  AffineExpr slot;               // physical slot = accumCnt % numBuffers
  AffineExpr iteration;          // ITERATION coordinate (accumCnt), symbolic
  Extent     writeExtent;        // writes only (full-overwrite owner => all cols)
  std::optional<int> loopStage;  // present pre-expander; reconstructed post-expander
};

struct AOGEdge {
  EdgeKind kind;                 // ProgramOrder | RAW | WAR
  AOGNode *src, *dst;
  int      reuseDistance;        // ITERATIONS between the two = dst.iteration - src.iteration
                                 //   (WAR: iters between prior read and the reusing write)
  Cadence  requiredCadence;
  bool     coveredByBDG;         // + the covering BDG edge/path
};
```

Iteration appears twice: as the node coordinate `iteration` (`= accumCnt`) and,
derived from it, as the edge attribute `reuseDistance`. The under-buffering test
(§4.4) is `WAR.reuseDistance + 1 (= requiredDepth) > buffer.copy` with no covering
BDG path. Post-expander, `iteration` is the reconstructed accumCnt/prologue/
version coordinate (§2.2); pre-expander it can also be read from `loopStage`.

### 4.1 Reuse-pair coverage rule (the doc's two named edges)

For each `buffer.id` and each ordered `(writer Wr, reader Rd)` where Rd reads a
value that Wr's next write (next tile/stage/iteration) clobbers and
`partition(Rd) != partition(next-writer)`, require the two edges the design doc
names for a 2-WSBuffer reuse group:

1. a data dependency from the **early WSBuffer's consumer** to the **late
   WSBuffer's producer**, and
2. an **additional sync from the late WSBuffer's commit/release to the early
   WSBuffer's producer_acquire**,

covering the write **extent** at the correct **cadence** (same loop level ->
wait right before the writer; reader at an outer level -> wait before the inner
loop, using the outer-loop phase). Flag a **race** when the backward edge is
**missing**, a **degenerate self-edge** (`dstTask == srcTask`, elided to a no-op
— the #9 signature), or **wrong cadence**. The working FA kernel carries a real
`direction="backward"` edge + backward barrier operand on the PV MMA; #2 is the
QK->P in-place TMEM reuse (`buffer.id=9`) that has only a self-edge, with the
writer at `loop.stage=0` and the reader MMA at `loop.stage=1`.

### 4.2 Column-packed TMEM coverage table

For each `buffer.id` with packed members, when the owner's producer is a
`tc_gen5_mma useAcc=false` (full-allocation zeroing write), emit a row per
aliased buffer the write touches; require a backward acquire on each sibling
whose consumer is in a **different partition**, before the overwrite, at the
sibling's cadence:

```
Physical buffer.id = 8 (owner: QK accumulator, 128 cols)
  Writer: tc_gen5_mma useAcc=false (task 1, inner loop)  extent: cols 0-127
    cols 0-63  QK result  consumer task 5 (same-partition) ✓ program order
    col  64    alpha      consumer task 0 (inner cadence)   ✓ per-iter backward barrier
    col  65    m_ij       consumer task 0 (outer cadence)   ✗ MISSING backward edge  → RACE
    col  66    l_i0       consumer task 0 (outer cadence)   ✗ MISSING backward edge  → RACE
```

Any ✗ is a race, invisible to arrive/wait balancing (each sibling token is
individually balanced; the defect is an *absent* edge across aliased columns).
Compiler guard: `isWholeAllocationOverwriteReuseOwner`
(`CodePartitionUtility.h`). Always print the full table (enumerate absences).

### 4.3 Redundant cross-partition accumulator init

For each operand-D accumulator whose `tc_gen5_mma` has `useAccumulator=false`
first iter (self-zeroes): if an explicit `ttng.tmem_store <0>` into the same
accumulator survives in a **different partition** and the enclosing structure is
persistent (`scf.for`/`scf.while`), flag a cross-tile race/hang — the store is a
redundant channel that should not exist. Fix: `removeRedundantTmemZeroStores`
(must recognize `scf::WhileOp` and forward the store's dep token). Structural
tell: ~4 accumulator barriers / 2 commits instead of ~2 / 1. Emit a row per
operand-D accumulator, even when clean.

### 4.4 Under-buffering / cross-stage depth (the reuse-distance rule)

This is the executable form of the access-order-graph WAR check (§4 intro) and
catches bug #8 (cross-stage SMEM downgraded to depth 1) and the QK↔P
single-buffer reuse hit at a >1-stage producer→consumer gap. For each reuse
group:

```
requiredDepth = maxStage - minStage + 1        (= reuse-distance + 1; §2.2)
under-buffered ⟺  requiredDepth > buffer.copy
             AND  no coarse backward WAR edge covers the whole multi-iteration
                  span at the matching cadence
```

In access-order-graph terms: a required **WAR edge whose reuse distance exceeds
the number of physical slots** (`buffer.copy`) cannot be satisfied — the slot is
physically reused before the prior reader has run — so the edge is *uncoverable*
by any BDG path → race/hang. Example: qk writer at stage 0, P consumer (feeding
PV) at stage 2, in-place `(qk, p)` reuse → `requiredDepth = 2-0+1 = 3` but
`buffer.copy = 1` → the qk MMA of iteration i+1/i+2 clobbers P before the
stage-delayed PV consumer reads it. This is exactly bug #8's
`getSmemCrossStageDepth` floor evaluated as a check.

Because `loop.stage` is gone post-expander (§2.2), the reuse distance is read off
the WAR edge span reconstructed from slot/phase stagger, prologue depth, or
loop-carried version count. This makes the check robust to **expander-introduced**
under-buffering (a mis-peel / mis-rotation shows up as an actual span mismatch,
not just intent); when the reconstruction is ambiguous the verdict is
**indeterminate** and the after-token-lowering run (explicit `loop.stage`) is
authoritative. Emit a per-group row: `buffer.id`, `requiredDepth`, `buffer.copy`,
covering-edge present?, verdict.

---

## 5. Worked example — the HSTU idle-acc-init deadlock

This walks the frozen lit
`ws_code_partition_dp_idle_acc_init_deadlock.mlir` (HSTU fwd, DP=2), which is the
minimal documented form of the cross-partition cycle (§3.2 condition 3). The
live P2426552821 has the same class in the gemm/load partitions; the frozen lit
is used here because its structure is minimal.

Reuse group = the O accumulators `acc_0` (`tmem_alloc buffer.copy=1 buffer.id=6`)
and `acc_1` (`buffer.id=7`); trace `buffer.id=6`. `buffer.copy=1` -> the pipeline
expander does not multibuffer or re-phase it (it only peels the multibuffered
q/k/v), so the post-expander handshake equals the post-code-partition one.

Real emitted barriers (from running code partition on the lit; labels E=epilogue
task 0, G=gemm task 1, number = program order):

- **E1** `ttng.wait_barrier %99, %101 {direction="backward", dstTask=1,
  async_task_id=0}` — backward EMPTY acquire, `%101 = NOT(accCnt & 1)` = 1 on
  first execution. Its barrier `%25` is `init_barrier ..., 1` (phase 0, **not
  pre-armed**).
- **E2** `ttng.tmem_store %90, %91[] ... tmem.start=2` (zero-init) +
  `nvws.producer_commit %35 {dstTask=1}` (forward init-ready arrive).
- **G1** forward wait on `acc0_INIT_FULL` (before first use of acc).
- **G2** `ttng.tc_gen5_mma ... %216[], %arg76, ...` (PV MMA into acc,
  `useAcc=%arg76`=false first iter -> self-zeroes).
- **G3** post-loop `ttng.tc_gen5_commit` (the only arrive on `%25`/`%21`, the acc
  EMPTY reuse barriers).

Edges (first-execution subgraph, after phase filter):

```
E1 ⇽ G3   (backward EMPTY wait, NOT pre-armed -> real wait-for; releaser is post-loop)
G1 ⇽ E2   (init FULL wait released by epilogue init store)
task-0 program order:  E1 → E2
task-1 program order:  G1 → G2 → G3
```

Cycle: `E1 ⇽ G3` -> G3 needs (task-1 order) G1,G2 to run -> `G1 ⇽ E2` -> E2 needs
(task-0 order) E1 to pass -> back to E1. Nontrivial SCC `{E1, E2, G1, G3}` with
cross-partition satisfy edges -> **deadlock at kernel entry** (both partitions
spin), matching the lit comment.

Why it is real, not a false positive: E1's EMPTY barrier is not pre-armed and has
no in-loop re-arm, so the phase filter (§3.1) keeps the `E1 ⇽ G3` edge. Why
FA-fwd DP=2 (same single-buffered acc layout) is **not** flagged: FA has an
in-loop acc consumer (alpha rescale) that re-arms the EMPTY barrier every
iteration, so its reuse release is loop-local; there is no long-range backward
edge spanning the whole loop -> no cross-partition cycle. The distinction is
purely graph structure + pre-arm/cadence, no kernel-specific rule.

Diagnostic:

```
error: WS deadlock cycle on reuse group buffer.id=6 (O accumulator, TMEM 128x128xf32):
  [task0 epilogue] wait_barrier acc0_EMPTY (backward), pre-loop   (not pre-armed)
    ⇽ released only by [task1 gemm] tc_gen5_commit acc0_EMPTY, post-loop
    which is program-ordered after
  [task1 gemm] wait_barrier acc0_INIT_FULL, pre-loop
    ⇽ released by [task0 epilogue] tmem_store<0>, program-ordered after the backward wait.
  Fix: idle producer (task0) must PRE-ARM acc0_EMPTY, not backward-wait it.
```

Cheap pre-filter (linear scan, matches the lit `CHECK-NOT`): a
`direction="backward"` wait in a partition idle inside the loop it guards, whose
satisfying arrive lies outside/after that loop.

---

## 6. Barrier -> buffer recovery (deferred for goal 1)

The hard part in general: a barrier op does not name the buffer it guards. Goal 1
assumes the mapping is given. The durable answer (Buffer-Reuse doc IR Option 1)
is a **first-class `WSBarrier` op** keyed by reuse group (`buffer.id`),
optionally naming its member WSBuffers — chosen over annotations because
attributes can be lost across transformations, and over Aref (which makes direct
barrier handling infeasible). The existing `constraints.WSBarrier` dict is the
bootstrap form (it survives token lowering + the expander in practice — verified
on P2426552821).

Recovery sources when the mapping is not given (also the TLX front-end):

1. MMA trailing barrier operand: `tc_gen5_mma ..., barriers(%bar)` names barrier
   + accumulator (-> `buffer.id`); `tc_gen5_commit %bar` after an MMA.
2. TMA copy / `barrier_expect`: `async_tma_copy_global_to_local ... %dstBuf,
   %bar` + `barrier_expect %bar, <bytes>` name dest buffer + barrier (byte count
   cross-checks merged sizing).
3. wait/arrive adjacency: nearest guarded `local_store`/`local_load`/`tmem_*`
   (arrive scans back, wait scans forward — `buildBarrierToMemoryOpMap`).
4. `buffer.id` via the barrier's `CommChannel` at emission
   (`producerBarrier`/`consumerBarriers` -> `Channel::getAllocOp()`); lost in
   final IR, so reconstructed from 1-3.

Recommendation: stamp `guardedBufferId`/`guardedBufferOffset` into
`constraints.WSBarrier` at emission (where `dstTask`/`channelGraph` are already
injected), and keep 1-3 as fallback. Trace the barrier SSA alias chain
(`local_alloc` -> block arg -> loop iter_arg -> `memdesc_index`) so all uses of
one logical mbarrier map to one node group. On failure, mark `buffer=unknown`
and emit **indeterminate**, never silently drop.

---

## 7. Pass placement & API

- **Placement:** after `add_pipeline` (goal 1). The doc's incremental plan also
  wants a run after token lowering and again after SWP to isolate
  SWP-introduced issues.
- **Entry points:**
  - `--nvgpu-verify-ws-barriers` — standalone `triton-opt` verify pass (mirrors
    `NVGPUTestWSCodePartitionPass`). Options: `emit-coverage-table={0,1}`,
    `severity={remark,warning,error}`, `check={deadlock,race,all}`.
  - in-tree invariant hook from `doCodePartitionPost`/post-pipeline under
    `TRITON_VERIFY_WS_BARRIERS`, reusing `WSBarrierAnalysis.h` +
    `CodePartitionUtility.h` structures.
  - `--dump-reuse-group=<buffer.id>` — focused per-reuse-group report (extends the
    existing `dumpTmemBufferLiveness` / `dumpSmemBufferLiveness` /
    `dumpCombinedGraph` helpers): owner + packed siblings, `buffer.copy`, every
    writer (op, partition, extent, slot/phase expr, prologue instances), every
    reader (op, partition, slot/phase, iteration offset from its writer), the BDG
    edges touching the buffer, `requiredDepth` vs `buffer.copy`, the required
    access-order edges and whether each is covered, and the verdict. This is the
    barrier-viz skill's Section 4 + coverage table made executable and scoped to
    one group — e.g. dump the `(qk, p)` group and read off "reuse distance 3 vs
    copy 1".
- **Inputs:** post-expander TTGIR (barriers, `async_task_id`, `buffer.*`
  present). Handles both NVWS token IR and lowered TTNG barrier IR
  (`isWSBarrierEndpoint`).
- **Outputs:** module unchanged (verify-only). Diagnostics via
  `emitError`/`emitWarning`/`emitRemark` on the offending op (works with
  FileCheck and `-verify-diagnostics`). Verdicts: safe / unsafe / indeterminate.

---

## 8. Diagnostics

- Deadlock (no producer): `error: WS deadlock: FULL wait_barrier on buffer.id=9
  has no producing arrive for phase parity 0`.
- Deadlock (cadence): `error: WS deadlock: cross-partition staging barrier
  (dstTask=2) has loop-invariant phase but its slot is reused across the
  persistent loop; parity never flips` (#11).
- Deadlock (cycle): the §5 message.
- Race (missing/self-edge): `error: WS race: buffer.id=9 written by tc_gen5_mma
  (task1, stage0), read by task4 (stage1); backward release is a degenerate
  self-edge (dstTask==srcTask)` (#2/#9).
- Under-buffering (§4.4): `error: WS under-buffered reuse group buffer.id=8
  (qk/p): requiredDepth=3 (reuse distance 2) > buffer.copy=1; WAR edge
  uncoverable` (#8).
- Column-packed: the §4.2 table + one summarizing error per ✗.
- Redundant acc init: the §4.3 row.
- Benign: `remark: redundant acquire (harmless over-synchronization)` for
  opposite-parity double-acquires — never an error.

---

## 9. Complexity, soundness, limitations

- **Complexity:** BDG build O(#barriers × alias depth); satisfy matching
  near-linear after bucketing by barrier alloc; SCC O(V+E); PBM O(#alloc users);
  the per-reuse-group access-order graph is O(#accesses in that group) and built
  on demand. Linear-ish in module size; fast enough for CI.
- **Soundness vs completeness:** biased toward soundness for the enumerated
  classes (few false negatives on #2/#3/#4/#8/#9/#11) and toward avoiding false
  positives (the §3.1 first-execution filter + opposite-parity classifier).
  Unknown-buffer / undischargeable cases are **indeterminate**, not pass/fail.
- **Limitations (indeterminate, not guessed):** `scf.if`-nested channels (invalid
  ordered-region metadata -> V1 fallback); data-dependent trip counts / dynamic
  persistence (CLC); partial-overwrite sub-column MMA extents (approximated
  conservatively); post-`memdesc_reinterpret` swizzle-changing aliasing enforced
  by token rather than static liveness (trust the emitted token, check
  presence/cadence).
- **Post-expander stage loss (the cross-stage/under-buffering check):**
  `loop.stage` does not survive `ExpandLoops` (verified), so `requiredDepth`
  (§2.2, §4.4) is reconstructed from the expander-realized structure (slot/phase
  stagger, prologue peel depth, loop-carried version count). This is the *right*
  place to catch expander-**introduced** under-buffering, but the reconstruction
  can be ambiguous (e.g. constant-folded indices) -> **indeterminate**; the
  after-token-lowering run, where `loop.stage` is explicit, is authoritative for
  this class. This is a concrete reason the verifier runs at both points (after
  token lowering AND after SWP).

---

## 10. Test strategy

Lit tests feed pre-code-partition IR, run code partitioning (+ pipeline for the
post-expander variant), then the verifier, and FileCheck the diagnostic:

```
// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=1 post-channel-creation=1" \
// RUN:   --nvgpu-verify-ws-barriers="check=all severity=error" \
// RUN:   -verify-diagnostics 2>&1 | FileCheck %s
```

Matrix (each cites its catalog entry):

1. Deadlock cycle — convert `ws_code_partition_dp_idle_acc_init_deadlock.mlir`
   (xfail) into `expected-error {{WS deadlock cycle}}`; fixed compiler -> silent
   (positive test asserts no diagnostic).
2. Column-packed race — `ws_code_partition_tmem_packed_reuse_backward.mlir`:
   back-edges present -> silent + all-✓ table; back-edges removed ->
   `expected-error {{missing sibling back-edge}}` + `CHECK: ✗ MISSING`.
3. Missing backward / self-edge (#2/#9) — QK->P in-place reuse with degenerate
   `dstTask==srcTask` guard -> `expected-error {{degenerate self-edge}}`;
   working FA shape -> silent.
4. Redundant acc init (#4) — pre-fix `matmul_kernel_tma_static_persistent_ws_while`
   -> `expected-error {{redundant cross-partition zero-store}}`; post-fix ->
   silent.
5. Pinned-phase deadlock (#11) — collapsed constant-phase fwd `desc_o` staging
   -> `expected-error {{loop-invariant phase ... parity never flips}}`;
   continuous-accumCnt -> silent.
6. Under-buffering / cross-stage (#8, §4.4) — a reuse group whose producer and
   consumer span >1 pipeline stage with `buffer.copy < requiredDepth` (e.g. the
   `(qk, p)` in-place reuse at stage-diff 2, `requiredDepth=3` vs copy 1) ->
   `expected-error {{under-buffered reuse group}}`. Include both an
   after-token-lowering variant (explicit `loop.stage`) and a post-expander
   variant (reuse distance reconstructed from slot/phase stagger + prologue
   depth) to exercise both evaluation paths.
7. Benign redundant-acquire (false-positive guard) — single-buffered EMPTY
   barrier acquired twice per iteration with opposite parities -> only
   `remark: redundant acquire`, no error. The most important negative case.
8. Live HSTU DP=2 (P2426552821) as an e2e regression once reduced to a lit.

Run under the WarpSpecialization lit suite (`test/Hopper/WarpSpecialization/`).

---

## 11. Open questions

1. Where to canonicalize barrier->buffer: first-class `WSBarrier` op vs a
   `guardedBufferId` addition to the `constraints.WSBarrier` dict (§6).
2. Run pre-lowering (NVWS tokens) and/or post-lowering (TTNG barriers) and/or
   post-expander — which combination in CI.
3. Merged-barrier extent granularity vs a coarse "covers all members" model.
4. `scf.if` conditional channels: path-sensitive check vs conservative
   indeterminate (cf. #6 causal-mask).
5. CLC / dynamic-persistent cadence: extend the symbolic-`accumCnt` model or
   declare out of scope.
6. Shared/fused barriers (one WSBarrier for two buffers): the WSBuffer-list on
   WSBarrier is the representation; also verify the marked buffer-set matches the
   buffers actually touched (authors mismarking barriers is itself a defect).
7. Whether the verifier should assert `isWholeAllocationOverwriteReuseOwner` /
   `removeRedundantTmemZeroStores` post-conditions (couples to those passes) or
   stay an independent audit.

---

## 12. Implementation plan

Phased so each phase lands with its own lit tests and is independently useful.
Goal 1 (§0) — run after `add_pipeline`, autoWS only, barrier->`buffer.id` given.
The Buffer-Reuse doc's incremental outline (verify after token lowering, then
after SWP; then TLX; then barrier optimizations) is the outer roadmap; this is
the concrete build order for the autoWS verifier.

- **Phase 0 — scaffolding.** New pass `--nvgpu-verify-ws-barriers` (mirror
  `NVGPUTestWSCodePartitionPass` in `WSCodePartition.cpp`); `BDGNode` / `AOGNode`
  / `AOGEdge` structs; barrier SSA alias trace (`local_alloc` -> block arg ->
  iter_arg -> `memdesc_index`), reusing `WSBarrierAnalysis.h`. Goal 1 reads
  barrier->`buffer.id` as given.
- **Phase 1 — BDG builder.** Nodes for every barrier-touching op; satisfy edges
  (bucket by barrier alloc + slot) + program-order edges; symbolic slot/phase
  from `getBufferIdxAndPhase`; the phase/pre-arm filter (§3.1).
- **Phase 2 — deadlock (§3).** The three conditions (no-producer,
  parity-cadence, cross-partition SCC via Tarjan on the first-execution
  subgraph). Land tests 1 (idle-acc-init, converts the xfail), 5
  (pinned-phase #11), and 7 (the benign redundant-acquire false-positive guard).
- **Phase 3 — PBM + AOG race (§4.1-4.3).** PBM fact table (writers/readers/
  extents/versions per `buffer.id`); per-group AOG build; the two-named-edges
  coverage check + column-packed table + operand-D redundant-init. Land tests
  2 (column-packed), 3 (missing/self-edge #2/#9), 4 (redundant init #4).
- **Phase 4 — under-buffering (§4.4).** `requiredDepth` via the WAR
  reuse-distance, with post-expander reconstruction (slot/phase stagger,
  prologue depth, version carry) and the indeterminate fallback. Land test 6
  (both after-token-lowering and post-expander variants).
- **Phase 5 — diagnostics, dump, wiring.** `emitError/Warning/Remark` messages
  (§8); `--dump-reuse-group=<id>` (extend `dump*BufferLiveness`); wire the
  verifier after `add_pipeline` (and the after-token-lowering run) in
  `compiler.py`; run over the full WS lit suite; reduce live HSTU DP=2
  (P2426552821) to a lit (test 8).
- **Phase 6 (later, out of goal 1).** Barrier->buffer recovery (§6) for the TLX
  front-end; first-class `WSBarrier` op; run-and-diff after token lowering vs
  after SWP; barrier optimizations built on the same analysis.

Milestone gates: Phase 2 exit = all deadlock lits green + no regressions on the
WS suite; Phase 4 exit = the stage-diff-2 qk/p case flagged both pre- and
post-expander; Phase 5 exit = verifier runs in CI on every WS kernel with zero
false positives on the current passing suite.

---

## Key file references

- Emission / IR: `WSCodePartition.cpp` (barrier emit; `createBarrierAlloc`;
  `NVGPUTestWSCodePartitionPass`), `CodePartitionUtility.cpp`.
- Data structures: `CodePartitionUtility.h` (`Channel`, `ChannelPost`,
  `TmemDataChannel`, `TmemDataChannelPost`, `CommChannel`, `ReuseGroup`,
  `isWholeAllocationOverwriteReuseOwner`).
- Barrier metadata / graph / ordered regions: `WSBarrierAnalysis.h`.
- Constraint keys / reordering: `docs/BarrierConstraints.md`;
  `docs/WSBarrierOrderedRegionTracking.md`.
- Pipeline placement: `docs/CodePartition.md`;
  `third_party/nvidia/backend/compiler.py` (`add_hopper_warpspec` ->
  `add_pipeline`).
- Manual rules automated: `.claude/skills/barrier-visualization/SKILL.md`.
- Bug catalog: `.llms/rules/partition-scheduler-bugs.md`.
- Regression IR: `ws_code_partition_dp_idle_acc_init_deadlock.mlir`,
  `ws_code_partition_tmem_packed_reuse_backward.mlir`,
  `ws_code_partition_bwd_persist_staging_war.mlir`,
  `ws_code_partition_bwd_staging_slot_rotation.mlir`.
- Reference IR pastes: prior-code-partition
  [P2426552429](https://www.internalfb.com/intern/paste/P2426552429/);
  post-pipeline-expander
  [P2426552821](https://www.internalfb.com/intern/paste/P2426552821/).
