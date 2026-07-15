# Debugging AutoWS Accuracy / Deadlock Issues

A reusable triage process for AutoWS (automatic warp specialization) kernels
that produce **wrong results** or **hang**, especially when a known-good
reference exists (a TLX / manual-WS kernel, a non-persistent variant, or a
different config that works). Distilled from the FA-bwd persistent
staging-reuse race triage (see "Worked example" below).

The single most useful framing: **find the smallest delta between a passing and
a failing variant, then diff their synchronization.** Most AutoWS correctness
bugs are a *missing or degenerate barrier*, not wrong math.

---

## Tools & skills

| Tool / skill | Use |
|---|---|
| `autows-docs` skill | **Read first.** Maps the task to the right design doc (`ReuseGroups.md`, `TMAStoreWaitPipeline.md`, `TokenBarrierLowering.md`, `BarrierFusion.md`, `AccumulationCounters.md`). The docs encode the invariants you're about to check. |
| `ir-debugging` skill | `TRITON_KERNEL_DUMP=1` → `~/.triton/dump/` (`.ttir/.ttgir/.llir/.ptx/.cubin`); `MLIR_ENABLE_DUMP=1` for per-pass IR; `TRITON_ALWAYS_COMPILE=1` to bypass cache. |
| `barrier-visualization` skill | The **mbarrier phase model** (the key reasoning tool) + the coverage-table method + the column-packed TMEM aliasing hazard. |
| `autows-testing` skill | Correctness test commands (do **not** use the TLX `test_correctness.py` for AutoWS). |
| `compute-sanitizer` | `--tool racecheck` (SMEM data hazards), `--tool synccheck` (illegal/divergent barriers). At `/usr/local/cuda-*/bin/compute-sanitizer`. |
| `[ws-summary]` memory-planner dump | The one-screen SMEM/TMEM dataflow summary (loops, partition→key-ops, buffer producer→consumer(s) with `buffer.id`/reuse). See §"Dump the memory-planner summary" below. Fastest way to see how each accumulator/operand is partitioned and which buffers share an id, without reading the full TTGIR. |
| `debug-failing-gpu` skill | Recover from GPU-busy/OOM before/while running repros. |
| `git log -S` / `git show <rev>:<file>` | Find the *right* reference revision — the matching reference kernel may be an **older** version of a file (kernels get refactored). |

---

## Process

### 1. Reproduce and isolate the exact config
Pin the failing config precisely (variant, `bwd_config_idx`, head dim,
`early_tma_store_lowering`, causal, shapes). Many configs are gated by
`pytest.skip` — the failing combo may have **zero test coverage**, so reproduce
it directly by setting `kernel.configs = [...]; kernel.cache = {}` like
`test_op` does.

Write an **accuracy benchmark** that runs the AutoWS kernel and a known-good
reference (TLX) on identical inputs, compares both to a torch reference *and to
each other*, and **runs N times** to expose non-determinism. (Template:
`third_party/tlx/tutorials/accuracy_bench_bwd_autows_vs_tlx.py`.)

### 2. Narrow to the smallest passing/failing delta
Sweep the axes and find the one flip:
- **config idx** (which scheduling/reuse config fails?),
- **persistent vs non-persistent** (same config) — isolates outer-tile reuse,
- **AutoWS vs TLX** — isolates the compiler vs the algorithm,
- **head dim / shape**.

The delta localizes the bug. (Here: only `ws_persistent` + idx 2 failed → the
bug is in persistent cross-tile reuse, not the algorithm or the config's math.)

### 3. Race or deterministic?
Run-to-run **variance** ≈ 0 ⇒ deterministic logic bug. Large variance ⇒ a race.
A race that doesn't always corrupt output still corrupts *sometimes* — rely on
variance, not a single pass.

### 4. compute-sanitizer on a *small* compiling repro
`racecheck` finds SMEM hazards; `synccheck` finds illegal/divergent barriers.
Precompile once, then run the sanitizer (it's slow). **Caveats that will mislead
you if ignored:**
- racecheck **cannot see TMEM** races — only shared memory.
- racecheck **false-positives on mbarrier kernels** (it models
  `__syncthreads`/named barriers, not the producer/consumer phase protocol).
  *Cross-check:* run it on a **known-correct** kernel (e.g. the forward) — the
  same hazard class there means it's a false positive.
- Use the variance/output signal as the ground truth for "is it actually
  wrong"; use the sanitizer to *locate* candidate SMEM accesses.

### 5. Rule out hypotheses with the PTX
Check the lowered PTX before trusting an IR-level theory. Example: TMA-store
draining — `cp.async.bulk.wait_group.read 0` everywhere means every TMA store
fully drains, so any "store-still-in-flight" theory for that buffer is dead.

### 6. Diff against the TLX base implementation (structured)

Most AutoWS kernels are ported from a hand-written **TLX** kernel that encodes
the intended warp-specialized structure (partitions, buffer reuse, op schedule,
barriers) explicitly. That TLX kernel is the *ground-truth design* — the AutoWS
compiler is supposed to reproduce it. When AutoWS is wrong, diff the two in this
**fixed order** (stop as soon as a level diverges — an earlier-level mismatch
explains later ones):

**Gather the same information for both sides first.**
- **AutoWS side:** the `[ws-summary]` (see the section above) gives partitions →
  key ops and every buffer's producer→consumer(s) + `buffer.id`/reuse in one
  screen. Plus the final TTGIR for barriers/phases.
- **TLX side:** there is no `[ws-summary]` (TLX is manual WS, not the memory
  planner). Extract the equivalent by hand from the TLX kernel + its TTGIR:
  partitions = the `tlx.async_task` regions (`with tlx.async_task(...)`); buffers
  = the explicit `tlx.local_alloc(..., storage_kind.tmem/smem)` sizes and any
  `reuse=` aliasing; schedule = program order within each task; barriers =
  `tlx.alloc_barriers` + `barrier_wait`/`barrier_arrive`.

> **Dump BOTH final TTGIRs and run `barrier-visualization` on each, then diff
> them side by side.** This is the single most effective way to compare TLX vs
> AutoWS synchronization — TLX lowers to the *same* `ttg.warp_specialize` +
> `ttng.wait_barrier`/`arrive_barrier`/`init_barrier` ops AutoWS does, so the two
> final TTGIRs are directly comparable at the mbarrier level (do NOT stop at the
> Python/TLX-DSL vs compiler-IR mismatch). Get both with `TRITON_KERNEL_DUMP=1`
> (run TLX with meta-WS **off** and AutoWS with it **on**, in separate processes —
> see the skill). Then diff, cheapest-signal-first:
> 1. **Partitions + per-partition `num_warps`** and total warp count (must be
>    `<= 16`; AutoWS `optimize-partition-warps` can push a compute partition to 8
>    and blow the budget where TLX uses `num_warps=4`).
> 2. **`init_barrier` count histogram** (`grep -oE 'init_barrier %[^,]+, [0-9]+'`
>    ... `sort | uniq -c`). A count-`N` barrier means `N` arrivers must complete a
>    phase flip — so a barrier released by an `N`-warp partition must be `init N`,
>    not 1. A per-channel init-count mismatch (TLX `init 4` vs AutoWS `init 1` on
>    the same 4-warp-consumer channel) is a silent phase desync invisible to
>    arrive/wait balancing. (Caveat: TLX's CLC-persistent kernels also carry a
>    `clc_context` barrier `init num_consumers` that a non-persistent AutoWS
>    kernel simply won't have — don't mistake that for the bug.)
> 3. **Per-channel arrive/wait/phase — enumerate EVERY cross-partition channel,
>    not just the new/suspect one.** Build the full channel inventory on both
>    sides (`grep` the `wait_barrier`/`arrive_barrier` + their `WSBarrier`
>    `{direction, dstTask, channelGraph}` and group by barrier), then for each
>    channel (qk, p/dp, dv, dk, dq, and the load channels k/v/q/do) compare: which
>    task arrives, which waits, forward vs backward edge present, the phase expr
>    (continuous counter vs `~counter`), and whether producer/consumer derive the
>    phase from the *same* iteration count (the lockstep check). A deadlock can
>    sit in a channel a structural change *perturbed* (e.g. `merge_epilogue`
>    moves dk/dv into the compute partition), not only the obviously-new one — a
>    dq-only spot-check will miss it. A producer pipelined/peeled differently from
>    its consumer (e.g. TLX's one-step `prev_blk_idx` producer + flat consumer vs
>    an AutoWS flat producer + compiler-peeled consumer) breaks lockstep here.
>
> **Tip:** `ttgir2tlx` (`triton-opt --tlx-print-ttgir-to-tlx <final.ttgir>`, or
> `TRITON_DUMP_TTGIR_TO_TLX=1`) rewrites each final TTGIR back into readable
> TLX-Python (`tlx.barrier_wait`/`barrier_arrive`/`async_tma_reduce` per task), so
> the two kernels land in the *same* form and the per-channel diff is a text diff.

Then compare, in this **strict order** — do not move to the next level until the
current one matches. Most of the value is in levels 1–2; get those right before
touching synchronization:

**(A) Align the partitions first.** Line up the two partition sets by *role*
(load / gemm(MMA) / computation(softmax) / epilogue / reduction) — the partition
*numbers* won't match, so map them by what each does. AutoWS may even have a
different partition *count* than TLX (e.g. separate `reduction` and `epilogue`
partitions where TLX has one store task); note that up front.

**(B) Compare partition-by-partition that the key ops are the SAME.** For each
aligned partition, list its key ops (MMAs, `tmem_load`/`tmem_store`,
`local_store`, `async_tma_copy`/`async_tma_reduce`, `descriptor_load`) and check
they match the reference **in identity, partition, AND count**. This is the step
that most often finds the bug, and it is *not* about synchronization:

- **A key op that appears in a partition it shouldn't, or the SAME op emitted in
  two partitions, or missing from one** is a structural code-partition bug. A
  duplicated store/reduce double-counts; a duplicated consumer re-reads a shared
  buffer at an unsynchronized point (garbage). These are invisible to barrier
  analysis — the individual barriers are all balanced.
- **Worked example (T279388065):** the dq `async_tma_reduce(add)` epilogue was
  emitted in **two** partitions (the `reduction` task *and* the `epilogue` task,
  next to the dk stores), both reducing the **same** `desc_dq_reduce_staging`
  into global dq — while TLX has exactly **one**. The second (epilogue) copy
  never fills the staging (no `tmem_load`/`trans`/`local_store` before it), so it
  double-adds dq (2× over-count) or reduces stale/uninitialized staging
  (garbage). Purely a partition-by-partition key-op **count** mismatch (2 vs 1);
  the tell in the IR is also `buffer.tmaStaging = 2` on the staging alloc (two
  staging consumers). Sync/phase/layout analysis all looked identical to TLX and
  found nothing — only the key-op count diff exposed it.

**(C) Only after (A) and (B) match, check synchronization** (levels below):

1. **Buffer reuse decisions.** For every TMEM/SMEM buffer, compare `buffer.id`
   sharing and physical fit: which tiles share an id (reuse group), which are
   packed by column (`buffer.offset`) or stacked in the two 64-lane row-halves,
   which are single vs multi-buffered, and which accumulators are **shared** vs
   **per-partition/per-half**. A shared reduction accumulator written by multiple
   MMAs into ONE alloc (chained accumulator) is NOT a reuse group — see
   `ReuseGroups.md` §"Shapes that are NOT reuse groups" and `OperandDHandling.md`.
   Mismatched reuse (an alias the reference doesn't make, or a shared tile the
   compiler duplicated) is a common root cause.
2. **Per-partition deep dive: op schedule/order + SWP.** Within each partition,
   compare the **order** of the key ops and the **software-pipelining (SWP)
   schedule** — the `stage`/`order`/`cluster` each op is assigned (AutoWS: the
   `tt.autows`/`loop.stage`/`loop.cluster` attrs and the per-dot `attrs=` in the
   kernel; TLX: program order + which iteration each barrier phase belongs to).
   A dot placed in the wrong stage, or a consumer/producer wait on the wrong
   phase, shows up here.
3. **Barriers (coverage table).** Finally, drive the `barrier-visualization`
   **coverage table**: for every physically-reused/aliased/shared buffer,
   enumerate the ordering edges that *should* exist (forward producer→consumer
   **and** backward / cross-iteration consumer→next-producer) and mark each ✓
   present or ✗ MISSING against the TLX reference. Count-balancing
   (`#arrive == #wait`) gives false passes — an *absent* cross-alias edge, or one
   the reference places on the *first* writer but the compiler placed on the
   *last* (or split across two channels), is invisible to it.

If the obvious reference isn't the same shape (e.g. the current TLX bwd is
one-tile-per-CTA while AutoWS is grid-stride persistent), **`git log` the file**
for an older revision that matches. Mind config skew: the TLX and AutoWS variants
may run different `BLOCK_M`/`BLOCK_N` (which changes TMEM fit — 64-row tiles
stack, 128-row tiles don't), so compare **structure/decisions**, not raw byte
counts, unless you pin both to the same config.

### 7. Root-cause with the mbarrier phase model
Reason, don't tally. Genuine deadlock requires one of: (1) no arrive ever
produces the awaited phase, (2) parity-cadence starvation, (3) a cross-partition
cycle. Genuine race = a missing/degenerate ordering edge. Degeneracy signatures
seen in practice:
- **Constant phase** on a reuse wait (literal `0`/`1` instead of an
  `accumCnt`/IV-derived phase) ⇒ never alternates across the loop ⇒ no-op after
  the first iteration.
- **Same-task channel**: a producer/consumer pair in the *same* partition has
  its consumer-release **elided** by `WSLowerToken`'s same-partition elision, so
  the surviving acquire is a no-op. A cross-partition edge must be modeled with
  a token whose producer and consumer are in **different** tasks.
- **Different warp counts**: you cannot add a second consumer-release of a
  different warp count to a host buffer's empty barrier (`consumerWarps ==
  nWarps` assert) — use a dedicated token.

### 8. Fix and verify
Verify the *output* is correct **and deterministic** across many runs/seeds;
re-run `racecheck`/`synccheck`; run the full WarpSpecialization lit suite for
regressions; confirm the other configs and the passing variants are unchanged.
**Add regression coverage** for the previously-uncovered config.

---

## Dump the memory-planner summary (`[ws-summary]`)

When a wrong result is on a **specific buffer** (an accumulator, an operand tile),
the fastest orientation is the memory planner's text summary — it prints, in one
screen, the loop nest, each partition's key ops, and every buffer's
producer-partition → consumer-partition(s) with its `buffer.id` and reuse. Use it
to answer "which partition writes this, which reads it, and does it share an id
with anything?" before diving into the full TTGIR.

`DEBUG_TYPE` is `nvgpu-ws-memory-planner`; the summary lines are prefixed
`[ws-summary]` (emitted by `printWsSummary` in `WSMemoryPlanner.cpp`). The
memory planner runs *inside* `NVGPUWarpSpecialization` (as `doMemoryPlanner`), so
to drive it standalone with debug output, run the **test pass**
`nvgpu-test-ws-memory-planner` on the IR captured **just before** warp
specialization.

```bash
# 1. Get the final TTGIR + capture IR before/after every top-level pass.
#    Under MLIR_ENABLE_DUMP the WS pass runs with dump-intermediate-steps=true,
#    so the log ALSO contains its internal steps ("WarpSpec internal IR Dump
#    After: doTaskIdPropagate / ... / doHoistLoopInvariantTMEMStore / doMemoryPlanner").
MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 <repro cmd> 2> /tmp/passes.mlir
#    (final TTGIR also lands in ~/.triton/dump/<hash>/<kernel>.ttgir with
#     TRITON_KERNEL_DUMP=1 — see the ir-debugging skill.)

# 2. Extract the module dumped "WarpSpec internal IR Dump After:
#    doHoistLoopInvariantTMEMStore" — this is the INPUT to doMemoryPlanner and,
#    crucially, it already carries async_task_id. Save it to /tmp/pre_mp.mlir.
#    Do NOT use "IR Dump Before NVGPUWarpSpecialization": that IR still uses
#    ttg.partition (task-id propagation happens inside the WS pass), so the test
#    pass aborts with "handleOperandD: expected exactly one producer task ID, got 0".

# 3. Re-run just the memory planner on that IR with debug, grep the summary.
#    num-buffers must be >= 1 (matches num-stages) or the pass no-ops; pass the
#    same smem-budget the WS pass used (grep the pass invocation in /tmp/passes.mlir).
triton-opt --nvgpu-test-ws-memory-planner="num-buffers=1 smem-budget=232448" \
  -debug-only=nvgpu-ws-memory-planner /tmp/pre_mp.mlir 2>&1 \
  | grep -F '[ws-summary]'
```

The summary has three blocks: `==== loops ====` (scf.for nesting + depth),
`==== partition -> key ops ====` (what each task does), and
`==== buffers: producer-partition -> consumer-partition(s) ====` (per buffer:
`smem`/`tmem`, name, `buffer.id`, producer task → consumer task(s)). A buffer read
by a *different* partition than its producer is a cross-partition channel — the
usual site of an accumulator race. Two buffers with the **same `buffer.id`** are a
reuse group (see `ReuseGroups.md`); the **same `buffer.id` on one `allocOp`
written by multiple MMAs** is a chained accumulator lifecycle (see
`OperandDHandling.md`), *not* a reuse group.

---

## Lockstep instrumentation (cross-partition barrier counters)

When a cross-partition producer/consumer buffer (esp. an operand-D accumulator's
empty/full pair) races even though the barrier structure looks correct, the
suspect is that the two partitions' **phase counters drift out of lockstep**
across the loop nest, or that a release fires before the async op it guards
actually drains. "Lockstep instrumentation" is the set of checks that confirm or
refute this. It is what to run *after* step 6 shows the broken kernel is
structurally identical to the reference yet still races.

A cross-partition single-buffer WAR only works if, for every iteration N, the
producer's `producer_acquire`(iter N+1) waits on the *exact* `consumer_release`
of iter N — which requires (a) both partitions derive the phase from the **same**
iteration count, (b) the release is ordered after the consumer's read truly
completes, and (c) the intra-partition warp sync is uncorrupted.

### 1. Static counter lockstep (the core check)
For the two channels of the pair (e.g. `dq_empties` acquired in the gemm
partition, released in the reduction partition), trace the **phase iter_args in
both partitions' `scf.for` nests** and verify they advance identically:

- Inner iter_arg initializes from the **outer** iter_arg's carry
  (`iter_args(%acc = %outer_arg)`), increments `+1` per inner iteration
  (`%next = %acc + 1; scf.yield %next`), and the outer loop yields the **final
  inner count** back (`scf.yield %inner#final`) — so the counter is a *single
  continuous* global-inner-iteration count across the whole nest.
- Both partitions must show the *same* structure. If one resets per outer
  iteration while the other stays continuous, the phases desync after the first
  outer iteration → race for ≥2 outer iterations.
- The phase **formulas** should be the standard producer-inverted / consumer
  parity: producer `((cnt & 1) ^ 1)`, consumer `(cnt & 1)`.
- Check **`init_barrier` counts** — every barrier should init to its expected
  arrive count (usually 1). A wrong init lets a wait pass prematurely (race)
  with no phase discrepancy.

### 2. Async-drain ordering (PTX)
For an async TMEM read, the consumer-release must be emitted **after** the
`tcgen05.wait::ld` (not just after the `ttng.tmem_load` op). In the PTX, confirm
the order is `tcgen05.ld → tcgen05.wait::ld → (bar.sync) → mbarrier.arrive`
(the empty/reuse release). If the `mbarrier.arrive` precedes the `wait::ld`, the
next iteration's producer overwrites TMEM while the load is still draining.

### 3. Runtime loop-carry isolation (the decisive probe)
Shrink the loop trip counts and watch where correctness flips — this localizes
the racing barrier without any IR reading:

- Make the **inner** loop run **once** (e.g. shrink the Q length so the inner
  m-loop trips a single time). If 1 iteration is numerically correct but ≥2
  iterations are wrong, the bug is the **cross-iteration WAR/reuse** (the barrier
  that carries the accumulator between consecutive iterations), not a
  within-iteration hazard.
- Independently vary the **outer** loop count to separate inner- vs outer-loop
  carry. Always run each config N times and compare exact tensors
  (`torch.equal`) — a stable rel-L2 with differing bits is still a race.

### 4. Named-barrier collision (multi-warp partitions)
A partition with >1 warp that co-reads a wide tile (e.g. a 4-warp reduction
reading a `blockM=128` TMEM accumulator via `#linear` with `warp=[[32,0],[64,0]]`)
uses a `bar.sync <id>` for its intra-partition sync. Grep the PTX for the
`bar.sync`/`barrier.sync` **ids per partition** and confirm the id is **not
shared by another concurrently-active partition** — a collision desyncs the
warps, so a subset (e.g. 2 of 4) proceed early and read stale TMEM, producing a
**partial/half-tile** corruption.

### Outcome interpretation
If §1–§4 all pass (counters continuous and matched, release after `wait::ld`,
init counts correct, no `bar.sync` id collision) yet the runtime probe (§3) still
races, the defect is **not** in the barrier model — it is below TTGIR/PTX (a
tcgen05 accumulator-chain / cross-partition TMEM coherence behavior), and the
next step is SASS single-step / live mbarrier-state inspection or hardware-team
escalation. (This was the outcome for the HSTU 2-KV `dq` case, T279388065:
counters in lockstep, release correctly after `wait::ld`, init counts 1, and the
kernel is IR- and PTX-identical to the correct TLX `attn_bwd_ws_2kv` — yet the
cross-inner-iteration `dq` reuse still races.)

---

## Worked example: FA-bwd persistent staging-reuse race

- **Symptom:** `fused_attention_ws_device_tma` bwd, `ws_persistent` +
  `early_tma_store_lowering` + `bwd_config_idx=2`, hDim 128 — wrong, non-
  deterministic gradients (~3-4 vs ~3e-3). Non-persistent and TLX correct.
- **Isolation:** only `ws_persistent`+idx2 failed; non-persistent idx2 and TLX
  passed ⇒ persistent cross-tile reuse.
- **Ruled out:** the 3-group `{dpT,dq,dsT}` *TMEM* reuse (racecheck is TMEM-
  blind; its cross-iter WAR is present); `can_rotate_by_buffer_count` (PTX =
  `wait_group 0`, stores drain); operand↔MMA reuse (armed backward barriers
  present).
- **Reference:** the current TLX bwd is one-tile-per-CTA; the matching
  *persistent* TLX bwd was an **April** revision (`git log` the file) — CLC
  work-stealing with the same `sdv reuses v_tiles` / `sdk reuses k_tiles`
  pattern, guarded by a dedicated `k_empties` arrived *after* the store drain.
- **Root cause:** dv/dk TMA-staging buffers alias the v/do operand SMEM
  (`allocation.reuseTarget`); the Step-7.5 guard was degenerate (constant
  `phase=0`, and a same-task channel whose release is elided) ⇒ the next tile's
  load raced the previous tile's staging store.
- **Fix:** a dedicated single-buffered cross-partition reuse token — load task
  acquires at the outer-loop top, staging task releases at the bottom (loop-
  carried phase). See `TMAStoreWaitPipeline.md` §"Step 1 … Step 7.5" and commit
  `e9135d565`.
- **Lesson:** the bug was *not* where racecheck pointed (TMEM is invisible to
  it) nor where the first hypothesis aimed (TMA drain) — the
  passing-vs-failing-variant delta plus the TLX barrier diff localized it.

## Worked example: FA-fwd persistent deadlock (constant staging phase)

- **Symptom:** `fused_attention_ws_device_tma_dp` fwd, `ws_persistent`, non-
  causal, hDim 128 — **deadlock**. Compile+launch returns in ~12s, then
  `torch.cuda.synchronize()` hangs forever with GPU pinned at 100% (faulthandler
  shows Python stuck in `synchronize`/`assert_close`).
- **Isolation:** `git bisect` on the build — HEAD~1 (`27e57b931`) completes 4/4,
  HEAD (`cb9d56313`, the bug #10 fix) hangs 3/3 ⇒ the top commit caused it.
- **Diff the synchronization (step 6/7):** dumped the post-WS TTGIR at both
  commits (`TRITON_KERNEL_DUMP=1`). The `desc_o` output-staging producer wait
  went from `wait_barrier empty, phase=(tile_idx/2)&1` (HEAD~1, rotating) to
  `wait_barrier empty, %c1_i32` (HEAD, **constant phase**) — the canonical
  "constant phase ⇒ never alternates ⇒ deadlocks after tile 0" failure mode
  (see Process §7).
- **Root cause:** bug #10 made `getStaggeredAccumCnt` return the bare subtile
  index for *all* `buffer.tmaStaging` allocs. Correct for **same-partition**
  wait_group-drained staging (bwd `dk/dv/dq`), but `desc_o` is **cross-
  partition** (compute task produces, epilogue-store task consumes) — a
  producer/consumer mbarrier whose phase must rotate by the continuous
  `accumCnt`. The constant index pinned the phase. `buffer.tmaStaging` alone
  can't tell the two apart; the discriminator is producer-task == consumer-task.
- **Fix:** gate the bare-subtile-index path on same-partition staging
  (`getStaggeredAccumCnt`); cross-partition keeps continuous `accumCnt`. Plus a
  defensive `K | S` cap in `increaseFusedEpilogueCopies` for same-partition
  rings. See partition-scheduler-bugs.md #11 and `AccumulationCounters.md`
  §"TMA Staging Buffers".
- **Lesson:** a fix scoped by a coarse tag (`buffer.tmaStaging`) silently caught
  a second, structurally different staging case; the persistent-vs-parent commit
  diff + the IR phase diff localized it exactly. New compiler paths (fwd
  persistent here) need coverage *before* a tag-based change rides over them.
