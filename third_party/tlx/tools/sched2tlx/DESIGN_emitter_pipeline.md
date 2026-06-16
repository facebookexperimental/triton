# Design Doc: Emitter Software-Pipelining (general per-partition lowering)

## TL;DR

**Software pipelining is a general lowering technique, orthogonal to warp specialization.** Warp specialization is the *spatial* split — distribute a loop's ops across N warp groups (partitions). Software pipelining is the *temporal* split — within a single execution stream, overlap iteration `i+1`'s early stages with iteration `i`'s late stages (prologue / steady-state / epilogue + depth-N ring buffers + deferred waits). The two **compose**: a program is partitioned into warp groups (WS), and **each partition whose schedule spans ≥2 stages is independently software-pipelined**. The single-warp-group (no-WS) kernel is just the degenerate 1-partition instance of the same lowering.

The modulo schedule already produces multi-stage schedules (per-node `stage`, per-buffer `count`=depth, `II`, `maxStage`) — for single-WG kernels *and* for individual partitions of multi-WG kernels. But the **sched2tlx emitter only realizes pipelining via cross-WG producer/consumer barriers (SemIR); it does not software-pipeline within a partition's own body.** The clearest symptom is the single-WG path, which emits a flat, op-by-op loop: a **blocking** TMA load (`desc.load()`) + **blocking** TMA store (`wait(0)`), both on the critical path → case6 LayerNorm is **0.27–0.79× hand-written** (1506 vs 5418 GB/s at M=65536). The same gap exists, less visibly, inside any multi-WG partition that has internal stages but is emitted as a flat body.

This project builds **per-partition software pipelining** as a general emitter capability: given a partition's stages + buffer depths, emit prologue/steady/epilogue with ring buffers and deferred waits. Applied uniformly to (a) the single-WG case (first, simplest), and (b) each multi-WG partition. Target: case6 generated ≥ hand-written (≈5.4 TB/s); a partition with 2 internal stages emits a 2-stage pipeline; no regression to cases 1/2/3/5.

Staged into three capabilities (store deferral → load prefetch → general prologue/steady/epilogue), each landed and measured independently, then applied per-partition.

---

## 1. Problem & current state

### 1.1 Two kinds of overlap — and which the emitter realizes today
- **Cross-WG (spatial, WS)**: producer in one warp group, consumer in another, synchronized by SemIR producer/consumer mbarriers over depth-N buffers. The emitter **does this** (case2/3 ≥1.0× hand-written). This is overlap *between* partitions.
- **Intra-partition (temporal, SWP)**: within one warp group's own loop body, overlap iteration `i+1`'s early-stage ops with iteration `i`'s late-stage ops (prologue/steady/epilogue + ring + deferred waits). The emitter **does NOT do this** — every partition body is emitted as a **flat, op-by-op loop**.

The single-WG (no-WS) path (`emit()` bypass: `inner_loop is None and not crossloop_channels and len(real_wg_ids) <= 1` → one plain `@triton.jit` loop) is where the missing intra-partition SWP is most visible (case6), because there's no cross-WG overlap to hide it. But the same gap exists inside any multi-WG partition that spans ≥2 stages internally — its body is flat too.

### 1.2 What the single-WG path emits today (case6)
```python
L0_smem_0 = tlx.local_alloc((8, 512), tl.float16, 1)     # depth 1
for tile_id in range(...):
    v = x_desc.load([row, 0])          # BLOCKING TMA load → registers (no prefetch)
    ... reduce / normalize ...
    tlx.local_store(L0_smem_0[0], y)
    tlx.async_descriptor_store(y_desc, L0_smem_0[0], [row, 0])
    tlx.async_descriptor_store_wait(0) # BLOCKS on the store every iteration
```
Both the load (748 cyc) and store (546 cyc) are on the per-iteration critical path → ~II/iter, no overlap.

### 1.3 What the hand-written reference does (the target)
Single warp group, **no** `async_tasks`, but **manually software-pipelined**:
- double-buffered SMEM ring for X + producer/consumer mbarriers,
- **prologue** prefetches tile 0,
- steady state: wait-consumer → prefetch tile i+1 → wait-producer → `local_load` tile i → compute → store,
- output via `tl.store` (off the TMA engine).
Result: ~5.4 TB/s (3.6× the generated kernel).

### 1.4 The gap
The schedule graph already carries everything needed:
- `schedule_loop.II`, `maxStage` / `num_stages`,
- per-node `stage` (e.g. case6: load N1 stage 0, store N22 stage 1),
- per-buffer `count` (= depth from `floor(lifetime/II)+1`).

The single-WG emitter **ignores `stage` and `count`** — it emits one flat stage with depth-1 blocking ops. This project makes it consume them.

---

## 2. Scope & non-goals

**In scope**: **per-partition software pipelining** as a general emitter capability — given a partition's per-node `stage` + per-buffer `count`, emit prologue/steady/epilogue with ring buffers + deferred waits for that partition's body. Applied to:
1. the **single-WG (no-WS) path** first (simplest; the 1-partition instance), then
2. **each multi-WG partition** that spans ≥2 internal stages (the same lowering, run per partition, composed with the existing cross-WG SemIR barriers).

**Orthogonality (the core thesis)**: WS = *spatial* (split ops across warp groups); SWP = *temporal* (pipeline iterations within a stream). They compose independently: `WS picks the partitions → SWP pipelines each partition's body`. A partition with one stage emits a flat body (today's behavior); a partition with K stages emits a K-stage pipeline. The single-WG kernel and a 2-stage partition of a multi-WG kernel are the **same** lowering with N=1 vs N>1 partitions.

**Non-goals**:
- The cross-WG SemIR barrier machinery (already correct — this *composes with* it, doesn't replace it).
- The schedule/latency model (done in `latency_model/`; this consumes its output).
- New hardware modeling.

---

## 3. Design — three capabilities

Build incrementally; each is independently shippable and measurable.

### L1 — Store deferral (smallest; the `bench_store_subslice` +17%)
An in-loop register-fed `tt.descriptor_store` whose staging buffer has **depth ≥ 2**:
- allocate the staging buffer as a depth-`d` ring,
- per iteration: `async_descriptor_store_wait(d-1)` (keep ≤ d-1 in flight) → `local_store(ring[slot])` → `fence_async_shared` → `async_descriptor_store(... ring[slot] ...)` → advance slot,
- **post-loop**: `async_descriptor_store_wait(0)` to drain.
Today the emitter emits depth-1 + `wait(0)` every iter. A.7 already does a version of this for the case2 *subtiled* epilogue — generalize it to any depth-≥2 in-loop store on the single-WG path.
**Prereq**: the schedule must give the store staging buffer depth ≥ 2. For case6 the store fits one II (lat 546 < II 594 → depth 1), so L1 alone needs either (a) the store-occupancy model to push depth to 2, or (b) emit deferral even at depth-1 by deferring the wait to the *next* iteration's stage rather than `wait(0)` immediately. (b) is the robust form.

### L2 — Load prefetch for register-consumed loads (the big case6 win)
A `tt.descriptor_load` consumed **in registers** (no downstream `local_alloc`; e.g. `load → tl.sum`) that the schedule places at an **earlier stage than its consumer** (case6: load stage 0, consumer stage 1):
- convert the blocking `desc.load()` into the hand-written async pattern:
  allocate a depth-`d` SMEM ring + producer/consumer barriers,
  **prologue**: `barrier_expect_bytes` + `async_descriptor_load(ring[0], producer_bar[0])`,
  steady state: `barrier_wait(producer_bar[slot])` → `local_load(ring[slot])` → (consume) and, before consuming, issue the prefetch for tile `i+d` into the freed slot,
  toggle phases per the depth.
This is exactly `handwritten.py`. It is the dominant case6 win (the load is 748 cyc on the critical path).

### L3 — General prologue / steady-state / epilogue framework
A **partition body** (single-WG kernel, or one warp group of a multi-WG kernel), today emitted as one flat stage, becomes a modulo-pipelined loop:
- partition that body's nodes by `stage` (from the schedule),
- emit a **prologue** that issues stages `0..maxStage-1` for the first `maxStage` iterations,
- emit a **steady-state** body issuing each node's stage for the appropriate (shifted) iteration index, with `bufferIdx = accumCnt % count`, `phase = (accumCnt / count) & 1` (same `accumCnt` arithmetic the WS path uses),
- emit an **epilogue** draining stages.

This is a pure function of *one partition's* sub-schedule, so it runs **per partition**: the single-WG kernel is the N=1 call; a multi-WG kernel calls it once per partition (each with its own stages), and the cross-WG SemIR barriers (unchanged) stitch the partitions together. L1 and L2 are special cases of L3; doing them first (on the single-WG body) de-risks the general framework and delivers perf early, then L3 is applied uniformly per partition.

---

## 4. Architecture / where in `emitter.py`

- The loop-body emitter for **any one partition** is `_emit_warp_group` / `_emit_uwg_body_impl` (used by both the single-WG bypass at `emit()` ~L4496 and each warp group of the WS path). Today it walks nodes in `(stage, cluster, id)` order and renders each op once into a **flat loop**.
- **Change**: make this per-partition body emitter a software-pipeliner. When the partition's sub-schedule has `maxStage ≥ 1` or any consumed buffer `count ≥ 2`:
  - group that partition's nodes by `stage`,
  - emit ring allocs + barriers for depth-≥2 buffers (reuse the WS path's alloc helpers),
  - emit prologue/steady/epilogue instead of the flat body.
  Because the change is *inside* the per-partition body emitter, it applies to the single-WG kernel and to every WS partition with the same code — no separate path.
- Reuse, don't fork: the `accumCnt`→`bufferIdx/phase` helper, the `local_alloc` ring helper, and the descriptor-load/store renderers already exist. The new code is the **control-flow scaffold** (prologue/steady/epilogue) parameterized by a partition's stages.
- Renderers to extend: `_render_descriptor_load` (add the async-ring variant for register-consumed loads), the in-loop `tt.descriptor_store` handler (deferred-wait variant).
- Composition with WS: cross-WG SemIR barriers are emitted as today *around* each partition's (now-pipelined) body; the intra-partition prologue/epilogue nest inside the partition's loop, the cross-WG barriers span partitions. They don't interfere — different buffers, different barriers.

---

## 5. Milestones

- **M1 — Store deferral (L1).** Emit deferred double-buffered store on the single-WG path. Validate: case6 store no longer blocks; expect the `bench_store_subslice` +17% (≈2518→2941 GB/s). Small, isolated.
- **M2 — Load prefetch (L2).** Emit async double-buffered SMEM load for register-consumed loads at stage 0. Validate: case6 load off the critical path; expect the bulk of the gap to close toward hand-written (~5 TB/s). Biggest win.
- **M3 — General per-partition prologue/steady/epilogue (L3).** Refactor M1+M2 into a single stage-driven SWP scaffold inside the per-partition body emitter. Validate on the single-WG body: case6 ≥ hand-written; handles arbitrary stage counts.
- **M4 — Apply per-partition in multi-WG kernels.** Run the same SWP scaffold for each WS partition whose sub-schedule spans ≥2 stages, composed with the existing cross-WG SemIR barriers. Validate: a multi-WG kernel where one partition has 2 internal stages emits a 2-stage pipeline in that partition; cases 1/2/3/5 still pass and perf ≥ committed (and may improve where a partition was previously flat). This is where the "general lowering" thesis is proven end-to-end.
- **M5 — Generalize beyond the known cases.** A new memory-bound kernel (RMSNorm / softmax / copy) through the same path with no kernel-specific code. Confirms it's a general capability, not a case-specific patch.

---

## 6. Validation

| Claim | Test | Pass criterion |
|---|---|---|
| store deferral works | case6 generated, M1 | GB/s ≥ blocking baseline (+~15%) |
| load prefetch works | case6 generated, M2 | GB/s approaches hand-written (≥ ~4 TB/s) |
| matches hand-written | `perf_generated_vs_handwritten.py` (case6) | gen/hw ≥ 1.0× |
| correctness | `run_generated.py` (case6) | 3/3 PASS vs torch |
| per-partition SWP (multi-WG) | M4: multi-WG kernel with a ≥2-stage partition | that partition emits prologue/steady/epilogue; composes with cross-WG barriers |
| no regression | cases 1/2/3/5 re-emit + correctness + perf | all pass, perf ≥ committed |
| general | M5 new kernel | pipelined with no kernel-specific code |

Perf harness: `examples/case6_layernorm/perf_generated_vs_handwritten.py` (already written).

---

## 7. Risks & mitigations

- **Phase/barrier bugs** (the classic WS-barrier debug class). *Mitigation*: mirror the proven hand-written `handwritten.py` structure exactly for M2; reuse the WS path's `accumCnt`/phase helpers; lit-test the emitted control flow.
- **Register-consumed load needs an SMEM staging buffer the schedule didn't allocate** (case6's load has no SMEM buffer in the graph — it's register-direct). *Mitigation*: the emitter synthesizes the staging ring (the schedule says "depth-2 load lifetime"; the emitter owns the SMEM realization), analogous to how the WS path synthesizes register→SMEM cross-WG channels.
- **Disturbing the working WS path.** *Mitigation*: gate all new logic on the single-WG bypass; the multi-WG path is untouched. Regression sweep on 1/2/3/5.
- **Depth-1 stores can't defer with a single buffer.** *Mitigation*: L1 form (b) — defer the wait to the next iteration even at depth 1 — or have the schedule give depth 2 (store-occupancy model). Decide in M1.
- **SMEM budget**: extra rings cost SMEM. *Mitigation*: the schedule's depth already respects the 228 KB budget; the emitter allocates exactly `count`.

---

## 8. Relationship to the latency-model project

`latency_model/` made the *schedule* correct (II, stages, depths). This project makes the *emitter* realize it for the single-WG path. They compose: latency model = "what the pipeline should be"; emitter pipelining = "emit that pipeline." Neither alone improves case6 perf — both are required. (Verified: with the corrected schedule but the current flat emitter, case6 is still 0.27× hand-written.)
