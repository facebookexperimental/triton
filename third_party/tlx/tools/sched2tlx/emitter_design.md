# sched2tlx — Design + Implementation Status

Source-of-truth design document for the emitter. It describes
the algorithm, where each piece comes from in the auto-WS pipeline, and the
current implementation state.

For the input contract, see `SCHEMA.md`. For project rationale, see
`design.md`.

_Last updated: 2026-05-17._

## Principles

1. **Pure mechanical lowering.** No optimization, no heuristics, no pattern
   recognition. The schedule_graph is the contract; the emitter walks it
   node-by-node and emits the TLX equivalent.
2. **Compiler-agnostic output.** The generated TLX runs through any
   TLX-supporting Triton.
3. **One generic algorithm for all kernels.** Single GEMM, persistent GEMM,
   FA — they share the same emitter; only the input graph shape differs.
4. **Algorithm comes from auto-WS.** We're not reinventing the lowering;
   we're rewriting the auto-WS pipeline's output target from MLIR to TLX
   Python.
5. **Trust modulo's authoritative values.** Buffer counts, II, partition
   assignments — all come from `schedule_graph.json`. The emitter never
   second-guesses (no "force count=1" workarounds; if something doesn't fit,
   either the schedule is wrong or the emitter is wrong, but we don't
   silently override the schedule).
6. **Comments trace back to schedule_graph.** Every non-trivial emitted
   construct carries a one-line comment explaining its source (e.g.
   `# inner-loop buf 0: SMEM count=2 (modulo lifetime [640..1310], II=1280)`).

## Auto-WS algorithm we port

From `WarpSpecialization/docs/`:

1. **`collectAsyncChannels`** — discover cross-partition data deps. A
   "channel" is a (producer, consumer) pair where the two ops have different
   warp-group assignments.
2. **`groupChannels`** — group channels by producer (one buffer serves
   multiple consumers of the same producer) and by consumer (one mbarrier
   pair serves multiple producers feeding the same consumer).
3. **`createBuffer` / `hoistLocalAlloc`** — hoist allocations to function
   entry; multi-buffered with depth = `numBuffers`.
4. **`createToken`** — one mbarrier pair per channel group with `numBuffers`
   slots.
5. **`appendAccumCntsForOps`** — thread `accumCnt` (`i64` loop-carried) into
   each enclosing loop. `bufferIdx = accumCnt % numBuffers`,
   `phase = (accumCnt / numBuffers) & 1`.
6. **`insertAsyncComm`** — wrap each producer/consumer with
   `ProducerAcquire/Commit` + `ConsumerWait/Release`.
7. **`specializeRegion`** — clone ops into per-partition regions. Each
   region gets a **trimmed loop** (only the args/yields it uses) and
   **constants are rematerialized** per region.

### Barrier insertion via the Semaphore IR

Steps 1, 2, 4, and 6 above all run through a small symbolic IR
(`sched2tlx/semaphore_ir.py`) that mirrors the upstream NVWS abstraction
(https://github.com/triton-lang/triton/pull/10121). The pipeline is:

1. **`derive_semaphores(loop)`** — walk the schedule graph's
   `cross_wg_barriers` and create one `Semaphore(producers=[...], consumers=[...])`
   per edge, tagging each `ReleaseSite` with an `AsyncKind`
   (`TMA_LOAD` / `TC5_MMA` / `TMEM_COPY` / `NONE`) and each `AcquireSite`
   with an `AccessKind` (`READ` / `STORE` / `UNKNOWN`).
2. **`combine_semaphores(sems)`** — merge edges whose producers feed the
   SAME `(consumer, buffer)` into one semaphore with `pending_count =`
   number of distinct producer arrival kinds. This collapses, e.g., a
   `(loadA, loadB) → mma` pattern into one mbarrier pair the consumer
   waits on once.
3. **`assign_stage_phase(sems, loop)`** — token-walk classifier that
   assigns a `stage` and `phase_expr` to every site. READ keeps the stage,
   STORE advances it. Mirrors `appendAccumCntsForOps`.
4. **`lower_semaphore(sem)`** — mechanical emission of the full+empty
   mbarrier pair, the producer-side acquire/release, and the consumer-side
   wait/arrive. `AsyncKind` drives whether the release lowers to a plain
   `tlx.barrier_arrive(...)` (NONE) or piggy-backs on a TMA/MMA's
   `mBarriers=[...]` field (TMA_LOAD / TC5_MMA).

The SemIR is the default emission path; `TLX_EMIT_LEGACY=1` opts out and
uses the older direct-channel logic. The legacy path is retained for A/B
debugging only.

## Mapping to TLX emission

| Auto-WS construct | TLX Python equivalent | Status |
|---|---|---|
| `WarpSpecializeOp` region | `with tlx.async_task(...)` block | ✓ |
| `CreateTokenOp(numBuffers)` | `tlx.alloc_barriers(num_barriers=N, arrive_count=1)` × 2 (full + empty) | ✓ |
| `ProducerAcquireOp` | `tlx.barrier_wait({buf}_empty[idx], phase ^ 1)` | ✓ |
| `ProducerCommitOp` | TMA's barrier arg auto-arrives on full | ✓ |
| `ConsumerWaitOp` | `tlx.barrier_wait({buf}_full[idx], phase)` | ✓ |
| `ConsumerReleaseOp` | `tlx.barrier_arrive({buf}_empty[idx], 1)` | ✓ |
| `accumCnt` loop-carried `i64` | Python `smem_accum` / `tmem_accum_cnt` integer | ✓ |
| `LocalAllocOp` hoisted | `tlx.local_alloc((shape), dtype, count)` at top of kernel | ✓ |
| `TMEMAllocOp` hoisted | `tlx.local_alloc((shape), tl.float32, count, tlx.storage_kind.tmem)` | ✓ (count from outer-loop schedule) |
| Trimmed `scf::ForOp` per partition | `for iv in range(lb, ub, step):` inside each `tlx.async_task` body | ✓ |
| Constant rematerialization | re-emit `pid_m_c = ...` etc. at top of each task body | ✓ (`_collect_infra_deps_recursive` + `_replicate_infra_deps`) |
| Topological clone order | sort nodes by `(stage, cluster, id)` | ✓ |
| `iter_arg` of scf.for | substituted as `(iv - step)` (heuristic for tile_id_c pattern) | ✓ (case2) |
| `memdesc_trans` walk-through | resolved to underlying SMEM buffer; rendered as `tlx.local_trans(buf)` | ✓ |
| Per-task variable scoping | `rctx.op_var` snapshot/restore per `_emit_uwg_body` call | ✓ |
| Channel discovery / mbarrier emission | SemIR pipeline (`derive_semaphores` → `combine_semaphores` → `assign_stage_phase` → `lower_semaphore`); see "Barrier insertion via the Semaphore IR" below | ✓ (default; `TLX_EMIT_LEGACY=1` for legacy path) |
| End-of-K-loop signal from TC partition (`acc_tmem_full[tmem_buf]`) | `tlx.tcgen05_commit(...)` so the barrier fires only AFTER all in-flight `tcgen5.mma` complete | ✓ |
| Plain TMEM `local_store` followed by signal (rescale ferry, softmax → P bridge) | `tlx.barrier_arrive(...)` (NOT `tcgen05_commit` — `local_store` is sync, no async tcgen5 op pending) | ✓ |

## Kernel shape produced

```python
@triton.jit
def <kernel_name>(<deduped_args>):
    # 1. Preamble: function-scope ops in source order (constants, descriptors,
    #    grid math). Captured into each task body via Python closure.

    # 2. Allocs: hoisted from inner-loop schedule_loop.buffers + outer-loop
    #    TMEM buffer (when nested). Names prefixed with L<loop_id> to avoid
    #    cross-loop ID collision.

    # 3. Mbarriers: one full+empty pair per cross-WG channel + one per outer
    #    TMEM buffer (depth from outer schedule_loop.buffers[tmem].count).

    with tlx.async_tasks():
        for each unified warp_group:
            with tlx.async_task(<spec>):
                # Per-task pre-loop counters (smem_accum, tmem_accum_cnt = 0)

                # Per-task constant rematerialization (cheap deps not yet emitted)

                for tile_id in range(...):       # outer persistent loop (if any)
                    # Per-tile TMEM ring index: tmem_buf, tmem_phase
                    # Per-tile rematerialized infra (pid_m, pid_n, offs_am, ...)

                    # If TC and persistent: barrier_wait(acc_tmem_empty[tmem_buf], phase ^ 1)

                    for k in range(...):          # inner K-loop (where super-node sits)
                        # Per-K SMEM ring index: buf, phase (from smem_accum)
                        # Per-WG ops: TMA loads (MEM partition) or MMA (TC) etc.
                        smem_accum += 1

                    # If TC and persistent: barrier_arrive(acc_tmem_full[tmem_buf], 1)
                    # If default: tmem_load + truncf + descriptor_store
                    tmem_accum_cnt += 1
```

## Unified warp-group view

`loops[outer].warp_groups` and `loops[inner].warp_groups` are independent
numberings. We collapse them into ONE list across the kernel. For each
unified WG we emit one `tlx.async_task`.

For a 2-level nest with a super-node:

| Outer WG (role) | Linked inner WG (role) | Unified WG |
|---|---|---|
| outer wg with epilogue ops | (none — runs only outer loop body) | **U-default** (TC's consumer) |
| outer wg containing super-node | inner wg with TMA loads (MEM) | **U-MEM** |
| outer wg containing super-node | inner wg with MMA (TC) | **U-TC** |

For a single-loop kernel (case1):
- One U-default for the epilogue (function-scope tmem_load + descriptor_store)
- One U-{role} per inner warp group

Implementation: `_unified_warp_groups()` in `emitter.py`.

## NONE-pipeline bridges

In case1 the data flow is `load(MEM, wg0) → local_alloc(NONE, wg2) →
mma(TC, wg2)`. The `local_alloc` is NONE — Phase 4 doesn't assign it to a
warp group, but auto-WS's Step 1.5 propagates it to its consumer's group.

Current emitter: when computing channels and resolving operands, walks
through NONE-pipeline ops via `_resolve_alloc_to_buffer` and the
`memdesc_trans` walk-through. Channel discovery counts cross-WG edges based
on the propagated assignments rather than NONE bridges.

Limitation: Phase 4's cost model in modulo doesn't count cross-WG edges
through NONE-pipeline bridges, which can cause sub-optimal partitioning.
This is a modulo bug (task #39), not an emitter bug — see the case1
study at `studies/case1_load_partition_2wg_vs_3wg/`.

## AccumCnt → Python counter

| Counter | Scope | When incremented | Used for |
|---|---|---|---|
| `smem_accum` | per-task local, declared before outer loop | end of inner K-iter | `buf = smem_accum % depth, phase = (smem_accum // depth) & 1` for SMEM ring |
| `tmem_accum_cnt` | per-task local, declared before outer loop | end of outer-tile iter | `tmem_buf = tmem_accum_cnt % count, tmem_phase = (tmem_accum_cnt // count) & 1` for TMEM hand-off |

For persistent kernels, `smem_accum` **persists across tiles** — the SMEM
ring keeps rotating through outer iterations.

The TMEM `count` comes directly from the outer-loop's TMEM ScheduleBuffer
(`outer_loop.schedule.buffers[tmem].count`). Never from SMEM, never from
heuristics — modulo's authoritative value.

## Special op handling

### TMA load (`tt.descriptor_load`)
```python
tlx.barrier_wait({buf}_empty[buf_idx], phase ^ 1)            # ProducerAcquire
tlx.barrier_expect_bytes({buf}_full[buf_idx], num_bytes)
tlx.async_descriptor_load(desc, {buf}[buf_idx], [...offsets], {buf}_full[buf_idx])
# (TMA auto-arrives on full when the copy completes)
```

### MMA (`ttng.tc_gen5_mma`)
```python
tlx.barrier_wait({a_buf}_full[buf_idx], phase)
tlx.barrier_wait({b_buf}_full[buf_idx], phase)               # if A != B
use_acc = (iv > lb)                                           # implicit init
tlx.async_dot({a}[buf_idx], {b_expr}, acc_tmem[tmem_buf], use_acc=use_acc)
tlx.barrier_arrive({a_buf}_empty[buf_idx], 1)
tlx.barrier_arrive({b_buf}_empty[buf_idx], 1)
```
`{b_expr}` is `tlx.local_trans({b}[buf_idx])` if B was originally fed
through `ttg.memdesc_trans` (case2's B layout).

### TMA store (`tt.descriptor_store`) — default partition
```python
tlx.local_store(c_smem[0], value)
tlx.fence_async_shared()
tlx.async_descriptor_store(c_desc, c_smem[0], [...offsets])
tlx.async_descriptor_store_wait(0)
```

### TMEM hand-off across persistent tiles

Both TC and default partitions:
```python
tmem_accum_cnt = 0
for tile_id in range(...):
    tmem_buf = tmem_accum_cnt % tmem_count
    tmem_phase = (tmem_accum_cnt // tmem_count) & 1
    # ... use tmem_buf, tmem_phase ...
    tmem_accum_cnt += 1
```

TC partition (producer):
```python
tlx.barrier_wait(acc_tmem_empty[tmem_buf], tmem_phase ^ 1)   # wait for prior reader
# K-loop runs (MMA writes acc_tmem[tmem_buf])
tlx.tcgen05_commit(acc_tmem_full[tmem_buf])                  # fires AFTER all in-flight MMAs land
```

`tcgen05_commit` is required here (not `barrier_arrive`): the K-loop's last
`async_dot` is still HW-pending when we exit the K-loop, so a plain
`barrier_arrive` would race and the epilogue would read a partially-written
TMEM tile. `tcgen05_commit` ties the barrier firing to completion of all
prior async `tcgen05.mma` ops on this WG.

Default partition (consumer):
```python
tlx.barrier_wait(acc_tmem_full[tmem_buf], tmem_phase)        # wait for MMA done
acc = tlx.local_load(acc_tmem[tmem_buf])
tlx.barrier_arrive(acc_tmem_empty[tmem_buf], 1)
```

### `tcgen05_commit` vs `barrier_arrive` for TMEM `local_store`

In case3 (FA fwd), the rescale ferry (`tmem_load → mul → tmem_store`) and
the softmax → P TMEM bridge both write TMEM via `local_store` and signal
the consumer. Use **plain `barrier_arrive`** there, NOT `tcgen05_commit`:
`local_store` to TMEM is synchronous from the issuing warp's POV, so by the
time we reach the signal the store is already complete. `tcgen05_commit`
is defined to fire when prior async `tcgen5.mma` ops drain — with no
async op pending it can be a no-op, which deadlocks the consumer
deterministically at scale (≥2048 CTAs / large N_CTX). See "Known issues".

## Implementation status

### Done

| Milestone | What | Status |
|---|---|---|
| **M1** | Dedup duplicate kernel arg names in JSON dumper (`a_desc, a_desc_0, …`) | ✓ |
| **M2** | Renderer registry: arith binops (muli/addi/subi/divsi/divui/remsi/remui/andi/ori/xori), memdesc_trans, get_program_id with axis, make_tensor_descriptor with operandSegmentSizes | ✓ |
| **M3** | Trimmed-loop / per-WG specialization. unified_warp_groups, per-task outer-loop replication, super-node recursion, constant rematerialization, NONE-bridge resolution, persistent SMEM accum_cnt, TMEM hand-off with ring buffer indexing | ✓ |
| **M3.5** | Inner-loop iter_arg threading from WS path (case3 softmax `m_i`/`l_i` carried as plain Python locals across the K-loop and stashed in SMEM for the function-scope epilogue) | ✓ |
| **M4** | Float arith + math + tensor manip renderers (case3: `arith.mulf/subf/maxnumf`, `math.exp2`, `tt.expand_dims/broadcast/splat/reshape/trans/split/join`, `tt.reduce` with `max`/`sum` kinds) | ✓ |
| **M4-SemIR** | Symbolic Semaphore IR (`derive_semaphores` → `combine_semaphores` → `assign_stage_phase` → `lower_semaphore`); default emission path | ✓ |
| **M5** | case3 FA fwd correctness + cross-WG TMEM channel emission (softmax → P TMEM bridge, rescale ferry); case3 e2e PASS on B200 | ✓ |
| **M6** | case3 deadlock fix: `tcgen05_commit` → `barrier_arrive` for plain TMEM `local_store` (was deadlocking at ≥2048 CTAs) | ✓ |
| (extra) | Per-line traceability comments tying emitted code back to schedule_graph fields | ✓ |

### Validated

| Case | Shape coverage | Result |
|---|---|---|
| **case1** (single-loop GEMM, BLOCK=128×128×64) | 5 shapes (1024³, 2048³, 4096³, 1024×1024×960, 1024×1024×1152) | bit-exact vs `torch.matmul` |
| **case2** (persistent GEMM, BLOCK=128×128×64) | 6 shapes (256×256×128 → 8192³ + 1024×1024×16384) | bit-exact vs `torch.matmul` |
| **case3** (FA fwd, non-WS pre-modulo source; BLOCK_M=128, BLOCK_N=64, HEAD_DIM=128) | 6 shapes ((1,4,512), (1,8,1024), (2,16,2048), (1,16,4096), (2,16,4096), (1,32,8192)) | PASS vs `F.scaled_dot_product_attention` (rel < 1e-2) |
| **case5** (persistent GEMM + 2D bias add, BLOCK=128×128×64) | 6 shapes (256³ → 8192³ + 1024×1024×16384), non-WS + TLX-WS references | Reference kernels PASS vs torch (rel ≤ 7.3e-4); emitter incomplete (see "Open questions" #4) |

#### case3 perf vs hand-written TLX WS reference (B200)

| Shape | HW TFLOPS | GEN TFLOPS | GEN/HW |
|---|---|---|---|
| (1,4,512) | 28.5 | 29.6 | 1.04× |
| (1,8,1024) | 159.4 | 173.1 | 1.09× |
| (2,16,2048) | 421.2 | 470.7 | 1.12× |
| (1,16,4096) | 466.5 | 538.3 | 1.15× |
| (2,16,4096) | 526.9 | 559.2 | 1.06× |
| **(1,32,8192)** | **559.4** | **617.4** | **1.10×** |

Generated kernel beats hand-written TLX on every shape. The win comes
from the 6-WG split (Q+K MEM / V MEM / QK MMA / rescale / PV MMA /
softmax) that the cluster + cost-model refinements in D104110963 enable.
Run via `examples/case3_FA/perf_generated_vs_handwritten.py`.

### Deferred (M5+)

| Item | Why deferred | When needed |
|---|---|---|
| case4 (FA bwd) emitter wiring | Reference kernels are in place (`case4_FA_bwd/handwritten.py`, `handwritten_nows.py`); pre-modulo TTGIR + schedule_graph generation not yet done | Next milestone |
| Constexpr promotion (block sizes lifted from buffer shapes to kernel constexprs) | Literals work fine for fixed-shape; cosmetic + flexibility improvement | Polish |
| MMA splitting for `BLOCK_M > 128` (TMEM hardware limit) | case1/case2/case3 use ≤128; only relevant for 256-block schedules | When running 256-block schedules without hand-editing |
| CSE across scopes (e.g. duplicate `pid * BLOCK_M` in case1) | Triton's MLIR CSE handles it at compile time; only visual noise | Polish |
| Auto-generate runner stub (`run_generated.py`) alongside the kernel | Manual writing is fine for now | Polish |
| Reuse-group emission (multiple channels sharing one buffer.id) | Not exercised by case1/case2/case3 | case4+ |

### Known issues with workarounds

1. **`tile_id_c` iter_arg substitution** (case2): the `arith.addi(iter_arg, step)`
   pattern in case2 produces a "tile_id_c + step" value that's mathematically
   equal to the current `tile_id`. We substitute `iter_arg` as `(iv - step)`
   so the addition collapses to `iv`. Brittle to other iter_arg patterns
   (e.g. async tokens, accumulators); revisit when we hit one.

2. **`tcgen05_commit` is NOT a drop-in replacement for `barrier_arrive`**
   (lesson, now encoded in the emitter). For plain `local_store` to TMEM
   (rescale ferry, softmax → P bridge), use `barrier_arrive` — the store is
   sync from the warp's POV. For end-of-K-loop signals from a TC partition,
   use `tcgen05_commit` — the last `async_dot` is HW-pending, and
   `tcgen05_commit` ties barrier firing to its completion. Misuse the other
   way and the kernel deadlocks deterministically at scale (≥2048 CTAs);
   this was caught in case3 (1,32,8192). The emitter encodes both cases
   correctly today.

## File map

| File | Role |
|---|---|
| `sched2tlx/schedule_graph.py` (341 LOC) | JSON → typed dataclasses (Op, Buffer, Node, Edge, Loop, ScheduleGraph). |
| `sched2tlx/semaphore_ir.py` (800 LOC) | Symbolic semaphore IR: `derive_semaphores` / `combine_semaphores` / `assign_stage_phase` / `lower_semaphore`. Default barrier-emission path; `TLX_EMIT_LEGACY=1` opts out. |
| `sched2tlx/emitter.py` (3.9 KLOC) | The emitter. `emit(graph)` is the entry point; helpers handle topology detection, unified WGs, per-task body, op rendering, SemIR integration. |
| `sched2tlx/cli.py` | `python -m sched2tlx <schedule_graph.json> -o out.py` |
| `sched2tlx/__main__.py` | Module entry point. |
| `examples/case1_simple_gemm/` | Single-loop GEMM end-to-end. `schedule_graph.json` (real modulo dump), `generated.py` (emitter output), `handwritten.py` (manual reference target), `run_generated.py` (B200 runner). |
| `examples/case2_persistent_gemm/` | Persistent GEMM (nested loops + TMEM hand-off). Same file layout, plus `run_handwritten.py`. |
| `examples/case3_FA/` | FA forward (non-WS pre-modulo source). Adds `handwritten_nows.py` (non-WS reference) and `perf_generated_vs_handwritten.py` (perf sweep). |
| `examples/case4_FA_bwd/` | FA backward — gold-standard reference kernels (non-WS + TLX-WS) and runners. Emitter wiring is the next milestone; no `schedule_graph.json` / `generated.py` yet. |
| `examples/case5_addmm_bias/` | Persistent GEMM + 2D bias add. Pre-modulo TTGIR, dumped schedule graph, non-WS + TLX-WS reference kernels, runners, and a partially-working `generated.py` (placeholder for the bias TMA load — see Open Question #4). |
| `SCHEMA.md` | The JSON input contract. |
| `design.md` | Project overview / rationale. |
| `emitter_design.md` | This file (algorithm + status). |

## Open questions

1. **Reuse groups** — auto-WS supports buffer sharing via `buffer.id` matching.
   The schedule_graph already has `merge_group_id` per buffer. Not exercised
   by case1/case2/case3; defer.

2. **Outer warp-group merging** — modulo Phase 4 owns the partition
   decisions. The emitter is faithful to whatever Phase 4 outputs. For case3
   the cluster + cost-model refinements in D104110963 (latency-based
   heaviness, heavy-first BFS, barrier-aware cost, TMA-load buffer lowering)
   take case3 FA fwd from 218 TF to 617 TF on B200, so the partition output
   is now competitive — no emitter-side workarounds needed. The case1
   3-WG vs 2-WG question is parked at the same study path
   (`studies/case1_load_partition_2wg_vs_3wg/`) — doesn't affect runtime
   correctness or perf there.

3. **Multiple sibling loops at the same nesting level** — not in
   case1/case2/case3. Defer until we hit it.

4. **case4 (FA bwd) integration** — gold-standard references exist; next
   step is to dump pre-modulo TTGIR and run the emitter end-to-end. May
   surface new patterns (e.g. `atomic_add` for dQ accumulation, 5-MMA
   single-kernel layout).

5. **case5 (outer-loop epilogue TMA load)** — `_render_descriptor_load`
   currently returns `<tma_load_inline_unsupported>` when a `tt.descriptor_load`
   is consumed as a value expression in the outer-loop default partition
   body (case5's per-tile bias load). The K-loop TMA path is handled
   specially in `_emit_in_loop_node`; the outer-loop path needs the
   analogous treatment: hoist a SMEM buffer for the bias tile, allocate a
   full+empty mbarrier pair, and emit
   `barrier_wait(empty)` → `barrier_expect_bytes(full)` →
   `async_descriptor_load(...full)` → `barrier_wait(full)` → `local_load` →
   `barrier_arrive(empty)` per outer-tile iteration. The hand-written TLX
   target at `examples/case5_addmm_bias/handwritten.py` shows the exact
   structure the emitter should produce.

## Non-goals

- Not generating Triton C++ / MLIR — only Python source.
- Not optimizing the schedule — accept whatever modulo produces.
- Not handling kernels modulo can't schedule (loops without TMA, deeper than
  2 levels, etc.) — those go through the auto-WS path as before.
- Not a TLX→TLX rewriter — input is always a `schedule_graph.json`.

## References

For the original auto-WS algorithm:
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSCodePartition.cpp`
  (`collectAsyncChannels`, `groupChannels`, `createBuffer`, `createToken`,
  `insertAsyncComm`)
- `WSSpecialize.cpp` (`SpecializeForOp`, constant rematerialization)
- `WSBuffer.cpp` (`appendAccumCntsForOps`, `getBufferIdxAndPhase`)
- `WarpSpecialization/docs/Overview.md`, `CodePartition.md`,
  `CodeSpecialization.md`, `BarrierInsertion.md`, `AccumulationCounters.md`
