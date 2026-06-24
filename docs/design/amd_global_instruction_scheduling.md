# AMD Warp-Pipelined Global Instruction Scheduling Algorithm (MI350 / gfx950 / CDNA4)

This document describes a scheduling algorithm for AMD CDNA4 (MI350, gfx950) GPU kernels. It is the AMD counterpart to [WS Global Instruction Scheduling](./ws_global_instruction_scheduling.md), which targets NVIDIA Hopper/Blackwell. The two share a common skeleton — a modulo-scheduling core feeding a reconstruction pass feeding a code generator — but the hardware model and the realization of the schedule differ substantially. Read the NVIDIA doc first; this document focuses on the AMD-specific design and assumes familiarity with the shared concepts (DDG, ScheduleGraph, modulo scheduling, II).

> **Status.** This is a design proposal. The AMD backend today has a stage/cluster software pipeliner (`ScheduleLoops.cpp` → `LowerLoops.cpp` → `Pipeline.cpp`), a marker-driven warp-pipeline lowering (`WarpPipeliner.cpp` → `ConvertWarpPipeline.cpp`), and a hand-written `BlockPingpong.cpp` scheduler — but **no latency cost model** (`ScheduleLoops.cpp:125` passes an empty `unusedOpLatency`). The algorithm here introduces a latency-driven global scheduler on top of that existing scaffolding. Wherever it relies on something that does not yet exist, it is called out explicitly.

## Table of Contents

- [Overview](#overview)
  - [Why AMD Is Different](#why-amd-is-different)
  - [Central Data Structure](#central-data-structure)
  - [Implementation Layer: ScheduleGraph](#implementation-layer-schedulegraph)
  - [Algorithm Summary](#algorithm-summary)
  - [Algorithm Flow](#algorithm-flow)
  - [Worked Examples](#worked-examples)
  - [Limitations and Assumptions](#limitations-and-assumptions)
- [Inputs](#inputs)
  - [1. Instruction Dependency Graph (DDG)](#1-instruction-dependency-graph-ddg)
  - [2. Op Lowering](#2-op-lowering)
  - [3. Functional Unit Mapping](#3-functional-unit-mapping)
  - [4. Latency Table](#4-latency-table)
  - [5. Resource Model](#5-resource-model)
- [Pass A: Modulo Scheduling](#pass-a-modulo-scheduling)
  - [Step 1: Compute Minimum Initiation Interval (II)](#step-1-compute-minimum-initiation-interval-ii)
  - [Step 2: Modulo Reservation Table Scheduling](#step-2-modulo-reservation-table-scheduling)
  - [Step 2.5: Compute Cluster IDs](#step-25-compute-cluster-ids)
  - [Step 3: Derive Per-Region Pipeline Depth](#step-3-derive-per-region-pipeline-depth)
  - [Step 4: LDS Budget and the vmcnt Window](#step-4-lds-budget-and-the-vmcnt-window)
  - [Step 4.5: Lifetime-Aware LDS Buffer Merging](#step-45-lifetime-aware-lds-buffer-merging)
  - [Step 4.7: Warp-Group Partitioning](#step-47-warp-group-partitioning)
  - [Step 4.8: Derive s_setprio Priorities (Modulo-Reservation Priority)](#step-48-derive-s_setprio-priorities-modulo-reservation-priority)
  - [Step 5: Emit ScheduleGraph](#step-5-emit-schedulegraph)
- [Pass A.5: Data Partitioning (Optional)](#pass-a5-data-partitioning-optional)
- [Pass A.6: Scheduling Non-Loop Regions](#pass-a6-scheduling-non-loop-regions)
- [Pass A.7: Epilogue Subtiling](#pass-a7-epilogue-subtiling)
- [Pass B: Warp-Pipeline Reconstruction](#pass-b-warp-pipeline-reconstruction)
- [Pass C: Code Generation and Instruction Ordering](#pass-c-code-generation-and-instruction-ordering)
- [Integration with the Existing AMD Backend](#integration-with-the-existing-amd-backend)
- [Worked Example: gfx950 Warp-Pipelined GEMM](#worked-example-gfx950-warp-pipelined-gemm)
- [Worked Example: Fused addmm + GLU](#worked-example-fused-addmm--glu)
- [Worked Example: Flash Attention Forward](#worked-example-flash-attention-forward)
- [NVIDIA vs AMD: Key Differences](#nvidia-vs-amd-key-differences)
- [Complexity](#complexity)

## Overview

This algorithm:

1. **Discovers** the near-optimal multi-pipeline instruction schedule using **modulo scheduling**, treating each CDNA4 functional unit (MEM, MFMA, VALU, SALU/transcendental) as an independent pipeline resource.
2. **Derives** the per-region pipelining scheme — LDS buffer depth, the in-flight `vmcnt` window, prologue/epilogue — from the modulo schedule.
3. **Reconstructs** the AMD warp-pipeline: which ops go in which **stage**, how the loop body is split into **clusters**, where `s_barrier`/`cond_barrier`/`s_waitcnt vmcnt` synchronization is inserted, and what `s_setprio` priorities each cluster carries.

The algorithm is inspired by the hand-tuned AMD TLX kernels in this tree — `amd_gemm_warp_pipeline.py`, `amd_fa_pipelined.py`, `amd-addmm-glu-opt_test.py`, `amd_tdm_gemm_pipelined.py` — and by the existing `BlockPingpong.cpp` scheduler. It formalizes the decisions those kernels make by hand (buffer depth, the `async_load_wait_group` count, the `warp_pipeline_stage` split, `s_setprio` priorities) into a systematic modulo-scheduling framework.

**The algorithm is implemented entirely as AMD backend compiler passes — it does not generate TLX.** It consumes a plain pipelined `scf.for` in ordinary **TTGIR** (no `tlx.*` annotations required — the same kind of input `BlockPingpong.cpp` operates on) and emits the fully lowered warp-pipeline IR directly: the border-split loop, the `cond_barrier` phase shift, `s_setprio`, `s_waitcnt vmcnt`, and `sched.group.barrier`, all the way down to ROCDL/LLVM. TLX primitives (`tlx.async_load`, `tlx.warp_pipeline_stage`, `tlx.local_alloc`) appear in this document only as a **human-readable rendering** of that IR, because they map one-to-one onto the AMD ops the passes emit; the actual artifacts are TTGIR attributes/ops and ROCDL intrinsics consumed and produced by the AMD lowering passes (`WarpPipeliner.cpp`, `ConvertWarpPipeline.cpp`, `LoadStoreOpToLLVM.cpp`, `UpdateAsyncWaitCount.cpp`). Nothing round-trips back up to the TLX frontend.

### Why AMD Is Different

CDNA4 has no direct analog of several Hopper/Blackwell primitives the NVIDIA algorithm leans on. The table below is the conceptual mapping that drives every adaptation in this document.

| Concept | NVIDIA (Hopper/Blackwell) | AMD (CDNA4 / gfx950) | Consequence for the scheduler |
|---|---|---|---|
| Warp width | 32-lane warp | **64-lane wavefront** (`getWarpSize`, TargetInfo.cpp:72) | Tile/occupancy math uses 64; "warp group" = a set of wavefronts on adjacent SIMDs |
| Matrix unit | `wgmma` / `tcgen05.mma` | **MFMA** (`V_MFMA_*`, MFMA.cpp:75; `MfmaGroup.cpp`) | `matrix_instr_nonkdim`, `kpack` shape knobs; latencies differ |
| MMA accumulator storage | **TMEM** (dedicated tensor memory) | **AGPR/VGPR registers** | No TMEM buffers to allocate or merge; accumulator pressure is a *register* (occupancy) constraint, not an LDS one |
| Scratch / staging memory | SMEM | **LDS** (160 KB/CU, `getSharedMemorySize`, TargetInfo.cpp:87) | Larger budget than Hopper SMEM; same depth-vs-budget tradeoff |
| Async bulk copy | TMA (`cp.async.bulk`) | **direct-to-LDS** `global_load_lds` / `buffer_load ... lds` (LoadStoreOpToLLVM.cpp:1020, BufferOpsEmitter.cpp:111) | No descriptor on gfx950; no LDS scattering — each wavefront writes a contiguous chunk (laneId forced to warp base, LoadStoreOpToLLVM.cpp:501) |
| Async completion | **mbarrier** (phase-based) | **`s_waitcnt vmcnt`** (count-based, AsyncWaitOpConversion, LoadStoreOpToLLVM.cpp:2305) | No phase cycling; correctness = "≤ N loads still in flight". Commit groups are bookkeeping markers; the vmcnt is computed by `UpdateAsyncWaitCount.cpp` |
| Cross-group sync | named barriers (0–15) | **AMD named barriers** (`amdgcn.named.barrier`, up to 17 → ~15 groups, ConvertWarpSpecializeToLLVM.cpp:38) and `s_barrier` / **`cond_barrier`** (asymmetric) | Warp-pipeline uses `cond_barrier` to phase-shift one group ahead |
| Per-group register budget | `setmaxnreg` | **none** (`reallocRegisters` is a no-op, ConvertWarpSpecializeToLLVM.cpp:168) | Register budget is global per-CU; controlled by `waves_per_eu` occupancy, not per-group |
| Specialization model | **producer/consumer WS** (different code per warp group) | **warp-pipelining** (same staged code, two warp groups phase-offset by one stage) | Pass B reconstructs a *symmetric* phase-shifted pipeline, not asymmetric roles |
| Instruction ordering primitive | compiler ordering | **`sched.group.barrier`, `iglp_opt`, `s_setprio`, `sched.barrier`** (SchedInstructions.cpp; BlockPingpong.cpp:897) | Pass C emits these to realize the per-cluster order |

The single most important difference: **AMD's dominant specialization mechanism is warp-pipelining, not warp specialization.** `tlx.warp_pipeline_stage` (`warp_pipeline.py:1-46`, AMD-only, gated to HIP at code_generator.py:360) splits the loop body into stages and runs two warp groups *one stage apart* over the **same** code. The NVIDIA `tlx.async_tasks` producer/consumer model does lower on AMD (`ConvertWarpSpecializeToLLVM.cpp`), but every AMD tutorial kernel uses warp-pipelining, and without `setmaxnreg` the register-rebalancing benefit of true WS is absent. This algorithm therefore targets warp-pipelining as its primary output; warp specialization is treated as the degenerate case where stages are assigned disjoint pipelines.

### Central Data Structure

As in the NVIDIA design, the algorithm's central output is the **ScheduleGraph** — a DDG-based graph that accumulates all scheduling and resource decisions without mutating the IR. Each scheduled op carries a `(cycle, pipeline, stage, cluster)` tuple:

- **cycle**: when the op starts (within the II-length reservation table for loop regions; absolute for non-loop regions).
- **pipeline**: which CDNA4 unit executes it — **MEM, MFMA, VALU, SALU**, or NONE.
- **stage**: how many II periods the op is deferred relative to its owning iteration. On AMD this maps directly to the **warp-pipeline stage** and to the `cond_barrier` phase offset between the two warp groups.
- **cluster**: within-stage ordering derived from cycle. The cluster IDs become `warp_pipeline_stage` boundaries (`triton.warp_pipeline.border`) and `sched.group.barrier` groupings in Pass C.

Beyond per-op scheduling, the ScheduleGraph carries AMD-specific resource decisions:

- **LDS buffers** (`ScheduleBuffer`, `kind=LDS`) with shape, element type, buffer count, modular live interval, merge-group ID, and — crucially — a **padding policy** for bank-conflict avoidance (`tlx.padded_shared_layout_encoding`, types.py:111).
- **vmcnt counters** instead of paired phase barriers. A multi-buffered LDS allocation is associated with a target in-flight count `W` (the `async_load_wait_group(W)` argument); there is no per-buffer barrier object because completion is counted globally per wavefront.
- **Warp-group / stage assignments** and prologue/epilogue structure.

There are no TMEM buffers: MFMA accumulators live in registers, so accumulator depth shows up as a register-pressure annotation (feeding `waves_per_eu`), not as an allocated buffer.

### Implementation Layer: ScheduleGraph

The ScheduleGraph is built from the DDG and points into the TTGIR via `Operation*`. The AMD-specific field mapping:

| Type | Role | AMD lowering target |
|---|---|---|
| **ScheduleBuffer** (`kind=LDS`) | Multi-buffered LDS allocation: shape, elem type, count, live interval, merge group, padding policy | `ttg.local_alloc` with a `ttg.padded_shared` / swizzled encoding |
| **ScheduleNode** | A scheduled op with cycle/stage/pipeline/latency, buffer refs, warp-group, and an optional `s_setprio` priority | `ttg.async_copy_global_to_local` / `tt.dot` (→MFMA) / `ttg.local_load` (→`ds_read`) / `tt.store` |
| **ScheduleEdge** | Producer–consumer dependency with latency and loop-carried distance | enforced at runtime by `s_waitcnt vmcnt` / `s_barrier` placement |
| **ScheduleLoop** | A pipelined `scf.for` with II, maxStage, trip count, nodes, edges, LDS buffers, and the in-flight vmcnt window | a `tl.range(..., num_stages=1)` loop carrying `warp_pipeline_stage` regions |
| **ScheduleGraph** | Forest of ScheduleLoops with bottom-up order | the complete kernel |

**Phase mapping (mirrors NVIDIA, with AMD substitutions):**

```
Phase 0 (Schedule):   DDG + Rau's → ScheduleNode.cycle/stage
Phase 1 (Buffers):    Stage diffs → ScheduleBuffer.count + vmcnt window
Phase 1.5 (Partition):Separation cost + makespan → ScheduleNode.warpGroup/stage
                      MRT slack + occupancy      → ScheduleNode.s_setprio
Phase 2 (Expand):     Bottom-up → prologueNodes/epilogueNodes
Phase 3 (Lower):      ScheduleGraph → async copies + vmcnt waits + cond_barrier
                      + warp_pipeline_stage borders + sched.group.barrier + s_setprio
```

Phase 3 is where AMD diverges most: instead of emitting `mbarrier.{init,arrive,wait}`, it emits `global_load_lds` copies whose completion is tracked by `s_waitcnt vmcnt`, with the count back-filled by the existing `UpdateAsyncWaitCount` pass.

### Algorithm Summary

**Pass A — Scheduling (iterative).** Schedules all regions, derives LDS depths, checks the LDS budget, partitions ops into warp-pipeline stages/groups, and applies DDG transformations — re-running until stable. Loop regions use modulo scheduling (Rau's algorithm) to minimize II; non-loop regions use list scheduling. From the schedule it derives LDS buffer depths and the per-loop **vmcnt window** (Step 3), merges LDS buffers with non-overlapping lifetimes (Step 4.5), runs a kernel-wide LDS budget check (Step 4), and partitions ops into warp groups/stages using latency-aware multi-pipeline clustering (Step 4.7), then derives each cluster's `s_setprio` priority from the modulo reservation table (Step 4.8). DDG transformations — data partitioning (A.5) and epilogue subtiling (A.7) — can trigger a re-schedule. Converges in 1–2 iterations.

**Pass B — Warp-Pipeline Reconstruction.** Reads the stage/group partition from the ScheduleGraph and reconstructs the AMD two-warp-group phase-shifted pipeline: it groups same-stage ops into clusters, inserts the `cond_barrier` phase shift in the prelude (one group runs a stage ahead), places `s_waitcnt vmcnt` waits and `s_barrier` cluster barriers where the modulo schedule requires cross-cluster ordering, and assigns each cluster an `s_setprio` priority. This is the analog of NVIDIA Pass B, but symmetric (both groups run the same code) rather than role-specialized.

**Pass C — Code Generation and Instruction Ordering.** Takes the `(stage, cluster)` assignments and emits the prologue/kernel/epilogue loop with `warp_pipeline_stage` borders, then emits the per-cluster instruction-interleaving hints (`sched.group.barrier` to interleave MFMA with VALU/`ds_read`, optionally `iglp_opt`) and `s_setprio` transitions. The `vmcnt` values are finalized by `UpdateAsyncWaitCount`.

### Algorithm Flow

```
┌─────────────────────────────────────────────────────┐
│  Input: Kernel with loop and non-loop regions       │
│         DDG per region, AMD latency table, LDS budget│
└──────────────────────┬──────────────────────────────┘
                       ▼
┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
│         Pass A: Iterative Scheduling Loop           │
│   Schedule regions (modulo / list) → cluster IDs    │
│   Step 3: LDS depths + vmcnt window                 │
│   Step 4.5: merge non-overlapping LDS buffers       │
│   Step 4:  kernel-wide LDS budget check (160 KB)    │
│   Step 4.7: warp-group / stage partitioning         │
│   Step 4.8: s_setprio from modulo MRT (slack + occ) │
│   DDG transforms: A.5 data partition, A.7 subtile   │
│         ── any DDG changed? ── yes → re-run ─────────│
└ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤ no (converged) ─ ─ ─ ─ ─┘
                          ▼
┌─────────────────────────────────────────────────────┐
│  Step 5: Emit ScheduleGraph                         │
│   cycles, stages, LDS buffers + lifetimes + padding, │
│   merge groups, vmcnt windows, warp-group/stage      │
└──────────────────────┬──────────────────────────────┘
                       ▼  ScheduleGraph
┌─────────────────────────────────────────────────────┐
│  Pass B: Reconstruct warp-pipeline                  │
│   Read stages/groups; split body into clusters;     │
│   cond_barrier phase-shift; place s_waitcnt vmcnt +  │
│   s_barrier; assign s_setprio per cluster           │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Pass C: Code generation + instruction ordering     │
│   Loop: expand prologue/kernel/epilogue with         │
│   warp_pipeline_stage borders                        │
│   Per cluster: sched.group.barrier / iglp_opt /      │
│   s_setprio; UpdateAsyncWaitCount finalizes vmcnt    │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Output: warp-pipelined CDNA4 kernel                 │
└─────────────────────────────────────────────────────┘

Convergence: typically 1-2 iterations (same argument as NVIDIA —
DDG transforms are idempotent and monotonically improving).
```

### Worked Examples

Three examples of increasing complexity, all mapping to existing AMD TLX kernels:

1. **gfx950 Warp-Pipelined GEMM** (`amd_gemm_warp_pipeline.py`, `amd-addmm-glu-opt_test.py`): 2 active pipelines (MEM, MFMA), 3-deep LDS pipeline, two warp groups split into `mfma`/`mem` clusters. The canonical case.
2. **Fused addmm + GLU** (`amd-addmm-glu-opt_test.py`): GEMM body plus a fused register-resident epilogue (bias add + GLU `x + x*y`) with an async-prefetched, **padded** Y tile in LDS.
3. **Flash Attention Forward** (`amd-fa-pipelined_test.py`): 3 active pipelines (MEM, MFMA, VALU/transcendental), accumulator recurrence. Four variants from single-buffer (`async_simple`) up to the **rotated 4-cluster `warp_pipeline_stage`** kernel (`cluster`) — the full worked example: a depth-4 schedule over a 2-slot LDS buffer with the exp2 burst overlapped onto the PV MFMA.

### Limitations and Assumptions

1. **No existing latency model.** The AMD backend currently schedules by stage/cluster distance, not cost (`unusedOpLatency` is empty, ScheduleLoops.cpp:125). The latency table in this doc (MFMA, `ds_read`, `global_load_lds`) must be measured by microbenchmark and is net-new. The schedule is only as good as those numbers.
2. **vmcnt is coarser than mbarrier.** `s_waitcnt vmcnt(n)` waits until at most `n` *vector-memory* operations are outstanding — it cannot name *which* buffer completed, only how many loads remain. The scheduler must order independent loads so that "≤ n outstanding" implies "buffer k is ready" (the existing kernels rely on FIFO completion of `global_load_lds`). `UpdateAsyncWaitCount.cpp` computes the count by backward def-chain analysis; this doc assumes that pass remains the source of truth for the final count.
3. **No LDS scattering on gfx950.** Direct-to-LDS requires each wavefront to write a contiguous LDS chunk (LoadStoreOpToLLVM.cpp:501); valid load widths are `{128, 32}` bits (TargetInfo.cpp:702). Tile/layout choices that would need scatter are infeasible and must be excluded during op lowering.
4. **No per-group register budget.** Without `setmaxnreg`, the algorithm cannot rebalance registers between warp groups. Accumulator depth and live-register counts feed a single global `waves_per_eu` occupancy estimate; over-allocation causes VGPR spills the schedule did not anticipate.
5. **Barrier/`s_setprio` overhead not modeled in Pass A.** The cost of `s_barrier`, `cond_barrier`, and priority transitions is not in the latency table. For kernels with many cluster boundaries this shifts actual timings.
6. **Static latencies / no dynamic scheduling**, **single-CTA**, and **approximate register allocation** — same caveats as the NVIDIA design.

---

## Inputs

### 1. Instruction Dependency Graph (DDG)

Identical in structure to the NVIDIA DDG: nodes are operations, intra-iteration edges (distance=0) and loop-carried edges (distance=d) carry `(latency, distance)`. AMD example (warp-pipelined GEMM K-loop):

```
load_A ──→ ds_read_A ──→ mfma ──→ acc(next iter)
load_B ──→ ds_read_B ──↗
Loop-carried (distance=1):
  acc ──→ mfma (next iter)         // accumulator lives in registers
```

The accumulator recurrence is the same shape as NVIDIA's, but the accumulator is a **register value**, not a TMEM tile — so its only resource cost is register pressure.

### 2. Op Lowering

As in the NVIDIA design, DDG nodes are *lowered* to expose AMD-specific detail the IR does not represent. Op lowering does not modify the IR.

**Why lower (AMD specifics):**

- **`selfLatency` ≠ `latency`** for async ops. A `global_load_lds` issues in a few cycles (it occupies the MEM issue slot briefly) but the data is not in LDS until the full DRAM round-trip completes. The scheduler reserves `selfLatency` slots on MEM and uses `latency` as the edge weight to the `ds_read`/MFMA consumer.
- **Symbolic, unaliased LDS buffers.** Buffers are named (`buf_A`, `buf_B`), with index arithmetic (`i % NUM_BUFFERS`) deferred to Pass C — exactly as NVIDIA. This is what lets Step 4.5 merge buffers.
- **Direct-to-LDS has no intermediate register store.** `global_load_lds` writes straight to LDS, so the `tma_load`-equivalent node (`async_copy`) is itself the LDS buffer producer (`→buf`); there is no `local_store` between global memory and LDS. The synthetic `ds_read` (`local_load`) node is the consumer (`←buf`) that ends the buffer lifetime.

**DDG node → IR mapping (AMD):**

| DDG Node | `irOp` | Buffer Ref | Lowers to |
|---|---|---|---|
| `async_copy` (real) | `ttg.async_copy_global_to_local` | `→buf` | `global_load_lds` / `buffer_load ... lds` (LoadStoreOpToLLVM.cpp:1020, BufferOpsEmitter.cpp:111) |
| `ds_read` (synthetic, was `local_load`) | NULL | `←buf` | `ds_read` (ends lifetime; drives `vmcnt` placement) |
| `mfma` (real) | `tt.dot` → `ttng`-equivalent | — | `V_MFMA_*` (MFMA.cpp:75) |
| `local_store` (real, epilogue) | `ttg.local_store` | `→buf` | `ds_write` |
| `store` (real, epilogue) | `tt.store` | `←buf` | `global_store` / `buffer_store` |

There is **no async commit edge** that gates correctness: `ttg.async_commit_group` lowers to a dummy value on AMD (LoadStoreOpToLLVM.cpp:2366) and exists only so `UpdateAsyncWaitCount` can deduce the `vmcnt`. The DDG models the ordering constraint directly as a latency edge from `async_copy` to `ds_read`.

### 3. Functional Unit Mapping

Each op is assigned to exactly one CDNA4 pipeline:

| Pipeline | Operations |
|---|---|
| **MEM** | `global_load_lds` (async copy), `buffer_load`, `ds_read`/`ds_write` (LDS), global stores |
| **MFMA** | `V_MFMA_*` matrix instructions (the `tl.dot` lowering) |
| **VALU** | vector ALU: rowmax, rowsum, scale, accumulator update, type conversions, GLU mul/add |
| **SALU** | scalar ALU and the transcendental approximations that run on the VALU transcendental path (exp2, rsqrt) — AMD has no separate SFU; model transcendentals as long-latency VALU |
| **NONE** | synthetic buffer-lifetime endpoints |

Note the merge of "CUDA core" and "SFU" relative to NVIDIA: CDNA4 executes transcendentals on the vector unit, so SFU is not a physically separate pipeline. Keep transcendentals as a distinct *logical* pipeline only if microbenchmarks show they can co-issue with plain VALU; otherwise fold them into VALU (this is a measurement question — see Limitations).

### 4. Latency Table

**Net-new — must be measured on gfx950.** Representative placeholders (cycles); the real values come from microbenchmarks and depend on tile shape, `matrix_instr_nonkdim`, and `kpack`:

| Operation | selfLatency (issue) | latency (result) | Pipeline |
|---|---:|---:|---|
| `global_load_lds` 128×64 fp16 | ~40 | ~600 (DRAM round-trip) | MEM |
| `ds_read` 128×64 fp16 | ~50 | ~50 | MEM |
| `ds_write` (epilogue) | ~50 | ~50 | MEM |
| MFMA 16×16×16 fp16 | ~16 | ~64 | MFMA |
| MFMA 32×32×8 fp16 | ~32 | ~128 | MFMA |
| rowmax / rowsum (128×128) | varies | = selfLatency | VALU |
| exp2 (elementwise) | varies | = selfLatency | VALU(transcendental) |

`selfLatency` is the issue cost (how long the SIMD's dispatch slot is occupied). For `global_load_lds` it is small because the load executes asynchronously and completion is observed via `vmcnt`. `latency` is the edge weight to consumers. The MFMA `latency`/`selfLatency` ratio determines how many MFMAs are needed to hide one `global_load_lds` — i.e. the minimum useful pipeline depth.

### 5. Resource Model

- Each pipeline executes **one op at a time per warp group** (per SIMD).
- Distinct pipelines **overlap**: MEM + MFMA + VALU concurrent.
- An op occupies its pipeline for `selfLatency`, not full `latency` (async ops free the issue slot immediately).
- **LDS budget: 160 KB per CU** (`getSharedMemorySize`, TargetInfo.cpp:87) — substantially larger than Hopper's 228 KB-shared-across-more-warps reality, and far larger than the 64 KB of older CDNA. This budget gates total LDS buffer depth across all regions.
- **Registers, not TMEM, hold accumulators.** Accumulator depth and live-value count feed an occupancy estimate (`waves_per_eu`, compiler.py:49). There is no separate accumulator-memory budget.
- **vmcnt window**: the hardware tracks outstanding vector-memory ops via a 6-bit `vmcnt` (LoadStoreOpToLLVM.cpp:2305-2323). The scheduler must keep the number of in-flight `global_load_lds` ops within range and emit `s_waitcnt vmcnt(n)` to bound it.
- **Wavefront = 64 lanes**; a warp-pipeline group is `warpSize * 4` threads = 4 SIMDs × (1 wave/SIMD per group), with two groups per block (ConvertWarpPipeline.cpp:951).

---

## Pass A: Modulo Scheduling

Pass A is an iterative refinement loop, identical in control flow to the NVIDIA design (`pass_a` pseudocode in [ws_global_instruction_scheduling.md](./ws_global_instruction_scheduling.md#pass-a-scheduling-iterative)). Only the resource-model substitutions below differ.

### Step 1: Compute Minimum Initiation Interval (II)

`MinII = max(ResMII, RecMII)`.

**ResMII** — busiest pipeline. For a gfx950 GEMM K-loop (128×128 tile, BK=64):

```
MEM:  load_A(600) + load_B(600) + ds_read_A + ds_read_B   ≈ 1300
MFMA: sum of MFMA latencies for the BK=64 slice            ≈ 1100
ResMII = max(1300, 1100) = 1300   (MEM-bound)
```

GEMM on AMD is typically MEM-bound (as on NVIDIA), so the scheduler's job is to overlap enough K-iterations that MFMA hides the `global_load_lds` latency. This is exactly what the hand-written kernel encodes with `NUM_BUFFERS=3` and `async_load_wait_group(1)`.

**RecMII** — recurrence circuits, computed identically. For GEMM the only loop-carried edge is the register accumulator (`acc[i] → mfma[i+1]`, distance=1); its latency is the MFMA latency, well below ResMII, so GEMM stays MEM-bound. For Flash Attention the accumulator/softmax recurrence dominates, as on NVIDIA.

### Step 2: Modulo Reservation Table Scheduling

Rau's iterative modulo scheduling, unchanged. The reservation table has one row per AMD pipeline (MEM, MFMA, VALU, SALU). Loop-carried edges use the same `consumer_start ≥ producer_start + latency − d·II` constraint.

The **stage** assigned here is what becomes the warp-pipeline phase offset in Pass B: an op at stage `s` runs `s` II-periods after the iteration it belongs to, which the AMD lowering realizes by running the second warp group `s` stages ahead via `cond_barrier`.

### Step 2.5: Compute Cluster IDs

Within each stage, ops are assigned dense cluster IDs sorted by cycle. On AMD these cluster IDs have a direct lowering: they become `warp_pipeline_stage` boundaries. In `amd_gemm_warp_pipeline.py` the author writes exactly two clusters per iteration — `warp_pipeline_stage("mfma", priority=0)` then `warp_pipeline_stage("mem", priority=1)` — which is the cluster partition this step would compute automatically (MFMA ops in cluster 0, the async loads + `ds_read` of the next tile in cluster 1).

### Step 3: Derive Per-Region Pipeline Depth

LDS buffer depth from stage diffs, identical formula:

```
num_buffers(R) = floor(lifetime(R) / II) + 1
```

Plus the AMD-specific **vmcnt window**: the `async_load_wait_group(W)` argument is `W = (num_buffers − 1) × loads_per_iter − loads_consumed_this_iter`, i.e. how many `global_load_lds` ops may remain in flight after the wait. For the 3-buffer GEMM, the kernel issues 2 loads/iter and keeps the most-recent group in flight, so `W=1` group → the kernel calls `async_load_wait_group(1)` and the backend's `UpdateAsyncWaitCount` converts that to a concrete `vmcnt`. The comment in `amd-addmm-glu-opt_test.py:207-210` ("3 buffers lets the in-loop wait skip waiting on the directly-previous global read") is precisely this window calculation.

### Step 4: LDS Budget and the vmcnt Window

Kernel-wide LDS budget check, after all regions have depths:

```
total_LDS = Σ_buffers (count × padded_size_bytes)   ≤ 160 KB   (per CU)
```

`padded_size_bytes` must include the bank-conflict padding (Step 4.5 padding policy), so the budget check uses the *physical* padded footprint, not the logical tile size. If over budget, reduce the deepest pipeline first (same greedy reduction as NVIDIA), which raises II.

Unlike NVIDIA, **there is no TMEM budget** — accumulators are registers. Instead, a parallel *register* check estimates VGPR/AGPR pressure from accumulator depth + live values and lowers `waves_per_eu` if needed. Over-pressure here causes spills rather than a hard allocation failure.

### Step 4.5: Lifetime-Aware LDS Buffer Merging

Identical modular live-interval analysis: two LDS buffers can share one physical allocation if their live intervals don't overlap across any in-flight iteration. The physical size is `max(padded_size)` and count is `max(count)`.

AMD adds a **padding-compatibility** check: merged buffers must share a padding policy (or the merged allocation uses the most-padded layout), because the physical `ttg.padded_shared` encoding is per-allocation. Bank-conflict padding (`tlx.padded_shared_layout_encoding`, e.g. the `_y_padded_layout` in addmm-glu) changes the byte footprint, so merging two differently-padded buffers can cost more than keeping them separate — the merge is only accepted if it still reduces total LDS.

### Step 4.7: Warp-Group Partitioning

Latency-aware multi-pipeline clustering, as in NVIDIA Step 4.7 — compute a **separation cost** per cross-pipeline edge (barrier overhead relative to the cycle gap) and validate merged groups via multi-pipeline makespan. The AMD-specific realities:

- **Two groups, symmetric.** The dominant output is a two-group warp-pipeline (`threadsPerPipelineGroup = warpSize × 4`, ConvertWarpPipeline.cpp:951), where both groups run the same clusters phase-offset by one stage. This is the `BlockPingpong`/`warp_pipeline_stage` model: MFMA in one cluster, MEM in the other, the two groups ping-ponging so that while group 0 does MFMA, group 1 issues the next loads.
- **Barrier currency is `s_barrier` + `vmcnt`, not mbarrier.** Separation cost uses the `s_barrier`/`cond_barrier` cost. Because cross-cluster ordering on the MEM side is enforced by `s_waitcnt vmcnt` (cheap, no full block barrier) while LDS read/write ordering needs `ds`-wait + `s_barrier`, the cost model must distinguish the two — a MEM→MFMA edge across clusters is cheaper to separate than an LDS-RAW edge.
- **No register rebalancing benefit.** Without `setmaxnreg`, splitting into disjoint-pipeline groups does not free registers for the compute group. So the partitioner should prefer **fewer, phase-offset symmetric groups** (warp-pipelining) over many asymmetric specialized groups, unless a pipeline is so underutilized that a dedicated group still wins after paying full barrier cost.

The result for GEMM: a single two-group warp-pipeline with `{MFMA}` and `{MEM}` clusters. For FA: the same two groups, with VALU/transcendental ops co-scheduled into the compute cluster (mixed MFMA+VALU), mirroring how `BlockPingpong` interleaves ~3 SALU/VALU per MFMA via `sched.group.barrier` (BlockPingpong.cpp:897-901).

### Step 4.8: Derive s_setprio Priorities (Modulo-Reservation Priority)

`s_setprio` is the per-cluster issue priority Pass C emits. This step computes it **from the modulo reservation table (MRT) built in Step 2** — not from a memory-vs-compute label. The "memory cluster → 1, compute cluster → 0" rule that `BlockPingpong.cpp` hard-codes (BlockPingpong.cpp:61-62, 678-707) is the *output* of this derivation on GEMM/FA, not its premise.

**The lever.** The two warp groups run the same clusters ping-ponged by an offset Δ (Step 4.8b); at every cycle one group is in cluster `c` and the other in `c⊖Δ`. Where both occupy the **same** pipeline — almost always VALU (address arithmetic in the mem cluster, softmax/scale in the compute cluster) — only one can issue, and `s_setprio` picks it. Two quantities read off the MRT decide the right pick:

- **Monopolization** `M(c)` — `c`'s occupancy of the *contended* pipeline, straight from the MRT: `M(c) = Σ_p contended(p)·occ(c,p)`. A cluster dense on the contended unit must get *low* priority: if it wins the slot it monopolizes issue and the opposite warp cannot make progress, collapsing the overlap (BlockPingpong.cpp:689-692).
- **Issue urgency** `U(c)` — how much delaying `c`'s issue grows the makespan: the modulo slack of its ops plus the latency-criticality of any async load it issues (a `global_load_lds` must be issued early enough that its ~600-cycle round-trip is hidden). `U(c) = max_{op∈c} [ crit(op) + isAsyncLoad(op)·latencyPressure(op) ]`.

**Step 4.8a — MRT occupancy & slack.** From the Step 2 schedule, for each cluster `c` and pipeline `p` take `occ(c,p)` = cycles of `p` used within the II window. Under the same modulo constraints (`consumer ≥ producer + latency − d·II`) compute `slack(op) = ALAP(op) − ASAP(op)` and `crit(op) = 1 − slack(op)/(maxSlack + ε)` ∈ [0,1]. Ops on the recurrence circuit κ* that set RecMII have `crit ≈ 1`; ops that are only resource-bound (they set ResMII but lie on no tight cycle — e.g. the MFMA in a MEM-bound GEMM) keep positive slack and lower `crit`.

**Step 4.8b — Ping-pong offset Δ.** The offset is *derived*, not assumed to be "one stage": it is the cluster shift that minimizes contention in the overlapped MRT.

```
Δ* = argmin_Δ  Σ_c Σ_p  max(0, occ(c,p) + occ(c⊖Δ,p) − cap(p))
```

For the 2-cluster GEMM body Δ*=1 (≈ II/2: `mfma` opposite `mem`); for FA's 4-cluster body Δ*=2 (`dot1`↔`dot2`, `mem1`↔`mem2` never co-occupy). `contended(p)` is the set of pipelines with residual co-occupancy at Δ*. This Δ* is what the rest of the doc calls the one-stage phase offset.

**Step 4.8c — Priority.** Rank clusters by

```
pscore(c) = U(c) − λ·M(c)
```

and quantize to the `s_setprio` 0–3 range by rank; the two-group ping-pong collapses this to {0,1}. The value is stored on each cluster's ScheduleNodes (`s_setprio` field) and realized verbatim in Pass C.

**Reduction to mem=1 / dot=0 (why it matches the hand kernels).** On GEMM and FA the compute cluster is dense on the contended VALU/MFMA units (`M` high) and self-paced, while the mem cluster is sparse on VALU (`M` low) but issues the latency-critical `global_load_lds` (`U` high). So `pscore(mem) > pscore(dot)` → **mem=1, dot=0**, matching every hand-tuned kernel — computed from the MRT, not labelled. The rule is general: it elevates whichever cluster is the *sparse co-issuer* on the contended unit, so a compute-sparse / memory-dense loop (heavy LDS packing, tiny MFMA) flips the assignment — which a fixed op-type heuristic cannot.

### Step 5: Emit ScheduleGraph

Package all decisions: cycles, stages, LDS buffers with lifetimes + padding policy + merge groups, vmcnt windows, the two-group/stage partition, and the per-cluster `s_setprio` priorities. This graph is the sole input to Pass B.

---

## Pass A.5: Data Partitioning (Optional)

Same intent as NVIDIA: split an underutilized loop op into sub-tiles to expose parallelism. On AMD the most common application is **splitting a wide MFMA into k-sub-tiles** (`pingpong_2step`, BlockPingpong.cpp:646) so the two warp groups can interleave half-MFMAs with memory, rather than one warp group stalling on a monolithic MFMA. As on NVIDIA, this is a DDG transform that triggers a re-schedule.

## Pass A.6: Scheduling Non-Loop Regions

List scheduling for prologue/epilogue/straight-line code, `stage=0`, output format `(cycle, pipeline, 0, cluster)`. Used for the GEMM/addmm epilogue (bias add + GLU) and for the prologue that issues the first `NUM_BUFFERS` async loads.

## Pass A.7: Epilogue Subtiling

Split a monolithic global store into independent sub-chains so the store traffic overlaps compute. On AMD the epilogue is register→(optional LDS via `ds_write`)→global store; subtiling lets `ds_write` of sub-tile `s+1` overlap the `buffer_store` of sub-tile `s`. The freed LDS may enable a deeper K-loop pipeline, triggering a re-schedule.

---

## Pass B: Warp-Pipeline Reconstruction

This is where AMD diverges most from NVIDIA Pass B. NVIDIA reconstructs *asymmetric producer/consumer* warp groups; AMD reconstructs a *symmetric, phase-offset* warp-pipeline. Pass B makes no scheduling decisions — it realizes Pass A's stage/cluster/group assignments.

**Step 1 — Read stages and clusters.** From the ScheduleGraph, read each op's `(stage, cluster, warpGroup)`. Group same-stage ops into clusters by cluster ID. Each cluster becomes a `warp_pipeline_stage` region (`triton.warp_pipeline.border` marker, the input to `WarpPipeliner.cpp`).

**Step 1.5 — Replicate shared infrastructure ops.** Ops with `pipeline == NONE` (index math, loop-invariant address computation) are cloned into each cluster as needed, identical to NVIDIA.

**Step 2 — Insert synchronization.** Three AMD mechanisms, chosen per edge:

- **`s_waitcnt vmcnt(n)`** for async-load completion: placed before the `ds_read`/MFMA that consumes an in-flight `global_load_lds`. The concrete `n` is left symbolic (`async_load_wait_group(W)`) and finalized by `UpdateAsyncWaitCount.cpp` via backward def-chain analysis. This replaces the entire NVIDIA mbarrier wait/arrive pairing — there are **no per-buffer barrier objects**.
- **`s_barrier` (+ `ds`-wait)** for LDS read-after-write / write-after-read ordering across clusters, inserted only where the LDS-dependency analysis (`analyzePipelineDependencies`, ConvertWarpPipeline.cpp:198) finds a genuine LDS hazard.
- **`cond_barrier`** for the phase shift itself: the prelude (`emitPipelinePrelude`, ConvertWarpPipeline.cpp:277) flushes LDS, computes `warpID`, and issues `cond_barrier(warpHigh)` so the high warp group runs one stage ahead. The postlude (`emitPipelinePostlude`) resets `s_setprio 0` and issues `cond_barrier(warpLow)` to reconverge.

**Step 3 — Compute prologue/epilogue structure.** Prologue depth = max stage across all ops (drains the pipeline fill); epilogue drains the in-flight loads. Same loop-expansion math as NVIDIA, realized by the prologue async-load sequence + the `tl.static_range` drain loop seen in the hand-written kernels.

**Step 4 — Assign warp counts and priorities.** Two warp groups (`warpSize × 4` threads each). **No register reallocation** (`reallocRegisters` is a no-op, ConvertWarpSpecializeToLLVM.cpp:168) — instead, each cluster carries the `s_setprio` priority computed in [Pass A Step 4.8](#step-48-derive-s_setprio-priorities-modulo-reservation-priority) and stored on its ScheduleNodes; Pass B only threads it through to `warp_pipeline_stage(..., priority=)` (it makes no priority decision of its own). Occupancy is set via `waves_per_eu`.

**Step 5 — Emit the warp-pipeline IR (backend, not TLX).** Write the result directly into TTGIR: the prologue async-copy sequence, the `scf.for` body split into `scf.execute_region` clusters carrying `triton.warp_pipeline.border` (cluster label) + `triton.warp_pipeline.priority` (from Step 4.8) attributes, the inter-cluster `s_waitcnt vmcnt` / `s_barrier` ops, and the epilogue drain — then hand off to `WarpPipeliner.cpp` → `ConvertWarpPipeline.cpp` for the `cond_barrier` phase-shift lowering down to ROCDL. The borders, priorities, and vmcnt waits are **IR attributes and ops, never `tlx.*` calls**. The emitted structure is identical to what `amd_gemm_warp_pipeline.py` writes by hand, which is the only reason the worked examples below render it as TLX for readability.

---

## Pass C: Code Generation and Instruction Ordering

Pass C takes the `(stage, cluster)` assignments and the border-tagged loop from Pass B and produces the final TTGIR→LLVM ordering. It makes no scheduling decisions.

- **Loop regions:** expand prologue/kernel/epilogue. The `WarpPipeliner.cpp` pass splits the `scf.for` body at `warp_pipeline.border` markers into `scf.execute_region` clusters; `ConvertWarpPipeline.cpp` lowers them to the phase-shifted two-group schedule.
- **Within-cluster instruction interleaving:** emit `sched.group.barrier` (`ROCDL::SchedGroupBarrier`, SchedInstructions.cpp; BlockPingpong.cpp:897) to interleave MFMA with VALU/`ds_read` at a chosen ratio (e.g. ~3 VALU per MFMA), bracketed by `sched.barrier` so the LLVM scheduler cannot move ops across cluster boundaries. Optionally emit `iglp_opt` for the attention pattern (the only `instruction_sched_hint` variant currently implemented, SchedInstructions.cpp:82). Emit `s_setprio` transitions at cluster entry/exit (`ROCDL::SetPrioOp`).
- **vmcnt finalization:** `UpdateAsyncWaitCount.cpp` replaces each `ttg.async_wait` (commit-group count) with `amdgpu.async_wait` (intrinsic count) by counting direct-to-LDS instructions on the def chain — this produces the concrete `s_waitcnt vmcnt` value.

### Relationship Between Pass A and Pass C

Pass A decides *what overlaps* (stages, clusters, depths); Pass C decides *how the ISA expresses it* (`sched.group.barrier` ratios, `s_setprio`, `vmcnt`). The cluster IDs from Pass A Step 2.5 are the contract between them — Pass C never reorders across a cluster boundary.

---

## Integration with the Existing AMD Backend

This algorithm does not replace the AMD pass pipeline — it slots into it. The entire warp-pipeline is held together by **one IR attribute contract**, not by TLX, which is what makes the integration small.

### The seam: the border-marker attribute contract

The whole mechanism is carried by two attributes on a marker op in the loop body:

> **border marker** = an op (a `rocdl.sched.barrier`) carrying `triton.warp_pipeline.border` (StringAttr, the cluster label) and an optional `triton.warp_pipeline.priority` (IntegerAttr).

- The TLX frontend only ever *emits* these via `create_warp_pipeline_border`.
- `WarpPipeliner.cpp` *reads* them (`readBorderMarker`, WarpPipeliner.cpp:61/64), splits the loop body at borders into clusters, and propagates the priority onto each `scf.execute_region` (WarpPipeliner.cpp:157).
- `ConvertWarpPipeline.cpp` turns the priority into `ROCDL::SetPrioOp(intAttr.getInt())` (ConvertWarpPipeline.cpp:253) and emits the `cond_barrier(warpHigh/warpLow)` phase shift itself (ConvertWarpPipeline.cpp:295/305).

So connecting this algorithm to the backend reduces to: **a new TTGIR pass writes these two attributes onto marker ops; everything downstream is reused unchanged, and no `tlx.*` op is ever produced.**

### Pass A/B/C mapped onto the real pipeline

The pass order in `third_party/amd/backend/compiler.py` (`make_ttgir`, then `make_llir`):

| This doc | Existing AMD pass (call site) | Status |
|---|---|---|
| **Pass A** — modulo schedule, II, buffer depth, stage, cluster | `add_schedule_loops` (ScheduleLoops.cpp, make_ttgir:320) + `add_pipeline` (LowerLoops/Pipeline, make_ttgir:321) | **scaffolding exists, cost model missing** |
| **Pass A, Step 4.8** — `s_setprio` priority | nobody computes it automatically (only `BlockPingpong.cpp`'s hard-coded template) | **net-new**; output is the `.priority` integer |
| **Pass B** — emit borders + sync | marker emission (today done by the TLX frontend) + `add_warp_pipeline` (WarpPipeliner.cpp, make_ttgir:370) splits clusters | WarpPipeliner exists; **marker emission is the new piece** (it replaces the human writing `warp_pipeline_stage`) |
| **Pass C** — lower to `cond_barrier`/`s_setprio`/`vmcnt`/`sched.group.barrier` | `add_warp_pipeline_conversion` (ConvertWarpPipeline.cpp, make_llir:401) + `add_update_async_wait_count` (make_llir:400) + `lower_instruction_sched_hints` (make_llir:430) | **reused as-is** |

### Where the new pass goes

A single new TTGIR pass — `add_auto_warp_pipeline` — inserted in `make_ttgir` **after `add_pipeline` and immediately before `add_warp_pipeline`** (make_ttgir:370):

- *After `add_pipeline`*: the loop is already multi-buffered, with the prologue/epilogue and async copies materialized, so the marker ops have real ops to bracket.
- *Immediately before `add_warp_pipeline`*: that pass is deliberately placed last because the preceding `cse`/`dce` would otherwise strip the priority markers (compiler.py:364-369). The new emitter must run after that cleanup for the same reason (its markers survive because they hang on a side-effecting `sched.barrier`).

The pass does exactly three things — build the DDG, run the modulo analysis (II / slack / occupancy), compute clusters + Step 4.8 priorities — then writes the `border`/`priority` markers. After that, `add_warp_pipeline → add_update_async_wait_count → add_warp_pipeline_conversion → lower_instruction_sched_hints` take over automatically.

### Reused for free (the new scheduler never touches these)

- **Two warp groups + `cond_barrier` phase shift** — `emitPipelinePrelude`/`emitPipelinePostlude` (ConvertWarpPipeline.cpp:295/305).
- **`s_setprio`** — emitted from the priority attribute (ConvertWarpPipeline.cpp:253); Step 4.8 only supplies the integer.
- **`vmcnt`** — `UpdateAsyncWaitCount.cpp` derives it by backward def-chain analysis.
- **In-cluster `sched.group.barrier` / `iglp_opt`** — `lower_instruction_sched_hints`.

This is the concrete meaning of "Pass C makes no scheduling decisions": on AMD it already exists.

### Net-new work and blockers (priority order)

1. **Latency + resource cost model (the hard blocker).** `add_schedule_loops` passes an *empty* `unusedOpLatency` today (ScheduleLoops.cpp:125), so there is currently no II / slack / occupancy to compute against. Both Pass A and Step 4.8 stand on this gfx950-microbenchmarked table (Limitations #1). Without it the rest is moot.
2. **The marker-emitting pass** above — turning "human writes `warp_pipeline_stage`" into "compiler emits the attributes."
3. **Mutual exclusion with `BlockPingpong.cpp`.** `add_block_pingpong` (make_ttgir:335-336) is the existing *automatic* competitor: it also emits `s_setprio` + `cond_barrier`, also runs before `add_warp_pipeline`, but only matches its hard-coded templates (1–2 dots, fixed tile sizes). The new pass and BlockPingpong must never transform the same loop — gate them with a knob (mirroring `knobs.amd.use_block_pingpong`), with the new general path taking the loops BlockPingpong's templates do not cover.

### Subtleties

- **Opt-in is automatic.** `WarpPipeliner.cpp` is a no-op when a loop has no borders, so the new pass only affects loops it chooses to mark — the blast radius is bounded.
- **Gluon path.** `gluon_to_ttgir` runs its own `add_warp_pipeline` (compiler.py:387). Covering gluon kernels means running the emitter there too; the plain `triton.jit`/TTGIR path needs only the `make_ttgir` insertion.
- **Who owns buffer depth.** Multi-buffering is decided by `add_schedule_loops` + `add_pipeline` from `num_stages`. Pass A's depth decision therefore belongs *inside* ScheduleLoops (feed it the cost model and let it set depth) rather than in the marker emitter — which is the choice between folding Pass A into ScheduleLoops ("all the way down", more invasive) and a separate post-pipeline analysis pass (lower risk, matches the Pass A/B split here).

---

## Worked Example: gfx950 Warp-Pipelined GEMM

Maps to `amd_gemm_warp_pipeline.py` (kernel `matmul_kernel_warp_pipeline`, line 38).

**DDG (K-loop body):**
```
load_A →buf_A→ ds_read_A ┐
load_B →buf_B→ ds_read_B ┴→ mfma → acc(next, d=1)
```

**Step 1 — MinII:** MEM-bound, `ResMII ≈ load_A + load_B`. RecMII (register accumulator) is below ResMII → MEM-bound.

**Step 2 — Modulo schedule (3 stages):**
```
stage 0:  load_A, load_B            (MEM)     // global_load_lds → buf[i % 3]
stage 1:  ds_read_A, ds_read_B      (MEM)     // ds_read from buf
stage 2:  mfma                      (MFMA)
```

**Step 3 — Depth:** accumulator-to-MFMA lifetime spans the loads → `floor(lifetime/II)+1 = 3` buffers; vmcnt window `W=1` (keep newest load group in flight) → `async_load_wait_group(1)`.

**Step 4.7 — Partition:** two-group warp-pipeline, clusters `{mfma}` (priority 0) and `{mem: next loads + ds_read}` (priority 1).

**Pass B/C — Emitted IR (abbreviated; rendered as TLX for readability — the passes emit TTGIR borders + ROCDL, not `tlx.*`):**
```python
# Prologue: issue NUM_BUFFERS async loads (global_load_lds)
for i in range(NUM_BUFFERS):
    tok_a = tlx.async_load(a_ptr + ..., tlx.local_view(smemA, i), ...)
    tok_b = tlx.async_load(b_ptr + ..., tlx.local_view(smemB, i), ...)
    tlx.async_load_commit_group([tok_a, tok_b])
tlx.async_load_wait_group(1)                  # → s_waitcnt vmcnt(n)
a_tile = tlx.local_load(tlx.local_view(smemA, 0))   # ds_read

for i in tl.range(0, k_iters - NUM_BUFFERS, loop_unroll_factor=0):
    with tlx.warp_pipeline_stage("mfma", priority=0):    # cluster 0
        acc = tl.dot(a_tile, b_tile, acc)                # MFMA
    with tlx.warp_pipeline_stage("mem", priority=1):     # cluster 1
        tok_a = tlx.async_load(...)  ; tok_b = tlx.async_load(...)
        tlx.async_load_commit_group([tok_a, tok_b])      # next global_load_lds
        a_tile = tlx.local_load(tlx.local_view(smemA, next_buf))  # ds_read
    tlx.async_load_wait_group(1)                          # s_waitcnt vmcnt
# Epilogue: drain remaining in-flight tiles (static_range)
```

The two `warp_pipeline_stage` clusters + `cond_barrier` phase shift are exactly what Pass A computes; `num_stages=1` is required so the automatic pipeliner defers to the warp-pipeline (compiler.py guard).

## Worked Example: Fused addmm + GLU

Maps to `amd-addmm-glu-opt_test.py` (kernel `tlx_addmm_glu_kernel_optimized`, line 127). Same K-loop as GEMM, plus:

- **Non-loop epilogue (Pass A.6):** `bias add → x = acc + bias → GLU out = x + x*y → store`. The accumulator is register-resident; bias/Y are loaded in the epilogue. This is list-scheduled as a straight-line region.
- **Padded LDS for Y (Step 4.5 padding policy):** the async variant stages Y into a **padded** LDS buffer (`_y_padded_layout`, line 119: `with_identity_for([[BN, PAD]], [BM, BN], [1,0])`) so the `ds_read` of Y avoids bank conflicts. The padding adds `PAD` elements per row to the buffer's physical footprint, which the Step 4 budget check must count.
- **Streaming epilogue traffic:** Y and C are touched once (`cache_modifier=".cs"`), so they should not be cached — a Pass C codegen detail, not a scheduling one.

## Worked Example: Flash Attention Forward

The FA forward kernel in `amd-fa-pipelined_test.py` ships four variants of increasing sophistication; they form a natural ladder from "what the simple algorithm produces" to "what the full algorithm produces":

| Variant | Kernel | LDS depth | Pipelining mechanism |
|---|---|---|---|
| `async_simple` | `_attn_fwd_async_simple` | 1 (single-buffer) | plain async-load + `wait_group(0)`, `num_stages=0` |
| `async_prefetch` | `_attn_fwd_async_prefetch` | 2 (double-buffer) | hand modulo-scheduled prologue / hot-loop / epilogue |
| `persistent` | `_attn_fwd_persistent` | 2 | `async_prefetch` tile body + persistent XCD-pinned zig-zag scheduler |
| **`cluster`** | **`_attn_fwd_cluster_pipeline` → `_attn_inner_pipelined`** | **2 (2-slot), depth-4** | **rotated 4-cluster `warp_pipeline_stage`** |

The `cluster` variant is the canonical AMD warp-pipeline and the one this algorithm targets, so it gets the full worked-example treatment below. The simpler variants are what Pass A emits at shallower depth (they skip Step 4.7's multi-cluster partition).

### FA Forward Dependency Graph

The hot loop body decomposes into **eight logical sub-clusters** (the kernel names them in the header comment), connected by intra-iteration and loop-carried edges:

```
ACK ─→ LRK ─→ dot_qk ─→ VEC1 ──(p, alpha; d=1)──→ VEC2 ─→ dot_pv ─→ acc(d=1)
ACV ─→ LRV ───────────────────────────────────────────────↗
        sub-cluster   pipeline   role
        ───────────   ────────   ──────────────────────────────────────
        ACK / ACV     MEM        async-copy K/V  (global → LDS, global_load_lds)
        LRK / LRV     MEM        local-read K/V  (LDS → regs, ds_read; LRK transposes)
        dot_qk        MFMA       Q·Kᵀ → qk scores
        VEC1          VALU/SFU   softmax numerator: row-max + exp2 burst → p, alpha
        VEC2          VALU       softmax denominator: Σp, acc·alpha, l_i, p→fp16
        dot_pv        MFMA       P·V → acc
```

Two loop-carried edges (distance=1) make this a genuine software-pipelining problem, exactly as on NVIDIA FA:

- `acc[i] → dot_pv[i+1]` — the output accumulator (a **register** value on AMD, not a TMEM tile).
- `(p, alpha)[i] → VEC2[i]` produced by `VEC1` in the **previous** iteration: the kernel computes `p, alpha` in `vec1` one step ahead and consumes them in `vec2`/`dot_pv` the next step. This staggering is what lets the exp2 burst overlap the matrix engine.

### Pass A, Step 1: MinII

Active pipelines for the square anchor (BLOCK_M=256, D=128, BLOCK_N=64, bf16):

```
MEM:  ACK + ACV + LRK + LRV     (two global_load_lds + two ds_read per iter)
MFMA: dot_qk + dot_pv           (two MFMAs per iter)
VALU: VEC1(exp2 burst) + VEC2   (the transcendental row-max/exp2 dominates VALU)
```

FA forward is **balanced between MFMA and VALU** (the exp2 burst is heavy), so unlike GEMM it is *not* purely MEM-bound — `ResMII ≈ max(MFMA work, VALU work)`. The accumulator recurrence (`RecMII`) is broken by the one-iteration `(p, alpha)` stagger, so the schedule can approach `ResMII`. The scheduler's job is therefore to **overlap exp2 with the MFMAs**, which is precisely why VEC1 is co-scheduled with `dot_pv` (see Step 4.7).

### Pass A, Step 2: Modulo Schedule (rotated, depth-4)

The schedule rotates the eight sub-clusters across a **depth-4** pipeline over a **2-slot** LDS double buffer. One steady-state iteration processes K/V block `i` while three other blocks are in flight at different stages. The cluster assignment (cycle order within the iteration) is:

```
cluster 0  "dot1"  (MFMA, prio 0):  dot_qk[i+1]      ; VEC2[i]
   ── wait vmcnt: V[i] ready ──
cluster 1  "mem1"  (MEM,  prio 1):  LRV[i]           ; ACK[i+3]
cluster 2  "dot2"  (MFMA, prio 0):  dot_pv[i]        ; VEC1[i+1]   (exp2 lands after PV)
   ── wait vmcnt: K[i+2] ready ──
cluster 3  "mem2"  (MEM,  prio 1):  LRK[i+2] (trans) ; ACV[i+2]
```

The defining trick: **`dot_pv[i]` and `VEC1[i+1]` share cluster 2**, so the exp2 burst for the *next* block issues right after the PV MFMA of the *current* block — exp throughput on the VALU/transcendental unit overlaps the matrix engine instead of stalling it. Symmetrically, the cheaper `VEC2[i]` rides with `dot_qk[i+1]` in cluster 0. Each MEM cluster pairs one `ds_read` (LRK/LRV) with one `global_load_lds` (ACK/ACV) so the read of the just-arrived tile overlaps the issue of a future tile.

### Pass A, Step 3: Pipeline Depths

`BUF_DEPTH = 2` for both K and V (`local_alloc(..., BUF_DEPTH)`). The depth-4 *schedule* over a 2-slot *buffer* works because each slot is produced (ACK/ACV) and fully consumed (LRK/LRV) within two iterations — `floor(lifetime / II) + 1 = 2`. The prologue primes the pipeline by issuing K0, V0, K1 and then K2 into the *reused* slot 0 (commit order `K0, V0, K1, K2`), and the vmcnt window is tuned per-wait: `async_load_wait_group(1)` drains all but the newest commit so the consumed tile is guaranteed complete while the freshest prefetch stays in flight.

### Pass A, Step 4: LDS Budget

Two 2-slot buffers: `2 × (BLOCK_N × HEAD_DIM) × sizeof(bf16) × 2` = `2 × (64×128) × 2 × 2` = **64 KB**, comfortably inside the 160 KB/CU budget. This is also why D=128 forces `BLOCK_N=64` in the `async_prefetch`/`persistent` variants (the comment at `flash_attn_async_prefetch`): `BLOCK_N=128` at D=128 would blow the double-buffered K+V LDS footprint. No TMEM budget exists — the `[BLOCK_M, HEAD_DIM]` accumulator lives in registers, which is what makes register pressure (next note) the binding constraint instead.

### Pass A, Step 4.7: Warp-Group / Cluster Partition

Step 4.7 produces **four clusters**, realized directly by the kernel's `with tlx.warp_pipeline_stage(label, priority=...)` blocks; the `priority=` values themselves are assigned in [Step 4.8](#step-48-derive-s_setprio-priorities-modulo-reservation-priority), not here:

- **DOT clusters** (`dot1`, `dot2`). MFMA + the dependent VALU softmax ops are kept in one cluster because softmax must follow QK on the *same* data — separating them across warp groups would cost a full `s_barrier` per element of a long dependency chain (high separation cost).
- **MEM clusters** (`mem1`, `mem2`). The `ds_read` + `global_load_lds` pair, kept together so the read of the just-arrived tile overlaps the issue of a future tile.

Both warp groups run **all four clusters** phase-offset by one stage (`cond_barrier`), the AMD warp-pipeline model — not asymmetric producer/consumer roles. This is the same shape `BlockPingpong.cpp` builds by hand for GEMM, here generalized to FA's four clusters.

`num_warps = 8` at BLOCK_M=256 (the wrapper enforces `BLOCK_M ≥ num_warps × MFMA_M`, MFMA_M=32): fewer than 32 dot-rows per warp would be an invalid MFMA tiling.

### Pass A, Step 4.8: Priorities from the MRT

The four clusters fall into the kernel's two `priority` classes through the Step 4.8 derivation — not by labelling them mem vs compute:

- **Contended unit.** At Δ*=2 each mem cluster runs opposite a dot cluster. The pipeline they share is **VALU**: the mem clusters issue address arithmetic while the dot clusters run the exp2/scale softmax burst, so `contended = {VALU}`.
- **Monopolization `M`.** `M(dot1) = M(dot2)` is high — dense MFMA plus the heavy exp2 VEC burst saturate the VALU issue. `M(mem1) = M(mem2)` is low — only a handful of address-update VALU ops. A high-priority dot cluster would monopolize VALU and starve the opposite mem warp's address calc (BlockPingpong.cpp:689-692).
- **Urgency `U`.** `U(mem1) = U(mem2)` is high — each issues a `global_load_lds` (ACK/ACV) whose ~600-cycle round-trip must be hidden across the depth-4 pipeline, so its *issue timing* is latency-critical. `U(dot1) = U(dot2)` is lower — the one-iteration `(p, alpha)` stagger gives the compute clusters issue slack.

So `pscore(mem) = U − λM` exceeds `pscore(dot)` → **mem1/mem2 → priority 1, dot1/dot2 → priority 0** — exactly the `priority=` the kernel writes by hand, now produced by Step 4.8 rather than asserted. (If the loop were memory-dense and compute-sparse, the same arithmetic would flip it; that is the point of deriving rather than labelling.)

### Pass B, Step 2: Synchronization

Three AMD mechanisms appear, each mapping to a Pass B decision:

- **`s_waitcnt vmcnt`** via `tlx.async_load_wait_group(1)` before each `local_load` — the count-based completion that replaces NVIDIA's per-buffer mbarrier wait. The kernel comments spell out the window reasoning ("wait_group(1) drains all but the newest (K2), so K1 is complete before this LDS read; wait_group(2) would race"). `UpdateAsyncWaitCount.cpp` finalizes the literal `vmcnt`.
- **`s_barrier`** via `tl.debug_barrier()` at the two **WAR hazards**: where an `LRK`/`LRV` (`ds_read`) of a slot precedes an `ACK`/`ACV` (`global_load_lds` write) that *reuses* that slot. With only 2 slots and a depth-4 schedule, slot reuse is unavoidable, so these barriers are mandatory — the LDS-dependency analysis (`analyzePipelineDependencies`) is what would insert them automatically.
- **`cond_barrier`** for the one-stage phase offset between the two warp groups (prelude/postlude, emitted by `ConvertWarpPipeline.cpp`).

The `local_trans` on K is a **metadata-only memdesc transpose**: it makes `local_load` land directly in `dot_op(opIdx=1)` layout, skipping the register-shuffle + LDS round-trip that `tl.dot(q, k.T)` would otherwise emit. In the ScheduleGraph this is a layout annotation on the LRK buffer's consumer, not a scheduled op.

### Pass B/C, Step 5: Emitted IR (steady-state loop, abbreviated; rendered as TLX for readability)

```python
for block_n in tl.range(block_start, block_end - 3, num_stages=0):
    cur_slot = (block_n - block_start) % BUF_DEPTH
    nxt_slot = (block_n + 1 - block_start) % BUF_DEPTH

    with tlx.warp_pipeline_stage("dot1", priority=0):        # cluster 0
        qk = tl.dot(q, kt_dot)                               # dot_qk[i+1]
        state, p_dot = state.vec2(p_c, alpha_c, q.dtype)     # VEC2[i]

    tlx.async_load_wait_group(1)                             # vmcnt: V[i] ready

    with tlx.warp_pipeline_stage("mem1", priority=1):        # cluster 1
        v_dot = tlx.local_load(tlx.local_view(v_buf, cur_slot), relaxed=True)   # LRV[i]
        tok_k = tlx.async_load(k_ptrs + ack_n*stride_kn, tlx.local_view(k_buf, nxt_slot))
        tlx.async_load_commit_group([tok_k])                 # ACK[i+3]

    with tlx.warp_pipeline_stage("dot2", priority=0):        # cluster 2
        acc = tl.dot(p_dot, v_dot, state.acc)                # dot_pv[i]
        state = SoftmaxState(acc, state.l_i, state.m_i)
        state, p_c, alpha_c = state.vec1(qk, ahead_n, ...)   # VEC1[i+1]  (exp2 burst)

    tlx.async_load_wait_group(1)                             # vmcnt: K[i+2] ready

    with tlx.warp_pipeline_stage("mem2", priority=1):        # cluster 3
        kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, cur_slot)), relaxed=True)  # LRK[i+2]
        tok_v = tlx.async_load(v_ptrs + acv_n*stride_vn, tlx.local_view(v_buf, cur_slot))
        tlx.async_load_commit_group([tok_v])                 # ACV[i+2]
```

The depth-4 prologue (prime K0/V0/K1/K2) and the 3-tile drain (no OOB prefetch) are exactly the prologue/epilogue structure Pass B Step 3 computes from `maxStage`.

### Algorithm → Emitted IR Mapping Summary

(The right column shows the hand-written `cluster` kernel's TLX as a readable stand-in for the IR the backend passes emit — the passes produce the equivalent TTGIR borders / ROCDL, not this TLX source.)

| Algorithm concept | FA `cluster` kernel realization |
|---|---|
| `(cycle, pipeline, stage, cluster)` per op | the four `warp_pipeline_stage` blocks + their op order |
| Step 2.5 cluster IDs | `"dot1"/"mem1"/"dot2"/"mem2"` borders |
| Step 3 buffer depth | `BUF_DEPTH = 2`, slot = `(block_n - block_start) % BUF_DEPTH` |
| vmcnt window | `async_load_wait_group(1)` |
| Step 4.8 priority (modulo MRT) | `priority=0` (DOT) / `priority=1` (MEM) → `s_setprio` |
| Pass B WAR `s_barrier` | `tl.debug_barrier()` at slot-reuse points |
| Pass B phase offset | `cond_barrier` (emitted by `ConvertWarpPipeline.cpp`) |
| layout annotation | `tlx.local_trans` (K → dot-operand layout, no shuffle) |
| occupancy / register budget | `waves_per_eu` (see below) — no `setmaxnreg` |

### Notes Specific to FA on CDNA4

- **Register pressure is the binding constraint, not LDS.** The accumulator is register-resident and the NaN-propagating row-max (`_row_max`) raises VGPR pressure. The wrapper pins `waves_per_eu = 0 if causal else 2`: the **non-causal path must pin 2 or it spills ~8×**; the causal path is fastest unconstrained. This is the AMD analog of NVIDIA's TMEM-merging pressure — but resolved through occupancy, since there is no `setmaxnreg` and no TMEM to merge.
- **Causal = two pipelined regions, not masking inside the loop.** `_attn_fwd_cluster_pipeline` splits causal work into an unmasked below-diagonal region (`MASK_STEPS=False`, FMA-friendly) followed by a `BLOCK_M/BLOCK_N`-tile masked diagonal band (`MASK_STEPS=True`). Both run the *same* depth-4 rotated pipeline (each region must be ≥ 4 tiles, which the wrapper asserts). In the algorithm this is Pass A.5 data partitioning applied to the iteration space.
- **`iglp_opt` is the existing coarse alternative.** For the non-cluster variants, the AMD backend's only built-in instruction-scheduling hint is the `attention` `iglp_opt` strategy (SchedInstructions.cpp:82). The `cluster` kernel supersedes it with the explicit `warp_pipeline_stage` partition — i.e. it hand-codes the Step 4.7 / Pass C output this algorithm would generate.
- **Shallower variants = shallower schedules.** `async_simple` (depth-1) and `async_prefetch` (depth-2) are the same DDG scheduled at lower pipeline depth; the measured baselines in the file (`PERF_BASELINE_TFLOPS`) show the `cluster` variant winning at D=128 (e.g. N=16384 non-causal: 865 vs 724 for `async_simple`), quantifying what the deeper rotated schedule buys.

---

## NVIDIA vs AMD: Key Differences

| Dimension | NVIDIA (ws_global_instruction_scheduling.md) | AMD (this doc) |
|---|---|---|
| Specialization model | Asymmetric producer/consumer warp groups | **Symmetric phase-offset warp-pipeline** (two groups, same code, one stage apart) |
| Accumulator storage | TMEM buffers (allocated, merged) | **Registers** (occupancy/`waves_per_eu`, no buffer) |
| Async copy | TMA + mbarrier (phase) | **`global_load_lds` + `s_waitcnt vmcnt`** (count) |
| Sync objects | mbarrier pairs per buffer | **vmcnt counters** + `s_barrier`/`cond_barrier`; no per-buffer object |
| Register rebalancing | `setmaxnreg` per group | **none** (`reallocRegisters` no-op) |
| Scratch budget | SMEM (per-CTA) | **LDS 160 KB/CU**; no separate TMEM budget |
| Bank conflicts | swizzling | **padded LDS layouts** (`padded_shared`) — affects buffer footprint and merge |
| Instruction ordering | compiler | **`sched.group.barrier` / `iglp_opt` / `s_setprio`** |
| Existing scaffolding | WS passes | `WarpPipeliner.cpp`, `ConvertWarpPipeline.cpp`, `BlockPingpong.cpp`, `UpdateAsyncWaitCount.cpp` |
| Latency model | microbenchmark table | **net-new** (`unusedOpLatency` is empty today) |
| Load constraints | TMA box copies | **no LDS scatter on gfx950**; load widths `{128,32}` |

## Complexity

Same asymptotic profile as the NVIDIA design: modulo scheduling per region is the dominant cost (Rau's iterative placement, `O(|ops| × II × backtrack)`), the budget/merge checks are `O(buffers²)` over live-interval pairs, and Pass A converges in 1–2 iterations. The AMD-specific passes (vmcnt window computation, padding-aware budget, two-group partition, and the modulo-MRT `s_setprio` derivation of Step 4.8 — `O(|ops|)` for ASAP/ALAP slack plus `O(clusters² × pipelines)` for the Δ search) are linear-to-quadratic in the schedule size and do not change the overall complexity.
