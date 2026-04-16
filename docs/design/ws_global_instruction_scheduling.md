# Warp-Specialized Global Instruction Scheduling Algorithm

This document is based on the original design in [WS global instruction scheduling](https://docs.google.com/document/d/1vgHBxejxbF-IUydQh-2-kpKX6sF1_lQfZizY-kJsTyc/edit?tab=t.0#heading=h.n6jjdkke8lkz).

## Table of Contents

- [Overview](#overview)
  - [Central Data Structure](#central-data-structure)
  - [Implementation Layer: ScheduleGraph](#implementation-layer-schedulegraph)
  - [Algorithm Summary](#algorithm-summary)
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
    - [Background: Rau's Iterative Modulo Scheduling](#background-raus-iterative-modulo-scheduling)
  - [Step 2.5: Compute Cluster IDs from the Modulo Schedule](#step-25-compute-cluster-ids-from-the-modulo-schedule)
  - [Step 3: Derive Per-Region Pipeline Depth from the Modulo Schedule](#step-3-derive-per-region-pipeline-depth-from-the-modulo-schedule)
  - [Step 4: Handling Resource Pressure (SMEM/TMEM Budget)](#step-4-handling-resource-pressure-smemtmem-budget)
  - [Step 4.5: Lifetime-Aware Buffer Merging](#step-45-lifetime-aware-buffer-merging)
  - [Step 5: Emit ScheduleGraph](#step-5-emit-schedulegraph)
- [Pass A.5: Data Partitioning for Improved Overlap (Optional)](#pass-a5-data-partitioning-for-improved-overlap-optional)
- [Pass A.6: Scheduling Non-Loop Regions](#pass-a6-scheduling-non-loop-regions)
- [Pass A.7: Epilogue Subtiling](#pass-a7-epilogue-subtiling)
- [Pass B: Warp Specialization Reconstruction](#pass-b-warp-specialization-reconstruction)
  - [Step 1: Partition Ops into Warp Groups](#step-1-partition-ops-into-warp-groups)
  - [Step 1.5: Replicate Shared Infrastructure Ops](#step-15-replicate-shared-infrastructure-ops)
  - [Step 2: Insert Synchronization](#step-2-insert-synchronization)
  - [Step 3: Compute Per-Region Loop Structure](#step-3-compute-per-region-loop-structure)
  - [Step 4: Assign Warp Counts and Registers](#step-4-assign-warp-counts-and-registers)
  - [Step 5: Generate TLX Code Skeleton](#step-5-generate-tlx-code-skeleton)
- [Pass C: Code Generation and Instruction Ordering](#pass-c-code-generation-and-instruction-ordering)
  - [Relationship Between Pass A and Pass C](#relationship-between-pass-a-and-pass-c)
- [Worked Example: Blackwell GEMM Kernel](#worked-example-blackwell-gemm-kernel)
  - [GEMM Dependency Graph](#gemm-dependency-graph)
  - [Pass A, Step 1: Compute MinII](#pass-a-step-1-compute-minii)
  - [Pass A, Step 2: Modulo Schedule](#pass-a-step-2-modulo-schedule)
  - [Pass A, Step 3: Derive Pipeline Depths](#pass-a-step-3-derive-pipeline-depths)
  - [Pass A, Step 4: Memory Budget Check (Initial)](#pass-a-step-4-memory-budget-check-initial)
  - [Pass A.7 Applied: Epilogue Subtiling (EPILOGUE_SUBTILE=4)](#pass-a7-applied-epilogue-subtiling-epilogue_subtile4)
  - [Pass A, Step 4: Memory Budget Check (After A.7)](#pass-a-step-4-memory-budget-check-after-a7)
  - [Pass A, Step 5: Emit ScheduleGraph](#pass-a-step-5-emit-schedulegraph)
  - [Pass B, Step 1: Partition into Warp Groups](#pass-b-step-1-partition-into-warp-groups)
  - [Pass B, Step 2: Insert Synchronization](#pass-b-step-2-insert-synchronization)
  - [Pass B, Step 5: Generated TLX Code](#pass-b-step-5-generated-tlx-code)
  - [Algorithm → TLX Code Mapping Summary](#algorithm--tlx-code-mapping-summary)
  - [Pass B, Step 1: Partition into Warp Groups](#pass-b-step-1-partition-into-warp-groups)
  - [Pass B, Step 2: Insert Synchronization](#pass-b-step-2-insert-synchronization)
  - [Pass B, Step 5: Generated TLX Code](#pass-b-step-5-generated-tlx-code)
  - [Algorithm → TLX Code Mapping Summary](#algorithm--tlx-code-mapping-summary)
- [Worked Example: Blackwell Flash Attention Forward Kernel](#worked-example-blackwell-flash-attention-forward-kernel)
  - [FA Forward Dependency Graph](#fa-forward-dependency-graph)
  - [Pass A, Step 1: Compute MinII](#pass-a-step-1-compute-minii-1)
  - [Pass A.5 Applied: Data Partitioning (NUM_MMA_GROUPS=2)](#pass-a5-applied-data-partitioning-num_mma_groups2)
  - [Pass A, Step 2: Modulo Schedule](#pass-a-step-2-modulo-schedule-1)
  - [Pass A, Step 3: Derive Pipeline Depths](#pass-a-step-3-derive-pipeline-depths-1)
  - [Pass A, Step 4: Memory Budget Check](#pass-a-step-4-memory-budget-check-1)
  - [Pass B, Step 1: Partition into Warp Groups](#pass-b-step-1-partition-into-warp-groups-1)
  - [Pass B, Step 2: Insert Synchronization](#pass-b-step-2-insert-synchronization-1)
  - [Pass B, Step 5: Generated TLX Code](#pass-b-step-5-generated-tlx-code-1)
  - [Algorithm → TLX Code Mapping Summary](#algorithm--tlx-code-mapping-summary-1)
  - [Pass C Applied: In-Group Pipelining (blackwell_fa_ws_pipelined.py)](#pass-c-applied-in-group-pipelining-blackwell_fa_ws_pipelinedpy)
  - [GEMM vs FA Forward: Key Differences](#gemm-vs-fa-forward-key-differences)
- [Worked Example: Blackwell Flash Attention Backward Kernel](#worked-example-blackwell-flash-attention-backward-kernel)
  - [FA Backward Dependency Graph](#fa-backward-dependency-graph)
  - [Pass A, Step 1: Compute MinII](#pass-a-step-1-compute-minii-2)
  - [Pass A, Step 2: Modulo Schedule](#pass-a-step-2-modulo-schedule-2)
  - [Pass A, Step 3: Derive Pipeline Depths](#pass-a-step-3-derive-pipeline-depths-2)
  - [Pass A, Step 4: Memory Budget Check](#pass-a-step-4-memory-budget-check-2)
  - [Pass B, Step 1: Partition into Warp Groups](#pass-b-step-1-partition-into-warp-groups-2)
  - [Pass B, Step 2: Insert Synchronization](#pass-b-step-2-insert-synchronization-2)
  - [Pass B, Step 5: Generated TLX Code](#pass-b-step-5-generated-tlx-code-2)
  - [Algorithm → TLX Code Mapping Summary](#algorithm--tlx-code-mapping-summary-2)
  - [GEMM vs FA Forward vs FA Backward: Key Differences](#gemm-vs-fa-forward-vs-fa-backward-key-differences)
- [Complexity](#complexity)

## Overview

This document describes a scheduling algorithm for GPU kernels that:

1. **Discovers** the near-optimal multi-pipeline instruction schedule using **modulo scheduling**
2. **Derives** the per-region pipelining scheme (buffer depth, prologue/epilogue) from the modulo schedule
3. **Reconstructs** the warp specialization strategy, synchronization, and code structure

The algorithm is inspired by the scheduling patterns found in existing hand-tuned TLX kernels (`blackwell_gemm_ws`, `blackwell_fa_ws`, `blackwell_fa_ws_pipelined`, `blackwell_fa_ws_pipelined_persistent`) and formalizes them into a systematic framework based on modulo scheduling. The goal is to automate the decisions that kernel authors currently make by hand — buffer depths, warp group partitioning, barrier placement, in-group instruction interleaving — and reproduce (or improve upon) the performance of hand-written kernels.

The ultimate target of the algorithm is **TTGIR** (Triton GPU IR), the warp-specialized intermediate representation that the Triton compiler lowers to PTX. Throughout this document, TLX code is used for illustration because it maps closely to the hardware primitives (barriers, TMEM, TMA) and is easier to read than TTGIR, but the algorithm's output is a scheduling specification that can be lowered to either representation.

The algorithm treats each major GPU functional unit (Memory, Tensor Core, CUDA Core, SFU) as an independent pipeline resource and finds a steady-state schedule that overlaps iterations with a fixed **initiation interval (II)**.

### Central Data Structure

The algorithm's central output is the **ScheduleGraph** — a DDG-based graph that accumulates all scheduling and resource allocation decisions. At its core, each scheduled op carries a `(cycle, pipeline, stage, cluster)` tuple:

- **cycle**: When the op starts. For loop regions, this is within the II-length reservation table (0 ≤ cycle < II × max_stage). For non-loop regions, this is the absolute cycle from the start of the region.
- **pipeline**: Which hardware unit executes it (MEM, TC, CUDA, SFU)
- **stage**: For loop regions, how many II periods the op is deferred relative to its owning iteration (enables cross-iteration pipelining). For non-loop regions, always 0 — there is no iteration overlap.
- **cluster**: Within-stage ordering derived from cycle. Ops in the same stage are assigned dense cluster IDs sorted by cycle (lower cycle → lower cluster ID). The downstream code generator uses cluster IDs to determine instruction emission order within each stage, ensuring the generated code respects the schedule's optimal ordering rather than relying on arbitrary IR program order.

Beyond per-op scheduling, the ScheduleGraph also carries **resource allocation decisions**: multi-buffered memory allocations (`ScheduleBuffer`), paired barrier objects, buffer sharing/merging groups, warp group assignments, and prologue/epilogue structure. These are all accumulated on the graph without modifying the original IR — enabling iterative refinement where the schedule can be rebuilt from scratch if a DDG transformation changes the problem.

The schedule format is the same for both loop and non-loop regions. The difference is in how it's computed (modulo scheduling vs list scheduling) and how it's realized (prologue/kernel/epilogue expansion vs direct emission in cluster order). This unified representation allows the same downstream passes (warp group partitioning, barrier insertion, code generation) to handle both cases.

### Implementation Layer: ScheduleGraph

The design doc describes the algorithm using TLX (the Python DSL) for illustration because it maps closely to hardware primitives and is easy to read. For the actual compiler implementation at the **TTGIR level**, we introduce an intermediate abstraction called the **ScheduleGraph** — a DDG-based side data structure that captures all scheduling decisions without modifying the original IR.

**DDG-based construction:** The ScheduleGraph is built directly from the Data Dependence Graph (DDG). Each DDG node becomes a `ScheduleNode`, each DDG edge becomes a `ScheduleEdge`, and the graph inherits the DDG's dependency structure, pipeline classification, and latency information. The ScheduleGraph then *extends* the DDG with scheduling decisions: cycle/stage assignments from modulo scheduling, buffer allocations from lifetime analysis, warp group partitions from utilization analysis, and prologue/epilogue structure from loop expansion. In this sense, the ScheduleGraph is a **scheduled, annotated DDG** — the DDG provides the "what depends on what" foundation, and the scheduling algorithm fills in the "when, where, and how much buffering" decisions.

**Why a separate abstraction?** The algorithm produces many interdependent decisions: cycle assignments, buffer depths, warp group partitions, barrier placement, prologue/epilogue structure. Applying these incrementally to the IR is fragile — a later decision (e.g., SMEM budget reduction) can invalidate an earlier IR modification. The ScheduleGraph solves this by recording all decisions on a separate graph that *points into* the IR (via Operation pointers) but does not mutate it. Only after the schedule converges does a lowering pass apply the accumulated decisions to produce the final TTGIR. This also means the iterative refinement loop can simply rebuild the ScheduleGraph from a fresh DDG — no IR rollback needed.

**Relationship to TLX:** The ScheduleGraph is conceptually equivalent to TLX — both represent a pipelined loop with multi-buffered memory, barrier synchronization, and warp specialization. TLX expresses this at the Python language level (the kernel author writes `tlx.barrier_wait`, `tlx.tmem_alloc[2]`, etc.); the ScheduleGraph expresses the same concepts at the TTGIR implementation level (a `ScheduleBuffer` with `count=2` maps to a double-buffered `ttg.local_alloc`). The key difference: TLX is manually authored, while the ScheduleGraph is automatically constructed from the DDG by the scheduling algorithm.

**Core types** (implemented in `ModuloScheduleGraph.h`):

| Type | Role | TLX Equivalent |
|------|------|----------------|
| **ScheduleBuffer** | Multi-buffered memory allocation (SMEM, TMEM, or BARRIER) with shape, element type, buffer count, modular live interval (`liveStart`/`liveEnd` within II), merge group ID, and paired barrier references | `tlx.alloc_smem[num_buffers]`, `tlx.alloc_tmem[2]` |
| **ScheduleNode** | A scheduled operation wrapping an MLIR op with cycle, stage, pipeline, latency, buffer produce/consume refs, and warp group assignment | Individual TLX ops within an `async_task` |
| **ScheduleEdge** | Producer-consumer dependency with latency and loop-carried distance | Implicit in TLX barrier wait/arrive pairs |
| **ScheduleLoop** | A pipelined `scf.for` with II, maxStage, trip count, nodes, edges, buffers, and memory interface ports | A TLX `tl.range(..., warp_specialize=True)` loop |
| **ScheduleGraph** | Top-level container: a forest of ScheduleLoops with bottom-up processing order and parent-child relationships via super-nodes | The complete TLX kernel |

**How the algorithm phases map to the ScheduleGraph:**

```
Phase 0 (Schedule):   DDG + Rau's → populate ScheduleNode.cycle/stage
Phase 1 (Buffers):    Stage diffs → populate ScheduleBuffer.count
Phase 1.5 (WS):       Utilization → assign ScheduleNode.warpGroup
Phase 2 (Expand):     Bottom-up → populate prologueNodes/epilogueNodes
Phase 3 (Lower):      ScheduleGraph → replace MLIR ops with async copies + barriers
```

Phases 0-2 (Pass A + Pass B) operate entirely on the ScheduleGraph, accumulating decisions. Phase 3 (Pass C) reads the converged graph and emits the final TTGIR. This separation means the iterative refinement loop (re-scheduling when A.5 or A.7 transform a DDG) simply rebuilds the ScheduleGraph from scratch — no IR rollback needed.

**Nested loops:** For persistent kernels with outer tile loops and inner K-loops, the ScheduleGraph forms a tree. The inner K-loop becomes a child `ScheduleLoop` linked to the outer loop via a super-node `ScheduleNode`. The algorithm processes bottom-up: schedule the inner loop first, model it as a single super-node with latency = `prologueLatency + tripCount × II`, then schedule the outer loop.

**Full pass coverage:** Every pass in the algorithm maps to ScheduleGraph fields:

| Algorithm Step | ScheduleGraph Field(s) |
|----------------|----------------------|
| A.1 MinII → A.2 Modulo schedule | `ScheduleLoop.II`, `ScheduleNode.{cycle, stage}` |
| A.2.5 Cluster IDs | Derived from `ScheduleNode.cycle` within each stage |
| A.3 Buffer depths | `ScheduleBuffer.count` (from stage diffs) |
| A.4 SMEM/TMEM budget | `ScheduleBuffer.sizeBytes()` × `count` |
| A.4.5 Buffer merging | `ScheduleBuffer.mergeGroupId` (planned) |
| Step 5: Emit ScheduleGraph | All fields — packages accumulated decisions into the final graph output |
| A.5 Data partitioning | DDG transform → rebuild ScheduleGraph from fresh DDG |
| A.6 List scheduling | Same `ScheduleNode`/`ScheduleEdge`, stage always 0 |
| A.7 Epilogue subtiling | DDG transform → rebuild ScheduleGraph from fresh DDG |
| B.1 Warp group partition | `ScheduleNode.warpGroup` |
| B.1.5 Replicate infra ops | Ops with `pipeline == NONE` cloned per group |
| B.2 Barrier insertion | `ScheduleBuffer(kind=BARRIER, pairedBufferId)` |
| B.3 Prologue/epilogue structure | `ScheduleLoop.{prologueNodes, epilogueNodes, maxStage}` |
| B.4 Warp counts/registers | Per-group config (planned extension) |
| C Loop expansion | Read `ScheduleLoop` prologue/kernel/epilogue nodes |
| C Non-loop reorder | Sort `ScheduleNode` by cycle/cluster within block |

DDG transformations (A.5, A.7) modify the DDG, not the ScheduleGraph directly. The iterative loop simply rebuilds the ScheduleGraph from the transformed DDG — since the ScheduleGraph is built *from* the DDG, this is natural and requires no rollback.

**Encoding buffer sharing on the ScheduleGraph:** Buffer merging (Step 4.5) is represented by a `mergeGroupId` on each `ScheduleBuffer`. Buffers with the same `mergeGroupId` share a single physical allocation — the physical size is `max(sizeBytes)` across all merged buffers, and the physical count is `max(count)`. The merge is computed from modular live-interval analysis on the ScheduleGraph: two buffers can share physical memory if their live intervals (computed from producer/consumer cycles in the modulo schedule) do not overlap across any in-flight iteration. This is checked across all `(d1, d2)` pairs of buffer instances for buffers with depths `D1` and `D2`. The ScheduleGraph also tracks the implicit ordering constraint introduced by sharing: `last_consumer_of_A` must happen-before `producer_of_B` when A and B share a buffer, which is verified for cycle-freedom in the dependency graph before accepting the merge.

**Barrier encoding:** Each multi-buffered data buffer (`kind=SMEM` or `kind=TMEM` with `count > 1`) is paired with a `ScheduleBuffer(kind=BARRIER)` via `pairedBufferId`. The barrier has the same `count` as its data buffer. At runtime, barrier phase cycling ensures correctness: the producer signals `barrier[iter % count]` after writing, and the consumer waits on the same phase before reading. The ScheduleGraph records this pairing so that Phase 3 (lowering) can emit the correct `mbarrier.init`, `mbarrier.arrive`, and `mbarrier.wait` ops. In the `dump()` output, barriers appear as `%bar0 = modulo.alloc BARRIER [N] for buf0`.

**Cross-loop boundary ports:** For nested loops (persistent kernels with outer tile loop + inner K-loop), the `ScheduleLoop.inputs` and `ScheduleLoop.outputs` vectors track values that cross the loop boundary. **Inputs** are values consumed from the outer scope: iter_args (loop-carried values like accumulators), captured values (TMA descriptors, tile offsets), and multi-buffered resources from the parent loop. **Outputs** are values yielded back to the parent via `scf.yield`. These ports drive the parent loop's scheduling — the outer `ScheduleLoop` sees the inner loop as a super-node, and the ports tell it which buffers need to be multi-buffered at the outer level.

**Non-loop regions:** The ScheduleGraph represents straight-line code (prologue, epilogue, inter-loop regions) using the same `ScheduleNode`/`ScheduleEdge` types but with different parameters. For non-loop regions: `stage` is always 0 (no cross-iteration overlap), there is no `II` (the "II" field stores the makespan instead), and the DDG has no loop-carried edges (all `distance=0`). The scheduling algorithm dispatches to list scheduling instead of modulo scheduling, but the output format is identical — `(cycle, pipeline, stage=0, cluster)`. This means downstream passes (warp group partitioning, barrier insertion, code generation) handle loop and non-loop regions uniformly.

**Conditional ops (scf.if):** Persistent kernels wrap TMA loads in conditional blocks (`scf.if i < num_iter`) for boundary handling. The DDG builder walks into `scf.if` regions to find pipeline-relevant ops (TMA loads/stores). The enclosing `scf.if` becomes a single `ScheduleNode` that inherits the **dominant pipeline** (highest latency pipeline found inside) and the corresponding latency from its contents. For example, an `scf.if` containing a `tt.descriptor_load` becomes a MEM-pipeline node with the TMA load's latency. This ensures conditional prefetch blocks are visible to the scheduler rather than being treated as opaque zero-latency ops.

#### Concrete Example: GEMM K-loop ScheduleGraph

The `dump()` output for a Blackwell GEMM K-loop (128×128 tile, K=64 per iteration) shows the complete ScheduleGraph after Phase 0 (scheduling) and Phase 1 (buffer allocation):

```
modulo.schedule @loop0 {
  ii = 1038, max_stage = 2

  %buf0 = modulo.alloc SMEM [3 x 128x64 x f16]  live=[0, 1938)  // 24576 bytes total  (A tile)
  %buf1 = modulo.alloc SMEM [3 x 64x128 x f16]   live=[519, 2457)  // 24576 bytes total  (B tile)
  %bar0 = modulo.alloc BARRIER [3] for buf0        // 24 bytes total
  %bar1 = modulo.alloc BARRIER [3] for buf1        // 24 bytes total

  modulo.stage @s0 {
    %N0 = tt.descriptor_load  {pipe: MEM, cycle: 0, cluster: 0, latency: 519, selfLatency: 519, ->buf0}
    %N1 = tt.descriptor_load  {pipe: MEM, cycle: 519, cluster: 1, latency: 519, selfLatency: 519, ->buf1}
  }

  modulo.stage @s1 {
    %N2 = ttng.tc_gen5_mma  {pipe: TC, cycle: 1038, cluster: 0, latency: 900, selfLatency: 900, <-buf0, <-buf1}
  }

  modulo.stage @s2 {
    %N3 = ttng.tmem_load  {pipe: TC, cycle: 2076, cluster: 0, latency: 200, selfLatency: 200}
  }

  edges {
    N0 -> N2  lat=519  dist=0
    N1 -> N2  lat=519  dist=0
    N2 -> N3  lat=900  dist=0
  }
}
```

Key observations:
- **3 stages** (s0, s1, s2): loads at stage 0, MMA at stage 1, tmem_load at stage 2
- **Buffer count = 3**: `floor(lifetime / II) + 1` — the A tile is live from cycle 0 (LoadA) to cycle 1938 (MMA finish), lifetime = 1938, `floor(1938 / 1038) + 1 = 2 + 1 = 3`
- **Live intervals**: `live=[0, 1938)` on buf0 and `live=[519, 2457)` on buf1 record the absolute live range (producer start to last consumer end), used by Step 4.5 to determine whether buffers can share physical memory
- **Paired barriers**: each SMEM buffer gets its own barrier with the same count
- **Buffer produce/consume refs**: `->buf0` means the node produces into buf0, `<-buf0` means it consumes from buf0. The `local_alloc` that creates the SMEM allocation is not a scheduled node — it is the buffer itself (`defOp` on `ScheduleBuffer`)

### Algorithm Summary

The algorithm proceeds in three main passes:

**Pass A — Scheduling (iterative):** An iterative refinement loop that schedules all code regions, derives pipeline depths, checks resource budgets, and applies DDG transformations — re-running until the schedule stabilizes. DDG nodes are lowered during construction (see [Op Lowering](#2-op-lowering)): each node has target-accurate `selfLatency` (pipeline occupancy) and `latency` (edge weight), and synthetic `local_load`/`local_store` nodes make buffer access explicit with symbolic, unaliased buffer references. **Loop regions** use modulo scheduling (Rau's algorithm) to minimize II; **non-loop regions** use list scheduling to minimize makespan. Both produce the same `(cycle, pipeline, stage, cluster)` output. From the schedule, it derives buffer depths (with live intervals) for all regions and merges buffers with non-overlapping lifetimes (Step 4.5). Then it considers two DDG transformations: **data partitioning** (Pass A.5) splits underutilized loop ops into sub-tiles, and **epilogue subtiling** (Pass A.7) splits monolithic TMA stores into independent sub-chains. If either transformation modifies a DDG, Pass A re-runs from the top — the freed SMEM may enable higher pipeline depth, changing II and the entire schedule. Converges in 1-2 iterations. The final output is a **ScheduleGraph** (Step 5) that packages all accumulated decisions — cycles, stages, buffers with lifetimes, merge groups — into a single side data structure for downstream passes.

**Pass B — Warp Specialization Reconstruction:** Partitions ops into warp groups based on per-pipeline utilization derived from the schedule. Pipelines with >30% utilization get dedicated warp groups; underutilized pipelines are merged into the group with the most data dependency edges. The pass then inserts barrier synchronization at cross-group boundaries, computes prologue/epilogue loop structure (prolog depth = max stage across all ops), assigns warp counts and registers, and generates the warp-specialized code structure.

**Pass C — Code Generation and Instruction Ordering:** Takes the `(stage, cluster)` assignments from Pass A and the warp-specialized code skeleton from Pass B. For **loop regions**, generates the prologue/kernel/epilogue loop structure. For **non-loop regions**, reorders ops by cluster ID. Pass C makes no scheduling decisions — all ordering is determined by Pass A's cluster IDs.

### Algorithm Flow

```
┌─────────────────────────────────────────────────────┐
│  Input: Kernel with loop and non-loop regions       │
│         DDG per region, latency table, resources    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
│         Pass A: Iterative Scheduling Loop           │
│                                                     │
│  ┌────────────────────────────────────────────────┐ │
│  │  Schedule all regions:                         │ │
│  │    Loop regions → modulo schedule (Steps 1-2)  │ │
│  │    Non-loop regions → list schedule (A.6)      │ │
│  │    Compute cluster IDs (Step 2.5)              │ │
│  └───────────────────┬────────────────────────────┘ │
│                      │                              │
│                      ▼                              │
│  ┌────────────────────────────────────────────────┐ │
│  │  Step 3: Derive pipeline depths (all regions)  │ │
│  │    num_buffers(R) = floor(lifetime(R) / II) + 1│ │
│  │  Step 4: Memory budget check                   │ │
│  │  Step 4.5: Merge non-overlapping buffers       │ │
│  └───────────────────┬────────────────────────────┘ │
│                      │                              │
│                      ▼                              │
│  ┌────────────────────────────────────────────────┐ │
│  │  DDG transformations:                          │ │
│  │    A.5: Data partitioning (loop DDGs)          │ │
│  │    A.7: Epilogue subtiling (epilogue DDG)      │ │
│  └───────────────────┬────────────────────────────┘ │
│                      │                              │
│             ┌────────┴────────┐                     │
│             │  Any DDG        │                     │
│             │  changed?       │                     │
│             └────┬───────┬────┘                     │
│              Yes │       │ No                       │
│                  │       │                          │
│       ┌──────────┘       │                          │
│       │ (re-run from     │                          │
│       │  top — new DDG   │                          │
│       │  may change II,  │                          │
│       │  depths, budget) │                          │
│       └──────────────────┤                          │
└ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
                           │ Converged
                           ▼
┌─────────────────────────────────────────────────────┐
│  Step 5: Emit ScheduleGraph                         │
│    Package all decisions into a ScheduleGraph:      │
│    cycles, stages, buffers, lifetimes, merge groups  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼  ScheduleGraph
┌─────────────────────────────────────────────────────┐
│  Pass B: Reconstruct warp specialization            │
│    Input: ScheduleGraph from Pass A                 │
│    Step 1: Partition ops into warp groups            │
│    Step 2: Insert barriers at group boundaries      │
│    Step 3: Compute per-region loop structure         │
│    Step 4: Assign warp counts and registers         │
│    Step 5: Generate TLX code skeleton               │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Pass C: Apply reordering from Pass A               │
│    Loop regions: expand prologue/kernel/epilogue    │
│    Non-loop regions: reorder ops by cluster ID      │
│    Barriers from Pass B move with their ops         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Output: Warp-specialized kernel with               │
│    - ScheduleGraph (Pass A output):                 │
│      · Per-op (cycle, pipeline, stage, cluster)     │
│      · Per-buffer (count, liveStart, liveEnd)       │
│      · Buffer merge groups                          │
│    - Warp group partitioning (Pass B)               │
│    - Barrier synchronization (Pass B)               │
│    - Prologue/epilogue structure (Pass C)            │
│    - Per-warp instruction ordering (Pass C)         │
└─────────────────────────────────────────────────────┘

Convergence: typically 1-2 iterations. Iteration 1 computes the
initial schedule; if A.5 or A.7 transform a DDG, iteration 2
re-schedules with the refined DDG and updated SMEM budget.
Further iterations are rare — the transformations are idempotent
(a subtiled store won't be subtiled again).
```

### Worked Examples

The algorithm is illustrated with three worked examples of increasing complexity:

1. **Blackwell GEMM** (`blackwell_gemm_ws.py`): 2 active pipelines (MEM, TC), MEM-bound (II=1280), 3 warp groups. All ops at stage=0. The simplest case — no cross-iteration pipelining needed.

2. **Blackwell FA Forward** (`blackwell_fa_ws.py` and `blackwell_fa_ws_pipelined.py`): 4 active pipelines, TC-bound (II=1800), 4 warp groups. Data partitioning splits MMA ops into 2 groups. The pipelined variant assigns PV_g1 to stage=1, creating the in-group interleaving QK_g0[i] → PV_g1[i-1] → QK_g1[i] → PV_g0[i] that eliminates softmax stalls on the TC pipeline.

3. **Blackwell FA Backward** (`blackwell_fa_ws_pipelined_persistent.py`): 5 MMA ops per iteration, heavily TC-bound (II=4500), 4 warp groups. The MMA group uses a prolog/main/epilog structure to pipeline dK/dQ from iteration j-1 with QK/dP/dV from iteration j. TMEM buffer merging (dP/dQ share physical memory) is essential to fit within the 256KB limit.

### Limitations and Assumptions

The algorithm as described has several limitations:

1. **Static latencies**: The algorithm uses fixed cycle counts from microbenchmarks. In practice, latencies vary with memory access patterns (L2 hit vs miss), tile sizes, and occupancy. The schedule is optimal for the assumed latencies but may not be optimal at runtime.

2. **Multi-region scheduling**: The algorithm schedules each code region (loop or straight-line) independently. Kernels with nested loops (e.g., persistent kernels iterating over both tiles and K/V blocks) treat each loop as a separate scheduling problem. Cross-region interactions (e.g., epilogue-to-prologue overlap across tiles) are handled by the outer region's schedule, which models inner regions as super-nodes with known latency.

3. **No dynamic scheduling**: The schedule is computed at compile time and embedded in the generated code. It cannot adapt to runtime conditions like varying sequence lengths, cache behavior, or SM occupancy. The prolog/epilog structure is fixed.

4. **Barrier overhead not modeled in Pass A**: The modulo schedule does not account for the ~20-30 cycle cost of barrier wait/arrive operations. For kernels with many cross-group barriers per iteration (e.g., FA backward with ~20 barrier types), this overhead can shift actual timings relative to the schedule. A more accurate model would include barrier costs in the latency table.

5. **1:1 pipeline-to-warp-group assumption**: Pass A assumes each warp group maps to a single pipeline. This holds for MMA (TC) and producer (MEM) groups but breaks for mixed groups like the epilogue (CUDA + MEM) or compute (CUDA + SFU). For mixed groups, Pass A would need to schedule across multiple pipelines within one warp group.

6. **No multi-CTA or cluster-level scheduling**: The algorithm schedules within a single CTA. Multi-CTA kernels (e.g., `blackwell_gemm_2cta.py`) require additional coordination for cross-CTA B-tile sharing and cluster-level barrier synchronization, which is handled separately.

7. **Register allocation is approximate**: Pass B Step 4 estimates register usage from live variable counts but doesn't perform full register allocation. The actual register count is determined by the compiler backend (ptxas), which may differ from the estimate and cause spills that the schedule didn't anticipate.

---

## Inputs

### 1. Instruction Dependency Graph (DDG)

A **data dependency graph with loop-carried edges**:
- **Nodes** = operations (LoadK, LoadV, QK_MMA, Softmax sub-ops, PV_MMA, etc.)
- **Intra-iteration edges** (distance=0): producer-consumer within one iteration
  - e.g., LoadK[i] → QK[i], QK[i] → RowMax[i]
- **Loop-carried edges** (distance=d): cross-iteration dependencies
  - e.g., Acc[i] → AccUpdate[i+1] (distance=1)
  - e.g., m_i[i] → Alpha[i+1] (distance=1)

Example (Flash Attention forward, one iteration body):
```
LoadK ──→ QK ──→ RowMax ──→ Scale/Sub ──→ Exp2 ──→ RowSum ──→ AccUpdate ──→ PV
LoadV ───────────────────────────────────────────────────────────────────────→ PV
                                                                              │
Loop-carried edges (distance=1):                                              │
  Acc ─────────────────────────────────────────────→ AccUpdate (next iter)     │
  m_i ───→ Alpha (next iter)                                                  │
  l_i ───→ l_update (next iter)                                               │
```

Each edge `(u, v)` carries:
- `latency(u, v)`: minimum cycles between start of u and start of v
- `distance(u, v)`: iteration distance (0 = same iteration, 1 = next iteration, etc.)

### 2. Op Lowering

The DDG is not a literal mirror of the IR. During DDG construction, ops are **lowered** to expose target-specific details that the scheduler needs but the IR does not represent. **Op lowering does not modify the IR** — it only affects how DDG nodes are constructed.

#### Why Lower

1. **Fine-grained modeling**: The scheduler sees actual pipeline occupancy (`selfLatency`) separately from async completion time (`latency`). This enables better overlap — e.g., back-to-back TMA issues on the MEM pipeline instead of serialized loads that block for the full transfer time.

2. **Target portability**: The same DDG structure (nodes, edges, buffer references) works across targets. For AMDGPU, where memory ops have different pipeline characteristics, only the `selfLatency` / `latency` values change — the scheduling algorithm and buffer tracking are target-independent.

3. **Symbolic memory**: Buffers are named and unaliased in the DDG — no index arithmetic, no phase cycling, no `buf_idx = i % depth`. All buffer indexing is deferred to code generation (Pass C). This keeps the scheduling model clean and enables buffer merging (Step 4.5) without rewriting index expressions. The DDG reasons about `buf_A` and `buf_B` as abstract names; the physical layout is decided later.

#### DDG Node to IR Mapping

Each DDG node has an optional `irOp` pointer back to the TTGIR op it models:

- **Real nodes** (e.g., `tma_load`, `mma`, `local_store`): `irOp` points to the corresponding TTGIR op. Phase 3 (Pass C) uses this pointer to apply schedule decisions (cycle, stage, cluster) to the original IR.
- **Synthetic nodes** (e.g., `local_load`): `irOp = NULL` — there is no corresponding IR op. These nodes exist only in the DDG for buffer lifetime tracking and barrier placement. Pass C skips them.

Additionally, each node carries a buffer reference (`→buf` for producers, `←buf` for consumers) that connects it to the symbolic buffer it accesses. This is how the scheduler traces the data flow through SMEM/TMEM without relying on IR pointers.

| DDG Node | `irOp` | Buffer Ref | Used By |
|----------|--------|-----------|---------|
| `tma_load` (real) | → `tt.descriptor_load` | `→buf` (producer) | Pass C: schedule the IR op |
| `local_load` (synthetic) | NULL | `←buf` (consumer) | Step 3: end buffer lifetime; Pass B: place barrier |
| `mma` (real) | → `ttng.tc_gen5_mma` | — | Pass C: schedule the IR op |
| `local_store` (real) | → `ttg.local_store` | `→buf` (producer) | Pass C: schedule the IR op |
| `tma_store` (real) | → `tt.descriptor_store` | `←buf` (consumer) | Pass C: schedule the IR op |

#### Lowering Refinements

Lowering introduces two kinds of refinements:

1. **selfLatency ≠ latency**: A single DDG node with `selfLatency` (pipeline occupancy) shorter than `latency` (time until result is available). The modulo scheduler blocks `selfLatency` consecutive reservation table slots, while using `latency` as the edge weight to consumers. This models async ops like TMA loads without extra nodes.

2. **Synthetic DDG nodes**: Nodes with `irOp = NULL` that do not correspond to any IR op. Currently only `local_load` — it makes buffer consumption explicit so the scheduler can track buffer lifetimes precisely and Pass B can insert barriers at the correct producer-consumer boundaries.

#### Synthetic Nodes: local_load and local_store

The DDG introduces **synthetic nodes** that do not correspond to any IR op. These make buffer access explicit so the scheduler can track buffer lifetimes precisely.

- **`local_load`** (synthetic): Marks the point where an op **finishes reading** from a buffer. The buffer lifetime **ends** here. Has `selfLatency = 0` and `pipeline = NONE` — it doesn't occupy any hardware resource. It exists as the explicit buffer consumer that drives lifetime analysis and barrier insertion.

- **`local_store`** (real or synthetic): Marks the point where data is **written** to a buffer. For TMA loads, there is no synthetic `local_store` — the TMA hardware writes directly to SMEM, so the `tma_load` DDG node itself is the buffer producer (`→buf`). For the epilogue path, `local_store` corresponds to a real IR op (`ttg.local_store`) that writes registers to SMEM.

Each buffer reference is:
- **Symbolic**: Named (e.g., `buf_A`, `buf_B`), not a raw SMEM address
- **Trackable**: The scheduler can trace the full chain: `tma_load →buf→ local_load → consumer`
- **Unaliased**: Each symbolic buffer maps to exactly one logical allocation. No two buffer names alias the same memory — until Step 4.5 explicitly merges them via `mergeGroupId`

#### Example: GEMM K-loop with Lowered DDG

The IR has three ops: `tt.descriptor_load` (×2) and `ttng.tc_gen5_mma`. The lowered DDG exposes the buffer flow, matching the TLX `blackwell_gemm_ws` kernel where `async_descriptor_load` writes directly into SMEM buffers and `async_dot` reads from them:

```
IR ops (unchanged):          DDG nodes (lowered):

tt.descriptor_load A    →    tma_load_A  {pipe: MEM, selfLat: 20, lat: 520, →buf_A}
                             local_load_A {pipe: NONE, selfLat: 0, ←buf_A}  // synthetic

tt.descriptor_load B    →    tma_load_B  {pipe: MEM, selfLat: 20, lat: 520, →buf_B}
                             local_load_B {pipe: NONE, selfLat: 0, ←buf_B}  // synthetic

ttng.tc_gen5_mma        →    mma {pipe: TC, selfLat: 900, lat: 900}

Edges:
  tma_load_A → local_load_A (lat: 520)    // TMA writes directly to SMEM buf_A
  local_load_A → mma (lat: 0)             // MMA reads operand A from buf_A
  tma_load_B → local_load_B (lat: 520)
  local_load_B → mma (lat: 0)             // MMA reads operand B from buf_B

Buffer lifetimes (for Step 3):
  buf_A: live from tma_load_A (producer) to local_load_A (last consumer)
  buf_B: live from tma_load_B (producer) to local_load_B (last consumer)
```

The `tma_load` is the buffer **producer** — TMA writes directly to the SMEM buffer, no intermediate store. The synthetic `local_load` is the buffer **consumer** — it marks when MMA finishes reading from the buffer, ending the buffer's lifetime. This matches the TLX pattern where `async_descriptor_load` fills `buffers_A[buf]` and `async_dot` reads from it, with `mBarriers=[A_smem_empty_bars[buf]]` signaling when the read is done.

#### Epilogue Path: local_store as Real IR Op

In the epilogue, `local_store` corresponds to a real IR op (`ttg.local_store`). The data flows from TMEM through registers into SMEM, then out via TMA:

```
tmem_load {pipe: TC, selfLat: 200}
  → truncf {pipe: CUDA, selfLat: 100}
    → local_store {pipe: MEM, selfLat: 150, →buf_out}    // real IR op, writes to SMEM
      → tma_store {pipe: MEM, selfLat: 20, lat: 600, ←buf_out}
```

Here `local_store` is a real DDG node (not synthetic) with `pipeline = MEM` and real `selfLatency` because it's an actual SMEM write that occupies the MEM pipeline.

#### selfLatency / latency Summary (Blackwell)

| TTGIR Op | DDG Node(s) | selfLatency | latency | Pipeline |
|----------|------------|----------:|--------:|----------|
| `tt.descriptor_load` | `tma_load` (→buf) + `local_load` (←buf, synthetic) | 20 / 0 | 520 / 0 | MEM / NONE |
| `tt.descriptor_store` | `tma_store` (←buf) | 20 | 600 | MEM |
| `ttg.local_store` | `local_store` (→buf, real IR op) | 150 | 150 | MEM |
| `ttng.tc_gen5_mma` | `mma` | 900 | 900 | TC |
| `ttng.tmem_load` | `tmem_load` | 200 | 200 | TC |
| CUDA/SFU ops | 1:1 | varies | = selfLatency | CUDA/SFU |

### 3. Functional Unit Mapping

Each op is assigned to exactly one hardware pipeline:

| Pipeline | Operations |
|----------|-----------|
| **MEM** | TMA loads, TMA stores, local_store (real IR op) |
| **TC** | wgmma / tcgen05.mma, tmem_load |
| **CUDA** | rowmax, rowsum, scale, acc update, type conversions |
| **SFU** | exp2, rsqrt, other transcendentals |
| **NONE** | Synthetic local_load (buffer lifetime endpoint) |

### 4. Latency Table

Execution time per operation in cycles (from microbenchmarks):

| Operation | Latency (cycles) | Pipeline |
|-----------|----------------:|----------|
| TMA Load 128x64 | 640 | MEM |
| tcgen05.mma 128x128x128 | 900 | TC |
| tcgen05.mma 128x128x64 | 559 | TC |
| RowMax (QK) | 336 | CUDA |
| Scale & Subtract | 130 | CUDA |
| Exp2 (elementwise) | 662 | SFU |
| Alpha = Exp2(scalar) | 43 | SFU |
| RowSum (P) | 508 | CUDA |
| Acc x Alpha | 105 | CUDA |

### 5. Resource Model

- Each pipeline can execute **one op at a time** per warpgroup
- Distinct pipelines **can overlap** (MEM + TC + CUDA + SFU all concurrent)
- An op **occupies** its pipeline for its full latency duration

---

## Pass A: Scheduling (Iterative)

Pass A is an **iterative refinement loop**. It schedules all regions, derives pipeline depths, checks resource budgets, and then applies DDG transformations (data partitioning, epilogue subtiling) that may improve the schedule. If any transformation modifies a DDG, Pass A re-runs from the top — the new DDG may change II, pipeline depths, or SMEM budget, requiring a fresh schedule.

```python
def pass_a(kernel_regions, latency_model, memory_budget):
    """
    Iterative scheduling loop. Converges when no DDG transformation
    improves the schedule. Typically 1-2 iterations.

    Precondition: each DDG node has target-accurate selfLatency
    (pipeline occupancy) and latency (edge weight to consumers),
    set during DDG construction.
    """
    while True:
        # Schedule all regions
        for region in kernel_regions:
            if region.has_loop_carried_edges:
                # Steps 1-2: modulo schedule
                MinII = max(compute_ResMII(region.DDG), compute_RecMII(region.DDG))
                region.schedule, region.II = modulo_schedule(region.DDG, MinII)
            else:
                # A.6: list schedule
                region.schedule, region.makespan = list_schedule(region.DDG)

            # Step 2.5: cluster IDs
            region.cluster_ids = compute_cluster_ids(region.schedule, region.II)

        # Steps 3-4: pipeline depths + budget check
        pipeline_config = derive_pipeline_depths(kernel_regions)
        pipeline_config = adjust_for_memory(pipeline_config, memory_budget)
        pipeline_config = merge_buffers(pipeline_config)  # Step 4.5

        # DDG transformations
        ddg_changed = False

        # A.5: data partitioning (loop regions)
        for region in kernel_regions:
            if region.is_loop and has_underutilized_pipeline(region):
                if data_partition(region):
                    ddg_changed = True

        # A.7: epilogue subtiling (non-loop regions with TMA stores)
        for region in kernel_regions:
            if not region.is_loop and has_tma_store(region):
                S = try_epilogue_subtiling(region, pipeline_config, memory_budget)
                if S > 1:
                    split_epilogue_stores(region, S)
                    ddg_changed = True

        if not ddg_changed:
            break  # Converged

    # Step 5: Emit ScheduleGraph
    return build_schedule_graph(kernel_regions, pipeline_config)
```

The iteration converges because:
- DDG transformations are **idempotent**: a subtiled store won't be subtiled again, a partitioned op won't be partitioned again
- Each transformation **monotonically improves** the objective (lower makespan, lower SMEM, or both)
- The number of possible transformations is bounded (finite ops, finite subtile factors)

In practice, iteration 1 computes the initial schedule. If A.5 or A.7 transform a DDG, iteration 2 re-schedules with the refined DDG and updated SMEM budget. Iteration 3 is rare.

### Step 1: Compute Minimum Initiation Interval (II)

The II is the number of cycles between the start of consecutive iterations in steady state. It is bounded from below by two constraints:

#### Resource-constrained II (ResMII)

Each pipeline can only execute one op at a time. The minimum II is at least the total work on the busiest pipeline:

```python
def compute_ResMII(ops, latencies, unit_map):
    """
    ResMII = max over all pipelines of total latency on that pipeline.
    """
    pipe_load = defaultdict(int)
    for op in ops:
        pipe_load[unit_map[op]] += latencies[op]
    return max(pipe_load.values())
```

Example (FA forward, 128x128 tiles):
```
MEM:  LoadK(640) + LoadV(640)                           = 1280
TC:   QK(779) + PV(779)                                 = 1558
CUDA: RowMax(336) + Scale(130) + RowSum(508) + Acc(105)  = 1079
SFU:  Exp2(662) + Alpha(43)                              = 705

ResMII = max(1280, 1558, 1079, 705) = 1558  (TC-bound)
```

#### Recurrence-constrained II (RecMII)

Loop-carried dependencies form recurrence circuits. For each circuit, the II must be large enough that iteration i+d finishes its consumer after iteration i finishes its producer:

```python
def compute_RecMII(DDG, latencies):
    """
    RecMII = max over all recurrence circuits C of:
        sum(latency(e) for e in C) / sum(distance(e) for e in C)

    A recurrence circuit is a cycle in the DDG when loop-carried
    edges are included.
    """
    max_rec = 0
    for circuit in find_all_elementary_circuits(DDG):
        total_latency = sum(latencies[e.src] for e in circuit)
        total_distance = sum(e.distance for e in circuit)
        if total_distance > 0:
            max_rec = max(max_rec, ceil(total_latency / total_distance))
    return max_rec
```

Example (FA forward):
```
Recurrence: AccUpdate[i] ---(d=1)--→ AccUpdate[i+1]
  Path: AccUpdate → ... → PV → AccUpdate
  Total latency along path: 105 + ... + 779 ≈ 3982
  Distance: 1
  RecMII contribution: 3982

But this recurrence includes ALL ops in the iteration body, so:
  RecMII ≈ total_single_iteration_latency (for distance-1 loops)
```

For FA, the recurrence through the accumulator is effectively the entire iteration, so RecMII ≈ 3982 (sequential) before any overlap. The modulo schedule's job is to achieve II close to ResMII by overlapping multiple iterations.

#### MinII

```python
MinII = max(ResMII, RecMII)
```

In practice for FA, the RecMII through the accumulator is long but can be broken by **pipelining the accumulator** (multiple acc buffers), effectively reducing the recurrence distance. With 2 acc buffers, `distance=2`, cutting RecMII in half.

### Step 2: Modulo Reservation Table Scheduling

Schedule each op into a slot within the II-length reservation table. Multiple iterations overlap in steady state.

#### Background: Rau's Iterative Modulo Scheduling

Rau's algorithm (B. Ramakrishna Rau, "Iterative Modulo Scheduling: An Algorithm For Software Pipelining Loops", 1994) is the standard algorithm for **software pipelining** — overlapping multiple loop iterations on a set of hardware resources. The core idea:

1. **Modulo reservation table**: A table of length II (initiation interval) with one row per hardware resource (pipeline). A slot `[cycle % II][pipeline]` can hold at most one op. Because the table wraps modulo II, placing an op at cycle `t` means it occupies slot `t % II` — and this slot is reused by the *same* op from every subsequent iteration, spaced II cycles apart.

2. **Iterative placement**: Ops are placed one at a time in priority order (highest critical path first). For each op, compute the earliest cycle it can start (based on predecessor completion times and loop-carried distances), then scan forward for a free slot on its pipeline. If no slot is free within II cycles, either **eject** a less-critical op (backtracking) or increase II and restart.

3. **Loop-carried edges**: An edge with distance `d` means the consumer in iteration `i+d` depends on the producer in iteration `i`. The constraint becomes: `consumer_start >= producer_start + latency - d * II`. This allows the consumer to start *before* the producer in the modulo table (negative offset), because it's actually `d` iterations later in absolute time.

4. **Termination**: The algorithm is guaranteed to find a valid schedule if II is large enough (worst case: II = total latency of all ops on the busiest pipeline). In practice, it usually succeeds at or near MinII.

The algorithm is adapted here for GPU multi-pipeline scheduling, where the "resources" are the MEM, TC, CUDA, and SFU pipelines rather than traditional VLIW functional units.

```python
def modulo_schedule(DDG, latencies, unit_map, MinII):
    """
    Iterative modulo scheduling (Rau's algorithm adapted for multi-pipeline GPU).

    Returns:
        schedule: dict mapping op -> (cycle_within_II, pipeline)
        II: the achieved initiation interval
    """

    II = MinII

    while True:  # Increase II if scheduling fails
        # Reservation table: which pipeline slots are occupied
        # res_table[cycle_mod_II][pipeline] = op or None
        res_table = [[None] * NUM_PIPELINES for _ in range(II)]

        # Compute scheduling order: ops sorted by critical path height
        # (bottom-up, longest path to any sink including loop-carried)
        height = compute_heights(DDG, latencies)
        sorted_ops = sorted(DDG.nodes, key=lambda n: -height[n])

        schedule = {}
        success = True

        for op in sorted_ops:
            pipe = unit_map[op]

            # Compute earliest start time for this op
            earliest = 0
            for pred in predecessors(op):
                if pred in schedule:
                    pred_cycle = schedule[pred][0]
                    edge = DDG.edge(pred, op)
                    # Account for loop-carried distance:
                    # pred in iteration (i - distance) started at
                    # pred_cycle - distance * II
                    earliest = max(
                        earliest,
                        pred_cycle + latencies[pred] - edge.distance * II
                    )

            # Search for selfLatency consecutive free slots in
            # [earliest, earliest + II) on the required pipeline.
            # selfLatency is how long the op blocks the pipeline;
            # latency (used for edge weights) may be longer for
            # async ops like TMA loads.
            self_lat = self_latencies[op]
            placed = False
            for t in range(earliest, earliest + II):
                # Check that all slots [t, t+selfLatency) are free (mod II)
                if all(res_table[(t + d) % II][pipe] is None
                       for d in range(self_lat)):
                    for d in range(self_lat):
                        res_table[(t + d) % II][pipe] = op
                    schedule[op] = (t, pipe)
                    placed = True
                    break

            if not placed:
                # Try to eject a less-critical op (Rau's backtracking)
                ejected = eject_least_critical(res_table, pipe, earliest, II, height)
                if ejected:
                    # Re-place ejected op later
                    del schedule[ejected]
                    res_table[schedule[ejected][0] % II][pipe] = None
                    # Place current op
                    slot = earliest % II
                    res_table[slot][pipe] = op
                    schedule[op] = (earliest, pipe)
                    # Re-schedule ejected op (recursive)
                    # ... (standard Rau backtracking)
                else:
                    success = False
                    break

        if success:
            return schedule, II

        II += 1  # Try larger II
```

### Step 2.5: Compute Cluster IDs from the Modulo Schedule

After the modulo schedule assigns each op a `(cycle, pipeline)`, compute **cluster IDs** that encode within-stage instruction ordering for the downstream code generator.

```python
def compute_cluster_ids(schedule, II):
    """
    Assign dense cluster IDs to ops within each stage, sorted by cycle.

    Ops in the same stage but at different cycles get different cluster IDs.
    Ops at the same cycle within a stage share a cluster ID (they can be
    emitted in any order relative to each other).

    The code generator (Pass B Step 6) emits ops in (stage, cluster) order,
    so cluster IDs directly control the instruction emission sequence.

    Returns:
        cluster_ids: dict mapping op -> cluster_id
    """
    # Group ops by stage
    stage_ops = defaultdict(list)
    for op, (cycle, pipeline) in schedule.items():
        stage = cycle // II
        stage_ops[stage].append((cycle, op))

    cluster_ids = {}
    for stage, ops_with_cycles in stage_ops.items():
        # Sort by cycle, deduplicate cycle values, assign dense IDs
        unique_cycles = sorted(set(c for c, _ in ops_with_cycles))
        cycle_to_cluster = {c: i for i, c in enumerate(unique_cycles)}
        for cycle, op in ops_with_cycles:
            cluster_ids[op] = cycle_to_cluster[cycle]

    return cluster_ids
```

The full schedule output is now `schedule[op] = (cycle, pipeline, stage, cluster)` where `stage = cycle // II` and `cluster = dense_rank(cycle)` within each stage.

### Step 3: Derive Per-Region Pipeline Depth from the Modulo Schedule

This is the key question: **given the modulo schedule, how many pipeline stages does each shared resource need in each warp-specialized region?**

#### Core Principle

A shared resource (e.g., K tile in SMEM) is **live** from when its producer writes it to when its last consumer reads it. In the modulo schedule, the producer and consumer may be in different iterations. The number of buffers needed equals the maximum number of simultaneously live instances:

```python
def compute_pipeline_depth(schedule, DDG, latencies, II):
    """
    For each shared resource, compute the number of pipeline stages
    (multi-buffer depth) required by the modulo schedule.

    The key formula:
        num_buffers(R) = floor(lifetime(R) / II) + 1

    where lifetime(R) = time from producer start to last consumer end,
    measured within the modulo schedule.

    Returns:
        buffer_depths: dict mapping resource_name -> num_stages
    """
    buffer_depths = {}

    for resource in shared_resources(DDG):
        producer = resource.producer_op    # e.g., LoadK
        consumers = resource.consumer_ops  # e.g., [QK_MMA]

        # Producer writes at cycle schedule[producer][0]
        prod_time = schedule[producer][0]

        # Last consumer finishes reading at:
        last_consumer_end = max(
            schedule[c][0] + latencies[c]
            for c in consumers
        )

        # Lifetime: how long this resource instance stays live
        # across the modulo-scheduled timeline
        lifetime = last_consumer_end - prod_time

        # Number of iterations that overlap during this lifetime
        num_buffers = (lifetime // II) + 1

        buffer_depths[resource.name] = num_buffers

    return buffer_depths
```

#### Worked Example (FA Forward)

Suppose the modulo schedule achieves II = 1600 cycles:

```
Resource: K_tile (SMEM)
  Producer: LoadK at cycle 0, latency 640
  Consumer: QK_MMA at cycle 640, latency 779
  Last consumer end: 640 + 779 = 1419
  Lifetime: 1419 - 0 = 1419
  num_buffers = floor(1419 / 1600) + 1 = 0 + 1 = 1
  → Single-buffered (consumer finishes within same II)

Resource: V_tile (SMEM)
  Producer: LoadV at cycle 1280, latency 640
  Consumer: PV_MMA at cycle 3203, latency 779
  Last consumer end: 3203 + 779 = 3982
  Lifetime: 3982 - 1280 = 2702
  num_buffers = floor(2702 / 1600) + 1 = 1 + 1 = 2
  → Double-buffered (V from iter i still live when iter i+1 starts)

Resource: Accumulator (TMEM)
  Producer: AccUpdate at cycle 3098
  Consumer: AccUpdate at cycle 3098 + II = 4698 (next iteration, loop-carried)
  But PV_MMA also writes to acc at cycle 3203-3982
  Lifetime spans the full recurrence
  num_buffers depends on whether we can ping-pong:
    If acc[i] is consumed before acc[i+1] is produced → 1 buffer
    If they overlap → 2 buffers (ping-pong)
```

#### Per-Region Buffer Depth

When ops are partitioned into warp-specialized regions, the buffer depth for a resource **at the boundary between two regions** depends on the **cross-region latency**:

```python
def compute_per_region_pipeline_depth(schedule, regions, DDG, II):
    """
    For each cross-region resource transfer, compute the buffer depth
    needed at that specific boundary.

    A region boundary exists where a producer in region R_p sends data
    to a consumer in region R_c via shared memory + barrier.

    The buffer depth at this boundary =
        floor(cross_region_lifetime / II) + 1

    where cross_region_lifetime =
        (time consumer finishes using the buffer)
        - (time producer starts writing the buffer)
        + (barrier synchronization overhead)
    """
    boundary_depths = {}

    for resource in cross_region_resources(DDG, regions):
        producer_region = region_of(resource.producer_op, regions)
        consumer_region = region_of(resource.consumer_op, regions)

        # Time the producer starts writing (within its region's schedule)
        t_produce_start = schedule[resource.producer_op][0]

        # Time the consumer finishes reading
        t_consume_end = (
            schedule[resource.consumer_op][0]
            + latencies[resource.consumer_op]
        )

        # Cross-region lifetime includes:
        # 1. Producer write time
        # 2. Barrier signaling overhead
        # 3. Consumer wait + read time
        cross_lifetime = t_consume_end - t_produce_start

        # How many iterations of the producer can be in-flight
        # before the consumer releases the buffer?
        depth = (cross_lifetime // II) + 1

        boundary_depths[(producer_region, consumer_region, resource)] = depth

    return boundary_depths
```

#### Deriving Prologue and Epilogue Depth

The pipeline depth also determines the **prologue** (ramp-up) and **epilogue** (drain) of the software pipeline:

```python
def compute_prologue_epilogue(buffer_depths, II):
    """
    Prologue: number of iterations the producer must run ahead
    before the consumer can start.

    Epilogue: number of iterations the consumer must drain
    after the producer stops.

    For a resource with buffer depth D:
        prologue_depth = D - 1
            (producer fills D-1 buffers before consumer starts)
        epilogue_depth = D - 1
            (consumer processes D-1 remaining buffers after producer stops)
    """
    max_depth = max(buffer_depths.values())

    prologue_iters = max_depth - 1
    epilogue_iters = max_depth - 1

    # In practice, different resources may have different depths.
    # The prologue must satisfy ALL resources:
    # prologue_iters = max(depth - 1 for depth in buffer_depths.values())

    return prologue_iters, epilogue_iters
```

#### Putting It Together: Pipeline Configuration

```python
def derive_pipeline_config(schedule, DDG, latencies, regions, II):
    """
    Complete pipeline configuration from the modulo schedule.

    Returns:
        PipelineConfig with:
        - per-resource buffer depths
        - per-region prologue/epilogue structure
        - barrier phase cycling depth
    """
    # Step 1: Global buffer depths
    buffer_depths = compute_pipeline_depth(schedule, DDG, latencies, II)

    # Step 2: Per-region boundary depths
    boundary_depths = compute_per_region_pipeline_depth(
        schedule, regions, DDG, II
    )

    # Step 3: Prologue/epilogue
    prologue, epilogue = compute_prologue_epilogue(buffer_depths, II)

    # Step 4: Barrier phase cycling
    # Barriers cycle through phases 0, 1, ..., (depth-1)
    # Phase at iteration i = i % depth
    barrier_phases = {}
    for (prod_region, cons_region, resource), depth in boundary_depths.items():
        barrier_phases[(prod_region, cons_region)] = depth
        # Allocate 'depth' mbarriers for this boundary
        # Consumer waits on phase = i % depth
        # Producer signals phase = i % depth

    # Step 5: Validate resource constraints
    total_smem = sum(
        resource.size_bytes * buffer_depths[resource.name]
        for resource in shared_resources(DDG)
        if resource.storage == SMEM
    )
    assert total_smem <= MAX_SMEM, (
        f"Pipeline depth requires {total_smem}B SMEM, "
        f"exceeds limit {MAX_SMEM}B. Reduce II or buffer sizes."
    )

    total_tmem = sum(
        resource.size_bytes * buffer_depths[resource.name]
        for resource in shared_resources(DDG)
        if resource.storage == TMEM
    )
    assert total_tmem <= MAX_TMEM, (
        f"Pipeline depth requires {total_tmem}B TMEM, "
        f"exceeds limit {MAX_TMEM}B."
    )

    return PipelineConfig(
        buffer_depths=buffer_depths,
        boundary_depths=boundary_depths,
        prologue_iters=prologue,
        epilogue_iters=epilogue,
        barrier_phases=barrier_phases,
        II=II,
    )
```

### Step 4: Handling Resource Pressure (SMEM/TMEM Budget)

If the derived pipeline depth exceeds available SMEM or TMEM, the algorithm must back off:

```python
def adjust_pipeline_for_memory(pipeline_config, memory_budget):
    """
    If pipeline depth requires more SMEM/TMEM than available,
    reduce buffer depths and accept a larger II.

    Strategy: reduce depth of the resource with the largest
    size * depth product first.
    """
    while total_memory(pipeline_config) > memory_budget:
        # Find the most expensive resource
        worst = argmax(
            pipeline_config.buffer_depths,
            key=lambda r: resource_size(r) * pipeline_config.buffer_depths[r]
        )

        # Reduce its depth by 1
        pipeline_config.buffer_depths[worst] -= 1

        if pipeline_config.buffer_depths[worst] < 1:
            raise Error(f"Cannot fit {worst} even with depth=1")

        # Recompute: reduced depth means the producer must stall
        # until a buffer is freed → effective II increases
        new_lifetime = pipeline_config.buffer_depths[worst] * pipeline_config.II
        # The consumer must finish within new_lifetime cycles
        # If it can't, II must increase
        pipeline_config.II = recompute_II(pipeline_config)

    return pipeline_config
```

### Step 4.5: Lifetime-Aware Buffer Merging

SMEM and TMEM buffers can be **reused** between different logical resources if their live intervals do not overlap, **including across overlapping iterations** in the modulo schedule. This is analogous to register allocation by graph coloring, but applied to shared/tensor memory buffers.

Because the modulo schedule overlaps multiple iterations, a resource with buffer depth D has D instances in flight simultaneously, each offset by II cycles. Two resources can only share a physical buffer if **none** of their in-flight instances overlap — this requires checking all pairs of buffer instances across all in-flight iterations, not just within a single iteration.

#### Motivation

Consider Flash Attention forward where:
- **K tile** is live from cycle 0 to cycle 1419 (LoadK start → QK_MMA finish)
- **P tile** (softmax output for PV_MMA) is live from cycle ~2547 to cycle 3982

These two resources never overlap in time. Allocating them to the **same physical SMEM buffer** cuts memory usage without affecting correctness or throughput.

#### Algorithm

```python
def merge_buffers(schedule, DDG, latencies, buffer_depths, II):
    """
    Merge resources with non-overlapping lifetimes into shared
    physical buffers, similar to register allocation via
    interval graph coloring.

    Two resource instances can share a physical buffer if:
    1. They use the same storage type (both SMEM or both TMEM)
    2. Their live intervals do not overlap in the modulo schedule,
       including across all in-flight iterations (cross-iteration check)
    3. Merging does not introduce a dependency cycle
    """
    # Step 1: Compute modular live intervals for each resource
    intervals = {}
    for resource in shared_resources(DDG):
        prod_time = schedule[resource.producer_op][0]
        consume_end = max(
            schedule[c][0] + latencies[c]
            for c in resource.consumer_ops
        )
        intervals[resource.name] = ModularLiveInterval(
            start=prod_time % II,
            end=consume_end % II,
            size=resource.size_bytes,
            storage=resource.storage,
            depth=buffer_depths[resource.name],
        )

    # Step 2: Build conflict graph
    # Two resources conflict if they could be simultaneously live
    # across any combination of their in-flight buffer instances
    conflicts = {}
    for r1, iv1 in intervals.items():
        for r2, iv2 in intervals.items():
            if r1 >= r2:
                continue
            if iv1.storage != iv2.storage:
                continue
            # Check all pairs of buffer instances across in-flight iterations
            if any_instances_overlap(iv1, iv2, II):
                conflicts[(r1, r2)] = True

    # Step 3: Graph coloring = physical buffer assignment
    # Each color represents a physical buffer slot.
    # Resources assigned the same color share a physical buffer.
    coloring = greedy_color(intervals.keys(), conflicts)

    # Step 4: Verify no deadlock introduced
    # Sharing a buffer means: consumer_of_A must finish before
    # producer_of_B can write. This adds an implicit edge.
    # Reject any merge that would create a cycle in the
    # cross-group dependency graph.
    for color, resources in group_by_color(coloring).items():
        if introduces_dependency_cycle(resources, DDG):
            # Fall back: un-merge the conflicting pair
            split_color(coloring, resources)

    # Step 5: Compute physical buffer requirements
    physical_buffers = {}
    for color, resources in group_by_color(coloring).items():
        physical_buffers[color] = PhysicalBuffer(
            size=max(intervals[r].size for r in resources),
            depth=max(intervals[r].depth for r in resources),
            storage=intervals[resources[0]].storage,
            logical_resources=resources,
        )

    return physical_buffers
```

#### Modular Interval Overlap

In a modulo schedule, live intervals wrap around the II boundary. Two intervals `[a, b)` and `[c, d)` modulo II overlap if:

```python
def intervals_overlap_modular(a_start, a_end, b_start, b_end, II):
    """Check if two intervals overlap in modular arithmetic."""
    a_s, a_e = a_start % II, a_end % II
    b_s, b_e = b_start % II, b_end % II

    # Handle wrap-around intervals
    if a_s <= a_e:
        a_intervals = [(a_s, a_e)]
    else:
        a_intervals = [(a_s, II), (0, a_e)]

    if b_s <= b_e:
        b_intervals = [(b_s, b_e)]
    else:
        b_intervals = [(b_s, II), (0, b_e)]

    return any(
        s1 < e2 and s2 < e1
        for (s1, e1) in a_intervals
        for (s2, e2) in b_intervals
    )


def any_instances_overlap(iv1, iv2, II):
    """
    Check if any buffer instances of two resources overlap across
    all in-flight iterations.

    A resource R with depth D has D buffer instances in flight,
    corresponding to iterations offset by 0, II, 2*II, ..., (D-1)*II.
    Two resources can share a physical buffer only if NO pair of
    their in-flight instances overlaps.

    We check all (d1, d2) pairs where d1 ∈ [0, depth1) and d2 ∈ [0, depth2).
    The modulus is depth1 * depth2 * II to capture the full period
    of the combined buffer rotation.
    """
    for d1 in range(iv1.depth):
        for d2 in range(iv2.depth):
            offset = (d2 - d1) * II
            if intervals_overlap_modular(
                iv1.start, iv1.end,
                iv2.start + offset, iv2.end + offset,
                iv1.depth * iv2.depth * II,
            ):
                return True
    return False
```

#### Impact on Downstream Passes

1. **Memory budget check (Step 4)**: Now checks physical buffer totals instead of per-resource totals. Merging strictly reduces memory usage, so configurations that previously required depth reduction (and II increase) may now fit within budget.

2. **Barrier insertion (Pass B, Step 2)**: Merged buffers introduce implicit ordering constraints. When resource A and resource B share a physical buffer, an additional dependency edge is required:

   ```
   last_consumer_of_A  happens-before  producer_of_B
   ```

   This edge must be checked for cycle-freedom in the cross-group dependency graph. If it creates a cycle, the merge must be rejected.

3. **Code generation (Pass B, Step 5)**: Instead of separate `tlx.local_alloc` per logical resource, emit a single allocation for the physical buffer. Each logical resource becomes a view/reinterpret:

   ```python
   # Before merging:
   K_buf = tlx.local_alloc((128, 64), fp16, depth=2)
   P_buf = tlx.local_alloc((128, 128), fp16, depth=2)

   # After merging (K and P share a physical buffer):
   shared_buf_0 = tlx.local_alloc(max(K_size, P_size), uint8, depth=2)
   # K_buf and P_buf are views into shared_buf_0 at non-overlapping times
   ```

#### Constraints

- **Alignment**: TMA loads require 128-byte aligned SMEM, and tcgen05.mma has its own TMEM alignment rules. The physical buffer must satisfy the strictest alignment among all merged resources.
- **No partial overlap**: Two resources must be fully non-overlapping. If they overlap even partially, they cannot share a buffer regardless of size.
- **Deadlock safety**: Every proposed merge must pass the cycle-freedom check. This is a hard constraint — a deadlock is never acceptable, even if it would save significant memory.

### Step 5: Emit ScheduleGraph

After the iterative loop converges, all scheduling decisions are packaged into a **ScheduleGraph** — the sole output of Pass A. This graph carries every decision needed by downstream passes (B and C) without requiring them to re-derive anything from the IR or DDG.

#### ScheduleGraph Format

Each `ScheduleLoop` in the graph is emitted in the following format:

```
modulo.schedule @loop<id> {
  ii = <II>, max_stage = <maxStage>

  // Buffers: multi-buffered memory allocations with live intervals
  // live=[start, end) is the absolute cycle range: producer start to last consumer end
  %buf<id> = modulo.alloc <KIND> [<count> x <shape> x <dtype>]  live=[<start>, <end>)  // <size> bytes
  %bar<id> = modulo.alloc BARRIER [<count>] for buf<paired_id>

  // Merge groups (from Step 4.5): buffers sharing physical memory
  modulo.merge_group <group_id> { buf<id1>, buf<id2> }  // physical: <max_size> bytes x <max_count>

  // Stages: ops grouped by stage, ordered by cluster within each stage
  modulo.stage @s<N> {
    %N<id> = <mlir_op>  {pipe: <PIPE>, cycle: <C>, cluster: <K>, latency: <L>, selfLatency: <SL>, ->buf<id>, <-buf<id>}
  }

  // Edges: producer-consumer dependencies
  edges {
    N<src> -> N<dst>  lat=<L>  dist=<D>
  }
}
```

#### Field Reference

| Field | Populated by | Description |
|-------|-------------|-------------|
| `ii`, `max_stage` | Step 2 (Rau's) | Initiation interval and max pipeline stage |
| `%buf` kind, shape, dtype | DDG (`local_alloc` ops) | Memory allocation metadata |
| `%buf` count | Step 3 (`floor(lifetime / II) + 1`) | Multi-buffer depth for pipelining |
| `%buf` live=\[start, end) | Step 3 | Absolute cycle range: producer start cycle to last consumer end cycle. Buffer depth is derived from this (`floor((end - start) / II) + 1`). Step 4.5 projects onto `[0, II)` for modular overlap checks. |
| `%bar` | Step 3 | Paired barrier with same count as its data buffer |
| `merge_group` | Step 4.5 | Buffers sharing physical memory (non-overlapping lifetimes) |
| `pipe`, `cycle`, `cluster`, `stage` | Steps 1-2, 2.5 | Hardware pipeline, scheduled cycle, within-stage emission order, pipeline stage |
| `latency`, `selfLatency` | Latency model | Total latency and pipeline-occupancy latency |
| `->buf`, `<-buf` | DDG | Buffer produce/consume references |
| `lat`, `dist` | DDG | Edge latency and iteration distance |

#### Construction

```python
def build_schedule_graph(kernel_regions, pipeline_config):
    """
    Package all accumulated decisions into the ScheduleGraph.
    This is the sole output of Pass A — downstream passes read
    only the graph, never the raw DDG or schedule tables.
    """
    graph = ScheduleGraph()

    for region in kernel_regions:
        loop = graph.add_loop(region.loop_op)
        loop.II = region.II
        loop.maxStage = region.schedule.max_stage

        # Nodes: one per scheduled DDG node
        for node in region.DDG.nodes:
            sn = loop.add_node(node.op)
            sn.cycle = region.schedule[node]
            sn.stage = sn.cycle // loop.II
            sn.pipeline = node.pipeline
            sn.latency = node.latency
            sn.selfLatency = node.selfLatency

        # Edges: inherited from DDG
        for edge in region.DDG.edges:
            loop.add_edge(edge.src, edge.dst, edge.latency, edge.distance)

        # Buffers: with lifetimes from Step 3
        for resource in region.shared_resources:
            buf = loop.add_buffer(resource)
            buf.count = pipeline_config.buffer_depths[resource.name]
            buf.liveStart = pipeline_config.live_intervals[resource.name].start
            buf.liveEnd = pipeline_config.live_intervals[resource.name].end

            # Paired barrier
            bar = loop.add_buffer(MemoryKind.BARRIER, count=buf.count)
            bar.pairedBufferId = buf.id
            buf.pairedBufferId = bar.id

        # Merge groups: from Step 4.5
        for group_id, resources in pipeline_config.merge_groups.items():
            for resource in resources:
                loop.get_buffer(resource).mergeGroupId = group_id

    return graph
```

See [Concrete Example: GEMM K-loop ScheduleGraph](#concrete-example-gemm-k-loop-schedulegraph) for a complete instance of this format.

---

## Pass A.5: Data Partitioning for Improved Overlap (Optional)

When the schedule has significant idle gaps on some pipelines, split large ops into sub-tiles to create finer-grained scheduling opportunities.

```python
def data_partition_for_overlap(schedule, DDG, latencies, unit_map, II):
    """
    Split ops into sub-tiles when a pipeline has idle gaps > threshold.

    Splitting an op of latency L into N sub-ops of latency L/N
    allows interleaving with ops on other pipelines.

    Key constraint: splitting increases the number of barrier
    synchronizations and may increase SMEM usage.
    """
    # Compute per-pipeline utilization within II
    for pipe in [MEM, TC, CUDA, SFU]:
        busy = sum(latencies[op] for op in schedule if unit_map[op] == pipe)
        utilization = busy / II

        if utilization < 0.7:  # Pipeline underutilized
            # Find the largest op on this pipeline that could be split
            # to fill gaps on OTHER pipelines
            for op in sorted(schedule, key=lambda o: -latencies[o]):
                if unit_map[op] != pipe:
                    continue
                if not is_splittable(op):
                    continue

                # Split factor: match the gap size on the bottleneck pipe
                bottleneck_gap = find_largest_gap(schedule, bottleneck_pipe(schedule))
                N = ceil(latencies[op] / bottleneck_gap)
                N = min(N, max_split_factor(op))

                if N <= 1:
                    continue

                # Replace op with N sub-ops in the DDG
                sub_ops = split_op_in_DDG(op, N, DDG)
                for i, sub in enumerate(sub_ops):
                    latencies[sub] = latencies[op] // N
                    unit_map[sub] = pipe
                    if i > 0:
                        DDG.add_edge(sub_ops[i-1], sub, latency=latencies[sub], distance=0)

                # Reconnect consumers to appropriate sub-ops
                reconnect_dependencies(op, sub_ops, DDG)
                break  # Re-run scheduling with the refined DDG

    # Re-run modulo scheduling with the refined DDG
    return modulo_schedule(DDG, latencies, unit_map, compute_MinII(...))
```

### Example: Splitting 128x128 into 128x64 Sub-tiles

```
Before: LoadK (640 cycles), QK_MMA (779 cycles)
After:  LoadK(a) (320), LoadK(b) (320), QK(a) (389), QK(b) (389)
```

This reduces ResMII on the TC pipeline from 1558 to 778 per sub-tile, enabling tighter interleaving and a smaller effective II.

---

## Pass A.6: Scheduling Non-Loop Regions

The modulo scheduling framework (Pass A Steps 1-2) is designed for loops, where the goal is to overlap iterations and minimize the steady-state initiation interval (II). But GPU kernels also contain **non-loop regions** — straight-line code before, after, or between loops — that benefit from cross-pipeline scheduling. Examples include:

- **Epilogue**: After the K-loop — accumulator readout from TMEM, dtype conversion, store to global memory
- **Prologue**: Before the K-loop — descriptor creation, initial tile setup
- **Inter-loop regions**: Between nested loops in persistent kernels — tile index updates, boundary checks, accumulator resets

These regions contain ops on multiple pipelines (TC, CUDA, MEM) that can execute concurrently but are emitted sequentially in the IR. Without scheduling, the compiler backend (ptxas) must discover this parallelism, which it often fails to do across barrier boundaries or complex control flow.

### The Generalization: List Scheduling on the Same Infrastructure

The modulo scheduling algorithm degenerates naturally to **list scheduling** when there are no loop-carried edges and no modulo constraint. The same DDG, latency model, pipeline resources, and priority-based placement apply — the only differences are:

| Aspect | Loop (modulo scheduling) | Non-loop (list scheduling) |
|--------|-------------------------|---------------------------|
| **Goal** | Minimize II (steady-state throughput) | Minimize makespan (total latency) |
| **Reservation table** | Wraps at II (modulo) | Linear (no wrap) |
| **Loop-carried edges** | Distance > 0 edges constrain cross-iteration | None — all edges have distance 0 |
| **Stage** | 0..max_stage (cross-iteration overlap) | Always 0 (no iterations to overlap) |
| **Cluster** | Within-stage ordering by cycle | Ordering by cycle (same mechanism, stage is always 0) |
| **Output** | Prologue/kernel/epilogue loop structure | Straight-line code in cluster order |

The scheduling algorithm is identical to Pass A Step 2, except:

```python
def list_schedule(DDG, latencies, unit_map):
    """
    Schedule a DAG of straight-line ops across multiple pipelines.
    Minimizes makespan (total execution time).

    This is Rau's algorithm with II=∞ (no modulo wrap) and no
    loop-carried edges — it degenerates to priority list scheduling.

    Returns:
        schedule: dict mapping op -> (cycle, pipeline)
        makespan: total execution time
    """
    # No reservation table size limit — we're minimizing makespan, not II
    # Use a simple per-pipeline "next free" tracker instead
    pipe_free = defaultdict(int)  # pipeline -> earliest free cycle

    # Priority: longest critical path to any sink (same as modulo scheduling)
    height = compute_heights(DDG, latencies)
    sorted_ops = sorted(DDG.nodes, key=lambda n: -height[n])

    schedule = {}

    for op in sorted_ops:
        pipe = unit_map[op]

        # Earliest start: max of (all predecessors done, pipeline free)
        earliest = pipe_free[pipe]
        for pred in predecessors(op):
            if pred in schedule:
                pred_done = schedule[pred][0] + latencies[pred]
                earliest = max(earliest, pred_done)

        schedule[op] = (earliest, pipe)
        pipe_free[pipe] = earliest + latencies[op]

    makespan = max(
        schedule[op][0] + latencies[op] for op in schedule
    )
    return schedule, makespan
```

Cluster IDs are computed exactly as in Step 2.5 — dense rank by cycle (with stage always 0):

```python
def compute_cluster_ids_linear(schedule):
    """Assign cluster IDs for straight-line code. All ops are stage 0."""
    unique_cycles = sorted(set(cycle for cycle, _ in schedule.values()))
    cycle_to_cluster = {c: i for i, c in enumerate(unique_cycles)}
    return {op: cycle_to_cluster[cycle] for op, (cycle, _) in schedule.items()}
```

### Unified Scheduling Entry Point

The scheduling framework uses a single entry point that dispatches based on the code region:

```python
def schedule_region(region, DDG, latencies, unit_map):
    """
    Schedule a code region — loop or straight-line.

    The DDG structure determines the algorithm:
    - Loop-carried edges present → modulo scheduling (minimize II)
    - No loop-carried edges → list scheduling (minimize makespan)

    Returns the same (cycle, pipeline, stage, cluster) format in both cases.
    """
    has_loop_carried = any(e.distance > 0 for e in DDG.edges)

    if has_loop_carried:
        # Loop region: modulo scheduling (Pass A Steps 1-2)
        MinII = max(compute_ResMII(DDG), compute_RecMII(DDG))
        schedule, II = modulo_schedule(DDG, latencies, unit_map, MinII)
        stages = {op: cycle // II for op, (cycle, _) in schedule.items()}
        clusters = compute_cluster_ids(schedule, II)
    else:
        # Non-loop region: list scheduling (minimize makespan)
        schedule, makespan = list_schedule(DDG, latencies, unit_map)
        stages = {op: 0 for op in schedule}     # all stage 0
        clusters = compute_cluster_ids_linear(schedule)
        II = makespan  # no steady state — "II" is the total time

    return {
        op: (cycle, pipe, stages[op], clusters[op])
        for op, (cycle, pipe) in schedule.items()
    }, II
```

### How Non-Loop Schedules Are Realized (Pass C)

For loop regions, Pass C expands the schedule into prologue/kernel/epilogue. For non-loop regions, Pass C simply **emits ops in cluster order** — no expansion needed:

```python
def emit_region(region, schedule, cluster_ids):
    if region.is_loop:
        # Existing loop expansion: prologue/kernel/epilogue
        expand_and_emit(region, schedule, cluster_ids)
    else:
        # Straight-line: emit in cluster order
        sorted_ops = sorted(
            region.ops,
            key=lambda op: cluster_ids[op]
        )
        for op in sorted_ops:
            emit(op)
```

The cluster IDs encode the schedule's optimal ordering, so emitting in cluster order produces straight-line code with cross-pipeline overlap. No loop structure is generated.

### Worked Example: GEMM Epilogue

The GEMM epilogue after the K-loop (with TMA store) consists of:

```
DDG (no loop-carried edges):

  tmem_load ──→ truncf ──→ local_store ──→ TMA_store
    (TC, 500)    (CUDA, 200)  (MEM, 300)    (MEM, 600)
```

List scheduling places these ops:

```
Cycle:   0        500       700        1000       1600
         |---------|---------|----------|----------|
TC:      [tmem_load (500)]
CUDA:              [truncf (200)]
MEM:                         [local_store (300)][TMA_store (600)]

Schedule:
  tmem_load:   cycle=0,    pipeline=TC,   cluster=0
  truncf:      cycle=500,  pipeline=CUDA, cluster=1
  local_store: cycle=700,  pipeline=MEM,  cluster=2
  TMA_store:   cycle=1000, pipeline=MEM,  cluster=3

Makespan: 1600 cycles
```

This is a simple chain — no cross-pipeline overlap is possible because each op depends on the previous. But consider a more interesting case: **two independent stores** (e.g., storing C and D tiles, or a subtiled epilogue with independent slices):

```
DDG (two independent store paths, no loop-carried edges):

  tmem_load_0 ──→ truncf_0 ──→ local_store_0 ──→ TMA_store_0
    (TC, 250)      (CUDA, 100)   (MEM, 150)       (MEM, 300)
  tmem_load_1 ──→ truncf_1 ──→ local_store_1 ──→ TMA_store_1
    (TC, 250)      (CUDA, 100)   (MEM, 150)       (MEM, 300)
```

List scheduling finds the cross-pipeline overlap:

```
Cycle:  0     250    500   600  750   900  1050  1350
        |------|------|------|------|------|------|------|
TC:     [tmem_ld_0][tmem_ld_1]
CUDA:          [truncf_0][truncf_1]
MEM:                      [l_store_0][TMA_0  ][l_store_1][TMA_1  ]

Schedule:
  tmem_load_0:   cycle=0,    cluster=0
  tmem_load_1:   cycle=250,  cluster=1
  truncf_0:      cycle=250,  cluster=1  (same cycle as tmem_load_1, different pipe)
  truncf_1:      cycle=500,  cluster=2
  local_store_0: cycle=500,  cluster=2
  TMA_store_0:   cycle=650,  cluster=3
  local_store_1: cycle=950,  cluster=4
  TMA_store_1:   cycle=1100, cluster=5

Makespan: 1400 cycles (vs. 1600 sequential)
```

The key overlap: `tmem_load_1` runs on TC while `truncf_0` runs on CUDA, and `truncf_1` runs on CUDA while `local_store_0` runs on MEM. The list scheduler discovers this automatically using the same priority-based placement as modulo scheduling.

### Kernel-Wide Scheduling

A complete kernel is a sequence of regions:

```
[prologue region] → [K-loop region] → [epilogue region]
```

Each region is scheduled independently:
- **Prologue**: list scheduling (straight-line)
- **K-loop**: modulo scheduling (loop with loop-carried edges)
- **Epilogue**: list scheduling (straight-line)

For persistent kernels with an outer tile loop:

```
outer tile loop {
    [prologue region]     ← list scheduled
    [K-loop region]       ← modulo scheduled (inner)
    [epilogue region]     ← list scheduled
}
```

The outer tile loop is modulo scheduled with the inner regions as super-nodes. Each super-node's latency is the makespan (for straight-line regions) or the steady-state latency (for loop regions) computed by its inner schedule.

Pass A computes schedules bottom-up — inner regions first, then outer regions — so that each level has the correct makespan/latency for its super-nodes. However, Pass A **does not reorder ops in the IR**. The computed schedule metadata (cycle, cluster, makespan) is sufficient for outer region scheduling. The actual reordering is deferred to Pass C, after Pass B has inserted barriers.

### Impact on the Algorithm Flow

The generalization affects all three passes:

1. **Pass A**: The scheduling algorithm dispatches to modulo or list scheduling based on whether the DDG has loop-carried edges. The output format `(cycle, pipeline, stage, cluster)` is the same. For non-loop regions, Pass A computes and stores the schedule (cluster IDs on ops as attributes) but does not reorder the IR — the schedule metadata flows to outer region scheduling via super-node latencies.

2. **Pass B**: Warp group partitioning works identically — per-pipeline utilization is computed from the schedule regardless of whether it came from modulo or list scheduling. Barriers are inserted based on the schedule's cross-group boundaries.

3. **Pass C**: Applies all reorderings. For loop regions, expands into prologue/kernel/epilogue. For non-loop regions, reorders ops in the basic block by cluster ID. This runs after Pass B, so barriers are already in place and move with their associated ops.

---

## Pass A.7: Epilogue Subtiling

Epilogue subtiling is a **DDG transformation** for non-loop epilogue regions, analogous to how Pass A.5 (data partitioning) transforms loop DDGs. It splits a monolithic TMA store into S sub-stores along the N-dimension, creating independent ops that Pass A.6's list scheduler can overlap across pipelines.

### The Transformation

Without subtiling, the epilogue is a single chain — no cross-pipeline overlap is possible:

```
tmem_load(256×256) → truncf(256×256) → local_store(256×256) → TMA_store(256×256)
     TC                  CUDA                MEM                    MEM
```

With subtiling factor S=4, this becomes 4 independent sub-chains:

```
tmem_load_0(256×64) → truncf_0 → local_store_0 → TMA_store_0
tmem_load_1(256×64) → truncf_1 → local_store_1 → TMA_store_1
tmem_load_2(256×64) → truncf_2 → local_store_2 → TMA_store_2
tmem_load_3(256×64) → truncf_3 → local_store_3 → TMA_store_3
```

The sub-chains are independent (no edges between them), so Pass A.6's list scheduler interleaves them across pipelines:

```
TC:   [tmem_ld_0][tmem_ld_1][tmem_ld_2][tmem_ld_3]
CUDA:       [truncf_0][truncf_1][truncf_2][truncf_3]
MEM:              [l_st_0][TMA_0][l_st_1][TMA_1][l_st_2][TMA_2][l_st_3][TMA_3]
```

The MEM pipeline is the bottleneck (it has 2 ops per sub-chain), but TC and CUDA ops run concurrently in the gaps, reducing total makespan.

The sub-stores **share a single SMEM buffer** of size `[BLOCK_M, BLOCK_N/S]`. This is safe because only one sub-store writes to SMEM at a time (the list schedule serializes MEM ops). The SMEM footprint drops from `BLOCK_M × BLOCK_N` to `BLOCK_M × BLOCK_N/S`.

### Trigger Conditions

Pass A.7 considers epilogue subtiling when **either** condition holds:

1. **SMEM budget pressure**: Step 4 would need to reduce K-loop buffer depth to fit the epilogue's store buffer within budget. Subtiling by factor S reduces the store buffer by S×, potentially recovering the desired depth.

2. **Epilogue latency reduction**: The list-scheduled makespan of the subtiled epilogue is shorter than the sequential epilogue. This matters especially for persistent kernels where the epilogue is a super-node in the outer tile loop — a shorter epilogue reduces the outer II.

```python
def try_epilogue_subtiling(epilogue_DDG, pipeline_config, memory_budget):
    """
    Try subtiling the epilogue's TMA store.
    Returns the best subtiling factor, or 1 (no subtiling).
    """
    store_nodes = find_tma_stores(epilogue_DDG)
    if not store_nodes:
        return 1

    sequential_makespan = list_schedule(epilogue_DDG).makespan

    best_S, best_score = 1, 0

    for store in store_nodes:
        BLOCK_M, BLOCK_N = store.shape

        for S in [2, 4]:
            if BLOCK_N % S != 0 or BLOCK_N // S < 64:
                continue

            # Build subtiled DDG and schedule it
            subtiled_DDG = split_store(epilogue_DDG, store, S)
            subtiled_makespan = list_schedule(subtiled_DDG).makespan

            # Score: latency reduction + SMEM savings
            latency_benefit = sequential_makespan - subtiled_makespan
            smem_freed = store.smem_size() * (1 - 1 / S)
            smem_recovers_depth = (
                total_smem(pipeline_config) > memory_budget
                and total_smem(pipeline_config) - smem_freed <= memory_budget
            )

            score = latency_benefit
            if smem_recovers_depth:
                score += SMEM_DEPTH_BONUS

            if score > best_score:
                best_score = score
                best_S = S

    return best_S
```

### Algorithm

```python
def split_store(epilogue_DDG, store_node, S):
    """
    Replace a monolithic store path with S independent sub-store paths.

    Each sub-store path:
      tmem_load(BLOCK_M, BLOCK_N/S) → truncf → local_store → TMA_store

    The sub-store paths are independent (no edges between them).
    They share a single SMEM buffer — the list scheduler serializes
    MEM ops naturally, so no explicit ordering is needed.
    """
    BLOCK_M, BLOCK_N = store_node.shape
    sub_N = BLOCK_N // S

    # Find the full epilogue chain: tmem_load → truncf → local_store → TMA_store
    chain = find_producer_chain(store_node)  # [tmem_load, truncf, local_store, TMA_store]

    new_DDG = epilogue_DDG.clone()
    new_DDG.remove_chain(chain)

    for i in range(S):
        sub_chain = []
        for op in chain:
            sub_op = new_DDG.add_node(
                name=f"{op.name}_{i}",
                pipeline=op.pipeline,
                latency=op.latency // S,
                shape=(BLOCK_M, sub_N),
                n_offset=i * sub_N,
            )
            sub_chain.append(sub_op)

        # Intra-chain edges (within each sub-store path)
        for j in range(1, len(sub_chain)):
            new_DDG.add_edge(sub_chain[j-1], sub_chain[j],
                             latency=sub_chain[j-1].latency)

    # No inter-chain edges — sub-stores are independent
    # The list scheduler will serialize MEM ops on the MEM pipeline

    return new_DDG
```

### Integration with the Algorithm Flow

```
Pass A Steps 1-2: Schedule K-loop (modulo)
Pass A Step 3-4:  Pipeline depths, SMEM budget check
Pass A.5:         Data partitioning (optional, loop DDG)
Pass A.6:         List schedule epilogue (initial, monolithic)
Pass A.7:         Try subtiling → if beneficial:
                    Transform epilogue DDG (split store)
                    Re-run A.6 list schedule on transformed DDG
                    Update SMEM budget (store buffer shrinks)
Pass B:           Warp specialization, barriers
Pass C:           Reorder epilogue ops by cluster, expand loops
```

Pass A.7 runs after A.6's initial schedule so it can compare the sequential makespan against the subtiled makespan. If subtiling helps, it transforms the DDG and re-runs A.6. The resulting cluster IDs encode the interleaved order that Pass C will apply.

### Worked Example (256×256 GEMM, TMA Store, S=4)

```
Sequential epilogue (no subtiling):
  tmem_load(256×256): 500 cy (TC)
  truncf(256×256):    200 cy (CUDA)
  local_store:        300 cy (MEM)
  TMA_store:          600 cy (MEM)
  Makespan: 1600 cy
  SMEM: 256×256×2 = 128KB

Subtiled epilogue (S=4, list scheduled):
  Per sub-store: tmem_load 125 cy, truncf 50 cy, l_store 75 cy, TMA_store 150 cy

  TC:   [ld_0 125][ld_1 125][ld_2 125][ld_3 125]
  CUDA:      [tr_0 50][tr_1 50][tr_2 50][tr_3 50]
  MEM:            [ls_0 75][tma_0 150][ls_1 75][tma_1 150][ls_2 75][tma_2 150][ls_3 75][tma_3 150]

  Makespan: 125 + max(TC trail, MEM total)
    MEM total: 4 × (75 + 150) = 900 cy, starting at cycle 175
    MEM finish: 175 + 900 = 1075 cy
  Makespan: ~1075 cy (vs 1600 sequential, 33% reduction)
  SMEM: 256×64×2 = 32KB (75% reduction)

SMEM budget impact (K-loop depth=3):
  K-loop buffers: 192KB
  Without subtiling: 192 + 128 = 320KB > 232KB budget → forced to depth=1
  With S=4: 192 + 32 = 224KB ✓ → depth=3 maintained
```

---

## Pass B: Warp Specialization Reconstruction

Given the modulo schedule and pipeline configuration, reconstruct the warp-specialized program.

### Step 1: Partition Ops into Warp Groups

```python
def partition_into_warp_groups(schedule, unit_map, latencies, II):
    """
    Assign ops to warp groups based on pipeline affinity and workload balance.

    Heuristic:
    - Each pipeline with > 30% utilization gets its own warp group
    - Underutilized pipelines are merged with compatible groups
    """
    # Compute per-pipeline utilization
    pipe_util = {}
    for pipe in [MEM, TC, CUDA, SFU]:
        busy = sum(latencies[op] for op in schedule if unit_map[op] == pipe)
        pipe_util[pipe] = busy / II

    # Create warp groups for well-utilized pipelines
    groups = []
    for pipe in [MEM, TC, CUDA, SFU]:
        if pipe_util[pipe] > 0.3:
            groups.append(WarpGroup(
                pipeline=pipe,
                ops=[op for op in schedule if unit_map[op] == pipe],
            ))

    # Merge underutilized pipelines into nearest group
    for pipe in [MEM, TC, CUDA, SFU]:
        if pipe_util[pipe] <= 0.3:
            # Merge into the group whose ops have the most
            # data dependencies with this pipeline's ops
            best_group = argmax(groups, key=lambda g:
                count_cross_edges(pipe_ops, g.ops, DDG)
            )
            best_group.ops.extend(
                op for op in schedule if unit_map[op] == pipe
            )

    return groups
```

### Step 1.5: Replicate Shared Infrastructure Ops

Pass A's modulo schedule and Step 1's partitioning only cover **pipeline ops** — the operations that execute on MEM, TC, CUDA, or SFU. But a real kernel also contains **infrastructure ops** that don't belong to any pipeline: loop control flow, buffer index arithmetic, constants, scalar computations, and conditional logic. These ops must be present in every warp group that needs them.

#### Categories of Shared Ops

| Category | Examples | Why shared |
|----------|---------|-----------|
| **Loop control** | `for i in range(N)`, induction variable, bounds check | Each warp group runs its own loop with potentially different trip counts (prologue/epilogue differences) |
| **Buffer indexing** | `buf_idx = i % depth`, `phase = (i // depth) & 1` | Every warp group that touches multi-buffered resources must compute the same buffer index |
| **Constants** | `sm_scale`, `BLOCK_M`, `log2e` | Used by ops across multiple warp groups |
| **Scalar state** | Tile offsets, descriptor pointers, `accum_cnt` | Bookkeeping that must be consistent across groups |
| **Conditional logic** | Causal mask checks, boundary guards | May gate ops in multiple warp groups |

These ops have no pipeline assignment (`unit_map` doesn't cover them) and zero pipeline latency — they execute on the warp's general-purpose issue slot and are not modeled in the modulo schedule.

#### Replication Strategy

The algorithm handles shared ops by **replication**: each warp group gets its own copy of every infrastructure op it needs. This is correct because these ops are pure (no side effects, no shared mutable state) and cheap (scalar arithmetic, a few cycles each).

```python
def replicate_shared_ops(groups, DDG, all_ops):
    """
    For each warp group, identify infrastructure ops needed by its
    pipeline ops and clone them into the group.

    An op is "needed" by a group if:
    1. It is in the transitive def chain of any pipeline op in the group
    2. It is not itself a pipeline op (not in any unit_map entry)

    Infrastructure ops are replicated, not shared, because:
    - Each warp group is an independent thread of execution
    - Sharing would require synchronization (defeating the purpose)
    - The ops are cheap scalar arithmetic (no performance cost)
    """
    pipeline_ops = set()
    for g in groups:
        pipeline_ops.update(g.ops)

    for g in groups:
        needed_infra = set()
        worklist = list(g.ops)
        visited = set()

        while worklist:
            op = worklist.pop()
            if op in visited:
                continue
            visited.add(op)

            for pred in predecessors(op, DDG):
                if pred not in pipeline_ops:
                    # This is an infrastructure op — replicate it
                    needed_infra.add(pred)
                    worklist.append(pred)

        g.infra_ops = needed_infra
```

#### What Gets Replicated vs. What Gets Specialized

Not all infrastructure is identical across groups. Some ops are **specialized per group**:

| Replicated identically | Specialized per group |
|----------------------|---------------------|
| `sm_scale`, constants | `accum_cnt` (each group may increment at different rates) |
| `buf_idx = cnt % depth` (same formula) | Trip count (producer runs `N` iters, consumer runs `N - prologue`) |
| Descriptor base pointers | Loop bounds (offset by prologue depth) |

The specialized ops are **derived** from the pipeline configuration (buffer depths, prologue/epilogue structure) rather than copied from the original program. For example, the producer group's loop runs `for k in range(k_tiles)` while the consumer group's loop runs `for k in range(k_tiles - prologue_depth)` with an offset start.

#### Impact on Code Size

Replication increases per-group code size but not execution cost. In practice, the replicated infrastructure ops are a small fraction of each group's total work — typically 10-20 scalar instructions per iteration vs. hundreds of cycles on the pipeline ops. The I-cache cost is negligible because each warp group's instruction stream fits comfortably within the SM's instruction cache.

#### Relation to the Implementation

In the compiler implementation (`WSCodePartition.cpp`), shared op replication is handled during code partitioning: the pass clones ops into each async task region that uses them. The `propagatePartitions` pass in `PartitionSchedulingMeta.cpp` handles the assignment side — unassigned ops (those not on any pipeline) are clustered based on their def-use relationships and assigned to the partition(s) that need them, with cloning when multiple partitions require the same op.

### Step 2: Insert Synchronization

```python
def insert_synchronization(groups, DDG, pipeline_config):
    """
    For each cross-group dependency, insert the appropriate barrier type.

    Barrier type selection:
    - SMEM transfer (TMA load → MMA read): mbarrier with expect_bytes
    - TMEM transfer (MMA write → CUDA read): named barrier
    - Control dependency (iteration gating): mbarrier phase
    """
    barriers = []

    for (u, v) in cross_group_edges(groups, DDG):
        depth = pipeline_config.boundary_depths.get(
            (group_of(u), group_of(v)), 1
        )

        if communicates_via_smem(u, v):
            # Allocate 'depth' mbarriers for this boundary
            # They cycle through phases: phase = iter % depth
            bar_array = AllocBarriers(
                num=depth,
                arrive_count=1,
                expect_bytes=resource_size(u, v),
            )
            barriers.append(CrossGroupBarrier(
                producer_op=u,
                consumer_op=v,
                barrier=bar_array,
                depth=depth,
                type="mbarrier",
            ))

        elif communicates_via_tmem(u, v):
            # Named barriers for TMEM (no phase cycling needed,
            # TMEM ops are warp-group scoped)
            bar_id = allocate_named_barrier_id()
            barriers.append(CrossGroupBarrier(
                producer_op=u,
                consumer_op=v,
                barrier=bar_id,
                depth=1,
                type="named",
            ))

    return barriers
```

### Step 3: Compute Per-Region Loop Structure

Each warp group runs its own loop, but the loops are coupled by barriers. The modulo schedule determines the relative timing:

```python
def compute_region_loop_structure(groups, pipeline_config, schedule, II):
    """
    For each warp group, determine:
    - How many iterations to run ahead in the prologue
    - The steady-state loop body (what ops execute per iteration)
    - The epilogue drain

    The producer group's prologue fills the pipeline:
        prologue_iters = max_buffer_depth - 1

    The consumer group's loop starts after the prologue,
    and runs an extra epilogue_iters iterations to drain.
    """
    # Find the producer group (typically MEM)
    producer_group = find_group_by_pipeline(groups, MEM)

    # Find consumer groups (typically TC, CUDA+SFU)
    consumer_groups = [g for g in groups if g != producer_group]

    max_depth = max(pipeline_config.buffer_depths.values())

    # Producer prologue: fill pipeline
    producer_group.prologue_iters = max_depth - 1
    producer_group.steady_state_body = producer_group.ops  # per iteration
    producer_group.epilogue_iters = 0  # producer stops first

    # Consumer groups: offset start, drain at end
    for cg in consumer_groups:
        # Consumer starts after producer has filled enough buffers
        # The offset depends on which resources this consumer reads
        relevant_depths = [
            pipeline_config.boundary_depths[(producer_group, cg, res)]
            for res in resources_between(producer_group, cg)
        ]
        cg.start_offset = max(relevant_depths) - 1  # iterations behind producer
        cg.prologue_iters = 0
        cg.steady_state_body = cg.ops
        cg.epilogue_iters = cg.start_offset  # drain remaining buffers

    return groups
```

### Step 4: Assign Warp Counts and Registers

```python
def assign_warp_resources(groups, latencies, II):
    """
    Determine num_warps and num_regs for each group.

    num_warps is driven by:
    1. Issue throughput: does the group have enough warps to
       issue all its ops within II cycles?
    2. Occupancy: more warps can hide intra-warp latency

    num_regs is driven by:
    1. Live variables within the group's ops
    2. Spill avoidance: keep below hardware limit per warp
    """
    for g in groups:
        # Total issue slots needed per iteration
        total_work = sum(latencies[op] for op in g.ops)

        # Each warp can issue one instruction per cycle
        # But ops occupy their pipeline for 'latency' cycles
        # The group needs enough warps to keep the pipeline fed
        g.num_warps = max(1, ceil(total_work / II))

        # Register estimation
        live_vars = compute_max_live_variables(g.ops)
        g.num_regs = min(
            ceil(live_vars * bytes_per_var / (g.num_warps * 32)),
            MAX_REGS_PER_THREAD
        )

    # Validate total warps don't exceed hardware limit
    total_warps = sum(g.num_warps for g in groups)
    assert total_warps <= MAX_WARPS_PER_CTA, (
        f"Total warps {total_warps} exceeds limit {MAX_WARPS_PER_CTA}"
    )

    return groups
```

### Step 5: Generate TLX Code Skeleton

```python
def generate_tlx_code(groups, pipeline_config, barriers):
    """
    Emit the TLX warp-specialized kernel structure.
    """

    # Buffer allocations
    for resource, depth in pipeline_config.buffer_depths.items():
        emit(f"{resource.name} = tlx.local_alloc("
             f"{resource.shape}, {resource.dtype}, {depth}"
             f"{', tlx.storage_kind.tmem' if resource.storage == TMEM else ''})")

    # Barrier allocations
    for bar in barriers:
        if bar.type == "mbarrier":
            emit(f"bar_{bar.name} = tlx.alloc_barriers({bar.depth}, "
                 f"arrive_count={bar.arrive_count})")

    # Warp-specialized regions
    emit("with tlx.async_tasks():")

    for g in groups:
        if g == default_group:
            emit(f"    with tlx.async_task('default'):")
        else:
            emit(f"    with tlx.async_task(num_warps={g.num_warps}, "
                 f"num_regs={g.num_regs}):")

        # Prologue
        if g.prologue_iters > 0:
            emit(f"        # Prologue: {g.prologue_iters} iterations")
            emit(f"        for _p in range({g.prologue_iters}):")
            for op in g.steady_state_body:
                emit(f"            {op.code}")
                emit_barriers(op, barriers, "prologue")

        # Steady-state loop
        emit(f"        # Steady state (II = {pipeline_config.II} cycles)")
        emit(f"        for i in range(N - {g.prologue_iters + g.epilogue_iters}):")
        emit(f"            buf_idx = i % {max(pipeline_config.buffer_depths.values())}")
        for op in g.steady_state_body:
            emit(f"            {op.code}")
            emit_barriers(op, barriers, "steady")

        # Epilogue
        if g.epilogue_iters > 0:
            emit(f"        # Epilogue: {g.epilogue_iters} iterations")
            emit(f"        for _e in range({g.epilogue_iters}):")
            for op in g.steady_state_body:
                emit(f"            {op.code}")
                emit_barriers(op, barriers, "epilogue")
```

---

## Pass C: Code Generation and Instruction Ordering

Pass C takes the `(stage, cluster)` assignments from Pass A and the warp-specialized code skeleton from Pass B (including barriers), and generates the final code with instructions in the order determined by the schedule.

**Pass C makes no scheduling decisions.** All ordering decisions were made by Pass A. Pass C applies them:

- **Loop regions**: Expand into prologue/kernel/epilogue using `(stage, cluster)` ordering
- **Non-loop regions**: Reorder ops in the basic block by cluster ID

Pass C runs after Pass B, so barriers are already inserted and move with their associated ops during reordering.

### Loop Regions

```python
def expand_loop_region(groups, schedule, cluster_ids, barriers, II):
    """
    Generate the prologue/kernel/epilogue loop structure.
    Ordering comes entirely from Pass A's modulo schedule via cluster IDs.
    """
    max_stage = max(schedule[op].stage for op in all_ops(groups))

    for g in groups:
        sorted_ops = sorted(
            g.ops,
            key=lambda op: (schedule[op].stage, cluster_ids[op])
        )

        # Prologue: ramp up the pipeline
        for s in range(max_stage):
            for op in sorted_ops:
                if schedule[op].stage <= s:
                    emit_with_barriers(op, barriers)

        # Kernel body: all stages active
        emit(f"for i in range(N - {max_stage}):")
        for op in sorted_ops:
            emit_with_barriers(op, barriers)

        # Epilogue: drain the pipeline
        for s in range(max_stage, 0, -1):
            for op in sorted_ops:
                if schedule[op].stage >= s:
                    emit_with_barriers(op, barriers)
```

### Non-Loop Regions

```python
def reorder_nonloop_region(region, cluster_ids):
    """
    Reorder ops in a basic block by cluster ID.
    All ops are stage 0 — just sort by cluster.
    Barriers inserted by Pass B move with their associated ops.
    """
    sorted_ops = sorted(
        region.ops,
        key=lambda op: cluster_ids[op]
    )
    reorder_ops_in_block(region.block, sorted_ops)
```

In the compiler implementation, the loop path corresponds to `PipelineExpander` reading `loop.stage` and `loop.cluster` attributes. The non-loop path reorders ops within a basic block by their `loop.cluster` attribute (all at `loop.stage = 0`).

### Relationship Between Pass A and Pass C

```
Pass A: schedule[op] = (cycle, pipeline, stage, cluster)
    → all scheduling decisions, annotates ops with attributes
    → computes makespan/latency for super-nodes (bottom-up)
Pass B: warp_groups[op] = group_id, barriers between groups
    → partitions ops, inserts synchronization
Pass C: apply reordering from Pass A's attributes
    → loop regions: expand into prologue/kernel/epilogue
    → non-loop regions: reorder ops in basic block by cluster
```

Pass A computes the optimal ordering via modulo scheduling. Pass C applies it. There is no heuristic refinement step — the cluster IDs from Pass A Step 2.5 are the final ordering.

---

## Worked Example: Blackwell GEMM Kernel

This section walks through the entire algorithm using a **Blackwell GEMM kernel** as the concrete input, showing what decisions each pass makes and what TLX code it produces. We use the config: `BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, NUM_SMEM_BUFFERS=3, NUM_TMEM_BUFFERS=1, EPILOGUE_SUBTILE=4`.

### GEMM Dependency Graph

GEMM's iteration body processes one K-tile per iteration:

```
LoadA[i] ──→ MMA[i]
LoadB[i] ──→ MMA[i]

Loop-carried edges (distance=1):
  Acc[i] → MMA[i+1]   (use_acc=True from iteration 1 onward)
```

**Functional unit mapping:**

| Pipeline | Operations |
|----------|-----------|
| **MEM** | LoadA, LoadB (TMA loads) |
| **TC** | MMA (tcgen05.mma) |
| **CUDA** | (none in main loop — epilogue only) |
| **SFU** | (none) |

GEMM only uses two pipelines in the inner loop (MEM and TC), unlike Flash Attention which uses all four.

### Pass A, Step 1: Compute MinII

```
LoadA (TMA 128×64 bf16):          ~320 cycles
LoadB (TMA 64×256 bf16):          ~640 cycles
MMA   (tcgen05.mma 128×256×64):   ~559 cycles
```

**ResMII** (resource-constrained):
```
MEM: LoadA(320) + LoadB(640) = 960
TC:  MMA(559)                = 559

ResMII = max(960, 559) = 960  (MEM-bound)
```

**RecMII** (recurrence-constrained):
The accumulator recurrence `Acc[i] → MMA[i+1]` has distance=1. The critical path is the MMA latency itself (559 cycles).
```
RecMII = 559
```

**MinII:**
```
MinII = max(ResMII, RecMII) = max(960, 559) = 960
```

The GEMM kernel is **memory-bound** — the TMA loads are the bottleneck.

### Pass A, Step 2: Modulo Schedule

Rau's algorithm places ops into a reservation table of length II=960:

```python
schedule = {
    "LoadA":  (0,   MEM),
    "LoadB":  (320, MEM),
    "MMA":    (320, TC),     # starts when LoadA finishes
}
II = 960
```

```
Cycle:   0         320              879   960 (=II)
         ├─────────┼────────────────┼─────┤
MEM:     [LoadA    ][  LoadB              ]
TC:                [  MMA            ]
```

MMA starts at cycle 320 (when LoadA's data is available) and finishes at cycle 879. LoadB finishes at cycle 960. Both fit within II — no cross-iteration wrap needed.

### Pass A, Step 3: Derive Pipeline Depths

**A tile (SMEM):**
```
Producer: LoadA at cycle 0, latency 320
Consumer: MMA finishes at cycle 879
Lifetime = 879 - 0 = 879
num_buffers = floor(879 / 960) + 1 = 0 + 1 = 1
```

A single buffer suffices for one iteration's data, but to keep the MEM pipeline busy (producer running ahead of MMA consumer), we need depth > 1. `NUM_SMEM_BUFFERS=3` allows the producer to run 2 iterations ahead:

```
Prologue depth = NUM_SMEM_BUFFERS - 1 = 2 iterations of prefetch
```

**B tile (SMEM):** Same analysis — `NUM_SMEM_BUFFERS=3`.

**Accumulator (TMEM):**
```
Producer: MMA writes over all K-iterations
Consumer: Epilogue reads after final K-iteration
NUM_TMEM_BUFFERS=1: single-buffered
  → Epilogue must finish before next tile's MMA can start
```

### Pass A, Step 4: Memory Budget Check (Initial)

```
SMEM:
  A buffers: 128 × 64 × 2B × 3 buffers  =  49,152 B
  B buffers:  64 × 256 × 2B × 3 buffers  =  98,304 B
  C epilogue: 128 × 256 × 2B × 2 buffers = 131,072 B  ← monolithic store
  Barriers:                               ~     96 B
  Total SMEM ≈ 278,624 B  (>> 228 KB limit ✗)

TMEM:
  Acc: 128 × 256 × 4B × 1 buffer = 131,072 B = 128 KB  (< 256 KB ✓)
```

The monolithic epilogue store buffer blows the SMEM budget. The store path (`tmem_load → truncf → local_store → TMA_store`) requires a `128×256 × 2B = 64 KB` SMEM buffer, and double-buffering doubles that to 128 KB.

### Pass A.7 Applied: Epilogue Subtiling (EPILOGUE_SUBTILE=4)

**Trigger:** Step 4 failed the SMEM budget check. The epilogue store buffer (128 KB) is the dominant cost.

**Transformation:** Split the epilogue chain into 4 independent sub-chains along the N-dimension:

```
Before:
  tmem_load(128×256) → truncf(128×256) → local_store(128×256) → TMA_store(128×256)
       TC                 CUDA                MEM                    MEM

After (S=4):
  tmem_load_0(128×64) → truncf_0 → local_store_0 → TMA_store_0
  tmem_load_1(128×64) → truncf_1 → local_store_1 → TMA_store_1
  tmem_load_2(128×64) → truncf_2 → local_store_2 → TMA_store_2
  tmem_load_3(128×64) → truncf_3 → local_store_3 → TMA_store_3
```

**Benefits:**
- **SMEM reduction**: store buffer shrinks from `128×256` to `128×64` (4×), from 64 KB to 16 KB
- **Cross-pipeline overlap**: Pass A.6's list scheduler interleaves sub-chains across TC/CUDA/MEM

Epilogue DDG changed → re-run from top. Steps 1-3 are unaffected (A.7 only transforms the epilogue DDG). Re-check Step 4:

### Pass A, Step 4: Memory Budget Check (After A.7)

```
SMEM (after A.7 subtiling):
  A buffers: 128 × 64 × 2B × 3 buffers  =  49,152 B
  B buffers:  64 × 256 × 2B × 3 buffers  =  98,304 B
  C epilogue: 128 × 64 × 2B × 2 buffers  =  32,768 B  (subtiled: 256/4=64)
  Barriers:                               ~     96 B
  Total SMEM ≈ 180,320 B  (< 228 KB limit ✓)

TMEM:
  Acc: 128 × 256 × 4B × 1 buffer = 131,072 B = 128 KB  (< 256 KB ✓)
```

No further DDG transforms needed → **converged**.

### Pass A, Step 5: Emit ScheduleGraph

The converged schedule is packaged into a ScheduleGraph. The GEMM kernel is a persistent kernel with three regions: an outer tile loop, an inner K-loop (modulo scheduled), and an epilogue (list scheduled on the subtiled DDG from A.7).

**Inner K-loop** (modulo scheduled):

```
modulo.pipeline @kloop {
  ii = 960, max_stage = 0

  %buf0 = modulo.alloc SMEM [3 x 128x64 x f16]   live=[0, 879)    // A tile
  %buf1 = modulo.alloc SMEM [3 x 64x256 x f16]   live=[320, 879)  // B tile
  %bar0 = modulo.alloc BARRIER [3] for buf0
  %bar1 = modulo.alloc BARRIER [3] for buf1
  %tmem0 = modulo.alloc TMEM [1 x 128x256 x f32]  live=[320, 879)  // Acc

  modulo.stage @s0 {
    %N0 = tt.descriptor_load  {pipe: MEM, cycle: 0, cluster: 0, latency: 320, selfLatency: 320, ->buf0}
    %N1 = tt.descriptor_load  {pipe: MEM, cycle: 320, cluster: 1, latency: 640, selfLatency: 640, ->buf1}
    %N2 = ttng.tc_gen5_mma    {pipe: TC, cycle: 320, cluster: 1, latency: 559, selfLatency: 559, <-buf0, <-buf1, ->tmem0}
  }

  edges {
    N0 -> N2  lat=320  dist=0    // LoadA → MMA
    N1 -> N2  lat=640  dist=0    // LoadB → MMA
    N2 -> N2  lat=559  dist=1    // Acc recurrence
  }
}
```

All ops are at stage 0 (`max_stage = 0`): the lifetime of each buffer is less than II=960. The `count=3` comes from the heuristic `NUM_SMEM_BUFFERS` parameter, which enables the producer to run 2 iterations ahead of the consumer.

**Epilogue region** (list scheduled, after subtiling with S=4):

Pass A.7 splits the monolithic epilogue store (128×256) into 4 independent sub-chains of (128×64) each. Pass A.6 list-schedules the subtiled DDG, interleaving sub-chains across pipelines. The cluster IDs encode the emission order — Pass C reorders ops by cluster to achieve cross-pipeline overlap:

```
modulo.pipeline @epilogue {
  ii = 0, max_stage = 0    // non-loop region: ii=0, makespan used instead
  makespan = 1075

  %c_smem = modulo.alloc SMEM [2 x 128x64 x f16]  live=[0, 1075)  // shared across sub-chains

  modulo.stage @s0 {
    // Ops listed in cluster order (the emission order Pass C uses).
    // Within the same cluster, ops are on different pipelines and execute concurrently.
    %E0  = ttng.tmem_load      {pipe: TC,   cycle: 0,   cluster: 0, latency: 125, selfLatency: 125, <-tmem0}
    %E4  = ttng.tmem_load      {pipe: TC,   cycle: 125, cluster: 1, latency: 125, selfLatency: 125, <-tmem0}
    %E1  = arith.truncf        {pipe: CUDA, cycle: 125, cluster: 1, latency: 50,  selfLatency: 50}
    %E2  = ttg.local_store     {pipe: MEM,  cycle: 175, cluster: 2, latency: 75,  selfLatency: 75,  ->c_smem}
    %E8  = ttng.tmem_load      {pipe: TC,   cycle: 250, cluster: 3, latency: 125, selfLatency: 125, <-tmem0}
    %E5  = arith.truncf        {pipe: CUDA, cycle: 250, cluster: 3, latency: 50,  selfLatency: 50}
    %E3  = tt.descriptor_store {pipe: MEM,  cycle: 250, cluster: 3, latency: 150, selfLatency: 150, <-c_smem}
    %E12 = ttng.tmem_load      {pipe: TC,   cycle: 375, cluster: 4, latency: 125, selfLatency: 125, <-tmem0}
    %E9  = arith.truncf        {pipe: CUDA, cycle: 375, cluster: 4, latency: 50,  selfLatency: 50}
    %E6  = ttg.local_store     {pipe: MEM,  cycle: 400, cluster: 5, latency: 75,  selfLatency: 75,  ->c_smem}
    %E13 = arith.truncf        {pipe: CUDA, cycle: 500, cluster: 6, latency: 50,  selfLatency: 50}
    %E7  = tt.descriptor_store {pipe: MEM,  cycle: 475, cluster: 6, latency: 150, selfLatency: 150, <-c_smem}
    %E10 = ttg.local_store     {pipe: MEM,  cycle: 625, cluster: 7, latency: 75,  selfLatency: 75,  ->c_smem}
    %E11 = tt.descriptor_store {pipe: MEM,  cycle: 700, cluster: 8, latency: 150, selfLatency: 150, <-c_smem}
    %E14 = ttg.local_store     {pipe: MEM,  cycle: 850, cluster: 9, latency: 75,  selfLatency: 75,  ->c_smem}
    %E15 = tt.descriptor_store {pipe: MEM,  cycle: 925, cluster: 10, latency: 150, selfLatency: 150, <-c_smem}
  }

  edges {
    // Intra-chain dependencies (4 independent chains)
    E0 -> E1  lat=125  dist=0     E4 -> E5  lat=125  dist=0
    E1 -> E2  lat=50   dist=0     E5 -> E6  lat=50   dist=0
    E2 -> E3  lat=75   dist=0     E6 -> E7  lat=75   dist=0
    E8 -> E9  lat=125  dist=0     E12 -> E13  lat=125  dist=0
    E9 -> E10 lat=50   dist=0     E13 -> E14  lat=50   dist=0
    E10 -> E11 lat=75  dist=0     E14 -> E15  lat=75   dist=0
    // No inter-chain edges — sub-chains are independent
  }
}
```

The cluster ordering interleaves sub-chains across pipelines. At cluster 1, `tmem_load_1` (TC) runs concurrently with `truncf_0` (CUDA). At cluster 3, `tmem_load_2` (TC), `truncf_1` (CUDA), and `TMA_store_0` (MEM) all run concurrently on different pipelines. Pass C emits ops in this cluster order — the hardware then overlaps ops on independent pipelines.

**Outer tile loop** (modulo scheduled, persistent kernel):

The outer loop sees the K-loop and epilogue as super-nodes:

```
modulo.pipeline @outer {
  ii = <tile_latency>, max_stage = 0

  modulo.stage @s0 {
    %T0 = scf.for [K-loop]  {pipe: TC, cycle: 0, latency: <k_tiles * II>, selfLatency: <k_tiles * II>}
    %T1 = epilogue           {pipe: MEM, cycle: <k_tiles * II>, latency: 1075, selfLatency: 1075}
  }

  edges {
    T0 -> T1  lat=<k_tiles * II>  dist=0    // epilogue after K-loop
    T1 -> T0  lat=1075             dist=1    // next tile after epilogue
  }
}
```

With `NUM_TMEM_BUFFERS=1`, the epilogue must complete before the next tile's MMA can start, so MMA/epilogue overlap is not possible. The outer loop is effectively sequential: each tile processes K-loop → epilogue → next tile.

### Pass B, Step 1: Partition into Warp Groups

Pipeline utilization within II=960:
```
MEM:  960/960 = 100%  → dedicated warp group  ✓  (> 30%)
TC:   559/960 =  58%  → dedicated warp group  ✓  (> 30%)
CUDA:   0/960 =   0%  → no inner-loop ops
SFU:    0/960 =   0%  → no ops
```

The epilogue (TMEM→registers→SMEM→TMA store) uses CUDA cores and MEM but runs *between* tiles, not in the inner K-loop. It gets its own warp group.

**Result: 3 warp groups:**

| Warp Group | Role | Pipeline | Warps | Regs |
|-----------|------|----------|-------|------|
| Producer | TMA loads of A and B | MEM | 1 | 24 |
| MMA | tcgen05.mma operations | TC | 1 | 24 |
| Epilogue | TMEM read + convert + TMA store | CUDA+MEM | default | — |

### Pass B, Step 2: Insert Synchronization

| Boundary | Resource | Direction | Barrier Type | Depth |
|----------|----------|-----------|-------------|-------|
| Producer → MMA | A tile in SMEM | data ready | `mbarrier` + `expect_bytes` | 3 |
| Producer → MMA | B tile in SMEM | data ready | `mbarrier` + `expect_bytes` | 3 |
| MMA → Producer | A tile consumed | buffer free | `mbarrier` (empty signal) | 3 |
| MMA → Epilogue | Accumulator in TMEM | data ready | `mbarrier` | 1 |
| Epilogue → MMA | TMEM buffer freed | buffer free | `mbarrier` | 1 |

Barriers cycle through phases using `(accum_cnt // NUM_BUFFERS) & 1`.

### Pass B, Step 5: Generated TLX Code

#### Buffer Allocations

```python
# A tile: (128, 64) × bf16 × 3 buffers
buffers_A = tlx.local_alloc(
    (BLOCK_M, BLOCK_K),            # (128, 64)
    tlx.dtype_of(a_desc),          # bf16
    NUM_SMEM_BUFFERS,              # 3
)

# B tile: (64, 256) × bf16 × 3 buffers
buffers_B = tlx.local_alloc(
    (BLOCK_K, BLOCK_N),            # (64, 256)
    tlx.dtype_of(b_desc),
    NUM_SMEM_BUFFERS,              # 3
)

# Accumulator in TMEM: (128, 256) × f32 × 1 buffer
tmem_buf = tlx.local_alloc(
    (BLOCK_M, BLOCK_N),            # (128, 256)
    tl.float32,
    NUM_TMEM_BUFFERS,              # 1
    tlx.storage_kind.tmem,
)

# Epilogue SMEM: (128, 64) × bf16 × 2 buffers (subtiled store)
c_smem = tlx.local_alloc(
    (BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE),  # (128, 64)
    tlx.dtype_of(c_desc),
    2,                                        # double-buffered
)
```

#### Barrier Allocations

```python
# Producer→MMA: "A tile loaded" / "A tile consumed"
A_full_bars  = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)   # 3
A_empty_bars = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)   # 3

# Producer→MMA: "B tile loaded"
B_full_bars  = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)   # 3

# MMA→Epilogue: "accumulator ready" / "TMEM buffer free"
tmem_full_bar  = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)           # 1
tmem_empty_bar = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=EPILOGUE_SUBTILE)  # 1
```

#### Warp-Specialized Kernel Structure

```python
with tlx.async_tasks():

    # ── Warp Group 1: Epilogue (TMEM → global) ──────────────────
    with tlx.async_task("default"):
        while tile_id < num_tiles:
            tlx.barrier_wait(tmem_full_bar[0], phase)             # wait for MMA

            # Subtiled epilogue: 4 slices of (128, 64), flattened in cluster order.
            # Pass C reorders ops by cluster to interleave sub-chains across pipelines.
            slice_n = BLOCK_N // EPILOGUE_SUBTILE                  # 64

            # cluster 0: tmem_load slice 0 (TC)
            r0 = tlx.local_load(tmem_buf[0], n_offset=0, n_size=slice_n)
            # cluster 1: tmem_load slice 1 (TC) + truncf slice 0 (CUDA)
            r1 = tlx.local_load(tmem_buf[0], n_offset=slice_n, n_size=slice_n)
            c0 = r0.to(output_dtype)
            # cluster 2: local_store slice 0 (MEM)
            tlx.local_store(c_smem, c0)
            # cluster 3: tmem_load slice 2 (TC) + truncf slice 1 (CUDA) + TMA_store slice 0 (MEM)
            r2 = tlx.local_load(tmem_buf[0], n_offset=2*slice_n, n_size=slice_n)
            c1 = r1.to(output_dtype)
            tlx.fence_async_shared()
            tlx.async_descriptor_store(c_desc, c_smem, [m, n])
            tlx.barrier_arrive(tmem_empty_bar[0], 1)               # 1 of 4 arrivals
            # cluster 4: tmem_load slice 3 (TC) + truncf slice 2 (CUDA)
            r3 = tlx.local_load(tmem_buf[0], n_offset=3*slice_n, n_size=slice_n)
            c2 = r2.to(output_dtype)
            # cluster 5: local_store slice 1 (MEM)
            tlx.local_store(c_smem, c1)
            # cluster 6: truncf slice 3 (CUDA) + TMA_store slice 1 (MEM)
            c3 = r3.to(output_dtype)
            tlx.fence_async_shared()
            tlx.async_descriptor_store(c_desc, c_smem, [m, n + slice_n])
            tlx.barrier_arrive(tmem_empty_bar[0], 1)               # 2 of 4 arrivals
            # cluster 7: local_store slice 2 (MEM)
            tlx.local_store(c_smem, c2)
            # cluster 8: TMA_store slice 2 (MEM)
            tlx.fence_async_shared()
            tlx.async_descriptor_store(c_desc, c_smem, [m, n + 2*slice_n])
            tlx.barrier_arrive(tmem_empty_bar[0], 1)               # 3 of 4 arrivals
            # cluster 9: local_store slice 3 (MEM)
            tlx.local_store(c_smem, c3)
            # cluster 10: TMA_store slice 3 (MEM)
            tlx.fence_async_shared()
            tlx.async_descriptor_store(c_desc, c_smem, [m, n + 3*slice_n])
            tlx.barrier_arrive(tmem_empty_bar[0], 1)               # 4 of 4 arrivals

            tile_id += NUM_SMS

    # ── Warp Group 2: MMA (SMEM → TMEM) ─────────────────────────
    with tlx.async_task(num_warps=1, num_regs=24):
        while tile_id < num_tiles:
            for k in range(k_tiles):
                buf, phase = _get_bufidx_phase(smem_cnt, NUM_SMEM_BUFFERS)

                tlx.barrier_wait(A_full_bars[buf], phase)          # wait for A
                tlx.barrier_wait(B_full_bars[buf], phase)          # wait for B
                tlx.barrier_wait(tmem_empty_bar[0], ...)           # wait for TMEM free

                tlx.async_dot(
                    buffers_A[buf], buffers_B[buf],
                    tmem_buf[0],
                    use_acc=(k > 0),
                    mBarriers=[A_empty_bars[buf]],                  # signal A consumed
                )
                smem_cnt += 1

            # Signal epilogue: accumulator is ready
            tlx.barrier_arrive(tmem_full_bar[0], 1)
            tile_id += NUM_SMS

    # ── Warp Group 3: Producer / TMA Load (global → SMEM) ───────
    with tlx.async_task(num_warps=1, num_regs=24):
        while tile_id < num_tiles:
            for k in range(k_tiles):
                buf, phase = _get_bufidx_phase(smem_cnt, NUM_SMEM_BUFFERS)

                # Load A
                tlx.barrier_wait(A_empty_bars[buf], phase ^ 1)    # wait for MMA to consume
                tlx.barrier_expect_bytes(A_full_bars[buf], ...)
                tlx.async_descriptor_load(a_desc, buffers_A[buf],
                                          [offs_m, offs_k],
                                          A_full_bars[buf])        # signal A loaded

                # Load B
                tlx.barrier_expect_bytes(B_full_bars[buf], ...)
                tlx.async_descriptor_load(b_desc, buffers_B[buf],
                                          [offs_k, offs_n],
                                          B_full_bars[buf])        # signal B loaded
                smem_cnt += 1
            tile_id += NUM_SMS
```

### Algorithm → TLX Code Mapping Summary

| Algorithm Decision | TLX Code |
|---|---|
| ResMII = 960 (MEM-bound) | Producer gets dedicated warp group with `tlx.async_task(num_warps=1, num_regs=24)` |
| NUM_SMEM_BUFFERS = 3 | `tlx.local_alloc(..., 3)` + 3 mbarriers cycling via `smem_cnt % 3` |
| NUM_TMEM_BUFFERS = 1 | `tlx.local_alloc(..., 1, tlx.storage_kind.tmem)` — no MMA/epilogue overlap |
| EPILOGUE_SUBTILE = 4 (A.7) | 4 sub-chains flattened in cluster order (Pass C); `arrive_count=EPILOGUE_SUBTILE` on `tmem_empty_bar` |
| 3 warp groups | 3 nested `tlx.async_task()` blocks |
| SMEM producer→consumer sync | `barrier_expect_bytes` + `async_descriptor_load` + `barrier_wait` pairs |
| TMEM MMA→epilogue sync | `tmem_full_bar` / `tmem_empty_bar` pair |
| Phase cycling | `_get_bufidx_phase()`: `bufIdx = cnt % depth`, `phase = (cnt // depth) & 1` |
| No explicit prologue loop | Producer runs ahead naturally — barrier back-pressure from `A_empty_bars` limits it to `NUM_SMEM_BUFFERS - 1` iterations ahead |

---

## Worked Example: Blackwell Flash Attention Forward Kernel

This section walks through the algorithm using a **Blackwell Flash Attention forward kernel** — a significantly more complex example than GEMM because it uses all four pipelines (MEM, TC, CUDA, SFU) and has multiple loop-carried recurrences. We use the config from `blackwell_fa_ws.py`: `BLOCK_M=256, BLOCK_N=128, HEAD_DIM=128, NUM_BUFFERS_KV=3, NUM_BUFFERS_QK=1, NUM_MMA_GROUPS=2`.

The resulting TLX code corresponds to `blackwell_fa_ws.py`.

### FA Forward Dependency Graph

Flash Attention iterates over K/V blocks. Each iteration computes one block of attention scores and updates the running softmax + output accumulator. The DDG per iteration is:

```
LoadK[i] ─────────→ QK_MMA[i] ──→ RowMax[i] ──→ Scale/Sub[i] ──→ Exp2[i] ──→ RowSum[i]
                                                                                    │
LoadV[i] ───────────────────────────────────────────────────────────────────────→ PV_MMA[i]
                                                                                    │
                                                                              AccUpdate[i]

Loop-carried edges (distance=1):
  m_i[i]   → Alpha[i+1]      (old max for correction factor)
  l_i[i]   → l_update[i+1]   (running sum for normalization)
  Acc[i]   → AccUpdate[i+1]  (output accumulator correction: acc *= alpha)
```

With `NUM_MMA_GROUPS=2`, Q is split into two 128×128 sub-tiles. Each group processes its own QK and PV independently, with its own softmax state (m_i, l_i, acc).

**Functional unit mapping:**

| Pipeline | Operations |
|----------|-----------|
| **MEM** | LoadK, LoadV (TMA loads), Q load (once, before loop) |
| **TC** | QK_MMA (Q @ K^T), PV_MMA (P @ V) |
| **CUDA** | RowMax, Scale/Subtract, RowSum, AccUpdate (acc *= alpha), type conversions |
| **SFU** | Exp2 (elementwise), Alpha = Exp2(scalar) |

Unlike GEMM, all four pipelines are active.

### Pass A, Step 1: Compute MinII

Using approximate Blackwell latencies (128×128 tiles):

```
LoadK       (TMA 128×128 bf16):        ~640 cycles
LoadV       (TMA 128×128 bf16):        ~640 cycles
QK_MMA      (tcgen05.mma 128×128×128): ~900 cycles
PV_MMA      (tcgen05.mma 128×128×128): ~900 cycles
RowMax      (128-wide reduce):         ~336 cycles
Scale/Sub   (elementwise):             ~130 cycles
Exp2        (elementwise transcend.):  ~662 cycles
Alpha       (Exp2 scalar):            ~43 cycles
RowSum      (128-wide reduce):         ~508 cycles
AccUpdate   (acc *= alpha):           ~105 cycles
```

**ResMII** (resource-constrained):
```
MEM:  LoadK(640) + LoadV(640)                           = 1280
TC:   QK(900) + PV(900)                                 = 1800
CUDA: RowMax(336) + Scale(130) + RowSum(508) + Acc(105)  = 1079
SFU:  Exp2(662) + Alpha(43)                              = 705

ResMII = max(1280, 1800, 1079, 705) = 1800  (TC-bound)
```

**RecMII** (recurrence-constrained):
The critical recurrence goes through the accumulator:
```
Recurrence: Acc[i] → AccUpdate[i+1] → ... → PV_MMA[i+1] → Acc[i+1]
  Path: AccUpdate(105) → [barrier] → PV_MMA waits for P → ...
  Total latency along path ≈ entire iteration body
  Distance: 1

For the m_i recurrence:
  m_i[i] → Alpha[i+1] → AccUpdate[i+1]
  Path: Alpha(43) + AccUpdate(105) = 148
  Distance: 1
  RecMII contribution: 148
```

The accumulator recurrence effectively spans the full iteration. However, warp specialization breaks this recurrence by placing AccUpdate on a separate warp group — the accumulator correction runs concurrently with the next iteration's QK_MMA and softmax.

**MinII:**
```
MinII = max(ResMII, RecMII_effective) = 1800  (TC-bound)
```

FA forward is **compute-bound** (TC pipeline is the bottleneck), unlike GEMM which was memory-bound.

### Pass A.5 Applied: Data Partitioning (NUM_MMA_GROUPS=2)

Data partitioning is **optional**. It is applied when the TC pipeline is fully utilized but has only a few large ops, limiting the modulo scheduler's ability to interleave them across iterations. For FA forward with `BLOCK_M=256`:

**Before splitting** (monolithic ops):
```
TC per iteration: QK_MMA(256×128×128) = 900 cycles + PV_MMA(256×128×128) = 900 cycles = 1800
```

The TC pipeline is fully utilized with just two large ops. But the softmax between QK and PV creates a dependency gap — QK must finish before softmax can run, and softmax must finish before PV can start. With monolithic 900-cycle ops, there's no room to interleave anything during the softmax wait.

**After splitting** with `NUM_MMA_GROUPS=2` (splitting along M):
```
QK_MMA(256×128×128) → QK_g0(128×128×128) + QK_g1(128×128×128)
PV_MMA(256×128×128) → PV_g0(128×128×128) + PV_g1(128×128×128)

TC per iteration: QK_g0(450) + QK_g1(450) + PV_g0(450) + PV_g1(450) = 1800
```

Now there are **4 smaller ops** instead of 2 large ones. This gives the modulo scheduler more flexibility to interleave them with softmax and across iterations. The split also creates independent softmax instances per group — g0's softmax can run while g1's QK is still computing.

The DDG after splitting:
```
LoadK[i] ──→ QK_g0[i] ──→ Softmax_g0[i] ──→ PV_g0[i]
         ──→ QK_g1[i] ──→ Softmax_g1[i] ──→ PV_g1[i]
LoadV[i] ─────────────────────────────────→ PV_g0[i]
         ─────────────────────────────────→ PV_g1[i]

Key: QK_g0 and QK_g1 share K (same SMEM buffer)
     PV_g0 and PV_g1 share V (same SMEM buffer)
     But Softmax_g0 and Softmax_g1 are INDEPENDENT
     (each has its own m_i, l_i, acc in registers/TMEM)
```

This independence is what enables the pipelined schedule: Softmax_g1 can run concurrently with PV_g0 or QK_g0 of the next iteration, because they're on different pipelines (CUDA/SFU vs TC) and operate on different data.

The modulo scheduler now sees 4 TC ops of 450 cycles each instead of 2 TC ops of 900 cycles. It can place them in any valid order within the II=1800 window, subject to dependency constraints. This produces the two schedules shown below.

### Pass A, Step 2: Modulo Schedule

With `NUM_MMA_GROUPS=2`, each MMA op is split into two sub-ops (g0 and g1), each taking ~450 cycles. The modulo schedule operates on these **split ops**, not the monolithic 900-cycle ops. This is critical — the in-group pipelining emerges directly from the modulo schedule's placement of split ops across overlapping iterations.

#### What the schedule stores

The schedule is a dict mapping each op to a tuple `(cycle, pipeline, stage)`:

- **cycle**: The cycle within the II-length reservation table (0 ≤ cycle < II) at which this op starts
- **pipeline**: Which hardware unit executes it
- **stage**: How many II periods *ahead* this op runs relative to the iteration that "owns" it. Stage 0 means the op executes during its own iteration's II window. Stage 1 means it is **deferred** by one II period — it executes during the *next* iteration's time window.

The stage is the key concept. If you print the schedule:

```python
def dump_schedule(schedule, II):
    print(f"II = {II}")
    print(f"{'Op':<20} {'Cycle':>6} {'Pipeline':>8} {'Stage':>6}  {'Absolute':>8}")
    print("-" * 60)
    for op, (cycle, pipe, stage) in sorted(
        schedule.items(), key=lambda x: x[1][0] + x[1][2] * II
    ):
        abs_cycle = cycle + stage * II
        print(f"{op:<20} {cycle:>6} {pipe:>8} {stage:>6}  {abs_cycle:>8}")
```

#### Basic schedule (blackwell_fa_ws.py)

All ops at stage=0 — no cross-iteration overlap:

```
II = 1800
Op                    Cycle Pipeline  Stage  Absolute
------------------------------------------------------------
LoadK                     0      MEM      0         0
QK_g0                     0       TC      0         0
RowMax_g0               450     CUDA      0       450
QK_g1                   450       TC      0       450
Exp2_g0                 580      SFU      0       580
LoadV                   640      MEM      0       640
PV_g0                   900       TC      0       900
RowMax_g1               900     CUDA      0       900
Exp2_g1                1030      SFU      0      1030
AccUpdate_g0           1200     CUDA      0      1200
PV_g1                  1350       TC      0      1350
AccUpdate_g1           1650     CUDA      0      1650
```

```python
schedule_basic = {
    "LoadK":        (0,    MEM,  0),
    "QK_g0":        (0,    TC,   0),
    "QK_g1":        (450,  TC,   0),
    "RowMax_g0":    (450,  CUDA, 0),
    "Exp2_g0":      (580,  SFU,  0),
    "LoadV":        (640,  MEM,  0),
    "PV_g0":        (900,  TC,   0),
    "RowMax_g1":    (900,  CUDA, 0),
    "Exp2_g1":      (1030, SFU,  0),
    "AccUpdate_g0": (1200, CUDA, 0),
    "PV_g1":        (1350, TC,   0),
    "AccUpdate_g1": (1650, CUDA, 0),
}
II = 1800
```

```
Cycle:   0        450      900      1350     1800 (=II)
         ├────────┼────────┼────────┼────────┤
TC:      [QK_g0  ][QK_g1  ][PV_g0  ][PV_g1  ]
MEM:     [ LoadK  ][ LoadV ]        ·  (idle)
CUDA:              [RowMax0][RowMax1][AccUpd0][AccUpd1]
SFU:             [Exp2_0 ][Exp2_1 ]
```

Problem: PV_g1 at cycle 1350 needs P1 from softmax g1. Softmax g1 starts at cycle 900 (after QK_g1) and takes ~450 cycles → finishes at ~1350. Zero slack — any softmax delay stalls the TC pipeline.

#### Pipelined schedule (blackwell_fa_ws_pipelined.py)

Rau's algorithm finds a better placement by assigning **stage=1** to PV_g1:

```
II = 1800
Op                    Cycle Pipeline  Stage  Absolute
------------------------------------------------------------
LoadK                     0      MEM      0         0
QK_g0                     0       TC      0         0
RowMax_g0               450     CUDA      0       450
PV_g1                   450       TC      1      2250  ← stage=1!
Exp2_g0                 580      SFU      0       580
LoadV                   640      MEM      0       640
QK_g1                   900       TC      0       900
RowMax_g1               900     CUDA      0       900
Exp2_g1                1030      SFU      0      1030
AccUpdate_g0           1200     CUDA      0      1200
PV_g0                  1350       TC      0      1350
AccUpdate_g1           1650     CUDA      0      1650
```

```python
schedule_pipelined = {
    "LoadK":        (0,    MEM,  0),
    "QK_g0":        (0,    TC,   0),
    "QK_g1":        (900,  TC,   0),
    "PV_g0":        (1350, TC,   0),
    "PV_g1":        (450,  TC,   1),   # ← stage=1: deferred by one II
    "RowMax_g0":    (450,  CUDA, 0),
    "Exp2_g0":      (580,  SFU,  0),
    "LoadV":        (640,  MEM,  0),
    "RowMax_g1":    (900,  CUDA, 0),
    "Exp2_g1":      (1030, SFU,  0),
    "AccUpdate_g0": (1200, CUDA, 0),
    "AccUpdate_g1": (1650, CUDA, 0),
}
II = 1800
```

**PV_g1 has stage=1.** This means: when iteration i starts at absolute cycle `i * II`, PV_g1 for iteration i runs at absolute cycle `i * II + 450 + 1 * 1800 = (i+1) * II + 450`. PV_g1 for iteration i is **deferred** to run during iteration i+1's time window.

The steady-state reservation table — what actually executes during one II window:

```
Cycle:   0        450      900      1350     1800 (=II)
         ├────────┼────────┼────────┼────────┤
TC:      [QK_g0[i]][PV_g1[i-1]][QK_g1[i]][PV_g0[i]]
                   ↑ stage=1 op from iter i-1 fills this slot
MEM:     [LoadK[i] ][ LoadV[i] ]   ·  (idle)
CUDA:               [RowMax0[i]][RowMax1[i]][AccUpd0[i]][AccUpd1[i]]
SFU:              [Exp2_0[i]][Exp2_1[i]]
```

The TC sequence in steady state: QK_g0[i], PV_g1[i-1], QK_g1[i], PV_g0[i]. This is exactly `blackwell_fa_ws_pipelined.py` lines 430–483.

#### Why stage=1 eliminates the stall

With stage=0 (basic): PV_g1[i] needs P1[i]. Softmax g1[i] finishes at absolute cycle ~`i*1800 + 1350`. PV_g1[i] starts at absolute `i*1800 + 1350`. **Zero slack.**

With stage=1 (pipelined): PV_g1[i] runs at absolute cycle `(i+1)*1800 + 450 = i*1800 + 2250`. Softmax g1[i] still finishes at `i*1800 + 1350`. **Slack = 2250 - 1350 = 900 cycles.** No stall possible.

The cost: PV_g1 for iteration i is delayed by one II period. This adds one iteration of **pipeline latency** (the loop needs one extra prolog iteration to fill the pipeline), but the steady-state throughput is unchanged.

#### How stage determines prolog/epilog

```python
max_stage = max(stage for _, _, stage in schedule_pipelined.values())  # = 1

# Prolog: max_stage iterations where higher-stage ops have no predecessor
#   Iteration 0: only stage=0 ops run
#     TC: QK_g0[0], QK_g1[0], PV_g0[0]        ← 3 ops (no PV_g1[-1])
#
# Steady state: all stages active
#   Iteration i (i >= 1):
#     TC: QK_g0[i], PV_g1[i-1], QK_g1[i], PV_g0[i]  ← 4 ops
#
# Epilog: drain deferred ops from the last iteration
#   After loop:
#     TC: PV_g1[last]                           ← 1 op
```

This maps directly to the pipelined kernel:
- **Lines 391–426**: Prolog — QK_g0[0], QK_g1[0], PV_g0[0]
- **Lines 430–483**: Main loop — QK_g0[i], PV_g1[i-1], QK_g1[i], PV_g0[i]
- **Lines 487–496**: Epilog — PV_g1[last]

#### What the schedule does NOT capture: in-group instruction ordering

The `(cycle, pipeline, stage)` schedule tells you **which TC slot each op occupies** and **which iteration it belongs to** (via stage). But it does not tell you the **order in which the MMA warp group issues these ops**. All four TC ops occupy consecutive 450-cycle slots on the same pipeline — the schedule says they tile the II window perfectly, but not which one the warp group's code emits first.

This is because the modulo schedule is a **resource-time map**, not an instruction sequence. It answers "at what absolute cycle does this op execute on the hardware?" — but a warp group is a single thread that issues `async_dot` calls sequentially. The TC pipeline executes them in FIFO order, so the issue order determines the execution order.

The in-group instruction ordering is determined by **Pass C**, which takes the schedule and produces a per-warp-group **instruction sequence**:

```python
# Pass C output for the MMA warp group:
mma_instruction_sequence = [
    # (op, iteration_offset, barrier_waits, barrier_signals)
    ("QK_g0",  0, [kv_fulls[k], q_fulls[0]],           [qk_fulls[0]]),
    ("PV_g1", -1, [p_fulls[1], acc_fulls[1], kv_fulls[v_prev]], [kv_empties[v_prev]]),
    ("QK_g1",  0, [],                                    [qk_fulls[1], kv_empties[k]]),
    ("PV_g0",  0, [p_fulls[0], acc_fulls[0], kv_fulls[v]],     []),
]
```

This sequence is what determines the actual TLX code. The `iteration_offset=-1` on PV_g1 means it uses data from the previous iteration (v_prev, p[3] instead of p[1]).

**How Pass C derives this sequence from the schedule:**

1. **Collect TC ops** from the schedule: QK_g0 (cycle=0, stage=0), QK_g1 (cycle=900, stage=0), PV_g0 (cycle=1350, stage=0), PV_g1 (cycle=450, stage=1)

2. **Compute absolute execution time** within one II window for steady state: ops from the current iteration use `cycle`, ops from the previous iteration (stage=1 deferred by one II) appear at `cycle` but logically belong to iteration i-1

3. **Sort by cycle** to get the TC pipeline execution order: 0 (QK_g0), 450 (PV_g1), 900 (QK_g1), 1350 (PV_g0)

4. **Insert barrier waits** before each op: each op waits on the barriers that its data dependencies require (e.g., PV_g1 waits for p_fulls and acc_fulls from iteration i-1)

5. **Insert barrier signals** after each op: each op signals the barriers that free resources for other warp groups (e.g., QK_g1 signals kv_empties to free the K buffer for the producer)

The result is the instruction sequence above, which maps 1:1 to the `async_dot` calls in `blackwell_fa_ws_pipelined.py`.

### Pass A, Step 3: Derive Pipeline Depths

**K tile (SMEM):**
```
Resource: K tile
  Producer: LoadK at cycle 0, latency 640
  Consumer: QK_MMA at cycle 640, latency 900
  Last consumer end: 640 + 900 = 1540
  Lifetime = 1540 - 0 = 1540
  num_buffers = floor(1540 / 1800) + 1 = 0 + 1 = 1
```

But K and V share a single `kv_tiles` buffer pool with `NUM_BUFFERS_KV=3`. Each iteration loads K then V into alternating slots from this pool. The 3 buffers allow the producer to stay ahead:

```
Iteration i:   K → slot 0, V → slot 1
Iteration i+1: K → slot 2, V → slot 0  (slot 0 freed after QK_MMA[i] consumed it)
```

**QK result (TMEM):**
```
Resource: QK result
  Producer: QK_MMA writes to TMEM
  Consumer: Softmax (RowMax, Scale, Exp2) reads from TMEM
  With NUM_BUFFERS_QK=1: single-buffered
    → Softmax must finish before next QK_MMA can write
```

**Accumulator (TMEM) — buffer merging applied:**
The `qk_tiles`, `p_tiles`, `alpha_tiles`, `l_tiles`, and `m_tiles` all declare `reuse=qk_tiles`, meaning they share the same physical TMEM buffer. This is exactly the **lifetime-aware buffer merging** from Pass A Step 4.5:

```
QK result:  live from QK_MMA start → softmax reads finish
P matrix:   live from Exp2 finish → PV_MMA finish
Alpha/l/m:  live from softmax compute → correction apply

These lifetimes are non-overlapping within the QK TMEM buffer:
  QK is consumed before P is produced (softmax converts QK → P)
  Alpha/l/m occupy only column 0 of the tile, coexisting with P in upper columns
```

This merging saves substantial TMEM — without it, separate buffers for QK, P, alpha, l, m would exceed the 256KB TMEM budget.

### Pass A, Step 4: Memory Budget Check

```
SMEM:
  Q tiles:  128 × 128 × 2B × 2 groups                  =  65,536 B
  KV tiles: 128 × 128 × 2B × 3 buffers                  =  98,304 B
  Barriers:                                              ~    256 B
  Total SMEM ≈ 164,096 B  (< 232 KB limit ✓)

TMEM:
  QK/P/alpha/l/m (merged): 128 × 128 × 4B × 2 groups   = 131,072 B
  Acc tiles:               128 × 128 × 4B × 2 groups    = 131,072 B
  Total TMEM = 262,144 B = 256 KB  (just fits ✓)
```

The buffer merging (`reuse=qk_tiles`) is essential — without it, QK + P + acc would require 384KB of TMEM, exceeding the limit.

### Pass B, Step 1: Partition into Warp Groups

Pipeline utilization within II=1800:
```
MEM:  1280/1800 = 71%  → dedicated warp group  ✓  (> 30%)
TC:   1800/1800 = 100% → dedicated warp group  ✓  (> 30%)
CUDA: 1079/1800 = 60%  → dedicated warp group  ✓  (> 30%)
SFU:   705/1800 = 39%  → dedicated warp group  ✓  (> 30%)
```

All four pipelines exceed the 30% threshold. However, SFU (Exp2) and CUDA (RowMax, Scale, RowSum, AccUpdate) are tightly coupled by data dependencies — Exp2 produces P which RowSum consumes, and both feed into AccUpdate. Merging them into a single "softmax" warp group is more practical than adding cross-group barriers between every softmax sub-op.

The actual kernel uses a different grouping that accounts for the recurrence structure:

**Result: 4 warp groups:**

| Warp Group | Role | Operations | Warps | Regs |
|-----------|------|-----------|-------|------|
| Producer | TMA loads | LoadQ (once), LoadK, LoadV | 1 | 24 |
| MMA | Tensor core ops | QK_MMA, PV_MMA | 1 | 24 |
| Softmax | Online softmax + P generation | RowMax, Scale, Exp2, RowSum, P conversion | 4 | 152 |
| Correction | Accumulator update + epilogue | AccUpdate (acc *= alpha), final normalization, store O | default | — |

The softmax group gets 4 warps and 152 registers because it performs register-heavy reductions (RowMax, RowSum) and elementwise compute (Exp2) across BLOCK_M_SPLIT=128 rows. The correction group is lightweight — it only scales the accumulator by alpha each iteration and handles the final epilogue.

### Pass B, Step 2: Insert Synchronization

The cross-group data flows are more complex than GEMM:

| Boundary | Resource | Direction | Barrier Type | Depth |
|----------|----------|-----------|-------------|-------|
| Producer → MMA | Q tile in SMEM | data ready | `mbarrier` | 1 per group (loaded once) |
| Producer → MMA | K/V tiles in SMEM | data ready | `mbarrier` (`kv_fulls`) | 3 (NUM_BUFFERS_KV) |
| MMA → Producer | K/V consumed | buffer free | `mbarrier` (`kv_empties`) | 3 |
| MMA → Softmax | QK result in TMEM | data ready | `mbarrier` (`qk_fulls`) | 1 per group |
| Softmax → MMA | P matrix in TMEM | data ready | `mbarrier` (`p_fulls`) | 1 per group |
| Softmax → Correction | Alpha in TMEM | data ready | `mbarrier` (`alpha_fulls`) | 1 per group |
| Correction → Softmax | Alpha consumed | buffer free | `mbarrier` (`alpha_empties`) | 1 per group |
| MMA → Correction | Acc updated by PV | data ready | `mbarrier` (`acc_fulls`) | 1 per group |
| Correction → MMA | Acc corrected | buffer free | `mbarrier` (`acc_empties`) | 1 per group |
| Softmax → Correction | l_i, m_i for epilogue | data ready | `mbarrier` (`l_fulls`) | 1 per group |

The circular dependency is: MMA produces QK → Softmax produces P and Alpha → MMA consumes P for PV, Correction consumes Alpha → Correction frees Acc → MMA can write Acc again. This forms the pipelined loop.

### Pass B, Step 5: Generated TLX Code

#### Buffer Allocations

```python
# Q tiles: loaded once before the loop, stays in SMEM
q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), dtype, NUM_MMA_GROUPS)  # 2

# K/V tiles: shared buffer pool, 3-deep for producer-consumer overlap
kv_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), dtype, NUM_BUFFERS_KV)       # 3

# QK result in TMEM (also reused for P, alpha, l, m via buffer merging)
qk_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tl.float32,
                             NUM_MMA_GROUPS * NUM_BUFFERS_QK,                 # 2
                             tlx.storage_kind.tmem)

# P matrix — shares physical TMEM with qk_tiles
p_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), dtype,
                            NUM_MMA_GROUPS * NUM_BUFFERS_QK * 2,              # 4
                            tlx.storage_kind.tmem, reuse=qk_tiles)

# Alpha, l, m scalars — share physical TMEM with qk_tiles
alpha_tiles = tlx.local_alloc((BLOCK_M_SPLIT, 1), tl.float32,
                               HEAD_DIM * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
                               tlx.storage_kind.tmem, reuse=qk_tiles)
l_tiles = tlx.local_alloc(...)   # same pattern, reuse=qk_tiles
m_tiles = tlx.local_alloc(...)   # same pattern, reuse=qk_tiles

# Output accumulator in TMEM (separate, not merged)
acc_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tl.float32,
                              NUM_MMA_GROUPS * NUM_BUFFERS_QK,                # 2
                              tlx.storage_kind.tmem)
```

#### Barrier Allocations

```python
# Producer → MMA: Q loaded (one-shot, before loop)
q_fulls = tlx.alloc_barriers(NUM_MMA_GROUPS)                                 # 2

# Producer → MMA: K/V loaded / consumed
kv_fulls   = tlx.alloc_barriers(NUM_BUFFERS_KV)                              # 3
kv_empties = tlx.alloc_barriers(NUM_BUFFERS_KV)                              # 3

# MMA → Softmax: QK result ready
qk_fulls = tlx.alloc_barriers(NUM_MMA_GROUPS * NUM_BUFFERS_QK)               # 2

# Softmax → MMA: P matrix ready
p_fulls = tlx.alloc_barriers(NUM_MMA_GROUPS * NUM_BUFFERS_QK)                # 2

# MMA → Correction / Correction → MMA: accumulator handoff
acc_fulls   = tlx.alloc_barriers(NUM_MMA_GROUPS * NUM_BUFFERS_QK)            # 2
acc_empties = tlx.alloc_barriers(NUM_MMA_GROUPS * NUM_BUFFERS_QK)            # 2

# Softmax → Correction: alpha / l / m handoff
alpha_fulls   = tlx.alloc_barriers(NUM_MMA_GROUPS * NUM_BUFFERS_QK)          # 2
alpha_empties = tlx.alloc_barriers(NUM_MMA_GROUPS * NUM_BUFFERS_QK)          # 2
l_fulls       = tlx.alloc_barriers(NUM_MMA_GROUPS)                           # 2
```

#### Warp-Specialized Kernel Structure

```python
with tlx.async_tasks():

    # ── Warp Group 1: Correction (acc *= alpha, epilogue) ─────
    with tlx.async_task("default"):
        for _ in range(lo, hi, BLOCK_N):
            for cid in range(NUM_MMA_GROUPS):
                # Wait for alpha from softmax
                tlx.barrier_wait(alpha_fulls[buf_idx], phase)
                alpha = tlx.local_load(alpha_tiles[cid * ...])
                tlx.barrier_arrive(alpha_empties[buf_idx])

                # Correct accumulator: acc *= alpha
                acc = tlx.local_load(acc_tiles[buf_idx])
                acc = acc * alpha
                tlx.local_store(acc_tiles[buf_idx], acc)
                tlx.barrier_arrive(acc_fulls[buf_idx])         # signal MMA

        # Epilogue: normalize by l_i and store output
        for cid in range(NUM_MMA_GROUPS):
            tlx.barrier_wait(l_fulls[cid], 0)
            l = tlx.local_load(l_tiles[...])
            acc = tlx.local_load(acc_tiles[cid])
            acc = acc / l
            desc_o.store([offset, 0], acc.to(output_dtype))

    # ── Warp Group 2: Softmax (online softmax + P) ────────────
    with tlx.async_task(num_warps=4, registers=152, replicate=NUM_MMA_GROUPS):
        m_i = -inf;  l_i = 1.0;  qk_scale = sm_scale * 1/log(2)
        cid = tlx.async_task_replica_id()

        for _ in range(lo, hi, BLOCK_N):
            # Wait for QK result from MMA
            tlx.barrier_wait(qk_fulls[buf_idx], phase)
            qk = tlx.local_load(qk_tiles[buf_idx])

            # Online softmax
            m_ij = max(m_i, rowmax(qk) * qk_scale)
            alpha = exp2(m_i - m_ij)

            # Send alpha to correction group
            tlx.barrier_wait(alpha_empties[buf_idx], prev_phase)
            tlx.local_store(alpha_tiles[...], alpha)
            tlx.barrier_arrive(alpha_fulls[buf_idx])

            # Compute P = exp2(qk * scale - m_ij)
            p = exp2(qk * qk_scale - m_ij)
            l_i = l_i * alpha + rowsum(p)
            p = p.to(input_dtype)

            # Send P to MMA for PV dot
            tlx.local_store(p_tiles[...], p)
            tlx.barrier_arrive(p_fulls[buf_idx])

            m_i = m_ij

        # Send final l_i, m_i to correction for epilogue
        tlx.local_store(l_tiles[...], l_i)
        tlx.local_store(m_tiles[...], m_i)
        tlx.barrier_arrive(l_fulls[cid])

    # ── Warp Group 3: MMA (QK and PV dots) ────────────────────
    with tlx.async_task(num_warps=1, registers=24):
        # Wait for Q to be loaded (one-shot)
        for cid in range(NUM_MMA_GROUPS):
            tlx.barrier_wait(q_fulls[cid], 0)

        for i in range(lo, hi, BLOCK_N):
            # -- QK dot: Q @ K^T --
            tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)     # wait for K
            k_tile = tlx.local_trans(kv_tiles[k_bufIdx])       # transpose K
            for cid in range(NUM_MMA_GROUPS):
                tlx.async_dot(q_tiles[cid], k_tile,
                              qk_tiles[buf_idx],
                              use_acc=False,
                              mBarriers=[qk_fulls[buf_idx],    # signal softmax
                                         kv_empties[k_bufIdx]])# free K buffer

            # -- PV dot: P @ V --
            tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)      # wait for V
            for cid in range(NUM_MMA_GROUPS):
                tlx.barrier_wait(p_fulls[buf_idx], phase)       # wait for P from softmax
                tlx.barrier_wait(acc_fulls[buf_idx], phase)     # wait for acc correction
                tlx.async_dot(p_tiles[...], kv_tiles[v_bufIdx],
                              acc_tiles[buf_idx],
                              use_acc=(i > 0),
                              mBarriers=[acc_empties[buf_idx],  # signal correction
                                         kv_empties[v_bufIdx]])# free V buffer

    # ── Warp Group 4: Producer / TMA Load ──────────────────────
    with tlx.async_task(num_warps=1, registers=24):
        # Load Q once (stays in SMEM for entire block)
        for cid in range(NUM_MMA_GROUPS):
            tlx.barrier_expect_bytes(q_fulls[cid], 2 * BLOCK_M_SPLIT * HEAD_DIM)
            tlx.async_descriptor_load(desc_q, q_tiles[cid], [...], q_fulls[cid])

        # Loop: load K and V alternately into kv_tiles pool
        for _ in range(lo, hi, BLOCK_N):
            # Load K
            tlx.barrier_wait(kv_empties[k_bufIdx], prev_phase)   # wait for MMA to consume
            tlx.barrier_expect_bytes(kv_fulls[k_bufIdx], 2 * BLOCK_N * HEAD_DIM)
            tlx.async_descriptor_load(desc_k, kv_tiles[k_bufIdx],
                                      [kv_offset, 0], kv_fulls[k_bufIdx])
            # Load V
            tlx.barrier_wait(kv_empties[v_bufIdx], prev_phase)
            tlx.barrier_expect_bytes(kv_fulls[v_bufIdx], 2 * BLOCK_N * HEAD_DIM)
            tlx.async_descriptor_load(desc_v, kv_tiles[v_bufIdx],
                                      [kv_offset, 0], kv_fulls[v_bufIdx])
            kv_offset += BLOCK_N
```

### Algorithm → TLX Code Mapping Summary

| Algorithm Decision | TLX Code |
|---|---|
| ResMII = 1800 (TC-bound) | MMA gets dedicated warp group; TC pipeline is the bottleneck |
| All 4 pipelines > 30% utilization | 4 warp groups (Producer, MMA, Softmax, Correction) |
| Softmax needs register-heavy reductions | `tlx.async_task(num_warps=4, registers=152, replicate=NUM_MMA_GROUPS)` |
| NUM_BUFFERS_KV = 3 | `kv_tiles = tlx.local_alloc(..., 3)` — K and V share a 3-deep pool |
| NUM_BUFFERS_QK = 1 | Single-buffered QK result — softmax must complete before next QK_MMA |
| Q loaded once (not per-iteration) | `q_tiles` loaded before the loop, stays in SMEM |
| TMEM buffer merging (Step 4.5) | `p_tiles`, `alpha_tiles`, `l_tiles`, `m_tiles` all use `reuse=qk_tiles` |
| Acc recurrence broken by warp specialization | Correction group runs `acc *= alpha` concurrently with next iter's QK |
| K/V interleaved in shared pool | `accum_cnt_kv` increments by 2 per iteration (K at even, V at odd slots) |
| `replicate=NUM_MMA_GROUPS` | Each MMA group gets its own softmax replica with independent m_i, l_i state |

### Pass C Applied: In-Group Pipelining (blackwell_fa_ws_pipelined.py)

The basic `blackwell_fa_ws.py` kernel processes MMA groups sequentially within each warp group. In the MMA group, group 0's QK dot finishes before group 1's QK dot starts. Similarly, in the load group, Q0 and Q1 are loaded one after another without interleaving with K/V loads.

The pipelined variant `blackwell_fa_ws_pipelined.py` applies **Pass C (Global Scheduling Refinement)** to reorder instructions *within* each warp group. This is intra-group instruction scheduling — the warp group structure from Pass B stays the same, but the operation ordering within the MMA and load groups changes to minimize cross-warp stalls.

#### MMA Group: Interleaving QK and PV Across Groups

**Before (basic — sequential within groups):**
```python
# Each iteration processes both groups in lockstep
for i in range(lo, hi, BLOCK_N):
    # QK dots for both groups, then PV dots for both groups
    tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
    k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
    for cid in range(NUM_MMA_GROUPS):
        tlx.async_dot(q_tiles[cid], k_tile, qk_tiles[...])    # QK g0, then QK g1
    for cid in range(NUM_MMA_GROUPS):
        tlx.barrier_wait(p_fulls[...])
        tlx.async_dot(p_tiles[...], kv_tiles[v_bufIdx], acc_tiles[...])  # PV g0, then PV g1
```

**After (pipelined — interleaved across groups and iterations):**
```python
# Prolog: QK g0, QK g1, PV g0 (no PV g1 yet — it will use iter 0's V)
tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
tlx.async_dot(q_tiles[0], k_tile, qk_tiles[0], mBarriers=[qk_fulls[0]])
tlx.async_dot(q_tiles[1], k_tile, qk_tiles[1], mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]])

tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
tlx.barrier_wait(p_fulls[0], qk_phase)
tlx.async_dot(p_tiles[1], kv_tiles[v_bufIdx], acc_tiles[0], use_acc=False)

# Main loop: 4 MMA ops interleaved across groups and iterations
for i in range(lo + BLOCK_N, hi, BLOCK_N):
    # 1. QK g0[i]           — start current iteration's QK for group 0
    tlx.async_dot(q_tiles[0], k_tile, qk_tiles[0], mBarriers=[qk_fulls[0]])

    # 2. PV g1[i-1]         — finish PREVIOUS iteration's PV for group 1
    tlx.barrier_wait(p_fulls[1], qk_phase_prev)
    tlx.async_dot(p_tiles[3], kv_tiles[v_bufIdx_prev], acc_tiles[1],
                  mBarriers=[kv_empties[v_bufIdx_prev]])

    # 3. QK g1[i]           — current iteration's QK for group 1
    tlx.async_dot(q_tiles[1], k_tile, qk_tiles[1],
                  mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]])

    # 4. PV g0[i]           — current iteration's PV for group 0
    tlx.barrier_wait(p_fulls[0], qk_phase)
    tlx.async_dot(p_tiles[1], kv_tiles[v_bufIdx], acc_tiles[0], use_acc=True)

# Epilog: PV g1[last] — finish the last iteration's group 1
tlx.async_dot(p_tiles[3], kv_tiles[v_bufIdx], acc_tiles[1], use_acc=acc1_init,
              mBarriers=[acc_empties[1], kv_empties[v_bufIdx]])
```

The key insight is that **PV g1 from iteration i-1 is interleaved with QK g0 from iteration i**. This works because:
- PV g1 uses the *previous* iteration's V tile and P tile — no dependency on the current iteration
- QK g0 uses the *current* iteration's K tile — no dependency on PV g1
- This overlap hides the softmax latency for group 1: while softmax computes P for g1, the MMA is already working on QK g0 for the next iteration

The prolog/epilog structure handles the boundary: iteration 0 has no previous PV g1 to interleave with, and the final iteration needs an extra PV g1 after the loop ends.

#### Load Group: Interleaving Q and K/V Loads

**Before (basic):**
```python
# All Q sub-tiles loaded together, then K/V loop
for cid in range(NUM_MMA_GROUPS):
    tlx.async_descriptor_load(desc_q, q_tiles[cid], ...)

for _ in range(lo, hi, BLOCK_N):
    tlx.async_descriptor_load(desc_k, kv_tiles[k_bufIdx], ...)
    tlx.async_descriptor_load(desc_v, kv_tiles[v_bufIdx], ...)
```

**After (pipelined):**
```python
# Interleave Q0, K, Q1, V to match MMA consumption order
tlx.async_descriptor_load(desc_q, q_tiles[0], ...)       # Q g0 — needed first by MMA

tlx.barrier_wait(kv_empties[k_bufIdx], k_phase ^ 1)
tlx.async_descriptor_load(desc_k, kv_tiles[k_bufIdx], ...)  # K — needed after Q g0

tlx.async_descriptor_load(desc_q, q_tiles[1], ...)       # Q g1 — needed after K

tlx.barrier_wait(kv_empties[v_bufIdx], v_phase ^ 1)
tlx.async_descriptor_load(desc_v, kv_tiles[v_bufIdx], ...)  # V — needed after QK finishes

# Steady-state loop: K, V in order (Q stays in SMEM)
for _ in range(lo + BLOCK_N, hi, BLOCK_N):
    tlx.async_descriptor_load(desc_k, kv_tiles[k_bufIdx], ...)
    tlx.async_descriptor_load(desc_v, kv_tiles[v_bufIdx], ...)
```

The load order is reordered to match the MMA group's consumption order: Q0 is needed before K (for QK g0), and K is needed before Q1 (since QK g0 starts before QK g1). This minimizes the time between load completion and consumption, reducing stalls.

#### Why This Matters: Cross-Warp Stall Reduction

The pipelined ordering directly addresses the Pass C priority function:

| Weight | Effect in FA pipelined |
|--------|----------------------|
| `W2` (global impact) | PV g1 is pulled earlier because acc_tiles[1] unblocks the correction group |
| `W1` (local critical path) | QK g0 is interleaved with PV g1 to keep the TC pipeline continuously fed |
| Barrier ordering | `kv_empties` is signaled as `mBarrier` on the *last* MMA that uses K (QK g1), not the first (QK g0). This frees the K buffer as soon as possible for the producer |

The net effect: the TC pipeline is kept closer to 100% utilization because the softmax latency for group 1 is hidden behind QK g0 of the next iteration, rather than stalling the TC pipeline while waiting.

### GEMM vs FA Forward: Key Differences

| Aspect | GEMM | Flash Attention Forward |
|--------|------|----------------------|
| Active pipelines | 2 (MEM, TC) | 4 (MEM, TC, CUDA, SFU) |
| Bottleneck | MEM (ResMII=1280) | TC (ResMII=1800) |
| Warp groups | 3 | 4 |
| Loop-carried state | Accumulator only | Accumulator + m_i + l_i |
| Buffer merging | None needed | Essential (QK/P/alpha/l/m share TMEM) |
| Q/A tile loading | Per K-iteration | Once before loop |
| KV buffer strategy | Separate A, B pools | Shared KV pool, K and V interleaved |
| Softmax | None | Online softmax with correction group |
| Recurrence breaking | Direct (use_acc flag) | Warp specialization (acc correction concurrent with next QK) |

---

## Worked Example: Blackwell Flash Attention Backward Kernel

This section walks through the algorithm using the **Flash Attention backward kernel** — the most complex of the three examples. The backward pass must compute three gradients (dQ, dK, dV) from the saved forward activations, requiring **5 concurrent matrix multiplies per inner-loop iteration** and heavy TMEM buffer reuse. We use the config from `blackwell_fa_ws_pipelined_persistent.py`: `BLOCK_M1=128, BLOCK_N1=128, HEAD_DIM=128, NUM_BUFFERS_KV=1, NUM_BUFFERS_Q=2, NUM_BUFFERS_DO=1, NUM_BUFFERS_DS=1, NUM_BUFFERS_TMEM=1`.

The resulting TLX code corresponds to `_attn_bwd_ws` in `blackwell_fa_ws_pipelined_persistent.py`.

### FA Backward Dependency Graph

The backward pass fixes a K/V block and iterates over Q/dO blocks (the inner M-loop). Each iteration computes:

```
1. qkT = K @ Q^T                → attention scores (transposed)
2. pT  = softmax(qkT)           → attention weights (transposed)
3. dpT = V @ dO^T               → gradient through attention weights
4. dsT = pT * (dpT - delta)     → gradient of scores (pre-softmax)
5. dV += pT @ dO                → gradient for V (accumulated)
6. dK += dsT @ Q                → gradient for K (accumulated)
7. dQ  = dsT^T @ K              → gradient for Q (per-block, atomically reduced)
```

```
LoadK ──→ (stays for all M-blocks)
LoadV ──→ (stays for all M-blocks)
  For each M-block:
    LoadQ[j]  ──→ QK_MMA: K @ Q^T[j] ──→ Softmax ──→ pT ──→ dV_MMA: pT @ dO[j]
    LoaddO[j] ──→ dP_MMA: V @ dO^T[j] ──→ ds = pT*(dpT-δ) ──→ dK_MMA: dsT @ Q[j]
                                                              ──→ dQ_MMA: dsT^T @ K

Loop-carried edges (distance=1, across M-blocks):
  dV[j] → dV[j+1]   (dV += pT @ dO, accumulated)
  dK[j] → dK[j+1]   (dK += dsT @ Q, accumulated)
```

**Key structural difference from forward:** K and V are loaded once per outer tile and stay in SMEM. Q and dO are loaded per inner iteration (they change with each M-block). The gradients dK and dV accumulate across M-blocks, while dQ is computed fresh each iteration and atomically added to global memory.

**Functional unit mapping:**

| Pipeline | Operations |
|----------|-----------|
| **MEM** | LoadK, LoadV (once per tile), LoadQ, LoaddO (per M-block), TMA stores for dQ |
| **TC** | QK_MMA (K @ Q^T), dP_MMA (V @ dO^T), dV_MMA (pT @ dO), dK_MMA (dsT @ Q), dQ_MMA (dsT^T @ K) |
| **CUDA** | Softmax (exp2, masking), ds computation (pT * (dpT - delta)), scale/convert |
| **SFU** | exp2 for softmax |

The TC pipeline has **5 matrix multiplies per iteration** — far more than forward's 2.

### Pass A, Step 1: Compute MinII

Using approximate Blackwell latencies (128×128 tiles):

```
LoadQ       (TMA 128×128 bf16):        ~640 cycles
LoaddO      (TMA 128×128 bf16):        ~640 cycles
QK_MMA      (K @ Q^T, 128×128×128):   ~900 cycles
dP_MMA      (V @ dO^T, 128×128×128):  ~900 cycles
dV_MMA      (pT @ dO, 128×128×128):   ~900 cycles
dK_MMA      (dsT @ Q, 128×128×128):   ~900 cycles
dQ_MMA      (dsT^T @ K, 128×128×128): ~900 cycles
Softmax     (exp2 + masking):          ~400 cycles
ds_compute  (pT*(dpT-δ), convert):    ~300 cycles
```

**ResMII** (resource-constrained):
```
MEM:  LoadQ(640) + LoaddO(640)                                      = 1280
TC:   QK(900) + dP(900) + dV(900) + dK(900) + dQ(900)              = 4500
CUDA: Softmax(400) + ds(300)                                        = 700
SFU:  exp2 within softmax (included in CUDA estimate above)          ≈ 0 (merged)

ResMII = max(1280, 4500, 700) = 4500  (heavily TC-bound)
```

**RecMII** (recurrence-constrained):
```
dV recurrence: dV[j] → dV_MMA[j+1]
  Distance: 1, latency: 900
  RecMII contribution: 900

dK recurrence: dK[j] → dK_MMA[j+1]
  Distance: 1, latency: 900
  RecMII contribution: 900
```

**MinII:**
```
MinII = max(4500, 900) = 4500  (heavily TC-bound)
```

The backward kernel is **extremely TC-bound** — the tensor core pipeline is 3.5× more loaded than MEM. This drives the key scheduling decisions.

### Pass A, Step 2: Modulo Schedule

With 5 MMA ops and II=4500, the modulo schedule must sequence them on the single TC pipeline. The exact schedule output:

```python
schedule = {
    # op:          (cycle, pipeline)
    # -- Iteration j's ops --
    "LoadQ":       (0,     MEM),
    "LoaddO":      (640,   MEM),
    "QK_MMA":      (0,     TC),      # K @ Q^T, needs Q ready
    "Softmax":     (900,   CUDA),    # exp2(qkT - m), after QK_MMA
    "dQ_MMA":      (900,   TC),      # dsT^T @ K, uses dsT from iter j-1
    "dK_MMA":      (1800,  TC),      # dsT @ Q, uses dsT from iter j-1
    "ds_compute":  (1300,  CUDA),    # pT*(dpT - delta), after softmax + dP
    "dP_MMA":      (2700,  TC),      # V @ dO^T, needs dO ready
    "dV_MMA":      (3600,  TC),      # pT @ dO, needs pT from softmax
}
II = 4500
```

Visualized on the reservation table:

```
Cycle:   0        900      1800     2700     3600    4500 (=II)
         ├────────┼────────┼────────┼────────┼───────┤
TC:      [QK_MMA ][dQ_MMA ][dK_MMA ][dP_MMA ][dV_MMA]
MEM:     [LoadQ  ][LoaddO ]·········(3220 cycles idle)·
CUDA:              [softmax][  ds  ]·························
```

The TC ordering is the critical insight. Notice that **dQ_MMA and dK_MMA (at cycles 900–2700) use dsT from the previous iteration**, while QK_MMA (at cycle 0) and dP_MMA/dV_MMA (at cycles 2700–4500) use the current iteration's data. This cross-iteration interleaving is why the actual TLX code has the prolog/main/epilog structure:

```python
# Prolog:  QK[0], dP[0], dV[0]       — no previous dsT available yet
# Main:    QK[j], dQ[j-1], dK[j-1], dP[j], dV[j]   — 5 MMA ops interleaved
# Epilog:  dK[last], dQ[last]         — drain remaining dsT
```

The schedule dict makes this explicit: `schedule["dQ_MMA"][0]` = 900 and `schedule["dK_MMA"][0]` = 1800 place them *after* `QK_MMA` at cycle 0 but *before* `dP_MMA` at cycle 2700. When Pass C projects this onto the MMA warp group, it directly produces the interleaved order seen in the code.

### Pass A, Step 3: Derive Pipeline Depths

**K, V tiles (SMEM):**
```
K and V are loaded once per outer tile (not per M-block iteration).
They stay in SMEM for all num_steps iterations.
NUM_BUFFERS_KV=1: single-buffered (K and V have separate allocations)
```

**Q tiles (SMEM):**
```
Producer: LoadQ per M-block, latency 640
Consumer: QK_MMA uses Q, dK_MMA uses Q (from previous iteration)
NUM_BUFFERS_Q=2: double-buffered
  → Producer loads Q[j+1] while MMA uses Q[j]
  → Q[j] is also needed for dK_MMA in the next iteration
```

Q requires double-buffering because the same Q block is consumed by two MMA ops across iterations: QK_MMA in iteration j and dK_MMA in iteration j+1.

**dO tiles (SMEM):**
```
NUM_BUFFERS_DO=1: single-buffered
  → dO is consumed by dP_MMA and dV_MMA within the same iteration
```

**QK / P / dP / dQ tiles (TMEM):**
```
NUM_BUFFERS_TMEM=1: single-buffered for all TMEM intermediates
  QK and P share TMEM via reuse=qk_tiles (non-overlapping lifetimes)
  dP and dQ share TMEM via reuse=dp_tiles (when REUSE_DP_FOR_DQ=True)
```

**dK, dV accumulators (TMEM):**
```
NUM_BUFFERS_KV=1: single-buffered accumulators
  dK and dV accumulate across all M-blocks, stored out once per tile
```

### Pass A, Step 4: Memory Budget Check

```
SMEM:
  K tiles:  128 × 128 × 2B × 1 buffer  =  32,768 B
  V tiles:  128 × 128 × 2B × 1 buffer  =  32,768 B
  Q tiles:  128 × 128 × 2B × 2 buffers =  65,536 B
  dO tiles: 128 × 128 × 2B × 1 buffer  =  32,768 B
  ds tiles: 128 × 128 × 2B × 1 buffer  =  32,768 B
  Barriers:                              ~    256 B
  Total SMEM ≈ 196,864 B  (< 232 KB limit ✓)

TMEM:
  qk/p (merged):  128 × 128 × 4B × 1  =  65,536 B
  dp/dq (merged): 128 × 128 × 4B × 1  =  65,536 B  (when REUSE_DP_FOR_DQ)
  dV:             128 × 128 × 4B × 1  =  65,536 B
  dK:             128 × 128 × 4B × 1  =  65,536 B
  Total TMEM = 262,144 B = 256 KB  (just fits ✓)
```

The `REUSE_DP_FOR_DQ` flag is **essential** for the 128×128 config — without it, dP and dQ would each need 64KB, pushing TMEM to 320KB (over the 256KB limit). This is another application of lifetime-aware buffer merging: dP is consumed before dQ is produced within the same iteration.

### Pass B, Step 1: Partition into Warp Groups

Pipeline utilization within II=4500:
```
MEM:  1280/4500 = 28%  → below 30% threshold
TC:   4500/4500 = 100% → dedicated warp group  ✓
CUDA:  700/4500 = 16%  → below 30% threshold
SFU:   merged with CUDA
```

With MEM and CUDA both below 30%, the algorithm merges them. But the actual kernel groups differently based on the dataflow structure:

**Result: 4 warp groups:**

| Warp Group | Role | Operations | Warps | Regs |
|-----------|------|-----------|-------|------|
| Producer | TMA loads | LoadK, LoadV (once), LoadQ, LoaddO (per M-block) | 1 | 88 |
| MMA | All 5 matrix multiplies | QK, dP, dV, dK, dQ MMA ops | 1 | 48 |
| Compute | Softmax + ds + dQ epilogue | exp2, masking, ds=pT*(dpT-δ), convert | 8 | 192 |
| Reduction | dQ atomic add + dK/dV store | TMEM→regs, scale, TMA store/atomic | default | — |

The compute group gets **8 warps and 192 registers** — more than FA forward's softmax group — because it must compute softmax, the ds gradient, and store the transposed ds to SMEM (which the MMA group reads as input for dK and dQ MMA ops).

### Pass B, Step 2: Insert Synchronization

The backward kernel has the most complex barrier structure of all three examples:

| Boundary | Resource | Direction | Barrier Type | Depth |
|----------|----------|-----------|-------------|-------|
| Producer → MMA | K tile in SMEM | data ready | `mbarrier` (`k_fulls`) | 1 |
| MMA → Producer | K consumed (end of tile) | buffer free | `mbarrier` (`k_empties`) | 1 |
| Producer → MMA | V tile in SMEM | data ready | `mbarrier` (`v_fulls`) | 1 |
| Producer → MMA | Q tile in SMEM | data ready | `mbarrier` (`q_fulls`) | 2 |
| MMA → Producer | Q consumed | buffer free | `mbarrier` (`q_empties`) | 2 |
| Producer → MMA | dO tile in SMEM | data ready | `mbarrier` (`do_fulls`) | 1 |
| MMA → Producer | dO consumed | buffer free | `mbarrier` (`do_empties`) | 1 |
| MMA → Compute | QK result in TMEM | data ready | `mbarrier` (`qk_fulls`) | 1 |
| Compute → MMA | QK consumed | buffer free | `mbarrier` (`qk_empties`) | 1 |
| MMA → Compute | dP result in TMEM | data ready | `mbarrier` (`dp_fulls`) | 1 |
| Compute → MMA | dP/dQ consumed | buffer free | `mbarrier` (`dp_empties`/`dq_empties`) | 1 |
| Compute → MMA | P (softmax output) in TMEM | data ready | `mbarrier` (`p_fulls`) | 1 |
| Compute → MMA | ds in SMEM | data ready | `mbarrier` (`ds_fulls`) | 1 |
| MMA → Reduction | dQ result in TMEM | data ready | `mbarrier` (`dq_fulls`) | 1 |
| Reduction → MMA | dQ consumed | buffer free | `mbarrier` (`dq_empties`) | 1 |
| MMA → Compute | dV result in TMEM | data ready | `mbarrier` (`dv_fulls`) | 1 |
| Compute → MMA | dV consumed | buffer free | `mbarrier` (`dv_empties`) | 1 |
| MMA → Compute | dK result in TMEM | data ready | `mbarrier` (`dk_fulls`) | 1 |
| Compute → MMA | dK consumed | buffer free | `mbarrier` (`dk_empties`) | 1 |

The critical circular dependency per iteration is:
```
MMA produces qkT ──→ Compute produces pT and dsT ──→ MMA consumes pT (for dV)
                                                  ──→ MMA consumes dsT (for dK, dQ)
                                                  ──→ Reduction consumes dQ
```

With `NUM_BUFFERS_TMEM=1`, all TMEM intermediates are single-buffered, meaning the compute group must finish processing qkT before the next iteration's QK_MMA can write. The MMA group pipelines around this by interleaving: it computes dQ and dK from the *previous* iteration's dsT while the current iteration's softmax runs.

### Pass B, Step 5: Generated TLX Code

#### Buffer Allocations

```python
# K, V: loaded once per tile, separate SMEM buffers
k_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), dtype, NUM_BUFFERS_KV)    # 1
v_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), dtype, NUM_BUFFERS_KV)    # 1

# Q: double-buffered (consumed across iterations for dK_MMA)
q_tiles = tlx.local_alloc((BLOCK_M1, HEAD_DIM), dtype, NUM_BUFFERS_Q)     # 2

# dO: single-buffered
do_tiles = tlx.local_alloc((BLOCK_M1, HEAD_DIM), dtype, NUM_BUFFERS_DO)   # 1

# ds: gradient of scores, stored in SMEM for MMA to consume
ds_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), dtype, NUM_BUFFERS_DS)   # 1

# QK result in TMEM (reused for P via buffer merging)
qk_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), tl.float32,
                             NUM_BUFFERS_TMEM, tlx.storage_kind.tmem)      # 1
p_tiles  = tlx.local_alloc(..., reuse=qk_tiles)                           # merged

# dP in TMEM (reused for dQ via buffer merging when REUSE_DP_FOR_DQ)
dp_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), tl.float32,
                             NUM_BUFFERS_TMEM, tlx.storage_kind.tmem)      # 1
dq_tiles = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tl.float32,
                             NUM_BUFFERS_TMEM, tlx.storage_kind.tmem,
                             reuse=dp_tiles)                                # merged

# dV, dK accumulators in TMEM
dv_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tl.float32,
                             NUM_BUFFERS_KV, tlx.storage_kind.tmem)        # 1
dk_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tl.float32,
                             NUM_BUFFERS_KV, tlx.storage_kind.tmem)        # 1
```

#### Warp-Specialized Kernel Structure

```python
with tlx.async_tasks():

    # ── Warp Group 1: Reduction (dQ atomic add, dK/dV store) ────
    with tlx.async_task("default"):
        for each tile:
            for each M-block:
                # Wait for dQ from MMA
                tlx.barrier_wait(dq_fulls[buf], phase)
                dq = tlx.local_load(dq_tiles[buf])
                dq = dq * LN2
                desc_dq.atomic_add([offset, 0], dq)   # atomic reduction
                tlx.barrier_arrive(dq_empties[buf])

            # After all M-blocks: store dV and dK
            tlx.barrier_wait(dv_fulls[buf], phase)
            dv = tlx.local_load(dv_tiles[buf])
            desc_dv.store([offset, 0], dv.to(output_dtype))
            tlx.barrier_arrive(dv_empties[buf])

            tlx.barrier_wait(dk_fulls[buf], phase)
            dk = tlx.local_load(dk_tiles[buf])
            dk *= sm_scale
            desc_dk.store([offset, 0], dk.to(output_dtype))
            tlx.barrier_arrive(dk_empties[buf])

    # ── Warp Group 2: Compute (softmax + ds gradient) ──────────
    with tlx.async_task(num_warps=8, registers=192, replicate=1):
        for each tile:
            for each M-block:
                m = tl.load(M + offs_m)          # saved from forward pass

                # Wait for qkT from MMA
                tlx.barrier_wait(qk_fulls[buf], phase)
                qkT = tlx.local_load(qk_tiles[buf])
                tlx.barrier_arrive(qk_empties[buf])

                # Recompute softmax: pT = exp2(qkT - m)
                pT = tl.math.exp2(qkT - m)
                pT = pT.to(input_dtype)
                tlx.local_store(p_tiles[buf], pT)     # for dV_MMA
                tlx.barrier_arrive(p_fulls[buf])

                # Wait for dpT from MMA
                delta = tl.load(D + offs_m)
                tlx.barrier_wait(dp_fulls[buf], phase)
                dpT = tlx.local_load(dp_tiles[buf])
                tlx.barrier_arrive(dp_empties[buf])

                # Compute ds = pT * (dpT - delta)
                dsT = pT * (dpT - delta)
                dsT = dsT.to(input_dtype)
                tlx.local_store(ds_tiles[buf], dsT)    # SMEM for MMA
                tlx.fence("async_shared")
                tlx.barrier_arrive(ds_fulls[buf])

            # Store dV, dK after all M-blocks
            tlx.barrier_wait(dv_fulls[buf], phase)
            dv = tlx.local_load(dv_tiles[buf])
            desc_dv.store(...)
            # ... (similar for dK)

    # ── Warp Group 3: MMA (5 matrix multiplies) ────────────────
    with tlx.async_task(num_warps=1, registers=48):
        for each tile:
            # Wait for K, V (loaded once per tile)
            tlx.barrier_wait(k_fulls[buf], phase)
            tlx.barrier_wait(v_fulls[buf], phase)

            # === Prolog (first M-block): 3 MMA ops ===
            # 1. qkT = K @ Q^T
            tlx.barrier_wait(q_fulls[q_buf], q_phase)
            tlx.barrier_wait(qk_empties[buf], prev_phase)
            qT = tlx.local_trans(q_tiles[q_buf])
            tlx.async_dot(k_tiles[kv_buf], qT, qk_tiles[buf],
                          use_acc=False, mBarriers=[qk_fulls[buf]])

            # 2. dpT = V @ dO^T
            tlx.barrier_wait(do_fulls[do_buf], do_phase)
            tlx.barrier_wait(dp_empties[buf], prev_phase)
            doT = tlx.local_trans(do_tiles[do_buf])
            tlx.async_dot(v_tiles[kv_buf], doT, dp_tiles[buf],
                          use_acc=False, mBarriers=[dp_fulls[buf]])

            # 3. dV += pT @ dO
            tlx.barrier_wait(p_fulls[buf], phase)
            tlx.barrier_wait(dv_empties[kv_buf], prev_phase)
            tlx.async_dot(p_tiles[buf], do_tiles[do_buf], dv_tiles[kv_buf],
                          use_acc=False, mBarriers=[do_empties[do_buf]])

            # === Main loop (M-blocks 1..N-1): 5 MMA ops ===
            for j in range(1, num_steps):
                # 1. qkT = K @ Q^T[j]         (current iteration)
                # 2. dQ = dsT^T @ K            (previous iteration's dsT)
                # 3. dK += dsT @ Q             (previous iteration's dsT)
                # 4. dpT = V @ dO^T[j]         (current iteration)
                # 5. dV += pT @ dO[j]          (current iteration's pT)

            # === Epilog: remaining dK, dQ from last iteration ===
            # dK += dsT @ Q  (last iteration)
            # dQ = dsT^T @ K (last iteration)
            tlx.tcgen05_commit(k_empties[kv_buf])

    # ── Warp Group 4: Producer / TMA Load ──────────────────────
    with tlx.async_task(num_warps=1, registers=88):
        for each tile:
            # Load K (once per tile)
            tlx.barrier_wait(k_empties[kv_buf], prev_phase)
            tlx.barrier_expect_bytes(k_fulls[kv_buf], ...)
            tlx.async_descriptor_load(desc_k, k_tiles[kv_buf], ...)

            # Load Q[0] and dO[0] (first M-block)
            tlx.barrier_wait(q_empties[q_buf], prev_phase)
            tlx.barrier_expect_bytes(q_fulls[q_buf], ...)
            tlx.async_descriptor_load(desc_q, q_tiles[q_buf], ...)

            # Load V (once per tile, no empty barrier needed)
            tlx.barrier_expect_bytes(v_fulls[kv_buf], ...)
            tlx.async_descriptor_load(desc_v, v_tiles[kv_buf], ...)

            tlx.barrier_wait(do_empties[do_buf], prev_phase)
            tlx.barrier_expect_bytes(do_fulls[do_buf], ...)
            tlx.async_descriptor_load(desc_do, do_tiles[do_buf], ...)

            # Load Q[j] and dO[j] for remaining M-blocks
            for j in range(1, num_steps):
                tlx.barrier_wait(q_empties[q_buf], prev_phase)
                tlx.async_descriptor_load(desc_q, q_tiles[q_buf], ...)
                tlx.barrier_wait(do_empties[do_buf], prev_phase)
                tlx.async_descriptor_load(desc_do, do_tiles[do_buf], ...)
```

### Algorithm → TLX Code Mapping Summary

| Algorithm Decision | TLX Code |
|---|---|
| ResMII = 4500 (heavily TC-bound) | 5 MMA ops sequenced on single TC pipeline; MEM 72% idle |
| 5 MMA ops per iteration | MMA group has prolog (3 ops) + main loop (5 ops) + epilog (2 ops) structure |
| Q consumed across iterations | `NUM_BUFFERS_Q=2` — double-buffered so Q[j] available for dK while Q[j+1] loads |
| K, V loaded once per tile | Single-buffered, `k_empties` signaled only at end of tile via `tlx.tcgen05_commit` |
| QK/P merged in TMEM | `p_tiles = tlx.local_alloc(..., reuse=qk_tiles)` — softmax converts in-place |
| dP/dQ merged in TMEM | `dq_tiles = tlx.local_alloc(..., reuse=dp_tiles)` when `REUSE_DP_FOR_DQ=True` |
| ds stored in SMEM (not TMEM) | `ds_tiles` in SMEM because MMA reads it as both `dsT` and `dsT^T` via `local_trans` |
| dQ atomically reduced | `desc_dq.atomic_add(...)` — each M-block contributes a partial dQ |
| Pipelined MMA structure | Iteration j's dK/dQ uses dsT from iteration j-1, overlapping with j's QK/dP |
| 8 warps, 192 regs for compute | Softmax recomputation + ds gradient + SMEM stores need high register pressure |

### GEMM vs FA Forward vs FA Backward: Key Differences

| Aspect | GEMM | FA Forward | FA Backward |
|--------|------|-----------|-------------|
| Active pipelines | 2 (MEM, TC) | 4 (MEM, TC, CUDA, SFU) | 3 (MEM, TC, CUDA) |
| Bottleneck | MEM (1280) | TC (1800) | TC (4500) |
| MMA ops per iteration | 2 | 2 | 5 |
| Warp groups | 3 | 4 | 4 |
| MEM utilization | 100% | 71% | 28% |
| TC utilization | 87% | 100% | 100% |
| Loop-carried state | Accumulator | Acc + m_i + l_i | dK + dV accumulators |
| TMEM merges | None | QK/P/alpha/l/m | QK/P and dP/dQ |
| Q/input loading | Per iteration | Once before loop | Per M-block (double-buffered) |
| Output strategy | Direct store | Direct store | dQ: atomic_add; dK/dV: direct store |
| MMA scheduling | Simple sequential | QK then PV | Prolog/main/epilog with cross-iteration pipelining |
| Compute group | None (GEMM has no softmax) | 4 warps, 152 regs | 8 warps, 192 regs |

---

## Complexity

| Pass | Time Complexity |
|------|----------------|
| MinII computation | O(V + E) for ResMII; O(V * E) for RecMII (cycle detection) |
| Modulo scheduling | O(V^2 * II) worst case with backtracking |
| Pipeline depth derivation | O(V + E) |
| Buffer merging (graph coloring) | O(R^2) where R = number of shared resources |
| Data partitioning | O(V) per split pass |
| WS reconstruction | O(V + E) |
| Global refinement | O(W * V * log V) where W = num warps |

Where V = number of ops, E = number of dependency edges.
