---
name: barrier-visualization
description: >
  Produce a structured barrier report for AutoWS (automatic warp specialization) IR.
  Use when the user wants to visualize, audit, or debug barrier usage across
  warp-specialized partitions. Covers mbarriers, named barriers, tcgen05 commit,
  TMA-implicit arrives, Aref-based synchronization, and producer/consumer
  barrier patterns.
---

# Barrier Visualization Report

When the user asks for a barrier visualization report, produce a structured
analysis of barrier usage in the given IR (either from a file, an IR dump, or
from running a compilation with `MLIR_ENABLE_DUMP`). The report has five
sections. Use the IR directly as input -- read the file or dump and analyze it.

## Report Format

### Section 1: Partition Summary

Label each partition by its **key ops** -- the operations that differentiate it.
Use short descriptive names. When multiple partitions contain similar ops, add
qualifying detail.

Format as a table:

```
| Partition   | Role             | Key Ops                        | Warps |
|-------------|------------------|--------------------------------|-------|
| default     | Acc correction   | tmem_load, tmem_store          | 4     |
| partition0  | MMA              | tc_gen5_mma x2                 | 4     |
| partition1  | TMA loads (Q,K,V)| async_tma_copy_global_to_local | 1     |
| partition2  | Output store     | descriptor_store               | 1     |
| partition3  | Softmax (QK_1)   | tmem_load, exp2, reduce        | 2     |
```

How to identify key ops:
- **MMA partition**: contains `tt.dot`, `warp_group_dot`, `tc_gen5_mma`, or `tc_gen5_mma_scaled`
- **TMA load partition**: contains `async_tma_copy_global_to_local` or `descriptor_load` feeding `local_alloc`
- **Store/epilogue partition**: contains `descriptor_store`, `tt.store`, `tmem_load` at loop exit
- **Softmax/reduction partition**: contains `tt.reduce`, `math.exp2`, `arith.maxf`
- **Accumulator correction**: contains `tmem_load` + `tmem_store` (re-scaling accumulators)

When two partitions both do TMA loads, differentiate by what they load:
- "TMA load (Q, K)" vs "TMA load (V, scales)"
- Use loc metadata or tensor shapes to identify operand names when available

### Section 2: Barrier Dependency Graph

Draw an ASCII diagram showing which partitions produce/consume through each
barrier. Use arrows to show data flow direction.

```
Barrier Dependency Graph
========================

  Forward barriers:

  partition1 (TMA loads)
      |
      | barrier_expect + async_tma_copy (mbarrier, SMEM buffers A, B)
      v
  partition0 (MMA)
      |
      | tc_gen5_commit (mbarrier on TMEM result)
      v
  partition3/4 (Softmax)
      |
      | aref.put / aref.get  (SMEM buffer for P)
      v
  partition0 (MMA, 2nd use)
      |
      | tc_gen5_commit
      v
  partition2 (Output store)

  Backwards barriers (next-iteration dependencies):

  partition2 (Output store)
      |
      | TMEM token (backward): tmem_load token → next iter's tmem_store
      v
  partition0 (MMA, next iteration)

  partition0 (MMA)
      |
      | mbarrier phase (backward, implicit): phase tracking prevents
      |   TMA re-arrival until MMA has consumed the buffer
      v
  partition1 (TMA loads, next iteration)
```

For each arrow, annotate:
- The barrier mechanism type (see table below)
- What data flows across (buffer name or tensor shape)
- The direction: **forward** (producer → consumer) or **backward** (consumer →
  producer, signaling resource reuse)

#### Backwards-Direction Barriers

In persistent kernels (those with an outer tile loop), downstream partitions
often need to signal upstream partitions that shared resources can be reused.
These "backwards" barriers create cycles in the dependency graph.

Common backwards barriers:
- **TMEM token chain**: `tmem_load` (epilogue) produces a token consumed by
  `tmem_store` (MMA) in the next iteration — prevents zeroing the accumulator
  before the epilogue finishes reading it.
- **consumer_release** (legacy WS): Consumer releases the mbarrier slot,
  allowing the producer to re-acquire it for the next iteration.
- **Phase-based mbarrier**: Multi-buffered SMEM implicitly handles backwards
  sync — the producer can't re-arrive on a slot until the consumer has waited
  on it (phase flip).

Show backwards barriers as upward arrows or annotated return edges in the
dependency graph. When a backwards token chain is expected but the SSA token
is unused (not loop-carried), flag it as a potential issue.

#### Barrier Mechanism Types

| Mechanism | Arrive Side | Wait Side | Notes |
|-----------|------------|-----------|-------|
| **mbarrier (TMA)** | `async_tma_copy_global_to_local` (implicit arrive) | `wait_barrier` with phase | TMA HW auto-arrives on mbarrier after copy completes. `barrier_expect` sets expected byte count. |
| **mbarrier (explicit)** | `arrive_barrier` | `wait_barrier` | Thread-side explicit arrive with count. |
| **tcgen05 commit** | `tc_gen5_commit` on barrier | `wait_barrier` | Tracks completion of prior async tcgen5 ops (MMA, tmem_copy). Arrive count = 1. Sequential ordering between commits. |
| **tc_gen5_mma barrier arg** | `tc_gen5_mma ... barriers(%bar)` | `wait_barrier` | MMA op directly arrives on given barrier(s) upon completion. |
| **Named barrier** | `arrive_barrier_named` | `wait_barrier_named` | HW barrier (index 0-15), no SMEM. Used for intra-CTA sync between warp groups. |
| **Producer/Consumer (legacy)** | `producer_acquire` + `producer_commit` | `consumer_wait` + `consumer_release` | Legacy Hopper WS. Producer acquires mbarrier slot, does copies, commits. Consumer waits then releases. |
| **Aref (new pipeline)** | `aref.put.enter` / `aref.put.exit` | `aref.get.enter` / `aref.get.exit` | Cross-partition SSA deps rewritten to SMEM multibuffers. Handles sync internally. `async_ops` attr on exit specifies what async ops to wait on. |
| **async_copy_mbarrier_arrive** | `async_copy_mbarrier_arrive` | `wait_barrier` | Arrives on mbarrier after all prior `cp.async` copies complete. |

### Section 3: Index and Phase Analysis

For each barrier instance, describe:
- **Buffer depth** (number of multibuffer slots, from `buffer.copy` attr or memdesc shape dim 0)
- **Index computation** (how the buffer/barrier slot index is derived -- typically `iteration % num_buffers`)
- **Phase tracking** (how the phase bit flips -- typically `iteration / num_buffers`)
- **Stagger offsets** (for data-partitioned barriers sharing `buffer.id`, each operand gets a different offset: `(accumCnt + offset) % num_buffers`)

Example:

```
Barrier: mbarrier for SMEM buffers A, B (buffer.id = 0, merged)
  Depth: 3 (triple-buffered)
  Index: accumCnt % 3
  Phase: accumCnt / 3 (1-bit: flips every 3 iterations)
  Merged: barrier_expect size = 49152 (128*64*2 + 64*256*2)

Barrier: mbarrier for data-partitioned operands a0, a1, b (buffer.id = 2)
  Depth: 3
  Index (a0): (accumCnt + 1) % 3
  Index (a1): (accumCnt + 2) % 3
  Index (b):  accumCnt % 3
  Phase: same for all, accumCnt / 3
```

Flag potential issues:
- Mismatched arrive/wait counts
- Missing phase tracking
- Barriers with `buffer.copy` = 1 (no pipelining)
- Merged barriers where byte counts don't match tensor sizes

### Section 4: Shared Data Description

For each barrier, describe what logical data it protects and which partitions
share it. Group by logical purpose.

```
Shared Data Map
===============

Buffer Group: "K tile" (SMEM)
  Storage: !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
  buffer.id: 0 (merged with V tile)
  Writer: partition1 (TMA load)
  Reader: partition0 (MMA operand A)
  Barrier: mbarrier[buffer.id=0], merged expect=49152

Buffer Group: "V tile" (SMEM)
  Storage: !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable>
  buffer.id: 0 (merged with K tile)
  Writer: partition1 (TMA load)
  Reader: partition0 (MMA operand B)
  Barrier: mbarrier[buffer.id=0], merged expect=49152

Buffer Group: "QK accumulator" (TMEM)
  Storage: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  buffer.id: 1
  Writer: partition0 (MMA result)
  Reader: partition3 (softmax tmem_load)
  Barrier: tc_gen5_commit

Buffer Group: "P matrix" (Aref)
  Storage: !ttg.memdesc<1x128x128xf16, #shared, #smem>
  Writer: partition3 (softmax output, via aref.put)
  Reader: partition0 (MMA 2nd operand, via aref.get)
  Barrier: Aref-internal sync
```

Note when:
- Multiple logical buffers share the same `buffer.id` (merged barriers)
- Data aliases exist (same physical storage, different views)
- TMEM vs SMEM vs register data flows

### Section 5: SSA Value to Barrier Mapping

List all SSA values that refer to the same logical barrier, tracing through
block arguments, iter_args, and aliases.

```
Barrier Alias Map
=================

Logical barrier "mbarrier_0" (buffer.id = 0):
  %bar_alloc   = ttg.local_alloc  (line 12)    -- allocation
  %arg35       = block argument   (line 45)     -- passed into loop body
  %bar_idx     = ttg.memdesc_index %arg35[%idx] -- indexed for iteration
  Used in:
    barrier_expect %bar_idx, 49152  (partition1, line 82)
    async_tma_copy ... %bar_idx     (partition1, line 84)
    wait_barrier %bar_idx, %phase   (partition0, line 67)

Logical barrier "named_bar_1":
  %c1 = arith.constant 1 : i32
  Used in:
    arrive_barrier_named %c1, 128  (default, line 50)
    wait_barrier_named %c1, 128    (partition0, line 55)
```

Include:
- The allocation site (local_alloc, or constant for named barriers)
- All aliases through block args, loop iter_args, memdesc_index, memdesc_subview
- Every use site with partition and line number
- For Arefs: the aref.create site and all enter/exit pairs

## How to Generate the Report

1. **Read the IR** from the file or dump the user provides.
2. **Identify all `ttg.warp_specialize` ops** -- these define the partition structure.
3. **Scan each partition region** for barrier-related ops (see mechanism table above).
4. **Trace SSA values** backward from barrier ops to their allocation sites.
   Follow block arguments and iter_args chains.
5. **Identify buffer.id attributes** on `local_alloc` and `tmem_alloc` ops to
   group related barriers.
6. **Check for merged barriers** -- multiple buffers sharing the same `buffer.id`
   with a single `barrier_expect` whose size is the sum of individual buffer sizes.
7. **Look for loc metadata** (e.g., `loc("a_desc")`, `loc("K")`) to name buffers.
8. **Check async_task_id attributes** on ops to determine partition membership
   when analyzing pre-code-partition IR.
9. **Identify backwards-direction barriers** in persistent kernels (outer tile
   loops). Check whether downstream partitions produce tokens or release barriers
   that upstream partitions consume in the next iteration:
   - TMEM: Does `tmem_load`'s output token feed back (via iter_arg) to the next
     iteration's `tmem_store`? If not, flag as a potential missing backward sync.
   - SMEM mbarrier: Is the buffer multi-buffered (depth > 1) with phase tracking?
     If so, backwards sync is implicit. If single-buffered, check for explicit
     backward barriers.
   - Legacy WS: Does `consumer_release` pair with the next `producer_acquire`?

## Example Reports

See `EXAMPLES.md` in this skill directory for two fully worked example reports:
1. **Blackwell GEMM with merged barriers** -- `@matmul_kernel_tma_persistent` from
   `ws_code_partition_merged_barrier.mlir`. Demonstrates merged `buffer.id`,
   TMEM token chains, and `tc_gen5_mma` barrier patterns.
2. **Hopper matmul with two consumers** -- `@matmul_kernel_two_consumers` from
   `ws_code_partition.mlir`. Demonstrates legacy producer/consumer barriers,
   shared SMEM buffers consumed by multiple partitions, and pre-code-partition
   `async_task_id` analysis.

## Reference Files

- Barrier op definitions: `include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td`
- NVWS Aref ops: `third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td`
- Code partition (legacy): `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSCodePartition.cpp`
- Code partition (new): `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/`
- Test IR examples:
  - `test/Hopper/WarpSpecialization/ws_code_partition.mlir` -- basic producer/consumer
  - `test/Hopper/WarpSpecialization/ws_code_partition_merged_barrier.mlir` -- merged barriers
  - `test/Hopper/WarpSpecialization/ws_code_partition_data_partition_barriers.mlir` -- staggered indices
  - `test/Hopper/WarpSpecialization/blackwell_fa_code_partition.mlir` -- complex multi-partition FA
  - `test/TritonGPU/rewrite-partition-dependencies.mlir` -- Aref-based barriers
