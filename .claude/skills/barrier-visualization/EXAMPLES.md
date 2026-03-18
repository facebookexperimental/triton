# Barrier Visualization -- Example Reports

These are example outputs generated from actual AutoWS test IR files.

---

## Example 1: Blackwell GEMM with Merged Barriers

**Source:** `test/Hopper/WarpSpecialization/ws_code_partition_merged_barrier.mlir`
(`@matmul_kernel_tma_persistent`)

This is a Blackwell (cuda:100) persistent GEMM with 3 partitions: MMA, TMA
producer, and epilogue store. Two SMEM buffers share a `buffer.id` so their
barriers are merged.

### Section 1: Partition Summary

| Partition  | Role          | Key Ops                                          | Warps |
|------------|---------------|--------------------------------------------------|-------|
| default    | MMA           | `tc_gen5_mma` (128x64 * 64x256 -> 128x256 TMEM) | 4     |
| partition0 | TMA loads (A, B) | `barrier_expect`, `async_tma_copy_global_to_local` x2 | (assigned by code partition) |
| partition1 | Epilogue store | `tmem_load`, `descriptor_store` x2              | (assigned by code partition) |

**Notes:** This is pre-code-partition IR analyzed via `async_task_id` attributes:
- Task 0 = MMA (`tc_gen5_mma`, `tmem_store`)
- Task 1 = TMA loads (`descriptor_load`, `local_store`)
- Task 2 = Epilogue (`tmem_load`, `descriptor_store`)

### Section 2: Barrier Dependency Graph

```
Barrier Dependency Graph
========================

  partition0 (TMA loads)
      |
      | mbarrier (TMA): barrier_expect 49152 bytes
      |   async_tma_copy_global_to_local x2 (A: 128x64xf16, B: 64x256xf16)
      |   [merged barrier -- single expect for both buffers]
      v
  default (MMA)
      |
      | TMEM token chain: tc_gen5_mma produces %token,
      |   tmem_load consumes %token
      v
  partition1 (Epilogue)
      |
      | (no downstream barrier -- writes to global via descriptor_store)
      v
  [global memory]
```

### Section 3: Index and Phase Analysis

```
Barrier: mbarrier for SMEM buffers A, B (buffer.id = 0, merged)
  Depth: 3 (triple-buffered, buffer.copy = 3)
  Index: managed by code partition (accumCnt % 3)
  Phase: accumCnt / 3 (1-bit)
  Merged expect: 49152 bytes = 128*64*2 (A) + 64*256*2 (B)
  Status: OK -- merged correctly, single barrier_expect prevents over-arrival

Barrier: TMEM accumulator token (buffer.id = 1)
  Depth: 1 (single-buffered, buffer.copy = 1)
  Mechanism: async token chain (%token from tmem_alloc -> tc_gen5_mma -> tmem_load)
  Phase: N/A (token-based, not phase-based)
  Status: OK -- single-buffered is correct for accumulator (reused in-place)
  Note: buffer.copy = 1 means no pipelining of accumulator; this is expected
        since the accumulator is initialized per outer loop iteration via tmem_store
```

**Potential issues:** None detected. Merged barrier byte count (49152) correctly
sums A (128\*64\*2 = 16384) + B (64\*256\*2 = 32768).

### Section 4: Shared Data Description

```
Shared Data Map
===============

Buffer Group: "A tile" (SMEM)
  Storage: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  buffer.id: 0 (merged with B tile)
  Allocation: %1 = ttg.local_alloc {buffer.copy = 3, buffer.id = 0}  (line 45)
  Writer: partition0 -- local_store from descriptor_load %arg0 (A matrix)
  Reader: default -- tc_gen5_mma operand A
  Barrier: mbarrier[buffer.id=0], merged expect=49152

Buffer Group: "B tile" (SMEM)
  Storage: !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
  buffer.id: 0 (merged with A tile)
  Allocation: %0 = ttg.local_alloc {buffer.copy = 3, buffer.id = 0}  (line 44)
  Writer: partition0 -- local_store from descriptor_load %arg5 (B matrix)
  Reader: default -- tc_gen5_mma operand B
  Barrier: mbarrier[buffer.id=0], merged expect=49152

Buffer Group: "Accumulator" (TMEM)
  Storage: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  buffer.id: 1
  Allocation: %result, %token = ttng.tmem_alloc {buffer.copy = 1, buffer.id = 1}  (line 46)
  Writer: default -- tc_gen5_mma accumulates into %result
  Reader: partition1 -- tmem_load %result (after k-loop completes)
  Barrier: TMEM async token chain
```

### Section 5: SSA Value to Barrier Mapping

```
Barrier Alias Map
=================

Logical barrier "SMEM mbarrier" (buffer.id = 0):
  [Created by code partition pass -- not yet present in input IR]
  Will protect:
    %0  = ttg.local_alloc {buffer.copy=3, buffer.id=0}  (line 44)  -- B tile SMEM
    %1  = ttg.local_alloc {buffer.copy=3, buffer.id=0}  (line 45)  -- A tile SMEM
  Writer ops (partition0 / task 1):
    ttg.local_store %44, %1  (line 85)  -- store A tile
    ttg.local_store %45, %0  (line 87)  -- store B tile
  Reader ops (default / task 0):
    ttng.tc_gen5_mma %1, %0, %result  (line 88)  -- MMA reads both

Logical barrier "TMEM token" (buffer.id = 1):
  %token    = ttng.tmem_alloc  (line 46)       -- initial token from allocation
  %23       = ttng.tmem_store %cst, %result[%token]  (line 81)  -- returns new token
  %arg23    = iter_arg in k-loop  (line 82)    -- loop-carried token
  %46       = ttng.tc_gen5_mma ... %result[%arg23]  (line 88)  -- MMA consumes & produces token
  %24#1     = scf.for result  (line 82)        -- final token from k-loop
  ttng.tmem_load %result[%24#1]  (line 102)    -- epilogue consumes final token
```

---

## Example 2: Hopper Matmul with Two Consumers (Legacy Producer/Consumer)

**Source:** `test/Hopper/WarpSpecialization/ws_code_partition.mlir`
(`@matmul_kernel_two_consumers`)

This is a Hopper (cuda:90) matmul where the K-dimension load (B matrix) is
shared between two independent MMA consumers computing separate dot products.

### Section 1: Partition Summary

| Partition  | Role              | Key Ops                                     | Warps |
|------------|-------------------|---------------------------------------------|-------|
| default    | Producer (loads)  | `tt.load` x3, `local_alloc` x3             | 4     |
| partition0 | MMA consumer 1    | `warp_group_dot` (%99 * %104 -> %arg10)     | 4     |
| partition1 | MMA consumer 2    | `warp_group_dot` (%106 * %104 -> %arg11)    | 4     |

**Notes:** Three loads feed two dots. Buffer %104 (B matrix, `64x128xf16`) is
shared between both consumers (`async_task_id = array<i32: 1, 2>`).

### Section 2: Barrier Dependency Graph

```
Barrier Dependency Graph
========================

  default (Producer)
      |
      +--[barrier_A]--> partition0 (MMA consumer 1)
      |   producer_acquire/commit
      |   Data: %99 (A1: 64x64xf16) + %104 (B: 64x128xf16)
      |
      +--[barrier_B]--> partition1 (MMA consumer 2)
      |   producer_acquire/commit
      |   Data: %106 (A2: 64x64xf16) + %104 (B: 64x128xf16, shared)
      |
      v
  partition0 --> tt.store %store_ptr1  (after loop)
  partition1 --> tt.store %store_ptr2  (after loop)
```

**Expected code-partition output** (from CHECK lines):
- default: `producer_acquire` -> `async_copy_global_to_local` -> `producer_commit`
  (repeated for each buffer group)
- partition0: `consumer_wait` x2 -> `warp_group_dot` -> `consumer_release` x2
- partition1: `consumer_wait` x2 -> `warp_group_dot` -> `consumer_release` x2

### Section 3: Index and Phase Analysis

```
Barrier: mbarrier for buffer A1 (%99, 64x64xf16)
  Depth: 1 (num-buffers=1 in test)
  Index: constant 0 (single-buffered)
  Phase: alternates each iteration (iter % 2)
  Consumers: partition0 only

Barrier: mbarrier for buffer B (%104, 64x128xf16, shared)
  Depth: 1 (num-buffers=1)
  Index: constant 0
  Phase: alternates each iteration
  Consumers: partition0 AND partition1
  Note: Two consumer_wait + consumer_release pairs needed (one per consumer)

Barrier: mbarrier for buffer A2 (%106, 64x64xf16)
  Depth: 1 (num-buffers=1)
  Index: constant 0
  Phase: alternates each iteration
  Consumers: partition1 only
```

**Potential issues:**
- `num-buffers=1` means no pipelining overlap between load and compute. This is
  the test configuration; production would use `num-buffers=3` or higher.
- Buffer B is consumed by two partitions -- the code partition must emit separate
  `consumer_wait`/`consumer_release` pairs in each consumer partition. The CHECK
  lines confirm this (2 waits + 2 releases per consumer).

### Section 4: Shared Data Description

```
Shared Data Map
===============

Buffer Group: "A1 tile" (SMEM)
  Storage: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
  Allocation: %99 = ttg.local_alloc %98  (line 119)
  Writer: default -- tt.load %arg12 (input_ptr1)
  Reader: partition0 -- warp_group_dot operand A
  Barrier: producer/consumer mbarrier (1 consumer)
  async_task_id: {1} (consumer 1 only)

Buffer Group: "B tile" (SMEM) -- SHARED between consumers
  Storage: !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory>
  Allocation: %104 = ttg.local_alloc %103  (line 124)
  Writer: default -- tt.load %arg13 (input_ptr2)
  Reader: partition0 -- warp_group_dot operand B
          partition1 -- warp_group_dot operand B
  Barrier: producer/consumer mbarrier (2 consumers)
  async_task_id: {1, 2} (both consumers)

Buffer Group: "A2 tile" (SMEM)
  Storage: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
  Allocation: %106 = ttg.local_alloc %105  (line 126)
  Writer: default -- tt.load %arg14 (input_ptr3)
  Reader: partition1 -- warp_group_dot operand A
  Barrier: producer/consumer mbarrier (1 consumer)
  async_task_id: {2} (consumer 2 only)
```

### Section 5: SSA Value to Barrier Mapping

```
Barrier Alias Map
=================

[Pre-code-partition IR -- barriers not yet materialized]
[Cross-partition data flow identified by async_task_id mismatches:]

Data flow "A1" (task 0 -> task 1):
  %98   = tt.load %arg12, ...  {async_task_id = array<i32: 0>}     (line 118) -- producer
  %99   = ttg.local_alloc %98  {async_task_id = array<i32: 1>}     (line 119) -- consumer alloc
  %107  = ttng.warp_group_dot %99, %104, ...  {async_task_id = array<i32: 1>}  (line 127) -- consumer use
  Will become: producer_acquire/copy/commit in default, consumer_wait/load in partition0

Data flow "B" (task 0 -> tasks 1,2):
  %103  = tt.load %arg13, ...  {async_task_id = array<i32: 0>}     (line 123) -- producer
  %104  = ttg.local_alloc %103 {async_task_id = array<i32: 1, 2>}  (line 124) -- shared alloc
  %107  = ttng.warp_group_dot %99, %104, ... {async_task_id = array<i32: 1>}  (line 127) -- consumer 1
  %108  = ttng.warp_group_dot %106, %104, ... {async_task_id = array<i32: 2>} (line 128) -- consumer 2
  Will become: 2 separate producer_acquire/commit groups, 2 consumer_wait/release in each partition

Data flow "A2" (task 0 -> task 2):
  %105  = tt.load %arg14, ...  {async_task_id = array<i32: 0>}     (line 125) -- producer
  %106  = ttg.local_alloc %105 {async_task_id = array<i32: 2>}     (line 126) -- consumer alloc
  %108  = ttng.warp_group_dot %106, %104, ... {async_task_id = array<i32: 2>} (line 128) -- consumer use
  Will become: producer_acquire/copy/commit in default, consumer_wait/load in partition1
```
