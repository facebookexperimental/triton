// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-alloc-algo=1 smem-budget=196608" | FileCheck %s --check-prefix=RESERVED
// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-alloc-algo=1 smem-budget=196608 reserve-auxiliary-smem=0" | FileCheck %s --check-prefix=UNRESERVED
// RUN: not triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-alloc-algo=1 smem-budget=1" 2>&1 | FileCheck %s --check-prefix=EXHAUSTED
// RUN: env TRITON_WS_MEM_PLAN_TOPK=3 TRITON_WS_MEM_PLAN_PICK=0 triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=163840 reserve-auxiliary-smem=0 smem-plan-search" | FileCheck %s --check-prefix=SEARCH-PICK0
// RUN: env TRITON_WS_MEM_PLAN_TOPK=3 TRITON_WS_MEM_PLAN_PICK=1 triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=163840 reserve-auxiliary-smem=0 smem-plan-search" | FileCheck %s --check-prefix=SEARCH-PICK1
// RUN: env TRITON_WS_MEM_PLAN_TOPK=3 TRITON_WS_MEM_PLAN_PICK=2 triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=163840 reserve-auxiliary-smem=0 smem-plan-search" | FileCheck %s --check-prefix=SEARCH-PICK2
// RUN: env TRITON_WS_MEM_PLAN_TOPK=3 TRITON_WS_MEM_PLAN_PICK=0 triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=180224 reserve-auxiliary-smem=0 smem-plan-search" | FileCheck %s --check-prefix=FIXED-PICK0
// RUN: env TRITON_WS_MEM_PLAN_TOPK=3 TRITON_WS_MEM_PLAN_PICK=1 triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=180224 reserve-auxiliary-smem=0 smem-plan-search" | FileCheck %s --check-prefix=FIXED-PICK1
// RUN: env TRITON_WS_MEM_PLAN_TOPK=3 TRITON_WS_MEM_PLAN_PICK=2 triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=180224 reserve-auxiliary-smem=0 smem-plan-search" | FileCheck %s --check-prefix=FIXED-PICK2

// EXHAUSTED: error: estimated auxiliary shared-memory allocation ({{[0-9]+}} bytes) exhausts the shared-memory budget (1 bytes)

// Test: Phase 4 chooses equal-priority multi-buffer candidates in producer
// usage order, not local_alloc order. W is allocated first, but X's TMA load is
// issued first. With only enough SMEM for one extra 128x256xf16 copy, X should
// get buffer.copy = 2 and W should stay at buffer.copy = 1.

// UNRESERVED-LABEL: @load_order_breaks_multibuffer_tie
// UNRESERVED: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = {{[0-9]+}} : i32}
// UNRESERVED-SAME: !ttg.memdesc<128x256xf16
// UNRESERVED: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = {{[0-9]+}} : i32}
// UNRESERVED-SAME: !ttg.memdesc<128x256xf16

// RESERVED-LABEL: @load_order_breaks_multibuffer_tie
// RESERVED: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = {{[0-9]+}} : i32}
// RESERVED-SAME: !ttg.memdesc<128x256xf16

// The search fixture has two independent 32-KiB TMA operand blocks. Each is
// consumed at stages 0 and 1, so A2/B2 is the hard floor (128 KiB). The 160-KiB
// budget admits exactly one additional copy. TOPK=3 must retain both asymmetric
// depth plans plus the floor plan, without operand-depth annotations.
// SEARCH-PICK0-LABEL: @copy_depth_frontier
// SEARCH-PICK0: ttg.local_alloc {buffer.copy = 2 : i32
// SEARCH-PICK0-SAME: !ttg.memdesc<64x256xf16
// SEARCH-PICK0: ttg.local_alloc {buffer.copy = 2 : i32
// SEARCH-PICK0-SAME: !ttg.memdesc<64x256xf16
// SEARCH-PICK1-LABEL: @copy_depth_frontier
// SEARCH-PICK1: ttg.local_alloc {buffer.copy = 3 : i32
// SEARCH-PICK1-SAME: !ttg.memdesc<64x256xf16
// SEARCH-PICK1: ttg.local_alloc {buffer.copy = 2 : i32
// SEARCH-PICK1-SAME: !ttg.memdesc<64x256xf16
// SEARCH-PICK2-LABEL: @copy_depth_frontier
// SEARCH-PICK2: ttg.local_alloc {buffer.copy = 2 : i32
// SEARCH-PICK2-SAME: !ttg.memdesc<64x256xf16
// SEARCH-PICK2: ttg.local_alloc {buffer.copy = 3 : i32
// SEARCH-PICK2-SAME: !ttg.memdesc<64x256xf16

// Fixed-group rank zero is the heuristic allocation. The staging ring remains
// pinned to one copy in every rank; only the singleton A/B operand depths vary.
// FIXED-PICK0-LABEL: @fixed_group_copy_frontier
// FIXED-PICK0: ttg.local_alloc {buffer.copy = 3 : i32
// FIXED-PICK0-SAME: !ttg.memdesc<64x256xf16
// FIXED-PICK0: ttg.local_alloc {buffer.copy = 2 : i32
// FIXED-PICK0-SAME: !ttg.memdesc<64x256xf16
// FIXED-PICK0: ttg.local_alloc {buffer.copy = 1 : i32
// FIXED-PICK0-SAME: buffer.tmaStaging = 1 : i32
// FIXED-PICK1-LABEL: @fixed_group_copy_frontier
// FIXED-PICK1: ttg.local_alloc {buffer.copy = 2 : i32
// FIXED-PICK1-SAME: !ttg.memdesc<64x256xf16
// FIXED-PICK1: ttg.local_alloc {buffer.copy = 2 : i32
// FIXED-PICK1-SAME: !ttg.memdesc<64x256xf16
// FIXED-PICK1: ttg.local_alloc {buffer.copy = 1 : i32
// FIXED-PICK1-SAME: buffer.tmaStaging = 1 : i32
// FIXED-PICK2-LABEL: @fixed_group_copy_frontier
// FIXED-PICK2: ttg.local_alloc {buffer.copy = 2 : i32
// FIXED-PICK2-SAME: !ttg.memdesc<64x256xf16
// FIXED-PICK2: ttg.local_alloc {buffer.copy = 3 : i32
// FIXED-PICK2-SAME: !ttg.memdesc<64x256xf16
// FIXED-PICK2: ttg.local_alloc {buffer.copy = 1 : i32
// FIXED-PICK2-SAME: buffer.tmaStaging = 1 : i32
// RESERVED: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = {{[0-9]+}} : i32}
// RESERVED-SAME: !ttg.memdesc<128x256xf16

#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @load_order_breaks_multibuffer_tie(
      %x_desc: !tt.tensordesc<128x256xf16, #shared>,
      %w_desc: !tt.tensordesc<128x256xf16, #shared>) {
    %W_smem = ttg.local_alloc : () -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
    %X_smem = ttg.local_alloc : () -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : i32
    %c10 = arith.constant {async_task_id = array<i32: 0, 1>} 10 : i32
    scf.for %iv = %c0 to %c10 step %c1 : i32 {
      %x = tt.descriptor_load %x_desc[%c0, %c0] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x256xf16, #shared> -> tensor<128x256xf16, #blocked1>
      ttg.local_store %x, %X_smem {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x256xf16, #blocked1> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
      %w = tt.descriptor_load %w_desc[%c0, %c0] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x256xf16, #shared> -> tensor<128x256xf16, #blocked1>
      ttg.local_store %w, %W_smem {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x256xf16, #blocked1> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
      %x_val = ttg.local_load %X_smem {async_task_id = array<i32: 0>} : !ttg.memdesc<128x256xf16, #shared, #smem, mutable> -> tensor<128x256xf16, #blocked1>
      %w_val = ttg.local_load %W_smem {async_task_id = array<i32: 0>} : !ttg.memdesc<128x256xf16, #shared, #smem, mutable> -> tensor<128x256xf16, #blocked1>
      %sum = arith.addf %x_val, %w_val {async_task_id = array<i32: 0>} : tensor<128x256xf16, #blocked1>
      scf.yield {async_task_id = array<i32: 0, 1>}
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize}
    tt.return
  }

  tt.func public @copy_depth_frontier(
      %a_desc: !tt.tensordesc<64x256xf16, #shared>,
      %b_desc: !tt.tensordesc<64x256xf16, #shared>) {
    %A_smem = ttg.local_alloc : () -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
    %B_smem = ttg.local_alloc : () -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : i32
    %c10 = arith.constant {async_task_id = array<i32: 0, 1>} 10 : i32
    scf.for %iv = %c0 to %c10 step %c1 : i32 {
      %a = tt.descriptor_load %a_desc[%c0, %c0] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
      ttg.local_store %a, %A_smem {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x256xf16, #blocked1> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
      %b = tt.descriptor_load %b_desc[%c0, %c0] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
      ttg.local_store %b, %B_smem {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x256xf16, #blocked1> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
      %a0 = ttg.local_load %A_smem {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x256xf16, #shared, #smem, mutable> -> tensor<64x256xf16, #blocked1>
      %b0 = ttg.local_load %B_smem {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x256xf16, #shared, #smem, mutable> -> tensor<64x256xf16, #blocked1>
      %a1 = ttg.local_load %A_smem {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x256xf16, #shared, #smem, mutable> -> tensor<64x256xf16, #blocked1>
      %b1 = ttg.local_load %B_smem {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x256xf16, #shared, #smem, mutable> -> tensor<64x256xf16, #blocked1>
      %sum0 = arith.addf %a0, %b0 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x256xf16, #blocked1>
      %sum1 = arith.addf %a1, %b1 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x256xf16, #blocked1>
      %sum = arith.addf %sum0, %sum1 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x256xf16, #blocked1>
      scf.yield {async_task_id = array<i32: 0, 1>}
    } {async_task_id = array<i32: 0, 1>, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
    tt.return
  }

  // A multi-store staging ring makes this function take the fixed-grouping
  // path. The staging depth must remain exactly as chosen by the heuristic,
  // while the two independent operand rings remain searchable.
  tt.func public @fixed_group_copy_frontier(
      %a_desc: !tt.tensordesc<64x256xf16, #shared>,
      %b_desc: !tt.tensordesc<64x256xf16, #shared>,
      %out_desc: !tt.tensordesc<64x64xf16, #shared>,
      %out: tensor<64x64xf16, #blocked1>) {
    %A_smem = ttg.local_alloc : () -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
    %B_smem = ttg.local_alloc : () -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
    %staging = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : i32
    %c10 = arith.constant {async_task_id = array<i32: 0, 1>} 10 : i32
    ttg.local_store %out, %staging {async_task_id = array<i32: 0>} : tensor<64x64xf16, #blocked1> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %store0 = ttng.async_tma_copy_local_to_global %out_desc[%c0, %c0] %staging {async_task_id = array<i32: 0>} : !tt.tensordesc<64x64xf16, #shared>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %store1 = ttng.async_tma_copy_local_to_global %out_desc[%c0, %c0] %staging {async_task_id = array<i32: 0>} : !tt.tensordesc<64x64xf16, #shared>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %store0 {async_task_id = array<i32: 0>} : !ttg.async.token
    ttng.async_tma_store_token_wait %store1 {async_task_id = array<i32: 0>} : !ttg.async.token
    scf.for %iv = %c0 to %c10 step %c1 : i32 {
      %a = tt.descriptor_load %a_desc[%c0, %c0] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
      ttg.local_store %a, %A_smem {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x256xf16, #blocked1> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
      %b = tt.descriptor_load %b_desc[%c0, %c0] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
      ttg.local_store %b, %B_smem {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x256xf16, #blocked1> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
      %a0 = ttg.local_load %A_smem {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x256xf16, #shared, #smem, mutable> -> tensor<64x256xf16, #blocked1>
      %b0 = ttg.local_load %B_smem {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x256xf16, #shared, #smem, mutable> -> tensor<64x256xf16, #blocked1>
      %a1 = ttg.local_load %A_smem {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x256xf16, #shared, #smem, mutable> -> tensor<64x256xf16, #blocked1>
      %b1 = ttg.local_load %B_smem {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x256xf16, #shared, #smem, mutable> -> tensor<64x256xf16, #blocked1>
      %sum0 = arith.addf %a0, %b0 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x256xf16, #blocked1>
      %sum1 = arith.addf %a1, %b1 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x256xf16, #blocked1>
      %sum = arith.addf %sum0, %sum1 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x256xf16, #blocked1>
      scf.yield {async_task_id = array<i32: 0, 1>}
    } {async_task_id = array<i32: 0, 1>, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
    tt.return
  }
}
