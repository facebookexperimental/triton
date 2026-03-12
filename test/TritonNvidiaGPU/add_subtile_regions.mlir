// RUN: triton-opt %s --nvgpu-test-add-subtile-regions | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 2, 32], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 32, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // The pass should wrap the tmem_load -> split tree -> per-tile ops into
  // a subtiled_region.

  // CHECK-LABEL: @epilogue_subtile
  // CHECK: ttng.subtiled_region
  // CHECK-SAME: barrier_annotations = []
  //
  // Setup region: tmem_load, then split tree (3 levels of reshape/trans/split).
  // CHECK: setup
  // CHECK: ttng.tmem_load
  // CHECK: tt.reshape
  // CHECK: tt.trans
  // CHECK: tt.split
  // CHECK: tt.reshape
  // CHECK: tt.trans
  // CHECK: tt.split
  // CHECK: tt.reshape
  // CHECK: tt.trans
  // CHECK: tt.split
  // CHECK: ttng.subtiled_region_yield
  //
  // Tile region: truncf -> convert_layout -> local_alloc template.
  // CHECK: tile
  // CHECK: arith.truncf
  // CHECK: ttg.convert_layout
  // CHECK: ttg.local_alloc
  // CHECK: ttng.subtiled_region_yield
  //
  // All original per-subtile ops should be erased.
  // CHECK: tt.return
  // CHECK-NOT: arith.truncf
  tt.func @epilogue_subtile(
      %acc_memdesc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %token: !ttg.async.token) {

    %acc, %token2 = ttng.tmem_load %acc_memdesc[%token] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

    // Level 1 split: 128x128 -> 2 x 128x64
    %reshaped = tt.reshape %acc : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
    %outLHS, %outRHS = tt.split %transposed : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5>

    // Level 2 split (LHS): 128x64 -> 2 x 128x32
    %lo_reshaped = tt.reshape %outLHS : tensor<128x64xf32, #blocked5> -> tensor<128x2x32xf32, #blocked6>
    %lo_transposed = tt.trans %lo_reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked6> -> tensor<128x32x2xf32, #blocked7>
    %lo_lhs, %lo_rhs = tt.split %lo_transposed : tensor<128x32x2xf32, #blocked7> -> tensor<128x32xf32, #blocked8>

    // Level 2 split (RHS): 128x64 -> 2 x 128x32
    %hi_reshaped = tt.reshape %outRHS : tensor<128x64xf32, #blocked5> -> tensor<128x2x32xf32, #blocked6>
    %hi_transposed = tt.trans %hi_reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked6> -> tensor<128x32x2xf32, #blocked7>
    %hi_lhs, %hi_rhs = tt.split %hi_transposed : tensor<128x32x2xf32, #blocked7> -> tensor<128x32xf32, #blocked8>

    // Per-subtile ops: truncf -> convert_layout -> local_alloc (x4)
    %c0 = arith.truncf %lo_lhs : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c0_cvt = ttg.convert_layout %c0 : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c0_alloc = ttg.local_alloc %c0_cvt : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>

    %c1 = arith.truncf %lo_rhs : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c1_cvt = ttg.convert_layout %c1 : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c1_alloc = ttg.local_alloc %c1_cvt : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>

    %c2 = arith.truncf %hi_lhs : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c2_cvt = ttg.convert_layout %c2 : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c2_alloc = ttg.local_alloc %c2_cvt : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>

    %c3 = arith.truncf %hi_rhs : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c3_cvt = ttg.convert_layout %c3 : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c3_alloc = ttg.local_alloc %c3_cvt : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>

    tt.return
  }

  // Same split-tree pattern as above, but with consistent async_task_id
  // attributes on all ops. The pass should still create a subtiled_region.

  // CHECK-LABEL: @epilogue_subtile_with_async_task_id
  // CHECK: ttng.subtiled_region
  // CHECK: setup
  // CHECK: ttng.tmem_load
  // CHECK: tile
  // CHECK: arith.truncf
  // CHECK: ttg.convert_layout
  // CHECK: ttg.local_alloc
  tt.func @epilogue_subtile_with_async_task_id(
      %acc_memdesc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %token: !ttg.async.token) {

    %acc, %token2 = ttng.tmem_load %acc_memdesc[%token] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

    // Level 1 split: 128x128 -> 2 x 128x64
    %reshaped = tt.reshape %acc {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>, async_task_id = array<i32: 0>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
    %outLHS, %outRHS = tt.split %transposed {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5>

    // Level 2 split (LHS): 128x64 -> 2 x 128x32
    %lo_reshaped = tt.reshape %outLHS {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked5> -> tensor<128x2x32xf32, #blocked6>
    %lo_transposed = tt.trans %lo_reshaped {order = array<i32: 0, 2, 1>, async_task_id = array<i32: 0>} : tensor<128x2x32xf32, #blocked6> -> tensor<128x32x2xf32, #blocked7>
    %lo_lhs, %lo_rhs = tt.split %lo_transposed {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #blocked7> -> tensor<128x32xf32, #blocked8>

    // Level 2 split (RHS): 128x64 -> 2 x 128x32
    %hi_reshaped = tt.reshape %outRHS {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked5> -> tensor<128x2x32xf32, #blocked6>
    %hi_transposed = tt.trans %hi_reshaped {order = array<i32: 0, 2, 1>, async_task_id = array<i32: 0>} : tensor<128x2x32xf32, #blocked6> -> tensor<128x32x2xf32, #blocked7>
    %hi_lhs, %hi_rhs = tt.split %hi_transposed {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #blocked7> -> tensor<128x32xf32, #blocked8>

    // Per-subtile ops: truncf -> convert_layout -> local_alloc (x4) — all task 0
    %c0 = arith.truncf %lo_lhs {async_task_id = array<i32: 0>} : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c0_cvt = ttg.convert_layout %c0 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c0_alloc = ttg.local_alloc %c0_cvt {async_task_id = array<i32: 0>} : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>

    %c1 = arith.truncf %lo_rhs {async_task_id = array<i32: 0>} : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c1_cvt = ttg.convert_layout %c1 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c1_alloc = ttg.local_alloc %c1_cvt {async_task_id = array<i32: 0>} : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>

    %c2 = arith.truncf %hi_lhs {async_task_id = array<i32: 0>} : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c2_cvt = ttg.convert_layout %c2 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c2_alloc = ttg.local_alloc %c2_cvt {async_task_id = array<i32: 0>} : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>

    %c3 = arith.truncf %hi_rhs {async_task_id = array<i32: 0>} : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c3_cvt = ttg.convert_layout %c3 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c3_alloc = ttg.local_alloc %c3_cvt {async_task_id = array<i32: 0>} : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>

    tt.return
  }

  // Same-task TMA store: split tree + truncf -> cvt -> local_alloc +
  // fence -> tma_copy -> token_wait, all without async_task_id (same task).
  // Should produce a single subtile region with buffer reuse.

  // CHECK-LABEL: @epilogue_subtile_tma_same_task
  // CHECK: ttng.subtiled_region
  //
  // Setup: tmem_load, split tree, and a mutable local_alloc (buffer for reuse).
  // CHECK: setup
  // CHECK: ttng.tmem_load
  // CHECK: tt.reshape
  // CHECK: tt.trans
  // CHECK: tt.split
  // CHECK: tt.reshape
  // CHECK: tt.trans
  // CHECK: tt.split
  // CHECK: tt.reshape
  // CHECK: tt.trans
  // CHECK: tt.split
  // CHECK: ttg.local_alloc
  // CHECK-NOT: %
  // CHECK-SAME: () -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK: ttng.subtiled_region_yield
  //
  // Tile: truncf -> cvt -> local_store -> fence -> tma_copy -> wait.
  // CHECK: tile
  // CHECK: arith.truncf
  // CHECK: ttg.convert_layout
  // CHECK: ttg.local_store
  // CHECK: ttng.fence_async_shared
  // CHECK: ttng.async_tma_copy_local_to_global
  // CHECK: ttng.async_tma_store_token_wait
  // CHECK: ttng.subtiled_region_yield
  //
  // All original per-subtile ops should be erased.
  // CHECK: tt.return
  // CHECK-NOT: arith.truncf
  // CHECK-NOT: ttng.fence_async_shared
  tt.func @epilogue_subtile_tma_same_task(
      %acc_memdesc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %token: !ttg.async.token,
      %desc: !tt.tensordesc<tensor<128x32xf16, #shared1>>,
      %x: i32, %y: i32) {

    %acc, %token2 = ttng.tmem_load %acc_memdesc[%token] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

    // Level 1 split: 128x128 -> 2 x 128x64
    %reshaped = tt.reshape %acc : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
    %outLHS, %outRHS = tt.split %transposed : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5>

    // Level 2 split (LHS): 128x64 -> 2 x 128x32
    %lo_reshaped = tt.reshape %outLHS : tensor<128x64xf32, #blocked5> -> tensor<128x2x32xf32, #blocked6>
    %lo_transposed = tt.trans %lo_reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked6> -> tensor<128x32x2xf32, #blocked7>
    %lo_lhs, %lo_rhs = tt.split %lo_transposed : tensor<128x32x2xf32, #blocked7> -> tensor<128x32xf32, #blocked8>

    // Level 2 split (RHS): 128x64 -> 2 x 128x32
    %hi_reshaped = tt.reshape %outRHS : tensor<128x64xf32, #blocked5> -> tensor<128x2x32xf32, #blocked6>
    %hi_transposed = tt.trans %hi_reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked6> -> tensor<128x32x2xf32, #blocked7>
    %hi_lhs, %hi_rhs = tt.split %hi_transposed : tensor<128x32x2xf32, #blocked7> -> tensor<128x32xf32, #blocked8>

    // Per-subtile: truncf -> cvt -> local_alloc -> fence -> tma_copy -> wait (x4)
    %c0 = arith.truncf %lo_lhs : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c0_cvt = ttg.convert_layout %c0 : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c0_alloc = ttg.local_alloc %c0_cvt : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %c0_tok = ttng.async_tma_copy_local_to_global %desc[%x, %y] %c0_alloc : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %c0_tok : !ttg.async.token

    %c1 = arith.truncf %lo_rhs : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c1_cvt = ttg.convert_layout %c1 : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c1_alloc = ttg.local_alloc %c1_cvt : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %c1_tok = ttng.async_tma_copy_local_to_global %desc[%x, %y] %c1_alloc : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %c1_tok : !ttg.async.token

    %c2 = arith.truncf %hi_lhs : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c2_cvt = ttg.convert_layout %c2 : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c2_alloc = ttg.local_alloc %c2_cvt : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %c2_tok = ttng.async_tma_copy_local_to_global %desc[%x, %y] %c2_alloc : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %c2_tok : !ttg.async.token

    %c3 = arith.truncf %hi_rhs : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c3_cvt = ttg.convert_layout %c3 : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c3_alloc = ttg.local_alloc %c3_cvt : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %c3_tok = ttng.async_tma_copy_local_to_global %desc[%x, %y] %c3_alloc : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %c3_tok : !ttg.async.token

    tt.return
  }

  // Different-task TMA store: epilogue compute ops have task 0,
  // TMA store ops have task 1. Should produce two subtile regions.

  // First subtile region: truncf -> cvt -> local_alloc -> fence (task 0).
  // The fence goes with the store, not with the TMA copy.
  // CHECK-LABEL: @epilogue_subtile_tma_different_task
  // CHECK: ttng.subtiled_region
  // CHECK: setup
  // CHECK: ttng.tmem_load
  // CHECK: tile
  // CHECK: arith.truncf
  // CHECK: ttg.convert_layout
  // CHECK: ttg.local_alloc
  // CHECK: ttng.fence_async_shared
  // CHECK: ttng.subtiled_region_yield
  //
  // Second subtile region: tma_copy -> wait (task 1, no fence).
  // CHECK: ttng.subtiled_region
  // CHECK: setup
  // CHECK: ttng.subtiled_region_yield
  // CHECK: tile
  // CHECK-NOT: ttng.fence_async_shared
  // CHECK: ttng.async_tma_copy_local_to_global
  // CHECK: ttng.async_tma_store_token_wait
  // CHECK: ttng.subtiled_region_yield
  //
  // Original TMA store ops should be erased.
  // CHECK: tt.return
  // CHECK-NOT: ttng.async_tma_copy_local_to_global
  tt.func @epilogue_subtile_tma_different_task(
      %acc_memdesc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %token: !ttg.async.token,
      %desc: !tt.tensordesc<tensor<128x32xf16, #shared1>>,
      %x: i32, %y: i32) {

    %acc, %token2 = ttng.tmem_load %acc_memdesc[%token] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

    // Level 1 split: 128x128 -> 2 x 128x64
    %reshaped = tt.reshape %acc {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>, async_task_id = array<i32: 0>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
    %outLHS, %outRHS = tt.split %transposed {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5>

    // Level 2 split (LHS): 128x64 -> 2 x 128x32
    %lo_reshaped = tt.reshape %outLHS {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked5> -> tensor<128x2x32xf32, #blocked6>
    %lo_transposed = tt.trans %lo_reshaped {order = array<i32: 0, 2, 1>, async_task_id = array<i32: 0>} : tensor<128x2x32xf32, #blocked6> -> tensor<128x32x2xf32, #blocked7>
    %lo_lhs, %lo_rhs = tt.split %lo_transposed {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #blocked7> -> tensor<128x32xf32, #blocked8>

    // Level 2 split (RHS): 128x64 -> 2 x 128x32
    %hi_reshaped = tt.reshape %outRHS {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked5> -> tensor<128x2x32xf32, #blocked6>
    %hi_transposed = tt.trans %hi_reshaped {order = array<i32: 0, 2, 1>, async_task_id = array<i32: 0>} : tensor<128x2x32xf32, #blocked6> -> tensor<128x32x2xf32, #blocked7>
    %hi_lhs, %hi_rhs = tt.split %hi_transposed {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #blocked7> -> tensor<128x32xf32, #blocked8>

    // Per-subtile: truncf -> cvt -> local_alloc (task 0) + fence -> tma_copy -> wait (task 1)
    %c0 = arith.truncf %lo_lhs {async_task_id = array<i32: 0>} : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c0_cvt = ttg.convert_layout %c0 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c0_alloc = ttg.local_alloc %c0_cvt {async_task_id = array<i32: 0>} : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
    ttng.fence_async_shared {async_task_id = array<i32: 1>, bCluster = false}
    %c0_tok = ttng.async_tma_copy_local_to_global %desc[%x, %y] %c0_alloc {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %c0_tok {async_task_id = array<i32: 1>} : !ttg.async.token

    %c1 = arith.truncf %lo_rhs {async_task_id = array<i32: 0>} : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c1_cvt = ttg.convert_layout %c1 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c1_alloc = ttg.local_alloc %c1_cvt {async_task_id = array<i32: 0>} : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
    ttng.fence_async_shared {async_task_id = array<i32: 1>, bCluster = false}
    %c1_tok = ttng.async_tma_copy_local_to_global %desc[%x, %y] %c1_alloc {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %c1_tok {async_task_id = array<i32: 1>} : !ttg.async.token

    %c2 = arith.truncf %hi_lhs {async_task_id = array<i32: 0>} : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c2_cvt = ttg.convert_layout %c2 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c2_alloc = ttg.local_alloc %c2_cvt {async_task_id = array<i32: 0>} : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
    ttng.fence_async_shared {async_task_id = array<i32: 1>, bCluster = false}
    %c2_tok = ttng.async_tma_copy_local_to_global %desc[%x, %y] %c2_alloc {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %c2_tok {async_task_id = array<i32: 1>} : !ttg.async.token

    %c3 = arith.truncf %hi_rhs {async_task_id = array<i32: 0>} : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c3_cvt = ttg.convert_layout %c3 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c3_alloc = ttg.local_alloc %c3_cvt {async_task_id = array<i32: 0>} : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
    ttng.fence_async_shared {async_task_id = array<i32: 1>, bCluster = false}
    %c3_tok = ttng.async_tma_copy_local_to_global %desc[%x, %y] %c3_alloc {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %c3_tok {async_task_id = array<i32: 1>} : !ttg.async.token

    tt.return
  }

  // Same-task TMA store with varying coordinates: each subtile stores to
  // different coordinates. The coordinates should become tile block args.

  // CHECK-LABEL: @epilogue_subtile_tma_varying_coords
  // CHECK: ttng.subtiled_region
  // CHECK: setup
  // CHECK: ttng.tmem_load
  // CHECK: ttg.local_alloc
  // CHECK-NOT: %
  // CHECK-SAME: () -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK: ttng.subtiled_region_yield
  // CHECK: tile
  // CHECK: arith.truncf
  // CHECK: ttg.convert_layout
  // CHECK: ttg.local_store
  // CHECK: ttng.fence_async_shared
  // CHECK: ttng.async_tma_copy_local_to_global
  // CHECK: ttng.async_tma_store_token_wait
  // CHECK: ttng.subtiled_region_yield
  // CHECK: tt.return
  // CHECK-NOT: arith.truncf
  tt.func @epilogue_subtile_tma_varying_coords(
      %acc_memdesc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %token: !ttg.async.token,
      %desc: !tt.tensordesc<tensor<128x32xf16, #shared1>>,
      %x: i32,
      %y0: i32, %y1: i32, %y2: i32, %y3: i32) {

    %acc, %token2 = ttng.tmem_load %acc_memdesc[%token] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

    // Level 1 split: 128x128 -> 2 x 128x64
    %reshaped = tt.reshape %acc : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
    %outLHS, %outRHS = tt.split %transposed : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5>

    // Level 2 split (LHS): 128x64 -> 2 x 128x32
    %lo_reshaped = tt.reshape %outLHS : tensor<128x64xf32, #blocked5> -> tensor<128x2x32xf32, #blocked6>
    %lo_transposed = tt.trans %lo_reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked6> -> tensor<128x32x2xf32, #blocked7>
    %lo_lhs, %lo_rhs = tt.split %lo_transposed : tensor<128x32x2xf32, #blocked7> -> tensor<128x32xf32, #blocked8>

    // Level 2 split (RHS): 128x64 -> 2 x 128x32
    %hi_reshaped = tt.reshape %outRHS : tensor<128x64xf32, #blocked5> -> tensor<128x2x32xf32, #blocked6>
    %hi_transposed = tt.trans %hi_reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked6> -> tensor<128x32x2xf32, #blocked7>
    %hi_lhs, %hi_rhs = tt.split %hi_transposed : tensor<128x32x2xf32, #blocked7> -> tensor<128x32xf32, #blocked8>

    // Per-subtile: truncf -> cvt -> local_alloc -> fence -> tma_copy -> wait (x4)
    // Each subtile has a different y coordinate.
    %c0 = arith.truncf %lo_lhs : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c0_cvt = ttg.convert_layout %c0 : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c0_alloc = ttg.local_alloc %c0_cvt : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %c0_tok = ttng.async_tma_copy_local_to_global %desc[%x, %y0] %c0_alloc : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %c0_tok : !ttg.async.token

    %c1 = arith.truncf %lo_rhs : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c1_cvt = ttg.convert_layout %c1 : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c1_alloc = ttg.local_alloc %c1_cvt : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %c1_tok = ttng.async_tma_copy_local_to_global %desc[%x, %y1] %c1_alloc : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %c1_tok : !ttg.async.token

    %c2 = arith.truncf %hi_lhs : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c2_cvt = ttg.convert_layout %c2 : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c2_alloc = ttg.local_alloc %c2_cvt : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %c2_tok = ttng.async_tma_copy_local_to_global %desc[%x, %y2] %c2_alloc : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %c2_tok : !ttg.async.token

    %c3 = arith.truncf %hi_rhs : tensor<128x32xf32, #blocked8> to tensor<128x32xf16, #blocked8>
    %c3_cvt = ttg.convert_layout %c3 : tensor<128x32xf16, #blocked8> -> tensor<128x32xf16, #blocked9>
    %c3_alloc = ttg.local_alloc %c3_cvt : (tensor<128x32xf16, #blocked9>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %c3_tok = ttng.async_tma_copy_local_to_global %desc[%x, %y3] %c3_alloc : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %c3_tok : !ttg.async.token

    tt.return
  }
}
