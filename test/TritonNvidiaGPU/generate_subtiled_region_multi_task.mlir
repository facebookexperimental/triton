// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-test-generate-subtiled-region | FileCheck %s

// Test: multi-task chain produces two SubtiledRegionOps.
// Compute ops (truncf) have task [3], store ops (async_tma_copy) have task [4].
// The transition is at local_alloc with data (explicit memory store).

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#blocked3d = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked_full = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2d = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @multi_task_with_memory_store
  // Two outer-scope empty SMEM allocations:
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x64xf16
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x64xf16
  //
  // First SubtiledRegionOp: compute + store to SMEM (task [3])
  // CHECK: ttng.subtiled_region
  // CHECK:   setup {
  // CHECK:     ttng.tmem_load
  // CHECK:     tt.reshape
  // CHECK:     tt.trans
  // CHECK:     tt.split
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   } tile{
  // CHECK:     arith.truncf
  // CHECK:     ttg.local_store
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   } teardown {
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  //
  // Second SubtiledRegionOp: TMA copy from SMEM (task [4])
  // CHECK: ttng.subtiled_region
  // CHECK:   setup {
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   } tile{
  // CHECK:     ttng.async_tma_copy_local_to_global
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   } teardown {
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  //
  // Original ops should be erased:
  // CHECK-NOT: tt.split
  // CHECK-NOT: ttg.local_alloc %
  tt.func @multi_task_with_memory_store(
      %tmem_buf: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token,
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %off0: i32, %off1: i32, %off2: i32) {
    %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_full>
    %reshaped = tt.reshape %loaded#0 : tensor<128x128xf32, #blocked_full> -> tensor<128x2x64xf32, #blocked3d>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3d> -> tensor<128x64x2xf32, #blocked3d_perm>
    %lhs, %rhs = tt.split %transposed : tensor<128x64x2xf32, #blocked3d_perm> -> tensor<128x64xf32, #blocked2d>

    // Chain 0 (from lhs): truncf{3} → local_alloc{3} → async_tma_copy{4}
    %trunc0 = arith.truncf %lhs {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked2d> to tensor<128x64xf16, #blocked2d>
    %smem0 = ttg.local_alloc %trunc0 {async_task_id = array<i32: 3>} : (tensor<128x64xf16, #blocked2d>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttng.async_tma_copy_local_to_global %desc[%off0, %off1] %smem0 {async_task_id = array<i32: 4>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    // Chain 1 (from rhs): truncf{3} → local_alloc{3} → async_tma_copy{4}
    %trunc1 = arith.truncf %rhs {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked2d> to tensor<128x64xf16, #blocked2d>
    %smem1 = ttg.local_alloc %trunc1 {async_task_id = array<i32: 3>} : (tensor<128x64xf16, #blocked2d>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttng.async_tma_copy_local_to_global %desc[%off0, %off2] %smem1 {async_task_id = array<i32: 4>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    tt.return
  }
}

// -----

// Test: single-task chain still produces one SubtiledRegionOp (backward compat).

#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#blocked3d2 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm2 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked_full2 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2d2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @single_task_no_split
  // Only one SubtiledRegionOp should be generated:
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK:     arith.truncf
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  // CHECK-NOT: ttng.subtiled_region tile_mappings
  tt.func @single_task_no_split(
      %tmem_buf: !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token) {
    %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] : !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_full2>
    %reshaped = tt.reshape %loaded#0 : tensor<128x128xf32, #blocked_full2> -> tensor<128x2x64xf32, #blocked3d2>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3d2> -> tensor<128x64x2xf32, #blocked3d_perm2>
    %lhs, %rhs = tt.split %transposed : tensor<128x64x2xf32, #blocked3d_perm2> -> tensor<128x64xf32, #blocked2d2>

    %trunc0 = arith.truncf %lhs : tensor<128x64xf32, #blocked2d2> to tensor<128x64xf16, #blocked2d2>
    %trunc1 = arith.truncf %rhs : tensor<128x64xf32, #blocked2d2> to tensor<128x64xf16, #blocked2d2>

    tt.return
  }
}

// -----

// Test: implicit buffer (option 2). No memory store at the transition;
// the pass creates SMEM buffers with local_store + local_load.

#tmem3 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#blocked3d3 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm3 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked_full3 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2d3 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2d3b = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @multi_task_implicit_buffer
  // Two outer-scope SMEM buffer allocations:
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x64xf16
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x64xf16
  //
  // First SubtiledRegionOp: truncf + store to SMEM
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK:     arith.truncf
  // CHECK:     ttg.local_store
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  //
  // Second SubtiledRegionOp: load from SMEM + convert_layout
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK:     ttg.local_load
  // CHECK:     ttg.convert_layout
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  //
  // CHECK-NOT: tt.split
  tt.func @multi_task_implicit_buffer(
      %tmem_buf: !ttg.memdesc<128x128xf32, #tmem3, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token) {
    %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] : !ttg.memdesc<128x128xf32, #tmem3, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_full3>
    %reshaped = tt.reshape %loaded#0 : tensor<128x128xf32, #blocked_full3> -> tensor<128x2x64xf32, #blocked3d3>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3d3> -> tensor<128x64x2xf32, #blocked3d_perm3>
    %lhs, %rhs = tt.split %transposed : tensor<128x64x2xf32, #blocked3d_perm3> -> tensor<128x64xf32, #blocked2d3>

    // Chain 0: truncf{3} → convert_layout{4} (no memory store at boundary)
    %trunc0 = arith.truncf %lhs {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked2d3> to tensor<128x64xf16, #blocked2d3>
    %cvt0 = ttg.convert_layout %trunc0 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked2d3> -> tensor<128x64xf16, #blocked2d3b>

    // Chain 1: truncf{3} → convert_layout{4}
    %trunc1 = arith.truncf %rhs {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked2d3> to tensor<128x64xf16, #blocked2d3>
    %cvt1 = ttg.convert_layout %trunc1 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked2d3> -> tensor<128x64xf16, #blocked2d3b>

    tt.return
  }
}

// -----

// Test: identity insertion. Chain1 has an extra arith.addi for offset
// computation; chain0 uses the base offset directly. The pass inserts a
// virtual identity (arith.addi %base, 0) in chain0's tile to make them
// structurally equivalent.

#tmem4 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#blocked3d4 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm4 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked_full4 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2d4 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @identity_insertion_addi
  // The tile body should include the arith.addi from the longer chain:
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK:     arith.truncf
  // CHECK:     arith.addi
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  tt.func @identity_insertion_addi(
      %tmem_buf: !ttg.memdesc<128x128xf32, #tmem4, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token,
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared4>>,
      %off_row: i32, %off_col: i32, %c64: i32) {
    %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] : !ttg.memdesc<128x128xf32, #tmem4, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_full4>
    %reshaped = tt.reshape %loaded#0 : tensor<128x128xf32, #blocked_full4> -> tensor<128x2x64xf32, #blocked3d4>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3d4> -> tensor<128x64x2xf32, #blocked3d_perm4>
    %lhs, %rhs = tt.split %transposed : tensor<128x64x2xf32, #blocked3d_perm4> -> tensor<128x64xf32, #blocked2d4>

    // Chain 0 (lhs): truncf → store at [off_row, off_col]
    %trunc0 = arith.truncf %lhs : tensor<128x64xf32, #blocked2d4> to tensor<128x64xf16, #blocked2d4>
    tt.descriptor_store %desc[%off_row, %off_col], %trunc0 : !tt.tensordesc<tensor<128x64xf16, #shared4>>, tensor<128x64xf16, #blocked2d4>

    // Chain 1 (rhs): truncf → addi offset → store at [off_row, off_col + 64]
    %trunc1 = arith.truncf %rhs : tensor<128x64xf32, #blocked2d4> to tensor<128x64xf16, #blocked2d4>
    %off_col2 = arith.addi %off_col, %c64 : i32
    tt.descriptor_store %desc[%off_row, %off_col2], %trunc1 : !tt.tensordesc<tensor<128x64xf16, #shared4>>, tensor<128x64xf16, #blocked2d4>

    tt.return
  }
}

// -----

// Test: identity insertion with descriptor_store epilogue (no early TMA store
// lowering). This mirrors the real addmm GEMM epilogue:
//   split → convert_layout → bias_load → extf → addf → truncf → descriptor_store
// Chain1 has an extra arith.addi for the second subtile's column offset.

#tmem5 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#blocked3d5 = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm5 = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked_full5 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2d5 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @identity_descriptor_store_epilogue
  // The tile body should include the full epilogue chain with arith.addi:
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK:     ttg.convert_layout
  // CHECK:     arith.addi
  // CHECK:     arith.extf
  // CHECK:     arith.addf
  // CHECK:     arith.truncf
  // CHECK:     tt.descriptor_store
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  // CHECK-NOT: tt.split
  tt.func @identity_descriptor_store_epilogue(
      %tmem_buf: !ttg.memdesc<128x256xf32, #tmem5, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token,
      %c_desc: !tt.tensordesc<tensor<128x128xf16, #shared5>>,
      %bias_desc: !tt.tensordesc<tensor<128x128xf16, #shared5>>,
      %off_m: i32, %off_n: i32, %c128: i32) {
    %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] : !ttg.memdesc<128x256xf32, #tmem5, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked_full5>
    %reshaped = tt.reshape %loaded#0 : tensor<128x256xf32, #blocked_full5> -> tensor<128x2x128xf32, #blocked3d5>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked3d5> -> tensor<128x128x2xf32, #blocked3d_perm5>
    %lhs, %rhs = tt.split %transposed : tensor<128x128x2xf32, #blocked3d_perm5> -> tensor<128x128xf32, #blocked2d5>

    // Chain 0 (lhs): cvt → bias_load → extf → addf → truncf → store
    %cvt0 = ttg.convert_layout %lhs : tensor<128x128xf32, #blocked2d5> -> tensor<128x128xf32, #blocked2d5>
    %bias0 = tt.descriptor_load %bias_desc[%off_m, %off_n] : !tt.tensordesc<tensor<128x128xf16, #shared5>> -> tensor<128x128xf16, #blocked2d5>
    %bias0_f32 = arith.extf %bias0 : tensor<128x128xf16, #blocked2d5> to tensor<128x128xf32, #blocked2d5>
    %acc0 = arith.addf %cvt0, %bias0_f32 : tensor<128x128xf32, #blocked2d5>
    %c0 = arith.truncf %acc0 : tensor<128x128xf32, #blocked2d5> to tensor<128x128xf16, #blocked2d5>
    tt.descriptor_store %c_desc[%off_m, %off_n], %c0 : !tt.tensordesc<tensor<128x128xf16, #shared5>>, tensor<128x128xf16, #blocked2d5>

    // Chain 1 (rhs): cvt → addi(offset) → bias_load → extf → addf → truncf → store
    %cvt1 = ttg.convert_layout %rhs : tensor<128x128xf32, #blocked2d5> -> tensor<128x128xf32, #blocked2d5>
    %off_n2 = arith.addi %off_n, %c128 : i32
    %bias1 = tt.descriptor_load %bias_desc[%off_m, %off_n2] : !tt.tensordesc<tensor<128x128xf16, #shared5>> -> tensor<128x128xf16, #blocked2d5>
    %bias1_f32 = arith.extf %bias1 : tensor<128x128xf16, #blocked2d5> to tensor<128x128xf32, #blocked2d5>
    %acc1 = arith.addf %cvt1, %bias1_f32 : tensor<128x128xf32, #blocked2d5>
    %c1 = arith.truncf %acc1 : tensor<128x128xf32, #blocked2d5> to tensor<128x128xf16, #blocked2d5>
    tt.descriptor_store %c_desc[%off_m, %off_n2], %c1 : !tt.tensordesc<tensor<128x128xf16, #shared5>>, tensor<128x128xf16, #blocked2d5>

    tt.return
  }
}

// -----

// Test: identity insertion combined with multi-task splitting (early TMA store
// lowering). Chain1 has an extra arith.addi AND the chain crosses partition
// boundaries at local_alloc. This should produce two SubtiledRegionOps:
//   1. compute + local_store (partition 4, uniform)
//   2. async_tma_copy + tma_store_token_wait (partition 3, uniform)

#tmem6 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#blocked3d6 = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm6 = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked_full6 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2d6 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared6 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem6 = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @identity_plus_multi_task_tma_store
  // Two outer-scope empty SMEM allocations:
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x128xf16
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x128xf16
  //
  // First SubtiledRegionOp: compute + store to SMEM (partition 4)
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK:     arith.truncf
  // CHECK:     arith.addi
  // CHECK:     ttg.local_store
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  //
  // Second SubtiledRegionOp: TMA copy + wait (partition 3)
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK:     ttng.async_tma_copy_local_to_global
  // CHECK:     ttng.async_tma_store_token_wait
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  //
  // CHECK-NOT: tt.split
  tt.func @identity_plus_multi_task_tma_store(
      %tmem_buf: !ttg.memdesc<128x256xf32, #tmem6, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token,
      %c_desc: !tt.tensordesc<tensor<128x128xf16, #shared6>>,
      %off_m: i32, %off_n: i32, %c128: i32) {
    %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] : !ttg.memdesc<128x256xf32, #tmem6, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked_full6>
    %reshaped = tt.reshape %loaded#0 : tensor<128x256xf32, #blocked_full6> -> tensor<128x2x128xf32, #blocked3d6>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked3d6> -> tensor<128x128x2xf32, #blocked3d_perm6>
    %lhs, %rhs = tt.split %transposed : tensor<128x128x2xf32, #blocked3d_perm6> -> tensor<128x128xf32, #blocked2d6>

    // Chain 0 (lhs): truncf{4} → local_alloc{4} → async_tma_copy{3} → wait{3}
    %trunc0 = arith.truncf %lhs {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked2d6> to tensor<128x128xf16, #blocked2d6>
    %smem0 = ttg.local_alloc %trunc0 {async_task_id = array<i32: 4>} : (tensor<128x128xf16, #blocked2d6>) -> !ttg.memdesc<128x128xf16, #shared6, #smem6, mutable>
    %tok0 = ttng.async_tma_copy_local_to_global %c_desc[%off_m, %off_n] %smem0 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared6>>, !ttg.memdesc<128x128xf16, #shared6, #smem6, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok0 {async_task_id = array<i32: 3>} : !ttg.async.token

    // Chain 1 (rhs): truncf{4} → addi{4} → local_alloc{4} → async_tma_copy{3} → wait{3}
    %trunc1 = arith.truncf %rhs {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked2d6> to tensor<128x128xf16, #blocked2d6>
    %off_n2 = arith.addi %off_n, %c128 {async_task_id = array<i32: 4>} : i32
    %smem1 = ttg.local_alloc %trunc1 {async_task_id = array<i32: 4>} : (tensor<128x128xf16, #blocked2d6>) -> !ttg.memdesc<128x128xf16, #shared6, #smem6, mutable>
    %tok1 = ttng.async_tma_copy_local_to_global %c_desc[%off_m, %off_n2] %smem1 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared6>>, !ttg.memdesc<128x128xf16, #shared6, #smem6, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok1 {async_task_id = array<i32: 3>} : !ttg.async.token

    tt.return
  }
}
