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
  // The tile body should include the arith.addi from the longer chain.
  // The split result and differing operands must use tile block arguments.
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK: ^bb0(%{{.*}}: tensor<{{.*}}>, %[[DIFF:.*]]: tensor<{{.*}}>, %[[VARY:.*]]: i32, %[[TIDX:.*]]: i32):
  // CHECK:     arith.truncf %[[DIFF]]
  // CHECK:     arith.addi %{{.*}}, %[[VARY]]
  // CHECK:     tt.descriptor_store
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
  // With recursive auxiliary collection, the full bias chain
  // (descriptor_load → extf) is pulled into the tile body. The bias tensor
  // is no longer a tile arg — descriptor_load produces it per tile.
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK: ^bb0(%{{.*}}: tensor<{{.*}}>, %[[SPLIT:.*]]: tensor<{{.*}}>, %[[VARY:.*]]: i32, %[[TIDX:.*]]: i32):
  // CHECK:     ttg.convert_layout %[[SPLIT]]
  // CHECK:     arith.addi %{{.*}}, %[[VARY]]
  // CHECK:     tt.descriptor_load
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

// Test: multi-task addmm epilogue with descriptor_store (no early TMA store
// lowering). The chain crosses 3 task boundaries (load→compute→store).
// Non-contiguous task 2 segments are merged and reordered by dependency,
// producing 3 SubtiledRegionOps: task 3 (bias load), task 2 (compute),
// task 1 (store), with SMEM transitions between them.

#tmem5mt = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#blocked3d5mt = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm5mt = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked_full5mt = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2d5mt = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared5mt = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @multi_task_addmm_descriptor_store
  // Two outer-scope SMEM buffer allocations (bias + output):
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x128xf16
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x128xf16
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x128xf16
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x128xf16
  //
  // First SubtiledRegionOp (task 3): bias descriptor_load + store to SMEM.
  // The addi uses the identity tile arg (%vary: 0 for tile 0, c128 for tile 1)
  // to compute the per-tile column offset, and descriptor_load uses that result.
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK: ^bb0(%{{.*}}: tensor<{{.*}}>, %[[VARY:.*]]: i32, %[[BUF:.*]]: !ttg.memdesc<{{.*}}>, %{{.*}}: i32):
  // CHECK:     %[[OFF:.*]] = arith.addi %{{.*}}, %[[VARY]]
  // CHECK:     %[[BIAS:.*]] = tt.descriptor_load %{{.*}}[%{{.*}}, %[[OFF]]]
  // CHECK:     ttg.local_store %[[BIAS]], %[[BUF]]
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  //
  // Second SubtiledRegionOp (task 2): compute (cvt + extf + addf + truncf)
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK:     ttg.local_load
  // CHECK:     ttg.convert_layout
  // CHECK:     arith.extf
  // CHECK:     arith.addf
  // CHECK:     arith.truncf
  // CHECK:     ttg.local_store
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  //
  // Third SubtiledRegionOp (task 1): descriptor_store from SMEM
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK:     ttg.local_load
  // CHECK:     tt.descriptor_store
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  //
  // CHECK-NOT: tt.split
  tt.func @multi_task_addmm_descriptor_store(
      %tmem_buf: !ttg.memdesc<128x256xf32, #tmem5mt, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token,
      %c_desc: !tt.tensordesc<tensor<128x128xf16, #shared5mt>>,
      %bias_desc: !tt.tensordesc<tensor<128x128xf16, #shared5mt>>,
      %off_m: i32, %off_n: i32, %c128: i32) {
    %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] {async_task_id = array<i32: 2>} : !ttg.memdesc<128x256xf32, #tmem5mt, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked_full5mt>
    %reshaped = tt.reshape %loaded#0 {async_task_id = array<i32: 2>} : tensor<128x256xf32, #blocked_full5mt> -> tensor<128x2x128xf32, #blocked3d5mt>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>, async_task_id = array<i32: 2>} : tensor<128x2x128xf32, #blocked3d5mt> -> tensor<128x128x2xf32, #blocked3d_perm5mt>
    %lhs, %rhs = tt.split %transposed {async_task_id = array<i32: 2>} : tensor<128x128x2xf32, #blocked3d_perm5mt> -> tensor<128x128xf32, #blocked2d5mt>

    // Chain 0 (lhs): cvt{2} → bias_load{3} → extf{2} → addf{2} → truncf{2} → store{1}
    %cvt0 = ttg.convert_layout %lhs {async_task_id = array<i32: 2>} : tensor<128x128xf32, #blocked2d5mt> -> tensor<128x128xf32, #blocked2d5mt>
    %bias0 = tt.descriptor_load %bias_desc[%off_m, %off_n] {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared5mt>> -> tensor<128x128xf16, #blocked2d5mt>
    %bias0_f32 = arith.extf %bias0 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked2d5mt> to tensor<128x128xf32, #blocked2d5mt>
    %acc0 = arith.addf %cvt0, %bias0_f32 {async_task_id = array<i32: 2>} : tensor<128x128xf32, #blocked2d5mt>
    %c0 = arith.truncf %acc0 {async_task_id = array<i32: 2>} : tensor<128x128xf32, #blocked2d5mt> to tensor<128x128xf16, #blocked2d5mt>
    tt.descriptor_store %c_desc[%off_m, %off_n], %c0 {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x128xf16, #shared5mt>>, tensor<128x128xf16, #blocked2d5mt>

    // Chain 1 (rhs): cvt{2} → addi{3} → bias_load{3} → extf{2} → addf{2} → truncf{2} → store{1}
    %cvt1 = ttg.convert_layout %rhs {async_task_id = array<i32: 2>} : tensor<128x128xf32, #blocked2d5mt> -> tensor<128x128xf32, #blocked2d5mt>
    %off_n2 = arith.addi %off_n, %c128 {async_task_id = array<i32: 3>} : i32
    %bias1 = tt.descriptor_load %bias_desc[%off_m, %off_n2] {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared5mt>> -> tensor<128x128xf16, #blocked2d5mt>
    %bias1_f32 = arith.extf %bias1 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked2d5mt> to tensor<128x128xf32, #blocked2d5mt>
    %acc1 = arith.addf %cvt1, %bias1_f32 {async_task_id = array<i32: 2>} : tensor<128x128xf32, #blocked2d5mt>
    %c1 = arith.truncf %acc1 {async_task_id = array<i32: 2>} : tensor<128x128xf32, #blocked2d5mt> to tensor<128x128xf16, #blocked2d5mt>
    tt.descriptor_store %c_desc[%off_m, %off_n2], %c1 {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x128xf16, #shared5mt>>, tensor<128x128xf16, #blocked2d5mt>

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

// -----

// Test: 4-tile subtiling via nested splits.

#tmem7 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#blocked3d7 = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm7 = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked_full7 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2d7 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3d7b = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm7b = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked2d7b = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared7 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @four_tile_nested_split
  // Should produce a single SubtiledRegionOp with 4 tile mappings.
  // CHECK: ttng.subtiled_region
  // CHECK-SAME: tile_mappings = [array<i32: 0,
  // CHECK-SAME: array<i32: 1,
  // CHECK-SAME: array<i32: 2,
  // CHECK-SAME: array<i32: 3,
  // CHECK:   setup {
  // CHECK:     tt.split
  // CHECK:     tt.split
  // CHECK:     tt.split
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   } tile{
  // CHECK:     arith.truncf
  // CHECK:     tt.descriptor_store
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  // CHECK-NOT: tt.split
  tt.func @four_tile_nested_split(
      %tmem_buf: !ttg.memdesc<128x256xf32, #tmem7, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token,
      %c_desc: !tt.tensordesc<tensor<128x64xf16, #shared7>>,
      %off_m: i32, %off_n: i32, %c64: i32, %c128: i32, %c192: i32) {
    %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] : !ttg.memdesc<128x256xf32, #tmem7, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked_full7>
    %reshaped = tt.reshape %loaded#0 : tensor<128x256xf32, #blocked_full7> -> tensor<128x2x128xf32, #blocked3d7>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked3d7> -> tensor<128x128x2xf32, #blocked3d_perm7>
    %lhs, %rhs = tt.split %transposed : tensor<128x128x2xf32, #blocked3d_perm7> -> tensor<128x128xf32, #blocked2d7>

    %lhs_r = tt.reshape %lhs : tensor<128x128xf32, #blocked2d7> -> tensor<128x2x64xf32, #blocked3d7b>
    %lhs_t = tt.trans %lhs_r {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3d7b> -> tensor<128x64x2xf32, #blocked3d_perm7b>
    %acc00, %acc01 = tt.split %lhs_t : tensor<128x64x2xf32, #blocked3d_perm7b> -> tensor<128x64xf32, #blocked2d7b>

    %rhs_r = tt.reshape %rhs : tensor<128x128xf32, #blocked2d7> -> tensor<128x2x64xf32, #blocked3d7b>
    %rhs_t = tt.trans %rhs_r {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3d7b> -> tensor<128x64x2xf32, #blocked3d_perm7b>
    %acc10, %acc11 = tt.split %rhs_t : tensor<128x64x2xf32, #blocked3d_perm7b> -> tensor<128x64xf32, #blocked2d7b>

    %c00 = arith.truncf %acc00 : tensor<128x64xf32, #blocked2d7b> to tensor<128x64xf16, #blocked2d7b>
    tt.descriptor_store %c_desc[%off_m, %off_n], %c00 : !tt.tensordesc<tensor<128x64xf16, #shared7>>, tensor<128x64xf16, #blocked2d7b>

    %c01 = arith.truncf %acc01 : tensor<128x64xf32, #blocked2d7b> to tensor<128x64xf16, #blocked2d7b>
    %off1 = arith.addi %off_n, %c64 : i32
    tt.descriptor_store %c_desc[%off_m, %off1], %c01 : !tt.tensordesc<tensor<128x64xf16, #shared7>>, tensor<128x64xf16, #blocked2d7b>

    %c10 = arith.truncf %acc10 : tensor<128x64xf32, #blocked2d7b> to tensor<128x64xf16, #blocked2d7b>
    %off2 = arith.addi %off_n, %c128 : i32
    tt.descriptor_store %c_desc[%off_m, %off2], %c10 : !tt.tensordesc<tensor<128x64xf16, #shared7>>, tensor<128x64xf16, #blocked2d7b>

    %c11 = arith.truncf %acc11 : tensor<128x64xf32, #blocked2d7b> to tensor<128x64xf16, #blocked2d7b>
    %off3 = arith.addi %off_n, %c192 : i32
    tt.descriptor_store %c_desc[%off_m, %off3], %c11 : !tt.tensordesc<tensor<128x64xf16, #shared7>>, tensor<128x64xf16, #blocked2d7b>

    tt.return
  }
}
