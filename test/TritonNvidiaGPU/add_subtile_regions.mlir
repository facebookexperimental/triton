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
}
