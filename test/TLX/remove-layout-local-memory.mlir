// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s

// Test that redundant layout conversion after local_load is removed

// CHECK: #[[$COALESCED:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: @local_load_coalesce
// CHECK: ttg.local_load %{{.*}} -> tensor<128x64xf16, #[[$COALESCED]]>
// CHECK-NOT: ttg.convert_layout
// CHECK: ttg.local_store

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func @local_load_coalesce(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem>, %arg1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>) {
  %0 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked1>
  %1 = ttg.convert_layout %0 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #blocked>
  ttg.local_store %1, %arg1 : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  tt.return
}
}

// -----

// Test layout conflict resolution when both tmem_load and local_load are in the
// same kernel with different layouts. The pass should prefer TMEM's layout with
// larger sizePerThread ([1, 128], score=128) for better memory access efficiency.
//
// After the pass, the larger layout ([1, 128]) should be selected for both loads,
// eliminating the need for intermediate convert_layout ops.

// CHECK: #[[$TMEM_LAYOUT:.*]] = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK-LABEL: @tmem_and_local_load_conflict_resolution
// Both loads should use the TMEM layout with higher score [1, 128]
// CHECK: ttng.tmem_load %{{.*}} -> tensor<128x128xf32, #[[$TMEM_LAYOUT]]>
// CHECK: ttg.local_load %{{.*}} -> tensor<128x128xbf16, #[[$TMEM_LAYOUT]]>
// The convert_layout to the original common layout should still exist at the end
// CHECK: ttg.convert_layout

#blocked_tmem = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_common = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_smem = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem1 = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
tt.func @tmem_and_local_load_conflict_resolution(
    %tmem_buf: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    %smem_buf: !ttg.memdesc<128x128xbf16, #shared1, #smem1>) -> tensor<128x128xf32, #blocked_common> {
  // TMEM load with large sizePerThread [1, 128], score = 128
  %result = ttng.tmem_load %tmem_buf : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_tmem>
  %result_cvt = ttg.convert_layout %result : tensor<128x128xf32, #blocked_tmem> -> tensor<128x128xf32, #blocked_common>
  // SMEM local_load with small sizePerThread [1, 8], score = 8
  %y = ttg.local_load %smem_buf : !ttg.memdesc<128x128xbf16, #shared1, #smem1> -> tensor<128x128xbf16, #blocked_smem>
  %y_cvt = ttg.convert_layout %y : tensor<128x128xbf16, #blocked_smem> -> tensor<128x128xbf16, #blocked_common>
  // Add them together (requires same layout)
  %y_ext = arith.extf %y_cvt : tensor<128x128xbf16, #blocked_common> to tensor<128x128xf32, #blocked_common>
  %z = arith.addf %result_cvt, %y_ext : tensor<128x128xf32, #blocked_common>
  tt.return %z : tensor<128x128xf32, #blocked_common>
}
}
