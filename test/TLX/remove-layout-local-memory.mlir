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
