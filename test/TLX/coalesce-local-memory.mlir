
// RUN: triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s

// Test that local_load is unchanged after coalescing pass

// CHECK-LABEL: @local_load_coalesce
// CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<128x64xf16, {{.*}}> -> tensor<128x64xf16, #{{.*}}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func @local_load_coalesce(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem>) -> tensor<128x64xf16, #blocked> {
  %0 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked>
  tt.return %0 : tensor<128x64xf16, #blocked>
}

}
