// RUN: triton-opt --tlx-dump-layout %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @assert_same_layout_kernel
  // CHECK-NOT: tlx.assert_same_layout
  tt.func public @assert_same_layout_kernel(%arg0: tensor<64xf32, #blocked>, %arg1: tensor<64xf32, #blocked>) {
    tlx.assert_same_layout %arg0, %arg1 : tensor<64xf32, #blocked>, tensor<64xf32, #blocked>
    tlx.assert_same_layout_expected %arg0 {expected = #blocked} : tensor<64xf32, #blocked>
    tt.return
  }
}
