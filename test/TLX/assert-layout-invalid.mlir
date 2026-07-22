// RUN: not triton-opt --tlx-dump-layout %s 2>&1 | FileCheck %s

#actual = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#expected = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: error: layout assertion failed: LinearLayouts differ
  // CHECK: lhs:
  // CHECK: rhs:
  tt.func public @assert_same_layout_kernel(%arg0: tensor<64xf32, #actual>) {
    tlx.assert_same_layout_expected %arg0 {expected = #expected} : tensor<64xf32, #actual>
    tt.return
  }
}
