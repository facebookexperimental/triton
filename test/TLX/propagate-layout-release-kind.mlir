// RUN: triton-opt --tlx-propagate-layout %s | FileCheck %s

#src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @release_boundary
  // CHECK: ttg.release_layout
  tt.func @release_boundary(%src: tensor<64x32xf32, #src>) -> tensor<64x32xf32, #dst> {
    %released = tlx.release_layout %src : tensor<64x32xf32, #src> -> tensor<64x32xf32, #dst>
    tt.return %released : tensor<64x32xf32, #dst>
  }

  // CHECK-LABEL: tt.func @identity_release_boundary
  // CHECK: ttg.release_layout
  tt.func @identity_release_boundary(%src: tensor<64x32xf32, #src>) -> tensor<64x32xf32, #src> {
    %released = tlx.release_layout %src : tensor<64x32xf32, #src> -> tensor<64x32xf32, #src>
    tt.return %released : tensor<64x32xf32, #src>
  }

  // CHECK-LABEL: tt.func @relaxed_release
  // CHECK-NOT: ttg.release_layout
  // CHECK: ttg.convert_layout
  tt.func @relaxed_release(%src: tensor<64x32xf32, #src>) -> tensor<64x32xf32, #dst> {
    %released = tlx.release_layout %src {relaxed = true} : tensor<64x32xf32, #src> -> tensor<64x32xf32, #dst>
    tt.return %released : tensor<64x32xf32, #dst>
  }
}
