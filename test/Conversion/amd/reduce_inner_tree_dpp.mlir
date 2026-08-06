// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

// Regression test for the AMD DPP warp-reduce ordering (D113540598).
//
// On gfx942/CDNA (wave64) a within-warp float reduction lowers to a DPP
// butterfly of `row_shr` steps. `ROW_SHR0` is 0x110 (272), so the printed
// dpp-control constant for `row_shr:n` is `272 + n`:
//   row_shr:1 -> 273, row_shr:2 -> 274, row_shr:4 -> 276, row_shr:8 -> 280.
// The two cross-row broadcast steps that follow are unchanged either way:
//   row_bcast:15 -> 322 (BCAST15, row_mask 0xa=10), row_bcast:31 -> 323 (BCAST31).
//
// A reduction with `reduction_ordering = "inner_tree"` must emit the row_shr
// steps in COUNT-UP order (1,2,4,8 -> 273,274,276,280) so adjacent lanes combine
// first, giving a fixed tree that is bitwise num_warps-invariant. The default
// (unordered) reduction is unchanged and stays COUNT-DOWN (8,4,2,1 ->
// 280,276,274,273).

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: reduce_inner_tree_sum
  tt.func @reduce_inner_tree_sum(%arg0: tensor<64xf32, #blocked>) {
    // Count-up within-row row_shr: 1, 2, 4, 8.
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 273, 15, 15, true : f32
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 274, 15, 15, true : f32
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 276, 15, 15, true : f32
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 280, 15, 15, true : f32
    // Cross-row broadcast steps unchanged.
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 322, 10, 15, true : f32
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 323, 15, 15, true : f32
    // CHECK: rocdl.readlane
    %0 = "tt.reduce"(%arg0) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) {axis = 0 : i32, reduction_ordering = "inner_tree"} : (tensor<64xf32, #blocked>) -> f32
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: reduce_unordered_sum
  tt.func @reduce_unordered_sum(%arg0: tensor<64xf32, #blocked>) {
    // Default (no reduction_ordering) stays count-down row_shr: 8, 4, 2, 1.
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 280, 15, 15, true : f32
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 276, 15, 15, true : f32
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 274, 15, 15, true : f32
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 273, 15, 15, true : f32
    // Cross-row broadcast steps unchanged.
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 322, 10, 15, true : f32
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 323, 15, 15, true : f32
    // CHECK: rocdl.readlane
    %0 = "tt.reduce"(%arg0) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) {axis = 0 : i32} : (tensor<64xf32, #blocked>) -> f32
    tt.return
  }
}
