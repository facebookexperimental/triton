// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -canonicalize | FileCheck %s

// Merging two reshapes that move elements would give a 96 -> 96 reshape whose
// only valid result is #l1, so both stay. The pow2 pair merges and folds.
// CHECK-LABEL: @product_pair_kept
//       CHECK:   %[[R:.+]] = tt.reshape %{{.+}} : tensor<96xf32, #{{.+}}> -> tensor<32x3xf32, #{{.+}}>
//       CHECK:   %[[S:.+]] = tt.reshape %[[R]] : tensor<32x3xf32, #{{.+}}> -> tensor<96xf32, #{{.+}}>
//       CHECK:   tt.store %{{.+}}, %[[S]]
#l1 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
#l2 = #ttg.linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [32, 3]}>
#l3 = #ttg.linear<{register = [[32]], lane = [[2], [1], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func @product_pair_kept(%p: !tt.ptr<f32>) {
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #l1>
    %v = tt.load %s : tensor<96x!tt.ptr<f32>, #l1>
    %r = tt.reshape %v : tensor<96xf32, #l1> -> tensor<32x3xf32, #l2>
    %t = tt.reshape %r : tensor<32x3xf32, #l2> -> tensor<96xf32, #l3>
    %d = tt.splat %p : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #l3>
    tt.store %d, %t : tensor<96x!tt.ptr<f32>, #l3>
    tt.return
}
}

// -----

// CHECK-LABEL: @pow2_pair_folds
//   CHECK-NOT:   tt.reshape
//       CHECK:   %[[V:.+]] = tt.load
//   CHECK-NOT:   tt.reshape
//       CHECK:   tt.store %{{.+}}, %[[V]]
#l1 = #ttg.linear<{register = [[1], [2]], lane = [[4], [8], [16], [32], [64]], warp = [], block = []}>
#l2 = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func @pow2_pair_folds(%p: !tt.ptr<f32>) {
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #l1>
    %v = tt.load %s : tensor<128x!tt.ptr<f32>, #l1>
    %r = tt.reshape %v : tensor<128xf32, #l1> -> tensor<32x4xf32, #l2>
    %t = tt.reshape %r : tensor<32x4xf32, #l2> -> tensor<128xf32, #l1>
    tt.store %s, %t : tensor<128x!tt.ptr<f32>, #l1>
    tt.return
}
}
