// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad='target=cuda:90 num-warps=4' -convert-triton-to-tritongpu='num-warps=4 target=cuda:90' -tritongpu-accelerate-matmul -tlx-insert-require-layout -tlx-propagate-layout -verify-diagnostics | FileCheck %s

// End to end: layout anchors on an NPOT tile keep the exact shape through
// the pad pass, TTGIR conversion and layout propagation (the per-stage
// guards are npot-pad-tlx-require-layout.mlir and
// npot-propagate-anchors.mlir).
// CHECK-LABEL: @anchors_e2e_96
// CHECK: ttng.warp_group_dot {{.*}} tensor<64x96xf32
// CHECK-NOT: tensor<64x128
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func @anchors_e2e_96(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>, %po: !tt.ptr<f32>) {
    %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<64x96x!tt.ptr<f16>>
    %a = tt.load %sa : tensor<64x96x!tt.ptr<f16>>
    %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<96x96x!tt.ptr<f16>>
    %b = tt.load %sb : tensor<96x96x!tt.ptr<f16>>
    %r = tlx.require_layout %a : tensor<64x96xf16> -> tensor<64x96xf16>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x96xf32>
    %acc = tt.dot %r, %b, %cst : tensor<64x96xf16> * tensor<96x96xf16> -> tensor<64x96xf32>
    %s = tlx.release_layout %acc : tensor<64x96xf32> -> tensor<64x96xf32>
    %so = tt.splat %po : !tt.ptr<f32> -> tensor<64x96x!tt.ptr<f32>>
    tt.store %so, %s : tensor<64x96x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Control: a power-of-two tile takes the same chain unchanged.
// CHECK-LABEL: @anchors_e2e_128
// CHECK: ttng.warp_group_dot {{.*}} tensor<64x128xf32
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func @anchors_e2e_128(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>, %po: !tt.ptr<f32>) {
    %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
    %a = tt.load %sa : tensor<64x128x!tt.ptr<f16>>
    %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>>
    %b = tt.load %sb : tensor<128x128x!tt.ptr<f16>>
    %r = tlx.require_layout %a : tensor<64x128xf16> -> tensor<64x128xf16>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %r, %b, %cst : tensor<64x128xf16> * tensor<128x128xf16> -> tensor<64x128xf32>
    %s = tlx.release_layout %acc : tensor<64x128xf32> -> tensor<64x128xf32>
    %so = tt.splat %po : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>>
    tt.store %so, %s : tensor<64x128x!tt.ptr<f32>>
    tt.return
  }
}
