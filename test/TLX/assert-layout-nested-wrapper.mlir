// RUN: triton-opt --tlx-dump-layout -split-input-file %s | FileCheck %s

// getEffectiveEncoding (used by assert_same_layout's comparison via
// unwrapTlxLayoutWrappers) peels *nested* TLX layout wrappers to a fixed point.
// A nested no_verify<user<L>> / user<no_verify<L>> must compare equal to the raw
// concrete L; if unwrapping stopped after a single layer the assert would fail
// compilation, so `CHECK-NOT: tlx.assert_same_layout` also asserts full peeling.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// TMEM-pin nesting order: no_verify outside, user inside.
#nv_user = #tlx.no_verify_layout<#tlx.user_layout<#blocked>>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @nested_no_verify_user_peeled
  // CHECK-NOT: tlx.assert_same_layout
  tt.func public @nested_no_verify_user_peeled(%arg0: tensor<64xf32, #nv_user>, %arg1: tensor<64xf32, #blocked>) {
    tlx.assert_same_layout %arg0, %arg1 : tensor<64xf32, #nv_user>, tensor<64xf32, #blocked>
    tlx.assert_same_layout_expected %arg0 {expected = #blocked} : tensor<64xf32, #nv_user>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// SMEM-pin nesting order: user outside, no_verify inside.
#user_nv = #tlx.user_layout<#tlx.no_verify_layout<#blocked>>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @nested_user_no_verify_peeled
  // CHECK-NOT: tlx.assert_same_layout
  tt.func public @nested_user_no_verify_peeled(%arg0: tensor<64xf32, #user_nv>, %arg1: tensor<64xf32, #blocked>) {
    tlx.assert_same_layout %arg0, %arg1 : tensor<64xf32, #user_nv>, tensor<64xf32, #blocked>
    tlx.assert_same_layout_expected %arg0 {expected = #blocked} : tensor<64xf32, #user_nv>
    tt.return
  }
}
