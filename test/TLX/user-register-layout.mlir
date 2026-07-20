// RUN: triton-opt -split-input-file -tritongpu-coalesce %s | FileCheck %s

// A user-pinned *register* layout on a local_load (#tlx.user_layout carrying
// PinnedEncodingTrait) must not be coalesced to a "full contiguity" blocked
// layout: the user chose this register layout on purpose. It stays wrapped
// through the layout-optimization passes (anchored via the trait) and is
// unwrapped later by tlx-finalize-user-layouts.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 32]], block = []}>
#user = #tlx.user_layout<#linear>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-DAG: #[[$USER:.*]] = #tlx.user_layout<
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @reg_pin
  tt.func @reg_pin(%buf: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>) -> tensor<128x128xf16, #user> {
    // Coalesce leaves the pinned result untouched (would otherwise force #blocked).
    // CHECK: ttg.local_load {{.*}} -> tensor<128x128xf16, #[[$USER]]>
    %y = ttg.local_load %buf : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #user>
    tt.return %y : tensor<128x128xf16, #user>
  }
}
