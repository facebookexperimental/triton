// RUN: triton-opt -split-input-file --tlx-propagate-layout --tlx-resolve-placeholder-layouts -tritongpu-remove-layout-conversions %s | FileCheck %s

// End-to-end anchor: a user-pinned shared layout survives the pass that actually
// rewrites layouts (tritongpu-remove-layout-conversions). The wrapper is unwrapped
// to the concrete swizzle in tlx-propagate-layout, and remove-layout-conversions
// does not rewrite a local_alloc's shared encoding, so the user's swizzle reaches
// the end unchanged.

#swizzled = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#user = #tlx.user_layout<#swizzled>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-DAG: #[[$SWZ:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>
// CHECK-NOT: #tlx.user_layout
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @survives
  tt.func @survives(%arg0: tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #blocked> {
    // CHECK: ttg.local_alloc {{.*}} -> !ttg.memdesc<128x64xf16, #[[$SWZ]], #smem, mutable>
    %0 = ttg.local_alloc %arg0 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #user, #smem, mutable>
    %1 = ttg.local_load %0 : !ttg.memdesc<128x64xf16, #user, #smem, mutable> -> tensor<128x64xf16, #blocked>
    tt.return %1 : tensor<128x64xf16, #blocked>
  }
}
