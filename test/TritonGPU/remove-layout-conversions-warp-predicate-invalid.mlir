// RUN: not triton-opt %s -tritongpu-remove-layout-conversions 2>&1 | FileCheck %s

#blocked_acc = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked_row = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#pinned_acc = #tlx.user_layout<#blocked_acc>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: error: cannot form a cross-lane-free warp_predicate layout island
  tt.func @pinned_boundary_conflicts_with_body_native_layout(
      %predicate: tensor<128xi1, #blocked_row>,
      %acc: tensor<128x128xf32, #pinned_acc>)
      -> tensor<128x128xf32, #pinned_acc> {
    %result = ttg.warp_predicate %predicate (%acc) {
      %acc_wave = ttg.convert_layout %acc : tensor<128x128xf32, #pinned_acc> -> tensor<128x128xf32, #mma>
      %next = arith.addf %acc_wave, %acc_wave : tensor<128x128xf32, #mma>
      %old = ttg.convert_layout %next : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #pinned_acc>
      ttg.predicate_yield %old : tensor<128x128xf32, #pinned_acc>
    } : (tensor<128xi1, #blocked_row>, tensor<128x128xf32, #pinned_acc>) -> tensor<128x128xf32, #pinned_acc>
    tt.return %result : tensor<128x128xf32, #pinned_acc>
  }
}
