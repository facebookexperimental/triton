// RUN: triton-opt %s -split-input-file --nvgpu-warp-specialization="num-stages=3 capability=100" | FileCheck %s

// Tests that warp specialization is skipped when num_warps < 4, since
// the default partition requires at least 4 warps for TMEM ops.
//
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: ttg.warp_return
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
module attributes {"ttg.num-warps" = 2 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func public @ws_skip_insufficient_warps(%arg0: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %result = scf.for %iv = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%acc = %arg0) -> (tensor<128x128xf32, #blocked>) : i32 {
      %add = arith.addf %acc, %acc : tensor<128x128xf32, #blocked>
      scf.yield %add : tensor<128x128xf32, #blocked>
    } {tt.warp_specialize}
    tt.return %result : tensor<128x128xf32, #blocked>
  }
}
