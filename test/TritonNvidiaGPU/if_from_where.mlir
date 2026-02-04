// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @test_if_from_where_basic
  tt.func @test_if_from_where_basic(%cond: tensor<128xi1, #blocked>, %else_val: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    // CHECK: ttng.if_from_where
    %result = ttng.if_from_where %cond, %else_val {
      %c1 = arith.constant dense<1.0> : tensor<128xf32, #blocked>
      ttng.if_from_where_yield %c1 : tensor<128xf32, #blocked>
    } : tensor<128xi1, #blocked>, tensor<128xf32, #blocked> -> tensor<128xf32, #blocked>
    tt.return %result : tensor<128xf32, #blocked>
  }
}
