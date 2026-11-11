// RUN: not triton-opt %s -tritongpu-accelerate-matmul 2>&1 | FileCheck %s

// A dot_scaled that accelerate-matmul name-rejects must fail the pass, not
// only emit the error. Otherwise the op reaches later passes, which fail with
// an unrelated message. -verify-diagnostics in accelerate-matmul.mlir does not
// see the pass result, so this test checks the exit code.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sm100_ue5m3_fp16_reject(%a: tensor<128x128xf16, #blocked2>, %scale_a: tensor<128x8xi8, #blocked1>, %b: tensor<64x128xi8, #blocked>, %scale_b: tensor<128x8xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: error: cannot decompose dot_scaled with UE5M3 (int8 block-16) scales and f16 compute type: scales up to 114688 overflow f16
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = fp16 rhs = e2m1 {fastMath = false, rhs_k_pack = true} : tensor<128x128xf16, #blocked2>, tensor<128x8xi8, #blocked1> * tensor<64x128xi8, #blocked>, tensor<128x8xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %d : tensor<128x128xf32, #blocked>
  }
}
