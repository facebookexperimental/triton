// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --tritongpu-accelerate-matmul -verify-diagnostics | FileCheck %s

// Under TRITON_ALLOW_NPOT, an sm_12x two-scale dot_scaled with an NPOT operand
// or scale is rejected by name before any rewrite. Power-of-two dots keep
// their native lowering.

#out = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @sm120_two_scale_k96(%a: tensor<64x48xi8, #a>, %sa: tensor<64x6xf8E4M3FN, #scale>, %b: tensor<48x64xi8, #out>, %sb: tensor<64x6xf8E4M3FN, #scale>, %c: tensor<64x64xf32, #out>) -> tensor<64x64xf32, #out> {
    // expected-error @below {{NPOT dot_scaled: decomposing a scaled operand or scale with a non-power-of-two extent is unsupported}}
    %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false, lhs_k_pack = true, rhs_k_pack = true} : tensor<64x48xi8, #a>, tensor<64x6xf8E4M3FN, #scale> * tensor<48x64xi8, #out>, tensor<64x6xf8E4M3FN, #scale> -> tensor<64x64xf32, #out>
    tt.return %d : tensor<64x64xf32, #out>
  }
}

// -----

#out = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @sm120_native_scaled_pot_n64
  // CHECK-NOT: ttg.fp4_to_fp
  // CHECK: tt.dot_scaled {{.*}} -> tensor<64x64xf32, #mma>
  tt.func @sm120_native_scaled_pot_n64(%a: tensor<64x32xi8, #a>, %sa: tensor<64x4xf8E4M3FN, #scale>, %b: tensor<32x64xi8, #out>, %sb: tensor<64x4xf8E4M3FN, #scale>, %c: tensor<64x64xf32, #out>) -> tensor<64x64xf32, #out> {
    %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false, lhs_k_pack = true, rhs_k_pack = true} : tensor<64x32xi8, #a>, tensor<64x4xf8E4M3FN, #scale> * tensor<32x64xi8, #out>, tensor<64x4xf8E4M3FN, #scale> -> tensor<64x64xf32, #out>
    tt.return %d : tensor<64x64xf32, #out>
  }
}
