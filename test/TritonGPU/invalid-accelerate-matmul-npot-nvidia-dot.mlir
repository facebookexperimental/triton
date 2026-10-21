// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --tritongpu-accelerate-matmul -verify-diagnostics | FileCheck %s

// Under TRITON_ALLOW_NPOT, an NVIDIA tt.dot whose result M or N is not a power
// of two is rejected by name before any rewrite (the predicate does not depend
// on the target or num-ctas). A power-of-two dot keeps its lowering.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @sm90_c1_m64n96k64(%a: tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<64x96xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %c: tensor<64x96xf32, #blocked>) -> tensor<64x96xf32, #blocked> {
    // expected-error @below {{NPOT NVIDIA dot: no MMA or FMA lowering admits a result M or N extent that is not a power of two}}
    %d = tt.dot %a, %b, %c : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x96xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x96xf32, #blocked>
    tt.return %d : tensor<64x96xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @sm90_pot_m64n64k64
  // CHECK: ttng.warp_group_dot
  tt.func @sm90_pot_m64n64k64(%a: tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %c: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
    %d = tt.dot %a, %b, %c : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }
}
