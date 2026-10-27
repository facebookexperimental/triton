// RUN: triton-opt %s -split-input-file -verify-diagnostics
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics

// Rejected in both modes, even when only a B operand uses the parent.
// expected-error@+1 {{NVIDIA MMA instrShape with an NPOT M or N is not yet supported, got instrShape = [16, 24]}}
#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 24]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @b_operand_n24(%x: f16) {
    %0 = tt.splat %x : f16 -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    tt.return
  }
}

// -----

// expected-error@+2 {{ttg.dot_op NPOT kWidth parameter is not yet supported for Ampere or Hopper MMA parent, got: 3}}
#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#dot_op = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 3}>

// -----

// Power-of-two control.
#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @b_operand_n8(%x: f16) {
    %0 = tt.splat %x : f16 -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    tt.return
  }
}
