// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics

// Under TRITON_ALLOW_NPOT only encoding-less TTIR tensors may be NPOT. Every
// tensor with an encoding keeps the power-of-two rule, including block
// arguments and results of ops without the tensor-size trait.

#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{Number of elements must be power-of-two, but <block argument> of type}}
  tt.func @arg_96(%x: tensor<96xf32, #b>) {
    tt.return
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @constant_96() {
    // expected-error @+1 {{Number of elements must be power-of-two, but}}
    %c = arith.constant dense<0.0> : tensor<96xf32, #b>
    tt.return
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{Number of elements must be power-of-two, but <block argument> of type}}
  tt.func @func_num_warps_arg_96(%x: tensor<96xf32, #b>) attributes {"ttg.num-warps" = 4 : i32} {
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @ttir_96_control(%x: tensor<96xf32>, %f: f32) -> tensor<96xf32> {
    %s = tt.splat %f : f32 -> tensor<96xf32>
    %c = arith.constant dense<1.0> : tensor<96xf32>
    %y = arith.addf %s, %c : tensor<96xf32>
    tt.return %y : tensor<96xf32>
  }
}
