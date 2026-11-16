// RUN: triton-opt %s --verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @invalid_input_type(
      %lhs: tensor<64xi32, #blocked>,
      %rhs: tensor<64xi32, #blocked>,
      %acc: tensor<64xf32, #blocked>) {
    // expected-error @+1 {{input_type must be f16 or bf16, got 'f32'}}
    %result = amdg.packed_dot2 %lhs, %rhs, %acc {input_type = f32} : tensor<64xi32, #blocked>, tensor<64xi32, #blocked>, tensor<64xf32, #blocked> -> tensor<64xf32, #blocked>
    tt.return
  }
}
