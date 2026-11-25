// RUN: triton-opt %s --split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx950 matrix-instruction-size=0" --verify-diagnostics -o /dev/null

// A packed FP6 dot whose K tile the native scaled MFMA cannot take must not
// fall back to the dense decompose upcast.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fp6_packed_k256(
      %a: tensor<128x256xi8, #blocked>,
      %b: tensor<256x128xi8, #blocked>,
      %sa: tensor<128x8xi8, #blocked1>,
      %sb: tensor<128x8xi8, #blocked1>,
      %out: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // expected-error @+1 {{amdg.fp6_packed dot cannot use the native scaled MFMA path: fp6 uses decompose upcast}}
    %0 = tt.dot_scaled %a scale %sa, %b scale %sb, %cst lhs = e2m3 rhs = e3m2 {amdg.fp6_packed, fastMath = false} : tensor<128x256xi8, #blocked>, tensor<128x8xi8, #blocked1> * tensor<256x128xi8, #blocked>, tensor<128x8xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.store %out, %0 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
