// RUN: triton-opt %s --split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1250" --verify-diagnostics -o /dev/null

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wrong_tiles_per_warp_rank(
      %a: tensor<256x256xf8E4M3FN, #blocked>,
      %b: tensor<256x256xf8E4M3FN, #blocked1>,
      %a_scale: tensor<256x8xi8, #blocked2>,
      %b_scale: tensor<256x8xi8, #blocked2>,
      %out: tensor<256x256x!tt.ptr<f32>, #blocked3>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #blocked3>
    // expected-error @+1 {{amdg.wmma_tiles_per_warp must have 2 entries}}
    %0 = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %acc lhs = e4m3 rhs = e4m3 {fastMath = false, amdg.wmma_tiles_per_warp = array<i32: 2>} : tensor<256x256xf8E4M3FN, #blocked>, tensor<256x8xi8, #blocked2> * tensor<256x256xf8E4M3FN, #blocked1>, tensor<256x8xi8, #blocked2> -> tensor<256x256xf32, #blocked3>
    tt.store %out, %0 : tensor<256x256x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @nonpositive_tiles_per_warp(
      %a: tensor<256x256xf8E4M3FN, #blocked>,
      %b: tensor<256x256xf8E4M3FN, #blocked1>,
      %a_scale: tensor<256x8xi8, #blocked2>,
      %b_scale: tensor<256x8xi8, #blocked2>,
      %out: tensor<256x256x!tt.ptr<f32>, #blocked3>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #blocked3>
    // expected-error @+1 {{amdg.wmma_tiles_per_warp entries must be positive}}
    %0 = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %acc lhs = e4m3 rhs = e4m3 {fastMath = false, amdg.wmma_tiles_per_warp = array<i32: 0, 2>} : tensor<256x256xf8E4M3FN, #blocked>, tensor<256x8xi8, #blocked2> * tensor<256x256xf8E4M3FN, #blocked1>, tensor<256x8xi8, #blocked2> -> tensor<256x256xf32, #blocked3>
    tt.store %out, %0 : tensor<256x256x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @overflowing_tiles_per_warp_extent(
      %a: tensor<256x256xf8E4M3FN, #blocked>,
      %b: tensor<256x256xf8E4M3FN, #blocked1>,
      %a_scale: tensor<256x8xi8, #blocked2>,
      %b_scale: tensor<256x8xi8, #blocked2>,
      %out: tensor<256x256x!tt.ptr<f32>, #blocked3>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #blocked3>
    // expected-error @+1 {{amdg.wmma_tiles_per_warp requires extent 4294967296 in dimension 0, which exceeds the result tile shape}}
    %0 = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %acc lhs = e4m3 rhs = e4m3 {fastMath = false, amdg.wmma_tiles_per_warp = array<i32: 134217728, 1>} : tensor<256x256xf8E4M3FN, #blocked>, tensor<256x8xi8, #blocked2> * tensor<256x256xf8E4M3FN, #blocked1>, tensor<256x8xi8, #blocked2> -> tensor<256x256xf32, #blocked3>
    tt.store %out, %0 : tensor<256x256x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}
