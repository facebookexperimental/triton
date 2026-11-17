// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx950 matrix-instruction-size=16" | FileCheck %s

// The MFMA rewrite re-creates `tt.dot` with MFMA layouts. The accumulator
// register-class request (`amdg.cd_regclass`, consumed by the MFMA lowering,
// see mfma-cd-regclass.mlir) has to survive on the re-created dot, and a dot
// without the request must not gain one.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_keeps_cd_regclass
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_keeps_cd_regclass(
      %arg0: tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x128xf32, #blocked>,
      %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    // CHECK: tt.dot {{.*}} {amdg.cd_regclass = "a"} : {{.*}} -> tensor<128x128xf32, #mma>
    %1 = tt.dot %arg0, %arg1, %arg2 {amdg.cd_regclass = "a"} : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg3, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_without_cd_regclass
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_without_cd_regclass(
      %arg0: tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x128xf32, #blocked>,
      %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    // CHECK-NOT: amdg.cd_regclass
    // CHECK: tt.dot {{.*}} -> tensor<128x128xf32, #mma>
    // CHECK-NOT: amdg.cd_regclass
    %1 = tt.dot %arg0, %arg1, %arg2 : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg3, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
