// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s
//
// Stochastic rounding (rounding = rs) on CDNA3 only covers FNUZ fp8
// (V_CVT_SR_FP8/BF8_F32 via rocdl.cvt.sr.*). OCP fp8 and f16/bf16
// stochastic rounding require CDNA4.

// CHECK-LABEL: f32_to_fnuz_f8_stochastic
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @f32_to_fnuz_f8_stochastic(%arg0: tensor<1024xf32, #blocked>,
                                     %rbits: tensor<1024xi32, #blocked>) {
    // CHECK: rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[0]
    // CHECK: rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[1]
    // CHECK: rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[2]
    // CHECK: rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[3]
    %0 = tt.fp_to_fp %arg0, rbits = %rbits : tensor<1024xi32, #blocked>, rounding = rs : tensor<1024xf32, #blocked> -> tensor<1024xf8E5M2FNUZ, #blocked>
    // CHECK: rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[0]
    // CHECK: rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[1]
    // CHECK: rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[2]
    // CHECK: rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[3]
    %1 = tt.fp_to_fp %arg0, rbits = %rbits : tensor<1024xi32, #blocked>, rounding = rs : tensor<1024xf32, #blocked> -> tensor<1024xf8E4M3FNUZ, #blocked>
    tt.return
  }
}
