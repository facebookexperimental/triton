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
    // Four elements per thread pack into one i32 through a byte-select
    // chain: each convert reads the previous convert's accumulator.
    // CHECK: %[[B0:.*]] = rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[0] : i32
    // CHECK: %[[B1:.*]] = rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %[[B0]][1] : i32
    // CHECK: %[[B2:.*]] = rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %[[B1]][2] : i32
    // CHECK: %[[B3:.*]] = rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %[[B2]][3] : i32
    %0 = tt.fp_to_fp %arg0, rbits = %rbits : tensor<1024xi32, #blocked>, rounding = rs : tensor<1024xf32, #blocked> -> tensor<1024xf8E5M2FNUZ, #blocked>
    // CHECK: %[[F0:.*]] = rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[0] : i32
    // CHECK: %[[F1:.*]] = rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %[[F0]][1] : i32
    // CHECK: %[[F2:.*]] = rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %[[F1]][2] : i32
    // CHECK: %[[F3:.*]] = rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %[[F2]][3] : i32
    // CHECK-NOT: rocdl.cvt.sr.
    // CHECK: llvm.return
    %1 = tt.fp_to_fp %arg0, rbits = %rbits : tensor<1024xi32, #blocked>, rounding = rs : tensor<1024xf32, #blocked> -> tensor<1024xf8E4M3FNUZ, #blocked>
    tt.return
  }
}
