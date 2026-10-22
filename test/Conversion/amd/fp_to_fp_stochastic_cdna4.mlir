// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s
//
// Stochastic rounding (rounding = rs) on CDNA4 lowers f32 to OCP fp8
// (V_CVT_SR_FP8/BF8_F32 via rocdl.cvt.sr.*) and to f16/bf16
// (V_CVT_SR_F16/BF16_F32 via llvm.amdgcn.cvt.sr.* intrinsics).

// CHECK-LABEL: f32_to_f8_stochastic
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @f32_to_f8_stochastic(%arg0: tensor<1024xf32, #blocked>,
                                %rbits: tensor<1024xi32, #blocked>) {
    // CHECK: rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[0]
    // CHECK: rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[1]
    // CHECK: rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[2]
    // CHECK: rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[3]
    %0 = tt.fp_to_fp %arg0, rbits = %rbits : tensor<1024xi32, #blocked>, rounding = rs : tensor<1024xf32, #blocked> -> tensor<1024xf8E5M2, #blocked>
    // CHECK: rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[0]
    // CHECK: rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[1]
    // CHECK: rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[2]
    // CHECK: rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[3]
    %1 = tt.fp_to_fp %arg0, rbits = %rbits : tensor<1024xi32, #blocked>, rounding = rs : tensor<1024xf32, #blocked> -> tensor<1024xf8E4M3FN, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: f32_to_f16_bf16_stochastic
#blocked2 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @f32_to_f16_bf16_stochastic(%arg0: tensor<1024xf32, #blocked2>,
                                      %rbits: tensor<1024xi32, #blocked2>) {
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.cvt.sr.f16.f32"
    %0 = tt.fp_to_fp %arg0, rbits = %rbits : tensor<1024xi32, #blocked2>, rounding = rs : tensor<1024xf32, #blocked2> -> tensor<1024xf16, #blocked2>
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.cvt.sr.bf16.f32"
    %1 = tt.fp_to_fp %arg0, rbits = %rbits : tensor<1024xi32, #blocked2>, rounding = rs : tensor<1024xf32, #blocked2> -> tensor<1024xbf16, #blocked2>
    tt.return
  }
}
