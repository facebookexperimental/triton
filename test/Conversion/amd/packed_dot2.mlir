// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @packed_bf16_dot2
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xbf16>
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xbf16>
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.fdot2.f32.bf16"
  // CHECK-NOT: llvm.inline_asm
  tt.func @packed_bf16_dot2(
      %lhs: tensor<64xi32, #blocked>,
      %rhs: tensor<64xi32, #blocked>,
      %acc: tensor<64xf32, #blocked>,
      %output: tensor<64x!tt.ptr<f32>, #blocked>) {
    %result = amdg.packed_dot2 %lhs, %rhs, %acc {input_type = bf16} : tensor<64xi32, #blocked>, tensor<64xi32, #blocked>, tensor<64xf32, #blocked> -> tensor<64xf32, #blocked>
    tt.store %output, %result : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: llvm.func @packed_fp16_dot2
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xf16>
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xf16>
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.fdot2"
  // CHECK-NOT: llvm.inline_asm
  tt.func @packed_fp16_dot2(
      %lhs: tensor<64xi32, #blocked>,
      %rhs: tensor<64xi32, #blocked>,
      %acc: tensor<64xf32, #blocked>,
      %output: tensor<64x!tt.ptr<f32>, #blocked>) {
    %result = amdg.packed_dot2 %lhs, %rhs, %acc {input_type = f16} : tensor<64xi32, #blocked>, tensor<64xi32, #blocked>, tensor<64xf32, #blocked> -> tensor<64xf32, #blocked>
    tt.store %output, %result : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
