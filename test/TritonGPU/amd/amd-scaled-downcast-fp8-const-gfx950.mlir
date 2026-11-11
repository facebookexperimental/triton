// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --canonicalize --cse | FileCheck %s

#four = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#two = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // scale_const = 119 is the E8M0 scale 2^-8 (f32 bits 0x3B800000), which the
  // conversion divides by, so it multiplies by 256 with no FP32 multiply. Both
  // pairs pack into one 32-bit word.
  // CHECK-LABEL: llvm.func @const_scale_four
  // CHECK: %[[BITS:.+]] = llvm.mlir.constant(998244352 : i32) : i32
  // CHECK-NOT: llvm.fmul
  // CHECK: %[[SCALE:.+]] = llvm.bitcast %[[BITS]] : i32 to f32
  // CHECK: %[[LOW:.+]] = rocdl.cvt.scalef32.pk.fp8.f32 %{{.+}}, %{{.+}}, %[[SCALE]] -> %{{.+}}[false] : vector<2xi16>
  // CHECK: rocdl.cvt.scalef32.pk.fp8.f32 %{{.+}}, %{{.+}}, %[[SCALE]] -> %[[LOW]][true] : vector<2xi16>
  // CHECK-NOT: rocdl.cvt.scalef32
  // CHECK: llvm.return
  tt.func public @const_scale_four(%input: tensor<1x256xf32, #four>, %output: tensor<1x256x!tt.ptr<f8E4M3FN>, #four>) {
    %result = amdg.scaled_downcast_fp8 %input {axis = 1 : i32, scale_const = 119 : i8} : tensor<1x256xf32, #four> -> tensor<1x256xf8E4M3FN, #four>
    tt.store %output, %result : tensor<1x256x!tt.ptr<f8E4M3FN>, #four>
    tt.return
  }

  // A constant scale imposes no layout constraint: two elements per thread
  // convert as one pair with no group-of-eight requirement.
  // CHECK-LABEL: llvm.func @const_scale_two
  // CHECK: %[[BITS:.+]] = llvm.mlir.constant(998244352 : i32) : i32
  // CHECK: %[[SCALE:.+]] = llvm.bitcast %[[BITS]] : i32 to f32
  // CHECK: rocdl.cvt.scalef32.pk.fp8.f32 %{{.+}}, %{{.+}}, %[[SCALE]] -> %{{.+}}[false] : vector<2xi16>
  // CHECK-NOT: rocdl.cvt.scalef32
  // CHECK-COUNT-2: llvm.store
  // CHECK-NOT: llvm.store
  tt.func public @const_scale_two(%input: tensor<1x128xf32, #two>, %output: tensor<1x128x!tt.ptr<f8E4M3FN>, #two>) {
    %result = amdg.scaled_downcast_fp8 %input {axis = 1 : i32, scale_const = 119 : i8} : tensor<1x128xf32, #two> -> tensor<1x128xf8E4M3FN, #two>
    tt.store %output, %result : tensor<1x128x!tt.ptr<f8E4M3FN>, #two>
    tt.return
  }
}
