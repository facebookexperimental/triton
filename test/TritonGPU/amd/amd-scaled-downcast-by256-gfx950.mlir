// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --canonicalize --cse | FileCheck %s

#four = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#two = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // The conversion divisor 2^-8 multiplies by 256 with no separate FP32 multiply.
  // CHECK-LABEL: llvm.func @by256_four
  // CHECK-DAG: %[[DIVISOR:.+]] = llvm.mlir.constant(3.906250e-03 : f32) : f32
  // CHECK-DAG: llvm.call_intrinsic "llvm.amdgcn.s.setreg"
  // CHECK-NOT: llvm.fmul
  // CHECK: %[[LOW:.+]] = rocdl.cvt.scalef32.pk.fp8.f32 %{{.+}}, %{{.+}}, %[[DIVISOR]] -> %{{.+}}[false] : vector<2xi16>
  // CHECK: rocdl.cvt.scalef32.pk.fp8.f32 %{{.+}}, %{{.+}}, %[[DIVISOR]] -> %[[LOW]][true] : vector<2xi16>
  // CHECK-NOT: rocdl.cvt.scalef32
  // CHECK: llvm.return
  tt.func public @by256_four(%input: tensor<1x256xf32, #four>, %output: tensor<1x256x!tt.ptr<f8E4M3FN>, #four>) {
    %result = amdg.scaled_downcast_fp8_by256 %input : tensor<1x256xf32, #four> -> tensor<1x256xf8E4M3FN, #four>
    tt.store %output, %result : tensor<1x256x!tt.ptr<f8E4M3FN>, #four>
    tt.return
  }

  // Two elements per thread exercise the partial four-element chunk: the
  // unused high pair converts undefined inputs and only two bytes are stored.
  // CHECK-LABEL: llvm.func @by256_tail
  // CHECK-DAG: %[[UNDEF:.+]] = llvm.mlir.undef : f32
  // CHECK-DAG: %[[DIVISOR:.+]] = llvm.mlir.constant(3.906250e-03 : f32) : f32
  // CHECK: %[[LOW:.+]] = rocdl.cvt.scalef32.pk.fp8.f32 %{{.+}}, %{{.+}}, %[[DIVISOR]] -> %{{.+}}[false] : vector<2xi16>
  // CHECK: rocdl.cvt.scalef32.pk.fp8.f32 %[[UNDEF]], %[[UNDEF]], %[[DIVISOR]] -> %[[LOW]][true] : vector<2xi16>
  // CHECK-COUNT-2: llvm.store
  // CHECK-NOT: llvm.store
  tt.func public @by256_tail(%input: tensor<1x128xf32, #two>, %output: tensor<1x128x!tt.ptr<f8E4M3FN>, #two>) {
    %result = amdg.scaled_downcast_fp8_by256 %input : tensor<1x128xf32, #two> -> tensor<1x128xf8E4M3FN, #two>
    tt.store %output, %result : tensor<1x128x!tt.ptr<f8E4M3FN>, #two>
    tt.return
  }
}
