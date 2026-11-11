// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -convert-triton-to-tritongpu="target=hip:gfx950 num-warps=4 threads-per-warp=64" -tritonamdgpu-accelerate-matmul="gfx-arch=gfx950 kPack=2" | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -convert-triton-to-tritongpu="target=hip:gfx950 num-warps=4 threads-per-warp=64" -tritonamdgpu-accelerate-matmul="gfx-arch=gfx950 matrix-instruction-size=64" -verify-diagnostics -o /dev/null

// An NPOT K keeps kBase-wide operands under kPack=2; a nonKDim with no MFMA
// fails the pass, since FMA has no NPOT dot.
// CHECK-LABEL: @dot_k96
// CHECK: tt.dot {{.*}} : tensor<128x96xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<96x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
tt.func @dot_k96(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>, %pc: !tt.ptr<f32>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x96x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x96x!tt.ptr<bf16>>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<96x128x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<96x128x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  // expected-error @+1 {{NPOT dot is not yet supported on FMA: matrix_instr_nonkdim 64 selects no MFMA for K 96}}
  %d = tt.dot %a, %b, %c : tensor<128x96xbf16> * tensor<96x128xbf16> -> tensor<128x128xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x128x!tt.ptr<f32>>
  tt.return
}

// Control: a pow2 K packs to kWidth 16.
// CHECK-LABEL: @dot_k128
// CHECK: tt.dot {{.*}} : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
tt.func @dot_k128(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>, %pc: !tt.ptr<f32>) {
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x128x!tt.ptr<f32>>
  tt.return
}
