// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=hip:gfx950 -tritonamdgpu-schedule-loops -verify-diagnostics | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad="target=hip:gfx950 matrix-instruction-size=16" -tritonamdgpu-schedule-loops -o /dev/null 2>&1 | FileCheck %s --check-prefix=NONK16

// On gfx950 a dot whose only NPOT dim is K stays exact when its pow2-padded K's
// MFMA divides K and issues it from exact operands.
// CHECK-LABEL: @dot_k96
// CHECK: tt.dot {{.*}} : tensor<128x96xbf16> * tensor<96x128xbf16> ->
tt.func @dot_k96(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x96x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x96x!tt.ptr<bf16>>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<96x128x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<96x128x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x96xbf16> * tensor<96x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----

// K=48 is exact on 32x32x16, but 16x16x32 doesn't divide it.
// CHECK-LABEL: @dot_k48
// CHECK: tt.dot {{.*}} : tensor<128x48xbf16> * tensor<48x128xbf16> ->
// NONK16: remark: NPOT dot is not yet supported on MFMA: the pow2-padded K's MFMA (rocdl.mfma.f32.16x16x32.bf16, kDim 32) doesn't divide K=48
tt.func @dot_k48(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x48x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x48x!tt.ptr<bf16>>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<48x128x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<48x128x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x48xbf16> * tensor<48x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----

// An NPOT M or N pads.
// CHECK-LABEL: @dot_n96
// CHECK: tt.dot {{.*}} : tensor<128x64xbf16> * tensor<64x128xbf16> ->
tt.func @dot_n96(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x64x!tt.ptr<bf16>>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<64x96x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<64x96x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported on MFMA except at a 2-D dot's K}}
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x96xbf16> -> tensor<128x96xf32>
  tt.return
}

// -----

// fp8 at K=192 takes the scaled f8f6f4 MFMA, which has no NPOT K yet.
// CHECK-LABEL: @dot_k192_fp8
// CHECK: tt.dot {{.*}} : tensor<128x256xf8E4M3FN> * tensor<256x128xf8E4M3FN> ->
tt.func @dot_k192_fp8(%pa: !tt.ptr<f8E4M3FN>, %pb: !tt.ptr<f8E4M3FN>) {
  %sa = tt.splat %pa : !tt.ptr<f8E4M3FN> -> tensor<128x192x!tt.ptr<f8E4M3FN>>
  %a = tt.load %sa : tensor<128x192x!tt.ptr<f8E4M3FN>>
  %sb = tt.splat %pb : !tt.ptr<f8E4M3FN> -> tensor<192x128x!tt.ptr<f8E4M3FN>>
  %b = tt.load %sb : tensor<192x128x!tt.ptr<f8E4M3FN>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported on rocdl.mfma.scale.f32.32x32x64.f8f6f4}}
  %d = tt.dot %a, %b, %c : tensor<128x192xf8E4M3FN> * tensor<192x128xf8E4M3FN> -> tensor<128x128xf32>
  tt.return
}
