// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=hip:gfx950 -tritonamdgpu-schedule-loops -verify-diagnostics | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad="target=hip:gfx950 matrix-instruction-size=16" -tritonamdgpu-schedule-loops -o /dev/null 2>&1 | FileCheck %s --check-prefix=NONK16
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=hip:gfx950 \
// RUN:   | not FileCheck %s --check-prefix=FP6-NATIVE

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

// An NPOT N pads when its warps' MFMA tiles overflow its power-of-two part.
// CHECK-LABEL: @dot_n96
// CHECK: tt.dot {{.*}} : tensor<128x64xbf16> * tensor<64x128xbf16> ->
tt.func @dot_n96(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x64x!tt.ptr<bf16>>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<64x96x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<64x96x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported with 2 warps of 32 on dim 1, whose power-of-two part is 32}}
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x96xbf16> -> tensor<128x96xf32>
  tt.return
}

// -----

// fp8 at K=192 issues three f8f6f4 MFMAs from exact operands.
// CHECK-LABEL: @dot_k192_fp8
// CHECK: tt.dot {{.*}} : tensor<128x192xf8E4M3FN> * tensor<192x128xf8E4M3FN> ->
tt.func @dot_k192_fp8(%pa: !tt.ptr<f8E4M3FN>, %pb: !tt.ptr<f8E4M3FN>) {
  %sa = tt.splat %pa : !tt.ptr<f8E4M3FN> -> tensor<128x192x!tt.ptr<f8E4M3FN>>
  %a = tt.load %sa : tensor<128x192x!tt.ptr<f8E4M3FN>>
  %sb = tt.splat %pb : !tt.ptr<f8E4M3FN> -> tensor<192x128x!tt.ptr<f8E4M3FN>>
  %b = tt.load %sb : tensor<192x128x!tt.ptr<f8E4M3FN>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x192xf8E4M3FN> * tensor<192x128xf8E4M3FN> -> tensor<128x128xf32>
  tt.return
}

// -----

// An fp4 lane holds 32 consecutive K (16 bytes) under one scale, and A's 48
// bytes per row don't fill whole 32-byte K tiles, so K=96 pads.
// CHECK-LABEL: @mxfp4_k96
// CHECK: tt.dot_scaled {{.*}} : tensor<128x64xi8>, tensor<128x4xi8> * tensor<64x128xi8>, tensor<128x4xi8> ->
tt.func @mxfp4_k96(%pa: !tt.ptr<i8>, %pb: !tt.ptr<i8>, %ps: !tt.ptr<i8>) {
  %sa = tt.splat %pa : !tt.ptr<i8> -> tensor<128x48x!tt.ptr<i8>>
  %a = tt.load %sa : tensor<128x48x!tt.ptr<i8>>
  %sb = tt.splat %pb : !tt.ptr<i8> -> tensor<48x128x!tt.ptr<i8>>
  %b = tt.load %sb : tensor<48x128x!tt.ptr<i8>>
  %ss = tt.splat %ps : !tt.ptr<i8> -> tensor<128x3x!tt.ptr<i8>>
  %s = tt.load %ss : tensor<128x3x!tt.ptr<i8>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported on rocdl.mfma.scale.f32.32x32x64.f8f6f4: operand 0 has no exact layout}}
  %d = tt.dot_scaled %a scale %s, %b scale %s, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x48xi8>, tensor<128x3xi8> * tensor<48x128xi8>, tensor<128x3xi8> -> tensor<128x128xf32>
  tt.return
}

// -----

// The third RUN expects fp6 K=96 to pad while the native pattern's gate
// "fp6 uses decompose upcast" holds; when the gate goes, drop the `not`, the
// padded CHECK and the remark.
// CHECK-LABEL: @fp6_k96
// CHECK: tt.dot_scaled {{.*}} : tensor<128x128xi8>, tensor<128x4xi8> * tensor<128x128xi8>, tensor<128x4xi8> ->
// FP6-NATIVE-LABEL: @fp6_k96
// FP6-NATIVE: tt.dot_scaled {{.*}} : tensor<128x96xi8>, tensor<128x3xi8> * tensor<96x128xi8>, tensor<128x3xi8> ->
tt.func @fp6_k96(%pa: !tt.ptr<i8>, %pb: !tt.ptr<i8>, %ps: !tt.ptr<i8>) {
  %sa = tt.splat %pa : !tt.ptr<i8> -> tensor<128x96x!tt.ptr<i8>>
  %a = tt.load %sa : tensor<128x96x!tt.ptr<i8>>
  %sb = tt.splat %pb : !tt.ptr<i8> -> tensor<96x128x!tt.ptr<i8>>
  %b = tt.load %sb : tensor<96x128x!tt.ptr<i8>>
  %ss = tt.splat %ps : !tt.ptr<i8> -> tensor<128x3x!tt.ptr<i8>>
  %s = tt.load %ss : tensor<128x3x!tt.ptr<i8>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  // expected-remark @+1 {{NPOT dot_scaled is not yet supported on MFMA: fp6 uses decompose upcast}}
  %d = tt.dot_scaled %a scale %s, %b scale %s, %c lhs = e2m3 rhs = e2m3 {fastMath = false} : tensor<128x96xi8>, tensor<128x3xi8> * tensor<96x128xi8>, tensor<128x3xi8> -> tensor<128x128xf32>
  tt.return
}

// -----

// The scaled MFMA pattern takes a blocked result only, and an NPOT M or N gets
// a mixed-radix one, so M=192 pads.
// CHECK-LABEL: @mxfp8_m192
// CHECK: tt.dot_scaled {{.*}} -> tensor<256x128xf32>
tt.func @mxfp8_m192(%pa: !tt.ptr<f8E4M3FN>, %pt: !tt.ptr<i8>, %b: tensor<128x128xf8E4M3FN>, %bs: tensor<128x4xi8>) {
  %sa = tt.splat %pa : !tt.ptr<f8E4M3FN> -> tensor<192x128x!tt.ptr<f8E4M3FN>>
  %a = tt.load %sa : tensor<192x128x!tt.ptr<f8E4M3FN>>
  %st = tt.splat %pt : !tt.ptr<i8> -> tensor<192x4x!tt.ptr<i8>>
  %as = tt.load %st : tensor<192x4x!tt.ptr<i8>>
  %c = arith.constant dense<0.0> : tensor<192x128xf32>
  // expected-remark @+1 {{NPOT dot_scaled is not yet supported on MFMA at an NPOT M or N}}
  %d = tt.dot_scaled %a scale %as, %b scale %bs, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<192x128xf8E4M3FN>, tensor<192x4xi8> * tensor<128x128xf8E4M3FN>, tensor<128x4xi8> -> tensor<192x128xf32>
  tt.return
}
