// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad | FileCheck %s

// A padded K is zero-filled on both operands.
// CHECK-LABEL: @dot_k
tt.func @dot_k(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>) {
  // CHECK: %[[A:.*]] = tt.load %{{.*}} : tensor<16x128x!tt.ptr<f16>>
  // CHECK: %[[B:.*]] = tt.load %{{.*}} : tensor<128x16x!tt.ptr<f16>>
  // CHECK: %[[ZA:.*]] = arith.constant dense<0.000000e+00> : tensor<16x128xf16>
  // CHECK: %[[SA:.*]] = arith.select %{{.*}}, %[[A]], %[[ZA]]
  // CHECK: %[[ZB:.*]] = arith.constant dense<0.000000e+00> : tensor<128x16xf16>
  // CHECK: %[[SB:.*]] = arith.select %{{.*}}, %[[B]], %[[ZB]]
  // CHECK: tt.dot %[[SA]], %[[SB]], %{{.*}} : tensor<16x128xf16> * tensor<128x16xf16> -> tensor<16x16xf32>
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<16x96x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<16x96x!tt.ptr<f16>>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<96x16x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<96x16x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<16x16xf32>
  %d = tt.dot %a, %b, %c : tensor<16x96xf16> * tensor<96x16xf16> -> tensor<16x16xf32>
  tt.return
}

// -----

// A scaled dot also pads its e8m0 scales with 127, a scale of 1.
// CHECK-LABEL: @dot_scaled_k
tt.func @dot_scaled_k(%pa: !tt.ptr<f8E4M3FN>, %ps: !tt.ptr<i8>) {
  // CHECK-COUNT-2: arith.constant dense<0.000000e+00> : tensor<{{16x128|128x16}}xf8E4M3FN>
  // CHECK: %[[ONE:.*]] = arith.constant dense<127> : tensor<16x4xi8>
  // CHECK: %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %[[ONE]] : tensor<16x4xi1>, tensor<16x4xi8>
  // CHECK: tt.dot_scaled %{{.*}} scale %[[S]], %{{.*}} scale %{{.*}}, %{{.*}} lhs = e4m3 rhs = e4m3
  %sa = tt.splat %pa : !tt.ptr<f8E4M3FN> -> tensor<16x96x!tt.ptr<f8E4M3FN>>
  %a = tt.load %sa : tensor<16x96x!tt.ptr<f8E4M3FN>>
  %sb = tt.splat %pa : !tt.ptr<f8E4M3FN> -> tensor<96x16x!tt.ptr<f8E4M3FN>>
  %b = tt.load %sb : tensor<96x16x!tt.ptr<f8E4M3FN>>
  %ss = tt.splat %ps : !tt.ptr<i8> -> tensor<16x3x!tt.ptr<i8>>
  %s = tt.load %ss : tensor<16x3x!tt.ptr<i8>>
  %c = arith.constant dense<0.0> : tensor<16x16xf32>
  %d = tt.dot_scaled %a scale %s, %b scale %s, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x96xf8E4M3FN>, tensor<16x3xi8> * tensor<96x16xf8E4M3FN>, tensor<16x3xi8> -> tensor<16x16xf32>
  tt.return
}

// -----

// A padded M or N needs no fill: the store masks the extra rows and columns.
// CHECK-LABEL: @dot_mn
tt.func @dot_mn(%pa: !tt.ptr<f16>) {
  // CHECK-NOT: arith.select
  // CHECK: tt.dot %{{.*}} : tensor<64x32xf16> * tensor<32x128xf16> -> tensor<64x128xf32>
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<48x32x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<48x32x!tt.ptr<f16>>
  %sb = tt.splat %pa : !tt.ptr<f16> -> tensor<32x80x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<32x80x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<48x80xf32>
  %d = tt.dot %a, %b, %c : tensor<48x32xf16> * tensor<32x80xf16> -> tensor<48x80xf32>
  tt.return
}
