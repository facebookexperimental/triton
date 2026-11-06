// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:90 2>/dev/null | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad="target=cuda:90 num-warps=8" -verify-diagnostics | FileCheck %s --check-prefix=W8

// One warpgroup spans N, so wgmma's N may be NPOT.
// CHECK-LABEL: @dot_64x192
// CHECK: tt.dot {{.*}} -> tensor<64x192xf32>
// W8-LABEL: @dot_64x192
// W8: tt.dot {{.*}} -> tensor<64x256xf32>
tt.func @dot_64x192(%pb: !tt.ptr<f16>) {
  %a = arith.constant dense<1.0> : tensor<64x64xf16>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x192x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x192x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<64x192xf32>
  // expected-remark @+1 {{NPOT wgmma N 96 is not yet supported with 2 warpgroups on N}}
  %d = tt.dot %a, %b, %c : tensor<64x64xf16> * tensor<64x192xf16> -> tensor<64x192xf32>
  tt.return
}

// -----

// A warpgroup's 96 rows aren't a multiple of m64, so the second goes on N.
// CHECK-LABEL: @dot_192x128
// CHECK: tt.dot {{.*}} -> tensor<192x128xf32>
// W8-LABEL: @dot_192x128
// W8: tt.dot {{.*}} -> tensor<192x128xf32>
tt.func @dot_192x128(%pa: !tt.ptr<f16>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<192x64x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<192x64x!tt.ptr<f16>>
  %b = arith.constant dense<1.0> : tensor<64x128xf16>
  %c = arith.constant dense<0.0> : tensor<192x128xf32>
  %d = tt.dot %a, %b, %c : tensor<192x64xf16> * tensor<64x128xf16> -> tensor<192x128xf32>
  tt.return
}

// -----

// B's 48-byte N-major rows take swizzle 0, which no wgmma descriptor matches.
// CHECK-LABEL: @dot_128x24
// CHECK: tt.dot {{.*}} -> tensor<128x32xf32>
tt.func @dot_128x24(%pb: !tt.ptr<f16>) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x24x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x24x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x24xf32>
  // expected-remark @+1 {{NPOT wgmma N 24 is not yet supported: B's shared layout has no core-matrix form}}
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x24xf16> -> tensor<128x24xf32>
  tt.return
}

// -----

// Control: B's 192-byte rows take swizzle 64.
// CHECK-LABEL: @dot_128x96
// CHECK: tt.dot {{.*}} -> tensor<128x96xf32>
tt.func @dot_128x96(%pb: !tt.ptr<f16>) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x96x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x96x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x96xf16> -> tensor<128x96xf32>
  tt.return
}

// -----

// There N/2 = 12 leaves each warpgroup less than an n8 atom's power-of-two part.
// W8-LABEL: @dot_192x24
// W8: tt.dot {{.*}} -> tensor<256x32xf32>
tt.func @dot_192x24(%pa: !tt.ptr<f16>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<192x64x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<192x64x!tt.ptr<f16>>
  %b = arith.constant dense<1.0> : tensor<64x24xf16>
  %c = arith.constant dense<0.0> : tensor<192x24xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported with 2 warps of 8 on dim 1, whose power-of-two part is 8}}
  %d = tt.dot %a, %b, %c : tensor<192x64xf16> * tensor<64x24xf16> -> tensor<192x24xf32>
  tt.return
}

// -----

// K's power-of-two part, 32, holds whole k16 atoms.
// W8-LABEL: @dot_k96
// W8: tt.dot {{.*}} : tensor<128x96xf16> * tensor<96x128xf16>
tt.func @dot_k96(%pa: !tt.ptr<f16>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x96x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<128x96x!tt.ptr<f16>>
  %b = arith.constant dense<1.0> : tensor<96x128xf16>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x96xf16> * tensor<96x128xf16> -> tensor<128x128xf32>
  tt.return
}

// -----

// W8-LABEL: @dot_k48_i8
// W8: tt.dot {{.*}} : tensor<128x64xi8> * tensor<64x128xi8>
tt.func @dot_k48_i8(%pa: !tt.ptr<i8>) {
  %sa = tt.splat %pa : !tt.ptr<i8> -> tensor<128x48x!tt.ptr<i8>>
  %a = tt.load %sa : tensor<128x48x!tt.ptr<i8>>
  %b = arith.constant dense<1> : tensor<48x128xi8>
  %c = arith.constant dense<0> : tensor<128x128xi32>
  // expected-remark @+1 {{NPOT dot is not yet supported on wgmma with K 48, whose power-of-two part 16 is less than k32}}
  %d = tt.dot %a, %b, %c : tensor<128x48xi8> * tensor<48x128xi8> -> tensor<128x128xi32>
  tt.return
}
