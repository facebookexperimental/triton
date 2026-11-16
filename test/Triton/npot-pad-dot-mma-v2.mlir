// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:80 -verify-diagnostics | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad="target=cuda:80 num-warps=2" 2>/dev/null | FileCheck %s --check-prefix=W2

// On sm80 a dot stays exact when its warps fit each dim's power-of-two part
// and match the padded dot's, here 2x2.
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

// CHECK-LABEL: @dot_128x128
// CHECK: tt.dot {{.*}} -> tensor<128x128xf32>
tt.func @dot_128x128(%pb: !tt.ptr<f16>) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x128x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
  tt.return
}

// -----

// 4 warps split 48x48 2x2, but M's power-of-two part, 16, holds one m16 atom.
// CHECK-LABEL: @dot_48x48
// CHECK: tt.dot {{.*}} -> tensor<64x64xf32>
// W2-LABEL: @dot_48x48
// W2: tt.dot {{.*}} -> tensor<48x48xf32>
tt.func @dot_48x48(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<48x64x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<48x64x!tt.ptr<f16>>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x48x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x48x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<48x48xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported with 2 warps of 16 on dim 0, whose power-of-two part is 16}}
  %d = tt.dot %a, %b, %c : tensor<48x64xf16> * tensor<64x48xf16> -> tensor<48x48xf32>
  tt.return
}

// -----

// 248's power-of-two part, 8, holds one n8 tile, so exact puts all 4 warps on
// M and each holds all of N; padded splits them 1x4.
// CHECK-LABEL: @dot_128x248
// CHECK: tt.dot {{.*}} -> tensor<128x256xf32>
tt.func @dot_128x248(%pb: !tt.ptr<f16>) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x248x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x248x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x248xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported on mma.sync with warps 4x1, where padded places them 1x4}}
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x248xf16> -> tensor<128x248xf32>
  tt.return
}

// -----

// Chained dots take the last dot's split: 96x64 puts all 4 warps on M, though
// alone it would split 2x2 and fit.
// CHECK-LABEL: @chain_96x64
// CHECK-COUNT-2: tt.dot {{.*}} -> tensor<128x64xf32>
tt.func @chain_96x64(%pq: !tt.ptr<f16>) {
  %sq = tt.splat %pq : !tt.ptr<f16> -> tensor<96x64x!tt.ptr<f16>>
  %q = tt.load %sq : tensor<96x64x!tt.ptr<f16>>
  %k = arith.constant dense<1.0> : tensor<64x64xf16>
  %z = arith.constant dense<0.0> : tensor<96x64xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported with 4 warps of 16 on dim 0, whose power-of-two part is 32}}
  %s = tt.dot %q, %k, %z : tensor<96x64xf16> * tensor<64x64xf16> -> tensor<96x64xf32>
  %p = arith.truncf %s : tensor<96x64xf32> to tensor<96x64xf16>
  // expected-remark @+1 {{NPOT dot is not yet supported with 4 warps of 16 on dim 0, whose power-of-two part is 32}}
  %o = tt.dot %p, %k, %z : tensor<96x64xf16> * tensor<64x64xf16> -> tensor<96x64xf32>
  tt.return
}

// -----

// A wider last dot puts the warps on N, so each holds all 96 rows; padded
// keeps them on M.
// CHECK-LABEL: @chain_96x128
// CHECK: tt.dot {{.*}} -> tensor<128x64xf32>
// CHECK: tt.dot {{.*}} -> tensor<128x128xf32>
tt.func @chain_96x128(%pq: !tt.ptr<f16>) {
  %sq = tt.splat %pq : !tt.ptr<f16> -> tensor<96x64x!tt.ptr<f16>>
  %q = tt.load %sq : tensor<96x64x!tt.ptr<f16>>
  %k = arith.constant dense<1.0> : tensor<64x64xf16>
  %z = arith.constant dense<0.0> : tensor<96x64xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported on mma.sync with warps 1x4, where padded places them 4x1}}
  %s = tt.dot %q, %k, %z : tensor<96x64xf16> * tensor<64x64xf16> -> tensor<96x64xf32>
  %p = arith.truncf %s : tensor<96x64xf32> to tensor<96x64xf16>
  %v = arith.constant dense<1.0> : tensor<64x128xf16>
  %zo = arith.constant dense<0.0> : tensor<96x128xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported on mma.sync with warps 1x4, where padded places them 4x1}}
  %o = tt.dot %p, %v, %zo : tensor<96x64xf16> * tensor<64x128xf16> -> tensor<96x128xf32>
  tt.return
}

// -----

// K's power-of-two part, 32, holds whole k16 operand tiles.
// CHECK-LABEL: @dot_k96
// CHECK: tt.dot {{.*}} : tensor<128x96xf16> * tensor<96x128xf16>
tt.func @dot_k96(%pa: !tt.ptr<f16>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x96x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<128x96x!tt.ptr<f16>>
  %b = arith.constant dense<1.0> : tensor<96x128xf16>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x96xf16> * tensor<96x128xf16> -> tensor<128x128xf32>
  tt.return
}

// -----

// Loaded as fp8, f16 operands take k32 tiles, which 48's 16 can't hold.
// CHECK-LABEL: @dot_k48_fp8
// CHECK: tt.dot {{.*}} : tensor<128x64xf16> * tensor<64x128xf16>
tt.func @dot_k48_fp8(%pa: !tt.ptr<f8E4M3FN>) {
  %sa = tt.splat %pa : !tt.ptr<f8E4M3FN> -> tensor<128x48x!tt.ptr<f8E4M3FN>>
  %a8 = tt.load %sa : tensor<128x48x!tt.ptr<f8E4M3FN>>
  %a = tt.fp_to_fp %a8 : tensor<128x48xf8E4M3FN> -> tensor<128x48xf16>
  %b = arith.constant dense<1.0> : tensor<48x128xf16>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported on mma.sync with K 48, whose power-of-two part 16 is less than k32}}
  %d = tt.dot %a, %b, %c : tensor<128x48xf16> * tensor<48x128xf16> -> tensor<128x128xf32>
  tt.return
}

// -----

// An f32 dot without tf32 runs on FMA, not mma.sync.
// CHECK-LABEL: @dot_fma
// CHECK: tt.dot {{.*}} -> tensor<128x128xf32>
tt.func @dot_fma(%pb: !tt.ptr<f32>) {
  %a = arith.constant dense<1.0> : tensor<128x32xf32>
  %sb = tt.splat %pb : !tt.ptr<f32> -> tensor<32x96x!tt.ptr<f32>>
  %b = tt.load %sb : tensor<32x96x!tt.ptr<f32>>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported on cuda:80}}
  %d = tt.dot %a, %b, %c, inputPrecision = ieee : tensor<128x32xf32> * tensor<32x96xf32> -> tensor<128x96xf32>
  tt.return
}
