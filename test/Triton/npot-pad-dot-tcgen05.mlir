// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:100 -verify-diagnostics | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:100 -convert-triton-to-tritongpu="target=cuda:100 num-warps=4" -tritongpu-accelerate-matmul 2>/dev/null | FileCheck %s --check-prefix=TC

// tcgen05 i8 needs N%16, so N=88 pads to a tcgen05 tile, not FMA.
// CHECK-LABEL: @dot_i8_128x88
// CHECK: tt.dot {{.*}} -> tensor<128x128xi32>
// TC-LABEL: @dot_i8_128x88
// TC-NOT: tt.dot
// TC: ttng.tc_gen5_mma {{.*}} !ttg.memdesc<64x128xi8, {{.*}} !ttg.memdesc<128x128xi32,
tt.func @dot_i8_128x88(%pb: !tt.ptr<i8>, %pd: !tt.ptr<i32>) {
  %a = arith.constant dense<1> : tensor<128x64xi8>
  %sb = tt.splat %pb : !tt.ptr<i8> -> tensor<64x88x!tt.ptr<i8>>
  %b = tt.load %sb : tensor<64x88x!tt.ptr<i8>>
  %c = arith.constant dense<0> : tensor<128x88xi32>
  // expected-remark @+1 {{NPOT MMAv5 shape 128x88 is unsupported: tcgen05.mma kind i8 SS MN-major B cta_group::1 per-CTA M=128 requires N to be a multiple of 16 in [16, 256], got N=88}}
  %d = tt.dot %a, %b, %c : tensor<128x64xi8> * tensor<64x88xi8> -> tensor<128x88xi32>
  %sd = tt.splat %pd : !tt.ptr<i32> -> tensor<128x88x!tt.ptr<i32>>
  tt.store %sd, %d : tensor<128x88x!tt.ptr<i32>>
  tt.return
}

// -----

// Control: f16 N=96 is a legal tcgen05 N.
// CHECK-LABEL: @dot_f16_128x96
// CHECK: tt.dot {{.*}} -> tensor<128x128xf32>
tt.func @dot_f16_128x96(%pb: !tt.ptr<f16>, %pd: !tt.ptr<f32>) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x96x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x96x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  // expected-remark @+1 {{NPOT MMAv5 accumulator N is not yet supported}}
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x96xf16> -> tensor<128x96xf32>
  %sd = tt.splat %pd : !tt.ptr<f32> -> tensor<128x96x!tt.ptr<f32>>
  tt.store %sd, %d : tensor<128x96x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @dot_f16_192x128
// CHECK: tt.dot {{.*}} -> tensor<256x128xf32>
tt.func @dot_f16_192x128(%pa: !tt.ptr<f16>, %pd: !tt.ptr<f32>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<192x64x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<192x64x!tt.ptr<f16>>
  %b = arith.constant dense<1.0> : tensor<64x128xf16>
  %c = arith.constant dense<0.0> : tensor<192x128xf32>
  // expected-remark @+1 {{NPOT MMAv5 M is not yet supported}}
  %d = tt.dot %a, %b, %c : tensor<192x64xf16> * tensor<64x128xf16> -> tensor<192x128xf32>
  %sd = tt.splat %pd : !tt.ptr<f32> -> tensor<192x128x!tt.ptr<f32>>
  tt.store %sd, %d : tensor<192x128x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @dot_f16_k96
// CHECK: tt.dot {{.*}} : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
tt.func @dot_f16_k96(%pa: !tt.ptr<f16>, %pd: !tt.ptr<f32>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x96x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<128x96x!tt.ptr<f16>>
  %b = arith.constant dense<1.0> : tensor<96x128xf16>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  // expected-remark @+1 {{NPOT MMAv5 K is not yet supported}}
  %d = tt.dot %a, %b, %c : tensor<128x96xf16> * tensor<96x128xf16> -> tensor<128x128xf32>
  %sd = tt.splat %pd : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.store %sd, %d : tensor<128x128x!tt.ptr<f32>>
  tt.return
}

// -----

// tcgen05 has no M=48, so the dot remarks tcgen05's reason.
// CHECK-LABEL: @dot_f16_48x96
// CHECK: tt.dot {{.*}} -> tensor<64x128xf32>
tt.func @dot_f16_48x96(%pd: !tt.ptr<f32>) {
  %a = arith.constant dense<1.0> : tensor<48x64xf16>
  %b = arith.constant dense<1.0> : tensor<64x96xf16>
  %c = arith.constant dense<0.0> : tensor<48x96xf32>
  // expected-remark @+1 {{NPOT MMAv5 shape 48x96 is unsupported: tcgen05.mma kind f16/tf32 SS MN-major B cta_group::1 per-CTA M=48 is not a legal M}}
  %d = tt.dot %a, %b, %c : tensor<48x64xf16> * tensor<64x96xf16> -> tensor<48x96xf32>
  %sd = tt.splat %pd : !tt.ptr<f32> -> tensor<48x96x!tt.ptr<f32>>
  tt.store %sd, %d : tensor<48x96x!tt.ptr<f32>>
  tt.return
}
