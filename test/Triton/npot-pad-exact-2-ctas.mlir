// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -triton-npot-pad='num-warps=1 num-ctas=2' -verify-diagnostics | FileCheck %s

// Across CTAs there is no mixed-radix layout, so 384 pads and says why.
// CHECK-LABEL: @copy_384
tt.func @copy_384(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %s: f32) {
  // CHECK: tt.load %{{.*}}, %{{.*}} : tensor<512x!tt.ptr<f32>>
  // expected-remark @+1 {{NPOT mixed-radix layout is not yet supported for tensor<384xi32>}}
  %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  %x = tt.load %p : tensor<384x!tt.ptr<f32>>
  %ss = tt.splat %s : f32 -> tensor<384xf32>
  %y = arith.addf %x, %ss : tensor<384xf32>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %q = tt.addptr %dp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  tt.store %q, %y : tensor<384x!tt.ptr<f32>>
  tt.return
}
