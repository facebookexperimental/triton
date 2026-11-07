// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:90 -verify-diagnostics | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:90 -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' -tritongpu-coalesce -mlir-print-local-scope -verify-diagnostics | FileCheck %s --check-prefix=TTG

// A TMA store's box can't be masked, so the copy stays exact at 3 messages
// against the padded tile's 2, and Coalesce keeps a mixed-radix layout.
// CHECK-LABEL: @copy_64x96
// CHECK: tt.descriptor_load {{.*}} -> tensor<64x96xf16>
// CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<64x96xf16>, tensor<64x96xf16>
// TTG-LABEL: @copy_64x96
// TTG: tt.descriptor_load {{.*}} -> tensor<64x96xf16, #ttg.linear<{register = {{\[\[}}0, 1], [0, 2], [0, 4]
tt.func @copy_64x96(%a: !tt.ptr<f16>, %c: !tt.ptr<f16>, %m: i32, %n: i32, %s: i64) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %da = tt.make_tensor_descriptor %a, [%m, %n], [%s, %c1] : !tt.ptr<f16>, !tt.tensordesc<64x96xf16>
  %dc = tt.make_tensor_descriptor %c, [%m, %n], [%s, %c1] : !tt.ptr<f16>, !tt.tensordesc<64x96xf16>
  %x = tt.descriptor_load %da[%c0, %c0] : !tt.tensordesc<64x96xf16> -> tensor<64x96xf16>
  tt.descriptor_store %dc[%c0, %c0], %x : !tt.tensordesc<64x96xf16>, tensor<64x96xf16>
  tt.return
}

// -----

// CHECK-LABEL: @copy_64x128
// CHECK: tt.descriptor_load {{.*}} -> tensor<64x128xf16>
// TTG-LABEL: @copy_64x128
// TTG: tt.descriptor_load {{.*}} -> tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8]
tt.func @copy_64x128(%a: !tt.ptr<f16>, %c: !tt.ptr<f16>, %m: i32, %n: i32, %s: i64) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %da = tt.make_tensor_descriptor %a, [%m, %n], [%s, %c1] : !tt.ptr<f16>, !tt.tensordesc<64x128xf16>
  %dc = tt.make_tensor_descriptor %c, [%m, %n], [%s, %c1] : !tt.ptr<f16>, !tt.tensordesc<64x128xf16>
  %x = tt.descriptor_load %da[%c0, %c0] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
  tt.descriptor_store %dc[%c0, %c0], %x : !tt.tensordesc<64x128xf16>, tensor<64x128xf16>
  tt.return
}

// -----

// With no TMA store, 3 messages against 2 pad the load's descriptor.
// CHECK-LABEL: @load_64x96
// CHECK: tt.make_tensor_descriptor {{.*}} : !tt.ptr<f16>, !tt.tensordesc<64x128xf16>
// CHECK: tt.descriptor_load {{.*}} -> tensor<64x128xf16>
// CHECK: tt.store {{.*}}, %{{.*}}, %{{.*}} : tensor<64x128x!tt.ptr<f16>>
tt.func @load_64x96(%a: !tt.ptr<f16>, %c: !tt.ptr<f16>, %m: i32, %n: i32, %s: i64) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %da = tt.make_tensor_descriptor %a, [%m, %n], [%s, %c1] : !tt.ptr<f16>, !tt.tensordesc<64x96xf16>
  %x = tt.descriptor_load %da[%c0, %c0] : !tt.tensordesc<64x96xf16> -> tensor<64x96xf16>
  %sc = tt.splat %c : !tt.ptr<f16> -> tensor<64x96x!tt.ptr<f16>>
  tt.store %sc, %x : tensor<64x96x!tt.ptr<f16>>
  tt.return
}

// -----

// 3 messages against 4 of the same width keep it exact.
// CHECK-LABEL: @load_128x192
// CHECK: tt.descriptor_load {{.*}} -> tensor<128x192xf16>
// CHECK: tt.store {{.*}} : tensor<128x192x!tt.ptr<f16>>
tt.func @load_128x192(%a: !tt.ptr<f16>, %c: !tt.ptr<f16>, %m: i32, %n: i32, %s: i64) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %da = tt.make_tensor_descriptor %a, [%m, %n], [%s, %c1] : !tt.ptr<f16>, !tt.tensordesc<128x192xf16>
  %x = tt.descriptor_load %da[%c0, %c0] : !tt.tensordesc<128x192xf16> -> tensor<128x192xf16>
  %sc = tt.splat %c : !tt.ptr<f16> -> tensor<128x192x!tt.ptr<f16>>
  tt.store %sc, %x : tensor<128x192x!tt.ptr<f16>>
  tt.return
}

// -----

// A rejected box pads the connected component, which its store can't follow.
tt.func @copy_sw0(%a: !tt.ptr<f8E4M3FN>, %c: !tt.ptr<f8E4M3FN>, %m: i32, %n: i32, %s: i64) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %da = tt.make_tensor_descriptor %a, [%m, %n], [%s, %c1] : !tt.ptr<f8E4M3FN>, !tt.tensordesc<128x48xf8E4M3FN>
  %dc = tt.make_tensor_descriptor %c, [%m, %n], [%s, %c1] : !tt.ptr<f8E4M3FN>, !tt.tensordesc<128x48xf8E4M3FN>
  // expected-remark @+1 {{NPOT TMA box pitch differs from its shared layout}}
  %x = tt.descriptor_load %da[%c0, %c0] : !tt.tensordesc<128x48xf8E4M3FN> -> tensor<128x48xf8E4M3FN>
  // expected-remark @+2 {{NPOT TMA box pitch differs from its shared layout}}
  // expected-error @+1 {{'tt.descriptor_store' op on an NPOT tensor is unsupported}}
  tt.descriptor_store %dc[%c0, %c0], %x : !tt.tensordesc<128x48xf8E4M3FN>, tensor<128x48xf8E4M3FN>
  tt.return
}

// -----

// CHECK-LABEL: @gather_rows24
// CHECK: tt.descriptor_gather {{.*}} -> tensor<32x64xf16>
tt.func @gather_rows24(%a: !tt.ptr<f16>, %c: !tt.ptr<f16>, %m: i32, %n: i32, %s: i64) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %da = tt.make_tensor_descriptor %a, [%m, %n], [%s, %c1] : !tt.ptr<f16>, !tt.tensordesc<1x64xf16>
  %rows = tt.make_range {end = 24 : i32, start = 0 : i32} : tensor<24xi32>
  // expected-remark @+1 {{NPOT TMA gather/scatter rows are not yet supported}}
  %x = tt.descriptor_gather %da[%rows, %c0] : (!tt.tensordesc<1x64xf16>, tensor<24xi32>, i32) -> tensor<24x64xf16>
  %sc = tt.splat %c : !tt.ptr<f16> -> tensor<24x64x!tt.ptr<f16>>
  tt.store %sc, %x : tensor<24x64x!tt.ptr<f16>>
  tt.return
}
