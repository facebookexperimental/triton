// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:90 -verify-diagnostics | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:90 -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' -tritongpu-coalesce -mlir-print-local-scope -verify-diagnostics | FileCheck %s --check-prefix=TTG

// A TMA store's box can't be masked, so the copy stays exact and Coalesce
// keeps a mixed-radix layout.
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

// With no TMA store, the load stays exact: TMA moves its exact tile.
// CHECK-LABEL: @load_64x96
// CHECK: tt.descriptor_load {{.*}} -> tensor<64x96xf16>
// CHECK: tt.store {{.*}} : tensor<64x96x!tt.ptr<f16>>
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

// A rejected box pads the connected component, so its store goes through
// pointers masked to the live tile and the descriptor's shape.
// CHECK-LABEL: @copy_sw0
// CHECK: arith.constant dense<48> : tensor<64xi32>
// CHECK: %[[LIVE:.*]] = tt.broadcast {{.*}} -> tensor<128x64xi1>
// CHECK: tt.make_tensor_descriptor %arg0
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK: %[[X:.*]] = tt.descriptor_load {{.*}} -> tensor<128x64xf8E4M3FN>
// CHECK: %[[M:.*]] = tt.splat %arg2 : i32 -> tensor<128xi32>
// CHECK: arith.cmpi slt, %{{.*}}, %[[M]]
// CHECK: %[[ROWS:.*]] = arith.andi %[[LIVE]], {{.*}} : tensor<128x64xi1>
// CHECK: %[[S:.*]] = tt.splat %arg4 : i64 -> tensor<128xi64>
// CHECK: arith.muli {{.*}}, %[[S]]
// CHECK: %[[N:.*]] = tt.splat %arg3 : i32 -> tensor<64xi32>
// CHECK: arith.cmpi slt, %{{.*}}, %[[N]]
// CHECK: %[[MASK:.*]] = arith.andi %[[ROWS]], {{.*}} : tensor<128x64xi1>
// CHECK-NOT: tt.descriptor_store
// CHECK: tt.store {{.*}}, %[[X]], %[[MASK]] : tensor<128x64x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT: tt.return
tt.func @copy_sw0(%a: !tt.ptr<f8E4M3FN>, %c: !tt.ptr<f8E4M3FN>, %m: i32, %n: i32, %s: i64) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %da = tt.make_tensor_descriptor %a, [%m, %n], [%s, %c1] : !tt.ptr<f8E4M3FN>, !tt.tensordesc<128x48xf8E4M3FN>
  %dc = tt.make_tensor_descriptor %c, [%m, %n], [%s, %c1] : !tt.ptr<f8E4M3FN>, !tt.tensordesc<128x48xf8E4M3FN>
  // expected-remark @+1 {{NPOT TMA box pitch differs from its shared layout}}
  %x = tt.descriptor_load %da[%c0, %c0] : !tt.tensordesc<128x48xf8E4M3FN> -> tensor<128x48xf8E4M3FN>
  // expected-remark @+1 {{NPOT TMA box pitch differs from its shared layout}}
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
