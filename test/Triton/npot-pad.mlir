// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @copy
tt.func @copy(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  // CHECK: %[[RANGE:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: %[[N:.*]] = arith.constant dense<96> : tensor<128xi32>
  // CHECK: %[[LIVE:.*]] = arith.cmpi slt, %[[RANGE]], %[[N]] : tensor<128xi32>
  // CHECK: tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @+1 {{NPOT tensors pad: their mixed-radix layouts take 3072 registers x lanes x warps, their pow2 layouts 1024}}
  %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<96x!tt.ptr<f32>>, tensor<96xi32>
  // CHECK: %[[X:.*]] = tt.load %{{.*}}, %[[LIVE]] : tensor<128x!tt.ptr<f32>>
  %x = tt.load %p : tensor<96x!tt.ptr<f32>>
  // CHECK: %[[Y:.*]] = arith.addf %[[X]], %{{.*}} : tensor<128xf32>
  %one = arith.constant dense<1.0> : tensor<96xf32>
  %y = arith.addf %x, %one : tensor<96xf32>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>>
  %q = tt.addptr %dp, %r : tensor<96x!tt.ptr<f32>>, tensor<96xi32>
  // CHECK: tt.store %{{.*}}, %[[Y]], %[[LIVE]] : tensor<128x!tt.ptr<f32>>
  tt.store %q, %y : tensor<96x!tt.ptr<f32>>
  tt.return
}

// -----

// A user mask is and-ed with the live mask; `other` is kept. [4,48] pads at four
// warps: its mixed-radix layout takes a larger total in-dim size than the
// padded one (384 vs 256).
// CHECK-LABEL: @user_mask
tt.func @user_mask(%ptr: !tt.ptr<f16>, %c: i1) {
  // CHECK: %[[RANGE:.*]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  // CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[RANGE]]
  // CHECK: %[[ROW:.*]] = tt.expand_dims %[[CMP]] {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
  // CHECK: %[[LIVE:.*]] = tt.broadcast %[[ROW]] : tensor<1x64xi1> -> tensor<4x64xi1>
  // CHECK: %[[MASK:.*]] = arith.andi %{{.*}}, %[[LIVE]] : tensor<4x64xi1>
  // CHECK: tt.load %{{.*}}, %[[MASK]], %{{.*}} : tensor<4x64x!tt.ptr<f16>>
  // expected-remark @+1 {{NPOT tensors pad: their mixed-radix layouts take 1536 registers x lanes x warps, their pow2 layouts 1024}}
  %p = tt.splat %ptr : !tt.ptr<f16> -> tensor<4x48x!tt.ptr<f16>>
  %m = tt.splat %c : i1 -> tensor<4x48xi1>
  %zero = arith.constant dense<0.0> : tensor<4x48xf16>
  %x = tt.load %p, %m, %zero : tensor<4x48x!tt.ptr<f16>>
  tt.return
}

// -----

// Two padded dims: one compare per dim, and-ed.
// CHECK-LABEL: @two_dims
tt.func @two_dims(%ptr: !tt.ptr<f32>) {
  // CHECK: tt.make_range {end = 4 : i32, start = 0 : i32}
  // CHECK: %[[ROWS:.*]] = tt.broadcast %{{.*}} : tensor<4x1xi1> -> tensor<4x128xi1>
  // CHECK: tt.make_range {end = 128 : i32, start = 0 : i32}
  // CHECK: %[[COLS:.*]] = tt.broadcast %{{.*}} : tensor<1x128xi1> -> tensor<4x128xi1>
  // CHECK: %[[LIVE:.*]] = arith.andi %[[ROWS]], %[[COLS]] : tensor<4x128xi1>
  // CHECK: tt.store %{{.*}}, %{{.*}}, %[[LIVE]] : tensor<4x128x!tt.ptr<f32>>
  // expected-remark @+1 {{NPOT tensors pad: their mixed-radix layouts take 2304 registers x lanes x warps, their pow2 layouts 1024}}
  %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<3x96x!tt.ptr<f32>>
  %v = arith.constant dense<1.0> : tensor<3x96xf32>
  tt.store %p, %v : tensor<3x96x!tt.ptr<f32>>
  tt.return
}

// -----

// Loop-carried values are retyped.
// CHECK-LABEL: @loop
tt.func @loop(%n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @+1 {{NPOT tensors pad: their mixed-radix layouts take 1536 registers x lanes x warps, their pow2 layouts 512}}
  %init = arith.constant dense<0.0> : tensor<96xf32>
  // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<128xf32>)
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (tensor<96xf32>) : i32 {
    %next = arith.addf %acc, %acc : tensor<96xf32>
    scf.yield %next : tensor<96xf32>
  }
  tt.return
}

// -----

// [3,32] -> [96] pads only the group's leading dim.
// CHECK-LABEL: @reshape_lead
tt.func @reshape_lead(%x: !tt.ptr<f32>) {
  // expected-remark @+1 {{NPOT tensors pad: their mixed-radix layouts take 1152 registers x lanes x warps, their pow2 layouts 384}}
  %s = tt.splat %x : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
  %v = tt.load %s : tensor<3x32x!tt.ptr<f32>>
  // CHECK: tt.reshape %{{.*}} : tensor<4x32xf32> -> tensor<128xf32>
  %r = tt.reshape %v : tensor<3x32xf32> -> tensor<96xf32>
  tt.return
}

// -----

// A pow2 kernel is unchanged.
// CHECK-LABEL: @pow2
tt.func @pow2(%p: tensor<128x!tt.ptr<f32>>) {
  // CHECK-NOT: arith.cmpi
  // CHECK: tt.load %{{.*}} : tensor<128x!tt.ptr<f32>>
  %x = tt.load %p : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----

// [32,3] pads to [32,4], which moves element k to lane 4*(k/3) + k%3, so live
// lane k of the result gathers that lane and the pad lanes gather lane 0.
// CHECK-LABEL: @reshape_trailing
// CHECK: %[[LOAD:.*]] = tt.load {{.*}} : tensor<32x4x!tt.ptr<f32>>
// CHECK: %[[K:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32}
// CHECK: arith.constant dense<96>
// CHECK: %[[LIVE:.*]] = arith.cmpi ult, %[[K]]
// CHECK: arith.constant dense<3>
// CHECK: %[[J:.*]] = arith.remui %[[K]]
// CHECK: arith.constant dense<3>
// CHECK: %[[I:.*]] = arith.divui %[[K]]
// CHECK: arith.constant dense<4>
// CHECK: %[[I4:.*]] = arith.muli %[[I]]
// CHECK: %[[IDX:.*]] = arith.addi %[[J]], %[[I4]]
// CHECK: %[[SEL:.*]] = arith.select %[[LIVE]], %[[IDX]]
// CHECK: %[[FLAT:.*]] = tt.reshape %[[LOAD]] : tensor<32x4xf32> -> tensor<128xf32>
// CHECK: tt.gather %[[FLAT]][%[[SEL]]] {axis = 0 : i32}
tt.func @reshape_trailing(%x: !tt.ptr<f32>) {
  // expected-remark @+1 {{NPOT tensors pad: their mixed-radix layouts take 1152 registers x lanes x warps, their pow2 layouts 384}}
  %s = tt.splat %x : !tt.ptr<f32> -> tensor<32x3x!tt.ptr<f32>>
  %v = tt.load %s : tensor<32x3x!tt.ptr<f32>>
  %r = tt.reshape %v : tensor<32x3xf32> -> tensor<96xf32>
  tt.return
}

// -----

// An exact connected component keeps the reshape that @reshape_trailing turns
// into a gather: at four warps the 384 range and its 128x3 view fill a total
// in-dim size of 384, where padded wastes 128.
// CHECK-LABEL: @reshape_trailing_exact
tt.func @reshape_trailing_exact() {
  %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  // CHECK: tt.reshape %{{.*}} : tensor<384xi32> -> tensor<128x3xi32>
  %s = tt.reshape %r : tensor<384xi32> -> tensor<128x3xi32>
  tt.return
}

// -----

// allow_reorder pads as an ordered reshape.
// CHECK-LABEL: @reshape_reorder
tt.func @reshape_reorder(%x: !tt.ptr<f32>) {
  // expected-remark @+1 {{NPOT tensors pad: their mixed-radix layouts take 1152 registers x lanes x warps, their pow2 layouts 384}}
  %s = tt.splat %x : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
  // CHECK: %[[V:.*]] = tt.load
  %v = tt.load %s : tensor<3x32x!tt.ptr<f32>>
  // CHECK: tt.reshape %[[V]] : tensor<4x32xf32> -> tensor<128xf32>
  %r = tt.reshape %v allow_reorder : tensor<3x32xf32> -> tensor<96xf32>
  tt.return
}

// -----

// Lane 4i+j of the [32,4] result gathers element 3i+j when j < 3, and the
// gathered order is kept.
// CHECK-LABEL: @reshape_reorder_trailing
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK: %[[L:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32}
// CHECK: arith.constant dense<4>
// CHECK: %[[J:.*]] = arith.remui %[[L]]
// CHECK: arith.constant dense<3>
// CHECK: %[[LIVE:.*]] = arith.cmpi ult, %[[J]]
// CHECK: arith.constant dense<4>
// CHECK: %[[I:.*]] = arith.divui %[[L]]
// CHECK: arith.constant dense<3>
// CHECK: %[[I3:.*]] = arith.muli %[[I]]
// CHECK: %[[IDX:.*]] = arith.addi %[[J]], %[[I3]]
// CHECK: %[[SEL:.*]] = arith.select %[[LIVE]], %[[IDX]]
// CHECK: %[[G:.*]] = tt.gather %[[LOAD]][%[[SEL]]] {axis = 0 : i32}
// CHECK: tt.reshape %[[G]] : tensor<128xf32> -> tensor<32x4xf32>
tt.func @reshape_reorder_trailing(%x: !tt.ptr<f32>) {
  // expected-remark @+1 {{NPOT tensors pad: their mixed-radix layouts take 1152 registers x lanes x warps, their pow2 layouts 384}}
  %s = tt.splat %x : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>>
  %v = tt.load %s : tensor<96x!tt.ptr<f32>>
  %r = tt.reshape %v allow_reorder : tensor<96xf32> -> tensor<32x3xf32>
  tt.return
}

// -----

// A pow2 allow_reorder reshape is unchanged.
// CHECK-LABEL: @reshape_reorder_pow2
tt.func @reshape_reorder_pow2(%v: tensor<4x32xf32>) {
  // CHECK: tt.reshape %{{.*}} allow_reorder : tensor<4x32xf32> -> tensor<128xf32>
  %r = tt.reshape %v allow_reorder : tensor<4x32xf32> -> tensor<128xf32>
  tt.return
}

// -----

tt.func @range_overflow() {
  // expected-remark @+2 {{NPOT tensors pad: their mixed-radix layouts take 384 registers x lanes x warps, their pow2 layouts 128}}
  // expected-error @+1 {{'tt.make_range' op with an NPOT extent whose padded end overflows i32 is unsupported}}
  %r = tt.make_range {end = 2147483643 : i32, start = 2147483547 : i32} : tensor<96xi32>
  tt.return
}

// -----

tt.func @non_splat() {
  // expected-remark @+2 {{NPOT tensors pad: their mixed-radix layouts take 384 registers x lanes x warps, their pow2 layouts 128}}
  // expected-error @+1 {{'arith.constant' op with a non-splat NPOT value is unsupported}}
  %c = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  tt.return
}

// -----

// expected-remark @+1 {{NPOT mixed-radix layout is not yet supported on tt.func}}
tt.func @impure_asm(%x: tensor<96xi32>) {
  // expected-error @+1 {{'tt.elementwise_inline_asm' op with side effects on an NPOT tensor is unsupported}}
  %y = tt.elementwise_inline_asm "mov.b32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %x : tensor<96xi32> -> tensor<96xi32>
  tt.return
}

// -----

// expected-remark @+2 {{NPOT mixed-radix layout is not yet supported on tt.func}}
// expected-error @+1 {{'tt.func' op with an NPOT tensor in its signature is unsupported}}
tt.func @signature(%x: tensor<96xf32>) {
  tt.return
}

// -----

// The converter keeps a given encoding, so the pad pass rejects an encoded NPOT
// tensor, even a mixed-radix one at a size whose exact connected component it
// would allow.
#linear = #ttg.linear<{register = [[128], [256], [512]], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [1536]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func @encoded(%x: !tt.ptr<f32>) {
  // expected-remark @+2 {{NPOT mixed-radix layout is not yet supported for tensor<1536x!tt.ptr<f32>, #ttg.linear}}
  // expected-error @+1 {{'tt.splat' op on an NPOT pointer-to-tensor, descriptor, memory descriptor or encoded tensor is unsupported}}
  %s = tt.splat %x : !tt.ptr<f32> -> tensor<1536x!tt.ptr<f32>, #linear>
  %v = tt.load %s : tensor<1536x!tt.ptr<f32>, #linear>
  tt.store %s, %v : tensor<1536x!tt.ptr<f32>, #linear>
  tt.return
}
}

// -----

// expected-remark @+1 {{NPOT mixed-radix layout is not yet supported on tt.func}}
tt.func @unknown_op(%a: tensor<48xf32>) {
  // expected-error @+1 {{'tt.print' op on an NPOT tensor is unsupported}}
  tt.print " a: " {hex = false, isSigned = array<i32: 0>} : %a : tensor<48xf32>
  tt.return
}

// -----

// A memdesc's first dim counts buffers: 3 barriers, and three 64 tiles.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: @multi_buffer
tt.func @multi_buffer(%x: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3xi64
  %bars = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared, #smem, mutable>
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x64xf32
  %a = ttg.local_alloc : () -> !ttg.memdesc<3x64xf32, #shared, #smem, mutable>
  // CHECK-COUNT-2: ttg.memdesc_index {{.*}} -> !ttg.memdesc<64xf32
  %b0 = ttg.memdesc_index %a[%c0] : !ttg.memdesc<3x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
  %b1 = ttg.memdesc_index %a[%c1] : !ttg.memdesc<3x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
  // CHECK: ttg.local_store %{{.*}} : tensor<64xf32> -> !ttg.memdesc<64xf32
  ttg.local_store %x, %b0 : tensor<64xf32> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
  // CHECK: ttg.local_load %{{.*}} -> tensor<64xf32>
  %y = ttg.local_load %b1 : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32>
  tt.return %y : tensor<64xf32>
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
tt.func @multi_buffer_npot_tile() {
  // expected-remark @+2 {{NPOT mixed-radix layout is not yet supported for !ttg.memdesc<2x96xf32}}
  // expected-error @+1 {{'ttg.local_alloc' op on an NPOT pointer-to-tensor, descriptor, memory descriptor or encoded tensor is unsupported}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<2x96xf32, #shared, #smem, mutable>
  tt.return
}
