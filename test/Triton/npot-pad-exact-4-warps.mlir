// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad='num-warps=4' -verify-diagnostics | FileCheck %s

// Vector first: at four warps exact can't keep padded's v4 on 384 (3 a
// thread): pad.
// CHECK-LABEL: @copy_384
tt.func @copy_384(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %s: f32) {
  // CHECK: tt.load %{{.*}}, %{{.*}} : tensor<512x!tt.ptr<f32>>
  %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  // expected-remark @+1 {{NPOT loads and stores pad: exact takes 768 vector accesses, padded 256}}
  %x = tt.load %p : tensor<384x!tt.ptr<f32>>
  %ss = tt.splat %s : f32 -> tensor<384xf32>
  %y = arith.addf %x, %ss : tensor<384xf32>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %q = tt.addptr %dp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  tt.store %q, %y : tensor<384x!tt.ptr<f32>>
  tt.return
}

// -----

// An unaligned copy, x[1:] to y[1:]: neither side proves more than one element,
// so total in-dim sizes decide (384 vs 512): exact.
// CHECK-LABEL: @unaligned_384
tt.func @unaligned_384(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK: tt.load %{{.*}} : tensor<384x!tt.ptr<f32>>
  // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<384x!tt.ptr<f32>>
  %c1 = arith.constant 1 : i32
  %s1 = tt.addptr %src, %c1 : !tt.ptr<f32>, i32
  %d1 = tt.addptr %dst, %c1 : !tt.ptr<f32>, i32
  %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  %sp = tt.splat %s1 : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  %x = tt.load %p : tensor<384x!tt.ptr<f32>>
  %dp = tt.splat %d1 : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %q = tt.addptr %dp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  tt.store %q, %x : tensor<384x!tt.ptr<f32>>
  tt.return
}

// -----

// Unaligned too, 216 takes more total in-dim size than its padded 256, so it
// pads and says why, on the first op with an NPOT value and a source line.
// CHECK-LABEL: @unaligned_216
tt.func @unaligned_216(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK: tt.load %{{.*}}, %{{.*}} : tensor<256x!tt.ptr<f32>>
  %zero = arith.constant dense<0> : tensor<216xi32> loc(unknown)
  %c1 = arith.constant 1 : i32
  %s1 = tt.addptr %src, %c1 : !tt.ptr<f32>, i32
  %d1 = tt.addptr %dst, %c1 : !tt.ptr<f32>, i32
  // expected-remark @+1 {{NPOT tensors pad: their mixed-radix layouts take 27648 registers x lanes x warps, their pow2 layouts 2048}}
  %r0 = tt.make_range {end = 216 : i32, start = 0 : i32} : tensor<216xi32>
  %r = arith.addi %r0, %zero : tensor<216xi32>
  %sp = tt.splat %s1 : !tt.ptr<f32> -> tensor<216x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<216x!tt.ptr<f32>>, tensor<216xi32>
  %x = tt.load %p : tensor<216x!tt.ptr<f32>>
  %dp = tt.splat %d1 : !tt.ptr<f32> -> tensor<216x!tt.ptr<f32>>
  %q = tt.addptr %dp, %r : tensor<216x!tt.ptr<f32>>, tensor<216xi32>
  tt.store %q, %x : tensor<216x!tt.ptr<f32>>
  tt.return
}

// -----

// At four warps 96 is scalar either way, and exact replicates it to a total
// in-dim size of 384: pad.
// CHECK-LABEL: @copy_96
tt.func @copy_96(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK: tt.load %{{.*}}, %{{.*}} : tensor<128x!tt.ptr<f32>>
  // expected-remark @+1 {{NPOT tensors pad: their mixed-radix layouts take 2304 registers x lanes x warps, their pow2 layouts 768}}
  %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<96x!tt.ptr<f32>>, tensor<96xi32>
  %x = tt.load %p : tensor<96x!tt.ptr<f32>>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>>
  %q = tt.addptr %dp, %r : tensor<96x!tt.ptr<f32>>, tensor<96xi32>
  tt.store %q, %x : tensor<96x!tt.ptr<f32>>
  tt.return
}

// -----

// [96, 64] at four warps keeps v4 along 64 and fills a total in-dim size of
// 6144, where padded wastes 2048 of 8192: exact.
// CHECK-LABEL: @copy_96x64
tt.func @copy_96x64(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK-NOT: tensor<128x64x
  // CHECK: tt.load %{{.*}} : tensor<96x64x!tt.ptr<f32>>
  %r = tt.make_range {end = 6144 : i32, start = 0 : i32} : tensor<6144xi32>
  %i = tt.reshape %r : tensor<6144xi32> -> tensor<96x64xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<96x64x!tt.ptr<f32>>
  %p = tt.addptr %sp, %i : tensor<96x64x!tt.ptr<f32>>, tensor<96x64xi32>
  %x = tt.load %p : tensor<96x64x!tt.ptr<f32>>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<96x64x!tt.ptr<f32>>
  %q = tt.addptr %dp, %i : tensor<96x64x!tt.ptr<f32>>, tensor<96x64xi32>
  tt.store %q, %x : tensor<96x64x!tt.ptr<f32>>
  tt.return
}

// -----

// The tile idiom at four warps: its 96-row index values take a larger total
// in-dim size exact than padded, but its [96, 64] values take 2048 fewer each,
// so the connected component keeps its shape: total in-dim sizes are summed
// over the connected component.
// CHECK-LABEL: @tile_96x64
tt.func @tile_96x64(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  // CHECK: tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32>
  // CHECK: tt.load %{{.*}} : tensor<96x64x!tt.ptr<f32>>
  // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<96x64x!tt.ptr<f32>>
  %c64 = arith.constant dense<64> : tensor<96x1xi32>
  %m = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32>
  %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<96xi32> -> tensor<96x1xi32>
  %row = arith.muli %m1, %c64 : tensor<96x1xi32>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %col = tt.broadcast %n1 : tensor<1x64xi32> -> tensor<96x64xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<96x1x!tt.ptr<f32>>
  %sr = tt.addptr %sp, %row : tensor<96x1x!tt.ptr<f32>>, tensor<96x1xi32>
  %sb = tt.broadcast %sr : tensor<96x1x!tt.ptr<f32>> -> tensor<96x64x!tt.ptr<f32>>
  %p = tt.addptr %sb, %col : tensor<96x64x!tt.ptr<f32>>, tensor<96x64xi32>
  %x = tt.load %p : tensor<96x64x!tt.ptr<f32>>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<96x1x!tt.ptr<f32>>
  %dr = tt.addptr %dp, %row : tensor<96x1x!tt.ptr<f32>>, tensor<96x1xi32>
  %db = tt.broadcast %dr : tensor<96x1x!tt.ptr<f32>> -> tensor<96x64x!tt.ptr<f32>>
  %q = tt.addptr %db, %col : tensor<96x64x!tt.ptr<f32>>, tensor<96x64xi32>
  tt.store %q, %x : tensor<96x64x!tt.ptr<f32>>
  tt.return
}

// -----

// Forwarded values join the connected component they reach: at four warps the
// accesses pad, so the constant arm and the result pad with them.
// CHECK-LABEL: @if_forwards
tt.func @if_forwards(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c: i1) {
  // CHECK: scf.if %{{.*}} -> (tensor<512xf32>)
  // CHECK: arith.constant dense<0.000000e+00> : tensor<512xf32>
  // CHECK: tt.load %{{.*}}, %{{.*}} : tensor<512x!tt.ptr<f32>>
  %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %q = tt.addptr %dp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  %acc = scf.if %c -> (tensor<384xf32>) {
    %z = arith.constant dense<0.000000e+00> : tensor<384xf32>
    scf.yield %z : tensor<384xf32>
  } else {
    // expected-remark @+1 {{NPOT loads and stores pad: exact takes 768 vector accesses, padded 256}}
    %x = tt.load %p : tensor<384x!tt.ptr<f32>>
    scf.yield %x : tensor<384xf32>
  }
  tt.store %q, %acc : tensor<384x!tt.ptr<f32>>
  tt.return
}

// -----

// So does a loop's iter-arg: the constant it yields pads with the loaded init.
// CHECK-LABEL: @for_forwards
tt.func @for_forwards(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
  // CHECK: scf.for {{.*}} -> (tensor<512xf32>)
  // CHECK: arith.constant dense<1.000000e+00> : tensor<512xf32>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  // expected-remark @+1 {{NPOT loads and stores pad: exact takes 768 vector accesses, padded 256}}
  %init = tt.load %p : tensor<384x!tt.ptr<f32>>
  %acc = scf.for %i = %c0 to %n step %c1 iter_args(%a = %init) -> (tensor<384xf32>) : i32 {
    %one = arith.constant dense<1.000000e+00> : tensor<384xf32>
    scf.yield %one : tensor<384xf32>
  }
  tt.store %p, %acc : tensor<384x!tt.ptr<f32>>
  tt.return
}

// -----

// A reduce lowers a mixed-radix layout, also with its odd digit on the axis; an
// inner_tree reduce has an unsupported reason, so its connected component pads.
// CHECK-LABEL: @reduce_32x96
tt.func @reduce_32x96(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK: tt.load %{{.*}} : tensor<32x96x!tt.ptr<f32>>
  // CHECK: tt.load %{{.*}}, %{{.*}} : tensor<32x128x!tt.ptr<f32>>
  %r = tt.make_range {end = 3072 : i32, start = 0 : i32} : tensor<3072xi32>
  %i = tt.reshape %r : tensor<3072xi32> -> tensor<32x96xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<32x96x!tt.ptr<f32>>
  %p = tt.addptr %sp, %i : tensor<32x96x!tt.ptr<f32>>, tensor<32x96xi32>
  %x = tt.load %p : tensor<32x96x!tt.ptr<f32>>
  %s = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.addf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<32x96xf32>) -> tensor<32xf32>
  %r2 = tt.make_range {end = 3072 : i32, start = 0 : i32} : tensor<3072xi32>
  %i2 = tt.reshape %r2 : tensor<3072xi32> -> tensor<32x96xi32>
  %sp2 = tt.splat %src : !tt.ptr<f32> -> tensor<32x96x!tt.ptr<f32>>
  %p2 = tt.addptr %sp2, %i2 : tensor<32x96x!tt.ptr<f32>>, tensor<32x96xi32>
  %y = tt.load %p2 : tensor<32x96x!tt.ptr<f32>>
  // expected-remark @+1 {{NPOT mixed-radix layout is not yet supported on an inner_tree reduce}}
  %t = "tt.reduce"(%y) <{axis = 1 : i32, reduction_ordering = "inner_tree"}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.addf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<32x96xf32>) -> tensor<32xf32>
  %u = arith.addf %s, %t : tensor<32xf32>
  %m = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %q = tt.addptr %dp, %m : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  tt.store %q, %u : tensor<32x!tt.ptr<f32>>
  tt.return
}

// -----

// At four warps [128, 80] puts its warps on 128; a reduce along either dim
// keeps its shape.
// CHECK-LABEL: @reduce_128x80
tt.func @reduce_128x80(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK: tt.load %{{.*}} : tensor<128x80x!tt.ptr<f32>>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
  // CHECK: tt.load %{{.*}} : tensor<128x80x!tt.ptr<f32>>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
  %r = tt.make_range {end = 10240 : i32, start = 0 : i32} : tensor<10240xi32>
  %i = tt.reshape %r : tensor<10240xi32> -> tensor<128x80xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<128x80x!tt.ptr<f32>>
  %p = tt.addptr %sp, %i : tensor<128x80x!tt.ptr<f32>>, tensor<128x80xi32>
  %x = tt.load %p : tensor<128x80x!tt.ptr<f32>>
  %s = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.addf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<128x80xf32>) -> tensor<128xf32>
  %m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %q = tt.addptr %dp, %m : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  tt.store %q, %s : tensor<128x!tt.ptr<f32>>
  %r2 = tt.make_range {end = 10240 : i32, start = 0 : i32} : tensor<10240xi32>
  %i2 = tt.reshape %r2 : tensor<10240xi32> -> tensor<128x80xi32>
  %sp2 = tt.splat %src : !tt.ptr<f32> -> tensor<128x80x!tt.ptr<f32>>
  %p2 = tt.addptr %sp2, %i2 : tensor<128x80x!tt.ptr<f32>>, tensor<128x80xi32>
  %y = tt.load %p2 : tensor<128x80x!tt.ptr<f32>>
  %t = "tt.reduce"(%y) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.addf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<128x80xf32>) -> tensor<80xf32>
  %n = tt.make_range {end = 80 : i32, start = 0 : i32} : tensor<80xi32>
  %ep = tt.splat %dst : !tt.ptr<f32> -> tensor<80x!tt.ptr<f32>>
  %e = tt.addptr %ep, %n : tensor<80x!tt.ptr<f32>>, tensor<80xi32>
  tt.store %e, %t : tensor<80x!tt.ptr<f32>>
  tt.return
}

// -----

// A scan lowers a mixed-radix layout along either dim of [96, 80], though the
// radix-3 digit on 96 is not most-major.
// CHECK-LABEL: @scan_96x80
tt.func @scan_96x80(%src: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  // CHECK: "tt.scan"(%{{.*}}) <{axis = 1 : i32, reverse = false}>
  // CHECK: }) : (tensor<96x80xi32>) -> tensor<96x80xi32>
  // CHECK: "tt.scan"(%{{.*}}) <{axis = 0 : i32, reverse = false}>
  // CHECK: }) : (tensor<96x80xi32>) -> tensor<96x80xi32>
  %r = tt.make_range {end = 7680 : i32, start = 0 : i32} : tensor<7680xi32>
  %i = tt.reshape %r : tensor<7680xi32> -> tensor<96x80xi32>
  %sp = tt.splat %src : !tt.ptr<i32> -> tensor<96x80x!tt.ptr<i32>>
  %p = tt.addptr %sp, %i : tensor<96x80x!tt.ptr<i32>>, tensor<96x80xi32>
  %x = tt.load %p : tensor<96x80x!tt.ptr<i32>>
  %s = "tt.scan"(%x) <{axis = 1 : i32, reverse = false}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.addi %a, %b : i32
    tt.scan.return %c : i32
  }) : (tensor<96x80xi32>) -> tensor<96x80xi32>
  tt.store %p, %s : tensor<96x80x!tt.ptr<i32>>
  %r2 = tt.make_range {end = 7680 : i32, start = 0 : i32} : tensor<7680xi32>
  %i2 = tt.reshape %r2 : tensor<7680xi32> -> tensor<96x80xi32>
  %sp2 = tt.splat %src : !tt.ptr<i32> -> tensor<96x80x!tt.ptr<i32>>
  %p2 = tt.addptr %sp2, %i2 : tensor<96x80x!tt.ptr<i32>>, tensor<96x80xi32>
  %y = tt.load %p2 : tensor<96x80x!tt.ptr<i32>>
  %t = "tt.scan"(%y) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.addi %a, %b : i32
    tt.scan.return %c : i32
  }) : (tensor<96x80xi32>) -> tensor<96x80xi32>
  tt.store %p2, %t : tensor<96x80x!tt.ptr<i32>>
  tt.return
}

// -----

// An atomic keeps a mixed-radix layout, but at four warps a [96] one
// broadcasts, so its connected component pads once its result is used.
// CHECK-LABEL: @atomic_96_result_used
tt.func @atomic_96_result_used(%dst: !tt.ptr<i32>) {
  // CHECK: tt.atomic_rmw add, relaxed, gpu, %{{.*}}, %{{.*}} : (tensor<96x!tt.ptr<i32>>
  // CHECK: tt.atomic_rmw add, relaxed, gpu, %{{.*}}, %{{.*}}, %{{.*}} : (tensor<128x!tt.ptr<i32>>
  %r = tt.make_range {end = 3072 : i32, start = 0 : i32} : tensor<3072xi32>
  %x = tt.reshape %r : tensor<3072xi32> -> tensor<96x32xi32>
  %s = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.addi %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<96x32xi32>) -> tensor<96xi32>
  %dp = tt.splat %dst : !tt.ptr<i32> -> tensor<96x!tt.ptr<i32>>
  %q = tt.addptr %dp, %s : tensor<96x!tt.ptr<i32>>, tensor<96xi32>
  %u = tt.atomic_rmw add, relaxed, gpu, %q, %s : (tensor<96x!tt.ptr<i32>>, tensor<96xi32>) -> tensor<96xi32>
  %r2 = tt.make_range {end = 3072 : i32, start = 0 : i32} : tensor<3072xi32>
  %x2 = tt.reshape %r2 : tensor<3072xi32> -> tensor<96x32xi32>
  %t = "tt.reduce"(%x2) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.addi %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<96x32xi32>) -> tensor<96xi32>
  %dp2 = tt.splat %dst : !tt.ptr<i32> -> tensor<96x!tt.ptr<i32>>
  %q2 = tt.addptr %dp2, %t : tensor<96x!tt.ptr<i32>>, tensor<96xi32>
  // expected-remark @+1 {{NPOT mixed-radix layout with broadcast elements is not yet supported on an atomic whose result is used}}
  %v = tt.atomic_rmw add, relaxed, gpu, %q2, %t : (tensor<96x!tt.ptr<i32>>, tensor<96xi32>) -> tensor<96xi32>
  tt.store %q2, %v : tensor<96x!tt.ptr<i32>>
  tt.return
}
