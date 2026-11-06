// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad='num-warps=1' -verify-diagnostics | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad='num-warps=1 num-ctas=2' -verify-diagnostics | FileCheck %s --check-prefix=CTAS
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad='num-warps=1' -convert-triton-to-tritongpu='target=cuda:90 num-warps=1' -verify-diagnostics | FileCheck %s --check-prefix=TTGIR
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad='num-warps=1' -convert-triton-to-tritongpu='target=cuda:90 num-warps=1' -tritongpu-coalesce -verify-diagnostics | FileCheck %s --check-prefix=V4
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad='num-warps=4' -verify-diagnostics | FileCheck %s --check-prefix=W4

// The row-pointer idiom (X + m)[:, None] + n[None, :]: only memory accesses
// must keep their vector, so the [96] row pointers don't force padding on the
// tile's v4.
// V4: #[[$V4:linear[0-9]*]] = #ttg.linear<{register = {{\[\[0, 1\], \[0, 2\],}}
// CHECK-LABEL: @row_pointer
// V4-LABEL: @row_pointer
tt.func @row_pointer(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK: tt.load %{{.*}} : tensor<96x64x!tt.ptr<f32>>
  // V4: tt.load %{{.*}} : tensor<96x64x!tt.ptr<f32>, #[[$V4]]>
  %c64 = arith.constant dense<64> : tensor<96xi32>
  %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32>
  %m = arith.muli %r, %c64 : tensor<96xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>>
  %row = tt.addptr %sp, %m : tensor<96x!tt.ptr<f32>>, tensor<96xi32>
  %row1 = tt.expand_dims %row {axis = 1 : i32} : tensor<96x!tt.ptr<f32>> -> tensor<96x1x!tt.ptr<f32>>
  %rows = tt.broadcast %row1 : tensor<96x1x!tt.ptr<f32>> -> tensor<96x64x!tt.ptr<f32>>
  %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %col = tt.broadcast %n1 : tensor<1x64xi32> -> tensor<96x64xi32>
  %p = tt.addptr %rows, %col : tensor<96x64x!tt.ptr<f32>>, tensor<96x64xi32>
  %x = tt.load %p : tensor<96x64x!tt.ptr<f32>>
  tt.store %p, %x : tensor<96x64x!tt.ptr<f32>>
  tt.return
}

// -----

// Vector first, then total in-dim size. At one warp 384 keeps padded's v4 (4 x
// 32 lanes x 3) and fills a total in-dim size of 384, where padded wastes 128
// of 512: exact. At four warps exact can't keep v4 (3 a thread); across CTAs no
// mixed-radix layout: pad.
// CHECK-LABEL: @copy_384
// CTAS-LABEL: @copy_384
// W4-LABEL: @copy_384
tt.func @copy_384(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %s: f32) {
  // CHECK-NOT: tensor<512x
  // CHECK: tt.load %{{.*}} : tensor<384x!tt.ptr<f32>>
  // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<384x!tt.ptr<f32>>
  // CTAS: tt.load %{{.*}}, %{{.*}} : tensor<512x!tt.ptr<f32>>
  // W4: tt.load %{{.*}}, %{{.*}} : tensor<512x!tt.ptr<f32>>
  // TTGIR: tt.load %{{.*}} : tensor<384x!tt.ptr<f32>, #linear>
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

// -----

// An unaligned copy, x[1:] to y[1:]: neither side proves more than one element,
// so total in-dim sizes decide (384 vs 512): exact.
// W4-LABEL: @unaligned_384
tt.func @unaligned_384(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // W4: tt.load %{{.*}} : tensor<384x!tt.ptr<f32>>
  // W4: tt.store %{{.*}}, %{{.*}} : tensor<384x!tt.ptr<f32>>
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

// 96 at one warp wastes less total in-dim size exact (0 vs 32), but its
// 32-element pow2 part only fills the lanes, so it loads 3 x 32b where padded
// loads v4: pad. At four warps both are scalar and exact replicates to a total
// in-dim size of 384: pad.
// CHECK-LABEL: @copy_96
// W4-LABEL: @copy_96
tt.func @copy_96(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK: tt.load %{{.*}}, %{{.*}} : tensor<128x!tt.ptr<f32>>
  // W4: tt.load %{{.*}}, %{{.*}} : tensor<128x!tt.ptr<f32>>
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

// [96, 2] contiguous along dim 0 at one warp: Coalesce vectorizes along 96,
// where padded loads v4 and exact only v2: pad.
// CHECK-LABEL: @column_contiguous
tt.func @column_contiguous(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK: tt.load {{.*}} : tensor<128x2x!tt.ptr<f32>>
  %c96 = arith.constant dense<96> : tensor<1x2xi32>
  %m = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32>
  %n = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<96xi32> -> tensor<96x1xi32>
  %row = tt.broadcast %m1 : tensor<96x1xi32> -> tensor<96x2xi32>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %col = arith.muli %n1, %c96 : tensor<1x2xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<1x2x!tt.ptr<f32>>
  %sc = tt.addptr %sp, %col : tensor<1x2x!tt.ptr<f32>>, tensor<1x2xi32>
  %sb = tt.broadcast %sc : tensor<1x2x!tt.ptr<f32>> -> tensor<96x2x!tt.ptr<f32>>
  %p = tt.addptr %sb, %row : tensor<96x2x!tt.ptr<f32>>, tensor<96x2xi32>
  %x = tt.load %p : tensor<96x2x!tt.ptr<f32>>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<1x2x!tt.ptr<f32>>
  %dc = tt.addptr %dp, %col : tensor<1x2x!tt.ptr<f32>>, tensor<1x2xi32>
  %db = tt.broadcast %dc : tensor<1x2x!tt.ptr<f32>> -> tensor<96x2x!tt.ptr<f32>>
  %q = tt.addptr %db, %row : tensor<96x2x!tt.ptr<f32>>, tensor<96x2xi32>
  tt.store %q, %x : tensor<96x2x!tt.ptr<f32>>
  tt.return
}

// -----

// [96, 64] at four warps keeps v4 along 64 and fills a total in-dim size of
// 6144, where padded wastes 2048 of 8192: exact.
// W4-LABEL: @copy_96x64
tt.func @copy_96x64(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // W4-NOT: tensor<128x64x
  // W4: tt.load %{{.*}} : tensor<96x64x!tt.ptr<f32>>
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
// W4-LABEL: @tile_96x64
tt.func @tile_96x64(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  // W4: tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32>
  // W4: tt.load %{{.*}} : tensor<96x64x!tt.ptr<f32>>
  // W4: tt.store %{{.*}}, %{{.*}} : tensor<96x64x!tt.ptr<f32>>
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

// An odd 1023 has no power-of-two part to spread, so exact replicates it over
// the lanes (a total in-dim size of 32736), where padded wastes 1: pad.
// CHECK-LABEL: @ranges
tt.func @ranges() {
  // CHECK: tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %r = tt.make_range {end = 1023 : i32, start = 0 : i32} : tensor<1023xi32>
  tt.return
}

// -----

// A padded connected component meets an exact one through a scalar: the reduce
// over 48 pads to 64, and its sum is splat into a 384 connected component that
// keeps its shape.
// CHECK-LABEL: @padded_meets_exact
// TTGIR-LABEL: @padded_meets_exact
tt.func @padded_meets_exact(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  // CHECK: tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
  // CHECK: (tensor<64xf32>) -> f32
  // CHECK: tt.splat %{{.*}} : f32 -> tensor<384xf32>
  // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<384x!tt.ptr<f32>>
  // TTGIR: tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>, #blocked>
  // TTGIR: tt.store %{{.*}}, %{{.*}} : tensor<384x!tt.ptr<f32>, #linear>
  %r48 = tt.make_range {end = 48 : i32, start = 0 : i32} : tensor<48xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<48x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r48 : tensor<48x!tt.ptr<f32>>, tensor<48xi32>
  %x = tt.load %p : tensor<48x!tt.ptr<f32>>
  %sum = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.addf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<48xf32>) -> f32
  %r384 = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  %v = tt.splat %sum : f32 -> tensor<384xf32>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %q = tt.addptr %dp, %r384 : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  tt.store %q, %v : tensor<384x!tt.ptr<f32>>
  tt.return
}

// -----

// Constants lower a mixed-radix layout, so `x * 2` keeps its shape.
// CHECK-LABEL: @literal
// TTGIR-LABEL: @literal
tt.func @literal(%src: !tt.ptr<i32>) {
  // CHECK: arith.constant dense<2> : tensor<384xi32>
  // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<384x!tt.ptr<i32>>
  // TTGIR: arith.constant dense<2> : tensor<384xi32, #linear>
  %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  %sp = tt.splat %src : !tt.ptr<i32> -> tensor<384x!tt.ptr<i32>>
  %p = tt.addptr %sp, %r : tensor<384x!tt.ptr<i32>>, tensor<384xi32>
  %x = tt.load %p : tensor<384x!tt.ptr<i32>>
  %two = arith.constant dense<2> : tensor<384xi32>
  %y = arith.muli %x, %two : tensor<384xi32>
  tt.store %p, %y : tensor<384x!tt.ptr<i32>>
  tt.return
}

// -----

// So do carried values: the loop keeps its shape.
// CHECK-LABEL: @loop_carried
// TTGIR-LABEL: @loop_carried
tt.func @loop_carried(%src: !tt.ptr<f32>, %n: i32) {
  // CHECK: scf.for {{.*}} -> (tensor<384xf32>)
  // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<384x!tt.ptr<f32>>
  // TTGIR: scf.for {{.*}} -> (tensor<384xf32, #linear>)
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  %init = tt.load %p : tensor<384x!tt.ptr<f32>>
  %acc = scf.for %i = %c0 to %n step %c1 iter_args(%a = %init) -> (tensor<384xf32>) : i32 {
    %next = arith.addf %a, %a : tensor<384xf32>
    scf.yield %next : tensor<384xf32>
  }
  tt.store %p, %acc : tensor<384x!tt.ptr<f32>>
  tt.return
}

// -----

// Forwarded values join the connected component they reach: at four warps the
// accesses pad, so the constant arm and the result pad with them.
// W4-LABEL: @if_forwards
tt.func @if_forwards(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c: i1) {
  // W4: scf.if %{{.*}} -> (tensor<512xf32>)
  // W4: arith.constant dense<0.000000e+00> : tensor<512xf32>
  // W4: tt.load %{{.*}}, %{{.*}} : tensor<512x!tt.ptr<f32>>
  %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %q = tt.addptr %dp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  %acc = scf.if %c -> (tensor<384xf32>) {
    %z = arith.constant dense<0.000000e+00> : tensor<384xf32>
    scf.yield %z : tensor<384xf32>
  } else {
    %x = tt.load %p : tensor<384x!tt.ptr<f32>>
    scf.yield %x : tensor<384xf32>
  }
  tt.store %q, %acc : tensor<384x!tt.ptr<f32>>
  tt.return
}

// -----

// So does a loop's iter-arg: the constant it yields pads with the loaded init.
// W4-LABEL: @for_forwards
tt.func @for_forwards(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
  // W4: scf.for {{.*}} -> (tensor<512xf32>)
  // W4: arith.constant dense<1.000000e+00> : tensor<512xf32>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  %init = tt.load %p : tensor<384x!tt.ptr<f32>>
  %acc = scf.for %i = %c0 to %n step %c1 iter_args(%a = %init) -> (tensor<384xf32>) : i32 {
    %one = arith.constant dense<1.000000e+00> : tensor<384xf32>
    scf.yield %one : tensor<384xf32>
  }
  tt.store %p, %acc : tensor<384x!tt.ptr<f32>>
  tt.return
}

// -----

// A non-entry block argument belongs to tt.func, which isn't a mixed-radix op,
// so the connected component of the cf.br feeding it pads, and the pass rejects
// cf.br by name.
tt.func @branch_rejects() {
  %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  // expected-error @+1 {{'cf.br' op on an NPOT tensor is unsupported}}
  cf.br ^bb1(%r : tensor<384xi32>)
^bb1(%y: tensor<384xi32>):
  tt.return
}

// -----

// A reduce lowers a mixed-radix layout, also with its odd digit on the axis; an
// inner_tree reduce has an unsupported reason, so its connected component pads.
// CHECK-LABEL: @reduce_32x96
// W4-LABEL: @reduce_32x96
tt.func @reduce_32x96(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK: tt.load %{{.*}} : tensor<32x96x!tt.ptr<f32>>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
  // CHECK: tt.load %{{.*}}, %{{.*}} : tensor<32x128x!tt.ptr<f32>>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32, reduction_ordering = "inner_tree"}>
  // W4: tt.load %{{.*}} : tensor<32x96x!tt.ptr<f32>>
  // W4: tt.load %{{.*}}, %{{.*}} : tensor<32x128x!tt.ptr<f32>>
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

// At four warps [128, 80] puts its warps on 128, so a reduce along 128 has an
// unsupported reason and pads; along 80 it keeps its shape.
// W4-LABEL: @reduce_128x80
tt.func @reduce_128x80(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // W4: tt.load %{{.*}} : tensor<128x80x!tt.ptr<f32>>
  // W4: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
  // W4: tt.load %{{.*}}, %{{.*}} : tensor<128x128x!tt.ptr<f32>>
  // W4: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
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
