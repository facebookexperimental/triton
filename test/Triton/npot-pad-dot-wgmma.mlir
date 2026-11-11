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
  // expected-remark @+1 {{NPOT dot is not yet supported with 2 warps of 64 on dim 1, whose power-of-two part is 64}}
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

// B's 640-byte rows take swizzle 128, so wgmma tiles N with n64, not n160.
// CHECK-LABEL: @dot_128x320
// CHECK: tt.dot {{.*}} -> tensor<128x320xf32>
// W8-LABEL: @dot_128x320
// W8: tt.dot {{.*}} -> tensor<128x320xf32>
tt.func @dot_128x320(%pb: !tt.ptr<f16>) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x320x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x320x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x320xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x320xf16> -> tensor<128x320xf32>
  tt.return
}

// -----

// At 8 warps, exact B loads 4-element vectors where the pow2 tile loads 8, so
// exact takes more vector accesses than padded.
// CHECK-LABEL: @dot_128x80_load
// CHECK: tt.dot {{.*}} -> tensor<128x80xf32>
// W8-LABEL: @dot_128x80_load
// W8: tt.dot {{.*}} -> tensor<128x128xf32>
tt.func @dot_128x80_load(%pb: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %stride = arith.constant dense<80> : tensor<64x1xi32>
  %row = arith.muli %k1, %stride : tensor<64x1xi32>
  %rowb = tt.broadcast %row : tensor<64x1xi32> -> tensor<64x80xi32>
  %n = tt.make_range {end = 80 : i32, start = 0 : i32} : tensor<80xi32>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<80xi32> -> tensor<1x80xi32>
  %nb = tt.broadcast %n1 : tensor<1x80xi32> -> tensor<64x80xi32>
  %off = arith.addi %rowb, %nb : tensor<64x80xi32>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x80x!tt.ptr<f16>>
  %ptr = tt.addptr %sb, %off : tensor<64x80x!tt.ptr<f16>>, tensor<64x80xi32>
  // expected-remark @+1 {{NPOT loads and stores pad: exact takes 1280 vector accesses, padded 1024}}
  %b = tt.load %ptr : tensor<64x80x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x80xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x80xf16> -> tensor<128x80xf32>
  tt.return
}

// -----

// At 8 warps exact B loads 4-element vectors, but the C store takes 144
// columns where the pow2 tile stores 256: fewer vector accesses in sum.
// W8-LABEL: @dot_128x144_store
// W8: tt.dot {{.*}} -> tensor<128x144xf32>
tt.func @dot_128x144_store(%pb: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pc: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %stride = arith.constant dense<144> : tensor<64x1xi32>
  %row = arith.muli %k1, %stride : tensor<64x1xi32>
  %rowb = tt.broadcast %row : tensor<64x1xi32> -> tensor<64x144xi32>
  %n = tt.make_range {end = 144 : i32, start = 0 : i32} : tensor<144xi32>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<144xi32> -> tensor<1x144xi32>
  %nb = tt.broadcast %n1 : tensor<1x144xi32> -> tensor<64x144xi32>
  %off = arith.addi %rowb, %nb : tensor<64x144xi32>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x144x!tt.ptr<f16>>
  %ptr = tt.addptr %sb, %off : tensor<64x144x!tt.ptr<f16>>, tensor<64x144xi32>
  %b = tt.load %ptr : tensor<64x144x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x144xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x144xf16> -> tensor<128x144xf32>
  %h = arith.truncf %d : tensor<128x144xf32> to tensor<128x144xf16>
  %m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %cstride = arith.constant dense<144> : tensor<128x1xi32>
  %crow = arith.muli %m1, %cstride : tensor<128x1xi32>
  %crowb = tt.broadcast %crow : tensor<128x1xi32> -> tensor<128x144xi32>
  %cn = tt.broadcast %n1 : tensor<1x144xi32> -> tensor<128x144xi32>
  %coff = arith.addi %crowb, %cn : tensor<128x144xi32>
  %sc = tt.splat %pc : !tt.ptr<f16> -> tensor<128x144x!tt.ptr<f16>>
  %cptr = tt.addptr %sc, %coff : tensor<128x144x!tt.ptr<f16>>, tensor<128x144xi32>
  tt.store %cptr, %h : tensor<128x144x!tt.ptr<f16>>
  tt.return
}

// -----

// Control: 96-wide rows keep 8-element vectors at 8 warps.
// CHECK-LABEL: @dot_128x96_load
// CHECK: tt.dot {{.*}} -> tensor<128x96xf32>
// W8-LABEL: @dot_128x96_load
// W8: tt.dot {{.*}} -> tensor<128x96xf32>
tt.func @dot_128x96_load(%pb: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %stride = arith.constant dense<96> : tensor<64x1xi32>
  %row = arith.muli %k1, %stride : tensor<64x1xi32>
  %rowb = tt.broadcast %row : tensor<64x1xi32> -> tensor<64x96xi32>
  %n = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<96xi32> -> tensor<1x96xi32>
  %nb = tt.broadcast %n1 : tensor<1x96xi32> -> tensor<64x96xi32>
  %off = arith.addi %rowb, %nb : tensor<64x96xi32>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x96x!tt.ptr<f16>>
  %ptr = tt.addptr %sb, %off : tensor<64x96x!tt.ptr<f16>>, tensor<64x96xi32>
  %b = tt.load %ptr : tensor<64x96x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x96xf16> -> tensor<128x96xf32>
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
