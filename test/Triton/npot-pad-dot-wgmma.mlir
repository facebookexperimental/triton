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

// -----

// B is a TMA-loaded tile, transposed: its 128-byte K-major rows take
// swizzle 128, so wgmma issues N 104 exactly.
// CHECK-LABEL: @dot_128x104_trans_b
// CHECK: tt.dot {{.*}} -> tensor<128x104xf32>
// W8-LABEL: @dot_128x104_trans_b
// W8: tt.dot {{.*}} -> tensor<128x104xf32>
tt.func @dot_128x104_trans_b(%pb: !tt.ptr<f16>, %n: i32, %k: i32, %s: i64) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %db = tt.make_tensor_descriptor %pb, [%n, %k], [%s, %c1] : !tt.ptr<f16>, !tt.tensordesc<104x64xf16>
  %bt = tt.descriptor_load %db[%c0, %c0] : !tt.tensordesc<104x64xf16> -> tensor<104x64xf16>
  %b = tt.trans %bt {order = array<i32: 1, 0>} : tensor<104x64xf16> -> tensor<64x104xf16>
  %c = arith.constant dense<0.0> : tensor<128x104xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x104xf16> -> tensor<128x104xf32>
  tt.return
}

// -----

// Control: the same B loaded N-contiguous has 208-byte rows, swizzle 0.
// CHECK-LABEL: @dot_128x104_load
// CHECK: tt.dot {{.*}} -> tensor<128x128xf32>
// W8-LABEL: @dot_128x104_load
// W8: tt.dot {{.*}} -> tensor<128x128xf32>
tt.func @dot_128x104_load(%pb: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %stride = arith.constant dense<104> : tensor<64x1xi32>
  %row = arith.muli %k1, %stride : tensor<64x1xi32>
  %rowb = tt.broadcast %row : tensor<64x1xi32> -> tensor<64x104xi32>
  %n = tt.make_range {end = 104 : i32, start = 0 : i32} : tensor<104xi32>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<104xi32> -> tensor<1x104xi32>
  %nb = tt.broadcast %n1 : tensor<1x104xi32> -> tensor<64x104xi32>
  %off = arith.addi %rowb, %nb : tensor<64x104xi32>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x104x!tt.ptr<f16>>
  %ptr = tt.addptr %sb, %off : tensor<64x104x!tt.ptr<f16>>, tensor<64x104xi32>
  %b = tt.load %ptr : tensor<64x104x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x104xf32>
  // expected-remark @+1 {{NPOT wgmma N 104 is not yet supported: B's shared layout has no core-matrix form}}
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x104xf16> -> tensor<128x104xf32>
  tt.return
}

// -----

// Pow2 control: the same transposed B at N 128 is untouched.
// CHECK-LABEL: @dot_128x128_trans_b
// CHECK: tt.dot {{.*}} -> tensor<128x128xf32>
tt.func @dot_128x128_trans_b(%pb: !tt.ptr<f16>, %n: i32, %k: i32, %s: i64) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %db = tt.make_tensor_descriptor %pb, [%n, %k], [%s, %c1] : !tt.ptr<f16>, !tt.tensordesc<128x64xf16>
  %bt = tt.descriptor_load %db[%c0, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
  %b = tt.trans %bt {order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
  tt.return
}

// -----

// The dot's result isn't loaded, so its C^T store predicates off duplicate
// threads and keeps 8-element vectors: the dot stays exact.
// W8-LABEL: @dot_128x104_trans_b_store
// W8: tt.dot {{.*}} -> tensor<128x104xf32>
tt.func @dot_128x104_trans_b_store(%pb: !tt.ptr<f16>, %pc: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %n: i32, %k: i32, %s: i64) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %db = tt.make_tensor_descriptor %pb, [%n, %k], [%s, %c1] : !tt.ptr<f16>, !tt.tensordesc<104x64xf16>
  %bt = tt.descriptor_load %db[%c0, %c0] : !tt.tensordesc<104x64xf16> -> tensor<104x64xf16>
  %b = tt.trans %bt {order = array<i32: 1, 0>} : tensor<104x64xf16> -> tensor<64x104xf16>
  %c = arith.constant dense<0.0> : tensor<128x104xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x104xf16> -> tensor<128x104xf32>
  %h = arith.truncf %d : tensor<128x104xf32> to tensor<128x104xf16>
  %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %r1 = tt.expand_dims %r {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %rb = tt.broadcast %r1 : tensor<128x1xi32> -> tensor<128x104xi32>
  %m = tt.make_range {end = 104 : i32, start = 0 : i32} : tensor<104xi32>
  %m1 = tt.expand_dims %m {axis = 0 : i32} : tensor<104xi32> -> tensor<1x104xi32>
  %stride = arith.constant dense<2560> : tensor<1x104xi32>
  %col = arith.muli %m1, %stride : tensor<1x104xi32>
  %cb = tt.broadcast %col : tensor<1x104xi32> -> tensor<128x104xi32>
  %off = arith.addi %rb, %cb : tensor<128x104xi32>
  %sc = tt.splat %pc : !tt.ptr<f16> -> tensor<128x104x!tt.ptr<f16>>
  %ptr = tt.addptr %sc, %off : tensor<128x104x!tt.ptr<f16>>, tensor<128x104xi32>
  tt.store %ptr, %h : tensor<128x104x!tt.ptr<f16>>
  tt.return
}

// -----

// A warp-group dot's accumulator is released to an unencoded tensor
// and waited on after the loop (T298865930). The accumulator's MMA
// encoding is legal at 160, so the component is layout-fixed: the
// epilogue's shared staging and bias load must not cost-pad it under
// an encoding computed for 160, and the wait on the unencoded
// accumulator is only ever retyped.
#s3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#s4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#mma160 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 160, 16]}>
#smem2 = #ttg.shared_memory
#swz = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @warp_group_dot_wait_160
  // CHECK: ttng.warp_group_dot {{.*}} -> tensor<64x160xf32,
  // CHECK: ttng.warp_group_dot_wait {{.*}} : tensor<64x160xf32, #
  // CHECK: ttng.warp_group_dot_wait {{.*}} : tensor<64x160xf32>
  // CHECK-NOT: tensor<64x256xf32
  // W8-LABEL: @warp_group_dot_wait_160
  // W8: ttng.warp_group_dot {{.*}} -> tensor<64x160xf32,
  // W8: ttng.warp_group_dot_wait {{.*}} : tensor<64x160xf32, #
  // W8: ttng.warp_group_dot_wait {{.*}} : tensor<64x160xf32>
  // W8-NOT: tensor<64x256xf32
  tt.func @warp_group_dot_wait_160(%pc: !tt.ptr<f32>, %pbias: !tt.ptr<f32>) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #s3, #smem2, mutable>
    %bt = ttg.local_alloc : () -> !ttg.memdesc<64x160xbf16, #s4, #smem2, mutable>
    %c = arith.constant dense<0.0> : tensor<64x160xf32>
    %cr = tlx.require_layout %c : tensor<64x160xf32> -> tensor<64x160xf32, #mma160>
    %d0 = ttng.warp_group_dot %a, %bt, %cr : !ttg.memdesc<64x64xbf16, #s3, #smem2, mutable> * !ttg.memdesc<64x160xbf16, #s4, #smem2, mutable> -> tensor<64x160xf32, #mma160>
    %r0 = tlx.release_layout %d0 {relaxed = true} : tensor<64x160xf32, #mma160> -> tensor<64x160xf32>
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %step = arith.constant 1 : index
    %acc = scf.for %i = %lb to %ub step %step iter_args(%it = %r0) -> (tensor<64x160xf32>) {
      %in = tlx.require_layout %it : tensor<64x160xf32> -> tensor<64x160xf32, #mma160>
      %d = ttng.warp_group_dot %a, %bt, %in : !ttg.memdesc<64x64xbf16, #s3, #smem2, mutable> * !ttg.memdesc<64x160xbf16, #s4, #smem2, mutable> -> tensor<64x160xf32, #mma160>
      %wi = ttng.warp_group_dot_wait %d {pendings = 1 : i32} : tensor<64x160xf32, #mma160>
      %r = tlx.release_layout %wi {relaxed = true} : tensor<64x160xf32, #mma160> -> tensor<64x160xf32>
      scf.yield %r : tensor<64x160xf32>
    }
    %w = ttng.warp_group_dot_wait %acc {pendings = 0 : i32} : tensor<64x160xf32>
    %buf = ttg.local_alloc %w : (tensor<64x160xf32>) -> !ttg.memdesc<64x160xf32, #swz, #smem2, mutable>
    %sb = tt.splat %pbias : !tt.ptr<f32> -> tensor<64x160x!tt.ptr<f32>>
    %bias = tt.load %sb : tensor<64x160x!tt.ptr<f32>>
    %sum = arith.addf %w, %bias : tensor<64x160xf32>
    %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<64x160x!tt.ptr<f32>>
    tt.store %sc, %sum : tensor<64x160x!tt.ptr<f32>>
    tt.return
  }
}
