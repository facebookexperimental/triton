// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad="target=hip:gfx950 threads-per-warp=64 num-stages=2 use-async-copy=true" -tritonamdgpu-schedule-loops=num_stages=1 -verify-diagnostics | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad="target=hip:gfx950 matrix-instruction-size=16" -tritonamdgpu-schedule-loops=num_stages=1 -o /dev/null 2>&1 | FileCheck %s --check-prefix=NONK16

// On gfx950 a dot whose only NPOT dim is K stays exact when its pow2-padded K's
// MFMA divides K and issues it from exact operands.
// CHECK-LABEL: @dot_k96
// CHECK: tt.dot {{.*}} : tensor<128x96xbf16> * tensor<96x128xbf16> ->
tt.func @dot_k96(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x96x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x96x!tt.ptr<bf16>>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<96x128x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<96x128x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x96xbf16> * tensor<96x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----

// K=48 is exact on 32x32x16, but 16x16x32 doesn't divide it.
// CHECK-LABEL: @dot_k48
// CHECK: tt.dot {{.*}} : tensor<128x48xbf16> * tensor<48x128xbf16> ->
// NONK16: remark: NPOT dot is not yet supported on MFMA: the pow2-padded K's MFMA (rocdl.mfma.f32.16x16x32.bf16, kDim 32) doesn't divide K=48
tt.func @dot_k48(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x48x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x48x!tt.ptr<bf16>>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<48x128x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<48x128x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x48xbf16> * tensor<48x128xbf16> -> tensor<128x128xf32>
  tt.return
}

// -----

// N=96's power-of-two part 32 fits two 16-wide warps, so an unset size asks
// for the 16x16 MFMA and the dot stays exact. B is K-contiguous.
// CHECK-LABEL: @dot_n96
// CHECK: tt.dot {{.*}} : tensor<128x64xbf16> * tensor<64x96xbf16> ->
tt.func @dot_n96(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x64x!tt.ptr<bf16>>
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %k2 = tt.expand_dims %k {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %kb = tt.broadcast %k2 : tensor<64x1xi32> -> tensor<64x96xi32>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<64x96x!tt.ptr<bf16>>
  %pk = tt.addptr %sb, %kb : tensor<64x96x!tt.ptr<bf16>>, tensor<64x96xi32>
  %b = tt.load %pk : tensor<64x96x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x96xbf16> -> tensor<128x96xf32>
  tt.return
}

// -----

// An N-contiguous B puts N's odd digit inside its contiguous LDS offset.
// CHECK-LABEL: @dot_n96_n_contiguous
// CHECK: tt.dot {{.*}} : tensor<128x64xbf16> * tensor<64x128xbf16> ->
tt.func @dot_n96_n_contiguous(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x64x!tt.ptr<bf16>>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<64x96x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<64x96x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  // expected-remark @+1 {{NPOT dot pads: N's odd digit sits inside B's contiguous LDS offset, so neither ds_read_tr nor direct-to-LDS can vectorize B}}
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x96xbf16> -> tensor<128x96xf32>
  tt.return
}

// -----

// N-major B goes to K-major LDS through a register transpose: K=192 leaves each
// thread a 2-row (32-bit) K run, padded K=256 an 8-row ds_write_b128.
// CHECK-LABEL: @dot_k192_n_major
// CHECK: tt.dot {{.*}} : tensor<64x256xbf16> * tensor<256x64xbf16> ->
tt.func @dot_k192_n_major(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<64x192x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<64x192x!tt.ptr<bf16>>
  %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %n2 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %nb = tt.broadcast %n2 : tensor<1x64xi32> -> tensor<192x64xi32>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<192x64x!tt.ptr<bf16>>
  %pn = tt.addptr %sb, %nb : tensor<192x64x!tt.ptr<bf16>>, tensor<192x64xi32>
  %b = tt.load %pn : tensor<192x64x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<64x64xf32>
  // expected-remark @+1 {{NPOT dot pads: N-major B is transposed to LDS in 32-bit K runs, padded in ds_write_b128}}
  %d = tt.dot %a, %b, %c : tensor<64x192xbf16> * tensor<192x64xbf16> -> tensor<64x64xf32>
  tt.return
}

// -----

// At K=96 padded K=128 reaches only 64 bits, so the exact dot stays.
// CHECK-LABEL: @dot_k96_n_major
// CHECK: tt.dot {{.*}} : tensor<64x96xbf16> * tensor<96x64xbf16> ->
tt.func @dot_k96_n_major(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<64x96x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<64x96x!tt.ptr<bf16>>
  %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %n2 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %nb = tt.broadcast %n2 : tensor<1x64xi32> -> tensor<96x64xi32>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<96x64x!tt.ptr<bf16>>
  %pn = tt.addptr %sb, %nb : tensor<96x64x!tt.ptr<bf16>>, tensor<96x64xi32>
  %b = tt.load %pn : tensor<96x64x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<64x64xf32>
  %d = tt.dot %a, %b, %c : tensor<64x96xbf16> * tensor<96x64xbf16> -> tensor<64x64xf32>
  tt.return
}

// -----

// N=80's power-of-two part 16 holds one 16-wide warp, so the others go on M
// and N's odd part is a register repeat; that spans less than padded.
// CHECK-LABEL: @dot_n80
// CHECK: tt.dot {{.*}} : tensor<128x64xbf16> * tensor<64x80xbf16> ->
tt.func @dot_n80(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x64x!tt.ptr<bf16>>
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %k2 = tt.expand_dims %k {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %kb = tt.broadcast %k2 : tensor<64x1xi32> -> tensor<64x80xi32>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<64x80x!tt.ptr<bf16>>
  %pk = tt.addptr %sb, %kb : tensor<64x80x!tt.ptr<bf16>>, tensor<64x80xi32>
  %b = tt.load %pk : tensor<64x80x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x80xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x80xbf16> -> tensor<128x80xf32>
  tt.return
}

// -----

// N=112 placed the same way reads more of A and B per warp than padded does
// (32+112 rows vs 64+64 at 16x16), so it pads.
// CHECK-LABEL: @dot_n112
// CHECK: tt.dot {{.*}} : tensor<128x64xbf16> * tensor<64x128xbf16> ->
// NONK16: remark: NPOT dot is not yet supported on MFMA with warps 4x1: each loads 144 A rows plus B columns, padded's 2x2 load 128
tt.func @dot_n112(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x64x!tt.ptr<bf16>>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<64x112x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<64x112x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x112xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported on MFMA with warps 1x4: each loads 156 A rows plus B columns, padded's 2x2 load 128}}
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x112xbf16> -> tensor<128x112xf32>
  tt.return
}

// -----

// fp8 at K=192 takes the scaled f8f6f4 MFMA, which has no NPOT K yet.
// CHECK-LABEL: @dot_k192_fp8
// CHECK: tt.dot {{.*}} : tensor<128x256xf8E4M3FN> * tensor<256x128xf8E4M3FN> ->
tt.func @dot_k192_fp8(%pa: !tt.ptr<f8E4M3FN>, %pb: !tt.ptr<f8E4M3FN>) {
  %sa = tt.splat %pa : !tt.ptr<f8E4M3FN> -> tensor<128x192x!tt.ptr<f8E4M3FN>>
  %a = tt.load %sa : tensor<128x192x!tt.ptr<f8E4M3FN>>
  %sb = tt.splat %pb : !tt.ptr<f8E4M3FN> -> tensor<192x128x!tt.ptr<f8E4M3FN>>
  %b = tt.load %sb : tensor<192x128x!tt.ptr<f8E4M3FN>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported on rocdl.mfma.scale.f32.32x32x64.f8f6f4}}
  %d = tt.dot %a, %b, %c : tensor<128x192xf8E4M3FN> * tensor<192x128xf8E4M3FN> -> tensor<128x128xf32>
  tt.return
}

// -----

// With NPOT M, A's lanes do not write contiguous LDS, so the pipeliner can't
// copy A direct-to-LDS; it can for the padded tile.
// CHECK-LABEL: @dot_m96_loop
// CHECK: tt.dot {{.*}} : tensor<128x64xf16> * tensor<64x128xf16> ->
tt.func @dot_m96_loop(%pa: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pb: !tt.ptr<f16>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %k2 = tt.expand_dims %k {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %kb = tt.broadcast %k2 : tensor<1x64xi32> -> tensor<96x64xi32>
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<96x64x!tt.ptr<f16>>
  %pk = tt.addptr %sa, %kb : tensor<96x64x!tt.ptr<f16>>, tensor<96x64xi32>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<96x128xf32>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c) -> (tensor<96x128xf32>) : i32 {
    %a = tt.load %pk : tensor<96x64x!tt.ptr<f16>>
    %b = tt.load %sb : tensor<64x128x!tt.ptr<f16>>
    // expected-remark @+1 {{NPOT dot pads: A loses direct-to-LDS loads: its lanes do not write contiguous LDS}}
    %d = tt.dot %a, %b, %acc : tensor<96x64xf16> * tensor<64x128xf16> -> tensor<96x128xf32>
    scf.yield %d : tensor<96x128xf32>
  }
  tt.return
}

// -----

// A one-stage loop's loads are not copied to LDS async.
// CHECK-LABEL: @dot_m96_one_stage
// CHECK: tt.dot {{.*}} : tensor<96x64xf16> * tensor<64x128xf16> ->
tt.func @dot_m96_one_stage(%pa: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pb: !tt.ptr<f16>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %k2 = tt.expand_dims %k {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %kb = tt.broadcast %k2 : tensor<1x64xi32> -> tensor<96x64xi32>
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<96x64x!tt.ptr<f16>>
  %pk = tt.addptr %sa, %kb : tensor<96x64x!tt.ptr<f16>>, tensor<96x64xi32>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<96x128xf32>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c) -> (tensor<96x128xf32>) : i32 {
    %a = tt.load %pk : tensor<96x64x!tt.ptr<f16>>
    %b = tt.load %sb : tensor<64x128x!tt.ptr<f16>>
    %d = tt.dot %a, %b, %acc : tensor<96x64xf16> * tensor<64x128xf16> -> tensor<96x128xf32>
    scf.yield %d : tensor<96x128xf32>
  } {tt.num_stages = 1 : i32}
  tt.return
}

// -----

// Two stages of padded 256x128 A and 128x128 B overflow LDS.
// CHECK-LABEL: @dot_m192_lds_overflow
// CHECK: tt.dot {{.*}} : tensor<192x128xf16> * tensor<128x128xf16> ->
tt.func @dot_m192_lds_overflow(%pa: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pb: !tt.ptr<f16>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %k = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %k2 = tt.expand_dims %k {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %kb = tt.broadcast %k2 : tensor<1x128xi32> -> tensor<192x128xi32>
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<192x128x!tt.ptr<f16>>
  %pk = tt.addptr %sa, %kb : tensor<192x128x!tt.ptr<f16>>, tensor<192x128xi32>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<192x128xf32>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c) -> (tensor<192x128xf32>) : i32 {
    %a = tt.load %pk : tensor<192x128x!tt.ptr<f16>>
    %b = tt.load %sb : tensor<128x128x!tt.ptr<f16>>
    %d = tt.dot %a, %b, %acc : tensor<192x128xf16> * tensor<128x128xf16> -> tensor<192x128xf32>
    scf.yield %d : tensor<192x128xf32>
  }
  tt.return
}

// -----

// A pow2 dot in a pipelined loop is unchanged.
// CHECK-LABEL: @dot_pow2_loop
// CHECK: tt.dot {{.*}} : tensor<128x64xf16> * tensor<64x128xf16> ->
tt.func @dot_pow2_loop(%pa: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pb: !tt.ptr<f16>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %k2 = tt.expand_dims %k {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %kb = tt.broadcast %k2 : tensor<1x64xi32> -> tensor<128x64xi32>
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
  %pk = tt.addptr %sa, %kb : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c) -> (tensor<128x128xf32>) : i32 {
    %a = tt.load %pk : tensor<128x64x!tt.ptr<f16>>
    %b = tt.load %sb : tensor<64x128x!tt.ptr<f16>>
    %d = tt.dot %a, %b, %acc : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
    scf.yield %d : tensor<128x128xf32>
  }
  tt.return
}
