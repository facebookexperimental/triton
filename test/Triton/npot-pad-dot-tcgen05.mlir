// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:100 -convert-triton-to-tritongpu="target=cuda:100 num-warps=4" -tritongpu-coalesce -tritongpu-accelerate-matmul -tritongpu-hoist-tmem-alloc -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline 2>/dev/null | FileCheck %s --check-prefix=PIPE
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

// A legal tcgen05 N stays exact: the accumulator's TMEM block covers N with
// a power of 2, and the MMA issues N columns. N=128 is the pow2 control.
// CHECK-LABEL: @dot_f16_npot_n
// CHECK: tt.dot {{.*}} -> tensor<128x96xf32>
// CHECK: tt.dot {{.*}} -> tensor<128x80xf32>
// CHECK: tt.dot {{.*}} -> tensor<128x112xf32>
// CHECK: tt.dot {{.*}} -> tensor<128x144xf32>
// CHECK: tt.dot {{.*}} -> tensor<128x160xf32>
// CHECK: tt.dot {{.*}} -> tensor<128x192xf32>
// CHECK: tt.dot {{.*}} -> tensor<128x128xf32>
// TC-DAG: #[[$B128:.*]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128,
// TC-DAG: #[[$B256:.*]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256,
// TC-LABEL: @dot_f16_npot_n
// TC: tc_gen5_mma {{.*}} !ttg.memdesc<128x96xf32, #[[$B128]], #ttng.tensor_memory, mutable, 128x128>
// TC: tc_gen5_mma {{.*}} !ttg.memdesc<128x80xf32, #[[$B128]], #ttng.tensor_memory, mutable, 128x128>
// TC: tc_gen5_mma {{.*}} !ttg.memdesc<128x112xf32, #[[$B128]], #ttng.tensor_memory, mutable, 128x128>
// TC: tc_gen5_mma {{.*}} !ttg.memdesc<128x144xf32, #[[$B256]], #ttng.tensor_memory, mutable, 128x256>
// TC: tc_gen5_mma {{.*}} !ttg.memdesc<128x160xf32, #[[$B256]], #ttng.tensor_memory, mutable, 128x256>
// TC: tc_gen5_mma {{.*}} !ttg.memdesc<128x192xf32, #[[$B256]], #ttng.tensor_memory, mutable, 128x256>
// TC: tc_gen5_mma {{.*}} !ttg.memdesc<128x128xf32, #[[$B128]], #ttng.tensor_memory, mutable>
// TC-NOT: tt.dot
tt.func @dot_f16_npot_n(%p: !tt.ptr<f32>) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %b0 = arith.constant dense<1.0> : tensor<64x96xf16>
  %c0 = arith.constant dense<0.0> : tensor<128x96xf32>
  %d0 = tt.dot %a, %b0, %c0 : tensor<128x64xf16> * tensor<64x96xf16> -> tensor<128x96xf32>
  %s0 = tt.splat %p : !tt.ptr<f32> -> tensor<128x96x!tt.ptr<f32>>
  tt.store %s0, %d0 : tensor<128x96x!tt.ptr<f32>>
  %b1 = arith.constant dense<1.0> : tensor<64x80xf16>
  %c1 = arith.constant dense<0.0> : tensor<128x80xf32>
  %d1 = tt.dot %a, %b1, %c1 : tensor<128x64xf16> * tensor<64x80xf16> -> tensor<128x80xf32>
  %s1 = tt.splat %p : !tt.ptr<f32> -> tensor<128x80x!tt.ptr<f32>>
  tt.store %s1, %d1 : tensor<128x80x!tt.ptr<f32>>
  %b2 = arith.constant dense<1.0> : tensor<64x112xf16>
  %c2 = arith.constant dense<0.0> : tensor<128x112xf32>
  %d2 = tt.dot %a, %b2, %c2 : tensor<128x64xf16> * tensor<64x112xf16> -> tensor<128x112xf32>
  %s2 = tt.splat %p : !tt.ptr<f32> -> tensor<128x112x!tt.ptr<f32>>
  tt.store %s2, %d2 : tensor<128x112x!tt.ptr<f32>>
  %b3 = arith.constant dense<1.0> : tensor<64x144xf16>
  %c3 = arith.constant dense<0.0> : tensor<128x144xf32>
  %d3 = tt.dot %a, %b3, %c3 : tensor<128x64xf16> * tensor<64x144xf16> -> tensor<128x144xf32>
  %s3 = tt.splat %p : !tt.ptr<f32> -> tensor<128x144x!tt.ptr<f32>>
  tt.store %s3, %d3 : tensor<128x144x!tt.ptr<f32>>
  %b4 = arith.constant dense<1.0> : tensor<64x160xf16>
  %c4 = arith.constant dense<0.0> : tensor<128x160xf32>
  %d4 = tt.dot %a, %b4, %c4 : tensor<128x64xf16> * tensor<64x160xf16> -> tensor<128x160xf32>
  %s4 = tt.splat %p : !tt.ptr<f32> -> tensor<128x160x!tt.ptr<f32>>
  tt.store %s4, %d4 : tensor<128x160x!tt.ptr<f32>>
  %b5 = arith.constant dense<1.0> : tensor<64x192xf16>
  %c5 = arith.constant dense<0.0> : tensor<128x192xf32>
  %d5 = tt.dot %a, %b5, %c5 : tensor<128x64xf16> * tensor<64x192xf16> -> tensor<128x192xf32>
  %s5 = tt.splat %p : !tt.ptr<f32> -> tensor<128x192x!tt.ptr<f32>>
  tt.store %s5, %d5 : tensor<128x192x!tt.ptr<f32>>
  %b6 = arith.constant dense<1.0> : tensor<64x128xf16>
  %c6 = arith.constant dense<0.0> : tensor<128x128xf32>
  %d6 = tt.dot %a, %b6, %c6 : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
  %s6 = tt.splat %p : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.store %s6, %d6 : tensor<128x128x!tt.ptr<f32>>
  tt.return
}

// -----

// At s3 the exact N=96 B operand of a tcgen05 dot goes through a 3-stage
// async copy, like a pow2 N.
// PIPE-LABEL: @loop_f16_n96
// PIPE: ttg.local_alloc : () -> !ttg.memdesc<3x64x96xf16
// PIPE: scf.for
// PIPE: ttng.tc_gen5_mma {{.*}} !ttg.memdesc<64x96xf16, {{.*}} !ttg.memdesc<128x96xf32, {{.*}}, 128x128>
// PIPE: ttg.async_copy_global_to_local {{.*}} -> <64x96xf16
tt.func @loop_f16_n96(%pb: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pd: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  %ld = arith.constant dense<96> : tensor<64x1xi32>
  %step = arith.constant dense<6144> : tensor<64x96xi32>
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %k2 = arith.muli %k1, %ld : tensor<64x1xi32>
  %k3 = tt.broadcast %k2 : tensor<64x1xi32> -> tensor<64x96xi32>
  %j = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32>
  %j1 = tt.expand_dims %j {axis = 0 : i32} : tensor<96xi32> -> tensor<1x96xi32>
  %j2 = tt.broadcast %j1 : tensor<1x96xi32> -> tensor<64x96xi32>
  %o = arith.addi %k3, %j2 : tensor<64x96xi32>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x96x!tt.ptr<f16>>
  %b0 = tt.addptr %sb, %o : tensor<64x96x!tt.ptr<f16>>, tensor<64x96xi32>
  %r:2 = scf.for %i = %c0 to %n step %c64 iter_args(%acc = %c, %q = %b0) -> (tensor<128x96xf32>, tensor<64x96x!tt.ptr<f16>>) : i32 {
    %b = tt.load %q : tensor<64x96x!tt.ptr<f16>>
    %d = tt.dot %a, %b, %acc : tensor<128x64xf16> * tensor<64x96xf16> -> tensor<128x96xf32>
    %nq = tt.addptr %q, %step : tensor<64x96x!tt.ptr<f16>>, tensor<64x96xi32>
    scf.yield %d, %nq : tensor<128x96xf32>, tensor<64x96x!tt.ptr<f16>>
  } {tt.num_stages = 3 : i32}
  %sd = tt.splat %pd : !tt.ptr<f32> -> tensor<128x96x!tt.ptr<f32>>
  tt.store %sd, %r#0 : tensor<128x96x!tt.ptr<f32>>
  tt.return
}

// -----

// B's 176-byte N-major rows take swizzle 0, which no tcgen05 descriptor
// matches; N=96 above is the control.
// CHECK-LABEL: @dot_f16_128x88
// CHECK: tt.dot {{.*}} -> tensor<128x128xf32>
tt.func @dot_f16_128x88(%p: !tt.ptr<f32>) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %b = arith.constant dense<1.0> : tensor<64x88xf16>
  %c = arith.constant dense<0.0> : tensor<128x88xf32>
  // expected-remark @+1 {{NPOT MMAv5 N 88 is not yet supported: B's shared layout has no core-matrix form}}
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x88xf16> -> tensor<128x88xf32>
  %s = tt.splat %p : !tt.ptr<f32> -> tensor<128x88x!tt.ptr<f32>>
  tt.store %s, %d : tensor<128x88x!tt.ptr<f32>>
  tt.return
}

// -----

// NPOT N at blockM=64 still pads.
// CHECK-LABEL: @dot_f16_64x96
// CHECK: tt.dot {{.*}} -> tensor<64x128xf32>
tt.func @dot_f16_64x96(%p: !tt.ptr<f32>) {
  %a = arith.constant dense<1.0> : tensor<64x64xf16>
  %b = arith.constant dense<1.0> : tensor<64x96xf16>
  %c = arith.constant dense<0.0> : tensor<64x96xf32>
  // expected-remark @+1 {{NPOT MMAv5 N with blockM=64 is not yet supported}}
  %d = tt.dot %a, %b, %c : tensor<64x64xf16> * tensor<64x96xf16> -> tensor<64x96xf32>
  %s = tt.splat %p : !tt.ptr<f32> -> tensor<64x96x!tt.ptr<f32>>
  tt.store %s, %d : tensor<64x96x!tt.ptr<f32>>
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

// A K in whole tcgen05.mma K steps stays exact.
// CHECK-LABEL: @dot_f16_k96
// CHECK: tt.dot {{.*}} : tensor<128x96xf16> * tensor<96x128xf16> -> tensor<128x128xf32>
// TC-LABEL: @dot_f16_k96
// TC: ttng.tc_gen5_mma {{.*}} : !ttg.memdesc<128x96xf16, {{.*}}, !ttg.memdesc<96x128xf16,
// TC-NOT: tt.dot
tt.func @dot_f16_k96(%pa: !tt.ptr<f16>, %pd: !tt.ptr<f32>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x96x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<128x96x!tt.ptr<f16>>
  %b = arith.constant dense<1.0> : tensor<96x128xf16>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x96xf16> * tensor<96x128xf16> -> tensor<128x128xf32>
  %sd = tt.splat %pd : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.store %sd, %d : tensor<128x128x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @dot_f16_k24
// CHECK: tt.dot {{.*}} : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
tt.func @dot_f16_k24(%pa: !tt.ptr<f16>, %pd: !tt.ptr<f32>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x24x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<128x24x!tt.ptr<f16>>
  %b = arith.constant dense<1.0> : tensor<24x128xf16>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  // expected-remark @+1 {{NPOT MMAv5 K=24 is unsupported: tcgen05.mma takes K in steps of 16}}
  %d = tt.dot %a, %b, %c : tensor<128x24xf16> * tensor<24x128xf16> -> tensor<128x128xf32>
  %sd = tt.splat %pd : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.store %sd, %d : tensor<128x128x!tt.ptr<f32>>
  tt.return
}
