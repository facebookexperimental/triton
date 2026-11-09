// RUN: split-file %s %t
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %t/nonk0.mlir -split-input-file -convert-triton-to-tritongpu="target=hip:gfx950 num-warps=4 threads-per-warp=64" -tritonamdgpu-accelerate-matmul="gfx-arch=gfx950" -verify-diagnostics | FileCheck %t/nonk0.mlir
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %t/nonk16.mlir -convert-triton-to-tritongpu="target=hip:gfx950 num-warps=4 threads-per-warp=64" -tritonamdgpu-accelerate-matmul="gfx-arch=gfx950 matrix-instruction-size=16" | FileCheck %t/nonk16.mlir

//--- nonk0.mlir

// An NPOT N takes MFMA with padded's warps when their tiles fit its
// power-of-two part.
// CHECK{LITERAL}: #mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
// CHECK-LABEL: @dot_n160
// CHECK: tt.dot {{.*}} -> tensor<1024x160xf32, #mma>
tt.func @dot_n160(%a: tensor<1024x64xbf16>, %pb: !tt.ptr<bf16>, %pc: !tt.ptr<f32>) {
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<64x160x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<64x160x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<1024x160xf32>
  %d = tt.dot %a, %b, %c : tensor<1024x64xbf16> * tensor<64x160xbf16> -> tensor<1024x160xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<1024x160x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<1024x160x!tt.ptr<f32>>
  tt.return
}

// -----

// Where the default 32x32 can't tile an NPOT N, the 16x16 MFMA that does is
// chosen, as the target dot-legality query admits it.
// CHECK{LITERAL}: #mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
// CHECK-LABEL: @dot_n96
// CHECK: tt.dot {{.*}} -> tensor<128x96xf32, #mma>
tt.func @dot_n96(%a: tensor<128x64xbf16>, %pb: !tt.ptr<bf16>, %pc: !tt.ptr<f32>) {
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<64x96x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<64x96x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x96xbf16> -> tensor<128x96xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x96x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x96x!tt.ptr<f32>>
  tt.return
}

// -----

// Where padded's warps would split an odd part, they go on the pow2 parts and
// the odd part is a register repeat: N=80 takes 4x1 warps of 16x16, 5 tiles
// each along N.
// CHECK{LITERAL}: #mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
// CHECK-LABEL: @dot_n80
// CHECK: tt.dot {{.*}} -> tensor<128x80xf32, #mma>
tt.func @dot_n80(%a: tensor<128x64xbf16>, %pb: !tt.ptr<bf16>, %pc: !tt.ptr<f32>) {
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<64x80x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<64x80x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x80xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x80xbf16> -> tensor<128x80xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x80x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x80x!tt.ptr<f32>>
  tt.return
}

// -----

// The default 32x32 keeps N=160's odd part 5 in registers with 4x1 warps.
// CHECK{LITERAL}: #mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
// CHECK-LABEL: @dot_m128_n160
// CHECK: tt.dot {{.*}} -> tensor<128x160xf32, #mma>
tt.func @dot_m128_n160(%a: tensor<128x64xbf16>, %pb: !tt.ptr<bf16>, %pc: !tt.ptr<f32>) {
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<64x160x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<64x160x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x160xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x160xbf16> -> tensor<128x160xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x160x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x160x!tt.ptr<f32>>
  tt.return
}

// -----

// 64x96's pow2 parts hold only two 32x32 warps, so 16x16 warps split 2x2
// across them and N=96 takes 3 tiles per warp.
// CHECK{LITERAL}: #mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
// CHECK-LABEL: @dot_m64_n96
// CHECK: tt.dot {{.*}} -> tensor<64x96xf32, #mma>
tt.func @dot_m64_n96(%a: tensor<64x64xbf16>, %pb: !tt.ptr<bf16>, %pc: !tt.ptr<f32>) {
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<64x96x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<64x96x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<64x96xf32>
  %d = tt.dot %a, %b, %c : tensor<64x64xbf16> * tensor<64x96xbf16> -> tensor<64x96xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<64x96x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<64x96x!tt.ptr<f32>>
  tt.return
}

// -----

// A chain head with N=16 takes two tiles per warp along M.
tt.func @chain_head_m192(%pa: !tt.ptr<bf16>, %b: tensor<64x16xbf16>, %v: tensor<16x64xbf16>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<192x64x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<192x64x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<192x16xf32>
  // expected-error @+1 {{NPOT dot is not yet supported on MFMA with 2x1 tiles per warp}}
  %s = tt.dot %a, %b, %c : tensor<192x64xbf16> * tensor<64x16xbf16> -> tensor<192x16xf32>
  %p = arith.truncf %s : tensor<192x16xf32> to tensor<192x16xbf16>
  %c2 = arith.constant dense<0.0> : tensor<192x64xf32>
  %o = tt.dot %p, %v, %c2 : tensor<192x16xbf16> * tensor<16x64xbf16> -> tensor<192x64xf32>
  tt.return
}

// -----

// Control: a pow2 dot.
// CHECK{LITERAL}: #mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
// CHECK-LABEL: @dot_pow2
// CHECK: tt.dot {{.*}} -> tensor<128x128xf32, #mma>
tt.func @dot_pow2(%a: tensor<128x64xbf16>, %b: tensor<64x128xbf16>, %pc: !tt.ptr<f32>) {
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x128x!tt.ptr<f32>>
  tt.return
}

//--- nonk16.mlir

// At nonKDim 16, two warps fit N=96's power-of-two part 32.
// CHECK{LITERAL}: #mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
// CHECK-LABEL: @dot_n96
// CHECK: tt.dot {{.*}} -> tensor<128x96xf32, #mma>
tt.func @dot_n96(%a: tensor<128x64xbf16>, %pb: !tt.ptr<bf16>, %pc: !tt.ptr<f32>) {
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<64x96x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<64x96x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x96xbf16> -> tensor<128x96xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x96x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x96x!tt.ptr<f32>>
  tt.return
}
