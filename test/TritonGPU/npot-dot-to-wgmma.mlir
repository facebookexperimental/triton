// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -convert-triton-to-tritongpu="target=cuda:90 num-warps=8" -tritongpu-accelerate-matmul | FileCheck %s

// A warpgroup's 96 rows aren't a multiple of m64, so the second goes on N.
// CHECK: #mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 64, 16]}>
// CHECK-LABEL: @dot_192x128
// CHECK: ttng.warp_group_dot {{.*}} -> tensor<192x128xf32, #mma>
tt.func @dot_192x128(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>, %pc: !tt.ptr<f32>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<192x64x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<192x64x!tt.ptr<f16>>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x128x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<192x128xf32>
  %d = tt.dot %a, %b, %c : tensor<192x64xf16> * tensor<64x128xf16> -> tensor<192x128xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<192x128x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<192x128x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK: #mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
// CHECK-LABEL: @dot_256x128
tt.func @dot_256x128(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>, %pc: !tt.ptr<f32>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<256x64x!tt.ptr<f16>>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x128x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<256x128xf32>
  %d = tt.dot %a, %b, %c : tensor<256x64xf16> * tensor<64x128xf16> -> tensor<256x128xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<256x128x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<256x128x!tt.ptr<f32>>
  tt.return
}

// -----

// An A computed in registers stays there, as a mixed-radix dot operand.
// CHECK-LABEL: @aregs_m192
// CHECK: %[[A:.*]] = ttg.convert_layout {{.*}} -> tensor<192x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
// CHECK: ttng.warp_group_dot %[[A]], {{.*}} -> tensor<192x128xf32, #mma>
tt.func @aregs_m192(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>, %pc: !tt.ptr<f32>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<192x64x!tt.ptr<f16>>
  %a0 = tt.load %sa : tensor<192x64x!tt.ptr<f16>>
  %a = arith.addf %a0, %a0 : tensor<192x64xf16>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x128x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<192x128xf32>
  %d = tt.dot %a, %b, %c : tensor<192x64xf16> * tensor<64x128xf16> -> tensor<192x128xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<192x128x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<192x128x!tt.ptr<f32>>
  tt.return
}

// -----

// K96's operands keep their layouts until accelerate-matmul allocs them.
// CHECK-LABEL: @dot_k96
// CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<128x96xf16, {{.*}}> * !ttg.memdesc<96x128xf16, {{.*}}> -> tensor<128x128xf32, #mma>
tt.func @dot_k96(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>, %pc: !tt.ptr<f32>) {
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x96x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<128x96x!tt.ptr<f16>>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<96x128x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<96x128x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x96xf16> * tensor<96x128xf16> -> tensor<128x128xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x128x!tt.ptr<f32>>
  tt.return
}
