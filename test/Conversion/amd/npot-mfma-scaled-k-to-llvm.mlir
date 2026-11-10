// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s

// A lane holds 32 K values per f8f6f4 MFMA: 32 fp8 bytes but 16 fp4 bytes,
// since two fp4 share a byte. The transposed layout passes B first.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 64], isTransposed = true}>
#da = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#db = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @fp4_x_fp8
  tt.func @fp4_x_fp8(%p: !tt.ptr<f32>) {
    %a = arith.constant dense<1> : tensor<32x32xi8, #da>
    %b = arith.constant dense<1.0> : tensor<64x32xf8E4M3FN, #db>
    %c = arith.constant dense<0.0> : tensor<32x32xf32, #mma>
    // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} fp8_e4m3, fp4_e2m1, {{.*}} : (vector<8xi32>, vector<4xi32>, vector<16xf32>, i32, i32)
    // CHECK-NOT: rocdl.mfma
    %d = tt.dot_scaled %a, %b, %c lhs = e2m1 rhs = e4m3 {fastMath = false} : tensor<32x32xi8, #da> * tensor<64x32xf8E4M3FN, #db> -> tensor<32x32xf32, #mma>
    %ps = tt.splat %p : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #mma>
    tt.store %ps, %d : tensor<32x32x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// K=96 takes two MFMAs, A's second zero past K's 48 registers per lane.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 64], isTransposed = true}>
#da = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#db = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @fp8_k96
  // CHECK-NOT: llvm.mlir.constant(0 : i8)
  // CHECK: llvm.bitcast {{.*}} : vector<32xi8> to vector<8xi32>
  // CHECK-NOT: llvm.mlir.constant(0 : i8)
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[IDX:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: llvm.insertelement %[[ZERO]], %{{.*}}[%[[IDX]] : i32] : vector<32xi8>
  // CHECK-COUNT-2: rocdl.mfma.scale.f32.32x32x64.f8f6f4
  // CHECK-NOT: rocdl.mfma
  tt.func @fp8_k96(%p: !tt.ptr<f32>) {
    %a = arith.constant dense<1.0> : tensor<32x96xf8E4M3FN, #da>
    %b = arith.constant dense<1.0> : tensor<96x32xf8E4M3FN, #db>
    %c = arith.constant dense<0.0> : tensor<32x32xf32, #mma>
    %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x96xf8E4M3FN, #da> * tensor<96x32xf8E4M3FN, #db> -> tensor<32x32xf32, #mma>
    %ps = tt.splat %p : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #mma>
    tt.store %ps, %d : tensor<32x32x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// MXFP8 K=96: lanes 32-63 shift the scale word a byte, so MFMA k's opsel 2k
// reads block 2k + 1 there; block 3, past K, is 127.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 64], isTransposed = true}>
#da = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#db = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
#sa = #ttg.linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[0, 0], [32, 0]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [64, 3]}>
#sb = #ttg.linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[32, 0], [0, 0]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [64, 3]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @mxfp8_k96
  // CHECK: llvm.mlir.constant(127 : i8)
  // CHECK: llvm.lshr
  // CHECK-COUNT-2: rocdl.mfma.scale.f32.32x32x64.f8f6f4
  // CHECK-NOT: rocdl.mfma
  tt.func @mxfp8_k96(%pas: !tt.ptr<i8>, %pbs: !tt.ptr<i8>, %p: !tt.ptr<f32>) {
    %sas = tt.splat %pas : !tt.ptr<i8> -> tensor<64x3x!tt.ptr<i8>, #sa>
    %as = tt.load %sas : tensor<64x3x!tt.ptr<i8>, #sa>
    %sbs = tt.splat %pbs : !tt.ptr<i8> -> tensor<64x3x!tt.ptr<i8>, #sb>
    %bs = tt.load %sbs : tensor<64x3x!tt.ptr<i8>, #sb>
    %a = arith.constant dense<1.0> : tensor<64x96xf8E4M3FN, #da>
    %b = arith.constant dense<1.0> : tensor<96x64xf8E4M3FN, #db>
    %c = arith.constant dense<0.0> : tensor<64x64xf32, #mma>
    %d = tt.dot_scaled %a scale %as, %b scale %bs, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x96xf8E4M3FN, #da>, tensor<64x3xi8, #sa> * tensor<96x64xf8E4M3FN, #db>, tensor<64x3xi8, #sb> -> tensor<64x64xf32, #mma>
    %ps = tt.splat %p : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #mma>
    tt.store %ps, %d : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}
