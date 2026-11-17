// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-accelerate-matmul | FileCheck %s

#ba = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#lb = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [0, 64]], lane = [[0, 8], [0, 16], [0, 32], [1, 0], [2, 0]], warp = [[4, 0], [8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 5], lane = [2, 2, 2, 2, 2], warp = [2, 2, 2], block = []}, outDimSizes = [64, 320]}>
#lc = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [64, 0], [0, 64]], lane = [[0, 8], [0, 16], [0, 32], [1, 0], [2, 0]], warp = [[4, 0], [8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 5], lane = [2, 2, 2, 2, 2], warp = [2, 2, 2], block = []}, outDimSizes = [128, 320]}>
// An n160 tile of an N-major B would start inside a 64-column swizzle
// pattern, so N is tiled with n64.
// CHECK: #mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
// CHECK: #shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
// CHECK-LABEL: @n_major
// CHECK: ttng.warp_group_dot {{.*}} !ttg.memdesc<64x320xf16, #shared, #smem> -> tensor<128x320xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @n_major(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>, %pc: !tt.ptr<f32>) {
    %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ba>
    %a = tt.load %sa : tensor<128x64x!tt.ptr<f16>, #ba>
    %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x320x!tt.ptr<f16>, #lb>
    %b = tt.load %sb : tensor<64x320x!tt.ptr<f16>, #lb>
    %c = arith.constant dense<0.0> : tensor<128x320xf32, #lc>
    %d = tt.dot %a, %b, %c : tensor<128x64xf16, #ba> * tensor<64x320xf16, #lb> -> tensor<128x320xf32, #lc>
    %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x320x!tt.ptr<f32>, #lc>
    tt.store %sc, %d : tensor<128x320x!tt.ptr<f32>, #lc>
    tt.return
  }
}

// -----

#ba = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#lb = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 32], [0, 64]], lane = [[8, 0], [16, 0], [32, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8], [0, 16]], block = [], inputRadices = {register = [2, 2, 2, 2, 5], lane = [2, 2, 2, 2, 2], warp = [2, 2, 2], block = []}, outDimSizes = [64, 320]}>
#lc = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [64, 0], [0, 64]], lane = [[0, 8], [0, 16], [0, 32], [1, 0], [2, 0]], warp = [[4, 0], [8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 5], lane = [2, 2, 2, 2, 2], warp = [2, 2, 2], block = []}, outDimSizes = [128, 320]}>
// Control: a K-major B's n160 tiles start on 8-row core matrices.
// CHECK: #mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 160, 16]}>
// CHECK: #shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
// CHECK-LABEL: @k_major
// CHECK: ttng.warp_group_dot {{.*}} !ttg.memdesc<64x320xf16, #shared1, #smem> -> tensor<128x320xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @k_major(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>, %pc: !tt.ptr<f32>) {
    %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ba>
    %a = tt.load %sa : tensor<128x64x!tt.ptr<f16>, #ba>
    %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x320x!tt.ptr<f16>, #lb>
    %b = tt.load %sb : tensor<64x320x!tt.ptr<f16>, #lb>
    %c = arith.constant dense<0.0> : tensor<128x320xf32, #lc>
    %d = tt.dot %a, %b, %c : tensor<128x64xf16, #ba> * tensor<64x320xf16, #lb> -> tensor<128x320xf32, #lc>
    %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x320x!tt.ptr<f32>, #lc>
    tt.store %sc, %d : tensor<128x320x!tt.ptr<f32>, #lc>
    tt.return
  }
}
