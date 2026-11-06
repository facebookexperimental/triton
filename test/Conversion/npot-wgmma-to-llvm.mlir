// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=83' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' | FileCheck %s

// B's mixed-radix registers store to shared memory in 16-byte runs. N=96 issues
// m64n96k16, 3/4 of the FLOPs of its 128x128 pad.
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [64, 96]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 96, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @wgmma_128x96
  // CHECK-COUNT-6: llvm.store {{.*}} : vector<4xi32>, !llvm.ptr<3>
  // CHECK-NOT: llvm.store {{.*}} !llvm.ptr<3>
  // CHECK-COUNT-8: nvg.wgmma %{{.*}} n = 96 : i32
  // CHECK-NOT: nvg.wgmma %
  tt.func @wgmma_128x96(%a: !ttg.memdesc<128x64xf16, #shared, #smem>, %pb: !tt.ptr<f16>) {
    %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x96x!tt.ptr<f16>, #linear>
    %b = tt.load %sb : tensor<64x96x!tt.ptr<f16>, #linear>
    %bs = ttg.local_alloc %b : (tensor<64x96xf16, #linear>) -> !ttg.memdesc<64x96xf16, #shared1, #smem>
    %c = arith.constant dense<0.0> : tensor<128x96xf32, #mma>
    %d = ttng.warp_group_dot %a, %bs, %c : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x96xf16, #shared1, #smem> -> tensor<128x96xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @wgmma_128x128
  // CHECK-COUNT-8: nvg.wgmma %{{.*}} n = 128 : i32
  // CHECK-NOT: nvg.wgmma %
  tt.func @wgmma_128x128(%a: !ttg.memdesc<128x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x128xf16, #shared, #smem>) {
    %c = arith.constant dense<0.0> : tensor<128x128xf32, #mma>
    %d = ttng.warp_group_dot %a, %b, %c : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<128x128xf32, #mma>
    tt.return
  }
}
