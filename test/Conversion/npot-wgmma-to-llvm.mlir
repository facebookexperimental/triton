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
  // CHECK: llvm.mlir.constant(33554432 : i32)
  // CHECK-COUNT-8: nvg.wgmma %{{.*}} n = 128 : i32
  // CHECK-NOT: nvg.wgmma %
  tt.func @wgmma_128x128(%a: !ttg.memdesc<128x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x128xf16, #shared, #smem>) {
    %c = arith.constant dense<0.0> : tensor<128x128xf32, #mma>
    %d = ttng.warp_group_dot %a, %b, %c : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<128x128xf32, #mma>
    tt.return
  }
}

// -----

// Dense NPOT allocations stack whole 128B swizzle panels at a stride of rows
// x 128B: K=96 B is 2 panels 12288B apart, so its LBO is 768 x 16B and the
// second warpgroup starts 1536 x 16B in.
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @wgmma_dense_k96
  // CHECK: llvm.mlir.constant(1536 : i32)
  // CHECK-NEXT: llvm.mul
  // CHECK-COUNT-12: llvm.mlir.constant(50331648 : i32)
  // CHECK-NOT: llvm.mlir.constant(50331648 : i32)
  tt.func @wgmma_dense_k96(%a: !ttg.memdesc<128x96xf16, #shared1, #smem>, %b: !ttg.memdesc<96x256xf16, #shared, #smem>) {
    %c = arith.constant dense<0.0> : tensor<128x256xf32, #mma>
    %d = ttng.warp_group_dot %a, %b, %c : !ttg.memdesc<128x96xf16, #shared1, #smem> * !ttg.memdesc<96x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    tt.return
  }
}

// -----

// K-major panels hold all rows: A's second K panel is at 192 x 128B and B's
// at 96 x 128B (1536 and 768 x 16B).
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 96, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#sharedT = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @wgmma_dense_m192_n96
  // CHECK: llvm.mlir.constant(1536 : i32)
  // CHECK: llvm.mlir.constant(768 : i32)
  // CHECK: nvg.wgmma %{{.*}} m = 64 : i32, n = 96 : i32
  tt.func @wgmma_dense_m192_n96(%a: !ttg.memdesc<192x128xf16, #shared, #smem>, %b: !ttg.memdesc<128x96xf16, #sharedT, #smem>) {
    %c = arith.constant dense<0.0> : tensor<192x96xf32, #mma>
    %d = ttng.warp_group_dot %a, %b, %c : !ttg.memdesc<192x128xf16, #shared, #smem> * !ttg.memdesc<128x96xf16, #sharedT, #smem> -> tensor<192x96xf32, #mma>
    tt.return
  }
}
