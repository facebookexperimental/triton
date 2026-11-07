// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory -test-print-membar | FileCheck %s

// A warp-scoped mixed-radix convert syncs only its warp, so a later read of
// shared memory written before it still needs a CTA barrier.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#src = #ttg.linear<{register = [[32], [16]], lane = [[1], [2], [4], [8], [0]], warp = [], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
#dst = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @warp_scope
  tt.func @warp_scope(%p: !tt.ptr<i32>) {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #ttg.shared_memory, mutable>
    %c = arith.constant dense<0.0> : tensor<32xf32>
    // CHECK: ttg.local_store
    ttg.local_store %c, %alloc : tensor<32xf32> -> !ttg.memdesc<32xf32, #shared, #ttg.shared_memory, mutable>
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #src>
    // CHECK: ttg.convert_layout {{.*}}allocation.size = 384 : i32
    %0 = ttg.convert_layout %r : tensor<96xi32, #src> -> tensor<96xi32, #dst>
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttg.local_load
    %1 = ttg.local_load %alloc : !ttg.memdesc<32xf32, #shared, #ttg.shared_memory, mutable> -> tensor<32xf32>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<96x!tt.ptr<i32>, #dst>
    tt.store %s, %0 : tensor<96x!tt.ptr<i32>, #dst>
    tt.return
  }
}

// -----

// A convert across warps ends with a CTA barrier, which orders the read.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#src = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16]], warp = [[32]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [192]}>
#dst = #ttg.linear<{register = [[64]], lane = [[2], [4], [8], [16], [32]], warp = [[1]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [192]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @block_scope
  tt.func @block_scope(%p: !tt.ptr<i32>) {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #ttg.shared_memory, mutable>
    %c = arith.constant dense<0.0> : tensor<32xf32>
    // CHECK: ttg.local_store
    ttg.local_store %c, %alloc : tensor<32xf32> -> !ttg.memdesc<32xf32, #shared, #ttg.shared_memory, mutable>
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #src>
    // CHECK: ttg.convert_layout {{.*}}allocation.size = 768 : i32
    %0 = ttg.convert_layout %r : tensor<192xi32, #src> -> tensor<192xi32, #dst>
    // CHECK-NEXT: ttg.local_load
    %1 = ttg.local_load %alloc : !ttg.memdesc<32xf32, #shared, #ttg.shared_memory, mutable> -> tensor<32xf32>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<192x!tt.ptr<i32>, #dst>
    tt.store %s, %0 : tensor<192x!tt.ptr<i32>, #dst>
    tt.return
  }
}

// -----

// An NPOT NVMMA tile sits on its pow2 alloc shape, so its region is exact and
// its store needs no barrier after the A store: two in the loop, not three.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [0, 64]], lane = [[0, 8], [0, 16], [0, 32], [1, 0], [2, 0]], warp = [[4, 0], [8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2, 2], block = []}, outDimSizes = [64, 192]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 192, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @nvmma_pow2_alloc_loop
  tt.func @nvmma_pow2_alloc_loop(%pa: tensor<128x64x!tt.ptr<f16>, #blocked>, %p: !tt.ptr<f16>, %po: !tt.ptr<f32>, %n: i32) {
    %pb = tt.splat %p : !tt.ptr<f16> -> tensor<64x192x!tt.ptr<f16>, #linear>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %acc0 = arith.constant dense<0.000000e+00> : tensor<128x192xf32, #mma>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %acc0) -> (tensor<128x192xf32, #mma>) : i32 {
      %a = tt.load %pa : tensor<128x64x!tt.ptr<f16>, #blocked>
      // CHECK: ttg.barrier local
      // CHECK-NEXT: ttg.local_alloc
      %sa = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // CHECK-NEXT: tt.load
      // CHECK-NEXT: ttg.local_alloc
      %b = tt.load %pb : tensor<64x192x!tt.ptr<f16>, #linear>
      %sb = ttg.local_alloc %b : (tensor<64x192xf16, #linear>) -> !ttg.memdesc<64x192xf16, #shared, #smem>
      ttng.fence_async_shared {bCluster = false}
      %d = ttng.warp_group_dot %sa, %sb, %acc : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x192xf16, #shared, #smem> -> tensor<128x192xf32, #mma>
      scf.yield %d : tensor<128x192xf32, #mma>
    }
    %ps = tt.splat %po : !tt.ptr<f32> -> tensor<128x192x!tt.ptr<f32>, #mma>
    tt.store %ps, %r : tensor<128x192x!tt.ptr<f32>, #mma>
    tt.return
  }
}
