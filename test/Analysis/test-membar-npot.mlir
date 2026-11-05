// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory -test-print-membar | FileCheck %s

// A warp-scoped Product convert syncs only its warp, so a later read of
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
