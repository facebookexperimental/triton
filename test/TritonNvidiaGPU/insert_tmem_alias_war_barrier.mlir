// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-insert-tmem-alias-war-barrier | FileCheck %s

// Cross-warp overlap: an f32 qk read reused as an aliased f16 P store (different
// frame -> row-level footprint overlap across warps) with no barrier between.
// A task-scoped ttg.barrier must be inserted between the read and the store.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @aliased_f32_read_f16_store
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: ttng.tmem_store
  tt.func @aliased_f32_read_f16_store(%val: tensor<128x128xf16, #linear>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%c1] : !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %qk = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttng.tmem_store %val, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// A dynamic view offset cannot prove non-aliasing and must conservatively keep
// the WAR protection.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @dynamic_index_conservative_insert
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: ttng.tmem_store
  tt.func @dynamic_index_conservative_insert(%val: tensor<128x128xf16, #linear>, %idx: i32, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%idx] : !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %read = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttng.tmem_store %val, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Same-frame accumulator read-modify-write (f32 read + f32 store, same memdesc):
// compared at (row, col) cell granularity; each warp only rewrites its own
// cells -> no cross-warp overlap -> NO barrier inserted.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @accumulator_rmw_no_barrier
  // CHECK-NOT: ttg.barrier
  tt.func @accumulator_rmw_no_barrier(%val: tensor<128x128xf32, #linear>, %pred: i1) {
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc = ttng.tmem_load %root : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttng.tmem_store %val, %root, %pred : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Already guarded by a barrier -> idempotent: exactly one barrier, no extra.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @already_guarded_idempotent
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NOT: ttg.barrier
  tt.func @already_guarded_idempotent(%val: tensor<128x128xf16, #linear>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%c1] : !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %qk = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttg.barrier all
    ttng.tmem_store %val, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// A named bar.sync covering all eight task warps is sufficient. The pass must
// preserve it without adding a second barrier.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @full_named_barrier_no_insert
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.wait_barrier_named
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NOT: ttg.barrier
  tt.func @full_named_barrier_no_insert(%val: tensor<128x128xf16, #linear>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bar = arith.constant 10 : i32
    %threads = arith.constant 256 : i32
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%c1] : !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %qk = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttng.wait_barrier_named %bar, %threads : i32, i32
    ttng.tmem_store %val, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Four of eight warps are insufficient, so a full-task barrier is still
// required after the partial named barrier.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @partial_named_barrier_still_inserts
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.wait_barrier_named
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: ttng.tmem_store
  tt.func @partial_named_barrier_still_inserts(%val: tensor<128x128xf16, #linear>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bar = arith.constant 10 : i32
    %threads = arith.constant 128 : i32
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%c1] : !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %qk = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttng.wait_barrier_named %bar, %threads : i32, i32
    ttng.tmem_store %val, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// A producer/consumer mbarrier does not rendezvous the eight consumer warps.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @producer_consumer_barrier_still_inserts
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.wait_barrier
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: ttng.tmem_store
  tt.func @producer_consumer_barrier_still_inserts(%val: tensor<128x128xf16, #linear>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%c1] : !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %qk = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.tmem_store %val, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// A later safe load must not replace an earlier hazardous pending read.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @multiple_pending_reads
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_load
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: ttng.tmem_store
  tt.func @multiple_pending_reads(%val: tensor<128x128xf16, #linear>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%c1] : !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %hazardous = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    %safe = ttng.tmem_load %wr : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf16, #linear>
    ttng.tmem_store %val, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// A non-overlapping store must not discard a pending read needed by a later
// aliased store.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @safe_store_keeps_pending_read
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: ttng.tmem_store
  tt.func @safe_store_keeps_pending_read(%f32: tensor<128x128xf32, #linear>, %f16: tensor<128x128xf16, #linear>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%c1] : !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %hazardous = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttng.tmem_store %f32, %rd, %pred : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %f16, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Different indexed views of one root can be physically disjoint even when
// their element types differ.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @disjoint_indexed_views_no_insert
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NOT: ttg.barrier
  tt.func @disjoint_indexed_views_no_insert(%val: tensor<128x128xf16, #linear>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<256x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<256x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<4x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%c2] : !ttg.memdesc<4x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %read = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttng.tmem_store %val, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
