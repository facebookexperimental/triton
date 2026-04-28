// RUN: triton-opt %s -allow-unregistered-dialect --nvws-lower-semaphore --tritongpu-partition-loops | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @cleanup_after_partitioned_user
  tt.func @cleanup_after_partitioned_user(%lb: i32, %ub: i32, %step: i32) {
    %value = arith.constant dense<1> : tensor<1xi32, #blocked>
    %backing = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[MBAR:%.*]] = ttg.local_alloc
    // CHECK: ttng.init_barrier
    %empty = nvws.semaphore.create %backing released = -1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>

    %token = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %view = nvws.semaphore.buffer %empty, %token {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    ttg.local_store %value, %view {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

    // CHECK-NOT: ttng.inval_barrier
    // CHECK-NOT: ttg.local_dealloc [[MBAR]]
    // CHECK: nvws.warp_group
    // CHECK: ttng.wait_barrier
    // CHECK: ttg.local_store
    scf.for %iv = %lb to %ub step %step : i32 {
      "loop_body"(%iv) {ttg.partition = array<i32: 0>} : (i32) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}

    // CHECK: ttng.inval_barrier
    // CHECK: ttg.local_dealloc [[MBAR]]
    ttg.local_dealloc %backing : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
