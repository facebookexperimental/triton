// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-semaphore | FileCheck %s --implicit-check-not=nvws.semaphore

// Arrive multiplicity (r S(n>1)): a single release satisfying a
// pending_count > 1 semaphore arrives N times — first-class via the
// release's arrive_count, passed through to ttng.arrive_barrier
// (fable/integrate-pending-count-plan.md).

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>
module attributes {"ttg.num-warps" = 4 : i32} {
  // One release, multiplicity 2: pending_count 2 is met by a single
  // arrive_barrier with count 2.
  // CHECK-LABEL: @arrive_count_two
  tt.func @arrive_count_two() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
    // CHECK-COUNT-2: ttng.init_barrier %{{.*}}, 2
    %sem = nvws.semaphore.create %buf released = -1 {pending_count = 2 : i32} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %v = ttg.local_load %view {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
    // CHECK: ttng.arrive_barrier {{.*}}, 2 {ttg.partition = array<i32: 0>}
    nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, arrive_count = 2 : i32} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // Asymmetric fan-in: pending_count 3 = one partition arriving twice
  // plus another arriving once.
  // CHECK-LABEL: @arrive_count_mixed_three
  tt.func @arrive_count_mixed_three() {
    %buf2 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[MBAR3:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
    // CHECK-COUNT-2: ttng.init_barrier %{{.*}}, 3
    %sem2 = nvws.semaphore.create %buf2 released = -1 {pending_count = 3 : i32} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    %tok2 = nvws.semaphore.acquire %sem2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: ttng.arrive_barrier {{.*}}, 2 {ttg.partition = array<i32: 0>}
    nvws.semaphore.release %sem2, %tok2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, arrive_count = 2 : i32} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: ttng.arrive_barrier {{.*}}, 1 {ttg.partition = array<i32: 1>}
    nvws.semaphore.release %sem2, %tok2 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>, arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf2 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
