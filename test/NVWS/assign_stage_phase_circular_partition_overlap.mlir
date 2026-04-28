// RUN: not triton-opt %s -allow-unregistered-dialect --nvws-assign-stage-phase 2>&1 | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK: error: circular semaphore has overlapping cross-partition physical-slot ownership
  tt.func @reject_overlapping_partition_owned_slots(%lb: i32, %ub: i32,
                                                     %step: i32) {
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 3 : i32, buffer.id = 306 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    %empty = nvws.semaphore.create %base released = -1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %a0 = arith.constant {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      %ta = nvws.semaphore.acquire %empty[%a0] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %ba = nvws.semaphore.buffer %empty[%a0], %ta {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%ba) {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()

      %b0 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %tb = nvws.semaphore.acquire %empty[%b0] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %bb = nvws.semaphore.buffer %empty[%b0], %tb {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%bb) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
    } {tt.scheduled_max_stage = 0 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 12 : i32}
    tt.return
  }
}
