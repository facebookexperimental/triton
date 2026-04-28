// RUN: not triton-opt %s -allow-unregistered-dialect --nvws-assign-stage-phase 2>&1 | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // D = 4, A = 2, and G = 2.  X owns class 1 in partition 1.
  // Y and Z both own class 0, but from partitions 2 and 1 respectively.
  // The per-partition phase-split proof accepts X/Z as disjoint; the global
  // partition-ownership proof must reject the Y/Z overlap.
  // CHECK: error: circular semaphore has overlapping cross-partition physical-slot ownership
  tt.func @reject_split_phase_cross_partition_slot_overlap(
      %lb: i32, %ub: i32, %step: i32) {
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 307 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<4x1xi32, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %base released = -1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]>
    %driver = nvws.semaphore.create %base released = -1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %zd0 = arith.constant {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %td0 = nvws.semaphore.acquire %driver[%zd0] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %bd0 = nvws.semaphore.buffer %driver[%zd0], %td0 {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%bd0) {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()

      %x = arith.constant {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %tx = nvws.semaphore.acquire %sem[%x] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token

      %zd1 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %td1 = nvws.semaphore.acquire %driver[%zd1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %bd1 = nvws.semaphore.buffer %driver[%zd1], %td1 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%bd1) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()

      %y = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      %ty = nvws.semaphore.acquire %sem[%y] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token

      %z = arith.constant {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %tz = nvws.semaphore.acquire %sem[%z] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    } {tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 13 : i32}
    tt.return
  }
}
