// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas \
// RUN:   --nvws-assign-stage-phase --nvws-lower-semaphore=num-stages=2 \
// RUN:   --tritongpu-partition-loops --nvws-lower-warp-group \
// RUN:   --tritongpu-schedule-loops=num-stages=2 \
// RUN:   --tritongpu-pipeline=num-stages=2 -o /dev/null

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @staged_tokenless_cross_stage_pou
  tt.func @staged_tokenless_cross_stage_pou(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32}
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 991 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // The cross-iteration EMPTY edge goes from stage 1 to stage 0.  POU keeps
    // both acquires inside the inner loop; neither loop carries an async token.
    // The consecutive checks after FULL also prove that no third semaphore or
    // pre-loop acquire was inserted.
    // CHECK-NEXT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
    scf.for %outer = %lb to %ub step %step : i32 {
      // CHECK-NEXT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
      scf.for %inner = %lb to %ub step %step : i32 {
        // CHECK-NEXT: [[P0_TOKEN:%.*]] = nvws.semaphore.acquire [[EMPTY]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
        // CHECK-NEXT: [[P0_BUFFER:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[P0_TOKEN]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
        // CHECK-NEXT: "touch0"([[P0_BUFFER]]) {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
        "touch0"(%alloc) {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
        // CHECK-NEXT: "touch1"([[P0_BUFFER]]) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
        // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[P0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
        "touch1"(%alloc) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
        // CHECK-NEXT: [[P1_TOKEN:%.*]] = nvws.semaphore.acquire [[FULL]] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[P1_BUFFER:%.*]] = nvws.semaphore.buffer [[FULL]], [[P1_TOKEN]] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: "touch2"([[P1_BUFFER]]) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[P1_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        "touch2"(%alloc) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
      // CHECK-NEXT: } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>}
      } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>}
    // CHECK-NEXT: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
