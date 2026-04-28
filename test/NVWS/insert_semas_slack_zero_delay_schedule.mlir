// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  // The q recurrence forces owner 2 one wave after owner 0:
  //   q store {2}, stage 2 -> q load {0}, stage 4
  //   q load {0}, stage 4 -> next q store {2}, distance 1, delay +1
  // The p current handoff has raw delay zero but one wave of solved slack:
  //   p store {0}, stage 4 -> p load {2}, stage 4
  // while the reverse recurrence is tight at delay -1. Schedule projection
  // must not turn the slack current handoff and tight recurrence into opposite
  // static cluster edges.
  // CHECK-LABEL: tt.func @slack_zero_delay_edge_is_not_projected
  // CHECK: [[Q_BACKING:%.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 430 : i32}
  // CHECK: [[Q_EMPTY:%.*]] = nvws.semaphore.create [[Q_BACKING]] released = -1
  // CHECK: [[Q_FULL:%.*]] = nvws.semaphore.create [[Q_BACKING]]
  // CHECK: [[P_BACKING:%.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 431 : i32}
  // CHECK: [[P_EMPTY:%.*]] = nvws.semaphore.create [[P_BACKING]] released = -1
  // CHECK: [[P_FULL:%.*]] = nvws.semaphore.create [[P_BACKING]]
  // CHECK: scf.for
  // CHECK: [[Q_WRITE_BUF:%.*]] = nvws.semaphore.buffer [[Q_EMPTY]], %{{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>}
  // CHECK: ttg.local_store %{{.*}}, [[Q_WRITE_BUF]] {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>}
  // CHECK: [[Q_READ_BUF:%.*]] = nvws.semaphore.buffer [[Q_FULL]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>}
  // CHECK: [[Q_READ:%.*]] = ttg.local_load [[Q_READ_BUF]] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>}
  // CHECK: [[P_WRITE_BUF:%.*]] = nvws.semaphore.buffer [[P_EMPTY]], %{{.*}} {loop.cluster = 1 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>}
  // CHECK: ttg.local_store [[Q_READ]], [[P_WRITE_BUF]] {loop.cluster = 1 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>}
  // CHECK: [[P_READ_BUF:%.*]] = nvws.semaphore.buffer [[P_FULL]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>}
  // CHECK: ttg.local_load [[P_READ_BUF]] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>}
  tt.func @slack_zero_delay_edge_is_not_projected(
      %lb: i32, %ub: i32, %step: i32) {
    %q = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 430 : i32} :
        () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %p = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 431 : i32} :
        () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

    scf.for %iv = %lb to %ub step %step : i32 {
      %q_value = "q_producer"() {loop.cluster = 0 : i32, loop.stage = 2 : i32,
                                  ttg.partition = array<i32: 2>} :
          () -> tensor<1xi32, #blocked>
      ttg.local_store %q_value, %q {loop.cluster = 0 : i32,
                                    loop.stage = 2 : i32,
                                    ttg.partition = array<i32: 2>} :
          tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %q_read = ttg.local_load %q {loop.cluster = 0 : i32,
                                   loop.stage = 4 : i32,
                                   ttg.partition = array<i32: 0>} :
          !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>

      ttg.local_store %q_read, %p {loop.cluster = 0 : i32,
                                   loop.stage = 4 : i32,
                                   ttg.partition = array<i32: 0>} :
          tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %p_read = ttg.local_load %p {loop.cluster = 0 : i32,
                                   loop.stage = 4 : i32,
                                   ttg.partition = array<i32: 2>} :
          !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      "consume"(%p_read) {loop.cluster = 0 : i32, loop.stage = 4 : i32,
                           ttg.partition = array<i32: 2>} :
          (tensor<1xi32, #blocked>) -> ()
    } {tt.scheduled_max_stage = 4 : i32, tt.warp_specialize,
       ttg.partition = array<i32: 0, 2>,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
