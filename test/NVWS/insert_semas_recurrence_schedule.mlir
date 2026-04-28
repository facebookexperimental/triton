// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas=num-stages=2 --nvws-lower-semaphore=num-stages=2 --tritongpu-partition-loops --nvws-lower-warp-group --tritongpu-schedule-loops=num-stages=2 --tritongpu-pipeline=num-stages=2 | FileCheck %s --check-prefix=PIPE

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // This is the one-slot Q shape from attention backward. The final read at
  // loop.stage 1 releases the slot reused by the loop.stage 0 store in a future
  // iteration. Because the loop-carried dependency distance is one, the final
  // read and next store execute in the same pipelined iteration; loop.cluster
  // must order the store and its first consumer after the final read.
  // CHECK-LABEL: @one_slot_recurrence
  // PIPE-LABEL: @one_slot_recurrence
  // PIPE: partition0
  // PIPE: ttg.local_load
  // PIPE: "consume_first"
  // PIPE: scf.for
  // PIPE: ttg.local_load
  // PIPE: ttng.arrive_barrier
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_load
  // PIPE: "consume_first"
  // PIPE: "consume_last"
  tt.func @one_slot_recurrence(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 420 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 420 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    scf.for %iv = %lb to %ub step %step : i32 {
      %value = "producer"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : () -> tensor<128x64xf16, #blocked>

      // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %value, %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V3]], [[V6]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[V7]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %first = ttg.local_load %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: "consume_first"({{.*}}) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      "consume_first"(%first) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()

      // CHECK: ttg.local_load [[V7]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %last = ttg.local_load %alloc {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: "consume_last"({{.*}}) {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      "consume_last"(%last) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
    } {tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 1, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // This is the same ownership cycle shifted across the two partitions. The
  // EMPTY handoff requires owner delay +1 and the FULL handoff contributes -1,
  // so the cycle is feasible but both handoffs meet in the same retimed wave.
  // Cluster legalization must still put the final read before the next write.
  // CHECK-LABEL: @retimed_zero_delay_cycle
  // PIPE-LABEL: @retimed_zero_delay_cycle
  // PIPE: partition0
  // PIPE: ttg.local_load
  // PIPE: "consume_first"
  // PIPE: scf.for
  // PIPE: ttg.local_load
  // PIPE: ttng.arrive_barrier
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_load
  // PIPE: "consume_first"
  // PIPE: "consume_last"
  tt.func @retimed_zero_delay_cycle(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 423 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 423 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    scf.for %iv = %lb to %ub step %step : i32 {
      %value = "producer"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : () -> tensor<128x64xf16, #blocked>
      // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %value, %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V3]], [[V6]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[V7]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %first = ttg.local_load %alloc {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: "consume_first"({{.*}}) {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      "consume_first"(%first) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()

      // CHECK: ttg.local_load [[V7]] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %last = ttg.local_load %alloc {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: "consume_last"({{.*}}) {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
      "consume_last"(%last) {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 1, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}
