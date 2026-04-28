// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas=num-stages=2 --nvws-lower-semaphore=num-stages=2 --tritongpu-partition-loops --nvws-lower-warp-group --tritongpu-schedule-loops=num-stages=2 --tritongpu-pipeline=num-stages=2 | FileCheck %s --check-prefix=PIPE

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // This is the one-slot Q shape from attention backward.  The final read in
  // stage 1 releases the slot consumed by the stage-0 producer in a future
  // iteration.  Because the recurrence distance is one, stage terms cancel;
  // the producer and its first consumer must be ordered after the final read.
  // CHECK-LABEL: @one_slot_recurrence
  // PIPE-LABEL: @one_slot_recurrence
  // PIPE: partition0
  // PIPE: ttg.local_load
  // PIPE: "consume_first"
  // PIPE: scf.for
  // PIPE: ttg.local_load
  // PIPE: ttng.arrive_barrier
  // PIPE: "consume_last"
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_load
  // PIPE: "consume_first"
  tt.func @one_slot_recurrence(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 420 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    scf.for %iv = %lb to %ub step %step : i32 {
      %value = "producer"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : () -> tensor<128x64xf16, #blocked>

      // CHECK: nvws.semaphore.acquire {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      // CHECK: ttg.local_store {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      ttg.local_store %value, %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

      // CHECK: ttg.local_load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      %first = ttg.local_load %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: "consume_first"({{.*}}) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      "consume_first"(%first) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()

      // CHECK: ttg.local_load {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      %last = ttg.local_load %alloc {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      "consume_last"(%last) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
      // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
    } {tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 1, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
