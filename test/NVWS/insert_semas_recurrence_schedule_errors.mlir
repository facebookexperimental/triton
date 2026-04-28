// RUN: not triton-opt %s -allow-unregistered-dialect --nvws-insert-semas 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The EMPTY handoff from partition 1 to partition 3 requires one iteration
  // of owner delay, while the reverse FULL handoff has zero delay. Their
  // positive cycle cannot be satisfied by cross-partition backpressure.
  // CHECK: error: nvws-insert-semas: fixed loop.stage assignments form an unsatisfiable semaphore handoff cycle (cycle requires 1 additional pipeline iteration)
  // CHECK-DAG: note: handoff {1} -> {3} has producer loop.stage 2, consumer loop.stage 0, loop-carried dependency distance 1, and required delay 1
  // CHECK-DAG: note: handoff {3} -> {1} has producer loop.stage 0, consumer loop.stage 0, loop-carried dependency distance 0, and required delay 0
  tt.func @invalid_loop_carried_schedule(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 421 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      %value = "producer"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : () -> tensor<128x64xf16, #blocked>
      ttg.local_store %value, %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %first = ttg.local_load %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      "consume_first"(%first) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
      %last = ttg.local_load %alloc {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      "consume_last"(%last) {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 1, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
