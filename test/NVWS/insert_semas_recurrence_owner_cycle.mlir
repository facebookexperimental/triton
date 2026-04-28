// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas=num-stages=4 --nvws-lower-semaphore=num-stages=4 --tritongpu-partition-loops --nvws-lower-warp-group --tritongpu-schedule-loops=num-stages=4 --tritongpu-pipeline=num-stages=4 | FileCheck %s --check-prefix=PIPE

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Two fresh writes advance a four-slot physical ring on every iteration.
  // Each logical buffer therefore reuses its slots after two iterations. Its
  // stage-3 EMPTY release followed by a stage-0 reacquire has required owner
  // delay +1, but the reverse FULL handoff has delay -3. The complete owner
  // cycle has delay -2 and is legal: the producer may block while the
  // independent consumer releases the old slot.
  // The pass folds both logical buffers onto one physical ring alloc and gives
  // every protocol op an explicit slot-offset operand: offset 0 rides the
  // current fresh-write cursor, while buffer a's consumer trio sits one fresh
  // write behind it (offset -1) because b's store advanced the cursor between
  // a's store and a's load.
  // CHECK-LABEL: @legal_cross_partition_backpressure
  // PIPE-LABEL: @legal_cross_partition_backpressure
  // PIPE: ttg.warp_specialize
  // PIPE: default {
  // PIPE: scf.for
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_load
  // PIPE: partition0
  // PIPE: scf.for
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_store
  // PIPE: partition1
  // PIPE: scf.for
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_store
  tt.func @legal_cross_partition_backpressure(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 422 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>
    // CHECK-NOT: iter_args
    %a = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 422 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 422 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    scf.for %iv = %lb to %ub step %step : i32 {
      %av = "producer_a"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : () -> tensor<128x64xf16, #blocked>
      // CHECK: [[V4:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]][[[V4]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V6:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V2]][[[V6]]], [[V5]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V7]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %av, %a {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: [[V8:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: nvws.semaphore.release [[V3]][[[V8]]], [[V5]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %bv = "producer_b"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : () -> tensor<128x64xf16, #blocked>
      // CHECK: [[V9:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V2]][[[V9]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V11:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V2]][[[V11]]], [[V10]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V12]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %bv, %b {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: [[V13:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: nvws.semaphore.release [[V3]][[[V13]]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token

      // CHECK: [[V14:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V3]][[[V14]]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V16:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: [[V17:%.*]] = nvws.semaphore.buffer [[V3]][[[V16]]], [[V15]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: [[V18:%.*]] = ttg.local_load [[V17]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %br = ttg.local_load %b {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: [[V19:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: nvws.semaphore.release [[V2]][[[V19]]], [[V15]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: "consume_b"([[V18]]) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      "consume_b"(%br) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> ()
      // CHECK: [[V20:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} -1 : i32
      // CHECK: [[V21:%.*]] = nvws.semaphore.acquire [[V3]][[[V20]]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V22:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} -1 : i32
      // CHECK: [[V23:%.*]] = nvws.semaphore.buffer [[V3]][[[V22]]], [[V21]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: [[V24:%.*]] = ttg.local_load [[V23]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %ar = ttg.local_load %a {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: [[V25:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} -1 : i32
      // CHECK: nvws.semaphore.release [[V2]][[[V25]]], [[V21]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: "consume_a"([[V24]]) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      "consume_a"(%ar) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> ()
    } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
