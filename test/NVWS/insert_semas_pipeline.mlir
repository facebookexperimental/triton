// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s --check-prefix=SEMA
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas=num-stages=2 --nvws-semaphore-optimize=num-stages=2 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore=num-stages=2 --tritongpu-partition-loops --nvws-lower-warp-group --tritongpu-schedule-loops=num-stages=2 --tritongpu-pipeline=num-stages=2 -cse | FileCheck %s --check-prefix=PIPE
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas='num-stages=2 use-meta-partitioner=true' --nvws-semaphore-optimize=num-stages=2 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore=num-stages=2 --tritongpu-partition-loops --nvws-lower-warp-group --tritongpu-schedule-loops='num-stages=2 use-meta-ws=true' --tritongpu-pipeline=num-stages=2 -cse | FileCheck %s --check-prefix=PIPE
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas=num-stages=4 --nvws-semaphore-optimize=num-stages=4 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore=num-stages=4 --tritongpu-partition-loops --nvws-lower-warp-group --tritongpu-schedule-loops=num-stages=4 --tritongpu-pipeline=num-stages=4 -cse | FileCheck %s --check-prefix=PIPE
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas=num-stages=2 --nvws-semaphore-optimize=num-stages=2 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore=num-stages=2 -cse | FileCheck %s --check-prefix=LOWER --implicit-check-not=nvws.descriptor_load
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas=num-stages=2 --nvws-semaphore-optimize=num-stages=2 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore=num-stages=2 --tritongpu-partition-loops | FileCheck %s --check-prefix=PARTITION

// Authored buffer depths and schedules survive both scheduling paths. Keep the
// nested startup relay, one-slot recurrence order, and four-slot backpressure.

#reg1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#reg = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory
#tm = #ttng.tensor_memory
!acc = !ttg.memdesc<128x128xf32, #tmem, #tm, mutable>
!tile = tensor<128x128xf32, #reg>
!lhs = !ttg.memdesc<128x64xf16, #shared, #smem>
!rhs = !ttg.memdesc<64x128xf16, #shared, #smem>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // PIPE-LABEL: @scheduled_reader_first
  // PIPE: [[PIPE_TMEM:%.*]] = ttng.tmem_alloc
  // PIPE: [[PIPE_READY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // PIPE: [[PIPE_TO_MMA:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // PIPE: ttg.warp_specialize
  // PIPE: default {
  // PIPE: [[PIPE_ENTRY:%.*]] = ttg.memdesc_index [[PIPE_READY]][
  // PIPE-NEXT: ttng.wait_barrier [[PIPE_ENTRY]],
  // PIPE-NEXT: {{.*}}scf.for
  // The pipeliner peels the first correction into the prologue. The enclosing
  // reservation must arrive before its first wait/read, not just a later wait
  // inside the recurrence. CSE reuses the entry arrival's barrier view.
  // PIPE-NOT: ttng.wait_barrier
  // PIPE-NOT: ttng.tmem_load
  // PIPE: [[PIPE_RELAY:%.*]] = ttg.memdesc_index [[PIPE_READY]][
  // PIPE-NEXT: ttng.arrive_barrier [[PIPE_RELAY]], 1
  // PIPE-NOT: scf.for
  // PIPE-NOT: ttng.wait_barrier
  // PIPE-NOT: ttng.tmem_load
  // PIPE: ttng.wait_barrier [[PIPE_RELAY]],
  // PIPE-NEXT: [[PIPE_FIRST_TILE:%.*]] = ttg.memdesc_index [[PIPE_TMEM]][
  // PIPE-NEXT: {{.*}}ttng.tmem_load [[PIPE_FIRST_TILE]][]
  // PIPE-NEXT: [[PIPE_FIRST_VALUE:%.*]] = math.exp2
  // PIPE-NEXT: {{.*}}ttng.tmem_store [[PIPE_FIRST_VALUE]], [[PIPE_FIRST_TILE]][]
  // PIPE-NEXT: [[PIPE_FIRST_DONE:%.*]] = ttg.memdesc_index [[PIPE_TO_MMA]][
  // PIPE-NEXT: ttng.arrive_barrier [[PIPE_FIRST_DONE]], 1
  // The inner recurrence keeps the same wait/correction/completion protocol.
  // PIPE-NEXT: [[PIPE_INNER:%[a-zA-Z0-9_]+]]:2 = scf.for
  // PIPE: [[PIPE_WAIT:%.*]] = ttg.memdesc_index [[PIPE_READY]][
  // PIPE-NEXT: ttng.wait_barrier [[PIPE_WAIT]],
  // PIPE-NEXT: [[PIPE_TILE:%.*]] = ttg.memdesc_index [[PIPE_TMEM]][
  // PIPE-NEXT: {{.*}}ttng.tmem_load [[PIPE_TILE]][]
  // PIPE-NEXT: [[PIPE_VALUE:%.*]] = math.exp2
  // PIPE-NEXT: {{.*}}ttng.tmem_store [[PIPE_VALUE]], [[PIPE_TILE]][]
  // PIPE-NEXT: [[PIPE_DONE:%.*]] = ttg.memdesc_index [[PIPE_TO_MMA]][
  // PIPE-NEXT: ttng.arrive_barrier [[PIPE_DONE]], 1
  // PIPE: scf.yield
  // The final observer reacquires READY at the returned slot before the next
  // outer iteration inherits that slot and the updated phase state.
  // PIPE: [[PIPE_FINAL_WAIT:%.*]] = ttg.memdesc_index [[PIPE_READY]][[[PIPE_INNER]]#0]
  // PIPE-NEXT: ttng.wait_barrier [[PIPE_FINAL_WAIT]],
  // PIPE-NEXT: [[PIPE_FINAL_TILE:%.*]] = ttg.memdesc_index [[PIPE_TMEM]][[[PIPE_INNER]]#0]
  // PIPE-NEXT: [[PIPE_FINAL_VALUE:%.*]], {{%.*}} = ttng.tmem_load [[PIPE_FINAL_TILE]][]
  // PIPE-NEXT: "consume"([[PIPE_FINAL_VALUE]])
  // PIPE-NEXT: scf.yield {{%.*}}, [[PIPE_INNER]]#0, {{%.*}} : !ttg.async.token, i32, i32
  // SEMA-LABEL: @scheduled_reader_first
  tt.func @scheduled_reader_first(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    // SEMA: [[READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.acquire [[READY]]
    // SEMA: scf.for {{.*}} iter_args([[OUTER_IN:%.*]] = [[ENTRY]])
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      // Scheduled reader and MMA stages require the same incoming-permit relay.
      // SEMA: nvws.semaphore.release [[READY]], [[OUTER_IN]] [#nvws.async_op<none>]
      // SEMA-NEXT: {{.*}}scf.for
      %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
        // SEMA: [[INNER_TOKEN:%.*]] = nvws.semaphore.acquire [[READY]]
        // SEMA: ttng.tmem_load
        %value, %read = ttng.tmem_load %acc[%carry] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !acc -> !tile
        %corrected = math.exp2 %value {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !tile
        %written = ttng.tmem_store %corrected, %acc[%read], %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !tile -> !acc
        // SEMA: nvws.semaphore.release [[TO_MMA:%.*]], [[INNER_TOKEN]] [#nvws.async_op<none>]
        // SEMA: [[MMA_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_MMA]]
        // SEMA: ttng.tc_gen5_mma
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %use_acc, %true {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        // SEMA: nvws.semaphore.release [[READY]], [[MMA_TOKEN]] [#nvws.async_op<tc5mma>]
        scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
      } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      // SEMA: [[RETURNED:%.*]] = nvws.semaphore.acquire [[READY]]
      %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      // SEMA: "consume"
      "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
      // SEMA: scf.yield {{.*}}[[RETURNED]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
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
  // This is the one-slot Q shape from attention backward. The final read at
  // loop.stage 1 releases the slot reused by the loop.stage 0 store in a future
  // iteration. Because the loop-carried dependency distance is one, the final
  // read and next store execute in the same pipelined iteration; loop.cluster
  // must order the store and its first consumer after the final read.
  // SEMA-LABEL: @one_slot_recurrence
  tt.func @one_slot_recurrence(%lb: i32, %ub: i32, %step: i32) {
    // SEMA: [[V1:%.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 420 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // SEMA: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // SEMA: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 420 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    scf.for %iv = %lb to %ub step %step : i32 {
      %value = "producer"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : () -> tensor<128x64xf16, #blocked>

      // SEMA: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %value, %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

      // SEMA: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // SEMA: [[V6:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[V7:%.*]] = nvws.semaphore.buffer [[V3]], [[V6]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_load [[V7]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %first = ttg.local_load %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // SEMA: "consume_first"({{.*}}) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      "consume_first"(%first) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()

      // SEMA: ttg.local_load [[V7]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %last = ttg.local_load %alloc {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // SEMA: nvws.semaphore.release [[V2]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // SEMA: "consume_last"({{.*}}) {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      "consume_last"(%last) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
    } {tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 1, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
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
  // SEMA-LABEL: @legal_cross_partition_backpressure
  tt.func @legal_cross_partition_backpressure(%lb: i32, %ub: i32, %step: i32) {
    // SEMA: [[V1:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 422 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    // SEMA: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 15 {pending_count = 1 : i32} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>
    // SEMA: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>
    // SEMA-NOT: iter_args
    %a = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 422 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 422 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    scf.for %iv = %lb to %ub step %step : i32 {
      %av = "producer_a"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : () -> tensor<128x64xf16, #blocked>
      // SEMA: [[V4:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[V5:%.*]] = nvws.semaphore.acquire [[V2]][[[V4]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[V7:%.*]] = nvws.semaphore.buffer [[V2]][[[V4]]], [[V5]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V7]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %av, %a {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: nvws.semaphore.release [[V3]][[[V4]]], [[V5]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %bv = "producer_b"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : () -> tensor<128x64xf16, #blocked>
      // SEMA: [[V9:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // SEMA: [[V10:%.*]] = nvws.semaphore.acquire [[V2]][[[V9]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[V12:%.*]] = nvws.semaphore.buffer [[V2]][[[V9]]], [[V10]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V12]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %bv, %b {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: nvws.semaphore.release [[V3]][[[V9]]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token

      // SEMA: [[V14:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} 0 : i32
      // SEMA: [[V15:%.*]] = nvws.semaphore.acquire [[V3]][[[V14]]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[V17:%.*]] = nvws.semaphore.buffer [[V3]][[[V14]]], [[V15]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: [[V18:%.*]] = ttg.local_load [[V17]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %br = ttg.local_load %b {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // SEMA: nvws.semaphore.release [[V2]][[[V14]]], [[V15]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // SEMA: "consume_b"([[V18]]) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      "consume_b"(%br) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> ()
      // SEMA: [[V20:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} -1 : i32
      // SEMA: [[V21:%.*]] = nvws.semaphore.acquire [[V3]][[[V20]]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[V23:%.*]] = nvws.semaphore.buffer [[V3]][[[V20]]], [[V21]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: [[V24:%.*]] = ttg.local_load [[V23]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %ar = ttg.local_load %a {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // SEMA: nvws.semaphore.release [[V2]][[[V20]]], [[V21]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // SEMA: "consume_a"([[V24]]) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      "consume_a"(%ar) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> ()
    } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // PIPE-LABEL: @post_ws_tmem_read_tag
  // SEMA-LABEL: @post_ws_tmem_read_tag
  // LOWER-LABEL: @post_ws_tmem_read_tag
  // PARTITION-LABEL: @post_ws_tmem_read_tag
  tt.func @post_ws_tmem_read_tag(
      %ub: i32,
      %lhs: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf16, #shared1, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // SEMA: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // SEMA-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32}
    // SEMA-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32}
    // SEMA-NEXT: [[HELD:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // SEMA-NEXT: [[INIT_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[HELD]]
    // SEMA-NEXT: ttng.tmem_store %{{.*}}, [[INIT_BUF]][], %{{.*}}
    // LOWER: [[READ:%.*]] = ttg.memdesc_index {{.*}} : !ttg.memdesc<1x128x128xf32
    // LOWER: ttng.tmem_store {{.*}}, [[READ]][]
    %init = ttng.tmem_store %cst, %acc[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // The same semaphore token remains live across the loop. Every MMA uses
    // its buffer, and one release after the loop tracks the final MMA.
    // SEMA-NEXT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
    // PARTITION: nvws.warp_group
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%carry = %init) -> (!ttg.async.token) : i32 {
      // SEMA-NEXT: [[BODY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[HELD]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, [[BODY_BUF]][], %{{.*}}, %{{.*}} {ttg.partition = array<i32: 1>}
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%carry], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 1>} %mma : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // SEMA: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // SEMA-NEXT: nvws.semaphore.release [[FULL]], [[HELD]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // SEMA-NEXT: [[READ:%.*]] = nvws.semaphore.acquire [[FULL]]
    // SEMA-NEXT: [[READ_BUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[READ]]
    // SEMA-NEXT: %{{.*}}, %{{.*}} = ttng.tmem_load [[READ_BUF]][]
    // LOWER: ttng.tc_gen5_commit {{.*}} {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // LOWER: ttng.wait_barrier {{.*}} :
    // LOWER: [[OUT:%.*]], {{%.*}} = ttng.tmem_load [[READ]][]
    // PARTITION: ttng.tc_gen5_commit {{.*}} {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // PARTITION: nvws.warp_group.return
    // PARTITION: [[POST_READ:%.*]] = ttg.memdesc_index {{.*}} : !ttg.memdesc<1x128x128xf32
    // PARTITION: [[POST_OUT:%.*]], {{%.*}} = ttng.tmem_load [[POST_READ]][]
    %out, %load_tok = ttng.tmem_load %acc[%loop] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // The post-loop read is the last access; no release of [[EMPTY]] follows it.
    // SEMA-NOT: nvws.semaphore.release
    // LOWER-NEXT: "use"([[OUT]])
    // PARTITION-NEXT: "use"([[POST_OUT]])
    "use"(%out) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}
