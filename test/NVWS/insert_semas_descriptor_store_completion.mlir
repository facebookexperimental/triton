// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s --check-prefix=SEMA
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --verify-each=false --nvws-insert-semas --nvws-lower-semaphore --triton-nvidia-tma-lowering -cse | FileCheck %s --check-prefix=LOWER

// TMA lowering does not propagate partition attrs to its new helper ops, so
// this synthetic pre-partition fixture disables per-pass verification only
// for the cross-pass order check. The production pipeline partitions first.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // SEMA-LABEL: @direct_descriptor_store_completion
  // LOWER-LABEL: sym_name = "direct_descriptor_store_completion"
  // LOWER: ttng.async_tma_copy_local_to_global
  // LOWER-NEXT: ttng.async_tma_store_wait
  // LOWER: ttng.arrive_barrier
  tt.func @direct_descriptor_store_completion(%desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %i: i32, %lb: i32, %ub: i32, %step: i32) {
    // SEMA: [[V1:%.*]] = ttg.local_alloc {buffer.id = 600 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // SEMA: [[EMPTY:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32}
    // SEMA-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32}
    %alloc = ttg.local_alloc {buffer.id = 600 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // SEMA-NEXT: [[LOOP:%.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[WRITE:%.*]] = [[ENTRY]]) -> (!ttg.async.token)  : i32 {
    scf.for %iv = %lb to %ub step %step : i32 {
      %first = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #blocked>
      // SEMA: [[WRITE_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[WRITE]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[WRITE_BUF]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: nvws.semaphore.release [[FULL]], [[WRITE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %first, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA-NEXT: [[READ:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: [[READ_BUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[READ]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: [[LOADED:%.*]] = ttg.local_load [[READ_BUF]] {ttg.partition = array<i32: 1>}
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // The consumer release comes AFTER the descriptor store it must cover.
      // SEMA-NEXT: tt.descriptor_store %{{.*}}[%{{.*}}, %{{.*}}], [[LOADED]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: nvws.semaphore.release [[EMPTY]], [[READ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      tt.descriptor_store %desc[%i, %i], %loaded {ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked>
      %next = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #blocked>
      // SEMA: [[NEXT_WRITE:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: [[NEXT_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[NEXT_WRITE]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[NEXT_BUF]] {ttg.partition = array<i32: 0>}
      ttg.local_store %next, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>} [[NEXT_WRITE]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    // SEMA: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // SEMA-LABEL: @already_lowered_tma_store_handoffs
  tt.func @already_lowered_tma_store_handoffs(
      %desc: !tt.tensordesc<tensor<128x64xf32, #shared>>,
      %lb: i32, %ub: i32, %step: i32) {
    %v0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %v1 = arith.constant dense<1.000000e+00> : tensor<128x64xf32, #blocked>
    // These exact-alias members model two consecutive output slices in one
    // depth-2 physical staging allocation.
    // SEMA: [[BASE:%.*]] = ttg.local_alloc
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] released = -1
    // SEMA-NEXT: [[COPY_READY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]]
    // SEMA-NEXT: [[M1_READY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]]
    // SEMA-NEXT: [[REDUCE_READY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]]
    %m0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 602 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    %m1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 602 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      // SEMA: [[ZERO_P0:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      // SEMA-NEXT: [[W0_TOKEN:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[ZERO_P0]]]
      // SEMA-NEXT: [[W0_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[ENTRY]], [[W0_TOKEN]]
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[W0_BUFFER]]#0
      // SEMA-NEXT: nvws.semaphore.release [[COPY_READY]][[[ZERO_P0]]], [[W0_TOKEN]]
      ttg.local_store %v0, %m0 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      // The TMA copy is a read of slot 0. Its release must stay after the
      // token wait and hand the next writer slot 1.
      // SEMA-NEXT: [[ZERO_P1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // SEMA-NEXT: [[COPY_TOKEN:%.*]] = nvws.semaphore.acquire [[COPY_READY]][[[ZERO_P1]]]
      // SEMA-NEXT: [[COPY_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[COPY_READY]], [[COPY_TOKEN]]
      // SEMA-NEXT: [[COPY:%.*]] = ttng.async_tma_copy_local_to_global %{{.*}} [[COPY_BUFFER]]#0
      // SEMA-NEXT: ttng.async_tma_store_token_wait [[COPY]]
      // SEMA-NEXT: [[TO_M1:%.*]] = arith.constant {{.*}} 1 : i32
      // SEMA-NEXT: nvws.semaphore.release [[M1_READY]][[[TO_M1]]], [[COPY_TOKEN]]
      %copy = ttng.async_tma_copy_local_to_global %desc[%i, %i] %m0 {ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x64xf32, #shared>>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %copy {ttg.partition = array<i32: 1>} : !ttg.async.token
      // SEMA-NEXT: [[W1_TOKEN:%.*]] = nvws.semaphore.acquire [[M1_READY]][[[ZERO_P0]]]
      // SEMA-NEXT: [[W1_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[M1_READY]], [[W1_TOKEN]]
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[W1_BUFFER]]#1
      // SEMA-NEXT: nvws.semaphore.release [[REDUCE_READY]][[[ZERO_P0]]], [[W1_TOKEN]]
      ttg.local_store %v1, %m1 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      // Async reduce has the same SMEM-read lifetime and must release only
      // after its completion wait, back to slot 0 of the next iteration.
      // SEMA-NEXT: [[REDUCE_TOKEN:%.*]] = nvws.semaphore.acquire [[REDUCE_READY]][[[ZERO_P1]]]
      // SEMA-NEXT: [[REDUCE_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[REDUCE_READY]], [[REDUCE_TOKEN]]
      // SEMA-NEXT: [[REDUCE:%.*]] = ttng.async_tma_reduce add, %{{.*}} [[REDUCE_BUFFER]]#1
      // SEMA-NEXT: ttng.async_tma_store_token_wait [[REDUCE]]
      // SEMA-NEXT: nvws.semaphore.release [[ENTRY]][[[TO_M1]]], [[REDUCE_TOKEN]]
      %reduce = ttng.async_tma_reduce add, %desc[%i, %i] %m1 {ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x64xf32, #shared>>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %reduce {ttg.partition = array<i32: 1>} : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // SEMA-LABEL: @converted_descriptor_store_completion
  tt.func @converted_descriptor_store_completion(%desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %i: i32, %lb: i32, %ub: i32, %step: i32) {
    // SEMA: [[V1:%.*]] = ttg.local_alloc {buffer.id = 601 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // SEMA: [[EMPTY:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32}
    // SEMA-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32}
    %alloc = ttg.local_alloc {buffer.id = 601 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // SEMA-NEXT: [[LOOP:%.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[WRITE:%.*]] = [[ENTRY]]) -> (!ttg.async.token)  : i32 {
    scf.for %iv = %lb to %ub step %step : i32 {
      %first = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #linear>
      // SEMA: [[WRITE_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[WRITE]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[WRITE_BUF]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: nvws.semaphore.release [[FULL]], [[WRITE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %first, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA-NEXT: [[READ:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: [[READ_BUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[READ]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: [[LOADED:%.*]] = ttg.local_load [[READ_BUF]] {ttg.partition = array<i32: 1>}
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #linear>
      // SEMA-NEXT: [[CONVERTED:%.*]] = ttg.convert_layout [[LOADED]] {ttg.partition = array<i32: 1>}
      %converted = ttg.convert_layout %loaded {ttg.partition = array<i32: 1>} : tensor<128x64xf16, #linear> -> tensor<128x64xf16, #blocked>
      // The consumer release comes AFTER the descriptor store even with the
      // intervening layout conversion between the load and the store.
      // SEMA-NEXT: tt.descriptor_store %{{.*}}[%{{.*}}, %{{.*}}], [[CONVERTED]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: nvws.semaphore.release [[EMPTY]], [[READ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      tt.descriptor_store %desc[%i, %i], %converted {ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked>
      %next = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #linear>
      // SEMA: [[NEXT_WRITE:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: [[NEXT_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[NEXT_WRITE]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[NEXT_BUF]] {ttg.partition = array<i32: 0>}
      ttg.local_store %next, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>} [[NEXT_WRITE]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    // SEMA: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
