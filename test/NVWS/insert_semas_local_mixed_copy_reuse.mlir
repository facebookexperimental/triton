// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_small = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @mixed_copy_same_backing
  // One 256x128 host covers the smaller two-copy 128x64 view.
  tt.func @mixed_copy_same_backing(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: %[[BASE:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 600 : i32} : () -> !ttg.memdesc<1x256x128xf16
    %host = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 600 : i32} : () -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    // CHECK-NEXT: %[[RING0:.*]] = ttg.memdesc_reinterpret %[[BASE]] : {{.*}} -> !ttg.memdesc<2x128x64xf16
    // CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i32
    // CHECK-NEXT: %[[SLOT0:.*]] = ttg.memdesc_index %[[RING0]][%[[ZERO]]]
    // CHECK-NEXT: %[[VIEW0:.*]] = ttg.memdesc_reinterpret %[[SLOT0]] : {{.*}} -> !ttg.memdesc<1x128x64xf16
    %slot0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 600 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK-NEXT: %[[RING1:.*]] = ttg.memdesc_reinterpret %[[BASE]] : {{.*}} -> !ttg.memdesc<2x128x64xf16
    // CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : i32
    // CHECK-NEXT: %[[SLOT1:.*]] = ttg.memdesc_index %[[RING1]][%[[ONE]]]
    // CHECK-NEXT: %[[VIEW1:.*]] = ttg.memdesc_reinterpret %[[SLOT1]] : {{.*}} -> !ttg.memdesc<1x128x64xf16
    // CHECK-NEXT: %[[ENTRY:.*]] = nvws.semaphore.create %[[BASE]], %[[VIEW0]], %[[VIEW1]] released = -1 {pending_count = 1 : i32}
    // CHECK-NEXT: %[[TO_R0:.*]] = nvws.semaphore.create %[[BASE]], %[[VIEW0]], %[[VIEW1]] {pending_count = 1 : i32}
    // CHECK-NEXT: %[[TO_W1:.*]] = nvws.semaphore.create %[[BASE]], %[[VIEW0]], %[[VIEW1]] {pending_count = 1 : i32}
    %slot1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 600 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %wide = arith.constant dense<0.0> : tensor<256x128xf16, #blocked>
    %half = arith.constant dense<1.0> : tensor<128x64xf16, #blocked64>
    // CHECK: %[[INIT:.*]] = nvws.semaphore.acquire %[[ENTRY]]
    // CHECK-NEXT: %[[LOOP_RESULT:.*]] = scf.for {{.*}} iter_args(%[[CARRY:.*]] = %[[INIT]])
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: %[[HOST_WRITE_BUF:.*]]:3 = nvws.semaphore.buffer %[[ENTRY]], %[[CARRY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store %{{.*}}, %[[HOST_WRITE_BUF]]#0 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release %[[TO_R0]], %[[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %wide, %host {ttg.partition = array<i32: 0>} : tensor<256x128xf16, #blocked> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
      // CHECK-NEXT: %[[R0_TOKEN:.*]] = nvws.semaphore.acquire %[[TO_R0]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: %[[R0_BUF:.*]]:3 = nvws.semaphore.buffer %[[TO_R0]], %[[R0_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: %[[R0_VALUE:.*]] = ttg.local_load %[[R0_BUF]]#1 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release %[[TO_W1]], %[[R0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %r0 = ttg.local_load %slot0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "use0"(%r0) {ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked64>) -> ()
      // CHECK: %[[W1_TOKEN:.*]] = nvws.semaphore.acquire %[[TO_W1]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: %[[W1_BUF:.*]]:3 = nvws.semaphore.buffer %[[TO_W1]], %[[W1_TOKEN]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttg.local_store %{{.*}}, %[[W1_BUF]]#2 {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release %[[ENTRY]], %[[W1_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      ttg.local_store %half, %slot1 {ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK-NEXT: %[[HOST_READ_TOKEN:.*]] = nvws.semaphore.acquire %[[ENTRY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: %[[HOST_READ_BUF:.*]]:3 = nvws.semaphore.buffer %[[ENTRY]], %[[HOST_READ_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: %[[HOST_VALUE:.*]] = ttg.local_load %[[HOST_READ_BUF]]#0 {ttg.partition = array<i32: 0>}
      %r1 = ttg.local_load %host {ttg.partition = array<i32: 0>} : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #blocked>
      "use1"(%r1) {ttg.partition = array<i32: 0>} : (tensor<256x128xf16, #blocked>) -> ()
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %[[HOST_READ_TOKEN]] : !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>,
       ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @same_copy_smaller_view
  tt.func @same_copy_smaller_view(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: %[[LARGE_BASE:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 602 : i32} : () -> !ttg.memdesc<1x128x128xf16
    %large = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 602 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK-NEXT: %[[SMALL_VIEW:.*]] = ttg.memdesc_reinterpret %[[LARGE_BASE]] : {{.*}} -> !ttg.memdesc<1x64x64xf16
    // CHECK-NEXT: %[[SMALL_ENTRY:.*]] = nvws.semaphore.create %[[LARGE_BASE]], %[[SMALL_VIEW]] released = -1 {pending_count = 1 : i32}
    // CHECK-NEXT: %[[SMALL_FULL:.*]] = nvws.semaphore.create %[[LARGE_BASE]], %[[SMALL_VIEW]] {pending_count = 1 : i32}
    %small = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 602 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %large_value = arith.constant dense<0.0> : tensor<128x128xf16, #blocked>
    // CHECK: scf.for
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: %[[LARGE_WRITE_TOKEN:.*]] = nvws.semaphore.acquire %[[SMALL_ENTRY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: %[[LARGE_WRITE_BUF:.*]]:2 = nvws.semaphore.buffer %[[SMALL_ENTRY]], %[[LARGE_WRITE_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store %{{.*}}, %[[LARGE_WRITE_BUF]]#0 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release %[[SMALL_FULL]], %[[LARGE_WRITE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %large_value, %large {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NEXT: %[[SMALL_TOKEN:.*]] = nvws.semaphore.acquire %[[SMALL_FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: %[[SMALL_READ_BUF:.*]]:2 = nvws.semaphore.buffer %[[SMALL_FULL]], %[[SMALL_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: %[[SMALL_VALUE:.*]] = ttg.local_load %[[SMALL_READ_BUF]]#1 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release %[[SMALL_ENTRY]], %[[SMALL_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %value = ttg.local_load %small {ttg.partition = array<i32: 1>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #blocked_small>
      "use_small"(%value) {ttg.partition = array<i32: 1>} : (tensor<64x64xf16, #blocked_small>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.partition.outputs = [], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }

  // CHECK-LABEL: @tokenless_copy_one_alias
  // CHECK-NOT: nvws.semaphore
  tt.func @tokenless_copy_one_alias(%lb: i32, %ub: i32, %step: i32,
                                     %value: tensor<128x64xf16, #blocked64>) {
    // CHECK: %[[ONE_BASE:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 601 : i32}
    // CHECK-NEXT: %[[A_ZERO:.*]] = arith.constant 0 : i32
    // CHECK-NEXT: %[[A_VIEW:.*]] = ttg.memdesc_index %[[ONE_BASE]][%[[A_ZERO]]]
    %a = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 601 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK-NEXT: %[[B_ZERO:.*]] = arith.constant 0 : i32
    // CHECK-NEXT: %[[B_VIEW:.*]] = ttg.memdesc_index %[[ONE_BASE]][%[[B_ZERO]]]
    // CHECK-NOT: nvws.semaphore
    %b = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 601 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: ttg.local_store %{{.*}}, %[[A_VIEW]] {ttg.partition = array<i32: 0>}
      ttg.local_store %value, %a {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK-NEXT: %[[LOADED:.*]] = ttg.local_load %[[B_VIEW]] {ttg.partition = array<i32: 0>}
      %loaded = ttg.local_load %b {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "use"(%loaded) {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>) -> ()
      scf.yield {ttg.partition = array<i32: 0>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0>,
       ttg.partition.outputs = [], ttg.warp_specialize.tag = 2 : i32}
    // CHECK-NOT: nvws.semaphore
    // CHECK: tt.return
    tt.return
  }
}
