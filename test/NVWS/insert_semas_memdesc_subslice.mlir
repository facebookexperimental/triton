// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>
!two = tensor<2xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_memdesc_subslice_alias
  tt.func @local_memdesc_subslice_alias(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[BASE:%[0-9]+]] = ttg.local_alloc {buffer.id = 9920 : i32} : () -> !ttg.memdesc<1x2xi32, #shared, #smem, mutable>
    // CHECK-NEXT: [[EMPTY:%[0-9]+]] = nvws.semaphore.create [[BASE]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]>
    // CHECK-NEXT: [[FULL:%[0-9]+]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 9920 : i32} : () -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[VALUE:%[0-9]+]] = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<2xi32, #blocked>
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !two
      // CHECK: [[EMPTY_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: [[WHOLE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[EMPTY]], [[EMPTY_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      // CHECK-NEXT: ttg.local_store [[VALUE]], [[WHOLE_BUFFER]] {ttg.partition = array<i32: 0>} : tensor<2xi32, #blocked> -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 0>} : !two -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]], [[EMPTY_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[FULL_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: [[WHOLE_READ_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[FULL]], [[FULL_TOKEN]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[SLICE:%[0-9]+]] = ttg.memdesc_subslice [[WHOLE_READ_BUFFER]][0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<2xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %view = ttg.memdesc_subslice %alloc[0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<2xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[LOADED:%[0-9]+]] = ttg.local_load [[SLICE]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %loaded = ttg.local_load %view {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      // CHECK: nvws.semaphore.release [[EMPTY]], [[FULL_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!one) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
