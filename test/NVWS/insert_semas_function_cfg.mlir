// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @managed_group_in_non_entry_block
  tt.func @managed_group_in_non_entry_block(
      %early: i1, %lb: i32, %ub: i32, %step: i32) {
    cf.cond_br %early, ^exit, ^work
  ^exit:
    tt.return
  ^work:
    // CHECK: ^bb2:
    // CHECK: [[BACKING:%.*]] = ttg.local_alloc {buffer.id = 1200 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BACKING]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BACKING]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 1200 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[WRITE_TOKEN:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[WRITE_BUFFER:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[WRITE_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[WRITE_BUFFER]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]], [[WRITE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[READ_TOKEN:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[READ_BUFFER:%.*]] = nvws.semaphore.buffer [[FULL]], [[READ_TOKEN]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[LOADED:%.*]] = ttg.local_load [[READ_BUFFER]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[EMPTY]], [[READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: "consume"([[LOADED]]) {ttg.partition = array<i32: 1>} : (tensor<1xi32, #blocked>) -> ()
      "consume"(%loaded) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
