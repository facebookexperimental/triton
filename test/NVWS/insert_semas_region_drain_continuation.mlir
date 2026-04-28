// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>
!two = tensor<2xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @root_continuation
  tt.func @root_continuation(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    // CHECK: [[ROOT_WHOLE:%[0-9]+]] = ttg.local_alloc {buffer.id = 9910 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[ROOT_LEFT:%[0-9]+]] = ttg.local_alloc {buffer.id = 9910 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[ROOT_RIGHT:%[0-9]+]] = ttg.local_alloc {buffer.id = 9910 : i32, buffer.offset = 1 : i32}
    // CHECK-NEXT: [[ROOT_EMPTY:%[0-9]+]] = nvws.semaphore.create [[ROOT_WHOLE]], [[ROOT_LEFT]], [[ROOT_RIGHT]] released = -1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[ROOT_FULL:%[0-9]+]] = nvws.semaphore.create [[ROOT_WHOLE]], [[ROOT_LEFT]], [[ROOT_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[ROOT_LEFT_READY:%[0-9]+]] = nvws.semaphore.create [[ROOT_WHOLE]], [[ROOT_LEFT]], [[ROOT_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[ROOT_RIGHT_READY:%[0-9]+]] = nvws.semaphore.create [[ROOT_WHOLE]], [[ROOT_LEFT]], [[ROOT_RIGHT]] {pending_count = 1 : i32}
    %whole = ttg.local_alloc {buffer.id = 9910 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    %left = ttg.local_alloc {buffer.id = 9910 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %right = ttg.local_alloc {buffer.id = 9910 : i32, buffer.offset = 1 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NEXT: [[ROOT_VALUE:%[0-9]+]] = "root.value"()
    %root = "root.value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[ROOT_LOOP_VALUE:%[0-9]+]] = "producer"() {ttg.partition = array<i32: 0>}
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !two
      // CHECK-NEXT: [[ROOT_WRITE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[ROOT_EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[ROOT_WRITE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[ROOT_EMPTY]], [[ROOT_WRITE_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store [[ROOT_LOOP_VALUE]], [[ROOT_WRITE_BUFFER]]#0 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[ROOT_FULL]], [[ROOT_WRITE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %value, %whole {ttg.partition = array<i32: 0>} : !two -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[ROOT_READ_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[ROOT_FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[ROOT_READ_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[ROOT_FULL]], [[ROOT_READ_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[ROOT_LOADED:%[0-9]+]] = ttg.local_load [[ROOT_READ_BUFFER]]#0 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[ROOT_EMPTY]], [[ROOT_READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %loaded = ttg.local_load %whole {ttg.partition = array<i32: 1>} : !ttg.memdesc<2xi32, #shared, #smem, mutable> -> !two
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!two) -> ()
    // CHECK: } {tt.warp_specialize
    // CHECK-NEXT: [[ROOT_DRAIN:%[0-9]+]] = nvws.semaphore.acquire [[ROOT_EMPTY]]
    // CHECK-NEXT: nvws.semaphore.release [[ROOT_RIGHT_READY]], [[ROOT_DRAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[ROOT_LEFT_READY]], [[ROOT_DRAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
    // CHECK-NEXT: [[ROOT_LEFT_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[ROOT_LEFT_READY]]
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.warp_specialize.tag = 0 : i32}
    scf.if %cond {
      // CHECK: scf.if
      // CHECK-NEXT: [[ROOT_THEN_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[ROOT_LEFT_READY]], [[ROOT_LEFT_TOKEN]]
      // CHECK-NEXT: ttg.local_store [[ROOT_VALUE]], [[ROOT_THEN_BUFFER]]#1
      ttg.local_store %root, %left : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } else {
      // CHECK: } else {
      // CHECK-NEXT: [[ROOT_ELSE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[ROOT_LEFT_READY]], [[ROOT_LEFT_TOKEN]]
      // CHECK-NEXT: ttg.local_store [[ROOT_VALUE]], [[ROOT_ELSE_BUFFER]]#1
      ttg.local_store %root, %left : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    }
    // CHECK: [[ROOT_RIGHT_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[ROOT_RIGHT_READY]]
    // CHECK-NEXT: [[ROOT_FINAL_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[ROOT_RIGHT_READY]], [[ROOT_RIGHT_TOKEN]]
    // CHECK-NEXT: ttg.local_store [[ROOT_VALUE]], [[ROOT_FINAL_BUFFER]]#2
    ttg.local_store %root, %right : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>
!two = tensor<2xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @partition_2_then_partition_3
  tt.func @partition_2_then_partition_3(%lb: i32, %ub: i32, %step: i32,
                                        %cond: i1) {
    // CHECK: [[P23_WHOLE:%[0-9]+]] = ttg.local_alloc {buffer.id = 9911 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[P23_LEFT:%[0-9]+]] = ttg.local_alloc {buffer.id = 9911 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[P23_RIGHT:%[0-9]+]] = ttg.local_alloc {buffer.id = 9911 : i32, buffer.offset = 1 : i32}
    // CHECK-NEXT: [[P23_EMPTY:%[0-9]+]] = nvws.semaphore.create [[P23_WHOLE]], [[P23_LEFT]], [[P23_RIGHT]] released = -1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[P23_FULL:%[0-9]+]] = nvws.semaphore.create [[P23_WHOLE]], [[P23_LEFT]], [[P23_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P23_ROOT_READY:%[0-9]+]] = nvws.semaphore.create [[P23_WHOLE]], [[P23_LEFT]], [[P23_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P23_THEN_READY:%[0-9]+]] = nvws.semaphore.create [[P23_WHOLE]], [[P23_LEFT]], [[P23_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P23_ELSE_READY:%[0-9]+]] = nvws.semaphore.create [[P23_WHOLE]], [[P23_LEFT]], [[P23_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P23_RIGHT_READY:%[0-9]+]] = nvws.semaphore.create [[P23_WHOLE]], [[P23_LEFT]], [[P23_RIGHT]] {pending_count = 1 : i32}
    %whole = ttg.local_alloc {buffer.id = 9911 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    %left = ttg.local_alloc {buffer.id = 9911 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %right = ttg.local_alloc {buffer.id = 9911 : i32, buffer.offset = 1 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NEXT: [[P23_VALUE:%[0-9]+]] = "root.value"()
    %root = "root.value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[P23_LOOP_VALUE:%[0-9]+]] = "producer"() {ttg.partition = array<i32: 0>}
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !two
      // CHECK-NEXT: [[P23_WRITE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P23_EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[P23_WRITE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P23_EMPTY]], [[P23_WRITE_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store [[P23_LOOP_VALUE]], [[P23_WRITE_BUFFER]]#0 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[P23_FULL]], [[P23_WRITE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %value, %whole {ttg.partition = array<i32: 0>} : !two -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[P23_READ_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P23_FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[P23_READ_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P23_FULL]], [[P23_READ_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[P23_LOADED:%[0-9]+]] = ttg.local_load [[P23_READ_BUFFER]]#0 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[P23_EMPTY]], [[P23_READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %loaded = ttg.local_load %whole {ttg.partition = array<i32: 1>} : !ttg.memdesc<2xi32, #shared, #smem, mutable> -> !two
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!two) -> ()
    // CHECK: } {tt.warp_specialize
    // CHECK-NEXT: [[P23_DRAIN:%[0-9]+]] = nvws.semaphore.acquire [[P23_EMPTY]]
    // CHECK-NEXT: nvws.semaphore.release [[P23_RIGHT_READY]], [[P23_DRAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[P23_ROOT_READY]], [[P23_DRAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
    // CHECK-NEXT: [[P23_ROOT_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P23_ROOT_READY]]
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>,
       ttg.warp_specialize.tag = 0 : i32}
    scf.if %cond {
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.release [[P23_THEN_READY]], [[P23_ROOT_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
      // CHECK-NEXT: [[P23_THEN_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P23_THEN_READY]] {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: [[P23_THEN_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P23_THEN_READY]], [[P23_THEN_TOKEN]] {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: ttg.local_store [[P23_VALUE]], [[P23_THEN_BUFFER]]#1 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
      ttg.local_store %root, %left {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } else {
      // CHECK: } else {
      // CHECK-NEXT: nvws.semaphore.release [[P23_ELSE_READY]], [[P23_ROOT_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
      // CHECK-NEXT: [[P23_ELSE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P23_ELSE_READY]] {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: [[P23_ELSE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P23_ELSE_READY]], [[P23_ELSE_TOKEN]] {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: ttg.local_store [[P23_VALUE]], [[P23_ELSE_BUFFER]]#1 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
      ttg.local_store %root, %left {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    }
    // CHECK: [[P23_RIGHT_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P23_RIGHT_READY]] {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: [[P23_FINAL_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P23_RIGHT_READY]], [[P23_RIGHT_TOKEN]] {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: ttg.local_store [[P23_VALUE]], [[P23_FINAL_BUFFER]]#2 {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32}
    ttg.local_store %root, %right {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>
!two = tensor<2xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @partition_1_then_partition_0
  tt.func @partition_1_then_partition_0(%lb: i32, %ub: i32, %step: i32,
                                        %cond: i1) {
    // CHECK: [[P10_WHOLE:%[0-9]+]] = ttg.local_alloc {buffer.id = 9912 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[P10_LEFT:%[0-9]+]] = ttg.local_alloc {buffer.id = 9912 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[P10_RIGHT:%[0-9]+]] = ttg.local_alloc {buffer.id = 9912 : i32, buffer.offset = 1 : i32}
    // CHECK-NEXT: [[P10_EMPTY:%[0-9]+]] = nvws.semaphore.create [[P10_WHOLE]], [[P10_LEFT]], [[P10_RIGHT]] released = -1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[P10_FULL:%[0-9]+]] = nvws.semaphore.create [[P10_WHOLE]], [[P10_LEFT]], [[P10_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P10_ROOT_READY:%[0-9]+]] = nvws.semaphore.create [[P10_WHOLE]], [[P10_LEFT]], [[P10_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P10_THEN_READY:%[0-9]+]] = nvws.semaphore.create [[P10_WHOLE]], [[P10_LEFT]], [[P10_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P10_ELSE_READY:%[0-9]+]] = nvws.semaphore.create [[P10_WHOLE]], [[P10_LEFT]], [[P10_RIGHT]] {pending_count = 1 : i32}
    %whole = ttg.local_alloc {buffer.id = 9912 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    %left = ttg.local_alloc {buffer.id = 9912 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %right = ttg.local_alloc {buffer.id = 9912 : i32, buffer.offset = 1 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NEXT: [[P10_VALUE:%[0-9]+]] = "root.value"()
    %root = "root.value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[P10_LOOP_VALUE:%[0-9]+]] = "producer"() {ttg.partition = array<i32: 0>}
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !two
      // CHECK-NEXT: [[P10_WRITE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P10_EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[P10_WRITE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P10_EMPTY]], [[P10_WRITE_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store [[P10_LOOP_VALUE]], [[P10_WRITE_BUFFER]]#0 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[P10_FULL]], [[P10_WRITE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %value, %whole {ttg.partition = array<i32: 0>} : !two -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[P10_READ_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P10_FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[P10_READ_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P10_FULL]], [[P10_READ_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[P10_LOADED:%[0-9]+]] = ttg.local_load [[P10_READ_BUFFER]]#0 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[P10_EMPTY]], [[P10_READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %loaded = ttg.local_load %whole {ttg.partition = array<i32: 1>} : !ttg.memdesc<2xi32, #shared, #smem, mutable> -> !two
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!two) -> ()
    // CHECK: } {tt.warp_specialize
    // CHECK-NEXT: [[P10_DRAIN:%[0-9]+]] = nvws.semaphore.acquire [[P10_EMPTY]]
    // CHECK-NEXT: nvws.semaphore.release [[P10_ROOT_READY]], [[P10_DRAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
    // CHECK-NEXT: [[P10_ROOT_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P10_ROOT_READY]]
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.warp_specialize.tag = 0 : i32}
    scf.if %cond {
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.release [[P10_THEN_READY]], [[P10_ROOT_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
      // CHECK-NEXT: [[P10_THEN_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P10_THEN_READY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: [[P10_THEN_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P10_THEN_READY]], [[P10_THEN_TOKEN]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: ttg.local_store [[P10_VALUE]], [[P10_THEN_BUFFER]]#1 {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
      ttg.local_store %root, %left {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } else {
      // CHECK: } else {
      // CHECK-NEXT: nvws.semaphore.release [[P10_ELSE_READY]], [[P10_ROOT_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
      // CHECK-NEXT: [[P10_ELSE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P10_ELSE_READY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: [[P10_ELSE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P10_ELSE_READY]], [[P10_ELSE_TOKEN]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: ttg.local_store [[P10_VALUE]], [[P10_ELSE_BUFFER]]#1 {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
      ttg.local_store %root, %left {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    }
    // CHECK: [[P10_FINAL_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P10_EMPTY]], [[P10_DRAIN]]
    // CHECK-NEXT: ttg.local_store [[P10_VALUE]], [[P10_FINAL_BUFFER]]#2 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    ttg.local_store %root, %right {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    tt.return
  }
}
