// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>
!two = tensor<2xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // One completed WS-region phase supplies both root writes. The drain acquire
  // is one exact token producer; the two handoff releases fan out from it.
  // CHECK-LABEL: @region_drain_fanout
  tt.func @region_drain_fanout(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[WHOLE:%.*]] = ttg.local_alloc
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[WHOLE]]
    %whole = ttg.local_alloc {buffer.id = 9903 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    %left = ttg.local_alloc {buffer.id = 9903 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %right = ttg.local_alloc {buffer.id = 9903 : i32, buffer.offset = 1 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %root = "root.value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !two
      ttg.local_store %value, %whole {ttg.partition = array<i32: 0>} : !two -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      %loaded = ttg.local_load %whole {ttg.partition = array<i32: 1>} : !ttg.memdesc<2xi32, #shared, #smem, mutable> -> !two
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!two) -> ()
    // CHECK: } {tt.warp_specialize
    // CHECK-NEXT: [[DRAIN:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: nvws.semaphore.release {{%.*}}, [[DRAIN]]
    // CHECK-NEXT: nvws.semaphore.release {{%.*}}, [[DRAIN]]
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.warp_specialize.tag = 0 : i32}
    ttg.local_store %root, %left : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    ttg.local_store %root, %right : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // A branch-only handoff drains the completed WS-region phase inside the
  // branch that executes. The two static drain sites are mutually exclusive.
  // CHECK-LABEL: @guarded_region_drain
  tt.func @guarded_region_drain(%lb: i32, %ub: i32, %step: i32,
                                %cond: i1) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
    // CHECK: [[BRANCH_EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]]
    %alloc = ttg.local_alloc {buffer.id = 9904 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !one
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 0>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!one) -> ()
    // CHECK: } {tt.warp_specialize
    // CHECK-NEXT: scf.if
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.warp_specialize.tag = 0 : i32}
    scf.if %cond {
      // CHECK-NEXT: [[THEN_DRAIN:%.*]] = nvws.semaphore.acquire [[BRANCH_EMPTY]]
      // CHECK-NEXT: nvws.semaphore.release {{%.*}}, [[THEN_DRAIN]]
      %then_value = ttg.local_load %alloc : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "then.use"(%then_value) : (!one) -> ()
    } else {
      // CHECK: } else {
      // CHECK-NEXT: [[ELSE_DRAIN:%.*]] = nvws.semaphore.acquire [[BRANCH_EMPTY]]
      // CHECK-NEXT: nvws.semaphore.release {{%.*}}, [[ELSE_DRAIN]]
      %else_value = ttg.local_load %alloc : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "else.use"(%else_value) : (!one) -> ()
    }
    tt.return
  }
}
