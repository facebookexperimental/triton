// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The start-1 allocation appears first, while producer order remains
  // start0, start1. Folding must move the canonical start-0 backing before the
  // merged semaphore creates.
  // CHECK-LABEL: @circular_start_zero_backing_dominates_creates
  tt.func @circular_start_zero_backing_dominates_creates(
      %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[S1_PAYLOAD:%.*]] = "make_start1"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %start1_payload = "make_start1"() {ttg.partition = array<i32: 1>} : () -> !ty
    // CHECK: [[S0_PAYLOAD:%.*]] = "make_start0"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %start0_payload = "make_start0"() {ttg.partition = array<i32: 1>} : () -> !ty
    %start1 = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 700 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 700 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %start0 = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 700 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      // CHECK: [[P1_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[S0_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[P1_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[S0_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[P1_STAGE]]{{\]}}, [[S0_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store [[S0_PAYLOAD]], [[S0_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[P1_STAGE]]{{\]}}, [[S0_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %start0_payload, %start0 {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[P2_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[S0_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[P2_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[S0_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[P2_STAGE]]{{\]}}, [[S0_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[S0_VAL:%.*]] = ttg.local_load [[S0_FULL_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[P2_STAGE]]{{\]}}, [[S0_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %start0_value = ttg.local_load %start0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: [[S1_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[P1_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[S1_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[P1_STAGE]]{{\]}}, [[S1_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store [[S1_PAYLOAD]], [[S1_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[P1_STAGE]]{{\]}}, [[S1_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %start1_payload, %start1 {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[S1_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[P2_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[S1_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[P2_STAGE]]{{\]}}, [[S1_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[S1_VAL:%.*]] = ttg.local_load [[S1_FULL_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[P2_STAGE]]{{\]}}, [[S1_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %start1_value = ttg.local_load %start1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: "use"([[S0_VAL]], [[S1_VAL]]) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
      "use"(%start0_value, %start1_value) {ttg.partition = array<i32: 2>} : (!ty, !ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
