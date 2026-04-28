// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// M2 PIN (nested-loop hold-rule extension, plan v3).
//
// A WS-tagged OUTER loop wrapping a non-WS INNER loop with ONE inner-confined
// ping-pong buffer (the persistent-FA qk shape, design v3 §8): tc_gen5_mma
// (partition 1) writes the tmem accumulator, tmem_load (partition 0) reads it,
// BOTH confined to the inner loop; nothing touches the buffer at the outer
// level.
//
// This pins the enabled native shape: no root entry acquire, the original
// token iter_args dropped from both loops, and the writer acquire at the
// first toucher inside the inner loop.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-LABEL:   tt.func @nested_ws_inner_loop(
  tt.func @nested_ws_inner_loop(%lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    // Alloc grows to 1x (single-buffered) and loses its token result; the two
    // ping-pong semaphores are created at the root: released EMPTY gates the
    // writer, blocked FULL the reader.
    // CHECK:           [[BUF_ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[EMPTY:%.*]] = nvws.semaphore.create [[BUF_ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[FULL:%.*]] = nvws.semaphore.create [[BUF_ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %res, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // Both loops drop their token iter_args entirely: no acquire hoisted to
    // the root, no carrier threaded through either loop.
    // CHECK:           scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
    %o = scf.for %iv0 = %lb to %ub step %step iter_args(%t0 = %tok) -> (!ttg.async.token) : i32 {
      // CHECK-NEXT:        scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
      %i = scf.for %iv = %lb to %ub step %step iter_args(%t1 = %t0) -> (!ttg.async.token) : i32 {
        // CHECK:               {{.*}} = "loadA"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %sA = "loadA"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        // CHECK:               {{.*}} = "loadB"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %sB = "loadB"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        // Writer (partition 1): acquire EMPTY at first toucher, buffer, MMA, release FULL.
        // CHECK:               [[WTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:               [[WBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[WTOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:               {{.*}} = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[WBUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %sA, %sB, %res[%t1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK:               nvws.semaphore.release [[FULL]], [[WTOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // Reader (partition 0): acquire FULL, buffer, tmem_load, release EMPTY.
        // CHECK:               [[RTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:               [[RBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[RTOK]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:               %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[RBUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %t2 = ttng.tmem_load %res[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK:               nvws.semaphore.release [[EMPTY]], [[RTOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK:               "use"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // The token results are gone, so neither loop yields a carrier.
        // CHECK-NOT:           scf.yield
        scf.yield {ttg.partition = array<i32: 0, 1>} %t2 : !ttg.async.token
      // CHECK:             } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // CHECK-NOT:           scf.yield
      scf.yield {ttg.partition = array<i32: 0, 1>} %i : !ttg.async.token
    // CHECK:           } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    // No post-loop TMEM access: nothing emitted after the outer loop.
    // CHECK:           tt.return
    tt.return
  }

  // The inner recurrence is identical to nested_ws_inner_loop, but the outer
  // body consumes the same allocation after the inner loop.  The recurrence
  // acquire stays next to the inner MMA; one final LOCAL_EMPTY acquire after
  // the inner loop hands the buffer to the outer continuation.
  // CHECK-LABEL:   tt.func @nested_ws_inner_loop_parent_continuation(
  tt.func @nested_ws_inner_loop_parent_continuation(%lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    // Four semaphores share the allocation: LOCAL EMPTY/FULL ping-pong the
    // inner recurrence, OUTER EMPTY/FULL fence the outer continuation.
    // CHECK:           [[LIVE_ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[LOCAL_EMPTY:%.*]] = nvws.semaphore.create [[LIVE_ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[OUTER_EMPTY:%.*]] = nvws.semaphore.create [[LIVE_ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[LOCAL_FULL:%.*]] = nvws.semaphore.create [[LIVE_ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[OUTER_FULL:%.*]] = nvws.semaphore.create [[LIVE_ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // The pre-loop entry acquire drains OUTER_EMPTY's initial permit so the
    // in-body tail acquire waits on the outer reader; its token is unused.
    // CHECK:           [[OUTER_ENTRY:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %res, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // Neither loop carries a token: the original iter_args are dropped.
    // CHECK:           scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
    %o = scf.for %iv0 = %lb to %ub step %step iter_args(%t0 = %tok) -> (!ttg.async.token) : i32 {
      // CHECK-NEXT:        scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
      %i = scf.for %iv = %lb to %ub step %step iter_args(%t1 = %t0) -> (!ttg.async.token) : i32 {
        %sA = "loadA"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %sB = "loadB"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        // Writer (partition 1): acquire LOCAL_EMPTY at first toucher, buffer, MMA, release LOCAL_FULL.
        // CHECK:               [[LACQ:%.*]] = nvws.semaphore.acquire [[LOCAL_EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:               [[LBUF:%.*]] = nvws.semaphore.buffer [[LOCAL_EMPTY]], [[LACQ]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:               {{.*}} = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[LBUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %sA, %sB, %res[%t1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK:               nvws.semaphore.release [[LOCAL_FULL]], [[LACQ]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // Reader (partition 0): acquire LOCAL_FULL, buffer, tmem_load, release LOCAL_EMPTY.
        // CHECK:               [[LRTOK:%.*]] = nvws.semaphore.acquire [[LOCAL_FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:               [[LRBUF:%.*]] = nvws.semaphore.buffer [[LOCAL_FULL]], [[LRTOK]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:               %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[LRBUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %t2 = ttng.tmem_load %res[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK:               nvws.semaphore.release [[LOCAL_EMPTY]], [[LRTOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK:               "use_inner"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        "use_inner"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK-NOT:           scf.yield
        scf.yield {ttg.partition = array<i32: 0, 1>} %t2 : !ttg.async.token
      // CHECK:             } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // Handoff: the final LOCAL_EMPTY acquire after the inner loop is the
      // token released to OUTER_FULL (async_op none: the last inner reader
      // already arrived after MMA completion).
      // CHECK-NEXT:        [[FINAL_LOCAL:%.*]] = nvws.semaphore.acquire [[LOCAL_EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT:        nvws.semaphore.release [[OUTER_FULL]], [[FINAL_LOCAL]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Outer reader (partition 0): acquire OUTER_FULL, buffer, tmem_load, release OUTER_EMPTY.
      // CHECK:             [[OACQ:%.*]] = nvws.semaphore.acquire [[OUTER_FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK:             [[OBUF:%.*]] = nvws.semaphore.buffer [[OUTER_FULL]], [[OACQ]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:             %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[OBUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %outerVal, %t3 = ttng.tmem_load %res[%i] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK:             nvws.semaphore.release [[OUTER_EMPTY]], [[OACQ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK:             "use_outer"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      "use_outer"(%outerVal) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // Bridge: regain OUTER_EMPTY and refeed LOCAL_EMPTY for the next outer
      // iteration; without this, iteration two deadlocks.
      // CHECK:             [[OUTER_TAIL:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT:        nvws.semaphore.release [[LOCAL_EMPTY]], [[OUTER_TAIL]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK-NOT:           scf.yield
      scf.yield {ttg.partition = array<i32: 0, 1>} %t3 : !ttg.async.token
    // CHECK:           } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 1 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 1 : i32}
    // CHECK:           tt.return
    tt.return
  }
}
