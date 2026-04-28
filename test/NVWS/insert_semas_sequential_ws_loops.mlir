// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// E4 golden (fable/rule-v2-corpus-verification.md §7.6 E4; plan M0.1):
// ONE buffer shared by TWO sequential ws-tagged loops. Pins the seam
// behavior of the CURRENT pass so any change shows up in review.
//
// What this pins (point-of-use construction): neither loop carries a
// token iter_arg. V2, loop A's full semaphore, is created initially
// released so it supplies iteration zero; A's store acquires it in-body
// directly before the store and A's consumer releases it each
// iteration. The seam is one acquire/release pair at function scope
// between the loops: `acquire V2` drains A's final permit and
// `release V5, <that token>` converts it into loop B's entry permit.
// Loop B repeats the same in-body shape on V5 (store acquire, consumer
// release) and holds one surplus permit on V5 at exit. V3 and V4 are
// the intra-loop store->load edges of A and B respectively, released
// then acquired within each iteration.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @sequential_ws_loops_shared_buffer
  tt.func @sequential_ws_loops_shared_buffer(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 950 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 950 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producerA"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V7]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V3]], [[V8]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V10:%.*]] = ttg.local_load [[V9]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %l = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V2]], [[V8]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "consumerA"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: nvws.semaphore.release [[V5]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    scf.for %j = %lb to %ub step %step : i32 {
      %w = "producerB"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V5]], [[V12]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V13]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %w, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[V4]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.buffer [[V4]], [[V14]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V16:%.*]] = ttg.local_load [[V15]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %m = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V5]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "consumerB"(%m) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 1 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}
