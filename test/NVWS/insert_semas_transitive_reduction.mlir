// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// Transitive reduction (spec TRANSITIVE REDUCTION section): implied
// same-chain edges are dropped pay-for-play — but never wave-opening
// acquires (the wave guard) and never across distinct destinations'
// closed waves. Corner cases pinned here:
//   1. serialized ring: the {0}->{2} fan-in arm is implied through
//      {0}->{1}->{2} and DROPPED (one release per handoff survives);
//   2. genuine fan-out to two reader partitions: both edges are wave
//      openers — NOTHING is dropped;
//   3. regain by the producer after a reader wave: kept (wave guard),
//      even though ordering is transitively implied.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @serialized_ring_reduces
  tt.func @serialized_ring_reduces(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant {ttg.partition = array<i32: 0>} dense<0.0> : tensor<128x128xf16, #blocked>
    %cst1 = arith.constant {ttg.partition = array<i32: 2>} dense<1.0> : tensor<128x128xf16, #blocked>
    // The two overlapping pieces (offset 0 and 64) of buffer.id 500 are
    // co-allocated into one semaphore set. One EMPTY (true) and three FULL
    // (false) handoff semaphores serialize the {0}->{1}->{2}->{0} chain.
    // CHECK: [[A:%.*]] = ttg.local_alloc {buffer.id = 500 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[B:%.*]] = ttg.local_alloc {buffer.id = 500 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[A]], [[B]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F01:%.*]] = nvws.semaphore.create [[A]], [[B]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F12:%.*]] = nvws.semaphore.create [[A]], [[B]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F20:%.*]] = nvws.semaphore.create [[A]], [[B]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // Overlapping pair (offset 0 and 64 of one buffer.id): {0} writes a,
    // {1} reads a, {2} writes b (overlaps a), {0} reads b. The W-after-R
    // edge {0}->{2} for the overlap piece is implied via {0}->{1}->{2}.
    // The minimal serialized chain survives: three handoffs plus the
    // carrier close — four releases. The implied {0}->{2} fan-in arm
    // (phase A) and the {2} regain whose ordering the following
    // traversal already implies (phase B, traversal closure) are gone.
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // {0} acquires EMPTY, writes a, releases FULL for {1}.
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF0:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>
      // CHECK: ttg.local_store %{{.*}}, [[BUF0]]#0 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[F01]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %a = ttg.local_alloc %cst0 {buffer.id = 500 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // {1} acquires FULL, reads a, releases FULL for {2} and EMPTY back.
      // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F01]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF1:%.*]]:2 = nvws.semaphore.buffer [[F01]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>
      // CHECK: ttg.local_load [[BUF1]]#0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[F12]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %va = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      // {2} acquires FULL, writes b, releases FULL for {0}.
      // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F12]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF2:%.*]]:2 = nvws.semaphore.buffer [[F12]], [[T2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[BUF2]]#1 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[F20]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %b = ttg.local_alloc %cst1 {buffer.id = 500 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // {0} acquires FULL, reads b. No re-release: the implied {2} regain
      // is dropped by transitive reduction (traversal closure).
      // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F20]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF3:%.*]]:2 = nvws.semaphore.buffer [[F20]], [[T3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[BUF3]]#1 {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      %vb = ttg.local_load %b {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      // Exactly four releases survive across the body (F01, F12+EMPTY, F20).
      // No carrier token threaded through scf.yield in this serialized ring.
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %{{.*}} : i32
      %j = arith.addi %i, %iv {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
      // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @fanout_not_reduced
  tt.func @fanout_not_reduced(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant {ttg.partition = array<i32: 0>} dense<0.0> : tensor<128x128xf16, #blocked>
    // One producer, two independent reader partitions: both edges open
    // their waves — the reduction must keep both reader acquires. The
    // EMPTY semaphore fans out (pending_count = 2) to two FULL sems.
    // CHECK: [[BUF:%.*]] = ttg.local_alloc {buffer.id = 501 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BUF]] true {pending_count = 2 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F1:%.*]] = nvws.semaphore.create [[BUF]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[BUF]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F0:%.*]] = nvws.semaphore.create [[BUF]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // Wave-opening acquire of EMPTY is hoisted before the loop and carried.
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: %{{.*}}:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, [[CARRY:%.*]] = [[T0]]) -> (i32, !ttg.async.token)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // {0} writes a from the carried EMPTY buffer, releases to both readers.
      // CHECK: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[CARRY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[PBUF]] {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[F1]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[F2]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %a = ttg.local_alloc %cst0 {buffer.id = 501 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Reader {1}: wave-opening acquire of F1 (KEPT, not reduced).
      // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[RBUF1:%.*]] = nvws.semaphore.buffer [[F1]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[RBUF1]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %v1 = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      // Reader {2}: wave-opening acquire of F2 (KEPT, not reduced).
      // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[RBUF2:%.*]] = nvws.semaphore.buffer [[F2]], [[T2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[RBUF2]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[F0]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %v2 = ttg.local_load %a {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
      // {0} re-reads a: acquire F0 (the {2}->{0} edge survives as wave guard).
      // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[RBUF0:%.*]] = nvws.semaphore.buffer [[F0]], [[T3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[RBUF0]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      %v0 = ttg.local_load %a {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%v0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      // Producer re-acquires EMPTY for next iteration and threads the token.
      // CHECK: [[T4:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %{{.*}}, [[T4]] : i32, !ttg.async.token
      %j = arith.addi %i, %iv {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
      // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>, array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
