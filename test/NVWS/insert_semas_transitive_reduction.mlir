// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// Transitive reduction (spec TRANSITIVE REDUCTION section): implied
// same-chain edges are dropped pay-for-play — but never wave-opening
// acquires (the wave guard) and never across distinct destinations'
// closed waves. Corner cases pinned here:
//   1. serialized ring: the {0}->{2} fan-in arm is implied through
//      {0}->{1}->{2} and DROPPED (one release per handoff survives);
//   2. genuine fan-out to two reader partitions: both edges are wave
//      openers — NOTHING is dropped;
//   3. a returning owner with an earlier token reuses that token without
//      adding a carrier-only handoff.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @serialized_ring_reduces
  tt.func @serialized_ring_reduces(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant {ttg.partition = array<i32: 0>} dense<0.0> : tensor<256x128xf16, #blocked>
    %cst1 = arith.constant {ttg.partition = array<i32: 2>} dense<1.0> : tensor<128x128xf16, #blocked>
    // Containment shape (the planner-realistic layout): the offset-0 owner
    // m0 spans [0,256) and the offset-64 reuser m1[64,192) nests inside it.
    // Both members of buffer.id 500 are co-allocated into one semaphore
    // set. One initially released EMPTY and three blocked FULL handoff semaphores
    // serialize the {0}->{1}->{2}->{0} chain.
    // CHECK: [[A:%.*]] = ttg.local_alloc {buffer.id = 500 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>
    // CHECK: [[B:%.*]] = ttg.local_alloc {buffer.id = 500 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[A]], [[B]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F01:%.*]] = nvws.semaphore.create [[A]], [[B]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F12:%.*]] = nvws.semaphore.create [[A]], [[B]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F20:%.*]] = nvws.semaphore.create [[A]], [[B]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // Owner/reuser pair of one buffer.id: {0} writes a (the spanning
    // owner), {1} reads a, {2} writes b (nested inside a), {0} reads b.
    // The W-after-R edge {0}->{2} for the overlap piece is implied via
    // {0}->{1}->{2} and dropped (in-chain sweep). The minimal serialized
    // chain survives: three handoffs plus the carrier close — four
    // releases. No piece is reuser-only, so every EXIT carries owner {0}
    // and no {2} regain edge is raised at all.
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // {0} acquires EMPTY, writes a, releases FULL for {1}.
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF0:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>
      // CHECK: ttg.local_store %{{.*}}, [[BUF0]]#0 {ttg.partition = array<i32: 0>} : tensor<256x128xf16, #blocked> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[F01]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %a = ttg.local_alloc %cst0 {buffer.id = 500 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
      // {1} acquires FULL, reads a, releases FULL for {2} and EMPTY back.
      // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F01]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF1:%.*]]:2 = nvws.semaphore.buffer [[F01]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>
      // CHECK: ttg.local_load [[BUF1]]#0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[F12]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %va = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #blocked>
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<256x128xf16, #blocked>) -> ()
      // {2} acquires FULL, writes b, releases FULL for {0}.
      // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F12]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF2:%.*]]:2 = nvws.semaphore.buffer [[F12]], [[T2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable, 1x256x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[BUF2]]#1 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[F20]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %b = ttg.local_alloc %cst1 {buffer.id = 500 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // {0} acquires FULL, reads b. No re-release: every piece's EXIT
      // carries owner {0}, so no {2} regain edge exists to close here.
      // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F20]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF3:%.*]]:2 = nvws.semaphore.buffer [[F20]], [[T3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable, 1x256x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
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

#scalar_blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#scalar_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#scalar_smem = #ttg.shared_memory
!scalar_ty = tensor<1xi32, #scalar_blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // The producer reread moves holder {0}'s WAR row, but it does not replace
  // the current version's RAW source. The foreign writer must wait for both
  // readers, including the producer reread.
  // CHECK-LABEL: @stable_producer_row_and_foreign_war
  tt.func @stable_producer_row_and_foreign_war(%lb: i32, %ub: i32,
                                                %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 991 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[READY:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[WAR:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 2 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // No pre-loop seed acquire and no token iter_arg: EMPTY is created
    // initially released and acquired at the point of use inside the body.
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
    %alloc = ttg.local_alloc {buffer.id = 991 : i32} : () -> !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %w0 = "w0"() {ttg.partition = array<i32: 0>} : () -> !scalar_ty
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[READY]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %w0, %alloc {ttg.partition = array<i32: 0>} : !scalar_ty -> !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable>
      // The reread uses the same producer token and buffer view, then
      // contributes a real WAR arrival to the later foreign writer.
      // CHECK: ttg.local_load [[B0]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      // CHECK: nvws.semaphore.release [[WAR]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %r0 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable> -> !scalar_ty
      "use0"(%r0) {ttg.partition = array<i32: 0>} : (!scalar_ty) -> ()
      // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[READY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[READY]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      // CHECK: nvws.semaphore.release [[WAR]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %r1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable> -> !scalar_ty
      "use1"(%r1) {ttg.partition = array<i32: 1>} : (!scalar_ty) -> ()
      %w2 = "w2"() {ttg.partition = array<i32: 2>} : () -> !scalar_ty
      // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[WAR]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[WAR]], [[T2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // The foreign writer closes the ring: its release of EMPTY supplies
      // the producer's next-iteration acquire.
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %w2, %alloc {ttg.partition = array<i32: 2>} : !scalar_ty -> !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#scalar_blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#scalar_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#scalar_smem = #ttg.shared_memory
!scalar_ty = tensor<1xi32, #scalar_blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // An outer write is represented by the child ENTER. The second child reader
  // fans out from ENTER, not from the first child reader or the outer row.
  // CHECK-LABEL: @region_resets_producer_row_to_enter
  tt.func @region_resets_producer_row_to_enter(%lb: i32, %ub: i32,
                                                %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 992 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // {2}'s turn semaphore is created before {1}'s; {0} hands off to [[R1]].
    // CHECK: [[R2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[R1:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 992 : i32} : () -> !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      %w0 = "w0"() {ttg.partition = array<i32: 0>} : () -> !scalar_ty
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[R1]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %w0, %alloc {ttg.partition = array<i32: 0>} : !scalar_ty -> !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable>
      // No token iter_arg on the inner loop: {1} acquires [[R1]] at its
      // point of use inside the body.
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
      scf.for %j = %lb to %ub step %step : i32 {
        // The entry acquire of [[R1]] is the inner chain's source: its
        // fan-out release to [[R2]] precedes the first reader's load, which
        // reuses the same token.
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[R1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: nvws.semaphore.release [[R2]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[R1]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        %r1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable> -> !scalar_ty
        "use1"(%r1) {ttg.partition = array<i32: 1>} : (!scalar_ty) -> ()
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[R2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[R2]], [[T2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B2]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        // {2}'s release of [[R1]] feeds the next inner iteration's entry
        // acquire.
        // CHECK: nvws.semaphore.release [[R1]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %r2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable> -> !scalar_ty
        "use2"(%r2) {ttg.partition = array<i32: 2>} : (!scalar_ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
      // Loop close: after the inner region, {1} re-acquires [[R1]] (the last
      // inner release) and returns EMPTY to the outer producer.
      // CHECK: } {ttg.partition = array<i32: 1, 2>}
      // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[R1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.warp_specialize.tag = 0 : i32}
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
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BUF]] released = -1 {pending_count = 2 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F1:%.*]] = nvws.semaphore.create [[BUF]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[BUF]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // No pre-loop seed acquire and no token iter_arg: the producer acquires
    // EMPTY at its point of use inside the body.
    // CHECK: %{{.*}} = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32) : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // {0} acquires EMPTY in-body, writes a, releases to both readers.
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[PBUF]] {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[F1]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[F2]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
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
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %v2 = ttg.local_load %a {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
      // {0} re-reads a with its retained producer token: no handoff edge.
      // CHECK: [[RBUF0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[RBUF0]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      %v0 = ttg.local_load %a {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%v0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      // No re-acquire and no token threaded through scf.yield: the next
      // iteration's in-body acquire waits on both reader releases.
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %{{.*}} : i32
      %j = arith.addi %i, %iv {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
      // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
