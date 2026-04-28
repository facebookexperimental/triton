// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s --check-prefix=SEMA
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas --nvws-assign-stage-phase -cse | FileCheck %s --check-prefix=ASP

// These tests cover slot replay across an atomic scheduled region.  The
// scf.if itself owns the stage-2 schedule; its child operations intentionally
// have no loop.stage/loop.cluster attributes.

#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked128 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Two fresh writes advance the depth-5 cursor from A's slot to B's slot.
  // The conditional C wave is a same-owner overwrite after B has been read,
  // so it reuses B's slot and does not advance the cursor.  The region-closing
  // release therefore supplies A three logical iterations later:
  //
  //   A(i) = 2i mod 5, B/C(i) = 2i+1 mod 5, A(i+3) = B/C(i).
  //
  // The only authored non-zero displacement is the A-read to B-write handoff.
  // The scf.if returns an owner-0 token: the then-branch hands the C-read
  // token back to owner 0, while the else-branch passes the B-read token
  // through.  One common ENTRY release follows the conditional.
  // SEMA-LABEL: @depth5_regular_atomic_if
  // ASP-LABEL: @depth5_regular_atomic_if
  tt.func @depth5_regular_atomic_if(%lb: i32, %ub: i32, %step: i32,
                                    %cond: i1) {
    // SEMA: [[A_BASE:%.*]] = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32}
    // SEMA: [[B_BASE:%.*]] = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32}
    // SEMA: [[C_BASE:%.*]] = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32}
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] released = 21 {pending_count = 1 : i32}
    // SEMA: [[A_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[A_TO_B:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[B_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[C_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[IF_BACK:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // ASP: [[ENTRY:%.*]] = nvws.semaphore.create {{.*}} released = 21 {pending_count = 1 : i32}
    // ASP: [[A_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // ASP: [[A_TO_B:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // ASP: [[B_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // ASP: [[C_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // ASP: [[IF_BACK:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    %a = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %a_value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked64>
    %b_value = arith.constant dense<1.000000e+00> : tensor<128x64xf16, #blocked64>
    %c_value = arith.constant dense<2.000000e+00> : tensor<128x128xf16, #blocked128>

    // No token iter_args: every acquire is at its point of use in the body.
    // After ASP the loop carries the cursor plus phase words, all i32.
    // SEMA: scf.for
    // ASP: %{{[0-9]+}}:7 = scf.for {{.*}} iter_args([[CURSOR:%.*]] = %{{[-A-Za-z0-9_.$#]+}},
    scf.for %iv = %lb to %ub step %step : i32 {
      // SEMA: [[ZA:%.*]] = arith.constant {{.*}} 0 : i32
      // SEMA: [[A_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[ZA]]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // SEMA: [[A_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[ENTRY]], [[A_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // SEMA: ttg.local_store {{.*}}, [[A_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // SEMA: [[ZAF:%.*]] = arith.constant {{.*}} 0 : i32
      // SEMA: nvws.semaphore.release [[A_FULL]][[[ZAF]]], [[A_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[A_SLOT:%.*]] = arith.select {{.*}} : i32
      // ASP: [[A_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[A_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[A_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[ENTRY]][[[A_SLOT]]], [[A_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: ttg.local_store {{.*}}, [[A_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: nvws.semaphore.release [[A_FULL]][[[A_SLOT]]], [[A_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      ttg.local_store %a_value, %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // The A read hands its slot off to the B write at displacement +1.
      // SEMA: [[ZAR:%.*]] = arith.constant {{.*}} 0 : i32
      // SEMA: [[A_READ_TOK:%.*]] = nvws.semaphore.acquire [[A_FULL]][[[ZAR]]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: [[A_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[A_FULL]], [[A_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: ttg.local_load [[A_READ_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: [[TO_B:%.*]] = arith.constant {{.*}} 1 : i32
      // SEMA: nvws.semaphore.release [[A_TO_B]][[[TO_B]]], [[A_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[A_READ_TOK:%.*]] = nvws.semaphore.acquire [[A_FULL]][[[A_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[A_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[A_FULL]][[[A_SLOT]]], [[A_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: ttg.local_load [[A_READ_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[TO_B_RAW:%.*]] = arith.addi [[A_SLOT]], {{%.*}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      // ASP: [[TO_B_REM:%.*]] = arith.remsi [[TO_B_RAW]], {{%.*}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      // ASP: [[TO_B_SLOT:%.*]] = arith.select {{.*}}, {{.*}}, [[TO_B_REM]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      // ASP: nvws.semaphore.release [[A_TO_B]][[[TO_B_SLOT]]], [[A_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %a_read = ttg.local_load %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_a"(%a_read) {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>) -> ()
      // B's fresh acquire advances the cursor to [[B_SLOT]] = [[A_SLOT]] + 1.
      // SEMA: [[ZB:%.*]] = arith.constant {{.*}} 0 : i32
      // SEMA: [[B_TOK:%.*]] = nvws.semaphore.acquire [[A_TO_B]][[[ZB]]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // SEMA: [[B_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[A_TO_B]], [[B_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // SEMA: ttg.local_store {{.*}}, [[B_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // SEMA: [[ZBF:%.*]] = arith.constant {{.*}} 0 : i32
      // SEMA: nvws.semaphore.release [[B_FULL]][[[ZBF]]], [[B_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[B_RAW:%.*]] = arith.addi [[A_SLOT]], {{%.*}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[B_SLOT:%.*]] = arith.select {{.*}}, {{.*}}, [[B_RAW]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[B_TOK:%.*]] = nvws.semaphore.acquire [[A_TO_B]][[[B_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[B_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[A_TO_B]][[[B_SLOT]]], [[B_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: ttg.local_store {{.*}}, [[B_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: nvws.semaphore.release [[B_FULL]][[[B_SLOT]]], [[B_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      ttg.local_store %b_value, %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: [[ZBR:%.*]] = arith.constant {{.*}} 0 : i32
      // SEMA: [[B_READ_TOK:%.*]] = nvws.semaphore.acquire [[B_FULL]][[[ZBR]]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: [[B_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[B_FULL]], [[B_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: ttg.local_load [[B_READ_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[B_READ_TOK:%.*]] = nvws.semaphore.acquire [[B_FULL]][[[B_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[B_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[B_FULL]][[[B_SLOT]]], [[B_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: ttg.local_load [[B_READ_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %b_read = ttg.local_load %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_b"(%b_read) {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>) -> ()
      // The scf.if yields its owner-0 token.  ASP also returns the shared slot
      // and the phase words updated inside the branch.
      // SEMA: [[IF_TOKEN:%.*]] = scf.if
      // ASP: [[IF_RESULTS:%.*]]:4 = scf.if
      scf.if %cond {
        // C overwrites B's slot under the still-held B-read token: it is
        // rendered through the C member of B_FULL's buffer tuple, so the
        // atomic region does not add a third cursor advance.
        // SEMA: [[C_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[B_FULL]], [[B_READ_TOK]] {ttg.partition = array<i32: 0>}
        // SEMA: ttg.local_store {{.*}}, [[C_VIEWS]]#2 {ttg.partition = array<i32: 0>}
        // SEMA: nvws.semaphore.release [[C_FULL]], [[B_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        // ASP: [[C_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[B_FULL]][[[B_SLOT]]], [[B_READ_TOK]] {ttg.partition = array<i32: 0>}
        // ASP: ttg.local_store {{.*}}, [[C_VIEWS]]#2 {ttg.partition = array<i32: 0>}
        // ASP: nvws.semaphore.release [[C_FULL]][[[B_SLOT]]], [[B_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        ttg.local_store %c_value, %c {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked128> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // The C read hands ownership back to owner 0 before the branch yields.
        // SEMA: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]] {ttg.partition = array<i32: 1>}
        // SEMA: [[C_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[C_FULL]], [[C_READ_TOK]] {ttg.partition = array<i32: 1>}
        // SEMA: ttg.local_load [[C_READ_VIEWS]]#2 {ttg.partition = array<i32: 1>}
        // SEMA: nvws.semaphore.release [[IF_BACK]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // SEMA: [[IF_THEN_TOKEN:%.*]] = nvws.semaphore.acquire [[IF_BACK]] {ttg.partition = array<i32: 0>}
        // SEMA: scf.yield {{.*}}[[IF_THEN_TOKEN]] : !ttg.async.token
        // ASP: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]][[[B_SLOT]], {{%.*}}] {ttg.partition = array<i32: 1>}
        // ASP: [[C_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[C_FULL]][[[B_SLOT]]], [[C_READ_TOK]] {ttg.partition = array<i32: 1>}
        // ASP: ttg.local_load [[C_READ_VIEWS]]#2 {ttg.partition = array<i32: 1>}
        // ASP: nvws.semaphore.release [[IF_BACK]][[[B_SLOT]]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // ASP: [[IF_THEN_TOKEN:%.*]] = nvws.semaphore.acquire [[IF_BACK]][[[B_SLOT]], {{%.*}}] {ttg.partition = array<i32: 0>}
        // ASP: scf.yield {{.*}}[[IF_THEN_TOKEN]], [[B_SLOT]],
        %c_read = ttg.local_load %c {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked128>
        "consume_c"(%c_read) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked128>) -> ()
      } else {
        // Without the C wave, the branch passes the B-read token through.
        // SEMA: scf.yield {{.*}}[[B_READ_TOK]] : !ttg.async.token
        // ASP: scf.yield {{.*}}[[B_READ_TOK]], [[B_SLOT]],
      } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      // SEMA: } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // ASP: } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>]}
      // SEMA: nvws.semaphore.release [[ENTRY]][{{%.*}}], [[IF_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[ENTRY]][[[IF_RESULTS]]#1], [[IF_RESULTS]]#0 [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: scf.yield {{.*}}[[IF_RESULTS]]#1,
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.partition.stages = [1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // Consecutive A/B writes by the same owner share one token and therefore
  // make one fresh depth-3 cursor advance.  Both reads and the atomic C region
  // use that stage.  The scf.if returns an owner-0 token from either the C read
  // or the unchanged A/B-read path.  One common release then returns the slot
  // to A after one full three-iteration orbit.
  // SEMA-LABEL: @depth3_same_owner_atomic_if
  // ASP-LABEL: @depth3_same_owner_atomic_if
  tt.func @depth3_same_owner_atomic_if(%lb: i32, %ub: i32, %step: i32,
                                       %cond: i1) {
    // SEMA: [[A_BASE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32}
    // SEMA: [[B_BASE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32}
    // SEMA: [[C_BASE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32}
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] released = -1 {pending_count = 1 : i32}
    // SEMA: [[AB_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[C_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[IF_BACK:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // ASP: [[ENTRY:%.*]] = nvws.semaphore.create {{.*}} released = -1 {pending_count = 1 : i32}
    // ASP: [[AB_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // ASP: [[C_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // ASP: [[IF_BACK:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    %a = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %a_value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked64>
    %b_value = arith.constant dense<1.000000e+00> : tensor<128x64xf16, #blocked64>
    %c_value = arith.constant dense<2.000000e+00> : tensor<128x128xf16, #blocked128>

    // SEMA: scf.for
    // ASP: %{{[0-9]+}}:5 = scf.for {{.*}} iter_args([[CURSOR:%.*]] = %{{[-A-Za-z0-9_.$#]+}},
    scf.for %iv = %lb to %ub step %step : i32 {
      // Both writes share the single ENTRY acquire of this iteration.
      // SEMA: [[AB_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // SEMA: [[AB_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[ENTRY]], [[AB_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // SEMA: ttg.local_store {{.*}}, [[AB_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // ASP: [[AB_SLOT:%.*]] = arith.select {{.*}} : i32
      // ASP: [[AB_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[AB_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // ASP: [[AB_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[ENTRY]][[[AB_SLOT]]], [[AB_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // ASP: ttg.local_store {{.*}}, [[AB_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      ttg.local_store %a_value, %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_store {{.*}}, [[AB_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // SEMA: nvws.semaphore.release [[AB_FULL]], [[AB_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // ASP: ttg.local_store {{.*}}, [[AB_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // ASP: nvws.semaphore.release [[AB_FULL]][[[AB_SLOT]]], [[AB_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      ttg.local_store %b_value, %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // Both reads share the single AB_FULL acquire.
      // SEMA: [[AB_READ_TOK:%.*]] = nvws.semaphore.acquire [[AB_FULL]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: [[AB_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[AB_FULL]], [[AB_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: ttg.local_load [[AB_READ_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[AB_READ_TOK:%.*]] = nvws.semaphore.acquire [[AB_FULL]][[[AB_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[AB_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[AB_FULL]][[[AB_SLOT]]], [[AB_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: ttg.local_load [[AB_READ_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %a_read = ttg.local_load %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      // SEMA: ttg.local_load [[AB_READ_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: ttg.local_load [[AB_READ_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %b_read = ttg.local_load %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_ab"(%a_read, %b_read) {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>, tensor<128x64xf16, #blocked64>) -> ()
      // The scf.if yields its owner-0 token.  ASP also returns the shared slot
      // and the phase words updated inside the branch.
      // SEMA: [[IF_TOKEN:%.*]] = scf.if
      // ASP: [[IF_RESULTS:%.*]]:4 = scf.if
      scf.if %cond {
        // C is rendered through the C member of AB_FULL's buffer tuple: the
        // atomic region reuses the shared slot without a cursor advance.
        // SEMA: [[C_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[AB_FULL]], [[AB_READ_TOK]] {ttg.partition = array<i32: 0>}
        // SEMA: ttg.local_store {{.*}}, [[C_VIEWS]]#2 {ttg.partition = array<i32: 0>}
        // SEMA: nvws.semaphore.release [[C_FULL]], [[AB_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        // ASP: [[C_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[AB_FULL]][[[AB_SLOT]]], [[AB_READ_TOK]] {ttg.partition = array<i32: 0>}
        // ASP: ttg.local_store {{.*}}, [[C_VIEWS]]#2 {ttg.partition = array<i32: 0>}
        // ASP: nvws.semaphore.release [[C_FULL]][[[AB_SLOT]]], [[AB_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        ttg.local_store %c_value, %c {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked128> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // The C read hands ownership back to owner 0 before the branch yields.
        // SEMA: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]] {ttg.partition = array<i32: 1>}
        // SEMA: [[C_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[C_FULL]], [[C_READ_TOK]] {ttg.partition = array<i32: 1>}
        // SEMA: ttg.local_load [[C_READ_VIEWS]]#2 {ttg.partition = array<i32: 1>}
        // SEMA: nvws.semaphore.release [[IF_BACK]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // SEMA: [[IF_THEN_TOKEN:%.*]] = nvws.semaphore.acquire [[IF_BACK]] {ttg.partition = array<i32: 0>}
        // SEMA: scf.yield {{.*}}[[IF_THEN_TOKEN]] : !ttg.async.token
        // ASP: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]][[[AB_SLOT]], {{%.*}}] {ttg.partition = array<i32: 1>}
        // ASP: [[C_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[C_FULL]][[[AB_SLOT]]], [[C_READ_TOK]] {ttg.partition = array<i32: 1>}
        // ASP: ttg.local_load [[C_READ_VIEWS]]#2 {ttg.partition = array<i32: 1>}
        // ASP: nvws.semaphore.release [[IF_BACK]][[[AB_SLOT]]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // ASP: [[IF_THEN_TOKEN:%.*]] = nvws.semaphore.acquire [[IF_BACK]][[[AB_SLOT]], {{%.*}}] {ttg.partition = array<i32: 0>}
        // ASP: scf.yield {{.*}}[[IF_THEN_TOKEN]], [[AB_SLOT]],
        %c_read = ttg.local_load %c {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked128>
        "consume_c"(%c_read) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked128>) -> ()
      } else {
        // Without the C wave, the branch passes the A/B-read token through.
        // SEMA: scf.yield {{.*}}[[AB_READ_TOK]] : !ttg.async.token
        // ASP: scf.yield {{.*}}[[AB_READ_TOK]], [[AB_SLOT]],
      } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      // SEMA: } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // ASP: } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0, 1, 2>, array<i32: 0>, array<i32: 1>]}
      // SEMA: nvws.semaphore.release [[ENTRY]], [[IF_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[ENTRY]][[[IF_RESULTS]]#1], [[IF_RESULTS]]#0 [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: scf.yield {{.*}}[[IF_RESULTS]]#1,
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}
