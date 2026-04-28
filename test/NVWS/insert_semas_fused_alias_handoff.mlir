// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s --check-prefix=SEMA
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas --nvws-assign-stage-phase -cse | FileCheck %s --check-prefix=ASP

// Two exact-alias epilogue members share one depth-2 physical allocation.
// The first member uses slot 0 and the second uses slot 1, so each
// read-to-next-write release must target the successor slot rather than the
// source read's slot.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SEMA-LABEL: @fused_alias_depth_two
  // ASP-LABEL: @fused_alias_depth_two
  tt.func @fused_alias_depth_two(%lb: i32, %ub: i32, %step: i32) {
    // Both member allocs collapse onto one fused depth-2 backing allocation;
    // every semaphore lists both (identical) member views as its buffers.
    // SEMA: [[BASE:%.*]] = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // SEMA: [[FULL0:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // SEMA: [[EMPTY1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // SEMA: [[FULL1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // ASP: [[BASE:%.*]] = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32}
    // ASP: [[ENTRY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] released = -1 {pending_count = 1 : i32}
    // ASP: [[FULL0:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32}
    // ASP: [[EMPTY1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32}
    // ASP: [[FULL1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32}
    %m0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %m1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %v1 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #blocked>

    // The loop-close release partition (2) differs from the first-acquire
    // partition (4), so no acquire token is threaded through iter_args at the
    // SEMA stage; ASP threads the slot cursor plus one phase word per
    // acquirer.
    // SEMA: scf.for
    // ASP: scf.for {{.*}} iter_args([[CURSOR:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PH_R0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PH_R1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PH_W0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PH_W1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}})
    scf.for %iv = %lb to %ub step %step : i32 {
      // Member 0 write: acquire ENTRY, store through view #0, release FULL0.
      // SEMA: [[W0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: [[W0_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[W0_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[W0_BUF:%.*]]:2 = nvws.semaphore.buffer [[ENTRY]], [[W0_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[W0_BUF]]#0 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // SEMA: [[W0_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: nvws.semaphore.release [[FULL0]][[[W0_REL_ZERO]]], [[W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[SLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: arith.shli {{%.*}}, [[SLOT0]] {ttg.partition = array<i32: 4>} : i32
      // ASP: [[PHN_W0:%.*]] = arith.xori [[PH_W0]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // ASP: [[W0_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[SLOT0]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // ASP: [[W0_BUF:%.*]]:2 = nvws.semaphore.buffer [[ENTRY]][[[SLOT0]]], [[W0_TOK]] {ttg.partition = array<i32: 4>}
      // ASP: ttg.local_store {{%.*}}, [[W0_BUF]]#0 {ttg.partition = array<i32: 4>}
      // ASP: nvws.semaphore.release [[FULL0]][[[SLOT0]]], [[W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      ttg.local_store %v0, %m0 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Member 0 read: acquire FULL0, load view #0, release EMPTY1 at the
      // successor slot (SLOT0 + 1).
      // SEMA: [[R0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[R0_TOK:%.*]] = nvws.semaphore.acquire [[FULL0]][[[R0_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[R0_BUF:%.*]]:2 = nvws.semaphore.buffer [[FULL0]], [[R0_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      // SEMA: ttg.local_load [[R0_BUF]]#0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // SEMA: [[TO_M1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // SEMA: nvws.semaphore.release [[EMPTY1]][[[TO_M1]]], [[R0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[PHN_R0:%.*]] = arith.xori [[PH_R0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[R0_TOK:%.*]] = nvws.semaphore.acquire [[FULL0]][[[SLOT0]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // ASP: [[R0_BUF:%.*]]:2 = nvws.semaphore.buffer [[FULL0]][[[SLOT0]]], [[R0_TOK]] {ttg.partition = array<i32: 2>}
      // ASP: ttg.local_load [[R0_BUF]]#0 {ttg.partition = array<i32: 2>}
      // ASP: [[TO_M1_RAW:%.*]] = arith.addi [[SLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TO_M1_REM:%.*]] = arith.remsi [[TO_M1_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TO_M1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_M1_REM]] {ttg.partition = array<i32: 2>} : i32
      // ASP: nvws.semaphore.release [[EMPTY1]][[[TO_M1]]], [[R0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      %r0 = ttg.local_load %m0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "consume0"(%r0) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
      // Member 1 write: acquire EMPTY1, store through view #1, release FULL1.
      // SEMA: [[W1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: [[W1_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY1]][[[W1_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[W1_BUF:%.*]]:2 = nvws.semaphore.buffer [[EMPTY1]], [[W1_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[W1_BUF]]#1 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // SEMA: [[W1_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: nvws.semaphore.release [[FULL1]][[[W1_REL_ZERO]]], [[W1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[NEXT_RAW:%.*]] = arith.addi [[SLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[SLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[NEXT_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: arith.shli {{%.*}}, [[SLOT1]] {ttg.partition = array<i32: 4>} : i32
      // ASP: [[PHN_W1:%.*]] = arith.xori [[PH_W1]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // ASP: [[W1_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY1]][[[SLOT1]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // ASP: [[W1_BUF:%.*]]:2 = nvws.semaphore.buffer [[EMPTY1]][[[SLOT1]]], [[W1_TOK]] {ttg.partition = array<i32: 4>}
      // ASP: ttg.local_store {{%.*}}, [[W1_BUF]]#1 {ttg.partition = array<i32: 4>}
      // ASP: nvws.semaphore.release [[FULL1]][[[SLOT1]]], [[W1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      ttg.local_store %v1, %m1 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Member 1 read: acquire FULL1, load view #1, close the loop by
      // releasing ENTRY at the successor slot (SLOT1 + 1) mod 2.
      // SEMA: [[R1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[R1_TOK:%.*]] = nvws.semaphore.acquire [[FULL1]][[[R1_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[R1_BUF:%.*]]:2 = nvws.semaphore.buffer [[FULL1]], [[R1_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_load [[R1_BUF]]#1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // SEMA: [[TO_NEXT_M0:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // SEMA: nvws.semaphore.release [[ENTRY]][[[TO_NEXT_M0]]], [[R1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[PHN_R1:%.*]] = arith.xori [[PH_R1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[R1_TOK:%.*]] = nvws.semaphore.acquire [[FULL1]][[[SLOT1]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // ASP: [[R1_BUF:%.*]]:2 = nvws.semaphore.buffer [[FULL1]][[[SLOT1]]], [[R1_TOK]] {ttg.partition = array<i32: 2>}
      // ASP: ttg.local_load [[R1_BUF]]#1 {ttg.partition = array<i32: 2>}
      // ASP: [[TO_M0_RAW:%.*]] = arith.addi [[SLOT1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TO_M0_REM:%.*]] = arith.remsi [[TO_M0_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TO_M0:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_M0_REM]] {ttg.partition = array<i32: 2>} : i32
      // ASP: nvws.semaphore.release [[ENTRY]][[[TO_M0]]], [[R1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // ASP: scf.yield {ttg.partition = array<i32: 2, 4>} [[SLOT1]], [[PHN_R0]], [[PHN_R1]], [[PHN_W0]], [[PHN_W1]] : i32, i32, i32, i32, i32
      %r1 = ttg.local_load %m1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "consume1"(%r1) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // Planner-authored aliases may be different views of one staged backing.
  // Here the smaller member covers the prefix of the larger member.  The
  // read-to-next-write handoff must still target the following physical slot.
  // SEMA-LABEL: @fused_partial_alias_depth_three
  // ASP-LABEL: @fused_partial_alias_depth_three
  tt.func @fused_partial_alias_depth_three(%lb: i32, %ub: i32, %step: i32) {
    // Entry stages 0 and 2 are acquired before their first release; stage 1
    // is released before its first acquire, so the bootstrap mask is 0b101.
    // SEMA: [[PLARGE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32} : () -> !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>
    // SEMA: [[PSMALL:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32} : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    // SEMA: [[PENTRY:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] released = 5 {pending_count = 1 : i32} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
    // SEMA: [[PFULL0:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
    // SEMA: [[PHANDOFF:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
    // SEMA: [[PFULL1:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
    // ASP: [[PLARGE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32}
    // ASP: [[PSMALL:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32}
    // ASP: [[PENTRY:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] released = 5 {pending_count = 1 : i32}
    // ASP: [[PFULL0:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32}
    // ASP: [[PHANDOFF:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32}
    // ASP: [[PFULL1:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32}
    %large = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32} : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %small = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %small_value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked64>
    %large_value = arith.constant dense<1.000000e+00> : tensor<256x64xf16, #blocked64>

    // SEMA: scf.for
    // ASP: scf.for {{.*}} iter_args([[PCURSOR:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PPH_R0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PPH_R1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PPH_W0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PPH_W1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}})
    scf.for %iv = %lb to %ub step %step : i32 {
      // Small-member write: acquire PENTRY, store through view #1 (the small
      // member), release PFULL0.
      // SEMA: [[PW0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: [[PW0_TOK:%.*]] = nvws.semaphore.acquire [[PENTRY]][[[PW0_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[PW0_BUF:%.*]]:2 = nvws.semaphore.buffer [[PENTRY]], [[PW0_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[PW0_BUF]]#1 {ttg.partition = array<i32: 4>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: [[PW0_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: nvws.semaphore.release [[PFULL0]][[[PW0_REL_ZERO]]], [[PW0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[PSLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[PPHN_W0:%.*]] = arith.xori [[PPH_W0]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // ASP: [[PW0_TOK:%.*]] = nvws.semaphore.acquire [[PENTRY]][[[PSLOT0]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // ASP: [[PW0_BUF:%.*]]:2 = nvws.semaphore.buffer [[PENTRY]][[[PSLOT0]]], [[PW0_TOK]] {ttg.partition = array<i32: 4>}
      // ASP: ttg.local_store {{%.*}}, [[PW0_BUF]]#1 {ttg.partition = array<i32: 4>}
      // ASP: nvws.semaphore.release [[PFULL0]][[[PSLOT0]]], [[PW0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      ttg.local_store %small_value, %small {ttg.partition = array<i32: 4>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // Small-member read: acquire PFULL0, load view #1, then hand off to the
      // large write at the following physical slot (PSLOT0 + 1) mod 3.
      // SEMA: [[PR0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[PR0_TOK:%.*]] = nvws.semaphore.acquire [[PFULL0]][[[PR0_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[PR0_BUF:%.*]]:2 = nvws.semaphore.buffer [[PFULL0]], [[PR0_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_load [[PR0_BUF]]#1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked1>
      // SEMA: [[TO_LARGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // SEMA: nvws.semaphore.release [[PHANDOFF]][[[TO_LARGE]]], [[PR0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[PPHN_R0:%.*]] = arith.xori [[PPH_R0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[PR0_TOK:%.*]] = nvws.semaphore.acquire [[PFULL0]][[[PSLOT0]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // ASP: [[PR0_BUF:%.*]]:2 = nvws.semaphore.buffer [[PFULL0]][[[PSLOT0]]], [[PR0_TOK]] {ttg.partition = array<i32: 2>}
      // ASP: ttg.local_load [[PR0_BUF]]#1 {ttg.partition = array<i32: 2>}
      // ASP: [[TO_LARGE_RAW:%.*]] = arith.addi [[PSLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TO_LARGE_REM:%.*]] = arith.remsi [[TO_LARGE_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TO_LARGE_SLOT:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_LARGE_REM]] {ttg.partition = array<i32: 2>} : i32
      // ASP: nvws.semaphore.release [[PHANDOFF]][[[TO_LARGE_SLOT]]], [[PR0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      %small_read = ttg.local_load %small {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_small"(%small_read) {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked64>) -> ()
      // Large-member write: acquire PHANDOFF at the successor slot, store
      // through view #0 (the large member), release PFULL1.
      // SEMA: [[PW1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: [[PW1_TOK:%.*]] = nvws.semaphore.acquire [[PHANDOFF]][[[PW1_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[PW1_BUF:%.*]]:2 = nvws.semaphore.buffer [[PHANDOFF]], [[PW1_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[PW1_BUF]]#0 {ttg.partition = array<i32: 4>} : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      // SEMA: [[PW1_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: nvws.semaphore.release [[PFULL1]][[[PW1_REL_ZERO]]], [[PW1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[PSLOT1_RAW:%.*]] = arith.addi [[PSLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[PSLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[PSLOT1_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[PPHN_W1:%.*]] = arith.xori [[PPH_W1]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // ASP: [[PW1_TOK:%.*]] = nvws.semaphore.acquire [[PHANDOFF]][[[PSLOT1]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // ASP: [[PW1_BUF:%.*]]:2 = nvws.semaphore.buffer [[PHANDOFF]][[[PSLOT1]]], [[PW1_TOK]] {ttg.partition = array<i32: 4>}
      // ASP: ttg.local_store {{%.*}}, [[PW1_BUF]]#0 {ttg.partition = array<i32: 4>}
      // ASP: nvws.semaphore.release [[PFULL1]][[[PSLOT1]]], [[PW1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      ttg.local_store %large_value, %large {ttg.partition = array<i32: 4>} : tensor<256x64xf16, #blocked64> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      // Large-member read: acquire PFULL1, load view #0, close the loop by
      // releasing PENTRY at the reader's own slot (constant 0 / PSLOT1: the
      // slot the next small write reaches two iterations later).
      // SEMA: [[PR1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[PR1_TOK:%.*]] = nvws.semaphore.acquire [[PFULL1]][[[PR1_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[PR1_BUF:%.*]]:2 = nvws.semaphore.buffer [[PFULL1]], [[PR1_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>
      // SEMA: ttg.local_load [[PR1_BUF]]#0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #blocked1>
      // SEMA: [[PBACK_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: nvws.semaphore.release [[PENTRY]][[[PBACK_ZERO]]], [[PR1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[PPHN_R1:%.*]] = arith.xori [[PPH_R1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[PR1_TOK:%.*]] = nvws.semaphore.acquire [[PFULL1]][[[PSLOT1]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // ASP: [[PR1_BUF:%.*]]:2 = nvws.semaphore.buffer [[PFULL1]][[[PSLOT1]]], [[PR1_TOK]] {ttg.partition = array<i32: 2>}
      // ASP: ttg.local_load [[PR1_BUF]]#0 {ttg.partition = array<i32: 2>}
      // ASP: nvws.semaphore.release [[PENTRY]][[[PSLOT1]]], [[PR1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // ASP: scf.yield {ttg.partition = array<i32: 2, 4>} [[PSLOT1]], [[PPHN_R0]], [[PPHN_R1]], [[PPHN_W0]], [[PPHN_W1]] : i32, i32, i32, i32, i32
      %large_read = ttg.local_load %large {ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #blocked64>
      "consume_large"(%large_read) {ttg.partition = array<i32: 2>} : (tensor<256x64xf16, #blocked64>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }

  // SEMA-LABEL: @tmem_fused_alias_depth_two
  // ASP-LABEL: @tmem_fused_alias_depth_two
  tt.func @tmem_fused_alias_depth_two(%lb: i32, %ub: i32, %step: i32) {
    %v0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %v1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>

    // The fused tmem allocation and its semaphores are hoisted to function
    // scope, ahead of the loop that contains the source tmem_allocs.
    // SEMA: [[TBASE:%.*]] = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // SEMA: [[TENTRY:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // SEMA: [[TFULL0:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // SEMA: [[TEMPTY1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // SEMA: [[TFULL1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // ASP: [[TBASE:%.*]] = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32}
    // ASP: [[TENTRY:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] released = -1 {pending_count = 1 : i32}
    // ASP: [[TFULL0:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32}
    // ASP: [[TEMPTY1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32}
    // ASP: [[TFULL1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32}
    // SEMA: scf.for
    // ASP: scf.for {{.*}} iter_args([[TCURSOR:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[TPH_R0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[TPH_R1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[TPH_W0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[TPH_W1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}})
    scf.for %iv = %lb to %ub step %step : i32 {
      // Member 0 write: the value-carrying tmem_alloc becomes a tmem_store
      // through view #0 with no token bracket.
      // SEMA: [[TW0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: [[TW0_TOK:%.*]] = nvws.semaphore.acquire [[TENTRY]][[[TW0_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // SEMA: [[TW0_BUF:%.*]]:2 = nvws.semaphore.buffer [[TENTRY]], [[TW0_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // SEMA: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[TW0_BUF]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // SEMA: [[TW0_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: nvws.semaphore.release [[TFULL0]][[[TW0_REL_ZERO]]], [[TW0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // ASP: [[TSLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[TPHN_W0:%.*]] = arith.xori [[TPH_W0]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // ASP: [[TW0_TOK:%.*]] = nvws.semaphore.acquire [[TENTRY]][[[TSLOT0]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // ASP: [[TW0_BUF:%.*]]:2 = nvws.semaphore.buffer [[TENTRY]][[[TSLOT0]]], [[TW0_TOK]] {ttg.partition = array<i32: 4>}
      // ASP: ttng.tmem_store {{%.*}}, [[TW0_BUF]]#0, {{%.*}} {ttg.partition = array<i32: 4>}
      // ASP: nvws.semaphore.release [[TFULL0]][[[TSLOT0]]], [[TW0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      %m0 = ttng.tmem_alloc %v0 {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
      // Member 0 read: acquire TFULL0, load view #0 with an empty token
      // bracket, release TEMPTY1 at the successor slot.
      // SEMA: [[TR0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[TR0_TOK:%.*]] = nvws.semaphore.acquire [[TFULL0]][[[TR0_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // SEMA: [[TR0_BUF:%.*]]:2 = nvws.semaphore.buffer [[TFULL0]], [[TR0_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // SEMA: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[TR0_BUF]]#0[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      // SEMA: [[T_TO_M1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // SEMA: nvws.semaphore.release [[TEMPTY1]][[[T_TO_M1]]], [[TR0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // ASP: [[TPHN_R0:%.*]] = arith.xori [[TPH_R0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TR0_TOK:%.*]] = nvws.semaphore.acquire [[TFULL0]][[[TSLOT0]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // ASP: [[TR0_BUF:%.*]]:2 = nvws.semaphore.buffer [[TFULL0]][[[TSLOT0]]], [[TR0_TOK]] {ttg.partition = array<i32: 2>}
      // ASP: ttng.tmem_load [[TR0_BUF]]#0[] {ttg.partition = array<i32: 2>}
      // ASP: [[T_TO_M1_RAW:%.*]] = arith.addi [[TSLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[T_TO_M1_REM:%.*]] = arith.remsi [[T_TO_M1_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[T_TO_M1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[T_TO_M1_REM]] {ttg.partition = array<i32: 2>} : i32
      // ASP: nvws.semaphore.release [[TEMPTY1]][[[T_TO_M1]]], [[TR0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      %r0, %t0 = ttng.tmem_load %m0[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      "consume0"(%r0) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
      // Member 1 write: acquire TEMPTY1, store through view #1, release
      // TFULL1.
      // SEMA: [[TW1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: [[TW1_TOK:%.*]] = nvws.semaphore.acquire [[TEMPTY1]][[[TW1_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // SEMA: [[TW1_BUF:%.*]]:2 = nvws.semaphore.buffer [[TEMPTY1]], [[TW1_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // SEMA: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[TW1_BUF]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // SEMA: [[TW1_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: nvws.semaphore.release [[TFULL1]][[[TW1_REL_ZERO]]], [[TW1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // ASP: [[TSLOT1_RAW:%.*]] = arith.addi [[TSLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[TSLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TSLOT1_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[TPHN_W1:%.*]] = arith.xori [[TPH_W1]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // ASP: [[TW1_TOK:%.*]] = nvws.semaphore.acquire [[TEMPTY1]][[[TSLOT1]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // ASP: [[TW1_BUF:%.*]]:2 = nvws.semaphore.buffer [[TEMPTY1]][[[TSLOT1]]], [[TW1_TOK]] {ttg.partition = array<i32: 4>}
      // ASP: ttng.tmem_store {{%.*}}, [[TW1_BUF]]#1, {{%.*}} {ttg.partition = array<i32: 4>}
      // ASP: nvws.semaphore.release [[TFULL1]][[[TSLOT1]]], [[TW1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      %m1 = ttng.tmem_alloc %v1 {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
      // Member 1 read: acquire TFULL1, load view #1, close the loop by
      // releasing TENTRY at the successor slot (TSLOT1 + 1) mod 2.
      // SEMA: [[TR1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[TR1_TOK:%.*]] = nvws.semaphore.acquire [[TFULL1]][[[TR1_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // SEMA: [[TR1_BUF:%.*]]:2 = nvws.semaphore.buffer [[TFULL1]], [[TR1_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // SEMA: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[TR1_BUF]]#1[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      // SEMA: [[T_TO_M0:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // SEMA: nvws.semaphore.release [[TENTRY]][[[T_TO_M0]]], [[TR1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // ASP: [[TPHN_R1:%.*]] = arith.xori [[TPH_R1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TR1_TOK:%.*]] = nvws.semaphore.acquire [[TFULL1]][[[TSLOT1]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // ASP: [[TR1_BUF:%.*]]:2 = nvws.semaphore.buffer [[TFULL1]][[[TSLOT1]]], [[TR1_TOK]] {ttg.partition = array<i32: 2>}
      // ASP: ttng.tmem_load [[TR1_BUF]]#1[] {ttg.partition = array<i32: 2>}
      // ASP: [[T_TO_M0_RAW:%.*]] = arith.addi [[TSLOT1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[T_TO_M0_REM:%.*]] = arith.remsi [[T_TO_M0_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[T_TO_M0:%.*]] = arith.select {{%.*}}, {{%.*}}, [[T_TO_M0_REM]] {ttg.partition = array<i32: 2>} : i32
      // ASP: nvws.semaphore.release [[TENTRY]][[[T_TO_M0]]], [[TR1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // ASP: scf.yield {ttg.partition = array<i32: 2, 4>} [[TSLOT1]], [[TPHN_R0]], [[TPHN_R1]], [[TPHN_W0]], [[TPHN_W1]] : i32, i32, i32, i32, i32
      %r1, %t1 = ttng.tmem_load %m1[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      "consume1"(%r1) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
