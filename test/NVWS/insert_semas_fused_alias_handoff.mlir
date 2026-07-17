// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s --check-prefix=SEMA
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas --nvws-assign-stage-phase -cse | FileCheck %s --check-prefix=ASP

// Two exact-alias epilogue members share one depth-2 physical allocation.
// The first member uses slot 0 and the second uses slot 1, so each
// read-to-next-write release must target the successor slot rather than the
// source read's slot.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared32 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SEMA-LABEL: @fused_alias_depth_two
  // ASP-LABEL: @fused_alias_depth_two
  tt.func @fused_alias_depth_two(%lb: i32, %ub: i32, %step: i32) {
    // Both member allocs collapse onto one fused depth-2 backing allocation;
    // every semaphore lists both (identical) member views as its buffers.
    // SEMA: [[BASE:%.*]] = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // SEMA: [[FULL0:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // SEMA: [[EMPTY1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // SEMA: [[FULL1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // ASP: [[BASE:%.*]] = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32}
    // ASP: [[ENTRY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] released = 3 {pending_count = 1 : i32}
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
    // SEMA: [[TENTRY:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // SEMA: [[TFULL0:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // SEMA: [[TEMPTY1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // SEMA: [[TFULL1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // ASP: [[TBASE:%.*]] = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32}
    // ASP: [[TENTRY:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] released = 3 {pending_count = 1 : i32}
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

  // A: both fresh-write epochs are inside one loop.
  // SEMA-LABEL: @case_a
  // ASP-LABEL: @case_a
  tt.func @case_a(%lb: i32, %ub: i32, %step: i32,
                  %lhs: !ttg.memdesc<128x64xf32, #shared32, #smem>,
                  %rhs: !ttg.memdesc<64x128xf32, #shared32, #smem>) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    // SEMA: [[A_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // SEMA: [[A_ENTRY:%.*]] = nvws.semaphore.create [[A_BASE]] released = 1 {pending_count = 1 : i32}
    // SEMA: [[A_NEXT:%.*]] = nvws.semaphore.create [[A_BASE]] released = 6 {pending_count = 1 : i32}
    // SEMA: [[A_INIT_TOK:%.*]] = nvws.semaphore.acquire [[A_ENTRY]] :
    // ASP: [[A_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // ASP: [[A_ENTRY:%.*]] = nvws.semaphore.create [[A_BASE]] released = 1 {pending_count = 1 : i32}
    // ASP: [[A_NEXT:%.*]] = nvws.semaphore.create [[A_BASE]] released = 6 {pending_count = 1 : i32}
    // ASP: [[A_INITIAL_CURRENT_STAGE:%.*]] = arith.constant 2 : i32
    // ASP: [[A_INITIAL_ONE:%.*]] = arith.constant 1 : i32
    // ASP: [[A_INITIAL_NEXT_RAW:%.*]] = arith.addi [[A_INITIAL_CURRENT_STAGE]], [[A_INITIAL_ONE]] : i32
    // ASP: [[A_INITIAL_DEPTH:%.*]] = arith.constant 3 : i32
    // ASP: [[A_INITIAL_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[A_INITIAL_NEXT_RAW]], [[A_INITIAL_DEPTH]] : i32
    // ASP: [[A_INITIAL_ZERO:%.*]] = arith.constant 0 : i32
    // ASP: [[A_INITIAL_NEXT_STAGE:%.*]] = arith.select [[A_INITIAL_NEEDS_WRAP]], [[A_INITIAL_ZERO]], [[A_INITIAL_NEXT_RAW]] : i32
    // ASP: [[A_INIT_TOK:%.*]] = nvws.semaphore.acquire [[A_ENTRY]][[[A_INITIAL_NEXT_STAGE]], {{%.*}}]
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // SEMA: scf.for {{.*}} iter_args([[A_W0_TOK:%.*]] = [[A_INIT_TOK]])
    // ASP: scf.for {{.*}} iter_args([[A_W0_TOK:%.*]] = [[A_INIT_TOK]], [[A_CURRENT_STAGE:%.*]] = [[A_INITIAL_NEXT_STAGE]],
    %outer = scf.for %i = %lb to %ub step %step iter_args(%carry = %tok) -> (!ttg.async.token) : i32 {
      // SEMA: [[A_W0_BUF:%.*]] = nvws.semaphore.buffer [[A_ENTRY]], [[A_W0_TOK]] {ttg.partition = array<i32: 0>}
      // SEMA: ttng.tmem_store {{%.*}}, [[A_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // SEMA: nvws.semaphore.release [[A_NEXT]], [[A_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[A_W0_BUF:%.*]] = nvws.semaphore.buffer [[A_ENTRY]][[[A_CURRENT_STAGE]]], [[A_W0_TOK]] {ttg.partition = array<i32: 0>}
      // ASP: ttng.tmem_store {{%.*}}, [[A_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[A_NEXT]][[[A_CURRENT_STAGE]]], [[A_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %w0 = ttng.tmem_store %zero, %acc[%carry], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // SEMA: [[A_W1_TOK:%.*]] = nvws.semaphore.acquire [[A_NEXT]] {ttg.partition = array<i32: 1>}
      // SEMA: [[A_W1_BUF:%.*]] = nvws.semaphore.buffer [[A_NEXT]], [[A_W1_TOK]] {ttg.partition = array<i32: 1>}
      // SEMA: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[A_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      // SEMA: nvws.semaphore.release [[A_ENTRY]], [[A_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[A_NEXT_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // ASP: [[A_NEXT_RAW:%.*]] = arith.addi [[A_CURRENT_STAGE]], [[A_NEXT_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[A_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // ASP: [[A_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[A_NEXT_RAW]], [[A_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[A_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // ASP: [[A_NEXT_STAGE:%.*]] = arith.select [[A_NEEDS_WRAP]], [[A_ZERO]], [[A_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[A_W1_TOK:%.*]] = nvws.semaphore.acquire [[A_NEXT]][[[A_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
      // ASP: [[A_W1_BUF:%.*]] = nvws.semaphore.buffer [[A_NEXT]][[[A_NEXT_STAGE]]], [[A_W1_TOK]] {ttg.partition = array<i32: 1>}
      // ASP: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[A_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      // ASP: nvws.semaphore.release [[A_ENTRY]][[[A_NEXT_STAGE]]], [[A_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %w1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%w0], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared32, #smem>, !ttg.memdesc<64x128xf32, #shared32, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // SEMA: [[A_READ_TOK:%.*]] = nvws.semaphore.acquire [[A_ENTRY]] {ttg.partition = array<i32: 0>}
      // SEMA: [[A_READ_BUF:%.*]] = nvws.semaphore.buffer [[A_ENTRY]], [[A_READ_TOK]] {ttg.partition = array<i32: 0>}
      // SEMA: {{%.*}}, {{%.*}} = ttng.tmem_load [[A_READ_BUF]][] {ttg.partition = array<i32: 0>}
      // ASP: [[A_READ_TOK:%.*]] = nvws.semaphore.acquire [[A_ENTRY]][[[A_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // ASP: [[A_READ_BUF:%.*]] = nvws.semaphore.buffer [[A_ENTRY]][[[A_NEXT_STAGE]]], [[A_READ_TOK]] {ttg.partition = array<i32: 0>}
      // ASP: {{%.*}}, {{%.*}} = ttng.tmem_load [[A_READ_BUF]][] {ttg.partition = array<i32: 0>}
      %value, %read = ttng.tmem_load %acc[%w1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_a"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %read : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 10 : i32}
    tt.return
  }

  // B: the first write precedes the loop, so the default all/none masks stay.
  // SEMA-LABEL: @case_b
  // ASP-LABEL: @case_b
  tt.func @case_b(%lb: i32, %ub: i32, %step: i32,
                  %lhs: !ttg.memdesc<128x64xf32, #shared32, #smem>,
                  %rhs: !ttg.memdesc<64x128xf32, #shared32, #smem>) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    // SEMA: [[B_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // SEMA: [[B_ENTRY:%.*]] = nvws.semaphore.create [[B_BASE]] released = 7 {pending_count = 1 : i32}
    // SEMA: [[B_FULL:%.*]] = nvws.semaphore.create [[B_BASE]] {pending_count = 1 : i32}
    // SEMA: [[B_INIT_TOK:%.*]] = nvws.semaphore.acquire [[B_ENTRY]] :
    // ASP: [[B_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // ASP: [[B_ENTRY:%.*]] = nvws.semaphore.create [[B_BASE]] released = 7 {pending_count = 1 : i32}
    // ASP: [[B_FULL:%.*]] = nvws.semaphore.create [[B_BASE]] {pending_count = 1 : i32}
    // ASP: [[B_INITIAL_CURRENT_STAGE:%.*]] = arith.constant 2 : i32
    // ASP: [[B_INITIAL_ONE:%.*]] = arith.constant 1 : i32
    // ASP: [[B_INITIAL_NEXT_RAW:%.*]] = arith.addi [[B_INITIAL_CURRENT_STAGE]], [[B_INITIAL_ONE]] : i32
    // ASP: [[B_INITIAL_DEPTH:%.*]] = arith.constant 3 : i32
    // ASP: [[B_INITIAL_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[B_INITIAL_NEXT_RAW]], [[B_INITIAL_DEPTH]] : i32
    // ASP: [[B_INITIAL_ZERO:%.*]] = arith.constant 0 : i32
    // ASP: [[B_INITIAL_NEXT_STAGE:%.*]] = arith.select [[B_INITIAL_NEEDS_WRAP]], [[B_INITIAL_ZERO]], [[B_INITIAL_NEXT_RAW]] : i32
    // ASP: [[B_INIT_TOK:%.*]] = nvws.semaphore.acquire [[B_ENTRY]][[[B_INITIAL_NEXT_STAGE]], {{%.*}}]
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // SEMA: [[B_W0_BUF:%.*]] = nvws.semaphore.buffer [[B_ENTRY]], [[B_INIT_TOK]]
    // SEMA: ttng.tmem_store {{%.*}}, [[B_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
    // ASP: [[B_W0_BUF:%.*]] = nvws.semaphore.buffer [[B_ENTRY]][[[B_INITIAL_NEXT_STAGE]]], [[B_INIT_TOK]]
    // ASP: ttng.tmem_store {{%.*}}, [[B_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
    %w0 = ttng.tmem_store %zero, %acc[%tok], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // SEMA: scf.for {{.*}} iter_args([[B_MMA_TOK:%.*]] = [[B_INIT_TOK]])
    // ASP: scf.for {{.*}} iter_args([[B_MMA_TOK:%.*]] = [[B_INIT_TOK]], [[B_CURRENT_STAGE:%.*]] = [[B_INITIAL_NEXT_STAGE]],
    %outer = scf.for %i = %lb to %ub step %step iter_args(%carry = %w0) -> (!ttg.async.token) : i32 {
      // SEMA: [[B_MMA_BUF:%.*]] = nvws.semaphore.buffer [[B_ENTRY]], [[B_MMA_TOK]] {ttg.partition = array<i32: 1>}
      // SEMA: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[B_MMA_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      // SEMA: nvws.semaphore.release [[B_FULL]], [[B_MMA_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[B_MMA_BUF:%.*]] = nvws.semaphore.buffer [[B_ENTRY]][[[B_CURRENT_STAGE]]], [[B_MMA_TOK]] {ttg.partition = array<i32: 1>}
      // ASP: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[B_MMA_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      // ASP: nvws.semaphore.release [[B_FULL]][[[B_CURRENT_STAGE]]], [[B_MMA_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %w1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%carry], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared32, #smem>, !ttg.memdesc<64x128xf32, #shared32, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // SEMA: [[B_READ_TOK:%.*]] = nvws.semaphore.acquire [[B_FULL]] {ttg.partition = array<i32: 0>}
      // SEMA: [[B_READ_BUF:%.*]] = nvws.semaphore.buffer [[B_FULL]], [[B_READ_TOK]] {ttg.partition = array<i32: 0>}
      // SEMA: {{%.*}}, {{%.*}} = ttng.tmem_load [[B_READ_BUF]][] {ttg.partition = array<i32: 0>}
      // SEMA: nvws.semaphore.release [[B_ENTRY]], [[B_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[B_READ_TOK:%.*]] = nvws.semaphore.acquire [[B_FULL]][[[B_CURRENT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // ASP: [[B_READ_BUF:%.*]] = nvws.semaphore.buffer [[B_FULL]][[[B_CURRENT_STAGE]]], [[B_READ_TOK]] {ttg.partition = array<i32: 0>}
      // ASP: {{%.*}}, {{%.*}} = ttng.tmem_load [[B_READ_BUF]][] {ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[B_ENTRY]][[[B_CURRENT_STAGE]]], [[B_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %value, %read = ttng.tmem_load %acc[%w1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_b"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // SEMA: [[B_NEXT_TOK:%.*]] = nvws.semaphore.acquire [[B_ENTRY]] {ttg.partition = array<i32: 1>}
      // ASP: [[B_NEXT_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // ASP: [[B_NEXT_RAW:%.*]] = arith.addi [[B_CURRENT_STAGE]], [[B_NEXT_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[B_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // ASP: [[B_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[B_NEXT_RAW]], [[B_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[B_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // ASP: [[B_NEXT_STAGE:%.*]] = arith.select [[B_NEEDS_WRAP]], [[B_ZERO]], [[B_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[B_NEXT_TOK:%.*]] = nvws.semaphore.acquire [[B_ENTRY]][[[B_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
      scf.yield {ttg.partition = array<i32: 0, 1>} %read : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 11 : i32}
    tt.return
  }

  // C: the loop-exit relay reserves the successor slot for the next W0.
  // SEMA-LABEL: @case_c
  // ASP-LABEL: @case_c
  tt.func @case_c(%lb: i32, %ub: i32, %step: i32,
                  %lhs: !ttg.memdesc<128x64xf32, #shared32, #smem>,
                  %rhs: !ttg.memdesc<64x128xf32, #shared32, #smem>) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    // SEMA: [[C_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // SEMA: [[C_ENTRY:%.*]] = nvws.semaphore.create [[C_BASE]] released = 1 {pending_count = 1 : i32}
    // SEMA: [[C_FULL:%.*]] = nvws.semaphore.create [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[C_FREE:%.*]] = nvws.semaphore.create [[C_BASE]] released = 6 {pending_count = 1 : i32}
    // ASP: [[C_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // ASP: [[C_ENTRY:%.*]] = nvws.semaphore.create [[C_BASE]] released = 1 {pending_count = 1 : i32}
    // ASP: [[C_FULL:%.*]] = nvws.semaphore.create [[C_BASE]] {pending_count = 1 : i32}
    // ASP: [[C_FREE:%.*]] = nvws.semaphore.create [[C_BASE]] released = 6 {pending_count = 1 : i32}
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // SEMA: scf.for
    // ASP: scf.for {{.*}} iter_args([[C_OUTER_CURRENT_STAGE:%[-A-Za-z0-9_.$#]+]] = {{%[-A-Za-z0-9_.$#]+}},
    %outer = scf.for %i = %lb to %ub step %step iter_args(%outer_token = %tok) -> (!ttg.async.token) : i32 {
      // SEMA: [[C_W0_TOK:%.*]] = nvws.semaphore.acquire [[C_ENTRY]] {ttg.partition = array<i32: 0>}
      // SEMA: [[C_W0_BUF:%.*]] = nvws.semaphore.buffer [[C_ENTRY]], [[C_W0_TOK]] {ttg.partition = array<i32: 0>}
      // SEMA: ttng.tmem_store {{%.*}}, [[C_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // SEMA: nvws.semaphore.release [[C_FREE]], [[C_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[C_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // ASP: [[C_OUTER_NEXT_RAW:%.*]] = arith.addi [[C_OUTER_CURRENT_STAGE]], [[C_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[C_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // ASP: [[C_OUTER_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[C_OUTER_NEXT_RAW]], [[C_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[C_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // ASP: [[C_NEXT_OF_OUTER_STAGE:%.*]] = arith.select [[C_OUTER_NEEDS_WRAP]], [[C_ZERO]], [[C_OUTER_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[C_W0_TOK:%.*]] = nvws.semaphore.acquire [[C_ENTRY]][[[C_NEXT_OF_OUTER_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // ASP: [[C_W0_BUF:%.*]] = nvws.semaphore.buffer [[C_ENTRY]][[[C_NEXT_OF_OUTER_STAGE]]], [[C_W0_TOK]] {ttg.partition = array<i32: 0>}
      // ASP: ttng.tmem_store {{%.*}}, [[C_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[C_FREE]][[[C_NEXT_OF_OUTER_STAGE]]], [[C_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %w0 = ttng.tmem_store %zero, %acc[%outer_token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // SEMA: scf.for
      // ASP: [[C_INNER_LOOP:%.*]]:3 = scf.for {{.*}} iter_args([[C_INNER_CURRENT_STAGE:%[-A-Za-z0-9_.$#]+]] = [[C_NEXT_OF_OUTER_STAGE]],
      %inner = scf.for %j = %lb to %ub step %step iter_args(%inner_token = %w0) -> (!ttg.async.token) : i32 {
        // SEMA: [[C_W1_TOK:%.*]] = nvws.semaphore.acquire [[C_FREE]] {ttg.partition = array<i32: 1>}
        // SEMA: [[C_W1_BUF:%.*]] = nvws.semaphore.buffer [[C_FREE]], [[C_W1_TOK]] {ttg.partition = array<i32: 1>}
        // SEMA: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[C_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
        // SEMA: nvws.semaphore.release [[C_FULL]], [[C_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // ASP: [[C_INNER_NEXT_RAW:%.*]] = arith.addi [[C_INNER_CURRENT_STAGE]], [[C_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
        // ASP: [[C_INNER_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[C_INNER_NEXT_RAW]], [[C_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
        // ASP: [[C_NEXT_OF_INNER_CURRENT_STAGE:%.*]] = arith.select [[C_INNER_NEEDS_WRAP]], [[C_ZERO]], [[C_INNER_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
        // ASP: [[C_W1_TOK:%.*]] = nvws.semaphore.acquire [[C_FREE]][[[C_NEXT_OF_INNER_CURRENT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
        // ASP: [[C_W1_BUF:%.*]] = nvws.semaphore.buffer [[C_FREE]][[[C_NEXT_OF_INNER_CURRENT_STAGE]]], [[C_W1_TOK]] {ttg.partition = array<i32: 1>}
        // ASP: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[C_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
        // ASP: nvws.semaphore.release [[C_FULL]][[[C_NEXT_OF_INNER_CURRENT_STAGE]]], [[C_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        %w1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%inner_token], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared32, #smem>, !ttg.memdesc<64x128xf32, #shared32, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // SEMA: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]] {ttg.partition = array<i32: 0>}
        // SEMA: [[C_READ_BUF:%.*]] = nvws.semaphore.buffer [[C_FULL]], [[C_READ_TOK]] {ttg.partition = array<i32: 0>}
        // SEMA: {{%.*}}, {{%.*}} = ttng.tmem_load [[C_READ_BUF]][] {ttg.partition = array<i32: 0>}
        // SEMA: nvws.semaphore.release [[C_FREE]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        // ASP: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]][[[C_NEXT_OF_INNER_CURRENT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
        // ASP: [[C_READ_BUF:%.*]] = nvws.semaphore.buffer [[C_FULL]][[[C_NEXT_OF_INNER_CURRENT_STAGE]]], [[C_READ_TOK]] {ttg.partition = array<i32: 0>}
        // ASP: {{%.*}}, {{%.*}} = ttng.tmem_load [[C_READ_BUF]][] {ttg.partition = array<i32: 0>}
        // ASP: nvws.semaphore.release [[C_FREE]][[[C_NEXT_OF_INNER_CURRENT_STAGE]]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        %value, %read = ttng.tmem_load %acc[%w1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "use_c"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // The accessless relay targets next(inner-stage) on both sides.
      // SEMA: [[C_ACQUIRE_AUTHORED_OFFSET_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // SEMA: [[C_EXIT_TOK:%.*]] = nvws.semaphore.acquire [[C_FREE]][[[C_ACQUIRE_AUTHORED_OFFSET_ONE]]] {ttg.partition = array<i32: 1>}
      // SEMA: [[C_RELEASE_AUTHORED_OFFSET_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // SEMA: nvws.semaphore.release [[C_ENTRY]][[[C_RELEASE_AUTHORED_OFFSET_ONE]]], [[C_EXIT_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[C_NEXT_OF_INNER_RAW:%.*]] = arith.addi [[C_INNER_LOOP]]#0, [[C_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[C_NEXT_OF_INNER_REM:%.*]] = arith.remsi [[C_NEXT_OF_INNER_RAW]], [[C_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[C_NEXT_OF_INNER_STAGE:%.*]] = arith.select {{.*}}, {{.*}}, [[C_NEXT_OF_INNER_REM]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[C_REL_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // ASP: [[C_EXIT_TOK:%.*]] = nvws.semaphore.acquire [[C_FREE]][[[C_NEXT_OF_INNER_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
      // ASP: [[C_RELEASE_NEXT_OF_INNER_RAW:%.*]] = arith.addi [[C_INNER_LOOP]]#0, [[C_REL_ONE]] {ttg.partition = array<i32: 1>} : i32
      // ASP: [[C_REL_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 3 : i32
      // ASP: [[C_RELEASE_NEXT_OF_INNER_REM:%.*]] = arith.remsi [[C_RELEASE_NEXT_OF_INNER_RAW]], [[C_REL_DEPTH]] {ttg.partition = array<i32: 1>} : i32
      // ASP: [[C_RELEASE_NEXT_OF_INNER_STAGE:%.*]] = arith.select {{.*}}, {{.*}}, [[C_RELEASE_NEXT_OF_INNER_REM]] {ttg.partition = array<i32: 1>} : i32
      // ASP: nvws.semaphore.release [[C_ENTRY]][[[C_RELEASE_NEXT_OF_INNER_STAGE]]], [[C_EXIT_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      scf.yield {ttg.partition = array<i32: 0, 1>} %inner : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 12 : i32}
    tt.return
  }

  // D: the acquire preceding the nonempty inner loop is the fresh epoch.
  // SEMA-LABEL: @case_d
  // ASP-LABEL: @case_d
  tt.func @case_d(%lb: i32, %ub: i32, %step: i32,
                  %lhs: !ttg.memdesc<128x64xf32, #shared32, #smem>,
                  %rhs: !ttg.memdesc<64x128xf32, #shared32, #smem>) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    // SEMA: [[D_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // SEMA: [[D_ENTRY:%.*]] = nvws.semaphore.create [[D_BASE]] released = 1 {pending_count = 1 : i32}
    // SEMA: [[D_NEXT:%.*]] = nvws.semaphore.create [[D_BASE]] released = 6 {pending_count = 1 : i32}
    // SEMA: [[D_INIT_TOK:%.*]] = nvws.semaphore.acquire [[D_ENTRY]] :
    // ASP: [[D_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // ASP: [[D_ENTRY:%.*]] = nvws.semaphore.create [[D_BASE]] released = 1 {pending_count = 1 : i32}
    // ASP: [[D_NEXT:%.*]] = nvws.semaphore.create [[D_BASE]] released = 6 {pending_count = 1 : i32}
    // ASP: [[D_INITIAL_CURRENT_STAGE:%.*]] = arith.constant 2 : i32
    // ASP: [[D_INITIAL_ONE:%.*]] = arith.constant 1 : i32
    // ASP: [[D_INITIAL_NEXT_RAW:%.*]] = arith.addi [[D_INITIAL_CURRENT_STAGE]], [[D_INITIAL_ONE]] : i32
    // ASP: [[D_INITIAL_DEPTH:%.*]] = arith.constant 3 : i32
    // ASP: [[D_INITIAL_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[D_INITIAL_NEXT_RAW]], [[D_INITIAL_DEPTH]] : i32
    // ASP: [[D_INITIAL_ZERO:%.*]] = arith.constant 0 : i32
    // ASP: [[D_INITIAL_NEXT_STAGE:%.*]] = arith.select [[D_INITIAL_NEEDS_WRAP]], [[D_INITIAL_ZERO]], [[D_INITIAL_NEXT_RAW]] : i32
    // ASP: [[D_INIT_TOK:%.*]] = nvws.semaphore.acquire [[D_ENTRY]][[[D_INITIAL_NEXT_STAGE]], {{%.*}}]
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // SEMA: scf.for {{.*}} iter_args([[D_W0_TOK:%.*]] = [[D_INIT_TOK]])
    // ASP: scf.for {{.*}} iter_args([[D_W0_TOK:%.*]] = [[D_INIT_TOK]], [[D_CURRENT_STAGE:%.*]] = [[D_INITIAL_NEXT_STAGE]],
    %outer = scf.for %i = %lb to %ub step %step iter_args(%outer_token = %tok) -> (!ttg.async.token) : i32 {
      // SEMA: [[D_W0_BUF:%.*]] = nvws.semaphore.buffer [[D_ENTRY]], [[D_W0_TOK]] {ttg.partition = array<i32: 0>}
      // SEMA: ttng.tmem_store {{%.*}}, [[D_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // SEMA: nvws.semaphore.release [[D_NEXT]], [[D_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[D_W0_BUF:%.*]] = nvws.semaphore.buffer [[D_ENTRY]][[[D_CURRENT_STAGE]]], [[D_W0_TOK]] {ttg.partition = array<i32: 0>}
      // ASP: ttng.tmem_store {{%.*}}, [[D_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[D_NEXT]][[[D_CURRENT_STAGE]]], [[D_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %w0 = ttng.tmem_store %zero, %acc[%outer_token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // SEMA: [[D_W1_TOK:%.*]] = nvws.semaphore.acquire [[D_NEXT]] {ttg.partition = array<i32: 1>}
      // SEMA: scf.for
      // ASP: [[D_NEXT_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // ASP: [[D_NEXT_RAW:%.*]] = arith.addi [[D_CURRENT_STAGE]], [[D_NEXT_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[D_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // ASP: [[D_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[D_NEXT_RAW]], [[D_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[D_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // ASP: [[D_NEXT_STAGE:%.*]] = arith.select [[D_NEEDS_WRAP]], [[D_ZERO]], [[D_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[D_W1_TOK:%.*]] = nvws.semaphore.acquire [[D_NEXT]][[[D_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
      // ASP: scf.for
      %inner = scf.for %j = %lb to %ub step %step iter_args(%inner_token = %w0) -> (!ttg.async.token) : i32 {
        // SEMA: [[D_W1_BUF:%.*]] = nvws.semaphore.buffer [[D_NEXT]], [[D_W1_TOK]] {ttg.partition = array<i32: 1>}
        // SEMA: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[D_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
        // ASP: [[D_W1_BUF:%.*]] = nvws.semaphore.buffer [[D_NEXT]][[[D_NEXT_STAGE]]], [[D_W1_TOK]] {ttg.partition = array<i32: 1>}
        // ASP: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[D_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
        %w1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%inner_token], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared32, #smem>, !ttg.memdesc<64x128xf32, #shared32, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1>} %w1 : !ttg.async.token
      } {ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
      // SEMA: nvws.semaphore.release [[D_ENTRY]], [[D_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // SEMA: [[D_READ_TOK:%.*]] = nvws.semaphore.acquire [[D_ENTRY]] {ttg.partition = array<i32: 0>}
      // SEMA: [[D_READ_BUF:%.*]] = nvws.semaphore.buffer [[D_ENTRY]], [[D_READ_TOK]] {ttg.partition = array<i32: 0>}
      // SEMA: {{%.*}}, {{%.*}} = ttng.tmem_load [[D_READ_BUF]][] {ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[D_ENTRY]][[[D_NEXT_STAGE]]], [[D_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[D_READ_TOK:%.*]] = nvws.semaphore.acquire [[D_ENTRY]][[[D_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // ASP: [[D_READ_BUF:%.*]] = nvws.semaphore.buffer [[D_ENTRY]][[[D_NEXT_STAGE]]], [[D_READ_TOK]] {ttg.partition = array<i32: 0>}
      // ASP: {{%.*}}, {{%.*}} = ttng.tmem_load [[D_READ_BUF]][] {ttg.partition = array<i32: 0>}
      %value, %read = ttng.tmem_load %acc[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_d"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %read : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 13 : i32}
    tt.return
  }
}
