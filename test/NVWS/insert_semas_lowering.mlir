// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas=num-stages=2 -cse | FileCheck %s --check-prefix=SEMA
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas=num-stages=2 --nvws-assign-semaphore-stage-phase -cse | FileCheck %s --check-prefix=ASP
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas=num-stages=2 --nvws-semaphore-optimize=num-stages=2 --nvws-assign-semaphore-stage-phase -cse | FileCheck %s --check-prefix=ASP
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas=num-stages=2 --nvws-semaphore-optimize=num-stages=2 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore=num-stages=2 -cse | FileCheck %s --check-prefix=LOWER --implicit-check-not=nvws.descriptor_load
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas=num-stages=2 --nvws-semaphore-optimize=num-stages=2 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore=num-stages=2 --triton-nvidia-tma-lowering --verify-each=false -cse | FileCheck %s --check-prefix=TMA

// Share each input across passes. The native attention capture (num-stages=2) is from
// python/test/unit/language/test_warp_specialization.py, with debug locations removed.


// Each fixture is shared by insertion, assignment, and lowering checks.
// SEMA-DAG: #[[$PARTIAL_LAYOUT:[a-zA-Z0-9_]+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared32 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Planner-authored aliases may be different views of one staged backing.
  // Here the smaller member covers the prefix of the larger member.  The
  // read-to-next-write handoff must still target the following physical slot.
  // SEMA-LABEL: @fused_partial_alias_depth_three
  // ASP-LABEL: @fused_partial_alias_depth_three
  // LOWER-LABEL: @fused_partial_alias_depth_three
  // TMA-LABEL: @fused_partial_alias_depth_three
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
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[PW0_BUF]]#1 {ttg.partition = array<i32: 4>} : tensor<128x64xf16, #[[$PARTIAL_LAYOUT]]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: nvws.semaphore.release [[PFULL0]][[[PW0_ZERO]]], [[PW0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
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
      // SEMA: ttg.local_load [[PR0_BUF]]#1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #[[$PARTIAL_LAYOUT]]>
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
      // SEMA: [[PW1_TOK:%.*]] = nvws.semaphore.acquire [[PHANDOFF]][[[PW0_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[PW1_BUF:%.*]]:2 = nvws.semaphore.buffer [[PHANDOFF]], [[PW1_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[PW1_BUF]]#0 {ttg.partition = array<i32: 4>} : tensor<256x64xf16, #[[$PARTIAL_LAYOUT]]> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      // SEMA: nvws.semaphore.release [[PFULL1]][[[PW0_ZERO]]], [[PW1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
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
      // SEMA: [[PR1_TOK:%.*]] = nvws.semaphore.acquire [[PFULL1]][[[PR0_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[PR1_BUF:%.*]]:2 = nvws.semaphore.buffer [[PFULL1]], [[PR1_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>
      // SEMA: ttg.local_load [[PR1_BUF]]#0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #[[$PARTIAL_LAYOUT]]>
      // SEMA: nvws.semaphore.release [[PENTRY]][[[PR0_ZERO]]], [[PR1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
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
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // SEMA-LABEL: @converted_descriptor_store_completion
  // ASP-LABEL: @converted_descriptor_store_completion
  // LOWER-LABEL: @converted_descriptor_store_completion
  // TMA-LABEL: @converted_descriptor_store_completion
  // Bind EMPTY through the initial producer wait, independently of FULL.
  // TMA: [[STORE_EMPTY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // TMA: [[STORE_ENTRY:%.*]] = ttg.memdesc_index [[STORE_EMPTY]][{{%.*}}] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
  // TMA-NEXT: ttng.wait_barrier [[STORE_ENTRY]],
  tt.func @converted_descriptor_store_completion(%desc: !tt.tensordesc<128x64xf16, #shared>, %i: i32, %lb: i32, %ub: i32, %step: i32) {
    // SEMA: [[V1:%.*]] = ttg.local_alloc {buffer.id = 601 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // SEMA: [[EMPTY:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32}
    // SEMA-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32}
    %alloc = ttg.local_alloc {buffer.id = 601 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // SEMA-NEXT: [[LOOP:%.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[WRITE:%.*]] = [[ENTRY]]) -> (!ttg.async.token)  : i32 {
    scf.for %iv = %lb to %ub step %step : i32 {
      %first = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #linear>
      // SEMA: [[WRITE_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[WRITE]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[WRITE_BUF]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: nvws.semaphore.release [[FULL]], [[WRITE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %first, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA-NEXT: [[READ:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: [[READ_BUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[READ]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: [[LOADED:%.*]] = ttg.local_load [[READ_BUF]] {ttg.partition = array<i32: 1>}
      // TMA: [[STORE_READ:%.*]] = ttg.memdesc_index {{%.*}}[[[STORE_SLOT:%[a-zA-Z0-9_]+]]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1x128x64xf16,
      // TMA-NEXT: {{%.*}} = ttg.local_load [[STORE_READ]]
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #linear>
      // SEMA-NEXT: [[CONVERTED:%.*]] = ttg.convert_layout [[LOADED]] {ttg.partition = array<i32: 1>}
      %converted = ttg.convert_layout %loaded {ttg.partition = array<i32: 1>} : tensor<128x64xf16, #linear> -> tensor<128x64xf16, #blocked>
      // The consumer release comes AFTER the descriptor store even with the
      // intervening layout conversion between the load and the store.
      // SEMA-NEXT: tt.descriptor_store %{{.*}}[%{{.*}}, %{{.*}}], [[CONVERTED]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: nvws.semaphore.release [[EMPTY]], [[READ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // The lowered descriptor read completes before its ownership release.
      // TMA: ttng.async_tma_copy_local_to_global
      // TMA-NEXT: ttng.async_tma_store_wait {pendings = 0 : i32}
      // TMA-NEXT: [[STORE_RETURN:%.*]] = ttg.memdesc_index [[STORE_EMPTY]][[[STORE_SLOT]]] {ttg.partition = array<i32: 1>}
      // TMA-NEXT: ttng.arrive_barrier [[STORE_RETURN]], 1 {ttg.partition = array<i32: 1>}
      tt.descriptor_store %desc[%i, %i], %converted {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared>, tensor<128x64xf16, #blocked>
      %next = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #linear>
      // SEMA: [[NEXT_WRITE:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: [[NEXT_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[NEXT_WRITE]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[NEXT_BUF]] {ttg.partition = array<i32: 0>}
      ttg.local_store %next, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>} [[NEXT_WRITE]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    // SEMA: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#reg1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#reg = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory
#tm = #ttng.tensor_memory
!acc = !ttg.memdesc<128x128xf32, #tmem, #tm, mutable>
!tile = tensor<128x128xf32, #reg>
!lhs = !ttg.memdesc<128x64xf16, #shared, #smem>
!rhs = !ttg.memdesc<64x128xf16, #shared, #smem>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SEMA-LABEL: @smem_nested_fanin
  // ASP-LABEL: @smem_nested_fanin
  // ASP: [[FANIN_ASP:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 2 : i32}
  // ASP: scf.for
  // ASP: nvws.semaphore.release [[FANIN_ASP]]{{.*}}[#nvws.async_op<none>] {arrive_count = 2 : i32,
  // ASP-NEXT: {{.*}}scf.for
  // LOWER-LABEL: @smem_nested_fanin
  // LOWER: [[FANIN_STORAGE:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // LOWER: ttng.init_barrier {{.*}}, 2
  // LOWER: scf.for
  // LOWER: [[FANIN_RELAY:%.*]] = ttg.memdesc_index [[FANIN_STORAGE]][
  // LOWER: ttng.arrive_barrier [[FANIN_RELAY]], 2
  // LOWER-NEXT: scf.for
  // LOWER: ttng.wait_barrier
  // LOWER: ttg.local_load
  // TMA-LABEL: @smem_nested_fanin
  tt.func @smem_nested_fanin(%value: tensor<1xi32, #reg1>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %buffer = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 102 : i32} : () -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
    // SEMA: [[FANIN_READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 2 : i32}
    // SEMA: [[FANIN_ENTRY:%.*]] = nvws.semaphore.acquire [[FANIN_READY]]
    // SEMA: ttg.local_store
    ttg.local_store %value, %buffer {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : tensor<1xi32, #reg1> -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
    // SEMA: scf.for {{.*}} iter_args([[FANIN_IN:%.*]] = [[FANIN_ENTRY]])
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      // The incoming permit supplies two arrivals for the two independent completions.
      // SEMA: nvws.semaphore.release [[FANIN_READY]], [[FANIN_IN]] [#nvws.async_op<none>] {arrive_count = 2 : i32, ttg.partition = array<i32: 2>}
      // SEMA-NEXT: scf.for
      scf.for %j = %c0 to %c2 step %c1 : i32 {
        // SEMA: nvws.semaphore.acquire [[FANIN_READY]]
        %first = ttg.local_load %buffer {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #reg1>
        "consume_first"(%first) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #reg1>) -> ()
        %second = ttg.local_load %buffer {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #reg1>
        %corrected = arith.addi %second, %second {ttg.partition = array<i32: 1>} : tensor<1xi32, #reg1>
        // SEMA: ttg.local_store
        ttg.local_store %corrected, %buffer {ttg.partition = array<i32: 1>} : tensor<1xi32, #reg1> -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
        // SEMA: nvws.semaphore.release [[FANIN_READY]]{{.*}}[#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // SEMA: ttg.local_load
        %last = ttg.local_load %buffer {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #reg1>
        // SEMA: nvws.semaphore.release [[FANIN_READY]]{{.*}}[#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        "consume_last"(%last) {ttg.partition = array<i32: 0>} : (tensor<1xi32, #reg1>) -> ()
      } {ttg.partition = array<i32: 0, 1, 2>}
      // SEMA: [[FANIN_OUT:%.*]] = nvws.semaphore.acquire [[FANIN_READY]]
      %after = ttg.local_load %buffer {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #reg1>
      // SEMA: "consume_final"
      "consume_final"(%after) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #reg1>) -> ()
      // SEMA: scf.yield {{.*}}[[FANIN_OUT]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#reg1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#reg = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory
#tm = #ttng.tensor_memory
!acc = !ttg.memdesc<128x128xf32, #tmem, #tm, mutable>
!tile = tensor<128x128xf32, #reg>
!lhs = !ttg.memdesc<128x64xf16, #shared, #smem>
!rhs = !ttg.memdesc<64x128xf16, #shared, #smem>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SEMA-LABEL: @reader_first_conditional
  // ASP-LABEL: @reader_first_conditional
  // ASP: nvws.semaphore.create
  // LOWER-LABEL: @reader_first_conditional
  // LOWER: [[NESTED_READY_STORAGE:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // LOWER: ttng.init_barrier
  // LOWER: scf.for
  // LOWER: scf.if
  // LOWER: scf.for
  // LOWER: [[NESTED_RELAY_VIEW:%.*]] = ttg.memdesc_index [[NESTED_READY_STORAGE]][
  // LOWER: ttng.arrive_barrier [[NESTED_RELAY_VIEW]], 1
  // LOWER-NEXT: {{.*}}scf.for
  // LOWER: [[NESTED_WAIT_VIEW:%.*]] = ttg.memdesc_index [[NESTED_READY_STORAGE]][
  // LOWER: ttng.wait_barrier [[NESTED_WAIT_VIEW]],
  // LOWER: ttng.tmem_load
  // LOWER: } else {
  // LOWER-NOT: ttng.arrive_barrier
  // LOWER: scf.yield
  // TMA-LABEL: @reader_first_conditional
  tt.func @reader_first_conditional(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    // SEMA: [[IF_READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
    // SEMA: [[IF_ENTRY:%.*]] = nvws.semaphore.acquire [[IF_READY]]
    // SEMA: scf.for {{.*}} iter_args([[IF_IN:%.*]] = [[IF_ENTRY]])
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      // SEMA: [[BRANCH:%.*]] = scf.if
      scf.if %guard {
        // SEMA: [[IF_MIDDLE:%.*]] = scf.for {{.*}} iter_args([[IF_MIDDLE_IN:%.*]] = [[IF_IN]])
        scf.for %middle = %c0 to %c2 step %c1 : i32 {
          // Supply belongs to the taken branch; the unchanged alternative keeps its token.
          // SEMA: nvws.semaphore.release [[IF_READY]], [[IF_MIDDLE_IN]] [#nvws.async_op<none>]
          // SEMA-NEXT: {{.*}}scf.for
          %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
            %value, %read = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 0>} : !acc -> !tile
            %corrected = math.exp2 %value {ttg.partition = array<i32: 0>} : !tile
            %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
            // SEMA: ttng.tc_gen5_mma
            %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %use_acc, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
            // SEMA: nvws.semaphore.release [[IF_READY]]{{.*}}[#nvws.async_op<tc5mma>]
            scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
          } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
          // SEMA: [[IF_OUT:%.*]] = nvws.semaphore.acquire [[IF_READY]]
          %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
          // SEMA: "consume"
          "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
          // SEMA: scf.yield {{.*}}[[IF_OUT]] : !ttg.async.token
        } {ttg.partition = array<i32: 0, 1>}
        // SEMA: scf.yield {{.*}}[[IF_MIDDLE]] : !ttg.async.token
      // SEMA: } else {
        // SEMA-NOT: nvws.semaphore.release
        // SEMA: scf.yield {{.*}}[[IF_IN]] : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>}
      // SEMA: scf.yield {{.*}}[[BRANCH]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#reg1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#reg = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory
#tm = #ttng.tensor_memory
!acc = !ttg.memdesc<128x128xf32, #tmem, #tm, mutable>
!tile = tensor<128x128xf32, #reg>
!lhs = !ttg.memdesc<128x64xf16, #shared, #smem>
!rhs = !ttg.memdesc<64x128xf16, #shared, #smem>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SEMA-LABEL: @initialized_reader_first
  // ASP-LABEL: @initialized_reader_first
  // ASP: nvws.semaphore.create
  // LOWER-LABEL: @initialized_reader_first
  // LOWER: ttng.init_barrier
  // TMA-LABEL: @initialized_reader_first
  tt.func @initialized_reader_first(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    // SEMA: [[INIT_READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
    // SEMA: [[INIT_ENTRY:%.*]] = nvws.semaphore.acquire [[INIT_READY]]
    // SEMA: scf.for {{.*}} iter_args([[INIT_IN:%.*]] = [[INIT_ENTRY]])
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      // SEMA: [[INIT_BUFFER:%.*]] = nvws.semaphore.buffer [[INIT_READY]], [[INIT_IN]]
      // SEMA: ttng.tmem_store {{.*}}, [[INIT_BUFFER]][]
      %init = ttng.tmem_store %zero, %acc[%alloc_tok], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
      // SEMA-NEXT: nvws.semaphore.release [[INIT_READY]], [[INIT_IN]] [#nvws.async_op<none>]
      // SEMA-NEXT: {{.*}}scf.for
      %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%carry = %init, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
        %value, %read = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 0>} : !acc -> !tile
        %corrected = math.exp2 %value {ttg.partition = array<i32: 0>} : !tile
        %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %use_acc, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        // SEMA: nvws.semaphore.release [[INIT_READY]]{{.*}}[#nvws.async_op<tc5mma>]
        scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      // SEMA: [[INIT_RETURN:%.*]] = nvws.semaphore.acquire [[INIT_READY]]
      %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      // SEMA: "consume"
      "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
      // SEMA: scf.yield {{.*}}[[INIT_RETURN]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared32 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // C: the loop-exit relay reserves the successor slot for the next W0.
  // SEMA-LABEL: @case_c
  // ASP-LABEL: @case_c
  // LOWER-LABEL: @case_c
  // TMA-LABEL: @case_c
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
      // SEMA: nvws.semaphore.release [[C_ENTRY]][[[C_ACQUIRE_AUTHORED_OFFSET_ONE]]], [[C_EXIT_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
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
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // SEMA-LABEL: @already_lowered_tma_store_handoffs
  // ASP-LABEL: @already_lowered_tma_store_handoffs
  // LOWER-LABEL: @already_lowered_tma_store_handoffs
  // TMA-LABEL: @already_lowered_tma_store_handoffs
  tt.func @already_lowered_tma_store_handoffs(
      %desc: !tt.tensordesc<128x64xf32, #shared>,
      %lb: i32, %ub: i32, %step: i32) {
    %v0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %v1 = arith.constant dense<1.000000e+00> : tensor<128x64xf32, #blocked>
    // These exact-alias members model two consecutive output slices in one
    // depth-2 physical staging allocation.
    // SEMA: [[BASE:%.*]] = ttg.local_alloc
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] released = 3
    // SEMA-NEXT: [[COPY_READY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]]
    // SEMA-NEXT: [[M1_READY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]]
    // SEMA-NEXT: [[REDUCE_READY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]]
    %m0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 602 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    %m1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 602 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      // SEMA: [[ZERO_P0:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      // SEMA-NEXT: [[W0_TOKEN:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[ZERO_P0]]]
      // SEMA-NEXT: [[W0_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[ENTRY]], [[W0_TOKEN]]
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[W0_BUFFER]]#0
      // SEMA-NEXT: nvws.semaphore.release [[COPY_READY]][[[ZERO_P0]]], [[W0_TOKEN]]
      ttg.local_store %v0, %m0 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      // The TMA copy is a read of slot 0. Its release must stay after the
      // token wait and hand the next writer slot 1.
      // SEMA-NEXT: [[ZERO_P1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // SEMA-NEXT: [[COPY_TOKEN:%.*]] = nvws.semaphore.acquire [[COPY_READY]][[[ZERO_P1]]]
      // SEMA-NEXT: [[COPY_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[COPY_READY]], [[COPY_TOKEN]]
      // SEMA-NEXT: [[COPY:%.*]] = ttng.async_tma_copy_local_to_global %{{.*}} [[COPY_BUFFER]]#0
      // SEMA-NEXT: ttng.async_tma_store_token_wait [[COPY]]
      // SEMA-NEXT: [[TO_M1:%.*]] = arith.constant {{.*}} 1 : i32
      // SEMA-NEXT: nvws.semaphore.release [[M1_READY]][[[TO_M1]]], [[COPY_TOKEN]]
      %copy = ttng.async_tma_copy_local_to_global %desc[%i, %i] %m0 {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xf32, #shared>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %copy {ttg.partition = array<i32: 1>} : !ttg.async.token
      // SEMA-NEXT: [[W1_TOKEN:%.*]] = nvws.semaphore.acquire [[M1_READY]][[[ZERO_P0]]]
      // SEMA-NEXT: [[W1_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[M1_READY]], [[W1_TOKEN]]
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[W1_BUFFER]]#1
      // SEMA-NEXT: nvws.semaphore.release [[REDUCE_READY]][[[ZERO_P0]]], [[W1_TOKEN]]
      ttg.local_store %v1, %m1 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      // Async reduce has the same SMEM-read lifetime and must release only
      // after its completion wait, back to slot 0 of the next iteration.
      // SEMA-NEXT: [[REDUCE_TOKEN:%.*]] = nvws.semaphore.acquire [[REDUCE_READY]][[[ZERO_P1]]]
      // SEMA-NEXT: [[REDUCE_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[REDUCE_READY]], [[REDUCE_TOKEN]]
      // SEMA-NEXT: [[REDUCE:%.*]] = ttng.async_tma_reduce add, %{{.*}} [[REDUCE_BUFFER]]#1
      // SEMA-NEXT: ttng.async_tma_store_token_wait [[REDUCE]]
      // SEMA-NEXT: nvws.semaphore.release [[ENTRY]][[[TO_M1]]], [[REDUCE_TOKEN]]
      %reduce = ttng.async_tma_reduce add, %desc[%i, %i] %m1 {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xf32, #shared>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %reduce {ttg.partition = array<i32: 1>} : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#reg1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#reg = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory
#tm = #ttng.tensor_memory
!acc = !ttg.memdesc<128x128xf32, #tmem, #tm, mutable>
!tile = tensor<128x128xf32, #reg>
!lhs = !ttg.memdesc<128x64xf16, #shared, #smem>
!rhs = !ttg.memdesc<64x128xf16, #shared, #smem>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SEMA-LABEL: @reader_first_depth_1
  // ASP-LABEL: @reader_first_depth_1
  // ASP: [[INITIAL_SLOT:%.*]] = arith.constant 0 : i32
  // ASP: [[READY_ASP:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
  // ASP: [[TO_MMA_ASP:%.*]] = nvws.semaphore.create
  // ASP: [[INITIAL_BITS:%.*]] = arith.constant -2 : i32
  // ASP: [[PHASE_ONE:%.*]] = arith.constant {{.*}} 1 : i32
  // ASP: [[INITIAL_BIT:%.*]] = arith.shli [[PHASE_ONE]], [[INITIAL_SLOT]]
  // ASP: [[ENTRY_BITS:%.*]] = arith.xori [[INITIAL_BITS]], [[INITIAL_BIT]]
  // ASP: [[INITIAL_SHIFT:%.*]] = arith.shrui [[ENTRY_BITS]], [[INITIAL_SLOT]]
  // ASP: [[ENTRY_PHASE:%.*]] = arith.andi [[INITIAL_SHIFT]], [[PHASE_ONE]]
  // ASP: [[ENTRY_ASP:%.*]] = nvws.semaphore.acquire [[READY_ASP]][[[INITIAL_SLOT]], [[ENTRY_PHASE]]]
  // ASP: scf.for {{.*}} iter_args([[OUTER_TOKEN_ASP:%.*]] = [[ENTRY_ASP]], [[OUTER_SLOT:%.*]] = [[INITIAL_SLOT]], [[OUTER_BITS:%.*]] = [[ENTRY_BITS]],
  // ASP: nvws.semaphore.release [[READY_ASP]][[[OUTER_SLOT]]], [[OUTER_TOKEN_ASP]] [#nvws.async_op<none>]
  // ASP: [[INNER_RESULT:%[a-zA-Z0-9_]+]]:{{[0-9]+}} = scf.for {{.*}} iter_args({{.*}}[[INNER_SLOT:%.*]] = [[OUTER_SLOT]], [[INNER_BITS:%.*]] = [[OUTER_BITS]],
  // ASP: [[INNER_BIT:%.*]] = arith.shli {{%.*}}, [[INNER_SLOT]]
  // ASP: [[NEXT_READER_BITS:%.*]] = arith.xori [[INNER_BITS]], [[INNER_BIT]]
  // ASP: [[INNER_SHIFT:%.*]] = arith.shrui [[NEXT_READER_BITS]], [[INNER_SLOT]]
  // ASP: [[INNER_PHASE:%.*]] = arith.andi [[INNER_SHIFT]],
  // ASP: [[READ_ASP:%.*]] = nvws.semaphore.acquire [[READY_ASP]][[[INNER_SLOT]], [[INNER_PHASE]]]
  // ASP: ttng.tmem_load
  // ASP: [[MMA_SLOT:%.*]] = arith.select
  // ASP: [[MMA_ASP:%.*]] = nvws.semaphore.acquire [[TO_MMA_ASP]][[[MMA_SLOT]],
  // ASP: ttng.tc_gen5_mma
  // ASP: nvws.semaphore.release [[READY_ASP]][[[MMA_SLOT]]], [[MMA_ASP]] [#nvws.async_op<tc5mma>]
  // ASP: scf.yield {{.*}}[[MMA_SLOT]], [[NEXT_READER_BITS]],
  // ASP: [[FINAL_ASP:%.*]] = nvws.semaphore.acquire [[READY_ASP]][[[INNER_RESULT]]#1,
  // ASP: "consume"
  // ASP: scf.yield {{.*}}[[FINAL_ASP]], [[INNER_RESULT]]#1,
  // LOWER-LABEL: @reader_first_depth_1
  // LOWER: [[READY_STORAGE:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // LOWER: ttng.init_barrier
  // LOWER: scf.for
  // LOWER: [[RELAY_VIEW:%.*]] = ttg.memdesc_index [[READY_STORAGE]][
  // LOWER: ttng.arrive_barrier [[RELAY_VIEW]], 1
  // LOWER-NEXT: {{.*}}scf.for
  // LOWER: [[FIRST_WAIT_VIEW:%.*]] = ttg.memdesc_index [[READY_STORAGE]][
  // LOWER: ttng.wait_barrier [[FIRST_WAIT_VIEW]],
  // LOWER: ttng.tmem_load
  // TMA-LABEL: @reader_first_depth_1
  tt.func @reader_first_depth_1(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    // The parent acquires once and carries that permit across outer iterations.
    // SEMA: [[READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.acquire [[READY]]
    // SEMA: scf.for {{.*}} iter_args([[OUTER_IN:%.*]] = [[ENTRY]])
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      // Relay the incoming permit before the nested reader acquires it.
      // SEMA: nvws.semaphore.release [[READY]], [[OUTER_IN]] [#nvws.async_op<none>]
      // SEMA-NEXT: {{.*}}scf.for
      %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
        // SEMA: [[INNER_TOKEN:%.*]] = nvws.semaphore.acquire [[READY]]
        // SEMA: ttng.tmem_load
        %value, %read = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 0>} : !acc -> !tile
        %corrected = math.exp2 %value {ttg.partition = array<i32: 0>} : !tile
        %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
        // SEMA: nvws.semaphore.release [[TO_MMA:%.*]], [[INNER_TOKEN]] [#nvws.async_op<none>]
        // SEMA: [[MMA_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_MMA]]
        // SEMA: ttng.tc_gen5_mma
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %use_acc, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        // SEMA: nvws.semaphore.release [[READY]], [[MMA_TOKEN]] [#nvws.async_op<tc5mma>]
        scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      // SEMA: [[RETURNED:%.*]] = nvws.semaphore.acquire [[READY]]
      %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      // SEMA: "consume"
      "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
    // The post-inner acquisition is the next outer iteration's incoming token.
      // SEMA: scf.yield {{.*}}[[RETURNED]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SEMA-LABEL: @cross_partition_accumulating_mma
  // ASP-LABEL: @cross_partition_accumulating_mma
  // The initialized accumulator keeps its acquired slot throughout the inner
  // recurrence: useAccumulator=true must not start a fresh-write epoch.
  // ASP: [[ACC_TRUE:%.*]] = arith.constant true
  // ASP: [[ACC_BACKING:%.*]] = ttng.tmem_alloc {buffer.copy = 2 : i32} : () -> !ttg.memdesc<2x128x128xf32,
  // ASP-NEXT: [[ACC_EMPTY:%.*]] = nvws.semaphore.create [[ACC_BACKING]] released = 3 {pending_count = 1 : i32}
  // ASP-NEXT: [[ACC_FULL:%.*]] = nvws.semaphore.create [[ACC_BACKING]] {pending_count = 1 : i32}
  // ASP: [[ACC_ENTRY:%.*]] = nvws.semaphore.acquire [[ACC_EMPTY]][[[ACC_SLOT:%[a-zA-Z0-9_]+]], {{%.*}}] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 2 : i32}
  // ASP-NEXT: {{.*}}scf.for {{.*}} iter_args([[ACC_OUTER:%[a-zA-Z0-9_]+]] = [[ACC_ENTRY]],
  // ASP-NEXT: [[ACC_INIT_VIEW:%.*]] = nvws.semaphore.buffer [[ACC_EMPTY]][[[ACC_SLOT]]], [[ACC_OUTER]] {ttg.partition = array<i32: 0>}
  // ASP-NEXT: {{%.*}} = ttng.tmem_store {{%.*}}, [[ACC_INIT_VIEW]][], [[ACC_TRUE]] {ttg.partition = array<i32: 0>}
  // ASP-NEXT: nvws.semaphore.release [[ACC_FULL]][[[ACC_SLOT]]], [[ACC_OUTER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
  // ASP: [[ACC_MMA_TOKEN:%.*]] = nvws.semaphore.acquire [[ACC_FULL]][[[ACC_SLOT]], {{%.*}}] {ttg.partition = array<i32: 2>}
  // ASP-NEXT: scf.for
  // ASP-NEXT: [[ACC_LHS:%.*]] = "load1"
  // ASP-NEXT: [[ACC_RHS:%.*]] = "load2"
  // ASP-NEXT: [[ACC_MMA_VIEW:%.*]] = nvws.semaphore.buffer [[ACC_FULL]][[[ACC_SLOT]]], [[ACC_MMA_TOKEN]] {ttg.partition = array<i32: 2>}
  // ASP-NEXT: {{%.*}} = ttng.tc_gen5_mma [[ACC_LHS]], [[ACC_RHS]], [[ACC_MMA_VIEW]][], [[ACC_TRUE]], [[ACC_TRUE]] {ttg.partition = array<i32: 2>}
  // ASP-NEXT: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = []}
  // ASP-NEXT: nvws.semaphore.release [[ACC_EMPTY]][[[ACC_SLOT]]], [[ACC_MMA_TOKEN]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
  // ASP: [[ACC_READ_TOKEN:%.*]] = nvws.semaphore.acquire [[ACC_EMPTY]][[[ACC_SLOT]], {{%.*}}] {ttg.partition = array<i32: 0>}
  // ASP-NEXT: [[ACC_READ_VIEW:%.*]] = nvws.semaphore.buffer [[ACC_EMPTY]][[[ACC_SLOT]]], [[ACC_READ_TOKEN]] {ttg.partition = array<i32: 0>}
  // ASP-NEXT: [[ACC_VALUE:%.*]], {{%.*}} = ttng.tmem_load [[ACC_READ_VIEW]][] {ttg.partition = array<i32: 0>}
  // ASP-NEXT: "use"([[ACC_VALUE]])
  // ASP-NEXT: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[ACC_READ_TOKEN]], {{%.*}}, {{%.*}} : !ttg.async.token, i32, i32
  // LOWER-LABEL: @cross_partition_accumulating_mma
  // TMA-LABEL: @cross_partition_accumulating_mma
  tt.func @cross_partition_accumulating_mma(%lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %res, %tok = ttng.tmem_alloc {buffer.copy = 2 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // SEMA: ttng.tmem_alloc {buffer.copy = 2 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%outerTok = %tok) -> (!ttg.async.token) : i32 {
      %storeTok = ttng.tmem_store %cst, %res[%outerTok], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %inner = scf.for %iv = %lb to %ub step %step iter_args(%innerTok = %storeTok) -> (!ttg.async.token) : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        %mmaTok = ttng.tc_gen5_mma %sA, %sB, %res[%innerTok], %true, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %mmaTok : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
      %value, %readTok = ttng.tmem_load %res[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %readTok : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // SEMA-LABEL: @attention_persistent_inner_loop_kernel(
  // ASP-LABEL: @attention_persistent_inner_loop_kernel(
  // LOWER-LABEL: @attention_persistent_inner_loop_kernel(
  // LOWER-SAME: [[CAP_Q:%[-A-Za-z0-9_.$#]+]]: !tt.tensordesc
  // LOWER-SAME: [[CAP_K:%[-A-Za-z0-9_.$#]+]]: !tt.tensordesc
  // LOWER-SAME: [[CAP_V:%[-A-Za-z0-9_.$#]+]]: !tt.tensordesc
  // LOWER-SAME: [[CAP_O:%[-A-Za-z0-9_.$#]+]]: !tt.tensordesc
  tt.func public @attention_persistent_inner_loop_kernel(
      %desc_q: !tt.tensordesc<128x128xf16, #shared>,
      %desc_q.shape.0: i32,
      %desc_q.shape.1: i32,
      %desc_q.stride.0: i64,
      %desc_q.stride.1: i64,
      %desc_k: !tt.tensordesc<128x128xf16, #shared>,
      %desc_k.shape.0: i32,
      %desc_k.shape.1: i32,
      %desc_k.stride.0: i64,
      %desc_k.stride.1: i64,
      %desc_v: !tt.tensordesc<128x128xf16, #shared>,
      %desc_v.shape.0: i32,
      %desc_v.shape.1: i32,
      %desc_v.stride.0: i64,
      %desc_v.stride.1: i64,
      %desc_acc: !tt.tensordesc<128x128xf16, #shared>,
      %desc_acc.shape.0: i32,
      %desc_acc.shape.1: i32,
      %desc_acc.stride.0: i64,
      %desc_acc.stride.1: i64,
      %l_i_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %m_i_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %qk_scale: f32) attributes {noinline = false} {
    // SEMA: [[CAP_FALSE:%.*]] = arith.constant false
    // SEMA: [[CAP_TRUE:%.*]] = arith.constant true
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %num_tiles = arith.constant 127 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %prog_id = tt.get_program_id x : i32
    %num_sm = tt.get_num_programs x : i32
    %num_tiles_2 = arith.addi %M, %num_tiles : i32
    %num_tiles_3 = arith.divsi %num_tiles_2, %c128_i32 : i32
    %tiles_per_sm = arith.divsi %num_tiles_3, %num_sm : i32
    %0 = arith.remsi %num_tiles_3, %num_sm : i32
    %1 = arith.cmpi slt, %prog_id, %0 : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_6 = arith.addi %tiles_per_sm, %c1_i32 : i32
      scf.yield %tiles_per_sm_6 : i32
    } else {
      scf.yield %tiles_per_sm : i32
    }
    %m_ij = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %qk = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #linear>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    // The initialized accumulator owns one slot; its permit crosses outer trips.
    // SEMA: [[CAP_ACC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // SEMA-NEXT: [[CAP_READY:%.*]] = nvws.semaphore.create [[CAP_ACC]] released = 1 {pending_count = 1 : i32}
    // SEMA-NEXT: [[CAP_DONE:%.*]] = nvws.semaphore.create [[CAP_ACC]] {pending_count = 1 : i32}
    // ASP: [[CAP_AACC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // ASP-NEXT: [[CAP_AREADY:%.*]] = nvws.semaphore.create [[CAP_AACC]] released = 1 {pending_count = 1 : i32}
    // ASP-NEXT: [[CAP_ADONE:%.*]] = nvws.semaphore.create [[CAP_AACC]] {pending_count = 1 : i32}
    // ASP: [[CAP_INITIAL_BITS:%.*]] = arith.constant -2 : i32
    // LOWER: [[CAP_LACC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // LOWER-NEXT: [[CAP_BARRIER:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
    // LOWER-NEXT: [[CAP_INITIAL_BARRIER:%.*]] = ttg.memdesc_index [[CAP_BARRIER]][
    // LOWER-NEXT: ttng.init_barrier [[CAP_INITIAL_BARRIER]], 1
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_4 = ub.poison : !ttg.async.token
    %acc_5 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %q = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %k = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // SEMA: [[CAP_ENTRY:%.*]] = nvws.semaphore.acquire [[CAP_READY]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // SEMA: scf.for {{.*}} iter_args({{[^,]+}}, [[CAP_IN:%.*]] = [[CAP_ENTRY]])
    // ASP: [[CAP_SLOT:%.*]] = arith.select
    // ASP: [[CAP_BIT:%.*]] = arith.shli [[CAP_ONE:%.*]], [[CAP_SLOT]]
    // ASP: [[CAP_ENTRY_BITS:%.*]] = arith.xori [[CAP_INITIAL_BITS]], [[CAP_BIT]]
    // ASP: [[CAP_SHIFT:%.*]] = arith.shrui [[CAP_ENTRY_BITS]], [[CAP_SLOT]]
    // ASP: [[CAP_PHASE:%.*]] = arith.andi [[CAP_SHIFT]], [[CAP_ONE]]
    // ASP: [[CAP_AENTRY:%.*]] = nvws.semaphore.acquire [[CAP_AREADY]][[[CAP_SLOT]], [[CAP_PHASE]]]
    // ASP: scf.for {{.*}} iter_args({{[^,]+}}, [[CAP_AIN:%.*]] = [[CAP_AENTRY]], [[CAP_OUT_SLOT:%.*]] = [[CAP_SLOT]], [[CAP_OUT_BITS:%.*]] = [[CAP_ENTRY_BITS]],
    // LOWER: scf.for
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_6 = %prog_id) -> (i32)  : i32 {
      %off_m = arith.muli %tile_idx_6, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      // LOWER: ttng.async_tma_copy_global_to_local [[CAP_Q]]{{.*}} {ttg.partition = array<i32: 2>}
      nvws.descriptor_load %desc_q[%off_m, %c0_i32] 32768 %q {multicast = false, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %qk_7, %qk_8 = ttng.tmem_alloc {ttg.partition = array<i32: 0, 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_9 = arith.constant {ttg.partition = array<i32: 0>} true
      // SEMA: [[CAP_INIT_BUF:%.*]] = nvws.semaphore.buffer [[CAP_READY]], [[CAP_IN]]
      // SEMA-NEXT: ttng.tmem_store {{%.*}}, [[CAP_INIT_BUF]], {{%.*}} {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: nvws.semaphore.release [[CAP_READY]], [[CAP_IN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[CAP_AINIT_BUF:%.*]] = nvws.semaphore.buffer [[CAP_AREADY]][[[CAP_OUT_SLOT]]], [[CAP_AIN]]
      // ASP-NEXT: ttng.tmem_store {{%.*}}, [[CAP_AINIT_BUF]],
      // ASP-NEXT: nvws.semaphore.release [[CAP_AREADY]][[[CAP_OUT_SLOT]]], [[CAP_AIN]] [#nvws.async_op<none>]
      // LOWER: [[CAP_LINIT_BUF:%.*]] = ttg.memdesc_index [[CAP_LACC]][[[CAP_LSLOT:%.*]]]
      // LOWER-NEXT: ttng.tmem_store {{%.*}}, [[CAP_LINIT_BUF]],
      // LOWER-NEXT: [[CAP_RELAY:%.*]] = ttg.memdesc_index [[CAP_BARRIER]][[[CAP_LSLOT]]]
      // LOWER-NEXT: ttng.arrive_barrier [[CAP_RELAY]], 1 {ttg.partition = array<i32: 0>}
      ttng.tmem_store %cst_1, %acc, %acc_9 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // SEMA: scf.for {{.*}} iter_args({{[^,]+}}, {{[^,]+}}, [[CAP_UPDATE:%.*]] = [[CAP_FALSE]])
      // ASP: [[CAP_INNER:%[-A-Za-z0-9_.$#]+]]:{{[0-9]+}} = scf.for {{.*}} iter_args({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, [[CAP_IN_SLOT:%.*]] = [[CAP_OUT_SLOT]], [[CAP_IN_BITS:%.*]] = [[CAP_OUT_BITS]],
      // LOWER: [[CAP_LINNER:%[-A-Za-z0-9_.$#]+]]:{{[0-9]+}} = scf.for
      %acc_10:5 = scf.for %start_n = %c0_i32 to %N step %c128_i32 iter_args(%m_i = %cst_0, %l_i = %cst, %acc_14 = %false, %qk_15 = %qk_8, %acc_16 = %acc_4) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, i1, !ttg.async.token, !ttg.async.token)  : i32 {
        // LOWER: ttng.async_tma_copy_global_to_local [[CAP_K]]{{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        nvws.descriptor_load %desc_k[%start_n, %c0_i32] 32768 %k {loop.cluster = 3 : i32, loop.stage = 0 : i32, multicast = false, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %k_17 = ttg.memdesc_trans %k {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
        // SEMA: ttng.tc_gen5_mma {{.*}}[], [[CAP_FALSE]], [[CAP_TRUE]] {loop.cluster = 2 : i32, loop.stage = 1 : i32,
        %qk_18 = ttng.tc_gen5_mma %q, %k_17, %qk_7[%qk_15], %false, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_19, %qk_20 = ttng.tmem_load %qk_7[%qk_18] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %m_ij_21 = "tt.reduce"(%qk_19) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_40: f32, %m_ij_41: f32):
          %m_ij_42 = arith.maxnumf %m_ij_40, %m_ij_41 {ttg.partition = array<i32: 0>} : f32
          tt.reduce.return %m_ij_42 {ttg.partition = array<i32: 0>} : f32
        }) {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_22 = arith.mulf %m_ij_21, %m_ij {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_23 = arith.maxnumf %m_i, %m_ij_22 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %qk_24 = arith.mulf %qk_19, %qk {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %qk_25 = tt.expand_dims %m_ij_23 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_26 = tt.broadcast %qk_25 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_27 = arith.subf %qk_24, %qk_26 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %p = math.exp2 %qk_27 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %alpha = arith.subf %m_i, %m_ij_23 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_28 = math.exp2 %alpha {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_40: f32, %l_ij_41: f32):
          %l_ij_42 = arith.addf %l_ij_40, %l_ij_41 {ttg.partition = array<i32: 0>} : f32
          tt.reduce.return %l_ij_42 {ttg.partition = array<i32: 0>} : f32
        }) {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc_29 = tt.expand_dims %alpha_28 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %acc_30 = tt.broadcast %acc_29 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        // Correction consumes the relayed permit and the same accumulator view.
        // SEMA: [[CAP_READ:%.*]] = nvws.semaphore.acquire [[CAP_READY]] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
        // SEMA-NEXT: [[CAP_READ_BUF:%.*]] = nvws.semaphore.buffer [[CAP_READY]], [[CAP_READ]]
        // SEMA-NEXT: [[CAP_VALUE:%.*]], {{%.*}} = ttng.tmem_load [[CAP_READ_BUF]][]
        // SEMA-NEXT: [[CAP_CORRECTED:%.*]] = arith.mulf [[CAP_VALUE]],
        // ASP: [[CAP_READ_BIT:%.*]] = arith.shli {{%.*}}, [[CAP_IN_SLOT]]
        // ASP-NEXT: [[CAP_READ_BITS:%.*]] = arith.xori [[CAP_IN_BITS]], [[CAP_READ_BIT]]
        // ASP-NEXT: [[CAP_READ_SHIFT:%.*]] = arith.shrui [[CAP_READ_BITS]], [[CAP_IN_SLOT]]
        // ASP-NEXT: [[CAP_READ_PHASE:%.*]] = arith.andi [[CAP_READ_SHIFT]],
        // ASP-NEXT: [[CAP_AREAD:%.*]] = nvws.semaphore.acquire [[CAP_AREADY]][[[CAP_IN_SLOT]], [[CAP_READ_PHASE]]]
        // LOWER: [[CAP_INNER_WAIT:%.*]] = ttg.memdesc_index [[CAP_BARRIER]][[[CAP_CORRECT_SLOT:%.*]]]
        // LOWER-NEXT: ttng.wait_barrier [[CAP_INNER_WAIT]], {{%.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
        // LOWER-NEXT: [[CAP_CORRECT_BUF:%.*]] = ttg.memdesc_index [[CAP_LACC]][[[CAP_CORRECT_SLOT]]]
        // LOWER-NEXT: {{%.*}}, {{%.*}} = ttng.tmem_load [[CAP_CORRECT_BUF]][]
        %acc_31, %acc_32 = ttng.tmem_load %acc[%acc_16] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %acc_33 = arith.mulf %acc_31, %acc_30 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        // LOWER: ttng.async_tma_copy_global_to_local [[CAP_V]]{{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>}
        nvws.descriptor_load %desc_v[%start_n, %c0_i32] 32768 %v {loop.cluster = 1 : i32, loop.stage = 2 : i32, multicast = false, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %p_34 = arith.truncf %p {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %acc_35 = arith.constant {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} true
        ttng.tmem_store %p_34, %acc_5, %acc_35 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        // SEMA: ttng.tmem_store [[CAP_CORRECTED]], [[CAP_READ_BUF]][], [[CAP_TRUE]]
        // SEMA-NEXT: nvws.semaphore.release [[CAP_DONE]], [[CAP_READ]] [#nvws.async_op<none>]
        // SEMA-NEXT: [[CAP_MMA:%.*]] = nvws.semaphore.acquire [[CAP_DONE]] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>}
        // SEMA-NEXT: [[CAP_MMA_BUF:%.*]] = nvws.semaphore.buffer [[CAP_DONE]], [[CAP_MMA]]
        // ASP: nvws.semaphore.release [[CAP_ADONE]][[[CAP_IN_SLOT]]], [[CAP_AREAD]] [#nvws.async_op<none>]
        // ASP: [[CAP_AMMA:%.*]] = nvws.semaphore.acquire [[CAP_ADONE]][[[CAP_MMA_SLOT:%.*]], {{%.*}}] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>}
        %acc_36 = ttng.tmem_store %acc_33, %acc[%acc_32], %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // SEMA: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[CAP_MMA_BUF]][], [[CAP_UPDATE]], [[CAP_TRUE]] {loop.cluster = 0 : i32, loop.stage = 3 : i32,
        // SEMA: nvws.semaphore.release [[CAP_READY]], [[CAP_MMA]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>}
        // ASP: nvws.semaphore.release [[CAP_AREADY]][[[CAP_MMA_SLOT]]], [[CAP_AMMA]] [#nvws.async_op<tc5mma>]
        // LOWER: ttng.tc_gen5_mma {{.*}} {is_async, loop.cluster = 0 : i32, loop.stage = 3 : i32,
        // LOWER: [[CAP_COMMIT:%.*]] = ttg.memdesc_index [[CAP_BARRIER]][
        // LOWER-NEXT: ttng.tc_gen5_commit [[CAP_COMMIT]] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>}
        %acc_37 = ttng.tc_gen5_mma %acc_5, %v, %acc[%acc_36], %acc_14, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_i_38 = arith.mulf %l_i, %alpha_28 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i_39 = arith.addf %l_i_38, %l_ij {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        // ASP: scf.yield {{.*}}, [[CAP_MMA_SLOT]], [[CAP_READ_BITS]],
        scf.yield {ttg.partition = array<i32: 0, 1, 2>} %m_ij_23, %l_i_39, %true, %qk_20, %acc_37 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, i1, !ttg.async.token, !ttg.async.token
      } {tt.scheduled_max_stage = 3 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 1>, array<i32: 0>]}
      // SEMA: } {tt.scheduled_max_stage = 3 : i32,
      // SEMA: [[CAP_FINAL:%.*]] = nvws.semaphore.acquire [[CAP_READY]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: [[CAP_FINAL_BUF:%.*]] = nvws.semaphore.buffer [[CAP_READY]], [[CAP_FINAL]]
      // SEMA-NEXT: [[CAP_FINAL_VALUE:%.*]], {{%.*}} = ttng.tmem_load [[CAP_FINAL_BUF]][]
      // ASP: } {tt.scheduled_max_stage = 3 : i32,
      // ASP: [[CAP_FINAL_BIT:%.*]] = arith.shli {{%.*}}, [[CAP_INNER]]#3
      // ASP-NEXT: [[CAP_FINAL_BITS:%.*]] = arith.xori [[CAP_INNER]]#4, [[CAP_FINAL_BIT]]
      // ASP-NEXT: [[CAP_FINAL_SHIFT:%.*]] = arith.shrui [[CAP_FINAL_BITS]], [[CAP_INNER]]#3
      // ASP-NEXT: [[CAP_FINAL_PHASE:%.*]] = arith.andi [[CAP_FINAL_SHIFT]],
      // ASP-NEXT: [[CAP_AFINAL:%.*]] = nvws.semaphore.acquire [[CAP_AREADY]][[[CAP_INNER]]#3, [[CAP_FINAL_PHASE]]]
      // LOWER: } {tt.scheduled_max_stage = 3 : i32,
      // LOWER: [[CAP_FINAL_WAIT:%.*]] = ttg.memdesc_index [[CAP_BARRIER]][[[CAP_LINNER]]#3]
      // LOWER-NEXT: ttng.wait_barrier [[CAP_FINAL_WAIT]], {{%.*}} {ttg.partition = array<i32: 0>}
      // LOWER-NEXT: [[CAP_LFINAL_BUF:%.*]] = ttg.memdesc_index [[CAP_LACC]][[[CAP_LINNER]]#3]
      // LOWER-NEXT: {{%.*}}, {{%.*}} = ttng.tmem_load [[CAP_LFINAL_BUF]][]
      %acc_11, %acc_12 = ttng.tmem_load %acc[%acc_10#4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      // SEMA-NEXT: [[CAP_TRUNC:%.*]] = arith.truncf [[CAP_FINAL_VALUE]]
      // SEMA-NEXT: [[CAP_LAYOUT:%.*]] = ttg.convert_layout [[CAP_TRUNC]]
      %4 = arith.truncf %acc_11 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      %5 = ttg.convert_layout %4 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      // SEMA-NEXT: tt.descriptor_store {{%.*}}[{{.*}}], [[CAP_LAYOUT]] {ttg.partition = array<i32: 0>}
      // LOWER: tt.descriptor_store [[CAP_O]][{{.*}}]{{.*}} {ttg.partition = array<i32: 0>}
      tt.descriptor_store %desc_acc[%off_m, %c0_i32], %5 {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x128xf16, #shared>, tensor<128x128xf16, #blocked1>
      %6 = tt.addptr %l_i_ptr, %off_m {ttg.partition = array<i32: 0>} : !tt.ptr<f16>, i32
      %7 = tt.splat %6 {ttg.partition = array<i32: 0>} : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
      %8 = tt.addptr %7, %3 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
      %9 = arith.truncf %acc_10#1 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> to tensor<128xf16, #ttg.slice<{dim = 1, parent = #linear}>>
      %10 = ttg.convert_layout %9 {ttg.partition = array<i32: 0>} : tensor<128xf16, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf16, #blocked>
      tt.store %8, %10 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f16>, #blocked>
      %11 = tt.addptr %m_i_ptr, %off_m {ttg.partition = array<i32: 0>} : !tt.ptr<f16>, i32
      %12 = tt.splat %11 {ttg.partition = array<i32: 0>} : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
      %13 = tt.addptr %12, %3 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
      %14 = arith.truncf %acc_10#0 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> to tensor<128xf16, #ttg.slice<{dim = 1, parent = #linear}>>
      %15 = ttg.convert_layout %14 {ttg.partition = array<i32: 0>} : tensor<128xf16, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf16, #blocked>
      tt.store %13, %15 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f16>, #blocked>
      %tile_idx_13 = arith.addi %tile_idx_6, %num_sm {ttg.partition = array<i32: 0, 2>} : i32
      // The epilogue token and evolved phase state seed the next tile.
      // SEMA: scf.yield {{.*}}, [[CAP_FINAL]] : i32, !ttg.async.token
      // ASP: scf.yield {{.*}}, [[CAP_AFINAL]], [[CAP_INNER]]#3, [[CAP_FINAL_BITS]],
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %tile_idx_13 : i32
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 2>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// SEMA-LABEL: @dynamic_initialized_carrier
// SEMA: [[TRUE:%[-a-zA-Z0-9_]+]] = arith.constant true
// SEMA: [[FALSE:%[-a-zA-Z0-9_]+]] = arith.constant false
// SEMA: [[BACKING:%[-a-zA-Z0-9_]+]] = ttng.tmem_alloc
// SEMA: [[READY:%[-a-zA-Z0-9_]+]] = nvws.semaphore.create [[BACKING]] released = 1 {pending_count = 1 : i32}
// SEMA: [[TO_MMA:%[-a-zA-Z0-9_]+]] = nvws.semaphore.create [[BACKING]] {pending_count = 1 : i32}
// SEMA: [[ENTRY:%[-a-zA-Z0-9_]+]] = nvws.semaphore.acquire [[READY]]
// SEMA: [[INIT:%[-a-zA-Z0-9_]+]] = nvws.semaphore.buffer [[READY]], [[ENTRY]]
// SEMA: ttng.tmem_store {{%[^,]+}}, [[INIT]][], [[TRUE]]
// SEMA: [[OUTER:%[-a-zA-Z0-9_]+]]:2 = scf.for {{.*}} iter_args([[OUTER_USE:%[-a-zA-Z0-9_]+]] = [[FALSE]], [[OUTER_TOKEN:%[-a-zA-Z0-9_]+]] = [[ENTRY]])
// SEMA-NEXT: nvws.semaphore.release [[READY]], [[OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
// SEMA-NEXT: [[INNER:%[-a-zA-Z0-9_]+]] = scf.for {{.*}} iter_args([[USE_ACC:%[-a-zA-Z0-9_]+]] = [[OUTER_USE]])
// SEMA-NEXT: [[READ:%[-a-zA-Z0-9_]+]] = nvws.semaphore.acquire [[READY]]
// SEMA-NEXT: [[RBUF:%[-a-zA-Z0-9_]+]] = nvws.semaphore.buffer [[READY]], [[READ]]
// SEMA-NEXT: [[VALUE:%[-a-zA-Z0-9_]+]], {{%[^ ]+}} = ttng.tmem_load [[RBUF]][]
// SEMA-NEXT: [[CORRECTED:%[-a-zA-Z0-9_]+]] = arith.mulf [[VALUE]],
// SEMA-NEXT: {{%[^ ]+}} = ttng.tmem_store [[CORRECTED]], [[RBUF]][], [[TRUE]]
// SEMA-NEXT: nvws.semaphore.release [[TO_MMA]], [[READ]] [#nvws.async_op<none>]
// SEMA-NEXT: [[MMA:%[-a-zA-Z0-9_]+]] = nvws.semaphore.acquire [[TO_MMA]]
// SEMA-NEXT: [[MBUF:%[-a-zA-Z0-9_]+]] = nvws.semaphore.buffer [[TO_MMA]], [[MMA]]
// SEMA-NEXT: {{%[^ ]+}} = ttng.tc_gen5_mma {{%[^,]+}}, {{%[^,]+}}, [[MBUF]][], [[USE_ACC]], [[TRUE]]
// SEMA-NEXT: nvws.semaphore.release [[READY]], [[MMA]] [#nvws.async_op<tc5mma>]
// SEMA-NEXT: scf.yield {{.*}} [[TRUE]] : i1
// SEMA-NEXT: }
// SEMA-NEXT: [[DONE:%[-a-zA-Z0-9_]+]] = nvws.semaphore.acquire [[READY]]
// SEMA-NEXT: [[DONEBUF:%[-a-zA-Z0-9_]+]] = nvws.semaphore.buffer [[READY]], [[DONE]]
// SEMA-NEXT: {{.*}}ttng.tmem_load [[DONEBUF]][]
// SEMA: scf.yield {{.*}} [[INNER]], [[DONE]] : i1, !ttg.async.token
// SEMA-NEXT: }
// SEMA-NEXT: [[FINAL:%[-a-zA-Z0-9_]+]] = nvws.semaphore.buffer [[READY]], [[OUTER]]#1
// SEMA-NEXT: {{.*}}ttng.tmem_load [[FINAL]][]
// SEMA: tt.return

// Both loop result tuples must forward their initial slot and phase states
// when empty. Check that the observer consumes those returned states, rather
// than restarting either semaphore's phase after a child loop.
// ASP-LABEL: @dynamic_initialized_carrier
// ASP: [[TRUE:%[-a-zA-Z0-9_]+]] = arith.constant true
// ASP: [[FALSE:%[-a-zA-Z0-9_]+]] = arith.constant false
// ASP: [[BACKING:%[-a-zA-Z0-9_]+]] = ttng.tmem_alloc
// ASP: [[READY:%[-a-zA-Z0-9_]+]] = nvws.semaphore.create [[BACKING]] released = 1
// ASP: [[TO_MMA:%[-a-zA-Z0-9_]+]] = nvws.semaphore.create [[BACKING]] {pending_count = 1 : i32}
// ASP: [[INITIAL_R:%[-a-zA-Z0-9_]+]] = arith.constant -2 : i32
// ASP: [[INITIAL_M:%[-a-zA-Z0-9_]+]] = arith.constant -1 : i32
// ASP: [[INITIAL_SLOT:%[-a-zA-Z0-9_]+]] = arith.select
// ASP: [[INITIAL_BIT:%[-a-zA-Z0-9_]+]] = arith.shli {{%[^,]+}}, [[INITIAL_SLOT]]
// ASP: [[INITIAL_BITS:%[-a-zA-Z0-9_]+]] = arith.xori [[INITIAL_R]], [[INITIAL_BIT]]
// ASP: [[INITIAL_SHIFT:%[-a-zA-Z0-9_]+]] = arith.shrui [[INITIAL_BITS]], [[INITIAL_SLOT]]
// ASP: [[INITIAL_PHASE:%[-a-zA-Z0-9_]+]] = arith.andi [[INITIAL_SHIFT]],
// ASP: [[ENTRY:%[-a-zA-Z0-9_]+]] = nvws.semaphore.acquire [[READY]][[[INITIAL_SLOT]], [[INITIAL_PHASE]]]
// ASP: [[INIT:%[-a-zA-Z0-9_]+]] = nvws.semaphore.buffer [[READY]][[[INITIAL_SLOT]]], [[ENTRY]]
// ASP: ttng.tmem_store {{%[^,]+}}, [[INIT]][], [[TRUE]]
// ASP: [[OUTER:%[-a-zA-Z0-9_]+]]:5 = scf.for {{.*}} iter_args([[OUTER_USE:%[-a-zA-Z0-9_]+]] = [[FALSE]], [[OUTER_TOKEN:%[-a-zA-Z0-9_]+]] = [[ENTRY]], [[OUTER_SLOT:%[-a-zA-Z0-9_]+]] = [[INITIAL_SLOT]], [[OUTER_R:%[-a-zA-Z0-9_]+]] = [[INITIAL_BITS]], [[OUTER_M:%[-a-zA-Z0-9_]+]] = [[INITIAL_M]])
// ASP-NEXT: nvws.semaphore.release [[READY]][[[OUTER_SLOT]]], [[OUTER_TOKEN]] [#nvws.async_op<none>]
// ASP-NEXT: [[INNER:%[-a-zA-Z0-9_]+]]:4 = scf.for {{.*}} iter_args([[USE_ACC:%[-a-zA-Z0-9_]+]] = [[OUTER_USE]], [[SLOT:%[-a-zA-Z0-9_]+]] = [[OUTER_SLOT]], [[R_BITS:%[-a-zA-Z0-9_]+]] = [[OUTER_R]], [[M_BITS:%[-a-zA-Z0-9_]+]] = [[OUTER_M]])
// ASP: [[R_BIT:%[-a-zA-Z0-9_]+]] = arith.shli {{%[^,]+}}, [[SLOT]]
// ASP: [[NEXT_R:%[-a-zA-Z0-9_]+]] = arith.xori [[R_BITS]], [[R_BIT]]
// ASP: [[R_SHIFT:%[-a-zA-Z0-9_]+]] = arith.shrui [[NEXT_R]], [[SLOT]]
// ASP: [[R_PHASE:%[-a-zA-Z0-9_]+]] = arith.andi [[R_SHIFT]],
// ASP: [[READ:%[-a-zA-Z0-9_]+]] = nvws.semaphore.acquire [[READY]][[[SLOT]], [[R_PHASE]]]
// ASP: nvws.semaphore.release [[TO_MMA]][[[SLOT]]], [[READ]] [#nvws.async_op<none>]
// ASP: [[NEXT_SLOT:%[-a-zA-Z0-9_]+]] = arith.select
// ASP: [[M_BIT:%[-a-zA-Z0-9_]+]] = arith.shli {{%[^,]+}}, [[NEXT_SLOT]]
// ASP: [[NEXT_M:%[-a-zA-Z0-9_]+]] = arith.xori [[M_BITS]], [[M_BIT]]
// ASP: [[M_SHIFT:%[-a-zA-Z0-9_]+]] = arith.shrui [[NEXT_M]], [[NEXT_SLOT]]
// ASP: [[M_PHASE:%[-a-zA-Z0-9_]+]] = arith.andi [[M_SHIFT]],
// ASP: [[MMA:%[-a-zA-Z0-9_]+]] = nvws.semaphore.acquire [[TO_MMA]][[[NEXT_SLOT]], [[M_PHASE]]]
// ASP: [[MBUF:%[-a-zA-Z0-9_]+]] = nvws.semaphore.buffer [[TO_MMA]][[[NEXT_SLOT]]], [[MMA]]
// ASP: ttng.tc_gen5_mma {{%[^,]+}}, {{%[^,]+}}, [[MBUF]][], [[USE_ACC]], [[TRUE]]
// ASP-NEXT: nvws.semaphore.release [[READY]][[[NEXT_SLOT]]], [[MMA]] [#nvws.async_op<tc5mma>]
// ASP-NEXT: scf.yield {{.*}} [[TRUE]], [[NEXT_SLOT]], [[NEXT_R]], [[NEXT_M]] : i1, i32, i32, i32
// ASP-NEXT: }
// ASP: [[DONE_BIT:%[-a-zA-Z0-9_]+]] = arith.shli {{%[^,]+}}, [[INNER]]#1
// ASP: [[DONE_BITS:%[-a-zA-Z0-9_]+]] = arith.xori [[INNER]]#2, [[DONE_BIT]]
// ASP: [[DONE_SHIFT:%[-a-zA-Z0-9_]+]] = arith.shrui [[DONE_BITS]], [[INNER]]#1
// ASP: [[DONE_PHASE:%[-a-zA-Z0-9_]+]] = arith.andi [[DONE_SHIFT]],
// ASP: [[DONE:%[-a-zA-Z0-9_]+]] = nvws.semaphore.acquire [[READY]][[[INNER]]#1, [[DONE_PHASE]]]
// ASP: [[DONEBUF:%[-a-zA-Z0-9_]+]] = nvws.semaphore.buffer [[READY]][[[INNER]]#1], [[DONE]]
// ASP: ttng.tmem_load [[DONEBUF]][]
// ASP: scf.yield {{.*}} [[INNER]]#0, [[DONE]], [[INNER]]#1, [[DONE_BITS]], [[INNER]]#3 : i1, !ttg.async.token, i32, i32, i32
// ASP-NEXT: }
// ASP-NEXT: [[FINAL:%[-a-zA-Z0-9_]+]] = nvws.semaphore.buffer [[READY]][[[OUTER]]#2], [[OUTER]]#1
// ASP-NEXT: {{.*}}ttng.tmem_load [[FINAL]][]
// ASP: tt.return

// LOWER-LABEL: @dynamic_initialized_carrier
// LOWER: [[BACKING:%[-a-zA-Z0-9_]+]] = ttng.tmem_alloc
// LOWER: [[READY:%[-a-zA-Z0-9_]+]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
// LOWER: [[TO_MMA:%[-a-zA-Z0-9_]+]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
// LOWER: [[OUTER:%[-a-zA-Z0-9_]+]]:5 = scf.for
// LOWER-NEXT: [[RELAY:%[-a-zA-Z0-9_]+]] = ttg.memdesc_index [[READY]][[[OUTER_SLOT:%[-a-zA-Z0-9_]+]]]
// LOWER-NEXT: ttng.arrive_barrier [[RELAY]], 1
// LOWER-NEXT: [[INNER:%[-a-zA-Z0-9_]+]]:4 = scf.for {{.*}} iter_args({{%[^,]+}}, [[SLOT:%[-a-zA-Z0-9_]+]] = [[OUTER_SLOT]],
// LOWER: [[READ_WAIT:%[-a-zA-Z0-9_]+]] = ttg.memdesc_index [[READY]][[[SLOT]]]
// LOWER-NEXT: ttng.wait_barrier [[READ_WAIT]],
// LOWER: ttng.tmem_load
// LOWER: [[CORRECTED:%[-a-zA-Z0-9_]+]] = arith.mulf
// LOWER-NEXT: {{%[^ ]+}} = ttng.tmem_store [[CORRECTED]],
// LOWER-NEXT: [[HANDOFF:%[-a-zA-Z0-9_]+]] = ttg.memdesc_index [[TO_MMA]][[[SLOT]]]
// LOWER-NEXT: ttng.arrive_barrier [[HANDOFF]], 1
// LOWER: [[MMA_WAIT:%[-a-zA-Z0-9_]+]] = ttg.memdesc_index [[TO_MMA]][[[MMA_SLOT:%[-a-zA-Z0-9_]+]]]
// LOWER-NEXT: ttng.wait_barrier [[MMA_WAIT]],
// LOWER: ttng.tc_gen5_mma
// LOWER-NEXT: [[COMMIT:%[-a-zA-Z0-9_]+]] = ttg.memdesc_index [[READY]][[[MMA_SLOT]]]
// LOWER-NEXT: ttng.tc_gen5_commit [[COMMIT]]
// LOWER: [[DONE_WAIT:%[-a-zA-Z0-9_]+]] = ttg.memdesc_index [[READY]][[[INNER]]#1]
// LOWER-NEXT: ttng.wait_barrier [[DONE_WAIT]],
// LOWER: scf.yield {{.*}} [[INNER]]#0, {{%[^,]+}}, [[INNER]]#1, {{%[^,]+}}, [[INNER]]#3 : i1, !ttg.async.token, i32, i32, i32
// LOWER: [[FINAL:%[-a-zA-Z0-9_]+]] = ttg.memdesc_index [[BACKING]][[[OUTER]]#2]
// LOWER: ttng.tmem_load [[FINAL]][]
// LOWER: tt.return

// Dynamic loop bounds retain both zero-trip paths in real structured IR.
// Scaling a zero initializer still permits a fresh first MMA; subsequent MMAs
// accumulate, including across outer iterations. The observer is outside both
// loops, so an empty outer loop must expose the initialized value and token.
#reg = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory
#tm = #ttng.tensor_memory
!acc = !ttg.memdesc<128x128xf32, #tmem, #tm, mutable>
!tile = tensor<128x128xf32, #reg>
!lhs = !ttg.memdesc<128x64xf16, #shared, #smem>
!rhs = !ttg.memdesc<64x128xf16, #shared, #smem>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @dynamic_initialized_carrier(%outer_ub: i32, %inner_ub: i32,
                                        %lhs: !lhs, %rhs: !rhs) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %scale = arith.constant dense<2.0> : !tile
    %acc, %allocated = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    %init = ttng.tmem_store %zero, %acc[%allocated], %true {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : !tile -> !acc
    %outer:2 = scf.for %i = %c0 to %outer_ub step %c1 iter_args(%outer_token = %init, %outer_use_acc = %false) -> (!ttg.async.token, i1) : i32 {
      %inner:2 = scf.for %j = %c0 to %inner_ub step %c1 iter_args(%inner_token = %outer_token, %use_acc = %outer_use_acc) -> (!ttg.async.token, i1) : i32 {
        %value, %read = ttng.tmem_load %acc[%inner_token] {ttg.partition = array<i32: 0>} : !acc -> !tile
        %corrected = arith.mulf %value, %scale {ttg.partition = array<i32: 0>} : !tile
        %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %use_acc, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      %tile, %done = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      "consume"(%tile) {ttg.partition = array<i32: 0>} : (!tile) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %done, %inner#1 : !ttg.async.token, i1
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result, %last = ttng.tmem_load %acc[%outer#0] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : !acc -> !tile
    "consume_final"(%result) {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : (!tile) -> ()
    tt.return
  }
}
