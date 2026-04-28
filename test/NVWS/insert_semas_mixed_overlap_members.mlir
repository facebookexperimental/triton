// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// Dedicated mirror of the meta-FA stats group (GROUP buffer.id=4 in
// insert_semas_meta_fa_fwd: m0[64,65) m1[66,67) m2[65,66) m3[0,128)
// m4[0,64)): many members on ONE backing buffer, with a MIX of
// overlapping and non-overlapping extents. A spanning member (the FA
// accumulator) covers the whole buffer, so the planner keeps every
// member on one backing and the group synchronizes as a single unit.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#half_blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#col_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // memory:   0        64  65  66  67      128
  // m3 (acc)  [==============================)    <- spans everything
  // m4 (p)    [========)
  // m0 (alpha)         [--)
  // m2 (l)                 [--)
  // m1 (m)                     [--)
  //
  // conflict graph (who overlaps whom):
  //
  //       m4 --- m3 --- m0       m3 is the HUB: it overlaps every
  //               | \            sliver, so every sliver must sync
  //              m2  m1          with m3's writes/reads.
  //                              No sliver<->sliver edge exists!
  //
  // m3 spans and bridges the whole group into one backing. One carrier
  // chain threads it: every sliver's W/R weaves acquire/release against
  // the shared semaphores, but slivers never sync with each other
  // directly - overlap is priced per shared piece.
  //
  // Owners form a ring: slivers W{1}->R{2}, acc W{2}->R{0}, p W{0}->R{1}.
  // CHECK-LABEL: @tmem_mixed_overlap_spanning_member
  tt.func @tmem_mixed_overlap_spanning_member(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #half_blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #col_blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = ttng.tmem_subslice [[V1]] {N = 0 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V3:%.*]] = ttg.memdesc_reinterpret [[V2]] : !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[V4:%.*]] = ttng.tmem_subslice [[V1]] {N = 65 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V5:%.*]] = ttg.memdesc_reinterpret [[V4]] : !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V6:%.*]] = ttng.tmem_subslice [[V1]] {N = 66 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V7:%.*]] = ttg.memdesc_reinterpret [[V6]] : !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = ttng.tmem_subslice [[V1]] {N = 64 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V9:%.*]] = ttg.memdesc_reinterpret [[V8]] : !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V10:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] released = -1 {pending_count = 2 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V11:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V13:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V15:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V16:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V17:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V19:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V18:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // m0: alpha sliver at column 64, produced by {1}, consumed by {2}.
      %alpha = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V20:%.*]] = nvws.semaphore.acquire [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V21:%.*]]:5 = nvws.semaphore.buffer [[V10]], [[V20]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V21]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      ttng.tmem_store %cst1, %alpha, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V11]], [[V20]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V22:%.*]] = nvws.semaphore.acquire [[V11]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V23:%.*]]:5 = nvws.semaphore.buffer [[V11]], [[V22]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V23]]#0[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      %av, %at = ttng.tmem_load %alpha[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_alpha"(%av) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // m1: m sliver at column 66, produced by {1}, consumed by {2}.
      %m = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 66 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V25:%.*]]:5 = nvws.semaphore.buffer [[V10]], [[V20]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V25]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      ttng.tmem_store %cst1, %m, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V13]], [[V20]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V26:%.*]] = nvws.semaphore.acquire [[V13]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V27:%.*]]:5 = nvws.semaphore.buffer [[V13]], [[V26]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V27]]#1[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      %mv, %mt = ttng.tmem_load %m[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_m"(%mv) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // m2: l sliver at column 65, produced by {1}, consumed by {2}.
      %l = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 65 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V29:%.*]]:5 = nvws.semaphore.buffer [[V10]], [[V20]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V29]]#2, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      ttng.tmem_store %cst1, %l, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V15]], [[V20]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V30:%.*]] = nvws.semaphore.acquire [[V15]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V31:%.*]]:5 = nvws.semaphore.buffer [[V15]], [[V30]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V31]]#2[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      %lv, %lt = ttng.tmem_load %l[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_l"(%lv) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // m3: the spanning accumulator [0,128), produced by {2}, consumed
      // by {0}. It overlaps ALL other members.
      %acc, %tacc = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 2>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: [[V32:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V31]]#3[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %acc0 = ttng.tmem_store %cst, %acc[%tacc], %true {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V16]], [[V30]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[V10]], [[V30]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V33:%.*]] = nvws.semaphore.acquire [[V16]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V34:%.*]]:5 = nvws.semaphore.buffer [[V16]], [[V33]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V34]]#3[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %accv, %acct = ttng.tmem_load %acc[%acc0] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V10]], [[V33]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_acc"(%accv) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      // m4: p at [0,64) - disjoint from every sliver, overlaps only the
      // accumulator. Produced by {0}, consumed by {1}.
      %p = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V34]]#4, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      ttng.tmem_store %cst64, %p, %true {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #half_blocked> -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V17]], [[V33]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V35:%.*]] = nvws.semaphore.acquire [[V17]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V36:%.*]]:5 = nvws.semaphore.buffer [[V17]], [[V35]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V36]]#4[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64> -> tensor<128x64xf32, #blocked1>
      %pv, %pt = ttng.tmem_load %p[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #half_blocked>
      "use_p"(%pv) {ttg.partition = array<i32: 1>} : (tensor<128x64xf32, #half_blocked>) -> ()

      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
