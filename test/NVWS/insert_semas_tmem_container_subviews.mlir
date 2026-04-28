// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// v4 container-pattern tests. All multi-member buffer.id groups in real
// Triton IR have one member that acts as the physical slot container
// (largest extent, covers [0, slot_size)); other members are sub-views
// inside the container. Union-find on overlap intervals unites all
// members through the container, so each buffer.id group collapses to
// a single resourceKey.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked256 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem128 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem256 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Container m0 = [0, 256). Disjoint sub-views m1 = [0, 128),
  // m2 = [128, 192), m3 = [192, 256). m0 unions with m1, m2, m3 via
  // overlap; m1, m2, m3 are pairwise disjoint. All collapse to one
  // resourceKey via m0.

  // CHECK-LABEL: @container_with_disjoint_subviews
  tt.func @container_with_disjoint_subviews(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst256 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked256>
    %cst128 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked64>
    %true = arith.constant true

    // Container alloc, then the three sub-view memdescs that union into it.
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc {buffer.id = 900 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[S3:%.*]] = ttng.tmem_subslice [[ALLOC]] {N = 192 : i32}
    // CHECK: [[M3:%.*]] = ttg.memdesc_reinterpret [[S3]] : !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[S2:%.*]] = ttng.tmem_subslice [[ALLOC]] {N = 128 : i32}
    // CHECK: [[M2:%.*]] = ttg.memdesc_reinterpret [[S2]] : !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[S1:%.*]] = ttng.tmem_subslice [[ALLOC]] {N = 0 : i32}
    // CHECK: [[M1:%.*]] = ttg.memdesc_reinterpret [[S1]] : !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>

    // One EMPTY semaphore (pending=3) and three FULL semaphores (pending=1),
    // all over the single collapsed resourceKey of four memdesc members.
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]], [[M3]] released = -1 {pending_count = 3 : i32} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[F0:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]], [[M3]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[F1:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]], [[M3]] {pending_count = 1 : i32}
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]], [[M3]] {pending_count = 1 : i32}

    // No carried token: EMPTY is initially released and acquired inside the loop body.
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {

      // Producer (p0): acquire EMPTY at point of use, buffer the container
      // slot, store the full extent, and release it to p1.
      // CHECK: [[A0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]]:4 = nvws.semaphore.buffer [[EMPTY]], [[A0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B0]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>
      %m0, %t0 = ttng.tmem_alloc %cst256 {buffer.id = 900 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x256xf32, #blocked256>) -> (!ttg.memdesc<128x256xf32, #tmem256, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %v0, %l0 = ttng.tmem_load %m0[%t0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x256xf32, #tmem256, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked256>
      %m1, %t1 = ttng.tmem_alloc %cst128 {buffer.id = 900 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m2, %t2 = ttng.tmem_alloc %cst64 {buffer.id = 900 : i32, buffer.offset = 128 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf32, #blocked64>) -> (!ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m3, %t3 = ttng.tmem_alloc %cst64 {buffer.id = 900 : i32, buffer.offset = 192 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf32, #blocked64>) -> (!ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // Whole-buffer reader p1 acquires F0 and reads all three pieces.
      // CHECK: nvws.semaphore.release [[F0]], [[A0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK-NOT: nvws.semaphore.release [[F1]], [[A0]]
      // CHECK-NOT: nvws.semaphore.release [[F2]], [[A0]]
      // CHECK: [[A1:%.*]] = nvws.semaphore.acquire [[F0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B1:%.*]]:4 = nvws.semaphore.buffer [[F0]], [[A1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B1]]#0[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256> -> tensor<128x256xf32, #blocked>

      // Reader p1 sends P1 and P2 to their writers, then writes P0 itself.
      // CHECK: nvws.semaphore.release [[F1]], [[A1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[F2]], [[A1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B1]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>

      // Sub-view writer p2 (m2 = [128,192)): acquire F1, buffer, store member #2.
      // CHECK: [[A2:%.*]] = nvws.semaphore.acquire [[F1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B2:%.*]]:4 = nvws.semaphore.buffer [[F1]], [[A2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B2]]#2, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>

      // Sub-view writer p3 (m3 = [192,256)): acquire F2, buffer, store member #3.
      // CHECK: [[A3:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B3:%.*]]:4 = nvws.semaphore.buffer [[F2]], [[A3]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B3]]#3, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      %v1, %l1 = ttng.tmem_load %m1[%t1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %v2, %l2 = ttng.tmem_load %m2[%t2] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked64>
      %v3, %l3 = ttng.tmem_load %m3[%t3] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked64>

      // Reader p1 reuses its retained F0 token, loads member #1, and recycles EMPTY.
      // CHECK: [[B4:%.*]]:4 = nvws.semaphore.buffer [[F0]], [[A1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B4]]#1[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[A1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // Reader p2 reuses its retained F1 token, loads member #2, and recycles EMPTY.
      // CHECK: [[B5:%.*]]:4 = nvws.semaphore.buffer [[F1]], [[A2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B5]]#2[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64> -> tensor<128x64xf32, #blocked1>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[A2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // Reader p3 reuses its retained F2 token, loads member #3, and recycles EMPTY.
      // CHECK: [[B6:%.*]]:4 = nvws.semaphore.buffer [[F2]], [[A3]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B6]]#3[] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64> -> tensor<128x64xf32, #blocked1>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[A3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use1"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      "use2"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x64xf32, #blocked64>) -> ()
      "use3"(%v3) {ttg.partition = array<i32: 3>} : (tensor<128x64xf32, #blocked64>) -> ()
      "use0"(%v0) {ttg.partition = array<i32: 1>} : (tensor<128x256xf32, #blocked256>) -> ()

      // No bottom re-acquire: the yield carries only the original i32.
      // CHECK-NOT: nvws.semaphore.acquire
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} %{{[-A-Za-z0-9_.$#]+}} : i32
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2, 3>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3>], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked256 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem128 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem256 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Container m0 = [0, 256). Two overlapping sub-views m1 = [0, 128),
  // m2 = [64, 128). m1 and m2 overlap each other AND both overlap m0.
  // All three collapse to one resourceKey.

  // CHECK-LABEL: @container_with_overlapping_subviews
  tt.func @container_with_overlapping_subviews(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst256 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked256>
    %cst128 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked64>
    %true = arith.constant true

    // Container alloc, then two overlapping sub-view memdescs that union into it.
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc {buffer.id = 901 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[S2:%.*]] = ttng.tmem_subslice [[ALLOC]] {N = 64 : i32}
    // CHECK: [[M2:%.*]] = ttg.memdesc_reinterpret [[S2]] : !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[S1:%.*]] = ttng.tmem_subslice [[ALLOC]] {N = 0 : i32}
    // CHECK: [[M1:%.*]] = ttg.memdesc_reinterpret [[S1]] : !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>

    // One EMPTY semaphore (pending=2) and three FULL semaphores (pending=1)
    // over the single collapsed resourceKey of three memdesc members.
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]] released = -1 {pending_count = 2 : i32} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[F0:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[F1:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]] {pending_count = 1 : i32}
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]] {pending_count = 1 : i32}

    // No carried token here: EMPTY is acquired inside the loop body.
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {

      // Producer (p0): acquire EMPTY, buffer the container slot, store full extent, release F0.
      // CHECK: [[A0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]]:3 = nvws.semaphore.buffer [[EMPTY]], [[A0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B0]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: nvws.semaphore.release [[F0]], [[A0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %m0, %t0 = ttng.tmem_alloc %cst256 {buffer.id = 901 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x256xf32, #blocked256>) -> (!ttg.memdesc<128x256xf32, #tmem256, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m1, %t1 = ttng.tmem_alloc %cst128 {buffer.id = 901 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m2, %t2 = ttng.tmem_alloc %cst64 {buffer.id = 901 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf32, #blocked64>) -> (!ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // Sub-view writer p1 (m1 = [0,128)): acquire F0, buffer, store member #1, release F1.
      // CHECK: [[A1:%.*]] = nvws.semaphore.acquire [[F0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B1:%.*]]:3 = nvws.semaphore.buffer [[F0]], [[A1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B1]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: nvws.semaphore.release [[F1]], [[A1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // Sub-view writer p2 (m2 = [64,128)): acquire F1, buffer, store member #2, release F2.
      // CHECK: [[A2:%.*]] = nvws.semaphore.acquire [[F1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B2:%.*]]:3 = nvws.semaphore.buffer [[F1]], [[A2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B2]]#2, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: nvws.semaphore.release [[F2]], [[A2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %v1, %l1 = ttng.tmem_load %m1[%t1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %v2, %l2 = ttng.tmem_load %m2[%t2] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked64>

      // Reader p1: acquire F2, buffer, load member #1, and recycle EMPTY.
      // CHECK: [[A3:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B3:%.*]]:3 = nvws.semaphore.buffer [[F2]], [[A3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B3]]#1[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[A3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // Reader p2 reuses its retained F1 token, loads member #2, and recycles EMPTY.
      // CHECK: [[B4:%.*]]:3 = nvws.semaphore.buffer [[F1]], [[A2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B4]]#2[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64> -> tensor<128x64xf32, #blocked1>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[A2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use1"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      "use2"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x64xf32, #blocked64>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %{{[-A-Za-z0-9_.$#]+}} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
