// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#scalar = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#local_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Each TMEM group sees its own partition-1 access at cluster 1 or 3; the
  // independent local buffer adds a partition-1 access at cluster 4. The
  // inner loop carries no tokens: each group's regain is emitted after the
  // inner loop in partition 1, acquiring the group's last-round semaphore
  // and releasing its entry semaphore for the next outer tile.
  // CHECK-LABEL: @cross_group_tail_acquire_schedule
  tt.func @cross_group_tail_acquire_schedule(
      %lhs: !ttg.memdesc<128x64xf32, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf32, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %frontier = ttg.local_alloc {buffer.id = 402 : i32} : () -> !ttg.memdesc<1xi32, #local_shared, #smem, mutable>

    // Both in-loop tmem_allocs are hoisted; each group gets an initially
    // released entry semaphore plus two in-loop semaphores.
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 400 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V5]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V7:%.*]] = nvws.semaphore.create [[V5]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]] = nvws.semaphore.create [[V5]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32) : i32
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0) -> (i32) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V2]], [[V9]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V10]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[V4]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %acc_a, %tok_a = ttng.tmem_alloc %cst {buffer.id = 400 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V6]], [[V11]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V12]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[V8]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %acc_b, %tok_b = ttng.tmem_alloc %cst {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32
      %inner:2 = scf.for %iv1 = %lb to %ub step %step iter_args(%a_token = %tok_a, %b_token = %tok_b) -> (!ttg.async.token, !ttg.async.token) : i32 {
        // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V4]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V14:%.*]] = nvws.semaphore.buffer [[V4]], [[V13]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V14]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: nvws.semaphore.release [[V3]], [[V13]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %mma_a = ttng.tc_gen5_mma %lhs, %rhs, %acc_a[%a_token], %true, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V16:%.*]] = nvws.semaphore.buffer [[V3]], [[V15]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V16]][] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V4]], [[V15]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %a_value, %a_read = ttng.tmem_load %acc_a[%mma_a] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "consume_a"(%a_value) {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

        // CHECK: [[V17:%.*]] = nvws.semaphore.acquire [[V8]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V18:%.*]] = nvws.semaphore.buffer [[V8]], [[V17]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V18]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: nvws.semaphore.release [[V7]], [[V17]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %mma_b = ttng.tc_gen5_mma %lhs, %rhs, %acc_b[%b_token], %true, %true {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: [[V19:%.*]] = nvws.semaphore.acquire [[V7]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V20:%.*]] = nvws.semaphore.buffer [[V7]], [[V19]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V20]][] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V8]], [[V19]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %b_value, %b_read = ttng.tmem_load %acc_b[%mma_b] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "consume_b"(%b_value) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

        %frontier_value = "frontier_value"() {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : () -> tensor<1xi32, #scalar>
        // CHECK: ttg.local_store {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
        ttg.local_store %frontier_value, %frontier {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<1xi32, #scalar> -> !ttg.memdesc<1xi32, #local_shared, #smem, mutable>

        scf.yield {ttg.partition = array<i32: 0, 1>} %a_read, %b_read : !ttg.async.token, !ttg.async.token
      } {tt.scheduled_max_stage = 0 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>]}
      // The token results and yields of the inner loop are dropped entirely.
      // CHECK: } {tt.scheduled_max_stage = 0 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}

      %next = arith.addi %tile, %c0 {ttg.partition = array<i32: 0>} : i32
      // Loop-close regains: partition 1 acquires each group's last-round
      // semaphore and releases its entry semaphore for the next outer tile.
      // CHECK: [[V21:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[V2]], [[V21]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V22:%.*]] = nvws.semaphore.acquire [[V8]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[V6]], [[V22]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}
