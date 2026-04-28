// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#scalar = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#local_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Each TMEM group sees its own partition-1 access at cluster 1 or 3. The
  // independent local buffer establishes the same lane's loop-exit frontier
  // at cluster 4. Both generated TMEM regains must use that global frontier.
  // CHECK-LABEL: @cross_group_tail_acquire_schedule
  tt.func @cross_group_tail_acquire_schedule(
      %lhs: !ttg.memdesc<128x64xf32, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf32, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %frontier = ttg.local_alloc {buffer.id = 402 : i32} : () -> !ttg.memdesc<1xi32, #local_shared, #smem, mutable>

    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0) -> (i32) : i32 {
      %acc_a, %tok_a = ttng.tmem_alloc %cst {buffer.id = 400 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_b, %tok_b = ttng.tmem_alloc %cst {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      %inner:2 = scf.for %iv1 = %lb to %ub step %step iter_args(%a_token = %tok_a, %b_token = %tok_b) -> (!ttg.async.token, !ttg.async.token) : i32 {
        %mma_a = ttng.tc_gen5_mma %lhs, %rhs, %acc_a[%a_token], %true, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %a_value, %a_read = ttng.tmem_load %acc_a[%mma_a] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "consume_a"(%a_value) {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

        %mma_b = ttng.tc_gen5_mma %lhs, %rhs, %acc_b[%b_token], %true, %true {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %b_value, %b_read = ttng.tmem_load %acc_b[%mma_b] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "consume_b"(%b_value) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

        %frontier_value = "frontier_value"() {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : () -> tensor<1xi32, #scalar>
        // CHECK: ttg.local_store {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
        ttg.local_store %frontier_value, %frontier {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<1xi32, #scalar> -> !ttg.memdesc<1xi32, #local_shared, #smem, mutable>

        // CHECK-NEXT: [[TAIL_A:%.*]] = nvws.semaphore.acquire {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[TAIL_B:%.*]] = nvws.semaphore.acquire {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: scf.yield {{.*}}[[TAIL_A]], [[TAIL_B]]
        scf.yield {ttg.partition = array<i32: 0, 1>} %a_read, %b_read : !ttg.async.token, !ttg.async.token
      } {tt.scheduled_max_stage = 0 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>]}

      %next = arith.addi %tile, %c0 {ttg.partition = array<i32: 0>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}
