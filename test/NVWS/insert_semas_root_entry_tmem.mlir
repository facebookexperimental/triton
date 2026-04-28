// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @root_entry_accumulator_uses_native_carried_pou
  tt.func @root_entry_accumulator_uses_native_carried_pou(
      %ub: i32,
      %lhs: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf16, #shared1, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true

    // Root initializes the accumulator before entering the WS loop. Native
    // carried POU passes that exact token into the loop. A zero-trip loop
    // returns it unchanged; a nonzero loop returns partition 1's final token.
    %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[ACC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-NEXT: [[ROOT:%.*]] = nvws.semaphore.create [[ACC]] released = -1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[TO_MMA:%.*]] = nvws.semaphore.create [[ACC]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[TO_ROOT:%.*]] = nvws.semaphore.create [[ACC]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[ROOT_TOKEN:%.*]] = nvws.semaphore.acquire [[ROOT]]
    // CHECK-NEXT: [[ROOT_BUFFER:%.*]] = nvws.semaphore.buffer [[ROOT]], [[ROOT_TOKEN]]
    // CHECK-NEXT: ttng.tmem_store %{{.*}}, [[ROOT_BUFFER]][], %{{.*}}
    %init = ttng.tmem_store %cst, %acc[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-NEXT: [[LOOP_RESULT:%.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[LOOP_TOKEN:%.*]] = [[ROOT_TOKEN]]) -> (!ttg.async.token)  : i32 {
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%carry = %init) -> (!ttg.async.token) : i32 {
      // CHECK-NEXT: [[LOOP_BUFFER:%.*]] = nvws.semaphore.buffer [[ROOT]], [[LOOP_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: %{{.*}}, %{{.*}} = ttng.tmem_load [[LOOP_BUFFER]][] {ttg.partition = array<i32: 1>}
      %loaded, %load_tok = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

      // CHECK: ttng.tmem_store %{{.*}}, [[LOOP_BUFFER]][], %{{.*}} {ttg.partition = array<i32: 1>}
      %store = ttng.tmem_store %loaded, %acc[%load_tok], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK-NEXT: nvws.semaphore.release [[TO_MMA]], [[LOOP_TOKEN]] [#nvws.async_op<none>]
      // CHECK-NEXT: [[MMA_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_MMA]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[MMA_BUFFER:%.*]] = nvws.semaphore.buffer [[TO_MMA]], [[MMA_TOKEN]] {ttg.partition = array<i32: 2>}
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%store], %true, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, [[MMA_BUFFER]][]
      // CHECK-NEXT: nvws.semaphore.release [[ROOT]], [[MMA_TOKEN]] [#nvws.async_op<tc5mma>]
      // CHECK-NEXT: [[NEXT_TOKEN:%.*]] = nvws.semaphore.acquire [[ROOT]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: scf.yield {{.*}}[[NEXT_TOKEN]] : !ttg.async.token
      scf.yield {ttg.partition = array<i32: 1, 2>} %mma : !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[TO_ROOT]], [[LOOP_RESULT]] [#nvws.async_op<none>]
    // CHECK-NEXT: [[OUT_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_ROOT]]
    // CHECK-NEXT: [[OUT_BUFFER:%.*]] = nvws.semaphore.buffer [[TO_ROOT]], [[OUT_TOKEN]]
    // CHECK-NEXT: %{{.*}}, %{{.*}} = ttng.tmem_load [[OUT_BUFFER]][]
    %out, %out_tok = ttng.tmem_load %acc[%loop] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%out) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}
