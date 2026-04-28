// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_loop_carried_linear_chain_no_exit_drain
  tt.func @tmem_loop_carried_linear_chain_no_exit_drain(
      %lb: i32, %ub: i32, %step: i32,
      %rhs: !ttg.memdesc<128x128xf16, #shared, #smem>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %cst_f16 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst_f32 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: [[V1:%.*]], [[V2:%.*]] = ttng.tmem_alloc %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc, %acc_tok = ttng.tmem_alloc %cst_f32 {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK: [[V3:%.*]] = ttng.tmem_alloc {buffer.id = 920 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V3]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V7:%.*]] = [[V2]]) -> (i32, !ttg.async.token)  : i32 {
    %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0, %tok = %acc_tok) -> (i32, !ttg.async.token) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V4]], [[V9]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V10]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 5>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %frag = ttng.tmem_alloc %cst_f16 {buffer.id = 920 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V5]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V5]], [[V11]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: [[V13:%.*]] = ttng.tc_gen5_mma [[V12]], %{{[-A-Za-z0-9_.$#]+}}, [[V1]]{{\[}}[[V7]]{{\]}}, %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %read = ttng.tc_gen5_mma %frag, %rhs, %acc[%tok], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %next_i = arith.addi %i, %c0 {ttg.partition = array<i32: 5>} : i32
      // CHECK: nvws.semaphore.release [[V4]], [[V11]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 5>} %{{[-A-Za-z0-9_.$#]+}}, [[V13]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 1, 5>} %next_i, %read : i32, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 5>, ttg.partition.outputs = [array<i32: 5>, array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}

    // No loop-exit drain: no acquire/release may appear after the loop.
    // CHECK-NOT: nvws.semaphore.acquire
    // CHECK-NOT: nvws.semaphore.release
    "use"(%loop#0) : (i32) -> ()
    tt.return
  }
}
