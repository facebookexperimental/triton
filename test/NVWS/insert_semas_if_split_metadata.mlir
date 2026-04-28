// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// Regression for branch-carried conditional metadata with non-default
// partitions. The taken branch returns ownership to partition 4, the other
// branch passes partition 4's token through, and the if result remains the
// loop-carried token. Nothing may assume partitions 0/1.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @if_split_workaround_nondefault_partitions
  tt.func @if_split_workaround_nondefault_partitions(%arg0: !tt.tensordesc<tensor<1x64xf16, #shared>>, %arg1: tensor<64x128x!tt.ptr<f16>, #blocked3> {tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    // Single-buffered (disallow_acc_multi_buffer): alloc grows to 1x, then the
    // EMPTY/FULL semaphore pair, the initial acquire of EMPTY, its buffer, and
    // the init store writing through that buffer.
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[INITTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[INITBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[INITTOK]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[INITBUF]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // The acquire token is threaded as the loop's third iter_arg.
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}, [[CARRYTOK:%.*]] = [[INITTOK]]) -> (i1, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.async.token)  : i32 {
    %1:3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %arg1, %arg5 = %0) -> (i1, tensor<64x128x!tt.ptr<f16>, #blocked3>, !ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 4, 5>} : (i32) -> (i32, tensor<64x128xi32, #blocked3>, i32)
      %3 = tt.splat %2#0 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : i32 -> tensor<128xi32, #blocked2>
      %4 = tt.descriptor_gather %arg0[%3, %2#2] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : (!tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #blocked2>, i32) -> tensor<128x64xf16, #blocked1>
      %5 = tt.addptr %arg4, %2#1 {loop.cluster = 3 : i32, loop.stage = 1 : i32, tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>, ttg.partition = array<i32: 4>} : tensor<64x128x!tt.ptr<f16>, #blocked3>, tensor<64x128xi32, #blocked3>
      %6 = tt.load %5 {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<64x128x!tt.ptr<f16>, #blocked3>
      %7 = ttg.local_alloc %4 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 5>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // MMA reads its accumulator through a buffer derived from the carried
      // EMPTY token.
      // CHECK: [[MMABUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[CARRYTOK]] {loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[MMABUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 5 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 4>}
      %9 = ttng.tc_gen5_mma %7, %8, %result[%arg5], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 4>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %10 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 4>} : i32
      %11 = arith.select %10, %false, %true {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 4>} : i1
      // The taken branch performs the complete {4}->{0}->{4} handoff. The
      // other branch passes the loop-carried partition-4 token through.
      // CHECK: [[NEXTTOK:%.*]] = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (!ttg.async.token) {
      // CHECK: nvws.semaphore.release [[FULL]], [[CARRYTOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[CONSTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[CONSBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[CONSTOK]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_load [[CONSBUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked1>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[CONSTOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[HAND_BACK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 4>} [[HAND_BACK]] : !ttg.async.token
      // CHECK: } else {
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 4>} [[CARRYTOK]] : !ttg.async.token
      // CHECK: } {loop.cluster = 4 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 4>, ttg.partition.outputs = [array<i32: 4>]}
      %12 = scf.if %10 -> (!ttg.async.token) {
        %result_0, %token_1 = ttng.tmem_load %result[%9] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 4>} %token_1 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 4>} %9 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 4>, ttg.partition.outputs = [array<i32: 4>]}
      // CHECK: scf.yield {{.*}}[[NEXTTOK]] : i1, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 4, 5>} %11, %5, %12 : i1, tensor<64x128x!tt.ptr<f16>, #blocked3>, !ttg.async.token
    // Loop close: the schedule/partition attrs are preserved verbatim.
    // CHECK: } {tt.disallow_acc_multi_buffer, tt.num_stages = 3 : i32, tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 4, 5>, ttg.partition.outputs = [array<i32: 4>, array<i32: 4>, array<i32: 4>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 3 : i32, tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 4, 5>, ttg.partition.outputs = [array<i32: 4>, array<i32: 4>, array<i32: 4>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    // After loop: no drain; there is no post-loop TMEM access.
    tt.return
  }

  // CHECK-LABEL: @if_split_yield_routing_three_partitions
  tt.func @if_split_yield_routing_three_partitions(%lhs: !ttg.memdesc<128x64xf16, #shared, #smem>, %rhs: !ttg.memdesc<64x128xf16, #shared, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant true
    %false = arith.constant false

    // Double-buffered (2x): alloc, then the EMPTY/FULL semaphore pair.
    // CHECK: [[ALLOC2:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY2:%.*]] = nvws.semaphore.create [[ALLOC2]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[FULL2:%.*]] = nvws.semaphore.create [[ALLOC2]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[INITIAL2:%.*]] = nvws.semaphore.acquire [[EMPTY2]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK: [[LOOP2:%.*]]:3 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[USE2:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[VALUE2:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[CARRY2:%.*]] = [[INITIAL2]]) -> (i1, i32, !ttg.async.token)  : i32 {
    %loop:3 = scf.for %iv = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%use_acc = %false, %tok = %acc_tok, %carry = %c0_i32) -> (i1, !ttg.async.token, i32) : i32 {
      // The MMA uses the token carried from the preceding iteration.
      // CHECK: [[BODYBUF:%.*]] = nvws.semaphore.buffer [[EMPTY2]], [[CARRY2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[BODYBUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok], %use_acc, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %aux = "p2_work"(%iv) {ttg.partition = array<i32: 2>} : (i32) -> i32
      "p2_sink"(%aux) {ttg.partition = array<i32: 2>} : (i32) -> ()
      %cond = arith.cmpi eq, %iv, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32

      // The branch-carried if must not pick up partition 2. The taken branch
      // performs {1}->{0}->{1}; the other branch passes [[CARRY2]] through.
      // CHECK: [[BRANCH2:%.*]]:3 = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (i32, i1, !ttg.async.token) {
      // CHECK: nvws.semaphore.release [[FULL2]], [[CARRY2]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[CONS2TOK:%.*]] = nvws.semaphore.acquire [[FULL2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[CONS2BUF:%.*]] = nvws.semaphore.buffer [[FULL2]], [[CONS2TOK]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_load [[CONS2BUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked1>
      // CHECK: nvws.semaphore.release [[EMPTY2]], [[CONS2TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[BACK2:%.*]] = nvws.semaphore.acquire [[EMPTY2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[BACK2]] : i32, i1, !ttg.async.token
      // CHECK: } else {
      // CHECK: scf.yield {{.*}}[[VALUE2]], [[USE2]], [[CARRY2]] : i32, i1, !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0, 1>, array<i32: 1>]}
      %epilogue:3 = scf.if %cond -> (i32, !ttg.async.token, i1) {
        %value, %load_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %iv, %load_tok, %true : i32, !ttg.async.token, i1
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %carry, %mma, %use_acc : i32, !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>, array<i32: 0, 1>]}
      %next = arith.addi %epilogue#0, %c1_i32 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: scf.yield {{.*}}[[BRANCH2]]#1, %{{[-A-Za-z0-9_.$#]+}}, [[BRANCH2]]#2 : i1, i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %epilogue#2, %epilogue#1, %next : i1, !ttg.async.token, i32
    // Loop close: the conditional token remains the third loop result.
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}
