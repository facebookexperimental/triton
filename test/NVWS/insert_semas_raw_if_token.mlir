// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @raw_edge_token_carried_if
  tt.func @raw_edge_token_carried_if(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]], [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V6:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // Different physical resources in the same logical buffer group force the
      // raw-edge scheduler. The taken branch hands ownership to partition 1 and
      // back to partition 0; the untouched branch passes partition 0's token
      // through the if.
      %a, %ta = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %b, %tb = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V9:%.*]]:2 = nvws.semaphore.buffer [[V2]], [[V8]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V10:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]]#0[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %a0 = ttng.tmem_store %cst0, %a[%ta], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %cond = arith.cmpi eq, %iv, %c0 {ttg.partition = array<i32: 0, 1>} : i32

      // CHECK: [[IF_TOKEN:%.*]] = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (!ttg.async.token) {
      %if_tok = scf.if %cond -> (!ttg.async.token) {
        // CHECK: nvws.semaphore.release [[V3]], [[V8]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V12:%.*]]:2 = nvws.semaphore.buffer [[V3]], [[V11]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V12]]#0[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %av, %at = ttng.tmem_load %a[%a0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V4]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "use_a"(%av) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[HAND_BACK:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[HAND_BACK]] : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %at : !ttg.async.token
      } else {
        // CHECK: } else {
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V8]] : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %a0 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // CHECK: [[V14:%.*]]:2 = nvws.semaphore.buffer [[V2]], [[IF_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V15:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V14]]#0[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %a1 = ttng.tmem_store %cst1, %a[%if_tok], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK: [[V16:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V14]]#1[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %b0 = ttng.tmem_store %cst0, %b[%tb], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V5]], [[IF_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V17:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V18:%.*]]:2 = nvws.semaphore.buffer [[V5]], [[V17]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V18]]#1[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %bv, %bt = ttng.tmem_load %b[%b0] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V17]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_b"(%bv) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: scf.yield {{.*}} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
