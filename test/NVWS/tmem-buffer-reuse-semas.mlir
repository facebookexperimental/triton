// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @sourceful_tokenless_alias
  tt.func @sourceful_tokenless_alias(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 7 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]], [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V4:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]]:2 = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V7]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %a = ttng.tmem_alloc %cst {buffer.id = 7 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V7]]#0[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %va, %ta = ttng.tmem_load %a[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V9:%.*]]:2 = nvws.semaphore.buffer [[V3]], [[V8]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %b = ttng.tmem_alloc %cst_0 {buffer.id = 7 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V9]]#1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %vb, %tb = ttng.tmem_load %b[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V8]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

  // CHECK-LABEL: @same_owner_alias_no_semaphore
  tt.func @same_owner_alias_no_semaphore(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: [[V2:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V1:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a = ttng.tmem_alloc %cst {buffer.id = 8 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load %{{[-A-Za-z0-9_.$#]+}}[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %va, %ta = ttng.tmem_load %a[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      %b = ttng.tmem_alloc %cst_0 {buffer.id = 8 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load %{{[-A-Za-z0-9_.$#]+}}[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %vb, %tb = ttng.tmem_load %b[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%vb) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 1>} : i32
      scf.yield {ttg.partition = array<i32: 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 1 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

  // CHECK-LABEL: @loop_token_slot_alias
  tt.func @loop_token_slot_alias(%lb: i32, %ub: i32, %step: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %a, %ta = ttng.tmem_alloc {buffer.id = 9 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %b, %tb = ttng.tmem_alloc {buffer.id = 9 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 9 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]], [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    %r:2 = scf.for %iv = %lb to %ub step %step iter_args(%t0 = %ta, %t1 = %tb) -> (!ttg.async.token, !ttg.async.token) : i32 {
      // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V5:%.*]]:2 = nvws.semaphore.buffer [[V2]], [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]]#0[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %s = ttng.tmem_store %cst, %a[%t0], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V8:%.*]]:2 = nvws.semaphore.buffer [[V3]], [[V7]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V8]]#1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %v, %l = ttng.tmem_load %b[%t1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V7]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use"(%v) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %s, %l : !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0>], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }

  // CHECK-LABEL: @if_branch_alias
  tt.func @if_branch_alias(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    // CHECK: [[BACKING:%.*]] = ttng.tmem_alloc {buffer.id = 10 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[ENTRY:%.*]] = nvws.semaphore.create [[BACKING]], [[BACKING]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[THEN_TO_0:%.*]] = nvws.semaphore.create [[BACKING]], [[BACKING]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[ELSE_TO_0:%.*]] = nvws.semaphore.create [[BACKING]], [[BACKING]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[THEN_BACK:%.*]] = nvws.semaphore.create [[BACKING]], [[BACKING]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[ELSE_BACK:%.*]] = nvws.semaphore.create [[BACKING]], [[BACKING]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[AFTER_TO_0:%.*]] = nvws.semaphore.create [[BACKING]], [[BACKING]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[LOOP:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[INDEX:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a, %ta = ttng.tmem_alloc {buffer.id = 10 : i32, ttg.partition = array<i32: 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %b, %tb = ttng.tmem_alloc {buffer.id = 10 : i32, ttg.partition = array<i32: 0>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // Each branch restores owner 1 before yielding; the conditional returns
      // that token directly to the following owner-1 store.
      // CHECK: [[INPUT_TOKEN:%.*]] = nvws.semaphore.acquire [[ENTRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: [[IF_TOKEN:%.*]] = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (!ttg.async.token) {
      %if:2 = scf.if %cond -> (!ttg.async.token, !ttg.async.token) {
        // CHECK: [[THEN_INPUT_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[ENTRY]], [[INPUT_TOKEN]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[THEN_INPUT_BUFFER]]#0[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %s0 = ttng.tmem_store %cst, %a[%ta], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: nvws.semaphore.release [[THEN_TO_0]], [[INPUT_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[THEN_0_TOKEN:%.*]] = nvws.semaphore.acquire [[THEN_TO_0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[THEN_0_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[THEN_TO_0]], [[THEN_0_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[THEN_0_BUFFER]]#1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %v0, %l0 = ttng.tmem_load %b[%tb] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[THEN_BACK]], [[THEN_0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "use"(%v0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[THEN_TOKEN:%.*]] = nvws.semaphore.acquire [[THEN_BACK]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: scf.yield {{.*}}[[THEN_TOKEN]] : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %s0, %l0 : !ttg.async.token, !ttg.async.token
      } else {
        // CHECK: [[ELSE_INPUT_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[ENTRY]], [[INPUT_TOKEN]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[ELSE_INPUT_BUFFER]]#0[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %v1, %l1 = ttng.tmem_load %a[%ta] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[ELSE_TO_0]], [[INPUT_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "use"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[ELSE_0_TOKEN:%.*]] = nvws.semaphore.acquire [[ELSE_TO_0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[ELSE_0_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[ELSE_TO_0]], [[ELSE_0_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[ELSE_0_BUFFER]]#1[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %s1 = ttng.tmem_store %cst, %b[%tb], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: nvws.semaphore.release [[ELSE_BACK]], [[ELSE_0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[ELSE_TOKEN:%.*]] = nvws.semaphore.acquire [[ELSE_BACK]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: scf.yield {{.*}}[[ELSE_TOKEN]] : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %l1, %s1 : !ttg.async.token, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: [[AFTER_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[ENTRY]], [[IF_TOKEN]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[AFTER_BUFFER]]#0[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %s2 = ttng.tmem_store %cst, %a[%if#0], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[AFTER_TO_0]], [[IF_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[AFTER_0_TOKEN:%.*]] = nvws.semaphore.acquire [[AFTER_TO_0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[AFTER_0_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[AFTER_TO_0]], [[AFTER_0_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[AFTER_0_BUFFER]]#1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %v2, %l2 = ttng.tmem_load %b[%if#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[ENTRY]], [[AFTER_0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use"(%v2) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} %{{[-A-Za-z0-9_.$#]+}} : i32
      scf.yield {ttg.partition = array<i32: 0, 1>} %j : i32
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 3 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 3 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

  // CHECK-LABEL: @accumulator_and_operand_alias_same_partition
  tt.func @accumulator_and_operand_alias_same_partition(
      %lhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #linear>
    %true = arith.constant true
    %acc0, %tok0 = ttng.tmem_alloc {buffer.id = 11 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc1, %tok1 = ttng.tmem_alloc {buffer.id = 12 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 11 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = ttng.tmem_subslice [[V1]] {N = 0 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V3:%.*]] = ttg.memdesc_reinterpret [[V2]] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]], [[V3]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V1]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, [[V6:%.*]] = ttng.tmem_alloc {buffer.id = 12 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V7:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 4 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V10:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V8:%.*]] = [[V6]], [[V9:%.*]] = [[V7]]) -> (!ttg.async.token, !ttg.async.token)  : i32 {
    %r:2 = scf.for %iv = %lb to %ub step %step iter_args(%t0 = %tok0, %t1 = %tok1) -> (!ttg.async.token, !ttg.async.token) : i32 {
      // CHECK: [[V11:%.*]]:2 = nvws.semaphore.buffer [[V4]], [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V11]]#0[], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %mma0 = ttng.tc_gen5_mma %lhs, %rhs, %acc0[%t0], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V5]], [[V9]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V13:%.*]]:2 = nvws.semaphore.buffer [[V5]], [[V12]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V13]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %alias = ttng.tmem_alloc %cst_1 {buffer.id = 11 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
      // CHECK: nvws.semaphore.release [[V4]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V15:%.*]]:2 = nvws.semaphore.buffer [[V4]], [[V14]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: [[V16:%.*]] = ttng.tc_gen5_mma [[V15]]#1, %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}{{\[}}[[V8]]], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %mma1 = ttng.tc_gen5_mma %alias, %rhs, %acc1[%t1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V15]]#0[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %v, %load = ttng.tmem_load %acc0[%mma0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%v) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[V16]], [[V14]] : !ttg.async.token, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 1, 2>} %load, %mma1 : !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>], ttg.warp_specialize.tag = 4 : i32}
    tt.return
  }

  // CHECK-LABEL: @singleton_staged_alloc_preserves_buffer_attrs
  tt.func @singleton_staged_alloc_preserves_buffer_attrs(
      %lhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 13 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 13 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %init = ttng.tmem_store %cst, %acc[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%t = %init) -> (!ttg.async.token) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V9]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%t], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V3]], [[V7]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.buffer [[V3]], [[V10]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V11]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %v, %load = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use"(%v) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V12]]
      scf.yield {ttg.partition = array<i32: 0, 1>} %load : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 5 : i32}
    tt.return
  }

  // CHECK-LABEL: @n_owner_alias_sequence
  tt.func @n_owner_alias_sequence(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 14 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]], [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V8:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V7:%.*]] = [[V5]]) -> (i32, !ttg.async.token)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[V9:%.*]]:2 = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %a = ttng.tmem_alloc %cst0 {buffer.id = 14 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V3]], [[V7]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V11:%.*]]:2 = nvws.semaphore.buffer [[V3]], [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V11]]#0[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %va, %ta = ttng.tmem_load %a[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V4]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V13:%.*]]:2 = nvws.semaphore.buffer [[V4]], [[V12]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V13]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %b = ttng.tmem_alloc %cst1 {buffer.id = 14 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V2]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V15:%.*]]:2 = nvws.semaphore.buffer [[V2]], [[V14]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V15]]#1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %vb, %tb = ttng.tmem_load %b[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: scf.yield {{.*}}[[V14]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 6 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
