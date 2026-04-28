// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @outer_sourceful_alloc_inner_loop_reentry
  tt.func @outer_sourceful_alloc_inner_loop_reentry(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V8:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V7:%.*]] = [[V5]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %acc, %tok = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // The inner loop carries no semaphore token: partition 1 acquires [[V4]]
      // at the point of use adjacent to the MMA on every iteration.
      // CHECK: nvws.semaphore.release [[V4]], [[V7]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      %inner = scf.for %iv1 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
        %lhs = "load1"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %rhs = "load2"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[V11:%.*]] = nvws.semaphore.buffer [[V4]], [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V11]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: nvws.semaphore.release [[V3]], [[V10]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V3]], [[V12]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V13]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V4]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // Post-loop bridge: partition 1 re-acquires [[V4]] after the last in-loop
      // read and releases [[V2]]; partition 0 then reads the final value and
      // carries the fresh [[V2]] permit to the next outer iteration.
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: nvws.semaphore.release [[V2]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V16:%.*]] = nvws.semaphore.buffer [[V2]], [[V15]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V16]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %out, %out_tok = ttng.tmem_load %acc[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %next = arith.addi %tile, %c0_i32 {ttg.partition = array<i32: 0>} : i32
      // CHECK: scf.yield {{.*}}[[V15]]
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Completion differs between the conditional's stage-1 path and its
  // passthrough path.  The inner loop carries no token: the first stage-0 MMA
  // acquires at the point of use.  The conditional itself returns an owner-1
  // token: the then-path hands the partition-0 read back to owner 1, and the
  // else-path passes the stage-0 MMA token through unchanged.
  // CHECK-LABEL: @branch_completion_requires_carrier
  tt.func @branch_completion_requires_carrier(
      %cond: i1, %lb: i32, %ub: i32, %step: i32,
      %lhs: !ttg.memdesc<128x64xf32, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf32, #shared, #smem>) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 920 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[ENTRY:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[TO_BRANCH_READ:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[BACK_TO_WRITER:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[TO_POST_READ:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[LOOP_BACK:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[OUTER_INPUT:%.*]] = nvws.semaphore.acquire [[ENTRY]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[OUTER_RESULTS:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[OUTER_SCALAR:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_TOKEN:%.*]] = [[OUTER_INPUT]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %i = %lb to %ub step %step iter_args(%tile = %c0) -> (i32) : i32 {
      // CHECK: [[OUTER_BUFFER:%.*]] = nvws.semaphore.buffer [[ENTRY]], [[OUTER_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_BUFFER]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[LOOP_BACK]], [[OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %acc, %tok = ttng.tmem_alloc %cst {buffer.copy = 1 : i32, buffer.id = 920 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      %inner = scf.for %j = %lb to %ub step %step iter_args(%iter = %tok) -> (!ttg.async.token) : i32 {
        // CHECK: [[INNER_TOKEN:%.*]] = nvws.semaphore.acquire [[LOOP_BACK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[INNER_BUFFER:%.*]] = nvws.semaphore.buffer [[LOOP_BACK]], [[INNER_TOKEN]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK-NEXT: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[INNER_BUFFER]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma0 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%iter], %true, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

        // The conditional returns owner 1 on both paths.
        // CHECK: [[BRANCH_TOKEN:%.*]] = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (!ttg.async.token) {
        %branch = scf.if %cond -> (!ttg.async.token) {
          // CHECK: [[BRANCH_BUFFER:%.*]] = nvws.semaphore.buffer [[LOOP_BACK]], [[INNER_TOKEN]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK-NEXT: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[BRANCH_BUFFER]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK-NEXT: nvws.semaphore.release [[TO_BRANCH_READ]], [[INNER_TOKEN]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          %mma1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%mma0], %true, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK: [[BRANCH_READ_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_BRANCH_READ]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK-NEXT: [[BRANCH_READ_BUFFER:%.*]] = nvws.semaphore.buffer [[TO_BRANCH_READ]], [[BRANCH_READ_TOKEN]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK-NEXT: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[BRANCH_READ_BUFFER]][] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
          // CHECK-NEXT: nvws.semaphore.release [[BACK_TO_WRITER]], [[BRANCH_READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          %value0, %read0 = ttng.tmem_load %acc[%mma1] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          "consume.branch"(%value0) {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          // CHECK: [[RETURN_TOKEN:%.*]] = nvws.semaphore.acquire [[BACK_TO_WRITER]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
          // CHECK-NEXT: scf.yield {{.*}}[[RETURN_TOKEN]] : !ttg.async.token
          scf.yield {ttg.partition = array<i32: 0, 1>} %read0 : !ttg.async.token
        } else {
          // The passthrough path returns the unchanged stage-0 MMA token.
          // CHECK: } else {
          // CHECK-NEXT: scf.yield {{.*}}[[INNER_TOKEN]] : !ttg.async.token
          scf.yield {ttg.partition = array<i32: 0, 1>} %mma0 : !ttg.async.token
        // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}

        // CHECK: nvws.semaphore.release [[TO_POST_READ]], [[BRANCH_TOKEN]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[POST_READ_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_POST_READ]] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[POST_READ_BUFFER:%.*]] = nvws.semaphore.buffer [[TO_POST_READ]], [[POST_READ_TOKEN]] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK-NEXT: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[POST_READ_BUFFER]][] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        // CHECK-NEXT: nvws.semaphore.release [[LOOP_BACK]], [[POST_READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %value1, %read1 = ttng.tmem_load %acc[%branch] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "consume"(%value1) {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read1 : !ttg.async.token
      // CHECK: } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // Post-loop bridge: partition 1 re-acquires LOOP_BACK after the last
      // in-loop read and releases ENTRY; partition 0 reads the final value
      // under a fresh ENTRY permit carried to the next outer iteration.
      // CHECK: [[POST_LOOP_TOKEN:%.*]] = nvws.semaphore.acquire [[LOOP_BACK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: nvws.semaphore.release [[ENTRY]], [[POST_LOOP_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[NEXT_OUTER_TOKEN:%.*]] = nvws.semaphore.acquire [[ENTRY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: [[NEXT_OUTER_BUFFER:%.*]] = nvws.semaphore.buffer [[ENTRY]], [[NEXT_OUTER_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK-NEXT: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[NEXT_OUTER_BUFFER]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %out, %out_tok = ttng.tmem_load %acc[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "consume.post"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %next = arith.addi %tile, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: scf.yield {{.*}}[[NEXT_OUTER_TOKEN]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "consume.outer"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The recurrence acquire is constructed at the point of use with the stage-0
  // MMA's own schedule, so no stale scheduling arc is left from the stage-1
  // read to the next stage-0 MMA.  The post-loop bridge stays at the
  // partition-1 boundary instead of moving to the partition-0 final read.
  // CHECK-LABEL: @scheduled_relocated_acquire_boundaries
  tt.func @scheduled_relocated_acquire_boundaries(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V8:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V7:%.*]] = [[V5]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %acc, %tok = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // The inner loop carries no token; the acquire sits next to the MMA.
      // CHECK: nvws.semaphore.release [[V4]], [[V7]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      %inner = scf.for %iv1 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
        %lhs = "load1"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %rhs = "load2"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V4]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[V11:%.*]] = nvws.semaphore.buffer [[V4]], [[V10]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK-NEXT: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V11]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: nvws.semaphore.release [[V3]], [[V10]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[V13:%.*]] = nvws.semaphore.buffer [[V3]], [[V12]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK-NEXT: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V13]][] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %read_tok = ttng.tmem_load %acc[%mma] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V4]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "use"(%val) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
      // CHECK: } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // Post-loop bridge at the partition-1 boundary.
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: nvws.semaphore.release [[V2]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: [[V16:%.*]] = nvws.semaphore.buffer [[V2]], [[V15]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK-NEXT: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V16]][] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %out, %out_tok = ttng.tmem_load %acc[%inner] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%out) {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %next = arith.addi %tile, %c0_i32 {ttg.partition = array<i32: 0>} : i32
      // CHECK: scf.yield {{.*}}[[V15]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @three_level_reentry_without_post_access
  tt.func @three_level_reentry_without_post_access(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V6:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V5:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[V7:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %acc, %tok = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // Neither the middle nor the inner loop carries a token.
      // CHECK: nvws.semaphore.release [[V4]], [[V7]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
      %middle:2 = scf.for %iv1 = %lb to %ub step %step iter_args(%mid = %c0_i32, %mtok = %tok) -> (i32, !ttg.async.token) : i32 {
        // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
        %inner = scf.for %iv2 = %lb to %ub step %step iter_args(%tok1 = %mtok) -> (!ttg.async.token) : i32 {
          %lhs = "load1"(%iv2) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
          %rhs = "load2"(%iv2) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
          // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK-NEXT: [[V10:%.*]] = nvws.semaphore.buffer [[V4]], [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V10]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK: nvws.semaphore.release [[V3]], [[V9]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V3]], [[V11]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V12]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
          %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          // CHECK: nvws.semaphore.release [[V4]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
        // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

        %mid_next = arith.addi %mid, %c0_i32 {ttg.partition = array<i32: 0>} : i32
        scf.yield {ttg.partition = array<i32: 0, 1>} %mid_next, %inner : i32, !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>]}

      %next = arith.addi %tile, %middle#0 {ttg.partition = array<i32: 0>} : i32
      // Without a post-loop access, partition 1 bridges from the last in-loop
      // read straight to the [[V2]] release for the next outer iteration.
      // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: nvws.semaphore.release [[V2]], [[V13]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @three_level_sourceful_alloc_reentry
  tt.func @three_level_sourceful_alloc_reentry(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // The innermost recurrence gets its own initial permit.  OUTER_EMPTY still
    // carries the sourceful allocation across outer iterations.
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[INNER_EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32}
    // CHECK: [[OUTER_EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32}
    // CHECK: [[INNER_FULL:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32}
    // CHECK: [[MIDDLE_FULL:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32}
    // CHECK: [[OUTER_TO_MIDDLE:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32}
    // CHECK: [[OUTER_ENTRY:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]]
    // CHECK: [[OUTER:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[OUTER_IV:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_TOKEN:%.*]] = [[OUTER_ENTRY]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[OUTER_BUF:%.*]] = nvws.semaphore.buffer [[OUTER_EMPTY]], [[OUTER_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_BUF]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>}
      %acc, %tok = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // Partition 1 blocks on the store handoff before the middle loop; the
      // acquired token itself is unused, and the middle loop carries no token.
      // CHECK: nvws.semaphore.release [[OUTER_TO_MIDDLE]], [[OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.acquire [[OUTER_TO_MIDDLE]] {ttg.partition = array<i32: 1>}
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
      %middle:2 = scf.for %iv1 = %lb to %ub step %step iter_args(%mid = %c0_i32, %mtok = %tok) -> (i32, !ttg.async.token) : i32 {
        // The innermost loop carries no semaphore token; its EMPTY acquire is
        // adjacent to the MMA on every iteration.
        // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
        %inner = scf.for %iv2 = %lb to %ub step %step iter_args(%tok1 = %mtok) -> (!ttg.async.token) : i32 {
          %lhs = "load1"(%iv2) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
          %rhs = "load2"(%iv2) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
          // CHECK: [[INNER_ACQ:%.*]] = nvws.semaphore.acquire [[INNER_EMPTY]] {ttg.partition = array<i32: 1>}
          // CHECK-NEXT: [[INNER_BUF:%.*]] = nvws.semaphore.buffer [[INNER_EMPTY]], [[INNER_ACQ]] {ttg.partition = array<i32: 1>}
          // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[INNER_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
          %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK: nvws.semaphore.release [[INNER_FULL]], [[INNER_ACQ]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
          // CHECK: [[INNER_READ:%.*]] = nvws.semaphore.acquire [[INNER_FULL]] {ttg.partition = array<i32: 0>}
          // CHECK: [[INNER_READ_BUF:%.*]] = nvws.semaphore.buffer [[INNER_FULL]], [[INNER_READ]] {ttg.partition = array<i32: 0>}
          // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[INNER_READ_BUF]][] {ttg.partition = array<i32: 0>}
          %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          // CHECK: nvws.semaphore.release [[INNER_EMPTY]], [[INNER_READ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
          "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

        // CHECK: [[FINAL_INNER:%.*]] = nvws.semaphore.acquire [[INNER_EMPTY]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: nvws.semaphore.release [[MIDDLE_FULL]], [[FINAL_INNER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK: [[MIDDLE_READ:%.*]] = nvws.semaphore.acquire [[MIDDLE_FULL]] {ttg.partition = array<i32: 0>}
        // CHECK: [[MIDDLE_READ_BUF:%.*]] = nvws.semaphore.buffer [[MIDDLE_FULL]], [[MIDDLE_READ]] {ttg.partition = array<i32: 0>}
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[MIDDLE_READ_BUF]][] {ttg.partition = array<i32: 0>}
        %mid_out, %mid_tok = ttng.tmem_load %acc[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[OUTER_TO_MIDDLE]], [[MIDDLE_READ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        "use"(%mid_out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        %mid_next = arith.addi %mid, %c0_i32 {ttg.partition = array<i32: 0>} : i32
        // The middle regain re-arms INNER_EMPTY for the next inner loop; it is
        // no longer yielded because the middle loop carries no token.
        // CHECK: [[MIDDLE_REGAIN:%.*]] = nvws.semaphore.acquire [[OUTER_TO_MIDDLE]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: nvws.semaphore.release [[INNER_EMPTY]], [[MIDDLE_REGAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        scf.yield {ttg.partition = array<i32: 0, 1>} %mid_next, %mid_tok : i32, !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>]}

      // Post-middle bridge: a fresh INNER_EMPTY acquire anchors the
      // OUTER_EMPTY release for the outer recurrence.
      // CHECK: [[POST_MIDDLE:%.*]] = nvws.semaphore.acquire [[INNER_EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[OUTER_EMPTY]], [[POST_MIDDLE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[OUTER_READ:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK: [[OUTER_READ_BUF:%.*]] = nvws.semaphore.buffer [[OUTER_EMPTY]], [[OUTER_READ]] {ttg.partition = array<i32: 0>}
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[OUTER_READ_BUF]][] {ttg.partition = array<i32: 0>}
      %out, %out_tok = ttng.tmem_load %acc[%middle#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %next = arith.addi %tile, %middle#0 {ttg.partition = array<i32: 0>} : i32
      // CHECK: scf.yield {{.*}}[[OUTER_READ]]
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}
