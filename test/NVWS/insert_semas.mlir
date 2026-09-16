// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas --verify-diagnostics -cse | FileCheck %s
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas="use-meta-partitioner=true" --verify-diagnostics -cse | FileCheck %s --check-prefix=META
// RUN: not triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas=num-stages=33 2>&1 | FileCheck %s --check-prefix=NUM-STAGES

// Core insertion contracts: ownership, buffer reuse, control flow, and metadata.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true, rank = 3}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @matmul_scaled_rhs_scales_tma
  tt.func @matmul_scaled_rhs_scales_tma(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf8E4M3FN, #shared2>, %arg4: !tt.tensordesc<128x64xf8E4M3FN, #shared2>, %arg5: !tt.tensordesc<128x8xi8, #shared3>) {
    %cst = arith.constant dense<127> : tensor<128x8xi8, #linear>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    // LHS scales (no semaphore - static)
    %result = ttng.tmem_alloc %cst : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    // ACC buffer: alloc, create, initial acquire+store
    %result_1, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[ACC_EMPTY:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[ACC_FULL:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[ACC_INIT:%.*]] = nvws.semaphore.acquire [[ACC_EMPTY]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[ACC_INIT_BUF:%.*]] = nvws.semaphore.buffer [[ACC_EMPTY]], [[ACC_INIT]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[ACC_INIT_BUF]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %0 = ttng.tmem_store %cst_0, %result_1[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // RHS scales: alloc + semaphore pair
    // CHECK: [[V7:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = nvws.semaphore.create [[V7]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V9:%.*]] = nvws.semaphore.create [[V7]] {pending_count = 1 : i32} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    %1 = scf.for %arg6 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg7 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg6, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %3 = tt.descriptor_load %arg3[%arg1, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf8E4M3FN, #shared2> -> tensor<128x64xf8E4M3FN, #blocked1>
      %4 = tt.descriptor_load %arg4[%arg2, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf8E4M3FN, #shared2> -> tensor<128x64xf8E4M3FN, #blocked1>
      %5 = tt.descriptor_load %arg5[%arg1, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x8xi8, #shared3> -> tensor<128x8xi8, #linear>
      %6 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem>
      %7 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem>
      %8 = ttg.memdesc_trans %7 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>
      // RHS scales: acquire SEMPTY, buffer, store, release SFULL
      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V8]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.buffer [[V8]], [[V10]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V11]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x8xi8, #linear> -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
      %result_2 = ttng.tmem_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      // ACC buffer + RHS scales buffer for MMA
      // CHECK: nvws.semaphore.release [[V9]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[ACC_BUF:%.*]] = nvws.semaphore.buffer [[ACC_EMPTY]], [[ACC_INIT]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V14:%.*]] = nvws.semaphore.buffer [[V9]], [[V13]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma_scaled %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[ACC_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, [[V14]], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 1>}
      %9 = ttng.tc_gen5_mma_scaled %6, %8, %result_1[%arg7], %result, %result_2, %true, %true lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem>, !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %9 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 9 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop: release FULL, acquire FULL, buffer, load; no trailing EMPTY release (no further access)
    // CHECK: nvws.semaphore.release [[V8]], [[V13]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 9 : i32}
    // CHECK: nvws.semaphore.release [[ACC_FULL]], [[ACC_INIT]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 9 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[FINAL:%.*]] = nvws.semaphore.acquire [[ACC_FULL]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[FINAL_BUF:%.*]] = nvws.semaphore.buffer [[ACC_FULL]], [[FINAL]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[FINAL_BUF]][] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %val, %tok = ttng.tmem_load %result_1[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // CHECK-NOT: nvws.semaphore.release
    "use"(%val) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @shmem_sink_iterator_invalidation
  tt.func @shmem_sink_iterator_invalidation(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf16, #shared>, %arg4: !tt.tensordesc<128x64xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    // Two TMEM allocs: ACC (single-buffered) + LHS (TMEM operand, single-buffered)
    // ACC semaphore pair
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // LHS semaphore pair
    // CHECK: [[V7:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = nvws.semaphore.create [[V7]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V9:%.*]] = nvws.semaphore.create [[V7]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %3 = tt.descriptor_load %arg4[%arg2, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg3[%arg1, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %5 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // CHECK: [[V10:%.*]] = ttg.local_load %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared3, #smem> -> tensor<128x64xf16, #blocked2>
      %6 = ttg.local_load %5 {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked2>
      %7 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.memdesc_trans %7 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // LHS: acquire, buffer, store, release
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V8]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V8]], [[V11]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 1x128x64>
      // CHECK: ttng.tmem_store [[V10]], [[V12]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked2> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 1x128x64>
      %result_2 = ttng.tmem_alloc %6 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory>
      // ACC buffer + LHS acquire for MMA
      // CHECK: nvws.semaphore.release [[V9]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.buffer [[V9]], [[V14]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 1x128x64>
      // CHECK: ttng.tc_gen5_mma [[V15]], %{{[-A-Za-z0-9_.$#]+}}, [[V13]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
      %9 = ttng.tc_gen5_mma %result_2, %8, %result[%arg6], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %9 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 19 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop: release FULL, acquire FULL, buffer, load; no trailing EMPTY release (no further access)
    // CHECK: nvws.semaphore.release [[V8]], [[V14]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 19 : i32}
    // CHECK: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 19 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V16:%.*]] = nvws.semaphore.acquire [[V3]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V17:%.*]] = nvws.semaphore.buffer [[V3]], [[V16]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V17]][] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // CHECK-NOT: nvws.semaphore.release
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_smem_fanout
  tt.func @local_smem_fanout(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 100 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 100 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 2 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V7:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = [[V5]]) -> (!ttg.async.token)  : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[V4]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V3]], [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V11:%.*]] = ttg.local_load [[V10]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %l1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V2]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "use_b"(%l1) {ttg.partition = array<i32: 1>} : (!ty) -> ()

      // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V4]], [[V12]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V14:%.*]] = ttg.local_load [[V13]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %l2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V2]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "use_c"(%l2) {ttg.partition = array<i32: 2>} : (!ty) -> ()

      %v2 = "producer2"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V16:%.*]] = nvws.semaphore.buffer [[V2]], [[V15]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V16]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v2, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: scf.yield {{.*}}[[V15]]
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @local_loop_carried_and_result
  tt.func @local_loop_carried_and_result(%lb: i32, %ub: i32, %step: i32, %init: !ty) {
    %iter_alloc = ttg.local_alloc {buffer.id = 104 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %result_alloc = ttg.local_alloc {buffer.id = 105 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 104 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = ttg.local_alloc {buffer.id = 105 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V4]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V4]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V8:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (tensor<1xi32, #blocked>)  : i32 {
    %r = scf.for %i = %lb to %ub step %step iter_args(%arg = %init) -> (!ty) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V2]], [[V9]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store [[V7]], [[V10]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %arg, %iter_alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V3]], [[V11]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V13:%.*]] = ttg.local_load [[V12]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %l = ttg.local_load %iter_alloc {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V2]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "use_iter"(%l) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (!ty) -> ()

      %next = "next"(%arg) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (!ty) -> !ty
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : !ty
    // CHECK: } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}

    // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V15:%.*]] = nvws.semaphore.buffer [[V5]], [[V14]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: ttg.local_store [[V8]], [[V15]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    ttg.local_store %r, %result_alloc {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

    // CHECK: nvws.semaphore.release [[V6]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: [[V16:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V17:%.*]] = nvws.semaphore.buffer [[V6]], [[V16]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V18:%.*]] = ttg.local_load [[V17]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
    %lr = ttg.local_load %result_alloc {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
    // No trailing EMPTY release: the result buffer is not accessed again
    // CHECK-NOT: nvws.semaphore.release
    "use_result"(%lr) {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : (!ty) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @cached_exact_reuse_after_release
  tt.func @cached_exact_reuse_after_release(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[BASE:%[-A-Za-z0-9_.$#]+]] = ttg.local_alloc {buffer.id = 9820 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[ENTRY:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.create [[BASE]] released = 1 {pending_count = 1 : i32}
    // CHECK: [[FULL:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32}
    %buf = ttg.local_alloc {buffer.id = 9820 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %value = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[P1_TOKEN:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.acquire [[ENTRY]] {ttg.partition = array<i32: 1>}
      // CHECK: [[P1_WRITE_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[ENTRY]], [[P1_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[P1_WRITE_BUF]] {ttg.partition = array<i32: 1>}
      ttg.local_store %value, %buf {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]], [[P1_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}

      // CHECK: [[P0_TOKEN:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK: [[P0_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[FULL]], [[P0_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK: [[R0:%[-A-Za-z0-9_.$#]+]] = ttg.local_load [[P0_BUF]] {ttg.partition = array<i32: 0>}
      %r0 = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[ENTRY]], [[P0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "use0"([[R0]])
      "use0"(%r0) {ttg.partition = array<i32: 0>} : (!ty) -> ()

      // The first post-release read materializes an approved exact-reuse
      // buffer.  The second read must reuse that same view without making the
      // emitted-IR verifier demand a second release.
      // CHECK-NEXT: [[P1_REUSE_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[ENTRY]], [[P1_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[R1A:%[-A-Za-z0-9_.$#]+]] = ttg.local_load [[P1_REUSE_BUF]] {ttg.partition = array<i32: 1>}
      %r1a = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK-NEXT: "use1a"([[R1A]])
      "use1a"(%r1a) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      // CHECK-NEXT: [[R1B:%[-A-Za-z0-9_.$#]+]] = ttg.local_load [[P1_REUSE_BUF]] {ttg.partition = array<i32: 1>}
      %r1b = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK-NEXT: "use1b"([[R1B]])
      "use1b"(%r1b) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2d = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_model_descriptor_load_mma
  // Guard every gap, including the loop body and tail: one backing
  // allocation and two semaphore creations are allowed in this function.
  // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
  tt.func @circular_model_descriptor_load_mma(
      %desc_k: !tt.tensordesc<128x128xf16, #shared>,
      %desc_v: !tt.tensordesc<128x128xf16, #shared>,
      %lhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %tok: !ttg.async.token) {
    %false = arith.constant false
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // CHECK: [[BASE:%[-A-Za-z0-9_.$#]+]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
    // CHECK-NEXT: [[EMPTY:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.create [[BASE]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
    // CHECK-NEXT: [[FULL:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
    %k = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
    scf.for %iv = %c0 to %c1 step %c1 : i32 {
      // CHECK: [[K_EMPTY_STAGE:%[-A-Za-z0-9_.$#]+]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: [[K_EMPTY_TOK:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: [[K_EMPTY_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: nvws.descriptor_load {{.*}} 32768 [[K_EMPTY_BUF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      nvws.descriptor_load %desc_k[%c0, %c0] 32768 %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kt = ttg.memdesc_trans %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared_t, #smem, mutable>
      // CHECK: [[V_EMPTY_STAGE:%[-A-Za-z0-9_.$#]+]] = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: [[V_EMPTY_TOK:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: [[V_EMPTY_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: nvws.descriptor_load {{.*}} 32768 [[V_EMPTY_BUF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      nvws.descriptor_load %desc_v[%c0, %c0] 32768 %v {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_FULL_STAGE:%[-A-Za-z0-9_.$#]+]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} -1 : i32
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: [[K_FULL_TOK:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: [[K_FULL_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: [[K_TRANS:%[-A-Za-z0-9_.$#]+]] = ttg.memdesc_trans [[K_FULL_BUF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: [[QK:%[-A-Za-z0-9_.$#]+]] = ttng.tc_gen5_mma {{.*}}, [[K_TRANS]], {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      %qk = ttng.tc_gen5_mma %lhs, %kt, %acc[%tok], %false, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared_t, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[V_FULL_STAGE:%[-A-Za-z0-9_.$#]+]] = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: [[V_FULL_TOK:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: [[V_FULL_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: [[PV:%[-A-Za-z0-9_.$#]+]] = ttng.tc_gen5_mma {{.*}}, [[V_FULL_BUF]], {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
      %pv = ttng.tc_gen5_mma %lhs, %v, %acc[%qk], %true, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      "use_token"(%pv) {ttg.partition = array<i32: 1>} : (!ttg.async.token) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 3>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: tt.return
    // CHECK-NOT: {{ttg\.local_alloc|nvws\.semaphore\.create}}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @conditional_multi_result_if_token
  tt.func @conditional_multi_result_if_token(%lhs: !ttg.memdesc<128x64xf16, #shared, #smem>, %rhs: !ttg.memdesc<64x128xf16, #shared, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant true
    %false = arith.constant false

    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[INITIAL:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[LOOP:%.*]]:3 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[USE_ACC:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[CARRY_VALUE:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[CARRY_TOKEN:%.*]] = [[INITIAL]]) -> (i1, i32, !ttg.async.token)  : i32 {
    %loop:3 = scf.for %iv = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%use_acc = %false, %tok = %acc_tok, %carry = %c0_i32) -> (i1, !ttg.async.token, i32) : i32 {
      // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[CARRY_TOKEN]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V8]][], [[USE_ACC]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok], %use_acc, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %cond = arith.cmpi eq, %iv, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32

      // CHECK: [[BRANCH:%.*]]:3 = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (i32, i1, !ttg.async.token) {
      %epilogue:3 = scf.if %cond -> (i32, !ttg.async.token, i1) {
        // CHECK: nvws.semaphore.release [[V3]], [[CARRY_TOKEN]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V3]], [[V9]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V10]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %value, %load_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V2]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[HAND_BACK:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {{.*}}[[HAND_BACK]] : i32, i1, !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %iv, %load_tok, %true : i32, !ttg.async.token, i1
      } else {
        // CHECK: scf.yield {{.*}}[[CARRY_VALUE]], [[USE_ACC]], [[CARRY_TOKEN]] : i32, i1, !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %carry, %mma, %use_acc : i32, !ttg.async.token, i1
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0, 1>, array<i32: 1>]}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>, array<i32: 0, 1>]}
      %next = arith.addi %epilogue#0, %c1_i32 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: scf.yield {{.*}}[[BRANCH]]#1, %{{[-A-Za-z0-9_.$#]+}}, [[BRANCH]]#2 : i1, i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %epilogue#2, %epilogue#1, %next : i1, !ttg.async.token, i32
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @if_split_yield_routing_three_partitions
  tt.func @if_split_yield_routing_three_partitions(%lhs: !ttg.memdesc<128x64xf16, #shared, #smem>, %rhs: !ttg.memdesc<64x128xf16, #shared, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant true
    %false = arith.constant false

    // Double-buffered (2x): alloc, then the EMPTY/FULL semaphore pair.
    // CHECK: [[ALLOC2:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY2:%.*]] = nvws.semaphore.create [[ALLOC2]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
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
      // CHECK: ttng.tmem_load [[CONS2BUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @live_tag_source_after_prior_loop_threading
  tt.func @live_tag_source_after_prior_loop_threading(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %scratch = ttg.local_alloc {buffer.id = 910 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 910 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = ttng.tmem_alloc {buffer.id = 900 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V4]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V4]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V7:%.*]] = nvws.semaphore.create [[V4]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V11:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V9:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V10:%.*]] = [[V8]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V5]], [[V10]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V12]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %acc, %tok = ttng.tmem_alloc %cst {buffer.id = 900 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: nvws.semaphore.release [[V7]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      %inner_tmem = scf.for %iv1 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
        %lhs = "load_lhs"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %rhs = "load_rhs"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V14:%.*]] = nvws.semaphore.buffer [[V7]], [[V13]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V14]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: nvws.semaphore.release [[V6]], [[V13]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V16:%.*]] = nvws.semaphore.buffer [[V6]], [[V15]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V16]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V7]], [[V15]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "use_tmem"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // CHECK: [[V17:%.*]] = nvws.semaphore.acquire [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[V5]], [[V17]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V18:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V19:%.*]] = nvws.semaphore.buffer [[V5]], [[V18]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V19]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %out, %out_tok = ttng.tmem_load %acc[%inner_tmem] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_tmem_post"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      %payload = "local_payload"() {ttg.partition = array<i32: 2>} : () -> tensor<128x128xf16, #blocked>
      // CHECK: [[V20:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V21:%.*]] = nvws.semaphore.buffer [[V2]], [[V20]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V21]] {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload, %scratch {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V20]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V22:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      scf.for %iv2 = %lb to %ub step %step : i32 {
        // CHECK: [[V23:%.*]] = nvws.semaphore.buffer [[V3]], [[V22]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // CHECK: [[V24:%.*]] = ttg.local_load [[V23]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
        %loaded = ttg.local_load %scratch {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
        "use_local"(%loaded) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      } {ttg.partition = array<i32: 1>}

      %next = arith.addi %tile, %c0_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: nvws.semaphore.release [[V2]], [[V22]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V18]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %next : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_small = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @mixed_copy_same_backing
  // One 256x128 host covers the smaller two-copy 128x64 view.
  tt.func @mixed_copy_same_backing(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: %[[BASE:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 600 : i32} : () -> !ttg.memdesc<1x256x128xf16
    %host = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 600 : i32} : () -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    // CHECK-NEXT: %[[RING0:.*]] = ttg.memdesc_reinterpret %[[BASE]] : {{.*}} -> !ttg.memdesc<2x128x64xf16
    // CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i32
    // CHECK-NEXT: %[[SLOT0:.*]] = ttg.memdesc_index %[[RING0]][%[[ZERO]]]
    // CHECK-NEXT: %[[VIEW0:.*]] = ttg.memdesc_reinterpret %[[SLOT0]] : {{.*}} -> !ttg.memdesc<1x128x64xf16
    %slot0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 600 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // The second slot reuses the same ring view after CSE.
    // CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : i32
    // CHECK-NEXT: %[[SLOT1:.*]] = ttg.memdesc_index %[[RING0]][%[[ONE]]]
    // CHECK-NEXT: %[[VIEW1:.*]] = ttg.memdesc_reinterpret %[[SLOT1]] : {{.*}} -> !ttg.memdesc<1x128x64xf16
    // CHECK-NEXT: %[[ENTRY:.*]] = nvws.semaphore.create %[[BASE]], %[[VIEW0]], %[[VIEW1]] released = 1 {pending_count = 1 : i32}
    // CHECK-NEXT: %[[TO_R0:.*]] = nvws.semaphore.create %[[BASE]], %[[VIEW0]], %[[VIEW1]] {pending_count = 1 : i32}
    // CHECK-NEXT: %[[TO_W1:.*]] = nvws.semaphore.create %[[BASE]], %[[VIEW0]], %[[VIEW1]] {pending_count = 1 : i32}
    %slot1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 600 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %wide = arith.constant dense<0.0> : tensor<256x128xf16, #blocked>
    %half = arith.constant dense<1.0> : tensor<128x64xf16, #blocked64>
    // CHECK: %[[INIT:.*]] = nvws.semaphore.acquire %[[ENTRY]]
    // CHECK-NEXT: %[[LOOP_RESULT:.*]] = scf.for {{.*}} iter_args(%[[CARRY:.*]] = %[[INIT]])
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: %[[HOST_WRITE_BUF:.*]]:3 = nvws.semaphore.buffer %[[ENTRY]], %[[CARRY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store %{{.*}}, %[[HOST_WRITE_BUF]]#0 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release %[[TO_R0]], %[[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %wide, %host {ttg.partition = array<i32: 0>} : tensor<256x128xf16, #blocked> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
      // CHECK-NEXT: %[[R0_TOKEN:.*]] = nvws.semaphore.acquire %[[TO_R0]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: %[[R0_BUF:.*]]:3 = nvws.semaphore.buffer %[[TO_R0]], %[[R0_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: %[[R0_VALUE:.*]] = ttg.local_load %[[R0_BUF]]#1 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release %[[TO_W1]], %[[R0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %r0 = ttg.local_load %slot0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "use0"(%r0) {ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked64>) -> ()
      // CHECK: %[[W1_TOKEN:.*]] = nvws.semaphore.acquire %[[TO_W1]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: %[[W1_BUF:.*]]:3 = nvws.semaphore.buffer %[[TO_W1]], %[[W1_TOKEN]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttg.local_store %{{.*}}, %[[W1_BUF]]#2 {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release %[[ENTRY]], %[[W1_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      ttg.local_store %half, %slot1 {ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK-NEXT: %[[HOST_READ_TOKEN:.*]] = nvws.semaphore.acquire %[[ENTRY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: %[[HOST_READ_BUF:.*]]:3 = nvws.semaphore.buffer %[[ENTRY]], %[[HOST_READ_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: %[[HOST_VALUE:.*]] = ttg.local_load %[[HOST_READ_BUF]]#0 {ttg.partition = array<i32: 0>}
      %r1 = ttg.local_load %host {ttg.partition = array<i32: 0>} : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #blocked>
      "use1"(%r1) {ttg.partition = array<i32: 0>} : (tensor<256x128xf16, #blocked>) -> ()
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %[[HOST_READ_TOKEN]] : !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>,
       ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

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
    // CHECK: [[V10:%.*]] = nvws.semaphore.create [[V1]], [[V1]], [[V1]], [[V1]], [[V1]] released = 3 {pending_count = 2 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V11:%.*]] = nvws.semaphore.create [[V1]], [[V1]], [[V1]], [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V13:%.*]] = nvws.semaphore.create [[V1]], [[V1]], [[V1]], [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V15:%.*]] = nvws.semaphore.create [[V1]], [[V1]], [[V1]], [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V16:%.*]] = nvws.semaphore.create [[V1]], [[V1]], [[V1]], [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V17:%.*]] = nvws.semaphore.create [[V1]], [[V1]], [[V1]], [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V19:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V18:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // m0: alpha sliver at column 64, produced by {1}, consumed by {2}.
      %alpha = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V20:%.*]] = nvws.semaphore.acquire [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V21:%.*]]:5 = nvws.semaphore.buffer [[V10]], [[V20]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V21_M0:%.*]] = ttng.tmem_subslice [[V21]]#0 {offset = 64 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V21_M0]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
      ttng.tmem_store %cst1, %alpha, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V11]], [[V20]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V22:%.*]] = nvws.semaphore.acquire [[V11]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V23:%.*]]:5 = nvws.semaphore.buffer [[V11]], [[V22]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V23_M0:%.*]] = ttng.tmem_subslice [[V23]]#0 {offset = 64 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V23_M0]][] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x1xf32, #blocked2>
      %av, %at = ttng.tmem_load %alpha[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_alpha"(%av) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // m1: m sliver at column 66, produced by {1}, consumed by {2}.
      %m = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 66 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V25:%.*]]:5 = nvws.semaphore.buffer [[V10]], [[V20]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V25_M1:%.*]] = ttng.tmem_subslice [[V25]]#1 {offset = 66 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V25_M1]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
      ttng.tmem_store %cst1, %m, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V13]], [[V20]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V26:%.*]] = nvws.semaphore.acquire [[V13]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V27:%.*]]:5 = nvws.semaphore.buffer [[V13]], [[V26]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V27_M1:%.*]] = ttng.tmem_subslice [[V27]]#1 {offset = 66 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V27_M1]][] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x1xf32, #blocked2>
      %mv, %mt = ttng.tmem_load %m[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_m"(%mv) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // m2: l sliver at column 65, produced by {1}, consumed by {2}.
      %l = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 65 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V29:%.*]]:5 = nvws.semaphore.buffer [[V10]], [[V20]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V29_M2:%.*]] = ttng.tmem_subslice [[V29]]#2 {offset = 65 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V29_M2]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
      ttng.tmem_store %cst1, %l, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V15]], [[V20]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V30:%.*]] = nvws.semaphore.acquire [[V15]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V31:%.*]]:5 = nvws.semaphore.buffer [[V15]], [[V30]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V31_M2:%.*]] = ttng.tmem_subslice [[V31]]#2 {offset = 65 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V31_M2]][] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x1xf32, #blocked2>
      %lv, %lt = ttng.tmem_load %l[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_l"(%lv) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // m3: the spanning accumulator [0,128), produced by {2}, consumed
      // by {0}. It overlaps ALL other members.
      %acc, %tacc = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 2>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: [[V32:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V31]]#3[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %acc0 = ttng.tmem_store %cst, %acc[%tacc], %true {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V16]], [[V30]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[V10]], [[V30]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V33:%.*]] = nvws.semaphore.acquire [[V16]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V34:%.*]]:5 = nvws.semaphore.buffer [[V16]], [[V33]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V34]]#3[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %accv, %acct = ttng.tmem_load %acc[%acc0] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V10]], [[V33]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_acc"(%accv) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      // m4: p at [0,64) - disjoint from every sliver, overlaps only the
      // accumulator. Produced by {0}, consumed by {1}.
      %p = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: [[V34_M4:%.*]] = ttng.tmem_subslice [[V34]]#4 {offset = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V34_M4]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
      ttng.tmem_store %cst64, %p, %true {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #half_blocked> -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V17]], [[V33]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V35:%.*]] = nvws.semaphore.acquire [[V17]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V36:%.*]]:5 = nvws.semaphore.buffer [[V17]], [[V35]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V36_M4:%.*]] = ttng.tmem_subslice [[V36]]#4 {offset = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V36_M4]][] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x64xf32, #blocked1>
      %pv, %pt = ttng.tmem_load %p[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #half_blocked>
      "use_p"(%pv) {ttg.partition = array<i32: 1>} : (tensor<128x64xf32, #half_blocked>) -> ()

      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The outer body consumes the same allocation after the inner writer/reader
  // recurrence. Its acquire stays next to the inner MMA; one final LOCAL_EMPTY
  // acquire after the inner loop hands the buffer to the outer continuation.
  // CHECK-LABEL:   tt.func @nested_ws_inner_loop_parent_continuation(
  tt.func @nested_ws_inner_loop_parent_continuation(%lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    // Four semaphores share the allocation: LOCAL EMPTY/FULL ping-pong the
    // inner recurrence, OUTER EMPTY/FULL fence the outer continuation.
    // CHECK:           [[LIVE_ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[LOCAL_EMPTY:%.*]] = nvws.semaphore.create [[LIVE_ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[OUTER_EMPTY:%.*]] = nvws.semaphore.create [[LIVE_ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[LOCAL_FULL:%.*]] = nvws.semaphore.create [[LIVE_ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[OUTER_FULL:%.*]] = nvws.semaphore.create [[LIVE_ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // The pre-loop entry acquire drains OUTER_EMPTY's initial permit so the
    // in-body tail acquire waits on the outer reader; its token is unused.
    // CHECK:           [[OUTER_ENTRY:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %res, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // Neither loop carries a token: the original iter_args are dropped.
    // CHECK:           scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
    %o = scf.for %iv0 = %lb to %ub step %step iter_args(%t0 = %tok) -> (!ttg.async.token) : i32 {
      // CHECK-NEXT:        scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
      %i = scf.for %iv = %lb to %ub step %step iter_args(%t1 = %t0) -> (!ttg.async.token) : i32 {
        %sA = "loadA"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %sB = "loadB"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        // Writer (partition 1): acquire LOCAL_EMPTY at first toucher, buffer, MMA, release LOCAL_FULL.
        // CHECK:               [[LACQ:%.*]] = nvws.semaphore.acquire [[LOCAL_EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:               [[LBUF:%.*]] = nvws.semaphore.buffer [[LOCAL_EMPTY]], [[LACQ]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:               {{.*}} = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[LBUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %sA, %sB, %res[%t1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK:               nvws.semaphore.release [[LOCAL_FULL]], [[LACQ]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // Reader (partition 0): acquire LOCAL_FULL, buffer, tmem_load, release LOCAL_EMPTY.
        // CHECK:               [[LRTOK:%.*]] = nvws.semaphore.acquire [[LOCAL_FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:               [[LRBUF:%.*]] = nvws.semaphore.buffer [[LOCAL_FULL]], [[LRTOK]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:               %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[LRBUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %t2 = ttng.tmem_load %res[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK:               nvws.semaphore.release [[LOCAL_EMPTY]], [[LRTOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK:               "use_inner"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        "use_inner"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK-NOT:           scf.yield
        scf.yield {ttg.partition = array<i32: 0, 1>} %t2 : !ttg.async.token
      // CHECK:             } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // Handoff: the final LOCAL_EMPTY acquire after the inner loop is the
      // token released to OUTER_FULL (async_op none: the last inner reader
      // already arrived after MMA completion).
      // CHECK-NEXT:        [[FINAL_LOCAL:%.*]] = nvws.semaphore.acquire [[LOCAL_EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT:        nvws.semaphore.release [[OUTER_FULL]], [[FINAL_LOCAL]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Outer reader (partition 0): acquire OUTER_FULL, buffer, tmem_load, release OUTER_EMPTY.
      // CHECK:             [[OACQ:%.*]] = nvws.semaphore.acquire [[OUTER_FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK:             [[OBUF:%.*]] = nvws.semaphore.buffer [[OUTER_FULL]], [[OACQ]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:             %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[OBUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %outerVal, %t3 = ttng.tmem_load %res[%i] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK:             nvws.semaphore.release [[OUTER_EMPTY]], [[OACQ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK:             "use_outer"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      "use_outer"(%outerVal) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // Bridge: regain OUTER_EMPTY and refeed LOCAL_EMPTY for the next outer
      // iteration; without this, iteration two deadlocks.
      // CHECK:             [[OUTER_TAIL:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT:        nvws.semaphore.release [[LOCAL_EMPTY]], [[OUTER_TAIL]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK-NOT:           scf.yield
      scf.yield {ttg.partition = array<i32: 0, 1>} %t3 : !ttg.async.token
    // CHECK:           } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 1 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 1 : i32}
    // CHECK:           tt.return
    tt.return
  }
}

// -----

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
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V5]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked256 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem128 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem256 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @container_with_overlapping_subviews
  tt.func @container_with_overlapping_subviews(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst256 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked256>
    %cst128 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked64>
    %true = arith.constant true

    // Keep the whole container in every tuple slot; select a stage before slicing.
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc {buffer.id = 901 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>

    // One EMPTY semaphore (pending=2) and three FULL semaphores (pending=1)
    // over the single collapsed resourceKey of three memdesc members.
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]], [[ALLOC]], [[ALLOC]] released = 3 {pending_count = 2 : i32} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[F0:%.*]] = nvws.semaphore.create [[ALLOC]], [[ALLOC]], [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[F1:%.*]] = nvws.semaphore.create [[ALLOC]], [[ALLOC]], [[ALLOC]] {pending_count = 1 : i32}
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]], [[ALLOC]], [[ALLOC]] {pending_count = 1 : i32}

    // No carried token here: EMPTY is acquired inside the loop body.
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {

      // Producer (p0): acquire EMPTY, buffer the container slot, store full extent, release F0.
      // CHECK: [[A0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]]:3 = nvws.semaphore.buffer [[EMPTY]], [[A0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B0]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: nvws.semaphore.release [[F0]], [[A0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %m0, %t0 = ttng.tmem_alloc %cst256 {buffer.id = 901 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x256xf32, #blocked256>) -> (!ttg.memdesc<128x256xf32, #tmem256, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m1, %t1 = ttng.tmem_alloc %cst128 {buffer.id = 901 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m2, %t2 = ttng.tmem_alloc %cst64 {buffer.id = 901 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf32, #blocked64>) -> (!ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // Sub-view writer p1 (m1 = [0,128)): acquire F0, buffer, store member #1, release F1.
      // CHECK: [[A1:%.*]] = nvws.semaphore.acquire [[F0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B1:%.*]]:3 = nvws.semaphore.buffer [[F0]], [[A1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: [[B1_M1:%.*]] = ttng.tmem_subslice [[B1]]#1 {offset = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B1_M1]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: nvws.semaphore.release [[F1]], [[A1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // Sub-view writer p2 (m2 = [64,128)): acquire F1, buffer, store member #2, release F2.
      // CHECK: [[A2:%.*]] = nvws.semaphore.acquire [[F1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B2:%.*]]:3 = nvws.semaphore.buffer [[F1]], [[A2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: [[B2_M2:%.*]] = ttng.tmem_subslice [[B2]]#2 {offset = 64 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B2_M2]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: nvws.semaphore.release [[F2]], [[A2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %v1, %l1 = ttng.tmem_load %m1[%t1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %v2, %l2 = ttng.tmem_load %m2[%t2] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked64>

      // Reader p1: acquire F2, buffer, load member #1, and recycle EMPTY.
      // CHECK: [[A3:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B3:%.*]]:3 = nvws.semaphore.buffer [[F2]], [[A3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: [[B3_M1:%.*]] = ttng.tmem_subslice [[B3]]#1 {offset = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B3_M1]][] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x256> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[A3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // Reader p2 reuses its retained F1 token, loads member #2, and recycles EMPTY.
      // CHECK: [[B4:%.*]]:3 = nvws.semaphore.buffer [[F1]], [[A2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: [[B4_M2:%.*]] = ttng.tmem_subslice [[B4]]#2 {offset = 64 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B4_M2]][] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x256> -> tensor<128x64xf32, #blocked1>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[A2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // A conditional inside the inner loop.
  // The store/load/store all live inside the conditional; a single carrier is
  // threaded from a function-level acquire through outer loop, inner loop, and
  // the scf.if (point-of-use store on the carried buffer, else passes through).
  // CHECK-LABEL: @uniform_hold_s9_if_inside_inner_loop
  tt.func @uniform_hold_s9_if_inside_inner_loop(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 989 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 989 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[ICARRY:%.*]] = [[OCARRY]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[IF:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
        scf.if %cond {
          // CHECK: "producer1"
          // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ICARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
          // CHECK: ttg.local_load [[B0]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F2]], [[ICARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"() {ttg.partition = array<i32: 2>} : () -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
        // CHECK: } else {
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[ICARRY]] : !ttg.async.token
        // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[IF]] : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[INNER]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_reinterpret_alias
  // META-LABEL: @tmem_reinterpret_alias
  // NUM-STAGES: error: nvws-insert-semas: num-stages must be in [1, 32], got 33
  tt.func @tmem_reinterpret_alias(%ub: i32) {
    // CHECK: [[V0:%.*]] = ub.poison : !ttg.async.token
    // META: [[V0:%.*]] = ub.poison : !ttg.async.token
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // META: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // META: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // META: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %alloc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V5:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V4:%.*]] = [[V0]]) -> (!ttg.async.token)  : i32 {
    // META: [[V5:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V4:%.*]] = [[V0]]) -> (!ttg.async.token)  : i32 {
    %r = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%t = %tok) -> (!ttg.async.token) : i32 {
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V8:%.*]] = ttg.memdesc_reinterpret [[V7]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // META: [[V6:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // META: [[V7:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // META: [[V8:%.*]] = ttg.memdesc_reinterpret [[V7]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %view0 = ttg.memdesc_reinterpret %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[V9:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // META: [[V9:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %t0 = ttng.tmem_store %cst, %view0[%t], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.buffer [[V3]], [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V12:%.*]] = ttg.memdesc_reinterpret [[V11]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // META: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // META: [[V10:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // META: [[V11:%.*]] = nvws.semaphore.buffer [[V3]], [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // META: [[V12:%.*]] = ttg.memdesc_reinterpret [[V11]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %view1 = ttg.memdesc_reinterpret %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V12]][] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // META: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V12]][] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %val, %t1 = ttng.tmem_load %view1[%t0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // META: nvws.semaphore.release [[V2]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use"(%val) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      // The source token thread is dropped: the yield forwards the poison
      // placeholder, not an acquire token (loop-close release partition 1
      // differs from the first-acquire partition 0, so no token is carried).
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V0]] : !ttg.async.token
      // META: scf.yield {ttg.partition = array<i32: 0, 1>} [[V0]] : !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %t1 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: "use_token"([[V5]])
    // META: "use_token"([[V5]])
    "use_token"(%r) : (!ttg.async.token) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @managed_group_in_non_entry_block
  // META-LABEL: @managed_group_in_non_entry_block
  tt.func @managed_group_in_non_entry_block(%early: i1, %lb: i32, %ub: i32, %step: i32) {
    // CHECK-NOT: nvws.semaphore.create
    // CHECK: ^bb2:
    // META-NOT: nvws.semaphore.create
    // META: ^bb2:
    cf.cond_br %early, ^exit, ^work
  ^exit:
    tt.return
  ^work:
    // CHECK: [[CFG_BACKING:%.*]] = ttg.local_alloc {buffer.id = 1200 : i32}
    // CHECK: nvws.semaphore.create [[CFG_BACKING]] released = 1
    // CHECK: nvws.semaphore.create [[CFG_BACKING]]
    // META: [[CFG_BACKING:%.*]] = ttg.local_alloc {buffer.id = 1200 : i32}
    // META: nvws.semaphore.create [[CFG_BACKING]] released = 1
    // META: nvws.semaphore.create [[CFG_BACKING]]
    %alloc = ttg.local_alloc {buffer.id = 1200 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: nvws.semaphore.acquire
      // CHECK: ttg.local_store
      // CHECK: nvws.semaphore.release
      // CHECK: nvws.semaphore.acquire
      // CHECK: ttg.local_load
      // META: nvws.semaphore.acquire
      // META: ttg.local_store
      // META: nvws.semaphore.release
      // META: nvws.semaphore.acquire
      // META: ttg.local_load
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "consume"(%loaded) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reject_managed_flow_across_cfg_blocks(%early: i1, %lb: i32, %ub: i32, %step: i32) {
    // expected-error @below {{nvws-insert-semas: managed memdesc flow across function CFG blocks is unsupported}}
    %alloc = ttg.local_alloc {buffer.id = 1201 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    cf.cond_br %early, ^exit, ^work
  ^exit:
    tt.return
  ^work:
    scf.for %i = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @reject_conflicting_tmem_copies(%lb: i32, %ub: i32, %step: i32) {
    // expected-note @below {{first buffer.copy value is 1}}
    %a = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 77 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @below {{nvws-insert-semas: TMEM allocations sharing buffer.id 77 have conflicting buffer.copy values 1 and 2}}
    %b = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 77 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reject_circular_missing_id(%lb: i32, %ub: i32, %step: i32) {
    // expected-error @below {{nvws-insert-semas: circular local alloc requires buffer.id}}
    %alloc = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #ttg.shared_memory, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reject_circular_missing_copy(%lb: i32, %ub: i32, %step: i32) {
    // expected-error @below {{nvws-insert-semas: circular local alloc requires buffer.copy}}
    %alloc = ttg.local_alloc {buffer.circular, buffer.id = 1202 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #ttg.shared_memory, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reject_circular_missing_start(%lb: i32, %ub: i32, %step: i32) {
    // expected-error @below {{nvws-insert-semas: circular local alloc requires buffer.start}}
    %alloc = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 1203 : i32} : () -> !ttg.memdesc<1xi32, #shared, #ttg.shared_memory, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reject_circular_offset(%lb: i32, %ub: i32, %step: i32) {
    // expected-error @below {{nvws-insert-semas: circular local alloc must not carry buffer.offset}}
    %alloc = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 1204 : i32, buffer.offset = 0 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #ttg.shared_memory, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
    } {tt.warp_specialize}
    tt.return
  }
}
