// RUN: triton-opt %s -split-input-file --nvgpu-test-taskid-propagate=num-warp-groups=2 | FileCheck %s

// Regression tests for B-20-F2 / T273501458.
//
// Operand-D init propagation must find the pre-loop TMEM store even when the
// MMA D operand is a descriptor derived from the TMEM allocation.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @direct_tmem_d_init_store_gets_mma_task
  // CHECK:       ttng.tmem_store {{.*}} {async_task_id = array<i32: 1>}
  // CHECK:       ttng.tc_gen5_mma {{.*}} {async_task_id = array<i32: 1>}
  tt.func public @direct_tmem_d_init_store_gets_mma_task(%a: !ttg.memdesc<128x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x128xf16, #shared1, #smem>, %n_tiles: i32) {
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %acc, %acc_token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_token = ttng.tmem_store %zero, %acc[%acc_token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %result = scf.for %iv = %c0 to %n_tiles step %c1 iter_args(%dep = %acc_token) -> (!ttg.async.token) : i32 {
      %mma_token = ttng.tc_gen5_mma %a, %b, %acc[%dep], %false, %true {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %mma_token : !ttg.async.token
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @subslice_tmem_d_init_store_gets_mma_task
  // CHECK:       ttng.tmem_subslice
  // CHECK:       ttng.tmem_store {{.*}} {async_task_id = array<i32: 1>}
  // CHECK:       ttng.tc_gen5_mma {{.*}} {async_task_id = array<i32: 1>}
  tt.func public @subslice_tmem_d_init_store_gets_mma_task(%a: !ttg.memdesc<128x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x64xf16, #shared1, #smem>, %n_tiles: i32) {
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %acc_base, %acc_token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc = ttng.tmem_subslice %acc_base {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
    %init_token = ttng.tmem_store %zero, %acc[%acc_token], %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
    %result = scf.for %iv = %c0 to %n_tiles step %c1 iter_args(%dep = %acc_token) -> (!ttg.async.token) : i32 {
      %mma_token = ttng.tc_gen5_mma %a, %b, %acc[%dep], %false, %true {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
      scf.yield %mma_token : !ttg.async.token
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @indexed_tmem_d_init_store_gets_mma_task
  // CHECK:       ttg.memdesc_index
  // CHECK:       ttng.tmem_store {{.*}} {async_task_id = array<i32: 1>}
  // CHECK:       ttng.tc_gen5_mma {{.*}} {async_task_id = array<i32: 1>}
  tt.func public @indexed_tmem_d_init_store_gets_mma_task(%a: !ttg.memdesc<128x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x128xf16, #shared1, #smem>, %n_tiles: i32) {
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %acc_base, %acc_token = ttng.tmem_alloc : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc = ttg.memdesc_index %acc_base[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %init_token = ttng.tmem_store %zero, %acc[%acc_token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %result = scf.for %iv = %c0 to %n_tiles step %c1 iter_args(%dep = %acc_token) -> (!ttg.async.token) : i32 {
      %mma_token = ttng.tc_gen5_mma %a, %b, %acc[%dep], %false, %true {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %mma_token : !ttg.async.token
    }
    tt.return
  }
}
