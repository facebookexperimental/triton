// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-hoist-tmem-store | FileCheck %s

// Test hoisting a loop-invariant TMEMStore out of an outer ForOp when the inner
// loop's MMA has useD=false (statically).
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @hoist_invariant_tmem_store
  // The store should be hoisted before the outer loop.
  // CHECK: %[[ZEROS:.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[HOISTED_TOK:.*]] = ttng.tmem_store %[[ZEROS]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[HOISTED_TOK]],
  // CHECK-NOT: ttng.tmem_store
  // CHECK:   scf.for
  // CHECK:     ttng.tc_gen5_mma
  // CHECK:   ttng.tmem_load
  tt.func public @hoist_invariant_tmem_store(
      %A_sh: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>,
      %B_sh: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>,
      %N: i32, %K: i32) -> tensor<128x128xf32, #blocked> {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %acc_tm, %tok0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %outer:2 = scf.for %i = %c0_i32 to %N step %c1_i32 iter_args(%tok = %tok0, %out = %cst) -> (!ttg.async.token, tensor<128x128xf32, #blocked>)  : i32 {
      // Zero the accumulator every outer iteration.
      %tok1 = ttng.tmem_store %cst, %acc_tm[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // Inner K-loop with useD=false.
      %inner = scf.for %j = %c0_i32 to %K step %c1_i32 iter_args(%inner_tok = %tok1) -> (!ttg.async.token)  : i32 {
        %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%inner_tok], %false, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %mma_tok : !ttg.async.token
      }
      %result, %load_tok = ttng.tmem_load %acc_tm[%inner] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %load_tok, %result : !ttg.async.token, tensor<128x128xf32, #blocked>
    }
    tt.return %outer#1 : tensor<128x128xf32, #blocked>
  }
}

// -----

// Test hoisting with a loop-carried useD flag that starts false.
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @hoist_loop_carried_use_d
  // CHECK: %[[ZEROS:.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[HOISTED_TOK:.*]] = ttng.tmem_store %[[ZEROS]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[HOISTED_TOK]],
  // CHECK-NOT: ttng.tmem_store
  // CHECK:   scf.for
  // CHECK:     ttng.tc_gen5_mma
  // CHECK:   ttng.tmem_load
  tt.func public @hoist_loop_carried_use_d(
      %A_sh: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>,
      %B_sh: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>,
      %N: i32, %K: i32) -> tensor<128x128xf32, #blocked> {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %acc_tm, %tok0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %outer:2 = scf.for %i = %c0_i32 to %N step %c1_i32 iter_args(%tok = %tok0, %out = %cst) -> (!ttg.async.token, tensor<128x128xf32, #blocked>)  : i32 {
      %tok1 = ttng.tmem_store %cst, %acc_tm[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %inner:2 = scf.for %j = %c0_i32 to %K step %c1_i32 iter_args(%inner_tok = %tok1, %useD = %false) -> (!ttg.async.token, i1)  : i32 {
        %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%inner_tok], %useD, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %mma_tok, %true : !ttg.async.token, i1
      }
      %result, %load_tok = ttng.tmem_load %acc_tm[%inner#0] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %load_tok, %result : !ttg.async.token, tensor<128x128xf32, #blocked>
    }
    tt.return %outer#1 : tensor<128x128xf32, #blocked>
  }
}

// -----

// Test hoisting when the dep token is defined outside the loop (not loop-carried).
// This is the pattern seen in the autoWS pipeline after doBufferAllocation.
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @hoist_non_loop_carried_dep
  // The store's dep token is from tmem_alloc, defined outside the loop.
  // CHECK: %[[ZEROS:.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: ttng.tmem_store %[[ZEROS]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: scf.for
  // CHECK-NOT: ttng.tmem_store
  // CHECK:   scf.for
  // CHECK:     ttng.tc_gen5_mma
  tt.func public @hoist_non_loop_carried_dep(
      %A_sh: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %B_sh: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %N: i32, %K: i32) {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %acc_tm, %alloc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    scf.for %i = %c0_i32 to %N step %c1_i32  : i32 {
      %store_tok = ttng.tmem_store %cst, %acc_tm[%alloc_tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %B_trans = ttg.memdesc_trans %B_sh {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
      %inner:2 = scf.for %j = %c0_i32 to %K step %c1_i32 iter_args(%inner_tok = %store_tok, %useD = %false) -> (!ttg.async.token, i1)  : i32 {
        %mma_tok = ttng.tc_gen5_mma %A_sh, %B_trans, %acc_tm[%inner_tok], %useD, %true : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %mma_tok, %true : !ttg.async.token, i1
      }
      %result, %load_tok = ttng.tmem_load %acc_tm[%inner#0] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    }
    tt.return
  }
}

// -----

// Negative test: the store source is NOT loop-invariant (it's a block arg), so
// the store must NOT be hoisted.
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @no_hoist_variant_store_src
  // The store source varies per iteration, so it must remain inside the loop.
  // CHECK: scf.for
  // CHECK:   ttng.tmem_store
  tt.func public @no_hoist_variant_store_src(
      %A_sh: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>,
      %B_sh: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>,
      %N: i32, %K: i32) -> tensor<128x128xf32, #blocked> {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %acc_tm, %tok0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %outer:2 = scf.for %i = %c0_i32 to %N step %c1_i32 iter_args(%tok = %tok0, %prev = %cst) -> (!ttg.async.token, tensor<128x128xf32, #blocked>)  : i32 {
      // Store from previous iteration's result — NOT loop invariant.
      %tok1 = ttng.tmem_store %prev, %acc_tm[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %inner = scf.for %j = %c0_i32 to %K step %c1_i32 iter_args(%inner_tok = %tok1) -> (!ttg.async.token)  : i32 {
        %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%inner_tok], %false, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %mma_tok : !ttg.async.token
      }
      %result, %load_tok = ttng.tmem_load %acc_tm[%inner] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %load_tok, %result : !ttg.async.token, tensor<128x128xf32, #blocked>
    }
    tt.return %outer#1 : tensor<128x128xf32, #blocked>
  }
}

// -----

// Negative test: the MMA uses useD=true, so the store is NOT redundant and
// must not be hoisted.
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @no_hoist_use_d_true
  // MMA accumulates (useD=true), so the per-iteration zero matters.
  // CHECK: scf.for
  // CHECK:   ttng.tmem_store
  tt.func public @no_hoist_use_d_true(
      %A_sh: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>,
      %B_sh: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>,
      %N: i32, %K: i32) -> tensor<128x128xf32, #blocked> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %acc_tm, %tok0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %outer:2 = scf.for %i = %c0_i32 to %N step %c1_i32 iter_args(%tok = %tok0, %out = %cst) -> (!ttg.async.token, tensor<128x128xf32, #blocked>)  : i32 {
      %tok1 = ttng.tmem_store %cst, %acc_tm[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %inner = scf.for %j = %c0_i32 to %K step %c1_i32 iter_args(%inner_tok = %tok1) -> (!ttg.async.token)  : i32 {
        %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%inner_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %mma_tok : !ttg.async.token
      }
      %result, %load_tok = ttng.tmem_load %acc_tm[%inner] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %load_tok, %result : !ttg.async.token, tensor<128x128xf32, #blocked>
    }
    tt.return %outer#1 : tensor<128x128xf32, #blocked>
  }
}
