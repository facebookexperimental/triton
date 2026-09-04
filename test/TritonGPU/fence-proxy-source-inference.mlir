// RUN: triton-opt %s --triton-nvidia-gpu-fence-insertion | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_ws = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma_a = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#mma_b = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @backing_ignores_alias_lifetime
  tt.func @backing_ignores_alias_lifetime(
      %src: tensor<64x32xf16, #blocked>,
      %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %alias = ttg.memdesc_reinterpret %a {tlx.logical_lifetime_boundary} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    ttg.local_store %src, %alias : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %bt = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %b = ttg.memdesc_trans %bt {order = array<i32: 1, 0>} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> -> !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>
    %false = arith.constant false
    %true = arith.constant true
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %a, %b, %acc, %false, %true {is_async} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func @alias_lifetime_sees_cross_partition_store
  tt.func @alias_lifetime_sees_cross_partition_store(
      %src: tensor<64x64xf16, #blocked_ws>) {
    %backing = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %a = ttg.memdesc_reinterpret %backing {tlx.logical_lifetime_boundary} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %a_view = ttg.memdesc_reinterpret %a : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_specialize(%a_view, %b, %acc, %src)
    default {
      ttg.warp_yield
    }
    partition0(%buf: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %rhs: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %dst: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %value: tensor<64x64xf16, #blocked_ws>) num_warps(4) {
      %true = arith.constant true
      %selected = scf.if %true -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable> {
        scf.yield %buf : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      } else {
        scf.yield %buf : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      }
      ttg.local_store %value, %selected : tensor<64x64xf16, #blocked_ws> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      ttg.warp_return
    }
    partition1(%lhs: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %rhs: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %dst: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %value: tensor<64x64xf16, #blocked_ws>) num_warps(4) {
      %true = arith.constant true
      // CHECK: ttng.fence_async_shared
      // CHECK-NEXT: ttng.tc_gen5_mma
      ttng.tc_gen5_mma %lhs, %rhs, %dst, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, tensor<64x64xf16, #blocked_ws>) -> ()
    tt.return
  }

  // CHECK-LABEL: tt.func @alias_lifetime_ignores_backing_and_sibling_stores
  tt.func @alias_lifetime_ignores_backing_and_sibling_stores(
      %src: tensor<64x32xf16, #blocked>,
      %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %backing = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    ttg.local_store %src, %backing : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %a = ttg.memdesc_reinterpret %backing {tlx.logical_lifetime_boundary} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %sibling = ttg.memdesc_reinterpret %backing {tlx.logical_lifetime_boundary} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    ttg.local_store %src, %sibling : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %bt = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %b = ttg.memdesc_trans %bt {order = array<i32: 1, 0>} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> -> !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>
    %false = arith.constant false
    %true = arith.constant true
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %a, %b, %acc, %false, %true {is_async} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func public @unknown_function_argument
  tt.func public @unknown_function_argument(
      %a: !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>,
      %b: !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>,
      %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %false = arith.constant false
    %true = arith.constant true
    // CHECK: ttng.fence_async_shared
    // CHECK-NEXT: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %a, %b, %acc, %false, %true {is_async} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  tt.func private @identity_buffer(
      %a: !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>) -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> {
    tt.return %a : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
  }

  // CHECK-LABEL: tt.func @unknown_call_and_if_results
  tt.func @unknown_call_and_if_results(
      %src: tensor<64x32xf16, #blocked>,
      %acc0: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc1: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %from_call = tt.call @identity_buffer(%a) : (!ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>) -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %bt = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %b = ttg.memdesc_trans %bt {order = array<i32: 1, 0>} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> -> !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>
    %false = arith.constant false
    %true = arith.constant true
    // CHECK: ttng.fence_async_shared
    // CHECK-NEXT: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %from_call, %b, %acc0, %false, %true {is_async} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %safe_backing = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    ttg.local_store %src, %safe_backing : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %safe_alias = ttg.memdesc_reinterpret %safe_backing {tlx.logical_lifetime_boundary} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %safe = scf.if %true -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> {
      scf.yield %safe_alias : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    } else {
      scf.yield %safe_alias : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    }
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %safe, %b, %acc1, %false, %true {is_async} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %unsafe_backing = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %unsafe_alias = ttg.memdesc_reinterpret %unsafe_backing {tlx.logical_lifetime_boundary} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %unsafe = scf.if %true -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> {
      ttg.local_store %src, %unsafe_alias : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
      scf.yield %unsafe_alias : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    } else {
      scf.yield %unsafe_alias : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    }
    // CHECK: ttng.fence_async_shared
    // CHECK-NEXT: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %unsafe, %b, %acc1, %false, %true {is_async} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func @safe_loop_carried_mma
  tt.func @safe_loop_carried_mma(
      %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %true = arith.constant true
    %a = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %bt = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %b = ttg.memdesc_trans %bt {order = array<i32: 1, 0>} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> -> !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>
    %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %a) -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> {
      // CHECK-NOT: ttng.fence_async_shared
      // CHECK: ttng.tc_gen5_mma
      ttng.tc_gen5_mma %iter, %b, %acc, %false, %true {is_async} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %iter : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    }
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %result, %b, %acc, %false, %true {is_async} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func @unsafe_loop_carried_mma
  tt.func @unsafe_loop_carried_mma(
      %src: tensor<64x32xf16, #blocked>,
      %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %true = arith.constant true
    %a = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %bt = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    %b = ttg.memdesc_trans %bt {order = array<i32: 1, 0>} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> -> !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>
    %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %a) -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable> {
      ttg.local_store %src, %iter : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
      // CHECK: ttng.fence_async_shared
      // CHECK-NEXT: ttng.tc_gen5_mma
      ttng.tc_gen5_mma %iter, %b, %acc, %false, %true {is_async} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %iter : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>
    }
    // CHECK: ttng.fence_async_shared
    // CHECK-NEXT: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %result, %b, %acc, %false, %true {is_async} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.fence_async_shared
    // CHECK-NEXT: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %a, %b, %acc, %false, %true {is_async} : !ttg.memdesc<64x32xf16, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf16, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
