// RUN: triton-opt -split-input-file --tlx-propagate-layout %s | FileCheck %s

// Test that warp-specialized TMEM paths remain valid after propagation. This
// covers memdesc constraints in the consumer partition together with
// tensor-side release/require cleanup around tmem_load/tmem_store.

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem_1 = #ttg.shared_memory
// Use two textual aliases for the same TMEM encoding so the input IR can model
// a partition-local memdesc constraint without introducing an unsupported TMEM
// layout mismatch.
#tmem_1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
#tmem_2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
// CHECK-DAG: #[[$TMEM:.*]] = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>

module attributes {tlx.has_explicit_local_mem_access = true, tlx.has_tlx_ops = true, tlx.has_warp_spec_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @warp_specialize_tmem_paths
  tt.func public @warp_specialize_tmem_paths() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x64x16xf16, #shared, #smem_1, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1x16x32xf16, #shared1, #smem_1, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<1x32x32xf16, #shared1, #smem_1, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>
    %result_0 = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x32xf16, #tmem_2, #ttng.tensor_memory, mutable>
    %result_1 = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>
    ttg.warp_specialize(%0, %result, %1, %2, %result_1, %result_0)
    default {
      ttg.warp_yield
    }
    partition0(%arg8: !ttg.memdesc<1x64x16xf16, #shared, #smem_1, mutable>, %arg9: !ttg.memdesc<1x64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>, %arg10: !ttg.memdesc<1x16x32xf16, #shared1, #smem_1, mutable>, %arg11: !ttg.memdesc<1x32x32xf16, #shared1, #smem_1, mutable>, %arg12: !ttg.memdesc<1x64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>, %arg13: !ttg.memdesc<1x64x32xf16, #tmem_2, #ttng.tensor_memory, mutable>) num_warps(1) {
      %true = arith.constant true
      %false = arith.constant false
      %c0_i32 = arith.constant 0 : i32
      %3 = ttg.memdesc_index %arg8[%c0_i32] : !ttg.memdesc<1x64x16xf16, #shared, #smem_1, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem_1, mutable>
      %4 = ttg.memdesc_index %arg10[%c0_i32] : !ttg.memdesc<1x16x32xf16, #shared1, #smem_1, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem_1, mutable>
      %5 = ttg.memdesc_index %arg9[%c0_i32] : !ttg.memdesc<1x64x32xf32, #tmem_1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>
      %6 = ttng.tc_gen5_mma %3, %4, %5[], %false, %true : !ttg.memdesc<64x16xf16, #shared, #smem_1, mutable>, !ttg.memdesc<16x32xf16, #shared1, #smem_1, mutable>, !ttg.memdesc<64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>
      // CHECK: %[[TMEM_F16:.*]] = ttg.memdesc_index %arg5[%c0_i32] : !ttg.memdesc<1x64x32xf16, #[[$TMEM]], #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf16, #[[$TMEM]], #ttng.tensor_memory, mutable>
      %7 = ttg.memdesc_index %arg13[%c0_i32] : !ttg.memdesc<1x64x32xf16, #tmem_2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf16, #tmem_2, #ttng.tensor_memory, mutable>
      %8 = ttg.memdesc_index %arg11[%c0_i32] : !ttg.memdesc<1x32x32xf16, #shared1, #smem_1, mutable> -> !ttg.memdesc<32x32xf16, #shared1, #smem_1, mutable>
      %9 = ttg.memdesc_index %arg12[%c0_i32] : !ttg.memdesc<1x64x32xf32, #tmem_1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>
      %10 = tlx.require_layout %7 : !ttg.memdesc<64x32xf16, #tmem_2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf16, #tmem_1, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma %[[TMEM_F16]], %{{.*}}, %{{.*}}[], %false, %true : !ttg.memdesc<64x32xf16, #[[$TMEM]], #ttng.tensor_memory, mutable>, !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>, !ttg.memdesc<64x32xf32, #[[$TMEM]], #ttng.tensor_memory, mutable>
      %11 = ttng.tc_gen5_mma %10, %8, %9[], %false, %true : !ttg.memdesc<64x32xf16, #tmem_1, #ttng.tensor_memory, mutable>, !ttg.memdesc<32x32xf16, #shared1, #smem_1, mutable>, !ttg.memdesc<64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>
      ttg.warp_return
    }
    partition1(%arg8: !ttg.memdesc<1x64x16xf16, #shared, #smem_1, mutable>, %arg9: !ttg.memdesc<1x64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>, %arg10: !ttg.memdesc<1x16x32xf16, #shared1, #smem_1, mutable>, %arg11: !ttg.memdesc<1x32x32xf16, #shared1, #smem_1, mutable>, %arg12: !ttg.memdesc<1x64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>, %arg13: !ttg.memdesc<1x64x32xf16, #tmem_2, #ttng.tensor_memory, mutable>) num_warps(4) {
      %true = arith.constant true
      %c0_i32 = arith.constant 0 : i32
      %3 = ttg.memdesc_index %arg9[%c0_i32] : !ttg.memdesc<1x64x32xf32, #tmem_1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>
      // CHECK: %[[TMEM_LOAD:.*]] = ttng.tmem_load %{{.*}} : !ttg.memdesc<64x32xf32, #[[$TMEM]], #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #{{.*}}>
      %result_2 = ttng.tmem_load %3 : !ttg.memdesc<64x32xf32, #tmem_1, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #blocked>
      // CHECK: %[[REL_CVT:.*]] = ttg.convert_layout %[[TMEM_LOAD]] : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #blocked1>
      %4 = tlx.release_layout %result_2 : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #blocked1>
      %5 = arith.truncf %4 : tensor<64x32xf32, #blocked1> to tensor<64x32xf16, #blocked1>
      %6 = ttg.memdesc_index %arg13[%c0_i32] : !ttg.memdesc<1x64x32xf16, #tmem_2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf16, #tmem_2, #ttng.tensor_memory, mutable>
      // CHECK: %[[STORE_CVT:.*]] = ttg.convert_layout %{{.*}} : tensor<64x32xf16, #blocked1> -> tensor<64x32xf16, #blocked>
      %7 = tlx.require_layout %5 : tensor<64x32xf16, #blocked1> -> tensor<64x32xf16, #blocked>
      // CHECK: ttng.tmem_store %[[STORE_CVT]], %{{.*}}, %true : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #[[$TMEM]], #ttng.tensor_memory, mutable>
      ttng.tmem_store %7, %6, %true : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #tmem_2, #ttng.tensor_memory, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1x64x16xf16, #shared, #smem_1, mutable>, !ttg.memdesc<1x64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x16x32xf16, #shared1, #smem_1, mutable>, !ttg.memdesc<1x32x32xf16, #shared1, #smem_1, mutable>, !ttg.memdesc<1x64x32xf32, #tmem_1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x64x32xf16, #tmem_2, #ttng.tensor_memory, mutable>) -> ()
    tt.return
  }
}
