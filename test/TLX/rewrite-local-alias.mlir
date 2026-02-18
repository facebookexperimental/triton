// RUN: triton-opt -split-input-file --tlx-rewrite-local-alias %s| FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>

// CHECK-DAG: #[[$SHARED:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
// CHECK-DAG: #[[$SHARED1:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
// CHECK-DAG: #[[$TMEM:.*]] = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>

module attributes {tlx.has_explicit_local_mem_access = true, tlx.has_tlx_ops = true, tlx.has_warp_spec_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tcgen5_fa_kernel
  tt.func public @tcgen5_fa_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK: %[[$LOCAL_ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x64x16xf16, #[[$SHARED]], #smem, mutable>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1x16x32xf16, #shared1, #smem, mutable>

    // CHECK-NOT: tlx.local_alias
    // CHECK: ttg.memdesc_reinterpret %[[$LOCAL_ALLOC]] : !ttg.memdesc<1x64x16xf16, #[[$SHARED]], #smem, mutable> -> !ttg.memdesc<1x32x32xf16, #[[$SHARED1]], #smem, mutable>
    %2 = tlx.local_alias %0 : !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<1x32x32xf16, #shared1, #smem, mutable>

    // CHECK: %[[$TMEM_ALLOC:.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x32xf32, #[[$TMEM]], #ttng.tensor_memory, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x32xf16, #tmem1, #ttng.tensor_memory, mutable>

    // CHECK-NOT: tlx.local_alias
    // CHECK: ttg.memdesc_reinterpret %[[$TMEM_ALLOC]] : !ttg.memdesc<1x64x32xf32, #[[$TMEM]], #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x32xf16, #[[$TMEM]], #ttng.tensor_memory, mutable>
    %result_0 = tlx.local_alias %result : !ttg.memdesc<1x64x32xf16, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_1 = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_specialize(%0, %result_0, %1, %2, %result_1, %result)
    default {
      ttg.warp_yield
    }
    partition0(%arg8: !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable>, %arg9: !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>, %arg10: !ttg.memdesc<1x16x32xf16, #shared1, #smem, mutable>, %arg11: !ttg.memdesc<1x32x32xf16, #shared1, #smem, mutable>, %arg12: !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>, %arg13: !ttg.memdesc<1x64x32xf16, #tmem1, #ttng.tensor_memory, mutable>) num_warps(1) {
      %true = arith.constant true
      %false = arith.constant false
      %c0_i32 = arith.constant 0 : i32
      %3 = ttg.memdesc_index %arg8[%c0_i32] : !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable>
      %4 = ttg.memdesc_index %arg10[%c0_i32] : !ttg.memdesc<1x16x32xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable>
      %5 = ttg.memdesc_index %arg9[%c0_i32] : !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
      %6 = ttng.tc_gen5_mma %3, %4, %5[], %false, %true : !ttg.memdesc<64x16xf16, #shared, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared1, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
      %7 = ttg.memdesc_index %arg13[%c0_i32] : !ttg.memdesc<1x64x32xf16, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf16, #tmem1, #ttng.tensor_memory, mutable>
      %8 = ttg.memdesc_index %arg11[%c0_i32] : !ttg.memdesc<1x32x32xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>
      %9 = ttg.memdesc_index %arg12[%c0_i32] : !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %false, %true : !ttg.memdesc<64x32xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
      ttg.warp_return
    }
    partition1(%arg8: !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable>, %arg9: !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>, %arg10: !ttg.memdesc<1x16x32xf16, #shared1, #smem, mutable>, %arg11: !ttg.memdesc<1x32x32xf16, #shared1, #smem, mutable>, %arg12: !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>, %arg13: !ttg.memdesc<1x64x32xf16, #tmem1, #ttng.tensor_memory, mutable>) num_warps(4) {
      %true = arith.constant true
      %c0_i32 = arith.constant 0 : i32
      %3 = ttg.memdesc_index %arg9[%c0_i32] : !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_2 = ttng.tmem_load %3 : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #blocked>
      %4 = ttg.convert_layout %result_2 : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #blocked1>
      %5 = arith.truncf %4 : tensor<64x32xf32, #blocked1> to tensor<64x32xf16, #blocked1>
      %6 = ttg.memdesc_index %arg13[%c0_i32] : !ttg.memdesc<1x64x32xf16, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf16, #tmem1, #ttng.tensor_memory, mutable>
      %7 = ttg.convert_layout %5 : tensor<64x32xf16, #blocked1> -> tensor<64x32xf16, #blocked>
      ttng.tmem_store %7, %6, %true : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #tmem1, #ttng.tensor_memory, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1x64x16xf16, #shared, #smem, mutable>, !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x16x32xf16, #shared1, #smem, mutable>, !ttg.memdesc<1x32x32xf16, #shared1, #smem, mutable>, !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x64x32xf16, #tmem1, #ttng.tensor_memory, mutable>) -> ()
    tt.return
  }
}
