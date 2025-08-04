// RUN: triton-opt %s -split-input-file --triton-nvidia-optimize-descriptor-encoding | FileCheck %s
// Test that gather/scatter are assigned swizzled encodings

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-DAG: #[[NVMMA_32:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
tt.func public @tma_gather(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: tensor<32xi32, #blocked> ) -> tensor<32x32xi8, #blocked1> {
  // CHECK: tt.make_tensor_descriptor {{.*}} : <i8>, <tensor<1x32xi8, #[[NVMMA_32]]>>
  // CHECK: tt.descriptor_gather {{.*}} : (!tt.tensordesc<tensor<1x32xi8, #[[NVMMA_32]]>>
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant dense<32> : tensor<8x1xi32>
  %c64_i32 = arith.constant 64 : i32
  %c8_i32 = arith.constant 8 : i32
  %0 = arith.extsi %arg2 : i32 to i64
  %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <i8>, <tensor<1x32xi8>>
  %2 = tt.descriptor_gather %1[%arg3, %c8_i32] : (!tt.tensordesc<tensor<1x32xi8>>, tensor<32xi32, #blocked>, i32) -> tensor<32x32xi8, #blocked1>
  tt.return %2 : tensor<32x32xi8, #blocked1>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-DAG: #[[NVMMA_32:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
tt.func public @tma_scatter(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: tensor<32xi32, #blocked>, %arg4: tensor<32x32xi8, #blocked1>) {
  // CHECK: tt.make_tensor_descriptor {{.*}} : <i8>, <tensor<1x32xi8, #[[NVMMA_32]]>>
  // CHECK: tt.descriptor_scatter {{.*}} : !tt.tensordesc<tensor<1x32xi8, #[[NVMMA_32]]>>, {{.*}}
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant dense<32> : tensor<8x1xi32>
  %c64_i32 = arith.constant 64 : i32
  %c8_i32 = arith.constant 8 : i32
  %0 = arith.extsi %arg2 : i32 to i64
  %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <i8>, <tensor<1x32xi8>>
  tt.descriptor_scatter %1[%arg3, %c8_i32], %arg4 : !tt.tensordesc<tensor<1x32xi8>>, tensor<32xi32, #blocked>, i32, tensor<32x32xi8, #blocked1>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-DAG: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[SWIZZLE_MMA:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
// CHECK-DAG: #[[SWIZZLE_2D:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
tt.func public @tma_scatter(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64) {
  // CHECK: tt.make_tensor_descriptor {{.*}} : <f32>, <tensor<1x256x32xf32, #[[SWIZZLE_MMA]]>>
  // CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<1x256x32xf32, #[[SWIZZLE_MMA]]>> -> tensor<256x32xf32, #[[BLOCKED]]>
  // CHECK: ttg.local_alloc %[[LOAD]] : (tensor<256x32xf32, #[[BLOCKED]]>) -> !ttg.memdesc<256x32xf32, #[[SWIZZLE_2D]], #smem>
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %0 = tt.make_tensor_descriptor %arg0, [%c1_i32, %arg1, %arg2], [%arg3, %arg4, %c1_i64] : <f32>, <tensor<1x256x32xf32>>
  %1 = tt.descriptor_load %0[%c1_i32, %c1_i32, %c1_i32] : !tt.tensordesc<tensor<1x256x32xf32>> -> tensor<256x32xf32, #blocked>
  %2 = ttg.local_alloc %1 : (tensor<256x32xf32, #blocked>) -> !ttg.memdesc<256x32xf32, #shared, #smem>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-DAG: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[NVMMA_64:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
tt.func public @descriptor_kernel_arg(%arg0: !tt.tensordesc<tensor<64x64xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64) {
  // CHECK: %arg0: !tt.tensordesc<tensor<64x64xf16, #[[NVMMA_64]]>>
  // CHECK: %[[LOAD:.*]] = tt.descriptor_load %arg0[{{.*}}] : !tt.tensordesc<tensor<64x64xf16, #[[NVMMA_64]]>> -> tensor<64x64xf16, #[[BLOCKED]]>
  // CHECK: ttg.local_alloc %[[LOAD]] : (tensor<64x64xf16, #[[BLOCKED]]>) -> !ttg.memdesc<64x64xf16, #[[NVMMA_64]], #smem>
  %c1_i32 = arith.constant 1 : i32
  %1 = tt.descriptor_load %arg0[%c1_i32, %c1_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #blocked>
  %2 = ttg.local_alloc %1 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK-DAG: #[[SHARED:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {tlx.has_explicit_local_mem_access = true, tlx.has_tlx_ops = true, tlx.has_warp_spec_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: %arg0: !tt.tensordesc<tensor<128x64xf16, #[[SHARED]]>>
  tt.func public @ttng_load_seed_from_default(%arg0: !tt.tensordesc<tensor<128x64xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<128x64xf16>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64) attributes {noinline = false} {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %2 = ttg.memdesc_subview %1[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %2, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %3 = ttg.memdesc_subview %1[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.warp_specialize(%arg5, %result)
    default {
      %4 = ttg.memdesc_subview %0[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: tensor_desc_to_tma_ptr{{.*}}!tt.tensordesc<tensor<128x64xf16, #[[SHARED]]>>
      %5 = ttng.tensor_desc_to_tma_ptr %arg0 : !tt.tensordesc<tensor<128x64xf16>> to !tt.ptr<i8>
      ttng.async_tma_copy_global_to_local %5[%c0_i32, %c0_i32] %4, %2, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    // CHECK: %arg10: !tt.tensordesc<tensor<128x64xf16, #[[SHARED]]>>
    partition0(%arg10: !tt.tensordesc<tensor<128x64xf16>>, %arg11: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
      %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
      %true_0 = arith.constant true
      %c0_i32_1 = arith.constant 0 : i32
      %4 = ttg.memdesc_subview %arg11[%c0_i32_1, %c0_i32_1, %c0_i32_1] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tmem_store %cst, %4, %true_0 : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttg.warp_return
    }
    // CHECK: %arg10: !tt.tensordesc<tensor<128x64xf16, #[[SHARED]]>>
    partition1(%arg10: !tt.tensordesc<tensor<128x64xf16>>, %arg11: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
      %c0_i32_0 = arith.constant 0 : i32
      %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
      ttg.warp_return
    // CHECK: (!tt.tensordesc<tensor<128x64xf16, #[[SHARED]]>>
    } : (!tt.tensordesc<tensor<128x64xf16>>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK: #[[SHARED:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {tlx.has_explicit_local_mem_access = true, tlx.has_tlx_ops = true, tlx.has_warp_spec_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: %arg0: !tt.tensordesc<tensor<128x64xf16, #[[SHARED]]>>
  tt.func public @ttng_store_seed_from_non_default(%arg0: !tt.tensordesc<tensor<128x64xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<128x128xf16>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %2 = ttg.memdesc_subview %1[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %2, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %3 = ttg.memdesc_subview %1[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.warp_specialize(%0, %arg5, %result)
    default {
      %4 = ttg.memdesc_subview %result[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tmem_store %cst, %4, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttg.warp_yield
    }
    // CHECK: %arg11: !tt.tensordesc<tensor<128x128xf16, #[[SHARED]]>>
    partition0(%arg10: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg11: !tt.tensordesc<tensor<128x128xf16>>, %arg12: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
      %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
      %true_1 = arith.constant true
      %c0_i32_2 = arith.constant 0 : i32
      %4 = ttg.memdesc_subview %arg12[%c0_i32_2, %c0_i32_2, %c0_i32_2] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tmem_store %cst_0, %4, %true_1 : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttg.warp_return
    }
    // CHECK: %arg11: !tt.tensordesc<tensor<128x128xf16, #[[SHARED]]>>
    partition1(%arg10: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg11: !tt.tensordesc<tensor<128x128xf16>>, %arg12: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
      %c0_i32_0 = arith.constant 0 : i32
      %4 = ttg.memdesc_subview %arg10[%c0_i32_0, %c0_i32_0, %c0_i32_0] : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: tensor_desc_to_tma_ptr{{.*}}!tt.tensordesc<tensor<128x128xf16, #[[SHARED]]>>
      %5 = ttng.tensor_desc_to_tma_ptr %arg11 : !tt.tensordesc<tensor<128x128xf16>> to !tt.ptr<i8>
      ttng.async_tma_copy_local_to_global %5[%c0_i32_0, %c0_i32_0] %4 : !tt.ptr<i8>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.warp_return
    // CHECK: !tt.tensordesc<tensor<128x128xf16, #[[SHARED]]>>
    } : (!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !tt.tensordesc<tensor<128x128xf16>>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
    tt.return
  }
}
