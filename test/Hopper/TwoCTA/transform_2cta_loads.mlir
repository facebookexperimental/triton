// RUN: triton-opt %s -split-input-file --nvgpu-2cta-transform-loads | FileCheck %s

// Test case: Transform B matrix loads for 2-CTA mode.
// When TCGen5MMAOp has two_ctas=true, this pass should:
// 1. Insert ClusterCTAIdOp to get CTA rank
// 2. Modify B loads to load BLOCK_N/2 columns with CTA-based offset

// CHECK-LABEL: @matmul_2cta_transform_loads
// CHECK: %[[CTA_ID:.*]] = nvgpu.cluster_id
// CHECK: arith.constant 2 : i32
// CHECK: arith.remsi %[[CTA_ID]]
// CHECK: arith.constant 64 : i32
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: tt.descriptor_load
// CHECK: ttng.tc_gen5_mma
// CHECK-SAME: two_ctas

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_2cta_transform_loads(
      %x_desc: !tt.tensordesc<tensor<128x64xf16>>,
      %w_desc: !tt.tensordesc<tensor<64x128xf16>>,
      %z_desc: !tt.tensordesc<tensor<128x128xf16>>,
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>

    %pid = tt.get_program_id x : i32
    %offs_xm = arith.muli %pid, %c128_i32 : i32
    %offs_wn = arith.muli %pid, %c128_i32 : i32

    %accumulator = scf.for %k = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>) : i32 {
      %offs_k = arith.muli %k, %c64_i32 : i32

      // A matrix load (not modified by 2-CTA)
      %x = tt.descriptor_load %x_desc[%offs_xm, %offs_k] : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked1>
      %x_smem = ttg.local_alloc %x : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

      // B matrix load (should be transformed for 2-CTA)
      %w = tt.descriptor_load %w_desc[%offs_k, %offs_wn] : !tt.tensordesc<tensor<64x128xf16>> -> tensor<64x128xf16, #blocked2>
      %w_smem = ttg.local_alloc %w : (tensor<64x128xf16, #blocked2>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

      %acc_layout = ttg.convert_layout %acc : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked3>
      %acc_tmem, %token = ttng.tmem_alloc %acc_layout : (tensor<128x128xf32, #blocked3>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // TCGen5MMA with two_ctas=true - this triggers the 2-CTA transformation
      %mma_token = ttng.tc_gen5_mma %x_smem, %w_smem, %acc_tmem[%token], %true, %true {two_ctas} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      %result, %load_token = ttng.tmem_load %acc_tmem[%mma_token] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked3>
      %result_layout = ttg.convert_layout %result : tensor<128x128xf32, #blocked3> -> tensor<128x128xf32, #blocked>

      scf.yield %result_layout : tensor<128x128xf32, #blocked>
    }

    %z = arith.truncf %accumulator : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %z_layout = ttg.convert_layout %z : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
    tt.descriptor_store %z_desc[%offs_xm, %offs_wn], %z_layout : !tt.tensordesc<tensor<128x128xf16>>, tensor<128x128xf16, #blocked2>
    tt.return
  }
}

// -----

// Test case: No two_ctas attribute - pass should do nothing
// CHECK-LABEL: @matmul_no_2cta
// CHECK: ttng.tc_gen5_mma
// CHECK-NOT: two_ctas

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_no_2cta(
      %x_desc: !tt.tensordesc<tensor<128x64xf16>>,
      %w_desc: !tt.tensordesc<tensor<64x128xf16>>,
      %z_desc: !tt.tensordesc<tensor<128x128xf16>>) attributes {noinline = false} {
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>

    %pid = tt.get_program_id x : i32
    %offs = arith.muli %pid, %c128_i32 : i32

    %x = tt.descriptor_load %x_desc[%offs, %c0_i32] : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked1>
    %x_smem = ttg.local_alloc %x : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

    %w = tt.descriptor_load %w_desc[%c0_i32, %offs] : !tt.tensordesc<tensor<64x128xf16>> -> tensor<64x128xf16, #blocked2>
    %w_smem = ttg.local_alloc %w : (tensor<64x128xf16, #blocked2>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

    %acc_layout = ttg.convert_layout %cst : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked3>
    %acc_tmem, %token = ttng.tmem_alloc %acc_layout : (tensor<128x128xf32, #blocked3>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // No two_ctas attribute - should pass through unchanged
    %mma_token = ttng.tc_gen5_mma %x_smem, %w_smem, %acc_tmem[%token], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    %result, %load_token = ttng.tmem_load %acc_tmem[%mma_token] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked3>
    %z = arith.truncf %result : tensor<128x128xf32, #blocked3> to tensor<128x128xf16, #blocked3>
    %z_layout = ttg.convert_layout %z : tensor<128x128xf16, #blocked3> -> tensor<128x128xf16, #blocked2>
    tt.descriptor_store %z_desc[%offs, %offs], %z_layout : !tt.tensordesc<tensor<128x128xf16>>, tensor<128x128xf16, #blocked2>
    tt.return
  }
}
