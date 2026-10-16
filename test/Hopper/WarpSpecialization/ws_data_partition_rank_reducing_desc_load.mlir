// RUN: triton-opt %s -split-input-file --nvgpu-ws-data-partition=num-warp-groups=1 | FileCheck %s

// A rank-3 descriptor with a [1, BK, BM] block is loaded as a 64x256 tensor
// (the unit dim is dropped) and feeds an MN-major A operand. The M split
// slices load result dim 1, which is descriptor coordinate 2. Partition 1 must
// shift that coordinate, not coordinate 1 (K): shifting K loaded the wrong K
// rows for half of every tile, a silent wrong answer.
// Reduced from the e74 fused kernel (BM=256, data_partition_factor=2).

// CHECK-LABEL: @rank_reducing_desc_load_m_split
// CHECK: tt.make_tensor_descriptor {{.*}} : !tt.ptr<bf16>, !tt.tensordesc<1x64x128xbf16
// CHECK: scf.for
// CHECK: %[[K:.*]] = arith.muli %{{.*}}, %c64_i32 : i32
// CHECK: %[[M:.*]] = arith.subi
// CHECK: %[[M1:.*]] = arith.addi %[[M]], %c128_i32 : i32
// CHECK: tt.descriptor_load %{{.*}}[%[[B:.*]], %[[K]], %[[M]]] : !tt.tensordesc<1x64x128xbf16, {{.*}}> -> tensor<64x128xbf16
// CHECK: tt.descriptor_load %{{.*}}[%[[B]], %[[K]], %[[M1]]] : !tt.tensordesc<1x64x128xbf16, {{.*}}> -> tensor<64x128xbf16

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank_reducing_desc_load_m_split(%x_ptr: !tt.ptr<bf16>, %bt_ptr: !tt.ptr<bf16>, %c_ptr: !tt.ptr<bf16>, %NB: i32, %S: i32, %M: i32, %N: i32, %K: i32) {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %tiles = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #linear>
    %x_desc = arith.muli %K, %S : i32
    %x_desc_0 = arith.extsi %x_desc : i32 to i64
    %x_desc_1 = arith.extsi %S : i32 to i64
    %x_desc_2 = tt.make_tensor_descriptor %x_ptr, [%NB, %K, %S], [%x_desc_0, %x_desc_1, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x64x256xbf16, #shared>
    %b_desc = arith.extsi %K : i32 to i64
    %b_desc_3 = tt.make_tensor_descriptor %bt_ptr, [%N, %K], [%b_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared1>
    %c_desc = arith.extsi %N : i32 to i64
    %c_desc_4 = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%c_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<256x128xbf16, #shared1>
    %tiles_5 = arith.addi %M, %tiles : i32
    %tiles_6 = arith.divsi %tiles_5, %c256_i32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = arith.addi %K, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    scf.for %t = %0 to %tiles_6 step %1  : i32 {
      %off_m = arith.muli %t, %c256_i32 : i32
      %acc, %acc_7 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_8 = ttng.tmem_store %cst, %acc[%acc_7], %true : tensor<256x128xf32, #linear> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_9:2 = scf.for %ki = %c0_i32 to %3 step %c1_i32 iter_args(%acc_12 = %false, %acc_13 = %acc_8) -> (i1, !ttg.async.token)  : i32 {
        %b_i = arith.divsi %off_m, %S : i32
        %at = arith.muli %ki, %c64_i32 : i32
        %at_14 = arith.muli %b_i, %S : i32
        %at_15 = arith.subi %off_m, %at_14 : i32
        %at_16 = tt.descriptor_load %x_desc_2[%b_i, %at, %at_15] : !tt.tensordesc<1x64x256xbf16, #shared> -> tensor<64x256xbf16, #blocked>
        %b = tt.descriptor_load %b_desc_3[%c0_i32, %at] : !tt.tensordesc<128x64xbf16, #shared1> -> tensor<128x64xbf16, #blocked1>
        %acc_17 = ttg.local_alloc %at_16 : (tensor<64x256xbf16, #blocked>) -> !ttg.memdesc<64x256xbf16, #shared1, #smem>
        %acc_18 = ttg.memdesc_trans %acc_17 {order = array<i32: 1, 0>} : !ttg.memdesc<64x256xbf16, #shared1, #smem> -> !ttg.memdesc<256x64xbf16, #shared2, #smem>
        %acc_19 = ttg.local_alloc %b : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared1, #smem>
        %acc_20 = ttg.memdesc_trans %acc_19 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared1, #smem> -> !ttg.memdesc<64x128xbf16, #shared2, #smem>
        %acc_21 = ttng.tc_gen5_mma %acc_18, %acc_20, %acc[%acc_13], %acc_12, %true : !ttg.memdesc<256x64xbf16, #shared2, #smem>, !ttg.memdesc<64x128xbf16, #shared2, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %acc_21 : i1, !ttg.async.token
      }
      %acc_10, %acc_11 = ttng.tmem_load %acc[%acc_9#1] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear>
      %4 = arith.truncf %acc_10 : tensor<256x128xf32, #linear> to tensor<256x128xbf16, #linear>
      %5 = ttg.convert_layout %4 : tensor<256x128xbf16, #linear> -> tensor<256x128xbf16, #blocked2>
      tt.descriptor_store %c_desc_4[%off_m, %c0_i32], %5 : !tt.tensordesc<256x128xbf16, #shared1>, tensor<256x128xbf16, #blocked2>
    } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}
    tt.return
  }
}

// -----

// The data-partition split also runs on a loop without warp specialization.

// CHECK-LABEL: @rank_reducing_desc_load_m_split_no_ws
// CHECK: scf.for
// CHECK: %[[K:.*]] = arith.muli %{{.*}}, %c64_i32 : i32
// CHECK: %[[M:.*]] = arith.subi
// CHECK: %[[M1:.*]] = arith.addi %[[M]], %c128_i32 : i32
// CHECK: tt.descriptor_load %{{.*}}[%[[B:.*]], %[[K]], %[[M]]] : !tt.tensordesc<1x64x128xbf16, {{.*}}> -> tensor<64x128xbf16
// CHECK: tt.descriptor_load %{{.*}}[%[[B]], %[[K]], %[[M1]]] : !tt.tensordesc<1x64x128xbf16, {{.*}}> -> tensor<64x128xbf16
// CHECK: } {tt.data_partition_factor = 2 : i32}

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank_reducing_desc_load_m_split_no_ws(%x_ptr: !tt.ptr<bf16>, %bt_ptr: !tt.ptr<bf16>, %c_ptr: !tt.ptr<bf16>, %NB: i32, %S: i32, %M: i32, %N: i32, %K: i32) {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %tiles = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #linear>
    %x_desc = arith.muli %K, %S : i32
    %x_desc_0 = arith.extsi %x_desc : i32 to i64
    %x_desc_1 = arith.extsi %S : i32 to i64
    %x_desc_2 = tt.make_tensor_descriptor %x_ptr, [%NB, %K, %S], [%x_desc_0, %x_desc_1, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x64x256xbf16, #shared>
    %b_desc = arith.extsi %K : i32 to i64
    %b_desc_3 = tt.make_tensor_descriptor %bt_ptr, [%N, %K], [%b_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared1>
    %c_desc = arith.extsi %N : i32 to i64
    %c_desc_4 = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%c_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<256x128xbf16, #shared1>
    %tiles_5 = arith.addi %M, %tiles : i32
    %tiles_6 = arith.divsi %tiles_5, %c256_i32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = arith.addi %K, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    scf.for %t = %0 to %tiles_6 step %1  : i32 {
      %off_m = arith.muli %t, %c256_i32 : i32
      %acc, %acc_7 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_8 = ttng.tmem_store %cst, %acc[%acc_7], %true : tensor<256x128xf32, #linear> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_9:2 = scf.for %ki = %c0_i32 to %3 step %c1_i32 iter_args(%acc_12 = %false, %acc_13 = %acc_8) -> (i1, !ttg.async.token)  : i32 {
        %b_i = arith.divsi %off_m, %S : i32
        %at = arith.muli %ki, %c64_i32 : i32
        %at_14 = arith.muli %b_i, %S : i32
        %at_15 = arith.subi %off_m, %at_14 : i32
        %at_16 = tt.descriptor_load %x_desc_2[%b_i, %at, %at_15] : !tt.tensordesc<1x64x256xbf16, #shared> -> tensor<64x256xbf16, #blocked>
        %b = tt.descriptor_load %b_desc_3[%c0_i32, %at] : !tt.tensordesc<128x64xbf16, #shared1> -> tensor<128x64xbf16, #blocked1>
        %acc_17 = ttg.local_alloc %at_16 : (tensor<64x256xbf16, #blocked>) -> !ttg.memdesc<64x256xbf16, #shared1, #smem>
        %acc_18 = ttg.memdesc_trans %acc_17 {order = array<i32: 1, 0>} : !ttg.memdesc<64x256xbf16, #shared1, #smem> -> !ttg.memdesc<256x64xbf16, #shared2, #smem>
        %acc_19 = ttg.local_alloc %b : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared1, #smem>
        %acc_20 = ttg.memdesc_trans %acc_19 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared1, #smem> -> !ttg.memdesc<64x128xbf16, #shared2, #smem>
        %acc_21 = ttng.tc_gen5_mma %acc_18, %acc_20, %acc[%acc_13], %acc_12, %true : !ttg.memdesc<256x64xbf16, #shared2, #smem>, !ttg.memdesc<64x128xbf16, #shared2, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %acc_21 : i1, !ttg.async.token
      }
      %acc_10, %acc_11 = ttng.tmem_load %acc[%acc_9#1] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear>
      %4 = arith.truncf %acc_10 : tensor<256x128xf32, #linear> to tensor<256x128xbf16, #linear>
      %5 = ttg.convert_layout %4 : tensor<256x128xbf16, #linear> -> tensor<256x128xbf16, #blocked2>
      tt.descriptor_store %c_desc_4[%off_m, %c0_i32], %5 : !tt.tensordesc<256x128xbf16, #shared1>, tensor<256x128xbf16, #blocked2>
    } {tt.data_partition_factor = 2 : i32}
    tt.return
  }
}
