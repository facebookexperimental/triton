// RUN: triton-opt %s -split-input-file --nvgpu-ws-data-partition=num-warp-groups=1 --verify-each 2>&1 | FileCheck %s

// CHECK: warning: skipping M-dimension data partitioning because slicing TMEM result from 128 to 64 rows would require updating tensor memory encoding blockM=128
// CHECK: warning: skipping M-dimension data partitioning because slicing TMEM result from 128 to 64 rows would require updating tensor memory encoding blockM=128
// CHECK-LABEL: @tmem_m_slice_requires_encoding_update
// CHECK: scf.for
// CHECK: ttng.tc_gen5_mma
// CHECK-SAME: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
// CHECK: } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}
// CHECK-NOT: !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_m_slice_requires_encoding_update(
      %a_smem: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b_smem: !ttg.memdesc<64x128xf16, #shared1, #smem>) {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %i = %c0_i32 to %c1_i32 step %c1_i32  : i32 {
      %acc, %acc_tok = ttng.tmem_alloc {async_task_id = array<i32: 1, 2>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %a_smem, %b_smem, %acc[%acc_tok], %false, %true {async_task_id = array<i32: 1, 2>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}
    tt.return
  }
}

// -----

// CHECK-LABEL: @tmem_m_slice_scaled_memdesc_reshape_requires_encoding_update
// CHECK: ttg.memdesc_reshape
// CHECK: ttng.tc_gen5_mma_scaled
// CHECK-SAME: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
// CHECK: } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 5}>
#shared3 = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0]]}, alignment = 128>
#shared4 = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0]]}, alignment = 128>
#shared5 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]]}, alignment = 128>
#shared6 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [128, 0]]}, alignment = 128>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_m_slice_scaled_memdesc_reshape_requires_encoding_update(
      %a_smem: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>,
      %b_smem: !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem>,
      %a_scale_smem: !ttg.memdesc<1x1x1x2x256xi8, #shared2, #smem>,
      %b_scale_smem: !ttg.memdesc<256x4xi8, #shared6, #smem>) {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %i = %c0_i32 to %c1_i32 step %c1_i32  : i32 {
      %acc, %acc_tok = ttng.tmem_alloc {async_task_id = array<i32: 1, 2>} : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %a_scale_0 = ttg.memdesc_reshape %a_scale_smem : !ttg.memdesc<1x1x1x2x256xi8, #shared2, #smem> -> !ttg.memdesc<1x1x32x4x4xi8, #shared3, #smem>
      %a_scale_1 = ttg.memdesc_trans %a_scale_0 {order = array<i32: 0, 3, 2, 1, 4>} : !ttg.memdesc<1x1x32x4x4xi8, #shared3, #smem> -> !ttg.memdesc<1x4x32x1x4xi8, #shared4, #smem>
      %a_scale = ttg.memdesc_reshape %a_scale_1 : !ttg.memdesc<1x4x32x1x4xi8, #shared4, #smem> -> !ttg.memdesc<128x4xi8, #shared5, #smem>
      %mma_tok = ttng.tc_gen5_mma_scaled %a_smem, %b_smem, %acc[%acc_tok], %a_scale, %b_scale_smem, %false, %true lhs = e4m3 rhs = e4m3 {async_task_id = array<i32: 1, 2>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x4xi8, #shared5, #smem>, !ttg.memdesc<256x4xi8, #shared6, #smem>
    } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
