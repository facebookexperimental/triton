// RUN: triton-opt %s --nvgpu-ws-data-partition=num-warp-groups=1 --verify-each 2>&1 | FileCheck %s

// CHECK: warning: skipping M-dimension data partitioning because slicing TMEM result from 128 to 64 rows would require updating tensor memory encoding blockM=128
// CHECK-LABEL: @tmem_m_slice_requires_encoding_update
// CHECK: scf.for
// CHECK-NOT: tt.warp_specialize
// CHECK: ttng.tc_gen5_mma
// CHECK-SAME: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
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
