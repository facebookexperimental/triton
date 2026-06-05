// RUN: triton-opt %s --nvgpu-ws-data-partition=num-warp-groups=3 | FileCheck %s

// Regression test for B-14-F1 / T273489733.
// Data partitioning is driven by dots with multiple consumer task ids. An
// unrelated dot with a single consumer task id should not be sliced merely
// because another dot in the same function needs partitioning.

// CHECK-LABEL: @single_task_dot_not_partitioned
// CHECK: ttng.warp_group_dot
// CHECK-SAME: {async_task_id = array<i32: 1>
// CHECK-SAME: -> tensor<64x256xf32, #mma>
// CHECK: ttng.warp_group_dot
// CHECK-SAME: {async_task_id = array<i32: 2>
// CHECK-SAME: -> tensor<64x256xf32, #mma>
// CHECK: ttng.warp_group_dot
// CHECK-SAME: {async_task_id = array<i32: 1>
// CHECK-SAME: !ttg.memdesc<128x64xf16, #shared, #smem>
// CHECK-SAME: !ttg.memdesc<64x256xf16, #shared, #smem>
// CHECK-SAME: -> tensor<128x256xf32, #mma>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @single_task_dot_not_partitioned(
      %desc_a_multi: !tt.tensordesc<tensor<128x64xf16>>,
      %desc_b_multi: !tt.tensordesc<tensor<64x256xf16>>,
      %desc_a_single: !tt.tensordesc<tensor<128x64xf16>>,
      %desc_b_single: !tt.tensordesc<tensor<64x256xf16>>,
      %out_multi: !tt.ptr<f16>,
      %out_single: !tt.ptr<f16>) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %acc_multi = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc_single = arith.constant {async_task_id = array<i32: 1>} dense<0.000000e+00> : tensor<128x256xf32, #mma>

    %a_multi = tt.descriptor_load %desc_a_multi[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked>
    %a_multi_smem = ttg.local_alloc %a_multi {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_multi = tt.descriptor_load %desc_b_multi[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #blocked1>
    %b_multi_smem = ttg.local_alloc %b_multi {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %dot_multi = ttng.warp_group_dot %a_multi_smem, %b_multi_smem, %acc_multi {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    %multi_f16 = arith.truncf %dot_multi {async_task_id = array<i32: 1, 2>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %multi_out = ttg.convert_layout %multi_f16 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    %multi_ptr = tt.splat %out_multi {async_task_id = array<i32: 1, 2>} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
    tt.store %multi_ptr, %multi_out {async_task_id = array<i32: 1, 2>} : tensor<128x256x!tt.ptr<f16>, #blocked1>

    %a_single = tt.descriptor_load %desc_a_single[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked>
    %a_single_smem = ttg.local_alloc %a_single {async_task_id = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_single = tt.descriptor_load %desc_b_single[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #blocked1>
    %b_single_smem = ttg.local_alloc %b_single {async_task_id = array<i32: 1>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %dot_single = ttng.warp_group_dot %a_single_smem, %b_single_smem, %acc_single {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    %single_f16 = arith.truncf %dot_single {async_task_id = array<i32: 1>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %single_out = ttg.convert_layout %single_f16 {async_task_id = array<i32: 1>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    %single_ptr = tt.splat %out_single {async_task_id = array<i32: 1>} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
    tt.store %single_ptr, %single_out {async_task_id = array<i32: 1>} : tensor<128x256x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}
