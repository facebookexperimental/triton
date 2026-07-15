// RUN: triton-opt %s --nvgpu-ws-data-partition=num-warp-groups=3 | FileCheck %s
// Regression test for B-14-F3 / T273489833.
// If partitioning a dot result through another dot's K operand is not backed
// by an atomic store/reduce, the pass should reject that trial direction
// deterministically and choose the legal N-dimension direction here.

// CHECK-LABEL: @dot_k_operand_without_atomic_store
// CHECK: ttng.warp_group_dot
// CHECK-SAME: !ttg.memdesc<128x64xf16, #shared, #smem>
// CHECK-SAME: !ttg.memdesc<64x128xf16, #shared, #smem>
// CHECK-SAME: -> tensor<128x128xf32, #mma>
// CHECK: ttng.warp_group_dot
// CHECK-SAME: !ttg.memdesc<128x64xf16, #shared, #smem>
// CHECK-SAME: !ttg.memdesc<64x128xf16, #shared, #smem>
// CHECK-SAME: -> tensor<128x128xf32, #mma>
// CHECK: ttng.warp_group_dot
// CHECK-SAME: !ttg.memdesc<128x128xf16, #shared, #smem>
// CHECK-SAME: !ttg.memdesc<128x128xf16, #shared, #smem>
// CHECK-SAME: -> tensor<128x128xf32, #mma>
// CHECK: ttng.warp_group_dot
// CHECK-SAME: !ttg.memdesc<128x128xf16, #shared, #smem>
// CHECK-SAME: !ttg.memdesc<128x128xf16, #shared, #smem>
// CHECK-SAME: -> tensor<128x128xf32, #mma>
// CHECK: tt.store

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_k_operand_without_atomic_store(
      %desc_a0: !tt.tensordesc<tensor<128x64xf16>>,
      %desc_b0: !tt.tensordesc<tensor<64x256xf16>>,
      %desc_a1: !tt.tensordesc<tensor<128x128xf16>>,
      %out: !tt.ptr<f16>) {
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 0 : i32
    %acc0 = arith.constant {ttg.partition = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc1 = arith.constant {ttg.partition = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>

    %a0 = tt.descriptor_load %desc_a0[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked1>
    %a0_smem = ttg.local_alloc %a0 {ttg.partition = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b0 = tt.descriptor_load %desc_b0[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #blocked>
    %b0_smem = ttg.local_alloc %b0 {ttg.partition = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %dot0 = ttng.warp_group_dot %a0_smem, %b0_smem, %acc0 {ttg.partition = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    %dot0_f16 = arith.truncf %dot0 {ttg.partition = array<i32: 1, 2>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %dot0_smem = ttg.local_alloc %dot0_f16 {ttg.partition = array<i32: 1, 2>} : (tensor<128x256xf16, #mma>) -> !ttg.memdesc<128x256xf16, #shared, #smem>

    %a1 = tt.descriptor_load %desc_a1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #blocked1>
    %a1_smem = ttg.local_alloc %a1 {ttg.partition = array<i32: 1, 2>} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %dot1 = ttng.warp_group_dot %a1_smem, %dot0_smem, %acc1 {ttg.partition = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem> * !ttg.memdesc<128x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    %dot1_f16 = arith.truncf %dot1 {ttg.partition = array<i32: 1, 2>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %dot1_out = ttg.convert_layout %dot1_f16 {ttg.partition = array<i32: 1, 2>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    %out_ptr = tt.splat %out {ttg.partition = array<i32: 1, 2>} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.store %out_ptr, %dot1_out {ttg.partition = array<i32: 1, 2>} : tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
