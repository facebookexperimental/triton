// RUN: triton-opt %s --nvgpu-test-ping-pong-prep="capability=90 num-stages=3" | FileCheck %s

// WGMMA has memory effects because it reads SMEM operands, but those endpoint
// effects are not intervening effects between two WGMMA ops in the same
// partition. All ten dots below should therefore be one ping-pong group.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @pingpong_group_wgmma_endpoints
// CHECK:      ttng.warp_group_dot
// CHECK-SAME: async_task_id = array<i32: 1>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST:[0-9]+]] : i32
// CHECK-SAME: pingpong_id = [[ID:[0-9]+]] : i32
// CHECK:      ttng.warp_group_dot
// CHECK-SAME: async_task_id = array<i32: 1>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST]] : i32
// CHECK-SAME: pingpong_id = [[ID]] : i32
// CHECK:      ttng.warp_group_dot
// CHECK-SAME: async_task_id = array<i32: 1>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST]] : i32
// CHECK-SAME: pingpong_id = [[ID]] : i32
// CHECK:      ttng.warp_group_dot
// CHECK-SAME: async_task_id = array<i32: 1>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST]] : i32
// CHECK-SAME: pingpong_id = [[ID]] : i32
// CHECK:      ttng.warp_group_dot
// CHECK-SAME: async_task_id = array<i32: 1>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST]] : i32
// CHECK-SAME: pingpong_id = [[ID]] : i32
// CHECK:      ttng.warp_group_dot
// CHECK-SAME: async_task_id = array<i32: 2>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST]] : i32
// CHECK-SAME: pingpong_id = [[ID]] : i32
// CHECK:      ttng.warp_group_dot
// CHECK-SAME: async_task_id = array<i32: 2>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST]] : i32
// CHECK-SAME: pingpong_id = [[ID]] : i32
// CHECK:      ttng.warp_group_dot
// CHECK-SAME: async_task_id = array<i32: 2>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST]] : i32
// CHECK-SAME: pingpong_id = [[ID]] : i32
// CHECK:      ttng.warp_group_dot
// CHECK-SAME: async_task_id = array<i32: 2>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST]] : i32
// CHECK-SAME: pingpong_id = [[ID]] : i32
// CHECK:      ttng.warp_group_dot
// CHECK-SAME: async_task_id = array<i32: 2>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST]] : i32
// CHECK-SAME: pingpong_id = [[ID]] : i32
tt.func public @pingpong_group_wgmma_endpoints(
    %lhs0: !ttg.memdesc<64x64xf16, #shared, #smem>,
    %lhs1: !ttg.memdesc<64x64xf16, #shared, #smem>,
    %rhs: !ttg.memdesc<64x128xf16, #shared, #smem>,
    %num_tiles: i32
) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %init = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #mma>
  %result:2 = scf.for %iv = %c0 to %num_tiles step %c1
      iter_args(%acc0 = %init, %acc1 = %init)
      -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>) : i32 {
    %anchor = arith.addi %iv, %c0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
    %dot0 = ttng.warp_group_dot %lhs0, %rhs, %acc0 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>
    %dot1 = ttng.warp_group_dot %lhs0, %rhs, %dot0 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>
    %dot2 = ttng.warp_group_dot %lhs0, %rhs, %dot1 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>
    %dot3 = ttng.warp_group_dot %lhs0, %rhs, %dot2 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>
    %dot4 = ttng.warp_group_dot %lhs0, %rhs, %dot3 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>
    %dot5 = ttng.warp_group_dot %lhs1, %rhs, %acc1 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>
    %dot6 = ttng.warp_group_dot %lhs1, %rhs, %dot5 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>
    %dot7 = ttng.warp_group_dot %lhs1, %rhs, %dot6 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>
    %dot8 = ttng.warp_group_dot %lhs1, %rhs, %dot7 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>
    %dot9 = ttng.warp_group_dot %lhs1, %rhs, %dot8 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>
    scf.yield {async_task_id = array<i32: 1, 2>} %dot4, %dot9 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>
  } {tt.scheduled_max_stage = 1 : i32}
  tt.return
}

} // module
