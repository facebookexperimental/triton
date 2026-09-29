// RUN: triton-opt %s --nvgpu-ws-data-partition=num-warp-groups=3 -verify-diagnostics | FileCheck %s

// A 64x128 accumulator cannot be split two ways (32 rows is below the minimum
// M slice and 64 columns below the minimum N slice), so data partitioning is
// skipped with a remark and the tile stays whole instead of failing the pass.

// CHECK-LABEL: @small_tile_skips_partitioning
// CHECK: ttng.warp_group_dot {{.*}} -> tensor<64x128xf32, #mma>
// CHECK: tt.store {{.*}} : tensor<64x128x!tt.ptr<f32>, #mma>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @small_tile_skips_partitioning(%desc_a: !tt.tensordesc<64x64xf16>, %desc_b: !tt.tensordesc<64x128xf16>, %out: !tt.ptr<f32>) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %a = tt.descriptor_load %desc_a[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked>
    %a_smem = ttg.local_alloc %a {async_task_id = array<i32: 1, 2>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %b = tt.descriptor_load %desc_b[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked1>
    %b_smem = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    // expected-remark @below {{skipping data partitioning: a 64x128 accumulator is too small to split 2 ways}}
    %dot = ttng.warp_group_dot %a_smem, %b_smem, %acc {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>
    %ptr = tt.splat %out {async_task_id = array<i32: 1, 2>} : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #mma>
    tt.store %ptr, %dot {async_task_id = array<i32: 1, 2>} : tensor<64x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}
