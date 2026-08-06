// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-alloc-algo=1 smem-budget=196608" | FileCheck %s

// Hoist descriptor-load staging allocations reached through memdesc views.

// CHECK-LABEL: @hoist_tma_load_staging_through_view
// CHECK: %[[ALLOC:.*]] = ttg.local_alloc {{.*}}!ttg.memdesc<1x128x64xf16
// CHECK: scf.for
// CHECK: ttg.memdesc_index %[[ALLOC]]

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hoist_tma_load_staging_through_view(
      %desc: !tt.tensordesc<128x64xf16, #shared>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : i32
    %channel_buffer = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %iv = %c0 to %c1 step %c1 : i32 {
      %staging_buffer = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
      %staging_view = ttg.memdesc_index %staging_buffer[%c0] : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      ttg.local_store %tile, %staging_view {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %tile, %channel_buffer {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %staging_value = ttg.local_load %staging_view {async_task_id = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %channel_value = ttg.local_load %channel_buffer {async_task_id = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %sum = arith.addf %staging_value, %channel_value {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked>
      scf.yield {async_task_id = array<i32: 0, 1>}
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize}
    tt.return
  }
}
