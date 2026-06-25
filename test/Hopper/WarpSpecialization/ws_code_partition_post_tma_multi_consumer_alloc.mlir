// RUN: triton-opt %s --nvgpu-test-ws-code-partition='num-buffers=2 post-channel-creation=1' | FileCheck %s

// Regression test for a post-allocated TMA load whose SMEM tile feeds multiple
// consumers. The memory planner can materialize one local_store/local_alloc
// pair per consumer even when both allocs map to the same physical buffer.
// Code partitioning should merge those stores before TMA lowering so the
// descriptor load becomes one async TMA copy instead of tripping the
// single-local-store invariant in optimizeTMALoads.

// CHECK-LABEL: @post_tma_load_multi_consumer_allocs
// CHECK-COUNT-1: ttng.async_tma_copy_global_to_local
// CHECK: ttng.wait_barrier {{.*}}async_task_id = array<i32: 1>
// CHECK: ttg.local_load {{.*}}async_task_id = array<i32: 1>
// CHECK: ttng.wait_barrier {{.*}}async_task_id = array<i32: 2>
// CHECK: ttg.local_load {{.*}}async_task_id = array<i32: 2>
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @post_tma_load_multi_consumer_allocs(%desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %out0: !tt.ptr<f16>, %out1: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %ptrs0 = tt.splat %out0 {async_task_id = array<i32: 1>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptrs1 = tt.splat %out1 {async_task_id = array<i32: 2>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
    %alloc0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %alloc1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttg.local_store %tile, %alloc0 {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttg.local_store %tile, %alloc1 {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %loaded0 = ttg.local_load %alloc0 {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
    %loaded1 = ttg.local_load %alloc1 {async_task_id = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
    %consumer0 = arith.addf %loaded0, %loaded0 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked>
    %consumer1 = arith.subf %loaded1, %loaded1 {async_task_id = array<i32: 2>} : tensor<128x64xf16, #blocked>
    tt.store %ptrs0, %consumer0 {async_task_id = array<i32: 1>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.store %ptrs1, %consumer1 {async_task_id = array<i32: 2>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
