// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation '--nvgpu-test-ws-memory-planner=num-buffers=2 smem-budget=200000' '--nvgpu-test-ws-code-partition=num-buffers=2' | FileCheck %s
// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation '--nvgpu-test-ws-memory-planner=num-buffers=2 smem-budget=200000' '--nvgpu-test-ws-code-partition=num-buffers=2' | FileCheck %s --check-prefix=TASK1
// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation '--nvgpu-test-ws-memory-planner=num-buffers=2 smem-budget=200000' '--nvgpu-test-ws-code-partition=num-buffers=2' | FileCheck %s --check-prefix=TASK2

// Buffer creation should reuse the descriptor load's existing SMEM allocation.

// CHECK-LABEL: @post_tma_load_multi_consumer_allocs
// CHECK-NOT: tt.descriptor_load
// CHECK-COUNT-1: ttng.async_tma_copy_global_to_local
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.return

// TASK1-LABEL: @post_tma_load_multi_consumer_allocs
// TASK1: ttng.wait_barrier {{.*}}async_task_id = array<i32: 1>
// TASK1: ttg.local_load {{.*}}async_task_id = array<i32: 1>

// TASK2-LABEL: @post_tma_load_multi_consumer_allocs
// TASK2: ttng.wait_barrier {{.*}}async_task_id = array<i32: 2>
// TASK2: ttg.local_load {{.*}}async_task_id = array<i32: 2>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @post_tma_load_multi_consumer_allocs(%desc: !tt.tensordesc<128x64xf16, #shared>, %out0: !tt.ptr<f16>, %out1: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %ptrs0 = tt.splat %out0 {async_task_id = array<i32: 1>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptrs1 = tt.splat %out1 {async_task_id = array<i32: 2>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    %alloc0 = ttg.local_alloc %tile {async_task_id = array<i32: 0, 1>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %loaded0 = ttg.local_load %alloc0 {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
    %consumer0 = arith.addf %loaded0, %loaded0 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked>
    %consumer1 = arith.subf %tile, %tile {async_task_id = array<i32: 2>} : tensor<128x64xf16, #blocked>
    tt.store %ptrs0, %consumer0 {async_task_id = array<i32: 1>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.store %ptrs1, %consumer1 {async_task_id = array<i32: 2>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// An allocation in an ancestor block can serve a direct consumer in a nested
// region when it dominates that consumer.

// CHECK-LABEL: @post_tma_load_nested_consumer
// CHECK-NOT: tt.descriptor_load
// CHECK-COUNT-1: ttng.async_tma_copy_global_to_local
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @post_tma_load_nested_consumer(%desc: !tt.tensordesc<128x64xf16, #shared>, %out0: !tt.ptr<f16>, %out1: !tt.ptr<f16>, %lb: i32, %ub: i32, %step: i32) attributes {noinline = false} {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %ptrs0 = tt.splat %out0 {async_task_id = array<i32: 1>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptrs1 = tt.splat %out1 {async_task_id = array<i32: 2>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    %alloc = ttg.local_alloc %tile {async_task_id = array<i32: 0, 1>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %loaded = ttg.local_load %alloc {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked>
    tt.store %ptrs0, %loaded {async_task_id = array<i32: 1>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    scf.for %iv = %lb to %ub step %step : i32 {
      %consumer = arith.addf %tile, %tile {async_task_id = array<i32: 2>} : tensor<128x64xf16, #blocked>
      tt.store %ptrs1, %consumer {async_task_id = array<i32: 2>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    } {async_task_id = array<i32: 0, 2>}
    tt.return
  }
}
