// RUN: triton-opt %s --nvgpu-test-ws-buffer-allocation '--nvgpu-test-ws-memory-planner=num-buffers=2 smem-budget=200000' '--nvgpu-test-ws-code-partition=num-buffers=2' | FileCheck %s

// Regression test for a TMA load whose SMEM tile feeds multiple consumers. The
// buffer-allocation pre-pass should avoid materializing one local_alloc/local_store
// chain per consumer when both chains come from the same descriptor_load value.
// That lets code partitioning discover one SMEM buffer with two consumer tasks
// without relying on buffer.id, which is also used for logical buffer reuse.

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
  tt.func public @post_tma_load_multi_consumer_allocs(%desc: !tt.tensordesc<128x64xf16, #shared>, %out0: !tt.ptr<f16>, %out1: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %ptrs0 = tt.splat %out0 {async_task_id = array<i32: 1>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptrs1 = tt.splat %out1 {async_task_id = array<i32: 2>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    %alloc0 = ttg.local_alloc %tile {async_task_id = array<i32: 0, 1>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %alloc1 = ttg.local_alloc %tile {async_task_id = array<i32: 0, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %loaded0 = ttg.local_load %alloc0 {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
    %loaded1 = ttg.local_load %alloc1 {async_task_id = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
    %consumer0 = arith.addf %loaded0, %loaded0 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked>
    %consumer1 = arith.subf %loaded1, %loaded1 {async_task_id = array<i32: 2>} : tensor<128x64xf16, #blocked>
    tt.store %ptrs0, %consumer0 {async_task_id = array<i32: 1>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.store %ptrs1, %consumer1 {async_task_id = array<i32: 2>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
