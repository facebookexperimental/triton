// RUN: triton-opt %s --nvgpu-test-ws-code-partition=num-buffers=2 | FileCheck %s

// Regression test for B-16-F2 / T273493463.
//
// A TMA descriptor load feeding multiple consumer tasks needs the replacement
// local_load to survive in every consumer partition. The fused barrier path
// creates waits for each task; the data load must carry the same full consumer
// task-id set instead of inheriting only the last extra task.

// CHECK-LABEL: @tma_multi_consumer_task_ids
// CHECK: ttng.wait_barrier {{.*}}async_task_id = array<i32: 1>
// CHECK: %[[LOCAL0:.*]] = ttg.local_load {{.*}}async_task_id = array<i32: 1>
// CHECK: arith.addf %[[LOCAL0]], %[[LOCAL0]] {{.*}}async_task_id = array<i32: 1>
// CHECK: ttng.wait_barrier {{.*}}async_task_id = array<i32: 2>
// CHECK: %[[LOCAL1:.*]] = ttg.local_load {{.*}}async_task_id = array<i32: 2>
// CHECK: arith.subf %[[LOCAL1]], %[[LOCAL1]] {{.*}}async_task_id = array<i32: 2>
// CHECK-NOT: tt.descriptor_load
// CHECK-NOT: nvws.descriptor_load

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_multi_consumer_task_ids(%desc: !tt.tensordesc<128x64xf16, #shared>, %out0: !tt.ptr<f16>, %out1: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c4 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 4 : i32
    %ptrs0 = tt.splat %out0 {async_task_id = array<i32: 1>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptrs1 = tt.splat %out1 {async_task_id = array<i32: 2>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %i0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : index
    %i1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : index
    %i4 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 4 : index
    %buffer = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %iv = %i0 to %i4 step %i1 {
      %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      ttg.local_store %tile, %buffer {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %local0 = ttg.local_load %buffer {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %consumer0 = arith.addf %local0, %local0 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked>
      %local1 = ttg.local_load %buffer {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %consumer1 = arith.subf %local1, %local1 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked>
      tt.store %ptrs0, %consumer0 {async_task_id = array<i32: 1>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      tt.store %ptrs1, %consumer1 {async_task_id = array<i32: 2>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      scf.yield
    } {async_task_id = array<i32: 0, 1, 2>, tt.warp_specialize}
    tt.return
  }
}
