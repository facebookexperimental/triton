// RUN: triton-opt %s --nvgpu-test-ws-code-partition='num-buffers=2 post-channel-creation=1' | FileCheck %s
// XFAIL: *

// Regression test for B-1-F2 / T273462656.
// Multi-task SMEM producers can have consumers in multiple task partitions.
// The post channel collector should filter per partition instead of asserting
// when both the producer and consumer task-id sets contain more than one id.

// CHECK-LABEL: @multi_task_smem_producer_multiple_consumers
// CHECK: tt.return
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @multi_task_smem_producer_multiple_consumers(%src: tensor<16xf32, #blocked>) attributes {noinline = false} {
    %alloc = ttg.local_alloc {async_task_id = array<i32: 0, 1>, buffer.copy = 2 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    ttg.local_store %src, %alloc {async_task_id = array<i32: 0, 1>} : tensor<16xf32, #blocked> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    %loaded0 = ttg.local_load %alloc {async_task_id = array<i32: 0>} : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
    %loaded1 = ttg.local_load %alloc {async_task_id = array<i32: 1>} : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
    %used = arith.addf %loaded0, %loaded1 {async_task_id = array<i32: 0, 1>} : tensor<16xf32, #blocked>
    tt.return
  }
}
