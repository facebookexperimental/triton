// RUN: triton-opt %s --nvgpu-test-ws-code-partition='num-buffers=2' | FileCheck %s
// XFAIL: *

// Regression test for B-1-F1 / T273462573.
// Same-partition SMEM producer/consumer pairs are not cross-partition channels.
// The post code partition path should leave the pre-planned allocation shape
// alone instead of creating a multi-buffered ChannelPost for an empty consumer
// task set.

// CHECK-LABEL: @same_partition_smem_post_no_channel
// CHECK-NOT: memdesc<2x16xf32
// CHECK: tt.return
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @same_partition_smem_post_no_channel(%src: tensor<16xf32, #blocked>) attributes {noinline = false} {
    %alloc = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 2 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    ttg.local_store %src, %alloc {async_task_id = array<i32: 0>} : tensor<16xf32, #blocked> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    %loaded = ttg.local_load %alloc {async_task_id = array<i32: 0>} : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
    %used = arith.addf %loaded, %loaded {async_task_id = array<i32: 0>} : tensor<16xf32, #blocked>
    tt.return
  }
}
