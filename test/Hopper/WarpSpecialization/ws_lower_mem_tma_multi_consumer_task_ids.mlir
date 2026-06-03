// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=2" | FileCheck %s
// XFAIL: *

// Regression test for B-16-F2 / T273493463.
//
// A TMA descriptor load feeding multiple consumer tasks needs the replacement
// local_load to survive in every consumer partition. The fused barrier path
// already creates waits for each task; the data load must carry the same full
// consumer task-id set instead of inheriting only the last extra task.

// CHECK-LABEL: @tma_multi_consumer_task_ids
// CHECK: ttng.wait_barrier {{.*}}async_task_id = array<i32: 1}
// CHECK: ttng.wait_barrier {{.*}}async_task_id = array<i32: 2}
// CHECK: %[[LOCAL:.*]] = ttg.local_load {{.*}}async_task_id = array<i32: 1, 2}
// CHECK: arith.addf %[[LOCAL]], %[[LOCAL]] {{.*}}async_task_id = array<i32: 1}
// CHECK: arith.subf %[[LOCAL]], %[[LOCAL]] {{.*}}async_task_id = array<i32: 2}

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_multi_consumer_task_ids(%desc: !tt.tensordesc<tensor<128x64xf16, #shared>>) attributes {noinline = false} {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c4 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 4 : i32
    scf.for %iv = %c0 to %c4 step %c1 {
      %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
      %consumer0 = arith.addf %tile, %tile {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked>
      %consumer1 = arith.subf %tile, %tile {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked>
      scf.yield
    } {async_task_id = array<i32: 0, 1, 2>, tt.warp_specialize}
    tt.return
  }
}
