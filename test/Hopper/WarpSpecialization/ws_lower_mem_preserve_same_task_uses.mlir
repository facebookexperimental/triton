// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=2" | FileCheck %s

// Regression test for B-16-F1 / T273493462.
//
// A producer load can have both same-task and cross-task consumers. WSLowerMem
// must only lower the cross-task channel operand through SMEM; the same-task
// use must stay on the original load result.

// CHECK-LABEL: @preserve_same_task_load_use
// CHECK: [[LOAD:%.*]] = tt.load
// CHECK: arith.addf [[LOAD]], [[LOAD]]
// CHECK-SAME: ttg.partition = array<i32: 0>}
// CHECK: [[LOCAL:%.*]] = ttg.local_load
// CHECK: arith.addf [[LOCAL]], [[LOCAL]]
// CHECK-SAME: ttg.partition = array<i32: 1>}

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @preserve_same_task_load_use(%ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %ptrs0 = tt.splat %ptr {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %ptrs1 = tt.splat %ptr {ttg.partition = array<i32: 1>} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %loaded = tt.load %ptrs0 {ttg.partition = array<i32: 0>} : tensor<16x!tt.ptr<f32>, #blocked>
    %same_task = arith.addf %loaded, %loaded {ttg.partition = array<i32: 0>} : tensor<16xf32, #blocked>
    tt.store %ptrs0, %same_task {ttg.partition = array<i32: 0>} : tensor<16x!tt.ptr<f32>, #blocked>
    %cross_task = arith.addf %loaded, %loaded {ttg.partition = array<i32: 1>} : tensor<16xf32, #blocked>
    tt.store %ptrs1, %cross_task {ttg.partition = array<i32: 1>} : tensor<16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
