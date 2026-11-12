// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=1" | FileCheck %s

// Untagged ops after the warp-specialized loop read memory the loop wrote, so
// they must stay after the warp_specialize op.

// CHECK-LABEL: @post_loop_ops_after_ws
// CHECK-NOT: tt.load
// CHECK: ttg.warp_specialize
// CHECK: partition0
// CHECK: tt.store
// CHECK: } : (
// CHECK-NEXT: %[[V:.*]] = tt.load
// CHECK-NEXT: tt.store %{{.*}}, %[[V]]
// CHECK-NEXT: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @post_loop_ops_after_ws(%a: tensor<128x!tt.ptr<f32>, #blocked>, %b: tensor<128x!tt.ptr<f32>, #blocked>, %out: tensor<128x!tt.ptr<f32>, #blocked>, %n: index) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : index
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : index
    %one = arith.constant {async_task_id = array<i32: 0>} dense<1.0> : tensor<128xf32, #blocked>
    %alloc = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    scf.for %iv = %c0 to %n step %c1 {
      tt.store %a, %one {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      ttg.local_store %one, %alloc {async_task_id = array<i32: 0>} : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {async_task_id = array<i32: 1>} : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
      tt.store %b, %loaded {async_task_id = array<i32: 1>} : tensor<128x!tt.ptr<f32>, #blocked>
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["compute", "compute"]}
    %v = tt.load %a : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %out, %v : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
