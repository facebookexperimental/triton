// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=2" | FileCheck %s
// XFAIL: *

// Regression test for B-12-F1 / T273486918.
//
// The only post-code-partition SMEM channel is directly inside the else block
// of an scf.if. WSBuffer accumulator rewriting must leave the then branch's
// accumulator unchanged and increment the else branch's accumulator. The
// current implementation increments the then yield and leaves else unchanged.

// CHECK-LABEL: @else_direct_smem_channel
// CHECK: scf.if {{.*}} -> (i64) {
// CHECK-NEXT: %[[THEN_ZERO:.*]] = arith.constant{{.*}} 0 : i64
// CHECK-NEXT: scf.yield {{.*}}%[[THEN_ZERO]] : i64
// CHECK-NEXT: } else {
// CHECK: %[[ELSE_ONE:.*]] = arith.constant{{.*}} 1 : i64
// CHECK-NEXT: scf.yield {{.*}}%[[ELSE_ONE]] : i64
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @else_direct_smem_channel(%src: tensor<16xf32, #blocked>) attributes {noinline = false} {
    %alloc = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 2 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : i32
    %pred = arith.cmpi eq, %c0_i32, %c1_i32 {async_task_id = array<i32: 0, 1>} : i32
    scf.if %pred {
      scf.yield
    } else {
      %stored = arith.addf %src, %src {async_task_id = array<i32: 0>} : tensor<16xf32, #blocked>
      ttg.local_store %stored, %alloc {async_task_id = array<i32: 0>} : tensor<16xf32, #blocked> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {async_task_id = array<i32: 1>} : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
      %used = arith.addf %loaded, %loaded {async_task_id = array<i32: 1>} : tensor<16xf32, #blocked>
      scf.yield
    } {async_task_id = array<i32: 0, 1>}
    tt.return
  }
}
