// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=1 post-channel-creation=1" | FileCheck %s

// Regression test for B-19-F1 / T273499456.
//
// The task-1 partition consumes the result of an scf.if even though the then
// branch yields an untagged rematerializable value. Specialization must keep
// the if result in partition0 instead of dropping it based on the producer of
// the then-yielded value.

// CHECK-LABEL: @if_result_used_by_task
// CHECK: ttg.warp_specialize
// CHECK: partition0
// CHECK: %[[IF:.*]] = scf.if {{.*}} -> (i32) {
// CHECK: scf.yield {{.*}} : i32
// CHECK: } else {
// CHECK: scf.yield {{.*}} : i32
// CHECK: }
// CHECK: %[[USE:.*]] = arith.addi %[[IF]],
// CHECK: tt.splat %[[USE]]

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @if_result_used_by_task(%src: tensor<16xf32, #blocked>, %dst: tensor<16x!tt.ptr<f32>, #blocked>) {
    %c0 = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : index
    %c1 = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : index
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
    %c1_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
    %zero_vec = arith.constant {ttg.partition = array<i32: 1>} dense<0> : tensor<16xi32, #blocked>
    %alloc = ttg.local_alloc {ttg.partition = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    scf.for %iv = %c0 to %c1 step %c1 {
      %c7_i32 = arith.constant 7 : i32
      %pred = arith.cmpi eq, %c0_i32, %c1_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %selected = scf.if %pred -> (i32) {
        scf.yield {ttg.partition = array<i32: 0, 1>} %c7_i32 : i32
      } else {
        %task_value = arith.addi %c0_i32, %c1_i32 {ttg.partition = array<i32: 1>} : i32
        scf.yield {ttg.partition = array<i32: 0, 1>} %task_value : i32
      } {ttg.partition = array<i32: 0, 1>}
      %used = arith.addi %selected, %c1_i32 {ttg.partition = array<i32: 1>} : i32
      %used_vec = tt.splat %used {ttg.partition = array<i32: 1>} : i32 -> tensor<16xi32, #blocked>
      %mask = arith.cmpi sge, %used_vec, %zero_vec {ttg.partition = array<i32: 1>} : tensor<16xi32, #blocked>
      %stored = arith.addf %src, %src {ttg.partition = array<i32: 0>} : tensor<16xf32, #blocked>
      ttg.local_store %stored, %alloc {ttg.partition = array<i32: 0>} : tensor<16xf32, #blocked> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
      tt.store %dst, %loaded, %mask {ttg.partition = array<i32: 1>} : tensor<16x!tt.ptr<f32>, #blocked>
    } {ttg.partition = array<i32: 0, 1>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["compute", "compute"]}
    tt.return
  }
}
